import os
import random
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from sunpy.map import Map
import sunpy.map
from astropy.visualization import ImageNormalize, LogStretch, LinearStretch, PercentileInterval
import astropy.io.ascii as ascii
from astropy.table import QTable
import csv
from scipy.ndimage import zoom, gaussian_filter, label, find_objects
from scipy import ndimage
import pandas as pd  # 新增：用于生成汇总表格

# 导入BID反卷积函数
from deconv import deconvolve_bid

# 项目配置
from config import get_config
from model import PhysicsAwareUNet

# ====================== 全局汇总变量：存储所有波段功率谱数据 ======================
band_power_summary = {}  # 键：波段(int)，值：该波段的功率谱数据
# 3×3子图固定位置映射（严格满足：335在第三行第二个）
POSITION_MAP = {
    94: (0, 0),   # 第一行第一列
    131: (0, 1),  # 第一行第二列
    171: (0, 2),  # 第一行第三列
    193: (1, 0),  # 第二行第一列
    211: (1, 1),  # 第二行第二列
    304: (1, 2),  # 第二行第三列
    335: (2, 1)   # 第三行第二个 ✅ 核心要求
}
# ============================================================================

def calculate_fwhm(
    profile: np.ndarray,
    sigma_smooth: float = 1.0
) -> float:
    """
    ✅ 天文标准版：基于峰值双向搜索的亚像素 FWHM 计算
    能够完美免疫外围噪声干扰，准确捕捉中心单一结构的真实分辨率。
    
    Args:
        profile: 一维强度剖面（slit 提取的原始数据）
        sigma_smooth: 用于寻找峰值的轻微高斯平滑标准差
    Returns:
        fwhm: 真实的半高全宽（像素尺度）
    """
    # 1. 基础数据清洗
    profile = np.nan_to_num(profile, nan=0.0)
    if len(profile) < 5:
        return 0.0
        
    # 2. 全局轻微平滑（仅用于抑制极端噪点对峰值定位的干扰）
    smoothed = gaussian_filter(profile, sigma=sigma_smooth)
    
    # 3. 动态寻找真正的结构峰值（不再盲目假设中心是峰值）
    peak_idx = np.argmax(smoothed)
    peak_val = smoothed[peak_idx]
    
    # 4. 健壮的背景估算：以整条剖面两端最暗的区域作为真实背景基线
    bg = np.min(smoothed)
    
    # 5. 计算物理意义上的净峰值与半高切线
    peak_corrected = peak_val - bg
    if peak_corrected < 1e-3:  # 几乎无信号
        return 0.0
    half_max_corrected = peak_corrected / 2.0 + bg  # 映射回含背景的坐标系
    
    # 6. 从峰值点开始，【向左】寻找第一个穿过半高线的交叉点
    left_idx = peak_idx
    while left_idx > 0 and smoothed[left_idx] >= half_max_corrected:
        left_idx -= 1
        
    # 左边界亚像素级线性插值
    if left_idx >= 0 and smoothed[left_idx] < half_max_corrected:
        y1, y2 = smoothed[left_idx], smoothed[left_idx + 1]
        x1, x2 = left_idx, left_idx + 1
        left_pos = x1 + (half_max_corrected - y1) * (x2 - x1) / (y2 - y1 + 1e-8)
    else:
        left_pos = float(left_idx)

    # 7. 从峰值点开始，【向右】寻找第一个穿过半高线的交叉点
    right_idx = peak_idx
    while right_idx < len(smoothed) - 1 and smoothed[right_idx] >= half_max_corrected:
        right_idx += 1
        
    # 右边界亚像素级线性插值
    if right_idx < len(smoothed) and smoothed[right_idx] < half_max_corrected:
        y1, y2 = smoothed[right_idx - 1], smoothed[right_idx]
        x1, x2 = right_idx - 1, right_idx
        right_pos = x1 + (half_max_corrected - y1) * (x2 - x1) / (y2 - y1 + 1e-8)
    else:
        right_pos = float(right_idx)
        
    # 8. 计算最终的物理 FWHM
    fwhm = right_pos - left_pos
    
    # 排除异常，若结构未完全闭合则返回 0
    if fwhm <= 0 or left_idx == 0 or right_idx == len(smoothed) - 1:
        return 0.0
        
    return round(fwhm, 2)

def save_fwhm_metrics(band: int, file_idx: int, fwhm_dict: dict, save_root: str):
    """保存FWHM量化指标到CSV文件"""
    csv_path = os.path.join(save_root, "fwhm_metrics.csv")
    header = ["Band(Å)", "File_Index", "Obverse_FWHM(px)", "Proposed_FWHM(px)", "RL_FWHM(px)", "BID_FWHM(px)"]
    row = [
        band, file_idx,
        round(fwhm_dict['Obverse'], 4),
        round(fwhm_dict['Proposed'], 4),
        round(fwhm_dict['RL'], 4),
        round(fwhm_dict['BID'], 4)
    ]
    
    # ====================== 修改1：每次写入前覆盖文件，清空旧数据 ======================
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:  # 把 'a' 改成 'w'
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow(row)
    
    print(f"📊 FWHM指标保存完成 | Obverse: {row[2]:.2f} | Proposed: {row[3]:.2f} | RL: {row[4]:.2f} | BID: {row[5]:.2f}")

# ====================== 【新增】全波段FWHM汇总函数 ======================
def summarize_fwhm_all_bands(output_root):
    """
    汇总所有波段的FWHM指标，生成Markdown表格+CSV汇总文件
    【已修复】读取局部耀斑FWHM，计算平均值，彻底抛弃旧全局数据
    """
    summary_data = []
    # 遍历所有波段文件夹
    for band_folder in os.listdir(output_root):
        if not band_folder.startswith("band_"):
            continue
        band = int(band_folder.split("_")[1])
        csv_path = os.path.join(output_root, band_folder, "fwhm_metrics.csv")
        if not os.path.exists(csv_path):
            print(f"⚠️ 未找到波段{band}的FWHM文件")
            continue
        # 读取单波段FWHM数据
        df = pd.read_csv(csv_path)
        # ====================== 修改2：计算均值（支持多张图，单张图也兼容） ======================
        obverse_fwhm = df["Obverse_FWHM(px)"].mean()
        proposed_fwhm = df["Proposed_FWHM(px)"].mean()
        rl_fwhm = df["RL_FWHM(px)"].mean()
        bid_fwhm = df["BID_FWHM(px)"].mean()
        
        summary_data.append([band, obverse_fwhm, proposed_fwhm, rl_fwhm, bid_fwhm])
    
    # 按波段数值排序
    summary_data = sorted(summary_data, key=lambda x: x[0])
    # 生成汇总DataFrame
    summary_df = pd.DataFrame(
        summary_data,
        columns=["波段(Å)", "观测图像(Obverse)FWHM(px)", "Proposed方法FWHM(px)", "RL方法FWHM(px)", "BID方法FWHM(px)"]
    )
    
    # 保存汇总CSV到根目录
    summary_csv_path = os.path.join(output_root, "FWHM_SUMMARY_TABLE.csv")
    summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ FWHM汇总表格已保存：{summary_csv_path}")
    
    # 控制台打印格式化Markdown表格
    print("\n" + "="*85)
    print("📋 全波段FWHM分辨率指标汇总表（耀斑局部区域）")
    print("="*85)
    print("| 波段(Å) | 观测图像(Obverse)FWHM(px) | Proposed方法FWHM(px) | RL方法FWHM(px) | BID方法FWHM(px) |")
    print("|---------|--------------------------|----------------------|---------------|----------------|")
    for row in summary_data:
        print(f"| {row[0]:>3}     | {row[1]:>24.2f} | {row[2]:>20.2f} | {row[3]:>13.2f} | {row[4]:>14.2f} |")
    print("="*85)
    return summary_df
# ============================================================================

# ====================== 独立PSF退化工具类 ======================
class PSFDegradation:
    def __init__(self, psf_file: str, psf_crop_size: int = 64, debug: bool = False):
        self.debug = debug
        self.psf_crop_size = psf_crop_size
        self.psf_tensor = self._load_and_preprocess_psf(psf_file)
        self.K = self.psf_tensor.shape[-1]
        self.psf_fft = None
        self.pad_h = None
        self.pad_w = None
        self.precomputed_H = None
        self.precomputed_W = None

    def _load_and_preprocess_psf(self, psf_file: str) -> torch.Tensor:
        try:
            with fits.open(psf_file) as hdul:
                psf_data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data
                psf_data = psf_data.astype(np.float32)

            h, w = psf_data.shape
            target_size = self.psf_crop_size
            if h > target_size:
                start = (h - target_size) // 2
                psf_data = psf_data[start:start+target_size, start:start+target_size]
            elif h < target_size:
                pad = (target_size - h) // 2
                psf_data = np.pad(psf_data, pad, mode='constant')

            psf_sum = np.sum(psf_data)
            if psf_sum == 0:
                raise RuntimeError("PSF数据总和为0")
            psf_data = psf_data / psf_sum

            psf_tensor = torch.from_numpy(psf_data).unsqueeze(0).unsqueeze(0).float()
            return psf_tensor

        except Exception as e:
            raise RuntimeError(f"加载PSF失败: {str(e)}")

    def _compute_psf_fft(self, H: int, W: int, device: torch.device):
        psf = self.psf_tensor.to(device)
        K = self.K
        pad_h = H + K - 1
        pad_w = W + K - 1

        psf_pad = torch.zeros((1, 1, pad_h, pad_w), dtype=psf.dtype, device=device)
        center_h, center_w = pad_h // 2, pad_w // 2
        start_h = center_h - (K // 2)
        start_w = center_w - (K // 2)
        psf_pad[:, :, start_h:start_h+K, start_w:start_w+K] = psf

        psf_pad_shifted = torch.fft.ifftshift(psf_pad, dim=(-2, -1))
        psf_fft = torch.fft.fft2(psf_pad_shifted)
        return psf_fft, pad_h, pad_w

    def degrade_numpy(self, img_np: np.ndarray) -> np.ndarray:
        """
        新增：直接对numpy图像做PSF退化（适配RL/BID的numpy输出）
        Args:
            img_np: [H, W] numpy数组
        Returns:
            degraded_np: PSF退化后的[H, W] numpy数组
        """
        # 转为模型输入格式 [1,1,H,W]
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).float().to(device)
        degraded_tensor = self.degrade(img_tensor)
        return degraded_tensor.squeeze().cpu().numpy()

    def degrade(self, img: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img.shape
        device = img.device

        if self.precomputed_H != H or self.precomputed_W != W:
            self.psf_fft, self.pad_h, self.pad_w = self._compute_psf_fft(H, W, device)
            self.precomputed_H, self.precomputed_W = H, W

        img_pad = torch.nn.functional.pad(img, (0, self.pad_w - W, 0, self.pad_h - H), mode='constant', value=0.0)
        img_fft = torch.fft.fft2(img_pad)
        degraded_fft = img_fft * self.psf_fft.to(device)
        degraded_img = torch.fft.ifft2(degraded_fft).real
        
        degraded_img = degraded_img[:, :, :H, :W]
        degraded_img = torch.clamp(degraded_img, min=0.0)
        return degraded_img
# ======================================================================================

# 数据根目录
TEST_DATA_ROOT = "./data/test/test_classified/"
MODEL_ROOT = "./outputs"
CUSTOM_OUTPUT_ROOT = "./outputs/test_results_zoom_416/"

# 适配XXXX-HR目录的波段提取函数
def get_available_bands(data_root):
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")
    
    band_folders = [
        f for f in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, f)) 
        and f.endswith("-HR")
    ]
    
    bands = []
    for folder in band_folders:
        band_str = folder.split("-")[0]
        try:
            band = int(band_str)
            bands.append(band)
        except ValueError:
            print(f"⚠️ 跳过无效波段文件夹: {folder}")
            continue
    
    if not bands:
        raise ValueError(f"在 {data_root} 下未找到有效波段文件夹")
    
    return sorted(bands)

BANDS = get_available_bands(TEST_DATA_ROOT)
print(f"✅ 自动识别到待处理波段: {BANDS}")

START_TIME = "2025-06-20 00:00:00"
END_TIME   = "2025-06-20 23:59:59"
MAX_FILES_PER_BAND = 5

# ====================== 配置项 ======================
USE_MANUAL_ZOOM = True
MANUAL_ZOOM_START = 50
MANUAL_ZOOM_END = 300
FLARE_ZOOM_SIZE = 200
FLARE_INTENSITY_PERCENTILE = 95

PROFILE_USE_MANUAL_YLIM = False
PROFILE_YLIM_MIN = None
PROFILE_YLIM_MAX = None

IMAGE_USE_MANUAL_CLIM = False
IMAGE_MANUAL_VMIN = None
IMAGE_MANUAL_VMAX = None
IMAGE_VMIN_PERCENTILE = 1.0
IMAGE_VMAX_PERCENTILE = 99.5

DIFF_USE_MANUAL_CLIM = False
DIFF_MANUAL_VMIN = None
DIFF_MANUAL_VMAX = None
DIFF_VMAX_PERCENTILE = 99.5

REF_USE_MANUAL_CLIM = False
REF_MANUAL_VMIN = None
REF_MANUAL_VMAX = None
REF_VMIN_PERCENTILE = 0.0
REF_VMAX_PERCENTILE = 99.5
# ====================================================

PROFILE_FIGSIZE = (14, 8)
PROFILE_DPI = 300
POWER_SPECTRUM_FIGSIZE = (18, 12)  # 放大适配3×3汇总图
# 新增：单波段PSD图配置
SINGLE_PSD_FIGSIZE = (12, 8)
SINGLE_PSD_DPI = 300

COLOR_OBVERSE = '#2E8B57'  # 统一替换Ref为Obverse
COLOR_MODEL_OUTPUT = "#FF1100"
COLOR_RL_DECONV = "#F78605"
COLOR_BID_DECONV = "#0077FF"

PROFILE_COLORS = {
    'Obverse': COLOR_OBVERSE,
    'Proposed': COLOR_MODEL_OUTPUT,
    'RL': COLOR_RL_DECONV,
    'BID': COLOR_BID_DECONV
}

PROFILE_STYLES = {
    'Obverse': {'linestyle': '-', 'marker': None, 'markersize': 0, 'linewidth': 2.0},
    'Proposed': {'linestyle': '--', 'marker': None, 'markersize': 0, 'linewidth': 2.0},
    'RL': {'linestyle': '-.', 'marker': None, 'markersize': 0, 'linewidth': 2.0},
    'BID': {'linestyle': ':', 'marker': None, 'markersize': 0, 'linewidth': 2.0}
}

WAVELENGTH_TO_CMAP = {
    94: 'sdoaia94', 131: 'sdoaia131', 171: 'sdoaia171',
    193: 'sdoaia193', 211: 'sdoaia211', 304: 'sdoaia304', 335: 'sdoaia335'
}
AIA_OFFICIAL_CONFIG = {
    94: (0, 99.5, True),131: (2, 99.5, True),171: (0, 99.5, False),
    193: (1, 99.5, False),211: (1, 99.5, False),304: (5, 99.5, True),335: (0, 99.5, True)
}
SUPPORTED_WAVELENGTHS = list(AIA_OFFICIAL_CONFIG.keys())
DIFF_PERCENTILE = 99

aia_band_params = {
    "94": {"HR_DIR_NAME": "0094-HR","QUIET_HR_DIR_NAME": "0094-HR","GLOBAL_MIN": 0.0,"GLOBAL_MAX": 806,"REAL_HR_NORM_MEAN": 0.0052,"VALID_SIGNAL_THRESH": 0.0045,"PSF_FILE": "./PSF/psf_aia_0094.fits"},
    "131":{"HR_DIR_NAME": "0131-HR","QUIET_HR_DIR_NAME": "0131-HR","GLOBAL_MIN": 0.0,"GLOBAL_MAX": 1165,"REAL_HR_NORM_MEAN": 0.0086,"VALID_SIGNAL_THRESH": 0.0045,"PSF_FILE": "./PSF/psf_aia_0131.fits"},
    "171":{"HR_DIR_NAME": "0171-HR","QUIET_HR_DIR_NAME": "0171-HR","GLOBAL_MIN": 0.0,"GLOBAL_MAX": 4619,"REAL_HR_NORM_MEAN": 0.0466,"VALID_SIGNAL_THRESH": 0.0045,"PSF_FILE": "./PSF/psf_aia_0171.fits"},
    "193":{"HR_DIR_NAME": "0193-HR","QUIET_HR_DIR_NAME": "0193-HR","GLOBAL_MIN": 0.0,"GLOBAL_MAX": 5195,"REAL_HR_NORM_MEAN": 0.0518,"VALID_SIGNAL_THRESH": 0.0045,"PSF_FILE": "./PSF/psf_aia_0193.fits"},
    "211":{"HR_DIR_NAME": "0211-HR","QUIET_HR_DIR_NAME": "0211-HR","GLOBAL_MIN": 0.0,"GLOBAL_MAX": 3457,"REAL_HR_NORM_MEAN": 0.0505,"VALID_SIGNAL_THRESH": 0.0045,"PSF_FILE": "./PSF/psf_aia_0211.fits"},
    "304":{"HR_DIR_NAME": "0304-HR","QUIET_HR_DIR_NAME": "0304-HR","GLOBAL_MIN": 0.0,"GLOBAL_MAX": 226,"REAL_HR_NORM_MEAN": 0.0355,"VALID_SIGNAL_THRESH": 0.0045,"PSF_FILE": "./PSF/psf_aia_0304.fits"},
    "335":{"HR_DIR_NAME": "0335-HR","QUIET_HR_DIR_NAME": "0335-HR","GLOBAL_MIN": 0.0,"GLOBAL_MAX": 158,"REAL_HR_NORM_MEAN": 0.0310,"VALID_SIGNAL_THRESH": 0.0045,"PSF_FILE": "./PSF/psf_aia_0335.fits"},
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_psf(wavelength: str, target_size: tuple):
    psf_path = aia_band_params[wavelength]["PSF_FILE"]
    with fits.open(psf_path) as hdul:
        psf = hdul[0].data.astype(np.float32)
    zoom_factor = (target_size[0]/psf.shape[0], target_size[1]/psf.shape[1])
    psf_resampled = zoom(psf, zoom_factor, order=1)
    psf_resampled = psf_resampled / np.sum(psf_resampled)
    psf_resampled = np.maximum(psf_resampled, 0.0)
    psf_tensor = torch.from_numpy(psf_resampled).to(device).float()
    psf_tensor = torch.fft.ifftshift(psf_tensor)
    psf_fft = torch.fft.rfft2(psf_tensor)
    psf_adj_fft = torch.conj(psf_fft)
    return psf_fft, psf_adj_fft, psf_resampled

# ====================== RL迭代次数：10次======================
def richardson_lucy_deconv_gpu(image: np.ndarray, psf_fft: torch.Tensor, psf_adj_fft: torch.Tensor, 
                               num_iter: int = 10, eps: float = 1e-8):
    img_tensor = torch.from_numpy(image).to(device).float()
    H, W = img_tensor.shape
    deconv = img_tensor.clone()
    
    for i in range(num_iter):
        deconv_fft = torch.fft.rfft2(deconv)
        simulated = torch.fft.irfft2(deconv_fft * psf_fft, s=(H, W))
        relative_blur = img_tensor / (simulated + eps)
        correction = torch.fft.irfft2(torch.fft.rfft2(relative_blur) * psf_adj_fft, s=(H, W))
        deconv = deconv * correction
        deconv = torch.clamp(deconv, min=0.0)
        
        if (i+1) % 10 == 0:
            print(f"RL迭代: {i+1}/{num_iter}")
    return deconv.cpu().numpy()

# ====================== BID迭代次数：10次 ======================
BID_PARAMS = {
    "iterations": 10,
    "tolerance": 0.1,
    "mask": None,
    "use_gpu": False,
    "large_psf": True,
    "pad": True,
    "estimate_background": True,
    "constrain_positive": True
}
PSF_CROP_SIZE = (2048, 2048)
CROP_SIZE = 200

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_manual_or_percentile_clim(img_np, use_manual, manual_vmin, manual_vmax, pmin, pmax):
    img_np = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)
    if use_manual and manual_vmin is not None and manual_vmax is not None:
        return manual_vmin, manual_vmax
    return float(np.percentile(img_np, pmin)), float(np.percentile(img_np, pmax))

def get_diff_clim(diff_np, use_manual, manual_vmin, manual_vmax, vmax_percentile=99.5):
    diff_np = np.nan_to_num(diff_np, nan=0.0, posinf=0.0, neginf=0.0)
    if use_manual and manual_vmin is not None and manual_vmax is not None:
        return manual_vmin, manual_vmax
    vmax = float(np.percentile(np.abs(diff_np), vmax_percentile))
    if vmax == 0:
        vmax = 1.0
    return -vmax, vmax

def compute_power_spectrum(img: np.ndarray):
    img = np.nan_to_num(img, nan=0.0)
    fft = np.fft.fft2(img)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2
    return power

def radial_average(power: np.ndarray):
    y, x = np.indices(power.shape)
    center = np.array(power.shape) / 2
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(np.int32)
    
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    radial_profile = tbin / (nr + 1e-8)
    return radial_profile

# ====================== 【新增】单波段独立PSD绘图函数 ======================
def plot_single_band_psd(band: int, power_data: dict, save_path: str):
    """
    为单个波段绘制独立的PSD（功率谱密度）对比图
    包含Obverse/Proposed/RL/BID四条曲线，对数坐标，高清保存
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=SINGLE_PSD_FIGSIZE, dpi=SINGLE_PSD_DPI)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 功率谱默认对数坐标
    plt.tick_params(axis='both', labelsize=16)
    
    # 绘制四条功率谱曲线
    for label, (radial_ps, freq) in power_data.items():
        style = PROFILE_STYLES[label]
        plt.plot(freq, radial_ps, 
                 color=PROFILE_COLORS[label],
                 linestyle=style['linestyle'],
                 linewidth=style['linewidth'],
                 label=label)
    
    # 图表标注
    plt.title(f'AIA {band}Å | Power Spectral Density (PSD) Comparison', 
              fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Spatial Frequency (Pixel⁻¹)', fontsize=16, fontweight='bold')
    plt.ylabel('Log Power', fontsize=16, fontweight='bold')
    plt.legend(fontsize=14, loc='upper right', frameon=False)
    
    # 保存并关闭
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ {band}Å 独立PSD功率谱图保存完成：{save_path}")
# ============================================================================

# ====================== 【修改版】3×3功率谱汇总函数 ======================
def plot_3x3_power_spectrum_summary(band_power_data, save_path):
    """
    3×3汇总功率谱图：7个波段，335在第三行第二个
    ✅ 每个子图独立图例 | ✅ 波段标签顶部居中
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建3×3子图
    fig, axes = plt.subplots(3, 3, figsize=POWER_SPECTRUM_FIGSIZE, dpi=PROFILE_DPI)
    axes = axes.flatten()  # 展平方便索引
    
    # 隐藏所有子图，后续仅激活有效位置
    for ax in axes:
        ax.set_visible(False)
    
    # 遍历波段，绘制对应位置
    for band, power_data in band_power_data.items():
        if band not in POSITION_MAP:
            continue
        row, col = POSITION_MAP[band]
        ax = axes[row * 3 + col]
        ax.set_visible(True)
        
        # 绘图样式：对数坐标、网格
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=18)
        
        # 绘制四条曲线
        handles = []
        for label, (radial_ps, freq) in power_data.items():
            style = PROFILE_STYLES[label]
            line = ax.plot(freq, radial_ps, 
                     color=PROFILE_COLORS[label],
                     linestyle=style['linestyle'],
                     linewidth=style['linewidth'],
                     label=label)[0]
            handles.append(line)
        
        # ✅ 【修改1】波段标签：顶部中心标注（核心要求）
        ax.text(0.5, 0.98, f'{band}Å', 
                transform=ax.transAxes, 
                va='top', ha='center',  # 水平居中、垂直顶部
                fontsize=16, fontweight='bold',
                color='black')
        
        # ✅ 【修改2】每个子图独立图例（小巧、无框、右上角）
        ax.legend(handles=handles, 
                  fontsize=14,        # 小字体
                  loc='upper right', # 图例位置
                  frameon=False,     # 无边框
                  handlelength=1.5)  # 线型长度缩短
        
        # 统一坐标轴标签（仅底部/左侧子图显示，整洁美观）
        if row == 2:  # 最后一行
            ax.set_xlabel('Spatial Frequency (Pixel⁻¹)', fontsize=14)
        if col == 0:  # 第一列
            ax.set_ylabel('Power', fontsize=14)
    
    # 紧凑布局，保存
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=PROFILE_DPI, facecolor='white')
    plt.close()
    print(f"✅ 3×3全波段功率谱汇总图保存完成：{save_path}")
# ======================================================================================

def auto_detect_flare_region(img_np, zoom_size=200, intensity_percentile=95):
    img = np.nan_to_num(img_np, nan=0.0)
    H, W = img.shape
    
    img_smooth = gaussian_filter(img, sigma=2)
    threshold = np.percentile(img_smooth, intensity_percentile)
    flare_mask = img_smooth > threshold
    
    labeled, num_features = label(flare_mask)
    if num_features == 0:
        center_y, center_x = H // 2, W // 2
    else:
        regions = find_objects(labeled)
        areas = [np.sum(labeled == i+1) for i in range(num_features)]
        max_area_idx = np.argmax(areas)
        region = regions[max_area_idx]
        center_y = (region[0].start + region[0].stop) // 2
        center_x = (region[1].start + region[1].stop) // 2
    
    half_size = zoom_size // 2
    y_start = max(0, center_y - half_size)
    y_end = min(H, center_y + half_size)
    x_start = max(0, center_x - half_size)
    x_end = min(W, center_x + half_size)
    
    if y_end - y_start < zoom_size:
        y_end = y_start + zoom_size
    if x_end - x_start < zoom_size:
        x_end = x_start + zoom_size
    
    return (center_y, center_x), (y_start, y_end, x_start, x_end)

def plot_flare_zoom_comparison(obverse_img, model_img, rl_img, bid_img, band, crop_region, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    y_start, y_end, x_start, x_end = crop_region
    
    obverse_crop = obverse_img[y_start:y_end, x_start:x_end]
    model_crop = model_img[y_start:y_end, x_start:x_end]
    rl_crop = rl_img[y_start:y_end, x_start:x_end]
    bid_crop = bid_img[y_start:y_end, x_start:x_end]

    diff_model = model_crop - obverse_crop
    diff_rl = rl_crop - obverse_crop
    diff_bid = bid_crop - obverse_crop
    diff_zero = obverse_crop - obverse_crop

    vmin, vmax = get_manual_or_percentile_clim(
        obverse_crop, IMAGE_USE_MANUAL_CLIM, IMAGE_MANUAL_VMIN, IMAGE_MANUAL_VMAX,
        IMAGE_VMIN_PERCENTILE, IMAGE_VMAX_PERCENTILE
    )
    if band not in SUPPORTED_WAVELENGTHS:
        band = 94
    cmap = WAVELENGTH_TO_CMAP[band]
    use_log = AIA_OFFICIAL_CONFIG[band][2]
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch() if use_log else LinearStretch())

    all_diffs = np.concatenate([diff_model.flatten(), diff_rl.flatten(), diff_bid.flatten()])
    diff_vmin, diff_vmax = get_diff_clim(all_diffs, DIFF_USE_MANUAL_CLIM, DIFF_MANUAL_VMIN, DIFF_MANUAL_VMAX, DIFF_VMAX_PERCENTILE)

    fig, axes = plt.subplots(2, 4, figsize=(24, 10), dpi=250)
    titles_top = ["Obverse Flare Zoom", "Proposed Flare Zoom", "RL Flare Zoom", "BID Flare Zoom"]
    images_top = [obverse_crop, model_crop, rl_crop, bid_crop]
    titles_bottom = ["Residual", "Proposed - Obverse", "RL - Obverse", "BID - Obverse"]
    images_bottom = [diff_zero, diff_model, diff_rl, diff_bid]

    for j in range(4):
        ax = axes[0, j]
        im = ax.imshow(images_top[j], origin='lower', cmap=cmap, norm=norm)
        ax.set_title(titles_top[j], fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for j in range(4):
        ax = axes[1, j]
        im = ax.imshow(images_bottom[j], origin='lower', cmap='seismic', vmin=diff_vmin, vmax=diff_vmax)
        ax.set_title(titles_bottom[j], fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f'Auto-detected Flare Region | Zoom + Residual | AIA {band}Å', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=250, facecolor='white')
    plt.close(fig)
    print(f"✅ 耀斑局部放大+残差图保存完成")

def get_full_profile(img_np):
    h, w = img_np.shape
    profile_row = h // 2
    profile = img_np[profile_row, :]
    pos = np.arange(w)
    return profile, pos, profile_row

# 修改：剖面绘图新增FWHM标注
def plot_full_profile(profile_data, fwhm_dict, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    obverse_profile, obverse_pos, profile_row = profile_data['Obverse']
    w = len(obverse_pos)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=PROFILE_FIGSIZE, dpi=PROFILE_DPI)

    ax1.set_title(f'Intensity Profile (Row {profile_row}) | FWHM: Obverse={fwhm_dict["Obverse"]:.2f} Proposed={fwhm_dict["Proposed"]:.2f} RL={fwhm_dict["RL"]:.2f} BID={fwhm_dict["BID"]:.2f}', fontsize=18)
    ax1.set_ylabel('Intensity [DN]', fontsize=20, fontweight='bold')
    ax1.set_xlabel('Pixel Position (Left → Right)', fontsize=20, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=18, width=2)

    for label, (profile, pos, _) in profile_data.items():
        style = PROFILE_STYLES[label]
        ax1.plot(pos, profile, 
                 color=PROFILE_COLORS[label],
                 linestyle=style['linestyle'],
                 linewidth=style['linewidth'],
                 label=f"{label} (FWHM={fwhm_dict[label]:.2f})")

    if PROFILE_USE_MANUAL_YLIM:
        ax1.set_ylim(PROFILE_YLIM_MIN, PROFILE_YLIM_MAX)

    zoom_min, zoom_max = MANUAL_ZOOM_START, MANUAL_ZOOM_END
    ax2.set_xlabel('Pixel Position (Zoomed Region)', fontsize=20, fontweight='bold')
    ax2.set_ylabel('Intensity [DN]', fontsize=20, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(zoom_min, zoom_max)
    ax2.tick_params(axis='both', labelsize=18, width=2)

    for label, (profile, pos, _) in profile_data.items():
        style = PROFILE_STYLES[label]
        mask = (pos >= zoom_min) & (pos <= zoom_max)
        ax2.plot(pos[mask], profile[mask], 
                 color=PROFILE_COLORS[label],
                 linestyle=style['linestyle'],
                 linewidth=style['linewidth'],
                 label=f"{label} (FWHM={fwhm_dict[label]:.2f})")

    if PROFILE_USE_MANUAL_YLIM:
        ax2.set_ylim(PROFILE_YLIM_MIN, PROFILE_YLIM_MAX)

    ax1.legend(fontsize=14, loc='upper right')
    ax2.legend(fontsize=14, loc='upper right')
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=PROFILE_DPI, facecolor='white')
    plt.close(fig)
    print(f"✅ 四方法对比剖面（含FWHM标注）保存完成")

def plot_ref_with_profile_line(img_np, band, profile_row, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if band not in SUPPORTED_WAVELENGTHS:
        band = 94

    vmin, vmax = get_manual_or_percentile_clim(
        img_np, REF_USE_MANUAL_CLIM, REF_MANUAL_VMIN, REF_MANUAL_VMAX,
        REF_VMIN_PERCENTILE, REF_VMAX_PERCENTILE
    )

    use_log = AIA_OFFICIAL_CONFIG[band][2]
    cmap = WAVELENGTH_TO_CMAP[band]
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch() if use_log else LinearStretch())

    plt.figure(figsize=(10, 10), dpi=200)
    im = plt.imshow(img_np, cmap=cmap, origin='lower', norm=norm)
    plt.axhline(y=profile_row, color='red', linewidth=4, label=f'Profile Row: {profile_row}')
    plt.colorbar(im, shrink=0.8, pad=0.02)
    plt.title(f'AIA {band}Å Image with Profile Line', fontsize=14, pad=20)
    plt.legend(fontsize=12, loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200, facecolor='white')
    plt.close()
    print(f"✅ Obverse图剖线标记完成")

# ====================== 原有2×4结果图（完全保留） ======================
def plot_four_method_grid(obverse_img, model_img, rl_img, bid_img, band, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    obverse_img = np.nan_to_num(obverse_img, nan=0.0)
    model_img = np.nan_to_num(model_img, nan=0.0)
    rl_img = np.nan_to_num(rl_img, nan=0.0)
    bid_img = np.nan_to_num(bid_img, nan=0.0)

    img_vmin, img_vmax = get_manual_or_percentile_clim(
        obverse_img, IMAGE_USE_MANUAL_CLIM, IMAGE_MANUAL_VMIN, IMAGE_MANUAL_VMAX,
        IMAGE_VMIN_PERCENTILE, IMAGE_VMAX_PERCENTILE
    )

    diff_model = model_img - obverse_img
    diff_rl = rl_img - obverse_img
    diff_bid = bid_img - obverse_img
    diff_zero = obverse_img - obverse_img

    if band not in SUPPORTED_WAVELENGTHS:
        band = 94
    cmap = WAVELENGTH_TO_CMAP[band]
    use_log = AIA_OFFICIAL_CONFIG[band][2]
    norm_top = ImageNormalize(vmin=img_vmin, vmax=img_vmax, stretch=LogStretch() if use_log else LinearStretch())

    fig, axes = plt.subplots(2, 4, figsize=(28, 14), dpi=220)
    titles_top = ["Obverse", "Proposed", "RL", "BID"]
    images_top = [obverse_img, model_img, rl_img, bid_img]
    titles_bottom = ["", "Proposed - Obverse", "RL - Obverse", "BID - Obverse"]
    diffs_bottom = [diff_zero, diff_model, diff_rl, diff_bid]

    for j in range(4):
        ax = axes[0, j]
        im = ax.imshow(images_top[j], origin='lower', cmap=cmap, norm=norm_top)
        ax.set_title(titles_top[j], fontsize=15, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    for j in range(4):
        ax = axes[1, j]
        diff = diffs_bottom[j]
        if j == 0:
            curr_vmin, curr_vmax = -1, 1
        else:
            curr_vmin, curr_vmax = get_diff_clim(diff, DIFF_USE_MANUAL_CLIM, DIFF_MANUAL_VMIN, DIFF_MANUAL_VMAX, DIFF_VMAX_PERCENTILE)
        
        im = ax.imshow(diff, origin='lower', cmap='seismic', vmin=curr_vmin, vmax=curr_vmax)
        ax.set_title(titles_bottom[j], fontsize=15, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f'Four-Method Comparison | AIA {band}Å', fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=220, facecolor='white')
    plt.close(fig)
    print(f"✅ 2×4 可视化保存完成")

# ====================== 【新增】专属PSF退化2×4对比图（单独保存） ======================
def plot_psf_degraded_2x4(obverse_img, model_deg, rl_deg, bid_deg, band, save_path):
    """
    单独绘制：PSF退化结果 vs 原始观测 2行4列对比图
    上排：原始观测 | Proposed退化 | RL退化 | BID退化
    下排：0差分 | 退化-Obverse | 退化-Obverse | 退化-Obverse
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 统一显示范围
    vmin, vmax = get_manual_or_percentile_clim(
        obverse_img, IMAGE_USE_MANUAL_CLIM, IMAGE_MANUAL_VMIN, IMAGE_MANUAL_VMAX,
        IMAGE_VMIN_PERCENTILE, IMAGE_VMAX_PERCENTILE
    )

    # 计算差分
    diff_model = model_deg - obverse_img
    diff_rl = rl_deg - obverse_img
    diff_bid = bid_deg - obverse_img
    diff_zero = np.zeros_like(obverse_img)

    # 波段配色
    if band not in SUPPORTED_WAVELENGTHS:
        band = 94
    cmap = WAVELENGTH_TO_CMAP[band]
    use_log = AIA_OFFICIAL_CONFIG[band][2]
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch() if use_log else LinearStretch())

    # 统一差分色标
    all_diffs = np.concatenate([diff_model.flatten(), diff_rl.flatten(), diff_bid.flatten()])
    diff_vmin, diff_vmax = get_diff_clim(all_diffs, DIFF_USE_MANUAL_CLIM, DIFF_MANUAL_VMIN, DIFF_MANUAL_VMAX, DIFF_VMAX_PERCENTILE)

    # 2行4列绘图
    fig, axes = plt.subplots(2, 4, figsize=(28, 14), dpi=220)
    
    # 上排：原始 + 3个PSF退化结果
    titles_top = ["Obverse(Observed)", "Proposed PSF Degraded", "RL PSF Degraded", "BID PSF Degraded"]
    images_top = [obverse_img, model_deg, rl_deg, bid_deg]
    
    # 下排：差分图
    titles_bottom = ["", "ProposedDeg - Obverse", "RLDeg - Obverse", "BIDeg - Obverse"]
    images_bottom = [diff_zero, diff_model, diff_rl, diff_bid]

    # 绘制上排
    for j in range(4):
        ax = axes[0, j]
        im = ax.imshow(images_top[j], origin='lower', cmap=cmap, norm=norm)
        ax.set_title(titles_top[j], fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    # 绘制下排差分
    for j in range(4):
        ax = axes[1, j]
        im = ax.imshow(images_bottom[j], origin='lower', cmap='seismic', vmin=diff_vmin, vmax=diff_vmax)
        ax.set_title(titles_bottom[j], fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(f'PSF Degraded Results vs Original Observation | AIA {band}Å', fontsize=20, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches='tight', dpi=220, facecolor='white')
    plt.close(fig)
    print(f"✅ PSF退化专属2×4对比图单独保存完成")
# ======================================================================================

def preprocess_psf_for_bid(wavelength):
    psf_path = aia_band_params[str(wavelength)]["PSF_FILE"]
    with fits.open(psf_path) as psf_hdul:
        psf = psf_hdul[0].data
    psf = np.squeeze(psf)
    psf_h, psf_w = psf.shape
    start_h = (psf_h - PSF_CROP_SIZE[0]) // 2
    end_h = start_h + PSF_CROP_SIZE[0]
    start_w = (psf_w - PSF_CROP_SIZE[1]) // 2
    end_w = start_w + PSF_CROP_SIZE[1]
    psf_cropped = psf[start_h:end_h, start_w:end_w]
    psf_norm = psf_cropped / psf_cropped.sum()
    psf_shifted = np.fft.fftshift(psf_norm)
    psf_final = np.fft.ifftshift(psf_shifted)
    return psf_final

def run_bid_deconv(img_blurred, wavelength):
    psf = preprocess_psf_for_bid(wavelength)
    img_blurred = np.nan_to_num(img_blurred, nan=0.0)
    img_deconvolved = deconvolve_bid(img=img_blurred, psf=psf,** BID_PARAMS)
    img_deconvolved = np.nan_to_num(img_deconvolved, nan=0.0)
    return img_deconvolved

class SimpleTestDataset(Dataset):
    def __init__(self, data_dir):
        all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                           if f.endswith(('.fits', '.fits.gz'))])
        if MAX_FILES_PER_BAND is not None:
            self.files = all_files[:MAX_FILES_PER_BAND]
        else:
            self.files = all_files
        print(f"✅ 加载测试图：{len(self.files)} 张")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        with fits.open(file) as hdul:
            img = None
            for hdu in hdul:
                if hdu.data is not None:
                    img = hdu.data.astype(np.float32)
                    break
            if img is None:
                raise ValueError(f"文件 {file} 无有效数据")
        img = torch.from_numpy(img).unsqueeze(0)
        return {"y1": img, "y2": img}

@torch.no_grad()
def test_single_band(band, cfg):
    cfg.BASE.CURRENT_BAND = band

    band_output_dir = os.path.join(CUSTOM_OUTPUT_ROOT, f"band_{band}")
    os.makedirs(band_output_dir, exist_ok=True)

    data_dir = os.path.join(TEST_DATA_ROOT, f"{band:04d}-HR")

    print(f"\n{'=' * 60}")
    print(f"🔍 测试波段: {band}Å")
    test_set = SimpleTestDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    model = PhysicsAwareUNet(cfg).to(device)
    model_path = os.path.join(MODEL_ROOT, f"band_{band}", f"band_{band}_blur2blur", "models", "best_model.pth")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 初始化PSF退化器
    psf_degrader = PSFDegradation(psf_file=aia_band_params[str(band)]["PSF_FILE"])

    vis_done = False
    for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"测试中")):
        y1 = batch['y1'].to(device)
        x1_pred = model(y1)

        if not vis_done:
            input_img = y1[0, 0].cpu().numpy()
            model_img = x1_pred[0, 0].cpu().numpy()

            # 1. RL反卷积
            psf_fft, psf_adj_fft, _ = load_and_preprocess_psf(str(band), target_size=input_img.shape)
            rl_img = richardson_lucy_deconv_gpu(input_img, psf_fft, psf_adj_fft)

            # 2. BID反卷积
            bid_img = run_bid_deconv(input_img, band)

            # ====================== 核心：对3个方法分别做PSF退化 ======================
            print("🔄 对Proposed/RL/BID执行PSF退化...")
            model_degraded = psf_degrader.degrade_numpy(model_img)  # 模型输出退化
            rl_degraded = psf_degrader.degrade_numpy(rl_img)        # RL结果退化
            bid_degraded = psf_degrader.degrade_numpy(bid_img)      # BID结果退化
            print("✅ 全部PSF退化完成")
            # ========================================================================

            # 原有剖面数据
            profile_data = {
                'Obverse': get_full_profile(input_img),
                'Proposed': get_full_profile(model_img),
                'RL': get_full_profile(rl_img),
                'BID': get_full_profile(bid_img)
            }

            # 耀斑检测
            flare_center, flare_crop_region = auto_detect_flare_region(input_img)
            y_start, y_end, x_start, x_end = flare_crop_region
            obverse_local = input_img[y_start:y_end, x_start:x_end]
            model_local = model_img[y_start:y_end, x_start:x_end]
            rl_local = rl_img[y_start:y_end, x_start:x_end]
            bid_local = bid_img[y_start:y_end, x_start:x_end]

            # ====================== 新增：计算FWHM指标 ======================
            print("🔢 计算耀斑区域FWHM分辨率指标...")
            # 提取耀斑中心行剖面
            flare_y, flare_x = flare_center
            obverse_flare_profile = input_img[flare_y, :]
            model_flare_profile = model_img[flare_y, :]
            rl_flare_profile = rl_img[flare_y, :]
            bid_flare_profile = bid_img[flare_y, :]

            # 计算FWHM
            fwhm_dict = {
                'Obverse': calculate_fwhm(obverse_flare_profile),
                'Proposed': calculate_fwhm(model_flare_profile),
                'RL': calculate_fwhm(rl_flare_profile),
                'BID': calculate_fwhm(bid_flare_profile)
            }
            # 保存指标到CSV
            save_fwhm_metrics(band, batch_idx, fwhm_dict, band_output_dir)
            # ====================================================================

            # 功率谱：收集数据到全局汇总字典（不再单独绘图）
            power_data = {}
            max_freq = min(obverse_local.shape) // 2
            freq = np.arange(max_freq)
            for name, img_local in zip(['Obverse', 'Proposed', 'RL', 'BID'],
                                [obverse_local, model_local, rl_local, bid_local]):
                ps = compute_power_spectrum(img_local)
                power_data[name] = (radial_average(ps)[:max_freq], freq)
            band_power_summary[band] = power_data  # 存入全局字典

            # ====================== 【新增】保存单波段独立PSD图 ======================
            single_psd_save_path = os.path.join(band_output_dir, f"SINGLE_BAND_PSD.png")
            plot_single_band_psd(band, power_data, single_psd_save_path)
            # ========================================================================

            # ====================== 保存所有结果 ======================
            # 1. 原有2×4结果图
            grid_save_path = os.path.join(band_output_dir, f"FOUR_METHOD_COMPARE_GRID.png")
            plot_four_method_grid(input_img, model_img, rl_img, bid_img, band, grid_save_path)

            # 2. 【新增】单独保存PSF退化2×4对比图
            psf_save_path = os.path.join(band_output_dir, f"PSF_DEGRADED_COMPARE_GRID.png")
            plot_psf_degraded_2x4(input_img, model_degraded, rl_degraded, bid_degraded, band, psf_save_path)

            # 其余可视化
            profile_save_path = os.path.join(band_output_dir, f"FOUR_METHOD_COMPARE_PROFILE.png")
            plot_full_profile(profile_data, fwhm_dict, profile_save_path)
            
            ref_img_save_path = os.path.join(band_output_dir, f"OBVERSE_IMAGE_WITH_PROFILE_LINE.png")
            plot_ref_with_profile_line(input_img, band, profile_data['Obverse'][2], ref_img_save_path)
            
            flare_zoom_save_path = os.path.join(band_output_dir, f"FLARE_ZOOM_COMPARISON.png")
            plot_flare_zoom_comparison(input_img, model_img, rl_img, bid_img, band, flare_crop_region, flare_zoom_save_path)

            vis_done = True

    print(f"🏁 {band}Å 测试完成！")

def main():
    set_seed()
    cfg = get_config()
    global band_power_summary
    band_power_summary = {}  # 初始化全局功率谱数据字典

    print("\n" + "=" * 70)
    print("🚀 开始测试（新增FWHM分辨率指标 + PSF退化对比图 + 单波段PSD + 3×3功率谱汇总）")
    print("=" * 70)

    for i, band in enumerate(BANDS):
        print(f"\n📋 进度 {i + 1}/{len(BANDS)}")
        test_single_band(band, cfg)

    # ✅ 所有波段处理完成后，绘制3×3功率谱汇总图
    summary_save_path = os.path.join(CUSTOM_OUTPUT_ROOT, "ALL_BANDS_POWER_SPECTRUM_3x3_SUMMARY.png")
    plot_3x3_power_spectrum_summary(band_power_summary, summary_save_path)

    # ✅ 新增：汇总所有波段FWHM表格
    summarize_fwhm_all_bands(CUSTOM_OUTPUT_ROOT)

    print("\n🎉 全部测试完成！")

if __name__ == "__main__":
    main()