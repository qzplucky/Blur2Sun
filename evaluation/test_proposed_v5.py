import os
import random
import warnings
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from astropy.io import fits
import astropy.units as u
from sunpy.map import Map
from astropy.visualization import ImageNormalize, SqrtStretch
import matplotlib.colors as colors

# v5: 模型经 v5_common 加载（内部即修正 PSF 算子）
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'paper_tables_v5'))
from v5_common import load_v5_model

# Hand-set per-channel calibration constants. Applied ONLY to count-depleted
# short-exposure flare frames (set BLUR2SUN_SHORT_EXPOSURE=1); standard-exposure
# frames are deconvolved at native intensity (paper Section 4.4).
INPUT_OFFSET = {'94': 2.0, '131': 2.0, '171': 85.0, '193': 4.0,
                '211': 85.0, '304': 4.0, '335': 4.0}
SHORT_EXPOSURE = os.environ.get('BLUR2SUN_SHORT_EXPOSURE', '0') == '1'

# ====================== 【全局配置】仅修改这里 ======================
os.environ.setdefault("JSOC_EMAIL", "your_registered_email@example.com")
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')  # 服务器无界面绘图

TEST_DATA_ROOT = os.environ.get("BLUR2SUN_TEST", os.path.join(os.environ.get("BLUR2SUN_DATA", ""), "test", "test_classified")) + os.sep
MODEL_ROOT = os.environ.get("BLUR2SUN_MODELS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs"))
CUSTOM_OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "grid_7band_v5") + os.sep

# 降采样配置
TARGET_SIZE = 1024
SOURCE_SIZE = 4096

# 处理文件数量（None=全部）
MAX_FILES_PER_BAND = 1

# 可视化配置（紧凑布局，适配4列）
FONT_SIZE = 14
FIG_SIZE = (20, 24)  # 微调宽度适配第四列色标
DPI = 200

# AIA官方可视化配置（保留核心）
WAVELENGTH_TO_CMAP = {
    94: 'sdoaia94', 131: 'sdoaia131', 171: 'sdoaia171',
    193: 'sdoaia193', 211: 'sdoaia211', 304: 'sdoaia304', 335: 'sdoaia335'
}
AIA_OFFICIAL_CONFIG = {
    94: (0, 99.5, False),131: (0, 99.5, False),171: (0, 99.5, False),
    193: (1, 99.5, False),211: (1, 99.5, False),304: (0, 99.5, False),335: (0, 99.5, False)
}
SUPPORTED_WAVELANDS = list(AIA_OFFICIAL_CONFIG.keys())

# 设备配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====================== 适配新目录结构的波段识别 ======================
def get_available_bands(data_root):
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"数据根目录不存在: {data_root}")
    
    # 匹配格式: 0094-HR、0131-HR 等文件夹
    band_folders = [
        f for f in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, f)) 
        and f.endswith("-HR")
    ]
    
    bands = []
    for folder in band_folders:
        # 提取波段数字: 从 0094-HR → 0094 → 94
        band_str = folder.split("-")[0]
        try:
            band = int(band_str)
            bands.append(band)
        except ValueError:
            print(f"⚠️ 跳过无效波段文件夹: {folder}")
            continue
    
    if not bands:
        raise ValueError(f"在 {data_root} 下未找到有效波段文件夹（格式: XXXX-HR）")
    
    return sorted(bands)

BANDS = get_available_bands(TEST_DATA_ROOT)
print(f"✅ 自动识别波段: {BANDS}")

# ====================== 固定随机种子 ======================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)         
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ====================== 工具函数 ======================
def get_image_clim(img_np, pmin=1.0, pmax=99.5):
    """获取图像显示范围"""
    img_np = np.nan_to_num(img_np, 0)
    return float(np.percentile(img_np, pmin)), float(np.percentile(img_np, pmax))

def calculate_percentage_difference(original, predicted, eps=1.0):
    """计算相对原图的百分比差分: (Predicted - Original) / (Original + eps) * 100

    eps 防止暗区噪声导致分母为 0 时百分比爆炸。
    对于 AIA 典型 DN 范围 (~0.1~10000)，eps=1.0 有效抑制暗区伪影。
    """
    original = np.nan_to_num(original, 0.0)
    predicted = np.nan_to_num(predicted, 0.0)
    # 相对原图的百分比变化
    return (predicted - original) / (original + eps) * 100.0

# ====================== ✅ 修复版：图像与色标完全分离，第四列独立色标 ======================
def plot_7x3_visualization(band_data_list, save_path):
    """ 
    绘制7行4列可视化图
    ✅ 列1：原图 | 列2：预测图 | 列3：seismic普通差分图 | 列4：独立色标（每行专属）
    ✅ 图像与色标完全分离，无绑定，彻底解决列间距畸变问题
    ✅ 行列间距严格相等，极致紧凑布局
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_bands = len(band_data_list)

    # 核心修改：7行4列 | width_ratios 控制列宽（第四列仅放色标，极窄）
    fig, axes = plt.subplots(
        n_bands, 4, 
        figsize=FIG_SIZE, 
        dpi=DPI,
        gridspec_kw={
            'hspace': 0.01,   # 行间距
            'wspace': 0.01,   # 列间距（严格相等）
            'width_ratios': [1, 1, 1, 0.05]  # 前三列图像等宽，第四列窄色标 间距核心参数
        }
    )

    for row_idx, data in enumerate(band_data_list):
        band = data['band']
        orig = data['original']
        pred = data['predicted']
        
        # 计算相对原图的百分比差分图像
        diff_img = calculate_percentage_difference(orig, pred)

        # 1. 原图配置
        cmap = WAVELENGTH_TO_CMAP.get(band, 'sdoaia171')
        use_log = AIA_OFFICIAL_CONFIG[band][2]
        vmin, vmax = get_image_clim(orig)
        # 审稿规范：太阳图像统一用低/高分位截断的平方根拉伸
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

        # 列1：Original
        ax1 = axes[row_idx, 0]
        ax1.imshow(orig, origin='lower', cmap=cmap, norm=norm)
        ax1.text(0.02, 0.98, f'{band}Å', transform=ax1.transAxes,
                 fontsize=FONT_SIZE+2, fontweight='bold', color='white',
                 va='top', ha='left')
        ax1.axis('off')

        # 列2：Predicted
        ax2 = axes[row_idx, 1]
        ax2.imshow(pred, origin='lower', cmap=cmap, norm=norm)
        ax2.axis('off')

        # 动态计算百分比差分图的 1% 到 99.5% 分位数范围
        diff_vmin, diff_vmax = get_image_clim(diff_img, pmin=1.0, pmax=99.5)

        # 对称范围，保证 0% 对应白色中心
        abs_max = max(abs(diff_vmin), abs(diff_vmax))
        if abs_max == 0: abs_max = 1e-6
        # 夹到 ±100%：保留自适应（弱信号通道按自身分位），
        # 但防止 304 等暗区极端像素把色标撑到 ±150% 而压扁主体信息
        abs_max = min(abs_max, 100.0)

        # 列3：百分比差分图 (seismic: 蓝=负=暗于原图, 红=正=亮于原图)
        ax3 = axes[row_idx, 2]
        im3 = ax3.imshow(diff_img, origin='lower', cmap='seismic', vmin=-abs_max, vmax=abs_max)
        ax3.axis('off')

        # 列4：独立色标（完全分离，专属当前行）
        cbar_ax = axes[row_idx, 3]
        cbar = plt.colorbar(
            im3, 
            cax=cbar_ax,
            shrink=0.15,    
            pad=0.01,    
            aspect=1    
        )
        cbar.set_label('Diff [%]', fontsize=FONT_SIZE, labelpad=2)
        cbar.ax.tick_params(labelsize=FONT_SIZE, length=2)
        cbar.outline.set_visible(False)

        # 335Å 为最后一行：删除 colorbar 端点（±abs_max，即 ±100）的刻度数字
        if band == 335:
            ticks = cbar.ax.get_yticks()
            inner = [t for t in ticks if abs(t) < abs_max - 1e-6]
            cbar.ax.set_yticks(inner)

    # 紧凑布局，适配4列结构
    plt.subplots_adjust(left=0.01, right=0.45, top=0.85, bottom=0.01)  #色标卡导致间距出现问题的核心参数
    plt.savefig(save_path, bbox_inches='tight', facecolor='white', dpi=DPI)
    plt.close()
    print(f"✅ 7×4 可视化图保存完成 (差分改为原图百分比 + seismic色标): {save_path}")

# ====================== 数据集 ======================
class SimpleTestDataset(Dataset):
    def __init__(self, data_dir):
        all_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir)
                           if f.endswith(('.fits', '.fits.gz'))])
        self.files = all_files[:MAX_FILES_PER_BAND] if MAX_FILES_PER_BAND else all_files
        print(f"加载图像: {len(self.files)} 张")

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
        img = torch.from_numpy(img).unsqueeze(0)
        return {"y1": img}

# ====================== 适配新目录的预处理路径 ======================
def process_all_bands_l1_to_l15():
    print("🚀 开始数据预处理: L1 → 1024×1024")
    for band in BANDS:
        band_str = f"{band:04d}"
        input_dir = os.path.join(TEST_DATA_ROOT, f"{band_str}-HR")
        output_dir = os.path.join(input_dir, "level1_1024")
        os.makedirs(output_dir, exist_ok=True)

        fits_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".fits")])
        fits_files = fits_files[:MAX_FILES_PER_BAND] if MAX_FILES_PER_BAND else fits_files
        
        for file in tqdm(fits_files, desc=f"{band}Å 预处理"):
            out_path = os.path.join(output_dir, file.replace(".fits", "_1024.fits"))
            if os.path.exists(out_path):
                continue
            try:
                m = Map(os.path.join(input_dir, file))
                m_1024 = m.resample((TARGET_SIZE, TARGET_SIZE)*u.pixel, method='nearest')
                m_1024.save(out_path, overwrite=True)
            except Exception as e:
                print(f"处理失败 {file}: {str(e)}")
    print("✅ 所有波段预处理完成")

# ====================== 单波段模型推理 ======================
@torch.no_grad()
def infer_single_band(band):
    """仅执行模型推理，返回原图和预测图"""
    band_str = f"{band:04d}"
    data_dir = os.path.join(TEST_DATA_ROOT, f"{band_str}-HR/level1_1024")
    test_set = SimpleTestDataset(data_dir)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # v5 模型（修正 PSF 算子）；原生输入，输出扣宁静环带台 → clip
    model = load_v5_model(band)

    # 推理
    for batch in test_loader:
        y1 = batch['y1'][0, 0].numpy()
        offset = INPUT_OFFSET.get(str(band), 0.0) if SHORT_EXPOSURE else 0.0
        t = torch.from_numpy(np.ascontiguousarray(y1 + offset)).float().to(device)[None, None]
        with torch.no_grad():
            x = model(t)[0, 0].cpu().numpy() - offset
        H, W = x.shape
        yy, xx = np.mgrid[0:H, 0:W]
        ring = np.hypot(yy - H / 2, xx - W / 2) > 450
        x = np.clip(x - float(np.median(x[ring])), 0, None)
        orig_img = y1
        pred_img = x
        break

    print(f"✅ {band}Å 推理完成")
    return orig_img, pred_img

# ====================== 主函数 ======================
def main():
    set_seed()
    os.makedirs(CUSTOM_OUTPUT_ROOT, exist_ok=True)
    
    # 1. 预处理所有数据
    process_all_bands_l1_to_l15()
    
    # 2. 批量推理所有波段
    print("\n🚀 开始模型推理...")
    band_data_list = []
    for band in BANDS:
        orig, pred = infer_single_band(band)
        band_data_list.append({
            "band": band,
            "original": orig,
            "predicted": pred
        })
    
    # 3. 绘制可视化图
    save_path = os.path.join(CUSTOM_OUTPUT_ROOT, "AIA_7BANDS_7x4_VISUALIZATION.png")
    plot_7x3_visualization(band_data_list, save_path)
    
    print("\n🎉 全部任务完成！")

if __name__ == "__main__":
    main()