import os
import copy
import random
import warnings
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.io import fits
from astropy.visualization import ImageNormalize, LogStretch, LinearStretch
import sunpy.map

# 项目配置
from config import get_config
from model import PhysicsAwareIterativeNet

# ============================================================
# 全局配置
# ============================================================
warnings.filterwarnings('ignore')
plt.switch_backend('Agg')

TEST_DATA_ROOT = "./data/test/test_classified/"
MODEL_ROOT = "./outputs"
CUSTOM_OUTPUT_ROOT = "./outputs/band_94/ablation_test"

TARGET_BAND = 94
MAX_PAIRS = 15  # 最多测试 15 对相邻帧

FONT_SIZE = 14
DPI = 200
DIFF_PERCENTILE = 99
PATCH_SIZE = 256

WAVELENGTH_TO_CMAP = {94: 'sdoaia94'}
AIA_OFFICIAL_CONFIG = {94: (0, 99.5, True)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 选择要测试的消融模型组
# "T"     → T 迭代次数消融 (T=5 vs T=3)
# "step"  → 步长学习消融 (可学习 vs 固定)
# "module" → 原有模块消融 (Full vs No Detail)
# ============================================================
TEST_GROUP = "T"  # 改为 "step" 或 "module" 切换消融测试组

GROUP_MODEL_KEYS = {
    "T":     ["T=5 (Learnable Step)", "T=3 (Learnable Step)"],
    "step":  ["T=5 (Learnable Step)", "T=5 (Fixed Step)"],
    "module": ["Full (PSF+Detail)", "No Detail (PSF only)"],
}
ABLATION_MODELS = {
    # ===== 原有模块消融 =====
    "Full (PSF+Detail)": {
        "dir": "ab_full",
        "desc": "Full Model (PSF + Detail)",
        "flags": {"USE_PSF": True, "USE_DETAIL_BLOCK": True},
    },
    "No PSF (Detail only)": {
        "dir": "ab_no_psf",
        "desc": "No PSF (Detail only)",
        "flags": {"USE_PSF": False, "USE_DETAIL_BLOCK": True},
    },
    "No Detail (PSF only)": {
        "dir": "ab_no_detail",
        "desc": "No Detail (PSF only)",
        "flags": {"USE_PSF": True, "USE_DETAIL_BLOCK": False},
    },
    "Pure Iterative": {
        "dir": "ab_no_psf_detail",
        "desc": "No PSF + No Detail (Pure iterative)",
        "flags": {"USE_PSF": False, "USE_DETAIL_BLOCK": False},
    },

    # ===== T 数量消融（可学习步长, Full Model） =====
    "T=5 (Learnable Step)": {
        "dir": "ab_t5_full",
        "desc": "T=5 Full Model (Learnable Step)",
        "flags": {"USE_PSF": True, "USE_DETAIL_BLOCK": True,
                  "NUM_BID_ITER": 5, "LEARNABLE_STEP": True},
    },
    "T=3 (Learnable Step)": {
        "dir": "ab_t3_full",
        "desc": "T=3 Full Model (Learnable Step)",
        "flags": {"USE_PSF": True, "USE_DETAIL_BLOCK": True,
                  "NUM_BID_ITER": 3, "LEARNABLE_STEP": True},
    },

    # ===== 步长消融（T=5, Full Model） =====
    "T=5 (Fixed Step)": {
        "dir": "ab_t5_fixed",
        "desc": "T=5 Fixed Step",
        "flags": {"USE_PSF": True, "USE_DETAIL_BLOCK": True,
                  "NUM_BID_ITER": 5, "LEARNABLE_STEP": False},
    },
}

# ============================================================
# 基础工具
# ============================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def auto_detect_flare_roi(img_np, patch_size=256):
    """在图像中自动定位最亮的 patch 区域（耀斑活动区）"""
    img_safe = np.nan_to_num(img_np, nan=0.0, posinf=0.0, neginf=0.0)
    h, w = img_safe.shape
    step = 16
    max_sum = -1
    best_y, best_x = h // 2, w // 2
    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            current_sum = np.sum(img_safe[y:y + patch_size, x:x + patch_size])
            if current_sum > max_sum:
                max_sum = current_sum
                best_y, best_x = y, x
    return (best_y, best_y + patch_size, best_x, best_x + patch_size)


def compute_1d_psd(img_np):
    """计算 1D 径向功率谱密度"""
    img_np = np.nan_to_num(img_np, nan=0.0)
    h, w = img_np.shape
    f_shift = np.fft.fftshift(np.fft.fft2(img_np))
    psd2d = np.abs(f_shift) ** 2
    y, x = np.indices(psd2d.shape)
    center = (h // 2, w // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(int)
    tbin = np.bincount(r.ravel(), psd2d.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / np.maximum(nr, 1)
    return radialprofile[:min(h, w) // 2]


def get_image_clim(img_np, pmin=1.0, pmax=99.5):
    img_np = np.nan_to_num(img_np, nan=0.0)
    return float(np.percentile(img_np, pmin)), float(np.percentile(img_np, pmax))


def robust_data_range(ref_np, pmin=1.0, pmax=99.5):
    ref_np = np.nan_to_num(ref_np, nan=0.0)
    return max(float(np.percentile(ref_np, pmax) - np.percentile(ref_np, pmin)), 1e-6)


# ============================================================
# 无真值场景下的逆一致性指标
# 核心思路：将去卷积结果 x̂ 退化回 PSF(x̂)，与原始观测 y 比较
# - PSNR(ŷ_degraded, y): 退化预测与观测的匹配度（越高越好）
# - ResMean: mean(|PSF(x̂) - y|)，平均残差幅度（越低越好）
# - ResStd:  std(|PSF(x̂) - y|)，残差波动幅度（越低越稳定）
# - FluxErr: |sum(PSF(x̂)) - sum(y)| / sum(y)，通量守恒（越低越好）
# ============================================================
def compute_inverse_consistency_metrics(x_pred_np, y_obs_np, psf_module):
    """
    无真值下的逆一致性评价：
    - PSNR: 退化后预测 vs 观测的峰值信噪比
    - ResMean: 残差绝对值的均值
    - ResStd: 残差绝对值的标准差
    - FluxErr: 相对通量误差
    """
    # 退化预测
    pred_t = torch.from_numpy(x_pred_np).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        pred_degraded = psf_module(pred_t).squeeze().cpu().numpy()
    pred_degraded = np.nan_to_num(pred_degraded, nan=0.0)
    y_obs = np.nan_to_num(y_obs_np, nan=0.0)

    # 残差
    residual = pred_degraded - y_obs
    abs_res = np.abs(residual)

    # PSNR
    dr = robust_data_range(y_obs)
    mse = np.mean(residual ** 2)
    psnr = float(20.0 * np.log10(dr) - 10.0 * np.log10(max(mse, 1e-12)))

    # ResMean / ResStd
    res_mean = float(np.mean(abs_res))
    res_std = float(np.std(abs_res))

    # FluxErr
    flux_pred = np.sum(pred_degraded)
    flux_obs = np.sum(y_obs)
    flux_err = float(abs(flux_pred - flux_obs) / max(abs(flux_obs), 1e-8))

    return {
        "psnr": psnr,
        "res_mean": res_mean,
        "res_std": res_std,
        "flux_err": flux_err,
    }


# ============================================================
# 测试数据加载：从 test_classified 目录加载相邻帧对
# 相邻帧对 ≈ 同一太阳场景 + 不同噪声实例（符合 Blur2Blur 假设）
# ============================================================
def load_test_pairs(target_band=94, max_pairs=15):
    band_str = f"{target_band:04d}"
    data_dir = os.path.join(TEST_DATA_ROOT, f"{band_str}-HR/")
    all_files = sorted([os.path.join(data_dir, f)
                        for f in os.listdir(data_dir)
                        if f.endswith('.fits')])
    if len(all_files) < 2:
        raise FileNotFoundError(f"测试数据不足: {data_dir} (需要至少 2 帧)")

    # 生成相邻帧对
    pairs = []
    for i in range(len(all_files) - 1):
        pairs.append((all_files[i], all_files[i + 1]))
    pairs = pairs[:max_pairs]

    print(f"📁 测试数据: {data_dir}")
    print(f"  总帧数: {len(all_files)}, 相邻帧对: {len(pairs)}")
    return pairs


def read_fits_image(file_path):
    with fits.open(file_path) as hdul:
        img = next((hdu.data.astype(np.float32)
                     for hdu in hdul if hdu.data is not None), None)
    if img is None:
        raise ValueError(f"无效 FITS: {file_path}")
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    return img


# ============================================================
# 模型初始化
# ============================================================
def init_model(model_dir, model_flags, cfg):
    """根据消融标志初始化 PhysicsAwareIterativeNet 并加载权重"""
    cfg_copy = copy.deepcopy(cfg)
    cfg_copy.BASE.CURRENT_BAND = str(TARGET_BAND)
    cfg_copy.BASE.EXP_NAME = model_dir

    # 应用消融标志
    cfg_copy.MODEL.USE_PSF = model_flags["USE_PSF"]
    cfg_copy.MODEL.USE_DETAIL_BLOCK = model_flags["USE_DETAIL_BLOCK"]
    cfg_copy.MODEL.USE_PRIOR_BLOCK = model_flags["USE_DETAIL_BLOCK"]
    cfg_copy.MODEL.USE_ITER = True
    cfg_copy.MODEL.ADAPTIVE_ITER = False

    # T 迭代次数
    if "NUM_BID_ITER" in model_flags:
        cfg_copy.MODEL.NUM_BID_ITER = model_flags["NUM_BID_ITER"]

    # 步长可学习开关
    if "LEARNABLE_STEP" in model_flags:
        cfg_copy.MODEL.LEARNABLE_STEP_SIZE = model_flags["LEARNABLE_STEP"]

    model_path = os.path.join(MODEL_ROOT, f"band_{TARGET_BAND}", model_dir,
                              "models", "best_model.pth")
    model = PhysicsAwareIterativeNet(cfg_copy).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model


def write_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 可视化
# ============================================================
def plot_ablation_comparison(y1_img, y2_img, pred_dict, zoom_roi, save_path):
    """
    无真值对比可视化:
    Row 0: y₁ (输入) | y₂ (目标) | 各模型 x̂_pred (去卷积结果)
    Row 1: 各模型 x̂_pred 的局部 Zoom
    Row 2: 残差 |PSF(x̂) - y₂| 全图
    Row 3: 残差 |PSF(x̂) - y₂| 局部 Zoom
    """
    model_names = list(pred_dict.keys())
    n_models = len(model_names)
    roi_y1, roi_y2, roi_x1, roi_x2 = zoom_roi

    fig, axes = plt.subplots(4, n_models + 2, figsize=(4 * (n_models + 2), 16),
                             dpi=DPI, gridspec_kw={'wspace': 0.02, 'hspace': 0.15})
    cmap = WAVELENGTH_TO_CMAP[TARGET_BAND]
    vmin, vmax = get_image_clim(y1_img)
    norm = ImageNormalize(vmin=vmin, vmax=vmax,
                          stretch=LogStretch() if AIA_OFFICIAL_CONFIG[TARGET_BAND][2]
                          else LinearStretch())
    diff_cmap = plt.get_cmap('inferno')

    col_labels = ['y1 (Input)', 'y2 (Target)'] + model_names
    # 前两列是 y1, y2；后 n_models 列是各模型的 x̂
    image_list = [y1_img, y2_img] + [pred_dict[n]["x_pred"] for n in model_names]
    # 残差：y1 列放 None，y2 列放 |y1 - y2|（噪声参考），后面是各模型的 |PSF(x̂) - y₂|
    residual_list = [None,
                     np.abs(y1_img - y2_img),
                     ] + [pred_dict[n]["residual"] for n in model_names]

    last_abs_max = 1.0
    for col in range(len(col_labels)):
        # Row 0: 全图
        axes[0, col].imshow(image_list[col], origin='lower', cmap=cmap, norm=norm)
        axes[0, col].add_patch(plt.Rectangle(
            (roi_x1, roi_y1), roi_x2 - roi_x1, roi_y2 - roi_y1,
            edgecolor='red', linewidth=3, fill=False))
        axes[0, col].set_title(col_labels[col], fontsize=FONT_SIZE - 1, fontweight='bold')

        # Row 1: 局部 Zoom
        zoom_img = image_list[col][roi_y1:roi_y2, roi_x1:roi_x2]
        axes[1, col].imshow(zoom_img, origin='lower', cmap=cmap, norm=norm)
        axes[1, col].set_title('Local Zoom', fontsize=FONT_SIZE - 2)

        # Row 2: 残差全图
        if residual_list[col] is not None:
            res_img = residual_list[col]
            valid = res_img[~np.isnan(res_img)]
            abs_max = np.percentile(np.abs(valid), DIFF_PERCENTILE) if len(valid) else 1.0
            axes[2, col].imshow(res_img, origin='lower', cmap=diff_cmap,
                                norm=ImageNormalize(vmin=0, vmax=abs_max, stretch=LinearStretch()))
            if col == 1:
                axes[2, col].set_title('|y1 - y2| (Noise Ref)', fontsize=FONT_SIZE - 2)
            else:
                axes[2, col].set_title(f'|PSF(x̂) - y2|', fontsize=FONT_SIZE - 2)
        else:
            axes[2, col].axis('off')

        # Row 3: 残差局部 Zoom
        if residual_list[col] is not None:
            local_res = residual_list[col][roi_y1:roi_y2, roi_x1:roi_x2]
            valid_l = local_res[~np.isnan(local_res)]
            abs_max_l = np.percentile(np.abs(valid_l), DIFF_PERCENTILE) if len(valid_l) else 1.0
            last_abs_max = abs_max_l
            axes[3, col].imshow(local_res, origin='lower', cmap=diff_cmap,
                                norm=ImageNormalize(vmin=0, vmax=abs_max_l, stretch=LinearStretch()))
            if col == 1:
                axes[3, col].set_title('Local |y1 - y2|', fontsize=FONT_SIZE - 2)
            else:
                axes[3, col].set_title('Local |PSF(x̂) - y2|', fontsize=FONT_SIZE - 2)
        else:
            axes[3, col].axis('off')

        for row in range(4):
            axes[row, col].axis('off')

    cbar_ax = fig.add_axes([0.94, 0.15, 0.012, 0.5])
    cbar = plt.colorbar(plt.cm.ScalarMappable(
        norm=ImageNormalize(vmin=0, vmax=last_abs_max), cmap=diff_cmap), cax=cbar_ax)
    cbar.set_label('|Residual| (DN)', fontsize=FONT_SIZE)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_combined_psd(y1_img, y2_img, pred_dict, zoom_roi, save_path):
    """PSD 对比：y1, y2, 各模型去卷积结果"""
    roi_y1, roi_y2, roi_x1, roi_x2 = zoom_roi

    slices = {
        'y1 (Input)': y1_img[roi_y1:roi_y2, roi_x1:roi_x2],
        'y2 (Target)': y2_img[roi_y1:roi_y2, roi_x1:roi_x2],
    }
    for name in pred_dict:
        slices[name] = pred_dict[name]["x_pred"][roi_y1:roi_y2, roi_x1:roi_x2]

    colors = ['gray', 'silver', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    linestyles = [':', ':', '-', '-', '-', '-']

    plt.figure(figsize=(10, 7), dpi=DPI)
    for i, (label, img_slice) in enumerate(slices.items()):
        psd_1d = compute_1d_psd(img_slice)
        plt.plot(np.arange(len(psd_1d)), psd_1d, label=label,
                 color=colors[i % len(colors)],
                 ls=linestyles[i % len(linestyles)],
                 lw=2.2 if i >= 2 else 1.8, alpha=0.9)

    plt.yscale('log')
    plt.title(f'Local Patch PSD (Deconv Results vs Observations)',
              fontsize=FONT_SIZE, fontweight='bold')
    plt.xlabel('Spatial Frequency (px$^{-1}$)', fontsize=FONT_SIZE - 1)
    plt.ylabel('Power Density', fontsize=FONT_SIZE - 1)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(fontsize=FONT_SIZE - 2, loc='upper right', frameon=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# 主程序
# ============================================================
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    cfg = get_config()
    ensure_dir(CUSTOM_OUTPUT_ROOT)

    # 加载测试数据（相邻帧对）
    test_pairs = load_test_pairs(TARGET_BAND, MAX_PAIRS)

    # 选择本次测试的模型
    active_keys = GROUP_MODEL_KEYS.get(TEST_GROUP, list(ABLATION_MODELS.keys()))
    active_models = {k: ABLATION_MODELS[k] for k in active_keys if k in ABLATION_MODELS}
    print(f"\n🧪 测试组: {TEST_GROUP} → 模型: {list(active_models.keys())}")

    # 先加载第一个模型获取 PSF 退化算子
    print("\n⚙️  初始化模型...")
    first_key = list(active_models.keys())[0]
    first_cfg = active_models[first_key]
    print(f"  参考模型: {first_cfg['desc']} (获取共享 PSF 算子)")
    ref_model = init_model(first_cfg["dir"], first_cfg["flags"], cfg)
    psf_module = ref_model.psf_degrade  # 所有模型共享同一个物理 PSF

    # 初始化所有待测消融模型
    models = {}
    for exp_name, info in active_models.items():
        if exp_name == first_key:
            models[exp_name] = ref_model
        else:
            print(f"  {info['desc']}...")
            models[exp_name] = init_model(info["dir"], info["flags"], cfg)

    print(f"\n🚀 开始测试。测试帧对数: {len(test_pairs)}")
    print(f"   📐 场景: 真实 AIA 观测, 无 GT, 相邻帧对 (y₁, y₂)")
    print(f"   📐 核心指标: PSNR(PSF(x̂₁), y₂) — 逆一致性, 越高越好")
    print(f"   📐 残差指标: ResMean = mean(|PSF(x̂₁) - y₂|) — 越低越好")
    print(f"   📐 稳定性指标: ResStd = std(|PSF(x̂₁) - y₂|) — 越低越稳定")
    print(f"   📐 通量守恒: FluxErr = |sum(PSF(x̂₁)) - sum(y₂)| / sum(y₂)")

    detail_rows = []

    for pair_idx, (file1, file2) in enumerate(tqdm(test_pairs, desc="Testing pairs")):
        sample_id = f"pair_{pair_idx:02d}"

        # 读取观测
        y1_img = read_fits_image(file1)
        y2_img = read_fits_image(file2)

        # 自动 ROI
        zoom_roi = auto_detect_flare_roi(y1_img, patch_size=PATCH_SIZE)

        # 所有模型以 y1 为输入进行推理
        pred_dict = {}
        input_tensor = torch.from_numpy(y1_img).unsqueeze(0).unsqueeze(0).to(device)

        for exp_name, model in models.items():
            with torch.no_grad():
                x_pred = model(input_tensor)[0, 0].cpu().numpy()

            # 计算逆一致性指标
            metrics = compute_inverse_consistency_metrics(x_pred, y2_img, model.psf_degrade)

            # 计算残差图（用于可视化）
            pred_t = torch.from_numpy(x_pred).unsqueeze(0).unsqueeze(0).float().to(device)
            with torch.no_grad():
                degraded = model.psf_degrade(pred_t).squeeze().cpu().numpy()
            residual = np.abs(np.nan_to_num(degraded, nan=0.0) - y2_img)

            pred_dict[exp_name] = {
                "x_pred": x_pred,
                "degraded": degraded,
                "residual": residual,
            }

            # 同时记录自洽性指标（vs y1 而非 y2）
            self_metrics = compute_inverse_consistency_metrics(x_pred, y1_img, model.psf_degrade)

            base_row = {
                "sample_id": sample_id,
                "file1": os.path.basename(file1),
                "file2": os.path.basename(file2),
                "experiment": exp_name,
                "band": TARGET_BAND,
                "roi_y1": zoom_roi[0], "roi_y2": zoom_roi[1],
                "roi_x1": zoom_roi[2], "roi_x2": zoom_roi[3],
                # 逆一致性 vs y2
                "psnr_vs_y2": metrics["psnr"],
                "res_mean_vs_y2": metrics["res_mean"],
                "res_std_vs_y2": metrics["res_std"],
                "flux_err_vs_y2": metrics["flux_err"],
                # 自洽性 vs y1
                "psnr_vs_y1": self_metrics["psnr"],
                "res_mean_vs_y1": self_metrics["res_mean"],
                "res_std_vs_y1": self_metrics["res_std"],
                "flux_err_vs_y1": self_metrics["flux_err"],
            }
            detail_rows.append(base_row)

        # 保存可视化
        img_sample_dir = os.path.join(CUSTOM_OUTPUT_ROOT, sample_id)
        ensure_dir(img_sample_dir)
        plot_ablation_comparison(
            y1_img, y2_img, pred_dict, zoom_roi,
            os.path.join(img_sample_dir, f"{TARGET_BAND}A_ABLATION_COMPARISON.png"))
        plot_combined_psd(
            y1_img, y2_img, pred_dict=pred_dict, zoom_roi=zoom_roi,
            save_path=os.path.join(img_sample_dir, f"{TARGET_BAND}A_ABLATION_PSD.png"))

    # ============================================================
    # 输出 CSV
    # ============================================================
    metric_keys = [
        "psnr_vs_y2", "res_mean_vs_y2", "res_std_vs_y2", "flux_err_vs_y2",
        "psnr_vs_y1", "res_mean_vs_y1", "res_std_vs_y1", "flux_err_vs_y1",
    ]
    fieldnames_detail = [
        "sample_id", "file1", "file2", "experiment", "band",
        "roi_y1", "roi_y2", "roi_x1", "roi_x2",
    ] + metric_keys
    write_csv(os.path.join(CUSTOM_OUTPUT_ROOT, f"{TARGET_BAND}A_metrics_detail.csv"),
              detail_rows, fieldnames_detail)

    # 汇总表
    summary_rows = []
    for exp_name in active_models.keys():
        rows = [r for r in detail_rows if r["experiment"] == exp_name]
        summary = {"experiment": exp_name, "n_pairs": len(rows)}
        for k in metric_keys:
            vals = np.array([float(r[k]) for r in rows if not np.isnan(r[k])], dtype=np.float64)
            summary[f"{k}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
            summary[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary_rows.append(summary)

    fieldnames_summary = ["experiment", "n_pairs"] + \
                         [f"{k}_mean" for k in metric_keys] + \
                         [f"{k}_std" for k in metric_keys]
    write_csv(os.path.join(CUSTOM_OUTPUT_ROOT, f"{TARGET_BAND}A_metrics_summary.csv"),
              summary_rows, fieldnames_summary)

    # ============================================================
    # 终端结果打印
    # ============================================================
    print("\n\n" + "=" * 130)
    print("📊 消融实验测试结果（无真值, 逆一致性评价）")
    print("   PSNR ↑ 越高越好 | ResMean ↓ 越低越好 | ResStd ↓ 越低越稳定 | FluxErr ↓ 越低越好")
    print("=" * 130)
    header = (f"{'Experiment':<28} | {'PSNR_y2':>8} | {'ResMean':>8} | "
              f"{'ResStd':>8} | {'FluxErr':>8} | "
              f"{'PSNR_y1':>8} | {'ResMean_y1':>8}")
    print(header)
    print("-" * 130)
    for row in summary_rows:
        print(f"{row['experiment']:<28} | "
              f"{row['psnr_vs_y2_mean']:>8.2f} | "
              f"{row['res_mean_vs_y2_mean']:>8.4f} | "
              f"{row['res_std_vs_y2_mean']:>8.4f} | "
              f"{row['flux_err_vs_y2_mean']:>8.4f} | "
              f"{row['psnr_vs_y1_mean']:>8.2f} | "
              f"{row['res_mean_vs_y1_mean']:>8.4f}")

    print(f"\n✅ 测试完成。结果保存至: {CUSTOM_OUTPUT_ROOT}")
    print(f"   详细数据: {TARGET_BAND}A_metrics_detail.csv")
    print(f"   汇总数据: {TARGET_BAND}A_metrics_summary.csv")


if __name__ == "__main__":
    main()
