import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# AIA可视化核心依赖
import sunpy.map
from astropy.visualization import ImageNormalize, LogStretch, LinearStretch
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 一、AIA可视化核心配置（保留原有专业配置）y
# -----------------------------------------------------------------------------
WAVELENGTH_TO_CMAP = {
    94: 'sdoaia94', 131: 'sdoaia131', 171: 'sdoaia171',
    193: 'sdoaia193', 211: 'sdoaia211', 304: 'sdoaia304', 335: 'sdoaia335'
}

# AIA_OFFICIAL_CONFIG = {
#     94: (0, 99.9, True), 131: (25, 99.9, True), 171: (0, 99.5, False),
#     193: (1, 99.5, False), 211: (1, 99.5, False), 304: (30, 99.9, True),
#     335: (0, 99.9, True)
# }
AIA_OFFICIAL_CONFIG = {
    94: (0, 99.5, True),    # 低计数，下限0，上限99.5，避免压缩暗区
    131: (2, 99.5, True),   # 中等偏低，下限2，平衡噪声和结构
    171: (0, 99.5, False),  # 高计数
    193: (1, 99.5, False),  # 高计数
    211: (1, 99.5, False),  # 高计数
    304: (5, 99.5, True),   # 中等，下限5，微调
    335: (0, 99.5, True)    # 低计数，下限0，上限99.5，避免压缩暗区
}
SUPPORTED_WAVELENGTHS = list(AIA_OFFICIAL_CONFIG.keys())

# 🆕 优化1：调整绘图风格，更紧凑、更适合顶刊（1行4列）
plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "figure.figsize": (20, 5),  # 1行4列，更紧凑
    "figure.dpi": 300, "savefig.dpi": 300, "axes.labelsize": 10,
    "axes.titlesize": 11, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.titlepad": 8, "figure.subplot.wspace": 0.3  # 增加子图间距，避免色标重叠
})

DIFF_PERCENTILE = 99

def get_aia_config(wavelength):
    pure_wave = int(''.join([c for c in str(wavelength) if c.isdigit()]) or 171)
    pure_wave = pure_wave if pure_wave in SUPPORTED_WAVELENGTHS else 171
    cmap = plt.get_cmap(WAVELENGTH_TO_CMAP[pure_wave])
    min_pct, max_pct, use_log = AIA_OFFICIAL_CONFIG[pure_wave]
    return cmap, min_pct, max_pct, use_log

def get_percentile_limits(data, min_pct, max_pct):
    valid_data = data[~np.isnan(data)]
    if len(valid_data) == 0:
        return 0, 1
    vmin = np.percentile(valid_data, min_pct)
    vmax = np.percentile(valid_data, max_pct)
    if vmin == vmax:
        vmax = vmin + 1e-6 if vmin == 0 else vmin * 1.01
    return vmin, vmax

def get_diff_norm(diff_data, use_symlog=False, linthresh=1e-3):
    """🆕 修正：分离astropy和matplotlib的norm，解决类型不匹配"""
    valid_diff = diff_data[~np.isnan(diff_data)]
    abs_max = np.percentile(np.abs(valid_diff), DIFF_PERCENTILE) if len(valid_diff) > 0 else 1
    cmap = plt.get_cmap('seismic')
    
    if use_symlog:
        # 🚨 修正点1：用matplotlib的SymLogNorm直接作为norm，而非astropy的stretch
        norm = colors.SymLogNorm(linthresh=linthresh, linscale=1, 
                                 vmin=-abs_max, vmax=abs_max, base=10)
    else:
        # 线性拉伸用astropy的ImageNormalize（保持原有逻辑）
        norm = ImageNormalize(vmin=-abs_max, vmax=abs_max, stretch=LinearStretch())
    return cmap, norm, -abs_max, abs_max

# -----------------------------------------------------------------------------
# 二、🆕 优化后的Blur2Blur可视化函数（核心修改）
# -----------------------------------------------------------------------------
def visualize_blur2blur_result(epoch, y1, y2, x1_pred, x2_pred, cfg, save_suffix='', use_frame1=True):
    """
    🆕 优化版：1行4列布局 + 统一色标 + sym-log差分 + 物理指标标注
    :param use_frame1: True用y1/x1_pred，False用y2/x2_pred（主图只用1帧，更紧凑）
    """
    band = cfg.BASE.CURRENT_BAND
    
    # 1. ✅ 修复核心：取 batch 第 0 张图 + 压缩维度 → 纯 2D 图像 (1024,1024)
    def to_np(x): 
        # 输入: torch.Tensor (B,1,H,W) → 取第0张 → 压缩维度 → numpy (H,W)
        if torch.is_tensor(x):
            return x[0].squeeze().cpu().numpy()  # 加 .cpu() 兼容 cuda 张量
        else:
            return np.squeeze(x[0])
    
    if use_frame1:
        y_np, x_pred_np = to_np(y1), to_np(x1_pred)
        frame_label = "Frame 1"
    else:
        y_np, x_pred_np = to_np(y2), to_np(x2_pred)
        frame_label = "Frame 2"
    
    # 2. 计算关键物理指标（直接标注在图上，增强说服力）
    flux_ratio = np.sum(x_pred_np) / (np.sum(y_np) + 1e-12)
    peak_y = np.max(y_np)
    peak_x = np.max(x_pred_np)
    peak_ratio = peak_x / (peak_y + 1e-12)
    
    # 打印数值分布（保留原有好习惯）
    print(f"\n📊 Blur2Blur Epoch {epoch} 物理指标 (Band {band}Å):")
    print(f"   全局通量比 (Sharp/Blur): {flux_ratio:.4f} (理想值: 1.2~1.3)")
    print(f"   峰值强度比 (Sharp/Blur): {peak_ratio:.4f}")
    print(f"   模糊帧: min={y_np.min():.4f}, max={y_np.max():.4f}, mean={y_np.mean():.4f}")
    print(f"   预测帧: min={x_pred_np.min():.4f}, max={x_pred_np.max():.4f}, mean={x_pred_np.mean():.4f}")
    
    # 3. 计算差分图
    diff_np = x_pred_np - y_np
    
    # 4. 🆕 核心优化：统一色标（用模糊帧的vmin/vmax，直观对比强度提升）
    cmap, min_pct, max_pct, use_log = get_aia_config(band)
    stretch = LogStretch() if use_log else LinearStretch()
    
    # 统一用模糊帧的百分位限制，确保视觉对比的公平性
    vmin, vmax = get_percentile_limits(y_np, min_pct, max_pct)
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=stretch)
    
    # 5. 差分图配置（线性+sym-log双版本）
    diff_cmap, diff_norm, _, _ = get_diff_norm(diff_np, use_symlog=False)
    diff_symlog_cmap, diff_symlog_norm, _, _ = get_diff_norm(diff_np, use_symlog=True, linthresh=1e-3)
    
    # 6. 创建输出目录
    vis_dir = os.path.join(cfg.BASE.VIS_DIR, f"epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 7. 🆕 优化布局：1行4列（主图版本，适合论文）
    fig = plt.figure(figsize=(22, 5.5))
    
    # 7.1 原始模糊帧
    ax1 = plt.subplot(1, 4, 1)
    im1 = ax1.imshow(y_np, cmap=cmap, norm=norm)
    ax1.set_title(f'Original Blur\n(AIA {band}Å, {frame_label})', fontweight='bold')
    ax1.axis('off')
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.85)
    cbar1.set_label('Normalized Intensity', fontsize=9)
    
    # 7.2 预测清晰帧（用同一套色标！）
    ax2 = plt.subplot(1, 4, 2)
    im2 = ax2.imshow(x_pred_np, cmap=cmap, norm=norm)
    ax2.set_title(f'Predicted Sharp\n(Blur2Blur)', fontweight='bold')
    # 🆕 标注物理指标在图上
    ax2.text(0.02, 0.98, f'Flux Ratio: {flux_ratio:.2f}\nPeak Ratio: {peak_ratio:.2f}', 
             transform=ax2.transAxes, fontsize=9, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax2.axis('off')
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.85)
    cbar2.set_label('Normalized Intensity', fontsize=9)
    
    # 7.3 线性差分图（看亮区的正差值）
    ax3 = plt.subplot(1, 4, 3)
    im3 = ax3.imshow(diff_np, cmap=diff_cmap, norm=diff_norm)
    ax3.set_title(f'Difference (Linear)\n(Sharp - Blur)', fontweight='bold')
    ax3.axis('off')
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.85)
    cbar3.set_label('Intensity Difference', fontsize=9)
    
    # 7.4 🆕 新增：sym-log差分图（看全局的微弱负差值，呼应你的物理讨论）
    ax4 = plt.subplot(1, 4, 4)
    # 🚨 修正点2：直接传入matplotlib的SymLogNorm，而非astropy的ImageNormalize
    im4 = ax4.imshow(diff_np, cmap=diff_symlog_cmap, norm=diff_symlog_norm)
    ax4.set_title(f'Difference (Sym-Log)\n(Global View)', fontweight='bold')
    ax4.axis('off')
    cbar4 = plt.colorbar(im4, ax=ax4, shrink=0.85)
    cbar4.set_label('Intensity Difference', fontsize=9)
    
    # 8. 保存主图（论文用）
    plt.tight_layout()
    fig_path = os.path.join(vis_dir, f"epoch_{epoch}_blur2blur_main_band{band}{save_suffix}.png")
    fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    # 9. 🆕 补充：保存局部放大图（专门看耀斑周围，放在补充材料）
    # 自动找亮区中心（耀斑位置）
    bright_y, bright_x = np.unravel_index(np.argmax(y_np), y_np.shape)
    crop_size = 200  # 局部放大200×200
    # 边界检查
    start_y = max(0, bright_y - crop_size//2)
    start_x = max(0, bright_x - crop_size//2)
    end_y = min(y_np.shape[0], start_y + crop_size)
    end_x = min(y_np.shape[1], start_x + crop_size)
    
    # 裁剪局部区域
    y_crop = y_np[start_y:end_y, start_x:end_x]
    x_crop = x_pred_np[start_y:end_y, start_x:end_x]
    diff_crop = diff_np[start_y:end_y, start_x:end_x]
    
    # 绘制局部放大图
    fig_zoom = plt.figure(figsize=(16, 5))
    
    ax_zoom1 = plt.subplot(1, 3, 1)
    im_zoom1 = ax_zoom1.imshow(y_crop, cmap=cmap, norm=norm)
    ax_zoom1.set_title(f'Original Blur (Zoom)', fontweight='bold')
    ax_zoom1.axis('off')
    plt.colorbar(im_zoom1, ax=ax_zoom1, shrink=0.85)
    
    ax_zoom2 = plt.subplot(1, 3, 2)
    im_zoom2 = ax_zoom2.imshow(x_crop, cmap=cmap, norm=norm)
    ax_zoom2.set_title(f'Predicted Sharp (Zoom)', fontweight='bold')
    ax_zoom2.axis('off')
    plt.colorbar(im_zoom2, ax=ax_zoom2, shrink=0.85)
    
    ax_zoom3 = plt.subplot(1, 3, 3)
    im_zoom3 = ax_zoom3.imshow(diff_crop, cmap=diff_cmap, norm=diff_norm)
    ax_zoom3.set_title(f'Difference (Zoom)', fontweight='bold')
    ax_zoom3.axis('off')
    plt.colorbar(im_zoom3, ax=ax_zoom3, shrink=0.85)
    
    fig_zoom_path = os.path.join(vis_dir, f"epoch_{epoch}_blur2blur_zoom_band{band}{save_suffix}.png")
    fig_zoom.savefig(fig_zoom_path, bbox_inches='tight', facecolor='white')
    plt.close(fig_zoom)
    
    print(f"✅ Blur2Blur Epoch {epoch} 可视化完成:")
    print(f"   主图: {fig_path}")
    print(f"   局部放大图: {fig_zoom_path}")

# -----------------------------------------------------------------------------
# 三、保留原有函数（兼容旧模式）
# -----------------------------------------------------------------------------
def visualize_result(epoch, original, masked, recovered, cfg, save_suffix=''):
    band = cfg.BASE.CURRENT_BAND
    
    # 1. 数据预处理
    def to_np(x): return x.squeeze().numpy() if torch.is_tensor(x) else np.squeeze(x)
    orig_np = to_np(original)
    masked_np = to_np(masked)
    rec_np = to_np(recovered)
    
    # 数值分布打印
    print(f"\n📊 Epoch {epoch} 数值分布 (Band {band}Å):")
    print(f"   原始图: min={orig_np.min():.8f}, max={orig_np.max():.8f}, mean={orig_np.mean():.8f}")
    print(f"   Masked图: min={masked_np.min():.8f}, max={masked_np.max():.8f}, mean={masked_np.mean():.8f}")
    print(f"   校正图: min={rec_np.min():.8f}, max={rec_np.max():.8f}, mean={rec_np.mean():.8f}")
    
    print(f"📏 可视化数据尺寸 - 原始整图: {orig_np.shape}, Mask后整图: {masked_np.shape}, 校正整图: {rec_np.shape}")
    
    # 2. 计算全局差分图
    diff_np = rec_np - orig_np
    
    # 3. AIA可视化配置
    cmap, min_pct, max_pct, use_log = get_aia_config(band)
    stretch = LogStretch() if use_log else LinearStretch()
    
    vmin_orig, vmax_orig = get_percentile_limits(orig_np, min_pct, max_pct)
    norm_orig = ImageNormalize(vmin=vmin_orig, vmax=vmax_orig, stretch=stretch)
    
    vmin_rec, vmax_rec = get_percentile_limits(rec_np, min_pct, max_pct)
    norm_rec = ImageNormalize(vmin=vmin_rec, vmax=vmax_rec, stretch=stretch)
    
    diff_cmap, diff_norm, _, _ = get_diff_norm(diff_np)
    
    # 4. 创建输出目录
    vis_dir = os.path.join(cfg.BASE.VIS_DIR, f"epoch_{epoch}")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 5. 绘制画布（1行5列）
    fig = plt.figure(figsize=(40, 8))
    
    # 5.1 原始观测
    ax1 = plt.subplot(1, 5, 1)
    im1 = ax1.imshow(orig_np, cmap=cmap, norm=norm_orig)
    ax1.set_title(f'Original Full Corona (AIA {band}Å)', fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    # 5.2 Mask后图像
    ax2 = plt.subplot(1, 5, 2)
    im2 = ax2.imshow(masked_np, cmap=cmap, norm=norm_orig)
    ax2.set_title(f'Masked Full Corona (AIA {band}Å)', fontweight='bold')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 5.3 校正结果
    ax3 = plt.subplot(1, 5, 3)
    im3 = ax3.imshow(rec_np, cmap=cmap, norm=norm_rec)
    ax3.set_title(f'Corrected Full Corona (Ours)', fontweight='bold')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # 5.4 全局差分图
    ax4 = plt.subplot(1, 5, 4)
    im4 = ax4.imshow(diff_np, cmap=diff_cmap, norm=diff_norm)
    ax4.set_title(f'Difference (Corrected - Original)', fontweight='bold')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    # 5.5 原始观测参考
    ax5 = plt.subplot(1, 5, 5)
    im5 = ax5.imshow(orig_np, cmap=cmap, norm=norm_orig)
    ax5.set_title(f'Full Corona Reference (AIA {band}Å)', fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    # 6. Mask分布图
    mask_fig = plt.figure(figsize=(8, 8))
    mask_ax = mask_fig.add_subplot(111)
    mask_matrix = np.where(masked_np == 0, 1, 0)
    MASK_CMAP = plt.get_cmap('gray_r')
    mask_im = mask_ax.imshow(mask_matrix, cmap=MASK_CMAP, vmin=0, vmax=1)
    mask_ax.set_title(f'Mask Distribution (Epoch {epoch}, Band {band}Å)', fontweight='bold')
    mask_ax.axis('off')
    plt.colorbar(mask_im, ax=mask_ax, shrink=0.8, label='Mask (1=Masked, 0=Valid)')
    mask_fig_path = os.path.join(vis_dir, f"epoch_{epoch}_mask_distribution_band{band}{save_suffix}.png")
    mask_fig.savefig(mask_fig_path, bbox_inches='tight', facecolor='white')
    plt.close(mask_fig)
    
    # 7. 保存结果
    plt.tight_layout()
    fig_path = os.path.join(vis_dir, f"epoch_{epoch}_full_corona_band{band}{save_suffix}.png")
    fig.savefig(fig_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    print(f"✅ Epoch {epoch} 整图可视化完成: {fig_path}")
    print(f"✅ Epoch {epoch} Mask分布图保存完成: {mask_fig_path}")

def plot_loss_curve(train_losses, val_losses, cfg):
    """保留原有Loss曲线函数，仅微调风格"""
    band = cfg.BASE.CURRENT_BAND
    epochs = list(range(1, len(train_losses)+1))
    
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_losses, label='Train Loss', c='blue', linewidth=2, marker='o', markersize=3)
    plt.plot(epochs, val_losses, label='Val Loss', c='red', linewidth=2, marker='s', markersize=3)
    
    plt.title(f'Training & Validation Loss (AIA {band}Å)', fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    loss_path = os.path.join(cfg.BASE.VIS_DIR, f"loss_curve_band{band}.png")
    plt.tight_layout()
    plt.savefig(loss_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Loss曲线保存完成: {loss_path}")