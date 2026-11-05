#!/usr/bin/env python3
"""Blur2Sun v5 一键演示：加载轻量演示数据（demo_data/）+ v5 模型 → 出论文级例图。

用法：
    export BLUR2SUN_MODELS=/path/to/outputs      # v5 权重目录（含 band_XX/map_full_12s_v5/models/）
    python3 demo/run_demo.py                     # 在 demo/output/ 生成三张图

依赖：仓库内 src/（模型+官方 BID）、psf/（修正 PSF）。演示数据共 ~3.6MB。
推理口径：标准曝光帧原生输入（不加底台）；仅短曝光耀斑帧加逐通道校准常数
（论文 §4.4）。
"""
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sunpy.visualization.colormaps  # noqa: F401  (注册 sdoaia* 色标)
from astropy.visualization import ImageNormalize, SqrtStretch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'baselines'))
sys.path.insert(0, os.path.join(ROOT, 'evaluation', 'paper_tables_v5'))

from v5_common import load_v5_model, bid_deconv  # noqa: E402

DATA = os.path.join(HERE, 'demo_data')
OUT = os.path.join(HERE, 'output')
# Hand-set per-channel calibration constants, applied ONLY to count-depleted
# short-exposure flare frames (Section 4.4 of the paper).
INPUT_OFFSET = {94: 2.0, 131: 2.0, 171: 85.0, 193: 4.0, 211: 85.0, 304: 4.0, 335: 4.0}

# 论文线型规范：全实线，仅图案区分
STYLES = {'obverse': dict(color='#555555', marker=None, ms=0, lw=1.2),
          'proposed': dict(color='#d62728', marker='^', ms=8, lw=1.8),
          'rl': dict(color='#ff7f0e', marker='v', ms=7, lw=1.8),
          'bid': dict(color='#1f77b4', marker='o', ms=6, lw=1.8)}
LABELS = {'obverse': 'Obverse', 'proposed': 'Proposed', 'rl': 'RL', 'bid': 'BID'}


def b2s_infer(model, img, band, short_exposure=False):
    """Native-intensity inference; input offset only for short-exposure frames."""
    H, W = img.shape
    offset = INPUT_OFFSET[band] if short_exposure else 0.0
    t = torch.from_numpy(np.ascontiguousarray(img + offset)).float().cuda()[None, None]
    with torch.no_grad():
        x = model(t)[0, 0].cpu().numpy() - offset
    yy, xx = np.mgrid[0:H, 0:W]
    ring = np.hypot(yy - H / 2, xx - W / 2) > min(H, W) * 0.44
    return np.clip(x - float(np.median(x[ring])), 0, None)


def demo_image(npz_name, out_name, title, short_exposure=False):
    d = dict(np.load(os.path.join(DATA, npz_name)))
    img, band = d['image'].astype(np.float64), int(d['band'])
    model = load_v5_model(band)
    t0 = time.perf_counter()
    x_b2s = b2s_infer(model, img, band, short_exposure)
    t_b2s = time.perf_counter() - t0
    t0 = time.perf_counter()
    x_bid = bid_deconv(img, band, iterations=30)
    t_bid = time.perf_counter() - t0
    print(f'{npz_name}: B2S {t_b2s:.2f}s | BID {t_bid:.2f}s '
          f'(1024-class frame, band {band}A)')

    vmin, vmax = np.percentile(img, [1, 99.8])
    norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
    # 核区放大（最亮像素 ±160 px）
    from scipy.ndimage import uniform_filter
    j, i = np.unravel_index(np.argmax(uniform_filter(img, 15)), img.shape)
    Z = 160
    sl = np.s_[max(0, j - Z):j + Z, max(0, i - Z):i + Z]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 10.5))
    for r, (region, s) in enumerate([('full frame', np.s_[...]), ('core zoom', sl)]):
        for c, (im, ttl) in enumerate([(img, 'Observed'), (x_b2s, 'Blur2Sun v5'),
                                       (x_bid, 'BID (30it)')]):
            ax = axes[r, c]
            ax.imshow(im[s], origin='lower', cmap=f'sdoaia{band}', norm=norm)
            ax.set_title(f'{ttl} — {region}', fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{title} — {band}A, SqrtStretch (paper convention), '
                 f'zoom {2*Z}px on the brightest core', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, out_name), dpi=140, facecolor='white')
    plt.close(fig)
    print('saved', os.path.join(OUT, out_name))


def demo_lunar():
    d = np.load(os.path.join(DATA, 'lunar_profiles_7bands.npz'))
    bands = sorted({k.split('_')[0] for k in d.files if k.endswith('_obverse_profile')})
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    # 左：193 剖线位置缩略图
    th = dict(np.load(os.path.join(DATA, 'lunar_thumb_193.npz')))
    axes[0].imshow(th['thumb'], origin='lower', cmap='gray',
                   vmin=np.percentile(th['thumb'], 5), vmax=np.percentile(th['thumb'], 99.5))
    row = int(th['lunar_row']) / 4
    axes[0].axhline(row, color='#FF2200', lw=1.2)
    axes[0].set_title('193A lunar-transit frame (thumbnail, red = profile row)')
    axes[0].set_xticks([]); axes[0].set_yticks([])
    # 右：月缘放大剖面（193）
    ax = axes[1]
    for m in ['obverse', 'proposed', 'rl', 'bid']:
        prof = d[f'193_{m}_profile']
        st = STYLES[m]
        pos = np.arange(len(prof))
        sel = (pos >= 625) & (pos <= 750)
        ax.plot(pos[sel], prof[sel], color=st['color'], linestyle='-',
                marker=st['marker'], markersize=st['ms'], linewidth=st['lw'],
                markevery=6, label=LABELS[m])
    ax.set_xlabel('Pixel position')
    ax.set_ylabel('Intensity [DN]')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_title('193A lunar-edge profiles (zoom 625-750)')
    fig.suptitle('Lunar-transit stray-light test (from 1D demo profiles)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, 'demo_lunar_193.png'), dpi=140, facecolor='white')
    plt.close(fig)
    print('saved', os.path.join(OUT, 'demo_lunar_193.png'))


if __name__ == '__main__':
    if not os.environ.get('BLUR2SUN_MODELS') and not os.path.isdir(
            os.path.join(ROOT, 'outputs')):
        sys.exit('请设置 BLUR2SUN_MODELS 指向 v5 权重目录'
                 '（含 band_XX/map_full_12s_v5/models/best_model.pth）')
    demo_image('flare_X93_094.npz', 'demo_flare_X93_094.png',
               'X9.3 flare frame (co-add x5)', short_exposure=True)
    demo_image('quiet_171.npz', 'demo_quiet_171.png', 'Quiet-Sun frame')
    demo_lunar()
    print('\n全部演示图已生成于', OUT)
