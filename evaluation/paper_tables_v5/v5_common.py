#!/usr/bin/env python3
"""论文表格 v5 重跑公共模块：v5 模型加载 + 修正 PSF（rebin2048）算子。

与论文 V1 脚本的关键差异（全部为 Minor-11 修正 PSF 的落实）：
  - B2S: v5 MAPIterativeNet（内部即用修正 PSF 2048）
  - RL : 修正 PSF（2048 crop）+ 与模型完全同构的 FFT padding（线性卷积）
  - BID: 官方 deconvolve_bid + 修正 PSF（2048 crop）
数据与指标口径与原论文脚本完全一致，只换模型与算子。
"""
import os
import sys
import copy
import importlib.util
import types

import numpy as np
import torch
import torch.nn.functional as F
from astropy.io import fits

# 仓库相对路径：本文件位于 <repo>/evaluation/paper_tables_v5/
_HERE = os.path.dirname(os.path.abspath(__file__))
V5_ROOT = os.path.dirname(os.path.dirname(_HERE))
B2S_ROOT = V5_ROOT  # 官方 BID (deconvolve_image.py) 已内置于 src/
BANDS_4 = {'94': '0094', '131': '0131', '171': '0171', '193': '0193',
           '211': '0211', '304': '0304', '335': '0335'}

# v5 src 优先（跨版本错载静默出垃圾——方法论红线）
sys.path.insert(0, os.path.join(V5_ROOT, 'src'))
sys.path.insert(0, os.path.join(V5_ROOT, 'src', 'baselines'))

try:
    import cupy  # noqa: F401
except ImportError:
    _cp = types.ModuleType('cupy')

    class _CpArray:
        pass
    _cp.ndarray = _CpArray
    sys.modules['cupy'] = _cp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_CFG_CACHE = {}


def _load_cfg():
    if 'cfg' not in _CFG_CACHE:
        spec = importlib.util.spec_from_file_location(
            'cm_v5', os.path.join(V5_ROOT, 'configs', 'config_map_12s_v5.py'))
        cm = importlib.util.module_from_spec(spec)
        sys.modules['cm_v5'] = cm
        spec.loader.exec_module(cm)
        _CFG_CACHE['cfg'] = cm.get_config()
    return _CFG_CACHE['cfg']


def load_v5_model(band):
    """band: int（94..335）。返回 eval 状态的 v5 MAPIterativeNet。"""
    import model2_map
    from model2_map import MAPIterativeNet
    # 防错载约束（2026-10-27 事故后加固）：
    #  1. 断言 model2_map 解析到本仓库 src/（防 sys.path 遮蔽到旧版伴随步实现）；
    #  2. 加载后断言 missing/unexpected 键全空（防 strict=False 静默丢参）。
    if not os.path.abspath(model2_map.__file__) \
            .startswith(os.path.abspath(os.path.join(V5_ROOT, 'src'))):
        raise RuntimeError(
            f'[model guard] model2_map shadowed: {model2_map.__file__}; '
            f'expected {V5_ROOT}/src/model2_map.py (log_k/Landweber data step)')
    cfg = copy.deepcopy(_load_cfg())
    cfg.BASE.CURRENT_BAND = str(int(band))
    cfg.MODEL.USE_MAP_PRIOR = True
    cfg.MODEL.ADAPTIVE_ITER = False
    model = MAPIterativeNet(cfg).to(device)
    _models = os.environ.get('BLUR2SUN_MODELS', os.path.join(V5_ROOT, 'outputs'))
    ckpt = os.path.join(_models, f'band_{int(band)}',
                        'map_full_12s_v5', 'models', 'best_model.pth')
    missing, unexpected = model.load_state_dict(
        torch.load(ckpt, map_location=device,
                   weights_only=False)['model_state_dict'],
        strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f'[model guard] incomplete checkpoint band={band}: '
            f'missing={list(missing)} unexpected={list(unexpected)}')
    model.eval()
    return model


def _round_up_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


def corr_psf_fft(band, H, W, dev=device):
    """修正 PSF（rebin2048 中心裁 2048，归一）的 FFT——与 v5 模型
    PSFDegradationModule 完全同构（pad=round_up_pow2(H+K-1)，核居中）。"""
    b4 = BANDS_4[str(int(band))]
    psf_file = os.path.join(V5_ROOT, 'psf', f'psf_aia_{b4}_rebin2048.fits')
    with fits.open(psf_file) as hdul:
        psf = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None \
            else hdul[0].data
    psf = np.asarray(psf, np.float64)
    K = 2048
    h, w = psf.shape
    sh, sw = (h - K) // 2, (w - K) // 2
    k = psf[sh:sh + K, sw:sw + K]
    k = k / k.sum()
    pad_h = _round_up_pow2(H + K - 1)
    pad_w = _round_up_pow2(W + K - 1)
    kp = torch.zeros((1, 1, pad_h, pad_w), dtype=torch.float64, device=dev)
    st_h, st_w = pad_h // 2 - K // 2, pad_w // 2 - K // 2
    kp[:, :, st_h:st_h + K, st_w:st_w + K] = torch.from_numpy(
        np.ascontiguousarray(k)).to(dev)[None, None]
    pf = torch.fft.fft2(torch.fft.ifftshift(kp, dim=(-2, -1)))
    return pf, pad_h, pad_w


def _pad_to(x, pad_h, pad_w, dev):
    t = torch.from_numpy(np.ascontiguousarray(
        np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))).float().to(dev)
    return F.pad(t[None, None], (0, pad_w - x.shape[1], 0, pad_h - x.shape[0]))


def rl_deconv(img, band, num_iter=30, eps=1e-8):
    """RL（与模型同算子的线性卷积版）。返回 numpy (H,W) 非负。"""
    H, W = img.shape
    pf, pad_h, pad_w = corr_psf_fft(band, H, W)
    x = _pad_to(img, pad_h, pad_w, device).clone()
    y = _pad_to(img, pad_h, pad_w, device)
    for _ in range(num_iter):
        sim = torch.fft.ifft2(torch.fft.fft2(x) * pf).real
        ratio = y / (sim + eps)
        corr = torch.fft.ifft2(torch.fft.fft2(ratio) * torch.conj(pf)).real
        x = torch.clamp(x * corr, min=0.0)
    return x[0, 0, :H, :W].cpu().numpy()


def bid_deconv(img, band, iterations=30, estimate_background=True, **_):
    """官方 BID + 修正 PSF（rebin2048 裁 2048，归一）。
    其余参数（tolerance/mask/use_gpu/large_psf/pad/constrain_positive）
    固定为论文口径，由 **_ 接收忽略。"""
    from deconvolve_image import deconvolve_bid
    b4 = BANDS_4[str(int(band))]
    with fits.open(os.path.join(V5_ROOT, 'psf',
                                f'psf_aia_{b4}_rebin2048.fits')) as hdul:
        psf = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None \
            else hdul[0].data
    psf = np.asarray(psf, np.float64)
    K = 2048
    h, w = psf.shape
    sh, sw = (h - K) // 2, (w - K) // 2
    k = psf[sh:sh + K, sw:sw + K]
    k = k / k.sum()
    out = deconvolve_bid(img=np.nan_to_num(img, nan=0.0, posinf=0.0,
                                           neginf=0.0),
                         psf=k, iterations=iterations, tolerance=0.1,
                         mask=None, use_gpu=False, large_psf=True, pad=True,
                         estimate_background=estimate_background,
                         constrain_positive=True)
    return np.nan_to_num(np.asarray(out, np.float64), nan=0.0, posinf=0.0,
                         neginf=0.0)
