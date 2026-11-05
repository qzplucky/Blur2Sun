"""
Bayesian MAP-based Deep Unfolding Network
==========================================
Replaces BID Deep Unfolding with MAP-based Deep Unfolding.

Each iteration solves:
    x_{k+1} = x_k - α_k * ∇_x D(x_k, y) - β_k * ∇_x log p(x_k)

where:
  - ∇_x D(x_k, y) = PSF^T(PSF(x_k) - y)   : Data (likelihood) gradient
  - ∇_x log p(x_k) ≈ PriorNet(x_k)          : Learned Bayesian Prior gradient
  - α_k, β_k are learnable step sizes

PriorNet does NOT predict an image — it approximates the gradient of the
log-prior, making the entire network interpretable as a Learned MAP Solver.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits
import numpy as np


# =========================================================================
# 基础工具
# =========================================================================
def _pick_gn_groups(channels: int, max_groups: int = 8) -> int:
    for g in [max_groups, 8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


def _make_gn(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(_pick_gn_groups(channels), channels)


def _ensure_1ch(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != 1:
        return x[:, 0:1, :, :]
    return x


# =========================================================================
# PSF 物理退化算子（与 model2.py 完全一致）
# =========================================================================
class PSFDegradationModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        psf_tensor = self._load_and_preprocess_psf()
        self.register_buffer("psf_tensor", psf_tensor)
        self.K = psf_tensor.shape[-1]

        default_H = int(cfg.PREPROCESS.ORIGINAL_SIZE)
        default_W = int(cfg.PREPROCESS.ORIGINAL_SIZE)
        self._precompute_psf_fft_for(default_H, default_W)

    def _load_and_preprocess_psf(self):
        current_band = str(self.cfg.BASE.CURRENT_BAND)
        psf_file = self.cfg.AIA_BAND_PARAMS[current_band]["PSF_FILE"]
        target_size = int(self.cfg.BASE.PSF_CROP_SIZE)

        try:
            with fits.open(psf_file) as hdul:
                psf_data = hdul[1].data if len(hdul) > 1 and hdul[1].data is not None else hdul[0].data
                psf_data = psf_data.astype(np.float32)

            h, w = psf_data.shape
            if h > target_size:
                start = (h - target_size) // 2
                psf_data = psf_data[start:start + target_size, start:start + target_size]
            elif h < target_size:
                pad_total = target_size - h
                pad_before = pad_total // 2
                pad_after = pad_total - pad_before
                psf_data = np.pad(
                    psf_data,
                    ((pad_before, pad_after), (pad_before, pad_after)),
                    mode="constant"
                )

            s = np.sum(psf_data)
            if s == 0:
                raise RuntimeError("PSF 总和为 0，检查 PSF 文件或裁剪逻辑。")

            psf_data = psf_data / s
            return torch.from_numpy(psf_data).unsqueeze(0).unsqueeze(0).float()

        except Exception as e:
            raise RuntimeError(f"加载 PSF 失败: {psf_file}, 错误: {e}")

    def _round_up_pow2(self, x):
        p = 1
        while p < x:
            p <<= 1
        return p

    def _precompute_psf_fft_for(self, H, W):
        psf = self.psf_tensor
        K = self.K

        pad_h = H + K - 1
        pad_w = W + K - 1

        pad_h = self._round_up_pow2(pad_h)
        pad_w = self._round_up_pow2(pad_w)

        psf_pad = torch.zeros((1, 1, pad_h, pad_w), dtype=psf.dtype, device=psf.device)
        start_h = pad_h // 2 - (K // 2)
        start_w = pad_w // 2 - (K // 2)
        psf_pad[:, :, start_h:start_h + K, start_w:start_w + K] = psf

        psf_fft = torch.fft.fft2(torch.fft.ifftshift(psf_pad, dim=(-2, -1)))
        self.register_buffer("psf_fft", psf_fft)
        self.pad_h, self.pad_w = pad_h, pad_w
        self._precomputed_H, self._precomputed_W = H, W

    def _compute_psf_fft_dynamic(self, H, W, device):
        psf = self.psf_tensor.to(device)
        K = self.K

        pad_h = H + K - 1
        pad_w = W + K - 1

        pad_h = self._round_up_pow2(pad_h)
        pad_w = self._round_up_pow2(pad_w)

        psf_pad = torch.zeros((1, 1, pad_h, pad_w), dtype=psf.dtype, device=device)
        start_h = pad_h // 2 - (K // 2)
        start_w = pad_w // 2 - (K // 2)
        psf_pad[:, :, start_h:start_h + K, start_w:start_w + K] = psf

        psf_fft = torch.fft.fft2(torch.fft.ifftshift(psf_pad, dim=(-2, -1)))
        return psf_fft, pad_h, pad_w

    def _get_fft(self, H, W, device):
        cached = (
            hasattr(self, "_precomputed_H")
            and hasattr(self, "_precomputed_W")
            and self._precomputed_H == H
            and self._precomputed_W == W
        )
        if cached and hasattr(self, "psf_fft"):
            if self.psf_fft.device == device:
                return self.psf_fft, self.pad_h, self.pad_w
            return self.psf_fft.to(device), self.pad_h, self.pad_w
        return self._compute_psf_fft_dynamic(H, W, device)

    def forward(self, img):
        img = _ensure_1ch(img)
        _, _, H, W = img.shape
        device = img.device

        psf_fft, pad_h, pad_w = self._get_fft(H, W, device)
        img_pad = F.pad(img, (0, pad_w - W, 0, pad_h - H), mode="constant", value=0.0)

        degraded = torch.fft.ifft2(torch.fft.fft2(img_pad) * psf_fft).real
        degraded = degraded[:, :, :H, :W].clamp(min=0.0)
        return degraded

    def backproject(self, residual):
        residual = _ensure_1ch(residual)
        _, _, H, W = residual.shape
        device = residual.device

        psf_fft, pad_h, pad_w = self._get_fft(H, W, device)
        res_pad = F.pad(residual, (0, pad_w - W, 0, pad_h - H), mode="constant", value=0.0)

        back = torch.fft.ifft2(torch.fft.fft2(res_pad) * torch.conj(psf_fft)).real
        back = back[:, :, :H, :W]
        return back


# =========================================================================
# PriorNet: Learned Bayesian Prior Gradient
# =========================================================================
# PriorNet 不预测图像，而是近似 MAP 中 Bayesian Prior 的梯度：
#   PriorNet(x) ≈ ∇_x log p(x)
#
# 输出是 prior gradient g_prior，用于 MAP 更新中的 prior term。
# 这与 score-based 模型中的 score function 概念一致。
# =========================================================================
class PriorNet(nn.Module):
    """
    Learned Bayesian Prior Gradient Network.

    Maps current estimate x_k → prior gradient g_prior ≈ ∇log p(x_k).

    Architecture: Lightweight CNN with residual connection.
    The output is the gradient of the log-prior, NOT a residual image.
    """
    def __init__(self, in_ch=1, hidden_ch=32, num_layers=4):
        super().__init__()
        self.in_ch = in_ch
        self.hidden_ch = hidden_ch

        # Input projection
        self.conv_in = nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False)
        self.gn_in = _make_gn(hidden_ch)

        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(
                nn.Sequential(
                    nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
                    _make_gn(hidden_ch),
                    nn.LeakyReLU(0.1, inplace=True),
                )
            )

        # Output projection: hidden_ch → 1 (gradient field)
        self.conv_out = nn.Conv2d(hidden_ch, in_ch, 3, padding=1, bias=True)

    def forward(self, x):
        x = _ensure_1ch(x)

        # Feature extraction
        feat = F.leaky_relu(self.gn_in(self.conv_in(x)), 0.1, inplace=True)

        for layer in self.hidden_layers:
            feat = layer(feat)

        # Output: prior gradient (same shape as input)
        g_prior = self.conv_out(feat)
        return g_prior


# =========================================================================
# MAP Block: 单步 MAP 迭代展开块（v5: BID 同款 Landweber 数据步）
# =========================================================================
# 2026-10-18 关键发现：官方 BID（Hofmeister 2024）的更新不是乘性 EM，而是
#   Landweber 加性迭代：  x <- x - k*(P(x) - y),  k=1，每步正性截断
#   （deconvolve_image.py 源码：img_decon -= k*(img_decon_con - img)）。
# 收敛率 (1-|P_hat|^2)^k，等效步长 1.0——是 v3 加性 GD（alpha 学到 ~0.21）
#   的 ~5 倍/步，且通量守恒特性好（不会像 EM 比值更新那样抬背景）。
# v5 每步：
#   1. Landweber 数据步:  x_lw = clamp(x - k*(P(x) - y), min=0)   k 可学习, init 1.0
#   2. 学习先验（加性小步）: x_{k+1} = x_lw - beta*PriorNet(x_lw)
# =========================================================================
class MAPBlock(nn.Module):
    """
    Single MAP iteration block (v5: BID-style Landweber data step).

        x_lw = clamp(x - k*(P(x) - y), min=0)   # k learnable, init 1.0
        x_next = x_lw - beta*PriorNet(x_lw)     # learned prior, additive small step
    """
    def __init__(self, cfg, use_prior=True):
        super().__init__()
        self.cfg = cfg
        self.use_prior = use_prior

        # Landweber 步长 k（log 域，init 0 -> k=1.0，与 BID 一致）
        self.log_k = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        if self.use_prior:
            self.log_beta = nn.Parameter(torch.tensor(-4.6, dtype=torch.float32))
            self.prior_net = PriorNet(
                in_ch=1,
                hidden_ch=cfg.MODEL.BASE_CHANNELS * 2,  # 32
                num_layers=4,
            )

    def forward(self, x, y, psf_module):
        """
        Args:
            x: current estimate (B, 1, H, W)
            y: observed blurred image (B, 1, H, W)
            psf_module: PSFDegradationModule for P(x) and P^T(residual)
        Returns:
            x_next: updated estimate
        """
        x = _ensure_1ch(x)
        y = _ensure_1ch(y)

        # ---- Landweber (BID) data step: x <- clamp(x - k*(P(x) - y), 0) ----
        k = torch.exp(self.log_k)
        x_lw = x - k * (psf_module(x) - y)
        x_lw = torch.clamp(x_lw, min=0.0)

        # ---- learned prior: additive small step ----
        if self.use_prior:
            g_prior = self.prior_net(x_lw)
            beta = torch.exp(self.log_beta)
            x_next = x_lw - beta * g_prior
        else:
            x_next = x_lw

        return x_next


# =========================================================================
# 主网络：MAPIterativeNet — Learned MAP Solver
# =========================================================================
class MAPIterativeNet(nn.Module):
    """
    Bayesian MAP-based Deep Unfolding Network.

    Approximates the MAP solution via learned gradient descent:
        x* = argmin_x { D(x, y) + λ * R(x) }

    where D(x, y) is the data fidelity term (PSF consistency) and
    R(x) is the learned Bayesian prior whose gradient is PriorNet(x).

    The network performs NUM_ITER iterations of MAP gradient descent,
    with per-iteration learnable step sizes and a shared or unshared
    prior gradient network.

    Interpretation: Learned MAP Solver, NOT Learned BID.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.num_iter = cfg.MODEL.NUM_BID_ITER  # reuse same config key
        self.use_prior = getattr(cfg.MODEL, "USE_MAP_PRIOR", True)
        self.psf_degrade = PSFDegradationModule(cfg)

        # Build MAP iteration blocks
        self.map_blocks = nn.ModuleList([
            MAPBlock(cfg, use_prior=self.use_prior)
            for _ in range(self.num_iter)
        ])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, y):
        y = _ensure_1ch(y)
        x = y.clone()  # Initialize from observation

        for i in range(self.num_iter):
            x = self.map_blocks[i](x, y, self.psf_degrade)

        return F.softplus(x)  # Non-negative output
