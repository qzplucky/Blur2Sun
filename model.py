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
# PSF 物理退化算子
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
        """向上取整到最近的 2 的幂（FFT 高效运算的关键）"""
        p = 1
        while p < x:
            p <<= 1
        return p

    def _precompute_psf_fft_for(self, H, W):
        psf = self.psf_tensor
        K = self.K

        pad_h = H + K - 1
        pad_w = W + K - 1

        # 向上取整到 2 的幂，避免 cuFFT 对非 2 幂尺寸使用 Bluestein 算法（内存暴增）
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

        # 向上取整到 2 的幂
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
            # psf_fft 已经是 register_buffer，和模型在同一设备，避免不必要的 .to()
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
# 固定高通边缘提取（不学习）
# =========================================================================
class FixedLaplacianEdge(nn.Module):
    def __init__(self):
        super().__init__()
        lap = torch.tensor(
            [[0.0, 1.0, 0.0],
             [1.0, -4.0, 1.0],
             [0.0, 1.0, 0.0]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("lap_kernel", lap)

    def forward(self, x):
        x = _ensure_1ch(x)
        return torch.abs(F.conv2d(x, self.lap_kernel, padding=1))


# =========================================================================
# 细节保真补偿块：固定高频 + 轻量残差映射
# =========================================================================
class DetailPreservationBlock(nn.Module):
    def __init__(self, in_ch=4, hidden_ch=16):
        super().__init__()
        self.edge_extractor = FixedLaplacianEdge()

        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1, bias=False),
            _make_gn(hidden_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(hidden_ch, hidden_ch, 3, padding=1, bias=False),
            _make_gn(hidden_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(hidden_ch, 1, 1, bias=True)
        )

    def forward(self, prior_in):
        o_dc = prior_in[:, 0:1, :, :]
        residual = prior_in[:, 1:2, :, :]
        backproj = prior_in[:, 2:3, :, :]

        edge = self.edge_extractor(o_dc)
        feat = torch.cat([o_dc, residual, backproj, edge], dim=1)

        delta = self.fuse(feat)
        return delta


# 兼容旧名字
HighFreqCompensationPriorBlock = DetailPreservationBlock



# =========================================================================
# BID 单步展开块
# =========================================================================
class BIDBlock(nn.Module):
    def __init__(self, cfg, k=1.0, dropout_rate=0.25):
        super().__init__()
        self.cfg = cfg

        self.use_psf = getattr(cfg.MODEL, "USE_PSF", True)

        self.use_detail = getattr(
            cfg.MODEL,
            "USE_DETAIL_BLOCK",
            getattr(cfg.MODEL, "USE_PRIOR_BLOCK", True)
        )

        self.learnable_step = getattr(cfg.MODEL, "LEARNABLE_STEP_SIZE", True)
        if self.learnable_step:
            self.k_base_raw = nn.Parameter(torch.tensor(float(k), dtype=torch.float32))
        else:
            self.register_buffer("k_base_raw", torch.tensor(float(k), dtype=torch.float32))

        if self.use_detail:
            self.detail_block = DetailPreservationBlock(in_ch=4, hidden_ch=16)

        # 最小可学路径：当 PSF 和 Detail 都关闭时提供基本的可训练能力
        if not self.use_psf and not self.use_detail:
            self.minimal_conv = nn.Sequential(
                nn.Conv2d(1, 8, 1, bias=False),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(8, 1, 1, bias=False),
            )
        else:
            self.minimal_conv = None

    def forward(self, o, i, psf_module):
        o = _ensure_1ch(o)
        i = _ensure_1ch(i)

        if not self.use_psf:
            if self.minimal_conv is not None:
                delta = self.minimal_conv(o)
                o_dc = o + 0.1 * torch.tanh(delta)
            else:
                o_dc = o
            residual = o_dc - o
            backproj = residual
        else:
            o_degraded = psf_module(o)
            residual = o_degraded - i
            backproj = psf_module.backproject(residual)

            k_base = torch.clamp(self.k_base_raw, min=0.01, max=0.5)
            o_dc = o - k_base * backproj

        prior_in = torch.cat([o_dc, residual, backproj], dim=1)

        detail_update = torch.zeros_like(o_dc)
        if self.use_detail:
            detail_update = self.detail_block(prior_in)
            detail_update = 0.2 * torch.tanh(detail_update)

        o_next = o_dc + detail_update
        return o_next


# =========================================================================
# 主网络：PhysicsAwareIterativeNet（纯迭代展开，无 UNet backbone）
# =========================================================================
class PhysicsAwareIterativeNet(nn.Module):
    def __init__(self, cfg, dropout_rate=0.25):
        super().__init__()
        self.cfg = cfg

        self.use_iter = cfg.MODEL.USE_ITER
        self.num_bid_iter = cfg.MODEL.NUM_BID_ITER if self.use_iter else 0
        self.psf_degrade = PSFDegradationModule(cfg)

        # 自适应迭代：残差收敛后提前终止
        self.use_adaptive = getattr(cfg.MODEL, "ADAPTIVE_ITER", True)
        self.adaptive_min_iter = getattr(cfg.MODEL, "ADAPTIVE_MIN_ITER", 4)
        self.adaptive_threshold = getattr(cfg.MODEL, "ADAPTIVE_THRESHOLD", 0.001)
        self.adaptive_patience = getattr(cfg.MODEL, "ADAPTIVE_PATIENCE", 2)

        if self.use_iter:
            self.bid_blocks = nn.ModuleList([
                BIDBlock(cfg, k=1.0, dropout_rate=dropout_rate)
                for _ in range(self.num_bid_iter)
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
        x = y.clone()

        if self.use_iter:
            if self.use_adaptive:
                # 自适应迭代：残差收敛后提前终止
                prev_residual = None
                stale_count = 0

                for i in range(self.num_bid_iter):
                    x = self.bid_blocks[i](x, y, self.psf_degrade)

                    # 收敛检测：至少在 min_iter 次迭代后才开始判断
                    if i >= self.adaptive_min_iter - 1:
                        degraded = self.psf_degrade(x)
                        current_res = torch.mean(torch.abs(degraded - y)).item()

                        if prev_residual is not None:
                            rel_improvement = (prev_residual - current_res) / max(prev_residual, 1e-8)
                            if rel_improvement < self.adaptive_threshold:
                                stale_count += 1
                            else:
                                stale_count = 0

                        prev_residual = current_res

                        if stale_count >= self.adaptive_patience:
                            break
            else:
                for i in range(self.num_bid_iter):
                    x = self.bid_blocks[i](x, y, self.psf_degrade)

        return F.softplus(x)


# 保留旧名字兼容（但标注为 deprecated）
PhysicsAwareUNet = PhysicsAwareIterativeNet