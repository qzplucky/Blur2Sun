import torch
import torch.nn as nn
import torch.nn.functional as F
from astropy.io import fits
import numpy as np


class Blur2BlurPhysicsLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # 使用基础的 L1Loss
        self.l1_loss = nn.L1Loss()

        # ===== PSF 物理算子加载 =====
        psf_tensor = self._load_and_preprocess_psf()
        self.register_buffer('psf_tensor', psf_tensor)
        self.K = psf_tensor.shape[-1]
        default_H = int(self.cfg.PREPROCESS.ORIGINAL_SIZE)
        default_W = int(self.cfg.PREPROCESS.ORIGINAL_SIZE)
        self._precompute_psf_fft_for(default_H, default_W)

    def _load_and_preprocess_psf(self):
        current_band = str(self.cfg.BASE.CURRENT_BAND)
        psf_file = self.cfg.AIA_BAND_PARAMS[current_band]['PSF_FILE']
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
                pad = (target_size - h) // 2
                psf_data = np.pad(psf_data, pad, mode='constant')
            s = np.sum(psf_data)
            if s == 0:
                raise RuntimeError("PSF 总和为0！")
            return torch.from_numpy(psf_data / s).unsqueeze(0).unsqueeze(0).float()
        except Exception as e:
            raise RuntimeError(f"加载PSF失败: {psf_file}, 错误: {e}")

    def _precompute_psf_fft_for(self, H, W):
        psf = self.psf_tensor
        K = psf.shape[-1]
        pad_h, pad_w = H + K - 1, W + K - 1
        psf_pad = torch.zeros((1, 1, pad_h, pad_w), dtype=psf.dtype)
        start_h, start_w = pad_h // 2 - K // 2, pad_w // 2 - K // 2
        psf_pad[:, :, start_h:start_h + K, start_w:start_w + K] = psf
        psf_fft = torch.fft.fft2(torch.fft.ifftshift(psf_pad, dim=(-2, -1)))
        self.register_buffer('psf_fft', psf_fft)
        self.pad_h, self.pad_w = pad_h, pad_w
        self._precomputed_H, self._precomputed_W = H, W

    def _apply_psf_degradation(self, img):
        B, C, H, W = img.shape
        device = img.device
        if hasattr(self, '_precomputed_H') and self._precomputed_H == H and self._precomputed_W == W:
            psf_fft = self.psf_fft.to(device)
            pad_h, pad_w = self.pad_h, self.pad_w
        else:
            psf = self.psf_tensor.to(device)
            K = psf.shape[-1]
            pad_h, pad_w = H + K - 1, W + K - 1
            psf_pad = torch.zeros((1, 1, pad_h, pad_w), dtype=psf.dtype, device=device)
            psf_pad[:, :, pad_h//2-K//2 : pad_h//2-K//2+K, pad_w//2-K//2 : pad_w//2-K//2+K] = psf
            psf_fft = torch.fft.fft2(torch.fft.ifftshift(psf_pad, dim=(-2, -1)))

        img_pad = F.pad(img, (0, pad_w - W, 0, pad_h - H), mode='constant', value=0.0)
        return torch.fft.ifft2(torch.fft.fft2(img_pad) * psf_fft).real[:, :, :H, :W].clamp(min=0.0)

    def forward(self, y1, y2, x1_pred, x2_pred):
        loss_dict = {}

        # 输入安全检查
        if torch.isnan(x1_pred).any() or torch.isnan(x2_pred).any():
            loss_dict['total_loss'] = torch.tensor(float('nan'), device=x1_pred.device)
            return loss_dict

        # PSF 退化后计算 L1 交叉一致性（全程 FP32）
        x1_degraded = self._apply_psf_degradation(x1_pred.float())
        x2_degraded = self._apply_psf_degradation(x2_pred.float())

        loss_1 = self.l1_loss(x1_degraded, y2)
        loss_2 = self.l1_loss(x2_degraded, y1)

        total_loss = (loss_1 + loss_2) / 2.0
        loss_dict['total_loss'] = total_loss

        return loss_dict


