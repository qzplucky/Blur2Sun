import torch
import torch.nn.functional as F
from math import exp

def calculate_flux_error(pred, target):
    """
    计算通量相对误差 (Relative Flux Error)
    Formula: |Sum(Pred) - Sum(Target)| / Sum(Target)
    """
    flux_pred = torch.sum(pred, dim=(-2, -1))
    flux_targ = torch.sum(target, dim=(-2, -1))
    # 防止除零
    rel_error = torch.abs(flux_pred - flux_targ) / (torch.abs(flux_targ) + 1e-7)
    return rel_error.mean().item()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    计算结构相似性 (SSIM)
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average).item()

def calculate_psnr(pred, target):
    """
    计算峰值信噪比 (PSNR)
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100
    # 假设图像归一化到 [0, 1]，max_pixel = 1
    return 10 * torch.log10(1.0 / mse).item()

def compute_all_metrics(x_pred, y_gt):
    """
    一次性计算所有指标
    """
    # 确保是单通道
    if x_pred.shape[1] != 1: x_pred = x_pred[:, :1, :, :]
    if y_gt.shape[1] != 1: y_gt = y_gt[:, :1, :, :]
        
    psnr = calculate_psnr(x_pred, y_gt)
    ssim = calculate_ssim(x_pred, y_gt)
    flux_err = calculate_flux_error(x_pred, y_gt)
    
    return psnr, ssim, flux_err