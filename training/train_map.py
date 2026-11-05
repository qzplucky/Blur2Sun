# Blur2Sun v3 训练入口（修正 PSF: 官方 rebin_psf 2048; Minor-11）
#!/usr/bin/env python3
"""Blur2Sun v3 — MAP 训练统一入口。
用法: python3 train_map.py --config configs/config_map_12s.py --gpu 0 --bands 94 131
PSF: 官方 rebin_psf 2048（回应 Minor-11）；EXP_NAME 由配置文件决定，脚本内禁止覆盖。
"""

import os
import sys
import datetime
# ---- 在导入 torch 前解析 --gpu / --config ----
_gpu = '0'
_config = 'configs/config_map_12s.py'
if '--gpu' in sys.argv:
    _gpu = sys.argv[sys.argv.index('--gpu') + 1]
if '--config' in sys.argv:
    _config = sys.argv[sys.argv.index('--config') + 1]
os.environ["CUDA_VISIBLE_DEVICES"] = _gpu
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import random
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import copy

import importlib.util as _ilutil
_spec = _ilutil.spec_from_file_location("cfg_mod", _config)
_cfgmod = _ilutil.module_from_spec(_spec)
sys.modules["cfg_mod"] = _cfgmod
_spec.loader.exec_module(_cfgmod)
get_config, create_output_dirs = _cfgmod.get_config, _cfgmod.create_output_dirs
from data_loader import AIABlur2BlurDataset
from loss import Blur2BlurPhysicsLoss
from model2_map import MAPIterativeNet
from visualization import visualize_blur2blur_result, plot_loss_curve
from metrics import compute_all_metrics


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0.0
    nan_count = 0
    valid_batches = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for batch in pbar:
        y1 = batch['y1'].to(device, non_blocking=True)
        y2 = batch['y2'].to(device, non_blocking=True)

        if torch.isnan(y1).any() or torch.isnan(y2).any() or torch.isinf(y1).any() or torch.isinf(y2).any():
            nan_count += 1
            continue

        optimizer.zero_grad()

        x1_pred = model(y1)
        x2_pred = model(y2)

        if torch.isnan(x1_pred).any() or torch.isnan(x2_pred).any():
            nan_count += 1
            continue

        loss_dict = criterion(y1, y2, x1_pred, x2_pred)
        loss_val = loss_dict['total_loss']

        if torch.isnan(loss_val) or torch.isinf(loss_val):
            nan_count += 1
            continue

        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_val.item()
        valid_batches += 1
        pbar.set_postfix({'Loss': f"{loss_val.item():.4f}"})

        del x1_pred, x2_pred, loss_dict, loss_val
        if valid_batches % 10 == 0:
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(valid_batches, 1)
    if nan_count > 0:
        print(f"  [WARN] Epoch {epoch+1}: 跳过 {nan_count} 个 NaN batch")
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, cfg):
    model.eval()
    total_loss = 0.0
    vis_done = False

    total_inv_ssim = 0.0
    total_flux_err = 0.0
    total_res_mean = 0.0
    total_res_std = 0.0
    num_batches = 0

    def normalize(img):
        img_min = torch.amin(img, dim=(1, 2), keepdim=True)
        img_max = torch.amax(img, dim=(1, 2), keepdim=True)
        return (img - img_min) / (img_max - img_min + 1e-8)

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val] ", leave=False)
    for batch in pbar:
        y1 = batch['y1'].to(device, non_blocking=True)
        y2 = batch['y2'].to(device, non_blocking=True)

        x1_pred = model(y1)
        x2_pred = model(y2)

        loss_dict = criterion(y1, y2, x1_pred, x2_pred)
        loss_val = loss_dict['total_loss'].item()
        if np.isnan(loss_val) or np.isinf(loss_val):
            continue
        total_loss += loss_val

        # 反一致性指标
        x1_degraded_for_eval = criterion._apply_psf_degradation(x1_pred)
        x1_degraded_norm = normalize(x1_degraded_for_eval)
        y2_norm = normalize(y2)
        _, inv_ssim, flux_err = compute_all_metrics(x1_degraded_norm, y2_norm)
        total_inv_ssim += inv_ssim
        total_flux_err += flux_err

        # 残差统计
        residual = (x1_degraded_for_eval - y2).abs()
        total_res_mean += residual.mean().item()
        total_res_std += residual.std().item()

        num_batches += 1

        if epoch % 5 == 0 and not vis_done:
            visualize_blur2blur_result(
                epoch, y1[0].cpu(), y2[0].cpu(),
                x1_pred[0].cpu(), x2_pred[0].cpu(), cfg, '_map_blur2blur'
            )
            vis_done = True

    avg_loss = total_loss / len(loader)
    avg_inv_ssim = total_inv_ssim / num_batches
    avg_flux_err = total_flux_err / num_batches
    avg_res_mean = total_res_mean / num_batches
    avg_res_std = total_res_std / num_batches

    if writer:
        writer.add_scalar('Loss/Val', avg_loss, epoch)
        writer.add_scalar('Metrics/Inv_SSIM', avg_inv_ssim, epoch)
        writer.add_scalar('Metrics/Flux_Err', avg_flux_err, epoch)
        writer.add_scalar('Metrics/Res_Mean', avg_res_mean, epoch)
        writer.add_scalar('Metrics/Res_Std', avg_res_std, epoch)

    return avg_loss, {
        'inv_ssim': avg_inv_ssim,
        'flux_err': avg_flux_err,
        'res_mean': avg_res_mean,
        'res_std': avg_res_std,
    }


# =========================================================================
# MAP 实验定义（仅 Full MAP，不做消融）
# =========================================================================
MAP_EXPERIMENT = {
    "dir": "map_full_12s",
    "desc": "Full MAP (PSF + PriorNet)",
    "flags": {"USE_MAP_PRIOR": True},
}


def run_map_experiment(band, cfg_template):
    """MAP 网络训练（仅 Full MAP）"""
    group = MAP_EXPERIMENT
    cfg = copy.deepcopy(cfg_template)
    cfg.BASE.CURRENT_BAND = band

    # MAP 开关：尊重配置文件的值（消融变体通过配置关闭先验）
    cfg.MODEL.USE_MAP_PRIOR = bool(getattr(cfg.MODEL, "USE_MAP_PRIOR", True))

    assert cfg.BASE.EXP_NAME.startswith("map_full_"), "EXP_NAME 安全断言失败"
    create_output_dirs(cfg)
    device = cfg.BASE.DEVICE
    print(f"\n[MAP] Band: {band}A | {group['desc']}")

    # 数据加载
    train_set = AIABlur2BlurDataset(cfg, mode='train')
    val_set = AIABlur2BlurDataset(cfg, mode='val')
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 模型 + 损失 + 优化器
    model = MAPIterativeNet(cfg).to(device)
    if getattr(cfg.MODEL, "FREEZE_K", False):
        n_frozen = 0
        for name, p_ in model.named_parameters():
            if "log_k" in name:
                p_.requires_grad_(False)
                n_frozen += 1
        print(f"[消融] FREEZE_K=True: 冻结 {n_frozen} 个 log_k 参数")
    criterion = Blur2BlurPhysicsLoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.EPOCHS,
                                                      eta_min=cfg.TRAIN.MIN_LR)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量: {total_params:,} total, {trainable_params:,} trainable")

    writer = SummaryWriter(log_dir=cfg.BASE.LOG_DIR)
    best_val_loss = float('inf')
    final_metrics = None

    early_stop_patience = cfg.TRAIN.EARLY_STOP_PATIENCE
    early_stop_counter = 0
    early_stop = False

    for epoch in range(cfg.TRAIN.EPOCHS):
        if early_stop:
            break

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                      device, epoch, writer)
        val_loss, metrics = validate(model, val_loader, criterion, device,
                                      epoch, writer, cfg)
        scheduler.step()

        final_metrics = metrics

        # 打印 alpha/beta 学习值 (exp for positivity)
        alpha_vals = [torch.exp(b.log_k).item() for b in model.map_blocks]
        beta_vals = [torch.exp(b.log_beta).item() if hasattr(b, 'log_beta') else 0.0
                     for b in model.map_blocks]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_metrics': metrics,
                'best_val_loss': best_val_loss,
                'config': cfg,
                'log_k_vals': [b.log_k.item() for b in model.map_blocks],
                'log_beta_vals': [b.log_beta.item() if hasattr(b, 'log_beta') else 0.0
                                  for b in model.map_blocks],
            }, os.path.join(cfg.BASE.MODEL_DIR, "best_model.pth"))
        else:
            early_stop_counter += 1
            print(f"[EarlyStop] Counter: {early_stop_counter}/{early_stop_patience} "
                  f"(best val_loss: {best_val_loss:.6f})")
            if early_stop_counter >= early_stop_patience:
                print(f"\n[EarlyStop] Triggered! Best Val Loss: {best_val_loss:.6f}")
                early_stop = True

        print(f"Epoch {epoch+1}/{cfg.TRAIN.EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Inv-SSIM: {metrics['inv_ssim']:.4f} | "
              f"ResMean: {metrics['res_mean']:.4f} | "
              f"Alpha: [{alpha_vals[0]:.4f}..{alpha_vals[-1]:.4f}] | "
              f"Beta: [{beta_vals[0]:.4f}..{beta_vals[-1]:.4f}]")

    writer.close()

    # 打印最终学习到的 alpha/beta
    print(f"\n[MAP] 最终学习到的 step sizes:")
    for i, (a, bt) in enumerate(zip(alpha_vals, beta_vals)):
        print(f"  Iter {i}: alpha={a:.6f}, beta={bt:.6f}")

    # === 显存释放 ===
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)
    model.to('cpu')
    criterion.to('cpu')
    del model, criterion, optimizer, scheduler
    if hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
        train_loader._iterator._shutdown_workers()
    del train_loader, val_loader, train_set, val_set
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_gib = (total_bytes - free_bytes) / (1024**3)
        total_gib = total_bytes / (1024**3)
        print(f"  GPU 清理后显存: {used_gib:.1f} GiB / {total_gib:.1f} GiB")
    except Exception:
        pass

    print(f"[MAP] {group['desc']} Done. Best Val Loss: {best_val_loss:.6f} | "
          f"Inv-SSIM: {final_metrics['inv_ssim']:.4f} | "
          f"ResMean: {final_metrics['res_mean']:.4f} +- {final_metrics['res_std']:.4f}")
    return {
        'band': band, 'exp_name': group['dir'], 'desc': group['desc'],
        'val_loss': best_val_loss,
        'inv_ssim': final_metrics['inv_ssim'],
        'flux_err': final_metrics['flux_err'],
        'res_mean': final_metrics['res_mean'],
        'res_std': final_metrics['res_std'],
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', type=int, nargs='+',
                    default=[94, 131, 171, 193, 211, 304, 335])
    ap.add_argument('--config', default=_config)
    ap.add_argument('--gpu', default=_gpu)
    args = ap.parse_args()

    cfg_template = get_config()
    set_seed(cfg_template.BASE.SEED)

    bands_to_train = args.bands
    all_results = []

    print(f"{'='*60}")
    print(f"Bayesian MAP Deep Unfolding Training")
    print(f"启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"实验: Full MAP (PSF + PriorNet) [12s paired data]")
    print(f"波段: {bands_to_train}")
    print(f"{'='*60}")

    print(f"\n网络解释: Learned MAP Solver (exp step sizes)")
    print(f"  Data term:   g_data = PSF^T(PSF(x) - y)")
    print(f"  Prior term:  g_prior = PriorNet(x)")
    print(f"  Update:      x = x - alpha*g_data - beta*g_prior")
    print()

    for band in bands_to_train:
        print(f"\n{'='*60}")
        print(f"[MAP] 波段: {band}A")
        print(f"{'='*60}")

        res = run_map_experiment(band, cfg_template)
        all_results.append(res)
        gc.collect()
        torch.cuda.empty_cache()

    # 打印结果表格
    print("\n\n" + "="*80)
    print("  Bayesian MAP Deep Unfolding 最终结果")
    print("="*80)
    header = (f"{'Band':<5} | {'ValLoss':<10} | {'Inv-SSIM':<10} | "
              f"{'FluxErr':<8} | {'ResMean':<8} | {'ResStd':<8}")
    print(header)
    print("-" * 80)

    for res in all_results:
        print(f"{res['band']:<5} | {res['val_loss']:<10.6f} | {res['inv_ssim']:<10.4f} | "
              f"{res['flux_err']:<8.4f} | {res['res_mean']:<8.4f} | "
              f"{res['res_std']:<8.4f}")


if __name__ == "__main__":
    main()
