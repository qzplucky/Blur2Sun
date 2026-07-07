import os
import sys
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import random
import gc
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import copy

from config import get_config, create_output_dirs
from data_loader import AIABlur2BlurDataset
from loss import Blur2BlurPhysicsLoss
from model import PhysicsAwareIterativeNet
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
        print(f"  ⚠️  Epoch {epoch+1}: 跳过 {nan_count} 个 NaN batch")
    return avg_loss


@torch.no_grad()
def validate(model, loader, criterion, device, epoch, writer, cfg):
    model.eval()
    total_loss = 0.0
    vis_done = False

    total_inv_ssim = 0.0       # 反一致性 SSIM（退化预测 vs 观测）
    total_flux_err = 0.0       # 通量误差
    total_res_mean = 0.0       # 残差均值 |degraded_pred - obs|
    total_res_std = 0.0        # 残差标准差
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

        # ===== 反一致性指标：退化后的预测 vs 观测 =====
        # 注意：SSIM 衡量的是"反退化后是否像原模糊图"，而非反卷积质量
        # 反卷积后图像会比模糊图更锐利，SSIM 天生不会太高
        x1_degraded_for_eval = criterion._apply_psf_degradation(x1_pred)
        x1_degraded_norm = normalize(x1_degraded_for_eval)
        y2_norm = normalize(y2)
        _, inv_ssim, flux_err = compute_all_metrics(x1_degraded_norm, y2_norm)
        total_inv_ssim += inv_ssim
        total_flux_err += flux_err

        # ===== 残差统计（更直接的反卷积质量指标） =====
        residual = (x1_degraded_for_eval - y2).abs()
        total_res_mean += residual.mean().item()
        total_res_std += residual.std().item()

        num_batches += 1

        if epoch % 5 == 0 and not vis_done:
            visualize_blur2blur_result(
                epoch, y1[0].cpu(), y2[0].cpu(),
                x1_pred[0].cpu(), x2_pred[0].cpu(), cfg, '_blur2blur'
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
# 消融实验定义：PSF / Detail / 组合
# ITER 始终开启（没有迭代则全流程退化到 trivial），自适应迭代关闭
# =========================================================================
ABLATION_GROUPS = {
    # ===== 原有模块消融（T=10, learnable step） =====
    "full": {
        "dir": "ab_full",
        "desc": "Full Model (PSF + Detail)",
        "flags": {"USE_PSF": True, "USE_DETAIL_BLOCK": True},
    },
    "no_psf": {
        "dir": "ab_no_psf",
        "desc": "No PSF (Detail only)",
        "flags": {"USE_PSF": False, "USE_DETAIL_BLOCK": True},
    },
    "no_detail": {
        "dir": "ab_no_detail",
        "desc": "No Detail (PSF only)",
        "flags": {"USE_PSF": True, "USE_DETAIL_BLOCK": False},
    },
    "no_psf_detail": {
        "dir": "ab_no_psf_detail",
        "desc": "No PSF + No Detail (Pure iterative)",
        "flags": {"USE_PSF": False, "USE_DETAIL_BLOCK": False},
    },

    # ===== T 数量消融：T=5 vs T=3（Full Model, learnable step） =====
    "t5_full": {
        "dir": "ab_t5_full",
        "desc": "T=5 Full Model (Learnable Step)",
        "flags": {"USE_PSF": True, "USE_DETAIL_BLOCK": True,
                  "NUM_BID_ITER": 5, "LEARNABLE_STEP": True},
    },
    "t3_full": {
        "dir": "ab_t3_full",
        "desc": "T=3 Full Model (Learnable Step)",
        "flags": {"USE_PSF": True, "USE_DETAIL_BLOCK": True,
                  "NUM_BID_ITER": 3, "LEARNABLE_STEP": True},
    },

    # ===== 步长消融：可学习 vs 固定步长（T=5, Full Model） =====
    "t5_fixed_step": {
        "dir": "ab_t5_fixed",
        "desc": "T=5 Fixed Step",
        "flags": {"USE_PSF": True, "USE_DETAIL_BLOCK": True,
                  "NUM_BID_ITER": 5, "LEARNABLE_STEP": False},
    },
}


def run_experiment(band, ablation_type, cfg_template):
    """ 消融实验：模块组合 / T 迭代次数 / 步长学习 """
    group = ABLATION_GROUPS[ablation_type]
    cfg = copy.deepcopy(cfg_template)
    cfg.BASE.CURRENT_BAND = band

    # ====== 消融模块开关 ======
    cfg.MODEL.USE_PSF = group["flags"]["USE_PSF"]
    cfg.MODEL.USE_DETAIL_BLOCK = group["flags"]["USE_DETAIL_BLOCK"]
    cfg.MODEL.USE_PRIOR_BLOCK = group["flags"]["USE_DETAIL_BLOCK"]

    # ====== T 迭代次数（NUM_BID_ITER） ======
    if "NUM_BID_ITER" in group["flags"]:
        cfg.MODEL.NUM_BID_ITER = group["flags"]["NUM_BID_ITER"]

    # ====== 步长可学习开关 ======
    if "LEARNABLE_STEP" in group["flags"]:
        cfg.MODEL.LEARNABLE_STEP_SIZE = group["flags"]["LEARNABLE_STEP"]

    # ====== ITER 始终开启，自适应迭代关闭 ======
    cfg.MODEL.USE_ITER = True
    cfg.MODEL.ADAPTIVE_ITER = False

    cfg.BASE.EXP_NAME = group["dir"]
    create_output_dirs(cfg)
    device = cfg.BASE.DEVICE
    print(f"\n🧪 [Ablation] Band: {band}Å | {group['desc']} | Flags: {group['flags']}")

    # 数据加载
    train_set = AIABlur2BlurDataset(cfg, mode='train')
    val_set = AIABlur2BlurDataset(cfg, mode='val')
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 模型 + 损失 + 优化器
    model = PhysicsAwareIterativeNet(cfg).to(device)
    criterion = Blur2BlurPhysicsLoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR,
                            weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.EPOCHS,
                                                      eta_min=cfg.TRAIN.MIN_LR)

    writer = SummaryWriter(log_dir=cfg.BASE.LOG_DIR)
    best_val_loss = float('inf')
    final_metrics = None

    early_stop_patience = 5
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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_metrics': metrics,
                'best_val_loss': best_val_loss,
                'config': cfg,
            }, os.path.join(cfg.BASE.MODEL_DIR, "best_model.pth"))
        else:
            early_stop_counter += 1
            print(f"⚠️  Early Stop Counter: {early_stop_counter}/{early_stop_patience} "
                  f"(best val_loss: {best_val_loss:.6f})")
            if early_stop_counter >= early_stop_patience:
                print(f"\n🛑 Early Stop Triggered! Best Val Loss: {best_val_loss:.6f}")
                early_stop = True

        print(f"Epoch {epoch+1}/{cfg.TRAIN.EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Inv-SSIM: {metrics['inv_ssim']:.4f} | "
              f"ResMean: {metrics['res_mean']:.4f}")

    writer.close()

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
        print(f"🧹 GPU 清理后显存占用: {used_gib:.1f} GiB / {total_gib:.1f} GiB")
    except Exception:
        pass

    print(f"✅ {group['desc']} Done. Best Val Loss: {best_val_loss:.6f} | "
          f"Inv-SSIM: {final_metrics['inv_ssim']:.4f} | "
          f"ResMean: {final_metrics['res_mean']:.4f} ± {final_metrics['res_std']:.4f}")
    return {
        'band': band, 'exp_name': group['dir'], 'desc': group['desc'],
        'val_loss': best_val_loss,
        'inv_ssim': final_metrics['inv_ssim'],
        'flux_err': final_metrics['flux_err'],
        'res_mean': final_metrics['res_mean'],
        'res_std': final_metrics['res_std'],
    }


def main():
    cfg_template = get_config()
    set_seed(cfg_template.BASE.SEED)

    # ====== 选择要运行的消融实验组 ======
    # 可选值：
    #   "t5_full","t3_full"         → T 迭代次数消融（T=5 vs T=3）
    #   "t5_full","t5_fixed_step"  → 步长学习消融（可学习 vs 固定）
    #   "full","no_detail"          → 原有模块消融（Full vs No Detail）
    ablation_list = [
        "t5_full", "t3_full",         # T 迭代次数消融
        "t5_full", "t5_fixed_step",   # 步长学习消融（t5_full 共用）
    ]
    # 去重：t5_full 只需要训练一次
    ablation_list = list(dict.fromkeys(ablation_list))

    bands_to_train = [94]  # 只在94波段进行消融实验
    all_results = []

    print(f"{'='*60}")
    print(f"消融实验 - 启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"消融组数: {len(ablation_list)} | 波段数: {len(bands_to_train)} | 总实验: {len(ablation_list)*len(bands_to_train)}")
    print(f"{'='*60}")
    for a in ablation_list:
        g = ABLATION_GROUPS[a]
        print(f"  {a:<18} → {g['desc']:<40}  flags={g['flags']}")
    print()

    for band in bands_to_train:
        print(f"\n{'='*60}")
        print(f"🔥 开始波段: {band}Å 消融实验")
        print(f"{'='*60}")

        for ablation_type in ablation_list:
            res = run_experiment(band, ablation_type, cfg_template)
            all_results.append(res)
            gc.collect()
            torch.cuda.empty_cache()

    # 打印结果表格
    print("\n\n" + "="*110)
    print("📊 网络模块消融实验最终结果（ITER 始终开启，自适应迭代关闭）")
    print("  注：Inv-SSIM 衡量退化后预测 vs 观测的一致性，非反卷积质量")
    print("      ResMean/ResStd 衡量残差统计，更直接反映反卷积精度")
    print("="*110)
    header = (f"{'Experiment':<22} | {'Desc':<28} | {'Band':<5} | "
              f"{'ValLoss':<10} | {'Inv-SSIM':<10} | {'FluxErr':<8} | "
              f"{'ResMean':<8} | {'ResStd':<8}")
    print(header)
    print("-" * 110)

    for res in all_results:
        print(f"{res['exp_name']:<22} | {res['desc']:<28} | {res['band']:<5} | "
              f"{res['val_loss']:<10.6f} | {res['inv_ssim']:<10.4f} | "
              f"{res['flux_err']:<8.4f} | {res['res_mean']:<8.4f} | "
              f"{res['res_std']:<8.4f}")

    # 按消融组汇总
    print("\n📊 按消融组汇总（所有波段平均）")
    print("-" * 80)
    print(f"{'Experiment':<22} | {'Desc':<28} | {'Avg Inv-SSIM':<12} | "
          f"{'Avg ResMean':<12} | {'Avg FluxErr':<10}")
    print("-" * 80)
    for a_type in ablation_list:
        group_results = [r for r in all_results if r['exp_name'] == ABLATION_GROUPS[a_type]['dir']]
        if group_results:
            avg_inv_ssim = np.mean([r['inv_ssim'] for r in group_results])
            avg_res_mean = np.mean([r['res_mean'] for r in group_results])
            avg_flux = np.mean([r['flux_err'] for r in group_results])
            print(f"{a_type:<22} | {ABLATION_GROUPS[a_type]['desc']:<28} | "
                  f"{avg_inv_ssim:<12.4f} | {avg_res_mean:<12.6f} | "
                  f"{avg_flux:<10.4f}")


if __name__ == "__main__":
    main()