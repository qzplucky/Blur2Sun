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


class Tee:
    """同时输出到控制台和日志文件（同时捕获 stdout 和 stderr）"""
    def __init__(self, filepath, stream_name='stdout'):
        self.file = open(filepath, 'a', encoding='utf-8')
        self.stream_name = stream_name
        self.original = sys.stdout if stream_name == 'stdout' else sys.stderr

    def write(self, message):
        self.original.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.original.flush()
        self.file.flush()

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

        # 输入 NaN 检查
        if torch.isnan(y1).any() or torch.isnan(y2).any() or torch.isinf(y1).any() or torch.isinf(y2).any():
            nan_count += 1
            pbar.set_postfix({'Loss': 'SKIP(NaN input)'})
            continue

        optimizer.zero_grad()

        # 全程 FP32，避免 FP16 FFT 溢出（NUM_BID_ITER=10，每次迭代 2 次 FFT）
        x1_pred = model(y1)
        x2_pred = model(y2)

        # 输出 NaN 检查
        if torch.isnan(x1_pred).any() or torch.isnan(x2_pred).any():
            nan_count += 1
            pbar.set_postfix({'Loss': 'SKIP(NaN output)'})
            continue

        loss_dict = criterion(y1, y2, x1_pred, x2_pred)
        loss_val = loss_dict['total_loss']

        # Loss NaN 检查
        if torch.isnan(loss_val) or torch.isinf(loss_val):
            nan_count += 1
            pbar.set_postfix({'Loss': 'SKIP(NaN loss)'})
            continue

        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss_val.item()
        valid_batches += 1
        pbar.set_postfix({'Loss': f"{loss_val.item():.4f}"})

        # 释放中间 tensor，防止内存累积（NUM_BID_ITER=10 的 autograd 图很重）
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

    total_ssim = 0.0
    total_flux_err = 0.0
    num_batches = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Val] ", leave=False)
    for batch in pbar:
        y1 = batch['y1'].to(device, non_blocking=True)
        y2 = batch['y2'].to(device, non_blocking=True)

        # 全程 FP32，避免 FFT NaN
        x1_pred = model(y1)
        x2_pred = model(y2)

        loss_dict = criterion(y1, y2, x1_pred, x2_pred)
        loss_val = loss_dict['total_loss'].item()
        if np.isnan(loss_val) or np.isinf(loss_val):
            continue
        total_loss += loss_val

        def normalize(img):
            img_min = torch.amin(img, dim=(1, 2), keepdim=True)
            img_max = torch.amax(img, dim=(1, 2), keepdim=True)
            return (img - img_min) / (img_max - img_min + 1e-8)

        # Blur2Blur 评估：退化后的预测与观测比较（同质比较才有意义）
        # 对 x1_pred 施加 PSF 退化后再与 y2 算 SSIM
        x1_degraded_for_eval = criterion._apply_psf_degradation(x1_pred)
        x1_degraded_norm = normalize(x1_degraded_for_eval)
        y2_norm = normalize(y2)
        _, ssim, flux_err = compute_all_metrics(x1_degraded_norm, y2_norm)
        total_ssim += ssim
        total_flux_err += flux_err
        num_batches += 1

        if epoch % 5 == 0 and not vis_done:
            visualize_blur2blur_result(
                epoch, y1[0].cpu(), y2[0].cpu(),
                x1_pred[0].cpu(), x2_pred[0].cpu(), cfg, '_blur2blur'
            )
            vis_done = True

    avg_loss = total_loss / len(loader)
    avg_ssim = total_ssim / num_batches
    avg_flux_err = total_flux_err / num_batches

    if writer:
        writer.add_scalar('Loss/Val', avg_loss, epoch)
        writer.add_scalar('Metrics/SSIM', avg_ssim, epoch)
        writer.add_scalar('Metrics/Flux_Err', avg_flux_err, epoch)

    return avg_loss, {'ssim': avg_ssim, 'flux_err': avg_flux_err}


def run_train(band, cfg_template):
    """ Blur2Blur 双帧训练（共享真值 + E[n]=0 假设） + 早停机制 """
    cfg = copy.deepcopy(cfg_template)
    cfg.BASE.CURRENT_BAND = band

    cfg.MODEL.USE_DETAIL_BLOCK = True
    cfg.MODEL.USE_PSF = True
    cfg.MODEL.USE_ITER = True

    cfg.BASE.EXP_NAME = f"band_{band}_blur2blur"
    create_output_dirs(cfg)
    device = cfg.BASE.DEVICE
    print(f"\n🚀 [Blur2Blur Training] Band: {band}Å | 双帧共享真值 | E[n]=0")

    # 数据加载
    train_set = AIABlur2BlurDataset(cfg, mode='train')
    val_set = AIABlur2BlurDataset(cfg, mode='val')
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # 模型 + 损失 + 优化器（FP32 全精度训练，避免 FFT NaN）
    model = PhysicsAwareIterativeNet(cfg).to(device)
    criterion = Blur2BlurPhysicsLoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.TRAIN.EPOCHS, eta_min=cfg.TRAIN.MIN_LR)

    writer = SummaryWriter(log_dir=cfg.BASE.LOG_DIR)
    best_val_loss = float('inf')
    final_metrics = None

    early_stop_patience = 5
    early_stop_counter = 0
    early_stop = False

    for epoch in range(cfg.TRAIN.EPOCHS):
        if early_stop:
            break

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        val_loss, metrics = validate(model, val_loader, criterion, device, epoch, writer, cfg)
        scheduler.step()

        final_metrics = metrics

        # 早停用 validation loss（训练目标=退模糊后交叉一致性，SSIM 比较清晰图vs模糊图无意义）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_metrics': metrics,
                'best_val_loss': best_val_loss,
                'config': cfg
            }, os.path.join(cfg.BASE.MODEL_DIR, "best_model.pth"))
        else:
            early_stop_counter += 1
            print(f"⚠️  Early Stop Counter: {early_stop_counter}/{early_stop_patience} "
                  f"(best val_loss: {best_val_loss:.6f})")
            if early_stop_counter >= early_stop_patience:
                print(f"\n🛑 Early Stop Triggered! Best Val Loss: {best_val_loss:.6f}")
                early_stop = True

        print(f"Epoch {epoch+1}/{cfg.TRAIN.EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | SSIM: {metrics['ssim']:.4f}")

    writer.close()

    # === 显式释放 GPU 内存，防止跨波段 OOM ===
    # 关键：必须先将模型移到 CPU，PyTorch 才会真正释放 GPU tensor 的物理内存
    # Python 的 del + gc.collect() 无法穿透 PyTorch CUDA allocator 释放显存
    torch.cuda.synchronize()
    optimizer.zero_grad(set_to_none=True)  # 释放梯度 tensor

    # 1. 核心步骤：将模型和损失函数移到 CPU，强制释放 GPU tensor
    model.to('cpu')
    criterion.to('cpu')

    # 2. 删除所有 GPU 相关对象
    del model, criterion, optimizer, scheduler
    # 3. 关闭 DataLoader workers 并删除
    if hasattr(train_loader, '_iterator') and train_loader._iterator is not None:
        train_loader._iterator._shutdown_workers()
    del train_loader, val_loader, train_set, val_set
    # 4. 多重 GC + 清空缓存
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()

    # 5. 验证清理效果 (mem_get_info 返回 bytes)
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        used_gib = (total_bytes - free_bytes) / (1024**3)
        total_gib = total_bytes / (1024**3)
        print(f"🧹 GPU 清理后显存占用: {used_gib:.1f} GiB / {total_gib:.1f} GiB")
    except Exception:
        pass  # 兼容旧版 PyTorch

    print(f"✅ Band {band}Å Training Done. Best Val Loss: {best_val_loss:.6f} | "
          f"Best SSIM: {final_metrics['ssim']:.4f} (退化后对比)")
    return {'band': band, 'exp_name': cfg.BASE.EXP_NAME,
            'val_loss': best_val_loss, 'ssim': final_metrics['ssim'],
            'flux_err': final_metrics['flux_err']}


def main():
    cfg_template = get_config()

    # === 训练日志：同时输出到 outputs/train.log（含 stdout + stderr）===
    log_path = os.path.join(cfg_template.BASE.OUTPUT_ROOT, "train.log")
    os.makedirs(cfg_template.BASE.OUTPUT_ROOT, exist_ok=True)
    sys.stdout = Tee(log_path, 'stdout')
    sys.stderr = Tee(log_path, 'stderr')

    print(f"{'='*60}")
    print(f"Blur2Blur 训练日志 - 启动时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    set_seed(cfg_template.BASE.SEED)

    # 所有训练波段
    all_bands = cfg_template.BASE.TRAIN_BANDS
    all_results = []

    # 跳过已完成的波段（路径与 run_train 中 create_output_dirs 一致）
    bands_to_train = []
    for band in all_bands:
        model_path = os.path.join(cfg_template.BASE.OUTPUT_ROOT, f"band_{band}",
                                   f"band_{band}_blur2blur", "models", "best_model.pth")
        if os.path.exists(model_path):
            print(f"⏭️ 波段 {band}Å 已完成，跳过")
        else:
            bands_to_train.append(band)

    if len(bands_to_train) == 0:
        print("✅ 所有波段已完成训练！")
        return

    print(f"🔥 剩余待训练波段: {bands_to_train}")

    for band in bands_to_train:
        print(f"\n{'='*60}")
        print(f"🔥 开始训练波段: {band}Å (Blur2Blur 双帧共享真值)")
        print(f"{'='*60}")
        res = run_train(band, cfg_template)
        all_results.append(res)
        # 额外一层保险：确保跨波段显存彻底释放
        gc.collect()
        torch.cuda.empty_cache()

    print("\n\n" + "="*80)
    print("📊 7 波段 Blur2Blur 双帧训练结果汇总")
    print("="*80)
    print(f"{'Experiment':<30} | {'Band':<5} | {'SSIM':<8} | {'FluxErr':<8}")
    print("-" * 80)
    for res in all_results:
        print(f"{res['exp_name']:<30} | {res['band']:<5} | {res['ssim']:<8.4f} | {res['flux_err']:<8.4f}")


if __name__ == "__main__":
    main()
