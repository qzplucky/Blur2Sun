import os
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits
from data_utils import get_sorted_fits_files, generate_non_overlapping_pairs


# ==================== Blur2Blur 双帧数据集（共享真值假设） ====================
class AIABlur2BlurDataset(Dataset):
    """双帧数据集：每个样本为 (y1, y2)，满足 y1=P*x+n1, y2=P*x+n2
    共享真值假设：连续相邻帧对应相同的太阳场景 x，噪声 n1, n2 独立且 E[n]=0
    Blur2Blur 损失：||P*f(y1) - y2|| + ||P*f(y2) - y1||
    """
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.mode = mode
        self.band = cfg.BASE.CURRENT_BAND
        self.band_cfg = cfg.AIA_BAND_PARAMS[str(self.band)]
        
        all_raw_pairs = []
        
        # --- 加载爆发期数据 ---
        flare_sorted_files = self._get_sorted_files_from_root(
            root_dir=cfg.BASE.FLARE_DATA_ROOT,
            dir_name=self.band_cfg['HR_DIR_NAME']
        )
        if len(flare_sorted_files) >= 2:
            flare_split_files = self._split_train_val(flare_sorted_files, mode)
            flare_save_path = os.path.join(cfg.BASE.LOG_DIR, f"band_{self.band}_{mode}_flare_pairs.txt")
            flare_pairs = generate_non_overlapping_pairs(
                flare_split_files, save_path=flare_save_path,
                **self._pair_kwargs())
            print(f"✅ 爆发期数据: 原始 {len(flare_split_files)} 帧 → 生成 {len(flare_pairs)} 对")
            all_raw_pairs.extend(flare_pairs)
        
        # --- 加载宁静期数据 ---
        if hasattr(cfg.PREPROCESS, 'USE_QUIET_DATA') and cfg.PREPROCESS.USE_QUIET_DATA:
            quiet_dir_name = self.band_cfg.get('QUIET_HR_DIR_NAME', self.band_cfg['HR_DIR_NAME'])
            quiet_sorted_files = self._get_sorted_files_from_root(
                root_dir=cfg.BASE.QUIET_DATA_ROOT,
                dir_name=quiet_dir_name
            )
            if len(quiet_sorted_files) >= 2:
                quiet_split_files = self._split_train_val(quiet_sorted_files, mode)
                quiet_save_path = os.path.join(cfg.BASE.LOG_DIR, f"band_{self.band}_{mode}_quiet_pairs.txt")
                quiet_pairs = generate_non_overlapping_pairs(
                    quiet_split_files, save_path=quiet_save_path,
                    **self._pair_kwargs())
                print(f"✅ 宁静期数据: 原始 {len(quiet_split_files)} 帧 → 生成 {len(quiet_pairs)} 对")
                all_raw_pairs.extend(quiet_pairs)
        
        if len(all_raw_pairs) == 0:
            raise ValueError(f"未找到任何有效数据对！请检查数据路径: {cfg.BASE.FLARE_DATA_ROOT}, {cfg.BASE.QUIET_DATA_ROOT}")
        
        # 稳定性过滤：确保共享真值假设成立（场景不变，仅噪声不同）
        print(f"\n🔍 {self.band}Å {mode}集 - 稳定性过滤（共享真值验证）...")
        # 稳定性过滤 v3 最终版（回应 Major-3）：信号区 NCC + 本波段总体稳健阈值
        # 两遍法：先计算全部候选对的信号区 NCC，再按本波段总体（中位-4×MAD）
        # 自校准阈值——对低信噪比波段公平（阈值随通道噪声水平自适应），
        # 低尾 = 快速演化对（耀斑脉冲相）被正确剔除
        print(f"\n🔍 {self.band}Å {mode}集 - 稳定性过滤（信号区 NCC + 稳健阈值）...")
        nccs = np.array([self._signal_region_ncc(p[0], p[1])
                         for p in all_raw_pairs])
        # 按（波段×日）分组自校准阈值：不同日/处理链的噪声底不同，
        # 混合总体会拉低稳健阈值（quiet 日跨链伪影更显著）
        flare_root = os.path.normpath(self.cfg.BASE.FLARE_DATA_ROOT)
        groups = {}
        for i, p in enumerate(all_raw_pairs):
            key = 'flare' if os.path.normpath(p[0]).startswith(flare_root) else 'quiet'
            groups.setdefault(key, []).append(i)
        keep = np.zeros(len(all_raw_pairs), dtype=bool)
        thr_info = {}
        for key, idxs in groups.items():
            v = nccs[idxs]
            med = float(np.median(v))
            mad = float(1.4826 * np.median(np.abs(v - med)))
            thr = med - 4.0 * mad
            keep[idxs] = v >= thr
            thr_info[key] = f"中位={med:.4f} MAD={mad:.4f} 阈值={thr:.4f} 保留={100*keep[idxs].mean():.1f}%"
        self.pairs = [p for p, k in zip(all_raw_pairs, keep) if k]
        for key, info in thr_info.items():
            print(f"   [{key}] 信号区 NCC {info}")

        print(f"✅ 波段 {self.band}Å {mode}集 - 过滤后稳定成对数：{len(self.pairs)} "
              f"（原始 {len(all_raw_pairs)}，保留率约 {len(self.pairs)/len(all_raw_pairs):.1%}）")

        if len(self.pairs) == 0:
            raise ValueError(f"过滤后无稳定 pair！请检查 NCC 稳健阈值")
    
    def _pair_kwargs(self):
        """从配置读取配对时间间隔约束（12s 实验设 PAIR_INTERVAL_S=12；
        未配置则保持原行为不变）"""
        pp = self.cfg.PREPROCESS
        interval = getattr(pp, 'PAIR_INTERVAL_S', None)
        if interval:
            return {"interval_s": float(interval),
                    "tol_s": float(getattr(pp, 'PAIR_TOL_S', 2.0))}
        return {}

    def _get_sorted_files_from_root(self, root_dir, dir_name):
        """从指定根目录和子目录获取排序后的FITS文件"""
        data_dir = os.path.join(root_dir, dir_name)
        if not os.path.exists(data_dir):
            print(f"⚠️ 目录不存在: {data_dir}")
            return []
        try:
            return get_sorted_fits_files(self._create_dummy_cfg(root_dir), self.band)
        except:
            fits_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.fits')])
            fits_files = [f for f in fits_files if os.path.getsize(f) > 0]
            return fits_files
    
    def _create_dummy_cfg(self, data_root):
        from easydict import EasyDict as edict
        dummy_cfg = edict()
        dummy_cfg.BASE = edict()
        dummy_cfg.BASE.DATA_ROOT = data_root
        dummy_cfg.BASE.CURRENT_BAND = self.band
        dummy_cfg.AIA_BAND_PARAMS = self.cfg.AIA_BAND_PARAMS
        return dummy_cfg
    
    def _split_train_val(self, sorted_files, mode):
        """训练/验证划分（8:2）"""
        split_idx = int(0.8 * len(sorted_files))
        if mode == 'train':
            return sorted_files[:split_idx]
        else:
            return sorted_files[split_idx:]
    
    def __len__(self):
        return len(self.pairs)
    
    def _signal_region_ncc(self, file1, file2):
        """信号区 NCC（v3 最终版判据，回应审稿 Major-3）

        演进：mean/max 阈值（对处理链定标偏移与曝光差不免疫）
          -> 全局 NCC（对定标偏移免疫，但整盘 NCC 被低计数噪声区压低，
             对 94/131/335 等低信噪比波段不公平）
          -> 信号区 NCC：仅在两帧各自 70 分位以上像素的并集上计算。
        94Å 实测：静态 12s 对保留 97.3%，演化 120s 对仅 57.1% 通过（NCC>0.98）。
        阈值由本波段总体稳健统计自校准（中位 - 4×MAD），见数据集构建处。
        """
        img1 = self._read_fits(file1)
        img2 = self._read_fits(file2)
        i1 = np.nan_to_num(img1, nan=0.0, posinf=0.0, neginf=0.0)
        i2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
        m = (i1 >= np.percentile(i1, 70)) | (i2 >= np.percentile(i2, 70))
        if m.sum() < 1000:
            m = np.ones_like(m, dtype=bool)
        a, b = i1[m], i2[m]
        if a.std() < 1e-8 or b.std() < 1e-8:
            return -1.0
        return float(np.corrcoef(a, b)[0, 1])

    def _read_fits(self, file_path):
        """读取FITS文件并返回2D float32图像数组"""
        try:
            with fits.open(file_path) as hdul:
                if len(hdul) > 1 and hdul[1].data is not None:
                    img = hdul[1].data.astype(np.float32)
                else:
                    img = hdul[0].data.astype(np.float32)
                if len(img.shape) > 2:
                    img = img.squeeze()
                if len(img.shape) != 2:
                    raise ValueError(f"FITS文件数据不是2D图像: {file_path}, 形状: {img.shape}")
                if img.size == 0 or np.all(img == 0):
                    raise ValueError(f"FITS文件数据为空或全0: {file_path}")
                return img
        except Exception as e:
            raise RuntimeError(f"读取FITS文件失败: {file_path}, 错误: {str(e)}")
    
    def __getitem__(self, idx):
        file1, file2 = self.pairs[idx]
        img1 = self._read_fits(file1)
        img2 = self._read_fits(file2)
        
        # NaN/inf 安全检查：用 0 替换坏值
        if np.isnan(img1).any() or np.isinf(img1).any():
            img1 = np.nan_to_num(img1, nan=0.0, posinf=0.0, neginf=0.0)
        if np.isnan(img2).any() or np.isinf(img2).any():
            img2 = np.nan_to_num(img2, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 保留原始噪声分布（含负值），不截断
        # 原因：AIA Level 1.5 已扣除偏置和暗电流，负值来自零均值读出噪声
        #       截断(np.maximum)会破坏 E[n]=0 假设，在暗区引入~15%系统性偏置
        #       非负物理约束仅在推理/显示阶段施加
        
        # 轻量级强度对齐（仅当 mean 差异 >5% 时做轻微缩放）
        mean1 = img1.mean()
        mean2 = img2.mean()
        if mean1 > 1e-8 and mean2 > 1e-8:
            mean_ratio = abs(mean1 - mean2) / max(mean1, mean2)
            if mean_ratio > 0.05:
                target_mean = (mean1 + mean2) / 2.0
                img1 = img1 * (target_mean / mean1)
                img2 = img2 * (target_mean / mean2)
        
        return {
            "y1": torch.from_numpy(img1).unsqueeze(0),
            "y2": torch.from_numpy(img2).unsqueeze(0),
            "file1": file1,
            "file2": file2,
        }