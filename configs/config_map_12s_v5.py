"""Blur2Sun v3 — MAP 训练配置（12s 配对, 修正 PSF: 官方 rebin_psf 2048）
EXP=map_full_12s_v3; 与 v2 的唯一算法差异 = PSF 算子修正（回应 Minor-11）"""
"""
Config for Bayesian MAP-based Deep Unfolding experiments — 12s 间隔配对版.
回应审稿意见 Major-2/Minor-5：图像对间隔从 120s 改为 12s。
与 config2_map.py 的唯一区别：
  1. 数据根指向 Blur_data/paired_*（anchor+neighbor 交替排列的 12s 配对目录）
  2. PREPROCESS.PAIR_INTERVAL_S = 12（配对时显式校验 DATE-OBS 间隔）
  3. EXP_NAME = "map_full_12s"（输出目录隔离，不覆盖 2min 结果）
"""
import os
import torch
from easydict import EasyDict as edict


def get_config():
    cfg = edict()

    # ====================== 1. 基础路径与波段配置 ======================
    cfg.BASE = edict()
    cfg.BASE.FLARE_DATA_ROOT = os.environ.get("BLUR2SUN_DATA", "/mnt/zth24029/DataHDD/Hi-C/Blur_data") + "/paired_flare_20240515/"
    cfg.BASE.QUIET_DATA_ROOT = os.environ.get("BLUR2SUN_DATA", "/mnt/zth24029/DataHDD/Hi-C/Blur_data") + "/paired_quiet_20210218/"
    cfg.BASE.OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs") + os.sep
    cfg.BASE.SEED = 42
    cfg.BASE.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.BASE.PSF_CROP_SIZE = 2048
    cfg.BASE.TRAIN_BANDS = [94, 131, 171, 193, 211, 304, 335]

    # 实验名称
    cfg.BASE.EXP_NAME = "map_full_12s_v5"

    # 初始化路径
    cfg.BASE.MODEL_DIR = None
    cfg.BASE.LOG_DIR = None
    cfg.BASE.VIS_DIR = None
    cfg.BASE.INFERENCE_DIR = None
    cfg.BASE.CURRENT_BAND = None

    # 波段参数（与 config2_map.py 完全一致）
    aia_band_params = {
        "94": {"HR_DIR_NAME": "0094-HR", "QUIET_HR_DIR_NAME": "0094-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 806, "REAL_HR_NORM_MEAN": 0.0052, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "psf", "psf_aia_0094_rebin2048.fits")},
        "131": {"HR_DIR_NAME": "0131-HR", "QUIET_HR_DIR_NAME": "0131-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 1165, "REAL_HR_NORM_MEAN": 0.0086, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "psf", "psf_aia_0131_rebin2048.fits")},
        "171": {"HR_DIR_NAME": "0171-HR", "QUIET_HR_DIR_NAME": "0171-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 4619, "REAL_HR_NORM_MEAN": 0.0466, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "psf", "psf_aia_0171_rebin2048.fits")},
        "193": {"HR_DIR_NAME": "0193-HR", "QUIET_HR_DIR_NAME": "0193-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 5195, "REAL_HR_NORM_MEAN": 0.0518, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "psf", "psf_aia_0193_rebin2048.fits")},
        "211": {"HR_DIR_NAME": "0211-HR", "QUIET_HR_DIR_NAME": "0211-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 3457, "REAL_HR_NORM_MEAN": 0.0505, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "psf", "psf_aia_0211_rebin2048.fits")},
        "304": {"HR_DIR_NAME": "0304-HR", "QUIET_HR_DIR_NAME": "0304-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 226, "REAL_HR_NORM_MEAN": 0.0355, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "psf", "psf_aia_0304_rebin2048.fits")},
        "335": {"HR_DIR_NAME": "0335-HR", "QUIET_HR_DIR_NAME": "0335-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 158, "REAL_HR_NORM_MEAN": 0.0310, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "psf", "psf_aia_0335_rebin2048.fits")},
    }
    cfg.AIA_BAND_PARAMS = aia_band_params

    # ====================== 2. 数据预处理 ======================
    cfg.PREPROCESS = edict()
    cfg.PREPROCESS.ORIGINAL_SIZE = 1024
    cfg.PREPROCESS.PSF_REBIN_SIZE = 1024
    cfg.PREPROCESS.MASK_RATIO = 0.12
    cfg.PREPROCESS.SATURATION_THRESH = 14000
    cfg.PREPROCESS.USE_QUIET_DATA = True
    # --- 12s 配对约束（新增） ---
    # NCC screening uses the self-calibrated median - 4xMAD rule in
    # src/data_loader.py; no fixed absolute threshold is used.
    cfg.PREPROCESS.PAIR_INTERVAL_S = 12
    cfg.PREPROCESS.PAIR_TOL_S = 2.0

    # ====================== 3. MAP 网络架构 ======================
    cfg.MODEL = edict()
    cfg.MODEL.IN_CHANNELS = 1
    cfg.MODEL.OUT_CHANNELS = 1
    cfg.MODEL.BASE_CHANNELS = 16
    cfg.MODEL.NUM_BLOCKS = 4
    cfg.MODEL.NUM_GROUPS = 32
    cfg.MODEL.NUM_BID_ITER = 10  # MAP 迭代展开次数

    # MAP 消融开关
    cfg.MODEL.USE_MAP_PRIOR = True   # 是否使用 PriorNet（Bayesian Prior gradient）

    # 自适应迭代（关闭以保证可比性）
    cfg.MODEL.ADAPTIVE_ITER = False
    cfg.MODEL.ADAPTIVE_MIN_ITER = 4
    cfg.MODEL.ADAPTIVE_THRESHOLD = 0.001
    cfg.MODEL.ADAPTIVE_PATIENCE = 2

    # ====================== 4. 训练与测试基本超参 ======================
    cfg.TEST = edict()
    cfg.TEST.NUM_WORKERS = 1

    cfg.TRAIN = edict()
    cfg.TRAIN.EPOCHS = 50
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.LR = 1e-4
    cfg.TRAIN.WEIGHT_DECAY = 1e-5
    cfg.TRAIN.LR_SCHEDULER = "CosineAnnealingLR"
    cfg.TRAIN.MIN_LR = 1e-7
    cfg.TRAIN.VAL_INTERVAL = 1
    cfg.TRAIN.EARLY_STOP_PATIENCE = 5
    cfg.TRAIN.FREEZE_ENCODER_FIRST2 = False

    # ====================== 5. 损失配置 ======================
    cfg.LOSS = edict()
    cfg.LOSS.TYPE = 'L1'
    cfg.LOSS.SCALE = 1.0
    cfg.LOSS.MAIN_WEIGHT = 1.0
    cfg.LOSS.GRAD_WEIGHT = 0
    cfg.LOSS.DETAIL_MARGIN = 0

    return cfg


def create_output_dirs(cfg):
    if cfg.BASE.CURRENT_BAND is None:
        raise ValueError("请先设置 cfg.BASE.CURRENT_BAND 再创建目录！")

    band = cfg.BASE.CURRENT_BAND
    exp_name = cfg.BASE.EXP_NAME

    band_root_dir = os.path.join(cfg.BASE.OUTPUT_ROOT, f"band_{band}", exp_name)

    cfg.BASE.MODEL_DIR = os.path.join(band_root_dir, "models")
    cfg.BASE.LOG_DIR = os.path.join(band_root_dir, "logs")
    cfg.BASE.VIS_DIR = os.path.join(band_root_dir, "vis")
    cfg.BASE.INFERENCE_DIR = os.path.join(band_root_dir, "inference_results")

    dir_list = [cfg.BASE.MODEL_DIR, cfg.BASE.LOG_DIR, cfg.BASE.VIS_DIR, cfg.BASE.INFERENCE_DIR]
    for dir_path in dir_list:
        os.makedirs(dir_path, exist_ok=True)

    print(f"\n[MAP-12s] 实验目录：波段 {band}A | 实验 {exp_name}")
    print(f"   路径：{band_root_dir}")
