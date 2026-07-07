import os
import torch
from easydict import EasyDict as edict


def get_config():
    cfg = edict()

    # ====================== 1. 基础路径与波段配置 ======================
    cfg.BASE = edict()
    cfg.BASE.FLARE_DATA_ROOT = "./data/classified_aia_fits/"
    cfg.BASE.QUIET_DATA_ROOT = "./data/classified_aia_bands_fits/"
    cfg.BASE.OUTPUT_ROOT = "./outputs/"
    cfg.BASE.SEED = 42
    cfg.BASE.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.BASE.PSF_CROP_SIZE = 1024
    cfg.BASE.TRAIN_BANDS = [94, 131, 171, 193, 211, 304, 335]
    
    # 实验名称 (必须和消融对应，方便区分结果)
    cfg.BASE.EXP_NAME = "full_model" 
    
    # 初始化路径
    cfg.BASE.MODEL_DIR = None
    cfg.BASE.LOG_DIR = None
    cfg.BASE.VIS_DIR = None
    cfg.BASE.INFERENCE_DIR = None
    cfg.BASE.CURRENT_BAND = None

    # 波段参数
    aia_band_params = {
        "94": {"HR_DIR_NAME": "0094-HR", "QUIET_HR_DIR_NAME": "0094-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 806, "REAL_HR_NORM_MEAN": 0.0052, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": "./PSF/psf_aia_0094.fits"},
        "131": {"HR_DIR_NAME": "0131-HR", "QUIET_HR_DIR_NAME": "0131-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 1165, "REAL_HR_NORM_MEAN": 0.0086, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": "./PSF/psf_aia_0131.fits"},
        "171": {"HR_DIR_NAME": "0171-HR", "QUIET_HR_DIR_NAME": "0171-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 4619, "REAL_HR_NORM_MEAN": 0.0466, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": "./PSF/psf_aia_0171.fits"},
        "193": {"HR_DIR_NAME": "0193-HR", "QUIET_HR_DIR_NAME": "0193-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 5195, "REAL_HR_NORM_MEAN": 0.0518, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": "./PSF/psf_aia_0193.fits"},
        "211": {"HR_DIR_NAME": "0211-HR", "QUIET_HR_DIR_NAME": "0211-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 3457, "REAL_HR_NORM_MEAN": 0.0505, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": "./PSF/psf_aia_0211.fits"},
        "304": {"HR_DIR_NAME": "0304-HR", "QUIET_HR_DIR_NAME": "0304-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 226, "REAL_HR_NORM_MEAN": 0.0355, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": "./PSF/psf_aia_0304.fits"},
        "335": {"HR_DIR_NAME": "0335-HR", "QUIET_HR_DIR_NAME": "0335-HR", "GLOBAL_MIN": 0.0, "GLOBAL_MAX": 158, "REAL_HR_NORM_MEAN": 0.0310, "VALID_SIGNAL_THRESH": 0.0045, "PSF_FILE": "./PSF/psf_aia_0335.fits"},
    }
    cfg.AIA_BAND_PARAMS = aia_band_params

    # ====================== 2. 数据预处理 ======================
    cfg.PREPROCESS = edict()
    cfg.PREPROCESS.ORIGINAL_SIZE = 1024
    cfg.PREPROCESS.PSF_REBIN_SIZE = 1024
    cfg.PREPROCESS.MASK_RATIO = 0.12
    cfg.PREPROCESS.SATURATION_THRESH = 14000
    cfg.PREPROCESS.USE_QUIET_DATA = True

    # ====================== 3. 网络架构 + 精准消融开关 ======================
    cfg.MODEL = edict()
    cfg.MODEL.IN_CHANNELS = 1
    cfg.MODEL.OUT_CHANNELS = 1
    cfg.MODEL.BASE_CHANNELS = 16
    cfg.MODEL.NUM_BLOCKS = 4
    cfg.MODEL.NUM_GROUPS = 32
    cfg.MODEL.NUM_BID_ITER = 10

    # 消融实验开关
    cfg.MODEL.USE_PRIOR_BLOCK = True   # 多尺度先验块
    cfg.MODEL.USE_PSF = True          # PSF物理模块
    cfg.MODEL.USE_ITER = True         # BID迭代优化
    cfg.MODEL.LEARNABLE_STEP_SIZE = True  # 迭代步长 k 是否可学习（False 则固定为1.0）

    # 自适应迭代：残差收敛后提前终止，节省计算
    cfg.MODEL.ADAPTIVE_ITER = True    # 开启自适应迭代
    cfg.MODEL.ADAPTIVE_MIN_ITER = 4   # 最少迭代次数（前 N 次不检测收敛）
    cfg.MODEL.ADAPTIVE_THRESHOLD = 0.001  # 相对改进阈值：当前改进 < 该比例时视为停滞
    cfg.MODEL.ADAPTIVE_PATIENCE = 2   # 连续停滞次数后提前终止

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

    # ====================== 5. 适配新版自适应物理损失配置 ======================
    cfg.LOSS = edict()
    cfg.LOSS.TYPE = 'L1'
    cfg.LOSS.SCALE = 1.0            # 总损失缩放系数（供外层训练脚本聚合使用）
    cfg.LOSS.MAIN_WEIGHT = 1.0      # 核心亮度自适应一致性损失权重
    cfg.LOSS.GRAD_WEIGHT = 0    # 像素级细节下限损失权重（根除暗区黑影）
    cfg.LOSS.DETAIL_MARGIN = 0   # 细节防线安全边际（控制局部高频增益，推荐1.01~1.05）

    return cfg


def create_output_dirs(cfg):
    if cfg.BASE.CURRENT_BAND is None:
        raise ValueError("❌ 请先设置 cfg.BASE.CURRENT_BAND 再创建目录！")
    
    band = cfg.BASE.CURRENT_BAND
    exp_name = cfg.BASE.EXP_NAME
    
    # 输出目录自动区分消融实验
    band_root_dir = os.path.join(cfg.BASE.OUTPUT_ROOT, f"band_{band}", exp_name)
    
    cfg.BASE.MODEL_DIR = os.path.join(band_root_dir, "models")
    cfg.BASE.LOG_DIR = os.path.join(band_root_dir, "logs")
    cfg.BASE.VIS_DIR = os.path.join(band_root_dir, "vis")
    cfg.BASE.INFERENCE_DIR = os.path.join(band_root_dir, "inference_results")
    
    dir_list = [cfg.BASE.MODEL_DIR, cfg.BASE.LOG_DIR, cfg.BASE.VIS_DIR, cfg.BASE.INFERENCE_DIR]
    for dir_path in dir_list:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"\n📁 实验目录：波段 {band}Å | 实验 {exp_name}")
    print(f"   路径：{band_root_dir}")