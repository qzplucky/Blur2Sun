# data_utils.py
import os
import glob
import datetime
from tqdm import tqdm
from astropy.io import fits
from astropy.time import Time
import warnings
warnings.filterwarnings('ignore')

def parse_fits_time(fits_path):
    """解析FITS文件的DATE-OBS时间，返回datetime对象（兼容单/多HDU结构）"""
    try:
        with fits.open(fits_path) as hdul:
            if len(hdul) > 1 and hdul[1].data is not None:
                header = hdul[1].header
            else:
                header = hdul[0].header
            if 'DATE-OBS' in header:
                time_str = header['DATE-OBS']
                time_obj = Time(time_str).datetime
                return time_obj
            else:
                print(f"⚠️ 文件 {os.path.basename(fits_path)} 无DATE-OBS字段，跳过")
                return None
    except Exception as e:
        print(f"❌ 解析 {os.path.basename(fits_path)} 时间失败：{str(e)}")
        return None

def get_sorted_fits_files(cfg, band):
    """获取指定波段的FITS文件，按观测时间排序（过滤无效文件）"""
    band_str = str(band)
    if band_str not in cfg.AIA_BAND_PARAMS:
        raise ValueError(f"❌ 波段 {band} 不在配置中，支持：{list(cfg.AIA_BAND_PARAMS.keys())}")
    
    band_params = cfg.AIA_BAND_PARAMS[band_str]
    data_dir = os.path.join(cfg.BASE.DATA_ROOT, band_params["HR_DIR_NAME"])
    
    # 匹配AIA Level1.5的FITS文件（.fits）
    fits_pattern = os.path.join(data_dir, "*.fits")
    all_fits = glob.glob(fits_pattern)
    if len(all_fits) == 0:
        raise FileNotFoundError(f"❌ 波段 {band} 路径 {data_dir} 下未找到FITS文件")
    
    # 过滤空文件（复用你原有逻辑）
    all_fits = [f for f in all_fits if os.path.getsize(f) > 0]
    
    # 解析时间并排序
    valid_files_with_time = []
    print(f"\n📊 解析波段 {band}Å 的 {len(all_fits)} 个FITS文件时间...")
    for fits_path in tqdm(all_fits):
        time_obj = parse_fits_time(fits_path)
        if time_obj is not None:
            valid_files_with_time.append((fits_path, time_obj))
    
    valid_files_with_time.sort(key=lambda x: x[1])
    sorted_files = [item[0] for item in valid_files_with_time]
    
    print(f"✅ 波段 {band}Å 有效排序文件数：{len(sorted_files)}")
    return sorted_files

def generate_non_overlapping_pairs(sorted_files, save_path=None,
                                   interval_s=None, tol_s=2.0, times=None):
    """生成非重叠连续成对帧：(0,1), (2,3)...

    interval_s: 若给定（如 12），额外要求配对两帧的 DATE-OBS 之差接近该值
                （|dt - interval_s| <= tol_s），不满足则顺延配对，保证
                "图像对时间间隔" 严格受控（AIA EUV cadence 为 12s）。
    times: 可选，与 sorted_files 等长的 datetime 列表（避免重复解析 header）。
    """
    pairs = []
    total_files = len(sorted_files)

    if interval_s is not None:
        if times is None:
            times = [parse_fits_time(f) for f in sorted_files]
        i = 0
        n_reject = 0
        while i < total_files - 1:
            t1, t2 = times[i], times[i+1]
            if t1 is not None and t2 is not None and \
                    abs((t2 - t1).total_seconds() - interval_s) <= tol_s:
                pairs.append((sorted_files[i], sorted_files[i+1]))
                i += 2
            else:
                n_reject += 1
                i += 1
        if n_reject:
            print(f"⚠️ 间隔过滤：{n_reject} 帧未能配成 {interval_s}s 对，已顺延")
    else:
        for i in range(0, total_files - 1, 2):
            pairs.append((sorted_files[i], sorted_files[i+1]))
    
    # 处理奇数个文件
    if total_files % 2 != 0:
        print(f"⚠️ 总文件数{total_files}为奇数，最后一个文件无配对，已跳过")
    
    # 保存成对列表（可选）
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"非重叠成对帧列表（总数：{len(pairs)}）\n")
            f.write(f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for idx, (f1, f2) in enumerate(pairs):
                f.write(f"Pair {idx+1}: {os.path.basename(f1)} <-> {os.path.basename(f2)}\n")
        print(f"📝 成对列表保存至：{save_path}")
    
    return pairs


def generate_sliding_triplets(sorted_files, save_path=None):
    """生成滑动窗口三帧序列：(0,1,2), (1,2,3), (2,3,4)...
    每个 triplet 为 (y_prev, y_anchor, y_next)，y_anchor 是监督目标。
    """
    triplets = []
    total_files = len(sorted_files)
    for i in range(total_files - 2):
        triplets.append((sorted_files[i], sorted_files[i+1], sorted_files[i+2]))
    
    if total_files < 3:
        print(f"⚠️ 总文件数{total_files}不足3帧，无法生成triplet")
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"滑动窗口三帧序列（总数：{len(triplets)}）\n")
            f.write(f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for idx, (f0, f1, f2) in enumerate(triplets):
                f.write(f"Triplet {idx+1}: {os.path.basename(f0)} | {os.path.basename(f1)} | {os.path.basename(f2)}\n")
        print(f"📝 Triplet列表保存至：{save_path}")
    
    return triplets
