import os
DATA = os.environ.get("BLUR2SUN_TEST_DATA", os.path.join(os.environ.get("BLUR2SUN_DATA", ""), "test", "test_classified"))
OUT = os.environ.get("BLUR2SUN_RESULTS", os.path.dirname(os.path.abspath(__file__)))
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.time import Time
from aiapy.response import Channel
import fiasco

# ===================== 你的配置 =====================
os.environ['XUVTOP'] = '/home/zth2029/project/CHIANTI_11.0.2_database'
fits_files = {
    94: os.path.join(DATA, "0094-HR/AIA20250620_0806_0094.fits"),
    131: os.path.join(DATA, "0131-HR/AIA20250620_0806_0131.fits"),
    171: os.path.join(DATA, "0171-HR/AIA20250620_0806_0171.fits"),
    193: os.path.join(DATA, "0193-HR/AIA20250620_0806_0193.fits"),
    211: os.path.join(DATA, "0211-HR/AIA20250620_0806_0211.fits"),
    335: os.path.join(DATA, "0335-HR/AIA20250620_0806_0335.fits"),
}
# 温度网格
temperature = np.logspace(5, 8, 100) * u.K
# AIA 各通道对应核心离子（CHIANTI 11.0.2 标准）
AIA_IONS = {
    94: ['fe_8', 'fe_9', 'fe_10', 'fe_11'],
    131: ['fe_5', 'fe_6', 'ca_10', 'fe_13'],
    171: ['fe_5', 'fe_6', 'fe_7', 'fe_8'],
    193: ['fe_10', 'fe_11', 'fe_12', 'ca_14'],
    211: ['fe_11', 'fe_12', 'fe_13', 'ca_14'],
    335: ['fe_13', 'fe_14', 'fe_15', 'fe_16']
}
# =====================================================

def get_obstime_from_filename(fits_path):
    """从文件名提取观测时间"""
    filename = os.path.basename(fits_path)
    match = re.search(r'AIA(\d{8})_(\d{4})', filename)
    date_str, time_str = match.groups()
    obs_time_str = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]} {time_str[:2]}:{time_str[2:]}:00"
    return Time(obs_time_str)

def calculate_temperature_response(channel_wavelength):
    """
    🔥 终极兼容版：纯原生计算，无任何版本依赖
    """
    # 1. 初始化 AIA 通道 + 基础波长响应（无报错定义）
    channel = Channel(channel_wavelength * u.angstrom)
    wavelength = channel.wavelength
    wave_response = channel.effective_area * channel.gain

    # 2. 🔥 修复：原生 fiasco 计算发射率（兼容所有版本）
    ions = [fiasco.Ion(ion, temperature, abundance='sun_photospheric') for ion in AIA_IONS[channel_wavelength]]
    ion_collection = fiasco.IonCollection(*ions)
    # 计算发射率
    emission = ion_collection.emission(wavelength)
    emission = emission.to('cm^5 erg s^-1 Angstrom^-1')

    # 3. 积分计算温度响应函数
    temp_response = np.trapz(wave_response * emission, wavelength, axis=0)
    temp_response = (temp_response / (1.0 * u.erg)).to('DN cm^-5 s^-1 pix^-1')

    return temperature, temp_response

# ===================== 主程序 =====================
if __name__ == '__main__':
    response_data = {}
    colors = ['purple', 'blue', 'green', 'orange', 'red', 'brown']
    
    plt.figure(figsize=(12, 6))
    for i, wave in enumerate(fits_files.keys()):
        print(f"正在计算 {wave} Å 通道...")
        obstime = get_obstime_from_filename(fits_files[wave])
        print(f"✅ 观测时间: {obstime.iso}")
        
        temp, resp = calculate_temperature_response(wave)
        response_data[wave] = {'temperature': temp, 'response': resp}
        plt.plot(temp.value, resp.value, label=f'{wave} Å', color=colors[i], lw=2)

    # 绘图
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Response (DN cm$^{-5}$ s$^{-1}$ pix$^{-1}$)')
    plt.title('AIA Temperature Response Functions (CHIANTI 11.0.2)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'aia_temperature_response.png'), dpi=300)
    
    # 保存数据
    np.savez(os.path.join(OUT, 'aia_temp_response.npz'), **response_data)
    print("\n🎉 全部计算完成！结果已保存")