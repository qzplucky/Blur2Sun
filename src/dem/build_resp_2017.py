#!/usr/bin/env python3
"""为 2017-09-06 X9.3 耀斑构建 AIA 温度响应矩阵（SR 环境，fiasco/CHIANTI）。

输出: resp_matrix_2017.npy, shape (6, nT)，单位 DN/s per (cm^-5)，
      与 fasico_b2s.build_response_matrix 同一实现（含衰减校正表）。
温度网格: logT 5.5-7.5，31 边界（30 bins，至 31.6 MK）。
"""
import os
import sys
import numpy as np

DEM_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, DEM_DIR)
os.chdir(DEM_DIR)  # build_response_matrix 用相对路径读校正表 CSV
os.environ.setdefault('XUVTOP', os.environ.get('BLUR2SUN_CHIANTI', '/path/to/CHIANTI/database'))

from astropy.time import Time
from fasico_b2s import build_response_matrix

OBSTIME = Time('2017-09-06T12:29:54')
logt_edges = np.linspace(5.5, 7.5, 31)
logt_centers = 0.5 * (logt_edges[:-1] + logt_edges[1:])

print('构建 2017-09-06 响应矩阵 (6 channels x', len(logt_centers),
      'T bins)...', flush=True)
resp = build_response_matrix(OBSTIME, logt_centers)
np.save(os.path.join(DEM_DIR, 'resp_matrix_2017.npy'), resp)
print('已保存 resp_matrix_2017.npy, shape =', resp.shape)
for i, wl in enumerate([94, 131, 171, 193, 211, 335]):
    r = resp[i]
    if r.max() > 0:
        pk = 10 ** logt_centers[int(np.argmax(r))] / 1e6
        print(f'  {wl} A: 峰值响应 T = {pk:.2f} MK, max = {r.max():.3e}')
