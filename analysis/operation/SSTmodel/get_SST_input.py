"""
Download ERA5 monthly-mean input data for CAS-Canglong SST16 model.

Usage:
    修改 forecast_start 后直接运行:
    conda activate torch
    python analysis/operation/SSTmodel/get_SST_input.py

输入窗口为 forecast_start 之前的 16 个月。
例如 forecast_start='202602' → 输入 2024-10 ~ 2026-01，预测 2026-02 ~ 2027-05。

ERA5 CDS API 不支持跨年请求，因此按年拆分下载，最后合并。
单层变量(6个) + 气压层变量(2个) 按年各一个文件。
"""

import os
import zipfile
import tempfile
import cdsapi
import xarray as xr
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ============================================================
# 配置：修改这里的起报月份
# ============================================================
forecast_start = '202603'  # YYYYMM, 起报月份

# 输出目录
OUTPUT_DIR = '/data/lhwang/SST'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 计算16个月输入窗口
# ============================================================
fc_date = datetime.strptime(forecast_start, '%Y%m')
# 输入窗口: fc_date - 16 months ~ fc_date - 1 month
input_start = fc_date - relativedelta(months=16)
input_end = fc_date - relativedelta(months=1)

print(f'Forecast start: {fc_date:%Y-%m}')
print(f'Input window:   {input_start:%Y-%m} ~ {input_end:%Y-%m} (16 months)')

# 按年份分组月份
year_months = {}
cur = input_start
while cur <= input_end:
    y = str(cur.year)
    m = f'{cur.month:02d}'
    year_months.setdefault(y, []).append(m)
    cur += relativedelta(months=1)

print(f'Download plan: {len(year_months)} years × 2 datasets = {len(year_months) * 2} requests')
for y, ms in sorted(year_months.items()):
    print(f'  {y}: months {",".join(ms)}')

# ============================================================
# CDS API 请求定义
# ============================================================
SURFACE_VARS = [
    "sea_surface_temperature",
    "mean_sea_level_pressure",
    "surface_latent_heat_flux",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "surface_net_solar_radiation",
]

client = cdsapi.Client()

downloaded_surface = []
downloaded_pressure = []

for year, months in sorted(year_months.items()):
    # --- 单层变量 ---
    surf_file = os.path.join(OUTPUT_DIR, f'ERA5_surface_monthly_{year}_{forecast_start}.nc')
    if os.path.exists(surf_file):
        print(f'[skip] {surf_file} already exists')
    else:
        print(f'[download] Surface {year} months={months} ...')
        client.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": SURFACE_VARS,
                "year": [year],
                "month": months,
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            surf_file,
        )
        print(f'  -> {surf_file}')
    downloaded_surface.append(surf_file)

    # --- 气压层变量 (500hPa + 850hPa geopotential) ---
    pres_file = os.path.join(OUTPUT_DIR, f'ERA5_pressure_monthly_{year}_{forecast_start}.nc')
    if os.path.exists(pres_file):
        print(f'[skip] {pres_file} already exists')
    else:
        print(f'[download] Pressure {year} months={months} ...')
        client.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": ["monthly_averaged_reanalysis"],
                "variable": ["geopotential"],
                "pressure_level": ["500", "850"],
                "year": [year],
                "month": months,
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
            },
            pres_file,
        )
        print(f'  -> {pres_file}')
    downloaded_pressure.append(pres_file)

# ============================================================
# 合并各年文件 → 单个16月文件
# CDS 多变量请求会返回 zip (内含按 stepType 拆分的多个 NC)，
# 需要先解压再合并。
# ============================================================
print('\nMerging files ...')

import pandas as pd

def normalize_time(ds):
    """将 valid_time 统一到月初 00:00，消除 stepType 导致的 00:00/06:00 差异。"""
    tc = 'valid_time' if 'valid_time' in ds.dims else 'time'
    times = pd.to_datetime(ds[tc].values)
    # 截断到月初
    normalized = times.normalize()  # 去掉小时
    ds[tc] = normalized
    return ds

def open_cds_file(path):
    """打开 CDS 下载文件，自动处理 zip 和普通 NC。"""
    if zipfile.is_zipfile(path):
        datasets = []
        with zipfile.ZipFile(path) as z:
            tmpdir = tempfile.mkdtemp()
            for name in z.namelist():
                extracted = z.extract(name, tmpdir)
                datasets.append(normalize_time(xr.open_dataset(extracted)))
        # 合并同一年内按 stepType 拆分的多个 NC (时间已对齐)
        return xr.merge(datasets)
    else:
        return normalize_time(xr.open_dataset(path))

# 单层变量：各年打开后合并
surf_datasets = [open_cds_file(f) for f in sorted(downloaded_surface)]
ds_surf = xr.concat(surf_datasets, dim='valid_time').sortby('valid_time')
print(f'Surface merged: {dict(ds_surf.sizes)}, vars={list(ds_surf.data_vars)}')

# 气压层变量：各年打开后合并
pres_datasets = [open_cds_file(f) for f in sorted(downloaded_pressure)]
ds_pres = xr.concat(pres_datasets, dim='valid_time').sortby('valid_time')
print(f'Pressure merged: {dict(ds_pres.sizes)}, vars={list(ds_pres.data_vars)}')

# 保存合并后的文件
fc_tag = f'{input_start:%Y%m}-{input_end:%Y%m}'
surf_merged = os.path.join(OUTPUT_DIR, f'ERA5_surface_monthly_{fc_tag}.nc')
pres_merged = os.path.join(OUTPUT_DIR, f'ERA5_pressure_monthly_{fc_tag}.nc')

ds_surf.to_netcdf(surf_merged)
print(f'Saved: {surf_merged}')
ds_pres.to_netcdf(pres_merged)
print(f'Saved: {pres_merged}')

# 清理按年下载的中间文件
for f in downloaded_surface + downloaded_pressure:
    if os.path.exists(f):
        os.remove(f)
print('Cleaned up per-year intermediate files.')

# ============================================================
# 验证
# ============================================================
print(f'\n=== Verification ===')
ds_s = xr.open_dataset(surf_merged)
ds_p = xr.open_dataset(pres_merged)
tc_s = 'valid_time' if 'valid_time' in ds_s.dims else 'time'
tc_p = 'valid_time' if 'valid_time' in ds_p.dims else 'time'
print(f'Surface: {len(ds_s[tc_s])} months, vars={list(ds_s.data_vars)}')
print(f'  time: {ds_s[tc_s].values[0]} ~ {ds_s[tc_s].values[-1]}')
print(f'Pressure: {len(ds_p[tc_p])} months, levels={ds_p.pressure_level.values if "pressure_level" in ds_p else "?"}')
print(f'  time: {ds_p[tc_p].values[0]} ~ {ds_p[tc_p].values[-1]}')

n_surf = len(ds_s[tc_s])
n_pres = len(ds_p[tc_p])
assert n_surf == 16, f'Expected 16 surface months, got {n_surf}'
assert n_pres == 16, f'Expected 16 pressure months, got {n_pres}'
print('\nAll 16 months downloaded and merged successfully.')
print(f'Surface file: {surf_merged}')
print(f'Pressure file: {pres_merged}')
