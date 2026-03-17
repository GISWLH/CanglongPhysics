"""
Download ERA5 daily input data for CAS-Canglong S2S model.

Usage:
    conda activate torch
    python analysis/operation/get_s2s_input.py

说明:
    forecast_week: 当前最后一个已知观测周（起报周 - 1）。
    输入窗口为第(forecast_week-2)周至第(forecast_week-1)周，共 14 天。
    每天下载表面层和气压层两个文件，共 14×2 = 28 个文件。

    例: forecast_week=6, forecast_year=2026
        查找表 (平年) → week4: 0122-0128, week5: 0129-0204
        下载范围: 20260122 ~ 20260204
        API 按月分组 (Jan 22-31, Feb 1-4) → 4 次请求 → 拆成 28 日文件
"""

import os
import calendar
import zipfile
import tempfile
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import xarray as xr
import cdsapi

# ============================================================
# 配置：修改这里
# ============================================================
forecast_week = 10    # 最后一个已知观测周 (模型将预报第 forecast_week+1 周起)
forecast_year = 2026

OUTPUT_DIR = '/data/lhwang/ERA5_daily_s2s'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 查找表 → 确定输入日期范围
# ============================================================
is_leap = calendar.isleap(forecast_year)
script_dir = os.path.dirname(os.path.abspath(__file__))
lookup_file = os.path.join(
    script_dir,
    'lookup_weeks_leap.csv' if is_leap else 'lookup_weeks_common.csv',
)
df_weeks = pd.read_csv(lookup_file)
df_weeks['week'] = df_weeks['week'].astype(int)

def mmdd_to_date(year, mmdd_str):
    return datetime.strptime(f'{year}{mmdd_str.zfill(4)}', '%Y%m%d')

# 输入: 第(forecast_week-2)周 start  ~  第(forecast_week-1)周 end
w1 = forecast_week - 2
w2 = forecast_week - 1
row1 = df_weeks[df_weeks['week'] == w1].iloc[0]
row2 = df_weeks[df_weeks['week'] == w2].iloc[0]

date_start = mmdd_to_date(forecast_year, str(row1['start']))
date_end   = mmdd_to_date(forecast_year, str(row2['end']))
dates = [date_start + timedelta(days=i)
         for i in range((date_end - date_start).days + 1)]

print(f'Forecast year  : {forecast_year} ({"leap" if is_leap else "common"})')
print(f'Lookup table   : {os.path.basename(lookup_file)}')
print(f'Input weeks    : week {w1} ({row1["start"]}-{row1["end"]}) + '
      f'week {w2} ({row2["start"]}-{row2["end"]})')
print(f'Date range     : {date_start:%Y%m%d} ~ {date_end:%Y%m%d}  ({len(dates)} days)')
print(f'Total downloads: {len(dates)*2} files  (14 days × 2 datasets)')
print(f'Output dir     : {OUTPUT_DIR}')

# ============================================================
# CDS API 变量定义
# ============================================================
SURFACE_VARS = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
    "surface_pressure",
    "mean_eastward_turbulent_surface_stress",
    "mean_northward_turbulent_surface_stress",
    "mean_surface_latent_heat_flux",
    "mean_surface_net_long_wave_radiation_flux",
    "mean_surface_net_short_wave_radiation_flux",
    "mean_surface_sensible_heat_flux",
    "mean_top_net_long_wave_radiation_flux",
    "mean_top_net_short_wave_radiation_flux",
    "surface_net_solar_radiation",
    "surface_net_thermal_radiation",
    "total_cloud_cover",
    "total_column_cloud_ice_water",
    "runoff",
    "convective_rain_rate",
    "large_scale_rain_rate",
    "soil_temperature_level_1",
    "soil_temperature_level_2",
    "soil_temperature_level_3",
    "soil_temperature_level_4",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "boundary_layer_height",
    "sea_ice_cover",
]

PRESSURE_VARS = [
    "fraction_of_cloud_cover",
    "geopotential",
    "ozone_mass_mixing_ratio",
    "specific_cloud_ice_water_content",
    "specific_cloud_liquid_water_content",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

PRESSURE_LEVELS = ["200", "300", "500", "700", "850"]

# ============================================================
# 辅助函数
# ============================================================
def open_cds_download(path):
    """打开 CDS 下载文件，自动处理 zip（含多 NC）和普通 NC。"""
    if zipfile.is_zipfile(path):
        tmpdir = tempfile.mkdtemp()
        datasets = []
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.endswith('.nc'):
                    extracted = z.extract(name, tmpdir)
                    datasets.append(xr.open_dataset(extracted))
        return xr.merge(datasets) if len(datasets) > 1 else datasets[0]
    else:
        return xr.open_dataset(path)


def split_to_daily(ds, day_list, prefix, output_dir):
    """将多日 Dataset 按天拆分，保存为单日 NC 文件。"""
    # 时间轴可能叫 valid_time 或 time
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
    times = pd.to_datetime(ds[time_dim].values)

    for d in sorted(day_list):
        out_path = os.path.join(output_dir, f'{prefix}_{d.strftime("%Y%m%d")}.nc')
        if os.path.exists(out_path):
            print(f'  [skip] {os.path.basename(out_path)}')
            continue
        # 选取当天时间步（日均值应只有一个时间步）
        mask = [t.date() == d.date() for t in times]
        idxs = [i for i, m in enumerate(mask) if m]
        if not idxs:
            print(f'  [warn] no data for {d.date()} in downloaded file')
            continue
        day_ds = ds.isel({time_dim: idxs[0]})
        day_ds.to_netcdf(out_path)
        print(f'  -> {os.path.basename(out_path)}')

# ============================================================
# 按年-月分组，再切成 ≤7 天小块下载，再拆分为日文件
# CDS 对大请求降低优先级，且 derived 数据集每次最多约 8 天
# ============================================================
CHUNK_SIZE = 7  # 每次请求最多天数

def make_chunks(day_list, size):
    day_list = sorted(day_list)
    for i in range(0, len(day_list), size):
        yield day_list[i:i + size]

month_groups = defaultdict(list)
for d in dates:
    month_groups[(d.year, d.month)].append(d)

# 展开成 (year, month, chunk) 三元组列表
chunks = []
for (year, month), day_list in sorted(month_groups.items()):
    for chunk in make_chunks(day_list, CHUNK_SIZE):
        chunks.append((year, month, chunk))

print(f'\nDownload plan: {len(chunks)} chunks × 2 datasets = {len(chunks)*2} requests')
for year, month, chunk in chunks:
    days_str = ', '.join(f'{d.day:02d}' for d in chunk)
    print(f'  {year}-{month:02d}  days=[{days_str}]  ({len(chunk)} days)')

client = cdsapi.Client()

for year, month, chunk in chunks:
    yr  = str(year)
    mo  = f'{month:02d}'
    days = [f'{d.day:02d}' for d in chunk]
    tag  = f'{yr}{mo}{days[0]}-{days[-1]}'
    print(f'\n[{yr}-{mo} days {days[0]}-{days[-1]}]')

    # ---------- 表面层 ----------
    need_surf = [d for d in chunk
                 if not os.path.exists(
                     os.path.join(OUTPUT_DIR, f'ERA5_surface_{d.strftime("%Y%m%d")}.nc'))]
    if need_surf:
        tmp_surf = os.path.join(OUTPUT_DIR, f'_tmp_surface_{tag}.download')
        print(f'  Downloading surface ({len(chunk)} days) ...')
        client.retrieve(
            "derived-era5-single-levels-daily-statistics",
            {
                "product_type": "reanalysis",
                "variable": SURFACE_VARS,
                "year": yr,
                "month": [mo],
                "day": days,
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "6_hourly",
            },
        ).download(tmp_surf)
        ds_surf = open_cds_download(tmp_surf)
        split_to_daily(ds_surf, chunk, 'ERA5_surface', OUTPUT_DIR)
        ds_surf.close()
        os.remove(tmp_surf)
    else:
        print('  [skip] all surface files in this chunk already exist')

    # ---------- 气压层 ----------
    need_pres = [d for d in chunk
                 if not os.path.exists(
                     os.path.join(OUTPUT_DIR, f'ERA5_pressure_{d.strftime("%Y%m%d")}.nc'))]
    if need_pres:
        tmp_pres = os.path.join(OUTPUT_DIR, f'_tmp_pressure_{tag}.download')
        print(f'  Downloading pressure ({len(chunk)} days) ...')
        client.retrieve(
            "derived-era5-pressure-levels-daily-statistics",
            {
                "product_type": "reanalysis",
                "variable": PRESSURE_VARS,
                "year": yr,
                "month": [mo],
                "day": days,
                "pressure_level": PRESSURE_LEVELS,
                "daily_statistic": "daily_mean",
                "time_zone": "utc+00:00",
                "frequency": "6_hourly",
            },
        ).download(tmp_pres)
        ds_pres = open_cds_download(tmp_pres)
        split_to_daily(ds_pres, chunk, 'ERA5_pressure', OUTPUT_DIR)
        ds_pres.close()
        os.remove(tmp_pres)
    else:
        print('  [skip] all pressure files in this chunk already exist')

# ============================================================
# 验证
# ============================================================
print('\n=== Verification ===')
surf_files = sorted([f for f in os.listdir(OUTPUT_DIR)
                     if f.startswith('ERA5_surface_') and f.endswith('.nc')
                     and not f.startswith('_')])
pres_files = sorted([f for f in os.listdir(OUTPUT_DIR)
                     if f.startswith('ERA5_pressure_') and f.endswith('.nc')
                     and not f.startswith('_')])
print(f'Surface  files: {len(surf_files)}  (expected {len(dates)})')
print(f'Pressure files: {len(pres_files)}  (expected {len(dates)})')
for f in surf_files:
    print(f'  {f}')
