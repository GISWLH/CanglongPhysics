from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cmaps
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mplotutils as mpu
import numpy as np
import pandas as pd
import torch
import xarray as xr
from shapely import contains_xy
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT))

from utils import plot as china_plot
from canglong import CanglongV2_5
from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays


INPUT_DIR = Path('/data/lhwang/ERA5_daily_s2s')
MODEL_PATH = ROOT / 'model' / 'model_v3_5_continue_record_ft2_best.pth'
NORM_JSON = ROOT / 'code_v2' / 'ERA5_1940_2023_mean_std_v2.json'
CLIMATE_WEEKLY_PATH = ROOT / 'data' / 'climate_variables_2000_2023_weekly.nc'
CHINA_SHP = ROOT / 'code' / 'data' / 'china.shp'
OUTPUT_ROOT = ROOT / 'analysis' / 'operation' / 'output'
CACHE_DIR = ROOT / 'analysis' / 'operation' / 'cache'
MONTHLY_TP_CLIM_PATH = CACHE_DIR / 'tp_monthly_climatology_2000_2023.nc'
ECMWF_MONTHLY_TP_SOURCE_CANDIDATES = [
    CACHE_DIR / 'tp_origin_fromec_montly_all.py.nc',
    CACHE_DIR / 'tp_origin_fromec_monthly_all.py.nc',
]
ECMWF_MONTHLY_TP_CLIM_PATH = CACHE_DIR / 'china_tp_monthly_climatology_2000_2023_from_ecmwf_tp.nc'
ECMWF_CLIM_YEAR_START = 2000
ECMWF_CLIM_YEAR_END = 2023

FORECAST_YEAR = 2026
TARGET_MONTHS = [3, 4, 5]
CHINA_BOUNDS = {'lon_min': 70.0, 'lon_max': 140.0, 'lat_min': 15.0, 'lat_max': 55.0}
SOIL_DEPTHS = np.array([0.07, 0.21, 0.72, 1.89], dtype=np.float32)
SOIL_WEIGHTS = SOIL_DEPTHS / SOIL_DEPTHS.sum()
SECONDS_PER_DAY = 86400.0
EPS = 1.0e-8
SUBMAP_X_OFFSET = 0.775 + 0.01
SUBMAP_Y_OFFSET = 0.022
SUBMAP_WIDTH = 0.216
SUBMAP_HEIGHT = 0.264

SURFACE_VARS = [
    'avg_tnswrf', 'avg_tnlwrf', 'tciw', 'tcc', 'lsrr', 'crr', 'blh',
    'u10', 'v10', 'd2m', 't2m', 'avg_iews', 'avg_inss', 'slhf', 'sshf',
    'avg_snswrf', 'avg_snlwrf', 'ssr', 'str', 'sp', 'msl', 'siconc',
    'sst', 'ro', 'stl', 'swvl'
]
UPPER_VARS = ['o3', 'z', 't', 'u', 'v', 'w', 'q', 'cc', 'ciwc', 'clwc']
PRESSURE_LEVELS = [200, 300, 500, 700, 850]

SURFACE_DAILY_MAP = {
    'avg_tnswrf': 'avg_tnswrf',
    'avg_tnlwrf': 'avg_tnlwrf',
    'tciw': 'tciw',
    'tcc': 'tcc',
    'lsrr': 'lsrr',
    'crr': 'crr',
    'blh': 'blh',
    'u10': 'u10',
    'v10': 'v10',
    'd2m': 'd2m',
    't2m': 't2m',
    'avg_iews': 'avg_iews',
    'avg_inss': 'avg_inss',
    'avg_snswrf': 'avg_snswrf',
    'avg_snlwrf': 'avg_snlwrf',
    'ssr': 'ssr',
    'str': 'str',
    'sp': 'sp',
    'msl': 'msl',
    'siconc': 'siconc',
    'sst': 'sst',
    'ro': 'ro',
}

REGIONS = {
    'ChinaBBox': {'lon_min': 70.0, 'lon_max': 140.0, 'lat_min': 15.0, 'lat_max': 55.0},
    'Jiangnan': {'lon_min': 109.0, 'lon_max': 122.0, 'lat_min': 24.0, 'lat_max': 31.0},
    'LoessPlateau': {'lon_min': 103.0, 'lon_max': 111.0, 'lat_min': 34.0, 'lat_max': 40.0},
    'NorthChina': {'lon_min': 112.0, 'lon_max': 120.0, 'lat_min': 35.0, 'lat_max': 41.0},
    'SouthChina': {'lon_min': 105.0, 'lon_max': 120.0, 'lat_min': 20.0, 'lat_max': 27.0},
    'Northeast': {'lon_min': 120.0, 'lon_max': 135.0, 'lat_min': 42.0, 'lat_max': 50.0},
}


def month_range(year: int, month: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthEnd(1)
    return start, end


def overlap_days(start_a: pd.Timestamp, end_a: pd.Timestamp, start_b: pd.Timestamp, end_b: pd.Timestamp) -> int:
    start = max(start_a, start_b)
    end = min(end_a, end_b)
    if start > end:
        return 0
    return int((end - start).days) + 1


def find_input_dates(prefix: str) -> list[pd.Timestamp]:
    return [pd.Timestamp(path.stem.split('_')[-1]) for path in sorted(INPUT_DIR.glob(f'{prefix}_*.nc'))]


def find_existing_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def load_surface_day(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = xr.open_dataset(path)
    lat = ds['latitude'].values.astype(np.float32)
    lon = ds['longitude'].values.astype(np.float32)
    data = np.empty((len(SURFACE_VARS), lat.size, lon.size), dtype=np.float32)

    for idx, var in enumerate(SURFACE_VARS):
        if var == 'slhf':
            arr = ds['avg_slhtf'].values.astype(np.float32) * SECONDS_PER_DAY
        elif var == 'sshf':
            arr = ds['avg_ishf'].values.astype(np.float32) * SECONDS_PER_DAY
        elif var == 'stl':
            arr = sum(ds[f'stl{i + 1}'].values.astype(np.float32) * SOIL_WEIGHTS[i] for i in range(4))
        elif var == 'swvl':
            arr = sum(ds[f'swvl{i + 1}'].values.astype(np.float32) * SOIL_WEIGHTS[i] for i in range(4))
        else:
            arr = ds[SURFACE_DAILY_MAP[var]].values.astype(np.float32)
        data[idx] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    ds.close()
    return data, lat, lon


def load_upper_day(path: Path) -> np.ndarray:
    ds = xr.open_dataset(path)
    data = np.empty((len(UPPER_VARS), len(PRESSURE_LEVELS), ds['latitude'].size, ds['longitude'].size), dtype=np.float32)
    for idx, var in enumerate(UPPER_VARS):
        arr = ds[var].sel(pressure_level=PRESSURE_LEVELS).values.astype(np.float32)
        data[idx] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    ds.close()
    return data


def load_observed_daily_precip_china(start_date: pd.Timestamp, end_date: pd.Timestamp, lat_mask, lon_mask):
    current = start_date
    values = []
    dates = []
    while current <= end_date:
        ds = xr.open_dataset(INPUT_DIR / f'ERA5_surface_{current.strftime("%Y%m%d")}.nc')
        day_tp = (ds['lsrr'].values.astype(np.float32) + ds['crr'].values.astype(np.float32)) * SECONDS_PER_DAY
        values.append(day_tp[lat_mask][:, lon_mask])
        dates.append(current)
        ds.close()
        current += pd.Timedelta(days=1)
    return np.stack(values, axis=0), pd.DatetimeIndex(dates)


def build_two_week_inputs(surface_dates: list[pd.Timestamp], pressure_dates: list[pd.Timestamp]):
    if len(surface_dates) != 14 or len(pressure_dates) != 14:
        raise ValueError(f'Expected 14 days, got surface={len(surface_dates)}, pressure={len(pressure_dates)}')

    surface_days = []
    upper_days = []
    lat = None
    lon = None
    for surf_date, pres_date in zip(surface_dates, pressure_dates):
        surface_day, lat, lon = load_surface_day(INPUT_DIR / f'ERA5_surface_{surf_date.strftime("%Y%m%d")}.nc')
        upper_day = load_upper_day(INPUT_DIR / f'ERA5_pressure_{pres_date.strftime("%Y%m%d")}.nc')
        surface_days.append(surface_day)
        upper_days.append(upper_day)

    surface_days = np.stack(surface_days, axis=0)
    upper_days = np.stack(upper_days, axis=0)
    input_surface = np.stack([surface_days[:7].mean(axis=0), surface_days[7:].mean(axis=0)], axis=1)
    input_upper = np.stack([upper_days[:7].mean(axis=0), upper_days[7:].mean(axis=0)], axis=2)
    return input_surface.astype(np.float32), input_upper.astype(np.float32), lat, lon


def prepare_font():
    font_path = Path('/usr/share/fonts/arial/ARIAL.TTF')
    if font_path.exists():
        from matplotlib import font_manager
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        plt.rcParams['font.family'] = 'Arial'


def area_weighted_mean(field: np.ndarray, lat_subset: np.ndarray, lat_mask: np.ndarray, lon_mask: np.ndarray) -> float:
    region = field[lat_mask][:, lon_mask]
    if region.size == 0:
        return math.nan
    weights = np.cos(np.deg2rad(lat_subset[lat_mask])).astype(np.float64)
    weights_2d = np.repeat(weights[:, None], region.shape[1], axis=1)
    valid = np.isfinite(region)
    if not valid.any():
        return math.nan
    return float(np.sum(region[valid] * weights_2d[valid]) / np.sum(weights_2d[valid]))


def build_model(device, surface_mean_np, surface_std_np, upper_mean_np, upper_std_np):
    model = CanglongV2_5(
        surface_mean=torch.from_numpy(surface_mean_np),
        surface_std=torch.from_numpy(surface_std_np),
        upper_mean=torch.from_numpy(upper_mean_np),
        upper_std=torch.from_numpy(upper_std_np),
    )
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def build_or_load_monthly_tp_climatology() -> xr.Dataset:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if MONTHLY_TP_CLIM_PATH.exists():
        return xr.open_dataset(MONTHLY_TP_CLIM_PATH)

    ds = xr.open_dataset(CLIMATE_WEEKLY_PATH)
    tp = ds['tp']
    times = pd.DatetimeIndex(tp['time'].values)
    years = np.unique(times.year)
    lat = ds['lat'].values.astype(np.float32)
    lon = ds['lon'].values.astype(np.float32)
    accum = np.zeros((12, lat.size, lon.size), dtype=np.float64)

    print(f'Building monthly TP climatology from {CLIMATE_WEEKLY_PATH} ...')
    for idx, week_start in enumerate(times):
        arr = tp.isel(time=idx).values.astype(np.float32)
        week_end = week_start + pd.Timedelta(days=6)
        if week_start.month == week_end.month and week_start.year == week_end.year:
            accum[week_start.month - 1] += arr * 7.0
        else:
            first_start, first_end = month_range(week_start.year, week_start.month)
            days_first = overlap_days(week_start, week_end, first_start, first_end)
            if days_first > 0:
                accum[week_start.month - 1] += arr * days_first
            second_start, second_end = month_range(week_end.year, week_end.month)
            days_second = overlap_days(week_start, week_end, second_start, second_end)
            if days_second > 0:
                accum[week_end.month - 1] += arr * days_second

        if (idx + 1) % 200 == 0 or idx == len(times) - 1:
            print(f'  {idx + 1}/{len(times)} weeks accumulated')

    clim = accum / float(len(years))
    ds.close()

    out = xr.Dataset(
        data_vars={
            'tp_monthly_total_mm': (('month', 'lat', 'lon'), clim.astype(np.float32)),
            'year_count': (('month',), np.full(12, len(years), dtype=np.int32)),
        },
        coords={'month': np.arange(1, 13, dtype=np.int32), 'lat': lat, 'lon': lon},
        attrs={
            'description': 'Monthly total precipitation climatology derived from climate_variables_2000_2023_weekly.nc',
            'source': str(CLIMATE_WEEKLY_PATH),
            'years': f'{years.min()}-{years.max()}',
            'units': 'mm',
            'note': 'Computed by weighting each 7-day weekly mean (mm/day) by its overlap days with each calendar month.',
        },
    )
    out.to_netcdf(MONTHLY_TP_CLIM_PATH)
    return xr.open_dataset(MONTHLY_TP_CLIM_PATH)


def build_or_load_monthly_tp_climatology_ecmwf(lat_subset: np.ndarray, lon_subset: np.ndarray) -> xr.Dataset:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if ECMWF_MONTHLY_TP_CLIM_PATH.exists():
        return xr.open_dataset(ECMWF_MONTHLY_TP_CLIM_PATH)

    source_path = find_existing_path(ECMWF_MONTHLY_TP_SOURCE_CANDIDATES)
    if source_path is None:
        raise FileNotFoundError('ECMWF monthly tp source not found in analysis/operation/cache')

    ds = xr.open_dataset(source_path)
    tp = ds['tp'].sel(
        valid_time=slice(f'{ECMWF_CLIM_YEAR_START}-01-01', f'{ECMWF_CLIM_YEAR_END}-12-31'),
        latitude=slice(float(np.max(lat_subset)), float(np.min(lat_subset))),
        longitude=slice(float(np.min(lon_subset)), float(np.max(lon_subset))),
    )
    tp_mm_day = tp.astype(np.float32) * 1000.0
    days_in_month = xr.DataArray(
        tp['valid_time'].dt.days_in_month.astype(np.float32).values,
        dims=('valid_time',),
        coords={'valid_time': tp['valid_time']},
    )
    clim = (tp_mm_day * days_in_month).groupby('valid_time.month').mean('valid_time').load()
    clim = clim.interp(
        latitude=lat_subset.astype(np.float32),
        longitude=lon_subset.astype(np.float32),
    )
    ds.close()

    out = xr.Dataset(
        data_vars={
            'tp_monthly_total_mm': (('month', 'lat', 'lon'), clim.values.astype(np.float32)),
            'year_count': (('month',), np.full(12, ECMWF_CLIM_YEAR_END - ECMWF_CLIM_YEAR_START + 1, dtype=np.int32)),
        },
        coords={
            'month': np.arange(1, 13, dtype=np.int32),
            'lat': lat_subset.astype(np.float32),
            'lon': lon_subset.astype(np.float32),
        },
        attrs={
            'description': 'China monthly total precipitation climatology derived from ECMWF official monthly total precipitation source',
            'source': str(source_path),
            'years': f'{ECMWF_CLIM_YEAR_START}-{ECMWF_CLIM_YEAR_END}',
            'units': 'mm',
            'note': 'Computed directly on monthly data: tp [m] -> mm/day via *1000, then multiplied by actual days_in_month, then averaged by calendar month.',
        },
    )
    out.to_netcdf(ECMWF_MONTHLY_TP_CLIM_PATH)
    return xr.open_dataset(ECMWF_MONTHLY_TP_CLIM_PATH)


def plot_monthly_china_anomaly(anom_percent_da: xr.DataArray, figure_path: Path):
    prepare_font()
    china_shp = gpd.read_file(CHINA_SHP)
    china_geom = unary_union(china_shp.geometry)
    title_prefix = 'CAS-Canglong'
    data_cmap = cmaps.drought_severity_r
    levels = np.linspace(-100, 100, 11)
    norm = colors.Normalize(vmin=-100, vmax=100)
    projection = ccrs.LambertConformal(
        central_longitude=105,
        central_latitude=40,
        standard_parallels=(25.0, 47.0),
    )

    figure_path = Path(figure_path).resolve()
    cwd = Path.cwd()
    try:
        os.chdir(ROOT / 'code')
        fig = plt.figure(figsize=(24, 8.6))
        axes = []
        for idx in range(anom_percent_da.sizes['month']):
            ax = fig.add_subplot(1, 3, idx + 1, projection=projection)
            axes.append(ax)

        masked_list = []
        mappable = None
        for idx in range(anom_percent_da.sizes['month']):
            ax = axes[idx]
            current_data = anom_percent_da.isel(month=idx)
            lon2d, lat2d = np.meshgrid(current_data['lon'].values, current_data['lat'].values)
            china_mask = contains_xy(china_geom, lon2d, lat2d)
            masked = current_data.where(xr.DataArray(china_mask, coords=current_data.coords, dims=current_data.dims))
            masked = masked.clip(min=-100.0, max=100.0)
            masked_list.append(masked)
            mappable = china_plot.one_map_china(
                masked,
                ax,
                cmap=data_cmap,
                levels=levels,
                norm=norm,
                mask_ocean=False,
                add_coastlines=True,
                add_land=False,
                add_river=True,
                add_lake=True,
                add_stock=False,
                add_gridlines=True,
                colorbar=False,
                plotfunc='pcolormesh',
            )
            month_label = str(anom_percent_da['month'].values[idx])
            ax.set_title(f'{title_prefix} {month_label}', fontsize=20)

        cbar_ax = fig.add_axes([0.90, 0.15, 0.012, 0.70])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Precipitation Anomaly (%)', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        plt.subplots_adjust(left=0.025, right=0.88, top=0.92, bottom=0.08, wspace=0.20)
        mpu.set_map_layout(axes, width=80)

        for ax, masked in zip(axes, masked_list):
            pos = ax.get_position()
            ax2 = fig.add_axes(
                [pos.x0 + pos.width * SUBMAP_X_OFFSET, pos.y0 + pos.height * SUBMAP_Y_OFFSET, pos.width * SUBMAP_WIDTH, pos.height * SUBMAP_HEIGHT],
                projection=projection,
            )
            china_plot.sub_china_map(masked, ax2, cmap=data_cmap, levels=levels, norm=norm,
                                     add_coastlines=False, add_land=False)

        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    finally:
        os.chdir(cwd)


def run(make_plot: bool = True):
    surface_dates = find_input_dates('ERA5_surface')
    pressure_dates = find_input_dates('ERA5_pressure')
    input_surface_np, input_upper_np, lat, lon = build_two_week_inputs(surface_dates, pressure_dates)

    forecast_start = max(surface_dates) + pd.Timedelta(days=1)
    target_end = pd.Timestamp(year=FORECAST_YEAR, month=max(TARGET_MONTHS), day=1) + pd.offsets.MonthEnd(1)
    output_dir = OUTPUT_ROOT / f'{forecast_start.strftime("%Y%m%d")}_v35_ft2_best'
    output_dir.mkdir(parents=True, exist_ok=True)

    lat_mask = (lat <= CHINA_BOUNDS['lat_max']) & (lat >= CHINA_BOUNDS['lat_min'])
    lon_mask = (lon >= CHINA_BOUNDS['lon_min']) & (lon <= CHINA_BOUNDS['lon_max'])
    lat_china = lat[lat_mask]
    lon_china = lon[lon_mask]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Forecast start: {forecast_start:%Y-%m-%d}')
    print(f'Target end    : {target_end:%Y-%m-%d}')
    print(f'Model         : {MODEL_PATH.name}')
    print(f'Output dir    : {output_dir}')

    surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(str(NORM_JSON))
    surface_mean = torch.from_numpy(surface_mean_np).to(device=device, dtype=torch.float32)
    surface_std = torch.from_numpy(surface_std_np).to(device=device, dtype=torch.float32)
    upper_mean = torch.from_numpy(upper_mean_np).to(device=device, dtype=torch.float32)
    upper_std = torch.from_numpy(upper_std_np).to(device=device, dtype=torch.float32)

    model = build_model(device, surface_mean_np, surface_std_np, upper_mean_np, upper_std_np)

    current_surface = torch.from_numpy(input_surface_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    current_upper = torch.from_numpy(input_upper_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    current_surface = (current_surface - surface_mean) / surface_std
    current_upper = (current_upper - upper_mean) / upper_std

    monthly_clim_ds = build_or_load_monthly_tp_climatology()
    monthly_clim_mm = (
        monthly_clim_ds['tp_monthly_total_mm']
        .sel(month=TARGET_MONTHS)
        .sel(lat=slice(CHINA_BOUNDS['lat_max'], CHINA_BOUNDS['lat_min']), lon=slice(CHINA_BOUNDS['lon_min'], CHINA_BOUNDS['lon_max']))
        .values.astype(np.float32)
    )
    monthly_clim_sources = {f'{FORECAST_YEAR}-{TARGET_MONTHS[0]:02d}': str(MONTHLY_TP_CLIM_PATH.resolve())}
    climatology_mode = 'weekly_lsrr_crr_only'
    climatology_note = 'Monthly anomaly uses weekly lsrr+crr monthly climatology for all target months.'
    monthly_clim_ds.close()

    if len(TARGET_MONTHS) > 1:
        ecmwf_source = find_existing_path(ECMWF_MONTHLY_TP_SOURCE_CANDIDATES)
        if ecmwf_source is not None or ECMWF_MONTHLY_TP_CLIM_PATH.exists():
            ecmwf_clim_ds = build_or_load_monthly_tp_climatology_ecmwf(lat_china, lon_china)
            ecmwf_clim_mm = ecmwf_clim_ds['tp_monthly_total_mm'].sel(month=TARGET_MONTHS[1:]).values.astype(np.float32)
            monthly_clim_mm[1:] = ecmwf_clim_mm
            ecmwf_clim_ds.close()
            for month in TARGET_MONTHS[1:]:
                monthly_clim_sources[f'{FORECAST_YEAR}-{month:02d}'] = str(ECMWF_MONTHLY_TP_CLIM_PATH.resolve())
            climatology_mode = 'hybrid_first_old_rest_ecmwf'
            climatology_note = 'Monthly anomaly uses weekly lsrr+crr climatology for the first target month and ECMWF official monthly tp climatology for the remaining target months.'
            print(f'Hybrid climatology enabled: first month from {MONTHLY_TP_CLIM_PATH.name}, remaining months from {ECMWF_MONTHLY_TP_CLIM_PATH.name}')
        else:
            for month in TARGET_MONTHS[1:]:
                monthly_clim_sources[f'{FORECAST_YEAR}-{month:02d}'] = str(MONTHLY_TP_CLIM_PATH.resolve())

    forecast_steps = 0
    cursor = forecast_start
    while cursor <= target_end:
        forecast_steps += 1
        cursor += pd.Timedelta(days=7)

    lssr_idx = SURFACE_VARS.index('lsrr')
    crr_idx = SURFACE_VARS.index('crr')
    use_amp = device.type == 'cuda'

    forecast_times = []
    weekly_tp_rate_china = []
    weekly_tp_mm_day_china = []
    monthly_pred_mm = {month: np.zeros((lat_china.size, lon_china.size), dtype=np.float64) for month in TARGET_MONTHS}
    monthly_coverage_days = {month: 0 for month in TARGET_MONTHS}

    feb_obs_start = pd.Timestamp(year=FORECAST_YEAR, month=2, day=1)
    feb_obs_end = forecast_start - pd.Timedelta(days=1)
    if feb_obs_end >= feb_obs_start:
        observed_feb_mm, observed_feb_dates = load_observed_daily_precip_china(feb_obs_start, feb_obs_end, lat_mask, lon_mask)
        monthly_pred_mm[2] += observed_feb_mm.sum(axis=0).astype(np.float64)
        monthly_coverage_days[2] += len(observed_feb_dates)

    week_start = forecast_start
    with torch.inference_mode():
        for step in range(forecast_steps):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                output_surface, output_upper = model(current_surface, current_upper)

            output_surface = output_surface.float()
            output_upper = output_upper.float()
            pred_tp_rate = (
                output_surface[:, lssr_idx, 0] * surface_std[:, lssr_idx, 0] + surface_mean[:, lssr_idx, 0]
                + output_surface[:, crr_idx, 0] * surface_std[:, crr_idx, 0] + surface_mean[:, crr_idx, 0]
            )
            pred_tp_rate_china = pred_tp_rate[0, lat_mask][:, lon_mask].detach().cpu().numpy().astype(np.float32)
            pred_tp_mm_day_china = pred_tp_rate_china * SECONDS_PER_DAY
            weekly_tp_rate_china.append(pred_tp_rate_china)
            weekly_tp_mm_day_china.append(pred_tp_mm_day_china)
            forecast_times.append(week_start)

            week_end = week_start + pd.Timedelta(days=6)
            for month in TARGET_MONTHS:
                month_start, month_end = month_range(FORECAST_YEAR, month)
                days = overlap_days(week_start, week_end, month_start, month_end)
                if days > 0:
                    monthly_pred_mm[month] += pred_tp_mm_day_china.astype(np.float64) * days
                    monthly_coverage_days[month] += days

            current_surface = torch.cat([current_surface[:, :, 1:2], output_surface], dim=2)
            current_upper = torch.cat([current_upper[:, :, :, 1:2], output_upper], dim=3)
            week_start += pd.Timedelta(days=7)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f'[{step + 1:02d}/{forecast_steps:02d}] {forecast_times[-1]:%Y-%m-%d} done')

    weekly_tp_rate_china = np.stack(weekly_tp_rate_china, axis=0)
    weekly_tp_mm_day_china = np.stack(weekly_tp_mm_day_china, axis=0)

    month_labels = [f'{FORECAST_YEAR}-{month:02d}' for month in TARGET_MONTHS]
    monthly_pred_stack = np.stack([monthly_pred_mm[m].astype(np.float32) for m in TARGET_MONTHS], axis=0)
    monthly_anom_raw = (monthly_pred_stack - monthly_clim_mm) / (monthly_clim_mm + EPS) * 100.0

    weekly_ds = xr.Dataset(
        data_vars={
            'tp_rate': (('time', 'lat', 'lon'), weekly_tp_rate_china.astype(np.float32)),
            'tp_mm_day': (('time', 'lat', 'lon'), weekly_tp_mm_day_china.astype(np.float32)),
        },
        coords={'time': pd.DatetimeIndex(forecast_times), 'lat': lat_china, 'lon': lon_china},
        attrs={
            'model': MODEL_PATH.name,
            'forecast_start': forecast_start.strftime('%Y-%m-%d'),
            'forecast_end': forecast_times[-1].strftime('%Y-%m-%d'),
            'description': 'Weekly China precipitation forecast from CAS-Canglong v3.5 ft2 best checkpoint',
            'tp_rate_units': 'kg m^-2 s^-1',
            'tp_mm_day_units': 'mm day^-1',
        },
    )
    weekly_path = output_dir / 'china_weekly_tp_forecast.nc'
    weekly_ds.to_netcdf(weekly_path)

    monthly_ds = xr.Dataset(
        data_vars={
            'pred_total_mm': (('month', 'lat', 'lon'), monthly_pred_stack.astype(np.float32)),
            'clim_total_mm': (('month', 'lat', 'lon'), monthly_clim_mm.astype(np.float32)),
            'anom_percent': (('month', 'lat', 'lon'), monthly_anom_raw.astype(np.float32)),
            'coverage_days': (('month',), np.array([monthly_coverage_days[m] for m in TARGET_MONTHS], dtype=np.int32)),
        },
        coords={'month': month_labels, 'lat': lat_china, 'lon': lon_china},
        attrs={
            'model': MODEL_PATH.name,
            'forecast_start': forecast_start.strftime('%Y-%m-%d'),
            'monthly_climatology': json.dumps(monthly_clim_sources, ensure_ascii=False),
            'monthly_climatology_mode': climatology_mode,
            'notes': f'{climatology_note} February combines 2026-02-01 to 2026-02-04 observed input days and 2026-02-05 onward rolling forecast.',
        },
    )
    monthly_path = output_dir / 'china_monthly_tp_feb_mar_apr.nc'
    monthly_ds.to_netcdf(monthly_path)

    summary_rows = []
    for month_idx, month in enumerate(TARGET_MONTHS):
        for region_name, box in REGIONS.items():
            region_lat_mask = (lat_china <= box['lat_max']) & (lat_china >= box['lat_min'])
            region_lon_mask = (lon_china >= box['lon_min']) & (lon_china <= box['lon_max'])
            pred_mean = area_weighted_mean(monthly_pred_stack[month_idx], lat_china, region_lat_mask, region_lon_mask)
            clim_mean = area_weighted_mean(monthly_clim_mm[month_idx], lat_china, region_lat_mask, region_lon_mask)
            anom_mean = (pred_mean - clim_mean) / (clim_mean + EPS) * 100.0 if np.isfinite(pred_mean) and np.isfinite(clim_mean) else math.nan
            summary_rows.append({
                'month': f'{FORECAST_YEAR}-{month:02d}',
                'region': region_name,
                'pred_total_mm': pred_mean,
                'clim_total_mm': clim_mean,
                'anom_percent': anom_mean,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / 'china_region_monthly_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    fig_path = output_dir / 'china_monthly_tp_anomaly_feb_mar_apr.png'
    if make_plot:
        anom_da = xr.DataArray(
            np.clip(monthly_anom_raw.astype(np.float32), -100.0, 100.0),
            coords={'month': month_labels, 'lat': lat_china, 'lon': lon_china},
            dims=('month', 'lat', 'lon'),
            name='anom_percent_plot',
        )
        plot_monthly_china_anomaly(anom_da, fig_path)
    else:
        fig_path = None

    metadata = {
        'model': str(MODEL_PATH.resolve()),
        'forecast_start': forecast_start.strftime('%Y-%m-%d'),
        'forecast_steps': forecast_steps,
        'target_months': month_labels,
        'monthly_climatology_mode': climatology_mode,
        'monthly_climatology_paths': monthly_clim_sources,
        'weekly_output': str(weekly_path.resolve()),
        'monthly_output': str(monthly_path.resolve()),
        'summary_output': str(summary_path.resolve()),
        'figure_output': str(fig_path.resolve()) if fig_path is not None else None,
    }
    meta_path = output_dir / 'run_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print('Saved files:')
    print(f'  {weekly_path}')
    print(f'  {monthly_path}')
    print(f'  {summary_path}')
    if fig_path is not None:
        print(f'  {fig_path}')
    else:
        print('  figure skipped (--skip-plot)')
    print(f'  {meta_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run CAS-Canglong China precipitation operation workflow.')
    parser.add_argument('--skip-plot', action='store_true', help='Run inference and save NetCDF/CSV outputs without generating the PNG figure.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(make_plot=not args.skip_plot)
