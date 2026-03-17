from __future__ import annotations

import argparse
import calendar
import importlib.util
import json
import os
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

import cdsapi
import numpy as np
import pandas as pd
import torch
import xarray as xr
from scipy.special import gamma as gamma_function

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER_PATH = SCRIPT_DIR / 'run_china_precip_v35.py'
CHUNK_SIZE = 7
SECONDS_PER_DAY = 86400.0
MIN_HIST_SAMPLES = 10
SURFACE_HISTORY_VARS = [
    '2m_dewpoint_temperature',
    '2m_temperature',
    'convective_rain_rate',
    'large_scale_rain_rate',
]


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_china_precip_v35', RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def open_cds_download(path: Path):
    if zipfile.is_zipfile(path):
        tmpdir = tempfile.mkdtemp()
        datasets = []
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                if name.endswith('.nc'):
                    extracted = zf.extract(name, tmpdir)
                    datasets.append(xr.open_dataset(extracted))
        if not datasets:
            raise FileNotFoundError(f'No nc file found inside {path}')
        return xr.merge(datasets) if len(datasets) > 1 else datasets[0]
    return xr.open_dataset(path)


def split_to_daily(ds: xr.Dataset, day_list: list[pd.Timestamp], prefix: str, output_dir: Path):
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
    times = pd.to_datetime(ds[time_dim].values)
    for day in sorted(day_list):
        out_path = output_dir / f'{prefix}_{day.strftime("%Y%m%d")}.nc'
        if out_path.exists():
            continue
        mask = [ts.date() == day.date() for ts in times]
        idxs = [idx for idx, flag in enumerate(mask) if flag]
        if not idxs:
            print(f'  [warn] no daily record found for {day:%Y-%m-%d}')
            continue
        day_ds = ds.isel({time_dim: idxs[0]})
        day_ds.to_netcdf(out_path)
        print(f'  -> {out_path.name}')


def chunk_days(day_list: list[pd.Timestamp], size: int):
    day_list = sorted(day_list)
    for idx in range(0, len(day_list), size):
        yield day_list[idx:idx + size]


def ensure_surface_history(surface_dir: Path, start_date: pd.Timestamp, end_date: pd.Timestamp):
    all_days = [start_date + pd.Timedelta(days=offset) for offset in range((end_date - start_date).days + 1)]
    missing_days = [
        day for day in all_days
        if not (surface_dir / f'ERA5_surface_{day.strftime("%Y%m%d")}.nc').exists()
    ]
    if not missing_days:
        print('Antecedent surface history already complete.')
        return []

    print(f'Missing antecedent surface days: {[day.strftime("%Y-%m-%d") for day in missing_days]}')
    month_groups = defaultdict(list)
    for day in missing_days:
        month_groups[(day.year, day.month)].append(day)

    client = cdsapi.Client()
    downloaded = []
    for (year, month), days in sorted(month_groups.items()):
        for chunk in chunk_days(days, CHUNK_SIZE):
            yr = str(year)
            mo = f'{month:02d}'
            day_list = [f'{day.day:02d}' for day in chunk]
            tag = f'{yr}{mo}{day_list[0]}-{day_list[-1]}'
            tmp_path = surface_dir / f'_tmp_spei4_surface_{tag}.download'
            print(f'Downloading antecedent surface chunk {yr}-{mo} days {day_list[0]}-{day_list[-1]} ...')
            client.retrieve(
                'derived-era5-single-levels-daily-statistics',
                {
                    'product_type': 'reanalysis',
                    'variable': SURFACE_HISTORY_VARS,
                    'year': yr,
                    'month': [mo],
                    'day': day_list,
                    'daily_statistic': 'daily_mean',
                    'time_zone': 'utc+00:00',
                    'frequency': '6_hourly',
                },
            ).download(str(tmp_path))
            ds = open_cds_download(tmp_path)
            split_to_daily(ds, chunk, 'ERA5_surface', surface_dir)
            ds.close()
            tmp_path.unlink(missing_ok=True)
            downloaded.extend(chunk)
    return downloaded


def calculate_pet(temp_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    es = 0.618 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = 0.618 * np.exp(17.27 * dewpoint_c / (dewpoint_c + 237.3))
    ratio = np.full_like(temp_c, np.nan, dtype=np.float32)
    valid = es > 1.0e-9
    ratio[valid] = ea[valid] / es[valid]
    ratio = np.clip(ratio, None, 1.0)
    pet = 4.5 * np.power((1.0 + temp_c / 25.0), 2.0) * (1.0 - ratio)
    pet = np.maximum(pet, 0.0)
    return pet.astype(np.float32)


def calculate_pwm(series: np.ndarray):
    n = len(series)
    if n < 3:
        return np.nan, np.nan, np.nan
    sorted_series = np.sort(series)
    f_vals = (np.arange(1, n + 1) - 0.35) / n
    one_minus_f = 1.0 - f_vals
    w0 = np.mean(sorted_series)
    w1 = np.sum(sorted_series * one_minus_f) / n
    w2 = np.sum(sorted_series * (one_minus_f ** 2)) / n
    return w0, w1, w2


def calculate_loglogistic_params(w0, w1, w2):
    if np.isnan(w0) or np.isnan(w1) or np.isnan(w2):
        return np.nan, np.nan, np.nan
    numerator_beta = (2.0 * w1) - w0
    denominator_beta = (6.0 * w1) - w0 - (6.0 * w2)
    if np.isclose(denominator_beta, 0.0):
        return np.nan, np.nan, np.nan
    beta = numerator_beta / denominator_beta
    if beta <= 1.0:
        return np.nan, np.nan, np.nan
    try:
        term1 = gamma_function(1.0 + (1.0 / beta))
        term2 = gamma_function(1.0 - (1.0 / beta))
    except ValueError:
        return np.nan, np.nan, np.nan
    denom_alpha = term1 * term2
    if np.isclose(denom_alpha, 0.0):
        return np.nan, np.nan, np.nan
    alpha = ((w0 - (2.0 * w1)) * beta) / denom_alpha
    if alpha <= 0.0:
        return np.nan, np.nan, np.nan
    gamma_param = w0 - (alpha * denom_alpha)
    return alpha, beta, gamma_param


def loglogistic_cdf(x, alpha, beta, gamma_param):
    if np.isnan(alpha) or x <= gamma_param:
        return 1.0e-9
    term = (alpha / (x - gamma_param)) ** beta
    if np.isinf(term) or term > 1.0e18:
        return 1.0e-9
    cdf = 1.0 / (1.0 + term)
    return np.clip(cdf, 1.0e-9, 1.0 - 1.0e-9)


def cdf_to_spei(prob):
    if np.isnan(prob):
        return np.nan
    prob = float(np.clip(prob, 1.0e-9, 1.0 - 1.0e-9))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    if prob <= 0.5:
        w = np.sqrt(-2.0 * np.log(prob))
        return -(w - (c0 + c1 * w + c2 * w * w) / (1.0 + d1 * w + d2 * w * w + d3 * w * w * w))
    w = np.sqrt(-2.0 * np.log(1.0 - prob))
    return w - (c0 + c1 * w + c2 * w * w) / (1.0 + d1 * w + d2 * w * w + d3 * w * w * w)


def calculate_spei_for_pixel(historical_series: np.ndarray, current_value: float):
    if np.isnan(current_value):
        return np.nan
    valid_hist = historical_series[~np.isnan(historical_series)]
    if len(valid_hist) < MIN_HIST_SAMPLES:
        return np.nan
    w0, w1, w2 = calculate_pwm(valid_hist)
    if np.isnan(w0):
        return np.nan
    alpha, beta, gamma_param = calculate_loglogistic_params(w0, w1, w2)
    if np.isnan(alpha):
        return np.nan
    prob = loglogistic_cdf(current_value, alpha, beta, gamma_param)
    return cdf_to_spei(prob)


def get_week_of_year(date_array: xr.DataArray):
    day_of_year = date_array.dt.dayofyear
    week_num = ((day_of_year - 1) // 7) + 1
    return week_num.clip(max=52)


def subset_masks(lat: np.ndarray, lon: np.ndarray, bounds: dict):
    lat_mask = (lat >= bounds['lat_min']) & (lat <= bounds['lat_max'])
    lon_mask = (lon >= bounds['lon_min']) & (lon <= bounds['lon_max'])
    return lat_mask, lon_mask


def load_observed_week_records(op, history_start: pd.Timestamp, forecast_start: pd.Timestamp, lat_mask, lon_mask):
    records = []
    current_week_start = history_start
    while current_week_start < forecast_start:
        day_list = [current_week_start + pd.Timedelta(days=offset) for offset in range(7)]
        precip_days = []
        t2m_days = []
        d2m_days = []
        for day in day_list:
            ds = xr.open_dataset(op.INPUT_DIR / f'ERA5_surface_{day.strftime("%Y%m%d")}.nc')
            precip_days.append(((ds['lsrr'].values.astype(np.float32) + ds['crr'].values.astype(np.float32)) * SECONDS_PER_DAY)[lat_mask][:, lon_mask])
            t2m_days.append((ds['t2m'].values.astype(np.float32) - 273.15)[lat_mask][:, lon_mask])
            d2m_days.append((ds['d2m'].values.astype(np.float32) - 273.15)[lat_mask][:, lon_mask])
            ds.close()
        precip_week = np.stack(precip_days, axis=0).mean(axis=0).astype(np.float32)
        t2m_week = np.stack(t2m_days, axis=0).mean(axis=0).astype(np.float32)
        d2m_week = np.stack(d2m_days, axis=0).mean(axis=0).astype(np.float32)
        pet_week = calculate_pet(t2m_week, d2m_week)
        records.append(
            {
                'time': current_week_start,
                'source_flag': 0,
                'tp_mm_day': precip_week,
                't2m_c': t2m_week,
                'd2m_c': d2m_week,
                'pet_mm_day': pet_week,
                'water_balance_mm_day': (precip_week - pet_week).astype(np.float32),
            }
        )
        current_week_start += pd.Timedelta(days=7)
    return records


def build_forecast_week_records(op, forecast_start: pd.Timestamp, target_end: pd.Timestamp, lat_mask, lon_mask):
    surface_dates = set(op.find_input_dates('ERA5_surface'))
    pressure_dates = set(op.find_input_dates('ERA5_pressure'))
    common_dates = sorted(surface_dates & pressure_dates)
    if len(common_dates) < 14:
        raise ValueError(f'Need at least 14 common surface/pressure days, got {len(common_dates)}')
    input_dates = common_dates[-14:]
    input_surface_np, input_upper_np, _, _ = op.build_two_week_inputs(input_dates, input_dates)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device  : {device}')
    surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = op.load_normalization_arrays(str(op.NORM_JSON))
    surface_mean = torch.from_numpy(surface_mean_np).to(device=device, dtype=torch.float32)
    surface_std = torch.from_numpy(surface_std_np).to(device=device, dtype=torch.float32)
    upper_mean = torch.from_numpy(upper_mean_np).to(device=device, dtype=torch.float32)
    upper_std = torch.from_numpy(upper_std_np).to(device=device, dtype=torch.float32)

    model = op.build_model(device, surface_mean_np, surface_std_np, upper_mean_np, upper_std_np)
    current_surface = torch.from_numpy(input_surface_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    current_upper = torch.from_numpy(input_upper_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    current_surface = (current_surface - surface_mean) / surface_std
    current_upper = (current_upper - upper_mean) / upper_std

    lssr_idx = op.SURFACE_VARS.index('lsrr')
    crr_idx = op.SURFACE_VARS.index('crr')
    t2m_idx = op.SURFACE_VARS.index('t2m')
    d2m_idx = op.SURFACE_VARS.index('d2m')
    use_amp = device.type == 'cuda'

    forecast_steps = 0
    cursor = forecast_start
    while cursor <= target_end:
        forecast_steps += 1
        cursor += pd.Timedelta(days=7)

    records = []
    week_start = forecast_start
    with torch.inference_mode():
        for step in range(forecast_steps):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                output_surface, output_upper = model(current_surface, current_upper)
            output_surface = output_surface.float()
            surf_phys = output_surface * surface_std + surface_mean

            tp_mm_day = ((surf_phys[:, lssr_idx, 0] + surf_phys[:, crr_idx, 0]) * SECONDS_PER_DAY)[0].detach().cpu().numpy().astype(np.float32)
            t2m_c = (surf_phys[:, t2m_idx, 0][0] - 273.15).detach().cpu().numpy().astype(np.float32)
            d2m_c = (surf_phys[:, d2m_idx, 0][0] - 273.15).detach().cpu().numpy().astype(np.float32)

            tp_mm_day = tp_mm_day[lat_mask][:, lon_mask]
            t2m_c = t2m_c[lat_mask][:, lon_mask]
            d2m_c = d2m_c[lat_mask][:, lon_mask]
            pet_mm_day = calculate_pet(t2m_c, d2m_c)

            records.append(
                {
                    'time': week_start,
                    'source_flag': 1,
                    'tp_mm_day': tp_mm_day.astype(np.float32),
                    't2m_c': t2m_c.astype(np.float32),
                    'd2m_c': d2m_c.astype(np.float32),
                    'pet_mm_day': pet_mm_day.astype(np.float32),
                    'water_balance_mm_day': (tp_mm_day - pet_mm_day).astype(np.float32),
                }
            )
            current_surface = torch.cat([current_surface[:, :, 1:2], output_surface], dim=2)
            current_upper = torch.cat([current_upper[:, :, :, 1:2], output_upper], dim=3)
            week_start += pd.Timedelta(days=7)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f'Forecast week [{step + 1:02d}/{forecast_steps:02d}] {records[-1]["time"]:%Y-%m-%d} done')
    return records


def build_historical_water_balance(op, lat_mask, lon_mask):
    ds = xr.open_dataset(op.CLIMATE_WEEKLY_PATH)
    climate = ds[['tp', 'pet']].sel(lat=ds['lat'][lat_mask], lon=ds['lon'][lon_mask]).load()
    ds.close()
    d_hist = (climate['tp'] - climate['pet']).astype(np.float32)
    return d_hist


def compute_spei4(records: list[dict], d_hist: xr.DataArray):
    times = pd.DatetimeIndex([record['time'] for record in records])
    lat = d_hist['lat'].values.astype(np.float32)
    lon = d_hist['lon'].values.astype(np.float32)
    wb_stack = np.stack([record['water_balance_mm_day'] for record in records], axis=0).astype(np.float32)
    wb_da = xr.DataArray(wb_stack, coords={'time': times, 'lat': lat, 'lon': lon}, dims=('time', 'lat', 'lon'))

    hist_week_numbers = get_week_of_year(d_hist['time'])
    pred_week_numbers = get_week_of_year(wb_da['time'])
    hist_years = np.unique(pd.DatetimeIndex(d_hist['time'].values).year)

    forecast_times = []
    accum_list = []
    spei_list = []
    for idx in range(3, wb_da.sizes['time']):
        curr_accum = wb_da.isel(time=slice(idx - 3, idx + 1)).sum(dim='time')
        curr_week_num = int(pred_week_numbers.isel(time=idx).item())
        hist_4week_accum_list = []
        for year in hist_years:
            year_mask = d_hist['time'].dt.year == year
            year_data = d_hist.where(year_mask, drop=True)
            year_weeks = hist_week_numbers.where(year_mask, drop=True)
            week_indices = np.where(year_weeks.values == curr_week_num)[0]
            if len(week_indices) == 0:
                continue
            week_idx = int(week_indices[0])
            if week_idx >= 3:
                hist_4week_accum_list.append(year_data.isel(time=slice(week_idx - 3, week_idx + 1)).sum(dim='time'))
                continue
            if year <= hist_years.min():
                continue
            prev_year = year - 1
            if prev_year not in hist_years:
                continue
            prev_year_data = d_hist.where(d_hist['time'].dt.year == prev_year, drop=True)
            current_part = year_data.isel(time=slice(0, week_idx + 1)).sum(dim='time')
            weeks_needed = 4 - (week_idx + 1)
            if prev_year_data.sizes['time'] < weeks_needed:
                continue
            prev_part = prev_year_data.isel(time=slice(prev_year_data.sizes['time'] - weeks_needed, prev_year_data.sizes['time'])).sum(dim='time')
            hist_4week_accum_list.append((current_part + prev_part).astype(np.float32))

        if not hist_4week_accum_list:
            spei_map = xr.full_like(curr_accum, np.nan)
            hist_count = 0
        else:
            hist_4week_accum = xr.concat(hist_4week_accum_list, dim='time')
            hist_count = hist_4week_accum.sizes['time']
            if hist_count < MIN_HIST_SAMPLES:
                spei_map = xr.full_like(curr_accum, np.nan)
            else:
                spei_map = xr.apply_ufunc(
                    calculate_spei_for_pixel,
                    hist_4week_accum,
                    curr_accum,
                    input_core_dims=[['time'], []],
                    output_core_dims=[[]],
                    exclude_dims=set(('time',)),
                    vectorize=True,
                    output_dtypes=[np.float32],
                    keep_attrs=True,
                )
        forecast_times.append(times[idx])
        accum_list.append(curr_accum.astype(np.float32))
        spei_list.append(spei_map.astype(np.float32))
        print(f'SPEI-4 [{idx - 2:02d}/{wb_da.sizes["time"] - 3:02d}] {times[idx]:%Y-%m-%d} hist_samples={hist_count}')

    accum_da = xr.concat(accum_list, dim=xr.IndexVariable('forecast_time', pd.DatetimeIndex(forecast_times)))
    spei_da = xr.concat(spei_list, dim=xr.IndexVariable('forecast_time', pd.DatetimeIndex(forecast_times)))
    accum_da.name = 'water_balance_4week_mm_day'
    spei_da.name = 'spei4'
    return accum_da, spei_da


def parse_args():
    parser = argparse.ArgumentParser(description='Generate China weekly drought operation products with SPEI-4 for CAS-Canglong S2S.')
    parser.add_argument('--end-month', type=int, default=12, help='Forecast through this month of the forecast year.')
    return parser.parse_args()


def main():
    args = parse_args()
    op = load_runner_module()
    surface_dates = op.find_input_dates('ERA5_surface')
    if not surface_dates:
        raise FileNotFoundError(f'No ERA5 surface daily files found under {op.INPUT_DIR}')
    forecast_start = max(surface_dates) + pd.Timedelta(days=1)
    forecast_year = forecast_start.year
    if args.end_month < forecast_start.month:
        raise ValueError(f'end-month {args.end_month} is earlier than forecast start month {forecast_start.month}')
    target_end = pd.Timestamp(year=forecast_year, month=args.end_month, day=1) + pd.offsets.MonthEnd(1)

    sample_ds = xr.open_dataset(op.INPUT_DIR / f'ERA5_surface_{max(surface_dates).strftime("%Y%m%d")}.nc')
    lat = sample_ds['latitude'].values.astype(np.float32)
    lon = sample_ds['longitude'].values.astype(np.float32)
    sample_ds.close()
    lat_mask, lon_mask = subset_masks(lat, lon, op.CHINA_BOUNDS)
    lat_china = lat[lat_mask].astype(np.float32)
    lon_china = lon[lon_mask].astype(np.float32)

    antecedent_start = forecast_start - pd.Timedelta(days=21)
    antecedent_end = forecast_start - pd.Timedelta(days=1)
    downloaded_days = ensure_surface_history(op.INPUT_DIR, antecedent_start, antecedent_end)

    output_dir = op.OUTPUT_ROOT / f'{forecast_start.strftime("%Y%m%d")}_v35_ft2_best'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'operation_s2s_drought_weekly_spei4_china_{forecast_start.strftime("%Y%m%d")}_{target_end.strftime("%Y%m%d")}.nc'
    metadata_path = output_dir / f'operation_s2s_drought_metadata_{forecast_start.strftime("%Y%m%d")}.json'

    print(f'Forecast start : {forecast_start:%Y-%m-%d}')
    print(f'Forecast end   : {target_end:%Y-%m-%d}')
    print(f'Antecedent obs : {antecedent_start:%Y-%m-%d} to {antecedent_end:%Y-%m-%d}')
    print(f'Output         : {output_path}')

    observed_records = load_observed_week_records(op, antecedent_start, forecast_start, lat_mask, lon_mask)
    forecast_records = build_forecast_week_records(op, forecast_start, target_end, lat_mask, lon_mask)
    records = observed_records + forecast_records
    d_hist = build_historical_water_balance(op, lat_mask, lon_mask)
    accum_da, spei_da = compute_spei4(records, d_hist)

    all_times = pd.DatetimeIndex([record['time'] for record in records])
    source_flags = np.array([record['source_flag'] for record in records], dtype=np.int8)
    tp_stack = np.stack([record['tp_mm_day'] for record in records], axis=0).astype(np.float32)
    t2m_stack = np.stack([record['t2m_c'] for record in records], axis=0).astype(np.float32)
    d2m_stack = np.stack([record['d2m_c'] for record in records], axis=0).astype(np.float32)
    pet_stack = np.stack([record['pet_mm_day'] for record in records], axis=0).astype(np.float32)
    wb_stack = np.stack([record['water_balance_mm_day'] for record in records], axis=0).astype(np.float32)

    out = xr.Dataset(
        data_vars={
            'tp_mm_day': (('time', 'lat', 'lon'), tp_stack),
            't2m_c': (('time', 'lat', 'lon'), t2m_stack),
            'd2m_c': (('time', 'lat', 'lon'), d2m_stack),
            'pet_mm_day': (('time', 'lat', 'lon'), pet_stack),
            'water_balance_mm_day': (('time', 'lat', 'lon'), wb_stack),
            'source_flag': (('time',), source_flags),
            'water_balance_4week_mm_day': (('forecast_time', 'lat', 'lon'), accum_da.values.astype(np.float32)),
            'spei4': (('forecast_time', 'lat', 'lon'), spei_da.values.astype(np.float32)),
        },
        coords={
            'time': all_times,
            'forecast_time': pd.DatetimeIndex(accum_da['forecast_time'].values),
            'lat': lat_china,
            'lon': lon_china,
        },
        attrs={
            'model': op.MODEL_PATH.name,
            'forecast_start': forecast_start.strftime('%Y-%m-%d'),
            'forecast_end': target_end.strftime('%Y-%m-%d'),
            'description': 'China weekly drought operation products for CAS-Canglong S2S. First 3 weekly records are observed antecedent weeks; forecast_time begins at the first model forecast week.',
            'spei_method': 'Run.py PWM + log-logistic fit on 2000-2023 weekly climate water balance, 4-week accumulation.',
            'region': 'China bbox 70-140E, 15-55N',
            'note': 'tp and pet are weekly mean mm/day; water_balance_4week_mm_day is the sum of 4 weekly mean balances; source_flag 0=observed antecedent, 1=forecast.',
        },
    )
    out.to_netcdf(output_path)
    print(f'Saved drought NC: {output_path}')

    metadata = {
        'model': str(op.MODEL_PATH.resolve()),
        'forecast_start': forecast_start.strftime('%Y-%m-%d'),
        'forecast_end': target_end.strftime('%Y-%m-%d'),
        'antecedent_obs_start': antecedent_start.strftime('%Y-%m-%d'),
        'antecedent_obs_end': antecedent_end.strftime('%Y-%m-%d'),
        'downloaded_antecedent_days': [day.strftime('%Y-%m-%d') for day in downloaded_days],
        'output': str(output_path.resolve()),
        'climate_source': str(op.CLIMATE_WEEKLY_PATH.resolve()),
        'note': 'Surface antecedent history is auto-downloaded via CDS if missing. Pressure fields are not needed for antecedent SPEI-4 accumulation.',
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Saved metadata: {metadata_path}')


if __name__ == '__main__':
    main()
