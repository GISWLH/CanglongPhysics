from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = Path(__file__).resolve().parent / 'run_china_precip_v35.py'
G = 9.80665
GLOBAL_ECMWF_MONTHLY_TP_CLIM_PATH = ROOT / 'analysis' / 'operation' / 'cache' / 'tp_monthly_climatology_2000_2023_from_ecmwf_tp_global.nc'
DEFAULT_END_MONTH = 12
SEASON_DEFS = {
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'DJF': [12, 1, 2],
}


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_china_precip_v35', RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_or_load_global_monthly_tp_climatology_ecmwf(op, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    GLOBAL_ECMWF_MONTHLY_TP_CLIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    if GLOBAL_ECMWF_MONTHLY_TP_CLIM_PATH.exists():
        return xr.open_dataset(GLOBAL_ECMWF_MONTHLY_TP_CLIM_PATH)

    source_path = op.find_existing_path(op.ECMWF_MONTHLY_TP_SOURCE_CANDIDATES)
    if source_path is None:
        raise FileNotFoundError('ECMWF monthly tp source not found in analysis/operation/cache')

    ds = xr.open_dataset(source_path)
    tp = ds['tp'].sel(valid_time=slice(f'{op.ECMWF_CLIM_YEAR_START}-01-01', f'{op.ECMWF_CLIM_YEAR_END}-12-31'))
    tp_mm_day = tp.astype(np.float32) * 1000.0
    days_in_month = xr.DataArray(
        tp['valid_time'].dt.days_in_month.astype(np.float32).values,
        dims=('valid_time',),
        coords={'valid_time': tp['valid_time']},
    )
    clim = (tp_mm_day * days_in_month).groupby('valid_time.month').mean('valid_time').load()
    clim = clim.interp(latitude=lat.astype(np.float32), longitude=lon.astype(np.float32))
    ds.close()

    out = xr.Dataset(
        data_vars={
            'tp_monthly_total_mm': (('month', 'lat', 'lon'), clim.values.astype(np.float32)),
            'year_count': (('month',), np.full(12, op.ECMWF_CLIM_YEAR_END - op.ECMWF_CLIM_YEAR_START + 1, dtype=np.int32)),
        },
        coords={
            'month': np.arange(1, 13, dtype=np.int32),
            'lat': lat.astype(np.float32),
            'lon': lon.astype(np.float32),
        },
        attrs={
            'description': 'Global monthly total precipitation climatology derived from ECMWF official monthly total precipitation source',
            'source': str(source_path),
            'years': f'{op.ECMWF_CLIM_YEAR_START}-{op.ECMWF_CLIM_YEAR_END}',
            'units': 'mm',
            'note': 'Computed directly on monthly data: tp [m] -> mm/day via *1000, then multiplied by actual days_in_month, then averaged by calendar month.',
        },
    )
    out.to_netcdf(GLOBAL_ECMWF_MONTHLY_TP_CLIM_PATH)
    return xr.open_dataset(GLOBAL_ECMWF_MONTHLY_TP_CLIM_PATH)


def load_observed_daily_global_fields(op, start_date: pd.Timestamp, end_date: pd.Timestamp):
    records = []
    current = start_date
    levels = [200, 500, 850]
    while current <= end_date:
        surf_path = op.INPUT_DIR / f'ERA5_surface_{current.strftime("%Y%m%d")}.nc'
        pres_path = op.INPUT_DIR / f'ERA5_pressure_{current.strftime("%Y%m%d")}.nc'
        surf = xr.open_dataset(surf_path)
        pres = xr.open_dataset(pres_path)
        rec = {
            'date': current,
            'tp_mm_day': ((surf['lsrr'].values.astype(np.float32) + surf['crr'].values.astype(np.float32)) * op.SECONDS_PER_DAY),
            't2m_c': (surf['t2m'].values.astype(np.float32) - 273.15),
        }
        for level in levels:
            rec[f'h{level}_gpm'] = (pres['z'].sel(pressure_level=level).values.astype(np.float32) / G)
            rec[f'u{level}_ms'] = pres['u'].sel(pressure_level=level).values.astype(np.float32)
            rec[f'v{level}_ms'] = pres['v'].sel(pressure_level=level).values.astype(np.float32)
        surf.close()
        pres.close()
        records.append(rec)
        current += pd.Timedelta(days=1)
    return records


def build_month_labels(year: int, months: list[int]) -> list[str]:
    return [f'{year}-{month:02d}' for month in months]


def complete_seasons(year: int, target_months: list[int]) -> dict[str, list[int]]:
    month_set = set(target_months)
    seasons = {}
    for season_name, months in SEASON_DEFS.items():
        if season_name == 'DJF':
            if {12}.issubset(month_set):
                continue
        if set(months).issubset(month_set):
            seasons[f'{year}-{season_name}'] = months
    return seasons


def area_days_by_month(year: int, months: list[int]) -> dict[int, int]:
    return {month: int((pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)).day) for month in months}


def parse_args():
    parser = argparse.ArgumentParser(description='Generate atmosphere NC products for operation workflow; plotting is handled by separate scripts.')
    parser.add_argument('--end-month', type=int, default=DEFAULT_END_MONTH, help='Forecast through this month of the forecast year.')
    parser.add_argument('--save-weekly', action='store_true', help='Also save weekly global forecast fields to NetCDF.')
    return parser.parse_args()


def main():
    args = parse_args()
    op = load_runner_module()

    surface_dates = op.find_input_dates('ERA5_surface')
    pressure_dates = op.find_input_dates('ERA5_pressure')
    input_surface_np, input_upper_np, lat, lon = op.build_two_week_inputs(surface_dates, pressure_dates)

    forecast_start = max(surface_dates) + pd.Timedelta(days=1)
    forecast_year = forecast_start.year
    if args.end_month < forecast_start.month:
        raise ValueError(f'end-month {args.end_month} is earlier than forecast start month {forecast_start.month}')
    target_months = list(range(forecast_start.month, args.end_month + 1))
    target_end = pd.Timestamp(year=forecast_year, month=max(target_months), day=1) + pd.offsets.MonthEnd(1)
    month_labels = build_month_labels(forecast_year, target_months)
    days_in_month = area_days_by_month(forecast_year, target_months)

    output_dir = op.OUTPUT_ROOT / f'{forecast_start.strftime("%Y%m%d")}_v35_ft2_best'
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly_path = output_dir / f'operation_s2s_atmos_monthly_{forecast_year}{target_months[0]:02d}_{forecast_year}{target_months[-1]:02d}.nc'
    seasonal_path = output_dir / f'operation_s2s_atmos_seasonal_{forecast_year}.nc'
    weekly_path = output_dir / f'operation_s2s_atmos_weekly_{forecast_start.strftime("%Y%m%d")}_{target_end.strftime("%Y%m%d")}.nc'
    metadata_path = output_dir / f'operation_s2s_metadata_{forecast_start.strftime("%Y%m%d")}.json'

    print(f'Forecast start: {forecast_start:%Y-%m-%d}')
    print(f'Forecast end  : {target_end:%Y-%m-%d}')
    print(f'Months        : {month_labels}')
    print(f'Output dir    : {output_dir}')

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
    z_idx = op.UPPER_VARS.index('z')
    u_idx = op.UPPER_VARS.index('u')
    v_idx = op.UPPER_VARS.index('v')
    level_to_index = {level: op.PRESSURE_LEVELS.index(level) for level in [200, 500, 850]}
    use_amp = device.type == 'cuda'

    monthly_tp = {month: np.zeros((lat.size, lon.size), dtype=np.float64) for month in target_months}
    monthly_state_sums = {
        name: {month: np.zeros((lat.size, lon.size), dtype=np.float64) for month in target_months}
        for name in ['t2m_c', 'h200_gpm', 'u200_ms', 'v200_ms', 'h500_gpm', 'u500_ms', 'v500_ms', 'h850_gpm', 'u850_ms', 'v850_ms']
    }
    monthly_coverage_days = {month: 0 for month in target_months}

    weekly_times = []
    weekly_payload = {name: [] for name in ['tp_mm_day', 't2m_c', 'h200_gpm', 'u200_ms', 'v200_ms', 'h500_gpm', 'u500_ms', 'v500_ms', 'h850_gpm', 'u850_ms', 'v850_ms']} if args.save_weekly else None

    first_month_start = pd.Timestamp(year=forecast_year, month=target_months[0], day=1)
    observed_end = forecast_start - pd.Timedelta(days=1)
    if observed_end >= first_month_start:
        print(f'Including observed daily fields for {first_month_start:%Y-%m-%d} to {observed_end:%Y-%m-%d}')
        observed_records = load_observed_daily_global_fields(op, first_month_start, observed_end)
        for rec in observed_records:
            month = rec['date'].month
            if month not in monthly_coverage_days:
                continue
            monthly_tp[month] += rec['tp_mm_day'].astype(np.float64)
            for name in monthly_state_sums:
                monthly_state_sums[name][month] += rec[name].astype(np.float64)
            monthly_coverage_days[month] += 1

    forecast_steps = 0
    cursor = forecast_start
    while cursor <= target_end:
        forecast_steps += 1
        cursor += pd.Timedelta(days=7)

    week_start = forecast_start
    with torch.inference_mode():
        for step in range(forecast_steps):
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                output_surface, output_upper = model(current_surface, current_upper)

            output_surface = output_surface.float()
            output_upper = output_upper.float()
            surf_phys = output_surface * surface_std + surface_mean
            upper_phys = output_upper * upper_std + upper_mean

            tp_mm_day = ((surf_phys[:, lssr_idx, 0] + surf_phys[:, crr_idx, 0]) * op.SECONDS_PER_DAY)[0].detach().cpu().numpy().astype(np.float32)
            t2m_c = (surf_phys[:, t2m_idx, 0][0] - 273.15).detach().cpu().numpy().astype(np.float32)
            weekly_record = {
                'tp_mm_day': tp_mm_day,
                't2m_c': t2m_c,
            }
            for level in [200, 500, 850]:
                level_index = level_to_index[level]
                weekly_record[f'h{level}_gpm'] = (upper_phys[0, z_idx, level_index, 0] / G).detach().cpu().numpy().astype(np.float32)
                weekly_record[f'u{level}_ms'] = upper_phys[0, u_idx, level_index, 0].detach().cpu().numpy().astype(np.float32)
                weekly_record[f'v{level}_ms'] = upper_phys[0, v_idx, level_index, 0].detach().cpu().numpy().astype(np.float32)

            if args.save_weekly:
                weekly_times.append(week_start)
                for name, value in weekly_record.items():
                    weekly_payload[name].append(value)

            week_end = week_start + pd.Timedelta(days=6)
            for month in target_months:
                month_start, month_end = op.month_range(forecast_year, month)
                days = op.overlap_days(week_start, week_end, month_start, month_end)
                if days <= 0:
                    continue
                monthly_tp[month] += weekly_record['tp_mm_day'].astype(np.float64) * days
                for name in monthly_state_sums:
                    monthly_state_sums[name][month] += weekly_record[name].astype(np.float64) * days
                monthly_coverage_days[month] += days

            current_surface = torch.cat([current_surface[:, :, 1:2], output_surface], dim=2)
            current_upper = torch.cat([current_upper[:, :, :, 1:2], output_upper], dim=3)
            week_start += pd.Timedelta(days=7)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            print(f'[{step + 1:02d}/{forecast_steps:02d}] {(week_start - pd.Timedelta(days=7)):%Y-%m-%d} done')

    monthly_tp_stack = np.stack([monthly_tp[month].astype(np.float32) for month in target_months], axis=0)
    monthly_state_stack = {}
    for name, data_by_month in monthly_state_sums.items():
        stacked = []
        for month in target_months:
            days = max(monthly_coverage_days[month], 1)
            stacked.append((data_by_month[month] / float(days)).astype(np.float32))
        monthly_state_stack[name] = np.stack(stacked, axis=0)

    old_tp_clim_ds = op.build_or_load_monthly_tp_climatology()
    old_tp_clim = old_tp_clim_ds['tp_monthly_total_mm'].sel(month=target_months).values.astype(np.float32)
    old_tp_clim_ds.close()
    ecmwf_tp_clim_ds = build_or_load_global_monthly_tp_climatology_ecmwf(op, lat, lon)
    ecmwf_tp_clim = ecmwf_tp_clim_ds['tp_monthly_total_mm'].sel(month=target_months).values.astype(np.float32)
    ecmwf_tp_clim_ds.close()
    monthly_tp_clim = old_tp_clim.copy()
    if len(target_months) > 1:
        monthly_tp_clim[1:] = ecmwf_tp_clim[1:]
    tp_anom_percent = ((monthly_tp_stack - monthly_tp_clim) / (monthly_tp_clim + op.EPS) * 100.0).astype(np.float32)
    tp_clim_sources = {month_labels[0]: str(op.MONTHLY_TP_CLIM_PATH.resolve())}
    for month_label in month_labels[1:]:
        tp_clim_sources[month_label] = str(GLOBAL_ECMWF_MONTHLY_TP_CLIM_PATH.resolve())

    monthly_ds = xr.Dataset(
        data_vars={
            'tp_total_mm': (('month', 'lat', 'lon'), monthly_tp_stack),
            'tp_clim_total_mm': (('month', 'lat', 'lon'), monthly_tp_clim.astype(np.float32)),
            'tp_anom_percent': (('month', 'lat', 'lon'), tp_anom_percent),
            't2m_c': (('month', 'lat', 'lon'), monthly_state_stack['t2m_c']),
            'h200_gpm': (('month', 'lat', 'lon'), monthly_state_stack['h200_gpm']),
            'u200_ms': (('month', 'lat', 'lon'), monthly_state_stack['u200_ms']),
            'v200_ms': (('month', 'lat', 'lon'), monthly_state_stack['v200_ms']),
            'h500_gpm': (('month', 'lat', 'lon'), monthly_state_stack['h500_gpm']),
            'u500_ms': (('month', 'lat', 'lon'), monthly_state_stack['u500_ms']),
            'v500_ms': (('month', 'lat', 'lon'), monthly_state_stack['v500_ms']),
            'h850_gpm': (('month', 'lat', 'lon'), monthly_state_stack['h850_gpm']),
            'u850_ms': (('month', 'lat', 'lon'), monthly_state_stack['u850_ms']),
            'v850_ms': (('month', 'lat', 'lon'), monthly_state_stack['v850_ms']),
            'coverage_days': (('month',), np.array([monthly_coverage_days[month] for month in target_months], dtype=np.int32)),
        },
        coords={'month': month_labels, 'lat': lat.astype(np.float32), 'lon': lon.astype(np.float32)},
        attrs={
            'model': op.MODEL_PATH.name,
            'forecast_start': forecast_start.strftime('%Y-%m-%d'),
            'forecast_end': target_end.strftime('%Y-%m-%d'),
            'description': 'CAS-Canglong atmosphere monthly operation products; plotting is handled by separate scripts.',
            'tp_climatology_mode': 'hybrid_first_old_rest_ecmwf',
            'tp_climatology_paths': json.dumps(tp_clim_sources, ensure_ascii=False),
            'note': 'tp_total_mm is monthly total precipitation; t2m/h/u/v are monthly day-weighted means. Forecast start month includes observed daily fields before forecast start.',
        },
    )
    monthly_ds.to_netcdf(monthly_path)
    print(f'Saved monthly NC: {monthly_path}')

    seasons = complete_seasons(forecast_year, target_months)
    seasonal_ds = None
    if seasons:
        month_to_index = {month: idx for idx, month in enumerate(target_months)}
        seasonal_tp = []
        seasonal_tp_clim = []
        seasonal_tp_anom = []
        seasonal_t2m = []
        seasonal_h200 = []
        seasonal_u200 = []
        seasonal_v200 = []
        seasonal_h500 = []
        seasonal_u500 = []
        seasonal_v500 = []
        seasonal_h850 = []
        seasonal_u850 = []
        seasonal_v850 = []
        seasonal_labels = []

        for season_label, months in seasons.items():
            indices = [month_to_index[month] for month in months]
            weights = np.array([days_in_month[month] for month in months], dtype=np.float32)
            weight_sum = float(weights.sum())
            seasonal_labels.append(season_label)
            seasonal_tp_value = monthly_tp_stack[indices].sum(axis=0)
            seasonal_tp_clim_value = monthly_tp_clim[indices].sum(axis=0)
            seasonal_tp.append(seasonal_tp_value.astype(np.float32))
            seasonal_tp_clim.append(seasonal_tp_clim_value.astype(np.float32))
            seasonal_tp_anom.append(((seasonal_tp_value - seasonal_tp_clim_value) / (seasonal_tp_clim_value + op.EPS) * 100.0).astype(np.float32))
            weighted = lambda arr: np.tensordot(weights, arr[indices], axes=(0, 0)) / weight_sum
            seasonal_t2m.append(weighted(monthly_state_stack['t2m_c']).astype(np.float32))
            seasonal_h200.append(weighted(monthly_state_stack['h200_gpm']).astype(np.float32))
            seasonal_u200.append(weighted(monthly_state_stack['u200_ms']).astype(np.float32))
            seasonal_v200.append(weighted(monthly_state_stack['v200_ms']).astype(np.float32))
            seasonal_h500.append(weighted(monthly_state_stack['h500_gpm']).astype(np.float32))
            seasonal_u500.append(weighted(monthly_state_stack['u500_ms']).astype(np.float32))
            seasonal_v500.append(weighted(monthly_state_stack['v500_ms']).astype(np.float32))
            seasonal_h850.append(weighted(monthly_state_stack['h850_gpm']).astype(np.float32))
            seasonal_u850.append(weighted(monthly_state_stack['u850_ms']).astype(np.float32))
            seasonal_v850.append(weighted(monthly_state_stack['v850_ms']).astype(np.float32))

        seasonal_ds = xr.Dataset(
            data_vars={
                'tp_total_mm': (('season', 'lat', 'lon'), np.stack(seasonal_tp, axis=0)),
                'tp_clim_total_mm': (('season', 'lat', 'lon'), np.stack(seasonal_tp_clim, axis=0)),
                'tp_anom_percent': (('season', 'lat', 'lon'), np.stack(seasonal_tp_anom, axis=0)),
                't2m_c': (('season', 'lat', 'lon'), np.stack(seasonal_t2m, axis=0)),
                'h200_gpm': (('season', 'lat', 'lon'), np.stack(seasonal_h200, axis=0)),
                'u200_ms': (('season', 'lat', 'lon'), np.stack(seasonal_u200, axis=0)),
                'v200_ms': (('season', 'lat', 'lon'), np.stack(seasonal_v200, axis=0)),
                'h500_gpm': (('season', 'lat', 'lon'), np.stack(seasonal_h500, axis=0)),
                'u500_ms': (('season', 'lat', 'lon'), np.stack(seasonal_u500, axis=0)),
                'v500_ms': (('season', 'lat', 'lon'), np.stack(seasonal_v500, axis=0)),
                'h850_gpm': (('season', 'lat', 'lon'), np.stack(seasonal_h850, axis=0)),
                'u850_ms': (('season', 'lat', 'lon'), np.stack(seasonal_u850, axis=0)),
                'v850_ms': (('season', 'lat', 'lon'), np.stack(seasonal_v850, axis=0)),
            },
            coords={'season': seasonal_labels, 'lat': lat.astype(np.float32), 'lon': lon.astype(np.float32)},
            attrs={
                'model': op.MODEL_PATH.name,
                'forecast_start': forecast_start.strftime('%Y-%m-%d'),
                'forecast_end': target_end.strftime('%Y-%m-%d'),
                'description': 'CAS-Canglong atmosphere seasonal operation products derived from operation_all monthly outputs.',
                'note': 'tp_total_mm is seasonal total precipitation; t2m/h/u/v are day-weighted seasonal means.',
            },
        )
        seasonal_ds.to_netcdf(seasonal_path)
        print(f'Saved seasonal NC: {seasonal_path}')

    if args.save_weekly:
        weekly_ds = xr.Dataset(
            data_vars={name: (('time', 'lat', 'lon'), np.stack(values, axis=0).astype(np.float32)) for name, values in weekly_payload.items()},
            coords={'time': pd.DatetimeIndex(weekly_times), 'lat': lat.astype(np.float32), 'lon': lon.astype(np.float32)},
            attrs={
                'model': op.MODEL_PATH.name,
                'forecast_start': forecast_start.strftime('%Y-%m-%d'),
                'forecast_end': target_end.strftime('%Y-%m-%d'),
                'description': 'CAS-Canglong atmosphere weekly operation products; forecast weeks only.',
                'note': 'Weekly fields are direct weekly means/totals from the rolling S2S forecast and do not include observed days before forecast start.',
            },
        )
        weekly_ds.to_netcdf(weekly_path)
        print(f'Saved weekly NC: {weekly_path}')

    metadata = {
        'model': str(op.MODEL_PATH.resolve()),
        'forecast_start': forecast_start.strftime('%Y-%m-%d'),
        'forecast_end': target_end.strftime('%Y-%m-%d'),
        'target_months': month_labels,
        'monthly_output': str(monthly_path.resolve()),
        'seasonal_output': str(seasonal_path.resolve()) if seasonal_ds is not None else None,
        'weekly_output': str(weekly_path.resolve()) if args.save_weekly else None,
        'tp_climatology_mode': 'hybrid_first_old_rest_ecmwf',
        'tp_climatology_paths': tp_clim_sources,
        'note': 'SST is intentionally not produced here; use analysis/operation/SSTmodel for ocean-model products.',
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f'Saved metadata: {metadata_path}')


if __name__ == '__main__':
    main()
