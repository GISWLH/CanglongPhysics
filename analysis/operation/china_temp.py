from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path

import cartopy.crs as ccrs
import cmaps
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mplotutils as mpu
import numpy as np
import pandas as pd
import xarray as xr
from shapely import contains_xy
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
RUNNER_PATH = SCRIPT_DIR / 'run_china_precip_v35.py'
MONTHLY_T2M_CLIM_PATH = SCRIPT_DIR / 'cache' / 't2m_monthly_climatology_2000_2023.nc'
BIAS_CORR_CACHE_PATH = SCRIPT_DIR / 'cache' / 't2m_bias_correction_china_v3_weekly2000_2023.json'
HINDCAST_EVAL_PATH = ROOT / 'Infer' / 'eval' / 'model_v3.nc'
SUBMAP_X_OFFSET = 0.775 + 0.01
SUBMAP_Y_OFFSET = 0.022
SUBMAP_WIDTH = 0.216
SUBMAP_HEIGHT = 0.264
TITLE_FONT_SIZE = 23


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_china_precip_v35', RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_latest_monthly_file(output_root: Path) -> Path:
    patterns = ['*/operation_s2s_atmos_monthly_*.nc', '*/operation_all_atmos_monthly_*.nc']
    candidates = []
    for pattern in patterns:
        candidates.extend(output_root.glob(pattern))
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f'No monthly atmosphere NetCDF found under {output_root}')
    return candidates[0]


def normalize_months(month_labels: list[str] | None, available_months: list[str]) -> list[str]:
    if month_labels is None:
        return available_months[:3]
    return [str(month) for month in month_labels]


def month_token(month_label: str) -> str:
    ts = pd.Timestamp(f'{month_label}-01')
    return ts.strftime('%b').lower()


def default_output_path(monthly_nc: Path, selected_months: list[str], vmin: float, vmax: float, transparent_outside: bool, bias_corrected: bool) -> Path:
    month_part = '_'.join(month_token(month_label) for month_label in selected_months)
    suffix = '_bias_corrected' if bias_corrected else ''
    range_suffix = f'_range{int(abs(vmax))}' if float(abs(vmax)).is_integer() else f'_range{abs(vmax):g}'
    if transparent_outside:
        range_suffix += '_transparent'
    return monthly_nc.with_name(f'china_monthly_t2m_anomaly_{month_part}_operation_s2s{suffix}{range_suffix}.png')


def build_or_load_monthly_t2m_climatology(op, lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    MONTHLY_T2M_CLIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MONTHLY_T2M_CLIM_PATH.exists():
        cached = xr.open_dataset(MONTHLY_T2M_CLIM_PATH)
        cached_lat = cached['lat'].values.astype(np.float32)
        cached_lon = cached['lon'].values.astype(np.float32)
        if cached_lat.shape == lat.shape and cached_lon.shape == lon.shape and np.allclose(cached_lat, lat) and np.allclose(cached_lon, lon):
            return cached
        interp = cached.interp(lat=lat.astype(np.float32), lon=lon.astype(np.float32))
        cached.close()
        return interp

    ds = xr.open_dataset(op.CLIMATE_WEEKLY_PATH)
    t2m = ds['t2m']
    times = pd.DatetimeIndex(t2m['time'].values)
    climate_lat = ds['lat'].values.astype(np.float32)
    climate_lon = ds['lon'].values.astype(np.float32)
    weighted_sum = np.zeros((12, climate_lat.size, climate_lon.size), dtype=np.float64)
    total_days = np.zeros(12, dtype=np.float64)

    print(f'Building monthly T2m climatology from {op.CLIMATE_WEEKLY_PATH} ...')
    for idx, week_start in enumerate(times):
        arr = t2m.isel(time=idx).values.astype(np.float32)
        week_end = week_start + pd.Timedelta(days=6)
        if week_start.month == week_end.month and week_start.year == week_end.year:
            weighted_sum[week_start.month - 1] += arr * 7.0
            total_days[week_start.month - 1] += 7.0
            continue

        first_start, first_end = op.month_range(week_start.year, week_start.month)
        days_first = op.overlap_days(week_start, week_end, first_start, first_end)
        if days_first > 0:
            weighted_sum[week_start.month - 1] += arr * days_first
            total_days[week_start.month - 1] += days_first

        second_start, second_end = op.month_range(week_end.year, week_end.month)
        days_second = op.overlap_days(week_start, week_end, second_start, second_end)
        if days_second > 0:
            weighted_sum[week_end.month - 1] += arr * days_second
            total_days[week_end.month - 1] += days_second

    climatology = weighted_sum / np.maximum(total_days[:, None, None], 1.0)
    ds.close()

    out = xr.Dataset(
        data_vars={
            't2m_monthly_mean_c': (('month', 'lat', 'lon'), climatology.astype(np.float32)),
            'sample_days': (('month',), np.rint(total_days).astype(np.int32)),
        },
        coords={
            'month': np.arange(1, 13, dtype=np.int32),
            'lat': climate_lat,
            'lon': climate_lon,
        },
        attrs={
            'description': 'Global monthly 2m temperature climatology derived from weekly climate archive',
            'source': str(op.CLIMATE_WEEKLY_PATH),
            'years': '2000-2023',
            'units': 'degC',
            'note': 'Computed on monthly windows from weekly t2m means using overlap-day weighting, then averaged by calendar month.',
        },
    )
    out.to_netcdf(MONTHLY_T2M_CLIM_PATH)
    return xr.open_dataset(MONTHLY_T2M_CLIM_PATH)


def build_temperature_anomaly(monthly_ds: xr.Dataset, climatology_ds: xr.Dataset, selected_months: list[str]) -> xr.DataArray:
    t2m = monthly_ds['t2m_c'].sel(month=selected_months).load()
    clim_fields = []
    for month_label in selected_months:
        month_num = int(str(month_label).split('-')[1])
        clim_fields.append(climatology_ds['t2m_monthly_mean_c'].sel(month=month_num).load())
    clim = xr.concat(clim_fields, dim=xr.IndexVariable('month', np.asarray(selected_months, dtype=object)))
    clim = clim.transpose('month', 'lat', 'lon')
    anom = (t2m - clim).astype(np.float32)
    anom.name = 't2m_anom_c'
    anom.attrs['units'] = 'degC'
    anom.attrs['long_name'] = '2m temperature anomaly'
    return anom


def build_or_load_bias_correction_params(op) -> dict:
    BIAS_CORR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if BIAS_CORR_CACHE_PATH.exists():
        cached = json.loads(BIAS_CORR_CACHE_PATH.read_text(encoding='utf-8'))
        month_params = cached.get('month_band_params', {})
        if month_params and all('lead1_6' in month_params.get(str(month), {}) for month in range(1, 13)):
            return cached

    if not HINDCAST_EVAL_PATH.exists():
        raise FileNotFoundError(f'Hindcast eval file not found: {HINDCAST_EVAL_PATH}')

    print(f'Building T2m bias-correction cache from {HINDCAST_EVAL_PATH} ...')

    weekly_ds = xr.open_dataset(op.CLIMATE_WEEKLY_PATH)
    if weekly_ds['lat'][0] > weekly_ds['lat'][-1]:
        weekly_ds = weekly_ds.sel(lat=slice(55, 15), lon=slice(70, 140))
    else:
        weekly_ds = weekly_ds.sel(lat=slice(15, 55), lon=slice(70, 140))
    weekly_times = pd.DatetimeIndex(weekly_ds['time'].values)
    weekly_slot = np.clip(((weekly_times.dayofyear - 1) // 7).astype(int), 0, 51)
    weekly_clim = []
    for slot in range(52):
        weekly_clim.append(weekly_ds['t2m'].isel(time=np.where(weekly_slot == slot)[0]).mean('time').values.astype(np.float32))
    weekly_clim = np.stack(weekly_clim, axis=0)
    lat = weekly_ds['lat'].values.astype(np.float64)
    lat_weights = np.cos(np.deg2rad(lat)).astype(np.float64)[:, None]
    weekly_ds.close()

    eval_ds = xr.open_dataset(HINDCAST_EVAL_PATH)
    if eval_ds['lat'][0] > eval_ds['lat'][-1]:
        eval_ds = eval_ds.sel(lat=slice(55, 15), lon=slice(70, 140))
    else:
        eval_ds = eval_ds.sel(lat=slice(15, 55), lon=slice(70, 140))

    lead_bands = {
        'lead1_4': [1, 2, 3, 4],
        'lead5_6': [5, 6],
        'lead1_6': [1, 2, 3, 4, 5, 6],
    }
    params = {
        'description': 'China T2m anomaly linear bias correction derived from hindcast model_v3 vs 2000-2023 weekly climatology',
        'source_eval': str(HINDCAST_EVAL_PATH.resolve()),
        'source_weekly_clim': str(op.CLIMATE_WEEKLY_PATH.resolve()),
        'model': 'model_v3_5_continue_record_ft2_best.pth',
        'target_region': 'China 70-140E, 15-55N',
        'formula': 'obs_anom = a + b * pred_anom',
        'month_band_params': {},
    }

    def regress(month: int, leads: list[int]) -> dict:
        mask_t = eval_ds['time'].dt.month == month
        week_idx = eval_ds['woy'].where(mask_t, drop=True).values.astype(int) - 1
        obs = eval_ds['obs_t2m'].where(mask_t, drop=True).values.astype(np.float32) - (weekly_clim[week_idx] + 273.15)
        sum_w = 0.0
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0
        n = 0
        for lead in leads:
            pred = eval_ds[f'pred_t2m_lead{lead}'].where(mask_t, drop=True).values.astype(np.float32) - (weekly_clim[week_idx] + 273.15)
            valid = np.isfinite(pred) & np.isfinite(obs)
            wgt = np.broadcast_to(lat_weights, valid.shape[1:])[None, :, :]
            wgt = np.where(valid, wgt, 0.0)
            x = np.where(valid, pred, 0.0)
            y = np.where(valid, obs, 0.0)
            sum_w += float(wgt.sum())
            sum_x += float((wgt * x).sum())
            sum_y += float((wgt * y).sum())
            sum_xx += float((wgt * x * x).sum())
            sum_xy += float((wgt * x * y).sum())
            n += int(valid.sum())
        mean_x = sum_x / max(sum_w, 1.0)
        mean_y = sum_y / max(sum_w, 1.0)
        var_x = sum_xx / max(sum_w, 1.0) - mean_x * mean_x
        cov_xy = sum_xy / max(sum_w, 1.0) - mean_x * mean_y
        b = cov_xy / max(var_x, 1.0e-8)
        a = mean_y - b * mean_x
        return {
            'a': float(a),
            'b': float(b),
            'mean_pred': float(mean_x),
            'mean_obs': float(mean_y),
            'n': int(n),
        }

    for month in range(1, 13):
        params['month_band_params'][str(month)] = {}
        for band_name, leads in lead_bands.items():
            params['month_band_params'][str(month)][band_name] = regress(month, leads)

    eval_ds.close()
    BIAS_CORR_CACHE_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding='utf-8')
    return params


def apply_bias_correction(raw_anom: xr.DataArray, available_months: list[str], selected_months: list[str], params: dict) -> tuple[xr.DataArray, list[str]]:
    corrected = []
    notes = []
    for month_label in selected_months:
        month_num = int(str(month_label).split('-')[1])
        month_offset = available_months.index(month_label)
        current = raw_anom.sel(month=month_label)
        if month_offset == 0:
            current_corrected = current.astype(np.float32)
            corrected.append(current_corrected)
            notes.append(f'{month_label}: first_month_raw_kept')
            continue

        band = 'lead5_6'
        coeff = params['month_band_params'][str(month_num)][band]
        if month_offset <= 2:
            current_corrected = (coeff['a'] + coeff['b'] * current).astype(np.float32)
            corrected.append(current_corrected)
            notes.append(f'{month_label}: {band}_linear, a={coeff["a"]:.3f}, b={coeff["b"]:.3f}, n={coeff["n"]}')
            continue

        if month_num >= 9:
            coeff_all = params['month_band_params'][str(month_num)]['lead1_6']
            current_corrected = (coeff_all['a'] + coeff_all['b'] * current).astype(np.float32)
            corrected.append(current_corrected)
            notes.append(f'{month_label}: lead1_6_linear, a={coeff_all["a"]:.3f}, b={coeff_all["b"]:.3f}, n={coeff_all["n"]}')
            continue

        mean_shift = coeff['mean_obs'] - coeff['mean_pred']
        current_corrected = (current + mean_shift).astype(np.float32)
        corrected.append(current_corrected)
        notes.append(f'{month_label}: {band}_shift_only, shift={mean_shift:.3f}, n={coeff["n"]}')

    corrected_da = xr.concat(corrected, dim=xr.IndexVariable('month', np.asarray(selected_months, dtype=object)))
    corrected_da.name = 't2m_anom_c_bias_corrected'
    corrected_da.attrs['units'] = 'degC'
    corrected_da.attrs['long_name'] = '2m temperature anomaly (first month raw; Mar-Apr linear; JJA shift-only; SON-DJ all-leads linear)'
    return corrected_da, notes


def plot_china_temperature(monthly_nc: Path, output_png: Path, vmin: float, vmax: float, transparent_outside: bool, months: list[str] | None = None, bias_corrected: bool = False):
    op = load_runner_module()
    ds = xr.open_dataset(monthly_nc)
    available_months = [str(month) for month in ds['month'].values.tolist()]
    selected_months = normalize_months(months, available_months)
    clim_ds = build_or_load_monthly_t2m_climatology(op, ds['lat'].values.astype(np.float32), ds['lon'].values.astype(np.float32))
    raw_anom = build_temperature_anomaly(ds, clim_ds, selected_months)
    ds.close()
    if hasattr(clim_ds, 'close'):
        clim_ds.close()

    correction_notes = []
    if bias_corrected:
        params = build_or_load_bias_correction_params(op)
        anom, correction_notes = apply_bias_correction(raw_anom, available_months, selected_months, params)
    else:
        anom = raw_anom

    op.prepare_font()
    china_shp = gpd.read_file(op.CHINA_SHP)
    china_geom = unary_union(china_shp.geometry)
    projection = ccrs.LambertConformal(
        central_longitude=105,
        central_latitude=40,
        standard_parallels=(25.0, 47.0),
    )
    levels = np.linspace(vmin, vmax, 11)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = copy.copy(cmaps.BlueWhiteOrangeRed)
    if transparent_outside:
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    output_png = Path(output_png).resolve()
    cwd = Path.cwd()
    try:
        os.chdir(ROOT / 'code')
        fig = plt.figure(figsize=(24, 8.6))
        axes = []
        for idx in range(anom.sizes['month']):
            ax = fig.add_subplot(1, anom.sizes['month'], idx + 1, projection=projection)
            axes.append(ax)

        masked_list = []
        mappable = None
        for idx in range(anom.sizes['month']):
            ax = axes[idx]
            current = anom.isel(month=idx)
            lon2d, lat2d = np.meshgrid(current['lon'].values, current['lat'].values)
            china_mask = contains_xy(china_geom, lon2d, lat2d)
            masked = current.where(xr.DataArray(china_mask, coords=current.coords, dims=current.dims))
            if transparent_outside:
                masked = masked.where((masked >= vmin) & (masked <= vmax))
            else:
                masked = masked.clip(min=vmin, max=vmax)
            masked_list.append(masked)

            mappable = op.china_plot.one_map_china(
                masked,
                ax,
                cmap=cmap,
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
            month_label = str(anom['month'].values[idx])
            panel_title = f'CAS-Canglong {month_label}'
            ax.text(
                0.02,
                0.98,
                panel_title,
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=TITLE_FONT_SIZE,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.65, 'pad': 2.5},
                zorder=20,
            )

        cbar_ax = fig.add_axes([0.90, 0.15, 0.012, 0.70])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Temperature Anomaly (°C)', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        plt.subplots_adjust(left=0.025, right=0.88, top=0.92, bottom=0.08, wspace=0.20)
        mpu.set_map_layout(axes, width=80)

        for ax, masked in zip(axes, masked_list):
            pos = ax.get_position()
            ax2 = fig.add_axes(
                [
                    pos.x0 + pos.width * SUBMAP_X_OFFSET,
                    pos.y0 + pos.height * SUBMAP_Y_OFFSET,
                    pos.width * SUBMAP_WIDTH,
                    pos.height * SUBMAP_HEIGHT,
                ],
                projection=projection,
            )
            op.china_plot.sub_china_map(masked, ax2, cmap=cmap, levels=levels, norm=norm, add_coastlines=False, add_land=False)

        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
    finally:
        os.chdir(cwd)

    print(output_png)
    print(f'T2m climatology: {MONTHLY_T2M_CLIM_PATH.resolve()}')
    if bias_corrected:
        print(f'Bias-correction cache: {BIAS_CORR_CACHE_PATH.resolve()}')
        for note in correction_notes:
            print(note)
    return output_png


def parse_args():
    parser = argparse.ArgumentParser(description='Plot China 2m temperature anomaly from monthly CAS-Canglong output.')
    parser.add_argument('--input', type=Path, default=None, help='Monthly atmosphere NetCDF. Defaults to the latest file in analysis/operation/output.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--vmin', type=float, default=-5.0, help='Minimum plotted anomaly in degC.')
    parser.add_argument('--vmax', type=float, default=5.0, help='Maximum plotted anomaly in degC.')
    parser.add_argument('--months', nargs='+', default=None, help='Optional subset of month labels to plot, for example 2026-02 2026-03 2026-04.')
    parser.add_argument('--bias-corrected', action='store_true', help='Apply monthly anomaly linear bias correction derived from hindcast over China.')
    parser.set_defaults(transparent_outside=False)
    parser.add_argument('--transparent-outside', action='store_true', dest='transparent_outside', help='Mask anomalies outside [vmin, vmax] as transparent white instead of clipping.')
    parser.add_argument('--clip-outside', action='store_false', dest='transparent_outside', help='Clip anomalies outside [vmin, vmax] instead of masking them transparent.')
    return parser.parse_args()


def main():
    args = parse_args()
    monthly_nc = args.input if args.input is not None else find_latest_monthly_file(SCRIPT_DIR / 'output')

    ds = xr.open_dataset(monthly_nc)
    available_months = [str(month) for month in ds['month'].values.tolist()]
    selected_months = normalize_months(args.months, available_months)
    ds.close()

    output_png = args.output if args.output is not None else default_output_path(monthly_nc, selected_months, args.vmin, args.vmax, args.transparent_outside, args.bias_corrected)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_china_temperature(monthly_nc, output_png, args.vmin, args.vmax, args.transparent_outside, selected_months, args.bias_corrected)


if __name__ == '__main__':
    main()
