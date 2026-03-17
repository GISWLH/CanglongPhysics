from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path

import cartopy.crs as ccrs
import cmaps
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mplotutils as mpu
import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
RUNNER_PATH = SCRIPT_DIR / 'run_china_precip_v35.py'
TEMP_PATH = SCRIPT_DIR / 'china_temp.py'
TITLE_FONT_SIZE = 23
GLOBAL_BIAS_CORR_CACHE_PATH = SCRIPT_DIR / 'cache' / 't2m_bias_correction_global_land_v3_weekly2017_2021.json'
HINDCAST_EVAL_PATH = ROOT / 'Infer' / 'eval' / 'model_v3.nc'
LAND_COVER_PATH = ROOT / 'constant_masks' / 'land_cover.npy'
LAND_MASK_LAND_ANCHORS = [
    (30.0, 110.0),
    (50.0, 10.0),
    (0.0, 20.0),
    (40.0, 250.0),
    (-25.0, 135.0),
    (-15.0, 300.0),
]
LAND_MASK_OCEAN_ANCHORS = [
    (0.0, 180.0),
    (0.0, 330.0),
    (-30.0, 230.0),
    (-40.0, 20.0),
]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def month_token(month_label: str) -> str:
    return load_module(TEMP_PATH, 'china_temp').month_token(month_label)


def _sample_binary_mask(binary_mask: np.ndarray, lat: np.ndarray, lon: np.ndarray, point_lat: float, point_lon: float) -> bool:
    lat_idx = int(np.argmin(np.abs(lat - point_lat)))
    lon_idx = int(np.argmin(np.abs(lon - (point_lon % 360.0))))
    return bool(binary_mask[lat_idx, lon_idx])


def _score_land_mask_layout(binary_mask: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> int:
    score = 0
    for point_lat, point_lon in LAND_MASK_LAND_ANCHORS:
        score += int(_sample_binary_mask(binary_mask, lat, lon, point_lat, point_lon))
    for point_lat, point_lon in LAND_MASK_OCEAN_ANCHORS:
        score += int(not _sample_binary_mask(binary_mask, lat, lon, point_lat, point_lon))
    return score


def _load_aligned_land_mask(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, dict]:
    land = np.load(LAND_COVER_PATH)
    if land.shape != (lat.size, lon.size):
        raise ValueError(f'Land cover shape {land.shape} does not match target grid {(lat.size, lon.size)}')
    land_binary = land != 0
    native_score = _score_land_mask_layout(land_binary, lat, lon)
    rolled_binary = np.roll(land_binary, land.shape[1] // 2, axis=1)
    rolled_score = _score_land_mask_layout(rolled_binary, lat, lon)
    if rolled_score > native_score:
        return rolled_binary, {
            'alignment': 'rolled_180',
            'roll_cols': int(land.shape[1] // 2),
            'score_native': int(native_score),
            'score_aligned': int(rolled_score),
        }
    return land_binary, {
        'alignment': 'native',
        'roll_cols': 0,
        'score_native': int(native_score),
        'score_aligned': int(native_score),
    }


def build_global_land_mask(lat: np.ndarray, lon: np.ndarray) -> xr.DataArray:
    land_binary, _ = _load_aligned_land_mask(lat, lon)
    return xr.DataArray(land_binary, coords={'lat': lat.astype(np.float32), 'lon': lon.astype(np.float32)}, dims=('lat', 'lon'))


def build_or_load_global_bias_correction_params(op) -> dict:
    GLOBAL_BIAS_CORR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not HINDCAST_EVAL_PATH.exists():
        raise FileNotFoundError(f'Hindcast eval file not found: {HINDCAST_EVAL_PATH}')

    weekly_ds = xr.open_dataset(op.CLIMATE_WEEKLY_PATH)
    weekly_times = pd.DatetimeIndex(weekly_ds['time'].values)
    weekly_slot = np.clip(((weekly_times.dayofyear - 1) // 7).astype(int), 0, 51)
    weekly_clim = []
    for slot in range(52):
        weekly_clim.append(weekly_ds['t2m'].isel(time=np.where(weekly_slot == slot)[0]).mean('time').values.astype(np.float32))
    weekly_clim = np.stack(weekly_clim, axis=0)
    lat = weekly_ds['lat'].values.astype(np.float64)
    lon = weekly_ds['lon'].values.astype(np.float32)
    land_mask, land_mask_meta = _load_aligned_land_mask(lat.astype(np.float32), lon)
    lat_weights = np.cos(np.deg2rad(lat)).astype(np.float64)[:, None]
    spatial_weights = lat_weights * land_mask.astype(np.float64)
    weekly_ds.close()

    if GLOBAL_BIAS_CORR_CACHE_PATH.exists():
        cached = json.loads(GLOBAL_BIAS_CORR_CACHE_PATH.read_text(encoding='utf-8'))
        month_params = cached.get('month_band_params', {})
        cache_ok = (
            month_params
            and all('lead1_6' in month_params.get(str(month), {}) for month in range(1, 13))
            and cached.get('source_land_mask_alignment') == land_mask_meta['alignment']
            and int(cached.get('source_land_mask_roll_cols', -1)) == int(land_mask_meta['roll_cols'])
        )
        if cache_ok:
            return cached

    print(f'Building global land T2m bias-correction cache from {HINDCAST_EVAL_PATH} ...')

    eval_ds = xr.open_dataset(HINDCAST_EVAL_PATH)

    lead_bands = {
        'lead1_4': [1, 2, 3, 4],
        'lead5_6': [5, 6],
        'lead1_6': [1, 2, 3, 4, 5, 6],
    }
    params = {
        'description': 'Global-land T2m anomaly linear bias correction derived from hindcast model_v3 vs 2000-2023 weekly climatology',
        'source_eval': str(HINDCAST_EVAL_PATH.resolve()),
        'source_weekly_clim': str(op.CLIMATE_WEEKLY_PATH.resolve()),
        'source_land_mask': str(LAND_COVER_PATH.resolve()),
        'model': 'model_v3_5_continue_record_ft2_best.pth',
        'target_region': 'Global land',
        'formula': 'obs_anom = a + b * pred_anom',
        'month_band_params': {},
        'source_land_mask_alignment': land_mask_meta['alignment'],
        'source_land_mask_roll_cols': int(land_mask_meta['roll_cols']),
        'source_land_mask_score_native': int(land_mask_meta['score_native']),
        'source_land_mask_score_aligned': int(land_mask_meta['score_aligned']),
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
            valid = np.isfinite(pred) & np.isfinite(obs) & land_mask[None, :, :]
            wgt = np.broadcast_to(spatial_weights, valid.shape[1:])[None, :, :]
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
    GLOBAL_BIAS_CORR_CACHE_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding='utf-8')
    return params


def apply_global_bias_correction(raw_anom: xr.DataArray, available_months: list[str], selected_months: list[str], params: dict) -> tuple[xr.DataArray, list[str]]:
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

        coeff = params['month_band_params'][str(month_num)]['lead5_6']
        if month_offset <= 2:
            current_corrected = (coeff['a'] + coeff['b'] * current).astype(np.float32)
            corrected.append(current_corrected)
            notes.append(f'{month_label}: lead5_6_linear, a={coeff["a"]:.3f}, b={coeff["b"]:.3f}, n={coeff["n"]}')
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
        notes.append(f'{month_label}: lead5_6_shift_only, shift={mean_shift:.3f}, n={coeff["n"]}')

    corrected_da = xr.concat(corrected, dim=xr.IndexVariable('month', np.asarray(selected_months, dtype=object)))
    corrected_da.name = 't2m_anom_c_bias_corrected_global_land'
    corrected_da.attrs['units'] = 'degC'
    corrected_da.attrs['long_name'] = '2m temperature anomaly over global land (first month raw; Mar-Apr linear; JJA shift-only; SON-DJ all-leads linear)'
    return corrected_da, notes


def default_output_path(monthly_nc: Path, selected_months: list[str], vmin: float, vmax: float, transparent_outside: bool) -> Path:
    month_part = '_'.join(month_token(month_label) for month_label in selected_months)
    range_suffix = f'_range{int(abs(vmax))}' if float(abs(vmax)).is_integer() else f'_range{abs(vmax):g}'
    if transparent_outside:
        range_suffix += '_transparent'
    return monthly_nc.with_name(f'global_monthly_t2m_anomaly_{month_part}_operation_s2s{range_suffix}.png')


def plot_global_temperature(monthly_nc: Path, output_png: Path, month_labels: list[str] | None, vmin: float, vmax: float, transparent_outside: bool):
    op = load_module(RUNNER_PATH, 'run_china_precip_v35')
    temp = load_module(TEMP_PATH, 'china_temp')

    monthly_ds = xr.open_dataset(monthly_nc)
    available_months = [str(month) for month in monthly_ds['month'].values.tolist()]
    selected_months = temp.normalize_months(month_labels, available_months)
    clim_ds = temp.build_or_load_monthly_t2m_climatology(op, monthly_ds['lat'].values.astype(np.float32), monthly_ds['lon'].values.astype(np.float32))
    anom = temp.build_temperature_anomaly(monthly_ds, clim_ds, selected_months)
    monthly_ds.close()
    if hasattr(clim_ds, 'close'):
        clim_ds.close()

    land_mask = build_global_land_mask(anom['lat'].values.astype(np.float32), anom['lon'].values.astype(np.float32))

    op.prepare_font()
    projection = ccrs.PlateCarree(central_longitude=180)
    levels = np.linspace(vmin, vmax, 11)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = copy.copy(cmaps.BlueWhiteOrangeRed)
    if transparent_outside:
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    output_png = Path(output_png).resolve()
    cwd = Path.cwd()
    try:
        os.chdir(ROOT / 'code')
        fig = plt.figure(figsize=(24, 7.2))
        axes = []
        for idx in range(anom.sizes['month']):
            ax = fig.add_subplot(1, anom.sizes['month'], idx + 1, projection=projection)
            axes.append(ax)

        mappable = None
        for idx in range(anom.sizes['month']):
            ax = axes[idx]
            current = anom.isel(month=idx).where(land_mask)
            if transparent_outside:
                current = current.where((current >= vmin) & (current <= vmax))
            else:
                current = current.clip(min=vmin, max=vmax)
            current = mpu.cyclic_dataarray(current)

            mappable = op.china_plot.one_map_region(
                current,
                ax,
                cmap=cmap,
                levels=levels,
                norm=norm,
                mask_ocean=False,
                add_coastlines=True,
                add_land=False,
                add_river=False,
                add_lake=False,
                add_stock=False,
                add_gridlines=True,
                colorbar=False,
                plotfunc='pcolormesh',
                extents=[0, 360, -60, 75],
                interval=[60, 30],
            )
            month_label = str(anom['month'].values[idx])
            ax.text(
                0.02,
                0.98,
                f'CAS-Canglong {month_label}',
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=TITLE_FONT_SIZE,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.65, 'pad': 2.5},
                zorder=20,
            )

        cbar_ax = fig.add_axes([0.91, 0.17, 0.012, 0.64])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Temperature Anomaly (°C)', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        plt.subplots_adjust(left=0.03, right=0.89, top=0.92, bottom=0.11, wspace=0.08)

        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
    finally:
        os.chdir(cwd)

    print(output_png)
    return output_png


def parse_args():
    parser = argparse.ArgumentParser(description='Plot global 2m temperature anomaly from monthly CAS-Canglong operation_s2s output.')
    parser.add_argument('--input', type=Path, default=None, help='Monthly atmosphere NetCDF. Defaults to the latest operation_s2s monthly file.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--months', nargs='+', default=None, help='Optional subset of month labels, for example 2026-02 2026-03 2026-04.')
    parser.add_argument('--vmin', type=float, default=-5.0, help='Minimum plotted anomaly in degC.')
    parser.add_argument('--vmax', type=float, default=5.0, help='Maximum plotted anomaly in degC.')
    parser.set_defaults(transparent_outside=False)
    parser.add_argument('--transparent-outside', action='store_true', dest='transparent_outside', help='Mask anomalies outside [vmin, vmax] as transparent white instead of clipping.')
    parser.add_argument('--clip-outside', action='store_false', dest='transparent_outside', help='Clip anomalies outside [vmin, vmax] instead of masking them transparent.')
    return parser.parse_args()


def main():
    args = parse_args()
    temp = load_module(TEMP_PATH, 'china_temp')
    monthly_nc = args.input if args.input is not None else temp.find_latest_monthly_file(SCRIPT_DIR / 'output')
    with xr.open_dataset(monthly_nc) as ds:
        available_months = [str(month) for month in ds['month'].values.tolist()]
    selected_months = temp.normalize_months(args.months, available_months)
    output_png = args.output if args.output is not None else default_output_path(monthly_nc, selected_months, args.vmin, args.vmax, args.transparent_outside)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_global_temperature(monthly_nc, output_png, selected_months, args.vmin, args.vmax, args.transparent_outside)


if __name__ == '__main__':
    main()
