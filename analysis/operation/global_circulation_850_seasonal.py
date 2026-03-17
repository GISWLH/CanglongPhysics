from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import cartopy.crs as ccrs
import cmaps
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
RUNNER_PATH = SCRIPT_DIR / 'run_china_precip_v35.py'
CLIM_HELPER_PATH = ROOT / 'Infer' / 'compute_climatology.py'
MONTHLY_CLIM_PATH = SCRIPT_DIR / 'cache' / 'global_850hpa_monthly_climatology_2002_2016.nc'
Z500_SCALE_CACHE_PATH = SCRIPT_DIR / 'cache' / 'z500_global_height_scale_v3_weekly2017_2021.json'
HINDCAST_EVAL_PATH = ROOT / 'Infer' / 'eval' / 'model_v3.nc'
WEEKLY_CIRC_CLIM_PATH = ROOT / 'Infer' / 'eval' / 'climatology_2002_2016.nc'

G = 9.80665
INIT_LINE = 'from 1st Feb2026 (IAP-CIESM; 8-member mean)'
PANEL_TITLES = [
    ('MAM', '850 hPa HGT, U, V (Mar2026-May2026) forecast'),
    ('JJA', '850 hPa HGT, U, V (Jun2026-Aug2026) forecast'),
    ('SON', '850 hPa HGT, U, V (Sept2026-Nov2026) forecast'),
    ('DJ', '850 hPa HGT, U, V (Dec2026-Jan2027) forecast'),
]
PLOT_EXTENT = [0, 360, -60, 75]
PLOT_COARSEN_FACTOR = 8
VECTOR_SKIP_LAT = 5
VECTOR_SKIP_LON = 8
VECTOR_MIN_SPEED = 0.5
QUIVER_REF = 2.0
VECTOR_COLOR = '#1b8e3e'
HGT_SMOOTH_SIGMA = 1.0
WIND_SMOOTH_SIGMA = 0.9


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_latest_file(pattern: str) -> Path:
    candidates = sorted((SCRIPT_DIR / 'output').glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f'No file matched {pattern}')
    return candidates[0]


def add_cyclic_360(data: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if float(lon[-1]) >= 359.9:
        data_cyclic = np.concatenate([data, data[:, :1]], axis=1)
        lon_cyclic = np.concatenate([lon, np.array([360.0], dtype=np.float32)])
        return data_cyclic, lon_cyclic
    return data, lon


def weighted_mean_months(monthly_da: xr.DataArray, month_labels: list[str]) -> xr.DataArray:
    selected = monthly_da.sel(month=month_labels).load()
    weights = np.array([pd.Timestamp(f'{month_label}-01').days_in_month for month_label in month_labels], dtype=np.float32)
    values = np.tensordot(weights, selected.values.astype(np.float32), axes=(0, 0)) / float(weights.sum())
    return xr.DataArray(values.astype(np.float32), coords={'lat': selected['lat'], 'lon': selected['lon']}, dims=('lat', 'lon'))


def remove_global_mean_hgt(monthly_hgt_anom: xr.DataArray) -> xr.DataArray:
    lat_weights = np.cos(np.deg2rad(monthly_hgt_anom['lat'].values.astype(np.float64)))[:, None]
    corrected = []
    for month_label in monthly_hgt_anom['month'].values.tolist():
        current_da = monthly_hgt_anom.sel(month=month_label)
        current = current_da.values.astype(np.float32)
        valid = np.isfinite(current)
        weights_2d = np.broadcast_to(lat_weights, current.shape)
        mean_bias = float((current[valid] * weights_2d[valid]).sum() / np.maximum(weights_2d[valid].sum(), 1.0))
        corrected.append((current_da - mean_bias).astype(np.float32))
    corrected_da = xr.concat(corrected, dim=xr.IndexVariable('month', monthly_hgt_anom['month'].values))
    return corrected_da.transpose('month', 'lat', 'lon')


def coarsen_for_plot(da: xr.DataArray) -> xr.DataArray:
    return da.coarsen(lat=PLOT_COARSEN_FACTOR, lon=PLOT_COARSEN_FACTOR, boundary='trim').mean().astype(np.float32)


def smooth_for_plot(da: xr.DataArray, sigma: float) -> xr.DataArray:
    arr = gaussian_filter(da.values.astype(np.float32), sigma=(sigma, sigma), mode=('nearest', 'wrap'))
    return xr.DataArray(arr.astype(np.float32), coords=da.coords, dims=da.dims)


def coarsen_panel(panel: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    coarsened = {name: coarsen_for_plot(da) for name, da in panel.items()}
    return {
        'hgt': smooth_for_plot(coarsened['hgt'], HGT_SMOOTH_SIGMA),
        'u': smooth_for_plot(coarsened['u'], WIND_SMOOTH_SIGMA),
        'v': smooth_for_plot(coarsened['v'], WIND_SMOOTH_SIGMA),
    }


def build_or_load_z500_scale_params() -> dict:
    Z500_SCALE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if Z500_SCALE_CACHE_PATH.exists():
        cached = json.loads(Z500_SCALE_CACHE_PATH.read_text(encoding='utf-8'))
        month_params = cached.get('month_band_params', {})
        if month_params and all('lead1_6' in month_params.get(str(month), {}) for month in range(1, 13)):
            return cached

    eval_ds = xr.open_dataset(HINDCAST_EVAL_PATH)
    clim_ds = xr.open_dataset(WEEKLY_CIRC_CLIM_PATH)
    lat = eval_ds['lat'].values.astype(np.float64)
    lat_weights = np.cos(np.deg2rad(lat)).astype(np.float64)[:, None]
    lead_bands = {
        'lead1_6': [1, 2, 3, 4, 5, 6],
        'lead5_6': [5, 6],
    }
    params = {
        'description': 'Global Z500 anomaly linear scaling parameters derived from hindcast model_v3 vs weekly climatology',
        'source_eval': str(HINDCAST_EVAL_PATH.resolve()),
        'source_weekly_clim': str(WEEKLY_CIRC_CLIM_PATH.resolve()),
        'formula': 'scaled_anom = b * pred_anom',
        'month_band_params': {},
    }

    for month in range(1, 13):
        mask_t = eval_ds['time'].dt.month == month
        week_idx = eval_ds['woy'].where(mask_t, drop=True).values.astype(int)
        obs = (eval_ds['obs_z500'].where(mask_t, drop=True).values.astype(np.float32) - clim_ds['z500_clim'].values[week_idx]) / G
        params['month_band_params'][str(month)] = {}
        for band_name, leads in lead_bands.items():
            sum_w = 0.0
            sum_x = 0.0
            sum_y = 0.0
            sum_xx = 0.0
            sum_xy = 0.0
            n = 0
            for lead in leads:
                pred = (eval_ds[f'pred_z500_lead{lead}'].where(mask_t, drop=True).values.astype(np.float32) - clim_ds['z500_clim'].values[week_idx]) / G
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
            params['month_band_params'][str(month)][band_name] = {
                'a': float(a),
                'b': float(b),
                'mean_pred': float(mean_x),
                'mean_obs': float(mean_y),
                'n': int(n),
            }

    clim_ds.close()
    eval_ds.close()
    Z500_SCALE_CACHE_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding='utf-8')
    return params


def scale_hgt_like_z500(monthly_hgt_anom: xr.DataArray, available_months: list[str]) -> xr.DataArray:
    params = build_or_load_z500_scale_params()
    scaled = []
    for month_label in monthly_hgt_anom['month'].values.tolist():
        current = monthly_hgt_anom.sel(month=month_label)
        month_offset = available_months.index(str(month_label))
        if month_offset == 0:
            scaled.append(current.astype(np.float32))
            continue
        month_num = int(str(month_label).split('-')[1])
        coeff = params['month_band_params'][str(month_num)]['lead1_6']
        scaled.append((current * coeff['b']).astype(np.float32))
    return xr.concat(scaled, dim=xr.IndexVariable('month', monthly_hgt_anom['month'].values)).transpose('month', 'lat', 'lon')


def build_or_load_monthly_850_climatology(lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
    MONTHLY_CLIM_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MONTHLY_CLIM_PATH.exists():
        cached = xr.open_dataset(MONTHLY_CLIM_PATH)
        cached_lat = cached['lat'].values.astype(np.float32)
        cached_lon = cached['lon'].values.astype(np.float32)
        if cached_lat.shape == lat.shape and cached_lon.shape == lon.shape and np.allclose(cached_lat, lat) and np.allclose(cached_lon, lon):
            return cached
        interp = cached.interp(lat=lat.astype(np.float32), lon=lon.astype(np.float32))
        cached.close()
        return interp

    op = load_module(RUNNER_PATH, 'run_china_precip_v35')
    clim_helper = load_module(CLIM_HELPER_PATH, 'compute_climatology_helper')
    time_days = clim_helper.read_time_array(clim_helper.STORE_PATH)
    dates = pd.to_datetime(np.datetime64('1940-01-01') + time_days.astype('timedelta64[D]'))
    years = dates.year.values

    upper_arr = clim_helper.ZarrArray(clim_helper.STORE_PATH, 'upper_air')
    z_idx = op.UPPER_VARS.index('z')
    u_idx = op.UPPER_VARS.index('u')
    v_idx = op.UPPER_VARS.index('v')
    level_idx = op.PRESSURE_LEVELS.index(850)

    h850_sum = np.zeros((12, lat.size, lon.size), dtype=np.float64)
    u850_sum = np.zeros((12, lat.size, lon.size), dtype=np.float64)
    v850_sum = np.zeros((12, lat.size, lon.size), dtype=np.float64)
    total_days = np.zeros(12, dtype=np.float64)

    print(f'Building monthly 850 hPa climatology from {clim_helper.STORE_PATH} ...')
    total_count = 0
    for year in range(2002, 2017):
        idx_year = np.where(years == year)[0]
        if idx_year.size == 0:
            continue
        print(f'  Year {year}: {len(idx_year)} weekly chunks')
        for gi in idx_year:
            week_start = pd.Timestamp(dates[int(gi)])
            week_end = week_start + pd.Timedelta(days=6)
            upper = upper_arr.read_time(int(gi))
            h850 = (upper[z_idx, level_idx].astype(np.float32) / G).astype(np.float64)
            u850 = upper[u_idx, level_idx].astype(np.float64)
            v850 = upper[v_idx, level_idx].astype(np.float64)

            if week_start.month == week_end.month and week_start.year == week_end.year:
                month_idx = week_start.month - 1
                h850_sum[month_idx] += h850 * 7.0
                u850_sum[month_idx] += u850 * 7.0
                v850_sum[month_idx] += v850 * 7.0
                total_days[month_idx] += 7.0
            else:
                first_start, first_end = op.month_range(week_start.year, week_start.month)
                days_first = op.overlap_days(week_start, week_end, first_start, first_end)
                if days_first > 0:
                    month_idx = week_start.month - 1
                    h850_sum[month_idx] += h850 * days_first
                    u850_sum[month_idx] += u850 * days_first
                    v850_sum[month_idx] += v850 * days_first
                    total_days[month_idx] += days_first

                second_start, second_end = op.month_range(week_end.year, week_end.month)
                days_second = op.overlap_days(week_start, week_end, second_start, second_end)
                if days_second > 0:
                    month_idx = week_end.month - 1
                    h850_sum[month_idx] += h850 * days_second
                    u850_sum[month_idx] += u850 * days_second
                    v850_sum[month_idx] += v850 * days_second
                    total_days[month_idx] += days_second

            total_count += 1
        print(f'    Cumulative chunks: {total_count}')

    denom = np.maximum(total_days[:, None, None], 1.0)
    ds = xr.Dataset(
        data_vars={
            'h850_clim_gpm': (('month', 'lat', 'lon'), (h850_sum / denom).astype(np.float32)),
            'u850_clim_ms': (('month', 'lat', 'lon'), (u850_sum / denom).astype(np.float32)),
            'v850_clim_ms': (('month', 'lat', 'lon'), (v850_sum / denom).astype(np.float32)),
            'sample_days': (('month',), np.rint(total_days).astype(np.int32)),
        },
        coords={
            'month': np.arange(1, 13, dtype=np.int32),
            'lat': lat.astype(np.float32),
            'lon': lon.astype(np.float32),
        },
        attrs={
            'description': 'Monthly 850 hPa geopotential height and wind climatology derived from weekly ERA5 Zarr archive',
            'source': str(clim_helper.STORE_PATH),
            'years': '2002-2016',
            'note': 'Monthly climatology is computed directly on monthly windows from weekly means using overlap-day weighting.',
        },
    )
    ds.to_netcdf(MONTHLY_CLIM_PATH)
    return xr.open_dataset(MONTHLY_CLIM_PATH)


def build_monthly_anomalies(monthly_ds: xr.Dataset, clim_ds: xr.Dataset, selected_months: list[str]) -> dict[str, xr.DataArray]:
    def monthly_clim(var_name: str) -> xr.DataArray:
        fields = []
        for month_label in selected_months:
            month_num = int(str(month_label).split('-')[1])
            fields.append(clim_ds[var_name].sel(month=month_num).load())
        return xr.concat(fields, dim=xr.IndexVariable('month', np.asarray(selected_months, dtype=object))).transpose('month', 'lat', 'lon')

    hgt = monthly_ds['h850_gpm'].sel(month=selected_months).load()
    u = monthly_ds['u850_ms'].sel(month=selected_months).load()
    v = monthly_ds['v850_ms'].sel(month=selected_months).load()

    hgt_anom = (hgt - monthly_clim('h850_clim_gpm')).astype(np.float32)
    hgt_anom = remove_global_mean_hgt(hgt_anom)
    hgt_anom = scale_hgt_like_z500(hgt_anom, selected_months)
    return {
        'hgt': hgt_anom,
        'u': (u - monthly_clim('u850_clim_ms')).astype(np.float32),
        'v': (v - monthly_clim('v850_clim_ms')).astype(np.float32),
    }


def build_panel_data(monthly_ds: xr.Dataset) -> tuple[list[tuple[str, dict[str, xr.DataArray]]], str]:
    available_months = [str(month) for month in monthly_ds['month'].values.tolist()]
    clim_ds = build_or_load_monthly_850_climatology(
        monthly_ds['lat'].values.astype(np.float32),
        monthly_ds['lon'].values.astype(np.float32),
    )
    monthly_anom = build_monthly_anomalies(monthly_ds, clim_ds, available_months)
    if hasattr(clim_ds, 'close'):
        clim_ds.close()

    season_defs = {
        'MAM': ['2026-03', '2026-04', '2026-05'],
        'JJA': ['2026-06', '2026-07', '2026-08'],
        'SON': ['2026-09', '2026-10', '2026-11'],
    }
    panels = []
    for label, month_labels in season_defs.items():
        missing = [month_label for month_label in month_labels if month_label not in available_months]
        if missing:
            raise KeyError(f'Cannot build {label}; missing months: {missing}')
        panels.append((
            label,
            coarsen_panel({
                'hgt': weighted_mean_months(monthly_anom['hgt'], month_labels),
                'u': weighted_mean_months(monthly_anom['u'], month_labels),
                'v': weighted_mean_months(monthly_anom['v'], month_labels),
            }),
        ))

    if all(month_label in available_months for month_label in ['2026-12', '2027-01']):
        month_labels = ['2026-12', '2027-01']
        dj_panel = coarsen_panel({
            'hgt': weighted_mean_months(monthly_anom['hgt'], month_labels),
            'u': weighted_mean_months(monthly_anom['u'], month_labels),
            'v': weighted_mean_months(monthly_anom['v'], month_labels),
        })
        dj_note = 'monthly_dec_jan'
    elif '2026-12' in available_months:
        dj_panel = coarsen_panel({
            'hgt': monthly_anom['hgt'].sel(month='2026-12').load(),
            'u': monthly_anom['u'].sel(month='2026-12').load(),
            'v': monthly_anom['v'].sel(month='2026-12').load(),
        })
        dj_note = 'monthly_december_only'
    else:
        raise KeyError('Cannot build DJ panel from current monthly products')

    panels.append(('DJ', dj_panel))
    return panels, dj_note


def draw_panel(ax, panel: dict[str, xr.DataArray], cmap, norm):
    lat = panel['hgt']['lat'].values.astype(np.float32)
    lon = panel['hgt']['lon'].values.astype(np.float32)
    hgt = panel['hgt'].values.astype(np.float32)
    u = panel['u'].values.astype(np.float32)
    v = panel['v'].values.astype(np.float32)

    hgt_cyclic, lon_cyclic = add_cyclic_360(hgt, lon)
    mappable = ax.contourf(
        lon_cyclic,
        lat,
        hgt_cyclic,
        levels=np.arange(-22, 24, 2),
        cmap=cmap,
        norm=norm,
        extend='both',
        transform=ccrs.PlateCarree(),
    )
    ax.contour(
        lon_cyclic,
        lat,
        hgt_cyclic,
        levels=np.arange(-20, 21, 4),
        colors='white',
        linewidths=0.6,
        alpha=0.9,
        transform=ccrs.PlateCarree(),
    )

    lat_mask = (lat >= PLOT_EXTENT[2]) & (lat <= PLOT_EXTENT[3])
    lat_idx = np.where(lat_mask)[0][::VECTOR_SKIP_LAT]
    lon_idx = np.arange(0, lon.size, VECTOR_SKIP_LON)
    u_q = u[np.ix_(lat_idx, lon_idx)].copy()
    v_q = v[np.ix_(lat_idx, lon_idx)].copy()
    speed = np.hypot(u_q, v_q)
    u_q[speed < VECTOR_MIN_SPEED] = np.nan
    v_q[speed < VECTOR_MIN_SPEED] = np.nan

    quiver = ax.quiver(
        lon[lon_idx],
        lat[lat_idx],
        u_q,
        v_q,
        transform=ccrs.PlateCarree(),
        color=VECTOR_COLOR,
        scale=55,
        width=0.0016,
        headwidth=3.2,
        headlength=4.0,
        headaxislength=3.8,
        pivot='mid',
        zorder=10,
    )

    ax.coastlines(linewidth=0.9, color='0.2')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
    ax.set_xticks(np.arange(0, 361, 60), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-60, 91, 30), crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=False,
        linewidth=0.7,
        color='gray',
        alpha=0.5,
        linestyle='--',
        xlocs=np.arange(0, 361, 30),
        ylocs=np.arange(-60, 91, 15),
    )
    ax.spines['geo'].set_lw(0.5)
    ax.spines['geo'].set_color('0.5')
    return mappable, quiver


def default_output_path(monthly_nc: Path) -> Path:
    return monthly_nc.with_name('global_seasonal_850hpa_hgt_uv_anomaly_mam_jja_son_dj_operation_s2s.png')


def plot_global_850_circulation(monthly_nc: Path, output_png: Path, vmin: float, vmax: float):
    op = load_module(RUNNER_PATH, 'run_china_precip_v35')
    monthly_ds = xr.open_dataset(monthly_nc)
    panels, dj_note = build_panel_data(monthly_ds)
    monthly_ds.close()

    op.prepare_font()
    projection = ccrs.PlateCarree(central_longitude=180)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = copy.copy(cmaps.BlueWhiteOrangeRed)

    fig = plt.figure(figsize=(18, 10.8))
    axes = [fig.add_subplot(2, 2, idx + 1, projection=projection) for idx in range(4)]

    mappable = None
    quiver = None
    for ax, (_, title_head), (_, panel) in zip(axes, PANEL_TITLES, panels):
        mappable, quiver = draw_panel(ax, panel, cmap, norm)
        ax.tick_params(labelsize=10)
        ax.text(
            0.02, 1.03, f'{title_head}\n{INIT_LINE}',
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=13,
        )

    cbar_ax = fig.add_axes([0.21, 0.055, 0.58, 0.018])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('850 hPa HGT anomaly (gpm)', fontsize=14)
    cbar.set_ticks(np.arange(-22, 23, 2))
    cbar.ax.tick_params(labelsize=10)

    if quiver is not None:
        axes[-1].quiverkey(
            quiver,
            X=0.11,
            Y=0.064,
            U=QUIVER_REF,
            label='2 m/s',
            labelpos='E',
            coordinates='figure',
            fontproperties={'size': 13},
        )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.935, bottom=0.11, wspace=0.12, hspace=-0.1)
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(output_png)
    print(f'DJ source: {dj_note}')
    print(f'Climatology cache: {MONTHLY_CLIM_PATH.resolve()}')
    return output_png, dj_note


def parse_args():
    parser = argparse.ArgumentParser(description='Plot global seasonal 850 hPa circulation anomaly from operation_s2s monthly outputs.')
    parser.add_argument('--monthly-input', type=Path, default=None, help='Monthly NetCDF. Defaults to latest operation_s2s monthly file.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--vmin', type=float, default=-22.0, help='Minimum plotted 850 hPa HGT anomaly in gpm.')
    parser.add_argument('--vmax', type=float, default=22.0, help='Maximum plotted 850 hPa HGT anomaly in gpm.')
    return parser.parse_args()


def main():
    args = parse_args()
    monthly_nc = args.monthly_input if args.monthly_input is not None else find_latest_file('*/operation_s2s_atmos_monthly_*.nc')
    output_png = args.output if args.output is not None else default_output_path(monthly_nc)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_global_850_circulation(monthly_nc, output_png, args.vmin, args.vmax)


if __name__ == '__main__':
    main()
