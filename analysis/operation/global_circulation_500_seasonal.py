from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path

import cartopy.crs as ccrs
import cmaps
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
RUNNER_PATH = SCRIPT_DIR / 'run_china_precip_v35.py'
HELPER_850_PATH = SCRIPT_DIR / 'global_circulation_850_seasonal.py'
CLIM_HELPER_PATH = ROOT / 'Infer' / 'compute_climatology.py'
MONTHLY_CLIM_PATH = SCRIPT_DIR / 'cache' / 'global_500hpa_monthly_climatology_2002_2016.nc'

G = 9.80665
INIT_LINE = 'from 1st Feb2026 (IAP-CIESM; 8-member mean)'
PANEL_TITLES = [
    ('MAM', '500 hPa HGT, U, V (Mar2026-May2026) forecast'),
    ('JJA', '500 hPa HGT, U, V (Jun2026-Aug2026) forecast'),
    ('SON', '500 hPa HGT, U, V (Sept2026-Nov2026) forecast'),
    ('DJ', '500 hPa HGT, U, V (Dec2026-Jan2027) forecast'),
]
PLOT_EXTENT = [0, 360, -60, 75]
PLOT_COARSEN_FACTOR = 8
VECTOR_SKIP_LAT = 4
VECTOR_SKIP_LON = 6
VECTOR_MIN_SPEED = 0.8
QUIVER_REF = 4.0
VECTOR_COLOR = '#1b8e3e'
HGT_SMOOTH_SIGMA = 1.0
WIND_SMOOTH_SIGMA = 0.9
SUBHIGH_LEVEL = 5880.0
CLIM_CONTOUR_COLOR = '#6f63ff'
FORECAST_CONTOUR_COLOR = '#ff5e66'


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


def coarsen_for_plot(da: xr.DataArray) -> xr.DataArray:
    return da.coarsen(lat=PLOT_COARSEN_FACTOR, lon=PLOT_COARSEN_FACTOR, boundary='trim').mean().astype(np.float32)


def smooth_for_plot(da: xr.DataArray, sigma: float) -> xr.DataArray:
    arr = gaussian_filter(da.values.astype(np.float32), sigma=(sigma, sigma), mode=('nearest', 'wrap'))
    return xr.DataArray(arr.astype(np.float32), coords=da.coords, dims=da.dims)


def coarsen_and_smooth_panel(panel: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    coarsened = {name: coarsen_for_plot(da) for name, da in panel.items()}
    return {
        'hgt_anom': smooth_for_plot(coarsened['hgt_anom'], HGT_SMOOTH_SIGMA),
        'u_anom': smooth_for_plot(coarsened['u_anom'], WIND_SMOOTH_SIGMA),
        'v_anom': smooth_for_plot(coarsened['v_anom'], WIND_SMOOTH_SIGMA),
        'hgt_abs': smooth_for_plot(coarsened['hgt_abs'], HGT_SMOOTH_SIGMA),
        'hgt_clim_abs': smooth_for_plot(coarsened['hgt_clim_abs'], HGT_SMOOTH_SIGMA),
    }


def scale_h500_with_z500_bias(monthly_hgt_anom: xr.DataArray, selected_months: list[str]) -> xr.DataArray:
    helper850 = load_module(HELPER_850_PATH, 'circulation850_helper_for_500_scale')
    params = helper850.build_or_load_z500_scale_params()
    scaled = []
    for month_label in monthly_hgt_anom['month'].values.tolist():
        current = monthly_hgt_anom.sel(month=month_label)
        month_offset = selected_months.index(str(month_label))
        if month_offset == 0:
            scaled.append(current.astype(np.float32))
            continue
        month_num = int(str(month_label).split('-')[1])
        coeff = params['month_band_params'][str(month_num)]['lead5_6']
        scaled.append((current * coeff['b']).astype(np.float32))
    return xr.concat(scaled, dim=xr.IndexVariable('month', monthly_hgt_anom['month'].values)).transpose('month', 'lat', 'lon')


def build_or_load_monthly_500_climatology(lat: np.ndarray, lon: np.ndarray) -> xr.Dataset:
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
    clim_helper = load_module(CLIM_HELPER_PATH, 'compute_climatology_helper_500')
    time_days = clim_helper.read_time_array(clim_helper.STORE_PATH)
    dates = pd.to_datetime(np.datetime64('1940-01-01') + time_days.astype('timedelta64[D]'))
    years = dates.year.values

    upper_arr = clim_helper.ZarrArray(clim_helper.STORE_PATH, 'upper_air')
    z_idx = op.UPPER_VARS.index('z')
    u_idx = op.UPPER_VARS.index('u')
    v_idx = op.UPPER_VARS.index('v')
    level_idx = op.PRESSURE_LEVELS.index(500)

    h500_sum = np.zeros((12, lat.size, lon.size), dtype=np.float64)
    u500_sum = np.zeros((12, lat.size, lon.size), dtype=np.float64)
    v500_sum = np.zeros((12, lat.size, lon.size), dtype=np.float64)
    total_days = np.zeros(12, dtype=np.float64)

    print(f'Building monthly 500 hPa climatology from {clim_helper.STORE_PATH} ...')
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
            h500 = (upper[z_idx, level_idx].astype(np.float32) / G).astype(np.float64)
            u500 = upper[u_idx, level_idx].astype(np.float64)
            v500 = upper[v_idx, level_idx].astype(np.float64)

            if week_start.month == week_end.month and week_start.year == week_end.year:
                month_idx = week_start.month - 1
                h500_sum[month_idx] += h500 * 7.0
                u500_sum[month_idx] += u500 * 7.0
                v500_sum[month_idx] += v500 * 7.0
                total_days[month_idx] += 7.0
            else:
                first_start, first_end = op.month_range(week_start.year, week_start.month)
                days_first = op.overlap_days(week_start, week_end, first_start, first_end)
                if days_first > 0:
                    month_idx = week_start.month - 1
                    h500_sum[month_idx] += h500 * days_first
                    u500_sum[month_idx] += u500 * days_first
                    v500_sum[month_idx] += v500 * days_first
                    total_days[month_idx] += days_first

                second_start, second_end = op.month_range(week_end.year, week_end.month)
                days_second = op.overlap_days(week_start, week_end, second_start, second_end)
                if days_second > 0:
                    month_idx = week_end.month - 1
                    h500_sum[month_idx] += h500 * days_second
                    u500_sum[month_idx] += u500 * days_second
                    v500_sum[month_idx] += v500 * days_second
                    total_days[month_idx] += days_second
            total_count += 1
        print(f'    Cumulative chunks: {total_count}')

    denom = np.maximum(total_days[:, None, None], 1.0)
    ds = xr.Dataset(
        data_vars={
            'h500_clim_gpm': (('month', 'lat', 'lon'), (h500_sum / denom).astype(np.float32)),
            'u500_clim_ms': (('month', 'lat', 'lon'), (u500_sum / denom).astype(np.float32)),
            'v500_clim_ms': (('month', 'lat', 'lon'), (v500_sum / denom).astype(np.float32)),
            'sample_days': (('month',), np.rint(total_days).astype(np.int32)),
        },
        coords={
            'month': np.arange(1, 13, dtype=np.int32),
            'lat': lat.astype(np.float32),
            'lon': lon.astype(np.float32),
        },
        attrs={
            'description': 'Monthly 500 hPa geopotential height and wind climatology derived from weekly ERA5 Zarr archive',
            'source': str(clim_helper.STORE_PATH),
            'years': '2002-2016',
            'note': 'Monthly climatology is computed directly on monthly windows from weekly means using overlap-day weighting.',
        },
    )
    ds.to_netcdf(MONTHLY_CLIM_PATH)
    return xr.open_dataset(MONTHLY_CLIM_PATH)


def build_monthly_fields(monthly_ds: xr.Dataset, clim_ds: xr.Dataset, selected_months: list[str]) -> dict[str, xr.DataArray]:
    helper850 = load_module(HELPER_850_PATH, 'circulation850_helper_for_500')

    def monthly_clim(var_name: str) -> xr.DataArray:
        fields = []
        for month_label in selected_months:
            month_num = int(str(month_label).split('-')[1])
            fields.append(clim_ds[var_name].sel(month=month_num).load())
        return xr.concat(fields, dim=xr.IndexVariable('month', np.asarray(selected_months, dtype=object))).transpose('month', 'lat', 'lon')

    hgt_abs = monthly_ds['h500_gpm'].sel(month=selected_months).load()
    u_abs = monthly_ds['u500_ms'].sel(month=selected_months).load()
    v_abs = monthly_ds['v500_ms'].sel(month=selected_months).load()
    hgt_clim_abs = monthly_clim('h500_clim_gpm')
    hgt_anom = (hgt_abs - hgt_clim_abs).astype(np.float32)
    hgt_anom = helper850.remove_global_mean_hgt(hgt_anom)
    hgt_anom = scale_h500_with_z500_bias(hgt_anom, selected_months)

    return {
        'hgt_anom': hgt_anom,
        'u_anom': (u_abs - monthly_clim('u500_clim_ms')).astype(np.float32),
        'v_anom': (v_abs - monthly_clim('v500_clim_ms')).astype(np.float32),
        'hgt_abs': hgt_abs.astype(np.float32),
        'hgt_clim_abs': hgt_clim_abs.astype(np.float32),
    }


def build_panel_data(monthly_ds: xr.Dataset) -> tuple[list[tuple[str, dict[str, xr.DataArray]]], str]:
    available_months = [str(month) for month in monthly_ds['month'].values.tolist()]
    clim_ds = build_or_load_monthly_500_climatology(
        monthly_ds['lat'].values.astype(np.float32),
        monthly_ds['lon'].values.astype(np.float32),
    )
    monthly_fields = build_monthly_fields(monthly_ds, clim_ds, available_months)
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
            coarsen_and_smooth_panel({
                'hgt_anom': weighted_mean_months(monthly_fields['hgt_anom'], month_labels),
                'u_anom': weighted_mean_months(monthly_fields['u_anom'], month_labels),
                'v_anom': weighted_mean_months(monthly_fields['v_anom'], month_labels),
                'hgt_abs': weighted_mean_months(monthly_fields['hgt_abs'], month_labels),
                'hgt_clim_abs': weighted_mean_months(monthly_fields['hgt_clim_abs'], month_labels),
            }),
        ))

    if all(month_label in available_months for month_label in ['2026-12', '2027-01']):
        month_labels = ['2026-12', '2027-01']
        dj_panel = coarsen_and_smooth_panel({
            'hgt_anom': weighted_mean_months(monthly_fields['hgt_anom'], month_labels),
            'u_anom': weighted_mean_months(monthly_fields['u_anom'], month_labels),
            'v_anom': weighted_mean_months(monthly_fields['v_anom'], month_labels),
            'hgt_abs': weighted_mean_months(monthly_fields['hgt_abs'], month_labels),
            'hgt_clim_abs': weighted_mean_months(monthly_fields['hgt_clim_abs'], month_labels),
        })
        dj_note = 'monthly_dec_jan'
    elif '2026-12' in available_months:
        dj_panel = coarsen_and_smooth_panel({
            'hgt_anom': monthly_fields['hgt_anom'].sel(month='2026-12').load(),
            'u_anom': monthly_fields['u_anom'].sel(month='2026-12').load(),
            'v_anom': monthly_fields['v_anom'].sel(month='2026-12').load(),
            'hgt_abs': monthly_fields['hgt_abs'].sel(month='2026-12').load(),
            'hgt_clim_abs': monthly_fields['hgt_clim_abs'].sel(month='2026-12').load(),
        })
        dj_note = 'monthly_december_only'
    else:
        raise KeyError('Cannot build DJ panel from current monthly products')

    panels.append(('DJ', dj_panel))
    return panels, dj_note


def draw_panel(ax, panel: dict[str, xr.DataArray], cmap, norm):
    lat = panel['hgt_anom']['lat'].values.astype(np.float32)
    lon = panel['hgt_anom']['lon'].values.astype(np.float32)
    hgt = panel['hgt_anom'].values.astype(np.float32)
    u = panel['u_anom'].values.astype(np.float32)
    v = panel['v_anom'].values.astype(np.float32)
    hgt_abs = panel['hgt_abs'].values.astype(np.float32)
    hgt_clim_abs = panel['hgt_clim_abs'].values.astype(np.float32)

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

    hgt_abs_cyclic, lon_abs_cyclic = add_cyclic_360(hgt_abs, lon)
    hgt_clim_cyclic, lon_clim_cyclic = add_cyclic_360(hgt_clim_abs, lon)
    forecast_cs = ax.contour(
        lon_abs_cyclic,
        lat,
        hgt_abs_cyclic,
        levels=[SUBHIGH_LEVEL],
        colors=[FORECAST_CONTOUR_COLOR],
        linewidths=1.8,
        transform=ccrs.PlateCarree(),
        zorder=11,
    )
    clim_cs = ax.contour(
        lon_clim_cyclic,
        lat,
        hgt_clim_cyclic,
        levels=[SUBHIGH_LEVEL],
        colors=[CLIM_CONTOUR_COLOR],
        linewidths=1.6,
        transform=ccrs.PlateCarree(),
        zorder=11,
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
        scale=80,
        width=0.0017,
        headwidth=3.1,
        headlength=3.9,
        headaxislength=3.7,
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
    return mappable, quiver, forecast_cs, clim_cs


def default_output_path(monthly_nc: Path) -> Path:
    return monthly_nc.with_name('global_seasonal_500hpa_hgt_uv_anomaly_mam_jja_son_dj_operation_s2s.png')


def plot_global_500_circulation(monthly_nc: Path, output_png: Path, vmin: float, vmax: float):
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
        mappable, quiver, _, _ = draw_panel(ax, panel, cmap, norm)
        ax.tick_params(labelsize=10)
        ax.text(
            0.02, 1.03, f'{title_head}\n{INIT_LINE}',
            transform=ax.transAxes,
            ha='left', va='bottom',
            fontsize=13,
        )

    cbar_ax = fig.add_axes([0.21, 0.055, 0.58, 0.018])
    cbar = fig.colorbar(mappable, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('500 hPa HGT anomaly (gpm)', fontsize=14)
    cbar.set_ticks(np.arange(-22, 23, 2))
    cbar.ax.tick_params(labelsize=10)

    if quiver is not None:
        axes[-1].quiverkey(
            quiver,
            X=0.11,
            Y=0.064,
            U=QUIVER_REF,
            label='4 m/s',
            labelpos='E',
            coordinates='figure',
            fontproperties={'size': 13},
        )

    legend_handles = [
        Line2D([0], [0], color=CLIM_CONTOUR_COLOR, lw=2.2, label='Clim. 588 gpm'),
        Line2D([0], [0], color=FORECAST_CONTOUR_COLOR, lw=2.2, label='Fore. 588 gpm'),
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower left',
        bbox_to_anchor=(0.12, 0.035),
        ncol=2,
        frameon=False,
        fontsize=12,
        handlelength=2.6,
        columnspacing=1.8,
    )

    plt.subplots_adjust(left=0.05, right=0.95, top=0.935, bottom=0.11, wspace=0.12, hspace=-0.1)
    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(output_png)
    print(f'DJ source: {dj_note}')
    print(f'Climatology cache: {MONTHLY_CLIM_PATH.resolve()}')
    return output_png, dj_note


def parse_args():
    parser = argparse.ArgumentParser(description='Plot global seasonal 500 hPa circulation anomaly from operation_s2s monthly outputs.')
    parser.add_argument('--monthly-input', type=Path, default=None, help='Monthly NetCDF. Defaults to latest operation_s2s monthly file.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--vmin', type=float, default=-22.0, help='Minimum plotted 500 hPa HGT anomaly in gpm.')
    parser.add_argument('--vmax', type=float, default=22.0, help='Maximum plotted 500 hPa HGT anomaly in gpm.')
    return parser.parse_args()


def main():
    args = parse_args()
    monthly_nc = args.monthly_input if args.monthly_input is not None else find_latest_file('*/operation_s2s_atmos_monthly_*.nc')
    output_png = args.output if args.output is not None else default_output_path(monthly_nc)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_global_500_circulation(monthly_nc, output_png, args.vmin, args.vmax)


if __name__ == '__main__':
    main()
