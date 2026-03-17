from __future__ import annotations

import argparse
import copy
import importlib.util
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
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
RUNNER_PATH = SCRIPT_DIR / 'run_china_precip_v35.py'
CHINA_TEMP_PATH = SCRIPT_DIR / 'china_temp.py'
GLOBAL_TEMP_PATH = SCRIPT_DIR / 'global_temp.py'

INIT_LINE = 'from 1st Feb2026 (IAP-CIESM; 15-member mean)'
PANEL_TITLES = [
    ('MAM', 'SATA (Mar2026-May2026) forecast'),
    ('JJA', 'SATA (Jun2026-Aug2026) forecast'),
    ('SON', 'SATA (Sept2026-Nov2026) forecast'),
    ('DJ', 'SATA (Dec2026-Jan2027) forecast'),
]


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


def weighted_mean_months(monthly_da: xr.DataArray, month_labels: list[str]) -> xr.DataArray:
    selected = monthly_da.sel(month=month_labels).load()
    weights = np.array([pd.Timestamp(f'{month_label}-01').days_in_month for month_label in month_labels], dtype=np.float32)
    values = np.tensordot(weights, selected.values.astype(np.float32), axes=(0, 0)) / float(weights.sum())
    return xr.DataArray(values.astype(np.float32), coords={'lat': selected['lat'], 'lon': selected['lon']}, dims=('lat', 'lon'))


def build_panel_data(monthly_ds: xr.Dataset, bias_corrected: bool):
    op = load_module(RUNNER_PATH, 'run_china_precip_v35')
    temp = load_module(CHINA_TEMP_PATH, 'china_temp')
    global_temp = load_module(GLOBAL_TEMP_PATH, 'global_temp')

    available_months = [str(month) for month in monthly_ds['month'].values.tolist()]
    clim_ds = temp.build_or_load_monthly_t2m_climatology(op, monthly_ds['lat'].values.astype(np.float32), monthly_ds['lon'].values.astype(np.float32))
    monthly_anom = temp.build_temperature_anomaly(monthly_ds, clim_ds, available_months)
    if hasattr(clim_ds, 'close'):
        clim_ds.close()
    correction_notes = []
    if bias_corrected:
        params = global_temp.build_or_load_global_bias_correction_params(op)
        monthly_anom, correction_notes = global_temp.apply_global_bias_correction(monthly_anom, available_months, available_months, params)

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
        panels.append((label, weighted_mean_months(monthly_anom, month_labels)))

    if all(month_label in available_months for month_label in ['2026-12', '2027-01']):
        dj_da = weighted_mean_months(monthly_anom, ['2026-12', '2027-01'])
        dj_note = 'monthly_dec_jan'
    elif '2026-12' in available_months:
        dj_da = monthly_anom.sel(month='2026-12')
        dj_note = 'monthly_december_only'
    else:
        raise KeyError('Cannot build DJ panel from current monthly products')
    panels.append(('DJ', dj_da.load()))
    return panels, dj_note, correction_notes


def default_output_path(monthly_nc: Path) -> Path:
    return monthly_nc.with_name('global_seasonal_t2m_anomaly_mam_jja_son_dj_operation_s2s_land.png')


def draw_global_panel(ax, current: xr.DataArray, cmap, norm):
    mappable = current.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        add_colorbar=False,
        add_labels=False,
        rasterized=True,
    )
    ax.coastlines(linewidth=0.7, color='0.2')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_extent([0, 360, -60, 75], crs=ccrs.PlateCarree())
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
    return mappable


def add_cyclic_360(da: xr.DataArray) -> xr.DataArray:
    if float(da['lon'].values[-1]) >= 359.9:
        first = da.isel(lon=0).expand_dims(lon=[360.0])
        return xr.concat([da, first], dim='lon')
    return da


def plot_seasonal_global_temperature(monthly_nc: Path, output_png: Path, vmin: float, vmax: float, bias_corrected: bool):
    op = load_module(RUNNER_PATH, 'run_china_precip_v35')
    global_temp = load_module(GLOBAL_TEMP_PATH, 'global_temp')
    monthly_ds = xr.open_dataset(monthly_nc)
    panels, dj_note, correction_notes = build_panel_data(monthly_ds, bias_corrected)
    panel_arrays = [da.load() for _, da in panels]
    lon_native = panel_arrays[0]['lon'].values.astype(np.float32)
    lat = panel_arrays[0]['lat'].values.astype(np.float32)
    monthly_ds.close()

    land_mask = global_temp.build_global_land_mask(lat, lon_native)

    shifted_arrays = []
    for da in panel_arrays:
        current = da.where(land_mask)
        current = current.clip(min=vmin, max=vmax)
        shifted_arrays.append(add_cyclic_360(current))

    op.prepare_font()
    projection = ccrs.PlateCarree(central_longitude=180)
    levels = np.linspace(vmin, vmax, 11)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = copy.copy(cmaps.temp_diff_18lev)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    output_png = Path(output_png).resolve()
    cwd = Path.cwd()
    try:
        os.chdir(ROOT / 'code')
        fig = plt.figure(figsize=(18, 10.8))
        axes = []
        for idx in range(4):
            ax = fig.add_subplot(2, 2, idx + 1, projection=projection)
            axes.append(ax)

        mappable = None
        for ax, (_, title_head), current in zip(axes, PANEL_TITLES, shifted_arrays):
            mappable = draw_global_panel(ax, current, cmap, norm)
            ax.tick_params(labelsize=10)
            ax.text(
                0.02, 1.03, f'{title_head}\n{INIT_LINE}',
                transform=ax.transAxes,
                ha='left', va='bottom',
                fontsize=13,
            )

        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.68])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('SATA (°C)', fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        plt.subplots_adjust(left=0.05, right=0.89, top=0.935, bottom=0.07, wspace=0.12, hspace=-0.2)

        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
    finally:
        os.chdir(cwd)

    print(output_png)
    print(f'DJ source: {dj_note}')
    if bias_corrected:
        print(f'Bias-correction cache: {global_temp.GLOBAL_BIAS_CORR_CACHE_PATH.resolve()}')
        for note in correction_notes:
            print(note)
    return output_png, dj_note


def parse_args():
    parser = argparse.ArgumentParser(description='Plot global land-only seasonal 2m temperature anomaly from operation_s2s monthly outputs.')
    parser.add_argument('--monthly-input', type=Path, default=None, help='Monthly NetCDF. Defaults to latest operation_s2s monthly file.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--vmin', type=float, default=-5.0, help='Minimum plotted anomaly in degC.')
    parser.add_argument('--vmax', type=float, default=5.0, help='Maximum plotted anomaly in degC.')
    parser.add_argument('--bias-corrected', action='store_true', help='Apply global-land monthly bias correction before aggregating to seasons.')
    return parser.parse_args()


def main():
    args = parse_args()
    monthly_nc = args.monthly_input if args.monthly_input is not None else find_latest_file('*/operation_s2s_atmos_monthly_*.nc')
    output_png = args.output if args.output is not None else default_output_path(monthly_nc)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_seasonal_global_temperature(monthly_nc, output_png, args.vmin, args.vmax, args.bias_corrected)


if __name__ == '__main__':
    main()
