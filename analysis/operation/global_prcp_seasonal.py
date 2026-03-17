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
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
RUNNER_PATH = SCRIPT_DIR / 'run_china_precip_v35.py'
GLOBAL_TEMP_PATH = SCRIPT_DIR / 'global_temp.py'

INIT_LINE = 'from 1st Feb2026 (IAP-CIESM; 15-member mean)'
PANEL_TITLES = [
    ('MAM', 'Precipitation Anomaly (Mar2026-May2026) forecast'),
    ('JJA', 'Precipitation Anomaly (Jun2026-Aug2026) forecast'),
    ('SON', 'Precipitation Anomaly (Sept2026-Nov2026) forecast'),
    ('DJ', 'Precipitation Anomaly (Dec2026-Jan2027) forecast'),
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


def build_panel_data(seasonal_ds: xr.Dataset, monthly_ds: xr.Dataset):
    panels = []
    season_map = {
        'MAM': '2026-MAM',
        'JJA': '2026-JJA',
        'SON': '2026-SON',
    }
    for short_label, season_label in season_map.items():
        if season_label not in seasonal_ds['season'].values:
            raise KeyError(f'{season_label} not found in seasonal dataset')
        panels.append((short_label, seasonal_ds['tp_anom_percent'].sel(season=season_label).load()))

    if '2026-12' in monthly_ds['month'].values:
        dj_da = monthly_ds['tp_anom_percent'].sel(month='2026-12').load()
        dj_note = 'monthly_december_only'
    else:
        raise KeyError('Cannot build DJ panel from current monthly products')
    panels.append(('DJ', dj_da))
    return panels, dj_note


def default_output_path(seasonal_nc: Path) -> Path:
    return seasonal_nc.with_name('global_seasonal_tp_anomaly_mam_jja_son_dj_operation_s2s_land.png')


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


def plot_seasonal_global_precip(seasonal_nc: Path, monthly_nc: Path, output_png: Path, vmin: float, vmax: float):
    op = load_module(RUNNER_PATH, 'run_china_precip_v35')
    global_temp = load_module(GLOBAL_TEMP_PATH, 'global_temp')
    seasonal_ds = xr.open_dataset(seasonal_nc)
    monthly_ds = xr.open_dataset(monthly_nc)
    panels, dj_note = build_panel_data(seasonal_ds, monthly_ds)
    panel_arrays = [da for _, da in panels]
    seasonal_ds.close()
    monthly_ds.close()

    lon_native = panel_arrays[0]['lon'].values.astype(np.float32)
    lat = panel_arrays[0]['lat'].values.astype(np.float32)
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
    cmap = copy.copy(cmaps.drought_severity_r)
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
        cbar.set_label('Precipitation Anomaly (%)', fontsize=16)
        cbar.ax.tick_params(labelsize=12)
        plt.subplots_adjust(left=0.05, right=0.89, top=0.935, bottom=0.07, wspace=0.12, hspace=-0.3)

        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
    finally:
        os.chdir(cwd)

    print(output_png)
    print(f'DJ source: {dj_note}')
    return output_png, dj_note


def parse_args():
    parser = argparse.ArgumentParser(description='Plot global land-only seasonal precipitation anomaly from operation_s2s outputs.')
    parser.add_argument('--seasonal-input', type=Path, default=None, help='Seasonal NetCDF. Defaults to latest operation_s2s seasonal file.')
    parser.add_argument('--monthly-input', type=Path, default=None, help='Monthly NetCDF. Defaults to latest operation_s2s monthly file.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--vmin', type=float, default=-75.0, help='Minimum plotted anomaly percent.')
    parser.add_argument('--vmax', type=float, default=75.0, help='Maximum plotted anomaly percent.')
    return parser.parse_args()


def main():
    args = parse_args()
    seasonal_nc = args.seasonal_input if args.seasonal_input is not None else find_latest_file('*/operation_s2s_atmos_seasonal_*.nc')
    monthly_nc = args.monthly_input if args.monthly_input is not None else find_latest_file('*/operation_s2s_atmos_monthly_*.nc')
    output_png = args.output if args.output is not None else default_output_path(seasonal_nc)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_seasonal_global_precip(seasonal_nc, monthly_nc, output_png, args.vmin, args.vmax)


if __name__ == '__main__':
    main()
