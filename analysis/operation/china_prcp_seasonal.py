from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path

import cartopy.crs as ccrs
import cmaps
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mplotutils as mpu
import numpy as np
import xarray as xr
from shapely import contains_xy
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
RUNNER_PATH = SCRIPT_DIR / 'run_china_precip_v35.py'
CHINA_PRCP_PATH = SCRIPT_DIR / 'china_prcp.py'


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
        da = seasonal_ds['tp_anom_percent'].sel(season=season_label).rename({'season': 'panel'}) if 'season' in seasonal_ds['tp_anom_percent'].dims else seasonal_ds['tp_anom_percent'].sel(season=season_label)
        panels.append((short_label, da))

    if '2026-DJ' in seasonal_ds['season'].values:
        dj_da = seasonal_ds['tp_anom_percent'].sel(season='2026-DJ')
        dj_note = 'seasonal'
    elif '2026-DJF' in seasonal_ds['season'].values:
        dj_da = seasonal_ds['tp_anom_percent'].sel(season='2026-DJF')
        dj_note = 'seasonal'
    elif '2026-12' in monthly_ds['month'].values:
        dj_da = monthly_ds['tp_anom_percent'].sel(month='2026-12')
        dj_note = 'monthly_december_only'
    else:
        raise KeyError('Cannot build DJ panel from current monthly/seasonal products')
    panels.append(('DJ', dj_da))
    return panels, dj_note


def default_output_path(seasonal_nc: Path) -> Path:
    return seasonal_nc.with_name('china_seasonal_tp_anomaly_mam_jja_son_dj_operation_s2s_range75_transparent.png')


def plot_seasonal_china_precip(seasonal_nc: Path, monthly_nc: Path, output_png: Path, vmin: float, vmax: float, transparent_outside: bool):
    op = load_module(RUNNER_PATH, 'run_china_precip_v35')
    base = load_module(CHINA_PRCP_PATH, 'china_prcp')

    seasonal_ds = xr.open_dataset(seasonal_nc)
    monthly_ds = xr.open_dataset(monthly_nc)
    panels, dj_note = build_panel_data(seasonal_ds, monthly_ds)
    panel_labels = [label for label, _ in panels]
    panel_arrays = [da.load() for _, da in panels]
    seasonal_ds.close()
    monthly_ds.close()

    op.prepare_font()
    china_shp = gpd.read_file(op.CHINA_SHP)
    china_geom = unary_union(china_shp.geometry)
    projection = ccrs.LambertConformal(central_longitude=105, central_latitude=40, standard_parallels=(25.0, 47.0))
    levels = np.linspace(vmin, vmax, 11)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = copy.copy(cmaps.drought_severity_r)
    if transparent_outside:
        cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    output_png = Path(output_png).resolve()
    cwd = Path.cwd()
    try:
        os.chdir(ROOT / 'code')
        fig = plt.figure(figsize=(18, 15))
        axes = []
        for idx in range(4):
            ax = fig.add_subplot(2, 2, idx + 1, projection=projection)
            axes.append(ax)

        masked_list = []
        mappable = None
        for idx, (ax, label, current) in enumerate(zip(axes, panel_labels, panel_arrays)):
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
            ax.text(
                0.02, 0.98, f'CAS-Canglong {label}',
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=25,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.65, 'pad': 2.5},
                zorder=20,
            )

        cbar_ax = fig.add_axes([0.91, 0.13, 0.015, 0.72])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Precipitation Anomaly (%)', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        plt.subplots_adjust(left=0.03, right=0.89, top=0.94, bottom=0.06, wspace=0.10, hspace=0.14)
        mpu.set_map_layout(axes, width=80)

        for ax, masked in zip(axes, masked_list):
            pos = ax.get_position()
            ax2 = fig.add_axes(
                [pos.x0 + pos.width * base.SUBMAP_X_OFFSET, pos.y0 + pos.height * base.SUBMAP_Y_OFFSET, pos.width * base.SUBMAP_WIDTH, pos.height * base.SUBMAP_HEIGHT],
                projection=projection,
            )
            op.china_plot.sub_china_map(masked, ax2, cmap=cmap, levels=levels, norm=norm, add_coastlines=False, add_land=False)

        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
    finally:
        os.chdir(cwd)

    print(output_png)
    print(f'DJ source: {dj_note}')
    return output_png, dj_note


def parse_args():
    parser = argparse.ArgumentParser(description='Plot China seasonal precipitation anomaly (2x2: MAM, JJA, SON, DJ) from operation_s2s outputs.')
    parser.add_argument('--seasonal-input', type=Path, default=None, help='Seasonal NetCDF. Defaults to latest operation_s2s seasonal file.')
    parser.add_argument('--monthly-input', type=Path, default=None, help='Monthly NetCDF. Defaults to latest operation_s2s monthly file.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--vmin', type=float, default=-75.0, help='Minimum plotted anomaly percent.')
    parser.add_argument('--vmax', type=float, default=75.0, help='Maximum plotted anomaly percent.')
    parser.set_defaults(transparent_outside=True)
    parser.add_argument('--transparent-outside', action='store_true', dest='transparent_outside', help='Mask anomalies outside [vmin, vmax] as transparent white instead of clipping.')
    parser.add_argument('--clip-outside', action='store_false', dest='transparent_outside', help='Clip anomalies outside [vmin, vmax] instead of masking them transparent.')
    return parser.parse_args()


def main():
    args = parse_args()
    seasonal_nc = args.seasonal_input if args.seasonal_input is not None else find_latest_file('*/operation_s2s_atmos_seasonal_*.nc')
    monthly_nc = args.monthly_input if args.monthly_input is not None else find_latest_file('*/operation_s2s_atmos_monthly_*.nc')
    output_png = args.output if args.output is not None else default_output_path(seasonal_nc)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_seasonal_china_precip(seasonal_nc, monthly_nc, output_png, args.vmin, args.vmax, args.transparent_outside)


if __name__ == '__main__':
    main()
