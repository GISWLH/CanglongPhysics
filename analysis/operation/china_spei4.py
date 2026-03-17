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
import pandas as pd
import xarray as xr
from shapely import contains_xy
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
RUNNER_PATH = SCRIPT_DIR / 'run_china_precip_v35.py'
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


def find_latest_drought_file(output_root: Path) -> Path:
    candidates = sorted(
        output_root.glob('*/operation_s2s_drought_weekly_spei4_china_*.nc'),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f'No drought SPEI NetCDF found under {output_root}')
    return candidates[0]


def default_output_path(drought_nc: Path, weeks: int) -> Path:
    ds = xr.open_dataset(drought_nc)
    forecast_time = pd.DatetimeIndex(ds['forecast_time'].values)
    ds.close()
    weeks = min(weeks, len(forecast_time))
    start_date = forecast_time[0].strftime('%Y%m%d')
    end_date = (forecast_time[weeks - 1] + pd.Timedelta(days=6)).strftime('%Y%m%d')
    return drought_nc.with_name(f'china_spei4_next{weeks}weeks_{start_date}_{end_date}_operation_s2s.png')


def plot_china_spei4(drought_nc: Path, output_png: Path, weeks: int, vmin: float, vmax: float):
    op = load_runner_module()
    ds = xr.open_dataset(drought_nc)
    spei = ds['spei4'].isel(forecast_time=slice(0, weeks)).load()
    forecast_time = pd.DatetimeIndex(spei['forecast_time'].values)
    ds.close()

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
    cmap = copy.copy(cmaps.BlueWhiteOrangeRed_r)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    output_png = Path(output_png).resolve()
    cwd = Path.cwd()
    try:
        os.chdir(ROOT / 'code')
        fig = plt.figure(figsize=(24, 14.5))
        axes = []
        for idx in range(spei.sizes['forecast_time']):
            ax = fig.add_subplot(2, 3, idx + 1, projection=projection)
            axes.append(ax)

        masked_list = []
        mappable = None
        for idx in range(spei.sizes['forecast_time']):
            ax = axes[idx]
            current = spei.isel(forecast_time=idx)
            lon2d, lat2d = np.meshgrid(current['lon'].values, current['lat'].values)
            china_mask = contains_xy(china_geom, lon2d, lat2d)
            china_mask_da = xr.DataArray(china_mask, coords=current.coords, dims=current.dims)
            masked = current.where(china_mask_da).clip(min=vmin, max=vmax)
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
            week_start = forecast_time[idx]
            week_end = week_start + pd.Timedelta(days=6)
            ax.text(
                0.02,
                0.98,
                f'CAS-Canglong SPEI-4\n{week_start:%Y%m%d}-{week_end:%Y%m%d}',
                transform=ax.transAxes,
                ha='left',
                va='top',
                fontsize=TITLE_FONT_SIZE,
                bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.65, 'pad': 2.5},
                zorder=20,
            )

        cbar_ax = fig.add_axes([0.91, 0.13, 0.015, 0.72])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('SPEI-4', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        plt.subplots_adjust(left=0.03, right=0.89, top=0.94, bottom=0.06, wspace=0.10, hspace=0.14)
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
    return output_png


def parse_args():
    parser = argparse.ArgumentParser(description='Plot China SPEI-4 next 6 weeks from CAS-Canglong S2S drought output.')
    parser.add_argument('--input', type=Path, default=None, help='Drought NetCDF. Defaults to the latest file in analysis/operation/output.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--weeks', type=int, default=6, help='Number of forecast weeks to plot.')
    parser.add_argument('--vmin', type=float, default=-2.0, help='Minimum plotted SPEI.')
    parser.add_argument('--vmax', type=float, default=2.0, help='Maximum plotted SPEI.')
    return parser.parse_args()


def main():
    args = parse_args()
    drought_nc = args.input if args.input is not None else find_latest_drought_file(SCRIPT_DIR / 'output')
    output_png = args.output if args.output is not None else default_output_path(drought_nc, args.weeks)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_china_spei4(drought_nc, output_png, args.weeks, args.vmin, args.vmax)


if __name__ == '__main__':
    main()
