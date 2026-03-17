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
SUBMAP_X_OFFSET = 0.775 + 0.01
SUBMAP_Y_OFFSET = 0.022
SUBMAP_WIDTH = 0.216
SUBMAP_HEIGHT = 0.264


def load_runner_module():
    spec = importlib.util.spec_from_file_location('run_china_precip_v35', RUNNER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_latest_monthly_file(output_root: Path) -> Path:
    patterns = ['*/operation_s2s_atmos_monthly_*.nc', '*/operation_all_atmos_monthly_*.nc', '*/china_monthly_tp_feb_mar_apr*.nc']
    candidates = []
    for pattern in patterns:
        candidates.extend(output_root.glob(pattern))
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f'No monthly precipitation NetCDF found under {output_root}')
    return candidates[0]


def default_output_path(monthly_nc: Path, vmin: float, vmax: float, transparent_outside: bool) -> Path:
    suffix = f'_range{int(abs(vmax))}' if float(abs(vmax)).is_integer() else f'_range{abs(vmax):g}'
    if transparent_outside:
        suffix += '_transparent'
    return monthly_nc.with_name(f'{monthly_nc.stem}{suffix}.png')


def plot_china_precip(monthly_nc: Path, output_png: Path, vmin: float, vmax: float, transparent_outside: bool, months: list[str] | None = None):
    op = load_runner_module()
    ds = xr.open_dataset(monthly_nc)
    anom_name = 'tp_anom_percent' if 'tp_anom_percent' in ds.data_vars else 'anom_percent'
    anom = ds[anom_name].load()
    if months is not None:
        anom = anom.sel(month=months)
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
    cmap = copy.copy(cmaps.drought_severity_r)
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
            china_mask_da = xr.DataArray(china_mask, coords=current.coords, dims=current.dims)
            masked = current.where(china_mask_da)
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
            ax.set_title(f'CAS-Canglong {month_label}', fontsize=20)

        cbar_ax = fig.add_axes([0.90, 0.15, 0.012, 0.70])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label('Precipitation Anomaly (%)', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        plt.subplots_adjust(left=0.025, right=0.88, top=0.92, bottom=0.08, wspace=0.20)
        mpu.set_map_layout(axes, width=80)

        for ax, masked in zip(axes, masked_list):
            pos = ax.get_position()
            ax2 = fig.add_axes(
                [pos.x0 + pos.width * SUBMAP_X_OFFSET, pos.y0 + pos.height * SUBMAP_Y_OFFSET, pos.width * SUBMAP_WIDTH, pos.height * SUBMAP_HEIGHT],
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
    parser = argparse.ArgumentParser(description='Plot China precipitation anomaly from monthly CAS-Canglong output.')
    parser.add_argument('--input', type=Path, default=None, help='Monthly precipitation NetCDF. Defaults to the latest file in analysis/operation/output.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--vmin', type=float, default=-75.0, help='Minimum plotted anomaly percent.')
    parser.add_argument('--vmax', type=float, default=75.0, help='Maximum plotted anomaly percent.')
    parser.add_argument('--months', nargs='+', default=None, help='Optional subset of month labels to plot, for example 2026-02 2026-03 2026-04.')
    parser.set_defaults(transparent_outside=True)
    parser.add_argument('--transparent-outside', action='store_true', dest='transparent_outside', help='Mask anomalies outside [vmin, vmax] as transparent white instead of clipping.')
    parser.add_argument('--clip-outside', action='store_false', dest='transparent_outside', help='Clip anomalies outside [vmin, vmax] instead of masking them transparent.')
    return parser.parse_args()


def main():
    args = parse_args()
    monthly_nc = args.input if args.input is not None else find_latest_monthly_file(SCRIPT_DIR / 'output')
    output_png = args.output if args.output is not None else default_output_path(monthly_nc, args.vmin, args.vmax, args.transparent_outside)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_china_precip(monthly_nc, output_png, args.vmin, args.vmax, args.transparent_outside, months=args.months)


if __name__ == '__main__':
    main()
