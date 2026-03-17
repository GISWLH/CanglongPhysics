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
CHINA_TEMP_PATH = SCRIPT_DIR / 'china_temp.py'


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

    available_months = [str(month) for month in monthly_ds['month'].values.tolist()]
    clim_ds = temp.build_or_load_monthly_t2m_climatology(op, monthly_ds['lat'].values.astype(np.float32), monthly_ds['lon'].values.astype(np.float32))
    raw_monthly_anom = temp.build_temperature_anomaly(monthly_ds, clim_ds, available_months)
    if hasattr(clim_ds, 'close'):
        clim_ds.close()

    correction_notes = []
    if bias_corrected:
        params = temp.build_or_load_bias_correction_params(op)
        monthly_anom, correction_notes = temp.apply_bias_correction(raw_monthly_anom, available_months, available_months, params)
    else:
        monthly_anom = raw_monthly_anom

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

    if all(month_label in available_months for month_label in ['2026-12', '2027-01', '2027-02']):
        dj_da = weighted_mean_months(monthly_anom, ['2026-12', '2027-01', '2027-02'])
        dj_note = 'seasonal_djf'
    elif '2026-12' in available_months:
        dj_da = monthly_anom.sel(month='2026-12')
        dj_note = 'monthly_december_only'
    else:
        raise KeyError('Cannot build DJ panel from current monthly products')
    panels.append(('DJ', dj_da.load()))
    return panels, dj_note, correction_notes


def default_output_path(monthly_nc: Path, bias_corrected: bool, vmin: float, vmax: float, transparent_outside: bool) -> Path:
    suffix = '_bias_corrected' if bias_corrected else ''
    range_suffix = f'_range{int(abs(vmax))}' if float(abs(vmax)).is_integer() else f'_range{abs(vmax):g}'
    if transparent_outside:
        range_suffix += '_transparent'
    return monthly_nc.with_name(f'china_seasonal_t2m_anomaly_mam_jja_son_dj_operation_s2s{suffix}{range_suffix}.png')


def plot_seasonal_china_temperature(monthly_nc: Path, output_png: Path, vmin: float, vmax: float, transparent_outside: bool, bias_corrected: bool):
    op = load_module(RUNNER_PATH, 'run_china_precip_v35')
    temp = load_module(CHINA_TEMP_PATH, 'china_temp')

    monthly_ds = xr.open_dataset(monthly_nc)
    panels, dj_note, correction_notes = build_panel_data(monthly_ds, bias_corrected)
    panel_labels = [label for label, _ in panels]
    panel_arrays = [da.load() for _, da in panels]
    monthly_ds.close()

    op.prepare_font()
    china_shp = gpd.read_file(op.CHINA_SHP)
    china_geom = unary_union(china_shp.geometry)
    projection = ccrs.LambertConformal(central_longitude=105, central_latitude=40, standard_parallels=(25.0, 47.0))
    levels = np.linspace(vmin, vmax, 11)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = copy.copy(cmaps.BlueWhiteOrangeRed)
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
        for ax, label, current in zip(axes, panel_labels, panel_arrays):
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
        cbar.set_label('Temperature Anomaly (°C)', fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        plt.subplots_adjust(left=0.03, right=0.89, top=0.94, bottom=0.06, wspace=0.10, hspace=0.14)
        mpu.set_map_layout(axes, width=80)

        for ax, masked in zip(axes, masked_list):
            pos = ax.get_position()
            ax2 = fig.add_axes(
                [
                    pos.x0 + pos.width * temp.SUBMAP_X_OFFSET,
                    pos.y0 + pos.height * temp.SUBMAP_Y_OFFSET,
                    pos.width * temp.SUBMAP_WIDTH,
                    pos.height * temp.SUBMAP_HEIGHT,
                ],
                projection=projection,
            )
            op.china_plot.sub_china_map(masked, ax2, cmap=cmap, levels=levels, norm=norm, add_coastlines=False, add_land=False)

        fig.savefig(output_png, dpi=300, bbox_inches='tight')
        plt.close(fig)
    finally:
        os.chdir(cwd)

    print(output_png)
    print(f'DJ source: {dj_note}')
    if bias_corrected:
        print(f'Bias-correction cache: {temp.BIAS_CORR_CACHE_PATH.resolve()}')
        for note in correction_notes:
            print(note)
    return output_png, dj_note


def parse_args():
    parser = argparse.ArgumentParser(description='Plot China seasonal 2m temperature anomaly (2x2: MAM, JJA, SON, DJ) from operation_s2s monthly outputs.')
    parser.add_argument('--monthly-input', type=Path, default=None, help='Monthly NetCDF. Defaults to latest operation_s2s monthly file.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--vmin', type=float, default=-5.0, help='Minimum plotted anomaly in degC.')
    parser.add_argument('--vmax', type=float, default=5.0, help='Maximum plotted anomaly in degC.')
    parser.add_argument('--bias-corrected', action='store_true', help='Apply monthly bias correction before aggregating to seasons.')
    parser.set_defaults(transparent_outside=False)
    parser.add_argument('--transparent-outside', action='store_true', dest='transparent_outside', help='Mask anomalies outside [vmin, vmax] as transparent white instead of clipping.')
    parser.add_argument('--clip-outside', action='store_false', dest='transparent_outside', help='Clip anomalies outside [vmin, vmax] instead of masking them transparent.')
    return parser.parse_args()


def main():
    args = parse_args()
    monthly_nc = args.monthly_input if args.monthly_input is not None else find_latest_file('*/operation_s2s_atmos_monthly_*.nc')
    output_png = args.output if args.output is not None else default_output_path(monthly_nc, args.bias_corrected, args.vmin, args.vmax, args.transparent_outside)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_seasonal_china_temperature(monthly_nc, output_png, args.vmin, args.vmax, args.transparent_outside, args.bias_corrected)


if __name__ == '__main__':
    main()
