from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmaps
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import mplotutils as mpu
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib import font_manager


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / 'sst_forecast_seasonal_202603.nc'

SEASON_ORDER = ['MAM', 'JJA', 'SON', 'DJ']
SEASON_FALLBACK = {
    'MAM': ['2026-03', '2026-04', '2026-05'],
    'JJA': ['2026-06', '2026-07', '2026-08'],
    'SON': ['2026-09', '2026-10', '2026-11'],
    'DJ': ['2026-12', '2027-01'],
}


def prepare_font():
    font_path = Path('/usr/share/fonts/arial/ARIAL.TTF')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font_path)).get_name()
    else:
        plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['svg.fonttype'] = 'none'



def add_cyclic_360(da: xr.DataArray) -> xr.DataArray:
    lon = da['lon'].values.astype(np.float32)
    if float(lon[-1]) >= 359.9:
        first = da.isel(lon=0).expand_dims(lon=[360.0])
        return xr.concat([da, first], dim='lon')
    return da



def month_span_text(month_labels: list[str]) -> str:
    start = pd.Timestamp(f'{month_labels[0]}-01')
    end = pd.Timestamp(f'{month_labels[-1]}-01')
    if len(month_labels) == 2:
        return f'{start:%b%Y}-{end:%b%Y}'
    return f'{start:%b%Y}-{end:%b%Y}'



def weighted_mean_months(monthly_da: xr.DataArray, month_labels: list[str]) -> xr.DataArray:
    selected = monthly_da.sel(month=month_labels).load()
    weights = np.array([pd.Timestamp(f'{month_label}-01').days_in_month for month_label in month_labels], dtype=np.float32)
    values = np.tensordot(weights, selected.values.astype(np.float32), axes=(0, 0)) / float(weights.sum())
    return xr.DataArray(values.astype(np.float32), coords={'lat': selected['lat'], 'lon': selected['lon']}, dims=('lat', 'lon'))



def build_seasonal_from_monthly(monthly_ds: xr.Dataset) -> xr.Dataset:
    available_months = [str(month) for month in monthly_ds['month'].values.tolist()]
    seasonal_fields = []
    season_start = []
    season_end = []
    for season in SEASON_ORDER:
        month_labels = SEASON_FALLBACK[season]
        missing = [month_label for month_label in month_labels if month_label not in available_months]
        if missing:
            raise KeyError(f'Missing months for {season}: {missing}')
        seasonal_fields.append(weighted_mean_months(monthly_ds['ssta'], month_labels))
        season_start.append(month_labels[0])
        season_end.append(month_labels[-1])

    seasonal_da = xr.concat(seasonal_fields, dim=xr.IndexVariable('season', np.asarray(SEASON_ORDER, dtype=object)))
    ds = xr.Dataset(
        data_vars={'ssta': seasonal_da},
        coords={
            'season': np.asarray(SEASON_ORDER, dtype=object),
            'season_start': ('season', np.asarray(season_start, dtype=object)),
            'season_end': ('season', np.asarray(season_end, dtype=object)),
            'lat': monthly_ds['lat'].values.astype(np.float32),
            'lon': monthly_ds['lon'].values.astype(np.float32),
        },
        attrs=dict(monthly_ds.attrs),
    )
    return ds



def load_seasonal_dataset(input_path: Path) -> xr.Dataset:
    ds = xr.open_dataset(input_path)
    if 'season' in ds.dims and 'ssta' in ds:
        return ds
    if 'month' in ds.dims and 'ssta' in ds:
        built = build_seasonal_from_monthly(ds)
        ds.close()
        return built
    ds.close()
    raise ValueError(f'Unsupported input dataset: {input_path}')



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
    ax.add_feature(cfeature.LAND, facecolor='#D9D9D9', edgecolor='none', zorder=1)
    ax.coastlines(linewidth=0.7, color='0.2')
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
        alpha=0.45,
        linestyle='--',
        xlocs=np.arange(0, 361, 30),
        ylocs=np.arange(-60, 91, 15),
    )
    ax.spines['geo'].set_lw(0.5)
    ax.spines['geo'].set_color('0.5')
    return mappable



def default_output_path(input_path: Path) -> Path:
    stem_tail = input_path.stem.split('_')[-1]
    forecast_start = stem_tail if stem_tail.isdigit() and len(stem_tail) == 6 else 'unknown'
    return input_path.with_name(f'global_ssta_forecast_mam_jja_son_dj_{forecast_start}.png')



def plot_global_ssta(input_path: Path, output_png: Path, vmin: float, vmax: float):
    seasonal_ds = load_seasonal_dataset(input_path)
    projection = ccrs.PlateCarree(central_longitude=180)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = copy.copy(cmaps.BlueWhiteOrangeRed)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))

    forecast_start = seasonal_ds.attrs.get('forecast_start', '2026-02')
    ensemble_size = seasonal_ds.attrs.get('ensemble_size', '10')
    init_line = f'from {pd.Timestamp(forecast_start + "-01"):%b%Y} (CAS-Canglong SST16; {ensemble_size}-member mean)'

    prepare_font()
    fig = plt.figure(figsize=(18, 10.8))
    axes = [fig.add_subplot(2, 2, idx + 1, projection=projection) for idx in range(4)]

    mappable = None
    for ax, season in zip(axes, SEASON_ORDER):
        current = seasonal_ds['ssta'].sel(season=season).astype(np.float32)
        current = current.clip(min=vmin, max=vmax)
        current = add_cyclic_360(current)
        mappable = draw_global_panel(ax, current, cmap, norm)

        month_attr = seasonal_ds.attrs.get(f'season_months_{season}')
        if month_attr:
            month_labels = [month.strip() for month in month_attr.split(',') if month.strip()]
        else:
            month_labels = SEASON_FALLBACK[season]
        title = f'SSTA ({month_span_text(month_labels)}) forecast'
        ax.tick_params(labelsize=10)
        ax.text(
            0.02,
            1.03,
            f'{title}\n{init_line}',
            transform=ax.transAxes,
            ha='left',
            va='bottom',
            fontsize=13,
        )

    cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.68])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('Sea Surface Temperature Anomaly (°C)', fontsize=16)
    cbar.ax.tick_params(labelsize=12)
    plt.subplots_adjust(left=0.05, right=0.89, top=0.935, bottom=0.07, wspace=0.12, hspace=0.24)
    mpu.set_map_layout(axes, width=46)

    fig.savefig(output_png, dpi=300, bbox_inches='tight')
    fig.savefig(output_png.with_suffix('.svg'), bbox_inches='tight')
    plt.close(fig)
    seasonal_ds.close()
    print(output_png)
    return output_png



def parse_args():
    parser = argparse.ArgumentParser(description='Plot global seasonal SST anomaly forecast from CAS-Canglong SST operation products.')
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT, help='Input seasonal or monthly NetCDF file.')
    parser.add_argument('--output', type=Path, default=None, help='Output PNG path.')
    parser.add_argument('--vmin', type=float, default=-3.0, help='Minimum plotted anomaly in degC.')
    parser.add_argument('--vmax', type=float, default=3.0, help='Maximum plotted anomaly in degC.')
    return parser.parse_args()



def main():
    args = parse_args()
    output_png = args.output if args.output is not None else default_output_path(args.input)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plot_global_ssta(args.input, output_png, args.vmin, args.vmax)


if __name__ == '__main__':
    main()
