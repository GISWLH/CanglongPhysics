from __future__ import annotations

import os
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from cartopy.util import add_cyclic_point
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
IN_MAP_NC = ROOT / "analysis" / "NWP_AI" / "clean_group_tcc_tp_t2m_maps.nc"
LAND_COVER_FILE = ROOT / "constant_masks" / "land_cover.npy"
OUT_PNG = ROOT / "analysis" / "NWP_AI" / "tp_tcc_spatial_platecarree_land_stats_lead3_6_4x3.png"
OUT_PDF = ROOT / "analysis" / "NWP_AI" / "tp_tcc_spatial_platecarree_land_stats_lead3_6_4x3.pdf"

MODELS = ["CAS-Canglong", "FuXi-S2S", "ECMWF"]
LEADS = [3, 4, 5, 6]
VARIABLE = "tp"
THRESHOLD = 0.50
LAT_MIN = -60.0
MAP_EXTENT = [0.0, 360.0, LAT_MIN, 90.0]
MAP_BOX_ASPECT = 0.5
XTICKS = np.arange(0.0, 361.0, 60.0)
YTICKS = np.arange(-60.0, 91.0, 30.0)
HIST_XMIN = -0.60
HIST_XMAX = 1.00

TCC_CMAP = LinearSegmentedColormap.from_list(
    "tcc_standard_red",
    ["#fbf7f2", "#f3d9c7", "#e9b590", "#da7c56", "#bd442f", "#7f1d1d"],
    N=256,
)
TCC_CMAP.set_bad((1.0, 1.0, 1.0, 0.0))
TCC_NORM = mcolors.Normalize(vmin=0.00, vmax=1.00)

MAP_PROJECTION = ccrs.PlateCarree(central_longitude=180)
DATA_CRS = ccrs.PlateCarree()
OPENMP_STUBS = "/home/lhwang/anaconda3/pkgs/intel-openmp-2023.1.0-hdb19cb5_46306/lib/libiompstubs5.so"


def setup_matplotlib() -> None:
    font_path = "/usr/share/fonts/arial/ARIAL.TTF"
    try:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
    except Exception:
        font_name = "Arial"

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9.5,
            "axes.linewidth": 0.8,
            "axes.facecolor": "#fbfaf8",
            "axes.edgecolor": "#3f3f3f",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "figure.dpi": 250,
            "savefig.dpi": 450,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def require_openmp_stubs() -> None:
    preload = os.environ.get("LD_PRELOAD", "")
    preload_parts = [part for part in preload.split(":") if part]
    if OPENMP_STUBS not in preload_parts:
        raise RuntimeError(
            "This script must be run with Intel OpenMP stubs to avoid the local SHM2 crash.\n"
            f"Use:\nLD_PRELOAD={OPENMP_STUBS} python {Path(__file__).name}"
        )


def build_land_mask_on_shared_grid(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    if not LAND_COVER_FILE.exists():
        raise FileNotFoundError(f"Missing land-cover file: {LAND_COVER_FILE}")

    land_cover = np.load(LAND_COVER_FILE)
    if land_cover.shape != (721, 1440):
        raise ValueError(f"Unexpected land-cover shape: {land_cover.shape}")

    land_binary = land_cover != 0
    # The archived land mask is stored on a Greenwich-centered 0.25-degree grid.
    # Rolling by 180 degrees aligns it with the 0-360 longitude convention used here.
    land_binary = np.roll(land_binary, land_binary.shape[1] // 2, axis=1)

    land_lat = np.arange(90.0, -90.0 - 0.125, -0.25, dtype=np.float32)
    land_lon = np.arange(0.0, 360.0, 0.25, dtype=np.float32)
    if not np.array_equal(land_lat[::6], lat.astype(np.float32)):
        raise ValueError("1.5-degree latitude grid does not align with the 0.25-degree land mask.")
    if not np.array_equal(land_lon[::6], lon.astype(np.float32)):
        raise ValueError("1.5-degree longitude grid does not align with the 0.25-degree land mask.")

    return land_binary[::6, ::6]


def cosine_weighted_mean(field: np.ndarray, lat: np.ndarray) -> float:
    weights = np.cos(np.deg2rad(lat)).astype(np.float64)
    weights_2d = np.broadcast_to(weights[:, None], field.shape)
    valid = np.isfinite(field)
    if not np.any(valid):
        return float("nan")
    return float(np.average(field[valid], weights=weights_2d[valid]))


def compute_zonal_statistics(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(field)
    count = valid.sum(axis=1).astype(np.float64)
    field_zeroed = np.where(valid, field, 0.0)

    zonal_mean = np.full(field.shape[0], np.nan, dtype=np.float64)
    mean_ok = count > 0
    zonal_mean[mean_ok] = field_zeroed.sum(axis=1)[mean_ok] / count[mean_ok]

    sq_diff = np.where(valid, (field - zonal_mean[:, None]) ** 2, 0.0)
    zonal_std = np.full(field.shape[0], np.nan, dtype=np.float64)
    zonal_std[mean_ok] = np.sqrt(sq_diff.sum(axis=1)[mean_ok] / count[mean_ok])
    return zonal_mean, zonal_std


def smooth_histogram(hist_values: np.ndarray, sigma_bins: float = 1.1) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma_bins)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (offsets / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(hist_values, kernel, mode="same")


def add_histogram_inset(parent_ax: plt.Axes, values: np.ndarray) -> None:
    inset = parent_ax.inset_axes([0.035, 0.065, 0.30, 0.24])
    inset.set_facecolor("none")
    inset.patch.set_alpha(0.0)

    bin_edges = np.linspace(HIST_XMIN, HIST_XMAX, 26)
    hist_values, _ = np.histogram(values, bins=bin_edges, density=True)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_width = np.diff(bin_edges).mean() * 0.88

    inset.bar(
        centers,
        hist_values,
        width=bar_width,
        color=TCC_CMAP(TCC_NORM(centers)),
        edgecolor="white",
        linewidth=0.25,
        zorder=2,
    )

    smooth_values = smooth_histogram(hist_values)
    inset.fill_between(centers, 0.0, smooth_values, color="#8f7d70", alpha=0.10, zorder=1)
    inset.plot(centers, smooth_values, color="#342d26", linewidth=1.0, zorder=3)

    mean_val = float(np.nanmean(values))
    frac_high = float(np.nanmean(values > THRESHOLD))
    inset.axvline(THRESHOLD, color="#202020", linewidth=0.9, linestyle=(0, (4, 2)), zorder=4)

    inset.set_xlim(HIST_XMIN, HIST_XMAX)
    inset.set_xticks([-0.5, 0.0, 0.5, 1.0])
    inset.tick_params(labelsize=6, length=2.2, width=0.7, pad=1.5)
    inset.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    inset.grid(axis="y", color="#d8d5d1", linewidth=0.45, linestyle=(0, (2, 2)), alpha=0.65)
    inset.spines["top"].set_visible(False)
    inset.spines["right"].set_visible(False)
    inset.spines["left"].set_linewidth(0.7)
    inset.spines["bottom"].set_linewidth(0.7)
    inset.set_xlabel("TCC", fontsize=6.5, labelpad=0.8)
    inset.set_ylabel("")
    inset.text(
        0.98,
        0.96,
        f"mean {mean_val:.2f}\n>0.5 {frac_high:.0%}",
        transform=inset.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color="#303030",
    )


def draw_map_panel(
    ax: plt.Axes,
    field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    show_bottom_ticks: bool,
    show_left_ticks: bool,
) -> mpl.collections.QuadMesh:
    field_cyclic, lon_cyclic = add_cyclic_point(field, coord=lon)
    lon2d, lat2d = np.meshgrid(lon_cyclic, lat)

    mesh = ax.pcolormesh(
        lon2d,
        lat2d,
        field_cyclic,
        transform=DATA_CRS,
        cmap=TCC_CMAP,
        norm=TCC_NORM,
        shading="auto",
        rasterized=True,
    )

    ax.set_extent(MAP_EXTENT, crs=DATA_CRS)
    ax.set_aspect("auto")
    ax.set_box_aspect(MAP_BOX_ASPECT)
    ax.set_facecolor("#ffffff")
    ax.coastlines(resolution="110m", linewidth=0.48, color="#3a3a3a")
    ax.set_xticks(XTICKS, crs=DATA_CRS)
    ax.set_yticks(YTICKS, crs=DATA_CRS)
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=False))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.gridlines(
        crs=DATA_CRS,
        draw_labels=False,
        linewidth=0.45,
        color="#bcb8b2",
        alpha=0.55,
        linestyle=(0, (2, 2)),
        xlocs=np.arange(0.0, 361.0, 30.0),
        ylocs=np.arange(LAT_MIN, 91.0, 15.0),
    )
    ax.spines["geo"].set_linewidth(0.75)
    ax.spines["geo"].set_edgecolor("#474747")
    ax.tick_params(length=2.8, width=0.7, pad=1.7, labelsize=7)

    if not show_bottom_ticks:
        ax.tick_params(labelbottom=False)
    if not show_left_ticks:
        ax.tick_params(labelleft=False)

    return mesh


def draw_profile_panel(ax: plt.Axes, zonal_mean: np.ndarray, zonal_std: np.ndarray, lat: np.ndarray, show_bottom_ticks: bool) -> None:
    ax.set_facecolor("#fbfaf8")
    ax.axvline(THRESHOLD, color="#8e8a85", linewidth=0.85, linestyle=(0, (4, 3)), zorder=1)
    ax.fill_betweenx(
        lat,
        zonal_mean - zonal_std,
        zonal_mean + zonal_std,
        color="#ccd3d9",
        alpha=0.50,
        linewidth=0.0,
        zorder=1.5,
    )
    ax.plot(zonal_mean, lat, color="#2f2f2f", linewidth=1.05, zorder=3)

    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_xlim(0.0, 0.5)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels([])
    ax.set_xticks([0.0, 0.25, 0.5])
    ax.grid(axis="x", color="#d8d5d1", linewidth=0.45, linestyle=(0, (2, 2)), alpha=0.75)
    ax.tick_params(length=2.6, width=0.7, pad=1.7, labelsize=7)

    if show_bottom_ticks:
        ax.set_xlabel("Zonal mean TCC", fontsize=7.2, labelpad=1.4)
    else:
        ax.tick_params(labelbottom=False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_edgecolor("#474747")


def main() -> None:
    if not IN_MAP_NC.exists():
        raise FileNotFoundError(f"Missing input NetCDF: {IN_MAP_NC}")

    require_openmp_stubs()
    setup_matplotlib()

    with xr.open_dataset(IN_MAP_NC) as ds:
        data = ds["tcc_map"].sel(model=MODELS, variable=VARIABLE, lead=LEADS)
        lat = ds["lat"].values.astype(np.float64, copy=False)
        lon = ds["lon"].values.astype(np.float64, copy=False)
        time_range = ds.attrs.get("time_range", "2017-2021 common weeks")
        n_target_weeks = ds.attrs.get("n_target_weeks", "unknown")
        land_mask = build_land_mask_on_shared_grid(lat, lon)

        fig = plt.figure(figsize=(24.5, 15.5))
        outer = fig.add_gridspec(
            nrows=len(LEADS),
            ncols=len(MODELS),
            left=0.045,
            right=0.985,
            bottom=0.10,
            top=0.92,
            wspace=0.10,
            hspace=0.12,
        )

        mesh = None
        map_axes: list[plt.Axes] = []

        for row, lead in enumerate(LEADS):
            for col, model in enumerate(MODELS):
                inner = outer[row, col].subgridspec(1, 2, width_ratios=[5.3, 0.95], wspace=0.03)
                ax_map = fig.add_subplot(inner[0, 0], projection=MAP_PROJECTION)
                ax_profile = fig.add_subplot(inner[0, 1])
                map_axes.append(ax_map)

                field = data.sel(model=model, lead=lead).values.astype(np.float64, copy=False)
                field = np.where(land_mask, field, np.nan)
                field = np.where(lat[:, None] >= LAT_MIN, field, np.nan)
                values = field[np.isfinite(field)]
                mean_tcc = cosine_weighted_mean(field, lat)
                frac_high = float(np.mean(values > THRESHOLD)) if values.size else float("nan")
                zonal_mean, zonal_std = compute_zonal_statistics(field)

                mesh = draw_map_panel(
                    ax=ax_map,
                    field=field,
                    lat=lat,
                    lon=lon,
                    show_bottom_ticks=row == len(LEADS) - 1,
                    show_left_ticks=col == 0,
                )
                draw_profile_panel(
                    ax=ax_profile,
                    zonal_mean=zonal_mean,
                    zonal_std=zonal_std,
                    lat=lat,
                    show_bottom_ticks=row == len(LEADS) - 1,
                )
                add_histogram_inset(ax_map, values)

                if row == 0:
                    ax_map.set_title(model, pad=8.5, fontweight="bold")
                if col == 0:
                    ax_map.text(
                        -0.18,
                        0.50,
                        f"Lead {lead}",
                        rotation=90,
                        transform=ax_map.transAxes,
                        va="center",
                        ha="center",
                        fontsize=11.5,
                        fontweight="bold",
                        color="#1f1f1f",
                    )

                ax_map.text(
                    0.985,
                    0.985,
                    f"mean {mean_tcc:.3f}\n>0.5 {frac_high:.0%}",
                    transform=ax_map.transAxes,
                    ha="right",
                    va="top",
                    fontsize=7.0,
                    color="#222222",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.8},
                )

                print(
                    f"{model:12s} lead{lead}: "
                    f"weighted_mean={mean_tcc:.4f}, "
                    f"gt0.5_frac={frac_high:.4f}, "
                    f"min={np.nanmin(field):.4f}, max={np.nanmax(field):.4f}"
                )

        cax = fig.add_axes([0.27, 0.055, 0.46, 0.015])
        cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
        cbar.set_label("TCC", fontsize=10)
        cbar.set_ticks([0.0, 0.5, 1.0])
        cbar.ax.tick_params(labelsize=8, length=2.5, width=0.7)

        fig.suptitle(
            "Global Land TP TCC Spatial Distribution with Threshold Diagnostics",
            y=0.965,
            fontsize=17,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.936,
            f"Rows: lead 3-6 | Columns: CAS-Canglong, FuXi-S2S, ECMWF | Land-only statistics | 60S-90N | Common target weeks: {time_range} (n={n_target_weeks})",
            ha="center",
            va="center",
            fontsize=10.5,
        )
        fig.text(
            0.5,
            0.025,
            "PlateCarree (central_longitude=180). Ocean grid cells and Antarctica south of 60S are masked out. "
            "Inset dashed line marks TCC = 0.5. Right-side profile shows land-only zonal mean TCC with mean +/- std.",
            ha="center",
            va="center",
            fontsize=9.2,
            color="#303030",
        )

        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, bbox_inches="tight")
        fig.savefig(OUT_PDF, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved figure: {OUT_PNG}")
    print(f"Saved figure: {OUT_PDF}")


if __name__ == "__main__":
    main()
