from __future__ import annotations

from pathlib import Path

import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mplotutils as mpu
import numpy as np
import xarray as xr
from cartopy.util import add_cyclic_point
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
IN_MAP_NC = ROOT / "analysis" / "NWP_AI" / "clean_group_tcc_tp_t2m_maps.nc"
OUT_PNG = ROOT / "analysis" / "NWP_AI" / "tp_tcc_spatial_robinson_lead3_6_4x3.png"
OUT_PDF = ROOT / "analysis" / "NWP_AI" / "tp_tcc_spatial_robinson_lead3_6_4x3.pdf"

MODELS = ["CAS-Canglong", "FuXi-S2S", "ECMWF"]
LEADS = [3, 4, 5, 6]
VARIABLE = "tp"
CMAP = "RdBu_r"
VMIN = -0.30
VCENTER = 0.00
VMAX = 0.60


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
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def cosine_weighted_mean(field: np.ndarray, lat: np.ndarray) -> float:
    weights = np.cos(np.deg2rad(lat)).astype(np.float64)
    weights_2d = np.broadcast_to(weights[:, None], field.shape)
    valid = np.isfinite(field)
    if not np.any(valid):
        return float("nan")
    return float(np.average(field[valid], weights=weights_2d[valid]))


def main() -> None:
    if not IN_MAP_NC.exists():
        raise FileNotFoundError(f"Missing input NetCDF: {IN_MAP_NC}")

    setup_matplotlib()

    with xr.open_dataset(IN_MAP_NC) as ds:
        data = ds["tcc_map"].sel(model=MODELS, variable=VARIABLE, lead=LEADS)
        lat = ds["lat"].values.astype(np.float64, copy=False)
        lon = ds["lon"].values.astype(np.float64, copy=False)
        time_range = ds.attrs.get("time_range", "2017-2021 common weeks")
        n_target_weeks = ds.attrs.get("n_target_weeks", "unknown")

        fig, axes = plt.subplots(
            nrows=len(LEADS),
            ncols=len(MODELS),
            figsize=(17, 16),
            subplot_kw={"projection": ccrs.Robinson(central_longitude=180)},
        )

        norm = mcolors.TwoSlopeNorm(vmin=VMIN, vcenter=VCENTER, vmax=VMAX)
        mesh = None

        for row, lead in enumerate(LEADS):
            for col, model in enumerate(MODELS):
                ax = axes[row, col]
                field = data.sel(model=model, lead=lead).values.astype(np.float64, copy=False)
                field_cyclic, lon_cyclic = add_cyclic_point(field, coord=lon)
                lon2d, lat2d = np.meshgrid(lon_cyclic, lat)
                mean_tcc = cosine_weighted_mean(field, lat)

                mesh = ax.pcolormesh(
                    lon2d,
                    lat2d,
                    field_cyclic,
                    transform=ccrs.PlateCarree(),
                    cmap=CMAP,
                    norm=norm,
                    shading="auto",
                )
                ax.set_global()
                ax.coastlines(linewidth=0.45, color="#404040")

                if row == 0:
                    ax.set_title(model, pad=12, fontweight="bold")
                if col == 0:
                    ax.text(
                        -0.10,
                        0.50,
                        f"Lead {lead}",
                        rotation=90,
                        transform=ax.transAxes,
                        va="center",
                        ha="center",
                        fontsize=15,
                        fontweight="bold",
                    )

                print(
                    f"{model:12s} lead{lead}: "
                    f"weighted_mean={mean_tcc:.4f}, min={np.nanmin(field):.4f}, max={np.nanmax(field):.4f}"
                )

        fig.suptitle(
            "Global TP TCC Spatial Distribution\n"
            f"Common CAS-Canglong / FuXi-S2S / ECMWF target weeks, {time_range} (n={n_target_weeks})",
            y=0.97,
            fontsize=18,
            fontweight="bold",
        )
        fig.text(
            0.5,
            0.01,
            "Projection: Robinson | Variable: TP anomaly TCC | Rows: lead 3-6 | Columns: models",
            ha="center",
            va="bottom",
            fontsize=11,
        )
        fig.subplots_adjust(left=0.055, right=0.985, top=0.88, bottom=0.12, wspace=0.03, hspace=0.0)

        cbar = mpu.colorbar(
            mesh,
            ax=axes,
            orientation="horizontal",
            size=0.022,
            pad=0.22,
            shrink=0.02,
            extend="both",
        )
        cbar.set_label("TCC")

        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, bbox_inches="tight")
        fig.savefig(OUT_PDF, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved figure: {OUT_PNG}")
    print(f"Saved figure: {OUT_PDF}")


if __name__ == "__main__":
    main()
