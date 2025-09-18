import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmaps
from matplotlib import font_manager

from utils import plot

# Configure Arial font as required for all plots
font_path = "/usr/share/fonts/arial/ARIAL.TTF"
try:
    font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
except FileNotFoundError:
    plt.rcParams["font.family"] = "Arial"

# Common coordinates
lat = np.linspace(90, -90, 721)
lon = np.linspace(0, 359.75, 1440)

# Helper to move tensor data onto CPU numpy arrays
cpu_numpy = lambda tensor: tensor.detach().cpu().numpy()

# Prepare all fields before plotting
precip_mm_day = cpu_numpy(pred_total_precip)[0] * 86400.0
surface_np = cpu_numpy(output_surface)
upper_air_np = cpu_numpy(output_upper_air)

avg_tnlwrf = surface_np[0, 9, 0]
t2m = surface_np[0, 8, 0]
d2m = surface_np[0, 7, 0]
u200 = upper_air_np[0, 2, 0, 0]
u850 = upper_air_np[0, 2, 4, 0]

plot_configs = [
    {
        "data": xr.DataArray(
            precip_mm_day,
            coords=[("lat", lat), ("lon", lon)],
            name="Total precipitation (mm/day)",
        ),
        "levels": np.linspace(0, 60, num=19),
        "cmap": cmaps.cmocean_dense,
    },
    {
        "data": xr.DataArray(
            avg_tnlwrf,
            coords=[("lat", lat), ("lon", lon)],
            name="Avg. top net longwave flux (W m$^{-2}$)",
        ),
        "levels": np.linspace(-350, -100, num=19),
        "cmap": "Oranges",
    },
    {
        "data": xr.DataArray(
            t2m,
            coords=[("lat", lat), ("lon", lon)],
            name="2m temperature (K)",
        ),
        "levels": np.linspace(250, 310, num=19),
        "cmap": "RdBu_r",
    },
    {
        "data": xr.DataArray(
            d2m,
            coords=[("lat", lat), ("lon", lon)],
            name="2m dew point (K)",
        ),
        "levels": np.linspace(250, 305, num=19),
        "cmap": "RdYlBu_r",
    },
    {
        "data": xr.DataArray(
            u200,
            coords=[("lat", lat), ("lon", lon)],
            name="200 hPa zonal wind (m s$^{-1}$)",
        ),
        "levels": np.linspace(-40, 40, num=19),
        "cmap": "BrBG_r",
    },
    {
        "data": xr.DataArray(
            u850,
            coords=[("lat", lat), ("lon", lon)],
            name="850 hPa zonal wind (m s$^{-1}$)",
        ),
        "levels": np.linspace(-20, 20, num=19),
        "cmap": "BrBG_r",
    },
]

fig, axes = plt.subplots(
    nrows=2,
    ncols=3,
    figsize=(18, 9),
    subplot_kw={"projection": ccrs.Robinson()},
)
axes = axes.flatten()

for ax, cfg in zip(axes, plot_configs):
    plot.one_map_flat(
        cfg["data"],
        ax,
        levels=cfg["levels"],
        cmap=cfg["cmap"],
        mask_ocean=False,
        add_coastlines=True,
        add_land=False,
        plotfunc="pcolormesh",
        colorbar=True,
    )
    ax.set_title(cfg["data"].name, fontsize=10, pad=6)

for ax in axes[len(plot_configs):]:
    ax.set_visible(False)

plt.tight_layout()
plt.show()
