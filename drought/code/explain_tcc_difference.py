import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# Set up Arial font
font_path = "/usr/share/fonts/arial/ARIAL.TTF"
try:
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
except:
    plt.rcParams['font.family'] = 'Arial'

plt.style.use('seaborn-v0_8-talk')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'figure.dpi': 300,
    'figure.figsize': (14, 6),
})
mpl.rcParams['svg.fonttype'] = 'none'

print("Loading climatology and calculating weights...")
clim_path = 'E:/data/climate_variables_2000_2023_weekly.nc'
ds_clim = xr.open_dataset(clim_path)
week_of_year = ds_clim['time'].dt.isocalendar().week
tp_clim = ds_clim['tp'].groupby(week_of_year).mean('time')

# Create latitude weights
lat_rad = np.deg2rad(tp_clim.lat)
weights_lat = np.cos(lat_rad)
weights_lat = weights_lat / weights_lat.sum()

print("\nLoading Lead 1 data...")
ds = xr.open_dataset('Z:/Data/hindcast_2022_2023_lead1.nc')
tp_forecast = ds['total_precipitation']
tp_obs = ds['total_precipitation_obs']

# Rename coordinates
if 'latitude' in tp_forecast.dims:
    tp_forecast = tp_forecast.rename({'latitude': 'lat', 'longitude': 'lon'})
    tp_obs = tp_obs.rename({'latitude': 'lat', 'longitude': 'lon'})

time_coords = ds['time']
weeks = time_coords.dt.isocalendar().week

print("Calculating anomalies...")
tp_forecast_anom_list = []
tp_obs_anom_list = []

for t in range(len(time_coords)):
    week_num = int(weeks[t].values)
    if week_num > 52:
        week_num = 52

    clim_week = tp_clim.sel(week=week_num)
    tp_forecast_anom_list.append(tp_forecast[t] - clim_week)
    tp_obs_anom_list.append(tp_obs[t] - clim_week)

tp_forecast_anom = xr.concat(tp_forecast_anom_list, dim='time')
tp_obs_anom = xr.concat(tp_obs_anom_list, dim='time')

# Method 1: Gridpoint TCC (spatial mean of temporal correlations)
print("\nMethod 1: Calculating gridpoint-wise temporal correlation...")
n_time = len(time_coords)
n_lat = len(tp_forecast_anom.lat)
n_lon = len(tp_forecast_anom.lon)

forecast_flat = tp_forecast_anom.values.reshape(n_time, -1)
obs_flat = tp_obs_anom.values.reshape(n_time, -1)

# Calculate correlation for each grid point
tcc_grid = np.zeros(forecast_flat.shape[1])
for i in range(forecast_flat.shape[1]):
    if np.std(forecast_flat[:, i]) > 1e-10 and np.std(obs_flat[:, i]) > 1e-10:
        tcc_grid[i] = np.corrcoef(forecast_flat[:, i], obs_flat[:, i])[0, 1]
    else:
        tcc_grid[i] = 0

# Area-weighted mean
weights_2d = weights_lat.values[:, np.newaxis] * np.ones((n_lat, n_lon)) / n_lon
weights_flat = weights_2d.flatten()
valid_mask = ~np.isnan(tcc_grid)
gridpoint_tcc = np.average(tcc_grid[valid_mask], weights=weights_flat[valid_mask])

print(f"Method 1 TCC (gridpoint): {gridpoint_tcc:.4f}")

# Method 2: Time series TCC (correlation of spatial means)
print("\nMethod 2: Calculating time series correlation of spatial means...")
tp_forecast_anom_lon_mean = tp_forecast_anom.mean(dim='lon')
tp_obs_anom_lon_mean = tp_obs_anom.mean(dim='lon')

tp_forecast_anom_weighted = (tp_forecast_anom_lon_mean * weights_lat).sum(dim='lat')
tp_obs_anom_weighted = (tp_obs_anom_lon_mean * weights_lat).sum(dim='lat')

timeseries_tcc = np.corrcoef(tp_forecast_anom_weighted.values,
                              tp_obs_anom_weighted.values)[0, 1]

print(f"Method 2 TCC (time series): {timeseries_tcc:.4f}")

# Visualization
print("\nCreating visualization...")
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: TCC map (Method 1)
ax1 = axes[0]
tcc_map = tcc_grid.reshape(n_lat, n_lon)
im1 = ax1.imshow(tcc_map, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
ax1.set_title('Method 1: Gridpoint TCC Map\n(Temporal correlation at each point)',
              fontweight='bold')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')
plt.colorbar(im1, ax=ax1, label='TCC')
ax1.text(0.02, 0.98, f'Mean TCC = {gridpoint_tcc:.4f}',
         transform=ax1.transAxes, va='top', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 2: Time series (Method 2)
ax2 = axes[1]
time_axis = np.arange(len(time_coords))
ax2.plot(time_axis, tp_obs_anom_weighted.values, 'r-', linewidth=2,
         label='Observation', alpha=0.7)
ax2.plot(time_axis, tp_forecast_anom_weighted.values, 'b-', linewidth=1.5,
         label='CAS-Canglong', alpha=0.7)
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.4)
ax2.set_title('Method 2: Global Mean Time Series\n(Spatial mean then temporal correlation)',
              fontweight='bold')
ax2.set_xlabel('Time Index')
ax2.set_ylabel('Precipitation Anomaly (mm/day)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.text(0.02, 0.98, f'TCC = {timeseries_tcc:.4f}',
         transform=ax2.transAxes, va='top', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Plot 3: Scatter plot showing why they differ
ax3 = axes[2]
ax3.scatter(tp_obs_anom_weighted.values, tp_forecast_anom_weighted.values,
            alpha=0.6, s=50, c=time_axis, cmap='viridis')
ax3.plot([-0.05, 0.05], [-0.05, 0.05], 'k--', linewidth=2, alpha=0.5,
         label='Perfect correlation')

# Fit line
z = np.polyfit(tp_obs_anom_weighted.values, tp_forecast_anom_weighted.values, 1)
p = np.poly1d(z)
x_line = np.linspace(tp_obs_anom_weighted.values.min(),
                     tp_obs_anom_weighted.values.max(), 100)
ax3.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.7,
         label=f'Fitted line (slope={z[0]:.2f})')

ax3.set_title('Method 2: Scatter Plot\n(Shows negative correlation)', fontweight='bold')
ax3.set_xlabel('Observed Anomaly (mm/day)')
ax3.set_ylabel('Forecast Anomaly (mm/day)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
ax3.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('figures/tcc_explanation_lead1.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/tcc_explanation_lead1.svg', bbox_inches='tight')
print("\nFigure saved to: figures/tcc_explanation_lead1.png")
plt.show()

# Additional analysis
print("\n" + "="*70)
print("DETAILED COMPARISON")
print("="*70)
print(f"\nMethod 1 (Gridpoint TCC):")
print(f"  - Calculation: For each grid point, calculate correlation over time")
print(f"  - Then: Take area-weighted spatial mean of these correlations")
print(f"  - Result: {gridpoint_tcc:.4f} (POSITIVE)")
print(f"  - Interpretation: Most grid points show positive temporal correlation")

print(f"\nMethod 2 (Time Series TCC):")
print(f"  - Calculation: First calculate area-weighted spatial mean at each time")
print(f"  - Then: Calculate correlation of the two global-mean time series")
print(f"  - Result: {timeseries_tcc:.4f} (NEGATIVE)")
print(f"  - Interpretation: Global mean forecast and obs are out of phase")

print(f"\nWhy are they different?")
print(f"  - Gridpoint TCC: Captures local skill (many regions predict well)")
print(f"  - Time series TCC: Captures global mean skill (phase/magnitude errors)")
print(f"  - Spatial averaging can cancel out errors or amplify them")
print(f"  - Negative time series TCC suggests systematic phase/magnitude bias")

print("="*70)

ds.close()
ds_clim.close()
