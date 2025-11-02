import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set up Arial font for plotting
font_path = "/usr/share/fonts/arial/ARIAL.TTF"
try:
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
except:
    plt.rcParams['font.family'] = 'Arial'

# Set Nature style parameters
plt.style.use('seaborn-v0_8-talk')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'figure.figsize': (6, 4),
    'lines.linewidth': 1.5,
    'axes.linewidth': 1.0,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.edgecolor': '#454545',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'xtick.color': '#454545',
    'ytick.color': '#454545',
    'savefig.bbox': 'tight',
    'savefig.transparent': False
})
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['svg.hashsalt'] = 'hello'

print("Loading climatology data...")
# Load climatology data (24 years * 52 weeks = 1248 timesteps)
clim_path = 'E:/data/climate_variables_2000_2023_weekly.nc'
ds_clim = xr.open_dataset(clim_path)

# Calculate week-of-year for each timestep as a DataArray
week_of_year = ds_clim['time'].dt.isocalendar().week

# Calculate climatology: mean for each week across all years
print("Calculating weekly climatology (mean over 2000-2023)...")
tp_clim = ds_clim['tp'].groupby(week_of_year).mean('time')

print(f"Climatology shape: {tp_clim.shape}")
print(f"Week range: {tp_clim.week.min().values} to {tp_clim.week.max().values}")

# Create latitude weights (cosine of latitude)
# Latitude is in degrees, need to convert to radians for cos
lat_rad = np.deg2rad(tp_clim.lat)
weights = np.cos(lat_rad)

# Normalize weights so they sum to 1 for each longitude
weights = weights / weights.sum()

print(f"\nLatitude weights shape: {weights.shape}")
print(f"Weights range: {weights.min().values:.6f} to {weights.max().values:.6f}")
print(f"Weights sum: {weights.sum().values:.6f}")

# Initialize arrays to store TCC for each lead time
lead_times = range(1, 7)
tcc_values = []

print("\nCalculating area-weighted precipitation anomalies and TCC for each lead time...\n")

for lead in tqdm(lead_times, desc="Processing lead times"):
    # Load hindcast data
    hindcast_path = f'Z:/Data/hindcast_2022_2023_lead{lead}.nc'
    ds = xr.open_dataset(hindcast_path)

    # Extract precipitation forecast and observation
    tp_forecast = ds['total_precipitation']
    tp_obs = ds['total_precipitation_obs']

    # Rename coordinates to match climatology (latitude/longitude -> lat/lon)
    if 'latitude' in tp_forecast.dims:
        tp_forecast = tp_forecast.rename({'latitude': 'lat', 'longitude': 'lon'})
        tp_obs = tp_obs.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Get time coordinates and calculate week of year for each time step
    time_coords = ds['time']
    weeks = time_coords.dt.isocalendar().week

    # Initialize list for anomalies
    tp_forecast_anom_list = []
    tp_obs_anom_list = []

    # Calculate anomalies by subtracting climatology for each week
    for t in range(len(time_coords)):
        week_num = int(weeks[t].values)

        # Handle week 53 (rare case) - use week 52 climatology
        if week_num > 52:
            week_num = 52

        # Subtract climatology
        clim_week = tp_clim.sel(week=week_num)
        tp_forecast_anom_list.append(tp_forecast[t] - clim_week)
        tp_obs_anom_list.append(tp_obs[t] - clim_week)

    # Concatenate anomalies
    tp_forecast_anom = xr.concat(tp_forecast_anom_list, dim='time')
    tp_obs_anom = xr.concat(tp_obs_anom_list, dim='time')

    # Calculate temporal correlation coefficient (TCC) with area weighting
    # For each grid point, calculate weighted correlation over time

    # Flatten spatial dimensions
    n_time = len(time_coords)
    n_lat = len(tp_forecast_anom.lat)
    n_lon = len(tp_forecast_anom.lon)

    forecast_flat = tp_forecast_anom.values.reshape(n_time, -1)  # (time, lat*lon)
    obs_flat = tp_obs_anom.values.reshape(n_time, -1)

    # Create 2D weights array (lat, lon) and flatten
    weights_2d = weights.values[:, np.newaxis] * np.ones((n_lat, n_lon)) / n_lon
    weights_flat = weights_2d.flatten()

    # Calculate correlation for each grid point
    tcc_grid = np.zeros(forecast_flat.shape[1])
    for i in range(forecast_flat.shape[1]):
        # Only calculate if both forecast and obs have non-zero variance
        if np.std(forecast_flat[:, i]) > 1e-10 and np.std(obs_flat[:, i]) > 1e-10:
            tcc_grid[i] = np.corrcoef(forecast_flat[:, i], obs_flat[:, i])[0, 1]
        else:
            tcc_grid[i] = 0

    # Calculate area-weighted mean TCC
    # Filter out NaN values
    valid_mask = ~np.isnan(tcc_grid)
    if valid_mask.sum() > 0:
        weighted_tcc = np.average(tcc_grid[valid_mask], weights=weights_flat[valid_mask])
    else:
        weighted_tcc = 0.0

    tcc_values.append(weighted_tcc)

    print(f"Lead {lead}: Area-weighted TCC = {weighted_tcc:.4f}")

    ds.close()

ds_clim.close()

# Plot TCC decay curve
print("\nPlotting area-weighted TCC decay curve...")

fig, ax = plt.subplots(figsize=(7, 5))

# Plot TCC values
ax.plot(lead_times, tcc_values, marker='o', markersize=8,
        linewidth=2, color='#2E86AB', label='CAS-Canglong (Area-weighted)')

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Labels and title
ax.set_xlabel('Lead Time (weeks)', fontsize=11, fontweight='bold')
ax.set_ylabel('Temporal Correlation Coefficient (TCC)', fontsize=11, fontweight='bold')
ax.set_title('Precipitation Anomaly TCC Decay - Area Weighted (2022-2023)',
             fontsize=12, fontweight='bold', pad=15)

# Set x-axis
ax.set_xticks(lead_times)
ax.set_xlim(0.5, 6.5)

# Set y-axis limits
y_min = min(0, min(tcc_values) - 0.1)
y_max = max(tcc_values) + 0.1
ax.set_ylim(y_min, y_max)

# Add horizontal line at y=0
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Add legend
ax.legend(loc='best', frameon=True, fancybox=False,
          edgecolor='#454545', framealpha=1)

# Minor ticks
ax.minorticks_on()

plt.tight_layout()

# Save figure
output_path = 'figures/precipitation_tcc_decay_weighted.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Also save as SVG
output_path_svg = 'figures/precipitation_tcc_decay_weighted.svg'
plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
print(f"Figure saved to: {output_path_svg}")

plt.show()

# Print summary statistics
print("\n" + "="*60)
print("PRECIPITATION TCC SUMMARY (AREA-WEIGHTED)")
print("="*60)
for lead, tcc in zip(lead_times, tcc_values):
    print(f"Lead {lead} week: TCC = {tcc:.4f}")
print("="*60)

# Calculate TCC decay
tcc_decay = tcc_values[0] - tcc_values[-1]
print(f"\nTCC decay (Lead 1 to Lead 6): {tcc_decay:.4f}")
print(f"Average TCC decay per week: {tcc_decay/5:.4f}")

# Compare with non-weighted results
print("\n" + "="*60)
print("COMPARISON: Area-weighted vs Non-weighted TCC")
print("="*60)
non_weighted_tcc = [0.0285, 0.0303, 0.0125, 0.0141, 0.0121, 0.0130]
print(f"{'Lead':<10} {'Weighted':<12} {'Non-weighted':<15} {'Difference':<12}")
print("-"*60)
for lead, tcc_w, tcc_nw in zip(lead_times, tcc_values, non_weighted_tcc):
    diff = tcc_w - tcc_nw
    print(f"{lead:<10} {tcc_w:<12.4f} {tcc_nw:<15.4f} {diff:<12.4f}")
print("="*60)
