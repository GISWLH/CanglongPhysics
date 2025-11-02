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
    'legend.fontsize': 8,
    'figure.dpi': 600,
    'figure.figsize': (12, 8),
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
# Load climatology data
clim_path = 'E:/data/climate_variables_2000_2023_weekly.nc'
ds_clim = xr.open_dataset(clim_path)

# Calculate week-of-year for each timestep as a DataArray
week_of_year = ds_clim['time'].dt.isocalendar().week

# Calculate climatology: mean for each week across all years
print("Calculating weekly climatology (mean over 2000-2023)...")
tp_clim = ds_clim['tp'].groupby(week_of_year).mean('time')

print(f"Climatology shape: {tp_clim.shape}")

# Initialize dictionary to store anomalies for each lead time
lead_times = range(1, 7)
anomalies = {}

print("\nCalculating precipitation anomalies for each lead time...\n")

for lead in tqdm(lead_times, desc="Processing lead times"):
    # Load hindcast data
    hindcast_path = f'Z:/Data/hindcast_2022_2023_lead{lead}.nc'
    ds = xr.open_dataset(hindcast_path)

    # Extract precipitation forecast and observation
    tp_forecast = ds['total_precipitation']
    tp_obs = ds['total_precipitation_obs']

    # Rename coordinates to match climatology
    if 'latitude' in tp_forecast.dims:
        tp_forecast = tp_forecast.rename({'latitude': 'lat', 'longitude': 'lon'})
        tp_obs = tp_obs.rename({'latitude': 'lat', 'longitude': 'lon'})

    # Get time coordinates
    time_coords = ds['time']
    weeks = time_coords.dt.isocalendar().week

    # Initialize lists for anomalies
    tp_forecast_anom_list = []
    tp_obs_anom_list = []

    # Calculate anomalies by subtracting climatology for each week
    for t in range(len(time_coords)):
        week_num = int(weeks[t].values)

        # Handle week 53
        if week_num > 52:
            week_num = 52

        # Subtract climatology
        clim_week = tp_clim.sel(week=week_num)
        tp_forecast_anom_list.append(tp_forecast[t] - clim_week)
        tp_obs_anom_list.append(tp_obs[t] - clim_week)

    # Concatenate anomalies
    tp_forecast_anom = xr.concat(tp_forecast_anom_list, dim='time')
    tp_obs_anom = xr.concat(tp_obs_anom_list, dim='time')

    # Calculate spatial mean (global average)
    tp_forecast_anom_mean = tp_forecast_anom.mean(dim=['lat', 'lon'])
    tp_obs_anom_mean = tp_obs_anom.mean(dim=['lat', 'lon'])

    # Store results
    anomalies[f'lead{lead}'] = {
        'time': time_coords.values,
        'forecast': tp_forecast_anom_mean.values,
        'obs': tp_obs_anom_mean.values
    }

    ds.close()

ds_clim.close()

print("\nPlotting time series for all lead times...\n")

# Create subplots - 3 rows, 2 columns
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
axes = axes.flatten()

# Color palette
colors = {
    'forecast': '#2E86AB',
    'obs': '#E63946'
}

# Plot each lead time
for idx, lead in enumerate(lead_times):
    ax = axes[idx]

    data = anomalies[f'lead{lead}']
    time = data['time']
    forecast = data['forecast']
    obs = data['obs']

    # Plot lines
    ax.plot(time, obs, color=colors['obs'], linewidth=2.0,
            label='Observation', alpha=0.8)
    ax.plot(time, forecast, color=colors['forecast'], linewidth=1.5,
            label='CAS-Canglong', alpha=0.8, linestyle='-')

    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.4)

    # Calculate correlation
    correlation = np.corrcoef(forecast, obs)[0, 1]

    # Labels and title
    ax.set_title(f'Lead {lead} Week (TCC = {correlation:.3f})',
                 fontsize=11, fontweight='bold', pad=10)
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Precipitation Anomaly (mm/day)', fontsize=10)

    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Legend
    ax.legend(loc='best', frameon=True, fancybox=False,
              edgecolor='#454545', framealpha=1)

    # Format x-axis
    ax.tick_params(axis='x', rotation=45)

    # Minor ticks
    ax.minorticks_on()

# Overall title
fig.suptitle('Precipitation Anomaly Time Series: CAS-Canglong vs Observations (2022-2023)',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout()

# Save figure
output_path = 'figures/precipitation_anomaly_timeseries.png'
plt.savefig(output_path, dpi=600, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

output_path_svg = 'figures/precipitation_anomaly_timeseries.svg'
plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
print(f"Figure saved to: {output_path_svg}")

plt.show()

# Print statistics
print("\n" + "="*70)
print("PRECIPITATION ANOMALY TIME SERIES STATISTICS")
print("="*70)
for lead in lead_times:
    data = anomalies[f'lead{lead}']
    forecast = data['forecast']
    obs = data['obs']

    tcc = np.corrcoef(forecast, obs)[0, 1]
    rmse = np.sqrt(np.mean((forecast - obs)**2))
    mae = np.mean(np.abs(forecast - obs))

    print(f"\nLead {lead} week:")
    print(f"  TCC:  {tcc:.4f}")
    print(f"  RMSE: {rmse:.4f} mm/day")
    print(f"  MAE:  {mae:.4f} mm/day")
    print(f"  Obs mean:      {np.mean(obs):.4f} mm/day")
    print(f"  Obs std:       {np.std(obs):.4f} mm/day")
    print(f"  Forecast mean: {np.mean(forecast):.4f} mm/day")
    print(f"  Forecast std:  {np.std(forecast):.4f} mm/day")

print("="*70)
