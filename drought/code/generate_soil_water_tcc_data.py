"""
Generate soil water TCC analysis data for a specific lead time
Usage: python generate_soil_water_tcc_data.py --lead 1
"""
import xarray as xr
import numpy as np
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

def generate_soil_water_tcc_data(lead_time):
    """
    Generate soil water TCC data for a specific lead time

    Parameters:
    -----------
    lead_time : int
        Lead time (1-6)

    Returns:
    --------
    dict : Contains all data needed for plotting
    """
    print(f"="*70)
    print(f"Generating soil water TCC data for Lead {lead_time}")
    print(f"="*70)

    print(f"\n1. Loading Lead {lead_time} hindcast data...")
    hindcast_path = f'Z:/Data/hindcast_2022_2023_lead{lead_time}.nc'
    ds = xr.open_dataset(hindcast_path)
    
    # Extract soil water forecast and observation
    sw_forecast = ds['volumetric_soil_water_layer']
    sw_obs = ds['volumetric_soil_water_layer_obs']

    # Rename coordinates to match standard format
    if 'latitude' in sw_forecast.dims:
        sw_forecast = sw_forecast.rename({'latitude': 'lat', 'longitude': 'lon'})
        sw_obs = sw_obs.rename({'latitude': 'lat', 'longitude': 'lon'})

    time_coords = ds['time']
    
    print(f"   Data shape: {sw_forecast.shape}")
    print(f"   Time range: {time_coords.values[0]} to {time_coords.values[-1]}")

    print(f"\n2. Creating soil water climatology from 2022-2023 data...")
    # Since soil water climatology is not in the climate file, we use 2022-2023 as reference
    # Calculate weekly climatology from the observation data
    
    # Get week of year for each timestep
    weeks = time_coords.dt.isocalendar().week
    
    # Group observation data by week and calculate mean for each week
    sw_clim = sw_obs.groupby(weeks).mean('time')
    
    print(f"   Soil water climatology shape: {sw_clim.shape}")
    print(f"   Week range: {sw_clim.week.min().values} to {sw_clim.week.max().values}")

    # Create latitude weights for area-weighted calculations
    lat_rad = np.deg2rad(sw_forecast.lat)
    weights_lat = np.cos(lat_rad)
    weights_lat = weights_lat / weights_lat.sum()

    print(f"   Latitude weights sum: {weights_lat.sum().values:.6f}")

    print(f"\n3. Calculating soil water anomalies...")
    sw_forecast_anom_list = []
    sw_obs_anom_list = []

    # Calculate anomalies by subtracting climatology for each week
    for t in range(len(time_coords)):
        week_num = int(weeks[t].values)

        # Handle week 53 (rare case) - use week 52 climatology
        if week_num > 52:
            week_num = 52

        # Subtract climatology
        clim_week = sw_clim.sel(week=week_num)
        sw_forecast_anom_list.append(sw_forecast[t] - clim_week)
        sw_obs_anom_list.append(sw_obs[t] - clim_week)

    # Concatenate anomalies
    sw_forecast_anom = xr.concat(sw_forecast_anom_list, dim='time')
    sw_obs_anom = xr.concat(sw_obs_anom_list, dim='time')

    print(f"   Anomaly shape: {sw_forecast_anom.shape}")

    # Method 1: Gridpoint TCC (spatial mean of temporal correlations)
    print(f"\n4. Method 1: Calculating gridpoint-wise temporal correlation...")
    n_time = len(time_coords)
    n_lat = len(sw_forecast_anom.lat)
    n_lon = len(sw_forecast_anom.lon)

    forecast_flat = sw_forecast_anom.values.reshape(n_time, -1)
    obs_flat = sw_obs_anom.values.reshape(n_time, -1)

    # Calculate correlation for each grid point
    tcc_grid = np.zeros(forecast_flat.shape[1])
    for i in range(forecast_flat.shape[1]):
        # Only calculate if both forecast and obs have non-zero variance
        if np.std(forecast_flat[:, i]) > 1e-10 and np.std(obs_flat[:, i]) > 1e-10:
            tcc_grid[i] = np.corrcoef(forecast_flat[:, i], obs_flat[:, i])[0, 1]
        else:
            tcc_grid[i] = 0

    # Create ocean mask using land-sea mask (soil water should only exist over land)
    # For soil water, we need to mask out ocean points
    print(f"\n4a. Creating land mask for soil water (masking oceans)...")
    
    # Create a simple land mask: assume soil water > 0 indicates land
    # This is a proxy since actual soil water data should only exist over land
    mean_soil_water = np.mean(sw_obs.values, axis=0)  # Average over time
    land_mask = mean_soil_water > 0.01  # Threshold to identify land points
    
    # Apply land mask to TCC grid (set ocean points to NaN)
    tcc_grid_masked = tcc_grid.copy()
    tcc_grid_masked[~land_mask.flatten()] = np.nan
    
    print(f"   Land points: {np.sum(land_mask)} out of {land_mask.size}")
    print(f"   Ocean points masked: {np.sum(~land_mask)}")

    # Area-weighted mean (only over land points)
    weights_2d = weights_lat.values[:, np.newaxis] * np.ones((n_lat, n_lon)) / n_lon
    weights_2d_masked = weights_2d * land_mask  # Apply land mask to weights
    weights_flat = weights_2d_masked.flatten()
    valid_mask = ~np.isnan(tcc_grid_masked)
    
    if np.sum(valid_mask) > 0:
        gridpoint_tcc = np.average(tcc_grid_masked[valid_mask], weights=weights_flat[valid_mask])
    else:
        gridpoint_tcc = np.nan

    print(f"   Gridpoint TCC: {gridpoint_tcc:.4f}")

    # Method 2: Time series TCC (correlation of spatial means, over land only)
    print(f"\n5. Method 2: Calculating time series correlation of spatial means (over land only)...")
    
    # Apply land mask to anomalies
    sw_forecast_anom_masked = sw_forecast_anom.where(land_mask)
    sw_obs_anom_masked = sw_obs_anom.where(land_mask)
    
    sw_forecast_anom_lon_mean = sw_forecast_anom_masked.mean(dim='lon')
    sw_obs_anom_lon_mean = sw_obs_anom_masked.mean(dim='lon')

    sw_forecast_anom_weighted = (sw_forecast_anom_lon_mean * weights_lat).sum(dim='lat')
    sw_obs_anom_weighted = (sw_obs_anom_lon_mean * weights_lat).sum(dim='lat')

    timeseries_tcc = np.corrcoef(sw_forecast_anom_weighted.values,
                                  sw_obs_anom_weighted.values)[0, 1]

    print(f"   Time series TCC: {timeseries_tcc:.4f}")

    # Prepare output data
    print(f"\n6. Preparing output data...")
    output_data = {
        'lead_time': lead_time,
        'gridpoint_tcc': gridpoint_tcc,
        'timeseries_tcc': timeseries_tcc,
        'tcc_map': tcc_grid_masked.reshape(n_lat, n_lon),  # Use masked TCC map
        'tcc_map_original': tcc_grid.reshape(n_lat, n_lon),  # Also save original for reference
        'land_mask': land_mask,
        'lat': sw_forecast_anom.lat.values,
        'lon': sw_forecast_anom.lon.values,
        'time_index': np.arange(len(time_coords)),
        'time_coords': time_coords.values,
        'obs_timeseries': sw_obs_anom_weighted.values,
        'forecast_timeseries': sw_forecast_anom_weighted.values,
        'n_lat': n_lat,
        'n_lon': n_lon,
        'n_time': n_time
    }

    # Save data
    output_path = f'figures/soil_water_tcc_data_lead{lead_time}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n7. Data saved to: {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("SOIL WATER TCC SUMMARY")
    print("="*70)
    print(f"Lead {lead_time} week:")
    print(f"  Method 1 (Gridpoint TCC):    {gridpoint_tcc:.4f}")
    print(f"  Method 2 (Time Series TCC):  {timeseries_tcc:.4f}")
    print(f"  Data dimensions: {n_time} times x {n_lat} lats x {n_lon} lons")
    print("="*70)

    ds.close()

    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate soil water TCC analysis data')
    parser.add_argument('--lead', type=int, required=True, choices=[1,2,3,4,5,6],
                        help='Lead time (1-6)')

    args = parser.parse_args()

    generate_soil_water_tcc_data(args.lead)

    print("\n[OK] Soil water TCC data generation completed successfully!")
    print(f"  Next step: python plot_soil_water_tcc_map.py --lead {args.lead}")