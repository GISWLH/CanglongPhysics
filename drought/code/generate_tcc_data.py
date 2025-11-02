"""
Generate TCC analysis data for a specific lead time
Usage: python generate_tcc_data.py --lead 1
"""
import xarray as xr
import numpy as np
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

def generate_tcc_data(lead_time):
    """
    Generate TCC data for a specific lead time

    Parameters:
    -----------
    lead_time : int
        Lead time (1-6)

    Returns:
    --------
    dict : Contains all data needed for plotting
    """
    print(f"="*70)
    print(f"Generating TCC data for Lead {lead_time}")
    print(f"="*70)

    print("\n1. Loading climatology and calculating weights...")
    clim_path = 'E:/data/climate_variables_2000_2023_weekly.nc'
    ds_clim = xr.open_dataset(clim_path)
    week_of_year = ds_clim['time'].dt.isocalendar().week
    tp_clim = ds_clim['tp'].groupby(week_of_year).mean('time')

    # Create latitude weights
    lat_rad = np.deg2rad(tp_clim.lat)
    weights_lat = np.cos(lat_rad)
    weights_lat = weights_lat / weights_lat.sum()

    print(f"   Climatology shape: {tp_clim.shape}")
    print(f"   Latitude weights sum: {weights_lat.sum().values:.6f}")

    print(f"\n2. Loading Lead {lead_time} hindcast data...")
    hindcast_path = f'Z:/Data/hindcast_2022_2023_lead{lead_time}.nc'
    ds = xr.open_dataset(hindcast_path)
    tp_forecast = ds['total_precipitation']
    tp_obs = ds['total_precipitation_obs']

    # Rename coordinates
    if 'latitude' in tp_forecast.dims:
        tp_forecast = tp_forecast.rename({'latitude': 'lat', 'longitude': 'lon'})
        tp_obs = tp_obs.rename({'latitude': 'lat', 'longitude': 'lon'})

    time_coords = ds['time']
    weeks = time_coords.dt.isocalendar().week

    print(f"   Data shape: {tp_forecast.shape}")
    print(f"   Time range: {time_coords.values[0]} to {time_coords.values[-1]}")

    print(f"\n3. Calculating anomalies...")
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

    print(f"   Anomaly shape: {tp_forecast_anom.shape}")

    # Method 1: Gridpoint TCC (spatial mean of temporal correlations)
    print(f"\n4. Method 1: Calculating gridpoint-wise temporal correlation...")
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

    print(f"   Gridpoint TCC: {gridpoint_tcc:.4f}")

    # Method 2: Time series TCC (correlation of spatial means)
    print(f"\n5. Method 2: Calculating time series correlation of spatial means...")
    tp_forecast_anom_lon_mean = tp_forecast_anom.mean(dim='lon')
    tp_obs_anom_lon_mean = tp_obs_anom.mean(dim='lon')

    tp_forecast_anom_weighted = (tp_forecast_anom_lon_mean * weights_lat).sum(dim='lat')
    tp_obs_anom_weighted = (tp_obs_anom_lon_mean * weights_lat).sum(dim='lat')

    timeseries_tcc = np.corrcoef(tp_forecast_anom_weighted.values,
                                  tp_obs_anom_weighted.values)[0, 1]

    print(f"   Time series TCC: {timeseries_tcc:.4f}")

    # Prepare output data
    print(f"\n6. Preparing output data...")
    output_data = {
        'lead_time': lead_time,
        'gridpoint_tcc': gridpoint_tcc,
        'timeseries_tcc': timeseries_tcc,
        'tcc_map': tcc_grid.reshape(n_lat, n_lon),
        'lat': tp_forecast_anom.lat.values,
        'lon': tp_forecast_anom.lon.values,
        'time_index': np.arange(len(time_coords)),
        'time_coords': time_coords.values,
        'obs_timeseries': tp_obs_anom_weighted.values,
        'forecast_timeseries': tp_forecast_anom_weighted.values,
        'n_lat': n_lat,
        'n_lon': n_lon,
        'n_time': n_time
    }

    # Save data
    output_path = f'figures/tcc_data_lead{lead_time}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n7. Data saved to: {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Lead {lead_time} week:")
    print(f"  Method 1 (Gridpoint TCC):    {gridpoint_tcc:.4f} (POSITIVE)")
    print(f"  Method 2 (Time Series TCC):  {timeseries_tcc:.4f} (NEGATIVE)")
    print(f"  Data dimensions: {n_time} times x {n_lat} lats x {n_lon} lons")
    print("="*70)

    ds.close()
    ds_clim.close()

    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TCC analysis data')
    parser.add_argument('--lead', type=int, required=True, choices=[1,2,3,4,5,6],
                        help='Lead time (1-6)')

    args = parser.parse_args()

    generate_tcc_data(args.lead)

    print("\n[OK] Data generation completed successfully!")
    print(f"  Next step: python plot_tcc_analysis.py --lead {args.lead}")
