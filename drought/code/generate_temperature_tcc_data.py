"""
Generate temperature TCC analysis data for a specific lead time
Usage: python generate_temperature_tcc_data.py --lead 1
"""
import xarray as xr
import numpy as np
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

def generate_temperature_tcc_data(lead_time):
    """
    Generate temperature TCC data for a specific lead time

    Parameters:
    -----------
    lead_time : int
        Lead time (1-6)

    Returns:
    --------
    dict : Contains all data needed for plotting
    """
    print(f"="*70)
    print(f"Generating temperature TCC data for Lead {lead_time}")
    print(f"="*70)

    print("\n1. Loading climatology and calculating weights...")
    clim_path = 'E:/data/climate_variables_2000_2023_weekly.nc'
    ds_clim = xr.open_dataset(clim_path)
    week_of_year = ds_clim['time'].dt.isocalendar().week
    t2m_clim = ds_clim['t2m'].groupby(week_of_year).mean('time')

    # Create latitude weights
    lat_rad = np.deg2rad(t2m_clim.lat)
    weights_lat = np.cos(lat_rad)
    weights_lat = weights_lat / weights_lat.sum()

    print(f"   Climatology shape: {t2m_clim.shape}")
    print(f"   Latitude weights sum: {weights_lat.sum().values:.6f}")

    print(f"\n2. Loading Lead {lead_time} hindcast data...")
    hindcast_path = f'Z:/Data/hindcast_2022_2023_lead{lead_time}.nc'
    ds = xr.open_dataset(hindcast_path)
    t2m_forecast = ds['2m_temperature']
    t2m_obs = ds['2m_temperature_obs']

    # Rename coordinates
    if 'latitude' in t2m_forecast.dims:
        t2m_forecast = t2m_forecast.rename({'latitude': 'lat', 'longitude': 'lon'})
        t2m_obs = t2m_obs.rename({'latitude': 'lat', 'longitude': 'lon'})

    time_coords = ds['time']
    weeks = time_coords.dt.isocalendar().week

    print(f"   Data shape: {t2m_forecast.shape}")
    print(f"   Time range: {time_coords.values[0]} to {time_coords.values[-1]}")

    print(f"\n3. Calculating temperature anomalies...")
    t2m_forecast_anom_list = []
    t2m_obs_anom_list = []

    for t in range(len(time_coords)):
        week_num = int(weeks[t].values)
        if week_num > 52:
            week_num = 52

        clim_week = t2m_clim.sel(week=week_num)
        t2m_forecast_anom_list.append(t2m_forecast[t] - clim_week)
        t2m_obs_anom_list.append(t2m_obs[t] - clim_week)

    t2m_forecast_anom = xr.concat(t2m_forecast_anom_list, dim='time')
    t2m_obs_anom = xr.concat(t2m_obs_anom_list, dim='time')

    print(f"   Anomaly shape: {t2m_forecast_anom.shape}")

    # Method 1: Gridpoint TCC (spatial mean of temporal correlations)
    print(f"\n4. Method 1: Calculating gridpoint-wise temporal correlation...")
    n_time = len(time_coords)
    n_lat = len(t2m_forecast_anom.lat)
    n_lon = len(t2m_forecast_anom.lon)

    forecast_flat = t2m_forecast_anom.values.reshape(n_time, -1)
    obs_flat = t2m_obs_anom.values.reshape(n_time, -1)

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
    t2m_forecast_anom_lon_mean = t2m_forecast_anom.mean(dim='lon')
    t2m_obs_anom_lon_mean = t2m_obs_anom.mean(dim='lon')

    t2m_forecast_anom_weighted = (t2m_forecast_anom_lon_mean * weights_lat).sum(dim='lat')
    t2m_obs_anom_weighted = (t2m_obs_anom_lon_mean * weights_lat).sum(dim='lat')

    timeseries_tcc = np.corrcoef(t2m_forecast_anom_weighted.values,
                                  t2m_obs_anom_weighted.values)[0, 1]

    print(f"   Time series TCC: {timeseries_tcc:.4f}")

    # Prepare output data
    print(f"\n6. Preparing output data...")
    output_data = {
        'lead_time': lead_time,
        'gridpoint_tcc': gridpoint_tcc,
        'timeseries_tcc': timeseries_tcc,
        'tcc_map': tcc_grid.reshape(n_lat, n_lon),
        'lat': t2m_forecast_anom.lat.values,
        'lon': t2m_forecast_anom.lon.values,
        'time_index': np.arange(len(time_coords)),
        'time_coords': time_coords.values,
        'obs_timeseries': t2m_obs_anom_weighted.values,
        'forecast_timeseries': t2m_forecast_anom_weighted.values,
        'n_lat': n_lat,
        'n_lon': n_lon,
        'n_time': n_time
    }

    # Save data
    output_path = f'figures/temperature_tcc_data_lead{lead_time}.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)

    print(f"\n7. Data saved to: {output_path}")

    # Print summary
    print("\n" + "="*70)
    print("TEMPERATURE TCC SUMMARY")
    print("="*70)
    print(f"Lead {lead_time} week:")
    print(f"  Method 1 (Gridpoint TCC):    {gridpoint_tcc:.4f}")
    print(f"  Method 2 (Time Series TCC):  {timeseries_tcc:.4f}")
    print(f"  Data dimensions: {n_time} times x {n_lat} lats x {n_lon} lons")
    print("="*70)

    ds.close()
    ds_clim.close()

    return output_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate temperature TCC analysis data')
    parser.add_argument('--lead', type=int, required=True, choices=[1,2,3,4,5,6],
                        help='Lead time (1-6)')

    args = parser.parse_args()

    generate_temperature_tcc_data(args.lead)

    print("\n[OK] Temperature TCC data generation completed successfully!")
    print(f"  Next step: python plot_temperature_tcc_map.py --lead {args.lead}")