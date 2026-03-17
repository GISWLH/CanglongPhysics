"""
Calculate SPEI-4 for ERA5 observation data
Process: I:/ERA5_np/input_surface_norm_test_last100.pt
Start from week 5 (2022 week 9, Feb 26 - Mar 4)
Output to: Z:/Data/hindcast_spei_2022_2023_obs/
"""

import torch
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.special import gamma as gamma_function
from tqdm import tqdm
import os
import json

# ============================================================================
# Helper Functions
# ============================================================================

def get_week_of_year(date):
    """
    Calculate week number of year (1-52)
    Week 1: Jan 1-7, Week 2: Jan 8-14, ..., Week 52: Dec 24-30

    Parameters:
    -----------
    date : datetime or pd.Timestamp or xr.DataArray with datetime
        Date(s) to calculate week number

    Returns:
    --------
    int or xr.DataArray or None : Week number(s), or None for ignored days
    """
    if isinstance(date, xr.DataArray):
        day_of_year = date.dt.dayofyear
        week = ((day_of_year - 1) // 7) + 1
        week = week.where(week <= 52)
        return week
    else:
        day_of_year = date.timetuple().tm_yday
        if day_of_year > 364:
            return None
        return ((day_of_year - 1) // 7) + 1


def calculate_week_number(date):
    """Alias for get_week_of_year for consistency with hindcast_22_23_claude.py"""
    return get_week_of_year(date)


# Variable names for old 16-variable format
surface_var_names = [
    'large_scale_rain_rate',
    'convective_rain_rate',
    'total_column_cloud_ice_water',
    'total_cloud_cover',
    'top_net_solar_radiation_clear_sky',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'surface_latent_heat_flux',
    'surface_sensible_heat_flux',
    'surface_pressure',
    'volumetric_soil_water_layer',
    'mean_sea_level_pressure',
    'sea_ice_cover',
    'sea_surface_temperature'
]

# Variable mapping and statistics
var_mapping = {
    'large_scale_rain_rate': 'lsrr',
    'convective_rain_rate': 'crr',
    'total_column_cloud_ice_water': 'tciw',
    'total_cloud_cover': 'tcc',
    'top_net_solar_radiation_clear_sky': 'tsrc',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    '2m_dewpoint_temperature': 'd2m',
    '2m_temperature': 't2m',
    'surface_latent_heat_flux': 'surface_latent_heat_flux',
    'surface_sensible_heat_flux': 'surface_sensible_heat_flux',
    'surface_pressure': 'sp',
    'volumetric_soil_water_layer': 'swvl',
    'mean_sea_level_pressure': 'msl',
    'sea_ice_cover': 'siconc',
    'sea_surface_temperature': 'sst'
}

ordered_var_stats = {
    'lsrr': {'mean': 1.10E-05, 'std': 2.55E-05},
    'crr': {'mean': 1.29E-05, 'std': 2.97E-05},
    'tciw': {'mean': 0.022627383, 'std': 0.023428712},
    'tcc': {'mean': 0.673692584, 'std': 0.235167906},
    'tsrc': {'mean': 856148, 'std': 534222.125},
    'u10': {'mean': -0.068418466, 'std': 4.427545547},
    'v10': {'mean': 0.197138891, 'std': 3.09530735},
    'd2m': {'mean': 274.2094421, 'std': 20.45770073},
    't2m': {'mean': 278.7841187, 'std': 21.03286934},
    'surface_latent_heat_flux': {'mean': -5410301.5, 'std': 5349063.5},
    'surface_sensible_heat_flux': {'mean': -971651.375, 'std': 2276764.75},
    'sp': {'mean': 96651.14063, 'std': 9569.695313},
    'swvl': {'mean': 0.34216917, 'std': 0.5484813},
    'msl': {'mean': 100972.3438, 'std': 1191.102417},
    'siconc': {'mean': 0.785884917, 'std': 0.914535105},
    'sst': {'mean': 189.7337189, 'std': 136.1803131}
}


def denormalize_surface(normalized_surface):
    """
    Denormalize surface data using ordered_var_stats

    Parameters:
    -----------
    normalized_surface : np.ndarray
        Normalized data with shape (16, N, 721, 1440)

    Returns:
    --------
    np.ndarray : Denormalized data
    """
    normalized_surface = np.asarray(normalized_surface, dtype=np.float32)
    surface_means = np.array([ordered_var_stats[var_mapping[var]]['mean'] for var in surface_var_names], dtype=np.float32)
    surface_stds = np.array([ordered_var_stats[var_mapping[var]]['std'] for var in surface_var_names], dtype=np.float32)
    surface_means = surface_means.reshape(-1, 1, 1, 1)
    surface_stds = surface_stds.reshape(-1, 1, 1, 1)
    return normalized_surface * surface_stds + surface_means


# ============================================================================
# SPEI Calculation Functions
# ============================================================================

def calculate_pwm(series):
    """Calculate Probability Weighted Moments"""
    n = len(series)
    if n < 3:
        return np.nan, np.nan, np.nan

    sorted_series = np.sort(series)
    F_vals = (np.arange(1, n + 1) - 0.35) / n
    one_minus_F = 1.0 - F_vals

    W0 = np.mean(sorted_series)
    W1 = np.sum(sorted_series * one_minus_F) / n
    W2 = np.sum(sorted_series * (one_minus_F**2)) / n

    return W0, W1, W2


def calculate_loglogistic_params(W0, W1, W2):
    """Calculate log-logistic distribution parameters"""
    if np.isnan(W0) or np.isnan(W1) or np.isnan(W2):
        return np.nan, np.nan, np.nan

    numerator_beta = (2 * W1) - W0
    denominator_beta = (6 * W1) - W0 - (6 * W2)

    if np.isclose(denominator_beta, 0):
        return np.nan, np.nan, np.nan
    beta = numerator_beta / denominator_beta

    if beta <= 1.0:
        return np.nan, np.nan, np.nan

    try:
        term_gamma1 = gamma_function(1 + (1 / beta))
        term_gamma2 = gamma_function(1 - (1 / beta))
    except ValueError:
        return np.nan, np.nan, np.nan

    denominator_alpha = term_gamma1 * term_gamma2
    if np.isclose(denominator_alpha, 0):
        return np.nan, np.nan, np.nan

    alpha = ((W0 - (2 * W1)) * beta) / denominator_alpha

    if alpha <= 0:
        return np.nan, np.nan, np.nan

    gamma_param = W0 - (alpha * denominator_alpha)

    return alpha, beta, gamma_param


def loglogistic_cdf(x, alpha, beta, gamma_param):
    """Calculate log-logistic CDF"""
    if np.isnan(alpha) or x <= gamma_param:
        return 1e-9

    term = (alpha / (x - gamma_param))**beta
    if np.isinf(term) or term > 1e18:
        return 1e-9

    cdf_val = 1.0 / (1.0 + term)
    return np.clip(cdf_val, 1e-9, 1.0 - 1e-9)


def cdf_to_spei(P):
    """Convert CDF probability to SPEI value"""
    if np.isnan(P):
        return np.nan
    if P <= 0.0:
        P = 1e-9
    if P >= 1.0:
        P = 1.0 - 1e-9

    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308

    if P <= 0.5:
        w = np.sqrt(-2.0 * np.log(P))
        spei = -(w - (c0 + c1 * w + c2 * w**2) / (1 + d1 * w + d2 * w**2 + d3 * w**3))
    else:
        w = np.sqrt(-2.0 * np.log(1.0 - P))
        spei = (w - (c0 + c1 * w + c2 * w**2) / (1 + d1 * w + d2 * w**2 + d3 * w**3))
    return spei


def calculate_spei_for_pixel(historical_D_series, current_D_value):
    """Calculate SPEI for a single pixel"""
    if np.isscalar(current_D_value):
        if np.isnan(current_D_value):
            return np.nan
    else:
        if np.all(np.isnan(current_D_value)):
            return np.nan

    valid_historical_D = historical_D_series[~np.isnan(historical_D_series)]
    if len(valid_historical_D) < 10:
        return np.nan

    W0, W1, W2 = calculate_pwm(valid_historical_D)
    if np.isnan(W0):
        return np.nan

    alpha, beta, gamma_p = calculate_loglogistic_params(W0, W1, W2)
    if np.isnan(alpha):
        return np.nan

    P = loglogistic_cdf(current_D_value, alpha, beta, gamma_p)
    spei_val = cdf_to_spei(P)

    return spei_val


# ============================================================================
# Main Processing Function
# ============================================================================

def process_observation_spei():
    """
    Process ERA5 observation data and calculate SPEI-4

    Data structure:
    - Input: I:/ERA5_np/input_surface_norm_test_last100.pt
    - Shape: (17, 100, 721, 1440) - normalized
    - Variables (17): 0-1: rain rates, 2-6: other vars, 7: d2m, 8: t2m, 9-16: other vars
    - Time: 100 weeks from 2022 week 1 (Jan 1) to 2023 week 48 (Dec 2-8)

    Output: Weekly NC files matching hindcast format
    - Start from week 5 (index 4, 2022-02-26)
    - Save to Z:/Data/hindcast_spei_2022_2023_obs/
    """

    print("=" * 80)
    print("SPEI-4 Calculation for ERA5 Observation Data")
    print("=" * 80)

    # Load observation data
    print("\nLoading observation data...")
    obs_data_path = '/data/lhwang/input_surface_norm_test_last100.pt'
    surface_norm = torch.load(obs_data_path).numpy()  # (16, 100, 721, 1440)
    print(f"  Loaded data shape: {surface_norm.shape}")

    # Denormalize using ordered_var_stats
    print("\nDenormalizing data using ordered_var_stats...")
    surface_physical = denormalize_surface(surface_norm)
    print(f"  Denormalized data shape: {surface_physical.shape}")

    # Process precipitation (m/hr to mm/day)
    # Variables 0-1: large_scale_rain_rate, convective_rain_rate (m/hr after denormalization)
    print("\nProcessing precipitation...")
    large_scale_rain = surface_physical[0]  # (100, 721, 1440)
    convective_rain = surface_physical[1]   # (100, 721, 1440)

    # Convert from m/hr to mm/day
    m_hr_to_mm_day = 24.0 * 1000.0
    large_scale_rain = np.where(large_scale_rain >= 0, large_scale_rain, 0) * m_hr_to_mm_day
    convective_rain = np.where(convective_rain >= 0, convective_rain, 0) * m_hr_to_mm_day
    total_precip = large_scale_rain + convective_rain

    # Process temperature (K to Celsius)
    print("Processing temperature...")
    t2m_celsius = surface_physical[8] - 273.15  # (100, 721, 1440)
    d2m_celsius = surface_physical[7] - 273.15  # (100, 721, 1440)

    # Calculate PET
    print("Calculating PET...")
    es = 0.618 * np.exp(17.27 * t2m_celsius / (t2m_celsius + 237.3))
    ea = 0.618 * np.exp(17.27 * d2m_celsius / (d2m_celsius + 237.3))

    ratio_ea_es = np.full_like(t2m_celsius, np.nan)
    valid_es_mask = es > 1e-9
    ratio_ea_es[valid_es_mask] = ea[valid_es_mask] / es[valid_es_mask]
    ratio_ea_es = np.clip(ratio_ea_es, None, 1.0)

    pet = 4.5 * np.power((1 + t2m_celsius / 25.0), 2) * (1 - ratio_ea_es)
    pet = np.maximum(pet, 0)

    print(f"  PET range: {np.nanmin(pet):.2f} to {np.nanmax(pet):.2f} mm/day")
    print(f"  Precip range: {np.nanmin(total_precip):.2f} to {np.nanmax(total_precip):.2f} mm/day")

    # Calculate D = P - PET
    D_obs = total_precip - pet  # (100, 721, 1440)

    # Load historical climatology
    print("\nLoading historical climatology...")
    climate_ds = xr.open_dataset('/data/lhwang/climate_variables_2000_2023_weekly.nc')
    print(f"  Loaded {len(climate_ds.time)} weeks (2000-2023)")

    # Prepare historical D for climatology
    D_hist = climate_ds['tp'] - climate_ds['pet']
    hist_week_numbers = get_week_of_year(climate_ds['time'])

    # Setup directories - we will ADD spei_obs to existing hindcast files
    hindcast_dir = '/data/lhwang/hindcast_spei_2022_2023/'
    print(f"\nHindcast directory (will add spei_obs to existing files): {hindcast_dir}")

    # Create coordinate arrays (matching ERA5 grid)
    lat = np.linspace(90, -90, 721)
    lon = np.linspace(0, 359.75, 1440)

    # Define time mapping:
    # IMPORTANT: week_idx=0 corresponds to 2022-01-29 (week 1 of 2022)
    # This matches the data structure in hindcast_22_23_claude.py
    base_date = datetime(2022, 1, 29)

    # Hindcast files start from week09 (2022-02-26), which is week_idx=4
    # Calculation: 2022-01-29 + 4 weeks = 2022-02-26 (week09)
    start_week_idx = 4

    # Quick check: identify which files need processing
    print(f"\nChecking existing files for spei_obs...")
    weeks_to_process = []
    weeks_skipped = []

    for week_idx in range(start_week_idx, surface_norm.shape[1]):
        # Calculate date using base_date + weeks offset
        week_start_date = base_date + timedelta(weeks=week_idx)
        year = week_start_date.year
        week_num = calculate_week_number(week_start_date)

        # Skip if week number is invalid (None means day > 364)
        if week_num is None or week_num > 52:
            continue

        # Find corresponding hindcast file
        date_str = week_start_date.strftime('%Y-%m-%d')
        hindcast_filename = f'hindcast_{year}_week{week_num:02d}_surface_{date_str}.nc'
        hindcast_path = os.path.join(hindcast_dir, hindcast_filename)

        # Check if file exists
        if not os.path.exists(hindcast_path):
            continue

        # Quick check if spei_obs already exists
        try:
            with xr.open_dataset(hindcast_path) as ds:
                if 'spei_obs' in ds.data_vars:
                    weeks_skipped.append(week_idx)
                else:
                    weeks_to_process.append(week_idx)
        except Exception as e:
            print(f"  Warning: Error checking {hindcast_filename}: {e}")
            weeks_to_process.append(week_idx)

    print(f"  Found {len(weeks_to_process)} files to process")
    print(f"  Found {len(weeks_skipped)} files already with spei_obs (will skip)")

    if len(weeks_to_process) == 0:
        print("\n" + "=" * 80)
        print("All files already have spei_obs! Nothing to do.")
        print("=" * 80)
        return

    print(f"\nProcessing {len(weeks_to_process)} weeks...")

    for week_idx in tqdm(weeks_to_process, desc="Processing weeks"):
        # Calculate date using base_date + weeks offset
        week_start_date = base_date + timedelta(weeks=week_idx)
        year = week_start_date.year
        week_num = calculate_week_number(week_start_date)

        # Skip if week number is invalid
        if week_num is None or week_num > 52:
            print(f"  Skipping week_idx {week_idx} (invalid week number)")
            continue

        # Get current week D
        curr_week_D = D_obs[week_idx]  # (721, 1440)

        # Get previous 3 weeks D (for SPEI-4)
        prev_3_weeks_D = []
        for i in range(3, 0, -1):
            prev_idx = week_idx - i
            if prev_idx >= 0:
                prev_3_weeks_D.append(D_obs[prev_idx])
            else:
                # Need to handle case where we don't have enough previous weeks
                # Fill with NaN or skip
                prev_3_weeks_D.append(np.full_like(curr_week_D, np.nan))

        # Accumulate 4 weeks of D
        week_4_accum = prev_3_weeks_D[0] + prev_3_weeks_D[1] + prev_3_weeks_D[2] + curr_week_D

        # Get historical climatology for same week number
        hist_4week_accum_list = []
        hist_years = np.unique(D_hist.time.dt.year)

        for hist_year in hist_years:
            year_data = D_hist.where(D_hist.time.dt.year == hist_year, drop=True)
            year_weeks = hist_week_numbers.where(D_hist.time.dt.year == hist_year, drop=True)

            # Find current week in this year
            week_indices = np.where(year_weeks == week_num)[0]
            if len(week_indices) > 0:
                week_idx_hist = week_indices[0]

                # Check if we have enough previous weeks
                if week_idx_hist >= 3:
                    # Accumulate 4 weeks
                    accum_D = (year_data.isel(time=week_idx_hist-3) +
                              year_data.isel(time=week_idx_hist-2) +
                              year_data.isel(time=week_idx_hist-1) +
                              year_data.isel(time=week_idx_hist))
                    hist_4week_accum_list.append(accum_D)
                else:
                    # Cross-year case: need previous year's data
                    prev_year_data = D_hist.where(D_hist.time.dt.year == hist_year - 1, drop=True)
                    if len(prev_year_data.time) > 0:
                        weeks_needed_from_prev_year = 3 - week_idx_hist
                        accum_D = year_data.isel(time=slice(0, week_idx_hist+1)).sum(dim='time')

                        # Add weeks from previous year (counting backwards from week 52)
                        for j in range(weeks_needed_from_prev_year):
                            prev_year_week = 52 - j
                            prev_week_data = prev_year_data.where(
                                get_week_of_year(prev_year_data.time) == prev_year_week, drop=True)
                            if len(prev_week_data.time) > 0:
                                accum_D += prev_week_data.isel(time=0)

                        hist_4week_accum_list.append(accum_D)

        # Skip if no historical data
        if not hist_4week_accum_list:
            print(f"  Warning: No historical data for week {week_num}, skipping")
            continue

        # Combine historical data
        hist_4week_accum = xr.concat(hist_4week_accum_list, dim='time')

        # Calculate SPEI
        if len(hist_4week_accum.time) < 10:
            print(f"  Warning: Insufficient historical samples for week {week_num}")
            spei_map = np.full_like(curr_week_D, np.nan)
        else:
            # Align coordinates
            if 'latitude' in hist_4week_accum.coords and 'lat' not in hist_4week_accum.coords:
                hist_4week_accum = hist_4week_accum.rename({'latitude': 'lat', 'longitude': 'lon'})

            # Convert to xarray for apply_ufunc
            week_4_accum_da = xr.DataArray(
                week_4_accum,
                dims=['lat', 'lon'],
                coords={'lat': lat, 'lon': lon}
            )

            spei_map = xr.apply_ufunc(
                calculate_spei_for_pixel,
                hist_4week_accum,
                week_4_accum_da,
                input_core_dims=[['time'], []],
                output_core_dims=[[]],
                exclude_dims=set(('time',)),
                vectorize=True,
                output_dtypes=[float],
                keep_attrs=True
            ).values

        # Find corresponding hindcast file (already verified to exist in pre-check)
        date_str = week_start_date.strftime('%Y-%m-%d')
        hindcast_filename = f'hindcast_{year}_week{week_num:02d}_surface_{date_str}.nc'
        hindcast_path = os.path.join(hindcast_dir, hindcast_filename)

        # Load existing hindcast file (with spei from Canglong predictions)
        with xr.open_dataset(hindcast_path) as ds_hindcast:
            # Load data into memory and close file
            ds_hindcast_copy = ds_hindcast.load()

        # Add spei_obs to the dataset copy
        ds_hindcast_copy['spei_obs'] = (('latitude', 'longitude'), spei_map)
        ds_hindcast_copy['spei_obs'].attrs = {
            'units': 'dimensionless',
            'long_name': 'Standardized Precipitation-Evapotranspiration Index (SPEI-4, ERA5 observation)',
            'description': '4-week accumulated SPEI calculated from ERA5 observations'
        }

        # Save back to the same file (overwrite) - create a temp file first
        temp_path = hindcast_path + '.tmp'
        ds_hindcast_copy.to_netcdf(temp_path)
        ds_hindcast_copy.close()

        # Replace original file with temp file
        os.replace(temp_path, hindcast_path)

        print(f"  Added spei_obs to: {hindcast_filename}")

    print("\n" + "=" * 80)
    print(f"Processing complete! Added spei_obs to files in: {hindcast_dir}")
    print("=" * 80)


if __name__ == "__main__":
    process_observation_spei()
