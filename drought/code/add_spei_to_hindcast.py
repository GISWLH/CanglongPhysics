"""
Calculate SPEI-4 (4-week) for hindcast files
Only calculate spei_obs for lead1 (reuse for other leads)
Uses simplified standardization method with 2000-2023 climatology
"""

import xarray as xr
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def calculate_pet(temp_c, dewpoint_c):
    """
    Calculate PET using vapor pressure deficit method (from run.py)

    Parameters:
    -----------
    temp_c : array-like (lat, lon)
        Air temperature in Celsius (NOT Kelvin!)
    dewpoint_c : array-like (lat, lon)
        Dewpoint temperature in Celsius (NOT Kelvin!)

    Returns:
    --------
    pet : array-like (lat, lon)
        Potential evapotranspiration in mm/day
    """
    # 计算饱和水汽压和实际水汽压
    es = 0.618 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    ea = 0.618 * np.exp(17.27 * dewpoint_c / (dewpoint_c + 237.3))

    # 计算比率，避免除零错误
    ratio_ea_es = np.full_like(temp_c, np.nan)
    valid_es_mask = es > 1e-9
    ratio_ea_es[valid_es_mask] = ea[valid_es_mask] / es[valid_es_mask]
    ratio_ea_es = np.clip(ratio_ea_es, None, 1.0)

    # 计算PET
    pet = 4.5 * np.power((1 + temp_c / 25.0), 2) * (1 - ratio_ea_es)
    pet = np.maximum(pet, 0)

    return pet


def load_climate_data(climate_file='E:/data/climate_variables_2000_2023_weekly.nc'):
    """
    Load climate data (2000-2023) for SPEI calibration

    Returns:
    --------
    precip_climate : array (1248, 721, 1440)
        Precipitation climatology in mm/day
    pet_climate : array (1248, 721, 1440)
        PET climatology in mm/day
    """
    print(f"  Loading climate data from {climate_file}...")
    ds_climate = xr.open_dataset(climate_file)

    # tp is in mm/day, pet is in mm/day
    precip_climate = ds_climate['tp'].values
    pet_climate = ds_climate['pet'].values

    ds_climate.close()

    print(f"  Climate data loaded: {precip_climate.shape}")
    return precip_climate, pet_climate


def calculate_spei_simplified(precip_mm, pet_mm, precip_climate, pet_climate, scale=4):
    """
    Calculate SPEI using simplified standardization method
    Uses climatology (2000-2023) to establish the distribution parameters

    Parameters:
    -----------
    precip_mm : array (time, lat, lon)
        Precipitation in mm for target period
    pet_mm : array (time, lat, lon)
        PET in mm for target period
    precip_climate : array (1248, lat, lon)
        Precipitation climatology (2000-2023)
    pet_climate : array (1248, lat, lon)
        PET climatology (2000-2023)
    scale : int
        Time scale in weeks (default: 4)

    Returns:
    --------
    spei : array (time, lat, lon)
        SPEI values
    """
    n_time, n_lat, n_lon = precip_mm.shape
    spei = np.full((n_time, n_lat, n_lon), np.nan, dtype=np.float32)

    print(f"  Calculating SPEI for {n_lat}x{n_lon} grid points...")

    # Calculate water balance (P - PET) for climatology
    wb_climate = precip_climate - pet_climate  # (1248, 721, 1440)

    # Calculate rolling sum for climatology to get distribution parameters
    n_climate = wb_climate.shape[0]
    wb_climate_rolled = np.full_like(wb_climate, np.nan)

    for t in range(scale - 1, n_climate):
        wb_climate_rolled[t] = np.sum(wb_climate[t - scale + 1:t + 1], axis=0)

    # Calculate water balance for target period
    wb_target = precip_mm - pet_mm  # (98, 721, 1440)

    # Calculate rolling sum for target period
    wb_target_rolled = np.full_like(wb_target, np.nan)

    for t in range(scale - 1, n_time):
        wb_target_rolled[t] = np.sum(wb_target[t - scale + 1:t + 1], axis=0)

    # For each grid point, standardize using climatology statistics
    for i in tqdm(range(n_lat), desc="  Latitude", leave=False):
        for j in range(n_lon):
            # Get climatology distribution at this point
            wb_clim_point = wb_climate_rolled[:, i, j]

            # Skip if insufficient valid data
            valid_clim = ~np.isnan(wb_clim_point)
            if np.sum(valid_clim) < 20:  # Need at least 20 samples
                continue

            # Calculate mean and std from climatology
            mean_wb = np.nanmean(wb_clim_point)
            std_wb = np.nanstd(wb_clim_point)

            # Avoid division by zero
            if std_wb < 0.01:
                std_wb = 0.01

            # Standardize target period using climatology statistics
            wb_target_point = wb_target_rolled[:, i, j]
            spei[:, i, j] = (wb_target_point - mean_wb) / std_wb

    return spei


def process_single_file(file_path, lead_num, precip_climate, pet_climate, spei_obs_data=None):
    """Process a single hindcast file and add SPEI variables

    Parameters:
    -----------
    spei_obs_data : array or None
        Pre-calculated spei_obs from lead1 (for lead2-6)
    """
    print(f"\n{'='*80}")
    print(f"Processing lead{lead_num}: {file_path}")
    print(f"{'='*80}")

    # Open dataset
    print("Loading dataset...")
    ds = xr.open_dataset(file_path)

    print(f"Time steps: {len(ds.time)}")
    print(f"Spatial shape: {len(ds.latitude)} x {len(ds.longitude)}")

    # Check if SPEI already exists
    if 'spei' in ds.data_vars:
        print("WARNING: SPEI variables already exist. Removing old versions...")
        vars_to_drop = ['spei']
        if 'spei_obs' in ds.data_vars:
            vars_to_drop.append('spei_obs')
        ds = ds.drop_vars(vars_to_drop)

    # Extract variables - PREDICTION
    print("\nExtracting prediction variables...")
    temp_pred = ds['2m_temperature'].values  # Already in Celsius!
    dewpoint_pred = ds['2m_dewpoint_temperature'].values  # Already in Celsius!
    precip_pred = ds['total_precipitation'].values  # mm/day

    # Calculate PET for prediction
    print("\nCalculating PET for predictions...")
    pet_pred = np.zeros_like(temp_pred)
    for t in tqdm(range(len(ds.time)), desc="PET (prediction)"):
        pet_pred[t] = calculate_pet(temp_pred[t], dewpoint_pred[t])

    # Calculate SPEI-4 for prediction
    print("\nCalculating SPEI-4 for predictions (using 2000-2023 climatology)...")
    spei_pred = calculate_spei_simplified(precip_pred, pet_pred, precip_climate, pet_climate, scale=4)

    # Only calculate spei_obs for lead1
    if lead_num == 1:
        print("\nCalculating SPEI-4 for observations (lead1 only)...")

        # Extract observation variables
        temp_obs = ds['2m_temperature_obs'].values
        dewpoint_obs = ds['2m_dewpoint_temperature_obs'].values
        precip_obs = ds['total_precipitation_obs'].values

        # Calculate PET for observation
        print("Calculating PET for observations...")
        pet_obs = np.zeros_like(temp_obs)
        for t in tqdm(range(len(ds.time)), desc="PET (observation)"):
            pet_obs[t] = calculate_pet(temp_obs[t], dewpoint_obs[t])

        # Calculate SPEI-4 for observation
        spei_obs = calculate_spei_simplified(precip_obs, pet_obs, precip_climate, pet_climate, scale=4)
    else:
        print("\nUsing pre-calculated SPEI-4 for observations (from lead1)...")
        # Extract the matching time period from lead1's spei_obs
        n_time = len(ds.time)
        if spei_obs_data is not None:
            # Take the first n_time steps to match current lead's time dimension
            spei_obs = spei_obs_data[:n_time]
            print(f"  Extracted first {n_time} time steps from lead1 spei_obs (original: {spei_obs_data.shape[0]} steps)")
        else:
            print("  Warning: spei_obs_data is None, creating NaN array")
            spei_obs = np.full((n_time, ds.dims['latitude'], ds.dims['longitude']), np.nan, dtype=np.float32)

    # Add to dataset
    print("\nAdding SPEI variables to dataset...")
    ds['spei'] = xr.DataArray(
        spei_pred,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': ds.time, 'latitude': ds.latitude, 'longitude': ds.longitude},
        attrs={
            'long_name': 'Standardized Precipitation Evapotranspiration Index (4-week)',
            'units': 'dimensionless',
            'description': 'SPEI-4 calculated from model predictions using 2000-2023 climatology',
            'scale': '4 weeks'
        }
    )

    ds['spei_obs'] = xr.DataArray(
        spei_obs,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': ds.time, 'latitude': ds.latitude, 'longitude': ds.longitude},
        attrs={
            'long_name': 'Standardized Precipitation Evapotranspiration Index (4-week) - Observation',
            'units': 'dimensionless',
            'description': 'SPEI-4 calculated from observations using 2000-2023 climatology',
            'scale': '4 weeks'
        }
    )

    # Save back to file
    print(f"\nSaving updated file...")

    # Create encoding
    encoding = {var: {'zlib': True, 'complevel': 5} for var in ds.data_vars}

    # Close the original dataset first to release the file handle
    ds.close()

    # Create a temporary output file
    temp_path = file_path + '.tmp'

    # Reopen the dataset and save with new variables to temp file
    ds = xr.open_dataset(file_path)

    # Add SPEI variables again
    ds['spei'] = xr.DataArray(
        spei_pred,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': ds.time, 'latitude': ds.latitude, 'longitude': ds.longitude},
        attrs={
            'long_name': 'Standardized Precipitation Evapotranspiration Index (4-week)',
            'units': 'dimensionless',
            'description': 'SPEI-4 calculated from model predictions using 2000-2023 climatology',
            'scale': '4 weeks'
        }
    )

    ds['spei_obs'] = xr.DataArray(
        spei_obs,
        dims=['time', 'latitude', 'longitude'],
        coords={'time': ds.time, 'latitude': ds.latitude, 'longitude': ds.longitude},
        attrs={
            'long_name': 'Standardized Precipitation Evapotranspiration Index (4-week) - Observation',
            'units': 'dimensionless',
            'description': 'SPEI-4 calculated from observations using 2000-2023 climatology',
            'scale': '4 weeks'
        }
    )

    # Write to temp file
    ds.to_netcdf(temp_path, encoding=encoding, format='NETCDF4')
    ds.close()

    # Replace original file with temp file
    import os
    import shutil
    if os.path.exists(file_path):
        os.remove(file_path)
    shutil.move(temp_path, file_path)

    print(f"DONE Completed lead{lead_num}")
    print(f"  - SPEI range (prediction): [{np.nanmin(spei_pred):.2f}, {np.nanmax(spei_pred):.2f}]")
    print(f"  - SPEI range (observation): [{np.nanmin(spei_obs):.2f}, {np.nanmax(spei_obs):.2f}]")
    print(f"  - Valid data points (prediction): {np.sum(~np.isnan(spei_pred))}/{spei_pred.size}")
    print(f"  - Valid data points (observation): {np.sum(~np.isnan(spei_obs))}/{spei_obs.size}")

    # Return spei_obs from lead1 for reuse
    return spei_obs if lead_num == 1 else None


def main():
    """Main function to process all 6 lead files"""
    base_path = "Z:/Data/hindcast_2022_2023"

    print("="*80)
    print("SPEI-4 Calculation for Hindcast Files (2022-2023)")
    print("Using simplified standardization with 2000-2023 climatology")
    print("="*80)
    print("\nThis script will:")
    print("1. Load climatology data (2000-2023) for SPEI calibration")
    print("2. Calculate PET using vapor pressure deficit method")
    print("3. Calculate SPEI-4 (4-week) using simplified standardization")
    print("4. Calculate spei_obs only for lead1 (reuse for lead2-6)")
    print("5. Add 'spei' and 'spei_obs' variables to each file")
    print(f"\nProcessing 6 files from {base_path}\n")

    # Load climatology data once for all files
    print("\n" + "="*80)
    print("Loading climatology data (2000-2023)...")
    print("="*80)
    precip_climate, pet_climate = load_climate_data()

    # Load spei_obs from lead1 (already processed)
    print("\n" + "="*80)
    print("Loading spei_obs from lead1 (already processed)...")
    print("="*80)

    spei_obs_lead1 = None
    try:
        ds_lead1 = xr.open_dataset(f"{base_path}/hindcast_2022_2023_lead1.nc")
        if 'spei_obs' in ds_lead1.data_vars:
            spei_obs_lead1 = ds_lead1['spei_obs'].values
            print(f"Successfully loaded spei_obs from lead1: shape {spei_obs_lead1.shape}")
        ds_lead1.close()
    except Exception as e:
        print(f"Warning: Could not load spei_obs from lead1: {str(e)}")

    # Process each lead file (only lead4)
    for lead in [4]:
        file_path = f"{base_path}/hindcast_2022_2023_lead{lead}.nc"

        try:
            result = process_single_file(file_path, lead, precip_climate, pet_climate, spei_obs_lead1)

        except Exception as e:
            print(f"\nERROR processing lead{lead}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*80)
    print("All files processed!")
    print("="*80)

    # Verification
    print("\nVerifying SPEI variables in all files...")
    for lead in range(1, 7):
        file_path = f"{base_path}/hindcast_2022_2023_lead{lead}.nc"
        try:
            ds = xr.open_dataset(file_path)
            has_spei = 'spei' in ds.data_vars
            has_spei_obs = 'spei_obs' in ds.data_vars
            print(f"  Lead {lead}: spei={has_spei}, spei_obs={has_spei_obs}")
            ds.close()
        except Exception as e:
            print(f"  Lead {lead}: ERROR - {str(e)}")


if __name__ == "__main__":
    main()
