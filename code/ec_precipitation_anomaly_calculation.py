#!/usr/bin/env python3
"""
EC Precipitation Anomaly Percentage Calculation and Plotting

This script calculates precipitation anomaly percentages for ECMWF-S2S forecasts
at coarse resolution and generates visualization plots for China region.

Based on the CanglongPhysics project SPEI forecasting workflow.
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from ftplib import FTP

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cartopy.crs as ccrs
import salem
import geopandas as gpd
import cmaps
import mplotutils as mpu

# Add utils module path
sys.path.append('../')
from utils import plot

class ECPrecipitationAnomalyCalculator:
    """
    Calculator for EC precipitation anomaly percentages
    """
    
    def __init__(self, demo_end_time, ftp_host="10.168.39.193", 
                 ftp_user="Longhao_WANG", ftp_password="123456789"):
        """
        Initialize the calculator
        
        Args:
            demo_end_time (str): End time in format 'YYYY-MM-DD'
            ftp_host (str): FTP server address
            ftp_user (str): FTP username  
            ftp_password (str): FTP password
        """
        self.demo_end_time = demo_end_time
        self.ftp_host = ftp_host
        self.ftp_user = ftp_user
        self.ftp_password = ftp_password
        
        # Load historical climate data
        self.climate = self._load_climate_data()
        
    def _load_climate_data(self):
        """Load historical climate data for anomaly calculation"""
        try:
            climate = xr.open_dataset('/data/lhwang/npy/climate_variables_2000_2023_weekly.nc')
            return climate
        except FileNotFoundError:
            print("Warning: Climate data file not found. Using dummy data.")
            return None
    
    def download_ec_forecast_data(self):
        """
        Download EC forecast data from FTP server
        
        Returns:
            tuple: (precipitation_data, dewpoint_temp_data, avg_temp_data)
        """
        print(f"Downloading EC forecast data for {self.demo_end_time}")
        
        # Connect to FTP and download files
        with FTP(self.ftp_host) as ftp:
            ftp.login(self.ftp_user, self.ftp_password)
            
            # Define remote file paths
            remote_paths = [
                f'/Projects/data_NRT/S2S/Control forecast/P/P_{self.demo_end_time}_weekly.tif',
                f'/Projects/data_NRT/S2S/Control forecast/T/Tdew_{self.demo_end_time}_weekly.tif',
                f'/Projects/data_NRT/S2S/Control forecast/T/Tavg_{self.demo_end_time}_weekly.tif'
            ]
            
            temp_paths = []
            for remote_path in remote_paths:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    try:
                        # Download file
                        ftp.retrbinary(f'RETR {remote_path}', temp_file.write)
                        temp_paths.append(temp_file.name)
                        print(f"Downloaded: {os.path.basename(remote_path)}")
                    except Exception as e:
                        print(f"Error downloading {remote_path}: {e}")
                        temp_paths.append(None)
        
        # Read data using rioxarray
        try:
            data_prcp = rioxarray.open_rasterio(temp_paths[0]) if temp_paths[0] else None
            data_d2m = rioxarray.open_rasterio(temp_paths[1]) if temp_paths[1] else None
            data_t2m = rioxarray.open_rasterio(temp_paths[2]) if temp_paths[2] else None
            
            # Clean up temporary files
            for temp_path in temp_paths:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
            return data_prcp, data_d2m, data_t2m
            
        except Exception as e:
            print(f"Error reading downloaded data: {e}")
            return None, None, None
    
    def process_ec_data(self, data_prcp, data_d2m, data_t2m):
        """
        Process EC forecast data into standard format
        
        Args:
            data_prcp: Precipitation data
            data_d2m: Dewpoint temperature data
            data_t2m: Average temperature data
            
        Returns:
            xr.Dataset: Processed dataset
        """
        if any(data is None for data in [data_prcp, data_d2m, data_t2m]):
            print("Error: Missing EC forecast data")
            return None
            
        print("Processing EC forecast data...")
        
        # Create time coordinates for 6 weeks
        time_coords = pd.date_range(start=self.demo_end_time, periods=6, freq='7D')
        
        # Reshape data into standard format
        data_prcp_reshaped = data_prcp.values.reshape(6, data_prcp.y.size, data_prcp.x.size)
        data_d2m_reshaped = data_d2m.values.reshape(6, data_d2m.y.size, data_d2m.x.size)
        data_t2m_reshaped = data_t2m.values.reshape(6, data_t2m.y.size, data_t2m.x.size)
        
        # Create dataset with standard dimensions
        standard_data = xr.Dataset(
            data_vars={
                'total_precipitation': (('time', 'latitude', 'longitude'), data_prcp_reshaped),
                '2m_dewpoint_temperature': (('time', 'latitude', 'longitude'), data_d2m_reshaped),
                '2m_temperature': (('time', 'latitude', 'longitude'), data_t2m_reshaped)
            },
            coords={
                'time': time_coords,
                'latitude': data_prcp.y.values,
                'longitude': data_prcp.x.values
            }
        )
        
        return standard_data
    
    def calculate_potential_evapotranspiration(self, dataset):
        """
        Calculate potential evapotranspiration using Hargreaves-Samani method
        
        Args:
            dataset (xr.Dataset): Dataset with temperature variables
            
        Returns:
            xr.Dataset: Dataset with added PET variable
        """
        print("Calculating potential evapotranspiration...")
        
        t2m_celsius = dataset['2m_temperature'].values
        d2m_celsius = dataset['2m_dewpoint_temperature'].values
        
        # Calculate saturated and actual vapor pressure
        es = 0.618 * np.exp(17.27 * t2m_celsius / (t2m_celsius + 237.3))
        ea = 0.618 * np.exp(17.27 * d2m_celsius / (d2m_celsius + 237.3))
        
        # Calculate ratio, avoiding division by zero
        ratio_ea_es = np.full_like(t2m_celsius, np.nan)
        valid_es_mask = es > 1e-9
        ratio_ea_es[valid_es_mask] = ea[valid_es_mask] / es[valid_es_mask]
        ratio_ea_es = np.clip(ratio_ea_es, None, 1.0)
        
        # Calculate PET using simplified Hargreaves-Samani equation
        pet = 4.5 * np.power((1 + t2m_celsius / 25.0), 2) * (1 - ratio_ea_es)
        pet = np.maximum(pet, 0)
        
        # Add to dataset
        dataset['potential_evapotranspiration'] = (('time', 'latitude', 'longitude'), pet)
        
        return dataset
    
    def get_week_of_year(self, date):
        """
        Calculate week number of year (1st week = Jan 1-7)
        
        Args:
            date: xarray datetime
            
        Returns:
            Week number
        """
        day_of_year = date.dt.dayofyear
        return ((day_of_year - 1) // 7) + 1
    
    def calculate_precipitation_anomaly_percent(self, ec_data):
        """
        Calculate precipitation anomaly percentages
        
        Args:
            ec_data (xr.Dataset): EC forecast data
            
        Returns:
            xr.DataArray: Precipitation anomaly percentages
        """
        if self.climate is None:
            print("Warning: No climate data available for anomaly calculation")
            return None
            
        print("Calculating precipitation anomaly percentages...")
        
        # Resample climate data to match EC resolution
        climate_coarsened = self.climate.coarsen(lat=6, lon=6, boundary='trim').mean()
        
        # Crop to match EC data extent
        climate_cropped = climate_coarsened.sel(
            lat=slice(54.0, 15.0),
            lon=slice(70.5, 139.5)
        )
        
        # Interpolate to match EC grid
        target_lat = np.arange(54.0, 14.5, -1.5)
        target_lon = np.arange(70.5, 140.0, 1.5)
        
        climate_cropped = climate_cropped.interp(
            lat=target_lat,
            lon=target_lon,
            method='linear'
        )
        
        # Extract precipitation data
        precip_hist = climate_cropped['tp']
        precip_pred = ec_data['total_precipitation']
        precip_pred = precip_pred.rename({'latitude': 'lat', 'longitude': 'lon'})
        
        # Calculate week numbers
        hist_week_numbers = self.get_week_of_year(climate_cropped['time'])
        pred_week_numbers = self.get_week_of_year(ec_data['time'])
        
        # Calculate anomaly percentages for each week
        precip_anomaly_percent_list = []
        
        for i, t in enumerate(ec_data['time']):
            # Get current prediction week number
            curr_week_num = pred_week_numbers.isel(time=i).item()
            
            # Extract historical precipitation for same week
            hist_precip_same_week = precip_hist.where(hist_week_numbers == curr_week_num, drop=True)
            
            # Calculate historical mean precipitation
            hist_mean = hist_precip_same_week.mean(dim='time')
            
            # Calculate percentage anomaly
            # Formula: (current - historical_mean) / historical_mean * 100
            curr_precip = precip_pred.isel(time=i)
            
            # Avoid division by very small values
            epsilon = 0.0001
            precip_anomaly_percent = ((curr_precip - hist_mean) / 
                                    (hist_mean + epsilon)) * 100
            
            # Clip extreme values for better visualization
            precip_anomaly_percent = precip_anomaly_percent.clip(-200, 200)
            
            precip_anomaly_percent_list.append(precip_anomaly_percent)
        
        # Combine results
        precip_anomaly_percent = xr.concat(precip_anomaly_percent_list, dim='time')
        precip_anomaly_percent = precip_anomaly_percent.assign_coords(time=ec_data['time'])
        
        return precip_anomaly_percent
    
    def plot_precipitation_anomaly_china(self, precip_anomaly_percent, output_path=None):
        """
        Plot precipitation anomaly percentages for China region
        
        Args:
            precip_anomaly_percent (xr.DataArray): Precipitation anomaly data
            output_path (str, optional): Path to save the plot
        """
        print("Creating precipitation anomaly plots for China...")
        
        # Set font
        try:
            font_path = "/usr/share/fonts/arial/ARIAL.TTF"
            from matplotlib import font_manager
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams['font.family'] = font_name
        except:
            print("Warning: Could not set Arial font")
        
        # Select China region
        china_precip_anomaly = precip_anomaly_percent.sel(
            lon=slice(70, 140),
            lat=slice(55, 15)
        )
        
        # Load China boundary shapefile
        try:
            china_shp = gpd.read_file('data/china.shp')
        except:
            print("Warning: Could not load China shapefile")
            china_shp = None
        
        # Set color scale and range
        vmin, vmax = -100, 100
        unit_label = '%'
        title_prefix = 'ECMWF-S2S'
        data_cmap = cmaps.drought_severity_r
        
        # Create figure and projection
        fig = plt.figure(figsize=(42, 28))
        axes = []
        for i in range(6):
            ax = fig.add_subplot(2, 3, i+1, projection=ccrs.LambertConformal(
                central_longitude=105,
                central_latitude=40,
                standard_parallels=(25.0, 47.0)
            ))
            axes.append(ax)
        
        levels = np.linspace(vmin, vmax, 11)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
        # Plot each week
        mappable = None
        for t in range(6):
            ax = axes[t]
            current_data = china_precip_anomaly.isel(time=t)
            
            # Apply China mask if shapefile available
            if china_shp is not None:
                ds_t = salem.DataArrayAccessor(current_data)
                masked_data_t = ds_t.roi(shape=china_shp)
            else:
                masked_data_t = current_data
            
            # Plot
            try:
                mappable = plot.one_map_china(
                    masked_data_t, 
                    ax, 
                    cmap=data_cmap, 
                    levels=levels,
                    norm=norm,
                    mask_ocean=False, 
                    add_coastlines=True, 
                    add_land=False, 
                    add_river=True, 
                    add_lake=True, 
                    add_stock=False, 
                    add_gridlines=True, 
                    colorbar=False, 
                    plotfunc="pcolormesh"
                )
            except:
                # Fallback to simple plotting if custom plot function fails
                im = ax.pcolormesh(
                    masked_data_t.lon, masked_data_t.lat, masked_data_t.values,
                    cmap=data_cmap, norm=norm, transform=ccrs.PlateCarree()
                )
                mappable = im
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)
            
            # Add small inset map if possible
            try:
                ax2 = fig.add_axes([0.222 + (t % 3) * 0.291, 0.0500 + (1 - t // 3) * 0.4800, 0.06, 0.09], 
                                  projection=ccrs.LambertConformal(
                                      central_longitude=105,
                                      central_latitude=40,
                                      standard_parallels=(25.0, 47.0)
                                  ))
                plot.sub_china_map(masked_data_t, ax2, cmap=data_cmap, add_coastlines=False, add_land=False)
            except:
                pass
            
            # Time label
            current_time = china_precip_anomaly.time.values[t]
            start_date = np.datetime_as_string(current_time - np.timedelta64(6, 'D'), unit='D').replace('-', '')
            end_date = np.datetime_as_string(current_time, unit='D').replace('-', '')
            ax.set_title(f'{title_prefix} for {start_date}-{end_date}', fontsize=24)
        
        # Add colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label(f'Precipitation Anomaly ({unit_label})', fontsize=24)
        cbar.ax.tick_params(labelsize=24)
        
        # Adjust layout
        plt.subplots_adjust(left=0.025, right=0.85, top=0.9, bottom=0.05, wspace=0.2, hspace=0.3)
        
        try:
            mpu.set_map_layout(axes, width=80)
        except:
            pass
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        
        plt.show()
    
    def upload_to_ftp(self, local_file_path, ftp_directory='/Projects/data_NRT/Canglong'):
        """
        Upload file to FTP server
        
        Args:
            local_file_path (str): Local file path
            ftp_directory (str): FTP target directory
            
        Returns:
            bool: Success status
        """
        try:
            # Connect to FTP server
            with FTP(self.ftp_host) as ftp:
                ftp.login(user=self.ftp_user, passwd=self.ftp_password)
                
                # Change to target directory
                ftp.cwd(ftp_directory)
                
                # Get filename
                filename = os.path.basename(local_file_path)
                
                # Upload file
                with open(local_file_path, 'rb') as file:
                    ftp.storbinary(f'STOR {filename}', file)
                
                print(f"File {filename} uploaded successfully to {ftp_directory}")
                return True
                
        except Exception as e:
            print(f"Error uploading file: {e}")
            return False

    def save_and_upload_results(self, precip_anomaly_percent, output_dir='../../data', 
                               upload_to_ftp=True, ftp_directory='/Projects/data_NRT/Canglong'):
        """
        Save precipitation anomaly results to NetCDF file and upload to FTP
        
        Args:
            precip_anomaly_percent (xr.DataArray): Results to save
            output_dir (str): Output directory
            upload_to_ftp (bool): Whether to upload to FTP server
            ftp_directory (str): FTP target directory
            
        Returns:
            str: Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        input_end_date = datetime.strptime(self.demo_end_time, '%Y-%m-%d')
        start_date = input_end_date + timedelta(days=1)
        end_date = input_end_date + timedelta(days=6*7)
        
        filename = f'EC_PrcpAnomPercent_forecast_{start_date.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}.nc'
        filepath = os.path.join(output_dir, filename)
        
        # Save locally
        precip_anomaly_percent.to_netcdf(filepath)
        print(f"Results saved locally to: {filepath}")
        
        # Upload to FTP if requested
        if upload_to_ftp:
            success = self.upload_to_ftp(filepath, ftp_directory)
            if success:
                print(f"Results uploaded to FTP: {ftp_directory}/{filename}")
            else:
                print("Warning: Failed to upload to FTP, but local file saved successfully")
        
        return filepath

    def save_results(self, precip_anomaly_percent, output_dir='../../data'):
        """
        Save precipitation anomaly results to NetCDF file (legacy method)
        
        Args:
            precip_anomaly_percent (xr.DataArray): Results to save
            output_dir (str): Output directory
            
        Returns:
            str: Path to saved file
        """
        return self.save_and_upload_results(precip_anomaly_percent, output_dir, 
                                          upload_to_ftp=False)
    
    def run_full_analysis(self, output_dir='../../data', create_plots=True, upload_to_ftp=True,
                         ftp_directory='/Projects/data_NRT/Canglong'):
        """
        Run complete EC precipitation anomaly analysis
        
        Args:
            output_dir (str): Directory for output files
            create_plots (bool): Whether to create visualization plots
            upload_to_ftp (bool): Whether to upload results to FTP server
            ftp_directory (str): FTP target directory
            
        Returns:
            tuple: (precip_anomaly_percent, output_filepath)
        """
        print("Starting EC precipitation anomaly analysis...")
        
        # Download EC forecast data
        data_prcp, data_d2m, data_t2m = self.download_ec_forecast_data()
        
        if data_prcp is None:
            print("Error: Could not download EC forecast data")
            return None, None
        
        # Process EC data
        ec_data = self.process_ec_data(data_prcp, data_d2m, data_t2m)
        
        if ec_data is None:
            return None, None
        
        # Calculate potential evapotranspiration
        ec_data = self.calculate_potential_evapotranspiration(ec_data)
        
        # Calculate precipitation anomaly percentages
        precip_anomaly_percent = self.calculate_precipitation_anomaly_percent(ec_data)
        
        if precip_anomaly_percent is None:
            return None, None
        
        # Save results with optional FTP upload
        output_filepath = self.save_and_upload_results(
            precip_anomaly_percent, 
            output_dir, 
            upload_to_ftp=upload_to_ftp,
            ftp_directory=ftp_directory
        )
        
        # Create plots
        if create_plots:
            plot_path = output_filepath.replace('.nc', '_china_plot.png')
            self.plot_precipitation_anomaly_china(precip_anomaly_percent, plot_path)
        
        print("EC precipitation anomaly analysis completed successfully!")
        
        return precip_anomaly_percent, output_filepath


def main():
    """
    Main function to run EC precipitation anomaly calculation
    """
    # Configuration
    demo_end_time = '2025-06-24'  # Replace with actual end time
    
    # Create calculator instance
    calculator = ECPrecipitationAnomalyCalculator(demo_end_time)
    
    # Run full analysis with FTP upload
    results, output_file = calculator.run_full_analysis(
        output_dir='../../data',
        create_plots=True,
        upload_to_ftp=True,
        ftp_directory='/Projects/data_NRT/Canglong'
    )
    
    if results is not None:
        print(f"Analysis completed. Results shape: {results.shape}")
        print(f"Output file: {output_file}")
    else:
        print("Analysis failed.")


if __name__ == "__main__":
    main()