#!/usr/bin/env python3
"""
调试ECMWF数据计算NaN的问题
"""
import xarray as xr
import numpy as np
import rioxarray as rxr
import salem
from pathlib import Path

# 配置路径
local_data_dir = Path('/home/lhwang/Desktop/CanglongPhysics/data')
shapefile_path = '/home/lhwang/Desktop/CanglongPhysics/code/data/china.shp'

def crop_data_with_shapefile(data, shapefile_path):
    """使用shapefile裁剪数据"""
    try:
        shp_data = salem.read_shapefile(shapefile_path)
        cropped_data = data.salem.roi(shape=shp_data)
        return cropped_data
    except Exception as e:
        print(f"裁剪失败: {e}")
        return data

def calculate_rmse(forecast, observation):
    """计算RMSE"""
    f_flat = np.array(forecast).flatten()
    o_flat = np.array(observation).flatten()
    
    print(f"Forecast shape: {f_flat.shape}, range: {np.nanmin(f_flat):.3f} to {np.nanmax(f_flat):.3f}")
    print(f"Observation shape: {o_flat.shape}, range: {np.nanmin(o_flat):.3f} to {np.nanmax(o_flat):.3f}")
    
    valid_mask = ~(np.isnan(f_flat) | np.isnan(o_flat))
    print(f"Valid pixels: {valid_mask.sum()} / {len(valid_mask)}")
    
    if valid_mask.sum() == 0:
        return np.nan
    diff = f_flat[valid_mask] - o_flat[valid_mask]
    rmse = np.sqrt(np.mean(diff**2))
    return rmse

# 1. 加载观测数据
print("=== 加载观测数据 ===")
obs_file = local_data_dir / 'hind_obs' / 'obs_with_dewpoint_2025-07-09_to_2025-07-15.nc'
obs_data = xr.open_dataset(obs_file)
obs_temp = obs_data['2m_temperature']
obs_precip = obs_data['total_precipitation']

print(f"观测温度范围: {obs_temp.min().values:.2f} 到 {obs_temp.max().values:.2f}")
print(f"观测降水范围: {obs_precip.min().values:.2f} 到 {obs_precip.max().values:.2f}")

# 使用shapefile裁剪观测数据
obs_temp_cropped = crop_data_with_shapefile(obs_temp, shapefile_path)
obs_precip_cropped = crop_data_with_shapefile(obs_precip, shapefile_path)

print(f"裁剪后观测温度形状: {obs_temp_cropped.shape}")
print(f"裁剪后观测降水形状: {obs_precip_cropped.shape}")

# 2. 加载ECMWF数据
print("\n=== 加载ECMWF数据 ===")
ecmwf_temp_file = local_data_dir / 'ecmwf' / 'T' / 'Tavg_2025-07-09_weekly.tif'
ecmwf_precip_file = local_data_dir / 'ecmwf' / 'P' / 'P_2025-07-09_weekly.tif'

temp_data = rxr.open_rasterio(str(ecmwf_temp_file))
precip_data = rxr.open_rasterio(str(ecmwf_precip_file))

# 选择第1周的数据 (band=0)
temp_celsius = temp_data.isel(band=0)
precip_mm_day = precip_data.isel(band=0)

print(f"ECMWF温度形状: {temp_celsius.shape}, 范围: {temp_celsius.min().values:.2f} 到 {temp_celsius.max().values:.2f}")
print(f"ECMWF降水形状: {precip_mm_day.shape}, 范围: {precip_mm_day.min().values:.2f} 到 {precip_mm_day.max().values:.2f}")

# 转换为dataset以便使用salem
temp_ds = temp_celsius.to_dataset(name='temperature')
precip_ds = precip_mm_day.to_dataset(name='precipitation')

# 裁剪ECMWF数据
print("\n=== 裁剪ECMWF数据 ===")
temp_cropped = crop_data_with_shapefile(temp_ds, shapefile_path)['temperature']
precip_cropped = crop_data_with_shapefile(precip_ds, shapefile_path)['precipitation']

print(f"ECMWF裁剪后温度形状: {temp_cropped.shape}")
print(f"ECMWF裁剪后降水形状: {precip_cropped.shape}")
print(f"ECMWF裁剪后温度范围: {temp_cropped.min().values:.2f} 到 {temp_cropped.max().values:.2f}")
print(f"ECMWF裁剪后降水范围: {precip_cropped.min().values:.2f} 到 {precip_cropped.max().values:.2f}")

# 3. 检查数据对齐
print("\n=== 检查数据对齐 ===")
print(f"观测数据坐标:")
print(f"  latitude: {obs_temp_cropped.latitude.shape}, 范围: {obs_temp_cropped.latitude.min().values:.2f} 到 {obs_temp_cropped.latitude.max().values:.2f}")
print(f"  longitude: {obs_temp_cropped.longitude.shape}, 范围: {obs_temp_cropped.longitude.min().values:.2f} 到 {obs_temp_cropped.longitude.max().values:.2f}")

print(f"ECMWF数据坐标:")
print(f"  y: {temp_cropped.y.shape}, 范围: {temp_cropped.y.min().values:.2f} 到 {temp_cropped.y.max().values:.2f}")
print(f"  x: {temp_cropped.x.shape}, 范围: {temp_cropped.x.min().values:.2f} 到 {temp_cropped.x.max().values:.2f}")

# 4. 尝试重采样对齐
print("\n=== 尝试数据插值对齐 ===")
try:
    # 将ECMWF数据插值到观测数据的网格
    temp_interp = temp_cropped.interp(
        y=obs_temp_cropped.latitude,
        x=obs_temp_cropped.longitude,
        method='linear'
    )
    precip_interp = precip_cropped.interp(
        y=obs_precip_cropped.latitude, 
        x=obs_precip_cropped.longitude,
        method='linear'
    )
    
    print(f"插值后ECMWF温度形状: {temp_interp.shape}")
    print(f"插值后ECMWF降水形状: {precip_interp.shape}")
    
    # 计算RMSE
    print("\n=== 计算指标 ===")
    temp_rmse = calculate_rmse(temp_interp, obs_temp_cropped)
    precip_rmse = calculate_rmse(precip_interp, obs_precip_cropped)
    
    print(f"温度RMSE: {temp_rmse:.3f}")
    print(f"降水RMSE: {precip_rmse:.3f}")
    
except Exception as e:
    print(f"插值失败: {e}")
    print("尝试直接比较...")
    
    temp_rmse = calculate_rmse(temp_cropped, obs_temp_cropped)
    precip_rmse = calculate_rmse(precip_cropped, obs_precip_cropped)
    
    print(f"温度RMSE (直接): {temp_rmse:.3f}")
    print(f"降水RMSE (直接): {precip_rmse:.3f}")

if __name__ == "__main__":
    print("开始调试ECMWF数据问题...")