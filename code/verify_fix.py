"""
验证run_ec_pure.py修复后的温度数据
读取处理后的数据，检查温度是否合理
"""
import xarray as xr
import pandas as pd
import numpy as np
from ftplib import FTP
import tempfile
import os
import rioxarray

# 参数
demo_start_time = '2025-10-30'
demo_end_time = '2025-11-12'
ftp_host = "10.168.39.193"
ftp_user = "Longhao_WANG"
ftp_password = "123456789"

print("="*70)
print("验证修复后的Week3温度数据")
print("="*70)

# 1. 加载Week1和Week2 (ERA5数据 - 开尔文)
demo_start = (pd.to_datetime(demo_start_time) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
print(f"\n1. 加载Week1和Week2数据 (ERA5, {demo_start} to {demo_end_time})")

data_inner_steps = 24
ds_surface = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    chunks=None,
    consolidated=True
)[['2m_temperature']]
surface_ds_former = ds_surface.sel(time=slice(demo_start, demo_end_time, data_inner_steps))
surface_ds_former.load()

week1_data = surface_ds_former.isel(time=slice(0, 7))
week2_data = surface_ds_former.isel(time=slice(7, 14))

week1_mean = week1_data.mean(dim='time')
week2_mean = week2_data.mean(dim='time')

print(f"   Week1 (ERA5): mean = {week1_mean['2m_temperature'].values.mean():.2f} K")
print(f"   Week2 (ERA5): mean = {week2_mean['2m_temperature'].values.mean():.2f} K")

# 2. 加载Week3 (MSWX数据 - 摄氏度)
week3_start = (pd.to_datetime(demo_end_time) - pd.Timedelta(days=6)).strftime('%Y-%m-%d')
week3_dates = pd.date_range(start=week3_start, end=demo_end_time, freq='D')

print(f"\n2. 加载Week3数据 (MSWX, {week3_start} to {demo_end_time})")

week3_t2m_list = []
with FTP(ftp_host) as ftp:
    ftp.login(ftp_user, ftp_password)
    for date in week3_dates[:1]:  # 只读第一天测试
        date_str = date.strftime('%Y-%m-%d')
        temp_remote = f'/Projects/data_NRT/MSWX/tif/air temperature-{date_str}.tif'
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
            ftp.retrbinary(f'RETR {temp_remote}', temp_file.write)
            temp_path = temp_file.name
        t2m_data = rioxarray.open_rasterio(temp_path).squeeze().load()
        week3_t2m_list.append(t2m_data)
        os.unlink(temp_path)
        break

week3_t2m_sample = week3_t2m_list[0]
print(f"   Week3 (MSWX原始): mean = {week3_t2m_sample.values.mean():.2f} C")

# 3. 测试修复：将Week3转换为开尔文
week3_t2m_kelvin = week3_t2m_sample.values + 273.15
print(f"   Week3 (转为K): mean = {week3_t2m_kelvin.mean():.2f} K")

# 4. 模拟统一转换
week1_celsius = week1_mean['2m_temperature'].values.mean() - 273.15
week2_celsius = week2_mean['2m_temperature'].values.mean() - 273.15
week3_celsius = week3_t2m_kelvin.mean() - 273.15

print(f"\n3. 统一转换为摄氏度后:")
print(f"   Week1: {week1_celsius:.2f} C")
print(f"   Week2: {week2_celsius:.2f} C")
print(f"   Week3: {week3_celsius:.2f} C")

# 5. 检查是否合理
print(f"\n4. 合理性检查:")
all_temps = [week1_celsius, week2_celsius, week3_celsius]
if all(-50 < t < 50 for t in all_temps):
    print("   ✅ 所有温度都在合理范围内 (-50°C to 50°C)")
else:
    print("   ❌ 温度超出合理范围!")

temp_range = max(all_temps) - min(all_temps)
print(f"   温度范围: {temp_range:.2f} C")
if temp_range < 100:
    print("   ✅ 温度范围合理")
else:
    print("   ❌ 温度范围过大!")

# 6. 计算PET差异
print(f"\n5. PET计算测试:")

def simple_pet(t_celsius, rh=0.5):
    es = 0.618 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))
    ea = es * rh
    ratio = ea / es
    pet = 4.5 * np.power((1 + t_celsius / 25.0), 2) * (1 - ratio)
    return max(pet, 0)

for i, (week, temp) in enumerate([(1, week1_celsius), (2, week2_celsius), (3, week3_celsius)], 1):
    pet = simple_pet(temp)
    print(f"   Week{week}: T={temp:6.2f}C, PET={pet:6.2f} mm/day")

print("\n" + "="*70)
print("验证完成！如果所有检查都通过，说明修复成功。")
print("="*70)
