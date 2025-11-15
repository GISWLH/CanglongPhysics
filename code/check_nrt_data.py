"""
检查网盘中MSWX数据的分辨率、单位等信息
"""
import rioxarray
from ftplib import FTP
import tempfile
import os
import numpy as np

# FTP连接信息
ftp_host = "10.168.39.193"
ftp_user = "Longhao_WANG"
ftp_password = "123456789"

# 测试日期
test_date = '2025-11-06'

print("="*60)
print("检查网盘MSWX数据属性")
print("="*60)

# 连接FTP
with FTP(ftp_host) as ftp:
    ftp.login(ftp_user, ftp_password)

    # 1. 检查气温数据
    print("\n1. 气温数据 (air temperature)")
    print("-"*60)
    temp_remote = f'/Projects/data_NRT/MSWX/tif/air temperature-{test_date}.tif'
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
        ftp.retrbinary(f'RETR {temp_remote}', temp_file.write)
        temp_path = temp_file.name

    t2m_data = rioxarray.open_rasterio(temp_path)
    print(f"File: {temp_remote}")
    print(f"Dimensions: {t2m_data.dims}")
    print(f"Shape: {t2m_data.shape}")
    print(f"Coordinate ranges:")
    print(f"  - x (longitude): {t2m_data.x.min().values:.4f} to {t2m_data.x.max().values:.4f}")
    print(f"  - y (latitude): {t2m_data.y.min().values:.4f} to {t2m_data.y.max().values:.4f}")
    print(f"Resolution:")
    if len(t2m_data.x) > 1:
        print(f"  - x direction: {np.abs(t2m_data.x.values[1] - t2m_data.x.values[0]):.6f} degrees")
    if len(t2m_data.y) > 1:
        print(f"  - y direction: {np.abs(t2m_data.y.values[1] - t2m_data.y.values[0]):.6f} degrees")
    print(f"Data statistics:")
    print(f"  - Min: {t2m_data.values.min():.2f}")
    print(f"  - Max: {t2m_data.values.max():.2f}")
    print(f"  - Mean: {t2m_data.values.mean():.2f}")
    print(f"  - Median: {np.median(t2m_data.values):.2f}")
    print(f"CRS: {t2m_data.rio.crs}")
    print(f"NoData value: {t2m_data.rio.nodata}")

    # 判断单位
    mean_val = t2m_data.values.mean()
    if mean_val > 100:
        print(f"Inferred unit: Kelvin (K) - mean={mean_val:.2f}")
    elif -50 < mean_val < 50:
        print(f"Inferred unit: Celsius (C) - mean={mean_val:.2f}")
    else:
        print(f"Inferred unit: Unknown - mean={mean_val:.2f}")

    t2m_data.close()
    os.unlink(temp_path)

    # 2. 检查降水数据
    print("\n2. 降水数据 (MSWEP)")
    print("-"*60)
    prcp_remote = f'/Projects/data_NRT/MSWX/tif/MSWEP_{test_date}.tif'
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
        ftp.retrbinary(f'RETR {prcp_remote}', temp_file.write)
        temp_path = temp_file.name

    prcp_data = rioxarray.open_rasterio(temp_path)
    print(f"File: {prcp_remote}")
    print(f"Dimensions: {prcp_data.dims}")
    print(f"Shape: {prcp_data.shape}")
    print(f"Coordinate ranges:")
    print(f"  - x (longitude): {prcp_data.x.min().values:.4f} to {prcp_data.x.max().values:.4f}")
    print(f"  - y (latitude): {prcp_data.y.min().values:.4f} to {prcp_data.y.max().values:.4f}")
    print(f"Resolution:")
    if len(prcp_data.x) > 1:
        print(f"  - x direction: {np.abs(prcp_data.x.values[1] - prcp_data.x.values[0]):.6f} degrees")
    if len(prcp_data.y) > 1:
        print(f"  - y direction: {np.abs(prcp_data.y.values[1] - prcp_data.y.values[0]):.6f} degrees")
    print(f"Data statistics:")
    print(f"  - Min: {prcp_data.values.min():.4f}")
    print(f"  - Max: {prcp_data.values.max():.4f}")
    print(f"  - Mean: {prcp_data.values.mean():.4f}")
    print(f"  - Median: {np.median(prcp_data.values):.4f}")
    print(f"  - Non-zero percentage: {(prcp_data.values > 0).sum() / prcp_data.values.size * 100:.2f}%")
    print(f"CRS: {prcp_data.rio.crs}")
    print(f"NoData value: {prcp_data.rio.nodata}")

    # 判断单位
    mean_val = prcp_data.values.mean()
    max_val = prcp_data.values.max()
    if max_val < 1:
        print(f"Inferred unit: m/day - mean={mean_val:.4f}, max={max_val:.4f}")
    elif max_val < 100:
        print(f"Inferred unit: mm/day - mean={mean_val:.4f}, max={max_val:.4f}")
    else:
        print(f"Inferred unit: unknown - mean={mean_val:.4f}, max={max_val:.4f}")

    prcp_data.close()
    os.unlink(temp_path)

    # 3. 检查相对湿度数据
    print("\n3. 相对湿度数据 (relative humidity)")
    print("-"*60)
    rh_remote = f'/Projects/data_NRT/MSWX/tif/relative humidity-{test_date}.tif'
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
        ftp.retrbinary(f'RETR {rh_remote}', temp_file.write)
        temp_path = temp_file.name

    rh_data = rioxarray.open_rasterio(temp_path)
    print(f"File: {rh_remote}")
    print(f"Dimensions: {rh_data.dims}")
    print(f"Shape: {rh_data.shape}")
    print(f"Coordinate ranges:")
    print(f"  - x (longitude): {rh_data.x.min().values:.4f} to {rh_data.x.max().values:.4f}")
    print(f"  - y (latitude): {rh_data.y.min().values:.4f} to {rh_data.y.max().values:.4f}")
    print(f"Resolution:")
    if len(rh_data.x) > 1:
        print(f"  - x direction: {np.abs(rh_data.x.values[1] - rh_data.x.values[0]):.6f} degrees")
    if len(rh_data.y) > 1:
        print(f"  - y direction: {np.abs(rh_data.y.values[1] - rh_data.y.values[0]):.6f} degrees")
    print(f"Data statistics:")
    print(f"  - Min: {rh_data.values.min():.4f}")
    print(f"  - Max: {rh_data.values.max():.4f}")
    print(f"  - Mean: {rh_data.values.mean():.4f}")
    print(f"  - Median: {np.median(rh_data.values):.4f}")
    print(f"CRS: {rh_data.rio.crs}")
    print(f"NoData value: {rh_data.rio.nodata}")

    # 判断单位
    mean_val = rh_data.values.mean()
    max_val = rh_data.values.max()
    if max_val <= 1:
        print(f"Inferred unit: fraction (0-1) - mean={mean_val:.4f}, max={max_val:.4f}")
    elif max_val <= 100:
        print(f"Inferred unit: percentage (0-100) - mean={mean_val:.4f}, max={max_val:.4f}")
    else:
        print(f"Inferred unit: unknown - mean={mean_val:.4f}, max={max_val:.4f}")

    rh_data.close()
    os.unlink(temp_path)

# 4. 对比ERA5数据的分辨率
print("\n4. 对比ERA5数据分辨率")
print("-"*60)
print("ERA5数据:")
print("  - 全球范围: 90°S-90°N, 0°E-360°E")
print("  - 分辨率: 0.25° x 0.25°")
print("  - 网格大小: 721 x 1440")

print("\n" + "="*60)
print("检查完成")
print("="*60)
