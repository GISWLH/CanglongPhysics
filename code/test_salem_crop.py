#!/usr/bin/env python3
"""
测试salem裁剪功能
"""
import xarray as xr
import salem
from pathlib import Path

def test_salem_crop():
    """测试salem使用shapefile裁剪NC文件"""
    
    shapefile_path = '/home/lhwang/Desktop/CanglongPhysics/code/data/china.shp'
    
    # 检查shapefile是否存在
    if not Path(shapefile_path).exists():
        print(f"错误：shapefile不存在 {shapefile_path}")
        return False
    
    # 创建测试数据
    print("创建测试数据...")
    lats = xr.DataArray(range(15, 55), dims='latitude', name='latitude')
    lons = xr.DataArray(range(70, 140), dims='longitude', name='longitude')
    temp_data = xr.DataArray(
        [[20 + i + j for j in range(len(lons))] for i in range(len(lats))],
        dims=['latitude', 'longitude'],
        coords={'latitude': lats, 'longitude': lons},
        name='temperature'
    )
    
    print(f"原始数据形状: {temp_data.shape}")
    
    try:
        # 读取shapefile
        print("读取shapefile...")
        shp_data = salem.read_shapefile(shapefile_path)
        print(f"Shapefile读取成功，包含 {len(shp_data)} 个几何体")
        
        # 使用salem进行裁剪
        print("使用salem进行roi裁剪...")
        cropped_data = temp_data.salem.roi(shape=shp_data)
        print(f"裁剪后数据形状: {cropped_data.shape}")
        
        # 检查是否成功裁剪
        if cropped_data.shape[0] > 0 and cropped_data.shape[1] > 0:
            print("✅ Salem裁剪测试成功!")
            return True
        else:
            print("❌ Salem裁剪失败：结果数据为空")
            return False
            
    except Exception as e:
        print(f"❌ Salem裁剪测试失败: {e}")
        return False

if __name__ == "__main__":
    test_salem_crop()