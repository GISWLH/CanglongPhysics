#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
import os
import datetime
from datetime import timedelta
import numpy as np
from osgeo import gdal

# 手动设置起报日期
time_info = '2025-07-09'
print(f"起报日期: {time_info}")

server = ECMWFDataServer()

# T和P变量的步长
T_step = "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104"
P_step = "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240/246/252/258/264/270/276/282/288/294/300/306/312/318/324/330/336/342/348/354/360/366/372/378/384/390/396/402/408/414/420/426/432/438/444/450/456/462/468/474/480/486/492/498/504/510/516/522/528/534/540/546/552/558/564/570/576/582/588/594/600/606/612/618/624/630/636/642/648/654/660/666/672/678/684/690/696/702/708/714/720/726/732/738/744/750/756/762/768/774/780/786/792/798/804/810/816/822/828/834/840/846/852/858/864/870/876/882/888/894/900/906/912/918/924/930/936/942/948/954/960/966/972/978/984/990/996/1002/1008/1014/1020/1026/1032/1038/1044/1050/1056/1062/1068/1074/1080/1086/1092/1098/1104"

# 创建输出目录
if not os.path.exists('../data/ecmwf/T'):
    os.makedirs('../data/ecmwf/T')
if not os.path.exists('../data/ecmwf/P'):
    os.makedirs('../data/ecmwf/P')

# 1. 下载温度 (Ta_Tdew)
print("开始下载温度数据...")
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": time_info,
    "expver": "prod",
    "area": "55/70/15/140",
    "levtype": "sfc",
    "model": "glob",
    "origin": "ecmf",
    "param": "168/167",
    "step": T_step,
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
    "target": f"../data/ecmwf/T/Ta_Tdew_{time_info}.grib",
})
print(f"{time_info}的温度数据下载完成")

# 2. 下载降水
print("开始下载降水数据...")
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": time_info,
    "expver": "prod",
    "area": "55/70/15/140",
    "levtype": "sfc",
    "model": "glob",
    "origin": "ecmf",
    "param": "228228",
    "step": P_step,
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
    "target": f"../data/ecmwf/P/P_{time_info}.grib",
})
print(f"{time_info}的降水数据下载完成")

# 3. 转换为weekly tif格式
print("开始转换为weekly tif格式...")

# 变量映射
abbreviation_mapping = {
    'T': {
        'Ta_Tdew': {
            'Dew point temperature [C]': 'Tdew',
            'Temperature [C]': 'Tavg'
        }
    },
    'P': {
        'P': {
            'Total precipitation rate [kg/(m^2*s)]': 'P'
        }
    }
}

for folder, second_level_dict in abbreviation_mapping.items():
    print(f"处理{folder}数据")
    for file, third_level_dict in second_level_dict.items():
        for variable, abbr in third_level_dict.items():
            print(f"处理变量: {variable} -> {abbr}")
            
            grib_path = f"../data/ecmwf/{folder}/{file}_{time_info}.grib"
            output_tif_path = f"../data/ecmwf/{folder}/{abbr}_{time_info}_weekly.tif"
            
            grib_ds = gdal.Open(grib_path)
            weeks = range(1, 7)
            
            if grib_ds is not None:
                geotransform = grib_ds.GetGeoTransform()
                projection = grib_ds.GetProjection()
                
                first_band = grib_ds.GetRasterBand(1)
                data = first_band.ReadAsArray()
                rows, cols = data.shape
                
                # 创建新的 GeoTIFF 文件
                driver = gdal.GetDriverByName('GTiff')
                output_ds = driver.Create(output_tif_path, cols, rows, len(weeks), gdal.GDT_Float32)
                output_ds.SetGeoTransform(geotransform)
                output_ds.SetProjection(projection)
                
                band_count = grib_ds.RasterCount
                
                # 获取符合条件的波段索引
                valid_bands = []
                for band_num in range(1, band_count + 1):
                    band = grib_ds.GetRasterBand(band_num)
                    metadata = band.GetMetadata()
                    match = True
                    for key, value in {'GRIB_COMMENT': variable}.items():
                        if key not in metadata or metadata[key] != value:
                            match = False
                            break
                    if match:
                        valid_bands.append(band_num)
                
                if abbr == 'P':
                    for week in weeks:
                        start_num = valid_bands[0] + (week - 1) * 28
                        end_num = 1 + week * 28
                        start_band = grib_ds.GetRasterBand(start_num)
                        end_band = grib_ds.GetRasterBand(end_num)
                        
                        start_data = start_band.ReadAsArray()
                        end_data = end_band.ReadAsArray()
                        
                        week_data = np.maximum(end_data - start_data, 0.0)
                        output_band = output_ds.GetRasterBand(week)
                        output_band.WriteArray(week_data)
                        output_band.SetDescription(f'week_{week}')
                
                elif abbr == 'Tdew' or abbr == 'Tavg':
                    for week in weeks:
                        start_num = valid_bands[0] + (week - 1) * 14
                        end_num = valid_bands[0] + week * 14 - 1
                        
                        sum_data = None
                        band_count_sum = 0
                        for band_num in range(start_num, end_num):
                            band = grib_ds.GetRasterBand(band_num)
                            current_data = band.ReadAsArray()
                            if sum_data is None:
                                sum_data = current_data
                            else:
                                sum_data += current_data
                            band_count_sum += 1
                        
                        if band_count_sum > 0:
                            week_data = sum_data / band_count_sum
                        
                        output_band = output_ds.GetRasterBand(week)
                        output_band.WriteArray(week_data)
                        output_band.SetDescription(f'week_{week}')
                
                output_ds.FlushCache()
                output_ds = None
                grib_ds = None
                
                print(f"{abbr}的weekly tif文件已生成: {output_tif_path}")

print("所有数据处理完成！")