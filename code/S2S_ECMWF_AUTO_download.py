#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
import os
import datetime
from datetime import timedelta
import numpy as np
from osgeo import gdal

# 获取当前时间
time_info = (datetime.datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
print(f"两天前时间: {time_info}")

server = ECMWFDataServer()

#  以下为各变量的步长，不同变量的步长命名格式不一样，T_step是气温和露点温度的步长、P_step降水 ...
T_step = "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104"
P_step = "0/6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240/246/252/258/264/270/276/282/288/294/300/306/312/318/324/330/336/342/348/354/360/366/372/378/384/390/396/402/408/414/420/426/432/438/444/450/456/462/468/474/480/486/492/498/504/510/516/522/528/534/540/546/552/558/564/570/576/582/588/594/600/606/612/618/624/630/636/642/648/654/660/666/672/678/684/690/696/702/708/714/720/726/732/738/744/750/756/762/768/774/780/786/792/798/804/810/816/822/828/834/840/846/852/858/864/870/876/882/888/894/900/906/912/918/924/930/936/942/948/954/960/966/972/978/984/990/996/1002/1008/1014/1020/1026/1032/1038/1044/1050/1056/1062/1068/1074/1080/1086/1092/1098/1104"
TmaxTmin_step = "6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240/246/252/258/264/270/276/282/288/294/300/306/312/318/324/330/336/342/348/354/360/366/372/378/384/390/396/402/408/414/420/426/432/438/444/450/456/462/468/474/480/486/492/498/504/510/516/522/528/534/540/546/552/558/564/570/576/582/588/594/600/606/612/618/624/630/636/642/648/654/660/666/672/678/684/690/696/702/708/714/720/726/732/738/744/750/756/762/768/774/780/786/792/798/804/810/816/822/828/834/840/846/852/858/864/870/876/882/888/894/900/906/912/918/924/930/936/942/948/954/960/966/972/978/984/990/996/1002/1008/1014/1020/1026/1032/1038/1044/1050/1056/1062/1068/1074/1080/1086/1092/1098/1104"
Pa_step = "0/24/48/72/96/120/144/168/192/216/240/264/288/312/336/360/384/408/432/456/480/504/528/552/576/600/624/648/672/696/720/744/768/792/816/840/864/888/912/936/960/984/1008/1032/1056/1080/1104"

file_folder = r'G:\Thesis\Original data\key_RD_auto\data\S2S'

# 1. Control forecast
# 1.1 Tdew and Ta
if not os.path.exists(file_folder + '\Control forecast/T'):
    os.makedirs(file_folder + '\Control forecast/T')
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": time_info,   # 起报时间
    "expver": "prod",
    "area": "55/70/15/140",  # 覆盖范围
    "levtype": "sfc",
    "model": "glob",
    "origin": "ecmf",  # 模型名字
    "param": "168/167",  # 变量编码
    "step": T_step,  #  变量步长
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
    "target": file_folder + '\Control forecast\T/Ta_Tdew_' + time_info + '.grib',   #  存储路径
})
print(time_info,"的控制实验气温和露点气温下载完成",'1/8')

# 1.2 Pre
if not os.path.exists(file_folder + '\Control forecast/P'):
    os.makedirs(file_folder + '\Control forecast/P')
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": time_info,
    "expver": "prod",
    "area": "55/70/15/140",  # 覆盖范围
    "levtype": "sfc",
    "model": "glob",
    "origin": "ecmf",  # 模型名字
    "param": "228228",
    "step": P_step,
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
    "target": file_folder + '\Control forecast\P/P_' + time_info + '.grib',
})
print(time_info,"的控制实验降水下载完成",'2/8')

# 1.3 u10 and v10
if not os.path.exists(file_folder + '\Control forecast/U'):
    os.makedirs(file_folder + '\Control forecast/U')
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": time_info,
    "expver": "prod",
    "area": "55/70/15/140",  # 覆盖范围
    "levtype": "sfc",
    "model": "glob",
    "origin": "ecmf",  # 模型名字
    "param": "165/166",
    "step": P_step,
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
    "target": file_folder + '\Control forecast/U/U10_V10_' + time_info + '.grib',
})
print(time_info,"的控制实验风速下载完成",'3/8')

# 1.4 Tmax
if not os.path.exists(file_folder + '\Control forecast/T'):
    os.makedirs(file_folder + '\Control forecast/T')
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": time_info,
    "expver": "prod",
    "area": "55/70/15/140",  # 覆盖范围
    "levtype": "sfc",
    "model": "glob",
    "origin": "ecmf",  # 模型名字
    "param": "121",
    "step": TmaxTmin_step,
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
    "target": file_folder + '\Control forecast\T/Tmax_' + time_info + '.grib',
})
print(time_info,"的控制实验最高最低气温下载完成",'4/8')

# 1.5 Tmin
if not os.path.exists(file_folder + '\Control forecast/T'):
    os.makedirs(file_folder + '\Control forecast/T')
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": time_info,
    "expver": "prod",
    "area": "55/70/15/140",  # 覆盖范围
    "levtype": "sfc",
    "model": "glob",
    "origin": "ecmf",  # 模型名字
    "param": "122",
    "step": TmaxTmin_step,
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
    "target": file_folder + '\Control forecast\T/Tmin_' + time_info + '.grib',
})
print(time_info,"的控制实验最高最低气温下载完成",'5/8')

# 1.6 Pa
if not os.path.exists(file_folder + '\Control forecast/Pa'):
    os.makedirs(file_folder + '\Control forecast/Pa')
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": time_info,
    "expver": "prod",
    "area": "55/70/15/140",  # 覆盖范围
    "levtype": "sfc",
    "model": "glob",
    "origin": "ecmf",  # 模型名字
    "param": "134",
    "step": Pa_step,
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
    "target": file_folder + '\Control forecast\Pa/Pa_' + time_info + '.grib',
})
print(time_info,"的控制实验气压下载完成",'6/8')

# 1.7 Rnl Rns Rs
if not os.path.exists(file_folder + '\Control forecast/Rn'):
    os.makedirs(file_folder + '\Control forecast/Rn')
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": time_info,
    "expver": "prod",
    "area": "55/70/15/140",  # 覆盖范围
    "levtype": "sfc",
    "model": "glob",
    "origin": "ecmf",  # 模型名字
    "param": "176/177/169",
    "step": Pa_step,
    "stream": "enfo",
    "time": "00:00:00",
    "type": "cf",
    "target": file_folder + '\Control forecast\Rn/Rnl_Rns_Rs_' + time_info + '.grib',
})
print(time_info,"的控制实验辐射下载完成",'7/8')

# 2. # Perturbed forecast
# 2.1 Pre
if not os.path.exists(file_folder + '\Perturbed forecast/P'):
    os.makedirs(file_folder + '\Perturbed forecast/P')
server.retrieve({
    "class": "s2",
    "dataset": "s2s",
    "date": time_info,
    "expver": "prod",
    "area": "55/70/15/140",  # 覆盖范围
    "levtype": "sfc",
    "model": "glob",
    "number": "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50",
    "origin": "ecmf",  # 模型名字
    "param": "228228",
    "step": P_step,
    "stream": "enfo",
    "time": "00:00:00",
    "type": "pf",
    "target": file_folder + '\Perturbed forecast\P/P_' + time_info + '.grib',
})
print(time_info,"的扰动实验降水下载完成",'8/8')

# 3. grib to tif
date = time_info

##### daily value convert
# 打开 GRIB 文件
vars = ['P','Pa','Rn','T','U']

abbreviation_mapping = {
    'P':{
        'P':{
            'Total precipitation rate [kg/(m^2*s)]':'P'
        }
    },
    'Pa':{
        'Pa':{
            'Pressure [Pa]':'Pa'
        }
    },
    'Rn':{
        'Rnl_Rns_Rs':{
            'Net short wave radiation flux [W/(m^2)]': 'Rns',
            'Net long wave radiation flux [W/(m^2)]': 'Rnl',
            'Downward short-wave radiation flux [W/(m^2)]': 'Rs'
        }
    },
    'T':{
        'Ta_Tdew':{
            'Dew point temperature [C]': 'Tdew',
            'Temperature [C]': 'Tavg'
        },
        'Tmax':{
            'Temperature [C]':'Tmax'
        },
        'Tmin':{
            'Temperature [C]':'Tmin'
        }
    },
    'U':{
        'U10_V10':{
            'wind speed 2m': 'u2'
        }
    }
}

for folder, second_level_dict in abbreviation_mapping.items():
    print(folder)
    for file, third_level_dict in second_level_dict.items():
        print(file)
        for variable, abbr in third_level_dict.items():
            print(variable,abbr)

            # abbr == 'Tavg'
            grib_path = rf'G:\Thesis\Original data\key_RD_auto\data\S2S\Control forecast/{folder}/{file}_{date}.grib'
            output_tif_path = rf'G:\Thesis\Original data\key_RD_auto\data\S2S\Control forecast/{folder}/{abbr}_{date}.tif'

            grib_ds = gdal.Open(grib_path)
            days = range(1, 47)

            if grib_ds is not None:
                geotransform = grib_ds.GetGeoTransform()
                projection = grib_ds.GetProjection()

                # 获取第一个波段的数据形状，用于确定输出文件的大小
                first_band = grib_ds.GetRasterBand(1)
                data = first_band.ReadAsArray()
                rows, cols = data.shape

                # 创建新的 GeoTIFF 文件
                driver = gdal.GetDriverByName('GTiff')
                output_ds = driver.Create(output_tif_path, cols, rows, len(days), gdal.GDT_Float32)
                output_ds.SetGeoTransform(geotransform)
                output_ds.SetProjection(projection)

                band_count = grib_ds.RasterCount

                if folder != 'U':# 获取文件的总波段数
                    valid_bands = []  # 用于存储符合条件的波段索引

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
                        for day in days:
                            # 获取指定波段
                            start_num = 1 + (day - 1) * 4  # 每天起点数据
                            end_num = 1 + day * 4  # 每天终点数据
                            start_band = grib_ds.GetRasterBand(start_num)
                            end_band = grib_ds.GetRasterBand(end_num)
                            # 读取该波段的数据
                            start_data = start_band.ReadAsArray()
                            end_data = end_band.ReadAsArray()

                            day_data = np.maximum(end_data - start_data,0.0)   # 负值设为0
                            output_band = output_ds.GetRasterBand(day)

                            # 将计算得到的数据写入当前波段
                            output_band.WriteArray(day_data)

                            # 给波段命名
                            output_band.SetDescription(f'day_{day}')

                            # 刷新缓存并关闭数据集
                        output_ds.FlushCache()
                        output_ds = None
                        grib_ds = None

                    if abbr == 'Pa':
                        for day in days:
                            # 获取指定波段
                            band_num = day+1
                            band = grib_ds.GetRasterBand(band_num)
                            # 读取该波段的数据
                            day_data = band.ReadAsArray() / 1e3     #   转换为kPa
                            output_band = output_ds.GetRasterBand(day)
                            # 将计算得到的数据写入当前波段
                            output_band.WriteArray(day_data)
                            # 给波段命名
                            output_band.SetDescription(f'day_{day}')

                            # 刷新缓存并关闭数据集
                        output_ds.FlushCache()
                        output_ds = None
                        grib_ds = None
                    if abbr == 'Rns' or abbr == 'Rnl' or abbr == 'Rs':
                        for day in days:
                            print(day)
                            # 重新计算符合条件的起始和结束波段索引

                            start_num = valid_bands[0] + (day - 1) * 3
                            end_num = valid_bands[0] + day * 3

                            start_band = grib_ds.GetRasterBand(start_num)
                            end_band = grib_ds.GetRasterBand(end_num)

                            start_data = start_band.ReadAsArray()
                            end_data = end_band.ReadAsArray()

                            day_data = (end_data - start_data) / 86400           # 将W/m2的累计值转换为瞬时值
                            output_band = output_ds.GetRasterBand(day)

                            # 将计算得到的数据写入当前波段
                            output_band.WriteArray(day_data)

                            # 给波段命名
                            output_band.SetDescription(f'day_{day}')

                        # 刷新缓存并关闭数据集
                        output_ds.FlushCache()
                        output_ds = None
                        grib_ds = None
                    if abbr == 'Tdew' or abbr == 'Tavg':
                        for day in days:
                            print(day)
                            # 重新计算符合条件的起始和结束波段索引

                            band_num = valid_bands[0]+(day-1)*2
                            band = grib_ds.GetRasterBand(band_num)
                            # 读取该波段的数据
                            day_data = band.ReadAsArray()
                            output_band = output_ds.GetRasterBand(day)
                            # 将计算得到的数据写入当前波段
                            output_band.WriteArray(day_data)
                            # 给波段命名
                            output_band.SetDescription(f'day_{day}')

                        # 刷新缓存并关闭数据集
                        output_ds.FlushCache()
                        output_ds = None
                        grib_ds = None
                    if abbr == 'Tmax' or abbr == 'Tmin':
                        for day in days:
                            # 获取指定波段
                            start_num = valid_bands[0] + (day - 1) * 4  # 每天起点数据
                            end_num = valid_bands[0] + day * 4  # 每天终点数据

                            data = None
                            band_indices = np.zeros((rows, cols), dtype=np.float32)
                            #print(start_num, end_num)
                            for num in range(start_num, end_num):
                                current_band = grib_ds.GetRasterBand(num)
                                current_data = current_band.ReadAsArray()

                                if data is None:
                                    data = current_data
                                    band_indices[:] = band_num
                                else:
                                    if abbr == 'Tmax':
                                        mask = current_data > data
                                        data[mask] = current_data[mask]
                                        band_indices[mask] = num
                                    if abbr == 'Tmin':
                                        mask = current_data < data
                                        data[mask] = current_data[mask]
                                        band_indices[mask] = num
                                # 计算当前 day 的最大值和最小值在输出文件中的波段索引

                            # 获取输出文件的对应波段
                            output_band = output_ds.GetRasterBand(day)

                            # 将最大值和最小值数据写入输出文件的对应波段
                            output_band.WriteArray(data)

                            # 给波段命名
                            output_band.SetDescription(f'day_{day}')

                            # 刷新缓存并关闭数据集
                        output_ds.FlushCache()
                        output_ds = None
                        grib_ds = None

                elif folder == 'U':
                    valid_bands_u = []  # 用于存储符合条件的波段索引
                    valid_bands_v = []
                    for band_num in range(1, band_count + 1):
                        band = grib_ds.GetRasterBand(band_num)
                        metadata = band.GetMetadata()
                        match_u = True
                        for key, value in {'GRIB_COMMENT': 'u-component of wind [m/s]'}.items():
                            if key not in metadata or metadata[key] != value:
                                match_u = False
                                break
                        if match_u:
                            valid_bands_u.append(band_num)

                        match_v = True
                        for key, value in {'GRIB_COMMENT': 'v-component of wind [m/s]'}.items():
                            if key not in metadata or metadata[key] != value:
                                match_v = False
                                break
                        if match_v:
                            valid_bands_v.append(band_num)

                    for day in days:
                        print(day)
                        # 重新计算符合条件的起始和结束波段索引

                        time_nums_u = [valid_bands_u[i] + (day - 1) * 8 for i in range(1, 5)]
                        time_nums_v = [valid_bands_v[i] + (day - 1) * 8 for i in range(1, 5)]
                        # 存储各个波段的数据v
                        data_list_u = []
                        data_list_v = []
                        for time_num_u in time_nums_u:
                            band_u = grib_ds.GetRasterBand(time_num_u)
                            data_u = band_u.ReadAsArray()
                            data_list_u.append(data_u)

                        for time_num_v in time_nums_v:
                            band_v = grib_ds.GetRasterBand(time_num_v)
                            data_v = band_v.ReadAsArray()
                            data_list_v.append(data_v)

                        mean_data_u = np.mean(data_list_u, axis=0)
                        mean_data_v = np.mean(data_list_v, axis=0)

                        u10 = np.sqrt(mean_data_u ** 2 + mean_data_v ** 2)
                        u2 = u10 * 4.87 / np.log(67.8 * 10 - 5.42)
                        output_band = output_ds.GetRasterBand(day)
                        # 将计算得到的数据写入当前波段
                        output_band.WriteArray(u2)

                        # 给波段命名
                        output_band.SetDescription(f'day_{day}')

                    # 刷新缓存并关闭数据集
                    output_ds.FlushCache()
                    output_ds = None
                    grib_ds = None


##### weekly convert
for folder, second_level_dict in abbreviation_mapping.items():
    print(folder)
    for file, third_level_dict in second_level_dict.items():
        print(file)
        for variable, abbr in third_level_dict.items():
            print(variable,abbr)

            # abbr == 'Tavg'
            grib_path = rf'G:\Thesis\Original data\key_RD_auto\data\S2S\Control forecast/{folder}/{file}_{date}.grib'
            output_tif_path = rf'G:\Thesis\Original data\key_RD_auto\data\S2S\Control forecast/{folder}/{abbr}_{date}_weekly.tif'

            grib_ds = gdal.Open(grib_path)
            weeks = range(1, 7)

            if grib_ds is not None:
                geotransform = grib_ds.GetGeoTransform()
                projection = grib_ds.GetProjection()

                # 获取第一个波段的数据形状，用于确定输出文件的大小
                first_band = grib_ds.GetRasterBand(1)
                data = first_band.ReadAsArray()
                rows, cols = data.shape

                # 创建新的 GeoTIFF 文件
                driver = gdal.GetDriverByName('GTiff')
                output_ds = driver.Create(output_tif_path, cols, rows, len(weeks), gdal.GDT_Float32)
                output_ds.SetGeoTransform(geotransform)
                output_ds.SetProjection(projection)

                band_count = grib_ds.RasterCount

                if folder != 'U':# 获取文件的总波段数
                    valid_bands = []  # 用于存储符合条件的波段索引

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
                            # 获取指定波段
                            start_num = valid_bands[0] + (week - 1) * 28  # 每天起点数据
                            end_num = 1 + week * 28  # 每天终点数据
                            start_band = grib_ds.GetRasterBand(start_num)
                            end_band = grib_ds.GetRasterBand(end_num)
                            # 读取该波段的数据
                            start_data = start_band.ReadAsArray()
                            end_data = end_band.ReadAsArray()

                            week_data = np.maximum(end_data - start_data,0.0)    # 负值设为0
                            output_band = output_ds.GetRasterBand(week)

                            # 将计算得到的数据写入当前波段
                            output_band.WriteArray(week_data)

                            # 给波段命名
                            output_band.SetDescription(f'week_{week}')

                            # 刷新缓存并关闭数据集
                        output_ds.FlushCache()
                        output_ds = None
                        grib_ds = None

                    if abbr == 'Pa':
                        for week in weeks:
                            # 获取指定波段
                            start_num = valid_bands[1]+(week-1)*7
                            end_num = valid_bands[1] + week * 7

                            sum_data = None
                            band_count = 0
                            for band_num in range(start_num,end_num):
                                band = grib_ds.GetRasterBand(band_num)
                                current_data = band.ReadAsArray()
                                if sum_data is None:
                                    sum_data = current_data
                                else:
                                    sum_data += current_data
                                band_count += 1

                            if band_count > 0:
                                week_data = sum_data / band_count
                            output_band = output_ds.GetRasterBand(week)
                            # 将计算得到的数据写入当前波段
                            output_band.WriteArray(week_data / 1e3)   #   转换为kPa
                            # 给波段命名
                            output_band.SetDescription(f'week_{week}')

                            # 刷新缓存并关闭数据集
                        output_ds.FlushCache()
                        output_ds = None
                        grib_ds = None
                    if abbr == 'Rns' or abbr == 'Rnl' or abbr == 'Rs':
                        for week in weeks:
                            # 重新计算符合条件的起始和结束波段索引
                            start_num = valid_bands[0] + (week - 1) * 21
                            end_num = valid_bands[0] + week * 21

                            start_band = grib_ds.GetRasterBand(start_num)
                            end_band = grib_ds.GetRasterBand(end_num)

                            start_data = start_band.ReadAsArray()
                            end_data = end_band.ReadAsArray()

                            week_data = (end_data - start_data) / 7 / 86400   # 转换为W/m2瞬时值
                            output_band = output_ds.GetRasterBand(week)

                            # 将计算得到的数据写入当前波段
                            output_band.WriteArray(week_data)

                            # 给波段命名
                            output_band.SetDescription(f'week_{week}')

                        # 刷新缓存并关闭数据集
                        output_ds.FlushCache()
                        output_ds = None
                        grib_ds = None
                    if abbr == 'Tdew' or abbr == 'Tavg':
                        for week in weeks:
                            start_num = valid_bands[0]+(week-1)*14
                            end_num = valid_bands[0]+week*14-1

                            sum_data = None
                            band_count = 0
                            for band_num in range(start_num, end_num):
                                band = grib_ds.GetRasterBand(band_num)
                                current_data = band.ReadAsArray()
                                if sum_data is None:
                                    sum_data = current_data
                                else:
                                    sum_data += current_data
                                band_count += 1

                            if band_count > 0:
                                week_data = sum_data / band_count

                            # 读取该波段的数据
                            week_data = band.ReadAsArray()
                            output_band = output_ds.GetRasterBand(week)
                            # 将计算得到的数据写入当前波段
                            output_band.WriteArray(week_data)
                            # 给波段命名
                            output_band.SetDescription(f'week_{week}')

                        # 刷新缓存并关闭数据集
                        output_ds.FlushCache()
                        output_ds = None
                        grib_ds = None
                    if abbr == 'Tmax' or abbr == 'Tmin':
                        for week in weeks:
                            # 获取指定波段
                            start_num = valid_bands[0] + (week - 1) * 28  # 每天起点数据
                            end_num = valid_bands[0] + week * 28  # 每天终点数据

                            data = None
                            band_indices = np.zeros((rows, cols), dtype=np.float32)
                            for num in range(start_num, end_num+1):
                                current_band = grib_ds.GetRasterBand(num)
                                current_data = current_band.ReadAsArray()

                                if data is None:
                                    data = current_data
                                    band_indices[:] = band_num
                                else:
                                    if abbr == 'Tmax':
                                        mask = current_data > data
                                        data[mask] = current_data[mask]
                                        band_indices[mask] = num
                                    if abbr == 'Tmin':
                                        mask = current_data < data
                                        data[mask] = current_data[mask]
                                        band_indices[mask] = num
                                # 计算当前 day 的最大值和最小值在输出文件中的波段索引

                            # 获取输出文件的对应波段
                            output_band = output_ds.GetRasterBand(week)

                            # 将最大值和最小值数据写入输出文件的对应波段
                            output_band.WriteArray(data)

                            # 给波段命名
                            output_band.SetDescription(f'week_{week}')

                            # 刷新缓存并关闭数据集
                        output_ds.FlushCache()
                        output_ds = None
                        grib_ds = None

                elif folder == 'U':
                    valid_bands_u = []  # 用于存储符合条件的波段索引
                    valid_bands_v = []
                    for band_num in range(1, band_count + 1):
                        band = grib_ds.GetRasterBand(band_num)
                        metadata = band.GetMetadata()
                        match_u = True
                        for key, value in {'GRIB_COMMENT': 'u-component of wind [m/s]'}.items():
                            if key not in metadata or metadata[key] != value:
                                match_u = False
                                break
                        if match_u:
                            valid_bands_u.append(band_num)

                        match_v = True
                        for key, value in {'GRIB_COMMENT': 'v-component of wind [m/s]'}.items():
                            if key not in metadata or metadata[key] != value:
                                match_v = False
                                break
                        if match_v:
                            valid_bands_v.append(band_num)

                    for week in weeks:
                        start_num_u = valid_bands_u[1]+(week-1)*56
                        end_num_u = valid_bands_u[1]+week*56-1

                        start_num_v = valid_bands_v[1]+(week-1)*56
                        end_num_v = valid_bands_v[1]+week*56-1

                        sum_data_u = None
                        band_count_u = 0
                        for band_num_u in range(start_num_u, end_num_u):
                            band_u = grib_ds.GetRasterBand(band_num_u)
                            current_data_u = band_u.ReadAsArray()
                            if sum_data_u is None:
                                sum_data_u = current_data_u
                            else:
                                sum_data_u += current_data_u
                            band_count_u += 1

                        if band_count_u > 0:
                            week_data_u = sum_data_u / band_count_u

                        sum_data_v = None
                        band_count_v = 0
                        for band_num_v in range(start_num_v, end_num_v):
                            band_v = grib_ds.GetRasterBand(band_num_v)
                            current_data_v = band_v.ReadAsArray()
                            if sum_data_v is None:
                                sum_data_v = current_data_v
                            else:
                                sum_data_v += current_data_v
                            band_count_v += 1

                        if band_count_v > 0:
                            week_data_v = sum_data_v / band_count_v

                        u10 = np.sqrt(week_data_u ** 2 + week_data_v ** 2)
                        u2 = u10 * 4.87 / np.log(67.8 * 10 - 5.42)
                        output_band = output_ds.GetRasterBand(week)
                        # 将计算得到的数据写入当前波段
                        output_band.WriteArray(u2)

                        # 给波段命名
                        output_band.SetDescription(f'week_{week}')

                    # 刷新缓存并关闭数据集
                    output_ds.FlushCache()
                    output_ds = None
                    grib_ds = None

##### daily perturbed value convert
# 打开 GRIB 文件
vars = ['P']

abbreviation_mapping = {
    'P':{
        'P':{
            'Total precipitation rate [kg/(m^2*s)]':'P'
        }
    }
}

for folder, second_level_dict in abbreviation_mapping.items():
    print(folder)
    for file, third_level_dict in second_level_dict.items():
        print(file)
        for variable, abbr in third_level_dict.items():
            print(variable,abbr)

            # abbr == 'Tavg'
            grib_path = rf'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P/{file}_{date}.grib'

            grib_ds = gdal.Open(grib_path)
            days = range(1, 47)

            if grib_ds is not None:
                geotransform = grib_ds.GetGeoTransform()
                projection = grib_ds.GetProjection()

                # 获取第一个波段的数据形状，用于确定输出文件的大小
                first_band = grib_ds.GetRasterBand(1)
                data = first_band.ReadAsArray()
                rows, cols = data.shape
                band_count = grib_ds.RasterCount

                # 成员数循环遍历
                for num in range(1,51):
                    # 创建新的 GeoTIFF 文件
                    output_tif_path = rf'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P/tif/{abbr}_{date}_pert_num_' + str(num) + '.tif'

                    driver = gdal.GetDriverByName('GTiff')
                    output_ds = driver.Create(output_tif_path, cols, rows, len(days), gdal.GDT_Float32)
                    output_ds.SetGeoTransform(geotransform)
                    output_ds.SetProjection(projection)

                    for day in days:
                        # 获取指定波段
                        start_num = num + (day - 1) * 4 * 50  # 每天起点数据
                        end_num = num + day * 4 * 50  # 每天终点数据
                        start_band = grib_ds.GetRasterBand(start_num)
                        end_band = grib_ds.GetRasterBand(end_num)
                        # 读取该波段的数据
                        start_data = start_band.ReadAsArray()
                        end_data = end_band.ReadAsArray()

                        day_data = np.maximum(end_data - start_data,0.0)   # 负值设为0
                        output_band = output_ds.GetRasterBand(day)

                        # 将计算得到的数据写入当前波段
                        output_band.WriteArray(day_data)

                        # 给波段命名
                        output_band.SetDescription(f'day_{day}')

                        # 刷新缓存并关闭数据集
                    output_ds.FlushCache()
                    output_ds = None
                    pass
                grib_ds = None

##### weekly convert
for folder, second_level_dict in abbreviation_mapping.items():
    print(folder)
    for file, third_level_dict in second_level_dict.items():
        print(file)
        for variable, abbr in third_level_dict.items():
            print(variable,abbr)

            # abbr == 'Tavg'
            grib_path = rf'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P/{file}_{date}.grib'

            grib_ds = gdal.Open(grib_path)
            weeks = range(1, 7)

            if grib_ds is not None:
                geotransform = grib_ds.GetGeoTransform()
                projection = grib_ds.GetProjection()

                # 获取第一个波段的数据形状，用于确定输出文件的大小
                first_band = grib_ds.GetRasterBand(1)
                data = first_band.ReadAsArray()
                rows, cols = data.shape
                band_count = grib_ds.RasterCount

                for num in range(1,51):
                    output_tif_path = rf'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P/tif/{abbr}_{date}_pert_num_' + str(num) + '_weekly.tif'

                    # 创建新的 GeoTIFF 文件
                    driver = gdal.GetDriverByName('GTiff')
                    output_ds = driver.Create(output_tif_path, cols, rows, len(weeks), gdal.GDT_Float32)
                    output_ds.SetGeoTransform(geotransform)
                    output_ds.SetProjection(projection)

                    for week in weeks:
                        # 获取指定波段
                        start_num = num + (week - 1) * 28 * 50  # 每天起点数据
                        end_num = num + week * 28 * 50  # 每天终点数据
                        start_band = grib_ds.GetRasterBand(start_num)
                        end_band = grib_ds.GetRasterBand(end_num)
                        # 读取该波段的数据
                        start_data = start_band.ReadAsArray()
                        end_data = end_band.ReadAsArray()

                        week_data = np.maximum(end_data - start_data,0.0)    # 负值设为0
                        output_band = output_ds.GetRasterBand(week)

                        # 将计算得到的数据写入当前波段
                        output_band.WriteArray(week_data)

                        # 给波段命名
                        output_band.SetDescription(f'week_{week}')

                        # 刷新缓存并关闭数据集
                    output_ds.FlushCache()
                    output_ds = None
                    pass
                grib_ds = None

##### calculate median and std daily
cont = gdal.Open(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Control forecast\P/P_' + date + '.tif')

driver_median = gdal.GetDriverByName('GTiff')
ds_median = driver_median.Create(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_' + date + '_median.tif', cols, rows, 46, gdal.GDT_Float32)
ds_median.SetGeoTransform(geotransform)
ds_median.SetProjection(projection)

driver_std = gdal.GetDriverByName('GTiff')
ds_std = driver_std.Create(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_' + date + '_std.tif', cols, rows, 46, gdal.GDT_Float32)
ds_std.SetGeoTransform(geotransform)
ds_std.SetProjection(projection)

driver_std_positive = gdal.GetDriverByName('GTiff')
ds_std_positive = driver_std_positive.Create(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_' + date + '_median+std.tif', cols, rows, 46, gdal.GDT_Float32)
ds_std_positive.SetGeoTransform(geotransform)
ds_std_positive.SetProjection(projection)

driver_std_negative = gdal.GetDriverByName('GTiff')
ds_std_negative = driver_std_negative.Create(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_' + date + '_median-std.tif', cols, rows, 46, gdal.GDT_Float32)
ds_std_negative.SetGeoTransform(geotransform)
ds_std_negative.SetProjection(projection)

for i in range(1,47):
    daily_mat = np.zeros((51, rows, cols))
    for j in range(1,51):
        pert = gdal.Open(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_' + date + '_pert_num_' + str(j) + '.tif')
        daily_mat[j-1,:,:] = pert.GetRasterBand(i).ReadAsArray()
        pass
    daily_mat[50,:,:] = cont.GetRasterBand(1).ReadAsArray()

    daily_median = np.nanmedian(daily_mat,axis=0)
    daily_std = np.nanstd(daily_mat,axis=0)

    median_band = ds_median.GetRasterBand(i)
    median_band.WriteArray(daily_median)

    std_band = ds_std.GetRasterBand(i)
    std_band.WriteArray(daily_std)

    std_positive_band = ds_std_positive.GetRasterBand(i)
    std_positive_band.WriteArray(daily_median + daily_std)

    std_negative_band = ds_std_negative.GetRasterBand(i)
    std_negative_band.WriteArray(np.maximum(0.0,daily_median - daily_std))

    pass
ds_median.FlushCache()
ds_median = None

ds_std.FlushCache()
ds_std = None

ds_std_positive.FlushCache()
ds_std_positive = None

ds_std_negative.FlushCache()
ds_std_negative = None

##### calculate median and std weekly
cont = gdal.Open(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Control forecast\P/P_' + date + '_weekly.tif')

driver_median = gdal.GetDriverByName('GTiff')
ds_median = driver_median.Create(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_' + date + '_median_weekly.tif', cols, rows, 46, gdal.GDT_Float32)
ds_median.SetGeoTransform(geotransform)
ds_median.SetProjection(projection)

driver_std = gdal.GetDriverByName('GTiff')
ds_std = driver_std.Create(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_' + date + '_std_weekly.tif', cols, rows, 46, gdal.GDT_Float32)
ds_std.SetGeoTransform(geotransform)
ds_std.SetProjection(projection)

driver_std_positive = gdal.GetDriverByName('GTiff')
ds_std_positive = driver_std_positive.Create(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_' + date + '_median+std_weekly.tif', cols, rows, 46, gdal.GDT_Float32)
ds_std_positive.SetGeoTransform(geotransform)
ds_std_positive.SetProjection(projection)

driver_std_negative = gdal.GetDriverByName('GTiff')
ds_std_negative = driver_std_negative.Create(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_' + date + '_median-std_weekly.tif', cols, rows, 46, gdal.GDT_Float32)
ds_std_negative.SetGeoTransform(geotransform)
ds_std_negative.SetProjection(projection)

for i in range(1,7):
    weekly_mat = np.zeros((51, rows, cols))
    for j in range(1,51):
        pert = gdal.Open(r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_' + date + '_pert_num_' + str(j) + '_weekly.tif')
        weekly_mat[j-1,:,:] = pert.GetRasterBand(i).ReadAsArray()
        pass
    weekly_mat[50,:,:] = cont.GetRasterBand(1).ReadAsArray()

    weekly_median = np.nanmedian(weekly_mat,axis=0)
    weekly_std = np.nanstd(weekly_mat,axis=0)

    median_band = ds_median.GetRasterBand(i)
    median_band.WriteArray(weekly_median)

    std_band = ds_std.GetRasterBand(i)
    std_band.WriteArray(weekly_std)

    std_positive_band = ds_std_positive.GetRasterBand(i)
    std_positive_band.WriteArray(weekly_median + weekly_std)

    std_negative_band = ds_std_negative.GetRasterBand(i)
    std_negative_band.WriteArray(np.maximum(0.0,weekly_median - weekly_std))

    pass
ds_median.FlushCache()
ds_median = None

ds_std.FlushCache()
ds_std = None

ds_std_positive.FlushCache()
ds_std_positive = None

ds_std_negative.FlushCache()
ds_std_negative = None

##### 删除扰动预报的周平均tif
for i in range(1,51):
    file_path = r'G:\Thesis\Original data\key_RD_auto\data\S2S\Perturbed forecast\P\tif/P_'+ date+'_pert_num_' + str(i) + '_weekly.tif'
    if os.path.exists(file_path):
        try:
            os.remove(file_path)  # 删除文件
            print(f"文件 {file_path} 已删除")
        except Exception as e:
            print(f"删除失败: {str(e)}")
    else:
        print("文件不存在")