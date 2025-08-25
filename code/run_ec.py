import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import gcsfs
import zarr
import xarray

import ftplib
import os
from io import BytesIO
import xarray as xr

def upload_to_ftp(ftp_host, ftp_user, ftp_password, local_file_path, ftp_directory):
    """
    将文件上传到FTP服务器
    
    Parameters:
    -----------
    ftp_host : str
        FTP服务器地址
    ftp_user : str
        FTP用户名
    ftp_password : str
        FTP密码
    local_file_path : str
        本地文件路径
    ftp_directory : str
        FTP目标目录
    """
    try:
        # 连接到FTP服务器
        ftp = ftplib.FTP(ftp_host)
        ftp.login(user=ftp_user, passwd=ftp_password)
        
        # 切换到目标目录
        ftp.cwd(ftp_directory)
        
        # 获取文件名
        filename = os.path.basename(local_file_path)
        
        # 上传文件
        with open(local_file_path, 'rb') as file:
            ftp.storbinary(f'STOR {filename}', file)
        
        print(f"File {filename} uploaded successfully to {ftp_directory}")
        
        # 关闭FTP连接
        ftp.quit()
        return True
        
    except Exception as e:
        print(f"Error uploading file: {e}")
        return False
    
demo_start_time = '2025-08-06'
demo_end_time = '2025-08-19'

import rioxarray
from ftplib import FTP
import tempfile
import os

# FTP连接信息
ftp_host = "10.168.39.193"
ftp_user = "Longhao_WANG" 
ftp_password = "123456789"

# 连接FTP并下载文件到临时目录
with FTP(ftp_host) as ftp:
    ftp.login(ftp_user, ftp_password)
    
    # 远程文件路径
    remote_paths = [
        '/Projects/data_NRT/S2S/Control forecast/P/P_' + demo_end_time + '_weekly.tif',
        '/Projects/data_NRT/S2S/Control forecast/T/Tdew_' + demo_end_time + '_weekly.tif',
        '/Projects/data_NRT/S2S/Control forecast/T/Tavg_' + demo_end_time + '_weekly.tif'
    ]
    
    temp_paths = []
    for remote_path in remote_paths:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            # 下载文件
            ftp.retrbinary(f'RETR {remote_path}', temp_file.write)
            temp_paths.append(temp_file.name)

# 使用rioxarray读取数据        
data_prcp = rioxarray.open_rasterio(temp_paths[0])
data_d2m = rioxarray.open_rasterio(temp_paths[1]) 
data_t2m = rioxarray.open_rasterio(temp_paths[2])

# 删除临时文件
for temp_path in temp_paths:
    os.unlink(temp_path)

import xarray as xr
import pandas as pd
import numpy as np

# Create time coordinates for 6 weeks starting from demo_end_time
time_coords = pd.date_range(start=demo_end_time, periods=6, freq='7D')

# Reshape data into standard format
data_prcp_reshaped = data_prcp.values.reshape(6, data_prcp.y.size, data_prcp.x.size)
data_d2m_reshaped = data_d2m.values.reshape(6, data_d2m.y.size, data_d2m.x.size)
data_t2m_reshaped = data_t2m.values.reshape(6, data_t2m.y.size, data_t2m.x.size)

# Create new dataset with standard dimensions
standard_data = xr.Dataset(
    data_vars={
        'total_precipitation': (('time', 'latitude', 'longitude'), 
            data_prcp_reshaped),
        '2m_dewpoint_temperature': (('time', 'latitude', 'longitude'), 
            data_d2m_reshaped),
        '2m_temperature': (('time', 'latitude', 'longitude'), 
            data_t2m_reshaped)
    },
    coords={
        'time': time_coords,
        'latitude': data_prcp.y.values,  # Use .values to get just the coordinate values
        'longitude': data_prcp.x.values   # Use .values to get just the coordinate values
    }
)

# 计算潜在蒸散发
t2m_celsius = standard_data['2m_temperature'].values
d2m_celsius = standard_data['2m_dewpoint_temperature'].values

# 计算饱和水汽压和实际水汽压
es = 0.618 * np.exp(17.27 * t2m_celsius / (t2m_celsius + 237.3))
ea = 0.618 * np.exp(17.27 * d2m_celsius / (d2m_celsius + 237.3))

# 计算比率，避免除零错误
ratio_ea_es = np.full_like(t2m_celsius, np.nan)
valid_es_mask = es > 1e-9
ratio_ea_es[valid_es_mask] = ea[valid_es_mask] / es[valid_es_mask]
ratio_ea_es = np.clip(ratio_ea_es, None, 1.0)

# 计算PET
pet = 4.5 * np.power((1 + t2m_celsius / 25.0), 2) * (1 - ratio_ea_es)
pet = np.maximum(pet, 0)

# 添加到数据集
standard_data['potential_evapotranspiration'] = (('time', 'latitude', 'longitude'), pet)

import xarray as xr
import numpy as np

# Open the dataset
temp = xr.open_dataset('/data/lhwang/npy/temp.nc', engine='netcdf4')
climate = xr.open_dataset('/data/lhwang/npy/climate_variables_2000_2023_weekly.nc')

import xarray as xr
import pandas as pd

demo_start = (pd.to_datetime(demo_start_time) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')

data_inner_steps = 24
ds_surface = xr.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    chunks=None,
    consolidated=True
)[['large_scale_rain_rate', 'convective_rain_rate', '2m_dewpoint_temperature', '2m_temperature']]
surface_ds_former = ds_surface.sel(time=slice(demo_start, demo_end_time, data_inner_steps))
surface_ds_former.load()

# 更简单的方法：由于数据恰好是3周，直接分成三个7天
week1_data = surface_ds_former.isel(time=slice(0, 7))    # 第1-7天
week2_data = surface_ds_former.isel(time=slice(7, 14))   # 第8-14天
week3_data = surface_ds_former.isel(time=slice(14, 21))  # 第15-21天

# 计算每周的平均值
week1_mean = week1_data.mean(dim='time')
week2_mean = week2_data.mean(dim='time')
week3_mean = week3_data.mean(dim='time')

# 合并为一个新的数据集，包含三周的平均值
ds_former_means = xr.concat([week1_mean, week2_mean, week3_mean], 
                         dim=pd.DatetimeIndex([
                             pd.to_datetime(week1_data.time.values[0]),
                             pd.to_datetime(week2_data.time.values[0]),
                             pd.to_datetime(week3_data.time.values[0])
                         ], name='time'))

# 加载预测数据集
import xarray as xr
from datetime import datetime, timedelta

# 计算文件名的起止时间
end_date = datetime.strptime(demo_end_time, '%Y-%m-%d')
start_date = end_date + timedelta(days=1)
end_date = end_date + timedelta(weeks=6)

# 加载预测数据集
import xarray as xr
surface_ds = ds_former_means

# 处理温度数据 - 从开尔文转换为摄氏度
surface_ds['2m_temperature'] = surface_ds['2m_temperature'] - 273.15
surface_ds['2m_dewpoint_temperature'] = surface_ds['2m_dewpoint_temperature'] - 273.15

# 处理降水数据 - 从 m/hr 转换为 mm/day
precip_vars = ['large_scale_rain_rate', 'convective_rain_rate']
m_hr_to_mm_day_factor = 24.0 * 1000.0
for var in precip_vars:
    if var in surface_ds:
        surface_ds[var] = surface_ds[var].where(surface_ds[var] >= 0, 0)
        surface_ds[var] = surface_ds[var] * m_hr_to_mm_day_factor



surface_ds['total_precipitation'] = surface_ds['large_scale_rain_rate'] + surface_ds['convective_rain_rate']


t2m_celsius = surface_ds['2m_temperature'].values
d2m_celsius = surface_ds['2m_dewpoint_temperature'].values

# 计算饱和水汽压和实际水汽压
es = 0.618 * np.exp(17.27 * t2m_celsius / (t2m_celsius + 237.3))
ea = 0.618 * np.exp(17.27 * d2m_celsius / (d2m_celsius + 237.3))

# 计算比率，避免除零错误
ratio_ea_es = np.full_like(t2m_celsius, np.nan)
valid_es_mask = es > 1e-9
ratio_ea_es[valid_es_mask] = ea[valid_es_mask] / es[valid_es_mask]
ratio_ea_es = np.clip(ratio_ea_es, None, 1.0)

# 计算PET
pet = 4.5 * np.power((1 + t2m_celsius / 25.0), 2) * (1 - ratio_ea_es)
pet = np.maximum(pet, 0)

# 添加到数据集
surface_ds['potential_evapotranspiration'] = (('time', 'latitude', 'longitude'), pet)

# 使用列表语法正确提取多个变量
ds_sub = surface_ds[['total_precipitation', 'potential_evapotranspiration', '2m_temperature']]

# 6x6变粗平均采样
ds_sub_coarsened = ds_sub.coarsen(latitude=6, longitude=6, boundary='trim').mean()

# 截取到指定的经纬度范围
ds_sub_cropped = ds_sub_coarsened.sel(
    latitude=slice(54.0, 15.0),
    longitude=slice(70.5, 139.5)
)

# 将ds_sub_cropped的网格插值到与standard_data一致的网格
# 定义目标网格坐标（与standard_data一致）
target_lat = np.arange(54.0, 14.5, -1.5)  # 从54.0到15.0，步长1.5
target_lon = np.arange(70.5, 140.0, 1.5)  # 从70.5到139.5，步长1.5

# 使用xarray的interp方法进行双线性插值
ds_sub_cropped = ds_sub_cropped.interp(
    latitude=target_lat,
    longitude=target_lon,
    method='linear'
)

combined_ds = xr.concat(
    [ds_sub_cropped, standard_data[['total_precipitation', 'potential_evapotranspiration', '2m_temperature']]],
    dim='time'
)
# 对climate数据进行6x6变粗平均采样
climate_coarsened = climate.coarsen(lat=6, lon=6, boundary='trim').mean()

# 截取到指定的经纬度范围
climate_cropped = climate_coarsened.sel(
    lat=slice(54.0, 15.0),
    lon=slice(70.5, 139.5)
)

# 将climate_cropped的网格插值到与standard_data一致的网格
# 定义目标网格坐标（与standard_data一致）
target_lat = np.arange(54.0, 14.5, -1.5)  # 从54.0到15.0，步长1.5
target_lon = np.arange(70.5, 140.0, 1.5)  # 从70.5到139.5，步长1.5

# 使用xarray的interp方法进行双线性插值
climate_cropped = climate_cropped.interp(
    lat=target_lat,
    lon=target_lon,
    method='linear'
)

## 拟合函数SPEI
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_function
import pandas as pd

def calculate_pwm(series):
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
    if np.isnan(alpha) or x <= gamma_param:
        return 1e-9

    term = (alpha / (x - gamma_param))**beta
    if np.isinf(term) or term > 1e18:
        return 1e-9
        
    cdf_val = 1.0 / (1.0 + term)
    return np.clip(cdf_val, 1e-9, 1.0 - 1e-9)

def cdf_to_spei(P):
    if np.isnan(P): return np.nan
    if P <= 0.0: P = 1e-9
    if P >= 1.0: P = 1.0 - 1e-9

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

# 主程序
# 1. 计算历史D
D_hist = climate_cropped['tp'] - climate_cropped['pet']

# 2. 计算ds_sub的D
D_pred = combined_ds['total_precipitation'] - combined_ds['potential_evapotranspiration']
D_pred = D_pred.rename({'latitude': 'lat', 'longitude': 'lon'})

# 3. 自定义函数计算周号（按照固定的周定义：1月1-7日为第1周，1月8-14日为第2周...）
def get_week_of_year(date):
    # 计算一年中的周数，第一周为1月1-7日
    day_of_year = date.dt.dayofyear
    return ((day_of_year - 1) // 7) + 1

# 只计算第4周及之后的SPEI（即预测部分的SPEI）
start_pred_idx = 3  # 从第4个时间点开始计算SPEI
spei_pred_list = []
MIN_HIST_SAMPLES = 10

# 计算预测数据的周号
pred_week_numbers = get_week_of_year(D_pred.time)
# 计算历史数据的周号
hist_week_numbers = get_week_of_year(D_hist.time)

for i in range(start_pred_idx, len(D_pred.time)):
    # 计算当前时间点及前3个时间点的D值的累积
    curr_week_accum = 0
    for j in range(4):  # 累积4周的D值
        if i-j >= 0:  # 确保索引有效
            curr_week_accum += D_pred.isel(time=i-j)
    
    # 获取当前预测点的周号
    curr_week_num = pred_week_numbers.isel(time=i).item()
    
    # 提取历史数据中相同周号的数据
    hist_4week_accum_list = []
    
    # 按年份分组处理历史数据
    hist_years = np.unique(D_hist.time.dt.year)
    
    for year in hist_years:
        # 获取该年的数据
        year_data = D_hist.where(D_hist.time.dt.year == year, drop=True)
        year_weeks = hist_week_numbers.where(D_hist.time.dt.year == year, drop=True)
        
        # 找到当前周的索引
        week_indices = np.where(year_weeks == curr_week_num)[0]
        if len(week_indices) > 0:
            week_idx = week_indices[0]
            # 确保有足够的前置周
            if week_idx >= 3:  # 需要前3周的数据
                # 累积当前周和前3周的D值
                accum_D = 0
                for j in range(4):
                    accum_D += year_data.isel(time=week_idx-j)
                hist_4week_accum_list.append(accum_D)
    
    # 合并历史累积D值
    if hist_4week_accum_list:
        hist_4week_accum = xr.concat(hist_4week_accum_list, dim='time')
    else:
        # 如果没有找到历史数据，创建空DataArray
        hist_4week_accum = xr.DataArray(
            np.zeros((0,) + D_pred.isel(time=0).shape),
            coords={'time': [], **{dim: D_pred[dim] for dim in D_pred.dims if dim != 'time'}},
            dims=D_pred.dims
        )
    
    # 计算SPEI
    if len(hist_4week_accum.time) < MIN_HIST_SAMPLES:
        print(f"警告: 时间点 {D_pred.time.values[i]} (周 {curr_week_num}) 的历史样本数量不足: {len(hist_4week_accum.time)}")
        spei_map = xr.full_like(D_pred.isel(time=i), np.nan)
    elif np.isnan(curr_week_accum).all():
        spei_map = xr.full_like(D_pred.isel(time=i), np.nan)
    else:
        # 使用vectorize=True启用numpy的向量化处理
        spei_map = xr.apply_ufunc(
            calculate_spei_for_pixel,
            hist_4week_accum,
            curr_week_accum,
            input_core_dims=[['time'], []],
            output_core_dims=[[]],
            exclude_dims=set(('time',)),
            vectorize=True,
            output_dtypes=[float],
            keep_attrs=True
        )
    
    spei_pred_list.append(spei_map)

# 合并所有时间点的SPEI结果
spei_pred = xr.concat(spei_pred_list, dim='time')
# 只保留预测部分的时间坐标
spei_pred = spei_pred.assign_coords(time=D_pred.time[start_pred_idx:])

# 5. 可视化第一个预测时间点的SPEI
plt.figure(figsize=(10,4))
spei_pred.isel(time=0).plot(vmin=-2, vmax=2, cmap='RdBu')
plt.title(f"SPEI-4 {str(spei_pred.time.values[0])[:10]}")
plt.show()

# 计算 SPEI 并可视化中国区域6周SPEI
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmaps
import cartopy.crs as ccrs
import salem
import geopandas as gpd
import mplotutils as mpu
from utils import plot

# 设置全局字体为 Times New Roman
from matplotlib import font_manager
import os

font_path = "/usr/share/fonts/arial/ARIAL.TTF"
font_manager.fontManager.addfont(font_path)
font_name = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams['font.family'] = font_name


# 4. 选取中国区域
china_spei = spei_pred.sel(
    lon=slice(70, 140),
    lat=slice(55, 15)
)

# 5. 读取中国边界shapefile
china_shp = gpd.read_file('data/china.shp')

# 6. 设定色标和范围
vmin, vmax = -2, 2
unit_label = ''
title_prefix = 'ECMWF-S2S'
data_cmap = cmaps.BlueWhiteOrangeRed_r  # SPEI常用红蓝色卡

# 7. 创建图形和投影
fig = plt.figure(figsize=(42, 28))  # 或 (48, 32) 视分辨率和屏幕而定
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

# 8. 遍历6周
mappable = None
for t in range(6):
    ax = axes[t]
    current_data = china_spei.isel(time=t)
    # 使用salem创建掩膜，只显示中国陆地
    ds_t = salem.DataArrayAccessor(current_data)
    masked_data_t = ds_t.roi(shape=china_shp)
    
    # 绘图
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
    
    # 添加小地图（九段线）
    ax2 = fig.add_axes([0.222 + (t % 3) * 0.291, 0.0500 + (1 - t // 3) * 0.4800, 0.06, 0.09], 
                      projection=ccrs.LambertConformal(
                          central_longitude=105,
                          central_latitude=40,
                          standard_parallels=(25.0, 47.0)
                      ))
    plot.sub_china_map(masked_data_t, ax2, cmap=data_cmap, levels=levels, add_coastlines=False, add_land=False)
    
    # 时间标签 - 修改为显示一周的日期范围
    current_time = china_spei.time.values[t]
    start_date = np.datetime_as_string(current_time - np.timedelta64(6, 'D'), unit='D').replace('-', '')
    end_date = np.datetime_as_string(current_time, unit='D').replace('-', '')
    ax.set_title(f'{title_prefix} for {start_date}-{end_date}', fontsize=24, fontfamily='Arial')

# 9. 使用mpu添加色标
cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])  # 调整色标位置和大小
cbar = fig.colorbar(mappable, cax=cbar_ax)
cbar.set_label('SPEI', fontsize=24)
cbar.ax.tick_params(labelsize=24)  # 设置刻度标签字体大小

# 调整布局
plt.subplots_adjust(left=0.025, right=0.85, top=0.9, bottom=0.05, wspace=0.2, hspace=0.3)
mpu.set_map_layout(axes, width=80)
plt.show()

def save_and_upload_figure(fig, local_file_path, ftp_host, ftp_user, ftp_password, ftp_directory):
    """
    保存图片到本地并上传到FTP服务器
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        要保存的图片
    local_file_path : str
        本地保存路径（包含文件名）
    ftp_host : str
        FTP服务器地址
    ftp_user : str
        FTP用户名
    ftp_password : str
        FTP密码
    ftp_directory : str
        FTP目标目录
    """
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # 保存图片到本地文件
        fig.savefig(local_file_path)
        print(f"Figure saved locally to {local_file_path}")
        
        # 上传到FTP
        success = upload_to_ftp(
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_password=ftp_password,
            local_file_path=local_file_path,
            ftp_directory=ftp_directory
        )
        
        return success
        
    except Exception as e:
        print(f"Error saving and uploading figure: {e}")
        return False


# 构建文件名（使用第一周的开始日期和最后一周的结束日期）
start_date = np.datetime_as_string(china_spei.time.values[0] - np.timedelta64(6, 'D'), unit='D')
end_date = np.datetime_as_string(china_spei.time.values[-1], unit='D')
filename = f'EC_spei1_forecast_{start_date}_{end_date}.png'
local_file_path = os.path.join('../../data', filename)

# 保存并上传图片
ftp_host = "10.168.39.193"
ftp_user = "Longhao_WANG"
ftp_password = "123456789"  # 请替换为实际的密码
ftp_directory = '/Projects/data_NRT/Canglong/figure'

success = save_and_upload_figure(fig, local_file_path, ftp_host, ftp_user, ftp_password, ftp_directory)

input_end_date = datetime.strptime(demo_end_time, '%Y-%m-%d')
start_date1 = input_end_date + timedelta(days=1)
end_date = input_end_date + timedelta(days=6*7)
filename = f'../../data/EC_spei1_forecast_{start_date1.strftime("%Y-%m-%d")}_{end_date.strftime("%Y-%m-%d")}.nc'
spei_pred.to_netcdf(filename)
local_file_path = os.path.join('../../data', filename)
ftp_directory = '/Projects/data_NRT/Canglong'

def save_and_upload_dataset(dataset, local_file_path, ftp_host, ftp_user, ftp_password, ftp_directory):
    """
    将数据集保存到本地并上传到FTP服务器
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        要保存的数据集
    local_file_path : str
        本地保存路径（包含文件名）
    ftp_host : str
        FTP服务器地址
    ftp_user : str
        FTP用户名
    ftp_password : str
        FTP密码
    ftp_directory : str
        FTP目标目录
    """
    try:
        # 确保目标目录存在
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        # 保存数据集到本地文件
        dataset.to_netcdf(local_file_path)
        print(f"File saved locally to {local_file_path}")
        
        # 上传到FTP
        success = upload_to_ftp(
            ftp_host=ftp_host,
            ftp_user=ftp_user,
            ftp_password=ftp_password,
            local_file_path=local_file_path,
            ftp_directory=ftp_directory
        )
        
        return success
        
    except Exception as e:
        print(f"Error saving and uploading file: {e}")
        return False
    
# 保存并上传数据
success = save_and_upload_dataset(
    dataset=spei_pred,
    local_file_path=local_file_path,
    ftp_host=ftp_host,
    ftp_user=ftp_user,
    ftp_password=ftp_password,
    ftp_directory=ftp_directory
)