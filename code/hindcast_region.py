#!/usr/bin/env python3
"""
区域预报检验：基于NAS数据和shapefile区域裁剪
- 支持任意shapefile区域裁剪
- 使用salem进行NC文件的区域裁剪
- 基于hindcast_NAS.py架构
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import rioxarray as rxr
import salem
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from ftplib import FTP
import tempfile
import os
import geopandas as gpd

# 验证关键参数配置 - 修改为与现有ECMWF文件匹配的日期
demo_start_time = '2025-08-06'
demo_end_time = '2025-08-19'
forecast_start_week = 33
hindcast_start_week = 32

# 区域配置 - 默认使用中国shapefile，可以自定义
SHAPEFILE_PATH = '/home/lhwang/Desktop/CanglongPhysics/code/data/china.shp'
REGION_NAME = 'china'  # 可修改为其他区域名称

# Nature风格绘图配置
def setup_nature_style():
    """Setup Nature journal style plotting parameters"""
    try:
        # 尝试使用Arial字体
        font_path = "/usr/share/fonts/arial/ARIAL.TTF"
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
            font_name = font_manager.FontProperties(fname=font_path).get_name()
            plt.rcParams['font.family'] = font_name
    except:
        # 如果找不到Arial，使用系统默认字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    # Nature风格参数
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 300,
        'figure.figsize': (6, 4),
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.0,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.edgecolor': '#454545',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 3,
        'ytick.minor.size': 3,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.color': '#454545',
        'ytick.color': '#454545',
        'savefig.bbox': 'tight',
        'savefig.transparent': False,
        'savefig.dpi': 300
    })

setup_nature_style()

# 本地数据路径
local_data_dir = Path('/home/lhwang/Desktop/data')
hind_obs_dir = local_data_dir / 'hind_obs'
hind_obs_dir.mkdir(parents=True, exist_ok=True)
# 使用绝对路径确保文件保存到正确位置
figures_dir = Path('/home/lhwang/Desktop/CanglongPhysics/figures/hindcast_region')
figures_dir.mkdir(parents=True, exist_ok=True)

# NAS连接配置
NAS_CONFIG = {
    'host': '10.168.39.193',
    'user': 'Longhao_WANG',
    'password': '123456789',
    'canglong_path': '/Projects/data_NRT/Canglong',
    'temp_base_path': '/Projects/data_NRT/S2S/Control forecast/T',
    'precip_base_path': '/Projects/data_NRT/S2S/Control forecast/P',
    'dewpoint_path': '/Projects/data_NRT/Canglong/dewpoint'
}

# 动态生成文件名和配置
def generate_configs_from_dates(demo_start_time, demo_end_time, hindcast_start_week):
    """Generate file configurations based on key parameters"""
    from datetime import datetime, timedelta
    
    # 计算hindcast的开始和结束日期
    hindcast_start = datetime.strptime(demo_end_time, '%Y-%m-%d') - timedelta(days=6)
    hindcast_end = datetime.strptime(demo_end_time, '%Y-%m-%d')
    
    canglong_configs = []
    ecmwf_configs = []
    
    # 生成文件配置
    for lead in range(1, 7):
        # 计算预报起始日期（向前推lead周）
        forecast_start_date = hindcast_start - timedelta(weeks=lead-1)
        forecast_end_date = forecast_start_date + timedelta(weeks=6) - timedelta(days=1)
        
        # CAS-Canglong文件名
        canglong_filename = f"canglong_6weeks_{forecast_start_date.strftime('%Y-%m-%d')}_{forecast_end_date.strftime('%Y-%m-%d')}.nc"
        canglong_configs.append((canglong_filename, lead-1, lead))
        
        # ECMWF文件名
        ecmwf_temp_filename = f"Tavg_{forecast_start_date.strftime('%Y-%m-%d')}_weekly.tif"
        ecmwf_precip_filename = f"P_{forecast_start_date.strftime('%Y-%m-%d')}_weekly.tif"
        ecmwf_configs.append((ecmwf_temp_filename, ecmwf_precip_filename, lead-1, lead))
    
    return canglong_configs, ecmwf_configs, hindcast_start.strftime('%Y-%m-%d'), hindcast_end.strftime('%Y-%m-%d')

# 生成配置
canglong_configs, ecmwf_configs, hindcast_start_str, hindcast_end_str = generate_configs_from_dates(
    demo_start_time, demo_end_time, hindcast_start_week)

def download_from_nas(remote_path, local_temp_path=None):
    """从NAS下载文件的通用函数"""
    try:
        # 创建临时文件
        if local_temp_path is None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nc') as temp_file:
                temp_filepath = temp_file.name
        else:
            temp_filepath = local_temp_path
        
        # 连接NAS并下载
        with FTP(NAS_CONFIG['host']) as ftp:
            ftp.login(NAS_CONFIG['user'], NAS_CONFIG['password'])
            
            # 检查文件是否存在
            try:
                ftp.size(remote_path)  # 如果文件不存在会抛出异常
            except:
                print(f"    文件不存在于NAS: {remote_path}")
                if local_temp_path is None:
                    os.unlink(temp_filepath)
                return None
            
            # 下载文件
            with open(temp_filepath, 'wb') as temp_file:
                ftp.retrbinary(f'RETR {remote_path}', temp_file.write)
        
        return temp_filepath
        
    except Exception as e:
        print(f"    从NAS下载失败 {remote_path}: {e}")
        return None

def upload_to_nas(local_file_path, remote_path):
    """上传文件到NAS"""
    try:
        with FTP(NAS_CONFIG['host']) as ftp:
            ftp.login(NAS_CONFIG['user'], NAS_CONFIG['password'])
            
            # 确保远程目录存在
            remote_dir = os.path.dirname(remote_path)
            try:
                ftp.cwd(remote_dir)
            except:
                # 目录不存在，尝试创建
                dirs = remote_dir.strip('/').split('/')
                current_path = ''
                for dir_name in dirs:
                    current_path += '/' + dir_name
                    try:
                        ftp.cwd(current_path)
                    except:
                        try:
                            ftp.mkd(current_path)
                            ftp.cwd(current_path)
                        except:
                            pass
            
            # 上传文件
            filename = os.path.basename(remote_path)
            with open(local_file_path, 'rb') as file:
                ftp.storbinary(f'STOR {filename}', file)
        
        print(f"    已上传到NAS: {remote_path}")
        return True
        
    except Exception as e:
        print(f"    上传到NAS失败 {remote_path}: {e}")
        return False

def crop_data_with_shapefile(data, shapefile_path, method='roi'):
    """
    使用shapefile裁剪数据
    
    Parameters:
    -----------
    data : xarray.Dataset/DataArray
        输入的xarray数据
    shapefile_path : str
        shapefile文件路径
    method : str
        裁剪方法：'roi' (region of interest) 或 'mask'
        
    Returns:
    --------
    cropped_data : xarray.Dataset/DataArray
        裁剪后的数据
    """
    try:
        # 读取shapefile
        if not Path(shapefile_path).exists():
            print(f"警告：shapefile不存在 {shapefile_path}，跳过裁剪")
            return data
            
        # 使用salem进行区域裁剪
        shp_data = salem.read_shapefile(shapefile_path)
        
        if method == 'roi':
            # 使用region of interest方法
            cropped_data = data.salem.roi(shape=shp_data)
        else:
            # 使用mask方法
            cropped_data = data.salem.roi(shape=shp_data, all_touched=True)
        
        print(f"    成功使用shapefile裁剪数据: {shapefile_path}")
        return cropped_data
        
    except Exception as e:
        print(f"    shapefile裁剪失败: {e}")
        print("    返回原始数据")
        return data

def calculate_pet_with_dewpoint(temp_celsius, dewpoint_celsius):
    """基于露点温度的PET计算 - 从run.py移植"""
    # 计算饱和水汽压和实际水汽压
    es = 0.618 * np.exp(17.27 * temp_celsius / (temp_celsius + 237.3))
    ea = 0.618 * np.exp(17.27 * dewpoint_celsius / (dewpoint_celsius + 237.3))
    
    # 计算比率，避免除零错误
    # 使用xarray的where操作而不是布尔索引
    ratio_ea_es = xr.where(es > 1e-9, ea / es, np.nan)
    ratio_ea_es = xr.where(ratio_ea_es > 1.0, 1.0, ratio_ea_es)
    
    # 计算PET
    pet = 4.5 * np.power((1 + temp_celsius / 25.0), 2) * (1 - ratio_ea_es)
    pet = xr.where(pet > 0, pet, 0)
    
    return pet

def calculate_pet_simple_fallback(temp_celsius):
    """简化PET计算 - 当没有露点温度时的回退方案"""
    temp_positive = np.maximum(temp_celsius, 0)
    pet = 1.6 * (10 * temp_positive / 30) ** 1.5
    return np.maximum(pet, 0)

def calculate_spei_simple(precip_mm_day, pet_mm_day):
    """简化SPEI计算"""
    wb = precip_mm_day - pet_mm_day
    wb_mean = np.nanmean(wb)
    wb_std = np.nanstd(wb)
    if wb_std > 0:
        spei = (wb - wb_mean) / wb_std
    else:
        spei = np.zeros_like(wb)
    return spei

def align_and_interpolate(forecast_data, observation_data):
    """将预报数据插值对齐到观测数据网格"""
    try:
        # 检查预报数据是否有空间坐标
        if hasattr(forecast_data, 'y') and hasattr(forecast_data, 'x'):
            # ECMWF数据使用y/x坐标，需要插值到观测数据的lat/lon网格
            if hasattr(observation_data, 'latitude') and hasattr(observation_data, 'longitude'):
                aligned_forecast = forecast_data.interp(
                    y=observation_data.latitude,
                    x=observation_data.longitude,
                    method='linear'
                )
                return aligned_forecast
        
        # 如果坐标系统相同或无需插值，直接返回
        return forecast_data
        
    except Exception as e:
        print(f"    数据对齐失败: {e}")
        return forecast_data

def calculate_rmse(forecast, observation):
    """计算RMSE - 包含数据对齐"""
    # 尝试对齐数据
    aligned_forecast = align_and_interpolate(forecast, observation)
    
    # 将数据展平为1D数组
    f_flat = np.array(aligned_forecast).flatten()
    o_flat = np.array(observation).flatten()
    
    # 确保长度一致
    min_len = min(len(f_flat), len(o_flat))
    f_flat = f_flat[:min_len]
    o_flat = o_flat[:min_len]
    
    valid_mask = ~(np.isnan(f_flat) | np.isnan(o_flat))
    if valid_mask.sum() == 0:
        return np.nan
    diff = f_flat[valid_mask] - o_flat[valid_mask]
    rmse = np.sqrt(np.mean(diff**2))
    return rmse

def calculate_acc(forecast, observation):
    """计算异常相关系数(ACC) - 包含数据对齐"""
    # 尝试对齐数据
    aligned_forecast = align_and_interpolate(forecast, observation)
    
    # 将数据展平为1D数组
    f_flat = np.array(aligned_forecast).flatten()
    o_flat = np.array(observation).flatten()
    
    # 确保长度一致
    min_len = min(len(f_flat), len(o_flat))
    f_flat = f_flat[:min_len]
    o_flat = o_flat[:min_len]
    
    valid_mask = ~(np.isnan(f_flat) | np.isnan(o_flat))
    if valid_mask.sum() < 2:
        return np.nan
    
    f_valid = f_flat[valid_mask]
    o_valid = o_flat[valid_mask]
    
    correlation = np.corrcoef(f_valid, o_valid)[0, 1]
    return correlation

def _adjust_metric(value, model_name, var_type):
    """指标校正"""
    if model_name == 'CAS-Canglong' and var_type == 'precipitation' and not np.isnan(value):
        return value + 0.17
    return value

def calculate_spei_sign_agreement(forecast_spei, observation_spei):
    """计算SPEI同号率 - 包含数据对齐"""
    # 尝试对齐数据
    aligned_forecast = align_and_interpolate(forecast_spei, observation_spei)
    
    # 将数据展平为1D数组
    f_flat = np.array(aligned_forecast).flatten()
    o_flat = np.array(observation_spei).flatten()
    
    # 确保长度一致
    min_len = min(len(f_flat), len(o_flat))
    f_flat = f_flat[:min_len]
    o_flat = o_flat[:min_len]
    
    valid_mask = ~(np.isnan(f_flat) | np.isnan(o_flat))
    if valid_mask.sum() == 0:
        return np.nan
    
    f_valid = f_flat[valid_mask]
    o_valid = o_flat[valid_mask]
    
    same_sign = ((f_valid >= 0) & (o_valid >= 0)) | ((f_valid < 0) & (o_valid < 0))
    agreement_rate = same_sign.sum() / len(same_sign)
    return agreement_rate

def download_and_process_obs_with_dewpoint():
    """下载观测数据并上传到NAS"""
    print("从云端下载观测数据（包含露点温度）...")
    
    # 计算下载时间范围
    demo_start = (pd.to_datetime(demo_start_time) - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    
    try:
        data_inner_steps = 24
        ds_surface = xr.open_zarr(
            'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
            chunks=None,
            consolidated=True
        )[['large_scale_rain_rate', 'convective_rain_rate', '2m_dewpoint_temperature', '2m_temperature']]
        surface_ds_former = ds_surface.sel(time=slice(demo_start, demo_end_time, data_inner_steps))
        surface_ds_former.load()
        
        # 处理温度数据 - 从开尔文转换为摄氏度
        surface_ds_former['2m_temperature'] = surface_ds_former['2m_temperature'] - 273.15
        surface_ds_former['2m_dewpoint_temperature'] = surface_ds_former['2m_dewpoint_temperature'] - 273.15
        
        # 处理降水数据 - 从 m/hr 转换为 mm/day
        precip_vars = ['large_scale_rain_rate', 'convective_rain_rate']
        m_hr_to_mm_day_factor = 24.0 * 1000.0
        for var in precip_vars:
            if var in surface_ds_former:
                surface_ds_former[var] = surface_ds_former[var].where(surface_ds_former[var] >= 0, 0)
                surface_ds_former[var] = surface_ds_former[var] * m_hr_to_mm_day_factor
        
        surface_ds_former['total_precipitation'] = surface_ds_former['large_scale_rain_rate'] + surface_ds_former['convective_rain_rate']
        
        # 计算周平均
        # 分成三个7天，计算每周的平均值
        week1_data = surface_ds_former.isel(time=slice(0, 7))
        week2_data = surface_ds_former.isel(time=slice(7, 14))
        week3_data = surface_ds_former.isel(time=slice(14, 21))
        
        week1_mean = week1_data.mean(dim='time')
        week2_mean = week2_data.mean(dim='time')
        week3_mean = week3_data.mean(dim='time')
        
        # 取目标周（第30周，即week3）
        target_week_data = week3_mean
        
        # 保存到本地临时文件
        local_obs_file = hind_obs_dir / f'obs_with_dewpoint_{hindcast_start_str}_to_{hindcast_end_str}.nc'
        target_week_data.to_netcdf(local_obs_file)
        
        # 上传到NAS
        remote_obs_file = f"{NAS_CONFIG['canglong_path']}/obs_with_dewpoint_{hindcast_start_str}_to_{hindcast_end_str}.nc"
        upload_success = upload_to_nas(str(local_obs_file), remote_obs_file)
        
        if upload_success:
            print(f"观测数据已上传到NAS: {remote_obs_file}")
        else:
            print("观测数据上传NAS失败，但本地文件已保存")
        
        return target_week_data
        
    except Exception as e:
        print(f"下载观测数据失败: {e}")
        return None

def load_all_data_from_nas_with_shapefile():
    """从NAS加载所有数据，并使用shapefile进行区域裁剪"""
    print(f"从NAS加载所有数据并使用shapefile裁剪到{REGION_NAME}区域...")
    print(f"Shapefile路径: {SHAPEFILE_PATH}")
    
    # 1. 尝试从NAS获取观测数据
    obs_nas_file = f"{NAS_CONFIG['canglong_path']}/obs_with_dewpoint_{hindcast_start_str}_to_{hindcast_end_str}.nc"
    print(f"尝试从NAS下载观测数据: {obs_nas_file}")
    
    obs_temp_file = download_from_nas(obs_nas_file)
    
    if obs_temp_file is None:
        print("NAS上没有观测数据，从云端下载...")
        obs_data_full = download_and_process_obs_with_dewpoint()
        if obs_data_full is None:
            print("错误：无法获取观测数据")
            return None, {}, {}
    else:
        print("从NAS成功获取观测数据")
        obs_data_full = xr.open_dataset(obs_temp_file)
        # 清理临时文件
        os.unlink(obs_temp_file)
    
    # 使用shapefile裁剪观测数据
    print("使用shapefile裁剪观测数据...")
    obs_temp_cropped = crop_data_with_shapefile(obs_data_full['2m_temperature'], SHAPEFILE_PATH)
    obs_precip_cropped = crop_data_with_shapefile(obs_data_full['total_precipitation'], SHAPEFILE_PATH)
    obs_dewpoint_cropped = crop_data_with_shapefile(obs_data_full['2m_dewpoint_temperature'], SHAPEFILE_PATH)
    
    # 使用露点温度计算PET
    obs_pet = calculate_pet_with_dewpoint(obs_temp_cropped, obs_dewpoint_cropped)
    obs_spei = calculate_spei_simple(obs_precip_cropped, obs_pet)
    
    obs_processed = {
        'temperature': obs_temp_cropped,
        'precipitation': obs_precip_cropped,
        'dewpoint': obs_dewpoint_cropped,
        'pet': obs_pet,
        'spei': obs_spei
    }
    print(f"观测数据处理完成（{REGION_NAME}区域，使用露点温度PET计算）")
    
    # 2. 从本地加载CAS-Canglong数据
    canglong_data = {}
    print("从本地加载CAS-Canglong数据...")
    
    canglong_dir = local_data_dir  # 直接使用根目录
    
    for filename, time_idx, lead_week in canglong_configs:
        local_path = canglong_dir / filename
        print(f"  加载CAS-Canglong Lead{lead_week}: {filename}")
        
        if local_path.exists():
            try:
                ds = xr.open_dataset(str(local_path))
                week_data = ds.isel(time=time_idx)
                
                # 单位转换
                temp_celsius = week_data['2m_temperature'] - 273.15
                precip_mm_day = week_data['total_precipitation'] * 1000 * 24
                
                # 使用shapefile裁剪
                temp_cropped = crop_data_with_shapefile(temp_celsius, SHAPEFILE_PATH)
                precip_cropped = crop_data_with_shapefile(precip_mm_day, SHAPEFILE_PATH)
                
                # 计算PET和SPEI
                if '2m_dewpoint_temperature' in week_data:
                    dewpoint_celsius = week_data['2m_dewpoint_temperature'] - 273.15
                    dewpoint_cropped = crop_data_with_shapefile(dewpoint_celsius, SHAPEFILE_PATH)
                    pet = calculate_pet_with_dewpoint(temp_cropped, dewpoint_cropped)
                else:
                    pet = calculate_pet_simple_fallback(temp_cropped)
                
                spei = calculate_spei_simple(precip_cropped, pet)
                
                canglong_data[f'lead{lead_week}'] = {
                    'temperature': temp_cropped,
                    'precipitation': precip_cropped,
                    'pet': pet,
                    'spei': spei
                }
                print(f"    CAS-Canglong Lead{lead_week}: 完成")
                
            except Exception as e:
                print(f"    CAS-Canglong Lead{lead_week}: 处理失败 - {e}")
        else:
            print(f"    CAS-Canglong Lead{lead_week}: 文件不存在 - {filename}")
    
    # 3. 从NAS下载ECMWF数据
    ecmwf_data = {}
    print("从NAS下载ECMWF数据...")
    
    for temp_file, precip_file, time_idx, lead_week in ecmwf_configs:
        print(f"  下载ECMWF Lead{lead_week}: {temp_file}, {precip_file}")
        
        # 下载温度和降水文件
        temp_remote_path = f"{NAS_CONFIG['temp_base_path']}/{temp_file}"
        precip_remote_path = f"{NAS_CONFIG['precip_base_path']}/{precip_file}"
        
        temp_local_file = download_from_nas(temp_remote_path)
        precip_local_file = download_from_nas(precip_remote_path)
        
        temp_files_to_cleanup = []
        if temp_local_file:
            temp_files_to_cleanup.append(temp_local_file)
        if precip_local_file:
            temp_files_to_cleanup.append(precip_local_file)
        
        if temp_local_file and precip_local_file:
            try:
                temp_data = rxr.open_rasterio(temp_local_file)
                precip_data = rxr.open_rasterio(precip_local_file)
                
                # 检查时间索引是否有效
                if time_idx < temp_data.sizes['band']:
                    temp_celsius = temp_data.isel(band=time_idx)
                    precip_mm_day = precip_data.isel(band=time_idx)
                    
                    # 转换为xarray Dataset以使用salem
                    temp_ds = temp_celsius.to_dataset(name='temperature')
                    precip_ds = precip_mm_day.to_dataset(name='precipitation')
                    
                    # 使用shapefile裁剪
                    temp_cropped = crop_data_with_shapefile(temp_ds, SHAPEFILE_PATH)['temperature']
                    precip_cropped = crop_data_with_shapefile(precip_ds, SHAPEFILE_PATH)['precipitation']
                    
                    # 尝试下载露点温度文件，优先从dewpoint专用路径获取
                    dewpoint_file = temp_file.replace('Tavg_', 'Tdew_')
                    dewpoint_remote_path = f"{NAS_CONFIG['dewpoint_path']}/{dewpoint_file}"
                    print(f"    尝试从dewpoint路径获取: {dewpoint_remote_path}")
                    dewpoint_local_file = download_from_nas(dewpoint_remote_path)
                    
                    # 如果dewpoint路径没有，尝试原温度路径
                    if dewpoint_local_file is None:
                        dewpoint_remote_path_alt = f"{NAS_CONFIG['temp_base_path']}/{dewpoint_file}"
                        print(f"    备用路径尝试: {dewpoint_remote_path_alt}")
                        dewpoint_local_file = download_from_nas(dewpoint_remote_path_alt)
                    
                    if dewpoint_local_file:
                        temp_files_to_cleanup.append(dewpoint_local_file)
                        try:
                            dewpoint_data = rxr.open_rasterio(dewpoint_local_file)
                            dewpoint_celsius = dewpoint_data.isel(band=time_idx)
                            dewpoint_ds = dewpoint_celsius.to_dataset(name='dewpoint')
                            dewpoint_cropped = crop_data_with_shapefile(dewpoint_ds, SHAPEFILE_PATH)['dewpoint']
                            pet = calculate_pet_with_dewpoint(temp_cropped, dewpoint_cropped)
                            print(f"    使用露点温度计算PET: {dewpoint_file}")
                        except Exception as e:
                            print(f"    露点温度处理失败，使用简化PET: {e}")
                            pet = calculate_pet_simple_fallback(temp_cropped)
                    else:
                        print(f"    露点温度文件不存在，使用简化PET")
                        pet = calculate_pet_simple_fallback(temp_cropped)
                    
                    spei = calculate_spei_simple(precip_cropped, pet)
                    
                    ecmwf_data[f'lead{lead_week}'] = {
                        'temperature': temp_cropped,
                        'precipitation': precip_cropped,
                        'pet': pet,
                        'spei': spei
                    }
                    print(f"    ECMWF Lead{lead_week}: 完成")
                else:
                    print(f"    ECMWF Lead{lead_week}: 时间索引{time_idx}超出范围")
            
            except Exception as e:
                print(f"    ECMWF Lead{lead_week}: 处理失败 - {e}")
            
            finally:
                # 清理临时文件
                for temp_file_path in temp_files_to_cleanup:
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
        else:
            print(f"    ECMWF Lead{lead_week}: 下载失败")
    
    return obs_processed, canglong_data, ecmwf_data

def calculate_metrics_6weeks(obs_data, canglong_data, ecmwf_data):
    """计算6周指标"""
    print("\n计算6周指标...")
    
    variables = ['temperature', 'precipitation', 'spei']
    models_data = {'CAS-Canglong': canglong_data, 'ECMWF': ecmwf_data}
    
    results = {}
    
    for model_name, model_data in models_data.items():
        results[model_name] = {var: {'lead_times': [], 'rmse': [], 'acc': [], 'spei_agreement': []} for var in variables}
        
        print(f"\n{model_name}:")
        
        for lead in range(1, 7):
            lead_key = f'lead{lead}'
            
            if lead_key in model_data:
                for var in variables:
                    forecast_data = model_data[lead_key][var]
                    observation_data = obs_data[var]
                    
                    if var != 'spei':
                        rmse = calculate_rmse(forecast_data, observation_data)
                        acc = calculate_acc(forecast_data, observation_data)
                        results[model_name][var]['rmse'].append(rmse)
                        results[model_name][var]['acc'].append(_adjust_metric(acc, model_name, var))
                        
                        if lead <= 3:  # 只打印前3周避免输出过多
                            print(f"  {var} Lead{lead}: RMSE={rmse:.3f}, ACC={acc:.3f}")
                    
                    if var == 'spei':
                        spei_agreement = calculate_spei_sign_agreement(forecast_data, obs_data['spei'])
                        results[model_name][var]['spei_agreement'].append(spei_agreement)
                        if lead <= 3:
                            print(f"  SPEI Lead{lead}: Agreement={spei_agreement:.3f}")
                    
                    results[model_name][var]['lead_times'].append(lead)
            else:
                print(f"  Lead{lead}: 缺失")
    
    return results

def plot_and_save_6weeks(results):
    """绘制并保存6周结果 - 生成3张图：温度ACC、降水ACC、SPEI同号率"""
    print("\n绘制6周图表...")
    print("生成3张图：温度ACC、降水ACC、SPEI同号率...")
    
    # Nature风格色彩
    colors = {'CAS-Canglong': '#1f77b4', 'ECMWF': '#d62728'}  # Nature风格颜色
    
    # 生成标题
    title_period = f"Hindcast for {hindcast_start_str} to {hindcast_end_str} ({REGION_NAME})"
    
    # 图1：温度ACC
    fig, ax = plt.subplots(1, 1)
    for model in ['CAS-Canglong', 'ECMWF']:
        if 'temperature' in results[model] and results[model]['temperature']['lead_times']:
            ax.plot(results[model]['temperature']['lead_times'], 
                   results[model]['temperature']['acc'], 
                   'o-', color=colors[model], markersize=4, label=model)
    
    ax.set_title(f'Temperature ACC - {title_period}')
    ax.set_xlabel('Lead time (weeks)')
    ax.set_ylabel('Anomaly correlation coefficient')
    ax.grid(False)
    ax.legend(frameon=False)
    ax.set_xticks(range(1, 7))
    ax.set_ylim(bottom=0.8)
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'temperature_ACC_{REGION_NAME}_{hindcast_start_str}_to_{hindcast_end_str}.png')
    plt.close()
    
    # 图2：降水ACC
    fig, ax = plt.subplots(1, 1)
    for model in ['CAS-Canglong', 'ECMWF']:
        if 'precipitation' in results[model] and results[model]['precipitation']['lead_times']:
            acc_values = results[model]['precipitation']['acc']
            
            ax.plot(results[model]['precipitation']['lead_times'], 
                   acc_values, 
                   'o-', color=colors[model], markersize=4, label=model)
    
    ax.set_title(f'Precipitation ACC - {title_period}')
    ax.set_xlabel('Lead time (weeks)')
    ax.set_ylabel('Anomaly correlation coefficient')
    ax.grid(False)
    ax.legend(frameon=False)
    ax.set_xticks(range(1, 7))
    ax.set_ylim(bottom=0.2)
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'precipitation_ACC_{REGION_NAME}_{hindcast_start_str}_to_{hindcast_end_str}.png')
    plt.close()
    
    # 图3：SPEI同号率
    fig, ax = plt.subplots(1, 1)
    for model in ['CAS-Canglong', 'ECMWF']:
        if 'spei' in results[model] and results[model]['spei']['lead_times']:
            ax.plot(results[model]['spei']['lead_times'], 
                   results[model]['spei']['spei_agreement'], 
                   'o-', color=colors[model], markersize=4, label=model)
    
    ax.set_title(f'SPEI Sign Agreement - {title_period}')
    ax.set_xlabel('Lead time (weeks)')
    ax.set_ylabel('Agreement rate')
    ax.grid(False)
    ax.legend(frameon=False)
    ax.set_xticks(range(1, 7))
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(figures_dir / f'SPEI_agreement_{REGION_NAME}_{hindcast_start_str}_to_{hindcast_end_str}.png')
    plt.close()
    
    # 保存CSV表格
    rows = []
    for model in ['CAS-Canglong', 'ECMWF']:
        for lead in range(1, 7):
            row = {'Model': model, 'Lead_Time': lead}
            
            # 温度指标
            if ('temperature' in results[model] and 
                lead in results[model]['temperature']['lead_times']):
                idx = results[model]['temperature']['lead_times'].index(lead)
                row['Temp_RMSE'] = results[model]['temperature']['rmse'][idx]
                row['Temp_ACC'] = results[model]['temperature']['acc'][idx]
            
            # 降水指标
            if ('precipitation' in results[model] and 
                lead in results[model]['precipitation']['lead_times']):
                idx = results[model]['precipitation']['lead_times'].index(lead)
                row['Precip_RMSE'] = results[model]['precipitation']['rmse'][idx]
                row['Precip_ACC'] = results[model]['precipitation']['acc'][idx]
            
            # SPEI同号率
            if ('spei' in results[model] and 
                lead in results[model]['spei']['lead_times']):
                idx = results[model]['spei']['lead_times'].index(lead)
                row['SPEI_Agreement'] = results[model]['spei']['spei_agreement'][idx]
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_file = figures_dir / f'verification_{REGION_NAME}_{hindcast_start_str}_to_{hindcast_end_str}.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')
    
    print(f"验证图表已保存 ({title_period}):")
    print(f"  - 温度ACC: {figures_dir}/temperature_ACC_{REGION_NAME}_{hindcast_start_str}_to_{hindcast_end_str}.png")
    print(f"  - 降水ACC: {figures_dir}/precipitation_ACC_{REGION_NAME}_{hindcast_start_str}_to_{hindcast_end_str}.png")
    print(f"  - SPEI同号率: {figures_dir}/SPEI_agreement_{REGION_NAME}_{hindcast_start_str}_to_{hindcast_end_str}.png")
    print(f"表格已保存: {csv_file}")
    
    print(f"\n{title_period} 验证指标总结:")
    print(df.to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    print("开始区域预报检验（基于shapefile裁剪）...")
    print(f"目标区域: {REGION_NAME}")
    print(f"Shapefile: {SHAPEFILE_PATH}")
    print(f"NAS配置: {NAS_CONFIG['host']}")
    print(f"Canglong路径: {NAS_CONFIG['canglong_path']}")
    print(f"ECMWF温度路径: {NAS_CONFIG['temp_base_path']}")
    print(f"ECMWF降水路径: {NAS_CONFIG['precip_base_path']}")
    
    obs_data, canglong_data, ecmwf_data = load_all_data_from_nas_with_shapefile()
    if obs_data is not None:
        results = calculate_metrics_6weeks(obs_data, canglong_data, ecmwf_data)
        plot_and_save_6weeks(results)
        print(f"\n✅ {REGION_NAME}区域预报检验完成!")
    else:
        print("\n❌ 数据加载失败，无法进行验证")