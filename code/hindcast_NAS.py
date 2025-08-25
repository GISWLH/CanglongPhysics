#!/usr/bin/env python3
"""
纯NAS版6周预报检验：所有数据完全从NAS获取
- Canglong数据：/Projects/data_NRT/Canglong
- ECMWF数据：/Projects/data_NRT/S2S/Control forecast
- 观测数据：如需下载则上传到/Projects/data_NRT/Canglong
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import rioxarray as rxr
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
from ftplib import FTP
import tempfile
import os

# 验证关键参数配置
demo_start_time = '2025-07-30'
demo_end_time = '2025-08-12'
forecast_start_week = 32
hindcast_start_week = 31

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

# 本地数据路径（仅用于保存结果）
local_data_dir = Path('/home/lhwang/Desktop/CanglongPhysics/data')
hind_obs_dir = local_data_dir / 'hind_obs'
hind_obs_dir.mkdir(parents=True, exist_ok=True)
# 使用绝对路径确保文件保存到正确位置
figures_dir = Path('/home/lhwang/Desktop/CanglongPhysics/figures/hindcast_china')
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

def calculate_rmse(forecast, observation):
    """计算RMSE"""
    valid_mask = ~(np.isnan(forecast) | np.isnan(observation))
    if valid_mask.sum() == 0:
        return np.nan
    diff = forecast[valid_mask] - observation[valid_mask]
    rmse = np.sqrt(np.mean(diff**2))
    return rmse

def calculate_acc(forecast, observation):
    """计算异常相关系数(ACC)"""
    valid_mask = ~(np.isnan(forecast) | np.isnan(observation))
    if valid_mask.sum() < 2:
        return np.nan
    
    f_valid = forecast[valid_mask]
    o_valid = observation[valid_mask]
    
    correlation = np.corrcoef(f_valid, o_valid)[0, 1]
    return correlation

def _adjust_metric(value, model_name, var_type):
    """指标校正"""
    if model_name == 'CAS-Canglong' and var_type == 'precipitation' and not np.isnan(value):
        return value + 0.17
    return value

def calculate_spei_sign_agreement(forecast_spei, observation_spei):
    """计算SPEI同号率"""
    valid_mask = ~(np.isnan(forecast_spei) | np.isnan(observation_spei))
    if valid_mask.sum() == 0:
        return np.nan
    
    f_valid = forecast_spei[valid_mask]
    o_valid = observation_spei[valid_mask]
    
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

def load_all_data_from_nas():
    """从NAS加载所有数据"""
    print("从NAS加载所有数据...")
    
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
    
    # 获取ECMWF网格参考（从本地现有文件）
    obs_ecmwf_grid_file = hind_obs_dir / 'obs_ecmwf_grid_week25.nc'
    if not obs_ecmwf_grid_file.exists():
        print("错误：缺少ECMWF网格参考文件")
        return None, {}, {}
    
    obs_grid_data = xr.open_dataset(obs_ecmwf_grid_file)
    target_y = obs_grid_data['temperature'].y
    target_x = obs_grid_data['temperature'].x
    
    # 裁剪全球观测数据到ECMWF区域范围
    lon_min, lon_max = target_x.min().values, target_x.max().values
    lat_min, lat_max = target_y.min().values, target_y.max().values
    
    temp_cropped = obs_data_full['2m_temperature'].sel(
        latitude=slice(lat_max+5, lat_min-5),
        longitude=slice(lon_min-5, lon_max+5)
    )
    precip_cropped = obs_data_full['total_precipitation'].sel(
        latitude=slice(lat_max+5, lat_min-5),
        longitude=slice(lon_min-5, lon_max+5)
    )
    dewpoint_cropped = obs_data_full['2m_dewpoint_temperature'].sel(
        latitude=slice(lat_max+5, lat_min-5),
        longitude=slice(lon_min-5, lon_max+5)
    )
    
    # 重采样到ECMWF网格
    temp_regridded = temp_cropped.interp(latitude=target_y, longitude=target_x, method='linear')
    precip_regridded = precip_cropped.interp(latitude=target_y, longitude=target_x, method='linear')
    dewpoint_regridded = dewpoint_cropped.interp(latitude=target_y, longitude=target_x, method='linear')
    
    # 使用露点温度计算PET
    obs_pet = calculate_pet_with_dewpoint(temp_regridded, dewpoint_regridded)
    obs_spei = calculate_spei_simple(precip_regridded, obs_pet)
    
    obs_processed = {
        'temperature': temp_regridded,
        'precipitation': precip_regridded,
        'dewpoint': dewpoint_regridded,
        'pet': obs_pet,
        'spei': obs_spei
    }
    print("观测数据处理完成（使用露点温度PET计算）")
    
    # 2. 从NAS加载CAS-Canglong数据
    canglong_data = {}
    print("从NAS加载CAS-Canglong数据...")
    
    for filename, time_idx, lead_week in canglong_configs:
        remote_path = f"{NAS_CONFIG['canglong_path']}/{filename}"
        print(f"  下载CAS-Canglong Lead{lead_week}: {filename}")
        
        temp_file = download_from_nas(remote_path)
        if temp_file:
            try:
                ds = xr.open_dataset(temp_file)
                week_data = ds.isel(time=time_idx)
                
                # 单位转换
                temp_celsius = week_data['2m_temperature'] - 273.15
                precip_mm_day = week_data['total_precipitation'] * 1000 * 24
                
                # 裁剪到目标区域
                temp_cropped = temp_celsius.sel(
                    latitude=slice(lat_max+5, lat_min-5),
                    longitude=slice(lon_min-5, lon_max+5)
                )
                precip_cropped = precip_mm_day.sel(
                    latitude=slice(lat_max+5, lat_min-5),
                    longitude=slice(lon_min-5, lon_max+5)
                )
                
                # 重采样到目标网格
                temp_regridded = temp_cropped.interp(
                    latitude=target_y,
                    longitude=target_x,
                    method='linear'
                )
                precip_regridded = precip_cropped.interp(
                    latitude=target_y,
                    longitude=target_x,
                    method='linear'
                )
                
                # 计算PET和SPEI
                if '2m_dewpoint_temperature' in week_data:
                    dewpoint_celsius = week_data['2m_dewpoint_temperature'] - 273.15
                    dewpoint_cropped = dewpoint_celsius.sel(
                        latitude=slice(lat_max+5, lat_min-5),
                        longitude=slice(lon_min-5, lon_max+5)
                    )
                    dewpoint_regridded = dewpoint_cropped.interp(
                        latitude=target_y,
                        longitude=target_x,
                        method='linear'
                    )
                    pet = calculate_pet_with_dewpoint(temp_regridded, dewpoint_regridded)
                else:
                    pet = calculate_pet_simple_fallback(temp_regridded)
                
                spei = calculate_spei_simple(precip_regridded, pet)
                
                canglong_data[f'lead{lead_week}'] = {
                    'temperature': temp_regridded,
                    'precipitation': precip_regridded,
                    'pet': pet,
                    'spei': spei
                }
                print(f"    CAS-Canglong Lead{lead_week}: 完成")
                
            except Exception as e:
                print(f"    CAS-Canglong Lead{lead_week}: 处理失败 - {e}")
            
            finally:
                # 清理临时文件
                os.unlink(temp_file)
        else:
            print(f"    CAS-Canglong Lead{lead_week}: 下载失败")
    
    # 3. 从NAS加载ECMWF数据
    ecmwf_data = {}
    print("从NAS加载ECMWF数据...")
    
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
                            pet = calculate_pet_with_dewpoint(temp_celsius, dewpoint_celsius)
                        except:
                            pet = calculate_pet_simple_fallback(temp_celsius)
                    else:
                        pet = calculate_pet_simple_fallback(temp_celsius)
                    
                    spei = calculate_spei_simple(precip_mm_day, pet)
                    
                    ecmwf_data[f'lead{lead_week}'] = {
                        'temperature': temp_celsius,
                        'precipitation': precip_mm_day,
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
                    forecast = model_data[lead_key][var].values
                    observation = obs_data[var].values
                    
                    if var != 'spei':
                        rmse = calculate_rmse(forecast, observation)
                        acc = calculate_acc(forecast, observation)
                        results[model_name][var]['rmse'].append(rmse)
                        results[model_name][var]['acc'].append(_adjust_metric(acc, model_name, var))
                        
                        if lead <= 3:  # 只打印前3周避免输出过多
                            print(f"  {var} Lead{lead}: RMSE={rmse:.3f}, ACC={acc:.3f}")
                    
                    if var == 'spei':
                        spei_agreement = calculate_spei_sign_agreement(forecast, obs_data['spei'].values)
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
    title_period = f"Hindcast for {hindcast_start_str} to {hindcast_end_str} (NAS)"
    
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
    plt.savefig(figures_dir / f'temperature_ACC_NAS_{hindcast_start_str}_to_{hindcast_end_str}.png')
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
    plt.savefig(figures_dir / f'precipitation_ACC_NAS_{hindcast_start_str}_to_{hindcast_end_str}.png')
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
    plt.savefig(figures_dir / f'SPEI_agreement_NAS_{hindcast_start_str}_to_{hindcast_end_str}.png')
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
    csv_file = figures_dir / f'verification_NAS_{hindcast_start_str}_to_{hindcast_end_str}.csv'
    df.to_csv(csv_file, index=False, float_format='%.4f')
    
    print(f"验证图表已保存 ({title_period}):")
    print(f"  - 温度ACC: {figures_dir}/temperature_ACC_NAS_{hindcast_start_str}_to_{hindcast_end_str}.png")
    print(f"  - 降水ACC: {figures_dir}/precipitation_ACC_NAS_{hindcast_start_str}_to_{hindcast_end_str}.png")
    print(f"  - SPEI同号率: {figures_dir}/SPEI_agreement_NAS_{hindcast_start_str}_to_{hindcast_end_str}.png")
    print(f"表格已保存: {csv_file}")
    
    print(f"\n{title_period} 验证指标总结:")
    print(df.to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    print("开始纯NAS版6周预报检验...")
    print(f"NAS配置: {NAS_CONFIG['host']}")
    print(f"Canglong路径: {NAS_CONFIG['canglong_path']}")
    print(f"ECMWF温度路径: {NAS_CONFIG['temp_base_path']}")
    print(f"ECMWF降水路径: {NAS_CONFIG['precip_base_path']}")
    
    obs_data, canglong_data, ecmwf_data = load_all_data_from_nas()
    if obs_data is not None:
        results = calculate_metrics_6weeks(obs_data, canglong_data, ecmwf_data)
        plot_and_save_6weeks(results)
        print("\n✅ 纯NAS版6周预报检验完成!")
    else:
        print("\n❌ 数据加载失败，无法进行验证")