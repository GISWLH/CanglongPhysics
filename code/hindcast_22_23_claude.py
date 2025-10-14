"""
Hindcast 2022-2023 批量回报脚本
基于run_temp.py改编，用于批量处理2022-2023年全年数据
使用预加载的输入数据，无需从Google Cloud下载
"""

import torch
import numpy as np
import xarray as xr
import pandas as pd
import os
from tqdm import tqdm
from datetime import datetime, timedelta
from scipy.special import gamma as gamma_function

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 常量定义
forecast_weeks = 6
data_inner_steps = 24

# 变量列表
surface_var_names = [
    'large_scale_rain_rate',
    'convective_rain_rate',
    'total_column_cloud_ice_water',
    'total_cloud_cover',
    'top_net_solar_radiation_clear_sky',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    '2m_dewpoint_temperature',
    '2m_temperature',
    'surface_latent_heat_flux',
    'surface_sensible_heat_flux',
    'surface_pressure',
    'volumetric_soil_water_layer',
    'mean_sea_level_pressure',
    'sea_ice_cover',
    'sea_surface_temperature'
]

upper_air_vars = [
    'geopotential',
    'vertical_velocity',
    'u_component_of_wind',
    'v_component_of_wind',
    'fraction_of_cloud_cover',
    'temperature',
    'specific_humidity'
]

# 变量映射和统计信息
var_mapping = {
    'large_scale_rain_rate': 'lsrr',
    'convective_rain_rate': 'crr',
    'total_column_cloud_ice_water': 'tciw',
    'total_cloud_cover': 'tcc',
    'top_net_solar_radiation_clear_sky': 'tsrc',
    '10m_u_component_of_wind': 'u10',
    '10m_v_component_of_wind': 'v10',
    '2m_dewpoint_temperature': 'd2m',
    '2m_temperature': 't2m',
    'surface_latent_heat_flux': 'surface_latent_heat_flux',
    'surface_sensible_heat_flux': 'surface_sensible_heat_flux',
    'surface_pressure': 'sp',
    'volumetric_soil_water_layer': 'swvl',
    'mean_sea_level_pressure': 'msl',
    'sea_ice_cover': 'siconc',
    'sea_surface_temperature': 'sst'
}

ordered_var_stats = {
    'lsrr': {'mean': 1.10E-05, 'std': 2.55E-05},
    'crr': {'mean': 1.29E-05, 'std': 2.97E-05},
    'tciw': {'mean': 0.022627383, 'std': 0.023428712},
    'tcc': {'mean': 0.673692584, 'std': 0.235167906},
    'tsrc': {'mean': 856148, 'std': 534222.125},
    'u10': {'mean': -0.068418466, 'std': 4.427545547},
    'v10': {'mean': 0.197138891, 'std': 3.09530735},
    'd2m': {'mean': 274.2094421, 'std': 20.45770073},
    't2m': {'mean': 278.7841187, 'std': 21.03286934},
    'surface_latent_heat_flux': {'mean': -5410301.5, 'std': 5349063.5},
    'surface_sensible_heat_flux': {'mean': -971651.375, 'std': 2276764.75},
    'sp': {'mean': 96651.14063, 'std': 9569.695313},
    'swvl': {'mean': 0.34216917, 'std': 0.5484813},
    'msl': {'mean': 100972.3438, 'std': 1191.102417},
    'siconc': {'mean': 0.785884917, 'std': 0.914535105},
    'sst': {'mean': 189.7337189, 'std': 136.1803131},

    'geopotential': {
        '300': {'mean': 13763.50879, 'std': 1403.990112},
        '500': {'mean': 28954.94531, 'std': 2085.838867},
        '700': {'mean': 54156.85547, 'std': 3300.384277},
        '850': {'mean': 89503.79688, 'std': 5027.79541}
    },
    'vertical_velocity': {
        '300': {'mean': 0.011849277, 'std': 0.126232564},
        '500': {'mean': 0.002759292, 'std': 0.097579598},
        '700': {'mean': 0.000348145, 'std': 0.072489716},
        '850': {'mean': 0.000108061, 'std': 0.049831692}
    },
    'u_component_of_wind': {
        '300': {'mean': 1.374536991, 'std': 6.700420856},
        '500': {'mean': 3.290786982, 'std': 7.666454315},
        '700': {'mean': 6.491596222, 'std': 9.875613213},
        '850': {'mean': 11.66026878, 'std': 14.00845909}
    },
    'v_component_of_wind': {
        '300': {'mean': 0.146550566, 'std': 3.75399971},
        '500': {'mean': 0.022800878, 'std': 4.179731846},
        '700': {'mean': -0.025720235, 'std': 5.324173927},
        '850': {'mean': -0.027837994, 'std': 7.523460865}
    },
    'fraction_of_cloud_cover': {
        '300': {'mean': 0.152513072, 'std': 0.15887706},
        '500': {'mean': 0.106524825, 'std': 0.144112185},
        '700': {'mean': 0.105878539, 'std': 0.112193666},
        '850': {'mean': 0.108120449, 'std': 0.108371623}
    },
    'temperature': {
        '300': {'mean': 274.8048401, 'std': 15.28209305},
        '500': {'mean': 267.6254578, 'std': 14.55300999},
        '700': {'mean': 253.1627655, 'std': 12.77071381},
        '850': {'mean': 229.0860138, 'std': 10.5536499}
    },
    'specific_humidity': {
        '300': {'mean': 0.004610791, 'std': 0.003879665},
        '500': {'mean': 0.002473272, 'std': 0.002312181},
        '700': {'mean': 0.000875093, 'std': 0.000944978},
        '850': {'mean': 0.000130984, 'std': 0.000145811}
    }
}

## Paste model here

## End paste model
        
# 加载模型
print("加载模型...")
import sys
import canglong.embed_old as embed_old
sys.modules['canglong.embed'] = embed_old
import canglong.recovery_old as recovery_old
sys.modules['canglong.recovery'] = recovery_old

model_path = 'F:/model/weather_model_epoch_500.pt'
model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()
print("模型加载成功")

# 加载预处理的输入数据
print("加载预处理数据...")
surface_input = torch.load('I:/ERA5_np/input_surface_norm_test_last100.pt')  # (16, 100, 721, 1440)
upper_air_input = torch.load('I:/ERA5_np/input_upper_air_norm_test_last100.pt')  # (7, 4, 100, 721, 1440)
print(f"Surface输入形状: {surface_input.shape}")
print(f"Upper air输入形状: {upper_air_input.shape}")

# 加载气候态数据用于SPEI计算
print("加载气候态数据...")
climate = xr.open_dataset('E:/data/climate_variables_2000_2023_weekly.nc')

# 定义周号计算函数
def get_week_of_year(date):
    """用于xarray时间维度的周数计算"""
    day_of_year = date.dt.dayofyear
    return ((day_of_year - 1) // 7) + 1

def calculate_week_number(date):
    """计算单个datetime对象的周数（1-52）"""
    day_of_year = date.timetuple().tm_yday
    return ((day_of_year - 1) // 7) + 1

# SPEI计算辅助函数
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

def denormalize_surface(normalized_surface):
    """反标准化surface数据"""
    surface_means = np.array([ordered_var_stats[var_mapping[var]]['mean'] for var in surface_var_names])
    surface_stds = np.array([ordered_var_stats[var_mapping[var]]['std'] for var in surface_var_names])
    surface_means = surface_means.reshape(-1, 1, 1, 1)
    surface_stds = surface_stds.reshape(-1, 1, 1, 1)
    return normalized_surface * surface_stds + surface_means

def data_to_xarray(denormalized_surface, start_date, forecast_weeks=6):
    """将反标准化数据转换为xarray格式"""
    forecast_dates = [start_date + timedelta(days=(i+1)*7-1) for i in range(forecast_weeks)]
    lat = np.linspace(90, -90, 721)
    lon = np.linspace(0, 359.75, 1440)

    surface_ds = xr.Dataset(coords={
        'variable': surface_var_names,
        'time': forecast_dates,
        'latitude': lat,
        'longitude': lon
    })

    surface_data_array = xr.DataArray(
        denormalized_surface,
        dims=['variable', 'time', 'latitude', 'longitude'],
        coords={'variable': surface_var_names, 'time': forecast_dates, 'latitude': lat, 'longitude': lon}
    )

    surface_ds['data'] = surface_data_array
    for i, var_name in enumerate(surface_var_names):
        surface_ds[var_name] = surface_data_array.sel(variable=var_name)

    # 单位转换
    surface_ds['2m_temperature'] = surface_ds['2m_temperature'] - 273.15
    surface_ds['2m_dewpoint_temperature'] = surface_ds['2m_dewpoint_temperature'] - 273.15

    # 降水转换: m/hr -> mm/day
    m_hr_to_mm_day = 24.0 * 1000.0
    surface_ds['large_scale_rain_rate'] = surface_ds['large_scale_rain_rate'].where(
        surface_ds['large_scale_rain_rate'] >= 0, 0) * m_hr_to_mm_day
    surface_ds['convective_rain_rate'] = surface_ds['convective_rain_rate'].where(
        surface_ds['convective_rain_rate'] >= 0, 0) * m_hr_to_mm_day

    surface_ds['total_precipitation'] = surface_ds['large_scale_rain_rate'] + surface_ds['convective_rain_rate']

    return surface_ds

def calculate_pet_and_spei(surface_ds, climate_data, input_surface_ds=None, start_pred_idx=0):
    """计算PET和SPEI

    Args:
        surface_ds: 预报数据（6周）
        climate_data: 气候态数据
        input_surface_ds: 输入数据（2周），用于计算前3周的SPEI
        start_pred_idx: 开始计算SPEI的索引（默认0）
    """
    # 计算PET
    t2m_celsius = surface_ds['2m_temperature'].values
    d2m_celsius = surface_ds['2m_dewpoint_temperature'].values

    es = 0.618 * np.exp(17.27 * t2m_celsius / (t2m_celsius + 237.3))
    ea = 0.618 * np.exp(17.27 * d2m_celsius / (d2m_celsius + 237.3))

    ratio_ea_es = np.full_like(t2m_celsius, np.nan)
    valid_es_mask = es > 1e-9
    ratio_ea_es[valid_es_mask] = ea[valid_es_mask] / es[valid_es_mask]
    ratio_ea_es = np.clip(ratio_ea_es, None, 1.0)

    pet = 4.5 * np.power((1 + t2m_celsius / 25.0), 2) * (1 - ratio_ea_es)
    pet = np.maximum(pet, 0)

    surface_ds['potential_evapotranspiration'] = (('time', 'latitude', 'longitude'), pet)

    # 计算D (降水 - 蒸散发)
    D_pred = surface_ds['total_precipitation'] - surface_ds['potential_evapotranspiration']
    D_pred = D_pred.rename({'latitude': 'lat', 'longitude': 'lon'})

    # 如果提供了输入数据，合并以计算前3周的SPEI
    if input_surface_ds is not None:
        # 为输入数据计算PET
        t2m_input = input_surface_ds['2m_temperature'].values
        d2m_input = input_surface_ds['2m_dewpoint_temperature'].values

        es_input = 0.618 * np.exp(17.27 * t2m_input / (t2m_input + 237.3))
        ea_input = 0.618 * np.exp(17.27 * d2m_input / (d2m_input + 237.3))

        ratio_input = np.full_like(t2m_input, np.nan)
        valid_mask = es_input > 1e-9
        ratio_input[valid_mask] = ea_input[valid_mask] / es_input[valid_mask]
        ratio_input = np.clip(ratio_input, None, 1.0)

        pet_input = 4.5 * np.power((1 + t2m_input / 25.0), 2) * (1 - ratio_input)
        pet_input = np.maximum(pet_input, 0)

        input_surface_ds['potential_evapotranspiration'] = (('time', 'latitude', 'longitude'), pet_input)

        # 计算输入数据的D
        D_input = input_surface_ds['total_precipitation'] - input_surface_ds['potential_evapotranspiration']
        D_input = D_input.rename({'latitude': 'lat', 'longitude': 'lon'})

        # 合并输入和预报数据
        D_combined = xr.concat([D_input, D_pred], dim='time')
        start_calc_idx = 2  # 从合并数据的第3个时间点开始计算（即预报的第1周）
    else:
        D_combined = D_pred
        start_calc_idx = 3  # 从预报数据的第4个时间点开始计算

    D_hist = climate_data['tp'] - climate_data['pet']

    # 计算SPEI
    if len(D_combined.time) < 4:
        print(f"警告: 时间点不足4个（当前{len(D_combined.time)}个），无法计算SPEI")
        return surface_ds

    spei_pred_list = []
    pred_week_numbers = get_week_of_year(D_combined.time)
    hist_week_numbers = get_week_of_year(D_hist.time)

    # 从start_calc_idx + 1开始计算（确保有至少3周历史数据）
    for i in range(max(3, start_calc_idx + 1), len(D_combined.time)):
        # 累积当前及前3周的D值（从合并数据中）
        curr_week_accum = sum([D_combined.isel(time=i-j) for j in range(4) if i-j >= 0])
        curr_week_num = pred_week_numbers.isel(time=i).item()

        # 提取历史同期数据
        hist_4week_accum_list = []
        hist_years = np.unique(D_hist.time.dt.year)

        for year in hist_years:
            year_data = D_hist.where(D_hist.time.dt.year == year, drop=True)
            year_weeks = hist_week_numbers.where(D_hist.time.dt.year == year, drop=True)
            week_indices = np.where(year_weeks == curr_week_num)[0]
            if len(week_indices) > 0:
                week_idx = week_indices[0]
                if week_idx >= 3:
                    accum_D = sum([year_data.isel(time=week_idx-j) for j in range(4)])
                    hist_4week_accum_list.append(accum_D)

        if hist_4week_accum_list:
            hist_4week_accum = xr.concat(hist_4week_accum_list, dim='time')
        else:
            hist_4week_accum = xr.DataArray(
                np.zeros((0,) + D_pred.isel(time=0).shape),
                coords={'time': [], **{dim: D_pred[dim] for dim in D_pred.dims if dim != 'time'}},
                dims=D_pred.dims
            )

        if len(hist_4week_accum.time) < 10:
            spei_map = xr.full_like(D_pred.isel(time=i), np.nan)
        elif np.isnan(curr_week_accum).all():
            spei_map = xr.full_like(D_pred.isel(time=i), np.nan)
        else:
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

    if spei_pred_list:
        spei_pred = xr.concat(spei_pred_list, dim='time')

        # 如果提供了输入数据，SPEI从合并数据的第3个点开始（对应预报的第1周）
        # 否则从第4个点开始（对应预报的第4周）
        if input_surface_ds is not None:
            # 有输入数据：计算了全部6周预报的SPEI（从合并数据的索引3开始）
            spei_pred = spei_pred.assign_coords(time=D_pred.time[:len(spei_pred_list)])
        else:
            # 无输入数据：只计算了后3周的SPEI
            spei_pred = spei_pred.assign_coords(time=D_pred.time[start_pred_idx + 3:])

        surface_ds['spei'] = spei_pred.rename({'lat': 'latitude', 'lon': 'longitude'})

    return surface_ds

# 主循环：处理2022-2023年每一周
output_dir = 'Z:/Data/hindcast_2022_2023'
os.makedirs(output_dir, exist_ok=True)

# 定义2022年的起始周（从索引4开始，即2022-02-26至03-04）
start_year = 2022
start_week = 4  # 索引4对应2022年第9周 (2.26-3.4，2022是平年)
total_weeks = 100  # 数据中有100周

print(f"\n开始批量回报 2022-2023 年...")
print(f"起始周: 索引{start_week}，日期2022-02-26至2022-03-04")
print(f"总周数: {total_weeks - start_week}")

# 记录所有处理的文件
processed_files = []

with torch.no_grad():
    for week_idx in tqdm(range(start_week, total_weeks), desc="回报进度"):
        try:
            # 计算日期 (2022-01-29是第0周)
            base_date = datetime(2022, 1, 29)
            current_date = base_date + timedelta(weeks=week_idx)
            year = current_date.year
            week_of_year = calculate_week_number(current_date)

            # 提取前两周的输入数据 (形成2周输入)
            if week_idx < 2:
                continue  # 跳过前两周，因为没有足够的历史数据

            # 提取输入数据：需要前两周（week_idx-2 和 week_idx-1）
            input_surface_week = surface_input[:, [week_idx-2, week_idx-1], :, :]  # (16, 2, 721, 1440)
            input_upper_air_week = upper_air_input[:, :, [week_idx-2, week_idx-1], :, :]  # (7, 4, 2, 721, 1440)

            # 转换为tensor并添加batch维度
            input_surface_tensor = torch.tensor(input_surface_week, dtype=torch.float32).unsqueeze(0).to(device)
            input_upper_air_tensor = torch.tensor(input_upper_air_week, dtype=torch.float32).unsqueeze(0).to(device)

            # 滚动预报6周
            current_input_surface = input_surface_tensor
            current_input_upper_air = input_upper_air_tensor

            all_surface_predictions = []
            all_upper_air_predictions = []

            for week in range(forecast_weeks):
                output_surface, output_upper_air = model(current_input_surface, current_input_upper_air)
                all_surface_predictions.append(output_surface[:, :, 0:1, :, :])
                all_upper_air_predictions.append(output_upper_air[:, :, :, 0:1, :, :])

                if week < forecast_weeks - 1:
                    new_input_surface = torch.cat([
                        current_input_surface[:, :, 1:2, :, :],
                        output_surface[:, :, 0:1, :, :]
                    ], dim=2)
                    new_input_upper_air = torch.cat([
                        current_input_upper_air[:, :, :, 1:2, :, :],
                        output_upper_air[:, :, :, 0:1, :, :]
                    ], dim=3)
                    current_input_surface = new_input_surface
                    current_input_upper_air = new_input_upper_air

            # 合并预测结果
            all_weeks_surface_predictions = torch.cat(all_surface_predictions, dim=2)

            # 反标准化预测结果
            surface_predictions_np = all_weeks_surface_predictions.cpu().numpy()
            denormalized_surface = denormalize_surface(surface_predictions_np[0])

            # 反标准化输入数据（用于SPEI计算）
            input_surface_np = input_surface_tensor.cpu().numpy()
            denormalized_input = denormalize_surface(input_surface_np[0])

            # 转换预测为xarray
            forecast_start_date = current_date + timedelta(days=1)
            surface_ds = data_to_xarray(denormalized_surface, forecast_start_date, forecast_weeks)

            # 转换输入为xarray（需要构造时间坐标）
            # current_date是当前周的开始日期，前两周分别结束于current_date-8天和current_date-1天
            input_week1_start = current_date - timedelta(days=14)  # 第一周开始日期
            input_surface_ds = data_to_xarray(denormalized_input, input_week1_start, forecast_weeks=2)

            # 计算PET和SPEI（传入输入数据）
            surface_ds = calculate_pet_and_spei(surface_ds, climate, input_surface_ds=input_surface_ds, start_pred_idx=0)

            # 只保留需要的4个变量
            output_ds = xr.Dataset({
                '2m_temperature': surface_ds['2m_temperature'],
                '2m_dewpoint_temperature': surface_ds['2m_dewpoint_temperature'],
                'total_precipitation': surface_ds['total_precipitation'],
                'spei': surface_ds['spei'] if 'spei' in surface_ds else xr.full_like(surface_ds['2m_temperature'], np.nan)
            })

            # 添加元数据
            output_ds.attrs['description'] = 'Hindcast for temperature, dewpoint, precipitation, and SPEI'
            output_ds.attrs['year'] = year
            output_ds.attrs['week_of_year'] = week_of_year
            output_ds.attrs['start_date'] = current_date.strftime('%Y-%m-%d')
            output_ds.attrs['forecast_start'] = forecast_start_date.strftime('%Y-%m-%d')

            # 保存文件
            filename = f"hindcast_{year}_week{week_of_year:02d}_surface_{current_date.strftime('%Y-%m-%d')}.nc"
            output_path = os.path.join(output_dir, filename)
            output_ds.to_netcdf(output_path)

            processed_files.append({
                'year': year,
                'week': week_of_year,
                'start_date': current_date.strftime('%Y-%m-%d'),
                'surface_file': output_path
            })

            # 每10周清理一次内存
            if (week_idx - start_week) % 10 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n处理第 {week_idx} 周 ({current_date.strftime('%Y-%m-%d')}) 失败: {str(e)}")
            continue

# 保存索引文件
index_df = pd.DataFrame(processed_files)
index_path = os.path.join(output_dir, 'hindcast_index_2022_2023.csv')
index_df.to_csv(index_path, index=False)

print(f"\n完成！")
print(f"成功处理 {len(processed_files)} 周")
print(f"文件保存在: {output_dir}")
print(f"索引文件: {index_path}")
