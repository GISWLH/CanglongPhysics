#!/usr/bin/env python
"""
测试改进后的局地温度方程在52周训练数据上的闭合率和误差
使用8核并行处理
"""

import torch
import numpy as np
import h5py as h5
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

sys.path.insert(0, '/home/CanglongPhysics')
from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays

# 温度平衡方程权重参数（通过ERA5数据搜索优化得到）
ALPHA = 0.01  # 水平平流项权重
BETA = 0.01   # 垂直运动项权重
GAMMA = 0.1   # 非绝热加热项权重

# 全局变量用于并行处理
SURFACE_DATA = None
UPPER_AIR_DATA = None


def init_worker(surface_data, upper_air_data):
    """初始化工作进程的全局变量"""
    global SURFACE_DATA, UPPER_AIR_DATA
    SURFACE_DATA = surface_data
    UPPER_AIR_DATA = upper_air_data


def process_week(week):
    """处理单个周的数据，返回统计结果"""
    global SURFACE_DATA, UPPER_AIR_DATA

    # 物理常数
    R_d = 287.0
    c_p = 1004.0
    g = 9.8
    L_v = 2.5e6

    # 变量索引
    idx_t, idx_u, idx_v, idx_w = 2, 3, 4, 5
    idx_q, idx_o3, idx_clwc, idx_ciwc = 6, 0, 9, 8
    idx_tnswrf, idx_tnlwrf = 0, 1
    idx_lsrr, idx_crr = 4, 5
    idx_sshf = 14
    idx_snswrf, idx_snlwrf = 15, 16

    # 提取数据
    surface_t0 = SURFACE_DATA[week]      # (26, 721, 1440)
    surface_t1 = SURFACE_DATA[week + 1]  # (26, 721, 1440)
    upper_t0 = UPPER_AIR_DATA[week]      # (10, 5, 721, 1440)
    upper_t1 = UPPER_AIR_DATA[week + 1]  # (10, 5, 721, 1440)

    # 时间步长
    dt = 7 * 24 * 3600  # seconds

    # 提取温度场 (5, 721, 1440)
    t_t0 = upper_t0[idx_t]
    t_t1 = upper_t1[idx_t]

    # 风场和其他变量
    u = upper_t1[idx_u]
    v = upper_t1[idx_v]
    w = upper_t1[idx_w]
    t = upper_t1[idx_t]
    q = upper_t1[idx_q]
    o3 = upper_t1[idx_o3]
    clwc = upper_t1[idx_clwc]
    ciwc = upper_t1[idx_ciwc]

    # 1. 温度局地变化率
    dT_dt_observed = (t_t1 - t_t0) / dt  # (5, 721, 1440)

    # 2. 水平平流项
    dlat = 0.25 * np.pi / 180
    dlon = 0.25 * np.pi / 180
    R_earth = 6.371e6

    lat_values = np.linspace(-90, 90, 721) * np.pi / 180
    cos_lat = np.cos(lat_values).reshape(1, -1, 1)

    # ∂T/∂x (东西向，循环边界)
    t_padded_x = np.pad(t, ((0, 0), (0, 0), (1, 1)), mode='wrap')
    dT_dx = (t_padded_x[:, :, 2:] - t_padded_x[:, :, :-2]) / (2 * R_earth * dlon * cos_lat)

    # ∂T/∂y (南北向)
    t_padded_y = np.pad(t, ((0, 0), (1, 1), (0, 0)), mode='edge')
    dT_dy = (t_padded_y[:, 2:, :] - t_padded_y[:, :-2, :]) / (2 * R_earth * dlat)

    horizontal_advection = -(u * dT_dx + v * dT_dy)

    # 3. 垂直运动项
    pressure_levels = np.array([200, 300, 500, 700, 850]) * 100  # Pa
    p_3d = pressure_levels.reshape(5, 1, 1)

    dT_dp = np.zeros_like(t)
    for i in range(5):
        if i == 0:
            dT_dp[i] = (t[i+1] - t[i]) / (pressure_levels[i+1] - pressure_levels[i])
        elif i == 4:
            dT_dp[i] = (t[i] - t[i-1]) / (pressure_levels[i] - pressure_levels[i-1])
        else:
            dT_dp[i] = (t[i+1] - t[i-1]) / (pressure_levels[i+1] - pressure_levels[i-1])

    adiabatic_term = (R_d * t / (c_p * p_3d) - dT_dp) * w
    vertical_motion_term = -adiabatic_term

    # 4. 非绝热加热项
    tnswrf = surface_t1[idx_tnswrf]
    tnlwrf = surface_t1[idx_tnlwrf]
    snswrf = surface_t1[idx_snswrf]
    snlwrf = surface_t1[idx_snlwrf]

    A_sw = tnswrf - snswrf
    A_lw = tnlwrf - snlwrf

    w_sw = 0.5 * q + 0.3 * o3 + 0.2 * (clwc + ciwc)
    w_sw_sum = np.sum(w_sw, axis=0, keepdims=True)
    w_sw_sum = np.maximum(w_sw_sum, 1e-10)
    w_sw_norm = w_sw / w_sw_sum

    w_lw = 0.7 * q + 0.3 * (clwc + ciwc)
    w_lw_sum = np.sum(w_lw, axis=0, keepdims=True)
    w_lw_sum = np.maximum(w_lw_sum, 1e-10)
    w_lw_norm = w_lw / w_lw_sum

    dp = np.zeros(5)
    for i in range(5):
        if i == 0:
            dp[i] = (pressure_levels[0] + pressure_levels[1]) / 2 - 0
        elif i == 4:
            dp[i] = 100000 - (pressure_levels[3] + pressure_levels[4]) / 2
        else:
            dp[i] = (pressure_levels[i-1] + pressure_levels[i]) / 2 - (pressure_levels[i] + pressure_levels[i+1]) / 2
    dp = dp.reshape(5, 1, 1)

    Q_rad_sw = (g / c_p) * A_sw * w_sw_norm / (dp / 100)
    Q_rad_lw = (g / c_p) * A_lw * w_lw_norm / (dp / 100)
    Q_rad = (Q_rad_sw + Q_rad_lw) / 1000

    lsrr = surface_t1[idx_lsrr]
    crr = surface_t1[idx_crr]
    total_precip = lsrr + crr

    latent_profile = np.array([0.1, 0.2, 0.4, 0.2, 0.1]).reshape(5, 1, 1)
    Q_latent = (L_v / c_p) * total_precip * latent_profile / 1000

    sshf = surface_t1[idx_sshf]
    F_sen = sshf * 7 / dt

    sensible_profile = np.array([0.0, 0.0, 0.0, 0.1, 0.9]).reshape(5, 1, 1)
    rho_surface = pressure_levels[-1] / (R_d * t[-1])

    Q_sensible = np.zeros_like(t)
    Q_sensible[-1] = F_sen / (rho_surface * c_p * 1000) / 100

    Q_diabatic = Q_rad + Q_latent + Q_sensible * sensible_profile

    # 5. 理论温度趋势（使用权重参数）
    dT_dt_theoretical = (ALPHA * horizontal_advection +
                         BETA * vertical_motion_term +
                         GAMMA * Q_diabatic)

    residual = dT_dt_observed - dT_dt_theoretical

    # 统计各层
    pressure_list = [200, 300, 500, 700, 850]
    layer_results = {}
    for i, p in enumerate(pressure_list):
        obs = dT_dt_observed[i]
        theo = dT_dt_theoretical[i]
        res = residual[i]

        obs_abs_mean = np.mean(np.abs(obs))
        theo_abs_mean = np.mean(np.abs(theo))
        res_abs_mean = np.mean(np.abs(res))
        closure = 1 - res_abs_mean / (obs_abs_mean + 1e-10)

        layer_results[p] = {
            'observed': obs_abs_mean,
            'theoretical': theo_abs_mean,
            'residual': res_abs_mean,
            'closure': closure
        }

    # 全场统计
    obs_all = dT_dt_observed
    theo_all = dT_dt_theoretical
    res_all = residual

    rmse = np.sqrt(np.mean(res_all ** 2))

    # 相关系数
    obs_flat = obs_all.flatten()
    theo_flat = theo_all.flatten()
    corr = np.corrcoef(obs_flat, theo_flat)[0, 1]

    closure_rate = 1 - np.mean(np.abs(res_all)) / (np.mean(np.abs(obs_all)) + 1e-10)

    return {
        'week': week,
        'observed_mean': np.mean(obs_all),
        'observed_std': np.std(obs_all),
        'theoretical_mean': np.mean(theo_all),
        'theoretical_std': np.std(theo_all),
        'residual_mean': np.mean(res_all),
        'residual_std': np.std(res_all),
        'rmse': rmse,
        'correlation': corr,
        'closure_rate': closure_rate,
        'layer_results': layer_results
    }


def test_52_weeks_parallel():
    """在52周数据上并行测试温度趋势方程"""

    print("="*70)
    print("温度局地变化方程测试 - 52周训练数据 (8核并行)")
    print("="*70)
    print(f"权重参数: ALPHA={ALPHA}, BETA={BETA}, GAMMA={GAMMA}")
    print("="*70)

    # 加载数据
    print("\nLoading ERA5 2023 weekly data (52 weeks)...")
    with h5.File('/gz-data/ERA5_2023_weekly_new.h5', 'r') as f:
        surface_data = f['surface'][:]
        upper_air_data = f['upper_air'][:]

    print(f"Surface shape: {surface_data.shape}")  # (52, 26, 721, 1440)
    print(f"Upper air shape: {upper_air_data.shape}")  # (52, 10, 5, 721, 1440)

    n_weeks = surface_data.shape[0] - 1  # 51个可用的周对
    print(f"\nProcessing {n_weeks} week pairs with 8 workers...")

    # 压力层
    pressure_levels = [200, 300, 500, 700, 850]

    # 存储统计结果
    all_results = {
        'observed_mean': [],
        'observed_std': [],
        'theoretical_mean': [],
        'theoretical_std': [],
        'residual_mean': [],
        'residual_std': [],
        'closure_rate': [],
        'rmse': [],
        'correlation': []
    }

    # 按层统计
    layer_stats = {p: {'observed': [], 'theoretical': [], 'residual': [], 'closure': []}
                   for p in pressure_levels}

    # 并行处理
    n_workers = 8
    with ProcessPoolExecutor(max_workers=n_workers,
                            initializer=init_worker,
                            initargs=(surface_data, upper_air_data)) as executor:

        # 提交所有任务
        futures = {executor.submit(process_week, week): week for week in range(n_weeks)}

        # 使用tqdm显示进度
        results_list = []
        with tqdm(total=n_weeks, desc="Processing weeks") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results_list.append(result)
                pbar.update(1)
                pbar.set_postfix({
                    'week': result['week'],
                    'closure': f"{result['closure_rate']*100:.1f}%"
                })

    # 按周排序结果
    results_list.sort(key=lambda x: x['week'])

    # 汇总结果
    for result in results_list:
        all_results['observed_mean'].append(result['observed_mean'])
        all_results['observed_std'].append(result['observed_std'])
        all_results['theoretical_mean'].append(result['theoretical_mean'])
        all_results['theoretical_std'].append(result['theoretical_std'])
        all_results['residual_mean'].append(result['residual_mean'])
        all_results['residual_std'].append(result['residual_std'])
        all_results['rmse'].append(result['rmse'])
        all_results['correlation'].append(result['correlation'])
        all_results['closure_rate'].append(result['closure_rate'])

        for p in pressure_levels:
            layer_stats[p]['observed'].append(result['layer_results'][p]['observed'])
            layer_stats[p]['theoretical'].append(result['layer_results'][p]['theoretical'])
            layer_stats[p]['residual'].append(result['layer_results'][p]['residual'])
            layer_stats[p]['closure'].append(result['layer_results'][p]['closure'])

    # 打印总结
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)

    print("\n【全场统计】")
    print(f"  观测温度趋势 |∂T/∂t|:")
    print(f"    均值: {np.mean(all_results['observed_mean']):.2e} K/s")
    print(f"    标准差: {np.mean(all_results['observed_std']):.2e} K/s")
    print(f"  理论温度趋势:")
    print(f"    均值: {np.mean(all_results['theoretical_mean']):.2e} K/s")
    print(f"    标准差: {np.mean(all_results['theoretical_std']):.2e} K/s")
    print(f"  残差:")
    print(f"    均值: {np.mean(all_results['residual_mean']):.2e} K/s")
    print(f"    标准差: {np.mean(all_results['residual_std']):.2e} K/s")
    print(f"  RMSE: {np.mean(all_results['rmse']):.2e} K/s")
    print(f"  相关系数: {np.nanmean(all_results['correlation']):.4f}")
    print(f"  平均闭合率: {np.mean(all_results['closure_rate'])*100:.1f}%")

    print("\n【各压力层统计】")
    print("-"*70)
    print(f"{'压力层':<10} {'|观测|均值':<15} {'|理论|均值':<15} {'|残差|均值':<15} {'闭合率':<10}")
    print("-"*70)
    for p in pressure_levels:
        obs_mean = np.mean(layer_stats[p]['observed'])
        theo_mean = np.mean(layer_stats[p]['theoretical'])
        res_mean = np.mean(layer_stats[p]['residual'])
        closure_mean = np.mean(layer_stats[p]['closure'])
        print(f"{p:>6} hPa  {obs_mean:<15.2e} {theo_mean:<15.2e} {res_mean:<15.2e} {closure_mean*100:>6.1f}%")
    print("-"*70)

    # 季节变化分析
    print("\n【季节变化分析】")
    seasons = {
        'DJF (冬)': list(range(0, 9)) + list(range(48, 51)),
        'MAM (春)': list(range(9, 22)),
        'JJA (夏)': list(range(22, 35)),
        'SON (秋)': list(range(35, 48)),
    }

    for season, weeks in seasons.items():
        valid_weeks = [w for w in weeks if w < len(all_results['closure_rate'])]
        if valid_weeks:
            season_closure = np.mean([all_results['closure_rate'][w] for w in valid_weeks])
            season_rmse = np.mean([all_results['rmse'][w] for w in valid_weeks])
            print(f"  {season}: 闭合率={season_closure*100:.1f}%, RMSE={season_rmse:.2e}")

    # 每周详细结果
    print("\n【每周闭合率详情】")
    print("-"*70)
    for i, result in enumerate(results_list):
        if i % 5 == 0:
            print(f"Week {result['week']+1:2d}: {result['closure_rate']*100:5.1f}%", end="  ")
        else:
            print(f"Week {result['week']+1:2d}: {result['closure_rate']*100:5.1f}%", end="  ")
        if (i + 1) % 5 == 0:
            print()
    print()
    print("-"*70)

    print("\n" + "="*70)
    print("测试完成")
    print("="*70)

    return all_results, layer_stats


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    all_results, layer_stats = test_52_weeks_parallel()
