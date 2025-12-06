# ablation_all_experiments_v2.py
# 物理约束消融实验：全部7组实验
# 添加更多评估指标：ACC, Spatial Gradient MSE, Power Spectrum, SSIM
# 添加物理闭合率指标（5种物理约束的物理一致性）

import sys
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import h5py as h5
import torch.fft as fft

from canglong import Canglong, CanglongV3  # Canglong=V1(无风向约束), CanglongV3(有风向约束)

# ============ 全局缓存的常量掩码 ============
MASK_DIR = PROJECT_ROOT / 'constant_masks'
_CACHED_MASKS = {}

def get_cached_masks(device):
    """获取缓存的常量掩码"""
    global _CACHED_MASKS
    device_key = str(device)
    if device_key not in _CACHED_MASKS:
        print(f"Loading constant masks to {device}...")

        land_path = os.path.join(MASK_DIR, 'is_land.pt')
        land_mask = torch.load(land_path, map_location=device, weights_only=True).float()

        basin_path = os.path.join(MASK_DIR, 'hydrobasin_exorheic_mask.pt')
        basin_mask = torch.load(basin_path, map_location=device, weights_only=True).float()

        ocean_mask = 1.0 - land_mask

        cs_path_corrected = os.path.join(MASK_DIR, 'csol_bulk_025deg_721x1440_corrected.pt')
        cs_path_original = os.path.join(MASK_DIR, 'csol_bulk_025deg_721x1440.pt')

        if os.path.exists(cs_path_corrected):
            cs_soil_bulk = torch.load(cs_path_corrected, map_location=device, weights_only=True).float()
            cs_soil_bulk = torch.clamp(cs_soil_bulk, min=1e6, max=1e7)
        elif os.path.exists(cs_path_original):
            cs_soil_bulk = torch.load(cs_path_original, map_location=device, weights_only=True).float()
            nan_mask = torch.isnan(cs_soil_bulk)
            if not nan_mask.all():
                valid_values = cs_soil_bulk[~nan_mask]
                if valid_values.median() < 1e5:
                    cs_soil_bulk[~nan_mask] *= 1000
            cs_soil_bulk = torch.where(nan_mask | (cs_soil_bulk < 1e5),
                                       torch.tensor(2e6, device=device),
                                       cs_soil_bulk)
            cs_soil_bulk = torch.clamp(cs_soil_bulk, min=1e6, max=1e7)
        else:
            cs_soil_bulk = torch.ones(721, 1440, device=device) * 2e6

        dem_path = os.path.join(MASK_DIR, 'DEM.pt')
        if os.path.exists(dem_path):
            dem = torch.load(dem_path, map_location=device, weights_only=True).float()
        else:
            dem = torch.zeros(721, 1440, device=device)

        _CACHED_MASKS[device_key] = {
            'land_mask': land_mask,
            'basin_mask': basin_mask,
            'ocean_mask': ocean_mask,
            'cs_soil_bulk': cs_soil_bulk,
            'dem': dem
        }
        print(f"Loaded masks: land {land_mask.shape}, basin {basin_mask.shape}")

    return _CACHED_MASKS[device_key]


# ============ 辅助函数 ============
def denormalize_surface(tensor, surface_mean, surface_std):
    return tensor * surface_std + surface_mean

def denormalize_upper(tensor, upper_mean, upper_std):
    return tensor * upper_std + upper_mean


# ============ 空间评估指标 ============

def calculate_acc(pred, target):
    """
    计算异常相关系数 (Anomaly Correlation Coefficient)
    ACC = sum((pred - pred_mean) * (target - target_mean)) /
          sqrt(sum((pred - pred_mean)^2) * sum((target - target_mean)^2))
    """
    # 展平空间维度
    pred_flat = pred.reshape(pred.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)

    # 去除均值（计算异常）
    pred_anom = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_anom = target_flat - target_flat.mean(dim=1, keepdim=True)

    # 计算相关系数
    numerator = (pred_anom * target_anom).sum(dim=1)
    denominator = torch.sqrt((pred_anom ** 2).sum(dim=1) * (target_anom ** 2).sum(dim=1))

    acc = numerator / (denominator + 1e-8)
    return acc.mean().item()


def calculate_spatial_gradient_mse(pred, target):
    """
    计算空间梯度场的MSE
    评估空间结构的相似性
    """
    # 计算x方向梯度 (东西向)
    pred_grad_x = pred[..., :, 1:] - pred[..., :, :-1]
    target_grad_x = target[..., :, 1:] - target[..., :, :-1]

    # 计算y方向梯度 (南北向)
    pred_grad_y = pred[..., 1:, :] - pred[..., :-1, :]
    target_grad_y = target[..., 1:, :] - target[..., :-1, :]

    # 计算梯度MSE
    mse_x = torch.nn.functional.mse_loss(pred_grad_x, target_grad_x)
    mse_y = torch.nn.functional.mse_loss(pred_grad_y, target_grad_y)

    return (mse_x + mse_y).item() / 2


def calculate_power_spectrum_error(pred, target):
    """
    计算功率谱误差
    评估不同尺度的能量分布
    """
    # 取最后两个维度做2D FFT
    pred_2d = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    target_2d = target.reshape(-1, target.shape[-2], target.shape[-1])

    # 计算2D功率谱
    pred_fft = fft.fft2(pred_2d)
    target_fft = fft.fft2(target_2d)

    pred_power = torch.abs(pred_fft) ** 2
    target_power = torch.abs(target_fft) ** 2

    # 计算功率谱的相对误差
    error = torch.abs(pred_power - target_power) / (target_power + 1e-8)

    return error.mean().item()


def calculate_ssim(pred, target, window_size=11):
    """
    计算结构相似性指数 (SSIM)
    简化版本，逐通道计算
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # 展平为2D
    pred_2d = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    target_2d = target.reshape(-1, target.shape[-2], target.shape[-1])

    # 计算均值
    mu_pred = pred_2d.mean(dim=(-2, -1), keepdim=True)
    mu_target = target_2d.mean(dim=(-2, -1), keepdim=True)

    # 计算方差和协方差
    sigma_pred = ((pred_2d - mu_pred) ** 2).mean(dim=(-2, -1))
    sigma_target = ((target_2d - mu_target) ** 2).mean(dim=(-2, -1))
    sigma_pred_target = ((pred_2d - mu_pred) * (target_2d - mu_target)).mean(dim=(-2, -1))

    # 计算SSIM
    ssim = ((2 * mu_pred.squeeze() * mu_target.squeeze() + C1) * (2 * sigma_pred_target + C2)) / \
           ((mu_pred.squeeze() ** 2 + mu_target.squeeze() ** 2 + C1) * (sigma_pred + sigma_target + C2))

    return ssim.mean().item()


# ============ 物理闭合率计算函数 ============

def calculate_water_closure_rate(input_surface_normalized, output_surface_normalized,
                                  surface_mean, surface_std):
    """
    计算水量平衡闭合率
    闭合率 = 1 - |残差| / |左侧项|
    """
    input_physical = denormalize_surface(input_surface_normalized, surface_mean, surface_std)
    output_physical = denormalize_surface(output_surface_normalized, surface_mean, surface_std)

    device = output_physical.device
    masks = get_cached_masks(device)
    land_mask = masks['land_mask']
    basin_mask = masks['basin_mask']

    idx_lsrr, idx_crr, idx_slhf, idx_ro, idx_swvl = 4, 5, 13, 23, 25
    week_seconds = 7 * 24 * 3600
    L_v = 2.5e6
    SOIL_DEPTH = 2.89

    t0 = input_physical[:, :, -1, :, :]
    t1 = output_physical[:, :, 0, :, :]

    # 左侧项：土壤水变化
    delta_soil_water = (t1[:, idx_swvl] - t0[:, idx_swvl]) * SOIL_DEPTH

    # 右侧项：P - E - R
    P_land = (t1[:, idx_lsrr] + t1[:, idx_crr]) * week_seconds / 1000.0 * land_mask
    E_land = torch.abs(t1[:, idx_slhf]) * 7 / L_v / 1000.0 * land_mask
    R = t1[:, idx_ro] * 7 * land_mask
    rhs = (P_land - E_land - R) * basin_mask

    # 残差
    residual = (delta_soil_water - rhs) * basin_mask

    # 闭合率 = 1 - |残差| / max(|左侧|, |右侧|)
    lhs_magnitude = torch.abs(delta_soil_water * basin_mask).mean()
    rhs_magnitude = torch.abs(rhs).mean()
    residual_magnitude = torch.abs(residual).mean()

    denominator = torch.max(lhs_magnitude, rhs_magnitude) + 1e-10
    closure_rate = 1.0 - residual_magnitude / denominator

    return closure_rate.item()


def calculate_energy_closure_rate(input_surface_normalized, output_surface_normalized,
                                   surface_mean, surface_std):
    """
    计算能量平衡闭合率
    """
    device = output_surface_normalized.device
    masks = get_cached_masks(device)
    land_mask = masks['land_mask']
    cs_soil_bulk = masks['cs_soil_bulk']

    input_physical = denormalize_surface(input_surface_normalized, surface_mean, surface_std)
    output_physical = denormalize_surface(output_surface_normalized, surface_mean, surface_std)

    idx_slhf, idx_sshf, idx_sw_net, idx_lw_net = 13, 14, 15, 16
    idx_stl, idx_swvl = 24, 25

    t0 = input_physical[:, :, -1, :, :]
    t1 = output_physical[:, :, 0, :, :]

    week_seconds = 7 * 24 * 3600
    D = 2.89
    c_w = 4.184e6

    # 左侧项：净辐射 R_n
    sw_net = t1[:, idx_sw_net] * land_mask
    lw_net = t1[:, idx_lw_net] * land_mask
    R_n = sw_net + lw_net

    # 右侧项：LE + H + G
    LE = -t1[:, idx_slhf] * 7 / week_seconds * land_mask
    H = -t1[:, idx_sshf] * 7 / week_seconds * land_mask

    delta_T_soil = (t1[:, idx_stl] - t0[:, idx_stl]) * land_mask
    theta = t1[:, idx_swvl] * land_mask
    C_soil = (cs_soil_bulk.unsqueeze(0) + theta * c_w) * land_mask.unsqueeze(0)
    G = C_soil * D * delta_T_soil / week_seconds

    rhs = (LE + H + G) * land_mask.unsqueeze(0)

    # 残差
    residual = (R_n - rhs.squeeze(1)) * land_mask
    residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    # 闭合率
    lhs_magnitude = torch.abs(R_n).mean()
    rhs_magnitude = torch.abs(rhs).mean()
    residual_magnitude = torch.abs(residual).mean()

    denominator = torch.max(lhs_magnitude, rhs_magnitude) + 1e-10
    closure_rate = 1.0 - residual_magnitude / denominator

    return closure_rate.item()


def calculate_hydrostatic_closure_rate(output_upper_normalized, output_surface_normalized,
                                        upper_mean, upper_std, surface_mean, surface_std):
    """
    计算静力平衡闭合率
    """
    output_upper_physical = denormalize_upper(output_upper_normalized, upper_mean, upper_std)
    device = output_upper_physical.device

    R_d = 287

    phi_all = output_upper_physical[:, 1, :, 0, :, :]
    temp_all = output_upper_physical[:, 2, :, 0, :, :]

    total_closure = 0.0
    count = 0

    layer_pairs = [
        (0, 1, 200, 300),
        (1, 2, 300, 500),
        (2, 3, 500, 700),
        (3, 4, 700, 850),
    ]

    for idx_upper, idx_lower, p_upper, p_lower in layer_pairs:
        phi_upper = phi_all[:, idx_upper, :, :]
        phi_lower = phi_all[:, idx_lower, :, :]
        temp_upper = temp_all[:, idx_upper, :, :]
        temp_lower = temp_all[:, idx_lower, :, :]

        # 模型预测的位势厚度
        delta_phi_model = phi_upper - phi_lower

        # 物理计算的位势厚度
        temp_avg = (temp_upper + temp_lower) / 2
        delta_phi_physical = R_d * temp_avg * torch.log(torch.tensor(p_lower/p_upper, device=device))

        residual = delta_phi_model - delta_phi_physical
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

        lhs_magnitude = torch.abs(delta_phi_model).mean()
        rhs_magnitude = torch.abs(delta_phi_physical).mean()
        residual_magnitude = torch.abs(residual).mean()

        denominator = torch.max(lhs_magnitude, rhs_magnitude) + 1e-10
        closure = 1.0 - residual_magnitude / denominator

        total_closure += closure.item()
        count += 1

    return total_closure / max(count, 1)


# 温度方程权重
ALPHA = 0.01
BETA = 0.01
GAMMA = 0.1

def calculate_temperature_closure_rate(input_upper_normalized, output_upper_normalized,
                                        input_surface_normalized, output_surface_normalized,
                                        upper_mean, upper_std, surface_mean, surface_std):
    """
    计算温度局地变化方程闭合率
    """
    device = input_upper_normalized.device

    R_d = 287.0
    c_p = 1004.0
    g = 9.8
    L_v = 2.5e6

    input_upper_physical = input_upper_normalized * upper_std + upper_mean
    output_upper_physical = output_upper_normalized * upper_std + upper_mean
    output_surface_physical = output_surface_normalized * surface_std + surface_mean

    idx_t, idx_u, idx_v, idx_w = 2, 3, 4, 5
    idx_q, idx_o3, idx_clwc, idx_ciwc = 6, 0, 9, 8
    idx_tnswrf, idx_tnlwrf = 0, 1
    idx_lsrr, idx_crr = 4, 5
    idx_snswrf, idx_snlwrf = 15, 16

    t_t1 = input_upper_physical[:, idx_t, :, -1, :, :]
    t_t2 = output_upper_physical[:, idx_t, :, 0, :, :]

    u = output_upper_physical[:, idx_u, :, 0, :, :]
    v = output_upper_physical[:, idx_v, :, 0, :, :]
    w = output_upper_physical[:, idx_w, :, 0, :, :]
    t = output_upper_physical[:, idx_t, :, 0, :, :]
    q = output_upper_physical[:, idx_q, :, 0, :, :]
    o3 = output_upper_physical[:, idx_o3, :, 0, :, :]
    clwc = output_upper_physical[:, idx_clwc, :, 0, :, :]
    ciwc = output_upper_physical[:, idx_ciwc, :, 0, :, :]

    dt = 7 * 24 * 3600

    # 左侧项：观测的温度变化率
    dT_dt_observed = (t_t2 - t_t1) / dt

    dlat = 0.25 * np.pi / 180
    dlon = 0.25 * np.pi / 180
    R_earth = 6.371e6

    lat_values = torch.linspace(-90, 90, 721, device=device) * np.pi / 180
    cos_lat = torch.cos(lat_values).view(1, 1, -1, 1)

    t_padded_x = torch.nn.functional.pad(t, (1, 1, 0, 0), mode='circular')
    dT_dx = (t_padded_x[:, :, :, 2:] - t_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)

    t_padded_y = torch.nn.functional.pad(t, (0, 0, 1, 1), mode='replicate')
    dT_dy = (t_padded_y[:, :, 2:, :] - t_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    horizontal_advection = -(u * dT_dx + v * dT_dy)

    pressure_levels = torch.tensor([200, 300, 500, 700, 850], device=device, dtype=torch.float32) * 100
    p_3d = pressure_levels.view(1, 5, 1, 1).expand_as(t)

    dT_dp = torch.zeros_like(t)
    for i in range(5):
        if i == 0:
            dT_dp[:, i] = (t[:, i+1] - t[:, i]) / (pressure_levels[i+1] - pressure_levels[i])
        elif i == 4:
            dT_dp[:, i] = (t[:, i] - t[:, i-1]) / (pressure_levels[i] - pressure_levels[i-1])
        else:
            dT_dp[:, i] = (t[:, i+1] - t[:, i-1]) / (pressure_levels[i+1] - pressure_levels[i-1])

    adiabatic_term = (R_d * t / (c_p * p_3d) - dT_dp) * w
    vertical_motion_term = -adiabatic_term

    tnswrf = output_surface_physical[:, idx_tnswrf, 0, :, :]
    tnlwrf = output_surface_physical[:, idx_tnlwrf, 0, :, :]
    snswrf = output_surface_physical[:, idx_snswrf, 0, :, :]
    snlwrf = output_surface_physical[:, idx_snlwrf, 0, :, :]

    A_sw = tnswrf - snswrf
    A_lw = tnlwrf - snlwrf

    w_sw = 0.5 * q + 0.3 * o3 + 0.2 * (clwc + ciwc)
    w_sw_sum = w_sw.sum(dim=1, keepdim=True).clamp(min=1e-10)
    w_sw_norm = w_sw / w_sw_sum

    w_lw = 0.7 * q + 0.3 * (clwc + ciwc)
    w_lw_sum = w_lw.sum(dim=1, keepdim=True).clamp(min=1e-10)
    w_lw_norm = w_lw / w_lw_sum

    dp = torch.zeros(5, device=device)
    for i in range(5):
        if i == 0:
            dp[i] = (pressure_levels[0] + pressure_levels[1]) / 2 - 0
        elif i == 4:
            dp[i] = 100000 - (pressure_levels[3] + pressure_levels[4]) / 2
        else:
            dp[i] = (pressure_levels[i-1] + pressure_levels[i]) / 2 - (pressure_levels[i] + pressure_levels[i+1]) / 2
    dp = dp.view(1, 5, 1, 1)

    Q_rad_sw = (g / c_p) * A_sw.unsqueeze(1) * w_sw_norm / (dp / 100)
    Q_rad_lw = (g / c_p) * A_lw.unsqueeze(1) * w_lw_norm / (dp / 100)
    Q_rad = (Q_rad_sw + Q_rad_lw) / 1000

    lsrr = output_surface_physical[:, idx_lsrr, 0, :, :]
    crr = output_surface_physical[:, idx_crr, 0, :, :]
    total_precip = lsrr + crr

    latent_profile = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device).view(1, 5, 1, 1)
    Q_latent = (L_v / c_p) * total_precip.unsqueeze(1) * latent_profile / 1000

    Q_diabatic = Q_rad + Q_latent

    # 右侧项：理论温度变化率（考虑系数）
    dT_dt_theoretical = (ALPHA * horizontal_advection +
                         BETA * vertical_motion_term +
                         GAMMA * Q_diabatic)

    residual = dT_dt_observed - dT_dt_theoretical

    lhs_magnitude = torch.abs(dT_dt_observed).mean()
    rhs_magnitude = torch.abs(dT_dt_theoretical).mean()
    residual_magnitude = torch.abs(residual).mean()

    denominator = torch.max(lhs_magnitude, rhs_magnitude) + 1e-10
    closure_rate = 1.0 - residual_magnitude / denominator

    return closure_rate.item()


PGF_COEFFICIENTS = [0.99, 0.99, 0.93, 0.81, 0.39]
NS_LAYER_WEIGHTS = [1.0, 1.0, 1.2, 0.8, 0.3]

def calculate_momentum_closure_rate(input_upper_normalized, output_upper_normalized,
                                     input_surface_normalized, output_surface_normalized,
                                     upper_mean, upper_std, surface_mean, surface_std):
    """
    计算动量方程（纳维-斯托克斯）闭合率
    """
    device = input_upper_normalized.device

    OMEGA = 7.2921e-5
    R_d = 287.0
    R_earth = 6.371e6

    input_upper_physical = input_upper_normalized * upper_std + upper_mean
    output_upper_physical = output_upper_normalized * upper_std + upper_mean
    output_surface_physical = output_surface_normalized * surface_std + surface_mean

    idx_z, idx_t, idx_u, idx_v, idx_w = 1, 2, 3, 4, 5
    idx_iews, idx_inss = 11, 12

    dt = 7 * 24 * 3600

    u_t1 = input_upper_physical[:, idx_u, :, -1, :, :]
    u_t2 = output_upper_physical[:, idx_u, :, 0, :, :]
    v_t1 = input_upper_physical[:, idx_v, :, -1, :, :]
    v_t2 = output_upper_physical[:, idx_v, :, 0, :, :]
    w = output_upper_physical[:, idx_w, :, 0, :, :]
    t = output_upper_physical[:, idx_t, :, 0, :, :]
    phi = output_upper_physical[:, idx_z, :, 0, :, :]

    dlat = 0.25 * np.pi / 180
    dlon = 0.25 * np.pi / 180

    lat_values = torch.linspace(-90, 90, 721, device=device) * np.pi / 180
    cos_lat = torch.cos(lat_values).view(1, 1, -1, 1).clamp(min=0.01)
    f = 2 * OMEGA * torch.sin(lat_values).view(1, 1, -1, 1)

    u = u_t2
    v = v_t2

    # 右侧项：科氏力 + 气压梯度力
    coriolis_u = f * v
    coriolis_v = -f * u

    phi_padded_x = torch.nn.functional.pad(phi, (1, 1, 0, 0), mode='circular')
    dphi_dx = (phi_padded_x[:, :, :, 2:] - phi_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)

    phi_padded_y = torch.nn.functional.pad(phi, (0, 0, 1, 1), mode='replicate')
    dphi_dy = (phi_padded_y[:, :, 2:, :] - phi_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    pgf_u = -dphi_dx
    pgf_v = -dphi_dy

    pgf_coeffs = torch.tensor(PGF_COEFFICIENTS, device=device, dtype=torch.float32).view(1, 5, 1, 1)
    pgf_u = pgf_u * pgf_coeffs
    pgf_v = pgf_v * pgf_coeffs

    # 左侧项
    du_dt_observed = (u_t2 - u_t1) / dt
    dv_dt_observed = (v_t2 - v_t1) / dt

    u_padded_x = torch.nn.functional.pad(u, (1, 1, 0, 0), mode='circular')
    du_dx = (u_padded_x[:, :, :, 2:] - u_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)
    u_padded_y = torch.nn.functional.pad(u, (0, 0, 1, 1), mode='replicate')
    du_dy = (u_padded_y[:, :, 2:, :] - u_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    v_padded_x = torch.nn.functional.pad(v, (1, 1, 0, 0), mode='circular')
    dv_dx = (v_padded_x[:, :, :, 2:] - v_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)
    v_padded_y = torch.nn.functional.pad(v, (0, 0, 1, 1), mode='replicate')
    dv_dy = (v_padded_y[:, :, 2:, :] - v_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    u_advection_h = u * du_dx + v * du_dy
    v_advection_h = u * dv_dx + v * dv_dy

    pressure_levels = torch.tensor([200, 300, 500, 700, 850], device=device, dtype=torch.float32) * 100

    du_dp = torch.zeros_like(u)
    dv_dp = torch.zeros_like(v)

    for i in range(5):
        if i == 0:
            du_dp[:, i] = (u[:, i+1] - u[:, i]) / (pressure_levels[i+1] - pressure_levels[i])
            dv_dp[:, i] = (v[:, i+1] - v[:, i]) / (pressure_levels[i+1] - pressure_levels[i])
        elif i == 4:
            du_dp[:, i] = (u[:, i] - u[:, i-1]) / (pressure_levels[i] - pressure_levels[i-1])
            dv_dp[:, i] = (v[:, i] - v[:, i-1]) / (pressure_levels[i] - pressure_levels[i-1])
        else:
            du_dp[:, i] = (u[:, i+1] - u[:, i-1]) / (pressure_levels[i+1] - pressure_levels[i-1])
            dv_dp[:, i] = (v[:, i+1] - v[:, i-1]) / (pressure_levels[i+1] - pressure_levels[i-1])

    u_advection_v = w * du_dp
    v_advection_v = w * dv_dp

    tau_x = output_surface_physical[:, idx_iews, 0, :, :]
    tau_y = output_surface_physical[:, idx_inss, 0, :, :]

    p_850 = pressure_levels[4]
    t_850 = t[:, 4, :, :]
    rho_850 = p_850 / (R_d * t_850)

    h_bl = 1000.0
    friction_u = torch.zeros_like(u)
    friction_v = torch.zeros_like(v)
    friction_u[:, 4] = tau_x / (rho_850 * h_bl)
    friction_v[:, 4] = tau_y / (rho_850 * h_bl)

    lhs_u = du_dt_observed + u_advection_h + u_advection_v
    lhs_v = dv_dt_observed + v_advection_h + v_advection_v

    rhs_u = coriolis_u + pgf_u + friction_u
    rhs_v = coriolis_v + pgf_v + friction_v

    residual_u = lhs_u - rhs_u
    residual_v = lhs_v - rhs_v

    residual_u = torch.nan_to_num(residual_u, nan=0.0, posinf=0.0, neginf=0.0)
    residual_v = torch.nan_to_num(residual_v, nan=0.0, posinf=0.0, neginf=0.0)

    lhs_magnitude = (torch.abs(lhs_u).mean() + torch.abs(lhs_v).mean()) / 2
    rhs_magnitude = (torch.abs(rhs_u).mean() + torch.abs(rhs_v).mean()) / 2
    residual_magnitude = (torch.abs(residual_u).mean() + torch.abs(residual_v).mean()) / 2

    denominator = torch.max(lhs_magnitude, rhs_magnitude) + 1e-10
    closure_rate = 1.0 - residual_magnitude / denominator

    return closure_rate.item()


# ============ 物理约束损失函数 ============

def calculate_water_balance_loss(input_surface_normalized, output_surface_normalized,
                                surface_mean, surface_std):
    """水量平衡损失"""
    input_physical = denormalize_surface(input_surface_normalized, surface_mean, surface_std)
    output_physical = denormalize_surface(output_surface_normalized, surface_mean, surface_std)

    device = output_physical.device
    masks = get_cached_masks(device)
    land_mask = masks['land_mask']
    basin_mask = masks['basin_mask']

    idx_lsrr, idx_crr, idx_slhf, idx_ro, idx_swvl = 4, 5, 13, 23, 25
    week_seconds = 7 * 24 * 3600
    L_v = 2.5e6
    SOIL_DEPTH = 2.89

    t0 = input_physical[:, :, -1, :, :]
    t1 = output_physical[:, :, 0, :, :]

    delta_soil_water = (t1[:, idx_swvl] - t0[:, idx_swvl]) * SOIL_DEPTH
    P_land = (t1[:, idx_lsrr] + t1[:, idx_crr]) * week_seconds / 1000.0 * land_mask
    E_land = torch.abs(t1[:, idx_slhf]) * 7 / L_v / 1000.0 * land_mask
    R = t1[:, idx_ro] * 7 * land_mask

    residual_land = (delta_soil_water - (P_land - E_land - R)) * basin_mask
    return torch.nn.functional.mse_loss(residual_land, torch.zeros_like(residual_land))


def calculate_energy_balance_loss(input_surface_normalized, output_surface_normalized,
                                 surface_mean, surface_std):
    """能量平衡损失"""
    device = output_surface_normalized.device
    masks = get_cached_masks(device)
    land_mask = masks['land_mask']
    cs_soil_bulk = masks['cs_soil_bulk']

    input_physical = denormalize_surface(input_surface_normalized, surface_mean, surface_std)
    output_physical = denormalize_surface(output_surface_normalized, surface_mean, surface_std)

    idx_slhf, idx_sshf, idx_sw_net, idx_lw_net = 13, 14, 15, 16
    idx_stl, idx_swvl = 24, 25

    t0 = input_physical[:, :, -1, :, :]
    t1 = output_physical[:, :, 0, :, :]

    week_seconds = 7 * 24 * 3600
    D = 2.89
    c_w = 4.184e6

    sw_net = t1[:, idx_sw_net] * land_mask
    lw_net = t1[:, idx_lw_net] * land_mask
    R_n = sw_net + lw_net

    LE_raw = t1[:, idx_slhf] * 7 / week_seconds * land_mask
    H_raw = t1[:, idx_sshf] * 7 / week_seconds * land_mask
    LE = -LE_raw
    H = -H_raw

    delta_T_soil = (t1[:, idx_stl] - t0[:, idx_stl]) * land_mask
    theta = t1[:, idx_swvl] * land_mask
    C_soil = (cs_soil_bulk.unsqueeze(0) + theta * c_w) * land_mask.unsqueeze(0)
    G = C_soil * D * delta_T_soil / week_seconds

    residual_land = (R_n - LE - H - G) * land_mask.unsqueeze(0)
    residual_land = torch.nan_to_num(residual_land, nan=0.0, posinf=0.0, neginf=0.0)

    valid_pixels = land_mask.sum()
    if valid_pixels > 0:
        mse_loss = (residual_land ** 2).sum() / valid_pixels / output_physical.shape[0]
    else:
        mse_loss = torch.tensor(0.0, device=device)

    return mse_loss


def calculate_hydrostatic_balance_loss(output_upper_normalized, output_surface_normalized,
                                      upper_mean, upper_std, surface_mean, surface_std):
    """静力平衡损失"""
    output_upper_physical = denormalize_upper(output_upper_normalized, upper_mean, upper_std)
    output_surface_physical = denormalize_surface(output_surface_normalized, surface_mean, surface_std)
    device = output_upper_physical.device
    masks = get_cached_masks(device)
    dem = masks['dem']

    R_d = 287
    g = 9.80665

    phi_all = output_upper_physical[:, 1, :, 0, :, :]
    temp_all = output_upper_physical[:, 2, :, 0, :, :]

    total_loss = torch.tensor(0.0, device=device)
    loss_count = 0

    layer_pairs = [
        (0, 1, 200, 300),
        (1, 2, 300, 500),
        (2, 3, 500, 700),
        (3, 4, 700, 850),
    ]

    for idx_upper, idx_lower, p_upper, p_lower in layer_pairs:
        phi_upper = phi_all[:, idx_upper, :, :]
        phi_lower = phi_all[:, idx_lower, :, :]
        temp_upper = temp_all[:, idx_upper, :, :]
        temp_lower = temp_all[:, idx_lower, :, :]

        delta_phi_model = phi_upper - phi_lower
        temp_avg = (temp_upper + temp_lower) / 2
        delta_phi_physical = R_d * temp_avg * torch.log(torch.tensor(p_lower/p_upper, device=device))

        residual = delta_phi_model - delta_phi_physical
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

        loss = torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))
        total_loss = total_loss + loss
        loss_count += 1

    if dem.sum() > 0:
        t2m = output_surface_physical[:, 10, 0, :, :]
        sp = output_surface_physical[:, 19, 0, :, :]
        phi_850 = phi_all[:, 4, :, :]
        temp_850 = temp_all[:, 4, :, :]
        phi_surface = g * dem.unsqueeze(0)
        delta_phi_model_sfc = phi_850 - phi_surface
        valid_mask = (sp > 85000).float()
        temp_avg_sfc = (t2m + temp_850) / 2
        p_ratio = (sp / 85000.0).clamp(min=0.9, max=1.2)
        delta_phi_physical_sfc = R_d * temp_avg_sfc * torch.log(p_ratio) * valid_mask
        residual_sfc = (delta_phi_model_sfc - delta_phi_physical_sfc) * valid_mask
        residual_sfc = torch.nan_to_num(residual_sfc, nan=0.0, posinf=0.0, neginf=0.0)
        valid_pixels = valid_mask.sum()
        if valid_pixels > 0:
            loss_sfc = (residual_sfc ** 2).sum() / valid_pixels / output_surface_physical.shape[0]
        else:
            loss_sfc = torch.tensor(0.0, device=device)
        total_loss = total_loss + loss_sfc
        loss_count += 1

    return total_loss / max(loss_count, 1)


def calculate_temperature_tendency_loss(input_upper_normalized, output_upper_normalized,
                                       input_surface_normalized, output_surface_normalized,
                                       upper_mean, upper_std, surface_mean, surface_std):
    """温度局地变化方程约束"""
    device = input_upper_normalized.device

    R_d = 287.0
    c_p = 1004.0
    g = 9.8
    L_v = 2.5e6

    input_upper_physical = input_upper_normalized * upper_std + upper_mean
    output_upper_physical = output_upper_normalized * upper_std + upper_mean
    output_surface_physical = output_surface_normalized * surface_std + surface_mean

    idx_t, idx_u, idx_v, idx_w = 2, 3, 4, 5
    idx_q, idx_o3, idx_clwc, idx_ciwc = 6, 0, 9, 8
    idx_tnswrf, idx_tnlwrf = 0, 1
    idx_lsrr, idx_crr = 4, 5
    idx_snswrf, idx_snlwrf = 15, 16

    t_t1 = input_upper_physical[:, idx_t, :, -1, :, :]
    t_t2 = output_upper_physical[:, idx_t, :, 0, :, :]

    u = output_upper_physical[:, idx_u, :, 0, :, :]
    v = output_upper_physical[:, idx_v, :, 0, :, :]
    w = output_upper_physical[:, idx_w, :, 0, :, :]
    t = output_upper_physical[:, idx_t, :, 0, :, :]
    q = output_upper_physical[:, idx_q, :, 0, :, :]
    o3 = output_upper_physical[:, idx_o3, :, 0, :, :]
    clwc = output_upper_physical[:, idx_clwc, :, 0, :, :]
    ciwc = output_upper_physical[:, idx_ciwc, :, 0, :, :]

    dt = 7 * 24 * 3600
    dT_dt_observed = (t_t2 - t_t1) / dt

    dlat = 0.25 * np.pi / 180
    dlon = 0.25 * np.pi / 180
    R_earth = 6.371e6

    lat_values = torch.linspace(-90, 90, 721, device=device) * np.pi / 180
    cos_lat = torch.cos(lat_values).view(1, 1, -1, 1)

    t_padded_x = torch.nn.functional.pad(t, (1, 1, 0, 0), mode='circular')
    dT_dx = (t_padded_x[:, :, :, 2:] - t_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)

    t_padded_y = torch.nn.functional.pad(t, (0, 0, 1, 1), mode='replicate')
    dT_dy = (t_padded_y[:, :, 2:, :] - t_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    horizontal_advection = -(u * dT_dx + v * dT_dy)

    pressure_levels = torch.tensor([200, 300, 500, 700, 850], device=device, dtype=torch.float32) * 100
    p_3d = pressure_levels.view(1, 5, 1, 1).expand_as(t)

    dT_dp = torch.zeros_like(t)
    for i in range(5):
        if i == 0:
            dT_dp[:, i] = (t[:, i+1] - t[:, i]) / (pressure_levels[i+1] - pressure_levels[i])
        elif i == 4:
            dT_dp[:, i] = (t[:, i] - t[:, i-1]) / (pressure_levels[i] - pressure_levels[i-1])
        else:
            dT_dp[:, i] = (t[:, i+1] - t[:, i-1]) / (pressure_levels[i+1] - pressure_levels[i-1])

    adiabatic_term = (R_d * t / (c_p * p_3d) - dT_dp) * w
    vertical_motion_term = -adiabatic_term

    tnswrf = output_surface_physical[:, idx_tnswrf, 0, :, :]
    tnlwrf = output_surface_physical[:, idx_tnlwrf, 0, :, :]
    snswrf = output_surface_physical[:, idx_snswrf, 0, :, :]
    snlwrf = output_surface_physical[:, idx_snlwrf, 0, :, :]

    A_sw = tnswrf - snswrf
    A_lw = tnlwrf - snlwrf

    w_sw = 0.5 * q + 0.3 * o3 + 0.2 * (clwc + ciwc)
    w_sw_sum = w_sw.sum(dim=1, keepdim=True).clamp(min=1e-10)
    w_sw_norm = w_sw / w_sw_sum

    w_lw = 0.7 * q + 0.3 * (clwc + ciwc)
    w_lw_sum = w_lw.sum(dim=1, keepdim=True).clamp(min=1e-10)
    w_lw_norm = w_lw / w_lw_sum

    dp = torch.zeros(5, device=device)
    for i in range(5):
        if i == 0:
            dp[i] = (pressure_levels[0] + pressure_levels[1]) / 2 - 0
        elif i == 4:
            dp[i] = 100000 - (pressure_levels[3] + pressure_levels[4]) / 2
        else:
            dp[i] = (pressure_levels[i-1] + pressure_levels[i]) / 2 - (pressure_levels[i] + pressure_levels[i+1]) / 2
    dp = dp.view(1, 5, 1, 1)

    Q_rad_sw = (g / c_p) * A_sw.unsqueeze(1) * w_sw_norm / (dp / 100)
    Q_rad_lw = (g / c_p) * A_lw.unsqueeze(1) * w_lw_norm / (dp / 100)
    Q_rad = (Q_rad_sw + Q_rad_lw) / 1000

    lsrr = output_surface_physical[:, idx_lsrr, 0, :, :]
    crr = output_surface_physical[:, idx_crr, 0, :, :]
    total_precip = lsrr + crr

    latent_profile = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device).view(1, 5, 1, 1)
    Q_latent = (L_v / c_p) * total_precip.unsqueeze(1) * latent_profile / 1000

    Q_diabatic = Q_rad + Q_latent

    dT_dt_theoretical = (ALPHA * horizontal_advection +
                         BETA * vertical_motion_term +
                         GAMMA * Q_diabatic)

    residual = dT_dt_observed - dT_dt_theoretical
    layer_weights = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0], device=device).view(1, 5, 1, 1)
    weighted_residual = residual * layer_weights

    loss = (weighted_residual ** 2).mean()
    return loss


def calculate_navier_stokes_loss(input_upper_normalized, output_upper_normalized,
                                 input_surface_normalized, output_surface_normalized,
                                 upper_mean, upper_std, surface_mean, surface_std):
    """纳维-斯托克斯方程约束"""
    device = input_upper_normalized.device

    OMEGA = 7.2921e-5
    R_d = 287.0
    R_earth = 6.371e6

    input_upper_physical = input_upper_normalized * upper_std + upper_mean
    output_upper_physical = output_upper_normalized * upper_std + upper_mean
    output_surface_physical = output_surface_normalized * surface_std + surface_mean

    idx_z, idx_t, idx_u, idx_v, idx_w = 1, 2, 3, 4, 5
    idx_iews, idx_inss = 11, 12

    dt = 7 * 24 * 3600

    u_t1 = input_upper_physical[:, idx_u, :, -1, :, :]
    u_t2 = output_upper_physical[:, idx_u, :, 0, :, :]
    v_t1 = input_upper_physical[:, idx_v, :, -1, :, :]
    v_t2 = output_upper_physical[:, idx_v, :, 0, :, :]
    w = output_upper_physical[:, idx_w, :, 0, :, :]
    t = output_upper_physical[:, idx_t, :, 0, :, :]
    phi = output_upper_physical[:, idx_z, :, 0, :, :]

    dlat = 0.25 * np.pi / 180
    dlon = 0.25 * np.pi / 180

    lat_values = torch.linspace(-90, 90, 721, device=device) * np.pi / 180
    cos_lat = torch.cos(lat_values).view(1, 1, -1, 1).clamp(min=0.01)
    f = 2 * OMEGA * torch.sin(lat_values).view(1, 1, -1, 1)

    u = u_t2
    v = v_t2

    coriolis_u = f * v
    coriolis_v = -f * u

    phi_padded_x = torch.nn.functional.pad(phi, (1, 1, 0, 0), mode='circular')
    dphi_dx = (phi_padded_x[:, :, :, 2:] - phi_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)

    phi_padded_y = torch.nn.functional.pad(phi, (0, 0, 1, 1), mode='replicate')
    dphi_dy = (phi_padded_y[:, :, 2:, :] - phi_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    pgf_u = -dphi_dx
    pgf_v = -dphi_dy

    pgf_coeffs = torch.tensor(PGF_COEFFICIENTS, device=device, dtype=torch.float32).view(1, 5, 1, 1)
    pgf_u = pgf_u * pgf_coeffs
    pgf_v = pgf_v * pgf_coeffs

    du_dt_observed = (u_t2 - u_t1) / dt
    dv_dt_observed = (v_t2 - v_t1) / dt

    u_padded_x = torch.nn.functional.pad(u, (1, 1, 0, 0), mode='circular')
    du_dx = (u_padded_x[:, :, :, 2:] - u_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)
    u_padded_y = torch.nn.functional.pad(u, (0, 0, 1, 1), mode='replicate')
    du_dy = (u_padded_y[:, :, 2:, :] - u_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    v_padded_x = torch.nn.functional.pad(v, (1, 1, 0, 0), mode='circular')
    dv_dx = (v_padded_x[:, :, :, 2:] - v_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)
    v_padded_y = torch.nn.functional.pad(v, (0, 0, 1, 1), mode='replicate')
    dv_dy = (v_padded_y[:, :, 2:, :] - v_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    u_advection_h = u * du_dx + v * du_dy
    v_advection_h = u * dv_dx + v * dv_dy

    pressure_levels = torch.tensor([200, 300, 500, 700, 850], device=device, dtype=torch.float32) * 100

    du_dp = torch.zeros_like(u)
    dv_dp = torch.zeros_like(v)

    for i in range(5):
        if i == 0:
            du_dp[:, i] = (u[:, i+1] - u[:, i]) / (pressure_levels[i+1] - pressure_levels[i])
            dv_dp[:, i] = (v[:, i+1] - v[:, i]) / (pressure_levels[i+1] - pressure_levels[i])
        elif i == 4:
            du_dp[:, i] = (u[:, i] - u[:, i-1]) / (pressure_levels[i] - pressure_levels[i-1])
            dv_dp[:, i] = (v[:, i] - v[:, i-1]) / (pressure_levels[i] - pressure_levels[i-1])
        else:
            du_dp[:, i] = (u[:, i+1] - u[:, i-1]) / (pressure_levels[i+1] - pressure_levels[i-1])
            dv_dp[:, i] = (v[:, i+1] - v[:, i-1]) / (pressure_levels[i+1] - pressure_levels[i-1])

    u_advection_v = w * du_dp
    v_advection_v = w * dv_dp

    tau_x = output_surface_physical[:, idx_iews, 0, :, :]
    tau_y = output_surface_physical[:, idx_inss, 0, :, :]

    p_850 = pressure_levels[4]
    t_850 = t[:, 4, :, :]
    rho_850 = p_850 / (R_d * t_850)

    h_bl = 1000.0
    friction_u = torch.zeros_like(u)
    friction_v = torch.zeros_like(v)
    friction_u[:, 4] = tau_x / (rho_850 * h_bl)
    friction_v[:, 4] = tau_y / (rho_850 * h_bl)

    lhs_u = du_dt_observed + u_advection_h + u_advection_v
    lhs_v = dv_dt_observed + v_advection_h + v_advection_v

    rhs_u = coriolis_u + pgf_u + friction_u
    rhs_v = coriolis_v + pgf_v + friction_v

    residual_u = lhs_u - rhs_u
    residual_v = lhs_v - rhs_v

    layer_weights = torch.tensor(NS_LAYER_WEIGHTS, device=device, dtype=torch.float32).view(1, 5, 1, 1)
    weighted_residual_u = residual_u * layer_weights
    weighted_residual_v = residual_v * layer_weights

    weighted_residual_u = torch.nan_to_num(weighted_residual_u, nan=0.0, posinf=0.0, neginf=0.0)
    weighted_residual_v = torch.nan_to_num(weighted_residual_v, nan=0.0, posinf=0.0, neginf=0.0)

    loss_u = (weighted_residual_u ** 2).mean()
    loss_v = (weighted_residual_v ** 2).mean()

    return loss_u + loss_v


# ============ Focus和Tweedie损失 (用于评估) ============

def calculate_focus_variable_loss(output_surface_norm, target_surface_norm,
                                  output_upper_norm, target_upper_norm):
    """重要变量额外MSE损失"""
    precip_pred = output_surface_norm[:, 4, ...] + output_surface_norm[:, 5, ...]
    precip_target = target_surface_norm[:, 4, ...] + target_surface_norm[:, 5, ...]
    precip_loss = torch.nn.functional.mse_loss(precip_pred, precip_target)

    surface_focus_indices = [1, 10]
    surface_focus_pred = output_surface_norm[:, surface_focus_indices, ...]
    surface_focus_target = target_surface_norm[:, surface_focus_indices, ...]
    surface_focus_loss = torch.nn.functional.mse_loss(surface_focus_pred, surface_focus_target)

    upper_u_pred = output_upper_norm[:, 3, [0, 4], ...]
    upper_u_target = target_upper_norm[:, 3, [0, 4], ...]
    upper_focus_loss = torch.nn.functional.mse_loss(upper_u_pred, upper_u_target)

    return precip_loss + surface_focus_loss + upper_focus_loss


TWEEDIE_P_LSRR = 1.54
TWEEDIE_P_CRR = 1.59

def tweedie_deviance(y_true, y_pred, p):
    """Tweedie deviance损失"""
    eps = 1e-8
    mu = torch.clamp(y_pred, min=eps)
    y = torch.clamp(y_true, min=0.0)

    one_minus_p = 1.0 - p
    two_minus_p = 2.0 - p

    term1 = torch.pow(y + eps, two_minus_p) / (one_minus_p * two_minus_p)
    term2 = y * torch.pow(mu, one_minus_p) / one_minus_p
    term3 = torch.pow(mu, two_minus_p) / two_minus_p

    deviance = 2.0 * (term1 - term2 + term3)
    deviance = torch.nan_to_num(deviance, nan=0.0, posinf=0.0, neginf=0.0)

    return deviance.mean()


def calculate_tweedie_loss(output_surface_norm, target_surface_norm,
                           surface_mean, surface_std):
    """Tweedie降水损失"""
    output_physical = output_surface_norm * surface_std + surface_mean
    target_physical = target_surface_norm * surface_std + surface_mean

    idx_lsrr, idx_crr = 4, 5

    lsrr_pred = torch.relu(output_physical[:, idx_lsrr, 0, :, :])
    lsrr_target = torch.relu(target_physical[:, idx_lsrr, 0, :, :])

    crr_pred = torch.relu(output_physical[:, idx_crr, 0, :, :])
    crr_target = torch.relu(target_physical[:, idx_crr, 0, :, :])

    loss_lsrr = tweedie_deviance(lsrr_target, lsrr_pred, TWEEDIE_P_LSRR)
    loss_crr = tweedie_deviance(crr_target, crr_pred, TWEEDIE_P_CRR)

    return loss_lsrr + loss_crr


# ============ 数据集 ============

class WeatherDataset(Dataset):
    def __init__(self, surface_data, upper_air_data, start_idx, end_idx):
        self.surface_data = surface_data
        self.upper_air_data = upper_air_data
        self.length = end_idx - start_idx - 2
        self.start_idx = start_idx
        print(f"Dataset from index {start_idx} to {end_idx}, sample count: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx

        input_surface = self.surface_data[actual_idx:actual_idx+2]
        input_surface = np.transpose(input_surface, (1, 0, 2, 3))

        input_upper_air = self.upper_air_data[actual_idx:actual_idx+2]
        input_upper_air = np.transpose(input_upper_air, (1, 2, 0, 3, 4))

        target_surface = self.surface_data[actual_idx+2:actual_idx+3]
        target_surface = np.transpose(target_surface, (1, 0, 2, 3))

        target_upper_air = self.upper_air_data[actual_idx+2:actual_idx+3]
        target_upper_air = np.transpose(target_upper_air, (1, 2, 0, 3, 4))

        return input_surface, input_upper_air, target_surface, target_upper_air


# ============ 统一训练函数 ============

def run_experiment(exp_id, exp_name, exp_type, lambda_dict,
                   model, train_loader, valid_loader,
                   surface_mean, surface_std, upper_mean, upper_std,
                   device, num_epochs=20, lr=0.0005, save_dir=None):
    """
    运行单个消融实验（包含扩展指标）
    """
    print(f"\n{'='*60}")
    print(f"Experiment {exp_id}: {exp_name}")
    if exp_type == 'single':
        print(f"Lambda: {lambda_dict}")
    elif exp_type == 'full':
        print(f"Lambdas: {lambda_dict}")
    print(f"{'='*60}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    results = []

    for epoch in range(num_epochs):
        # ====== 训练阶段 ======
        model.train()
        train_metrics = {
            'mse_total': 0.0, 'mse_surface': 0.0, 'mse_upper_air': 0.0,
            'focus_loss': 0.0, 'tweedie_loss': 0.0,
            'acc_surface': 0.0, 'acc_upper': 0.0,
            'grad_mse_surface': 0.0, 'grad_mse_upper': 0.0,
            'power_err_surface': 0.0, 'power_err_upper': 0.0,
            'ssim_surface': 0.0, 'ssim_upper': 0.0,
            'closure_water': 0.0, 'closure_energy': 0.0,
            'closure_hydrostatic': 0.0, 'closure_temperature': 0.0,
            'closure_momentum': 0.0
        }

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for input_surface_batch, input_upper_air_batch, target_surface_batch, target_upper_air_batch in train_pbar:
            input_surface_batch = input_surface_batch.float().to(device)
            input_upper_air_batch = input_upper_air_batch.float().to(device)
            target_surface_batch = target_surface_batch.float().to(device)
            target_upper_air_batch = target_upper_air_batch.float().to(device)

            input_surface_norm = (input_surface_batch - surface_mean) / surface_std
            input_upper_air_norm = (input_upper_air_batch - upper_mean) / upper_std
            target_surface_norm = (target_surface_batch - surface_mean) / surface_std
            target_upper_air_norm = (target_upper_air_batch - upper_mean) / upper_std

            optimizer.zero_grad()

            output_surface, output_upper_air = model(input_surface_norm, input_upper_air_norm)

            loss_surface = criterion(output_surface, target_surface_norm)
            loss_upper_air = criterion(output_upper_air, target_upper_air_norm)
            mse_total = loss_surface + loss_upper_air

            # 对于baseline实验，MSE权重除以3，使其与有物理约束的实验损失量级一致
            # 有物理约束时：loss = MSE + physics，假设比例约为 2:1
            # 无物理约束时：loss = MSE，需要降低权重使优化强度一致
            if exp_type == 'baseline':
                loss = mse_total
            else:
                loss = mse_total

            if exp_type == 'single':
                constraint_name = list(lambda_dict.keys())[0]
                lambda_val = lambda_dict[constraint_name]

                if constraint_name == 'water':
                    loss_phys = calculate_water_balance_loss(input_surface_norm, output_surface, surface_mean, surface_std)
                elif constraint_name == 'energy':
                    loss_phys = calculate_energy_balance_loss(input_surface_norm, output_surface, surface_mean, surface_std)
                elif constraint_name == 'hydrostatic':
                    loss_phys = calculate_hydrostatic_balance_loss(output_upper_air, output_surface, upper_mean, upper_std, surface_mean, surface_std)
                elif constraint_name == 'temperature':
                    loss_phys = calculate_temperature_tendency_loss(input_upper_air_norm, output_upper_air, input_surface_norm, output_surface, upper_mean, upper_std, surface_mean, surface_std)
                elif constraint_name == 'momentum':
                    loss_phys = calculate_navier_stokes_loss(input_upper_air_norm, output_upper_air, input_surface_norm, output_surface, upper_mean, upper_std, surface_mean, surface_std)

                loss = loss + lambda_val * loss_phys

            elif exp_type == 'full':
                loss_water = calculate_water_balance_loss(input_surface_norm, output_surface, surface_mean, surface_std)
                loss_energy = calculate_energy_balance_loss(input_surface_norm, output_surface, surface_mean, surface_std)
                loss_hydrostatic = calculate_hydrostatic_balance_loss(output_upper_air, output_surface, upper_mean, upper_std, surface_mean, surface_std)
                loss_temperature = calculate_temperature_tendency_loss(input_upper_air_norm, output_upper_air, input_surface_norm, output_surface, upper_mean, upper_std, surface_mean, surface_std)
                loss_momentum = calculate_navier_stokes_loss(input_upper_air_norm, output_upper_air, input_surface_norm, output_surface, upper_mean, upper_std, surface_mean, surface_std)

                loss = loss + \
                       lambda_dict['water'] * loss_water + \
                       lambda_dict['energy'] * loss_energy + \
                       lambda_dict['hydrostatic'] * loss_hydrostatic + \
                       lambda_dict['temperature'] * loss_temperature + \
                       lambda_dict['momentum'] * loss_momentum

            loss_focus = calculate_focus_variable_loss(output_surface, target_surface_norm, output_upper_air, target_upper_air_norm)
            loss_tweedie = calculate_tweedie_loss(output_surface, target_surface_norm, surface_mean, surface_std)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 累加基础指标
            train_metrics['mse_total'] += mse_total.item()
            train_metrics['mse_surface'] += loss_surface.item()
            train_metrics['mse_upper_air'] += loss_upper_air.item()
            train_metrics['focus_loss'] += loss_focus.item()
            train_metrics['tweedie_loss'] += loss_tweedie.item()

            # 计算空间指标（每batch）
            with torch.no_grad():
                train_metrics['acc_surface'] += calculate_acc(output_surface, target_surface_norm)
                train_metrics['acc_upper'] += calculate_acc(output_upper_air, target_upper_air_norm)
                train_metrics['grad_mse_surface'] += calculate_spatial_gradient_mse(output_surface, target_surface_norm)
                train_metrics['grad_mse_upper'] += calculate_spatial_gradient_mse(output_upper_air, target_upper_air_norm)
                train_metrics['power_err_surface'] += calculate_power_spectrum_error(output_surface, target_surface_norm)
                train_metrics['power_err_upper'] += calculate_power_spectrum_error(output_upper_air, target_upper_air_norm)
                train_metrics['ssim_surface'] += calculate_ssim(output_surface, target_surface_norm)
                train_metrics['ssim_upper'] += calculate_ssim(output_upper_air, target_upper_air_norm)

                # 计算物理闭合率
                train_metrics['closure_water'] += calculate_water_closure_rate(input_surface_norm, output_surface, surface_mean, surface_std)
                train_metrics['closure_energy'] += calculate_energy_closure_rate(input_surface_norm, output_surface, surface_mean, surface_std)
                train_metrics['closure_hydrostatic'] += calculate_hydrostatic_closure_rate(output_upper_air, output_surface, upper_mean, upper_std, surface_mean, surface_std)
                train_metrics['closure_temperature'] += calculate_temperature_closure_rate(input_upper_air_norm, output_upper_air, input_surface_norm, output_surface, upper_mean, upper_std, surface_mean, surface_std)
                train_metrics['closure_momentum'] += calculate_momentum_closure_rate(input_upper_air_norm, output_upper_air, input_surface_norm, output_surface, upper_mean, upper_std, surface_mean, surface_std)

            train_pbar.set_postfix({"mse": f"{mse_total.item():.4f}"})

        n_train = len(train_loader)
        for k in train_metrics:
            train_metrics[k] /= n_train

        # ====== 验证阶段 ======
        model.eval()
        valid_metrics = {
            'mse_total': 0.0, 'mse_surface': 0.0, 'mse_upper_air': 0.0,
            'focus_loss': 0.0, 'tweedie_loss': 0.0,
            'acc_surface': 0.0, 'acc_upper': 0.0,
            'grad_mse_surface': 0.0, 'grad_mse_upper': 0.0,
            'power_err_surface': 0.0, 'power_err_upper': 0.0,
            'ssim_surface': 0.0, 'ssim_upper': 0.0,
            'closure_water': 0.0, 'closure_energy': 0.0,
            'closure_hydrostatic': 0.0, 'closure_temperature': 0.0,
            'closure_momentum': 0.0
        }

        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
            for input_surface_batch, input_upper_air_batch, target_surface_batch, target_upper_air_batch in valid_pbar:
                input_surface_batch = input_surface_batch.float().to(device)
                input_upper_air_batch = input_upper_air_batch.float().to(device)
                target_surface_batch = target_surface_batch.float().to(device)
                target_upper_air_batch = target_upper_air_batch.float().to(device)

                input_surface_norm = (input_surface_batch - surface_mean) / surface_std
                input_upper_air_norm = (input_upper_air_batch - upper_mean) / upper_std
                target_surface_norm = (target_surface_batch - surface_mean) / surface_std
                target_upper_air_norm = (target_upper_air_batch - upper_mean) / upper_std

                output_surface, output_upper_air = model(input_surface_norm, input_upper_air_norm)

                loss_surface = criterion(output_surface, target_surface_norm)
                loss_upper_air = criterion(output_upper_air, target_upper_air_norm)
                mse_total = loss_surface + loss_upper_air

                loss_focus = calculate_focus_variable_loss(output_surface, target_surface_norm, output_upper_air, target_upper_air_norm)
                loss_tweedie = calculate_tweedie_loss(output_surface, target_surface_norm, surface_mean, surface_std)

                # 基础指标
                valid_metrics['mse_total'] += mse_total.item()
                valid_metrics['mse_surface'] += loss_surface.item()
                valid_metrics['mse_upper_air'] += loss_upper_air.item()
                valid_metrics['focus_loss'] += loss_focus.item()
                valid_metrics['tweedie_loss'] += loss_tweedie.item()

                # 空间指标
                valid_metrics['acc_surface'] += calculate_acc(output_surface, target_surface_norm)
                valid_metrics['acc_upper'] += calculate_acc(output_upper_air, target_upper_air_norm)
                valid_metrics['grad_mse_surface'] += calculate_spatial_gradient_mse(output_surface, target_surface_norm)
                valid_metrics['grad_mse_upper'] += calculate_spatial_gradient_mse(output_upper_air, target_upper_air_norm)
                valid_metrics['power_err_surface'] += calculate_power_spectrum_error(output_surface, target_surface_norm)
                valid_metrics['power_err_upper'] += calculate_power_spectrum_error(output_upper_air, target_upper_air_norm)
                valid_metrics['ssim_surface'] += calculate_ssim(output_surface, target_surface_norm)
                valid_metrics['ssim_upper'] += calculate_ssim(output_upper_air, target_upper_air_norm)

                # 物理闭合率
                valid_metrics['closure_water'] += calculate_water_closure_rate(input_surface_norm, output_surface, surface_mean, surface_std)
                valid_metrics['closure_energy'] += calculate_energy_closure_rate(input_surface_norm, output_surface, surface_mean, surface_std)
                valid_metrics['closure_hydrostatic'] += calculate_hydrostatic_closure_rate(output_upper_air, output_surface, upper_mean, upper_std, surface_mean, surface_std)
                valid_metrics['closure_temperature'] += calculate_temperature_closure_rate(input_upper_air_norm, output_upper_air, input_surface_norm, output_surface, upper_mean, upper_std, surface_mean, surface_std)
                valid_metrics['closure_momentum'] += calculate_momentum_closure_rate(input_upper_air_norm, output_upper_air, input_surface_norm, output_surface, upper_mean, upper_std, surface_mean, surface_std)

                valid_pbar.set_postfix({"mse": f"{mse_total.item():.4f}"})

        n_valid = len(valid_loader)
        for k in valid_metrics:
            valid_metrics[k] /= n_valid

        # 记录结果
        result = {'epoch': epoch + 1}
        for k, v in train_metrics.items():
            result[f'train_{k}'] = v
        for k, v in valid_metrics.items():
            result[f'valid_{k}'] = v
        results.append(result)

        print(f"Epoch {epoch+1}:")
        print(f"  Train - MSE: {train_metrics['mse_total']:.6f}, ACC_sfc: {train_metrics['acc_surface']:.4f}, SSIM_sfc: {train_metrics['ssim_surface']:.4f}")
        print(f"  Valid - MSE: {valid_metrics['mse_total']:.6f}, ACC_sfc: {valid_metrics['acc_surface']:.4f}, SSIM_sfc: {valid_metrics['ssim_surface']:.4f}")
        print(f"  Closure - Water: {valid_metrics['closure_water']:.4f}, Energy: {valid_metrics['closure_energy']:.4f}, Hydro: {valid_metrics['closure_hydrostatic']:.4f}")

    # 保存CSV
    df = pd.DataFrame(results)
    if save_dir:
        csv_path = os.path.join(save_dir, f'ablation_exp{exp_id}_{exp_name}_v2.csv')
        df.to_csv(csv_path, index=False)
        print(f"Results saved to {csv_path}")

    return df


# ============ 主函数 ============

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data...")
    h5_file = h5.File('/gz-data/ERA5_2023_weekly_new.h5', 'r')
    input_surface = h5_file['surface']
    input_upper_air = h5_file['upper_air']
    print(f"Surface data shape: {input_surface.shape}")
    print(f"Upper air data shape: {input_upper_air.shape}")

    print("Loading normalization parameters...")
    sys.path.append(str(PROJECT_ROOT / 'code_v2'))
    from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays
    json_path = str(PROJECT_ROOT / 'code_v2/ERA5_1940_2023_mean_std_v2.json')
    surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(json_path)

    surface_mean = torch.from_numpy(surface_mean_np).to(device=device, dtype=torch.float32)
    surface_std = torch.from_numpy(surface_std_np).to(device=device, dtype=torch.float32)
    upper_mean = torch.from_numpy(upper_mean_np).to(device=device, dtype=torch.float32)
    upper_std = torch.from_numpy(upper_std_np).to(device=device, dtype=torch.float32)

    total_samples = 28
    train_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=0, end_idx=total_samples)
    valid_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=total_samples, end_idx=total_samples+12)

    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=12)

    save_dir = str(PROJECT_ROOT / 'analysis/physical/ablation_results')
    os.makedirs(save_dir, exist_ok=True)

    # 实验设计说明：
    # - base0: CanglongV1，无风向约束，无物理约束（纯MSE基线）
    # - baseline: CanglongV3，有风向约束，无物理约束
    # - exp1-5: CanglongV3，有风向约束，单一物理约束（lambda×5以保持与full相同的MSE:物理比例）
    # - full: CanglongV3，有风向约束，全部5种物理约束
    #
    # 单一约束实验(exp1-5)的lambda乘以5，保持与全耦合实验(exp6)的MSE:物理约束比例一致
    # 全耦合模型中各项比例约为 surface:upper_air:physics = 1.5:1:2.5
    # 单一约束若不乘5，则比例为 1.5:1:0.5，物理约束占比过小
    #
    # model_type: 'v1' = Canglong (无风向约束), 'v3' = CanglongV3 (有风向约束)
    # 完整实验列表（已注释掉已完成的实验）
    # experiments = [
    #     (-1, 'base0', 'baseline', {}, 'v1'),              # V1: 无风向约束，无物理约束 [已完成]
    #     (0, 'baseline', 'baseline', {}, 'v3'),            # V3: 有风向约束，无物理约束 [已完成]
    #     (1, 'water', 'single', {'water': 40}, 'v3'),           # 8 × 5 = 40 [已完成]
    #     (2, 'energy', 'single', {'energy': 1e-4}, 'v3'),       # 2e-5 × 5 = 1e-4 [已完成]
    #     (3, 'hydrostatic', 'single', {'hydrostatic': 2.5e-6}, 'v3'),  # 5e-7 × 5 = 2.5e-6
    #     (4, 'temperature', 'single', {'temperature': 1.5e-1}, 'v3'),  # 3e-2 × 5 = 0.15
    #     (5, 'momentum', 'single', {'momentum': 5e1}, 'v3'),    # 1e1 × 5 = 50
    #     (6, 'full', 'full', {
    #         'water': 8,
    #         'energy': 2e-5,
    #         'hydrostatic': 5e-7,
    #         'temperature': 3e-2,
    #         'momentum': 1e1
    #     }, 'v3'),
    # ]

    # 仅重新运行baseline（MSE权重除以3）
    # base0已完成，其他实验(exp1-6)已完成，无需重跑
    experiments = [
        (0, 'baseline', 'baseline', {}, 'v3'), # V3: 有风向约束，无物理约束，MSE/3
    ]

    all_results = {}

    for exp_id, exp_name, exp_type, lambda_dict, model_type in experiments:
        # 根据model_type选择模型
        if model_type == 'v1':
            model = Canglong()
            print(f"Using Canglong (V1) - No wind direction constraint")
        else:
            model = CanglongV3()
            print(f"Using CanglongV3 - With wind direction constraint")

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
        model.to(device)

        df = run_experiment(
            exp_id=exp_id,
            exp_name=exp_name,
            exp_type=exp_type,
            lambda_dict=lambda_dict,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            surface_mean=surface_mean,
            surface_std=surface_std,
            upper_mean=upper_mean,
            upper_std=upper_std,
            device=device,
            num_epochs=20,
            lr=0.0005*0.2,
            save_dir=save_dir
        )
        all_results[exp_name] = df

        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print("All 7 experiments completed!")
    print(f"Results saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
