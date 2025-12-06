# train_v3_test.py
# 基于梯度平衡的物理约束训练脚本
# 参考: https://www.zhihu.com/question/375794498/answer/3562750282
# 核心思想: 让每个loss对共享层贡献的梯度大小接近，通过动态调整权重实现

# Block 1
# Init CanglongV3 model
import torch
from canglong import CanglongV3

# Block 2
# Physical constraint
"""
基于梯度平衡的物理约束损失函数
通过监控每个loss对共享层输出的梯度贡献，动态调整权重
"""

import os
import torch
import numpy as np

# ============ 全局缓存的常量掩码 ============
MASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'constant_masks')
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


# ============ 物理约束损失函数 ============
# （从train_v3.py复制，保持一致）

def calculate_land_water_balance_loss(input_physical, output_physical):
    """陆地水量平衡损失"""
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


def calculate_water_balance_loss(input_surface_normalized, output_surface_normalized,
                                surface_mean, surface_std):
    """综合水量平衡损失"""
    input_physical = denormalize_surface(input_surface_normalized, surface_mean, surface_std)
    output_physical = denormalize_surface(output_surface_normalized, surface_mean, surface_std)
    return calculate_land_water_balance_loss(input_physical, output_physical)


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
    device = output_upper_physical.device

    R_d = 287
    pressure_levels = torch.tensor([200.0, 300.0, 500.0, 700.0, 850.0], device=device)

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

    return total_loss / max(loss_count, 1)


# 温度方程权重
ALPHA = 0.01
BETA = 0.01
GAMMA = 0.1

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
    idx_sshf = 14
    idx_snswrf, idx_snlwrf = 15, 16
    idx_blh = 6

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

    # 简化非绝热加热
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


PGF_COEFFICIENTS = [0.99, 0.99, 0.93, 0.81, 0.39]
NS_LAYER_WEIGHTS = [1.0, 1.0, 1.2, 0.8, 0.3]

def calculate_navier_stokes_loss(input_upper_normalized, output_upper_normalized,
                                 input_surface_normalized, output_surface_normalized,
                                 upper_mean, upper_std, surface_mean, surface_std,
                                 use_geostrophic_only=False):
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

    if use_geostrophic_only:
        residual_u = coriolis_u - pgf_u
        residual_v = coriolis_v - pgf_v
        lat_mask = (lat_values.abs() > 15 * np.pi / 180).view(1, 1, -1, 1)
        residual_u = residual_u * lat_mask
        residual_v = residual_v * lat_mask
    else:
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


# ============ Tweedie损失函数 (Hunt 2025) ============
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


# Block 3
# Prepare Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import h5py as h5
import sys

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


# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
print("Loading data...")
input_surface, input_upper_air = h5.File('/gz-data/ERA5_2023_weekly_new.h5')['surface'], h5.File('/gz-data/ERA5_2023_weekly_new.h5')['upper_air']
print(f"Surface data shape: {input_surface.shape}")
print(f"Upper air data shape: {input_upper_air.shape}")

# 加载标准化参数
print("Loading normalization parameters...")
from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays
json = '/home/CanglongPhysics/code_v2/ERA5_1940_2023_mean_std_v2.json'
surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(json)

surface_mean = torch.from_numpy(surface_mean_np).to(device=device, dtype=torch.float32)
surface_std = torch.from_numpy(surface_std_np).to(device=device, dtype=torch.float32)
upper_mean = torch.from_numpy(upper_mean_np).to(device=device, dtype=torch.float32)
upper_std = torch.from_numpy(upper_std_np).to(device=device, dtype=torch.float32)

print(f"Surface mean shape: {surface_mean.shape}")
print(f"Upper mean shape: {upper_mean.shape}")

# 创建数据集
total_samples = 28
train_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=0, end_idx=total_samples)
valid_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=total_samples, end_idx=total_samples+12)

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
print(f"Total training samples: {len(train_dataset)}")


# Block 4
# Training with Gradient Balancer
model = CanglongV3()

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.MSELoss()

save_dir = 'checkpoints_v3_test'
os.makedirs(save_dir, exist_ok=True)

num_epochs = 50
best_valid_loss = float('inf')

# ============ 固定权重（与train_v3-origin相同） ============
lambda_water = 8
lambda_energy = 2e-5
lambda_pressure = 5e-7
lambda_temperature = 3e-2
lambda_momentum = 1e1
lambda_focus = 0.2
lambda_tweedie = 8.0

print("=" * 60)
print("Training with FIXED Weights (Same as train_v3-origin)")
print("=" * 60)
print(f"lambda_water={lambda_water}, lambda_energy={lambda_energy}, lambda_pressure={lambda_pressure}")
print(f"lambda_temperature={lambda_temperature}, lambda_momentum={lambda_momentum}")
print(f"lambda_focus={lambda_focus}, lambda_tweedie={lambda_tweedie}")
print("=" * 60)

# 训练循环
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    surface_loss_total = 0.0
    upper_air_loss_total = 0.0
    water_loss_total = 0.0
    energy_loss_total = 0.0
    pressure_loss_total = 0.0
    temperature_loss_total = 0.0
    momentum_loss_total = 0.0
    focus_loss_total = 0.0
    tweedie_loss_total = 0.0

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for input_surface_batch, input_upper_air_batch, target_surface_batch, target_upper_air_batch in train_pbar:
        # 移动到设备
        input_surface_batch = input_surface_batch.float().to(device)
        input_upper_air_batch = input_upper_air_batch.float().to(device)
        target_surface_batch = target_surface_batch.float().to(device)
        target_upper_air_batch = target_upper_air_batch.float().to(device)

        # 标准化
        input_surface_norm = (input_surface_batch - surface_mean) / surface_std
        input_upper_air_norm = (input_upper_air_batch - upper_mean) / upper_std
        target_surface_norm = (target_surface_batch - surface_mean) / surface_std
        target_upper_air_norm = (target_upper_air_batch - upper_mean) / upper_std

        optimizer.zero_grad()

        # 前向传播
        output_surface, output_upper_air = model(input_surface_norm, input_upper_air_norm)

        # 计算MSE损失
        loss_surface = criterion(output_surface, target_surface_norm)
        loss_upper_air = criterion(output_upper_air, target_upper_air_norm)

        # Focus损失
        loss_focus = calculate_focus_variable_loss(
            output_surface, target_surface_norm,
            output_upper_air, target_upper_air_norm
        )

        # Tweedie损失
        loss_tweedie = calculate_tweedie_loss(
            output_surface, target_surface_norm,
            surface_mean, surface_std
        )

        # 物理约束损失
        loss_water = calculate_water_balance_loss(
            input_surface_norm, output_surface,
            surface_mean, surface_std
        )
        loss_energy = calculate_energy_balance_loss(
            input_surface_norm, output_surface,
            surface_mean, surface_std
        )
        loss_pressure = calculate_hydrostatic_balance_loss(
            output_upper_air, output_surface,
            upper_mean, upper_std, surface_mean, surface_std
        )
        loss_temperature = calculate_temperature_tendency_loss(
            input_upper_air_norm, output_upper_air,
            input_surface_norm, output_surface,
            upper_mean, upper_std, surface_mean, surface_std
        )
        loss_momentum = calculate_navier_stokes_loss(
            input_upper_air_norm, output_upper_air,
            input_surface_norm, output_surface,
            upper_mean, upper_std, surface_mean, surface_std
        )

        # 总损失（与train_v3-origin相同）
        loss = loss_surface + loss_upper_air + \
               lambda_water * loss_water + \
               lambda_energy * loss_energy + \
               lambda_pressure * loss_pressure + \
               lambda_temperature * loss_temperature + \
               lambda_momentum * loss_momentum + \
               lambda_focus * loss_focus + \
               lambda_tweedie * loss_tweedie

        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 累加损失
        batch_loss = loss.item()
        train_loss += batch_loss
        surface_loss_total += loss_surface.item()
        upper_air_loss_total += loss_upper_air.item()
        water_loss_total += loss_water.item()
        energy_loss_total += loss_energy.item()
        pressure_loss_total += loss_pressure.item()
        temperature_loss_total += loss_temperature.item()
        momentum_loss_total += loss_momentum.item()
        focus_loss_total += loss_focus.item()
        tweedie_loss_total += loss_tweedie.item()

        # 更新进度条
        train_pbar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "surf": f"{loss_surface.item():.4f}",
            "upper": f"{loss_upper_air.item():.4f}",
            "focus": f"{loss_focus.item():.4f}"
        })

    # 计算平均损失
    n_batches = len(train_loader)
    train_loss /= n_batches
    surface_loss_total /= n_batches
    upper_air_loss_total /= n_batches
    water_loss_total /= n_batches
    energy_loss_total /= n_batches
    pressure_loss_total /= n_batches
    temperature_loss_total /= n_batches
    momentum_loss_total /= n_batches
    focus_loss_total /= n_batches
    tweedie_loss_total /= n_batches

    # 验证阶段
    model.eval()
    valid_loss = 0.0
    valid_surface_loss = 0.0
    valid_upper_air_loss = 0.0
    valid_water_loss = 0.0
    valid_energy_loss = 0.0
    valid_pressure_loss = 0.0
    valid_temperature_loss = 0.0
    valid_momentum_loss = 0.0
    valid_focus_loss = 0.0
    valid_tweedie_loss = 0.0

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

            # 计算损失
            loss_surface = criterion(output_surface, target_surface_norm)
            loss_upper_air = criterion(output_upper_air, target_upper_air_norm)
            loss_focus = calculate_focus_variable_loss(
                output_surface, target_surface_norm,
                output_upper_air, target_upper_air_norm
            )
            loss_tweedie = calculate_tweedie_loss(
                output_surface, target_surface_norm,
                surface_mean, surface_std
            )
            loss_water = calculate_water_balance_loss(
                input_surface_norm, output_surface,
                surface_mean, surface_std
            )
            loss_energy = calculate_energy_balance_loss(
                input_surface_norm, output_surface,
                surface_mean, surface_std
            )
            loss_pressure = calculate_hydrostatic_balance_loss(
                output_upper_air, output_surface,
                upper_mean, upper_std, surface_mean, surface_std
            )
            loss_temperature = calculate_temperature_tendency_loss(
                input_upper_air_norm, output_upper_air,
                input_surface_norm, output_surface,
                upper_mean, upper_std, surface_mean, surface_std
            )
            loss_momentum = calculate_navier_stokes_loss(
                input_upper_air_norm, output_upper_air,
                input_surface_norm, output_surface,
                upper_mean, upper_std, surface_mean, surface_std
            )

            total_loss = loss_surface + loss_upper_air + \
                         lambda_water * loss_water + \
                         lambda_energy * loss_energy + \
                         lambda_pressure * loss_pressure + \
                         lambda_temperature * loss_temperature + \
                         lambda_momentum * loss_momentum + \
                         lambda_focus * loss_focus + \
                         lambda_tweedie * loss_tweedie

            valid_loss += total_loss.item()
            valid_surface_loss += loss_surface.item()
            valid_upper_air_loss += loss_upper_air.item()
            valid_water_loss += loss_water.item()
            valid_energy_loss += loss_energy.item()
            valid_pressure_loss += loss_pressure.item()
            valid_temperature_loss += loss_temperature.item()
            valid_momentum_loss += loss_momentum.item()
            valid_focus_loss += loss_focus.item()
            valid_tweedie_loss += loss_tweedie.item()

            valid_pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "surf": f"{loss_surface.item():.4f}",
                "upper": f"{loss_upper_air.item():.4f}"
            })

    # 计算平均验证损失
    valid_loss /= len(valid_loader)
    valid_surface_loss /= len(valid_loader)
    valid_upper_air_loss /= len(valid_loader)
    valid_water_loss /= len(valid_loader)
    valid_energy_loss /= len(valid_loader)
    valid_pressure_loss /= len(valid_loader)
    valid_temperature_loss /= len(valid_loader)
    valid_momentum_loss /= len(valid_loader)
    valid_focus_loss /= len(valid_loader)
    valid_tweedie_loss /= len(valid_loader)

    # 打印损失
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train - Total: {train_loss:.6f}")
    print(f"         MSE - Surface: {surface_loss_total:.6f}, Upper Air: {upper_air_loss_total:.6f}")
    print(f"         Focus - Raw: {focus_loss_total:.6f}, Weighted: {lambda_focus*focus_loss_total:.6f}")
    print(f"         Tweedie - Raw: {tweedie_loss_total:.2e}, Weighted: {lambda_tweedie*tweedie_loss_total:.6f}")
    print(f"         Physical Raw - Water: {water_loss_total:.2e}, Energy: {energy_loss_total:.2e}, Pressure: {pressure_loss_total:.2e}, Temp: {temperature_loss_total:.2e}, Mom: {momentum_loss_total:.2e}")
    print(f"         Physical Weighted - Water: {lambda_water*water_loss_total:.6f}, Energy: {lambda_energy*energy_loss_total:.6f}, Pressure: {lambda_pressure*pressure_loss_total:.6f}, Temp: {lambda_temperature*temperature_loss_total:.6f}, Mom: {lambda_momentum*momentum_loss_total:.6f}")
    print(f"  Valid - Total: {valid_loss:.6f}")
    print(f"         MSE - Surface: {valid_surface_loss:.6f}, Upper Air: {valid_upper_air_loss:.6f}")
    print(f"         Focus - Raw: {valid_focus_loss:.6f}, Weighted: {lambda_focus*valid_focus_loss:.6f}")
    print(f"         Tweedie - Raw: {valid_tweedie_loss:.2e}, Weighted: {lambda_tweedie*valid_tweedie_loss:.6f}")
    print(f"         Physical Raw - Water: {valid_water_loss:.2e}, Energy: {valid_energy_loss:.2e}, Pressure: {valid_pressure_loss:.2e}, Temp: {valid_temperature_loss:.2e}, Mom: {valid_momentum_loss:.2e}")
    print(f"         Physical Weighted - Water: {lambda_water*valid_water_loss:.6f}, Energy: {lambda_energy*valid_energy_loss:.6f}, Pressure: {lambda_pressure*valid_pressure_loss:.6f}, Temp: {lambda_temperature*valid_temperature_loss:.6f}, Mom: {lambda_momentum*valid_momentum_loss:.6f}")

    # 每10个epoch保存一次checkpoint
    if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pt')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"  => Saved checkpoint: {checkpoint_path}")

# 保存最终模型
torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pt'))
print(f"\nTraining completed! Model saved to {os.path.join(save_dir, 'final_model.pt')}")
