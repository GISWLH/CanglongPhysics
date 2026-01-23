"""
CAS-Canglong V3.5 Training Script
V3.5: V2.5 backbone + physical constraints from train_v3
"""

import argparse
import json
import math
import os
import sys
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
import numcodecs
import torch
import torch.distributed as dist
from torch import nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from canglong import CanglongV2_5
from canglong.wind_aware_shift import WIND_DIR_NAMES, get_dominant_direction


# ============ Debug switches ============
DEBUG_NAN = False
STOP_ON_NAN = False
MAX_NAN_REPORTS = 5
IS_MAIN = True

# Surface / upper variable names (for debug print only)
SURF_VARS = [
    'avg_tnswrf', 'avg_tnlwrf', 'tciw', 'tcc', 'lsrr', 'crr', 'blh',
    'u10', 'v10', 'd2m', 't2m', 'avg_iews', 'avg_inss', 'slhf', 'sshf',
    'avg_snswrf', 'avg_snlwrf', 'ssr', 'str', 'sp', 'msl', 'siconc',
    'sst', 'ro', 'stl', 'swvl'
]
UPPER_VARS = ['o3', 'z', 't', 'u', 'v', 'w', 'q', 'cc', 'ciwc', 'clwc']


def _nonfinite_channels(tensor):
    if tensor.ndim < 2:
        return []
    bad = ~torch.isfinite(tensor)
    bad = bad.reshape(bad.shape[0], bad.shape[1], -1)
    bad_any = bad.any(dim=-1).any(dim=0)
    return torch.nonzero(bad_any, as_tuple=False).flatten().tolist()


def _read_rss_mb():
    rss_kb = 0
    try:
        with open(f"/proc/{os.getpid()}/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_kb = int(parts[1])
                    break
    except FileNotFoundError:
        return 0.0
    return rss_kb / 1024.0


def _fd_count():
    try:
        return len(os.listdir(f"/proc/{os.getpid()}/fd"))
    except Exception:
        return -1


# ============ 常量掩码缓存 ============
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
def calculate_land_water_balance_loss(input_physical, output_physical):
    """陆地水量平衡损失"""
    device = output_physical.device
    masks = get_cached_masks(device)
    land_mask = masks['land_mask']
    basin_mask = masks['basin_mask']

    idx_lsrr, idx_crr, idx_slhf, idx_ro, idx_swvl = 4, 5, 13, 23, 25
    week_seconds = 7 * 24 * 3600
    L_v = 2.5e6
    soil_depth = 2.89

    t0 = input_physical[:, :, -1, :, :]
    t1 = output_physical[:, :, 0, :, :]

    delta_soil_water = (t1[:, idx_swvl] - t0[:, idx_swvl]) * soil_depth
    p_land = (t1[:, idx_lsrr] + t1[:, idx_crr]) * week_seconds / 1000.0 * land_mask
    e_land = torch.abs(t1[:, idx_slhf]) * 7 / L_v / 1000.0 * land_mask
    r = t1[:, idx_ro] * 7 * land_mask

    residual_land = (delta_soil_water - (p_land - e_land - r)) * basin_mask
    residual_land = torch.nan_to_num(residual_land, nan=0.0, posinf=0.0, neginf=0.0)
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
    depth = 2.89
    c_w = 4.184e6

    sw_net = t1[:, idx_sw_net] * land_mask
    lw_net = t1[:, idx_lw_net] * land_mask
    r_n = sw_net + lw_net

    le_raw = t1[:, idx_slhf] * 7 / week_seconds * land_mask
    h_raw = t1[:, idx_sshf] * 7 / week_seconds * land_mask
    le = -le_raw
    h = -h_raw

    delta_t_soil = (t1[:, idx_stl] - t0[:, idx_stl]) * land_mask
    theta = t1[:, idx_swvl] * land_mask
    c_soil = (cs_soil_bulk.unsqueeze(0) + theta * c_w) * land_mask.unsqueeze(0)
    g_flux = c_soil * depth * delta_t_soil / week_seconds

    residual_land = (r_n - le - h - g_flux) * land_mask.unsqueeze(0)
    residual_land = torch.nan_to_num(residual_land, nan=0.0, posinf=0.0, neginf=0.0)

    valid_pixels = land_mask.sum()
    if valid_pixels > 0:
        mse_loss = (residual_land ** 2).sum() / valid_pixels / output_physical.shape[0]
    else:
        mse_loss = torch.tensor(0.0, device=device)

    return mse_loss


def calculate_hydrostatic_balance_loss(output_upper_normalized, output_surface_normalized,
                                       upper_mean, upper_std, surface_mean, surface_std):
    """表面气压静力平衡损失"""
    output_surface_physical = denormalize_surface(output_surface_normalized, surface_mean, surface_std)
    device = output_surface_physical.device
    masks = get_cached_masks(device)
    dem = masks['dem']
    land_mask = masks['land_mask']

    g = 9.80665
    r_d = 287.0

    idx_sp = 19
    idx_msl = 20
    idx_t2m = 10

    sp_model = output_surface_physical[:, idx_sp, 0, :, :]
    msl = output_surface_physical[:, idx_msl, 0, :, :]
    t2m = output_surface_physical[:, idx_t2m, 0, :, :]

    t_v = torch.clamp(t2m, min=200.0, max=330.0)
    sp_physical = msl * torch.exp(-g * dem.unsqueeze(0) / (r_d * t_v))

    residual = (sp_model - sp_physical) * land_mask.unsqueeze(0)
    residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

    valid_pixels = land_mask.sum()
    if valid_pixels > 0:
        mse_loss = (residual ** 2).sum() / valid_pixels / output_surface_physical.shape[0]
    else:
        mse_loss = torch.tensor(0.0, device=device)

    return mse_loss


# 温度方程权重
ALPHA = 0.01
BETA = 0.01
GAMMA = 0.1


def calculate_temperature_tendency_loss(input_upper_normalized, output_upper_normalized,
                                        input_surface_normalized, output_surface_normalized,
                                        upper_mean, upper_std, surface_mean, surface_std):
    """温度局地变化方程约束"""
    device = input_upper_normalized.device

    r_d = 287.0
    c_p = 1004.0
    g = 9.8
    l_v = 2.5e6

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
    r_earth = 6.371e6

    lat_values = torch.linspace(-90, 90, 721, device=device) * np.pi / 180
    cos_lat = torch.cos(lat_values).view(1, 1, -1, 1).clamp(min=0.01)

    t_padded_x = torch.nn.functional.pad(t, (1, 1, 0, 0), mode='circular')
    dT_dx = (t_padded_x[:, :, :, 2:] - t_padded_x[:, :, :, :-2]) / (2 * r_earth * dlon * cos_lat)

    t_padded_y = torch.nn.functional.pad(t, (0, 0, 1, 1), mode='replicate')
    dT_dy = (t_padded_y[:, :, 2:, :] - t_padded_y[:, :, :-2, :]) / (2 * r_earth * dlat)

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

    adiabatic_term = (r_d * t / (c_p * p_3d) - dT_dp) * w
    vertical_motion_term = -adiabatic_term

    tnswrf = output_surface_physical[:, idx_tnswrf, 0, :, :]
    tnlwrf = output_surface_physical[:, idx_tnlwrf, 0, :, :]
    snswrf = output_surface_physical[:, idx_snswrf, 0, :, :]
    snlwrf = output_surface_physical[:, idx_snlwrf, 0, :, :]

    a_sw = tnswrf - snswrf
    a_lw = tnlwrf - snlwrf

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

    q_rad_sw = (g / c_p) * a_sw.unsqueeze(1) * w_sw_norm / (dp / 100)
    q_rad_lw = (g / c_p) * a_lw.unsqueeze(1) * w_lw_norm / (dp / 100)
    q_rad = (q_rad_sw + q_rad_lw) / 1000

    lsrr = output_surface_physical[:, idx_lsrr, 0, :, :]
    crr = output_surface_physical[:, idx_crr, 0, :, :]
    total_precip = lsrr + crr

    latent_profile = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device).view(1, 5, 1, 1)
    q_latent = (l_v / c_p) * total_precip.unsqueeze(1) * latent_profile / 1000

    q_diabatic = q_rad + q_latent

    dT_dt_theoretical = (ALPHA * horizontal_advection +
                         BETA * vertical_motion_term +
                         GAMMA * q_diabatic)

    residual = dT_dt_observed - dT_dt_theoretical
    residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
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

    omega = 7.2921e-5
    r_d = 287.0
    r_earth = 6.371e6

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
    f = 2 * omega * torch.sin(lat_values).view(1, 1, -1, 1)

    u = u_t2
    v = v_t2

    coriolis_u = f * v
    coriolis_v = -f * u

    phi_padded_x = torch.nn.functional.pad(phi, (1, 1, 0, 0), mode='circular')
    dphi_dx = (phi_padded_x[:, :, :, 2:] - phi_padded_x[:, :, :, :-2]) / (2 * r_earth * dlon * cos_lat)

    phi_padded_y = torch.nn.functional.pad(phi, (0, 0, 1, 1), mode='replicate')
    dphi_dy = (phi_padded_y[:, :, 2:, :] - phi_padded_y[:, :, :-2, :]) / (2 * r_earth * dlat)

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
        du_dx = (u_padded_x[:, :, :, 2:] - u_padded_x[:, :, :, :-2]) / (2 * r_earth * dlon * cos_lat)
        u_padded_y = torch.nn.functional.pad(u, (0, 0, 1, 1), mode='replicate')
        du_dy = (u_padded_y[:, :, 2:, :] - u_padded_y[:, :, :-2, :]) / (2 * r_earth * dlat)

        v_padded_x = torch.nn.functional.pad(v, (1, 1, 0, 0), mode='circular')
        dv_dx = (v_padded_x[:, :, :, 2:] - v_padded_x[:, :, :, :-2]) / (2 * r_earth * dlon * cos_lat)
        v_padded_y = torch.nn.functional.pad(v, (0, 0, 1, 1), mode='replicate')
        dv_dy = (v_padded_y[:, :, 2:, :] - v_padded_y[:, :, :-2, :]) / (2 * r_earth * dlat)

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
        rho_850 = p_850 / (r_d * t_850)

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
    output_surface_norm = torch.nan_to_num(output_surface_norm, nan=0.0, posinf=0.0, neginf=0.0)
    output_upper_norm = torch.nan_to_num(output_upper_norm, nan=0.0, posinf=0.0, neginf=0.0)
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
    output_surface_norm = torch.nan_to_num(output_surface_norm, nan=0.0, posinf=0.0, neginf=0.0)
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


def _load_zarr_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _build_blosc(codecs):
    blosc_cfg = None
    for codec in codecs:
        if codec.get("name") == "blosc":
            blosc_cfg = codec.get("configuration", {})
            break
    if not blosc_cfg:
        return None
    shuffle = blosc_cfg.get("shuffle", 1)
    if shuffle == "shuffle":
        shuffle = 1
    elif shuffle == "bitshuffle":
        shuffle = 2
    elif shuffle == "noshuffle":
        shuffle = 0
    return numcodecs.Blosc(
        cname=blosc_cfg.get("cname", "lz4"),
        clevel=blosc_cfg.get("clevel", 5),
        shuffle=shuffle,
        blocksize=blosc_cfg.get("blocksize", 0)
    )


class ZarrArraySpec:
    def __init__(self, store_path, name):
        meta = _load_zarr_json(os.path.join(store_path, name, "zarr.json"))
        self.shape = tuple(meta["shape"])
        self.chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
        if self.chunk_shape[0] != 1:
            raise ValueError(f"{name} time chunk must be 1, got {self.chunk_shape[0]}")
        self.dtype = np.dtype(meta["data_type"])
        endian = None
        for codec in meta.get("codecs", []):
            if codec.get("name") == "bytes":
                endian = codec.get("configuration", {}).get("endian", "little")
                break
        if endian == "little":
            self.dtype = self.dtype.newbyteorder("<")
        elif endian == "big":
            self.dtype = self.dtype.newbyteorder(">")
        self.compressor = _build_blosc(meta.get("codecs", []))
        self.array_dir = os.path.join(store_path, name)
        self.chunk_tail = ["0"] * (len(self.shape) - 1)

    def read_time_chunk(self, t_idx):
        chunk_path = os.path.join(self.array_dir, "c", str(t_idx), *self.chunk_tail)
        with open(chunk_path, "rb") as f:
            raw = f.read()
        if self.compressor:
            raw = self.compressor.decode(raw)
        arr = np.frombuffer(raw, dtype=self.dtype).reshape(self.chunk_shape)
        return arr


def _read_time_array(store_path):
    meta = _load_zarr_json(os.path.join(store_path, "time", "zarr.json"))
    dtype = np.dtype(meta["data_type"])
    endian = None
    for codec in meta.get("codecs", []):
        if codec.get("name") == "bytes":
            endian = codec.get("configuration", {}).get("endian", "little")
            break
    if endian == "little":
        dtype = dtype.newbyteorder("<")
    elif endian == "big":
        dtype = dtype.newbyteorder(">")
    compressor = _build_blosc(meta.get("codecs", []))
    chunk_path = os.path.join(store_path, "time", "c", "0")
    with open(chunk_path, "rb") as f:
        raw = f.read()
    if compressor:
        raw = compressor.decode(raw)
    return np.frombuffer(raw, dtype=dtype)


class ZarrDataset(Dataset):
    def __init__(self, store_specs):
        self._nan_reports = 0
        self.store_specs = []
        counts = []
        for spec in store_specs:
            if spec["count"] <= 0:
                continue
            self.store_specs.append(spec)
            counts.append(spec["count"])
        self._offsets = np.cumsum([0] + counts).tolist()
        self.length = int(self._offsets[-1]) if self._offsets else 0
        if IS_MAIN:
            print(f"Dataset stores: {len(self.store_specs)}, sample count: {self.length}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range for dataset length {self.length}")

        store_idx = bisect_right(self._offsets, idx) - 1
        local_idx = idx - self._offsets[store_idx]
        spec = self.store_specs[store_idx]
        t_idx = spec["start"] + local_idx

        surface_spec = spec["surface"]
        upper_spec = spec["upper"]
        surface_seq = np.empty((3,) + surface_spec.shape[1:], dtype=surface_spec.dtype)
        upper_seq = np.empty((3,) + upper_spec.shape[1:], dtype=upper_spec.dtype)

        for step in range(3):
            surface_seq[step] = surface_spec.read_time_chunk(t_idx + step)[0]
            upper_seq[step] = upper_spec.read_time_chunk(t_idx + step)[0]

        input_surface = surface_seq[:2]
        input_upper_air = upper_seq[:2]
        target_surface = surface_seq[2]
        target_upper_air = upper_seq[2]

        if DEBUG_NAN and self._nan_reports < MAX_NAN_REPORTS:
            if not np.isfinite(input_surface).all():
                bad = ~np.isfinite(input_surface)
                bad_vars = np.where(bad.reshape(bad.shape[0], bad.shape[1], -1).any(-1).any(0))[0].tolist()
                print(f"[NaN] input_surface idx {t_idx}: vars {bad_vars}")
                self._nan_reports += 1
            if not np.isfinite(target_surface).all():
                bad = ~np.isfinite(target_surface)
                bad_vars = np.where(bad.reshape(bad.shape[0], -1).any(-1))[0].tolist()
                print(f"[NaN] target_surface idx {t_idx+2}: vars {bad_vars}")
                self._nan_reports += 1
            if not np.isfinite(input_upper_air).all():
                bad = ~np.isfinite(input_upper_air)
                bad_vars = np.where(bad.reshape(bad.shape[0], bad.shape[1], -1).any(-1).any(0))[0].tolist()
                print(f"[NaN] input_upper_air idx {t_idx}: vars {bad_vars}")
                self._nan_reports += 1
            if not np.isfinite(target_upper_air).all():
                bad = ~np.isfinite(target_upper_air)
                bad_vars = np.where(bad.reshape(bad.shape[0], -1).any(-1))[0].tolist()
                print(f"[NaN] target_upper_air idx {t_idx+2}: vars {bad_vars}")
                self._nan_reports += 1

        if not np.isfinite(input_surface).all():
            input_surface = np.nan_to_num(input_surface, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(input_upper_air).all():
            input_upper_air = np.nan_to_num(input_upper_air, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(target_surface).all():
            target_surface = np.nan_to_num(target_surface, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.isfinite(target_upper_air).all():
            target_upper_air = np.nan_to_num(target_upper_air, nan=0.0, posinf=0.0, neginf=0.0)

        return input_surface, input_upper_air, target_surface, target_upper_air


class ContiguousDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, drop_last=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.drop_last = drop_last
        self.num_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[-padding_size:]
        else:
            indices = indices[:self.total_size]
        start = self.rank * self.num_samples
        end = start + self.num_samples
        return iter(indices[start:end])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--min-lr', type=float, default=1e-5)
    parser.add_argument('--warmup-epochs', type=int, default=2)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--prefetch-factor', type=int, default=2)
    parser.add_argument('--persistent-workers', action='store_true')
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--checkpoint-interval', type=int, default=5)
    parser.add_argument('--log-file', type=str, default='log.txt')
    parser.add_argument('--log-interval', type=int, default=50)
    parser.add_argument('--log-append', action='store_true')
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--tf32', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume-checkpoint', type=str, default='')
    args = parser.parse_args()

    use_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    IS_MAIN = rank == 0

    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if IS_MAIN:
        print(f"Using device: {device}")

    log_file = args.log_file
    if log_file:
        if use_ddp and not IS_MAIN:
            base, ext = os.path.splitext(log_file)
            if not ext:
                ext = ".txt"
            log_file = f"{base}_rank{rank}{ext}"
        log_mode = "a" if args.log_append else "w"
        with open(log_file, log_mode) as f:
            f.write(f"start={time.strftime('%Y-%m-%d %H:%M:%S')} rank={rank} pid={os.getpid()} device={device}\n")

    def _write_log(message):
        if not log_file:
            return
        with open(log_file, "a") as f:
            f.write(message + "\n")

    if IS_MAIN:
        print("Loading data...")
    data_dir = '/gz-data'
    store_paths = [
        os.path.join(data_dir, 'ERA5_1940_1981_weekly.zarr'),
        os.path.join(data_dir, 'ERA5_1982_2023_weekly.zarr')
    ]

    missing = [p for p in store_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing data files: {missing}")

    if IS_MAIN:
        for path in store_paths:
            surface_meta = _load_zarr_json(os.path.join(path, "surface", "zarr.json"))
            upper_meta = _load_zarr_json(os.path.join(path, "upper_air", "zarr.json"))
            print(f"{os.path.basename(path)} surface shape: {surface_meta['shape']}")
            print(f"{os.path.basename(path)} upper air shape: {upper_meta['shape']}")

    if IS_MAIN:
        print("Loading normalization parameters...")
    sys.path.append('code_v2')
    from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays
    json_path = '/home/CanglongPhysics/code_v2/ERA5_1940_2023_mean_std_v2.json'
    surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(json_path)

    surface_mean = torch.from_numpy(surface_mean_np).to(device=device, dtype=torch.float32)
    surface_std = torch.from_numpy(surface_std_np).to(device=device, dtype=torch.float32)
    upper_mean = torch.from_numpy(upper_mean_np).to(device=device, dtype=torch.float32)
    upper_std = torch.from_numpy(upper_std_np).to(device=device, dtype=torch.float32)

    if IS_MAIN:
        print(f"Surface mean shape: {surface_mean.shape}")
        print(f"Upper mean shape: {upper_mean.shape}")

    stores = []
    for path in store_paths:
        surface_spec = ZarrArraySpec(path, "surface")
        upper_spec = ZarrArraySpec(path, "upper_air")
        if surface_spec.shape[0] != upper_spec.shape[0]:
            raise ValueError(f"Time dimension mismatch in {path}")
        stores.append({
            "path": path,
            "surface": surface_spec,
            "upper": upper_spec,
            "time_len": surface_spec.shape[0]
        })

    time_values = _read_time_array(store_paths[1])
    base = np.datetime64('1940-01-01')
    dates = base + time_values.astype('timedelta64[D]')
    years = dates.astype('datetime64[Y]').astype(int) + 1970
    valid_mask = years >= 2020
    if not valid_mask.any():
        raise RuntimeError("Validation split (year >= 2020) not found in ERA5_1982_2023_weekly.zarr")
    valid_start = int(np.argmax(valid_mask))
    store2_len = stores[1]["time_len"]
    valid_end = store2_len - 3
    train2_end = valid_start - 3

    def _make_spec(store, start, end):
        count = max(0, end - start + 1)
        return {
            "path": store["path"],
            "surface": store["surface"],
            "upper": store["upper"],
            "start": start,
            "end": end,
            "count": count
        }

    train_specs = [
        _make_spec(stores[0], 0, stores[0]["time_len"] - 3),
        _make_spec(stores[1], 0, train2_end)
    ]
    valid_specs = [
        _make_spec(stores[1], valid_start, valid_end)
    ]

    if IS_MAIN:
        print(f"Split store2: train idx 0-{train2_end}, valid idx {valid_start}-{valid_end}, "
              f"valid start date {str(dates[valid_start])}")

    train_dataset = ZarrDataset(train_specs)
    valid_dataset = ZarrDataset(valid_specs)

    batch_size = args.batch_size
    if not use_ddp and torch.cuda.device_count() > 1 and batch_size < torch.cuda.device_count() and IS_MAIN:
        print("Warning: batch_size < num_gpus; DataParallel will underutilize GPUs. Consider DDP or increase batch_size.")

    if use_ddp:
        if args.shuffle:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = ContiguousDistributedSampler(train_dataset, drop_last=False)

        valid_sampler = ContiguousDistributedSampler(valid_dataset, drop_last=False)
    else:
        train_sampler = None
        valid_sampler = None

    persistent_workers = args.persistent_workers and args.num_workers > 0
    prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None

    base_train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None and args.shuffle),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )
    base_valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor
    )

    if IS_MAIN:
        print(f"Total training samples: {len(train_dataset)}")
        print(f"Total validation samples: {len(valid_dataset)}")

    train_steps = len(base_train_loader)
    valid_steps = len(base_valid_loader)

    train_loader = base_train_loader
    valid_loader = base_valid_loader

    # Model configuration (same as train_v2_5.py)
    embed_dim = 192
    num_heads = (12, 24, 24, 12)
    depths = (4, 8, 8, 4)
    max_wind_dirs = 2
    max_wind_dirs_by_layer = (2, 1, 1, 1)
    use_wind_aware_shift_by_layer = (True, False, False, False)
    drop_path_max = 0.3

    model = CanglongV2_5(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depths=depths,
        max_wind_dirs=max_wind_dirs,
        max_wind_dirs_by_layer=max_wind_dirs_by_layer,
        use_wind_aware_shift_by_layer=use_wind_aware_shift_by_layer,
        drop_path_max=drop_path_max,
        surface_mean=torch.from_numpy(surface_mean_np),
        surface_std=torch.from_numpy(surface_std_np),
        upper_mean=torch.from_numpy(upper_mean_np),
        upper_std=torch.from_numpy(upper_std_np),
        use_checkpoint=True
    )

    model.to(device)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if IS_MAIN:
            print(f"Using DDP with world size {dist.get_world_size()}.")
    elif torch.cuda.device_count() > 1:
        if IS_MAIN:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    base_lr = args.lr
    min_lr = args.min_lr
    warmup_epochs = max(1, args.warmup_epochs)
    optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    scaler = GradScaler('cuda')
    use_amp = True
    if IS_MAIN:
        print(f"Mixed precision training: {use_amp}")

    save_dir = 'checkpoints_v3_5'
    os.makedirs(save_dir, exist_ok=True)

    resume = args.resume
    start_epoch = 0
    checkpoint_files = []
    if resume:
        if args.resume_checkpoint:
            checkpoint_files = [Path(args.resume_checkpoint)]
        else:
            checkpoint_files = sorted(Path(save_dir).glob('model_v3_5_epoch*.pth'),
                                      key=lambda p: int(p.stem.split('epoch')[1]) if 'epoch' in p.stem else -1)

    if resume and checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        try:
            state_dict = torch.load(latest_checkpoint, map_location=device, weights_only=True)
            if hasattr(model, 'module'):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            try:
                start_epoch = int(latest_checkpoint.stem.split('epoch')[1])
            except (IndexError, ValueError):
                start_epoch = 0
            if IS_MAIN:
                print(f"Resumed model weights from {latest_checkpoint} (epoch {start_epoch}).")
        except (FileNotFoundError, ValueError, KeyError) as err:
            if IS_MAIN:
                print(f"Failed to load checkpoint {latest_checkpoint}: {err}. Starting fresh training.")
            start_epoch = 0
    else:
        if IS_MAIN:
            print("No checkpoint found. Starting fresh training.")

    num_epochs = args.num_epochs
    checkpoint_interval = args.checkpoint_interval

    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        min_lr_ratio = min_lr / base_lr
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    for param_group in optimizer.param_groups:
        param_group.setdefault('initial_lr', base_lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=start_epoch - 1)

    # ============ 固定权重（与train_v3一致） ============
    lambda_water = 8
    lambda_energy = 2e-5
    lambda_pressure = 1e-8
    lambda_temperature = 6e-4
    lambda_momentum = 2e-1
    lambda_focus = 0.1
    lambda_tweedie = 12.0

    if IS_MAIN:
        print("=" * 70)
        print("Training V3.5 (V2.5 + Physical Constraints)")
        print("=" * 70)
        print(f"lambda_water={lambda_water}, lambda_energy={lambda_energy}, lambda_pressure={lambda_pressure}")
        print(f"lambda_temperature={lambda_temperature}, lambda_momentum={lambda_momentum}")
        print(f"lambda_focus={lambda_focus}, lambda_tweedie={lambda_tweedie}")
        print(f"lr={base_lr}, min_lr={min_lr}, warmup_epochs={warmup_epochs}, num_epochs={num_epochs}")
        print("=" * 70)

    wind_direction_counts = {}
    nan_reports = 0
    half_epochs = max(1, num_epochs // 2)

    def ddp_sum(value):
        if not use_ddp:
            return value
        tensor = torch.tensor(value, device=device, dtype=torch.float32)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item()

    for epoch in range(start_epoch, num_epochs):
        if use_ddp and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        phys_scale = max(0.0, 1.0 - (epoch / float(half_epochs)))
        lambda_water_t = lambda_water * phys_scale
        lambda_energy_t = lambda_energy * phys_scale
        lambda_pressure_t = lambda_pressure * phys_scale
        lambda_temperature_t = lambda_temperature * phys_scale
        lambda_momentum_t = lambda_momentum * phys_scale

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
        skipped_batches = 0

        train_pbar = tqdm(train_loader, total=train_steps, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", disable=not IS_MAIN)
        for batch_idx, (input_surface, input_upper_air, target_surface, target_upper_air) in enumerate(train_pbar):
            if args.log_interval > 0 and batch_idx % args.log_interval == 0:
                rss_mb = _read_rss_mb()
                fd_count = _fd_count()
                if torch.cuda.is_available():
                    cuda_alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
                    cuda_reserved = torch.cuda.memory_reserved(device) / (1024 * 1024)
                    cuda_max_alloc = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                    cuda_max_reserved = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
                else:
                    cuda_alloc = 0.0
                    cuda_reserved = 0.0
                    cuda_max_alloc = 0.0
                    cuda_max_reserved = 0.0
                _write_log(
                    f"time={time.strftime('%Y-%m-%d %H:%M:%S')} rank={rank} "
                    f"epoch={epoch+1} step={batch_idx}/{train_steps} "
                    f"rss_mb={rss_mb:.1f} fd={fd_count} "
                    f"cuda_alloc_mb={cuda_alloc:.1f} cuda_reserved_mb={cuda_reserved:.1f} "
                    f"cuda_max_alloc_mb={cuda_max_alloc:.1f} cuda_max_reserved_mb={cuda_max_reserved:.1f}"
                )
            input_surface = input_surface.float().to(device, non_blocking=True)
            input_upper_air = input_upper_air.float().to(device, non_blocking=True)
            target_surface = target_surface.float().to(device, non_blocking=True)
            target_upper_air = target_upper_air.float().to(device, non_blocking=True)

            input_surface_norm = (input_surface.permute(0, 2, 1, 3, 4) - surface_mean) / surface_std
            input_upper_air_norm = (input_upper_air.permute(0, 2, 3, 1, 4, 5) - upper_mean) / upper_std
            target_surface_norm = (target_surface.unsqueeze(2) - surface_mean) / surface_std
            target_upper_air_norm = (target_upper_air.unsqueeze(3) - upper_mean) / upper_std

            if DEBUG_NAN and nan_reports < MAX_NAN_REPORTS:
                if not torch.isfinite(target_surface_norm).all():
                    bad_vars = _nonfinite_channels(target_surface_norm)
                    names = [SURF_VARS[i] for i in bad_vars] if bad_vars else []
                    print(f"[NaN] target_surface_norm epoch {epoch+1} batch {batch_idx}: vars {bad_vars} {names}")
                    nan_reports += 1
                    if STOP_ON_NAN:
                        raise RuntimeError("Non-finite target_surface_norm")
                if not torch.isfinite(target_upper_air_norm).all():
                    bad_vars = _nonfinite_channels(target_upper_air_norm)
                    names = [UPPER_VARS[i] for i in bad_vars] if bad_vars else []
                    print(f"[NaN] target_upper_air_norm epoch {epoch+1} batch {batch_idx}: vars {bad_vars} {names}")
                    nan_reports += 1
                    if STOP_ON_NAN:
                        raise RuntimeError("Non-finite target_upper_air_norm")

            optimizer.zero_grad()

            with autocast('cuda', enabled=use_amp):
                output_surface, output_upper_air, wind_dir_id = model(
                    input_surface_norm, input_upper_air_norm, return_wind_info=True
                )

            if DEBUG_NAN and nan_reports < MAX_NAN_REPORTS:
                if not torch.isfinite(output_surface).all():
                    bad_vars = _nonfinite_channels(output_surface)
                    names = [SURF_VARS[i] for i in bad_vars] if bad_vars else []
                    print(f"[NaN] output_surface epoch {epoch+1} batch {batch_idx}: vars {bad_vars} {names}")
                    nan_reports += 1
                    if STOP_ON_NAN:
                        raise RuntimeError("Non-finite output_surface")
                if not torch.isfinite(output_upper_air).all():
                    bad_vars = _nonfinite_channels(output_upper_air)
                    names = [UPPER_VARS[i] for i in bad_vars] if bad_vars else []
                    print(f"[NaN] output_upper_air epoch {epoch+1} batch {batch_idx}: vars {bad_vars} {names}")
                    nan_reports += 1
                    if STOP_ON_NAN:
                        raise RuntimeError("Non-finite output_upper_air")

            output_surface_f = torch.nan_to_num(output_surface.float(), nan=0.0, posinf=0.0, neginf=0.0)
            output_upper_air_f = torch.nan_to_num(output_upper_air.float(), nan=0.0, posinf=0.0, neginf=0.0)

            loss_surface = criterion(output_surface_f, target_surface_norm)
            loss_upper_air = criterion(output_upper_air_f, target_upper_air_norm)

            # 物理损失与加权损失使用 float32 以减少数值误差
            loss_focus = calculate_focus_variable_loss(
                output_surface_f, target_surface_norm,
                output_upper_air_f, target_upper_air_norm
            )
            loss_tweedie = calculate_tweedie_loss(
                output_surface_f, target_surface_norm,
                surface_mean, surface_std
            )
            if phys_scale > 0.0:
                loss_water = calculate_water_balance_loss(
                    input_surface_norm, output_surface_f,
                    surface_mean, surface_std
                )
                loss_energy = calculate_energy_balance_loss(
                    input_surface_norm, output_surface_f,
                    surface_mean, surface_std
                )
                loss_pressure = calculate_hydrostatic_balance_loss(
                    output_upper_air_f, output_surface_f,
                    upper_mean, upper_std, surface_mean, surface_std
                )
                loss_temperature = calculate_temperature_tendency_loss(
                    input_upper_air_norm, output_upper_air_f,
                    input_surface_norm, output_surface_f,
                    upper_mean, upper_std, surface_mean, surface_std
                )
                loss_momentum = calculate_navier_stokes_loss(
                    input_upper_air_norm, output_upper_air_f,
                    input_surface_norm, output_surface_f,
                    upper_mean, upper_std, surface_mean, surface_std
                )
            else:
                loss_water = loss_surface.new_zeros(())
                loss_energy = loss_surface.new_zeros(())
                loss_pressure = loss_surface.new_zeros(())
                loss_temperature = loss_surface.new_zeros(())
                loss_momentum = loss_surface.new_zeros(())

            if DEBUG_NAN and nan_reports < MAX_NAN_REPORTS:
                losses_to_check = {
                    "loss_surface": loss_surface,
                    "loss_upper_air": loss_upper_air,
                    "loss_focus": loss_focus,
                    "loss_tweedie": loss_tweedie,
                    "loss_water": loss_water,
                    "loss_energy": loss_energy,
                    "loss_pressure": loss_pressure,
                    "loss_temperature": loss_temperature,
                    "loss_momentum": loss_momentum,
                }
                for name, val in losses_to_check.items():
                    if not torch.isfinite(val):
                        print(f"[NaN] {name} epoch {epoch+1} batch {batch_idx}: {val}")
                        nan_reports += 1
                        if STOP_ON_NAN:
                            raise RuntimeError(f"Non-finite {name}")
                        break

            loss = loss_surface + loss_upper_air + \
                   lambda_water_t * loss_water + \
                   lambda_energy_t * loss_energy + \
                   lambda_pressure_t * loss_pressure + \
                   lambda_temperature_t * loss_temperature + \
                   lambda_momentum_t * loss_momentum + \
                   lambda_focus * loss_focus + \
                   lambda_tweedie * loss_tweedie

            if not torch.isfinite(loss):
                print(f"Non-finite loss at epoch {epoch+1}, batch {batch_idx}. Skipping update.")
                train_pbar.set_postfix({"loss": "nan", "surf": "nan", "upper": f"{loss_upper_air.item():.4f}"})
                skipped_batches += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

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

            dominant_id = get_dominant_direction(wind_dir_id)
            dir_name = WIND_DIR_NAMES.get(dominant_id, 'Unknown')
            wind_direction_counts[dir_name] = wind_direction_counts.get(dir_name, 0) + 1

            train_pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "surf": f"{loss_surface.item():.4f}",
                "upper": f"{loss_upper_air.item():.4f}",
                "wind": dir_name
            })

        n_batches = train_steps
        effective_batches = max(1, n_batches - skipped_batches)
        if use_ddp:
            train_loss = ddp_sum(train_loss)
            surface_loss_total = ddp_sum(surface_loss_total)
            upper_air_loss_total = ddp_sum(upper_air_loss_total)
            water_loss_total = ddp_sum(water_loss_total)
            energy_loss_total = ddp_sum(energy_loss_total)
            pressure_loss_total = ddp_sum(pressure_loss_total)
            temperature_loss_total = ddp_sum(temperature_loss_total)
            momentum_loss_total = ddp_sum(momentum_loss_total)
            focus_loss_total = ddp_sum(focus_loss_total)
            tweedie_loss_total = ddp_sum(tweedie_loss_total)
            effective_batches = max(1.0, ddp_sum(effective_batches))
            skipped_batches = int(ddp_sum(skipped_batches))
        train_loss /= effective_batches
        surface_loss_total /= effective_batches
        upper_air_loss_total /= effective_batches
        water_loss_total /= effective_batches
        energy_loss_total /= effective_batches
        pressure_loss_total /= effective_batches
        temperature_loss_total /= effective_batches
        momentum_loss_total /= effective_batches
        focus_loss_total /= effective_batches
        tweedie_loss_total /= effective_batches

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
        valid_skipped = 0

        with torch.no_grad():
            valid_pbar = tqdm(valid_loader, total=valid_steps, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]", disable=not IS_MAIN)
            for input_surface, input_upper_air, target_surface, target_upper_air in valid_pbar:
                input_surface = input_surface.float().to(device, non_blocking=True)
                input_upper_air = input_upper_air.float().to(device, non_blocking=True)
                target_surface = target_surface.float().to(device, non_blocking=True)
                target_upper_air = target_upper_air.float().to(device, non_blocking=True)

                input_surface_norm = (input_surface.permute(0, 2, 1, 3, 4) - surface_mean) / surface_std
                input_upper_air_norm = (input_upper_air.permute(0, 2, 3, 1, 4, 5) - upper_mean) / upper_std
                target_surface_norm = (target_surface.unsqueeze(2) - surface_mean) / surface_std
                target_upper_air_norm = (target_upper_air.unsqueeze(3) - upper_mean) / upper_std

                with autocast('cuda', enabled=use_amp):
                    output_surface, output_upper_air = model(input_surface_norm, input_upper_air_norm)

                output_surface_f = torch.nan_to_num(output_surface.float(), nan=0.0, posinf=0.0, neginf=0.0)
                output_upper_air_f = torch.nan_to_num(output_upper_air.float(), nan=0.0, posinf=0.0, neginf=0.0)

                loss_surface = criterion(output_surface_f, target_surface_norm)
                loss_upper_air = criterion(output_upper_air_f, target_upper_air_norm)

                loss_focus = calculate_focus_variable_loss(
                    output_surface_f, target_surface_norm,
                    output_upper_air_f, target_upper_air_norm
                )
                loss_tweedie = calculate_tweedie_loss(
                    output_surface_f, target_surface_norm,
                    surface_mean, surface_std
                )
                if phys_scale > 0.0:
                    loss_water = calculate_water_balance_loss(
                        input_surface_norm, output_surface_f,
                        surface_mean, surface_std
                    )
                    loss_energy = calculate_energy_balance_loss(
                        input_surface_norm, output_surface_f,
                        surface_mean, surface_std
                    )
                    loss_pressure = calculate_hydrostatic_balance_loss(
                        output_upper_air_f, output_surface_f,
                        upper_mean, upper_std, surface_mean, surface_std
                    )
                    loss_temperature = calculate_temperature_tendency_loss(
                        input_upper_air_norm, output_upper_air_f,
                        input_surface_norm, output_surface_f,
                        upper_mean, upper_std, surface_mean, surface_std
                    )
                    loss_momentum = calculate_navier_stokes_loss(
                        input_upper_air_norm, output_upper_air_f,
                        input_surface_norm, output_surface_f,
                        upper_mean, upper_std, surface_mean, surface_std
                    )
                else:
                    loss_water = loss_surface.new_zeros(())
                    loss_energy = loss_surface.new_zeros(())
                    loss_pressure = loss_surface.new_zeros(())
                    loss_temperature = loss_surface.new_zeros(())
                    loss_momentum = loss_surface.new_zeros(())

                total_loss = loss_surface + loss_upper_air + \
                             lambda_water_t * loss_water + \
                             lambda_energy_t * loss_energy + \
                             lambda_pressure_t * loss_pressure + \
                             lambda_temperature_t * loss_temperature + \
                             lambda_momentum_t * loss_momentum + \
                             lambda_focus * loss_focus + \
                             lambda_tweedie * loss_tweedie

                if not torch.isfinite(total_loss):
                    valid_skipped += 1
                    valid_pbar.set_postfix({"loss": "nan", "surf": "nan", "upper": f"{loss_upper_air.item():.4f}"})
                    continue

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

        valid_batches = valid_steps
        valid_effective = max(1, valid_batches - valid_skipped)
        if use_ddp:
            valid_loss = ddp_sum(valid_loss)
            valid_surface_loss = ddp_sum(valid_surface_loss)
            valid_upper_air_loss = ddp_sum(valid_upper_air_loss)
            valid_water_loss = ddp_sum(valid_water_loss)
            valid_energy_loss = ddp_sum(valid_energy_loss)
            valid_pressure_loss = ddp_sum(valid_pressure_loss)
            valid_temperature_loss = ddp_sum(valid_temperature_loss)
            valid_momentum_loss = ddp_sum(valid_momentum_loss)
            valid_focus_loss = ddp_sum(valid_focus_loss)
            valid_tweedie_loss = ddp_sum(valid_tweedie_loss)
            valid_effective = max(1.0, ddp_sum(valid_effective))
            valid_skipped = int(ddp_sum(valid_skipped))
        valid_loss /= valid_effective
        valid_surface_loss /= valid_effective
        valid_upper_air_loss /= valid_effective
        valid_water_loss /= valid_effective
        valid_energy_loss /= valid_effective
        valid_pressure_loss /= valid_effective
        valid_temperature_loss /= valid_effective
        valid_momentum_loss /= valid_effective
        valid_focus_loss /= valid_effective
        valid_tweedie_loss /= valid_effective

        if IS_MAIN:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch+1}/{num_epochs} (lr={current_lr:.6e}, phys_scale={phys_scale:.3f})")
            print(f"  Train - Total: {train_loss:.6f}")
            print(f"         MSE - Surface: {surface_loss_total:.6f}, Upper Air: {upper_air_loss_total:.6f}")
            print(f"         Focus - Raw: {focus_loss_total:.6f}, Weighted: {lambda_focus*focus_loss_total:.6f}")
            print(f"         Tweedie - Raw: {tweedie_loss_total:.2e}, Weighted: {lambda_tweedie*tweedie_loss_total:.6f}")
            print(f"         Physical Raw - Water: {water_loss_total:.2e}, Energy: {energy_loss_total:.2e}, Pressure: {pressure_loss_total:.2e}, Temp: {temperature_loss_total:.2e}, Mom: {momentum_loss_total:.2e}")
            print(f"         Physical Weighted - Water: {lambda_water_t*water_loss_total:.6f}, Energy: {lambda_energy_t*energy_loss_total:.6f}, Pressure: {lambda_pressure_t*pressure_loss_total:.6f}, Temp: {lambda_temperature_t*temperature_loss_total:.6f}, Mom: {lambda_momentum_t*momentum_loss_total:.6f}")
            if skipped_batches > 0:
                print(f"         Skipped batches: {skipped_batches}")
            print(f"  Valid - Total: {valid_loss:.6f}")
            print(f"         MSE - Surface: {valid_surface_loss:.6f}, Upper Air: {valid_upper_air_loss:.6f}")
            print(f"         Focus - Raw: {valid_focus_loss:.6f}, Weighted: {lambda_focus*valid_focus_loss:.6f}")
            print(f"         Tweedie - Raw: {valid_tweedie_loss:.2e}, Weighted: {lambda_tweedie*valid_tweedie_loss:.6f}")
            print(f"         Physical Raw - Water: {valid_water_loss:.2e}, Energy: {valid_energy_loss:.2e}, Pressure: {valid_pressure_loss:.2e}, Temp: {valid_temperature_loss:.2e}, Mom: {valid_momentum_loss:.2e}")
            print(f"         Physical Weighted - Water: {lambda_water_t*valid_water_loss:.6f}, Energy: {lambda_energy_t*valid_energy_loss:.6f}, Pressure: {lambda_pressure_t*valid_pressure_loss:.6f}, Temp: {lambda_temperature_t*valid_temperature_loss:.6f}, Mom: {lambda_momentum_t*valid_momentum_loss:.6f}")
            if valid_skipped > 0:
                print(f"         Valid skipped batches: {valid_skipped}")

        if IS_MAIN and (epoch + 1) % checkpoint_interval == 0:
            save_path = os.path.join(save_dir, f"model_v3_5_epoch{epoch+1}.pth")
            if hasattr(model, 'module'):
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"  Saved checkpoint: {save_path}")

        scheduler.step()

    if IS_MAIN:
        print("Training completed!")
        final_path = os.path.join(save_dir, "model_v3_5_final.pth")
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), final_path)
        else:
            torch.save(model.state_dict(), final_path)
        print(f"Saved final checkpoint: {final_path}")
    if use_ddp:
        dist.destroy_process_group()
