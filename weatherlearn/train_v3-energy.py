# Block 1
# Init CanglongV3 model
import torch
from canglong import CanglongV3

# Block 2
# Physical constraint
"""
水量平衡物理约束
考虑ERA5周数据中累积量被平均的问题
"""

import os
import torch

# 延迟加载掩码
MASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'constant_masks')
LAND_MASK = None
BASIN_MASK = None

def load_masks(device):
    """加载陆地和流域掩码到指定设备"""
    global LAND_MASK, BASIN_MASK
    if LAND_MASK is None:
        land_path = os.path.join(MASK_DIR, 'is_land.pt')
        basin_path = os.path.join(MASK_DIR, 'hydrobasin_exorheic_mask.pt')
        LAND_MASK = torch.load(land_path, map_location=device)
        BASIN_MASK = torch.load(basin_path, map_location=device)
        print(f"✅ Loaded masks: land {LAND_MASK.shape}, basin {BASIN_MASK.shape}")
    return LAND_MASK, BASIN_MASK

def denormalize_surface(tensor, surface_mean, surface_std):
    """Convert normalized surface data back to physical units."""
    return tensor * surface_std + surface_mean

def denormalize_upper(tensor, upper_mean, upper_std):
    """Convert normalized upper-air data back to physical units."""
    return tensor * upper_std + upper_mean

def calculate_land_water_balance_loss(input_physical, output_physical):
    """
    计算陆地水量平衡损失（仅在外流区计算）
    陆地水量平衡: ΔS_soilwater = P_land - E_land - R

    修正单位处理：
    - lsrr, crr: 率变量 (kg m^-2 s^-1) -> 乘以week_seconds再除以1000转为m
    - slhf: 日累积被平均 (J m^-2) -> 乘以7恢复周累积
    - ro: 日累积被平均 (m) -> 乘以7恢复周累积
    - swvl: 瞬时值 (m³/m³) -> 乘以土壤深度2.89m

    Args:
        input_physical: 反标准化的输入表面变量 (B, 26, time, lat, lon)
        output_physical: 反标准化的输出表面变量 (B, 26, time, lat, lon)
    """
    device = output_physical.device
    land_mask, basin_mask = load_masks(device)

    # 变量索引（基于CLAUDE.md）
    idx_lsrr = 4   # 大尺度降雨率 (kg m^-2 s^-1)
    idx_crr = 5    # 对流降雨率 (kg m^-2 s^-1)
    idx_slhf = 13  # 潜热通量 (J m^-2, 日累积被平均)
    idx_ro = 23    # 径流 (m, 日累积被平均)
    idx_swvl = 25  # 土壤水 (m³/m³, 瞬时值)

    # 物理常数
    week_seconds = 7 * 24 * 3600  # 一周的秒数
    L_v = 2.5e6  # 汽化潜热 (J/kg)
    SOIL_DEPTH = 2.89  # 总土壤深度 (m)

    # 获取时刻
    t0 = input_physical[:, :, -1, :, :]  # 初始时刻
    t1 = output_physical[:, :, 0, :, :]  # 预测时刻

    # 1. 土壤水变化量 (m³/m³ -> m)
    delta_soil_water = (t1[:, idx_swvl] - t0[:, idx_swvl]) * SOIL_DEPTH

    # 2. 降水量（仅陆地）: kg m^-2 s^-1 -> m
    # kg/m²/s * s = kg/m² = mm, 再除以1000转为m
    P_land = (t1[:, idx_lsrr] + t1[:, idx_crr]) * week_seconds / 1000.0 * land_mask

    # 3. 蒸散发（仅陆地）: J m^-2 (日累积被平均) -> m
    # J/m² * 7 / (J/kg) / 1000 = m
    E_land = torch.abs(t1[:, idx_slhf]) * 7 / L_v / 1000.0 * land_mask

    # 4. 径流（仅陆地）: m (日累积被平均) -> m
    R = t1[:, idx_ro] * 7 * land_mask

    # 陆地水量平衡残差（仅在外流区计算）
    residual_land = (delta_soil_water - (P_land - E_land - R)) * basin_mask

    return torch.nn.functional.mse_loss(residual_land, torch.zeros_like(residual_land))

def calculate_ocean_water_balance_loss(input_physical, output_physical):
    """
    计算海洋水量平衡损失
    海洋水量平衡: d(S_ocean)/dt = P_ocean - E_ocean + R

    注：由于没有海洋储水量变量，简化为：P_ocean - E_ocean + R ≈ 0

    Args:
        input_physical: 反标准化的输入表面变量 (B, 26, time, lat, lon)
        output_physical: 反标准化的输出表面变量 (B, 26, time, lat, lon)
    """
    device = output_physical.device
    land_mask, _ = load_masks(device)
    ocean_mask = 1.0 - land_mask

    # 变量索引
    idx_lsrr = 4
    idx_crr = 5
    idx_slhf = 13
    idx_ro = 23

    # 物理常数
    week_seconds = 7 * 24 * 3600
    L_v = 2.5e6

    # 获取预测时刻
    t1 = output_physical[:, :, 0, :, :]

    # 1. 降水量（仅海洋）
    P_ocean = (t1[:, idx_lsrr] + t1[:, idx_crr]) * week_seconds / 1000.0 * ocean_mask

    # 2. 蒸发（仅海洋）
    E_ocean = torch.abs(t1[:, idx_slhf]) * 7 / L_v / 1000.0 * ocean_mask

    # 3. 从陆地流入的径流（简化处理）
    R_land = t1[:, idx_ro] * 7 * land_mask
    total_runoff = torch.sum(R_land)
    ocean_area = torch.sum(ocean_mask)
    runoff_to_ocean = total_runoff / (ocean_area + 1e-8)

    # 海洋水量平衡残差
    residual_ocean = (P_ocean - E_ocean + runoff_to_ocean) * ocean_mask

    return torch.nn.functional.mse_loss(residual_ocean, torch.zeros_like(residual_ocean))

def calculate_atmosphere_water_balance_loss(input_physical, output_physical):
    """
    计算大气水量平衡损失
    大气水量平衡: d(S_atmos)/dt = ET_total - P_total

    注：忽略大气储水量变化，简化为：ET_total - P_total ≈ 0

    Args:
        input_physical: 反标准化的输入表面变量 (B, 26, time, lat, lon)
        output_physical: 反标准化的输出表面变量 (B, 26, time, lat, lon)
    """
    device = output_physical.device

    # 变量索引
    idx_lsrr = 4
    idx_crr = 5
    idx_slhf = 13

    # 物理常数
    week_seconds = 7 * 24 * 3600
    L_v = 2.5e6

    # 获取预测时刻
    t1 = output_physical[:, :, 0, :, :]

    # 1. 总降水
    P_total = (t1[:, idx_lsrr] + t1[:, idx_crr]) * week_seconds / 1000.0

    # 2. 总蒸散发
    E_total = torch.abs(t1[:, idx_slhf]) * 7 / L_v / 1000.0

    # 大气水量平衡残差：ET_total - P_total ≈ 0
    residual_atmos = E_total - P_total

    return torch.nn.functional.mse_loss(residual_atmos, torch.zeros_like(residual_atmos))

def calculate_water_balance_loss(input_surface_normalized, output_surface_normalized,
                                surface_mean, surface_std):
    """
    计算综合水量平衡损失
    包含三个部分：陆地 + 海洋 + 大气
    根据全球水量守恒：d(S_ocean)/dt + d(S_land)/dt + d(S_atmos)/dt = 0

    Args:
        input_surface_normalized: 标准化的输入表面变量 (B, 26, time, lat, lon)
        output_surface_normalized: 标准化的输出表面变量 (B, 26, time, lat, lon)
        surface_mean: 表面变量均值 (1, 26, 1, 721, 1440)
        surface_std: 表面变量标准差 (1, 26, 1, 721, 1440)
    """
    # 反标准化
    input_physical = denormalize_surface(input_surface_normalized, surface_mean, surface_std)
    output_physical = denormalize_surface(output_surface_normalized, surface_mean, surface_std)

    # 计算三个子系统的水量平衡损失
    loss_land = calculate_land_water_balance_loss(input_physical, output_physical)
    loss_ocean = calculate_ocean_water_balance_loss(input_physical, output_physical)
    loss_atmos = calculate_atmosphere_water_balance_loss(input_physical, output_physical)

    # 组合损失（陆地权重最高，海洋和大气辅助）
    total_water_loss = loss_land + 0.1 * loss_ocean + 0.1 * loss_atmos

    return total_water_loss

def calculate_land_energy_balance_loss(input_physical, output_physical):
    """
    计算陆地能量平衡损失
    能量平衡方程: R_n = LE + H + G
    G = C_soil * D * (T_soil(t+1) - T_soil(t)) / Δt (土壤热通量)

    修正：
    1. slhf, sshf是日累积被平均，需要乘以7恢复周累积
    2. 正确处理符号（向上通量为负，转为正值）
    3. 土壤热容单位可能需要从kJ转换为J

    Args:
        input_physical: 反标准化的输入表面变量 (B, 26, time, lat, lon)
        output_physical: 反标准化的输出表面变量 (B, 26, time, lat, lon)
    """
    device = output_physical.device

    # 加载陆地掩码
    land_mask = torch.load('constant_masks/is_land.pt', map_location=device).float()  # (721, 1440)

    # 加载土壤固体热容（优先使用修正后的文件）
    cs_path_corrected = 'constant_masks/csol_bulk_025deg_721x1440_corrected.pt'
    cs_path_original = 'constant_masks/csol_bulk_025deg_721x1440.pt'

    if os.path.exists(cs_path_corrected):
        # 使用已修正的文件
        cs_soil_bulk = torch.load(cs_path_corrected, map_location=device).float()
        # 限制土壤热容在合理范围内 (1e6 到 1e7 J/(m³·K))
        # 典型值: 沙土 ~1.5e6, 粘土 ~3e6, 湿土 ~4e6
        cs_soil_bulk = torch.clamp(cs_soil_bulk, min=1e6, max=1e7)
    elif os.path.exists(cs_path_original):
        # 加载原始文件并修正
        cs_soil_bulk = torch.load(cs_path_original, map_location=device).float()
        nan_mask = torch.isnan(cs_soil_bulk)
        if not nan_mask.all():
            valid_values = cs_soil_bulk[~nan_mask]
            if valid_values.median() < 1e5:
                cs_soil_bulk[~nan_mask] *= 1000  # kJ -> J
        cs_soil_bulk = torch.where(nan_mask | (cs_soil_bulk < 1e5),
                                   torch.tensor(2e6, device=device),
                                   cs_soil_bulk)
        # 限制范围
        cs_soil_bulk = torch.clamp(cs_soil_bulk, min=1e6, max=1e7)
    else:
        # 默认值
        cs_soil_bulk = torch.ones(721, 1440, device=device) * 2e6

    # 变量索引
    idx_slhf = 13  # slhf - Surface Latent Heat Flux (J/m², 日累积被平均)
    idx_sshf = 14  # sshf - Surface Sensible Heat Flux (J/m², 日累积被平均)
    idx_sw_net = 15  # avg_snswrf - Mean Surface Net Short Wave Radiation Flux (W/m²)
    idx_lw_net = 16  # avg_snlwrf - Mean Surface Net Long Wave Radiation Flux (W/m²)
    idx_stl = 24  # stl - Soil Temperature Layer (K)
    idx_swvl = 25  # swvl - Volumetric Soil Water Layer (m³/m³)

    # 获取时间点
    t0 = input_physical[:, :, -1, :, :]   # 输入的最后一个时刻
    t1 = output_physical[:, :, 0, :, :]   # 输出的第一个时刻

    # 物理常数
    week_seconds = 7 * 24 * 3600
    D = 2.89  # 土壤深度 (m)
    c_w = 4.184e6  # 水的体积热容 (J/(m³·K))

    # 1. 净辐射 R_n (W/m²) - 使用平均通量
    sw_net = t1[:, idx_sw_net] * land_mask  # (W/m²)
    lw_net = t1[:, idx_lw_net] * land_mask  # (W/m²，通常为负值)
    R_n = sw_net + lw_net

    # 2. 潜热和感热通量 (J/m² -> W/m²)
    # 重要修正：日累积被平均，需要乘以7恢复周累积
    # ERA5约定：向上通量（蒸发、感热输出）为负值
    LE_raw = t1[:, idx_slhf] * 7 / week_seconds * land_mask  # W/m²
    H_raw = t1[:, idx_sshf] * 7 / week_seconds * land_mask   # W/m²

    # 在能量平衡中，我们需要正的能量输出值
    LE = -LE_raw  # 转为正值（能量输出）
    H = -H_raw    # 转为正值（能量输出）

    # 3. 土壤热通量 G
    delta_T_soil = (t1[:, idx_stl] - t0[:, idx_stl]) * land_mask
    theta = t1[:, idx_swvl] * land_mask
    C_soil = (cs_soil_bulk.unsqueeze(0) + theta * c_w) * land_mask.unsqueeze(0)
    G = C_soil * D * delta_T_soil / week_seconds

    # 4. 能量平衡残差
    # R_n（输入） = LE（输出） + H（输出） + G（储存）
    residual_land = (R_n - LE - H - G) * land_mask.unsqueeze(0)

    # 处理NaN和Inf值
    residual_land = torch.nan_to_num(residual_land, nan=0.0, posinf=0.0, neginf=0.0)

    # 只对有效陆地像素计算MSE，避免被mask的0值稀释
    valid_pixels = land_mask.sum()
    if valid_pixels > 0:
        mse_loss = (residual_land ** 2).sum() / valid_pixels / output_physical.shape[0]
    else:
        mse_loss = torch.tensor(0.0, device=residual_land.device)

    return mse_loss

def calculate_ocean_energy_balance_loss(input_physical, output_physical):
    """
    计算海洋能量平衡损失
    能量平衡方程: R_n = LE + H + Q_ocean
    其中 Q_ocean 是海洋储热变化

    修正：
    1. slhf, sshf是日累积被平均，需要乘以7恢复周累积
    2. 正确处理符号（向上通量为负）
    3. 添加SST异常值检查

    Args:
        input_physical: 反标准化的输入表面变量 (B, 26, time, lat, lon)
        output_physical: 反标准化的输出表面变量 (B, 26, time, lat, lon)
    """
    device = output_physical.device

    # 加载掩码
    land_mask = torch.load('constant_masks/is_land.pt', map_location=device).float()
    ocean_mask = 1.0 - land_mask

    # 变量索引
    idx_slhf = 13  # J/m², 日累积被平均
    idx_sshf = 14  # J/m², 日累积被平均
    idx_sw_net = 15  # W/m²
    idx_lw_net = 16  # W/m²
    idx_sst = 22  # Sea Surface Temperature (K)

    # 获取时间点
    t0 = input_physical[:, :, -1, :, :]
    t1 = output_physical[:, :, 0, :, :]

    # 物理常数
    week_seconds = 7 * 24 * 3600
    rho_w = 1025  # 海水密度 (kg/m³)
    c_p_ocean = 3985  # 海水比热容 (J/(kg·K))
    mixed_layer_depth = 50  # 混合层深度 (m)，简化假设

    # 1. 净辐射 R_n (W/m²)
    sw_net = t1[:, idx_sw_net] * ocean_mask
    lw_net = t1[:, idx_lw_net] * ocean_mask
    R_n = sw_net + lw_net

    # 2. 潜热和感热通量（修正：乘以7）
    LE_raw = t1[:, idx_slhf] * 7 / week_seconds * ocean_mask
    H_raw = t1[:, idx_sshf] * 7 / week_seconds * ocean_mask

    # 3. 创建有效海洋掩码（基于SST有效值）
    # SST在271K（-2°C）到308K（35°C）之间才是有效海洋
    valid_ocean_mask = (t1[:, idx_sst] > 271) & (t1[:, idx_sst] < 308)
    valid_ocean_mask = valid_ocean_mask.float() * ocean_mask  # 结合原始海洋掩码

    # 重新计算能量组分，只在有效海洋区域
    R_n = (sw_net + lw_net) * valid_ocean_mask
    LE = -LE_raw * valid_ocean_mask  # 转为正值（能量输出）
    H = -H_raw * valid_ocean_mask    # 转为正值（能量输出）

    # 海洋储热变化 (基于SST变化)
    delta_SST = (t1[:, idx_sst] - t0[:, idx_sst]) * valid_ocean_mask
    # 限制异常的SST周变化（不应超过1K）
    delta_SST = torch.clamp(delta_SST, -1.0, 1.0)

    Q_ocean = rho_w * c_p_ocean * mixed_layer_depth * delta_SST / week_seconds

    # 4. 海洋能量平衡残差（只在有效海洋区域）
    residual_ocean = (R_n - LE - H - Q_ocean) * valid_ocean_mask.unsqueeze(0)

    # 处理NaN和Inf值
    residual_ocean = torch.nan_to_num(residual_ocean, nan=0.0, posinf=0.0, neginf=0.0)

    # 只对有效海洋像素计算MSE
    valid_pixels = valid_ocean_mask.sum()
    if valid_pixels > 0:
        mse_loss = (residual_ocean ** 2).sum() / valid_pixels / output_physical.shape[0]
    else:
        mse_loss = torch.tensor(0.0, device=residual_ocean.device)

    return mse_loss

def calculate_energy_balance_loss(input_surface_normalized, output_surface_normalized,
                                 surface_mean, surface_std):
    """
    计算综合能量平衡损失
    包含陆地和海洋两部分

    Args:
        input_surface_normalized: 标准化的输入表面变量 (B, 26, time, lat, lon)
        output_surface_normalized: 标准化的输出表面变量 (B, 26, time, lat, lon)
        surface_mean: 表面变量均值 (1, 26, 1, 721, 1440)
        surface_std: 表面变量标准差 (1, 26, 1, 721, 1440)
    """
    # 反标准化
    input_physical = denormalize_surface(input_surface_normalized, surface_mean, surface_std)
    output_physical = denormalize_surface(output_surface_normalized, surface_mean, surface_std)

    # 计算陆地和海洋的能量平衡损失
    loss_land = calculate_land_energy_balance_loss(input_physical, output_physical)
    loss_ocean = calculate_ocean_energy_balance_loss(input_physical, output_physical)

    # 组合损失（可以调整权重）
    total_energy_loss = loss_land + 0.5 * loss_ocean

    return total_energy_loss

def calculate_hydrostatic_balance_loss(output_upper_normalized, upper_mean, upper_std):
    """
    计算静力平衡损失
    静力平衡: Δφ = R_d * T_avg * ln(p1/p2)
    
    Args:
        output_upper_normalized: 标准化的输出高空变量 (B, 7, levels, time, lat, lon)
        upper_mean: 高空变量均值 (7, levels, lat, lon)
        upper_std: 高空变量标准差 (7, levels, lat, lon)
    """
    output_physical = denormalize_upper(output_upper_normalized, upper_mean, upper_std)
    
    # 变量索引（基于CLAUDE.md）:
    # 1: Geopotential (φ)
    # 2: Temperature (T)
    # 压力层: 200, 300, 500, 700, 850 hPa (索引 0-4)
    
    # 计算 850 hPa (索引 4) 和 700 hPa (索引 3) 之间的静力平衡
    phi_850 = output_physical[:, 1, 4, 0, :, :]  # 850 hPa 位势 (m^2 s^-2)
    phi_700 = output_physical[:, 1, 3, 0, :, :]  # 700 hPa 位势
    temp_850 = output_physical[:, 2, 4, 0, :, :] # 850 hPa 温度 (K)
    temp_700 = output_physical[:, 2, 3, 0, :, :] # 700 hPa 温度
    
    # 模型预测的位势厚度
    delta_phi_model = phi_700 - phi_850
    
    # 物理计算的位势厚度
    R_d = 287  # 干空气气体常数 (J/(kg·K))
    temp_avg = (temp_700 + temp_850) / 2
    delta_phi_physical = R_d * temp_avg * torch.log(torch.tensor(850.0/700.0, device=temp_avg.device))
    
    # 静力平衡残差
    residual_hydrostatic = delta_phi_model - delta_phi_physical
    
    return torch.nn.functional.mse_loss(residual_hydrostatic, torch.zeros_like(residual_hydrostatic))

def calculate_focus_variable_loss(output_surface_norm, target_surface_norm,
                                  output_upper_norm, target_upper_norm):
    """Repeat key-variable MSE so their weight doubles in the total loss."""
    precip_pred = output_surface_norm[:, 0, ...] + output_surface_norm[:, 1, ...]
    precip_target = target_surface_norm[:, 0, ...] + target_surface_norm[:, 1, ...]
    precip_loss = torch.nn.functional.mse_loss(precip_pred, precip_target)

    # 1: Mean Top Net Long Wave Radiation Flux; 10: 2m Temperature
    surface_focus_indices = [1, 10]
    surface_focus_pred = output_surface_norm[:, surface_focus_indices, ...]
    surface_focus_target = target_surface_norm[:, surface_focus_indices, ...]
    surface_focus_loss = torch.nn.functional.mse_loss(surface_focus_pred, surface_focus_target)

    upper_u_pred = output_upper_norm[:, 4, [0, 4], ...]
    upper_u_target = target_upper_norm[:, 4, [0, 4], ...]
    upper_focus_loss = torch.nn.functional.mse_loss(upper_u_pred, upper_u_target)

    return precip_loss + surface_focus_loss + upper_focus_loss

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
        """
        初始化气象数据集 - 按时间序列顺序划分
        
        参数:
            surface_data: 表面数据，形状为 [time, 17, 721, 1440]
            upper_air_data: 高空数据，形状为 [time, 7, 5, 721, 1440]
            start_idx: 开始索引
            end_idx: 结束索引
        """
        self.surface_data = surface_data
        self.upper_air_data = upper_air_data
        self.length = end_idx - start_idx - 2  # 减2确保有足够的目标数据
        self.start_idx = start_idx
        
        print(f"Dataset from index {start_idx} to {end_idx}, sample count: {self.length}")
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        actual_idx = self.start_idx + idx
        
        # 提取输入数据 (t和t+1时刻)
        input_surface = self.surface_data[actual_idx:actual_idx+2]  # [2, 17, 721, 1440]
        # 添加batch维度并调整为 [17, 2, 721, 1440]
        input_surface = np.transpose(input_surface, (1, 0, 2, 3))  # [17, 2, 721, 1440]
        
        # 提取高空数据 (t和t+1时刻)
        input_upper_air = self.upper_air_data[actual_idx:actual_idx+2]  # [2, 7, 5, 721, 1440]
        # 调整为 [7, 5, 2, 721, 1440]
        input_upper_air = np.transpose(input_upper_air, (1, 2, 0, 3, 4))  # [7, 5, 2, 721, 1440]
        
        # 提取目标数据 (t+2时刻)
        target_surface = self.surface_data[actual_idx+2:actual_idx+3]  # [1, 17, 721, 1440]
        # 调整为 [17, 1, 721, 1440]
        target_surface = np.transpose(target_surface, (1, 0, 2, 3))  # [17, 1, 721, 1440]
        
        target_upper_air = self.upper_air_data[actual_idx+2:actual_idx+3]  # [1, 7, 5, 721, 1440]
        # 调整为 [7, 5, 1, 721, 1440]
        target_upper_air = np.transpose(target_upper_air, (1, 2, 0, 3, 4))  # [7, 5, 1, 721, 1440]
        
        return input_surface, input_upper_air, target_surface, target_upper_air

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# # Load constant masks on the selected device
# constant_path = 'constant_masks/Earth.pt'
# input_constant = torch.load(constant_path, map_location=device)
# input_constant = input_constant.to(device)

# 加载数据
print("Loading data...")
input_surface, input_upper_air = h5.File('/gz-data/ERA5_2023_weekly_new.h5')['surface'], h5.File('/gz-data/ERA5_2023_weekly_new.h5')['upper_air']
print(f"Surface data shape: {input_surface.shape}") #(52, 26, 721, 1440)
print(f"Upper air data shape: {input_upper_air.shape}") #(52, 10, 5, 721, 1440)

# 加载标准化参数
print("Loading normalization parameters...")
from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays
# 调用函数获取四个数组
json = '/home/CanglongPhysics/code_v2/ERA5_1940_2023_mean_std_v2.json'
surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(json)

# 转换为张量并移动到设备
surface_mean = torch.from_numpy(surface_mean_np).to(device=device, dtype=torch.float32)
surface_std = torch.from_numpy(surface_std_np).to(device=device, dtype=torch.float32)
upper_mean = torch.from_numpy(upper_mean_np).to(device=device, dtype=torch.float32)
upper_std = torch.from_numpy(upper_std_np).to(device=device, dtype=torch.float32)

print(f"Surface mean shape: {surface_mean.shape}")
print(f"Surface std shape: {surface_std.shape}")
print(f"Upper mean shape: {upper_mean.shape}")
print(f"Upper std shape: {upper_std.shape}")

# 使用全部数据进行训练 (不再划分验证集)
total_samples = 28
train_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=0, end_idx=total_samples)
valid_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=total_samples, end_idx=total_samples+12)

batch_size = 1  # 小batch size便于调试
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=12)
print(f"Created data loader with batch size {batch_size}")
print(f"Total training samples: {len(train_dataset)}")

# Block 4
# Training
model = CanglongV3()

# 多GPU训练
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# 将模型移动到设备
model.to(device)

# 创建优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005 * 2)
criterion = nn.MSELoss()

# 创建保存目录
save_dir = 'checkpoints_v3'
os.makedirs(save_dir, exist_ok=True)

# 训练参数
num_epochs = 50
best_valid_loss = float('inf')

# 物理约束权重（根据损失量级动态调整）
# 目标：让每个物理约束贡献约1-10的损失量级
lambda_water = 5e1    # Water loss ~1e11 -> weight 1e-11 -> contribution ~1
lambda_energy = 5e-4    # Energy loss ~1e12 -> weight 1e-12 -> contribution ~1  
lambda_pressure = 5e-5   # Pressure loss ~1e6 -> weight 1e-6 -> contribution ~1

# 训练循环
print("Starting training with physical constraints...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    surface_loss = 0.0
    upper_air_loss = 0.0
    water_loss_total = 0.0
    energy_loss_total = 0.0
    pressure_loss_total = 0.0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for input_surface, input_upper_air, target_surface, target_upper_air in train_pbar:
        # 将数据移动到设备并转换为float32
        input_surface = input_surface.float().to(device)
        input_upper_air = input_upper_air.float().to(device)
        target_surface = target_surface.float().to(device)
        target_upper_air = target_upper_air.float().to(device)
        
        # 标准化输入数据
        input_surface_norm = (input_surface - surface_mean) / surface_std
        input_upper_air_norm = (input_upper_air - upper_mean) / upper_std
        target_surface_norm = (target_surface - surface_mean) / surface_std
        target_upper_air_norm = (target_upper_air - upper_mean) / upper_std
        
        # 清除梯度
        optimizer.zero_grad()
        
        # 前向传播
        output_surface, output_upper_air = model(input_surface_norm, input_upper_air_norm)
        
        # 计算MSE损失
        loss_surface = criterion(output_surface, target_surface_norm)
        loss_upper_air = criterion(output_upper_air, target_upper_air_norm)
        
        # 计算物理约束损失
        loss_water = calculate_water_balance_loss(
            input_surface_norm, output_surface, 
            surface_mean, surface_std
        )
        loss_energy = calculate_energy_balance_loss(
            input_surface_norm, output_surface,
            surface_mean, surface_std
        )
        loss_pressure = calculate_hydrostatic_balance_loss(
            output_upper_air, upper_mean, upper_std
        )
        
        # 总损失
        loss = loss_surface + loss_upper_air + \
               lambda_water * loss_water + \
               lambda_energy * loss_energy + \
               lambda_pressure * loss_pressure
        
        # 反向传播和优化
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 累加损失
        batch_loss = loss.item()
        train_loss += batch_loss
        surface_loss += loss_surface.item()
        upper_air_loss += loss_upper_air.item()
        water_loss_total += loss_water.item()
        energy_loss_total += loss_energy.item()
        pressure_loss_total += loss_pressure.item()
        
        # 更新进度条
        train_pbar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "surf": f"{loss_surface.item():.4f}",
            "upper": f"{loss_upper_air.item():.4f}",
            "water": f"{loss_water.item():.2e}",
            "energy": f"{loss_energy.item():.2e}",
            "pressure": f"{loss_pressure.item():.2e}"
        })
    
    # 计算平均训练损失
    train_loss = train_loss / len(train_loader)
    surface_loss = surface_loss / len(train_loader)
    upper_air_loss = upper_air_loss / len(train_loader)
    water_loss_total = water_loss_total / len(train_loader)
    energy_loss_total = energy_loss_total / len(train_loader)
    pressure_loss_total = pressure_loss_total / len(train_loader)
    
    # 验证阶段
    model.eval()
    valid_loss = 0.0
    valid_surface_loss = 0.0
    valid_upper_air_loss = 0.0
    valid_water_loss = 0.0
    valid_energy_loss = 0.0
    valid_pressure_loss = 0.0
    
    with torch.no_grad():
        valid_pbar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        for input_surface, input_upper_air, target_surface, target_upper_air in valid_pbar:
            # 将数据移动到设备
            input_surface = input_surface.float().to(device)
            input_upper_air = input_upper_air.float().to(device)
            target_surface = target_surface.float().to(device)
            target_upper_air = target_upper_air.float().to(device)
            
            # 标准化输入数据
            input_surface_norm = (input_surface - surface_mean) / surface_std
            input_upper_air_norm = (input_upper_air - upper_mean) / upper_std
            target_surface_norm = (target_surface - surface_mean) / surface_std
            target_upper_air_norm = (target_upper_air - upper_mean) / upper_std
            
            # 前向传播
            output_surface, output_upper_air = model(input_surface_norm.float(), input_upper_air_norm)
            
            # 计算MSE损失
            loss_surface = criterion(output_surface, target_surface_norm)
            loss_upper_air = criterion(output_upper_air, target_upper_air_norm)
            
            # 计算物理约束损失
            loss_water = calculate_water_balance_loss(
                input_surface_norm, output_surface, 
                surface_mean, surface_std
            )
            loss_energy = calculate_energy_balance_loss(
                input_surface_norm, output_surface,
                surface_mean, surface_std
            )
            loss_pressure = calculate_hydrostatic_balance_loss(
                output_upper_air, upper_mean, upper_std
            )
            
            # 总损失
            loss = loss_surface + loss_upper_air + \
                   lambda_water * loss_water + \
                   lambda_energy * loss_energy + \
                   lambda_pressure * loss_pressure
            
            # 累加损失
            batch_loss = loss.item()
            valid_loss += batch_loss
            valid_surface_loss += loss_surface.item()
            valid_upper_air_loss += loss_upper_air.item()
            valid_water_loss += loss_water.item()
            valid_energy_loss += loss_energy.item()
            valid_pressure_loss += loss_pressure.item()
            
            # 更新进度条
            valid_pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "surf": f"{loss_surface.item():.4f}",
                "upper": f"{loss_upper_air.item():.4f}",
                "water": f"{loss_water.item():.2e}",
                "energy": f"{loss_energy.item():.2e}",
                "pressure": f"{loss_pressure.item():.2e}"
            })
    
    # 计算平均验证损失
    valid_loss = valid_loss / len(valid_loader)
    valid_surface_loss = valid_surface_loss / len(valid_loader)
    valid_upper_air_loss = valid_upper_air_loss / len(valid_loader)
    valid_water_loss = valid_water_loss / len(valid_loader)
    valid_energy_loss = valid_energy_loss / len(valid_loader)
    valid_pressure_loss = valid_pressure_loss / len(valid_loader)
    
    # 打印损失
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train - Total: {train_loss:.6f}")
    print(f"         MSE - Surface: {surface_loss:.6f}, Upper Air: {upper_air_loss:.6f}")
    print(f"         Physical Raw - Water: {water_loss_total:.2e}, Energy: {energy_loss_total:.2e}, Pressure: {pressure_loss_total:.2e}")
    print(f"         Physical Weighted - Water: {lambda_water*water_loss_total:.6f}, Energy: {lambda_energy*energy_loss_total:.6f}, Pressure: {lambda_pressure*pressure_loss_total:.6f}")
    print(f"  Valid - Total: {valid_loss:.6f}")
    print(f"         MSE - Surface: {valid_surface_loss:.6f}, Upper Air: {valid_upper_air_loss:.6f}")
    print(f"         Physical Raw - Water: {valid_water_loss:.2e}, Energy: {valid_energy_loss:.2e}, Pressure: {valid_pressure_loss:.2e}")
    print(f"         Physical Weighted - Water: {lambda_water*valid_water_loss:.6f}, Energy: {lambda_energy*valid_energy_loss:.6f}, Pressure: {lambda_pressure*valid_pressure_loss:.6f}")