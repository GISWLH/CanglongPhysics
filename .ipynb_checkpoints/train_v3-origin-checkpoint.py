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

# ============ 全局缓存的常量掩码 ============
# 避免每次计算损失函数时重复加载文件
MASK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'constant_masks')

# 缓存字典，按device存储
_CACHED_MASKS = {}

def get_cached_masks(device):
    """
    获取缓存的常量掩码，避免重复加载

    Returns:
        dict: 包含 land_mask, basin_mask, ocean_mask, cs_soil_bulk, dem 的字典
    """
    global _CACHED_MASKS

    device_key = str(device)
    if device_key not in _CACHED_MASKS:
        print(f" Loading constant masks to {device}...")

        # 陆地掩码
        land_path = os.path.join(MASK_DIR, 'is_land.pt')
        land_mask = torch.load(land_path, map_location=device, weights_only=True).float()

        # 流域掩码
        basin_path = os.path.join(MASK_DIR, 'hydrobasin_exorheic_mask.pt')
        basin_mask = torch.load(basin_path, map_location=device, weights_only=True).float()

        # 海洋掩码
        ocean_mask = 1.0 - land_mask

        # 土壤热容
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

        # DEM高程
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
        print(f"Loaded masks: land {land_mask.shape}, basin {basin_mask.shape}, cs_soil {cs_soil_bulk.shape}, dem {dem.shape}")

    return _CACHED_MASKS[device_key]


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
    masks = get_cached_masks(device)
    land_mask = masks['land_mask']
    basin_mask = masks['basin_mask']

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
    masks = get_cached_masks(device)
    land_mask = masks['land_mask']
    ocean_mask = masks['ocean_mask']

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

    # 使用缓存的掩码
    masks = get_cached_masks(device)
    land_mask = masks['land_mask']
    cs_soil_bulk = masks['cs_soil_bulk']

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

    # 使用缓存的掩码
    masks = get_cached_masks(device)
    land_mask = masks['land_mask']
    ocean_mask = masks['ocean_mask']

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

def calculate_hydrostatic_balance_loss(output_upper_normalized, output_surface_normalized,
                                      upper_mean, upper_std, surface_mean, surface_std):
    """
    计算完整的静力平衡损失
    包括所有相邻压力层之间的静力平衡以及地表到850hPa的约束
    静力平衡: Δφ = R_d * T_avg * ln(p1/p2)

    Args:
        output_upper_normalized: 标准化的输出高空变量 (B, 10, levels, time, lat, lon)
        output_surface_normalized: 标准化的输出表面变量 (B, 26, time, lat, lon)
        upper_mean: 高空变量均值 (1, 10, 5, 1, 721, 1440)
        upper_std: 高空变量标准差 (1, 10, 5, 1, 721, 1440)
        surface_mean: 表面变量均值 (1, 26, 1, 721, 1440)
        surface_std: 表面变量标准差 (1, 26, 1, 721, 1440)
    """
    # 反标准化
    output_upper_physical = denormalize_upper(output_upper_normalized, upper_mean, upper_std)
    output_surface_physical = denormalize_surface(output_surface_normalized, surface_mean, surface_std)

    device = output_upper_physical.device

    # 变量索引
    # Upper air: 1=Geopotential(z), 2=Temperature(t)
    # Surface: 10=t2m, 19=sp(surface pressure), 20=msl
    # 压力层: 200, 300, 500, 700, 850 hPa (索引 0-4)

    # 物理常数
    R_d = 287  # 干空气气体常数 (J/(kg·K))
    g = 9.80665  # 重力加速度 (m/s²)

    # 压力层值
    pressure_levels = torch.tensor([200.0, 300.0, 500.0, 700.0, 850.0], device=device)

    # 提取位势和温度 (B, 5, lat, lon)
    phi_all = output_upper_physical[:, 1, :, 0, :, :]  # 所有层的位势
    temp_all = output_upper_physical[:, 2, :, 0, :, :]  # 所有层的温度

    # 初始化总损失
    total_loss = torch.tensor(0.0, device=device)
    loss_count = 0

    # 1. 计算所有相邻压力层之间的静力平衡
    layer_pairs = [
        (0, 1, 200, 300),  # 200-300 hPa
        (1, 2, 300, 500),  # 300-500 hPa
        (2, 3, 500, 700),  # 500-700 hPa
        (3, 4, 700, 850),  # 700-850 hPa
    ]

    for idx_upper, idx_lower, p_upper, p_lower in layer_pairs:
        # 提取相邻层的位势和温度
        phi_upper = phi_all[:, idx_upper, :, :]
        phi_lower = phi_all[:, idx_lower, :, :]
        temp_upper = temp_all[:, idx_upper, :, :]
        temp_lower = temp_all[:, idx_lower, :, :]

        # 模型预测的位势厚度
        delta_phi_model = phi_upper - phi_lower

        # 物理计算的位势厚度
        temp_avg = (temp_upper + temp_lower) / 2
        delta_phi_physical = R_d * temp_avg * torch.log(torch.tensor(p_lower/p_upper, device=device))

        # 计算残差和损失
        residual = delta_phi_model - delta_phi_physical
        residual = torch.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)

        loss = torch.nn.functional.mse_loss(residual, torch.zeros_like(residual))
        total_loss = total_loss + loss
        loss_count += 1

    # 2. 地表到850hPa的静力平衡约束
    # 使用缓存的DEM高程数据
    masks = get_cached_masks(device)
    dem = masks['dem']

    if dem.sum() > 0:  # 确保DEM已加载
        # 提取地表变量
        t2m = output_surface_physical[:, 10, 0, :, :]  # 2m温度 (K)
        sp = output_surface_physical[:, 19, 0, :, :]   # 地表气压 (Pa)

        # 850hPa层的变量
        phi_850 = phi_all[:, 4, :, :]  # 850hPa位势
        temp_850 = temp_all[:, 4, :, :]  # 850hPa温度

        # 计算地表位势 (φ_surface = g * z)
        phi_surface = g * dem.unsqueeze(0)

        # 模型预测的位势差
        delta_phi_model_sfc = phi_850 - phi_surface

        # 物理计算的位势差（使用压高公式）
        # 只在地表压力 > 850hPa 的地方计算（避免地形太高的区域）
        valid_mask = (sp > 85000).float()  # 地表压力大于850hPa

        # 平均温度（地表2m温度和850hPa温度）
        temp_avg_sfc = (t2m + temp_850) / 2

        # 压力比 (注意单位转换)
        p_ratio = (sp / 85000.0).clamp(min=0.9, max=1.2)  # 限制范围避免极端值

        # 物理计算的位势差
        delta_phi_physical_sfc = R_d * temp_avg_sfc * torch.log(p_ratio) * valid_mask

        # 计算残差（只在有效区域）
        residual_sfc = (delta_phi_model_sfc - delta_phi_physical_sfc) * valid_mask
        residual_sfc = torch.nan_to_num(residual_sfc, nan=0.0, posinf=0.0, neginf=0.0)

        # 计算损失（只对有效像素）
        valid_pixels = valid_mask.sum()
        if valid_pixels > 0:
            loss_sfc = (residual_sfc ** 2).sum() / valid_pixels / output_surface_physical.shape[0]
        else:
            loss_sfc = torch.tensor(0.0, device=device)

        total_loss = total_loss + loss_sfc
        loss_count += 1

    # 3. 海平面气压修正约束
    # 使用更准确的压高公式: msl = sp * (1 + L*z/T)^(g/(R_d*L))
    # 其中T是地表温度，不是海平面标准温度
    msl = output_surface_physical[:, 20, 0, :, :]  # 海平面气压 (Pa)
    # sp 和 t2m 已经在上面提取过了

    # 标准大气参数
    L = 0.0065  # 温度递减率 (K/m)

    # 计算理论海平面气压
    # 使用地表温度t2m而不是海平面标准温度
    # 防止除零错误，给t2m加个下限
    t2m_safe = torch.clamp(t2m, min=200.0)  # 确保温度合理

    # 压高公式: msl = sp * (1 + L*z/T)^(g/(R_d*L))
    # 注意：这里dem是地形高度，向上为正
    msl_theoretical = sp * torch.pow(1 + L * dem.unsqueeze(0) / t2m_safe, g / (R_d * L))

    # 计算残差（主要在低海拔地区约束）
    # 同时限制DEM在合理范围内（避免海底和极高山区）
    valid_altitude_mask = ((dem >= 0) & (dem < 1500)).float()  # 0-1500m范围
    residual_msl = (msl - msl_theoretical) * valid_altitude_mask.unsqueeze(0)
    residual_msl = torch.nan_to_num(residual_msl, nan=0.0, posinf=0.0, neginf=0.0)

    # MSE损失
    valid_pixels_msl = valid_altitude_mask.sum()
    if valid_pixels_msl > 0:
        loss_msl = (residual_msl ** 2).sum() / valid_pixels_msl / output_surface_physical.shape[0]
    else:
        loss_msl = torch.tensor(0.0, device=device)

    total_loss = total_loss + 0.1 * loss_msl  # 降低权重，因为这是辅助约束

    # 返回平均损失
    return total_loss / max(loss_count, 1)

# 温度平衡方程权重参数（通过ERA5数据搜索优化得到）
# 由于周平均数据存在协方差损失问题，需要缩放各项
ALPHA = 0.01  # 水平平流项权重
BETA = 0.01   # 垂直运动项权重
GAMMA = 0.1   # 非绝热加热项权重



def calculate_temperature_tendency_loss(input_upper_normalized, output_upper_normalized,
                                       input_surface_normalized, output_surface_normalized,
                                       upper_mean, upper_std, surface_mean, surface_std):
    """
    计算温度局地变化方程约束
    基于大气热力学方程：∂T/∂t = α*(-V_h·∇_h T) + β*(-(R_d*T/(c_p*p) - ∂T/∂p)·ω) + γ*Q

    注意：由于使用周平均数据，存在协方差损失（mean(u·∂T/∂x) ≠ mean(u)·mean(∂T/∂x)）
    因此各项需要乘以经验权重参数进行缩放

    Args:
        input_upper_normalized: 标准化的输入高空变量 (B, 10, 5, time, lat, lon)
        output_upper_normalized: 标准化的输出高空变量 (B, 10, 5, time, lat, lon)
        input_surface_normalized: 标准化的输入表面变量 (B, 26, time, lat, lon)
        output_surface_normalized: 标准化的输出表面变量 (B, 26, time, lat, lon)
        upper_mean: 高空变量均值 (1, 10, 5, 1, 721, 1440)
        upper_std: 高空变量标准差 (1, 10, 5, 1, 721, 1440)
        surface_mean: 表面变量均值 (1, 26, 1, 721, 1440)
        surface_std: 表面变量标准差 (1, 26, 1, 721, 1440)

    Returns:
        温度趋势方程的MSE损失
    """
    device = input_upper_normalized.device

    # 物理常数
    R_d = 287.0  # J/(kg·K) 干空气气体常数
    c_p = 1004.0  # J/(kg·K) 定压比热
    g = 9.8  # m/s² 重力加速度
    L_v = 2.5e6  # J/kg 水的蒸发潜热

    # 反标准化获取物理值
    input_upper_physical = input_upper_normalized * upper_std + upper_mean
    output_upper_physical = output_upper_normalized * upper_std + upper_mean
    input_surface_physical = input_surface_normalized * surface_std + surface_mean
    output_surface_physical = output_surface_normalized * surface_std + surface_mean

    # 提取需要的变量索引
    # Upper air: 0=o3, 1=z, 2=t, 3=u, 4=v, 5=w, 6=q, 7=cc, 8=ciwc, 9=clwc
    idx_t, idx_u, idx_v, idx_w = 2, 3, 4, 5
    idx_q, idx_o3, idx_clwc, idx_ciwc = 6, 0, 9, 8

    # Surface: 根据ERA5变量顺序
    idx_tnswrf, idx_tnlwrf = 0, 1  # 顶部辐射通量
    idx_lsrr, idx_crr = 4, 5  # 降水率
    idx_slhf, idx_sshf = 13, 14  # 潜热和感热通量
    idx_snswrf, idx_snlwrf = 15, 16  # 表面辐射通量
    idx_blh = 6  # 边界层高度

    # 使用输入的最后时刻(t1)和输出的第一时刻(t2)计算时间导数
    t_t1 = input_upper_physical[:, idx_t, :, -1, :, :]  # (B, 5, lat, lon) at time t1
    t_t2 = output_upper_physical[:, idx_t, :, 0, :, :]   # (B, 5, lat, lon) at time t2

    # 使用t2时刻的场进行计算
    u = output_upper_physical[:, idx_u, :, 0, :, :]  # (B, 5, lat, lon)
    v = output_upper_physical[:, idx_v, :, 0, :, :]  # (B, 5, lat, lon)
    w = output_upper_physical[:, idx_w, :, 0, :, :]  # (B, 5, lat, lon) Pa/s
    t = output_upper_physical[:, idx_t, :, 0, :, :]  # (B, 5, lat, lon)
    q = output_upper_physical[:, idx_q, :, 0, :, :]  # 比湿
    o3 = output_upper_physical[:, idx_o3, :, 0, :, :]  # 臭氧
    clwc = output_upper_physical[:, idx_clwc, :, 0, :, :]  # 云液水
    ciwc = output_upper_physical[:, idx_ciwc, :, 0, :, :]  # 云冰水

    # 时间步长（1周 = 7天）
    dt = 7 * 24 * 3600  # seconds

    # ========== 1. 温度局地变化率 ∂T/∂t ==========
    dT_dt_observed = (t_t2 - t_t1) / dt  # K/s

    # ========== 2. 水平平流项 -V_h·∇_h T ==========
    # 网格间距
    dlat = 0.25 * np.pi / 180  # radians
    dlon = 0.25 * np.pi / 180  # radians
    R_earth = 6.371e6  # meters

    # 创建纬度权重
    lat_values = torch.linspace(-90, 90, 721, device=device) * np.pi / 180
    cos_lat = torch.cos(lat_values).view(1, 1, -1, 1)

    # 计算温度梯度 (中心差分)
    # ∂T/∂x (东西向)
    t_padded_x = torch.nn.functional.pad(t, (1, 1, 0, 0), mode='circular')
    dT_dx = (t_padded_x[:, :, :, 2:] - t_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)

    # ∂T/∂y (南北向)
    t_padded_y = torch.nn.functional.pad(t, (0, 0, 1, 1), mode='replicate')
    dT_dy = (t_padded_y[:, :, 2:, :] - t_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    # 水平平流
    horizontal_advection = -(u * dT_dx + v * dT_dy)

    # ========== 3. 垂直运动项 -(R_d*T/(c_p*p) - ∂T/∂p)·ω ==========
    # 压力层 (Pa)
    pressure_levels = torch.tensor([200, 300, 500, 700, 850], device=device, dtype=torch.float32) * 100
    p_3d = pressure_levels.view(1, 5, 1, 1).expand_as(t)

    # 计算垂直温度梯度 ∂T/∂p
    dT_dp = torch.zeros_like(t)
    for i in range(5):
        if i == 0:  # 200 hPa, 向前差分
            dT_dp[:, i] = (t[:, i+1] - t[:, i]) / (pressure_levels[i+1] - pressure_levels[i])
        elif i == 4:  # 850 hPa, 向后差分
            dT_dp[:, i] = (t[:, i] - t[:, i-1]) / (pressure_levels[i] - pressure_levels[i-1])
        else:  # 中间层，中心差分
            dT_dp[:, i] = (t[:, i+1] - t[:, i-1]) / (pressure_levels[i+1] - pressure_levels[i-1])

    # 绝热项
    adiabatic_term = (R_d * t / (c_p * p_3d) - dT_dp) * w
    vertical_motion_term = -adiabatic_term

    # ========== 4. 非绝热加热项 Q/c_p (简化处理) ==========

    # 4.1 辐射加热项
    # 大气柱辐射吸收
    # 提取t2时刻的表面辐射
    tnswrf = output_surface_physical[:, idx_tnswrf, 0, :, :]  # (B, lat, lon)
    tnlwrf = output_surface_physical[:, idx_tnlwrf, 0, :, :]  # (B, lat, lon)
    snswrf = output_surface_physical[:, idx_snswrf, 0, :, :]  # (B, lat, lon)
    snlwrf = output_surface_physical[:, idx_snlwrf, 0, :, :]  # (B, lat, lon)

    # 大气短波和长波吸收 (W/m²)
    A_sw = tnswrf - snswrf  # 顶部减表面
    A_lw = tnlwrf - snlwrf

    # 构建辐射加热权重廓线（简化版本）
    # 短波权重：主要由水汽、臭氧和云吸收
    w_sw = 0.5 * q + 0.3 * o3 + 0.2 * (clwc + ciwc)
    w_sw_sum = w_sw.sum(dim=1, keepdim=True).clamp(min=1e-10)
    w_sw_norm = w_sw / w_sw_sum

    # 长波权重：主要由水汽和云决定
    w_lw = 0.7 * q + 0.3 * (clwc + ciwc)
    w_lw_sum = w_lw.sum(dim=1, keepdim=True).clamp(min=1e-10)
    w_lw_norm = w_lw / w_lw_sum

    # 分配辐射加热到各层 (K/s)
    # 需要考虑层厚度
    dp = torch.zeros(5, device=device)
    for i in range(5):
        if i == 0:
            dp[i] = (pressure_levels[0] + pressure_levels[1]) / 2 - 0  # 顶层
        elif i == 4:
            dp[i] = 100000 - (pressure_levels[3] + pressure_levels[4]) / 2  # 底层
        else:
            dp[i] = (pressure_levels[i-1] + pressure_levels[i]) / 2 - (pressure_levels[i] + pressure_levels[i+1]) / 2

    dp = dp.view(1, 5, 1, 1)

    # 辐射加热率 (K/s)
    Q_rad_sw = (g / c_p) * A_sw.unsqueeze(1) * w_sw_norm / (dp / 100)  # 转换单位
    Q_rad_lw = (g / c_p) * A_lw.unsqueeze(1) * w_lw_norm / (dp / 100)
    Q_rad = (Q_rad_sw + Q_rad_lw) / 1000  # 缩放到合理范围

    # 4.2 潜热加热项（简化：与降水相关）
    lsrr = output_surface_physical[:, idx_lsrr, 0, :, :]  # kg/(m²·s)
    crr = output_surface_physical[:, idx_crr, 0, :, :]  # kg/(m²·s)
    total_precip = lsrr + crr  # 总降水率

    # 潜热加热主要在中层（简化分布）
    latent_profile = torch.tensor([0.1, 0.2, 0.4, 0.2, 0.1], device=device).view(1, 5, 1, 1)
    Q_latent = (L_v / c_p) * total_precip.unsqueeze(1) * latent_profile / 1000  # K/s

    # 4.3 感热加热项（主要在边界层）
    sshf = output_surface_physical[:, idx_sshf, 0, :, :]  # J/m²
    blh = output_surface_physical[:, idx_blh, 0, :, :]  # m

    # 感热转换为通量率 (W/m²)
    F_sen = sshf * 7 / dt  # 乘以7因为是周累积的日平均

    # 感热主要影响850hPa（近地面）
    sensible_profile = torch.tensor([0.0, 0.0, 0.0, 0.1, 0.9], device=device).view(1, 5, 1, 1)

    # 近地面空气密度估算 (kg/m³)
    rho_surface = p_3d[:, -1:, :, :] / (R_d * t[:, -1:, :, :])  # 使用850hPa

    # 感热加热率 (K/s)
    Q_sensible = torch.zeros_like(t)
    Q_sensible[:, -1] = F_sen / (rho_surface.squeeze(1) * c_p * 1000) / 100  # 缩放

    # 总非绝热加热
    Q_diabatic = Q_rad + Q_latent + Q_sensible * sensible_profile

    # ========== 5. 构建温度趋势方程 ==========
    # 理论温度趋势（根据动力和热力过程，使用权重参数缩放）
    # 权重参数补偿周平均数据的协方差损失
    dT_dt_theoretical = (ALPHA * horizontal_advection +
                         BETA * vertical_motion_term +
                         GAMMA * Q_diabatic)

    # 计算残差
    residual = dT_dt_observed - dT_dt_theoretical

    # 使用层权重（对流层权重更高）
    layer_weights = torch.tensor([0.5, 1.0, 1.0, 1.0, 1.0], device=device).view(1, 5, 1, 1)

    # 加权残差
    weighted_residual = residual * layer_weights

    # MSE损失
    loss = (weighted_residual ** 2).mean()

    return loss


# 地转平衡闭合率分析结果 (排除赤道±15°):
#   200 hPa: 21.8% (c_pgf=0.99)
#   300 hPa: 19.5% (c_pgf=0.99)
#   500 hPa: 16.3% (c_pgf=0.93) <- 最佳
#   700 hPa: 31.8% (c_pgf=0.81)
#   850 hPa: 72.6% (c_pgf=0.39) <- 边界层效应显著

# 分层气压梯度力系数 [200, 300, 500, 700, 850] hPa
PGF_COEFFICIENTS = [0.99, 0.99, 0.93, 0.81, 0.39]

# 分层权重：根据闭合率反向设置（闭合率好的层权重高）
NS_LAYER_WEIGHTS = [1.0, 1.0, 1.2, 0.8, 0.3]  # 500hPa最高权重，850hPa最低


def calculate_navier_stokes_loss(input_upper_normalized, output_upper_normalized,
                                 input_surface_normalized, output_surface_normalized,
                                 upper_mean, upper_std, surface_mean, surface_std,
                                 use_geostrophic_only=False):
    """
    计算纳维-斯托克斯方程（水平动量方程）约束

    基于ERA5_2023_weekly数据的闭合率分析，等式两侧偏差：
    - u分量地转平衡: 17-22% (高层), 58% (850hPa)
    - 完整NS方程: ~100% (因为|LHS| << |RHS|)

    方程形式 (等压面坐标):
    u分量: ∂u/∂t + u∂u/∂x + v∂u/∂y + w∂u/∂p = fv - ∂Φ/∂x + Fx
    v分量: ∂v/∂t + u∂v/∂x + v∂v/∂y + w∂v/∂p = -fu - ∂Φ/∂y + Fy

    其中:
    - f = 2Ω*sin(φ) 是科氏参数 (rad/s)
    - Φ 是位势 (m²/s²)
    - Fx, Fy 是摩擦力项 (m/s²)

    Args:
        input_upper_normalized: 标准化的输入高空变量 (B, 10, 5, time, lat, lon)
        output_upper_normalized: 标准化的输出高空变量 (B, 10, 5, time, lat, lon)
        input_surface_normalized: 标准化的输入表面变量 (B, 26, time, lat, lon)
        output_surface_normalized: 标准化的输出表面变量 (B, 26, time, lat, lon)
        upper_mean, upper_std: 高空变量标准化参数
        surface_mean, surface_std: 表面变量标准化参数
        use_geostrophic_only: 是否只使用地转平衡约束（更稳定，闭合率更好）

    Returns:
        纳维-斯托克斯方程的MSE损失
    """
    device = input_upper_normalized.device

    # 物理常数
    OMEGA = 7.2921e-5  # 地球自转角速度 (rad/s)
    R_d = 287.0  # 干空气气体常数 (J/(kg·K))
    R_earth = 6.371e6  # 地球半径 (m)

    # 反标准化获取物理值
    input_upper_physical = input_upper_normalized * upper_std + upper_mean
    output_upper_physical = output_upper_normalized * upper_std + upper_mean
    input_surface_physical = input_surface_normalized * surface_std + surface_mean
    output_surface_physical = output_surface_normalized * surface_std + surface_mean

    # 变量索引
    # Upper air: 0=o3, 1=z, 2=t, 3=u, 4=v, 5=w, 6=q
    idx_z, idx_t, idx_u, idx_v, idx_w = 1, 2, 3, 4, 5

    # Surface: 11=avg_iews (东向湍流应力), 12=avg_inss (北向湍流应力)
    idx_iews, idx_inss = 11, 12

    # 时间步长（1周 = 604800秒）
    dt = 7 * 24 * 3600  # seconds

    # 提取风场变量 (B, 5, lat, lon)
    u_t1 = input_upper_physical[:, idx_u, :, -1, :, :]
    u_t2 = output_upper_physical[:, idx_u, :, 0, :, :]
    v_t1 = input_upper_physical[:, idx_v, :, -1, :, :]
    v_t2 = output_upper_physical[:, idx_v, :, 0, :, :]
    w = output_upper_physical[:, idx_w, :, 0, :, :]  # 垂直速度 (Pa/s)
    t = output_upper_physical[:, idx_t, :, 0, :, :]  # 温度 (K)
    phi = output_upper_physical[:, idx_z, :, 0, :, :]  # 位势 (m²/s²)

    # 网格间距 (弧度)
    dlat = 0.25 * np.pi / 180
    dlon = 0.25 * np.pi / 180

    # 纬度数组
    lat_values = torch.linspace(-90, 90, 721, device=device) * np.pi / 180
    cos_lat = torch.cos(lat_values).view(1, 1, -1, 1).clamp(min=0.01)

    # 科氏参数 f = 2Ω*sin(φ)
    f = 2 * OMEGA * torch.sin(lat_values).view(1, 1, -1, 1)

    # 当前时刻风场
    u = u_t2
    v = v_t2

    # ========== 科氏力 ==========
    coriolis_u = f * v   # fv (m/s²)
    coriolis_v = -f * u  # -fu (m/s²)

    # ========== 气压梯度力（位势梯度） ==========
    # ∂Φ/∂x
    phi_padded_x = torch.nn.functional.pad(phi, (1, 1, 0, 0), mode='circular')
    dphi_dx = (phi_padded_x[:, :, :, 2:] - phi_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)

    # ∂Φ/∂y
    phi_padded_y = torch.nn.functional.pad(phi, (0, 0, 1, 1), mode='replicate')
    dphi_dy = (phi_padded_y[:, :, 2:, :] - phi_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    # 气压梯度力 = -∇Φ (m/s²)
    pgf_u = -dphi_dx
    pgf_v = -dphi_dy

    # 应用分层系数（基于闭合率分析）
    pgf_coeffs = torch.tensor(PGF_COEFFICIENTS, device=device, dtype=torch.float32).view(1, 5, 1, 1)
    pgf_u = pgf_u * pgf_coeffs
    pgf_v = pgf_v * pgf_coeffs

    if use_geostrophic_only:
        # ========== 仅地转平衡约束 ==========
        # 地转平衡: fv ≈ -∂Φ/∂x, -fu ≈ -∂Φ/∂y
        # 闭合率: 200-500hPa约17-22%, 更稳定

        residual_u = coriolis_u - pgf_u
        residual_v = coriolis_v - pgf_v

        # 排除赤道区域（±15°，f≈0导致地转平衡不适用）
        lat_mask = (lat_values.abs() > 15 * np.pi / 180).view(1, 1, -1, 1)
        residual_u = residual_u * lat_mask
        residual_v = residual_v * lat_mask

    else:
        # ========== 完整NS方程约束 ==========

        # 1. 时间导数 ∂u/∂t, ∂v/∂t (m/s²)
        du_dt_observed = (u_t2 - u_t1) / dt
        dv_dt_observed = (v_t2 - v_t1) / dt

        # 2. 水平平流项 (m/s²)
        # ∂u/∂x
        u_padded_x = torch.nn.functional.pad(u, (1, 1, 0, 0), mode='circular')
        du_dx = (u_padded_x[:, :, :, 2:] - u_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)

        # ∂u/∂y
        u_padded_y = torch.nn.functional.pad(u, (0, 0, 1, 1), mode='replicate')
        du_dy = (u_padded_y[:, :, 2:, :] - u_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

        # ∂v/∂x
        v_padded_x = torch.nn.functional.pad(v, (1, 1, 0, 0), mode='circular')
        dv_dx = (v_padded_x[:, :, :, 2:] - v_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)

        # ∂v/∂y
        v_padded_y = torch.nn.functional.pad(v, (0, 0, 1, 1), mode='replicate')
        dv_dy = (v_padded_y[:, :, 2:, :] - v_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

        # 水平平流
        u_advection_h = u * du_dx + v * du_dy
        v_advection_h = u * dv_dx + v * dv_dy

        # 3. 垂直平流项 w∂u/∂p (m/s²)
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

        # 垂直平流 (w是Pa/s)
        u_advection_v = w * du_dp
        v_advection_v = w * dv_dp

        # 4. 摩擦力项（仅850hPa近地面层）
        tau_x = output_surface_physical[:, idx_iews, 0, :, :]  # N/m² = Pa
        tau_y = output_surface_physical[:, idx_inss, 0, :, :]  # N/m²

        # 近地面空气密度 ρ = p/(R*T)
        p_850 = pressure_levels[4]
        t_850 = t[:, 4, :, :]
        rho_850 = p_850 / (R_d * t_850)  # kg/m³

        # 摩擦加速度 = τ/(ρ*h)，其中h是边界层厚度
        h_bl = 1000.0  # m
        friction_u = torch.zeros_like(u)
        friction_v = torch.zeros_like(v)
        friction_u[:, 4] = tau_x / (rho_850 * h_bl)  # m/s²
        friction_v[:, 4] = tau_y / (rho_850 * h_bl)

        # 5. 构建动量方程
        # LHS: ∂u/∂t + u∂u/∂x + v∂u/∂y + w∂u/∂p
        # RHS: fv - ∂Φ/∂x + Fx
        # 残差 = LHS - RHS

        lhs_u = du_dt_observed + u_advection_h + u_advection_v
        lhs_v = dv_dt_observed + v_advection_h + v_advection_v

        rhs_u = coriolis_u + pgf_u + friction_u
        rhs_v = coriolis_v + pgf_v + friction_v

        residual_u = lhs_u - rhs_u
        residual_v = lhs_v - rhs_v

    # ========== 应用层权重 ==========
    layer_weights = torch.tensor(NS_LAYER_WEIGHTS, device=device, dtype=torch.float32).view(1, 5, 1, 1)

    weighted_residual_u = residual_u * layer_weights
    weighted_residual_v = residual_v * layer_weights

    # 处理NaN和Inf
    weighted_residual_u = torch.nan_to_num(weighted_residual_u, nan=0.0, posinf=0.0, neginf=0.0)
    weighted_residual_v = torch.nan_to_num(weighted_residual_v, nan=0.0, posinf=0.0, neginf=0.0)

    # MSE损失
    loss_u = (weighted_residual_u ** 2).mean()
    loss_v = (weighted_residual_v ** 2).mean()

    return loss_u + loss_v


def calculate_geostrophic_loss(output_upper_normalized, upper_mean, upper_std):
    """
    简化版：仅计算地转平衡约束损失

    地转平衡: fv = -∂Φ/∂x, -fu = -∂Φ/∂y
    闭合率: 200-500hPa约17-22%（高层较好）

    Args:
        output_upper_normalized: 标准化的输出高空变量 (B, 10, 5, 1, lat, lon)
        upper_mean, upper_std: 高空变量标准化参数

    Returns:
        地转平衡MSE损失
    """
    device = output_upper_normalized.device

    OMEGA = 7.2921e-5
    R_earth = 6.371e6

    # 反标准化
    output_upper_physical = output_upper_normalized * upper_std + upper_mean

    # 变量索引
    idx_z, idx_u, idx_v = 1, 3, 4

    u = output_upper_physical[:, idx_u, :, 0, :, :]  # (B, 5, lat, lon)
    v = output_upper_physical[:, idx_v, :, 0, :, :]
    phi = output_upper_physical[:, idx_z, :, 0, :, :]

    # 网格
    dlat = 0.25 * np.pi / 180
    dlon = 0.25 * np.pi / 180
    lat_values = torch.linspace(-90, 90, 721, device=device) * np.pi / 180
    cos_lat = torch.cos(lat_values).view(1, 1, -1, 1).clamp(min=0.01)
    f = 2 * OMEGA * torch.sin(lat_values).view(1, 1, -1, 1)

    # 科氏力
    coriolis_u = f * v
    coriolis_v = -f * u

    # 气压梯度力
    phi_padded_x = torch.nn.functional.pad(phi, (1, 1, 0, 0), mode='circular')
    dphi_dx = (phi_padded_x[:, :, :, 2:] - phi_padded_x[:, :, :, :-2]) / (2 * R_earth * dlon * cos_lat)

    phi_padded_y = torch.nn.functional.pad(phi, (0, 0, 1, 1), mode='replicate')
    dphi_dy = (phi_padded_y[:, :, 2:, :] - phi_padded_y[:, :, :-2, :]) / (2 * R_earth * dlat)

    pgf_u = -dphi_dx
    pgf_v = -dphi_dy

    # 应用系数
    pgf_coeffs = torch.tensor(PGF_COEFFICIENTS, device=device, dtype=torch.float32).view(1, 5, 1, 1)
    pgf_u = pgf_u * pgf_coeffs
    pgf_v = pgf_v * pgf_coeffs

    # 地转平衡残差
    residual_u = coriolis_u - pgf_u
    residual_v = coriolis_v - pgf_v

    # 排除赤道
    lat_mask = (lat_values.abs() > 15 * np.pi / 180).view(1, 1, -1, 1)
    residual_u = residual_u * lat_mask
    residual_v = residual_v * lat_mask

    # 层权重
    layer_weights = torch.tensor(NS_LAYER_WEIGHTS, device=device, dtype=torch.float32).view(1, 5, 1, 1)
    weighted_residual_u = residual_u * layer_weights
    weighted_residual_v = residual_v * layer_weights

    # 处理NaN
    weighted_residual_u = torch.nan_to_num(weighted_residual_u, nan=0.0, posinf=0.0, neginf=0.0)
    weighted_residual_v = torch.nan_to_num(weighted_residual_v, nan=0.0, posinf=0.0, neginf=0.0)

    loss_u = (weighted_residual_u ** 2).mean()
    loss_v = (weighted_residual_v ** 2).mean()

    return loss_u + loss_v


def calculate_focus_variable_loss(output_surface_norm, target_surface_norm,
                                  output_upper_norm, target_upper_norm):
    """
    计算重要变量的额外MSE损失，使其权重加倍

    重点变量（基于S2S/MJO预报需求）:

    Surface层:
    - 索引1: avg_tnlwrf - Mean Top Net Long Wave Radiation Flux (OLR，MJO关键指标)
    - 索引4: lsrr - Large Scale Rain Rate
    - 索引5: crr - Convective Rain Rate
    - 索引10: t2m - 2m Temperature

    Upper Air层 (B, 10, 5, time, lat, lon):
    - 变量索引3 = u (U风分量)
    - 压力层索引0 = 200hPa, 压力层索引4 = 850hPa
    - 200hPa和850hPa纬向风是MJO的关键动力指标

    Args:
        output_surface_norm: 标准化的输出表面变量 (B, 26, 1, lat, lon)
        target_surface_norm: 标准化的目标表面变量 (B, 26, 1, lat, lon)
        output_upper_norm: 标准化的输出高空变量 (B, 10, 5, 1, lat, lon)
        target_upper_norm: 标准化的目标高空变量 (B, 10, 5, 1, lat, lon)

    Returns:
        重点变量的额外MSE损失
    """
    # 1. 降水损失 (lsrr + crr)
    # 索引4 = lsrr (大尺度降雨率), 索引5 = crr (对流降雨率)
    precip_pred = output_surface_norm[:, 4, ...] + output_surface_norm[:, 5, ...]
    precip_target = target_surface_norm[:, 4, ...] + target_surface_norm[:, 5, ...]
    precip_loss = torch.nn.functional.mse_loss(precip_pred, precip_target)

    # 2. OLR和2m温度损失
    # 索引1 = avg_tnlwrf (OLR), 索引10 = t2m (2米温度)
    surface_focus_indices = [1, 10]
    surface_focus_pred = output_surface_norm[:, surface_focus_indices, ...]
    surface_focus_target = target_surface_norm[:, surface_focus_indices, ...]
    surface_focus_loss = torch.nn.functional.mse_loss(surface_focus_pred, surface_focus_target)

    # 3. 200hPa和850hPa纬向风损失
    # Upper air维度: (B, 10, 5, time, lat, lon)
    # 变量索引3 = u, 压力层索引0 = 200hPa, 压力层索引4 = 850hPa
    upper_u_pred = output_upper_norm[:, 3, [0, 4], ...]  # 200hPa和850hPa的u风
    upper_u_target = target_upper_norm[:, 3, [0, 4], ...]
    upper_focus_loss = torch.nn.functional.mse_loss(upper_u_pred, upper_u_target)

    return precip_loss + surface_focus_loss + upper_focus_loss


# ============ Tweedie损失函数 (Hunt 2025) ============
# 基于论文: "Stop using root-mean-square error as a precipitation target!"
# Tweedie分布的方差函数 V(μ) = μ^p，对于 1 < p < 2 是复合Poisson-Gamma分布
# 适合降水数据：零膨胀、非负、重尾

# ERA5周数据拟合的Tweedie幂参数 (通过方差-均值幂律关系估计)
TWEEDIE_P_LSRR = 1.54  # 大尺度降水率, R² = 0.91
TWEEDIE_P_CRR = 1.59   # 对流降水率, R² = 0.95


def tweedie_deviance(y_true, y_pred, p):
    """
    计算Tweedie deviance损失 (Hunt 2025, Eq. 10)

    d_p(y, μ) = 2 * [y^(2-p) / ((1-p)(2-p)) - y*μ^(1-p) / (1-p) + μ^(2-p) / (2-p)]

    其中:
    - y 是观测值 (target)
    - μ 是预测值 (prediction), 必须 > 0
    - p 是Tweedie幂参数, 1 < p < 2 对应复合Poisson-Gamma分布

    Args:
        y_true: 观测值张量 (非负)
        y_pred: 预测值张量 (必须 > 0)
        p: Tweedie幂参数

    Returns:
        Tweedie deviance的均值
    """
    # 确保预测值为正（加小常数避免数值问题）
    eps = 1e-8
    mu = torch.clamp(y_pred, min=eps)
    y = torch.clamp(y_true, min=0.0)

    # 计算 (1-p) 和 (2-p)
    one_minus_p = 1.0 - p
    two_minus_p = 2.0 - p

    # Tweedie deviance公式 (Eq. 10)
    # d_p(y, μ) = 2 * [y^(2-p) / ((1-p)(2-p)) - y*μ^(1-p) / (1-p) + μ^(2-p) / (2-p)]
    term1 = torch.pow(y + eps, two_minus_p) / (one_minus_p * two_minus_p)
    term2 = y * torch.pow(mu, one_minus_p) / one_minus_p
    term3 = torch.pow(mu, two_minus_p) / two_minus_p

    deviance = 2.0 * (term1 - term2 + term3)

    # 处理NaN和Inf
    deviance = torch.nan_to_num(deviance, nan=0.0, posinf=0.0, neginf=0.0)

    return deviance.mean()


def calculate_tweedie_precipitation_loss(output_surface_physical, target_surface_physical):
    """
    计算降水变量的Tweedie损失

    根据Hunt 2025论文，使用Tweedie deviance替代MSE可以：
    1. 更好地处理零值（降水数据零膨胀）
    2. 更好地捕获极端降水事件
    3. 避免负值预测

    变量索引（Surface层）:
    - 索引4: lsrr - Large Scale Rain Rate (kg m^-2 s^-1)
    - 索引5: crr - Convective Rain Rate (kg m^-2 s^-1)

    Args:
        output_surface_physical: 反标准化的输出表面变量 (B, 26, time, lat, lon)
        target_surface_physical: 反标准化的目标表面变量 (B, 26, time, lat, lon)

    Returns:
        Tweedie降水损失
    """
    # 提取降水变量
    idx_lsrr = 4
    idx_crr = 5

    # 获取预测和目标（取第一个时间步）
    lsrr_pred = output_surface_physical[:, idx_lsrr, 0, :, :]
    lsrr_target = target_surface_physical[:, idx_lsrr, 0, :, :]

    crr_pred = output_surface_physical[:, idx_crr, 0, :, :]
    crr_target = target_surface_physical[:, idx_crr, 0, :, :]

    # 确保非负（降水不能为负）
    lsrr_pred = torch.relu(lsrr_pred)
    crr_pred = torch.relu(crr_pred)
    lsrr_target = torch.relu(lsrr_target)
    crr_target = torch.relu(crr_target)

    # 计算各自的Tweedie损失
    loss_lsrr = tweedie_deviance(lsrr_target, lsrr_pred, TWEEDIE_P_LSRR)
    loss_crr = tweedie_deviance(crr_target, crr_pred, TWEEDIE_P_CRR)

    return loss_lsrr + loss_crr


def calculate_tweedie_loss(output_surface_norm, target_surface_norm,
                           surface_mean, surface_std):
    """
    计算Tweedie降水损失的包装函数

    Args:
        output_surface_norm: 标准化的输出表面变量 (B, 26, time, lat, lon)
        target_surface_norm: 标准化的目标表面变量 (B, 26, time, lat, lon)
        surface_mean: 表面变量均值
        surface_std: 表面变量标准差

    Returns:
        Tweedie降水损失
    """
    # 反标准化获取物理值
    output_physical = output_surface_norm * surface_std + surface_mean
    target_physical = target_surface_norm * surface_std + surface_mean

    return calculate_tweedie_precipitation_loss(output_physical, target_physical)


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
num_epochs = 10
best_valid_loss = float('inf')

# 物理约束权重（根据损失量级动态调整）
# 目标：让每个物理约束贡献约1-10的损失量级
lambda_water = 8    # Water loss ~1e11 -> weight 1e-11 -> contribution ~1
lambda_energy = 2e-5    # Energy loss ~1e12 -> weight 1e-12 -> contribution ~1
lambda_pressure = 5e-8   # Pressure loss ~1e6 -> weight 1e-6 -> contribution ~1
lambda_temperature = 3e-2   # Temperature tendency loss weight
lambda_momentum = 1e1     # Navier-Stokes momentum loss weight
lambda_focus = 0.2        # Focus variable loss weight (使重要变量权重加倍)
lambda_tweedie = 2.0    # Tweedie precipitation loss weight (Hunt 2025)

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
    temperature_loss_total = 0.0
    momentum_loss_total = 0.0
    focus_loss_total = 0.0
    tweedie_loss_total = 0.0

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
        loss_focus = calculate_focus_variable_loss(
            output_surface, target_surface_norm,
            output_upper_air, target_upper_air_norm
        )
        loss_tweedie = calculate_tweedie_loss(
            output_surface, target_surface_norm,
            surface_mean, surface_std
        )

        # 总损失
        loss = loss_surface + loss_upper_air + \
               lambda_water * loss_water + \
               lambda_energy * loss_energy + \
               lambda_pressure * loss_pressure + \
               lambda_temperature * loss_temperature + \
               lambda_momentum * loss_momentum + \
               lambda_focus * loss_focus + \
               lambda_tweedie * loss_tweedie
        
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
        temperature_loss_total += loss_temperature.item()
        momentum_loss_total += loss_momentum.item()
        focus_loss_total += loss_focus.item()
        tweedie_loss_total += loss_tweedie.item()

        # 更新进度条
        train_pbar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "surf": f"{loss_surface.item():.4f}",
            "upper": f"{loss_upper_air.item():.4f}",
            "focus": f"{loss_focus.item():.4f}",
            "tweedie": f"{loss_tweedie.item():.2e}",
            "water": f"{loss_water.item():.2e}",
            "energy": f"{loss_energy.item():.2e}",
            "mom": f"{loss_momentum.item():.2e}"
        })
    
    # 计算平均训练损失
    train_loss = train_loss / len(train_loader)
    surface_loss = surface_loss / len(train_loader)
    upper_air_loss = upper_air_loss / len(train_loader)
    water_loss_total = water_loss_total / len(train_loader)
    energy_loss_total = energy_loss_total / len(train_loader)
    pressure_loss_total = pressure_loss_total / len(train_loader)
    temperature_loss_total = temperature_loss_total / len(train_loader)
    momentum_loss_total = momentum_loss_total / len(train_loader)
    focus_loss_total = focus_loss_total / len(train_loader)
    tweedie_loss_total = tweedie_loss_total / len(train_loader)

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
            loss_focus = calculate_focus_variable_loss(
                output_surface, target_surface_norm,
                output_upper_air, target_upper_air_norm
            )
            loss_tweedie = calculate_tweedie_loss(
                output_surface, target_surface_norm,
                surface_mean, surface_std
            )

            # 总损失
            loss = loss_surface + loss_upper_air + \
                   lambda_water * loss_water + \
                   lambda_energy * loss_energy + \
                   lambda_pressure * loss_pressure + \
                   lambda_temperature * loss_temperature + \
                   lambda_momentum * loss_momentum + \
                   lambda_focus * loss_focus + \
                   lambda_tweedie * loss_tweedie

            # 累加损失
            batch_loss = loss.item()
            valid_loss += batch_loss
            valid_surface_loss += loss_surface.item()
            valid_upper_air_loss += loss_upper_air.item()
            valid_water_loss += loss_water.item()
            valid_energy_loss += loss_energy.item()
            valid_pressure_loss += loss_pressure.item()
            valid_temperature_loss += loss_temperature.item()
            valid_momentum_loss += loss_momentum.item()
            valid_focus_loss += loss_focus.item()
            valid_tweedie_loss += loss_tweedie.item()

            # 更新进度条
            valid_pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "surf": f"{loss_surface.item():.4f}",
                "upper": f"{loss_upper_air.item():.4f}",
                "focus": f"{loss_focus.item():.4f}",
                "tweedie": f"{loss_tweedie.item():.2e}",
                "water": f"{loss_water.item():.2e}",
                "energy": f"{loss_energy.item():.2e}",
                "mom": f"{loss_momentum.item():.2e}"
            })
    
    # 计算平均验证损失
    valid_loss = valid_loss / len(valid_loader)
    valid_surface_loss = valid_surface_loss / len(valid_loader)
    valid_upper_air_loss = valid_upper_air_loss / len(valid_loader)
    valid_water_loss = valid_water_loss / len(valid_loader)
    valid_energy_loss = valid_energy_loss / len(valid_loader)
    valid_pressure_loss = valid_pressure_loss / len(valid_loader)
    valid_temperature_loss = valid_temperature_loss / len(valid_loader)
    valid_momentum_loss = valid_momentum_loss / len(valid_loader)
    valid_focus_loss = valid_focus_loss / len(valid_loader)
    valid_tweedie_loss = valid_tweedie_loss / len(valid_loader)

    # 打印损失
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train - Total: {train_loss:.6f}")
    print(f"         MSE - Surface: {surface_loss:.6f}, Upper Air: {upper_air_loss:.6f}")
    print(f"         Focus - Raw: {focus_loss_total:.6f}, Weighted: {lambda_focus*focus_loss_total:.6f}")
    print(f"         Tweedie (Hunt 2025) - Raw: {tweedie_loss_total:.2e}, Weighted: {lambda_tweedie*tweedie_loss_total:.6f}")
    print(f"         Physical Raw - Water: {water_loss_total:.2e}, Energy: {energy_loss_total:.2e}, Pressure: {pressure_loss_total:.2e}, Temp: {temperature_loss_total:.2e}, Mom: {momentum_loss_total:.2e}")
    print(f"         Physical Weighted - Water: {lambda_water*water_loss_total:.6f}, Energy: {lambda_energy*energy_loss_total:.6f}, Pressure: {lambda_pressure*pressure_loss_total:.6f}, Temp: {lambda_temperature*temperature_loss_total:.6f}, Mom: {lambda_momentum*momentum_loss_total:.6f}")
    print(f"  Valid - Total: {valid_loss:.6f}")
    print(f"         MSE - Surface: {valid_surface_loss:.6f}, Upper Air: {valid_upper_air_loss:.6f}")
    print(f"         Focus - Raw: {valid_focus_loss:.6f}, Weighted: {lambda_focus*valid_focus_loss:.6f}")
    print(f"         Tweedie (Hunt 2025) - Raw: {valid_tweedie_loss:.2e}, Weighted: {lambda_tweedie*valid_tweedie_loss:.6f}")
    print(f"         Physical Raw - Water: {valid_water_loss:.2e}, Energy: {valid_energy_loss:.2e}, Pressure: {valid_pressure_loss:.2e}, Temp: {valid_temperature_loss:.2e}, Mom: {valid_momentum_loss:.2e}")
    print(f"         Physical Weighted - Water: {lambda_water*valid_water_loss:.6f}, Energy: {lambda_energy*valid_energy_loss:.6f}, Pressure: {lambda_pressure*valid_pressure_loss:.6f}, Temp: {lambda_temperature*valid_temperature_loss:.6f}, Mom: {lambda_momentum*valid_momentum_loss:.6f}")