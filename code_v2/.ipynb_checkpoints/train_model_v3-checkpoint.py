import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import h5py as h5
import sys
sys.path.append('..')
sys.path.append('.')
from model_v3 import CanglongV3
from convert_dict_to_pytorch_arrays import load_normalization_arrays


def calculate_water_balance_loss(input_surface_normalized, output_surface_normalized, 
                                surface_mean, surface_std, delta_t=7*24*3600):
    """
    计算水量平衡损失
    水量平衡: ΔSoil_water = P_total - E
    
    Args:
        input_surface_normalized: 标准化的输入表面变量 (B, 17, time, lat, lon)
        output_surface_normalized: 标准化的输出表面变量 (B, 17, time, lat, lon)
        surface_mean: 表面变量均值 (17, lat, lon)
        surface_std: 表面变量标准差 (17, lat, lon)
        delta_t: 时间步长（秒）
    """
    # 反标准化到物理单位
    surface_mean = surface_mean.unsqueeze(0).unsqueeze(2)  # (1, 17, 1, lat, lon)
    surface_std = surface_std.unsqueeze(0).unsqueeze(2)    # (1, 17, 1, lat, lon)
    
    input_physical = input_surface_normalized * surface_std + surface_mean
    output_physical = output_surface_normalized * surface_std + surface_mean
    
    # 变量索引（基于CLAUDE.md）:
    # 0: large_scale_rain_rate, 1: convective_rain_rate
    # 10: surface_latent_heat_flux
    # 13: volumetric_soil_water_layer
    
    # 土壤水变化量
    delta_soil_water = output_physical[:, 13, 0, :, :] - input_physical[:, 13, -1, :, :]
    
    # 总降水量 - 从 kg m^-2 s^-1 转换为总量
    large_scale_rain = output_physical[:, 0, 0, :, :]
    convective_rain = output_physical[:, 1, 0, :, :]
    p_total = (large_scale_rain + convective_rain) * delta_t
    
    # 蒸发量 - 从 J m^-2 转换为水量
    L_v = 2.5e6  # 汽化潜热 (J/kg)
    latent_heat_flux = output_physical[:, 10, 0, :, :]
    evaporation = latent_heat_flux / L_v * delta_t
    
    # 水量平衡残差
    residual_water = delta_soil_water - (p_total - evaporation)
    
    return torch.nn.functional.mse_loss(residual_water, torch.zeros_like(residual_water))


def calculate_energy_balance_loss(output_surface_normalized, surface_mean, surface_std):
    """
    计算能量平衡损失
    能量平衡: SW_net - LW_net = SHF + LHF
    
    Args:
        output_surface_normalized: 标准化的输出表面变量 (B, 17, time, lat, lon)
        surface_mean: 表面变量均值 (17, lat, lon)
        surface_std: 表面变量标准差 (17, lat, lon)
    """
    # 反标准化到物理单位
    surface_mean = surface_mean.unsqueeze(0).unsqueeze(2)  # (1, 17, 1, lat, lon)
    surface_std = surface_std.unsqueeze(0).unsqueeze(2)    # (1, 17, 1, lat, lon)
    output_physical = output_surface_normalized * surface_std + surface_mean
    
    # 变量索引（基于CLAUDE.md）:
    # 4: top_net_solar_radiation_clear_sky (SW_net)
    # 9: mean_top_net_long_wave_radiation_flux (LW_net)
    # 10: surface_latent_heat_flux (LHF)
    # 11: surface_sensible_heat_flux (SHF)
    
    sw_net = output_physical[:, 4, 0, :, :]  # 净太阳辐射 (J m^-2)
    lw_net = output_physical[:, 9, 0, :, :]  # 净长波辐射 (W m^-2)
    shf = output_physical[:, 11, 0, :, :]    # 感热通量 (J m^-2)
    lhf = output_physical[:, 10, 0, :, :]    # 潜热通量 (J m^-2)
    
    # 能量平衡残差
    residual_energy = (sw_net - lw_net) - (shf + lhf)
    
    return torch.nn.functional.mse_loss(residual_energy, torch.zeros_like(residual_energy))


def calculate_hydrostatic_balance_loss(output_upper_normalized, upper_mean, upper_std):
    """
    计算静力平衡损失
    静力平衡: Δφ = R_d * T_avg * ln(p1/p2)
    
    Args:
        output_upper_normalized: 标准化的输出高空变量 (B, 7, levels, time, lat, lon)
        upper_mean: 高空变量均值 (7, levels, lat, lon)
        upper_std: 高空变量标准差 (7, levels, lat, lon)
    """
    # 反标准化到物理单位
    upper_mean = upper_mean.unsqueeze(0).unsqueeze(3)  # (1, 7, levels, 1, lat, lon)
    upper_std = upper_std.unsqueeze(0).unsqueeze(3)    # (1, 7, levels, 1, lat, lon)
    output_physical = output_upper_normalized * upper_std + upper_mean
    
    # 变量索引（基于CLAUDE.md）:
    # 0: Geopotential (φ)
    # 5: Temperature (T)
    # 压力层: 200, 300, 500, 700, 850 hPa (索引 0-4)
    
    # 计算 850 hPa (索引 4) 和 700 hPa (索引 3) 之间的静力平衡
    phi_850 = output_physical[:, 0, 4, 0, :, :]  # 850 hPa 位势 (m^2 s^-2)
    phi_700 = output_physical[:, 0, 3, 0, :, :]  # 700 hPa 位势
    temp_850 = output_physical[:, 5, 4, 0, :, :] # 850 hPa 温度 (K)
    temp_700 = output_physical[:, 5, 3, 0, :, :] # 700 hPa 温度
    
    # 模型预测的位势厚度
    delta_phi_model = phi_700 - phi_850
    
    # 物理计算的位势厚度
    R_d = 287  # 干空气气体常数 (J/(kg·K))
    temp_avg = (temp_700 + temp_850) / 2
    delta_phi_physical = R_d * temp_avg * torch.log(torch.tensor(850.0/700.0, device=temp_avg.device))
    
    # 静力平衡残差
    residual_hydrostatic = delta_phi_model - delta_phi_physical
    
    return torch.nn.functional.mse_loss(residual_hydrostatic, torch.zeros_like(residual_hydrostatic))


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

# 加载数据
print("Loading data...")
h5_file = h5.File('/gz-data/ERA5_2023_weekly.h5', 'r')
input_surface = h5_file['surface'][:]  # (52, 17, 721, 1440)
input_upper_air = h5_file['upper_air'][:]  # (52, 7, 5, 721, 1440)
h5_file.close()

print(f"Surface data shape: {input_surface.shape}")
print(f"Upper air data shape: {input_upper_air.shape}")

# 加载标准化参数
print("Loading normalization parameters...")
json_path = '/home/CanglongPhysics/code_v2/ERA5_1940_2019_combined_mean_std.json'
surface_mean_np, surface_std_np, upper_mean_np, upper_std_np = load_normalization_arrays(json_path)

# 移除额外维度并转换为张量
surface_mean_np = surface_mean_np.squeeze(0).squeeze(1)  # (17, 721, 1440)
surface_std_np = surface_std_np.squeeze(0).squeeze(1)
upper_mean_np = upper_mean_np.squeeze(0).squeeze(2)  # (7, 5, 721, 1440)
upper_std_np = upper_std_np.squeeze(0).squeeze(2)

surface_mean = torch.from_numpy(surface_mean_np).float().to(device)
surface_std = torch.from_numpy(surface_std_np).float().to(device)
upper_mean = torch.from_numpy(upper_mean_np).float().to(device)
upper_std = torch.from_numpy(upper_std_np).float().to(device)

print(f"Surface mean shape: {surface_mean.shape}")
print(f"Surface std shape: {surface_std.shape}")
print(f"Upper mean shape: {upper_mean.shape}")
print(f"Upper std shape: {upper_std.shape}")

# 计算数据集划分点 - 按照6:2:2的时间序列划分
total_samples = 52
train_end = 30
valid_end = 40

# 创建数据集
train_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=0, end_idx=train_end)
valid_dataset = WeatherDataset(input_surface, input_upper_air, start_idx=train_end, end_idx=valid_end)
batch_size = 1  # 小batch size便于调试
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
print(f"Created data loaders with batch size {batch_size}")

# 创建模型
print("Creating model...")
# 根据观察到的损失量级调整权重：
# Water loss ~ 1.42e+11, 需要权重 ~ 1e-11 来平衡到 ~1
# Energy loss ~ 2.54e+12, 需要权重 ~ 1e-12 来平衡到 ~1  
# Pressure loss ~ 1.76e+6, 需要权重 ~ 1e-6 来平衡到 ~1
model = CanglongV3(
    surface_mean=surface_mean,
    surface_std=surface_std,
    upper_mean=upper_mean,
    upper_std=upper_std,
    lambda_water=1e-11,      # 从0.01降到1e-11
    lambda_energy=1e-12,     # 从0.001降到1e-12
    lambda_pressure=1e-6     # 从0.0001降到1e-6
)

# 多GPU训练
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

# 将模型移动到设备
model.to(device)

# 创建优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

# 创建保存目录
save_dir = 'checkpoints_v3'
os.makedirs(save_dir, exist_ok=True)

# 训练参数
num_epochs = 2
best_valid_loss = float('inf')

# 物理约束权重（根据损失量级动态调整）
# 目标：让每个物理约束贡献约1-10的损失量级
lambda_water = 1e-11     # Water loss ~1e11 -> weight 1e-11 -> contribution ~1
lambda_energy = 1e-12    # Energy loss ~1e12 -> weight 1e-12 -> contribution ~1  
lambda_pressure = 1e-6   # Pressure loss ~1e6 -> weight 1e-6 -> contribution ~1

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
        # 将数据移动到设备
        input_surface = input_surface.float().to(device)
        input_upper_air = input_upper_air.float().to(device)
        target_surface = target_surface.float().to(device)
        target_upper_air = target_upper_air.float().to(device)
        
        # 标准化输入数据
        input_surface_norm = (input_surface - surface_mean.unsqueeze(0).unsqueeze(2)) / surface_std.unsqueeze(0).unsqueeze(2)
        input_upper_air_norm = (input_upper_air - upper_mean.unsqueeze(0).unsqueeze(3)) / upper_std.unsqueeze(0).unsqueeze(3)
        target_surface_norm = (target_surface - surface_mean.unsqueeze(0).unsqueeze(2)) / surface_std.unsqueeze(0).unsqueeze(2)
        target_upper_air_norm = (target_upper_air - upper_mean.unsqueeze(0).unsqueeze(3)) / upper_std.unsqueeze(0).unsqueeze(3)
        
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
            output_surface, surface_mean, surface_std
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
            input_surface_norm = (input_surface - surface_mean.unsqueeze(0).unsqueeze(2)) / surface_std.unsqueeze(0).unsqueeze(2)
            input_upper_air_norm = (input_upper_air - upper_mean.unsqueeze(0).unsqueeze(3)) / upper_std.unsqueeze(0).unsqueeze(3)
            target_surface_norm = (target_surface - surface_mean.unsqueeze(0).unsqueeze(2)) / surface_std.unsqueeze(0).unsqueeze(2)
            target_upper_air_norm = (target_upper_air - upper_mean.unsqueeze(0).unsqueeze(3)) / upper_std.unsqueeze(0).unsqueeze(3)
            
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
                output_surface, surface_mean, surface_std
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
    
    # 记录最佳模型（但不保存）
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        print(f"  → New best validation loss: {best_valid_loss:.6f}")
        # 不保存模型文件
        # save_path = os.path.join(save_dir, 'best_model_v3.pth')
        # torch.save(model.state_dict(), save_path)

print("\nTraining completed!")