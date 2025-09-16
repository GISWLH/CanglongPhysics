import sys
sys.path.append('/home/CanglongPhysics/code_v2')
from convert_dict_to_pytorch_arrays import load_normalization_arrays
import torch
import numpy as np

# 调用函数获取四个数组 (这些形状是正确的)
surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays('/home/CanglongPhysics/code_v2/ERA5_1940_2019_combined_mean_std.json')

print("Original numpy arrays:")
print(f"surface_mean: {surface_mean.shape}")  # (17, 721, 1440)
print(f"surface_std: {surface_std.shape}")    # (17, 721, 1440)  
print(f"upper_mean: {upper_mean.shape}")      # (7, 5, 721, 1440)
print(f"upper_std: {upper_std.shape}")        # (7, 5, 721, 1440)

# 转换为PyTorch tensor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
surface_mean = torch.from_numpy(surface_mean).to(device)
surface_std = torch.from_numpy(surface_std).to(device)
upper_mean = torch.from_numpy(upper_mean).to(device)  
upper_std = torch.from_numpy(upper_std).to(device)

# 在使用时进行正确的广播
def normalize_inputs(input_surface, input_upper_air):
    """
    正确的标准化方法
    input_surface: (batch, 17, time, 721, 1440)
    input_upper_air: (batch, 7, 5, time, 721, 1440)
    """
    # 扩展维度用于广播
    surface_mean_broadcast = surface_mean.unsqueeze(0).unsqueeze(2)  # (1, 17, 1, 721, 1440)
    surface_std_broadcast = surface_std.unsqueeze(0).unsqueeze(2)    # (1, 17, 1, 721, 1440)
    
    upper_mean_broadcast = upper_mean.unsqueeze(0).unsqueeze(3)      # (1, 7, 5, 1, 721, 1440)  
    upper_std_broadcast = upper_std.unsqueeze(0).unsqueeze(3)        # (1, 7, 5, 1, 721, 1440)
    
    # 标准化
    norm_surface = (input_surface - surface_mean_broadcast) / surface_std_broadcast
    norm_upper_air = (input_upper_air - upper_mean_broadcast) / upper_std_broadcast
    
    return norm_surface, norm_upper_air

def denormalize_outputs(output_surface, output_upper_air):
    """
    正确的反标准化方法
    output_surface: (batch, 17, 1, 721, 1440)
    output_upper_air: (batch, 7, 5, 1, 721, 1440)
    """
    # 扩展维度用于广播
    surface_mean_broadcast = surface_mean.unsqueeze(0).unsqueeze(2)  # (1, 17, 1, 721, 1440)
    surface_std_broadcast = surface_std.unsqueeze(0).unsqueeze(2)    # (1, 17, 1, 721, 1440)
    
    upper_mean_broadcast = upper_mean.unsqueeze(0).unsqueeze(3)      # (1, 7, 5, 1, 721, 1440)
    upper_std_broadcast = upper_std.unsqueeze(0).unsqueeze(3)        # (1, 7, 5, 1, 721, 1440)
    
    # 反标准化
    denorm_surface = output_surface * surface_std_broadcast + surface_mean_broadcast
    denorm_upper_air = output_upper_air * upper_std_broadcast + upper_mean_broadcast
    
    return denorm_surface, denorm_upper_air

# 测试
print("\n测试标准化和反标准化:")
input_surface = torch.randn(1, 17, 2, 721, 1440).to(device)
input_upper_air = torch.randn(1, 7, 5, 2, 721, 1440).to(device)

norm_surface, norm_upper_air = normalize_inputs(input_surface, input_upper_air)
print(f"Normalized surface: {norm_surface.shape}")
print(f"Normalized upper_air: {norm_upper_air.shape}")

# 模拟模型输出  
output_surface = torch.randn(1, 17, 1, 721, 1440).to(device)
output_upper_air = torch.randn(1, 7, 5, 1, 721, 1440).to(device)

denorm_surface, denorm_upper_air = denormalize_outputs(output_surface, output_upper_air)
print(f"Denormalized surface: {denorm_surface.shape}")
print(f"Denormalized upper_air: {denorm_upper_air.shape}")

print("\n✅ 标准化和反标准化成功!")