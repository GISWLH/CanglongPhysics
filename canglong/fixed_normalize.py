import sys
sys.path.append('/home/CanglongPhysics/code_v2')
from convert_dict_to_pytorch_arrays import load_normalization_arrays
import torch
import numpy as np

# 调用函数获取四个数组
json = '/home/CanglongPhysics/code_v2/ERA5_1940_2019_combined_mean_std.json'
surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(json)

# 转换为PyTorch tensor并移到GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
surface_mean = torch.from_numpy(surface_mean).to(device)  # (17, 721, 1440)
surface_std = torch.from_numpy(surface_std).to(device)    # (17, 721, 1440)
upper_mean = torch.from_numpy(upper_mean).to(device)      # (7, 5, 721, 1440)
upper_std = torch.from_numpy(upper_std).to(device)        # (7, 5, 721, 1440)

print("Normalization arrays loaded:")
print(f"surface_mean shape: {surface_mean.shape}")
print(f"surface_std shape: {surface_std.shape}")
print(f"upper_mean shape: {upper_mean.shape}")
print(f"upper_std shape: {upper_std.shape}")

# 创建模拟数据
input_surface = torch.randn(1, 17, 2, 721, 1440).to(device)
input_upper_air = torch.randn(1, 7, 5, 2, 721, 1440).to(device)

print("\nInput shapes:")
print(f"input_surface: {input_surface.shape}")
print(f"input_upper_air: {input_upper_air.shape}")

# 标准化输入数据 - 正确的广播维度
# input_surface: (1, 17, 2, 721, 1440)
# surface_mean: (17, 721, 1440) -> 需要扩展维度匹配
surface_mean_expanded = surface_mean.unsqueeze(0).unsqueeze(2)  # (1, 17, 1, 721, 1440)
surface_std_expanded = surface_std.unsqueeze(0).unsqueeze(2)    # (1, 17, 1, 721, 1440)

# input_upper_air: (1, 7, 5, 2, 721, 1440)
# upper_mean: (7, 5, 721, 1440) -> 需要扩展维度匹配
upper_mean_expanded = upper_mean.unsqueeze(0).unsqueeze(3)      # (1, 7, 5, 1, 721, 1440)
upper_std_expanded = upper_std.unsqueeze(0).unsqueeze(3)        # (1, 7, 5, 1, 721, 1440)

print("\nExpanded shapes for broadcasting:")
print(f"surface_mean_expanded: {surface_mean_expanded.shape}")
print(f"surface_std_expanded: {surface_std_expanded.shape}")
print(f"upper_mean_expanded: {upper_mean_expanded.shape}")
print(f"upper_std_expanded: {upper_std_expanded.shape}")

# 标准化
normalized_surface = (input_surface - surface_mean_expanded) / surface_std_expanded
normalized_upper_air = (input_upper_air - upper_mean_expanded) / upper_std_expanded

print("\nNormalized shapes:")
print(f"normalized_surface: {normalized_surface.shape}")
print(f"normalized_upper_air: {normalized_upper_air.shape}")

# 模拟模型输出 (time维度从2变为1)
output_surface = torch.randn(1, 17, 1, 721, 1440).to(device)
output_upper_air = torch.randn(1, 7, 5, 1, 721, 1440).to(device)

print("\nOutput shapes:")
print(f"output_surface: {output_surface.shape}")
print(f"output_upper_air: {output_upper_air.shape}")

# 反标准化输出 - 输出的time维度是1，所以标准化参数也要匹配
denormalized_surface = output_surface * surface_std_expanded + surface_mean_expanded
denormalized_upper_air = output_upper_air * upper_std_expanded + upper_mean_expanded

print("\nDenormalized shapes:")
print(f"denormalized_surface: {denormalized_surface.shape}")
print(f"denormalized_upper_air: {denormalized_upper_air.shape}")

print("\n✅ All operations successful! Broadcasting works correctly.")