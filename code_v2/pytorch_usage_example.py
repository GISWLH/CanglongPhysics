# PyTorch使用示例
import torch
import numpy as np

# 加载标准化参数
surface_mean = torch.from_numpy(np.load("surface_mean.npy")).cuda()  # (17, 721, 1440)
surface_std = torch.from_numpy(np.load("surface_std.npy")).cuda()    # (17, 721, 1440)
upper_air_mean = torch.from_numpy(np.load("upper_air_mean.npy")).cuda()  # (7, 5, 721, 1440)
upper_air_std = torch.from_numpy(np.load("upper_air_std.npy")).cuda()    # (7, 5, 721, 1440)

# 标准化输入数据
def normalize_data(input_surface, input_upper_air):
    """
    Args:
        input_surface: (batch, 17, time_steps, 721, 1440)
        input_upper_air: (batch, 7, 5, time_steps, 721, 1440)
    Returns:
        normalized tensors
    """
    # 广播标准化
    normalized_surface = (input_surface - surface_mean.unsqueeze(0).unsqueeze(2)) / surface_std.unsqueeze(0).unsqueeze(2)
    normalized_upper_air = (input_upper_air - upper_air_mean.unsqueeze(0).unsqueeze(3)) / upper_air_std.unsqueeze(0).unsqueeze(3)
    
    return normalized_surface, normalized_upper_air

# 反标准化输出数据
def denormalize_data(output_surface, output_upper_air):
    """
    Args:
        output_surface: (batch, 17, 1, 721, 1440)
        output_upper_air: (batch, 7, 5, 1, 721, 1440)
    Returns:
        denormalized tensors
    """
    denorm_surface = output_surface * surface_std.unsqueeze(0).unsqueeze(2) + surface_mean.unsqueeze(0).unsqueeze(2)
    denorm_upper_air = output_upper_air * upper_air_std.unsqueeze(0).unsqueeze(3) + upper_air_mean.unsqueeze(0).unsqueeze(3)
    
    return denorm_surface, denorm_upper_air

# 变量名映射 (用于调试和可视化)
surf_vars = ['lsrr', 'crr', 'tciw', 'tcc', 'tsrc', 'u10', 'v10', 'd2m', 't2m', 'avg_tnlwrf', 'slhf', 'sshf', 'sp', 'swvl', 'msl', 'siconc', 'sst']
upper_vars = ['z', 'w', 'u', 'v', 'cc', 't', 'q']
levels = [200, 300, 500, 700, 850]

print("标准化数组已准备就绪!")
print(f"Surface shape: {surface_mean.shape}")
print(f"Upper air shape: {upper_air_mean.shape}")
