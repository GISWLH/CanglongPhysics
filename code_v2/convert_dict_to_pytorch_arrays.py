#!/usr/bin/env python3
"""
4.2 字典转数组：将JSON格式的mean/std转换为PyTorch可用的张量格式
读取ERA5_1940_2019_combined_mean_std.json，输出标准化张量

目标格式：
- surface_mean/std: (17, 721, 1440) -> 对应input_surface/output_surface
- upper_air_mean/std: (7, 5, 721, 1440) -> 对应input_upper_air/output_upper_air
可以直接进行广播运算: (x - mean) / std
"""

import json
import numpy as np
import os

# 变量定义 - 必须与模型通道顺序严格对应
surf_vars = ['lsrr','crr','tciw','tcc','tsrc','u10','v10','d2m',
             't2m','avg_tnlwrf','slhf','sshf','sp','swvl','msl','siconc','sst']
upper_vars = ['z','w','u','v','cc','t','q']
levels = [200, 300, 500, 700, 850]

def load_statistics_from_json(json_path, verbose=True):
    """加载JSON格式的统计数据"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")
    
    with open(json_path, 'r') as f:
        stats_data = json.load(f)
    
    if verbose:
        print("JSON数据加载成功")
    
    # 验证数据结构
    if 'surface' not in stats_data or 'upper_air' not in stats_data:
        raise ValueError("JSON数据缺少surface或upper_air部分")
    
    # 检查变量完整性
    missing_surface = [var for var in surf_vars if var not in stats_data['surface']]
    if missing_surface and verbose:
        print(f"缺少Surface变量: {missing_surface}")
    
    missing_upper = [var for var in upper_vars if var not in stats_data['upper_air']]
    if missing_upper and verbose:
        print(f"缺少Upper air变量: {missing_upper}")
    
    if verbose:
        print(f"Surface变量数: {len(stats_data['surface'])}")
        print(f"Upper air变量数: {len(stats_data['upper_air'])}")
    
    return stats_data

def create_surface_arrays(surface_data, verbose=True):
    """创建Surface变量的mean/std数组
    
    返回:
        surface_mean: (17, 721, 1440)
        surface_std: (17, 721, 1440)
    """
    
    if verbose:
        print("创建Surface数组...")
    
    # 初始化数组
    surface_mean = np.zeros((17, 721, 1440), dtype=np.float32)
    surface_std = np.ones((17, 721, 1440), dtype=np.float32)  # 默认std为1，避免除零
    
    # 按固定顺序填充数据
    for i, var_name in enumerate(surf_vars):
        if var_name in surface_data:
            mean_val = surface_data[var_name]['mean']
            std_val = surface_data[var_name]['std']
            
            # 广播到所有空间位置
            surface_mean[i, :, :] = mean_val
            surface_std[i, :, :] = max(std_val, 1e-8)  # 避免std为0
            
            if verbose:
                print(f"  {i:2d}. {var_name:15s}: mean={mean_val:12.8f}, std={std_val:12.8f}")
        else:
            if verbose:
                print(f"  {i:2d}. {var_name:15s}: 缺失数据，使用默认值")
    
    if verbose:
        print(f"Surface数组创建完成: {surface_mean.shape}")
    return surface_mean[None, :, None, :, :], surface_std[None, :, None, :, :]

def create_upper_air_arrays(upper_air_data, verbose=True):
    """创建Upper air变量的mean/std数组
    
    返回:
        upper_air_mean: (7, 5, 721, 1440)
        upper_air_std: (7, 5, 721, 1440)
    """
    
    if verbose:
        print("创建Upper air数组...")
    
    # 初始化数组
    upper_air_mean = np.zeros((7, 5, 721, 1440), dtype=np.float32)
    upper_air_std = np.ones((7, 5, 721, 1440), dtype=np.float32)  # 默认std为1
    
    # 按固定顺序填充数据
    for i, var_name in enumerate(upper_vars):
        if verbose:
            print(f"  处理变量 {i}. {var_name}")
        
        if var_name in upper_air_data:
            for j, level in enumerate(levels):
                level_str = str(level)
                
                if level_str in upper_air_data[var_name]:
                    mean_val = upper_air_data[var_name][level_str]['mean']
                    std_val = upper_air_data[var_name][level_str]['std']
                    
                    # 广播到所有空间位置
                    upper_air_mean[i, j, :, :] = mean_val
                    upper_air_std[i, j, :, :] = max(std_val, 1e-8)  # 避免std为0
                    
                    if verbose:
                        print(f"    {level:3d}hPa: mean={mean_val:12.6f}, std={std_val:12.6f}")
                else:
                    if verbose:
                        print(f"    {level:3d}hPa: 缺失数据")
        else:
            if verbose:
                print(f"  变量 {var_name} 缺失数据")
    
    if verbose:
        print(f"Upper air数组创建完成: {upper_air_mean.shape}")
    return upper_air_mean[None, :, :, None, :, :], upper_air_std[None, :, :, None, :, :]

def save_pytorch_arrays(surface_mean, surface_std, upper_air_mean, upper_air_std, output_dir):
    """保存为PyTorch可用的格式"""
    
    print("保存PyTorch格式数组...")
    
    # 保存为npz格式
    npz_path = os.path.join(output_dir, "ERA5_1940_2019_normalization_arrays.npz")
    np.savez_compressed(
        npz_path,
        surface_mean=surface_mean,
        surface_std=surface_std,
        upper_air_mean=upper_air_mean,
        upper_air_std=upper_air_std
    )
    print(f"NPZ文件已保存: {npz_path}")
    
    # 保存为单独的npy文件（PyTorch加载更方便）
    np.save(os.path.join(output_dir, "surface_mean.npy"), surface_mean)
    np.save(os.path.join(output_dir, "surface_std.npy"), surface_std)
    np.save(os.path.join(output_dir, "upper_air_mean.npy"), upper_air_mean)
    np.save(os.path.join(output_dir, "upper_air_std.npy"), upper_air_std)
    print("单独npy文件已保存")
    
    # 创建使用示例代码
    example_code = f'''# PyTorch使用示例
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
surf_vars = {surf_vars}
upper_vars = {upper_vars}
levels = {levels}

print("标准化数组已准备就绪!")
print(f"Surface shape: {{surface_mean.shape}}")
print(f"Upper air shape: {{upper_air_mean.shape}}")
'''
    
    example_path = os.path.join(output_dir, "pytorch_usage_example.py")
    with open(example_path, 'w') as f:
        f.write(example_code)
    print(f"使用示例已保存: {example_path}")
    
    return npz_path

def validate_arrays(surface_mean, surface_std, upper_air_mean, upper_air_std):
    """验证生成的数组"""
    
    print("验证生成的数组...")
    
    # 检查形状
    assert surface_mean.shape == (17, 721, 1440), f"Surface mean形状错误: {surface_mean.shape}"
    assert surface_std.shape == (17, 721, 1440), f"Surface std形状错误: {surface_std.shape}"
    assert upper_air_mean.shape == (7, 5, 721, 1440), f"Upper air mean形状错误: {upper_air_mean.shape}"
    assert upper_air_std.shape == (7, 5, 721, 1440), f"Upper air std形状错误: {upper_air_std.shape}"
    
    # 检查数值范围
    assert not np.any(np.isnan(surface_mean)), "Surface mean包含NaN"
    assert not np.any(np.isnan(surface_std)), "Surface std包含NaN"
    assert not np.any(np.isnan(upper_air_mean)), "Upper air mean包含NaN"
    assert not np.any(np.isnan(upper_air_std)), "Upper air std包含NaN"
    
    assert np.all(surface_std > 0), "Surface std包含非正值"
    assert np.all(upper_air_std > 0), "Upper air std包含非正值"
    
    # 显示统计信息
    print("数组验证通过!")
    print(f"Surface mean范围: [{surface_mean.min():.6f}, {surface_mean.max():.6f}]")
    print(f"Surface std范围: [{surface_std.min():.6f}, {surface_std.max():.6f}]")
    print(f"Upper air mean范围: [{upper_air_mean.min():.6f}, {upper_air_mean.max():.6f}]")
    print(f"Upper air std范围: [{upper_air_std.min():.6f}, {upper_air_std.max():.6f}]")

def load_normalization_arrays(json_path, verbose=False):
    """
    从JSON文件加载标准化参数数组
    
    Args:
        json_path: JSON文件路径
        verbose: 是否打印详细信息，默认False（静默加载）
    
    Returns:
        [surface_mean, surface_std, upper_mean, upper_std]: 四个numpy数组
        - surface_mean: (17, 721, 1440)
        - surface_std: (17, 721, 1440) 
        - upper_mean: (7, 5, 721, 1440)
        - upper_std: (7, 5, 721, 1440)
    """
    try:
        # 1. 加载JSON数据
        stats_data = load_statistics_from_json(json_path, verbose=verbose)
        
        # 2. 创建Surface数组
        surface_mean, surface_std = create_surface_arrays(stats_data['surface'], verbose=verbose)
        
        # 3. 创建Upper air数组
        upper_mean, upper_std = create_upper_air_arrays(stats_data['upper_air'], verbose=verbose)
        
        return [surface_mean, surface_std, upper_mean, upper_std]
        
    except Exception as e:
        print(f"加载标准化数组时出错: {str(e)}")
        raise e

def main():
    """主函数"""
    
    print("开始4.2字典转数组工作")
    
    # 输入输出路径
    input_json = "/home/CanglongPhysics/code_v2/ERA5_1940_2019_combined_mean_std.json"
    output_dir = "/home/CanglongPhysics/code_v2"
    
    try:
        # 使用新的函数接口（详细模式）
        surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(input_json, verbose=True)
        
        # 保存结果
        npz_path = save_pytorch_arrays(surface_mean, surface_std, upper_mean, upper_std, output_dir)

        
        return True
        
    except Exception as e:
        print(f"转换过程中出错: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ 字典转数组成功完成!")
    else:
        print("\n❌ 转换失败，请检查日志")