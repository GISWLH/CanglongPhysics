#!/usr/bin/env python3
"""
字典转数组：将JSON格式的mean/std转换为PyTorch可用的张量格式
读取ERA5_1940_2023_mean_std_statistics.json，输出标准化张量

目标格式：
- surface_mean/std: (26, 721, 1440) -> 对应input_surface/output_surface
- upper_air_mean/std: (10, 5, 721, 1440) -> 对应input_upper_air/output_upper_air
可以直接进行广播运算: (x - mean) / std
"""

import json
import numpy as np
import os
# 变量定义 - 必须与模型通道顺序严格对应，按CLAUDE.md文档顺序
surf_vars = ['avg_tnswrf', 'avg_tnlwrf', 'tciw', 'tcc', 'lsrr', 'crr', 'blh',
             'u10', 'v10', 'd2m', 't2m', 'avg_iews', 'avg_inss', 'slhf', 'sshf',
             'avg_snswrf', 'avg_snlwrf', 'ssr', 'str', 'sp', 'msl', 'siconc',
             'sst', 'ro', 'stl', 'swvl']

upper_vars = ['o3', 'z', 't', 'u', 'v', 'w', 'q', 'cc', 'ciwc', 'clwc']

# nc文件中是[850, 700, 500, 300, 200]，h5中是[200, 300, 500, 700, 850]
levels = [200, 300, 500, 700, 850]

def load_statistics_from_json(json_path, verbose=True):
    """
    加载JSON格式的统计数据
    注意：新格式是平铺结构，所有变量在根级别
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"文件不存在: {json_path}")

    with open(json_path, 'r') as f:
        stats_data = json.load(f)

    if verbose:
        print("JSON数据加载成功")
        print(f"总变量数: {len(stats_data)}")

    # 新格式是平铺的，分析变量类型
    surface_vars_found = []
    upper_vars_found = []

    for key in stats_data.keys():
        if isinstance(stats_data[key], dict):
            # 检查是否有压力层
            if '200' in stats_data[key] or '850' in stats_data[key]:
                upper_vars_found.append(key)
            else:
                surface_vars_found.append(key)

    if verbose:
        print(f"Surface变量数: {len(surface_vars_found)}")
        print(f"Upper air变量数: {len(upper_vars_found)}")

        # 检查缺失变量
        missing_surface = [var for var in surf_vars if var not in surface_vars_found]
        if missing_surface:
            print(f"⚠️  缺少Surface变量: {missing_surface}")

        missing_upper = [var for var in upper_vars if var not in upper_vars_found]
        if missing_upper:
            print(f"⚠️  缺少Upper air变量: {missing_upper}")

    return stats_data

def create_surface_arrays(stats_data, verbose=True):
    """
    创建Surface变量的mean/std数组

    Args:
        stats_data: 平铺格式的JSON数据

    Returns:
        surface_mean: (26, 721, 1440)
        surface_std: (26, 721, 1440)
    """

    if verbose:
        print("\n创建Surface数组...")

    # 初始化数组
    surface_mean = np.zeros((26, 721, 1440), dtype=np.float32)
    surface_std = np.ones((26, 721, 1440), dtype=np.float32)  # 默认std为1，避免除零

    # 按固定顺序填充数据
    for i, var_name in enumerate(surf_vars):
        if var_name in stats_data:
            mean_val = stats_data[var_name]['mean']
            std_val = stats_data[var_name]['std']

            # 广播到所有空间位置
            surface_mean[i, :, :] = mean_val
            surface_std[i, :, :] = max(std_val, 1e-8)  # 避免std为0

            if verbose:
                print(f"  {i+1:2d}. {var_name:15s}: mean={mean_val:14.6f}, std={std_val:14.6f}")
        else:
            if verbose:
                print(f"  {i+1:2d}. {var_name:15s}: ⚠️  缺失数据，使用默认值")

    if verbose:
        print(f"✓ Surface数组创建完成: {surface_mean.shape}")

    return surface_mean, surface_std

def create_upper_air_arrays(stats_data, verbose=True):
    """
    创建Upper air变量的mean/std数组

    Args:
        stats_data: 平铺格式的JSON数据

    Returns:
        upper_air_mean: (10, 5, 721, 1440)
        upper_air_std: (10, 5, 721, 1440)
    """

    if verbose:
        print("\n创建Upper air数组...")

    # 初始化数组
    upper_air_mean = np.zeros((10, 5, 721, 1440), dtype=np.float32)
    upper_air_std = np.ones((10, 5, 721, 1440), dtype=np.float32)  # 默认std为1

    # 按固定顺序填充数据
    for i, var_name in enumerate(upper_vars):
        if verbose:
            print(f"  {i+1}. {var_name}:")

        if var_name in stats_data:
            for j, level in enumerate(levels):
                level_str = str(level)

                if level_str in stats_data[var_name]:
                    mean_val = stats_data[var_name][level_str]['mean']
                    std_val = stats_data[var_name][level_str]['std']

                    # 广播到所有空间位置
                    upper_air_mean[i, j, :, :] = mean_val
                    upper_air_std[i, j, :, :] = max(std_val, 1e-8)  # 避免std为0

                    if verbose:
                        print(f"     {level:4d}hPa: mean={mean_val:14.6e}, std={std_val:14.6e}")
                else:
                    if verbose:
                        print(f"     {level:4d}hPa: ⚠️  缺失数据")
        else:
            if verbose:
                print(f"     ⚠️  变量 {var_name} 缺失数据")

    if verbose:
        print(f"✓ Upper air数组创建完成: {upper_air_mean.shape}")

    return upper_air_mean, upper_air_std

def save_pytorch_arrays(surface_mean, surface_std, upper_air_mean, upper_air_std, output_dir):
    """保存为PyTorch可用的格式"""

    print("\n保存PyTorch格式数组...")

    os.makedirs(output_dir, exist_ok=True)

    # 保存为npz格式
    npz_path = os.path.join(output_dir, "ERA5_1940_2023_normalization_arrays.npz")
    np.savez_compressed(
        npz_path,
        surface_mean=surface_mean,
        surface_std=surface_std,
        upper_air_mean=upper_air_mean,
        upper_air_std=upper_air_std
    )
    print(f"✓ NPZ文件已保存: {npz_path}")
    file_size = os.path.getsize(npz_path) / (1024**2)
    print(f"  文件大小: {file_size:.2f} MB")

    # 保存为单独的npy文件（PyTorch加载更方便）
    np.save(os.path.join(output_dir, "surface_mean.npy"), surface_mean)
    np.save(os.path.join(output_dir, "surface_std.npy"), surface_std)
    np.save(os.path.join(output_dir, "upper_air_mean.npy"), upper_air_mean)
    np.save(os.path.join(output_dir, "upper_air_std.npy"), upper_air_std)
    print("✓ 单独npy文件已保存")

    # 创建使用示例代码
    example_code = f'''# PyTorch使用示例 - ERA5 1940-2023 标准化
import torch
import numpy as np

# 加载标准化参数
surface_mean = torch.from_numpy(np.load("surface_mean.npy")).cuda()  # (26, 721, 1440)
surface_std = torch.from_numpy(np.load("surface_std.npy")).cuda()    # (26, 721, 1440)
upper_air_mean = torch.from_numpy(np.load("upper_air_mean.npy")).cuda()  # (10, 5, 721, 1440)
upper_air_std = torch.from_numpy(np.load("upper_air_std.npy")).cuda()    # (10, 5, 721, 1440)

# 标准化输入数据
def normalize_data(input_surface, input_upper_air):
    """
    Args:
        input_surface: (batch, 26, time_steps, 721, 1440)
        input_upper_air: (batch, 10, 5, time_steps, 721, 1440)
    Returns:
        normalized tensors
    """
    # 广播标准化
    # surface_mean: (26, 721, 1440) -> (1, 26, 1, 721, 1440)
    normalized_surface = (input_surface - surface_mean.unsqueeze(0).unsqueeze(2)) / \\
                         surface_std.unsqueeze(0).unsqueeze(2)

    # upper_air_mean: (10, 5, 721, 1440) -> (1, 10, 5, 1, 721, 1440)
    normalized_upper_air = (input_upper_air - upper_air_mean.unsqueeze(0).unsqueeze(3)) / \\
                           upper_air_std.unsqueeze(0).unsqueeze(3)

    return normalized_surface, normalized_upper_air

# 反标准化输出数据
def denormalize_data(output_surface, output_upper_air):
    """
    Args:
        output_surface: (batch, 26, 1, 721, 1440)
        output_upper_air: (batch, 10, 5, 1, 721, 1440)
    Returns:
        denormalized tensors
    """
    denorm_surface = output_surface * surface_std.unsqueeze(0).unsqueeze(2) + \\
                     surface_mean.unsqueeze(0).unsqueeze(2)

    denorm_upper_air = output_upper_air * upper_air_std.unsqueeze(0).unsqueeze(3) + \\
                       upper_air_mean.unsqueeze(0).unsqueeze(3)

    return denorm_surface, denorm_upper_air

# 变量名映射 (用于调试和可视化)
surf_vars = {surf_vars}
upper_vars = {upper_vars}
levels = {levels}  # [200, 300, 500, 700, 850] hPa (从高到低)

print("标准化数组已准备就绪!")
print(f"Surface shape: {{surface_mean.shape}}")
print(f"Upper air shape: {{upper_air_mean.shape}}")

# 使用示例
if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 1
    input_surface = torch.randn(batch_size, 26, 2, 721, 1440).cuda()  # 2个时间步
    input_upper_air = torch.randn(batch_size, 10, 5, 2, 721, 1440).cuda()

    # 标准化
    norm_surface, norm_upper = normalize_data(input_surface, input_upper_air)
    print(f"\\n标准化后:")
    print(f"  Surface: {{norm_surface.shape}}")
    print(f"  Upper air: {{norm_upper.shape}}")

    # 模拟模型输出
    output_surface = torch.randn(batch_size, 26, 1, 721, 1440).cuda()
    output_upper_air = torch.randn(batch_size, 10, 5, 1, 721, 1440).cuda()

    # 反标准化
    denorm_surface, denorm_upper = denormalize_data(output_surface, output_upper_air)
    print(f"\\n反标准化后:")
    print(f"  Surface: {{denorm_surface.shape}}")
    print(f"  Upper air: {{denorm_upper.shape}}")
'''

    example_path = os.path.join(output_dir, "pytorch_usage_example_2023.py")
    with open(example_path, 'w') as f:
        f.write(example_code)
    print(f"✓ 使用示例已保存: {example_path}")

    return npz_path

def validate_arrays(surface_mean, surface_std, upper_air_mean, upper_air_std):
    """验证生成的数组"""

    print("\n验证生成的数组...")

    # 检查形状
    assert surface_mean.shape == (26, 721, 1440), f"Surface mean形状错误: {surface_mean.shape}"
    assert surface_std.shape == (26, 721, 1440), f"Surface std形状错误: {surface_std.shape}"
    assert upper_air_mean.shape == (10, 5, 721, 1440), f"Upper air mean形状错误: {upper_air_mean.shape}"
    assert upper_air_std.shape == (10, 5, 721, 1440), f"Upper air std形状错误: {upper_air_std.shape}"

    # 检查数值范围
    assert not np.any(np.isnan(surface_mean)), "Surface mean包含NaN"
    assert not np.any(np.isnan(surface_std)), "Surface std包含NaN"
    assert not np.any(np.isnan(upper_air_mean)), "Upper air mean包含NaN"
    assert not np.any(np.isnan(upper_air_std)), "Upper air std包含NaN"

    assert np.all(surface_std > 0), "Surface std包含非正值"
    assert np.all(upper_air_std > 0), "Upper air std包含非正值"

    # 显示统计信息
    print("✓ 数组验证通过!")
    print(f"  Surface mean范围: [{surface_mean.min():.6f}, {surface_mean.max():.6f}]")
    print(f"  Surface std范围: [{surface_std.min():.6f}, {surface_std.max():.6f}]")
    print(f"  Upper air mean范围: [{upper_air_mean.min():.6e}, {upper_air_mean.max():.6e}]")
    print(f"  Upper air std范围: [{upper_air_std.min():.6e}, {upper_air_std.max():.6e}]")

def load_normalization_arrays(json_path, verbose=False):
    """
    从JSON文件加载标准化参数数组（兼容旧版本维度格式）

    Args:
        json_path: JSON文件路径
        verbose: 是否打印详细信息，默认False（静默加载）

    Returns:
        [surface_mean, surface_std, upper_mean, upper_std]: 四个numpy数组
        - surface_mean: (1, 26, 1, 721, 1440)
        - surface_std: (1, 26, 1, 721, 1440)
        - upper_mean: (1, 10, 5, 1, 721, 1440)
        - upper_std: (1, 10, 5, 1, 721, 1440)

    注意：添加了额外的维度以兼容旧版本的代码
    """
    try:
        # 1. 加载JSON数据（平铺格式）
        stats_data = load_statistics_from_json(json_path, verbose=verbose)

        # 2. 创建Surface数组 (26, 721, 1440)
        surface_mean, surface_std = create_surface_arrays(stats_data, verbose=verbose)

        # 3. 创建Upper air数组 (10, 5, 721, 1440)
        upper_mean, upper_std = create_upper_air_arrays(stats_data, verbose=verbose)

        # 4. 添加额外维度以兼容旧版本
        # (26, 721, 1440) -> (1, 26, 1, 721, 1440)
        surface_mean = surface_mean[np.newaxis, :, np.newaxis, :, :]
        surface_std = surface_std[np.newaxis, :, np.newaxis, :, :]

        # (10, 5, 721, 1440) -> (1, 10, 5, 1, 721, 1440)
        upper_mean = upper_mean[np.newaxis, :, :, np.newaxis, :, :]
        upper_std = upper_std[np.newaxis, :, :, np.newaxis, :, :]

        if verbose:
            print(f"\n✓ 数组维度（兼容旧版本格式）：")
            print(f"  Surface mean: {surface_mean.shape}")
            print(f"  Surface std: {surface_std.shape}")
            print(f"  Upper air mean: {upper_mean.shape}")
            print(f"  Upper air std: {upper_std.shape}")

        return [surface_mean, surface_std, upper_mean, upper_std]

    except Exception as e:
        print(f"加载标准化数组时出错: {str(e)}")
        raise e

def main():
    """主函数"""

    print("="*80)
    print("字典转数组 - ERA5 1940-2023")
    print("="*80)

    # 输入输出路径
    input_json = "/data/lhwang/ERA5_raw/output/ERA5_1940_2023_mean_std_statistics.json"
    output_dir = "/data/lhwang/ERA5_raw/output"

    try:
        # 使用新的函数接口（详细模式）
        surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(
            input_json, verbose=True
        )

        # 验证数组
        validate_arrays(surface_mean, surface_std, upper_mean, upper_std)

        # 保存结果
        npz_path = save_pytorch_arrays(surface_mean, surface_std, upper_mean, upper_std, output_dir)

        print("\n" + "="*80)
        print("✅ 字典转数组成功完成!")
        print("="*80)
        print(f"\n输出文件:")
        print(f"  - {npz_path}")
        print(f"  - {output_dir}/surface_mean.npy")
        print(f"  - {output_dir}/surface_std.npy")
        print(f"  - {output_dir}/upper_air_mean.npy")
        print(f"  - {output_dir}/upper_air_std.npy")
        print(f"  - {output_dir}/pytorch_usage_example_2023.py")

        return True

    except Exception as e:
        print(f"\n转换过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ 转换失败，请检查日志")
        exit(1)
