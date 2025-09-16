#!/usr/bin/env python3
"""
测试convert_dict_to_pytorch_arrays.py的load_normalization_arrays函数
"""

import numpy as np
from convert_dict_to_pytorch_arrays import load_normalization_arrays

def test_load_arrays():
    """测试加载标准化数组"""
    
    print("测试load_normalization_arrays函数...")
    
    # JSON文件路径
    json_path = "/home/CanglongPhysics/code_v2/ERA5_1940_2019_combined_mean_std.json"
    
    try:
        # 调用函数（静默模式）
        surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(json_path, verbose=False)
        
        print("静默加载测试 - 无打印信息")
        
        # 调用函数（详细模式）
        print("\n详细加载测试 - 有打印信息:")
        surface_mean2, surface_std2, upper_mean2, upper_std2 = load_normalization_arrays(json_path, verbose=True)
        
        # 检查返回值类型和形状
        print("\n返回值检查:")
        print(f"surface_mean: {type(surface_mean)}, shape: {surface_mean.shape}")
        print(f"surface_std: {type(surface_std)}, shape: {surface_std.shape}")
        print(f"upper_mean: {type(upper_mean)}, shape: {upper_mean.shape}")
        print(f"upper_std: {type(upper_std)}, shape: {upper_std.shape}")
        
        # 检查数值范围
        print("\n数值范围检查:")
        print(f"surface_mean范围: [{surface_mean.min():.6f}, {surface_mean.max():.6f}]")
        print(f"surface_std范围: [{surface_std.min():.6f}, {surface_std.max():.6f}]")
        print(f"upper_mean范围: [{upper_mean.min():.6f}, {upper_mean.max():.6f}]")
        print(f"upper_std范围: [{upper_std.min():.6f}, {upper_std.max():.6f}]")
        
        # 检查是否有NaN或异常值
        print("\n异常值检查:")
        print(f"surface_mean包含NaN: {np.any(np.isnan(surface_mean))}")
        print(f"surface_std包含NaN: {np.any(np.isnan(surface_std))}")
        print(f"upper_mean包含NaN: {np.any(np.isnan(upper_mean))}")
        print(f"upper_std包含NaN: {np.any(np.isnan(upper_std))}")
        
        print(f"surface_std包含非正值: {np.any(surface_std <= 0)}")
        print(f"upper_std包含非正值: {np.any(upper_std <= 0)}")
        
        print("\n✅ 测试通过！函数正常工作")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_load_arrays()
    if success:
        print("\n函数可以正常使用，调用方式:")
        print("from convert_dict_to_pytorch_arrays import load_normalization_arrays")
        print("# 静默加载（无打印信息）")
        print("surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(json_path)")
        print("# 详细加载（有打印信息）") 
        print("surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(json_path, verbose=True)")