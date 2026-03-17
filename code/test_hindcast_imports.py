"""
测试 hindcast_22_23_claude.py 的导入和基本功能
"""
import sys
import os

print("测试导入...")

try:
    import torch
    print(f"✓ torch {torch.__version__}")
except ImportError as e:
    print(f"✗ torch: {e}")

try:
    import numpy as np
    print(f"✓ numpy {np.__version__}")
except ImportError as e:
    print(f"✗ numpy: {e}")

try:
    import xarray as xr
    print(f"✓ xarray {xr.__version__}")
except ImportError as e:
    print(f"✗ xarray: {e}")

try:
    import psutil
    print(f"✓ psutil {psutil.__version__}")
except ImportError as e:
    print(f"✗ psutil: {e}")

try:
    import pandas as pd
    print(f"✓ pandas {pd.__version__}")
except ImportError as e:
    print(f"✗ pandas: {e}")

try:
    from scipy.special import gamma as gamma_function
    print(f"✓ scipy")
except ImportError as e:
    print(f"✗ scipy: {e}")

try:
    from tqdm import tqdm
    print(f"✓ tqdm")
except ImportError as e:
    print(f"✗ tqdm: {e}")

print("\n测试CUDA...")
if torch.cuda.is_available():
    print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
else:
    print("✗ CUDA不可用")

print("\n测试数据文件...")
files_to_check = [
    '/data/lhwang/input_surface_norm_test_last100.pt',
    '/data/lhwang/input_upper_air_norm_test_last100.pt',
    '/data/lhwang/climate_variables_2000_2023_weekly.nc',
    '/home/lhwang/Desktop/model/weather_model_epoch_500.pt'
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / 1e6
        print(f"✓ {os.path.basename(file_path)} ({size_mb:.1f}MB)")
    else:
        print(f"✗ {file_path} 不存在")

print("\n测试完成！")
