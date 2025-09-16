import h5py as h5
import numpy as np

# 加载数据
print("Loading data...")
with h5.File('/gz-data/ERA5_2023_weekly.h5', 'r') as f:
    input_surface = f['surface'][:]
    input_upper_air = f['upper_air'][:]

print(f"Surface data shape: {input_surface.shape}")
print(f"Upper air data shape: {input_upper_air.shape}")

# 检查NaN数量
surface_nan_count = np.isnan(input_surface).sum()
upper_air_nan_count = np.isnan(input_upper_air).sum()
print(f"Surface NaN count: {surface_nan_count}")
print(f"Upper air NaN count: {upper_air_nan_count}")

# 将NaN转换为0
print("Converting NaN to 0...")
input_surface = np.nan_to_num(input_surface, nan=0.0)
input_upper_air = np.nan_to_num(input_upper_air, nan=0.0)

# 验证转换后没有NaN
surface_nan_after = np.isnan(input_surface).sum()
upper_air_nan_after = np.isnan(input_upper_air).sum()
print(f"Surface NaN after conversion: {surface_nan_after}")
print(f"Upper air NaN after conversion: {upper_air_nan_after}")

# 保存到新文件
print("Saving to new file...")
with h5.File('/gz-data/ERA5_2023_weekly1.h5', 'w') as f:
    f.create_dataset('surface', data=input_surface)
    f.create_dataset('upper_air', data=input_upper_air)

print("Data saved to ERA5_2023_weekly1.h5")