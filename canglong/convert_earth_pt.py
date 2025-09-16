import torch
import numpy as np

# 加载Earth.pt文件
print("Loading Earth.pt...")
earth_data = torch.load('/home/CanglongPhysics/constant_masks/Earth.pt', map_location='cpu')

print(f"Original data shape: {earth_data.shape}")
print(f"Original data type: {earth_data.dtype}")

# 检查NaN数量
nan_count = torch.isnan(earth_data).sum().item()
print(f"NaN count: {nan_count}")

# 将NaN转换为0
print("Converting NaN to 0...")
earth_data_clean = torch.nan_to_num(earth_data, nan=0.0)

# 验证转换后没有NaN
nan_after = torch.isnan(earth_data_clean).sum().item()
print(f"NaN after conversion: {nan_after}")

# 保存到新文件
print("Saving to Earth1.pt...")
torch.save(earth_data_clean, '/home/CanglongPhysics/constant_masks/Earth1.pt')

print("Earth1.pt saved successfully!")