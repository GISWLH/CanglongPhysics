import pickle
import numpy as np
import matplotlib.pyplot as plt

# 测试土壤水mask效果
data_path = 'figures/soil_water_tcc_data_lead1.pkl'
with open(data_path, 'rb') as f:
    data = pickle.load(f)

print("土壤水数据分析 (带海洋mask):")
print(f"Lead time: {data['lead_time']}")
print(f"Gridpoint TCC (land-only): {data['gridpoint_tcc']:.4f}")
print(f"Time series TCC: {data['timeseries_tcc']:.4f}")
print(f"Land mask shape: {data['land_mask'].shape}")
print(f"Land points: {np.sum(data['land_mask'])} out of {data['land_mask'].size}")
print(f"Land percentage: {100 * np.sum(data['land_mask']) / data['land_mask'].size:.1f}%")

# 检查TCC地图中的NaN值（海洋区域）
tcc_map = data['tcc_map']
nan_count = np.sum(np.isnan(tcc_map))
print(f"Ocean points (NaN) in TCC map: {nan_count}")
print(f"Land points (valid) in TCC map: {np.sum(~np.isnan(tcc_map))}")

# 简单的可视化检查
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 原始TCC地图
ax1 = axes[0]
im1 = ax1.imshow(data['tcc_map_original'], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax1.set_title('Original TCC Map (with ocean)')
plt.colorbar(im1, ax=ax1)

# Mask后的TCC地图  
ax2 = axes[1]
im2 = ax2.imshow(data['tcc_map'], cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax2.set_title('Masked TCC Map (land only)')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('figures/test_soil_water_mask_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\nMask comparison plot saved to: figures/test_soil_water_mask_comparison.png")