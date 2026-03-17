import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from canglong import CanglongSST16
from canglong.model_v1 import UpSample, DownSample, EarthAttention3D, EarthSpecificBlock, BasicLayer, Mlp

# ============================================================
# Pickle compatibility: old checkpoint references __main__.Canglong
# and helper classes from __main__. Aliasing them here so that
# torch.load can resolve the pickled class paths.
# ============================================================
Canglong = CanglongSST16

# ============================================================
# 常量
# ============================================================
monthly_avg_temp = np.load("/data/lhwang/monthly_avg_temp.npy")
ocean_mask = np.load('/data/lhwang/ocean_mask.npy')

mean_all = torch.tensor([[[[ 2.8679e+02]],
                          [[ 1.0096e+05]],
                          [[-5.3626e+06]],
                          [[-5.1725e-02]],
                          [[ 1.8698e-01]],
                          [[ 5.4089e+04]],
                          [[ 1.3745e+04]],
                          [[ 1.1180e+07]]]])
std_all = torch.tensor([[[[1.1627e+01]],
                         [[1.0610e+03]],
                         [[4.9920e+06]],
                         [[3.8811e+00]],
                         [[2.4887e+00]],
                         [[3.2341e+03]],
                         [[1.3297e+03]],
                         [[7.8841e+06]]]])

# ============================================================
# 1. 加载历史归一化数据 (160 months: Sep 2009 ~ Dec 2022)
# ============================================================
import xarray as xr
from datetime import datetime
from dateutil.relativedelta import relativedelta

normalized_data_old = torch.load(
    '/home/lhwang/Desktop/weather/code/normalized_data.pt',
    weights_only=False
)
print(f'Old data: {normalized_data_old.shape} (Sep 2009 ~ Dec 2022)')

# ============================================================
# 2. 从8个ERA5 NC文件加载新数据并归一化
# ============================================================
print('Loading ERA5 NC files...')
layer1 = xr.open_dataset('/data/lhwang/SST/ERA5_sea_surface_temperature_monthly_2023-2025.5.nc').sst.values
layer2 = xr.open_dataset('/data/lhwang/SST/ERA5_mean_sea_level_pressure_monthly_2023-2025.5.nc').msl.values
layer3 = xr.open_dataset('/data/lhwang/SST/ERA5_surface_latent_heat_flux_monthly_2023-2025.5.nc').slhf.values
layer4 = xr.open_dataset('/data/lhwang/SST/ERA5_10m_u_component_of_wind_monthly_2023-2025.5.nc').u10.values
layer5 = xr.open_dataset('/data/lhwang/SST/ERA5_10m_v_component_of_wind_monthly_2023-2025.5.nc').v10.values
layer6 = xr.open_dataset('/data/lhwang/SST/ERA5_500hPa_geopotential_height_monthly_2023-2025.5.nc').z.values.squeeze(1)
layer7 = xr.open_dataset('/data/lhwang/SST/ERA5_850hPa_geopotential_height_monthly_2023-2025.5.nc').z.values.squeeze(1)
layer8 = xr.open_dataset('/data/lhwang/SST/ERA5_surface_net_solar_radiation_2023-2025.5.nc').ssr.values

n_new_months = layer1.shape[0]
combined_input = np.stack([layer1, layer2, layer3, layer4, layer5, layer6, layer7, layer8], axis=1)
input_data = torch.from_numpy(combined_input).float()
normalized_data_new = torch.nan_to_num((input_data - mean_all) / std_all, nan=0.0)
print(f'New data:  {normalized_data_new.shape} ({n_new_months} months from NC files)')

# ============================================================
# 3. 拼接 → 完整时间序列
# ============================================================
normalized_data_all = torch.cat([normalized_data_old, normalized_data_new], dim=0)
print(f'Combined:  {normalized_data_all.shape}')

# ============================================================
# 自动计算日期
# Index 0 = Sep 2009, old data 有160个月到 Dec 2022
# New data 从 Jan 2023 开始，共 n_new_months 个月
# ============================================================
DATA_START = datetime(2009, 9, 1)  # index 0 对应的日期
total_months = normalized_data_all.shape[0]
data_end = DATA_START + relativedelta(months=total_months - 1)
print(f'Data range: {DATA_START:%Y-%m} ~ {data_end:%Y-%m}')

# 取最后16个月作为输入，预测接下来16个月
start_id = total_months - 16
start_month = start_id % 12 + 1
plot_obs = False

# 预测起始 = 输入窗口结束后一个月
pred_start_date = DATA_START + relativedelta(months=total_months)
pred_start_year = pred_start_date.year
pred_start_month = pred_start_date.month
pred_end_date = pred_start_date + relativedelta(months=15)

input_start_date = DATA_START + relativedelta(months=start_id)
input_end_date = DATA_START + relativedelta(months=start_id + 15)
print(f'Input window: index {start_id}~{start_id+15} ({input_start_date:%Y-%m} ~ {input_end_date:%Y-%m})')
print(f'Prediction: {pred_start_date:%Y-%m} ~ {pred_end_date:%Y-%m} (16 months)')

# ============================================================
# 加载模型并推理
# ============================================================
DEVICE = 'cuda:0'

the_model = torch.load(
    '/home/lhwang/Desktop/weather/model/canglong16_0005_600ep_base.pth',
    weights_only=False
)
the_model = the_model.module
input_former = normalized_data_all[start_id : start_id + 16, :, :, :]

the_model.to(DEVICE)
the_model.eval()

batch1 = input_former.permute(1, 0, 2, 3).unsqueeze(0)  # (1, 8, 16, 721, 1440)
print(f'Input shape: {batch1.shape}')

with torch.no_grad():
    output = the_model(batch1.to(DEVICE))
    # 反归一化 (仅SST通道)
    output = output.squeeze() * std_all[0, 0, 0, 0].item() + mean_all[0, 0, 0, 0].item()
    output = output.cpu().numpy()  # (16, 721, 1440)

print(f'Output shape: {output.shape}')

# ============================================================
# 计算SSTA和Nino3.4
# ============================================================
month_indices = (np.arange(16) + (pred_start_month - 1)) % 12
ssta = output - monthly_avg_temp[month_indices, :, :]
ssta[:, ocean_mask] = np.nan

# Nino3.4: 5S-5N, 170W-120W
nino34_lat_slice = slice(340, 380)
nino34_lon_slice = slice(760, 960)
nino34_pred = np.nanmean(ssta[:, nino34_lat_slice, nino34_lon_slice], axis=(1, 2))

print('\nNino3.4 SSTA Prediction:')
for i in range(16):
    m = (pred_start_month - 1 + i) % 12 + 1
    y = pred_start_year + (pred_start_month - 1 + i) // 12
    print(f'  {y}-{m:02d}: {nino34_pred[i]:.3f} °C')

# ============================================================
# Nature-style绘图
# ============================================================
font_path = "/usr/share/fonts/arial/ARIAL.TTF"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'

plt.style.use('seaborn-v0_8-talk')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'lines.linewidth': 1.0,
    'axes.linewidth': 1.0,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'axes.edgecolor': '#454545',
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 8,
    'ytick.major.size': 8,
    'xtick.minor.size': 4,
    'ytick.minor.size': 4,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.minor.width': 1.0,
    'ytick.minor.width': 1.0,
    'xtick.color': '#454545',
    'ytick.color': '#454545',
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
})

# 生成月份标签
month_labels = []
for i in range(16):
    m = (pred_start_month - 1 + i) % 12 + 1
    y = pred_start_year + (pred_start_month - 1 + i) // 12
    month_labels.append(f'{y}-{m:02d}')

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(16)

# 填充正负异常区域
ax.fill_between(x, 0, nino34_pred, where=(nino34_pred >= 0),
                color='#E74C3C', alpha=0.15)
ax.fill_between(x, 0, nino34_pred, where=(nino34_pred < 0),
                color='#3498DB', alpha=0.15)

# 绘制预测曲线
ax.plot(x, nino34_pred, color='#2C3E50', marker='o', markersize=4,
        linewidth=1.2, label='CAS-Canglong SST Prediction')

# El Nino / La Nina阈值线
ax.axhline(y=0.5, color='#E74C3C', linestyle='--', linewidth=0.6, alpha=0.7, label='El Nino threshold (+0.5)')
ax.axhline(y=-0.5, color='#3498DB', linestyle='--', linewidth=0.6, alpha=0.7, label='La Nina threshold (-0.5)')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.4)

ax.set_xticks(x)
ax.set_xticklabels(month_labels, rotation=45, ha='right')
ax.set_xlabel('Month')
ax.set_ylabel('Nino3.4 SSTA (°C)')
ax.set_title(f'CAS-Canglong SST Model: Nino3.4 Index Prediction ({pred_start_date:%b %Y} - {pred_end_date:%b %Y})')
ax.legend(loc='best', framealpha=0.9)
ax.set_xlim(-0.5, 15.5)

fig.tight_layout()
save_path = '/home/lhwang/Desktop/CanglongPhysics/analysis/operation/SSTmodel/nino34_prediction.png'
fig.savefig(save_path, dpi=600)
fig.savefig(save_path.replace('.png', '.svg'))
print(f'\nFigure saved to {save_path}')
plt.close()
