# CLAUDE.md

此文件为 Claude Code 在此代码库中工作时提供指导。

## 编码偏好与环境

- 运行代码所需的环境：`conda activate torch`，运行py代码前先激活环境
- 模型研发阶段在base环境中运行，无需activate torch
- timeout应大于10分钟，代码运行较慢，可以多给一些时间
- 不喜欢定义太过复杂的函数并运行main函数；偏好jupyter notebook风格的直接代码，简单函数定义可以接受
- 使用matplotlib可视化，绘图使用Arial字体（`/usr/share/fonts/arial/ARIAL.TTF`），图片标记都用英文
- 绘图采用Nature风格参数（见附录）

## 项目概述

CAS-Canglong 是一个基于物理信息的AI次季节-季节（S2S）天气预测系统。项目将Swin-Transformer架构与物理约束相结合，构建了四个递进版本的天气预报模型，面向6周滚动预报任务。

## 模型版本体系

项目包含四个模型版本，逐步递进：

| 版本 | 代号 | 类名 | 模型文件 | 训练文件 | 核心特性 |
|------|------|------|----------|----------|----------|
| **V0** | Lite | `Canglong` | `canglong/model_v0.py` | — | 初始版本，训练20年数据，参数少 |
| **V1** | Base | `Canglong` | `canglong/model_v1.py` | `train_v1.py` | 扩充变量和参数量，标准Swin-Transformer |
| **V2** | Wind | `CanglongV2` | `canglong/model_v2.py` | `train_v2.py` | 在V1基础上添加风向感知窗口移位 |
| **V3** | Full | `CanglongV3` | `canglong/model_v3.py` | `train_v3.py` | 在V2基础上添加PINN物理约束损失 |

### V0 Lite — 初始基线

最早版本，变量少、参数少，仅训练了20年ERA5数据。

- **高空变量**: 7个变量，4个压力层 (300, 500, 700, 850 hPa)
- **表面变量**: 16个
- **地球常数**: 4个 (in_channels=4)
- **堆叠分辨率**: (4, 181, 360)，其中 4 = 1(upper air) + 1(constant) + 2(surface time)
- **输入**: surface (B, 16, 2, 721, 1440), upper_air (B, 7, 4, 721, 1440)
- **使用旧版嵌入**: `embed_old.py`, `recovery_old.py` (注意与V1+不兼容)
- **常量硬编码**: `input_constant = torch.load(...).cuda()` 写在模块顶层

### V1 Base — 标准Swin-Transformer

扩充到完整ERA5变量集，参数量显著增加。

- **高空变量**: 10个变量，5个压力层 (200, 300, 500, 700, 850 hPa)
- **表面变量**: 26个
- **地球常数**: 64个 (in_channels=64)
- **堆叠分辨率**: (6, 181, 360)，其中 6 = 3(upper air) + 2(surface time) + 1(constant)
- **输入**: surface (B, 26, 2, 721, 1440), upper_air (B, 10, 5, 2, 721, 1440)
- **输出**: surface (B, 26, 1, 721, 1440), upper_air (B, 10, 5, 1, 721, 1440)
- **Earth constant动态加载**: 支持路径参数 `earth_constant_path`

### V2 Wind — 风向感知注意力

在V1基础上，利用风场先验知识驱动Swin-Transformer的窗口移位策略。

- **变量与V1完全相同**
- **新增组件**:
  - `WindDirectionProcessor` — 从原始u/v风分量计算离散风向ID (9个方向: 无移位 + 8个罗盘方向)
  - `WindAwareEarthSpecificBlock` — 风向感知的Transformer块
  - 分区域模式: 4×8=32个区域独立计算风向
- **风向提取**: 在encoder之前从原始物理空间提取
  - upper_air[:, 3:5, :, :, :, :] → 多层u/v
  - surface[:, 7:9, :, :, :] → 10m u/v
- **窗口移位**: Swin固定移位 + 风向额外移位的双重机制
- **forward签名**: `forward(surface, upper_air, return_wind_info=False)`

### V3 Full — 物理约束PINN

在V2基础上，训练时添加物理方程残差作为软约束损失。模型架构（forward）与V2相同，物理约束仅体现在训练损失函数中。

- **模型架构与V2相同**，forward签名一致
- **物理约束损失**（在训练脚本中独立定义，不嵌入模型）:
  - 水量平衡 (Water Balance)
  - 能量平衡 (Energy Balance)
  - 静力平衡 (Hydrostatic Balance)
- **总损失**: `L_total = L_MSE + λ_water·L_water + λ_energy·L_energy + λ_pressure·L_pressure`
- **V2.x子版本** (实验性): V2.1-V2.5 测试不同的风向移位策略，位于 `canglong/model_v2_*.py`

### 各版本详细分析与性能对比

#### V1 — 基础版 (model_v1.py, 617行)

标准3D Swin-Transformer基础架构，无风向感知或物理约束，作为所有后续版本的对比基线。

#### V2 → V2.5 — 风向感知系列

| 版本 | 核心创新 | 关键性能 (降水PCC) |
|------|---------|------------------|
| V2 | 区域风向掩码 (4×8=32区域)，预计算9个注意力掩码 | 0.391 (+42% vs V1) |
| V2.1 | 改为逐窗口全局位移，大幅简化设计 | 0.608 (+55% vs V1) |
| V2.2 | 逐窗口精细化掩码匹配，支持梯度检查点 | 0.605 |
| V2.3 | V2.2参数化子类，Top-4风向 | 0.609 |
| V2.4 | Top-3方向 + embed_dim扩大到192 | 0.591 (效果下降) |
| V2.5 | embed_dim=192，更深网络，逐层风向控制（仅L1层启用风向移位） | 0.601 (训练2轮后→0.662) |

**关键转折点**: V2.1是最重要的改进，将surface RMSE从12.5降到0.80 (-93%)。

#### V3 — 物理约束版 (model_v3.py, 460行)

在V2的风向感知基础上，加入三个软约束物理损失项：

`L_total = L_MSE + λ_w·L_water + λ_e·L_energy + λ_p·L_pressure`

| 约束 | 方程 |
|------|------|
| 水量平衡 | ΔSoil water = P - E |
| 能量平衡 | R_n = LE + H + G |
| 静力平衡 | Δφ(层间) = R_d × T_avg × ln(P下/P上) |

**已知问题**: 损失函数错误地写入了模型主体，标准化/反标准化混乱（3种不同方式）。

#### V3.5 — 最终融合版 (train_v3_5.py)

- **主干**: 使用 V2.5（更宽更深，选择性风向移位）
- **物理约束**: 继承V3的三个物理损失项（在训练脚本中正确实现）
- **工程增强**: DDP分布式训练、混合精度(AMP)、动态损失权重调整、NaN检测、完整checkpoint管理

#### 架构演进总结

```
V1 (基础) → V2 (区域风向掩码) → V2.1 (逐窗口全局移位, 质变提升)
→ V2.2-V2.5 (微调+加宽加深) → V3 (加物理约束但结构有问题)
→ V3.5 (V2.5主干 + 正确物理约束 + 工程优化)
```

## 模型架构详情 (V1/V2/V3 通用)

### 处理流程

```
输入:
├─ Surface (B, 26, 2, 721, 1440) → Encoder3D (Conv3D+ResNet) → (B, 96, 2, 181, 360)
├─ Upper Air (B, 10, 5, 2, 721, 1440) → PatchEmbed4D (Conv4D) → (B, 96, 3, 1, 181, 360) → squeeze → (B, 96, 3, 181, 360)
└─ Earth Constant (64, 721, 1440) → Conv2D → (B, 96, 181, 360) → unsqueeze → (B, 96, 1, 181, 360)

堆叠 (按顺序: upper_air, surface, constant):
└─ (B, 96, 6, 181, 360) → reshape → (B, 6×181×360, 96)

U-Transformer (Swin):
├─ Layer1: (6, 181, 360), dim=96, depth=2
├─ DownSample → (6, 91, 180), dim=192
├─ Layer2: depth=6
├─ Layer3: depth=6
├─ UpSample → (6, 181, 360), dim=96
├─ Layer4: depth=2
└─ Skip Connection → (B, 192, 6, 181, 360)

输出分离:
├─ output[:, :, 0:3, :, :] → upper_air → PatchRecovery4D → (B, 10, 5, 1, 721, 1440)
└─ output[:, :, 3:5, :, :] → surface → Decoder3D → (B, 26, 1, 721, 1440)
```

### 关键分辨率
- 高分辨率: (6, 181, 360)
- 低分辨率: (6, 91, 180)

## 核心目录结构

```
CanglongPhysics/
├── canglong/                    # 核心模型包
│   ├── __init__.py              # 导出所有模型和工具
│   ├── model_v0.py              # V0 Lite (Canglong)
│   ├── model_v1.py              # V1 Base (Canglong)
│   ├── model_v2.py              # V2 Wind (CanglongV2)
│   ├── model_v2_1~v2_5.py       # V2实验子版本
│   ├── model_v3.py              # V3 Full (CanglongV3)
│   ├── embed.py                 # 2D/3D/4D Patch嵌入
│   ├── embed_old.py             # V0专用旧版嵌入
│   ├── recovery.py              # Patch还原操作
│   ├── earth_position.py        # 地球特定位置编码
│   ├── shift_window.py          # 移位窗口注意力
│   ├── wind_direction.py        # 风向计算与离散化
│   ├── wind_aware_block.py      # 风向感知Transformer块
│   ├── wind_aware_shift.py      # 风向驱动窗口移位
│   ├── helper.py                # 共享构建块 (ResBlock, GroupNorm等)
│   ├── pad.py / crop.py         # 空间填充和裁剪
│   └── Conv4d.py                # 4D卷积操作
├── train_v1.py                  # V1训练脚本
├── train_v2.py                  # V2训练脚本
├── train_v3.py                  # V3训练脚本 (含物理约束损失)
├── train_v3_5.py                # V3.5扩展训练
├── code/
│   ├── run.py                   # 主要6周滚动预报推理管道
│   ├── run_ec_pure_zdx.py       # ECMWF对比运行
│   └── hindcast_verification_final.py  # 回报检验系统
├── code_v2/                     # 标准化数据和旧版模型定义
│   ├── ERA5_1940_2019_combined_mean_std.json  # 标准化参数
│   └── convert_dict_to_pytorch_arrays.py      # 加载标准化数组
├── constant_masks/              # 预计算地理常数
│   ├── Earth.pt / input_tensor.pt  # 64个地球常数 (64, 721, 1440)
│   ├── is_land.pt               # 陆地/海洋掩码 (land=1, ocean=0)
│   ├── hydrobasin_exorheic_mask.pt  # 外流区掩码 (外流区=1)
│   └── csol_bulk_025deg_721x1440_corrected.pt  # 土壤热容
├── weatherlearn/                # 参考模型 (Pangu, FuXi)
├── data/                        # 数据存储
│   ├── canglong_pre/            # Canglong预报NC文件
│   ├── ecmwf/                   # ECMWF预报TIF文件 (T/, P/)
│   └── hind_obs/                # 观测数据
├── figures/                     # 输出图片
│   ├── hindcast_china/          # 回报检验结果
│   └── hindcast_region/         # 区域检验结果
└── physical_constraint.md       # 物理约束方法详细文档
```

## ERA5 变量规范 (V1/V2/V3)

### 变量总览

| 类别 | 数量 | 维度 | 说明 |
|------|------|------|------|
| Surface变量 | 26 | (26, 721, 1440) | 地表单层变量 |
| Upper Air变量 | 10 | (10, 5, 721, 1440) | 高空多层变量 |
| 压力层 | 5 | — | 200, 300, 500, 700, 850 hPa |
| 空间网格 | — | 721×1440 | 0.25°分辨率全球网格 |

### Surface变量（严格顺序）

| 索引 | 变量名 | 英文全称 | 单位 | 典型范围 |
|------|--------|----------|------|----------|
| 0 | avg_tnswrf | Mean Top Net Short Wave Radiation Flux | W/m² | 0-400 |
| 1 | avg_tnlwrf | Mean Top Net Long Wave Radiation Flux | W/m² | -300--100 |
| 2 | tciw | Total Column Cloud Ice Water | kg/m² | 0-0.5 |
| 3 | tcc | Total Cloud Cover | 0-1 | 0-1 |
| 4 | lsrr | Large Scale Rain Rate | kg/m²/s | 0-0.01 |
| 5 | crr | Convective Rain Rate | kg/m²/s | 0-0.01 |
| 6 | blh | Boundary Layer Height | m | 100-3000 |
| 7 | u10 | 10m U Component of Wind | m/s | -50-50 |
| 8 | v10 | 10m V Component of Wind | m/s | -50-50 |
| 9 | d2m | 2m Dewpoint Temperature | K | 200-320 |
| 10 | t2m | 2m Temperature | K | 200-330 |
| 11 | avg_iews | Mean Eastward Turbulent Surface Stress | N/m² | -1-1 |
| 12 | avg_inss | Mean Northward Turbulent Surface Stress | N/m² | -1-1 |
| 13 | slhf | Surface Latent Heat Flux | J/m² | -1e7-1e7 |
| 14 | sshf | Surface Sensible Heat Flux | J/m² | -1e6-1e6 |
| 15 | avg_snswrf | Mean Surface Net Short Wave Radiation Flux | W/m² | 0-300 |
| 16 | avg_snlwrf | Mean Surface Net Long Wave Radiation Flux | W/m² | -150-0 |
| 17 | ssr | Surface Net Solar Radiation | J/m² | 0-1e6 |
| 18 | str | Surface Net Thermal Radiation | J/m² | -5e5-0 |
| 19 | sp | Surface Pressure | Pa | 50000-110000 |
| 20 | msl | Mean Sea Level Pressure | Pa | 95000-105000 |
| 21 | siconc | Sea Ice Concentration | 0-1 | 0-1 |
| 22 | sst | Sea Surface Temperature | K | 271-310 |
| 23 | ro | Runoff | m | 0-0.01 |
| 24 | stl | Soil Temperature Layer (加权) | K | 200-330 |
| 25 | swvl | Volumetric Soil Water Layer (加权) | m³/m³ | 0-1 |

索引24、25是土壤四层加权变量，层厚 d1=0.07m, d2=0.21m, d3=0.72m, d4=1.89m，总深度2.89m：
```python
stl = (stl1*0.07 + stl2*0.21 + stl3*0.72 + stl4*1.89) / 2.89
swvl = (swvl1*0.07 + swvl2*0.21 + swvl3*0.72 + swvl4*1.89) / 2.89
```

```python
surf_vars = ['avg_tnswrf', 'avg_tnlwrf', 'tciw', 'tcc', 'lsrr', 'crr', 'blh',
             'u10', 'v10', 'd2m', 't2m', 'avg_iews', 'avg_inss', 'slhf', 'sshf',
             'avg_snswrf', 'avg_snlwrf', 'ssr', 'str', 'sp', 'msl', 'siconc',
             'sst', 'ro', 'stl', 'swvl']
```

数组维度：输入 `(B, 26, time, 721, 1440)`，输出 `(B, 26, 1, 721, 1440)`，标准化 `(1, 26, 1, 721, 1440)`

### Upper Air变量（严格顺序）

| 索引 | 变量名 | 英文全称 | 单位 | 典型范围 |
|------|--------|----------|------|----------|
| 0 | o3 | Ozone Mass Mixing Ratio | kg/kg | 0-1e-5 |
| 1 | z | Geopotential | m²/s² | 0-120000 |
| 2 | t | Temperature | K | 180-320 |
| 3 | u | U Component of Wind | m/s | -100-100 |
| 4 | v | V Component of Wind | m/s | -100-100 |
| 5 | w | Vertical Velocity | Pa/s | -5-5 |
| 6 | q | Specific Humidity | kg/kg | 0-0.02 |
| 7 | cc | Fraction of Cloud Cover | 0-1 | 0-1 |
| 8 | ciwc | Specific Cloud Ice Water Content | kg/kg | 0-0.001 |
| 9 | clwc | Specific Cloud Liquid Water Content | kg/kg | 0-0.001 |

```python
upper_vars = ['o3', 'z', 't', 'u', 'v', 'w', 'q', 'cc', 'ciwc', 'clwc']
levels = [200, 300, 500, 700, 850]  # 从高到低
```

压力层: 索引0=200hPa (~12km), 1=300hPa (~9km), 2=500hPa (~5.5km), 3=700hPa (~3km), 4=850hPa (~1.5km)

数组维度：输入 `(B, 10, 5, time, 721, 1440)`，输出 `(B, 10, 5, 1, 721, 1440)`，标准化 `(1, 10, 5, 1, 721, 1440)`

### S2S关键变量权重

MJO预报重点关注以下变量，在损失函数中给予额外权重：

**Surface**: OLR (索引1), 降水 lsrr+crr (索引4+5), d2m (索引9), t2m (索引10)

**Upper Air**: 850hPa U风 [:, 3, 4, :, :, :], 200hPa U风 [:, 3, 0, :, :, :]

## 数据标准化

使用40年ERA5 (1940-2019) 统计量，通过 `code_v2/convert_dict_to_pytorch_arrays.py` 加载：

```python
from code_v2.convert_dict_to_pytorch_arrays import load_normalization_arrays
json_path = 'code_v2/ERA5_1940_2019_combined_mean_std.json'
surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(json_path)
# surface_mean.shape = (1, 26, 1, 721, 1440)
# upper_mean.shape   = (1, 10, 5, 1, 721, 1440)
```

训练循环中的标准化（统一规范，所有版本通用）：
```python
input_surface = ((input_surface.permute(0, 2, 1, 3, 4) - surface_mean) / surface_std).to(device)
input_upper_air = ((input_upper_air.permute(0, 2, 3, 1, 4, 5) - upper_mean) / upper_std).to(device)
target_surface = ((target_surface.unsqueeze(2) - surface_mean) / surface_std).to(device)
target_upper_air = ((target_upper_air.unsqueeze(3) - upper_mean) / upper_std).to(device)
```

反标准化（计算物理约束时需要）：
```python
output_surface_physical = output_surface * surface_std + surface_mean
output_upper_physical = output_upper_air * upper_std + upper_mean
```

## 物理约束 (V3)

物理约束作为软约束损失添加到训练中，不嵌入模型forward。详见 `physical_constraint.md`。

总损失: `L_total = L_MSE + λ_water·L_water + λ_energy·L_energy + λ_pressure·L_pressure`

所有物理约束在反标准化后的物理空间中计算。注意数据是周平均值，累积量需正确处理时间尺度 (delta_t = 7×24×3600秒)。

### 1. 水量平衡

∆Soil_water = P_total − E (仅在 hydrobasin_exorheic_mask 区域计算)

```python
delta_soil_water = output_physical[:, 25] - input_physical[:, 25]  # swvl, 需乘以深度2.89m
p_total = (output_physical[:, 4] + output_physical[:, 5]) * delta_t  # lsrr + crr
evaporation = output_physical[:, 13] / 2.5e6 * delta_t  # slhf → E
residual = delta_soil_water - (p_total - evaporation)
```

### 2. 能量平衡

R_n = LE + H (陆地表面，仅 is_land 区域)

```python
sw_net = output_physical[:, 15]   # avg_snswrf (W/m²)
lw_net = output_physical[:, 16]   # avg_snlwrf (W/m²)
shf = output_physical[:, 14]      # sshf (J/m²)
lhf = output_physical[:, 13]      # slhf (J/m²)
residual = (sw_net + lw_net) - (shf + lhf)
```

### 3. 静力平衡

Δφ = R_d × T_avg × ln(p₁/p₂)，逐相邻压力层计算

```python
phi_850 = output_upper_physical[:, 1, 4]  # z at 850hPa
phi_700 = output_upper_physical[:, 1, 3]  # z at 700hPa
T_avg = (output_upper_physical[:, 2, 4] + output_upper_physical[:, 2, 3]) / 2
residual = (phi_700 - phi_850) - 287 * T_avg * (log(850) - log(700))
```

## 数据来源与格式

### ERA5数据
- **来源**: Google Cloud Storage `gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`
- **分辨率**: 0.25° 全球 (721×1440)
- **时间**: 小时级，聚合为周平均用于S2S预报

### 推理输出
- Canglong预报: NetCDF格式，如 `canglong_6weeks_2025-06-18_2025-07-29.nc`
  - 温度: `ds['2m_temperature']` (K → 减273.15转℃)
  - 降水: `ds['total_precipitation']` (m/hr → ×24000转mm/day)

### ECMWF数据
- 位于 `data/ecmwf/T/` 和 `data/ecmwf/P/`
- 用 rioxarray 读取 `.tif` 文件
- 温度已是℃，降水已是mm/day
- 分辨率1.5°，中国区域 (band:6, y:27, x:47)

## 预报检验模式

### 周数划分

每年从1月1日起按7天划分为52周（12月31日或30日不计入）。

### 回报检验流程

以hindcast_start_week = 25（2025-06-18至06-24）为例，检验提前1-6周预报：

| 提前周数 | Canglong文件 | 提取 | ECMWF文件 | 提取 |
|----------|-------------|------|-----------|------|
| 1 | canglong_6weeks_2025-06-18_*.nc | time[0] | P/T_2025-06-18_weekly.tif | band[0] |
| 2 | canglong_6weeks_2025-06-11_*.nc | time[1] | P/T_2025-06-11_weekly.tif | band[1] |
| 3 | canglong_6weeks_2025-06-04_*.nc | time[2] | P/T_2025-06-04_weekly.tif | band[2] |
| 4 | canglong_6weeks_2025-05-28_*.nc | time[3] | P/T_2025-05-28_weekly.tif | band[3] |
| 5 | canglong_6weeks_2025-05-21_*.nc | time[4] | P/T_2025-05-21_weekly.tif | band[4] |
| 6 | canglong_6weeks_2025-05-14_*.nc | time[5] | P/T_2025-05-14_weekly.tif | band[5] |

### 验证指标
- **RMSE**: 均方根误差
- **ACC**: 异常相关系数（需气候态 `data/climate_variables_2000_2023_weekly.nc`）
- **SPEI同号率**: 干旱/湿润状态预报一致性

### 数据对齐策略
ECMWF是中国区域1.5°，Canglong是全球0.25°。统一插值到ECMWF的1.5°网格 (27×47) 进行比较。

### 执行
```bash
conda activate torch
python code/hindcast_verification_final.py
```

输出: `figures/hindcast_china/` 目录下的ACC/RMSE/SPEI对比图和CSV数据表。

## 标准化评估体系

### 评估数据集格式 — 目标周中心 (Target-Week-Centric)

评估数据集采用**目标周中心**的组织方式：时间轴为连续的目标周（2017-2021全部52×5=260周），对于每个目标周，存储该周的ERA5观测值，以及不同提前量（lead 1-6周）的模型预测值。

这种组织方式的优势：
- **无冗余**: 每个目标周的观测只存一份（旧格式按初始化组织，同一周的obs被重复存储多次）
- **时间连续**: 时间轴覆盖测试期全部周，支持时间序列分析
- **直观对比**: 对同一日期，可直接看到真值和各提前量的预测，方便跨模型（AI/NWP）对比
- **通用格式**: 任何预报模型（Canglong、ECMWF、Pangu等）均可用相同结构组织，统一对比

#### 数据结构

```text
Dimensions:
  time: 260 (目标周, 2017-2021连续)
  lat:  721 (0.25°, 90°N → 90°S)
  lon:  1440 (0.25°, 0° → 359.75°E)

Coordinates:
  time (time): datetime64 - 目标周日期 (CF-convention, xarray自动解析)
  lat (lat): float32 - degrees_north
  lon (lon): float32 - degrees_east

Auxiliary Coordinates:
  year (time): int32 - 目标周年份
  woy (time): int32 - week-of-year (0-indexed, 用于气候态查表)
  global_idx (time): int32 - Zarr全局时间索引

Data Variables (float32, dims (time, lat, lon)):
  obs_tp                   # 观测: 总降水 (lsrr+crr), kg/m²/s
  pred_tp_lead1 ~ lead6    # 预测: 提前1~6周的总降水预报
  obs_t2m                  # 观测: 2m温度, K
  pred_t2m_lead1 ~ lead6   # 预测: 提前1~6周的2m温度预报
  obs_olr                  # 观测: OLR, W/m² (V0 pred为NaN)
  pred_olr_lead1 ~ lead6
  obs_z500                 # 观测: 500hPa位势高度, m²/s²
  pred_z500_lead1 ~ lead6
  obs_u850                 # 观测: 850hPa纬向风, m/s (仅V3.5)
  pred_u850_lead1 ~ lead6
  obs_u200                 # 观测: 200hPa纬向风, m/s (仅V3.5)
  pred_u200_lead1 ~ lead6
```

V3.5: 6 obs + 36 pred = **42个变量**; V0: 4 obs + 24 pred = **28个变量**

#### pred_lead{L} 的含义

对于目标周 t，`pred_{var}_lead{L}` 是**提前 L 周**的自回归预报：

```text
pred_lead{L} 的生成过程:
  初始化点: t - L - 1 (输入两周obs: [t-L-1, t-L])
  自回归 L 步到达目标周 t

示例: target = 2017-01-01, pred_tp_lead3
  输入: [obs_2016-12-04, obs_2016-12-11]  (2周obs)
  Step 1: [obs_wk-4, obs_wk-3] → pred_wk-2
  Step 2: [obs_wk-3, pred_wk-2] → pred_wk-1
  Step 3: [pred_wk-2, pred_wk-1] → pred_2017-01-01  ← pred_tp_lead3
```

| Lead | 初始化点 | obs输入 | 自回归步数 | 纯pred输入步数 |
|------|---------|---------|-----------|---------------|
| 1 | t - 2 | [t-2, t-1] | 1 | 0 |
| 2 | t - 3 | [t-3, t-2] | 2 | 1 |
| 3 | t - 4 | [t-4, t-3] | 3 | 2 |
| 4 | t - 5 | [t-5, t-4] | 4 | 3 |
| 5 | t - 6 | [t-6, t-5] | 5 | 4 |
| 6 | t - 7 | [t-7, t-6] | 6 | 5 |

Lead越大，自回归链越长，误差累积越多，预报技巧下降。

#### 评估变量定义

| 变量名 | 说明 | 物理单位 | Zarr Surface索引 | Zarr Upper索引 |
|--------|------|----------|------------------|----------------|
| tp | 总降水 (lsrr+crr) | kg/m²/s | surface[4]+surface[5] | - |
| t2m | 2m温度 | K | surface[10] | - |
| olr | 顶层净长波辐射 | W/m² | surface[1] (avg_tnlwrf) | - |
| z500 | 500hPa位势高度 | m²/s² | - | upper[1, 2] |
| u850 | 850hPa纬向风 | m/s | - | upper[3, 4] |
| u200 | 200hPa纬向风 | m/s | - | upper[3, 0] |

#### 气候态文件

`Infer/eval/climatology_2002_2016.nc`: 52周气候平均值（2002-2016, 15年），用于计算异常和TCC。
- `tp_clim(week, lat, lon)`: 总降水气候态
- `t2m_clim(week, lat, lon)`: 2m温度气候态
- `olr_clim(week, lat, lon)`: OLR气候态
- `z500_clim(week, lat, lon)`: Z500气候态
- week为0-indexed week-of-year，与NC中的 `woy` 坐标对应

`Infer/eval/woy_map.npy`: Zarr全局索引 → week-of-year的映射数组。

#### 已生成文件清单

| 文件 | 大小 | 说明 |
|------|------|------|
| `Infer/eval/model_v3.nc` | 25 GB | V3.5: 6变量 (tp,t2m,olr,z500,u850,u200), 260周×6lead |
| `Infer/eval/model_v0.nc` | 13 GB | V0: 4变量 (tp,t2m,olr,z500), pred_olr为NaN |
| `Infer/eval/climatology_2002_2016.nc` | 398 MB | 52周气候态 (2002-2016, 15年) |
| `Infer/eval/woy_map.npy` | 8.7 KB | Zarr全局索引→week-of-year映射 |

#### 生成脚本

```bash
# 1. 气候态 (CPU)
cd /home/lhwang/Desktop/CanglongPhysics
/home/lhwang/anaconda3/envs/torch/bin/python Infer/compute_climatology.py

# 2. V3.5评估数据集 (GPU, 输出~25GB)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/lhwang/anaconda3/envs/torch/bin/python Infer/gen_eval_v3.py

# 3. V0评估数据集 (GPU, 输出~13GB, 必须从canglong/目录运行)
cd /home/lhwang/Desktop/CanglongPhysics/canglong
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/lhwang/anaconda3/envs/torch/bin/python ../Infer/gen_eval_v0.py
```

#### 离线指标计算（无需GPU）

```python
import xarray as xr
import numpy as np

ds = xr.open_dataset('Infer/eval/model_v3.nc')

# === 空间PCC (单样本, lead 1) ===
pred = ds['pred_t2m_lead1'].isel(time=0).values.ravel()
obs = ds['obs_t2m'].isel(time=0).values.ravel()
pcc = np.corrcoef(pred, obs)[0, 1]

# === 逐lead全球平均PCC ===
for lead in range(1, 7):
    pred = ds[f'pred_t2m_lead{lead}'].values  # (260, 721, 1440)
    obs = ds['obs_t2m'].values                # (260, 721, 1440)
    pccs = [np.corrcoef(pred[i].ravel(), obs[i].ravel())[0,1]
            for i in range(260)]
    print(f'lead{lead}: t2m PCC = {np.mean(pccs):.4f}')

# === RMSE ===
rmse = np.sqrt(np.mean((pred - obs)**2))
```

#### 跨模型对比

相同的目标周中心格式可用于组织任何预报模型的评估数据，方便统一对比：

```python
v0 = xr.open_dataset('Infer/eval/model_v0.nc')
v3 = xr.open_dataset('Infer/eval/model_v3.nc')

# 对比同一变量同一lead
for lead in range(1, 7):
    for name, ds in [('V0', v0), ('V3.5', v3)]:
        pred = ds[f'pred_t2m_lead{lead}'].values
        obs = ds['obs_t2m'].values
        pccs = [np.corrcoef(pred[i].ravel(), obs[i].ravel())[0,1]
                for i in range(260)]
        print(f'{name} lead{lead}: t2m PCC = {np.mean(pccs):.4f}')

# 扩展: 对比 Canglong vs NWP (相同格式)
# ecmwf = xr.open_dataset('Infer/eval/model_ecmwf.nc')
# diff = v3['pred_t2m_lead3'] - ecmwf['pred_t2m_lead3']
```

## MJO预测技巧评估

### 方法

MJO预测技巧通过双变量相关系数(bivariate COR)评估，基于实时多变量MJO指数(RMM)。

1. **EOF分解**: 对ERA5 2002-2016热带带(15°N-15°S) cos(lat)加权经向平均的OLR、U850、U200周异常场进行联合EOF分解
2. **RMM指数**: 将观测和预测的异常场投影到EOF1/EOF2上，得到RMM1和RMM2
3. **双变量COR**: `COR(τ) = Σ[a1·b1 + a2·b2] / √(Σ[a1²+a2²] · Σ[b1²+b2²])`，COR ≥ 0.5 为有效预测
4. **周尺度计算**: 所有变量在周平均尺度上处理

### EOF统计

- EOF1方差贡献: 23.8%, EOF2: 15.9%
- 场标准差: OLR=11.19 W/m², U850=1.58 m/s, U200=4.67 m/s

### V3.5 MJO COR (2017-2021)

| Lead | 1 week | 2 weeks | 3 weeks | 4 weeks | 5 weeks | 6 weeks |
|------|--------|---------|---------|---------|---------|---------|
| **COR** | 0.808 | 0.599 | 0.460 | 0.394 | 0.325 | 0.276 |
| **Skillful** | Yes | Yes | No | No | No | No |

**有效预测时限: ~2.7周** (COR > 0.5)

### 脚本与文件

```bash
# 首次运行需GPU (~50 min)，后续从缓存读取 (CPU)
cd /home/lhwang/Desktop/CanglongPhysics
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/lhwang/anaconda3/envs/torch/bin/python analysis/MJO/compute_mjo_skill.py
```

| 文件 | 说明 |
|------|------|
| `analysis/MJO/compute_mjo_skill.py` | MJO分析主脚本 (EOF + 推理 + RMM + COR + 绘图) |
| `analysis/MJO/mjo_cache_v35.npz` | RMM索引缓存 (后续分析无需GPU) |
| `analysis/MJO/mjo_cor_v35.png/svg` | COR vs Lead折线图 |
| `analysis/MJO/mjo_results_v35.csv` | COR数值表 |

注: V0模型因缺少OLR和200hPa层，无法计算标准RMM/MJO指标。

## 关键依赖

PyTorch, xarray, cartopy, salem, cmaps, rioxarray, timm, h5py

## 附录：Nature风格绘图参数

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

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
    'figure.figsize': (6, 4),
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
    'savefig.transparent': False
})
```
