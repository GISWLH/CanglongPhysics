# AGENTS.md

本文件用于维护 CanglongPhysics 项目的长期协作约定，替代旧版中大量一次性任务描述。

## 0. 运行与风格约定

- 默认 Python 环境：`conda activate torch`
- 运行慢任务时请使用足够长超时：`timeout` 建议大于 10 分钟
- 偏好 Notebook 风格代码：直接、可读，避免过度复杂的封装和 `main()` 套娃
- 可视化使用 `matplotlib`，字体为 Arial（Linux 需手动安装），图中标注使用英文

## 1. 项目目标

CanglongPhysics 聚焦“AI 天气预测 + 物理先验”：

- 基线网络：3D Transformer 风格天气预测模型
- 关键增强：风场先验驱动窗口注意力、PINN 物理约束
- 主要任务：S2S（次季节到季节）周尺度滚动预报（常用 6 周）

## 2. 模型版本体系（最新）

当前统一采用四代命名：

1. `v0lite`
2. `v1base`
3. `v2wind`
4. `v3full`

### 2.1 版本映射（代码文件）

- `v0lite`
  - 代表文件：`canglong/model_v0.py`、`code/run.py`
  - 特征：最早期轻量版，参数较少，历史上使用约 20 年数据训练
- `v1base`
  - 代表文件：`canglong/model_v1.py`、`train_v1.py`、`code_v2/model_v1.py`
  - 特征：扩展变量与层数，作为后续版本主干
- `v2wind`
  - 代表文件：`canglong/model_v2.py`、`model_v2_1.py`~`model_v2_5.py`、`train_v2*.py`
  - 特征：在 v1base 上加入风向感知窗口交换（wind-aware shift / mask）
- `v3full`
  - 代表文件：`canglong/model_v3.py`、`train_v3.py`、`train_v3_5.py`
  - 特征：在 v2wind 基础上引入 PINN 物理约束损失（软约束）

### 2.2 命名规则

后续文档与讨论优先使用 `v0lite/v1base/v2wind/v3full`，而不是“V2.3/V3.5”这类分支编号。
分支编号仅用于具体实验脚本区分。

## 3. 数据配置与变量规范

## 3.1 v0lite（历史轻量配置）

- Surface 变量数：16
- Upper Air 变量数：7
- 压力层数：4（300, 500, 700, 850 hPa）
- 参考脚本：`code/run.py`

### v0lite Surface 变量（16）

```python
surface_vars_v0 = [
    "large_scale_rain_rate",
    "convective_rain_rate",
    "total_column_cloud_ice_water",
    "total_cloud_cover",
    "top_net_solar_radiation_clear_sky",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_dewpoint_temperature",
    "2m_temperature",
    "surface_latent_heat_flux",
    "surface_sensible_heat_flux",
    "surface_pressure",
    "volumetric_soil_water_layer",
    "mean_sea_level_pressure",
    "sea_ice_cover",
    "sea_surface_temperature",
]
```

### v0lite Upper Air 变量（7）

```python
upper_vars_v0 = [
    "geopotential",
    "vertical_velocity",
    "u_component_of_wind",
    "v_component_of_wind",
    "fraction_of_cloud_cover",
    "temperature",
    "specific_humidity",
]
levels_v0 = [300, 500, 700, 850]
```

## 3.2 v1base/v2wind/v3full（当前主配置）

- Surface 变量数：26
- Upper Air 变量数：10
- 压力层数：5（200, 300, 500, 700, 850 hPa）

### Surface 变量顺序（严格）

```python
surf_vars = [
    "avg_tnswrf", "avg_tnlwrf", "tciw", "tcc", "lsrr", "crr", "blh",
    "u10", "v10", "d2m", "t2m", "avg_iews", "avg_inss", "slhf", "sshf",
    "avg_snswrf", "avg_snlwrf", "ssr", "str", "sp", "msl", "siconc",
    "sst", "ro", "stl", "swvl"
]
```

### Upper Air 变量顺序（严格）

```python
upper_vars = ["o3", "z", "t", "u", "v", "w", "q", "cc", "ciwc", "clwc"]
levels = [200, 300, 500, 700, 850]
```

### 主配置张量维度

- Surface 输入：`(B, 26, 2, 721, 1440)`
- Surface 输出：`(B, 26, 1, 721, 1440)`
- Upper 输入：`(B, 10, 5, 2, 721, 1440)`
- Upper 输出：`(B, 10, 5, 1, 721, 1440)`
- 标准化参数：
  - `surface_mean/std`: `(1, 26, 1, 721, 1440)`
  - `upper_mean/std`: `(1, 10, 5, 1, 721, 1440)`

### 加权土壤变量（主配置）

- `swvl` 与 `stl` 采用四层厚度加权，厚度：
  - `d1=0.07`, `d2=0.21`, `d3=0.72`, `d4=1.89`, 总深度 `2.89 m`

## 4. 模型结构摘要（v1base 及以上）

输入由三部分组成：

1. Upper Air 分支（4D patch embed）
2. Surface 分支（3D encoder）
3. Earth constant 分支（static mask/topography 等）

典型流程：

- 编码后拼接为 6 个“层切片”（`upper 3 + surface 2 + constant 1`）
- 经过 Earth-specific Swin Transformer
- 输出拆分：
  - `output_upper_air = output[:, :, :3, :, :]`
  - `output_surface = output[:, :, 3:5, :, :]`
- 最终恢复到物理变量空间，时间维压缩到 1

## 5. v2wind：风向感知窗口机制

在 v1base 上新增风场先验引导注意力窗口交换：

- 关键信号来源：
  - Upper: `u/v` = `upper[:, 3:5, ...]`
  - Surface: `u10/v10` = `surface[:, 7:9, ...]`
- 在 encoder 之前计算粗粒度主导风向（常见 4x4 下采样到 181x360 网格）
- 按风向选择对应窗口移位/掩码进行注意力计算

实现上允许多种子策略（`v2_1 ... v2_5`），但都归类为 `v2wind`。

## 6. v3full：PINN 物理约束

核心思想：将物理方程残差作为软约束加入总损失。

```text
L_total = L_MSE + λ_water*L_water + λ_energy*L_energy + λ_pressure*L_pressure + ...
```

### 核心约束（最低要求）

1. 水量平衡（基于 `swvl`, `lsrr+crr`, `slhf`，可结合流域掩码）
2. 能量平衡（`R_n = LE + H + G`，结合土壤热容与陆地掩码）
3. 气压/静力相关约束（`sp`, `msl`, `t2m`, DEM）

### 常量掩码

- `constant_masks/is_land.pt`
- `constant_masks/hydrobasin_exorheic_mask.pt`
- `constant_masks/csol_bulk_025deg_721x1440_corrected.pt`
- `constant_masks/DEM.pt`

## 7. 训练规范（重要）

### 7.1 模型与损失分离

- 模型类职责：仅做前向预测（`output_surface`, `output_upper_air`）
- 训练脚本职责：定义并组合 `MSE + 物理损失`
- 不要把 loss 逻辑塞入主模型 `forward()`

### 7.2 标准化统一规范

统一使用 `convert_dict_to_pytorch_arrays_v2.load_normalization_arrays` 返回的四个数组，
保持原始维度，不做 `squeeze`。

```python
surface_mean, surface_std, upper_mean, upper_std = load_normalization_arrays(json_path)
```

推荐广播做法（主配置）：

```python
input_surface_norm = (input_surface.permute(0, 2, 1, 3, 4) - surface_mean) / surface_std
input_upper_norm = (input_upper_air.permute(0, 2, 3, 1, 4, 5) - upper_mean) / upper_std
target_surface_norm = (target_surface.unsqueeze(2) - surface_mean) / surface_std
target_upper_norm = (target_upper_air.unsqueeze(3) - upper_mean) / upper_std
```

反标准化：

```python
surface_physical = surface_norm * surface_std + surface_mean
upper_physical = upper_norm * upper_std + upper_mean
```

## 8. 预报与检验（业务流程）

常规检验聚焦 CAS-Canglong vs ECMWF：

- 指标：`RMSE`, `ACC`, `SPEI` 同号率
- 变量：2m 温度、降水（必要时再扩展）
- 单位统一：
  - Canglong `t2m`：K -> degC
  - Canglong `tp`：`m/hr -> mm/day`
  - ECMWF 通常已是 `degC` 与 `mm/day`
- 空间统一：
  - ECMWF 常为中国区、1.5°
  - CAS-Canglong/观测常为全球、0.25°
  - 评估前统一重网格到 ECMWF 网格

## 9. 代码组织建议

- `canglong/`：模型定义与模块
- `train_v1.py`：v1base 主训练入口
- `train_v2*.py`：v2wind 各实验入口
- `train_v3.py` / `train_v3_5.py`：v3full 物理约束训练入口
- `code/run.py`：v0lite 风格推理/业务流程脚本

建议保持“一个脚本对应一个实验配置”的方式，避免把过多实验开关堆在单脚本里。

## 10. 维护原则

- AGENTS.md 仅保留“长期有效”的项目规范
- 临时开发任务不要写入本文件（例如“现在去改某个脚本第几行”）
- 版本演进时优先更新第 2~7 节，确保新同事可快速上手

