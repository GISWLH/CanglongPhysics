# Hindcast 2022-2023 回报脚本使用说明

## 脚本功能

`hindcast_22_23.py` 用于回报2022-2023年全年（2年×52周=104周）的天气数据，并计算SPEI指数。

## 主要特性

- ✅ 自动处理2022-2023年每一周（共104周）
- ✅ 使用CAS-Canglong模型进行6周滚动预报
- ✅ 输出全部surface和upper air变量
- ✅ 自动计算SPEI指数
- ✅ 结果保存到Z盘

## 运行要求

### 环境
```bash
conda activate torch
```

### 数据要求
- ERA5数据访问权限（通过Google Cloud Storage）
- 模型文件：`F:/model/weather_model_epoch_500.pt`
- 常量数据：`../constant_masks/input_tensor.pt`
- 气候态数据：`../data/climate_variables_2000_2023_weekly.nc`

## 运行方式

```bash
python hindcast_22_23.py
```

**注意**: 由于需要处理104周的数据，运行时间较长（预计数小时到数天，取决于硬件配置）。

## 输出文件

### 文件结构
```
Z:/Data/hindcast_2022_2023/
├── hindcast_2022_week01_surface_2022-01-01.nc
├── hindcast_2022_week01_upper_2022-01-01.nc
├── hindcast_2022_week02_surface_2022-01-08.nc
├── hindcast_2022_week02_upper_2022-01-08.nc
├── ...
├── hindcast_2023_week52_surface_2023-12-24.nc
├── hindcast_2023_week52_upper_2023-12-24.nc
└── hindcast_index_2022_2023.csv
```

### 文件命名规则
- **Surface变量**: `hindcast_{year}_week{num:02d}_surface_{start_date}.nc`
- **Upper air变量**: `hindcast_{year}_week{num:02d}_upper_{start_date}.nc`
- **索引文件**: `hindcast_index_2022_2023.csv`

### 包含的变量

#### Surface变量 (16个)
1. large_scale_rain_rate (mm/day)
2. convective_rain_rate (mm/day)
3. total_column_cloud_ice_water
4. total_cloud_cover
5. top_net_solar_radiation_clear_sky
6. 10m_u_component_of_wind
7. 10m_v_component_of_wind
8. 2m_dewpoint_temperature (°C)
9. 2m_temperature (°C)
10. surface_latent_heat_flux
11. surface_sensible_heat_flux
12. surface_pressure
13. volumetric_soil_water_layer
14. mean_sea_level_pressure
15. sea_ice_cover
16. sea_surface_temperature

#### 额外计算变量
- **total_precipitation**: 总降水量 (mm/day)
- **potential_evapotranspiration**: 潜在蒸散发
- **spei**: 标准化降水蒸散指数

#### Upper air变量 (7个, 4个层级: 300, 500, 700, 850 hPa)
1. geopotential
2. vertical_velocity
3. u_component_of_wind
4. v_component_of_wind
5. fraction_of_cloud_cover
6. temperature
7. specific_humidity

### 数据维度
- **时间**: 6周预报 (每周一个时间点)
- **空间**: 全球 0.25° (721×1440)
- **层级**: 4个气压层 (仅upper air)

## 数据格式

每个nc文件包含：
- **坐标**: time, latitude, longitude (upper air还包含level)
- **变量**: 见上述列表
- **属性**:
  - forecast_start_date: 预报起始日期
  - forecast_end_date: 预报结束日期
  - input_period: 输入数据时间范围
  - year: 年份
  - week_number: 周数

## 索引文件

`hindcast_index_2022_2023.csv` 包含所有回报任务的元数据：
- year: 年份
- week: 周数
- start_date: 起始日期
- surface_file: surface数据文件路径
- upper_file: upper air数据文件路径

## 错误处理

脚本包含异常处理，如果某一周处理失败，会：
1. 打印错误信息
2. 跳过该周
3. 继续处理下一周

## 预期运行时间

- 单周处理时间: ~3-10分钟（取决于网络速度和GPU性能）
- 总运行时间: ~5-17小时（104周）

## 监控进度

脚本会实时显示进度：
```
处理 2022 年...
2022年回报: 100%|██████████| 52/52 [XX:XX<00:00, X.XXs/it]
  Week 1 (2022-01-01) completed
  Week 2 (2022-01-08) completed
  ...
```

## 常见问题

### 1. 内存不足
如果遇到内存错误，可以：
- 减少同时处理的周数
- 增加系统交换空间
- 使用更大内存的机器

### 2. 网络超时
ERA5数据加载可能因网络问题超时，脚本会自动跳过并继续。

### 3. 磁盘空间
确保Z盘有足够空间（预计每周~2-5GB，总计~200-500GB）。

## 后续分析

使用生成的数据进行分析：
```python
import xarray as xr
import pandas as pd

# 读取索引
index = pd.read_csv('Z:/Data/hindcast_2022_2023/hindcast_index_2022_2023.csv')

# 读取某一周的数据
week_data = xr.open_dataset(index.loc[0, 'surface_file'])

# 查看SPEI
spei = week_data['spei']
```

## 作者信息

基于 run.py 改编，用于2022-2023年批量回报任务。
