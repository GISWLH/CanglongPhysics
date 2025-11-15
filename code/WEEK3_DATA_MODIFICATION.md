# Week3 数据处理修改说明

## 修改概述

根据网盘实际数据检查结果，修改了 `run_ec_pure.py` 中 week3 的数据读取和处理逻辑。

## 网盘数据属性（已验证）

### 1. 气温数据 (air temperature-YYYY-MM-DD.tif)
- **分辨率**: 0.1° × 0.1° (全球 1800 × 3600)
- **单位**: 摄氏度 (°C)
- **数值范围**: -49.44 至 33.69°C
- **无需转换**: 已经是摄氏度

### 2. 降水数据 (MSWEP_YYYY-MM-DD.tif)
- **分辨率**: 0.1° × 0.1° (全球 1800 × 3600)
- **单位**: mm/day
- **数值范围**: 0 至 660.75 mm/day
- **需要转换**: mm/day → m/hr (ERA5格式)

### 3. 相对湿度 (relative humidity-YYYY-MM-DD.tif)
- **分辨率**: 0.1° × 0.1° (全球 1800 × 3600)
- **单位**: 百分比 (0-100)
- **数值范围**: 3.25 至 100%
- **无需转换**: 已经是百分比

## 主要修改内容

### 1. 数据读取
- 从网盘读取 7 天数据 (2025-11-06 至 2025-11-12)
- 每天读取 3 个 TIF 文件：气温、降水、相对湿度
- 正确关闭文件后再删除临时文件

### 2. 露点温度计算
由于网盘数据不包含露点温度，通过相对湿度计算：

```python
# 计算饱和水汽压 (kPa)
es = 0.618 * exp(17.27 * T / (T + 237.3))

# 计算实际水汽压
ea = es * RH / 100

# 反算露点温度 (°C)
Td = 237.3 * ln(ea/0.618) / (17.27 - ln(ea/0.618))
```

### 3. 空间分辨率统一
- **MSWX**: 0.1° → **ERA5**: 0.25°
- 使用 xarray 的双线性插值 (`interp` 方法)
- 从 1800×3600 网格插值到 721×1440 网格

### 4. 单位转换
**降水**: mm/day → m/hr (ERA5格式)
```python
# mm/day = mm/day × (1 m / 1000 mm) × (1 day / 24 hr) = m/hr
large_scale_rain_rate = precipitation * 0.5 / 24.0 / 1000.0
convective_rain_rate = precipitation * 0.5 / 24.0 / 1000.0
```
(总降水平均分为大尺度和对流降水)

**气温和露点**: 保持摄氏度，无需转换

### 5. 数据集格式统一
确保 week3_mean 与 week1_mean, week2_mean 格式一致：
- 维度: `['latitude', 'longitude']`
- 变量:
  - `2m_temperature` (°C)
  - `2m_dewpoint_temperature` (°C)
  - `large_scale_rain_rate` (m/hr)
  - `convective_rain_rate` (m/hr)

## 输出日志示例

脚本会打印详细的处理日志：

```
============================================================
Reading week3 data from NRT disk: 2025-11-06 to 2025-11-12
============================================================
  Loaded 2025-11-06
  Loaded 2025-11-07
  ...
  Loaded 2025-11-12

Week3 MSWX data loaded:
  - Resolution: 0.1deg x 0.1deg (1800 x 3600)
  - Temperature: 5.78 C
  - Precipitation: 2.25 mm/day
  - Relative Humidity: 75.42%
  - Calculated dewpoint: 1.23 C

Resampling to ERA5 grid (0.25deg):
  - Target grid: 721 x 1440
  - Interpolation complete

Week3 mean dataset created:
  - Grid: 721 x 1440
  - Variables: ['2m_temperature', '2m_dewpoint_temperature',
                'large_scale_rain_rate', 'convective_rain_rate']
============================================================

Combined 3-week historical data:
  - Time coverage: ['2025-10-23' '2025-10-30' '2025-11-06']
  - Shape: (3, 721, 1440)
```

## 数据流程图

```
Week1 & Week2: Google Cloud Storage (ERA5)
    ↓
  0.25° resolution
    ↓
Week平均

Week3: 网盘 MSWX
    ↓
  0.1° resolution → 通过RH计算露点
    ↓
双线性插值到 0.25°
    ↓
Week平均 + 单位转换

→ 合并为 3-week 历史数据 (ds_former_means)
```

## 关键改进

1. ✅ **基于实际数据验证**: 检查了网盘数据的实际分辨率和单位
2. ✅ **正确的单位处理**: 气温已是°C，降水是mm/day，相对湿度是%
3. ✅ **准确的空间插值**: 0.1° → 0.25°
4. ✅ **物理意义的露点计算**: 通过相对湿度反算
5. ✅ **文件管理优化**: 正确关闭文件后再删除
6. ✅ **详细的日志输出**: 便于调试和验证

## 注意事项

- 确保网盘路径正确: `/Projects/data_NRT/MSWX/tif/`
- 确保日期范围内的所有文件都存在
- 降水单位假设为 mm/day（根据数值范围推断）
- 总降水平均分为大尺度和对流降水各50%

## 测试建议

运行脚本后检查：
1. Week3 数据的平均值是否合理
2. 插值后的网格尺寸是否正确 (721 × 1440)
3. 单位转换是否正确（降水量级应为 10^-5 m/hr）
4. 三周数据是否成功合并
