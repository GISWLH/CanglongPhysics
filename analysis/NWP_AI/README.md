# NWP / AI 统一评估数据集说明

这个目录用于定义 CAS-Canglong、FuXi、ECMWF、BoM、IFS 等不同模式在周尺度 S2S 评估时的统一数据组织格式。

核心目标不是保存某个模型专属的推理中间结果，而是生成一个可长期复用、可直接离线评估、可跨模型对齐比较的标准化评估数据集。外部案例可参考 `I:\FudanFuXi\target_example.nc`，但本仓库内的实际落地规范以 `Infer/eval` 现有实现为准。

## 1. 先看当前仓库里已有的几种数据模式

目前和评估直接相关的模式，实际有 4 类。

### 1.1 原始周数据源模式：ERA5 weekly Zarr

原始观测来自：

```text
/data/lhwang/ERA5_raw/weekly_data_zarr/ERA5_1982_2023_weekly.zarr
```

它是模型训练、推理、标准评估集生成的共同上游数据源，特点是：

- 按时间 chunk 存储，适合按周读取原始场
- 保存完整 surface / upper_air 变量，而不是只保存评估变量
- 更接近“模型输入数据源”，不是最终评估交换格式

这类数据适合：

- 模型训练
- 在线推理
- 生成标准评估集
- 计算气候态

这类数据不适合：

- 直接做多模型对比
- 直接做按 lead 的统一离线评估
- 直接给外部 NWP / AI 模型复用

### 1.2 在线滑窗样本模式：`infer_v*_2017_2021.py`

`Infer/infer_v3_2017_2021.py` 和 `Infer/infer_v0_2017_2021.py` 使用的是最直接的三周滑窗样本模式：

```text
[week t, week t+1] -> predict week t+2
```

每个 sample 本质上是一个训练/验证同构样本：

- 输入：前两周观测
- 输出：下一周真值
- 评估：在线算 PCC / ACC / RMSE

这个模式的优点是简单，适合：

- 快速检查 checkpoint
- 直接复用训练时的数据接口
- 做单步 one-step 预报能力测试

但它不适合作为统一评估数据标准，原因也很明确：

- 时间轴不是“目标周中心”，而是“样本窗口中心”
- 一个目标周在不同初始化下会被重复出现
- 多 lead 预报不在同一条连续 target time 轴上
- 跨模型对齐麻烦，不适合离线复评

### 1.3 标准离线评估模式：`Infer/eval/*.nc`

这是当前仓库里最重要、也最值得延续的模式。

现有脚本：

- `Infer/gen_eval_v3.py`
- `Infer/gen_eval_v0.py`

现有产物：

- `Infer/eval/model_v3.nc`
- `Infer/eval/model_v0.nc`

这个模式采用的是：

> 目标周中心，obs 存一份，pred 按 lead 展开为不同变量

也就是：

- 时间轴是连续目标周
- 对于每个目标周，只保存一份对应的观测
- 不同 lead 的预测，直接写成 `pred_xxx_lead1` 到 `pred_xxx_lead6`
- 任何指标都可以离线重算，不需要再跑 GPU 推理

这正是后续 NWP / AI 统一评估应当采用的标准格式。

### 1.4 辅助气候态模式：`climatology_2002_2016.nc` + `woy_map.npy`

现有脚本：

- `Infer/compute_climatology.py`
- `Infer/compute_tcc_v3.py`
- `Infer/compute_tcc_v0.py`

现有产物：

- `Infer/eval/climatology_2002_2016.nc`
- `Infer/eval/woy_map.npy`

它们不是模型预测文件，而是评估辅助数据：

- `climatology_2002_2016.nc`：52 周气候态，用于 anomaly / ACC / TCC
- `woy_map.npy`：全局时间索引到 week-of-year 的映射

因此，当前仓库的评估链条可以概括为：

```text
ERA5 weekly zarr
    -> infer_v*.py                # 在线滑窗评估
    -> gen_eval_v*.py             # 生成标准 target-week-centric 数据集
    -> compute_climatology.py     # 生成 52 周气候态和 woy 映射
    -> compute_tcc_v*.py          # 基于标准化结果离线计算 TCC
```

## 2. 我们要统一保留的标准：目标周中心

后续 `analysis/NWP_AI` 下讨论的标准评估数据集，统一采用：

> Target-Week-Centric，目标周中心组织格式

这个标准适用于：

- CAS-Canglong AI 模型
- FuXi S2S 数据
- ECMWF / BoM / IFS 等 NWP 模式
- 任意可被转成周尺度、且能映射到统一网格的预报系统

### 2.1 这样组织的优势

- 无冗余：同一目标周的观测只存一份
- 时间连续：2017-2021 的所有目标周可以组成一条完整时间序列
- 对比直观：同一目标周上，可以直接并列看 obs 和 lead1-6 的预测
- 评估通用：PCC、ACC、RMSE、TCC、区域平均、时间序列诊断都可离线复算
- 模型无关：AI / NWP 统一放进同一种文件结构

## 3. 周定义约定

这是整个标准里最需要写死的部分，否则不同来源数据很容易“周边界不一致”。

统一约定如下：

- 时间单位是连续自然日构成的 7 天周
- 每一年都从当年的 `1 月 1 日` 开始计第 1 个目标周
- 周编号采用每年内部的 `week of year`
- 不跨年
- 每年只保留前 `52` 周
- 年末不足 7 天的剩余日期直接舍去

这意味着：

- `woy` 使用 `0` 到 `51`
- `2017-2021` 共 `52 x 5 = 260` 个目标周
- 如果某年最后剩下 1 到 2 天，不并入下一年

具体例子：

- `2021-12-31` 直接舍去
- 不会把 `2021-12-31` 与 `2022-01-01` 到 `2022-01-06` 拼成跨年的一周
- `2022` 的 week 0 永远从 `2022-01-01` 重新开始

这也与当前 `Infer/compute_climatology.py` 中 `N_WEEKS = 52` 和 `woy_map.npy` 的生成逻辑一致：每年只给前 52 个周块分配 `woy`，其余周块不进入标准周索引。

## 4. 标准文件组织

### 4.1 推荐存储策略

标准评估文件建议按“一个模型一个文件”保存。

推荐逻辑结构如下：

```text
analysis/NWP_AI/
  README.md

重数据文件建议放在外部数据盘或共享存储，不进入 git：
  model_canglong_v3_2017_2021.nc
  model_canglong_v0_2017_2021.nc
  model_fuxi_2017_2021.nc
  model_ecmwf_2017_2021.nc
  model_bom_2017_2021.nc
  climatology_2002_2016.nc
  woy_map.npy
```

说明：

- `README.md` 保存在仓库中，负责定义规范
- `.nc` 和 `.npy` 大文件不建议提交到 git
- 文件名建议体现 `model name + period`
- 所有模型文件共享同一套目标周坐标定义

### 4.2 推荐命名

推荐文件名：

```text
model_{name}_{period}.nc
```

例如：

- `model_canglong_v3_2017_2021.nc`
- `model_fuxi_2017_2021.nc`
- `model_ecmwf_2017_2021.nc`

辅助文件：

- `climatology_2002_2016.nc`
- `woy_map.npy`

## 5. 标准数据结构

标准交换格式优先使用 NetCDF4，原因是：

- xarray 直接可读
- CF time 坐标成熟
- 对外共享方便
- 与当前 `Infer/eval/model_v*.nc` 实现一致

### 5.1 文件级约定

- 文件格式：NetCDF4
- `time` 使用 CF convention，可被 xarray 自动解码
- 主数据类型：`float32`
- 维度顺序统一为 `(time, lat, lon)`
- 压缩方式可参考当前实现：`zlib=True, complevel=4`
- chunk 建议按时间切块：`chunksizes=(1, lat, lon)`

### 5.2 标准维度

```text
Dimensions:
  time: 260  (2017-2021 连续目标周)
  lat:  721  (0.25°, 90N -> 90S)
  lon:  1440 (0.25°, 0E -> 359.75E)
```

说明：

- 这是本仓库当前主评估网格
- 若外部模式原始分辨率不同，先重网格到该统一网格，再写入标准文件

### 5.3 坐标变量

```text
Coordinates:
  time (time): datetime64 / CF-time
  lat  (lat):  float32, degrees_north
  lon  (lon):  float32, degrees_east
```

### 5.4 辅助坐标

```text
Auxiliary Coordinates:
  year (time): int32   - 目标周年份
  woy  (time): int32   - week-of-year, 0-indexed
  global_idx (time): int32 - 上游周数据源中的全局时间索引
```

其中：

- `year` 便于按年筛选
- `woy` 便于查 52 周气候态
- `global_idx` 便于回溯到上游周数据源

## 6. 标准变量设计

### 6.1 变量命名规则

对于每个评估变量 `{var}`：

```text
obs_{var}                # 目标周真值
pred_{var}_lead1         # 提前 1 周预报
pred_{var}_lead2
pred_{var}_lead3
pred_{var}_lead4
pred_{var}_lead5
pred_{var}_lead6
```

重要约束：

- 不设置单独的 `lead` 维
- lead 直接体现在变量名中
- 这样做更接近当前 `Infer/eval/model_v*.nc` 的实现
- 对多数离线分析脚本更直接

### 6.2 标准变量集合

当前主标准变量为：

```text
Data Variables (float32, dims = (time, lat, lon)):
  obs_tp
  pred_tp_lead1 ~ pred_tp_lead6

  obs_t2m
  pred_t2m_lead1 ~ pred_t2m_lead6

  obs_olr
  pred_olr_lead1 ~ pred_olr_lead6

  obs_z500
  pred_z500_lead1 ~ pred_z500_lead6

  obs_u850
  pred_u850_lead1 ~ pred_u850_lead6

  obs_u200
  pred_u200_lead1 ~ pred_u200_lead6
```

### 6.3 不同模型的变量覆盖

- `v3full / V3.5`：6 个 obs + 36 个 pred = 42 个变量
- `v0lite / V0`：4 个 obs + 24 个 pred = 28 个变量

当前仓库里的具体情况：

- `V3.5` 同时支持 `tp / t2m / olr / z500 / u850 / u200`
- `V0` 只支持 `tp / t2m / olr / z500`
- `V0` 的 `pred_olr_*` 实际写为 `NaN`，因为模型本身不预测 OLR
- `V0` 不包含 `u850 / u200`

因此，统一评估时遵循两个原则：

- 通用标准文件尽量使用完整变量集合
- 某模型没有的变量，可以显式写 `NaN` 或在文件级说明中声明缺失

如果要做跨模型横向对比，应默认对齐到“各模型共有变量的交集”。

## 7. `pred_lead{L}` 的严格语义

对于目标周 `t`，`pred_{var}_lead{L}` 表示：

> 提前 `L` 周启动、经过 `L` 步自回归后，到达目标周 `t` 的预报值

生成过程：

```text
初始化点: t - L - 1
输入两周: [t - L - 1, t - L]
自回归 L 步后到达目标周 t
```

对应关系如下：

| Lead | 初始化点 | obs 输入 | 自回归步数 | 纯 pred 输入步数 |
|------|----------|----------|------------|------------------|
| 1 | t - 2 | [t-2, t-1] | 1 | 0 |
| 2 | t - 3 | [t-3, t-2] | 2 | 1 |
| 3 | t - 4 | [t-4, t-3] | 3 | 2 |
| 4 | t - 5 | [t-5, t-4] | 4 | 3 |
| 5 | t - 6 | [t-6, t-5] | 5 | 4 |
| 6 | t - 7 | [t-7, t-6] | 6 | 5 |

示意：

```text
target week = t, pred_lead3
  Step 1: [obs_(t-4), obs_(t-3)]   -> pred_(t-2)
  Step 2: [obs_(t-3), pred_(t-2)]  -> pred_(t-1)
  Step 3: [pred_(t-2), pred_(t-1)] -> pred_t
```

因此：

- `lead` 越大，自回归链越长
- 输入中由模型预测替代观测的比例越高
- 误差会逐步累积
- 预报技巧通常随 lead 增大而下降

## 8. 评估变量定义

| 变量名 | 说明 | 物理单位 | Zarr Surface 索引 | Zarr Upper 索引 |
|--------|------|----------|-------------------|-----------------|
| `tp` | 总降水 (`lsrr + crr`) | `kg/m²/s` | `surface[4] + surface[5]` | - |
| `t2m` | 2m 温度 | `K` | `surface[10]` | - |
| `olr` | 顶层净长波辐射 | `W/m²` | `surface[1]` (`avg_tnlwrf`) | - |
| `z500` | 500hPa 位势高度 | `m²/s²` | - | `upper[1, 2]` |
| `u850` | 850hPa 纬向风 | `m/s` | - | `upper[3, 4]` |
| `u200` | 200hPa 纬向风 | `m/s` | - | `upper[3, 0]` |

## 9. 对 FuXi / NWP 日尺度数据的转换要求

FuXi 的原始数据是日尺度，不能直接并入上述周尺度标准文件，必须先做“日到周”的转换。

统一要求：

- 先按本说明中的周定义切分目标周
- 再把日尺度变量聚合到周尺度
- 最终写入的变量物理意义和单位必须与标准变量保持一致
- 写入前先完成统一网格化

聚合时需要额外注意：

- `t2m / z500 / u850 / u200 / olr` 这类状态量或平均通量，默认应写成目标周上的周平均值
- `tp` 在本标准中的单位是 `kg/m²/s`，语义应与当前 ERA5 周数据一致；如果原始数据是 `mm/day`、日累计降水或周累计降水，必须先换算后再写入
- 如果未来确实需要保存“周累计降水”，应使用新的变量名和单位，不要继续占用 `tp`

转换时最重要的不是“原始文件长什么样”，而是最终写入标准文件时必须满足：

- 相同的 `time / year / woy / global_idx`
- 相同的变量名
- 相同的物理单位
- 相同的目标周语义

只有这样，FuXi、Canglong、ECMWF 才能在同一套评估脚本下直接对比。

## 10. 推荐保留的辅助文件

### 10.1 `climatology_2002_2016.nc`

用于：

- ACC
- TCC
- anomaly time series

推荐变量：

```text
tp_clim
t2m_clim
olr_clim
z500_clim
u850_clim
u200_clim
count
```

维度：

```text
week: 52
lat: 721
lon: 1440
```

### 10.2 `woy_map.npy`

作用是把原始周数据源的全局时间索引映射到 `woy = 0..51`，便于：

- 回查目标周属于哪一个周序号
- 生成气候态
- 做 anomaly 匹配

## 11. 生成和接入新模型时的约定

新增一个模型时，推荐按当前脚本模式新增对应生成脚本，例如：

- `Infer/gen_eval_fuxi.py`
- `Infer/gen_eval_ecmwf.py`
- `Infer/gen_eval_bom.py`

生成脚本应当完成的事情只有三类：

- 读取该模型原始预报结果
- 转成统一的目标周中心格式
- 输出标准化 NetCDF 文件

不应在生成脚本里混入：

- 专用评估指标逻辑
- 画图逻辑
- 模型训练逻辑

## 12. 为什么不再采用“按初始化组织”的旧格式

旧格式通常长这样：

```text
init_time x lead x lat x lon
```

或者为每个初始化日期单独存一组预报结果。

这种方式的问题是：

- 同一个目标周的 obs 会被重复存很多次
- 时间序列分析不自然
- 对“同一目标周，不同 lead”横向比较不直观
- 跨模型统一麻烦

而目标周中心格式刚好反过来：

- 每个目标周只出现一次
- obs 唯一
- 所有 lead 直接并排
- 更适合 S2S 周尺度业务评估

## 13. 当前仓库中的对应实现

可以把下面几个脚本视为当前标准的实现参考：

- `Infer/compute_climatology.py`
  - 负责生成 `climatology_2002_2016.nc` 和 `woy_map.npy`
  - 明确采用每年 52 周、`woy` 从 0 开始的定义

- `Infer/gen_eval_v3.py`
  - 负责把 V3.5 自回归预报写成标准 `model_v3.nc`
  - 使用 `obs_{var}` 和 `pred_{var}_lead{1..6}` 命名

- `Infer/gen_eval_v0.py`
  - 负责把 V0 写成同样的标准格式
  - 演示了“模型变量不完整时如何处理”的情况

- `Infer/eval/README.md`
  - 给出了当前标准评估文件的英文版说明

## 14. 一句话结论

后续 `analysis/NWP_AI` 下所有 NWP / AI 对比数据，都应当统一为：

> 一模型一文件、目标周中心、obs 存一份、pred 按 lead 展开、每年固定 52 周且不跨年

这就是本项目后续标准评估数据集的唯一推荐组织格式。
