# Physical Constraint Ablation Study

## 实验目标
分析5种物理约束模块对神经网络预测能力的影响，每组实验仅添加一种物理约束。

## 实验设置

| 实验编号 | 名称 | 损失函数 | Lambda |
|---------|------|---------|--------|
| 1 | Water Balance | `MSE + λ * L_water` | 8 |
| 2 | Energy Balance | `MSE + λ * L_energy` | 2e-5 |
| 3 | Hydrostatic Balance | `MSE + λ * L_pressure` | 5e-7 |
| 4 | Temperature Tendency | `MSE + λ * L_temperature` | 3e-2 |
| 5 | Navier-Stokes | `MSE + λ * L_momentum` | 1e1 |

## 训练配置
- **Epochs**: 20
- **Batch Size**: 1
- **Learning Rate**: 0.0005
- **Optimizer**: Adam
- **数据**: ERA5_2023_weekly_new.h5
  - 训练集: 0-28
  - 验证集: 28-40

## 记录指标 (每个epoch)

### 训练集
| 指标 | 说明 |
|-----|------|
| `train_mse_total` | surface + upper_air |
| `train_mse_surface` | 地表变量MSE |
| `train_mse_upper_air` | 高空变量MSE |
| `train_focus_loss` | 重要变量损失 (MJO相关) |
| `train_tweedie_loss` | Tweedie降水损失 |

### 验证集
| 指标 | 说明 |
|-----|------|
| `valid_mse_total` | surface + upper_air |
| `valid_mse_surface` | 地表变量MSE |
| `valid_mse_upper_air` | 高空变量MSE |
| `valid_focus_loss` | 重要变量损失 (MJO相关) |
| `valid_tweedie_loss` | Tweedie降水损失 |

## 物理约束说明

### 1. Water Balance (水量平衡)
- 方程: `ΔS_soil = P - E - R`
- 变量: swvl(土壤水), lsrr/crr(降水), slhf(蒸发), ro(径流)
- 区域: 外流流域 (hydrobasin_exorheic_mask)

### 2. Energy Balance (能量平衡)
- 方程: `R_n = LE + H + G`
- R_n = 净辐射 (sw_net + lw_net)
- LE = 潜热通量, H = 感热通量, G = 土壤热通量
- 区域: 陆地

### 3. Hydrostatic Balance (静力平衡)
- 方程: `Δφ = R_d * T_avg * ln(p1/p2)`
- 包含: 相邻气压层约束 + 地表到850hPa约束 + 海平面气压修正
- 区域: 全球

### 4. Temperature Tendency (温度局地变化)
- 方程: `∂T/∂t = -V_h·∇T - (绝热项)·ω + Q/c_p`
- 包含: 水平平流 + 垂直运动 + 非绝热加热
- 层权重: [0.5, 1.0, 1.0, 1.0, 1.0]

### 5. Navier-Stokes (动量方程)
- 方程: `∂u/∂t + 平流 = fv - ∂Φ/∂x + F_x`
- 包含: 科氏力 + 气压梯度力 + 摩擦力
- PGF系数: [0.99, 0.99, 0.93, 0.81, 0.39]

## 输出文件
- `ablation_exp1_water.csv`
- `ablation_exp2_energy.csv`
- `ablation_exp3_hydrostatic.csv`
- `ablation_exp4_temperature.csv`
- `ablation_exp5_momentum.csv`

## 运行命令
```bash
conda activate torch
python analysis/physical/ablation_experiments.py
```
