# Physical Constraints Ablation Analysis

## 概述

这个目录包含对Canglong天气预测模型中各个物理约束贡献的消融实验分析。

## 实验设计

### 实验组设置

| 实验ID | 实验名称 | 物理约束配置 | 说明 |
|--------|---------|-------------|------|
| Exp-1 | base0 | 无风向约束 | 最基础的模型 |
| Exp0 | baseline | 仅风向约束 | 基线模型 |
| Exp1 | water | baseline + 水量平衡 | 单独测试水量平衡约束 |
| Exp2 | energy | baseline + 能量平衡 | 单独测试能量平衡约束 |
| Exp3 | hydrostatic | baseline + 静力平衡 | 单独测试静力平衡约束 |
| Exp4 | temperature | baseline + 温度局地变化 | 单独测试温度局地变化方程 |
| Exp5 | momentum | baseline + 动量方程 | 单独测试动量方程约束 |
| Exp6 | full | baseline + 所有物理约束 | 完整模型（所有约束） |

### 物理约束详解

1. **水量平衡约束 (Water Balance)**
   - 方程：∆Soil water = P_total − E − R
   - 包含陆地、海洋、大气水量平衡
   - 在外流流域（exorheic basins）计算

2. **能量平衡约束 (Energy Balance)**
   - 方程：R_n = LE + H + G
   - 辐射平衡与热量分配

3. **静力平衡约束 (Hydrostatic Balance)**
   - 垂直方向压力-温度-高度关系
   - 应用于多个压力层之间

4. **温度局地变化方程 (Temperature Local Change)**
   - 温度时间演变的物理一致性

5. **动量方程约束 (Momentum Equation)**
   - 风场演变的动力学约束

## 主要发现

### MSE改进总结 (Epoch 20)

| 约束类型 | MSE改进 (Δ MSE) | 改进百分比 |
|---------|----------------|----------|
| 水量平衡 | -0.045 | **退化** 4.1% |
| 能量平衡 | -0.107 | **退化** 9.7% |
| 静力平衡 | -0.238 | **退化** 21.7% |
| 温度局地变化 | -0.014 | **退化** 1.3% |
| 动量方程 | -0.032 | **退化** 2.9% |
| **全部组合** | **-0.167** | **退化** 15.3% |

> **注意**: 负值表示相对于baseline的MSE增加（即性能下降）

### 关键洞察

1. **单独物理约束的负面影响**
   - 所有单独的物理约束都导致MSE增加
   - 静力平衡约束影响最大（-0.238）
   - 温度局地变化约束影响最小（-0.014）

2. **组合效应**
   - 全部物理约束组合后的退化（-0.167）小于各单独约束退化之和
   - 说明物理约束之间存在互补效应
   - 但整体仍未达到baseline性能

3. **训练动态**
   - 随着训练进行（Epoch 5→20），所有实验的MSE都在下降
   - 但相对性能关系保持一致
   - baseline始终表现最好

### 可能的原因分析

1. **过度约束问题**
   - 物理约束可能过于严格，限制了模型的学习能力
   - 约束权重（λ参数）可能需要进一步调整

2. **数据-模型不匹配**
   - ERA5数据可能本身不完全满足这些理想化的物理方程
   - 模型架构可能需要调整以更好地整合物理约束

3. **训练策略**
   - 可能需要更长的训练时间让物理约束发挥作用
   - 可以考虑课程学习：先训练baseline，再逐步加入物理约束

## 文件说明

### 脚本文件
- `ablation_physical_constraints.py`: 主分析脚本
  - 从消融实验CSV文件加载数据
  - 计算各物理约束的增量贡献
  - 生成可视化图表

### 数据文件
- `ablation_results/`: 原始消融实验结果（CSV格式）
  - 包含每个epoch的详细指标（MSE, ACC, 物理闭合率等）

### 输出文件
- `figures/physical_constraints_ablation.png`: 主要分析图（双子图）
  - 左图：各物理约束的独立贡献柱状图
  - 右图：baseline vs full的累积效果对比
- `figures/physical_constraints_ablation.svg`: 矢量图版本
- `figures/physical_constraints_contributions.csv`: 数值结果表格

## 使用方法

```bash
# 激活环境
conda activate torch

# 运行分析脚本
python analysis/physical/ablation_physical_constraints.py
```

## 后续工作建议

1. **约束权重优化**
   - 使用网格搜索或贝叶斯优化调整λ参数
   - 考虑自适应权重策略

2. **渐进式约束**
   - 实施课程学习策略
   - 在训练后期逐步增加物理约束权重

3. **软约束vs硬约束**
   - 考虑将某些约束从损失函数移到模型架构中
   - 探索不同的约束实现方式

4. **物理一致性诊断**
   - 深入分析为什么物理约束导致性能下降
   - 检查物理闭合率指标（已在CSV中记录）

5. **混合策略**
   - 仅在特定变量或区域应用物理约束
   - 选择性约束：只保留有益的物理约束

## 引用

如果使用此分析，请引用：
```
Canglong Weather Prediction Model - Physical Constraints Ablation Study
CAS-Canglong Team, 2025
```
