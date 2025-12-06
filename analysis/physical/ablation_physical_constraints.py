"""
Physical Constraints Ablation Analysis
分析各个物理约束对模型性能的贡献

从消融实验结果中提取MSE数据，可视化各物理约束的增量贡献
Experiments:
  - Exp-1 (base0): 基础模型（无风向约束）
  - Exp0 (baseline): 基础模型+风向约束
  - Exp1 (water): baseline + 水量平衡
  - Exp2 (energy): baseline + 能量平衡
  - Exp3 (hydrostatic): baseline + 静力平衡
  - Exp4 (temperature): baseline + 温度局地变化
  - Exp5 (momentum): baseline + 动量方程
  - Exp6 (full): baseline + 全部物理约束
"""

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager


# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ABLATION_DIR = PROJECT_ROOT / 'analysis' / 'physical' / 'ablation_results'
OUTPUT_DIR = PROJECT_ROOT / 'analysis' / 'physical' / 'figures'


def configure_fonts():
    """配置Arial字体"""
    font_path = Path('/usr/share/fonts/arial/ARIAL.TTF')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams['font.family'] = font_name

    # 设置Nature风格参数
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['svg.hashsalt'] = 'hello'

    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 600,
        'figure.figsize': (8, 5),
        'lines.linewidth': 1.5,
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


def load_ablation_results(ablation_dir: Path, epochs_to_extract=[5, 10, 15, 20]):
    """
    加载消融实验结果

    Args:
        ablation_dir: 消融实验结果目录
        epochs_to_extract: 要提取的epoch列表

    Returns:
        dict: {experiment_name: {epoch: {metric: value}}}
    """
    experiments = {
        'base0': 'ablation_exp-1_base0_v2.csv',
        'baseline': 'ablation_exp0_baseline_v2.csv',
        'water': 'ablation_exp1_water_v2.csv',
        'energy': 'ablation_exp2_energy_v2.csv',
        'hydrostatic': 'ablation_exp3_hydrostatic_v2.csv',
        'temperature': 'ablation_exp4_temperature_v2.csv',
        'momentum': 'ablation_exp5_momentum_v2.csv',
        'full': 'ablation_exp6_full_v2.csv'
    }

    results = {}

    for exp_name, filename in experiments.items():
        filepath = ablation_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found, skipping...")
            continue

        df = pd.read_csv(filepath)
        results[exp_name] = {}

        for epoch in epochs_to_extract:
            if epoch in df['epoch'].values:
                row = df[df['epoch'] == epoch].iloc[0]
                results[exp_name][epoch] = {
                    'valid_mse_total': row['valid_mse_total'],
                    'valid_mse_surface': row['valid_mse_surface'],
                    'valid_mse_upper_air': row['valid_mse_upper_air'],
                    'valid_acc_surface': row.get('valid_acc_surface', np.nan),
                    'valid_acc_upper': row.get('valid_acc_upper', np.nan)
                }

    return results


def calculate_constraint_contributions(results, epochs=[5, 10, 15, 20]):
    """
    计算各个物理约束的增量贡献

    策略1 - 传统方法（相对于baseline）：
    - baseline = 基础模型（含风向约束）
    - 每个单独物理约束的贡献 = (baseline_MSE - constraint_MSE)
    - 组合效果 = full_MSE - baseline_MSE

    策略2 - 基于第一个epoch的MSE排名分配贡献：
    - MSE越低 → 该约束效果越好 → 贡献越大
    - 使用倒数排序来计算贡献占比

    Returns:
        DataFrame with contributions for each constraint
    """
    data = []

    for epoch in epochs:
        baseline_mse = results['baseline'][epoch]['valid_mse_total']

        # 获取各约束实验的MSE
        water_mse = results['water'][epoch]['valid_mse_total']
        energy_mse = results['energy'][epoch]['valid_mse_total']
        hydrostatic_mse = results['hydrostatic'][epoch]['valid_mse_total']
        temperature_mse = results['temperature'][epoch]['valid_mse_total']
        momentum_mse = results['momentum'][epoch]['valid_mse_total']
        full_mse = results['full'][epoch]['valid_mse_total']

        # 策略1：传统增量贡献（相对于baseline的改进）
        water_contrib = baseline_mse - water_mse
        energy_contrib = baseline_mse - energy_mse
        hydrostatic_contrib = baseline_mse - hydrostatic_mse
        temperature_contrib = baseline_mse - temperature_mse
        momentum_contrib = baseline_mse - momentum_mse
        full_contrib = baseline_mse - full_mse

        # 策略2：基于MSE排名的贡献占比
        # MSE越低，约束效果越好，贡献越大
        # 使用 1/MSE 作为权重，然后归一化
        mse_dict = {
            'water': water_mse,
            'energy': energy_mse,
            'hydrostatic': hydrostatic_mse,
            'temperature': temperature_mse,
            'momentum': momentum_mse
        }

        # 计算倒数权重（MSE越小，权重越大）
        inverse_weights = {name: 1.0/mse for name, mse in mse_dict.items()}
        total_inverse = sum(inverse_weights.values())

        # 归一化得到贡献占比（百分比）
        contribution_pct = {name: (weight/total_inverse)*100
                           for name, weight in inverse_weights.items()}

        data.append({
            'epoch': epoch,
            'baseline_mse': baseline_mse,
            'full_mse': full_mse,

            # 各约束的MSE
            'water_mse': water_mse,
            'energy_mse': energy_mse,
            'hydrostatic_mse': hydrostatic_mse,
            'temperature_mse': temperature_mse,
            'momentum_mse': momentum_mse,

            # 传统增量贡献
            'water_contribution': water_contrib,
            'energy_contribution': energy_contrib,
            'hydrostatic_contribution': hydrostatic_contrib,
            'temperature_contribution': temperature_contrib,
            'momentum_contribution': momentum_contrib,
            'full_contribution': full_contrib,

            # 基于MSE的贡献占比（%）
            'water_contribution_pct': contribution_pct['water'],
            'energy_contribution_pct': contribution_pct['energy'],
            'hydrostatic_contribution_pct': contribution_pct['hydrostatic'],
            'temperature_contribution_pct': contribution_pct['temperature'],
            'momentum_contribution_pct': contribution_pct['momentum']
        })

    return pd.DataFrame(data)


def plot_constraint_contributions(contributions_df, output_dir: Path):
    """
    绘制物理约束贡献的可视化（三子图）

    Args:
        contributions_df: 包含各约束贡献的DataFrame
        output_dir: 输出目录
    """
    configure_fonts()

    epochs = contributions_df['epoch'].values
    x = np.arange(len(epochs))

    # 提取各约束的贡献
    water = contributions_df['water_contribution'].values
    energy = contributions_df['energy_contribution'].values
    hydrostatic = contributions_df['hydrostatic_contribution'].values
    temperature = contributions_df['temperature_contribution'].values
    momentum = contributions_df['momentum_contribution'].values

    baseline_mse = contributions_df['baseline_mse'].values
    full_mse = contributions_df['full_mse'].values

    # 提取基于MSE的贡献占比（只用第一个epoch，即Epoch 5）
    water_pct = contributions_df['water_contribution_pct'].values
    energy_pct = contributions_df['energy_contribution_pct'].values
    hydrostatic_pct = contributions_df['hydrostatic_contribution_pct'].values
    temperature_pct = contributions_df['temperature_contribution_pct'].values
    momentum_pct = contributions_df['momentum_contribution_pct'].values

    # 创建图形（三子图）
    fig = plt.figure(figsize=(18, 5))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    # ========== 图1：单个约束的独立贡献（传统方法）==========
    width = 0.15
    colors = {
        'water': '#66c2a5',
        'energy': '#fc8d62',
        'hydrostatic': '#8da0cb',
        'temperature': '#e78ac3',
        'momentum': '#a6d854'
    }

    ax1.bar(x - 2*width, water, width, label='Water Balance', color=colors['water'])
    ax1.bar(x - width, energy, width, label='Energy Balance', color=colors['energy'])
    ax1.bar(x, hydrostatic, width, label='Hydrostatic Balance', color=colors['hydrostatic'])
    ax1.bar(x + width, temperature, width, label='Temperature Change', color=colors['temperature'])
    ax1.bar(x + 2*width, momentum, width, label='Momentum Equation', color=colors['momentum'])

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Epoch {e}' for e in epochs])
    ax1.set_ylabel('MSE Reduction (relative to baseline)')
    ax1.set_title('(a) Individual Constraint Contributions')
    ax1.legend(loc='upper left', frameon=False, fontsize=8)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.axhline(y=0, color='black', linewidth=0.8, linestyle='-')

    # ========== 图2：累积效果对比 ==========
    # 绘制baseline和full的MSE对比
    width2 = 0.35

    ax2.bar(x - width2/2, baseline_mse, width2, label='Baseline (with wind)',
            color='#8da0cb', alpha=0.7)
    ax2.bar(x + width2/2, full_mse, width2, label='Full (all constraints)',
            color='#fc8d62', alpha=0.7)

    # 标注MSE值
    for i, (base, full) in enumerate(zip(baseline_mse, full_mse)):
        ax2.text(x[i] - width2/2, base + 0.02, f'{base:.3f}',
                ha='center', va='bottom', fontsize=7)
        ax2.text(x[i] + width2/2, full + 0.02, f'{full:.3f}',
                ha='center', va='bottom', fontsize=7)

        # 绘制改进箭头
        improvement = base - full
        ax2.annotate('', xy=(x[i] + width2/2, full),
                    xytext=(x[i] - width2/2, base),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.6))

        # 标注改进百分比
        improvement_pct = (improvement / base) * 100
        mid_y = (base + full) / 2
        ax2.text(x[i], mid_y, f'-{improvement_pct:.1f}%',
                ha='center', va='center', fontsize=7, color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Epoch {e}' for e in epochs])
    ax2.set_ylabel('Validation MSE (Total)')
    ax2.set_title('(b) Baseline vs Full Constraints')
    ax2.legend(loc='upper right', frameon=False, fontsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # ========== 图3：基于MSE的贡献占比（饼图，使用Epoch 5数据）==========
    # 使用第一个epoch的贡献占比
    sizes = [water_pct[0], energy_pct[0], hydrostatic_pct[0],
             temperature_pct[0], momentum_pct[0]]
    labels = ['Water\nBalance', 'Energy\nBalance', 'Hydrostatic\nBalance',
              'Temperature\nChange', 'Momentum\nEquation']
    colors_list = [colors['water'], colors['energy'], colors['hydrostatic'],
                   colors['temperature'], colors['momentum']]

    # 按贡献从大到小排序
    sorted_indices = np.argsort(sizes)[::-1]
    sizes_sorted = [sizes[i] for i in sorted_indices]
    labels_sorted = [labels[i] for i in sorted_indices]
    colors_sorted = [colors_list[i] for i in sorted_indices]

    wedges, texts, autotexts = ax3.pie(sizes_sorted, labels=labels_sorted,
                                        colors=colors_sorted,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        textprops={'fontsize': 9})

    # 加粗百分比文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)

    ax3.set_title(f'(c) Constraint Contribution Share\n(based on MSE at Epoch {epochs[0]})')

    plt.tight_layout()

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / 'physical_constraints_ablation.png'
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f'Saved figure to {output_path}')

    # 同时保存SVG格式
    output_path_svg = output_dir / 'physical_constraints_ablation.svg'
    fig.savefig(output_path_svg, format='svg', bbox_inches='tight')
    print(f'Saved SVG to {output_path_svg}')

    plt.close(fig)


def print_summary_table(contributions_df):
    """打印汇总表格"""
    print("\n" + "="*100)
    print("Physical Constraints Ablation Summary")
    print("="*100)

    print("\n[Table 1] Traditional Contributions (relative to baseline)")
    print(f"\n{'Epoch':<10} {'Baseline':<12} {'Water':<10} {'Energy':<10} {'Hydro':<10} {'Temp':<10} {'Momentum':<10} {'Full':<12}")
    print(f"{'':10} {'MSE':<12} {'Δ MSE':<10} {'Δ MSE':<10} {'Δ MSE':<10} {'Δ MSE':<10} {'Δ MSE':<10} {'MSE':<12}")
    print("-"*100)

    for _, row in contributions_df.iterrows():
        print(f"{row['epoch']:<10.0f} "
              f"{row['baseline_mse']:<12.4f} "
              f"{row['water_contribution']:<10.4f} "
              f"{row['energy_contribution']:<10.4f} "
              f"{row['hydrostatic_contribution']:<10.4f} "
              f"{row['temperature_contribution']:<10.4f} "
              f"{row['momentum_contribution']:<10.4f} "
              f"{row['full_mse']:<12.4f}")

    print("\n" + "-"*100)
    print("\n[Table 2] MSE-based Contribution Share (%) - Lower MSE = Higher Contribution")
    print(f"\n{'Epoch':<10} {'Water':<15} {'Energy':<15} {'Hydro':<15} {'Temp':<15} {'Momentum':<15}")
    print(f"{'':10} {'MSE / Share%':<15} {'MSE / Share%':<15} {'MSE / Share%':<15} {'MSE / Share%':<15} {'MSE / Share%':<15}")
    print("-"*100)

    for _, row in contributions_df.iterrows():
        print(f"{row['epoch']:<10.0f} "
              f"{row['water_mse']:.3f}/{row['water_contribution_pct']:.1f}%  "
              f"{row['energy_mse']:.3f}/{row['energy_contribution_pct']:.1f}%  "
              f"{row['hydrostatic_mse']:.3f}/{row['hydrostatic_contribution_pct']:.1f}%  "
              f"{row['temperature_mse']:.3f}/{row['temperature_contribution_pct']:.1f}%  "
              f"{row['momentum_mse']:.3f}/{row['momentum_contribution_pct']:.1f}%")

    print("\n" + "="*100)

    # 特别强调第一个epoch的贡献排名
    first_row = contributions_df.iloc[0]
    print(f"\n[Key Finding] Constraint Ranking at Epoch {int(first_row['epoch'])} (MSE-based):")
    print("-"*100)

    constraint_data = [
        ('Water Balance', first_row['water_mse'], first_row['water_contribution_pct']),
        ('Energy Balance', first_row['energy_mse'], first_row['energy_contribution_pct']),
        ('Hydrostatic Balance', first_row['hydrostatic_mse'], first_row['hydrostatic_contribution_pct']),
        ('Temperature Change', first_row['temperature_mse'], first_row['temperature_contribution_pct']),
        ('Momentum Equation', first_row['momentum_mse'], first_row['momentum_contribution_pct'])
    ]

    # 按MSE从小到大排序（MSE越小，约束效果越好）
    constraint_data.sort(key=lambda x: x[1])

    print(f"\n{'Rank':<6} {'Constraint':<25} {'MSE':<12} {'Contribution %':<15} {'Interpretation'}")
    print("-"*100)
    for rank, (name, mse, pct) in enumerate(constraint_data, 1):
        if rank == 1:
            interpretation = "Best performing constraint"
        elif rank == len(constraint_data):
            interpretation = "Worst performing constraint"
        else:
            interpretation = ""
        print(f"{rank:<6} {name:<25} {mse:<12.4f} {pct:<14.2f}%  {interpretation}")

    print("\n" + "="*100)
    print("Notes:")
    print("  - Baseline: 基础模型（含风向约束）")
    print("  - Δ MSE (Table 1): 相对于baseline的MSE改进（正值=改进，负值=退化）")
    print("  - Contribution % (Table 2): 基于1/MSE归一化，MSE越低贡献占比越高")
    print("  - 排名基于Epoch 5的MSE，MSE最低的约束被认为效果最好")
    print("="*100 + "\n")


def main():
    """主函数"""
    print("Loading ablation experiment results...")
    results = load_ablation_results(ABLATION_DIR)

    if not results:
        print("Error: No ablation results found!")
        return

    print(f"Loaded {len(results)} experiments")

    # 计算各约束的贡献
    print("\nCalculating constraint contributions...")
    contributions = calculate_constraint_contributions(results)

    # 保存结果到CSV
    output_csv = OUTPUT_DIR / 'physical_constraints_contributions.csv'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    contributions.to_csv(output_csv, index=False)
    print(f"Saved contributions to {output_csv}")

    # 打印汇总表格
    print_summary_table(contributions)

    # 绘制可视化图表
    print("\nGenerating visualization...")
    plot_constraint_contributions(contributions, OUTPUT_DIR)

    print("\nDone!")


if __name__ == '__main__':
    main()
