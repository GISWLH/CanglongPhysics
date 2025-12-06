"""
Physical Constraints Decomposition for V3-V2 Gap
将V3-V2的物理贡献差异分解为5个物理约束的贡献

策略：
- V2 = baseline (含风向约束)
- V3 = V2 + 全部物理约束
- V3-V2的差异 = 物理约束的总贡献
- 使用消融实验的Epoch 1数据，通过各约束的MSE来分配贡献占比
- MSE越低 → 约束效果越好 → 在V3-V2差异中贡献越大
"""

import os
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
    """配置Arial字体 - Nature风格"""
    font_path = Path('/usr/share/fonts/arial/ARIAL.TTF')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams['font.family'] = font_name
    else:
        plt.rcParams['font.family'] = 'DejaVu Sans'

    # 设置Nature风格参数（与plot_ablation_v2.py一致）
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['svg.hashsalt'] = 'hello'

    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.5,      # 轴线宽度减细
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.edgecolor': '#454545',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,      # 刻度线减短
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.5,   # 刻度线减细
        'ytick.major.width': 0.5,
        'savefig.bbox': 'tight',
        'savefig.transparent': False
    })


def load_ablation_epoch1(ablation_dir: Path):
    """
    从消融实验结果加载Epoch 1的数据

    Returns:
        dict: {experiment_name: mse_value}
    """
    experiments = {
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
        # 读取Epoch 1的数据
        if 1 in df['epoch'].values:
            row = df[df['epoch'] == 1].iloc[0]
            results[exp_name] = {
                'mse': row['valid_mse_total'],
                'mse_surface': row['valid_mse_surface'],
                'mse_upper': row['valid_mse_upper_air']
            }

    return results


def decompose_v3_v2_gap(results):
    """
    将V3-V2的差异分解为5个物理约束的贡献

    策略：
    1. V2 = baseline, V3 = full
    2. 总物理贡献 = baseline_mse - full_mse (可能为负，表示性能下降)
    3. 使用各约束的MSE来分配贡献占比（MSE越低，贡献越大）
    4. 每个约束的贡献 = 总物理贡献 × 该约束的贡献占比

    Returns:
        dict: 分解结果
    """
    baseline_mse = results['baseline']['mse']
    full_mse = results['full']['mse']

    # V3-V2的总差异（物理约束的总效果）
    total_physics_contribution = baseline_mse - full_mse

    # 获取各约束的MSE
    constraint_mses = {
        'water': results['water']['mse'],
        'energy': results['energy']['mse'],
        'hydrostatic': results['hydrostatic']['mse'],
        'temperature': results['temperature']['mse'],
        'momentum': results['momentum']['mse']
    }

    # 计算倒数权重（MSE越小，权重越大）
    inverse_weights = {name: 1.0/mse for name, mse in constraint_mses.items()}
    total_inverse = sum(inverse_weights.values())

    # 归一化得到贡献占比
    contribution_pct = {name: (weight/total_inverse)*100
                       for name, weight in inverse_weights.items()}

    # 计算每个约束在V3-V2差异中的实际贡献
    constraint_contributions = {name: total_physics_contribution * (pct/100.0)
                               for name, pct in contribution_pct.items()}

    return {
        'baseline_mse': baseline_mse,
        'full_mse': full_mse,
        'total_physics_contribution': total_physics_contribution,
        'constraint_mses': constraint_mses,
        'contribution_pct': contribution_pct,
        'constraint_contributions': constraint_contributions
    }


def plot_v3_v2_decomposition(decomposition, output_dir: Path):
    """
    绘制V3-V2差异的堆叠柱状图分解

    Args:
        decomposition: 分解结果字典
        output_dir: 输出目录
    """
    configure_fonts()

    baseline_mse = decomposition['baseline_mse']
    full_mse = decomposition['full_mse']
    total_contrib = decomposition['total_physics_contribution']
    contributions = decomposition['constraint_contributions']
    percentages = decomposition['contribution_pct']
    constraint_mses = decomposition['constraint_mses']

    # 按贡献占比排序（从大到小）
    sorted_constraints = sorted(contributions.items(),
                               key=lambda x: percentages[x[0]],
                               reverse=True)

    constraint_names = [name for name, _ in sorted_constraints]
    constraint_values = [contributions[name] for name in constraint_names]

    # 定义颜色（与plot_ablation_v2.py一致）
    colors_map = {
        'water': '#3498db',      # 蓝色 - 水量
        'energy': '#e74c3c',     # 红色 - 能量
        'hydrostatic': '#6BAED6', # 浅蓝 - 静力
        'temperature': '#f39c12', # 橙色 - 温度
        'momentum': '#9b59b6'    # 紫色 - 动量
    }
    colors = [colors_map[name] for name in constraint_names]

    # 定义标签
    labels_map = {
        'water': 'Water Balance',
        'energy': 'Energy Balance',
        'hydrostatic': 'Hydrostatic Balance',
        'temperature': 'Temperature Change',
        'momentum': 'Momentum Equation'
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    # 绘制V2 (baseline)的基准柱
    x_pos = 0
    ax.bar(x_pos, baseline_mse, 0.6, color='#8da0cb', alpha=0.7,
           label='V2 (Baseline with Wind)', edgecolor='black', linewidth=1.2)
    ax.text(x_pos, baseline_mse + 0.05, f'V2\n{baseline_mse:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 绘制V3的堆叠柱状图（从baseline开始堆叠）
    x_pos = 1
    bottom = baseline_mse

    for i, name in enumerate(constraint_names):
        value = constraint_values[i]
        pct = percentages[name]
        mse = constraint_mses[name]

        # 绘制堆叠块
        ax.bar(x_pos, value, 0.6, bottom=bottom, color=colors[i],
               label=f'{labels_map[name]} ({pct:.1f}%)',
               edgecolor='white', linewidth=1.0)

        # 在堆叠块中标注百分比
        if abs(value) > 0.02:  # 只标注较大的块
            ax.text(x_pos, bottom + value/2, f'{pct:.1f}%',
                   ha='center', va='center', fontsize=8,
                   fontweight='bold', color='white')

        bottom += value

    # 标注V3的总值
    ax.text(x_pos, full_mse + 0.05, f'V3\n{full_mse:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 绘制箭头显示总变化
    ax.annotate('', xy=(x_pos + 0.35, full_mse), xytext=(x_pos + 0.35, baseline_mse),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))

    change_pct = (total_contrib / baseline_mse) * 100
    ax.text(x_pos + 0.45, (baseline_mse + full_mse) / 2,
            f'Δ = {total_contrib:.3f}\n({change_pct:+.1f}%)',
            ha='left', va='center', fontsize=9, color='red',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor='red', alpha=0.9))

    # 设置坐标轴
    ax.set_xlim(-0.5, 2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['V2\n(Baseline)', 'V3\n(V2 + Physics)'])
    ax.set_ylabel('Validation MSE (Total)')
    ax.set_title('Physical Constraints Decomposition: V3 - V2 Gap\n(Based on Epoch 1 MSE Rankings)')

    # 图例
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / 'model_constraints_ablation_v2.png'
    fig.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f'Saved figure to {output_path}')

    # 保存SVG
    output_path_svg = output_dir / 'model_constraints_ablation_v2.svg'
    fig.savefig(output_path_svg, format='svg', bbox_inches='tight')
    print(f'Saved SVG to {output_path_svg}')

    plt.close(fig)


def print_decomposition_summary(decomposition):
    """打印分解结果摘要"""
    print("\n" + "="*80)
    print("V3-V2 Gap Decomposition Summary (Epoch 1)")
    print("="*80)

    baseline_mse = decomposition['baseline_mse']
    full_mse = decomposition['full_mse']
    total_contrib = decomposition['total_physics_contribution']
    contributions = decomposition['constraint_contributions']
    percentages = decomposition['contribution_pct']
    constraint_mses = decomposition['constraint_mses']

    print(f"\nV2 (Baseline with Wind) MSE: {baseline_mse:.4f}")
    print(f"V3 (V2 + All Physics) MSE:   {full_mse:.4f}")
    print(f"Total Physics Contribution:  {total_contrib:+.4f} ({(total_contrib/baseline_mse)*100:+.2f}%)")

    print("\n" + "-"*80)
    print("Physical Constraints Breakdown:")
    print("-"*80)

    print(f"\n{'Constraint':<25} {'MSE (Epoch 1)':<15} {'Share %':<12} {'Contribution':<15}")
    print("-"*80)

    # 按贡献占比排序
    sorted_items = sorted(contributions.items(),
                         key=lambda x: percentages[x[0]],
                         reverse=True)

    labels_map = {
        'water': 'Water Balance',
        'energy': 'Energy Balance',
        'hydrostatic': 'Hydrostatic Balance',
        'temperature': 'Temperature Change',
        'momentum': 'Momentum Equation'
    }

    for name, contrib in sorted_items:
        label = labels_map[name]
        mse = constraint_mses[name]
        pct = percentages[name]
        print(f"{label:<25} {mse:<15.4f} {pct:<11.2f}%  {contrib:+.4f}")

    print("\n" + "="*80)
    print("Interpretation:")
    print("  - MSE越低的约束，在V3-V2差异中的贡献占比越大")
    print("  - 负贡献表示该约束导致MSE增加（性能下降）")
    print("  - 所有约束贡献之和 = V3-V2的总差异")
    print("="*80 + "\n")


def main():
    """主函数"""
    print("Loading ablation experiment results (Epoch 1)...")
    results = load_ablation_epoch1(ABLATION_DIR)

    if not results:
        print("Error: No ablation results found!")
        return

    print(f"Loaded {len(results)} experiments")

    # 分解V3-V2差异
    print("\nDecomposing V3-V2 gap using physical constraints...")
    decomposition = decompose_v3_v2_gap(results)

    # 打印摘要
    print_decomposition_summary(decomposition)

    # 绘制可视化
    print("Generating visualization...")
    plot_v3_v2_decomposition(decomposition, OUTPUT_DIR)

    print("\nDone!")


if __name__ == '__main__':
    main()
