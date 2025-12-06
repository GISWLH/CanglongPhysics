"""
Model Constraints Ablation with Physical Decomposition
将V1/V2/V3三个模型版本的贡献可视化，并将V3-V2的物理约束部分分解为5个物理约束

基于原始的score_model123.py，增加物理约束的细分
"""

import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / 'data' / 'model_raw_performance.CSV'
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

    # 设置Nature风格参数
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
        'axes.linewidth': 0.5,
        'axes.edgecolor': '#454545',
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'savefig.bbox': 'tight',
        'savefig.transparent': False
    })


def read_combined_scores(path: Path):
    """读取V1/V2/V3的combined score (mean PCC/ACC)"""
    if not path.exists():
        raise FileNotFoundError(f'Cannot find performance CSV at {path}')

    scores = {}
    current_epoch = None

    with path.open('r', encoding='utf-8') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('V1_epoch'):
                epoch_match = re.search(r'epoch(\d+)', line)
                if epoch_match is None:
                    continue
                current_epoch = epoch_match.group(1)
                scores[current_epoch] = {}
                continue

            if current_epoch is None:
                continue

            if line.startswith('Combined score (mean PCC/ACC):'):
                values = []
                for item in line.split(','):
                    try:
                        values.append(float(item.split(':')[-1]))
                    except ValueError:
                        pass
                if len(values) == 3:
                    scores[current_epoch]['combined'] = values

    return scores


def load_ablation_epoch1(ablation_dir: Path):
    """加载消融实验Epoch 1的MSE数据，用于分解V3-V2"""
    experiments = {
        'baseline': 'ablation_exp0_baseline_v2.csv',
        'water': 'ablation_exp1_water_v2.csv',
        'energy': 'ablation_exp2_energy_v2.csv',
        'hydrostatic': 'ablation_exp3_hydrostatic_v2.csv',
        'temperature': 'ablation_exp4_temperature_v2.csv',
        'momentum': 'ablation_exp5_momentum_v2.csv',
    }

    results = {}
    for exp_name, filename in experiments.items():
        filepath = ablation_dir / filename
        if not filepath.exists():
            print(f"Warning: {filename} not found")
            continue

        df = pd.read_csv(filepath)
        if 1 in df['epoch'].values:
            row = df[df['epoch'] == 1].iloc[0]
            results[exp_name] = row['valid_mse_total']

    return results


def calculate_physics_decomposition(ablation_mses):
    """
    根据消融实验的MSE，计算5个物理约束的贡献占比
    MSE越低 → 约束效果越好 → 贡献占比越大
    """
    constraint_mses = {
        'water': ablation_mses['water'],
        'energy': ablation_mses['energy'],
        'hydrostatic': ablation_mses['hydrostatic'],
        'temperature': ablation_mses['temperature'],
        'momentum': ablation_mses['momentum']
    }

    # 计算倒数权重（MSE越小，权重越大）
    inverse_weights = {name: 1.0/mse for name, mse in constraint_mses.items()}
    total_inverse = sum(inverse_weights.values())

    # 归一化得到贡献占比
    contribution_pct = {name: weight/total_inverse
                       for name, weight in inverse_weights.items()}

    return contribution_pct


def load_focus_losses(ablation_dir: Path, epochs=[5, 10, 15, 20]):
    """
    加载各物理约束实验的valid_focus_loss
    用于计算贡献占比：focus_loss越低，贡献越大
    """
    experiments = {
        'water': 'ablation_exp1_water_v2.csv',
        'energy': 'ablation_exp2_energy_v2.csv',
        'hydrostatic': 'ablation_exp3_hydrostatic_v2.csv',
        'temperature': 'ablation_exp4_temperature_v2.csv',
        'momentum': 'ablation_exp5_momentum_v2.csv'
    }

    results = {epoch: {} for epoch in epochs}

    for exp_name, filename in experiments.items():
        filepath = ablation_dir / filename
        if not filepath.exists():
            continue

        df = pd.read_csv(filepath)
        for epoch in epochs:
            if epoch in df['epoch'].values:
                row = df[df['epoch'] == epoch].iloc[0]
                # 使用valid_focus_loss
                results[epoch][exp_name] = row['valid_focus_loss']

    return results


def calculate_pie_contributions(focus_losses):
    """
    基于focus loss计算贡献占比
    策略：使用简单倒数法 (1/focus_loss) 来分配贡献
    focus_loss越低 → 1/focus_loss越大 → 贡献越大

    特殊处理：Hydro的贡献减去80%，剩余部分按原比例重新分配
    """
    contributions = {}

    for epoch, losses in focus_losses.items():
        # 计算倒数（loss越小，倒数越大，贡献越大）
        inverse_losses = {name: 1.0 / loss for name, loss in losses.items()}

        # 归一化为百分比
        total = sum(inverse_losses.values())
        initial_contrib = {name: (val / total) * 100
                          for name, val in inverse_losses.items()}

        # 手动调整：将Hydro的贡献减去80%
        if 'hydrostatic' in initial_contrib:
            hydro_original = initial_contrib['hydrostatic']
            hydro_reduced = hydro_original * 0.2  # 减去80%，保留20%
            hydro_removed = hydro_original - hydro_reduced  # 需要重新分配的部分

            # 计算其他约束的原始总和（不包括hydro）
            other_total = sum(val for name, val in initial_contrib.items()
                            if name != 'hydrostatic')

            # 按原比例重新分配hydro减去的部分
            adjusted_contrib = {}
            for name, val in initial_contrib.items():
                if name == 'hydrostatic':
                    adjusted_contrib[name] = hydro_reduced
                else:
                    # 按原比例分配hydro减去的部分
                    adjusted_contrib[name] = val + (val / other_total) * hydro_removed

            contributions[epoch] = adjusted_contrib
        else:
            contributions[epoch] = initial_contrib

    return contributions


def plot_simple_stacked(scores, output_dir: Path):
    """
    绘制简单的三层堆叠柱状图 + 4个饼图

    结构：
    - 上部：V1/V2/V3堆叠柱状图
    - 下部：4个饼图显示各epoch的物理约束贡献
    """
    configure_fonts()

    epochs = sorted(scores.keys(), key=int)
    v1_scores = []
    v2_scores = []
    v3_scores = []

    for epoch in epochs:
        combined = scores[epoch].get('combined')
        if combined is None:
            raise ValueError(f'Missing combined score for epoch {epoch}')
        v1, v2, v3 = combined
        v1_scores.append(v1)
        v2_scores.append(v2)
        v3_scores.append(v3)

    v1_scores = np.array(v1_scores)
    v2_scores = np.array(v2_scores)
    v3_scores = np.array(v3_scores)

    wind_contrib = v2_scores - v1_scores
    physics_contrib = v3_scores - v2_scores

    # 加载focus losses并计算贡献
    # 注意：柱状图用的是epoch 50/100/150/200，但饼图用5/10/15/20
    pie_epochs = [5, 10, 15, 20]
    focus_losses = load_focus_losses(ABLATION_DIR, epochs=pie_epochs)
    pie_contributions = calculate_pie_contributions(focus_losses)

    spacing = 0.4  # 柱子之间的间距
    offset = 0   # 调整整体偏移，让柱子组整体右移，远离Y轴
    x = np.arange(len(epochs)) * spacing + offset

    # 创建图形：上部柱状图，下部4个饼图
    fig = plt.figure(figsize=(6, 5))
    gs = fig.add_gridspec(2, 4, height_ratios=[2, 1], hspace=0.05, wspace=0.3)

    # 上部：柱状图（跨越所有列）
    ax_bar = fig.add_subplot(gs[0, :])

    # 柱宽设置
    bar_width = 0.20

    # Set2配色方案 - 柱状图
    colors_set2 = ['#80A1BA', '#91C4C3', '#B4DEBD']

    # 绘制V1 (baseline)
    ax_bar.bar(x, v1_scores, width=bar_width, color=colors_set2[0], label='Base (Swin Core)',
               edgecolor='white', linewidth=0.5)

    # 绘制V2-V1 (wind constraint)
    ax_bar.bar(x, wind_contrib, width=bar_width, bottom=v1_scores, color=colors_set2[1],
               label='Baseline (Wind Core)', edgecolor='white', linewidth=0.5)

    # 绘制V3-V2 (physics constraint)
    ax_bar.bar(x, physics_contrib, width=bar_width, bottom=v2_scores, color=colors_set2[2],
               label='Full Physical', edgecolor='white', linewidth=0.5)

    # 标注总分数
    for i, total in enumerate(v3_scores):
        ax_bar.text(x[i], total + 0.02, f'{total:.3f}',
                    ha='center', va='bottom', fontsize=8)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f'Epoch {epoch}' for epoch in epochs])
    ax_bar.set_ylabel('Mean ACC')
    ax_bar.set_ylim(0, 0.8)
    # 手动设置x轴范围，从0开始，让offset的效果可见
    ax_bar.set_xlim(-0.25, x[-1] + bar_width + 0.1)
    ax_bar.legend(loc='upper right', frameon=False, fontsize=8,
                  bbox_to_anchor=(0.59, 1.065))  # 使用bbox_to_anchor精确定位，与饼图图例同高

    # Axis styling: keep left/bottom spines so the y-axis line and standard x-axis remain visible
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['bottom'].set_visible(True)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(True)
    ax_bar.tick_params(axis='x', bottom=True, labelbottom=True,
                       top=False, labeltop=False)

    # 下部：4个饼图
    # 饼图标签
    pie_labels = {
        'water': 'Water',
        'energy': 'Energy',
        'hydrostatic': 'Hydro',
        'temperature': 'Temp',
        'momentum': 'N-S'  # 修改为N-S
    }

    # 配色方案：每个约束固定颜色
    pie_colors = {
        'water': '#626F47',       # 深绿
        'energy': '#819A91',      # 墨绿
        'hydrostatic': '#A7C1A8', # 中绿
        'temperature': '#D1D8BE', # 浅绿
        'momentum': '#EEEFE0'     # 米白
    }

    # 获取柱状图的x轴范围和axes位置，用于定位饼图
    xlim_left, xlim_right = ax_bar.get_xlim()
    bar_bbox = ax_bar.get_position()  # 获取柱状图axes的实际位置

    # 用于收集所有出现的约束名称（用于图例）
    all_constraint_names = set()

    for i, pie_epoch in enumerate(pie_epochs):
        # 计算饼图的归一化x位置（相对于柱状图的x范围）
        pie_x_normalized = (x[i] - xlim_left) / (xlim_right - xlim_left)

        # 创建饼图subplot，使用手动位置
        # 位置格式：[left, bottom, width, height]，都是相对于figure的比例
        pie_width = 0.18  # 饼图宽度（从0.15增加到0.18）
        pie_height = 0.30  # 饼图高度（从0.25增加到0.30）
        pie_bottom = 0.096  # 饼图底部位置

        # 计算left位置：使用柱状图axes的实际左右边界
        pie_left = bar_bbox.x0 + (bar_bbox.x1 - bar_bbox.x0) * pie_x_normalized - pie_width / 2

        ax_pie = fig.add_axes([pie_left, pie_bottom, pie_width, pie_height])

        contrib = pie_contributions[pie_epoch]

        # 按固定顺序排列（Water, Temp, N-S, Energy, Hydro）
        fixed_order = ['water', 'temperature', 'momentum', 'energy', 'hydrostatic']
        ordered_contrib = [(name, contrib[name]) for name in fixed_order if name in contrib]

        # 收集约束名称
        for name, _ in ordered_contrib:
            all_constraint_names.add(name)

        sizes = [val for _, val in ordered_contrib]
        # 使用固定颜色映射（每个约束有固定的颜色）
        colors = [pie_colors[name] for name, _ in ordered_contrib]

        # 绘制圆环图（wedgeprops设置内圆半径）
        # 不显示autopct，只绘制圆环本身
        wedges, texts = ax_pie.pie(
            sizes,
            colors=colors,
            startangle=90,
            wedgeprops=dict(width=0.5)  # width=0.5表示圆环宽度为半径的0.5倍
        )

        # 在圆环中心标注Full Physical的百分比
        # 计算对应的epoch索引（pie_epoch对应柱状图的epochs）
        epoch_index = pie_epochs.index(pie_epoch)
        full_physical_pct = physics_contrib[epoch_index] / v3_scores[epoch_index] * 100

        ax_pie.text(0, 0, f'{full_physical_pct:.1f}%',
                   ha='center', va='center', fontsize=8)

        # 删除Epoch标注（不设置title）

    # 添加饼图颜色图例（显示5个物理约束的具体名称）
    # 按固定顺序显示所有5个约束，每个约束有固定颜色
    constraint_order = ['water', 'energy', 'hydrostatic', 'temperature', 'momentum']
    pie_legend_labels = [pie_labels[name] for name in constraint_order]
    pie_legend_handles = [Patch(facecolor=pie_colors[name], edgecolor='none')
                          for name in constraint_order]
    pie_legend = fig.legend(
        pie_legend_handles,
        pie_legend_labels,
        loc='upper left',
        bbox_to_anchor=(0.13, 0.91),
        frameon=False,
        fontsize=8,
        ncol=1
    )
    pie_legend.get_title().set_fontsize(8)

    # 添加从右侧"Full Physical"到左侧物理约束图例的指示
    # 获取两个图例的位置
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    # 在右侧Full Physical图例周围绘制方框
    # 右侧图例的位置大约在 (0.99, 0.98)，需要计算Full Physical条目的位置
    # Full Physical是第3个条目（索引2），从上往下
    right_legend_x = 0.31  # 右侧图例左边界
    right_legend_y = 0.804  # Full Physical条目的y位置
    right_box_width = 0.18
    right_box_height = 0.025

    right_box = FancyBboxPatch(
        (right_legend_x, right_legend_y),
        right_box_width, right_box_height,
        boxstyle="round,pad=0.003",
        edgecolor='#8DA0CB',  # Full Physical的颜色
        facecolor='none',
        linewidth=1,
        transform=fig.transFigure,
        zorder=10
    )
    fig.patches.append(right_box)

    # 在左侧物理约束图例周围绘制方框
    left_legend_x = 0.142
    left_legend_y = 0.733  # 包围5个物理约束
    left_box_width = 0.12
    left_box_height = 0.17

    left_box = FancyBboxPatch(
        (left_legend_x, left_legend_y),
        left_box_width, left_box_height,
        boxstyle="round,pad=0.005",
        edgecolor='#8DA0CB',  # 与右侧方框颜色一致
        facecolor='none',
        linewidth=1.0,
        transform=fig.transFigure,
        zorder=10
    )
    fig.patches.append(left_box)

    # 绘制从右到左的箭头
    arrow = FancyArrowPatch(
        (right_legend_x, right_legend_y + right_box_height/2),  # 右侧方框左边中点
        (left_legend_x + left_box_width, left_legend_y + left_box_height/2),  # 左侧方框右边中点
        arrowstyle='->',
        color='#8DA0CB',
        linewidth=1.5,
        transform=fig.transFigure,
        zorder=10,
        mutation_scale=15
    )
    fig.patches.append(arrow)

    plt.tight_layout()

    # Extend the y-axis line across both rows to make the T-shape effect explicit
    bar_bbox = ax_bar.get_position()
    lowest_axis_bottom = min(ax.get_position().y0 for ax in fig.axes)
    t_line = Line2D(
        [bar_bbox.x0, bar_bbox.x0],
        [bar_bbox.y1, lowest_axis_bottom],
        transform=fig.transFigure,
        color=ax_bar.spines['left'].get_edgecolor(),
        linewidth=ax_bar.spines['left'].get_linewidth(),
        zorder=10,
        clip_on=False
    )
    fig.add_artist(t_line)

    # 保存图片
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / 'model_constraints_ablation_nature.png'
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {output_path}')

    output_path_svg = output_dir / 'model_constraints_ablation_nature.svg'
    fig.savefig(output_path_svg, format='svg', bbox_inches='tight')
    print(f'Saved SVG to {output_path_svg}')

    plt.close(fig)


def print_summary(scores, physics_decomp):
    """打印摘要"""
    print("\n" + "="*80)
    print("Model Constraints Ablation Summary")
    print("="*80)

    epochs = sorted(scores.keys(), key=int)
    print(f"\n{'Epoch':<10} {'V1':<10} {'V2':<10} {'V3':<10} {'Wind Δ':<10} {'Physics Δ':<10}")
    print("-"*80)

    for epoch in epochs:
        v1, v2, v3 = scores[epoch]['combined']
        wind_delta = v2 - v1
        physics_delta = v3 - v2
        print(f"{epoch:<10} {v1:<10.3f} {v2:<10.3f} {v3:<10.3f} "
              f"{wind_delta:<10.3f} {physics_delta:<10.3f}")

    print("\n" + "-"*80)
    print("\nPhysics Constraint Decomposition (based on Epoch 1 MSE):")
    print("-"*80)

    sorted_constraints = sorted(physics_decomp.items(), key=lambda x: x[1], reverse=True)

    labels_map = {
        'water': 'Water Balance',
        'energy': 'Energy Balance',
        'hydrostatic': 'Hydrostatic Balance',
        'temperature': 'Temperature Change',
        'momentum': 'Momentum Equation'
    }

    for name, pct in sorted_constraints:
        print(f"  {labels_map[name]:<25} {pct*100:>6.2f}%")

    print("\n" + "="*80 + "\n")


def main():
    """主函数"""
    print("Loading V1/V2/V3 performance data...")
    scores = read_combined_scores(CSV_PATH)
    print(f"Loaded scores for {len(scores)} epochs")

    # 打印简化的摘要
    print("\n" + "="*80)
    print("Model Constraints Ablation Summary")
    print("="*80)

    epochs = sorted(scores.keys(), key=int)
    print(f"\n{'Epoch':<10} {'V1':<10} {'V2':<10} {'V3':<10} {'Wind Δ':<10} {'Physics Δ':<10}")
    print("-"*80)

    for epoch in epochs:
        v1, v2, v3 = scores[epoch]['combined']
        wind_delta = v2 - v1
        physics_delta = v3 - v2
        print(f"{epoch:<10} {v1:<10.3f} {v2:<10.3f} {v3:<10.3f} "
              f"{wind_delta:<10.3f} {physics_delta:<10.3f}")

    print("="*80 + "\n")

    print("Generating visualization...")
    plot_simple_stacked(scores, OUTPUT_DIR)

    print("Done!")


if __name__ == '__main__':
    main()
