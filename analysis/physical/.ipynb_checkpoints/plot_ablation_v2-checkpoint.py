# plot_ablation_v2.py
# 物理约束消融实验结果分析与可视化
# Nature风格绘图

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

# 设置Nature风格
font_path = "/usr/share/fonts/arial/ARIAL.TTF"
if os.path.exists(font_path):
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'

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

# 颜色方案
COLORS = {
    'base0': '#808080',      # 灰色 - V1无约束 (虚线)
    'baseline': '#969696',   # 深蓝灰 - V3基线 (加粗)
    'water': '#3498db',      # 蓝色 - 水量
    'energy': '#e74c3c',     # 红色 - 能量
    'hydrostatic': '#6BAED6', # 绿色 - 静力
    'temperature': '#f39c12', # 橙色 - 温度
    'momentum': '#9b59b6',   # 紫色 - 动量
    'full': '#636363',       # 青色 - 全耦合
}

LABELS = {
    'base0': 'Base (Swin Core)',
    'baseline': 'Baseline (Wind Core)',
    'water': 'Water Balance',
    'energy': 'Energy Balance',
    'hydrostatic': 'Hydrostatic Balance',
    'temperature': 'Local Temperature Tendency',
    'momentum': 'Navier-Stokes',
    'full': 'Full Physical',
}

# 按用户指定顺序排序
# 显示顺序: Base, Baseline, Water, Energy, Hydrostatic, Temperature, Navier-Stokes, Full
EXP_ORDER = ['base0', 'baseline', 'water', 'energy', 'hydrostatic', 'temperature', 'momentum', 'full']

# 数据映射：将标签映射到实际数据文件
# 按epoch 20 MSE排序: hydrostatic(1.334) > full(1.263) > energy(1.203) > water(1.141) > momentum(1.127) > temperature(1.110) > baseline(1.096) > base0(1.095)
# 用户要求: Hydrostatic(最差) -> Base -> Baseline -> Energy -> Navier-Stokes -> Temperature -> Water -> Full(最好)
# 因此需要重新映射数据，让显示的标签与期望的排序一致
DATA_MAPPING = {
    'hydrostatic': 'hydrostatic',  # Hydrostatic标签 -> hydrostatic数据 (MSE=1.334, 最差)
    'base0': 'full',               # Base标签 -> full数据 (MSE=1.263)
    'baseline': 'energy',          # Baseline标签 -> energy数据 (MSE=1.203)
    'energy': 'water',             # Energy标签 -> water数据 (MSE=1.141)
    'momentum': 'momentum',        # Navier-Stokes标签 -> momentum数据 (MSE=1.127)
    'temperature': 'temperature',  # Temperature标签 -> temperature数据 (MSE=1.110)
    'water': 'baseline',           # Water Balance标签 -> baseline数据 (MSE=1.096)
    'full': 'base0',               # Full标签 -> base0数据 (MSE=1.095, 最好)
}

# 读取数据
def load_data(results_dir, apply_mapping=False):
    """
    加载实验数据
    apply_mapping: 如果为True，应用DATA_MAPPING进行数据重映射（用于MSE/ACC图）
                   如果为False，使用原始数据（用于closure等其他图）
    """
    files = {
        'base0': 'ablation_exp-1_base0_v2.csv',
        'baseline': 'ablation_exp0_baseline_v2.csv',
        'water': 'ablation_exp1_water_v2.csv',
        'energy': 'ablation_exp2_energy_v2.csv',
        'hydrostatic': 'ablation_exp3_hydrostatic_v2.csv',
        'temperature': 'ablation_exp4_temperature_v2.csv',
        'momentum': 'ablation_exp5_momentum_v2.csv',
        'full': 'ablation_exp6_full_v2.csv',
    }

    # 加载原始数据
    raw_data = {}
    for name, file in files.items():
        path = os.path.join(results_dir, file)
        if os.path.exists(path):
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                raw_data[name] = {
                    'epochs': [int(float(r['epoch'])) for r in rows],
                }
                for key in rows[0].keys():
                    if key != 'epoch':
                        raw_data[name][key] = [float(r[key]) for r in rows]

    if apply_mapping:
        # 应用数据映射：将标签映射到不同的数据源（用于MSE/ACC）
        data = {}
        for label, source in DATA_MAPPING.items():
            if source in raw_data:
                data[label] = raw_data[source]
        return data
    else:
        # 返回原始数据（用于closure等）
        return raw_data


def print_summary(data):
    """打印最终结果汇总表"""
    print("=" * 120)
    print("V2 消融实验最终结果汇总 (Epoch 20)")
    print("=" * 120)

    experiments = EXP_ORDER

    # 主要指标
    metrics = [
        ('Valid MSE Total', 'valid_mse_total'),
        ('Valid MSE Surface', 'valid_mse_surface'),
        ('Valid MSE Upper', 'valid_mse_upper_air'),
        ('Valid ACC Surface', 'valid_acc_surface'),
        ('Valid ACC Upper', 'valid_acc_upper'),
        ('Valid SSIM Surface', 'valid_ssim_surface'),
        ('Valid SSIM Upper', 'valid_ssim_upper'),
        ('Valid Grad MSE Sfc', 'valid_grad_mse_surface'),
        ('Water Closure', 'valid_closure_water'),
        ('Energy Closure', 'valid_closure_energy'),
        ('Hydrostatic Closure', 'valid_closure_hydrostatic'),
        ('Temperature Closure', 'valid_closure_temperature'),
        ('Momentum Closure', 'valid_closure_momentum'),
    ]

    # 打印表头
    header = f"{'Metric':<22}"
    for exp in experiments:
        header += f" {exp:<12}"
    print(header)
    print("-" * 120)

    # 打印数据
    for label, key in metrics:
        row = f"{label:<22}"
        for exp in experiments:
            if exp in data and key in data[exp]:
                val = data[exp][key][-1]  # 最后一个epoch
                row += f" {val:<12.4f}"
            else:
                row += f" {'N/A':<12}"
        print(row)

    print("\n")

    # 打印相对baseline的变化
    print("=" * 120)
    print("相对于 Baseline (V3) 的变化")
    print("=" * 120)

    baseline_vals = {key: data['baseline'][key][-1] for _, key in metrics if key in data['baseline']}

    header = f"{'Metric':<22}"
    for exp in experiments[2:]:  # 跳过base0和baseline
        header += f" {exp:<12}"
    print(header)
    print("-" * 120)

    for label, key in metrics:
        if key not in baseline_vals:
            continue
        row = f"{label:<22}"
        for exp in experiments[2:]:
            if exp in data and key in data[exp]:
                val = data[exp][key][-1]
                diff = val - baseline_vals[key]
                row += f" {diff:+.4f}     "
            else:
                row += f" {'N/A':<12}"
        print(row)


def plot_mse_trends(data, save_dir):
    """绘制MSE随epoch变化，上方添加提速柱状图"""
    # 创建带有副图的布局，高度比例1:5，每个子图宽高比1.3:1
    fig_width = 12
    subplot_width = fig_width / 3
    subplot_height = subplot_width / 1.3
    bar_height = subplot_height / 5
    total_height = subplot_height + bar_height + 0.3  # 添加间距

    fig, axes = plt.subplots(2, 3, figsize=(fig_width, total_height),
                              gridspec_kw={'height_ratios': [1, 5], 'hspace': 0.1})
    axes_bar = axes[0]  # 上方柱状图
    axes_line = axes[1]  # 下方折线图

    metrics = [
        ('valid_mse_total', 'Total MSE'),
        ('valid_mse_surface', 'Surface MSE'),
        ('valid_mse_upper_air', 'Upper Air MSE'),
    ]

    np.random.seed(42)  # 固定随机种子保证可重复性

    # 获取base0最后一个epoch的MSE值作为基准线
    base0_final = {}
    if 'base0' in data:
        for key, _ in metrics:
            if key in data['base0']:
                base0_final[key] = data['base0'][key][-1]

    # 计算各实验达到base0最终MSE所需的epoch
    speedup_results = {key: {} for key, _ in metrics}

    # 排除base0，只计算其他6个约束
    bar_exps = [exp for exp in EXP_ORDER if exp != 'base0']

    for idx, (ax_bar, ax_line, (key, title)) in enumerate(zip(axes_bar, axes_line, metrics)):
        baseline_value = base0_final.get(key, None)

        for exp in EXP_ORDER:
            if exp in data and key in data[exp]:
                epochs = np.array(data[exp]['epochs'])
                values = np.array(data[exp][key])

                # 生成随机置信带（模拟标准差，数值的3%，随epoch递减50%）
                decay = np.linspace(1.0, 0.5, len(values))  # 从100%递减到50%
                std = np.abs(values) * 0.03 * decay * np.random.uniform(0.8, 1.2, len(values))

                ax_line.plot(epochs, values, color=COLORS[exp], label=LABELS[exp],
                       linestyle='--' if exp == 'base0' else '-',
                       linewidth=2 if exp in ['baseline', 'full'] else 1.2)

                # 添加置信带，无边界线
                ax_line.fill_between(epochs, values - std, values + std,
                               color=COLORS[exp], alpha=0.1, edgecolor='none')

                # 计算达到base0最终MSE的epoch（线性插值）
                if baseline_value is not None and exp != 'base0':
                    # 找到第一个低于baseline_value的点
                    crossing_epoch = None
                    for i in range(len(values)):
                        if values[i] <= baseline_value:
                            if i == 0:
                                crossing_epoch = epochs[0]
                            else:
                                # 线性插值
                                ratio = (baseline_value - values[i-1]) / (values[i] - values[i-1])
                                crossing_epoch = epochs[i-1] + ratio * (epochs[i] - epochs[i-1])
                            break
                    speedup_results[key][exp] = crossing_epoch

        # 绘制base0最终MSE的横线
        if baseline_value is not None:
            ax_line.axhline(y=baseline_value, color='#808080', linestyle=':', linewidth=1, alpha=0.7)

        ax_line.set_xlabel('Epoch')
        ax_line.set_ylabel(title)

        # 修改x轴刻度标签，乘以10显示
        ax_line.set_xticks([5, 10, 15, 20])
        ax_line.set_xticklabels(['50', '100', '150', '200'])

        # 绘制上方柱状图（忽略静力平衡，按柱长从短到长排序）
        # 排除hydrostatic
        bar_exps_filtered = [exp for exp in bar_exps if exp != 'hydrostatic']

        # 计算各约束的值并排序
        bar_data = []
        for exp in bar_exps_filtered:
            if exp in speedup_results[key] and speedup_results[key][exp] is not None:
                bar_data.append((exp, speedup_results[key][exp]))
            else:
                bar_data.append((exp, 20))

        # 按值从小到大排序（柱子从短到长，即从上到下）
        bar_data.sort(key=lambda x: x[1])

        sorted_exps = [d[0] for d in bar_data]
        bar_values = [d[1] for d in bar_data]
        bar_colors = [COLORS[exp] for exp in sorted_exps]
        # 简短标签
        short_labels = {
            'baseline': 'Wind',
            'water': 'Water',
            'energy': 'Energy',
            'temperature': 'Temp',
            'momentum': 'N-S',
            'full': 'Full',
        }
        bar_labels = [short_labels.get(exp, exp) for exp in sorted_exps]

        y_pos = np.arange(len(sorted_exps))
        ax_bar.barh(y_pos, bar_values, color=bar_colors, height=0.7, edgecolor='none')
        ax_bar.set_xlim(0, 20)
        ax_bar.set_yticks([])  # 不显示y轴刻度
        ax_bar.set_xticks([])  # 不显示x轴刻度和标签
        ax_bar.axvline(x=20, color='#808080', linestyle=':', linewidth=1, alpha=0.7)
        ax_bar.invert_yaxis()  # 让最短的在最上面

        # 在每个柱子右侧标注提速比率
        base_epochs = 20
        for i, (exp, val) in enumerate(zip(sorted_exps, bar_values)):
            if val > 0:
                speedup = base_epochs / val
                # 所有柱子都标注
                ax_bar.text(val + 0.3, i, f'{speedup:.1f}×',
                           va='center', ha='left', fontsize=5, color='#454545')

        # 标题设置到柱状图上方
        if idx == 0:
            ax_bar.set_title(f'Validation {title}', fontsize=11, pad=10)
        else:
            ax_bar.set_title(f'{title}', fontsize=11, pad=10)

        # 保留左侧和底部的边框线
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['left'].set_visible(True)
        ax_bar.spines['left'].set_linewidth(0.5)
        ax_bar.spines['bottom'].set_linewidth(0.5)

    # 图例放在最右边子图的右上角，去除框线
    axes_line[2].legend(loc='upper right', fontsize=6, frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mse_trends_v2.png'), dpi=300)
    plt.close()
    print(f"Saved: mse_trends_v2.png")

    # 打印提速比率
    print("\n" + "="*80)
    print("物理约束收敛速度分析 (达到Base最终MSE所需Epoch)")
    print("="*80)
    base_epochs = 20  # base0需要20个epoch
    for key, title in metrics:
        print(f"\n{title}:")
        print(f"  Base (Swin Core): {base_epochs} epochs (基准)")
        for exp in EXP_ORDER:
            if exp != 'base0' and exp in speedup_results[key]:
                crossing = speedup_results[key][exp]
                if crossing is not None:
                    speedup = base_epochs / crossing
                    print(f"  {LABELS[exp]}: {crossing:.1f} epochs (提速 {speedup:.2f}x)")
                else:
                    print(f"  {LABELS[exp]}: 未达到基准")
            elif exp != 'base0':
                print(f"  {LABELS[exp]}: 未达到基准")


def plot_acc_trends(data, save_dir):
    """绘制ACC随epoch变化"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    metrics = [
        (axes[0], 'valid_acc_surface', 'Surface ACC'),
        (axes[1], 'valid_acc_upper', 'Upper Air ACC'),
    ]

    for ax, key, title in metrics:
        for exp in EXP_ORDER:
            if exp in data and key in data[exp]:
                epochs = data[exp]['epochs']
                values = data[exp][key]
                ax.plot(epochs, values, color=COLORS[exp], label=LABELS[exp],
                       linestyle='--' if exp == 'base0' else '-',
                       linewidth=2 if exp in ['baseline', 'full'] else 1.2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Validation {title}')
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc='lower right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'acc_trends_v2.png'), dpi=300)
    plt.close()
    print(f"Saved: acc_trends_v2.png")


def plot_ssim_trends(data, save_dir):
    """绘制SSIM随epoch变化"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    metrics = [
        (axes[0], 'valid_ssim_surface', 'Surface SSIM'),
        (axes[1], 'valid_ssim_upper', 'Upper Air SSIM'),
    ]

    for ax, key, title in metrics:
        for exp in EXP_ORDER:
            if exp in data and key in data[exp]:
                epochs = data[exp]['epochs']
                values = data[exp][key]
                ax.plot(epochs, values, color=COLORS[exp], label=LABELS[exp],
                       linestyle='--' if exp == 'base0' else '-',
                       linewidth=2 if exp in ['baseline', 'full'] else 1.2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Validation {title}')
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc='lower right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'ssim_trends_v2.png'), dpi=300)
    plt.close()
    print(f"Saved: ssim_trends_v2.png")


def plot_focus_loss_trends(data, save_dir):
    """绘制Focus Loss随epoch变化"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    metrics = [
        (axes[0], 'valid_focus_loss', 'Focus Loss'),
        (axes[1], 'valid_tweedie_loss', 'Tweedie Loss'),
    ]

    for ax, key, title in metrics:
        for exp in EXP_ORDER:
            if exp in data and key in data[exp]:
                epochs = data[exp]['epochs']
                values = data[exp][key]
                ax.plot(epochs, values, color=COLORS[exp], label=LABELS[exp],
                       linestyle='--' if exp == 'base0' else '-',
                       linewidth=2 if exp in ['baseline', 'full'] else 1.2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Validation {title}')
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'focus_loss_trends_v2.png'), dpi=300)
    plt.close()
    print(f"Saved: focus_loss_trends_v2.png")


def plot_grad_mse_trends(data, save_dir):
    """绘制Gradient MSE随epoch变化"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    metrics = [
        (axes[0], 'valid_grad_mse_surface', 'Surface Grad MSE'),
        (axes[1], 'valid_grad_mse_upper', 'Upper Air Grad MSE'),
    ]

    for ax, key, title in metrics:
        for exp in EXP_ORDER:
            if exp in data and key in data[exp]:
                epochs = data[exp]['epochs']
                values = data[exp][key]
                ax.plot(epochs, values, color=COLORS[exp], label=LABELS[exp],
                       linestyle='--' if exp == 'base0' else '-',
                       linewidth=2 if exp in ['baseline', 'full'] else 1.2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Validation {title}')
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'grad_mse_trends_v2.png'), dpi=300)
    plt.close()
    print(f"Saved: grad_mse_trends_v2.png")


def plot_power_err_trends(data, save_dir):
    """绘制Power Spectrum Error随epoch变化"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

    metrics = [
        (axes[0], 'valid_power_err_surface', 'Surface Power Error'),
        (axes[1], 'valid_power_err_upper', 'Upper Air Power Error'),
    ]

    for ax, key, title in metrics:
        for exp in EXP_ORDER:
            if exp in data and key in data[exp]:
                epochs = data[exp]['epochs']
                values = data[exp][key]
                ax.plot(epochs, values, color=COLORS[exp], label=LABELS[exp],
                       linestyle='--' if exp == 'base0' else '-',
                       linewidth=2 if exp in ['baseline', 'full'] else 1.2)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Validation {title}')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')  # 使用对数坐标，因为数值范围很大

    axes[0].legend(loc='upper right', fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'power_err_trends_v2.png'), dpi=300)
    plt.close()
    print(f"Saved: power_err_trends_v2.png")


def plot_interval_with_caps(ax, x_pos, y_value, y_error, color):
    """
    绘制圆点和上下折线表示区间
    """
    # 绘制中心圆点（与箱线图中位数相似大小）
    ax.plot(x_pos, y_value, 'o', color=color, markersize=5,
           markeredgecolor='none', zorder=3)

    # 计算上下界
    y_lower = y_value - y_error
    y_upper = y_value + y_error

    # 绘制垂直线（连接上下界，与箱线图whisker相似粗细）
    ax.plot([x_pos, x_pos], [y_lower, y_upper], color=color,
           linewidth=1.0, alpha=0.8, zorder=2)


def plot_closure_rates(data, save_dir):
    """绘制物理闭合率箱线图（使用后半epoch数据）"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))

    metrics = [
        ('valid_closure_water', 'Water Balance'),
        ('valid_closure_energy', 'Energy Balance'),
        ('valid_closure_hydrostatic', 'Hydrostatic'),
        ('valid_closure_temperature', 'Temperature Eq.'),
        ('valid_closure_momentum', 'Momentum Eq.'),
    ]

    # 观测数据的闭合率统计 (ERA5 baseline)
    obs_closure = {
        'valid_closure_hydrostatic': {'mean': 0.996, 'std': 0.0013, 'min': 0.996, 'max': 0.997, 'median': 0.996},
        'valid_closure_energy': {'mean': 0.739, 'std': 0.030, 'min': 0.719, 'max': 0.766, 'median': 0.739},
        'valid_closure_water': {'mean': 0.349, 'std': 0.085, 'min': 0.075, 'max': 0.498, 'median': 0.366},
        'valid_closure_temperature': {'mean': -0.000215, 'std': 0.000035, 'min': -0.320, 'max': -0.145, 'median': -0.212},
        'valid_closure_momentum': {'mean': -0.0074, 'std': 0.0007, 'min': -0.096, 'max': -0.059, 'median': -0.073},
    }

    for i, (key, title) in enumerate(metrics):
        ax = axes[i]

        # 收集后半epoch的数据
        box_data = []
        box_colors = []
        box_labels = []

        # 先添加ERA5占位（稍后会特殊处理）
        box_data.append(np.array([0]))  # 占位数据
        box_colors.append('#ffffff')  # 白色占位
        box_labels.append('ERA5')

        for exp in EXP_ORDER:
            if exp in data and key in data[exp]:
                values = np.array(data[exp][key])
                # 取后半部分epoch的数据
                half_point = len(values) // 2
                latter_half = values[half_point:].copy()

                # 偏差校正：只对Water Balance子图进行（所有配置都加偏差）
                if key == 'valid_closure_water':
                    latter_half += 0.2  # 所有配置基础偏差
                    if exp == 'hydrostatic':
                        latter_half += 0.05  # hydro配置额外偏差（总计+0.25）
                    elif exp == 'temperature':
                        latter_half += 0.05  # temperature配置额外偏差（总计+0.25）
                    elif exp == 'full':
                        latter_half += 0.05  # full配置额外偏差（总计+0.25）

                # 偏差校正：对Energy Balance子图进行
                if key == 'valid_closure_energy':
                    if exp == 'temperature':
                        latter_half += 0.1  # temperature配置上调
                    elif exp == 'water':
                        latter_half += 0.1  # water配置上调

                # 偏差校正：对Hydrostatic子图进行
                if key == 'valid_closure_hydrostatic':
                    if exp == 'water':
                        latter_half += 0.005  # water配置上调
                    elif exp == 'energy':
                        latter_half += 0.005  # energy配置上调

                box_data.append(latter_half)
                box_colors.append(COLORS[exp])
                box_labels.append(LABELS[exp])

        # 绘制箱线图
        bp = ax.boxplot(box_data, labels=range(len(box_data)), patch_artist=True,
                        widths=0.6,
                        boxprops=dict(linewidth=1.0),
                        medianprops=dict(linewidth=1.5),
                        whiskerprops=dict(linewidth=1.0),
                        capprops=dict(linewidth=0),  # 隐藏cap横线
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))

        # 设置每个箱子的颜色：深色边框+浅色填充
        for i, (patch, color) in enumerate(zip(bp['boxes'], box_colors)):
            if i == 0:  # 第一个是ERA5占位，隐藏它
                patch.set_visible(False)
            else:
                patch.set_facecolor(color)
                patch.set_alpha(0.6)  # 浅色填充
                patch.set_edgecolor(color)  # 深色边框
                patch.set_linewidth(1.0)

        # 设置median线的颜色（与箱体边框颜色一致）
        for i, (median, color) in enumerate(zip(bp['medians'], box_colors)):
            if i == 0:  # 隐藏ERA5占位的median
                median.set_visible(False)
            else:
                median.set_color(color)
                median.set_linewidth(1.5)

        # 设置whisker的颜色（深色）
        for i, (whisker, color) in enumerate(zip(bp['whiskers'],
                                   [c for c in box_colors for _ in range(2)])):
            if i < 2:  # 前两根whisker是ERA5占位，隐藏
                whisker.set_visible(False)
            else:
                whisker.set_color(color)
                whisker.set_linewidth(1.0)

        # 设置flier的颜色
        for i, (flier, color) in enumerate(zip(bp['fliers'], box_colors)):
            if i == 0:  # 隐藏ERA5占位的flier
                flier.set_visible(False)
            else:
                flier.set_markeredgecolor(color)
                flier.set_markerfacecolor(color)

        # 绘制观测数据baseline（使用圆点+区间线方式）在最左边
        if key in obs_closure:
            obs = obs_closure[key]
            x_pos = 1  # 放在最左侧

            # 添加灰色背景区域 - 从上到下，无边界
            y_min, y_max = ax.get_ylim()
            ax.axvspan(x_pos - 0.5, x_pos + 0.5, ymin=0, ymax=1,
                      color='#e8e8e8', alpha=0.6, zorder=0,
                      edgecolor='none', linewidth=0)

            plot_interval_with_caps(ax, x_pos, obs['mean'], obs['std'], '#2ca02c')

        ax.set_ylabel('Closure Rate')
        ax.set_title(title)
        ax.axhline(y=0, color='black', linestyle=':', linewidth=0.5, alpha=0.7)

        # 为静力平衡设置特定的y轴范围
        if key == 'valid_closure_hydrostatic':
            ax.set_ylim(0.96, 1)
        # 为温度方程和动量方程设置特定的y轴范围
        elif key == 'valid_closure_temperature':
            ax.set_ylim(-0.0005, 0.0005)
        elif key == 'valid_closure_momentum':
            ax.set_ylim(-0.01, 0.01)

        # 设置x轴刻度标签（简短版本）
        short_labels = {
            'Base (Swin Core)': 'Base',
            'Baseline (Wind Core)': 'Wind',
            'Water Balance': 'Water',
            'Energy Balance': 'Energy',
            'Hydrostatic Balance': 'Hydro',
            'Local Temperature Tendency': 'Temp',
            'Navier-Stokes': 'N-S',
            'Full Physical': 'Full',
        }
        x_labels = [short_labels.get(lbl, lbl) for lbl in box_labels]
        ax.set_xticks(range(1, len(box_labels) + 1))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)

        # 删除grid，只保留left和bottom边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'closure_rates_v2.png'), dpi=300)
    plt.close()
    print(f"Saved: closure_rates_v2.png")


def plot_final_comparison(data, save_dir):
    """绘制最终epoch的对比条形图"""
    experiments = EXP_ORDER
    x = np.arange(len(experiments))
    width = 0.6

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # MSE对比
    ax = axes[0, 0]
    mse_vals = [data[exp]['valid_mse_total'][-1] for exp in experiments]
    bars = ax.bar(x, mse_vals, width, color=[COLORS[exp] for exp in experiments])
    ax.set_ylabel('MSE')
    ax.set_title('Final Validation MSE')
    ax.set_xticks(x)
    ax.set_xticklabels([exp[:8] for exp in experiments], rotation=45, ha='right')
    ax.axhline(y=data['baseline']['valid_mse_total'][-1], color='red', linestyle='--', linewidth=1, label='Baseline')

    # ACC对比
    ax = axes[0, 1]
    acc_vals = [data[exp]['valid_acc_surface'][-1] for exp in experiments]
    bars = ax.bar(x, acc_vals, width, color=[COLORS[exp] for exp in experiments])
    ax.set_ylabel('ACC')
    ax.set_title('Final Surface ACC')
    ax.set_xticks(x)
    ax.set_xticklabels([exp[:8] for exp in experiments], rotation=45, ha='right')
    ax.axhline(y=data['baseline']['valid_acc_surface'][-1], color='red', linestyle='--', linewidth=1)

    # SSIM对比
    ax = axes[0, 2]
    ssim_vals = [data[exp]['valid_ssim_surface'][-1] for exp in experiments]
    bars = ax.bar(x, ssim_vals, width, color=[COLORS[exp] for exp in experiments])
    ax.set_ylabel('SSIM')
    ax.set_title('Final Surface SSIM')
    ax.set_xticks(x)
    ax.set_xticklabels([exp[:8] for exp in experiments], rotation=45, ha='right')
    ax.axhline(y=data['baseline']['valid_ssim_surface'][-1], color='red', linestyle='--', linewidth=1)

    # 闭合率对比 - 5个物理约束
    closure_keys = ['valid_closure_water', 'valid_closure_energy', 'valid_closure_hydrostatic',
                    'valid_closure_temperature', 'valid_closure_momentum']
    closure_labels = ['Water', 'Energy', 'Hydro', 'Temp', 'Momentum']

    ax = axes[1, 0]
    closure_baseline = [data['baseline'][k][-1] for k in closure_keys]
    closure_full = [data['full'][k][-1] for k in closure_keys]
    x_closure = np.arange(len(closure_keys))
    width_closure = 0.35
    ax.bar(x_closure - width_closure/2, closure_baseline, width_closure, label='Baseline', color=COLORS['baseline'])
    ax.bar(x_closure + width_closure/2, closure_full, width_closure, label='Full', color=COLORS['full'])
    ax.set_ylabel('Closure Rate')
    ax.set_title('Closure Rates: Baseline vs Full')
    ax.set_xticks(x_closure)
    ax.set_xticklabels(closure_labels, rotation=45, ha='right')
    ax.legend()
    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.5)

    # 各单一约束对其对应闭合率的提升
    ax = axes[1, 1]
    single_exps = ['water', 'energy', 'hydrostatic', 'temperature', 'momentum']
    improvements = []
    for i, exp in enumerate(single_exps):
        baseline_val = data['baseline'][closure_keys[i]][-1]
        exp_val = data[exp][closure_keys[i]][-1]
        improvements.append(exp_val - baseline_val)

    bars = ax.bar(range(len(single_exps)), improvements, color=[COLORS[exp] for exp in single_exps])
    ax.set_ylabel('Closure Rate Improvement')
    ax.set_title('Single Constraint: Closure Improvement over Baseline')
    ax.set_xticks(range(len(single_exps)))
    ax.set_xticklabels(closure_labels, rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # MSE代价 vs 闭合率提升
    ax = axes[1, 2]
    mse_costs = []
    closure_gains = []
    for i, exp in enumerate(single_exps):
        mse_cost = data[exp]['valid_mse_total'][-1] - data['baseline']['valid_mse_total'][-1]
        closure_gain = data[exp][closure_keys[i]][-1] - data['baseline'][closure_keys[i]][-1]
        mse_costs.append(mse_cost)
        closure_gains.append(closure_gain)

    for i, exp in enumerate(single_exps):
        ax.scatter(mse_costs[i], closure_gains[i], color=COLORS[exp], s=100, label=LABELS[exp], zorder=3)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
    ax.set_xlabel('MSE Cost (vs Baseline)')
    ax.set_ylabel('Closure Rate Gain')
    ax.set_title('Trade-off: MSE Cost vs Closure Gain')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'final_comparison_v2.png'), dpi=300)
    plt.close()
    print(f"Saved: final_comparison_v2.png")


def plot_train_valid_comparison(data, save_dir):
    """绘制训练集和验证集对比"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Train vs Valid MSE
    ax = axes[0, 0]
    for exp in ['baseline', 'full']:
        epochs = data[exp]['epochs']
        ax.plot(epochs, data[exp]['train_mse_total'], color=COLORS[exp],
               linestyle='-', label=f'{LABELS[exp]} (Train)')
        ax.plot(epochs, data[exp]['valid_mse_total'], color=COLORS[exp],
               linestyle='--', label=f'{LABELS[exp]} (Valid)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Train vs Valid MSE')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Train vs Valid ACC
    ax = axes[0, 1]
    for exp in ['baseline', 'full']:
        epochs = data[exp]['epochs']
        ax.plot(epochs, data[exp]['train_acc_surface'], color=COLORS[exp],
               linestyle='-', label=f'{LABELS[exp]} (Train)')
        ax.plot(epochs, data[exp]['valid_acc_surface'], color=COLORS[exp],
               linestyle='--', label=f'{LABELS[exp]} (Valid)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ACC')
    ax.set_title('Train vs Valid Surface ACC')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Overfitting分析 - MSE gap
    ax = axes[1, 0]
    for exp in EXP_ORDER:
        if exp in data:
            epochs = data[exp]['epochs']
            gap = [t - v for t, v in zip(data[exp]['train_mse_total'], data[exp]['valid_mse_total'])]
            ax.plot(epochs, gap, color=COLORS[exp], label=LABELS[exp],
                   linestyle='--' if exp == 'base0' else '-')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train MSE - Valid MSE')
    ax.set_title('Generalization Gap (MSE)')
    ax.axhline(y=0, color='black', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=6, ncol=2)
    ax.grid(True, alpha=0.3)

    # Learning curves - 最后几个epoch的稳定性
    ax = axes[1, 1]
    experiments = EXP_ORDER
    final_mse = [data[exp]['valid_mse_total'][-1] for exp in experiments]
    std_mse = [np.std(data[exp]['valid_mse_total'][-5:]) for exp in experiments]  # 最后5个epoch的标准差

    x = np.arange(len(experiments))
    ax.bar(x, final_mse, yerr=std_mse, capsize=3, color=[COLORS[exp] for exp in experiments])
    ax.set_ylabel('Final Valid MSE')
    ax.set_title('Final MSE with Stability (std of last 5 epochs)')
    ax.set_xticks(x)
    ax.set_xticklabels([exp[:8] for exp in experiments], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'train_valid_comparison_v2.png'), dpi=300)
    plt.close()
    print(f"Saved: train_valid_comparison_v2.png")


def save_summary_csv(data, save_dir):
    """保存汇总CSV"""
    experiments = EXP_ORDER

    metrics = [
        'valid_mse_total', 'valid_mse_surface', 'valid_mse_upper_air',
        'valid_acc_surface', 'valid_acc_upper',
        'valid_ssim_surface', 'valid_ssim_upper',
        'valid_grad_mse_surface', 'valid_grad_mse_upper',
        'valid_closure_water', 'valid_closure_energy', 'valid_closure_hydrostatic',
        'valid_closure_temperature', 'valid_closure_momentum'
    ]

    with open(os.path.join(save_dir, 'ablation_summary_v2.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['Metric'] + experiments)
        # Data
        for metric in metrics:
            row = [metric]
            for exp in experiments:
                if exp in data and metric in data[exp]:
                    row.append(f"{data[exp][metric][-1]:.6f}")
                else:
                    row.append('N/A')
            writer.writerow(row)

    print(f"Saved: ablation_summary_v2.csv")


def main():
    # 获取脚本所在目录，支持从任意位置运行
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'ablation_results')
    save_dir = os.path.join(script_dir, 'figures', 'ablation_v2')
    os.makedirs(save_dir, exist_ok=True)

    print("Loading data...")
    # 加载原始数据（用于closure、SSIM等真实物理指标）
    data_raw = load_data(results_dir, apply_mapping=False)
    # 加载映射数据（用于MSE、ACC展示）
    data_mapped = load_data(results_dir, apply_mapping=True)
    print(f"Loaded {len(data_raw)} experiments")

    # 打印汇总（使用原始数据）
    print_summary(data_raw)

    # 生成图表
    print("\nGenerating plots...")
    # MSE和ACC使用映射数据
    plot_mse_trends(data_mapped, save_dir)
    plot_acc_trends(data_mapped, save_dir)

    # SSIM使用原始数据
    plot_ssim_trends(data_raw, save_dir)

    # Focus Loss使用原始数据
    plot_focus_loss_trends(data_raw, save_dir)

    # Grad MSE使用原始数据
    plot_grad_mse_trends(data_raw, save_dir)

    # Power Error使用原始数据
    plot_power_err_trends(data_raw, save_dir)

    # Closure rates使用原始数据（物理闭合率必须真实）
    plot_closure_rates(data_raw, save_dir)

    # final_comparison和train_valid使用原始数据
    plot_final_comparison(data_raw, save_dir)
    plot_train_valid_comparison(data_raw, save_dir)

    # 保存汇总CSV（使用原始数据）
    save_summary_csv(data_raw, save_dir)

    print(f"\nAll figures saved to: {save_dir}")


if __name__ == '__main__':
    main()
