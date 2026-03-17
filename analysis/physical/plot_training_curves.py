"""
绘制四个模型版本的训练曲线对比图
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['svg.hashsalt'] = 'hello'

# Nature风格参数
plt.style.use('seaborn-v0_8-talk')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'figure.figsize': (6, 4),
    'lines.linewidth': 1.0,
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

# 读取数据
df_v2_3 = pd.read_csv('training_logs_v2_3.csv')
df_v2_4 = pd.read_csv('training_logs_v2_4.csv')
df_v2_5 = pd.read_csv('training_logs_v2_5.csv')
df_v3_5 = pd.read_csv('training_logs_v3_5.csv')

# 颜色定义
color_train = '#2E86AB'  # 蓝色
color_valid = '#E94F37'  # 红色

def plot_training_curve(df, model_name, train_col, valid_col, ylabel, filename):
    """绘制单个模型的训练曲线"""
    fig, ax = plt.subplots(figsize=(6, 4))

    epochs = df['epoch']
    train_loss = df[train_col]
    valid_loss = df[valid_col]

    ax.plot(epochs, train_loss, color=color_train, label='Train Loss', linewidth=1.5)
    ax.plot(epochs, valid_loss, color=color_valid, label='Valid Loss', linewidth=1.5)

    # 标记最低验证损失点
    min_valid_idx = valid_loss.idxmin()
    min_valid_epoch = epochs[min_valid_idx]
    min_valid_value = valid_loss[min_valid_idx]
    ax.scatter([min_valid_epoch], [min_valid_value], color=color_valid, s=50, zorder=5, marker='o')
    ax.annotate(f'Min: {min_valid_value:.3f}\n(Epoch {min_valid_epoch})',
                xy=(min_valid_epoch, min_valid_value),
                xytext=(min_valid_epoch + 5, min_valid_value + 0.05),
                fontsize=8, color=color_valid)

    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{model_name} Training Curve')
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#454545')
    ax.set_xlim(0, 51)

    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(filename, dpi=600, bbox_inches='tight')
    plt.close()
    print(f'Saved: {filename}')

# 绘制四张图
plot_training_curve(df_v2_3, 'CanglongV2.3', 'train_total', 'valid_total',
                    'Total Loss', 'training_curve_v2_3.png')

plot_training_curve(df_v2_4, 'CanglongV2.4', 'train_total', 'valid_total',
                    'Total Loss', 'training_curve_v2_4.png')

plot_training_curve(df_v2_5, 'CanglongV2.5', 'train_total', 'valid_total',
                    'Total Loss', 'training_curve_v2_5.png')

plot_training_curve(df_v3_5, 'CanglongV3.5 (Physical Constraints)', 'train_total', 'valid_total',
                    'Total Loss', 'training_curve_v3_5.png')

print('\nAll training curves saved!')

# 打印分析摘要
print('\n' + '='*60)
print('Training Analysis Summary')
print('='*60)

for name, df in [('v2_3', df_v2_3), ('v2_4', df_v2_4), ('v2_5', df_v2_5), ('v3_5', df_v3_5)]:
    final_train = df['train_total'].iloc[-1]
    final_valid = df['valid_total'].iloc[-1]
    min_valid = df['valid_total'].min()
    min_valid_epoch = df.loc[df['valid_total'].idxmin(), 'epoch']
    gap = final_valid - final_train

    print(f'\n{name}:')
    print(f'  Final Train Loss: {final_train:.4f}')
    print(f'  Final Valid Loss: {final_valid:.4f}')
    print(f'  Min Valid Loss: {min_valid:.4f} (Epoch {min_valid_epoch})')
    print(f'  Train-Valid Gap: {gap:.4f}')

    # 过拟合判断
    if gap > 0.3:
        print(f'  Status: OVERFITTING (gap > 0.3)')
    elif gap > 0.2:
        print(f'  Status: Mild overfitting')
    else:
        print(f'  Status: Good generalization')
