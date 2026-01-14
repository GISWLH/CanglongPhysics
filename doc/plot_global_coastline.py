"""
绘制全球海岸线基线图
使用cartopy绘制PlateCarree投影的全球海岸线，带矩形外框
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.font_manager as font_manager

# 设置字体为Arial
font_path = "/usr/share/fonts/arial/ARIAL.TTF"
if font_path:
    try:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_name
    except:
        plt.rcParams['font.family'] = 'Arial'

# 设置图形参数
plt.rcParams.update({
    'font.size': 10,
    'axes.linewidth': 1.0,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 创建图形和坐标轴
fig = plt.figure(figsize=(12, 6))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置全球范围
ax.set_global()

# 添加海岸线
ax.coastlines(resolution='110m', linewidth=0.8, color='black')

# 添加国界线（可选）
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='gray', alpha=0.5)

# 添加经纬度网格线
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                  alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 9}
gl.ylabel_style = {'size': 9}

# 添加矩形外框
ax.spines['geo'].set_linewidth(1.5)
ax.spines['geo'].set_edgecolor('black')

# 设置标题
ax.set_title('Global Coastline Baseline (PlateCarree Projection)',
             fontsize=12, fontweight='bold', pad=10)

# 保存图形
output_path = 'global_coastline_baseline.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"图形已保存到: {output_path}")

# 同时保存为SVG格式（矢量图）
output_svg = 'global_coastline_baseline.svg'
plt.savefig(output_svg, format='svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"SVG格式已保存到: {output_svg}")

plt.close()
print("完成！")
