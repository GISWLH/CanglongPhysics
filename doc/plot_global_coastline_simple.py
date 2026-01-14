"""
绘制简洁版全球海岸线基线图
仅包含海岸线和矩形外框
"""
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 设置图形参数
plt.rcParams.update({
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 创建图形和坐标轴
fig = plt.figure(figsize=(14, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置全球范围
ax.set_global()

# 仅添加海岸线 - 加粗
ax.coastlines(resolution='50m', linewidth=2.0, color='black', zorder=3)

# 添加矩形外框 - 加粗
ax.spines['geo'].set_linewidth(2.5)
ax.spines['geo'].set_edgecolor('black')

# 关闭坐标轴刻度
ax.set_xticks([])
ax.set_yticks([])

# 保存图形
output_path = 'global_coastline_simple.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.1)
print(f"Simple map saved: {output_path}")

# 同时保存为SVG格式（矢量图）
output_svg = 'global_coastline_simple.svg'
plt.savefig(output_svg, format='svg', bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.1)
print(f"SVG format saved: {output_svg}")

plt.close()
print("Complete!")
