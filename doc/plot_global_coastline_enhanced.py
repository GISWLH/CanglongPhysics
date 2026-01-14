"""
绘制增强版全球海岸线基线图
使用cartopy绘制PlateCarree投影的全球海岸线，带矩形外框和更多地理要素
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
    'axes.linewidth': 1.5,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# 创建图形和坐标轴
fig = plt.figure(figsize=(14, 7))
ax = plt.axes(projection=ccrs.PlateCarree())

# 设置全球范围
ax.set_global()

# 添加海洋和陆地底色
ax.add_feature(cfeature.OCEAN, facecolor='#e0f3ff', zorder=0)
ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', zorder=1)

# 添加海岸线 - 使用更高分辨率
ax.coastlines(resolution='50m', linewidth=1.0, color='black', zorder=3)

# 添加国界线
ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='#666666',
               alpha=0.6, linestyle='--', zorder=2)

# 添加湖泊
ax.add_feature(cfeature.LAKES, facecolor='#e0f3ff', edgecolor='black',
               linewidth=0.5, alpha=0.8, zorder=2)

# 添加主要河流
ax.add_feature(cfeature.RIVERS, edgecolor='#4da6ff',
               linewidth=0.5, alpha=0.7, zorder=2)

# 添加经纬度网格线
gl = ax.gridlines(draw_labels=True, linewidth=0.8, color='gray',
                  alpha=0.5, linestyle='--', zorder=4)
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 10, 'color': 'black'}
gl.ylabel_style = {'size': 10, 'color': 'black'}

# 添加矩形外框 - 更粗的边框
ax.spines['geo'].set_linewidth(2.0)
ax.spines['geo'].set_edgecolor('black')

# 设置标题
ax.set_title('Global Coastline Baseline Map (PlateCarree Projection)',
             fontsize=14, fontweight='bold', pad=15)

# 添加副标题
ax.text(0.5, -0.08, 'Resolution: 50m | Features: Coastlines, Borders, Lakes, Rivers',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=9, style='italic', color='#666666')

# 保存图形
output_path = 'global_coastline_enhanced.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"Enhanced map saved: {output_path}")

# 同时保存为SVG格式（矢量图）
output_svg = 'global_coastline_enhanced.svg'
plt.savefig(output_svg, format='svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"SVG format saved: {output_svg}")

plt.close()
print("Complete!")
