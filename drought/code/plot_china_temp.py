import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import cmaps
import cartopy.crs as ccrs
import salem
import geopandas as gpd
import mplotutils as mpu
from utils import plot
import matplotlib as mpl
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 设置全局字体为 Arial
try:
    # 尝试直接设置为Arial
    plt.rcParams['font.family'] = 'Arial'
    # 检查Arial是否可用
    if 'Arial' not in set(f.name for f in font_manager.fontManager.ttflist):
        raise ValueError("Arial not found in system fonts.")
except Exception:
    # 如果Arial不可用，则加载指定路径的字体
    font_path = "/usr/share/fonts/arial/ARIAL.TTF"
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name

# 设置Nature风格参数
plt.style.use('seaborn-v0_8-talk')
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'figure.figsize': (6, 4),
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
mpl.rcParams['svg.fonttype'] = 'none'
mpl.rcParams['svg.hashsalt'] = 'hello'

print("Loading temperature data for 2022-07-02...")

print("Loading temperature climatology from E:/data/climate_variables_2000_2023_weekly.nc...")
# 使用与降水相同的气候态数据源
clim_path = 'E:/data/climate_variables_2000_2023_weekly.nc'
ds_clim = xr.open_dataset(clim_path)

# 计算每个时间步的年中周数作为DataArray
week_of_year = ds_clim['time'].dt.isocalendar().week

# 计算气候态：每年同一周的平均值
print("Calculating weekly temperature climatology (mean over 2000-2023)...")
t2m_clim = ds_clim['t2m'].groupby(week_of_year).mean('time')

print(f"Temperature climatology shape: {t2m_clim.shape}")
print(f"Week range: {t2m_clim.week.min().values} to {t2m_clim.week.max().values}")

# 关闭气候态数据集
ds_clim.close()

# 目标日期
target_date = '2022-07-02'

# 1. 读取中国边界shapefile
china_shp = gpd.read_file('data/china.shp')

# 2. 打开数据集并筛选2022-07-02这一天的气温数据
lead1 = xr.open_dataset('Z:/Data/hindcast_2022_2023/hindcast_2022_2023_lead1.nc')['2m_temperature'].sel(time=target_date).rename({'latitude': 'lat', 'longitude': 'lon'})
lead2 = xr.open_dataset('Z:/Data/hindcast_2022_2023/hindcast_2022_2023_lead2.nc')['2m_temperature'].sel(time=target_date).rename({'latitude': 'lat', 'longitude': 'lon'})
lead3 = xr.open_dataset('Z:/Data/hindcast_2022_2023/hindcast_2022_2023_lead3.nc')['2m_temperature'].sel(time=target_date).rename({'latitude': 'lat', 'longitude': 'lon'})
lead4 = xr.open_dataset('Z:/Data/hindcast_2022_2023/hindcast_2022_2023_lead4.nc')['2m_temperature'].sel(time=target_date).rename({'latitude': 'lat', 'longitude': 'lon'})
lead5 = xr.open_dataset('Z:/Data/hindcast_2022_2023/hindcast_2022_2023_lead5.nc')['2m_temperature'].sel(time=target_date).rename({'latitude': 'lat', 'longitude': 'lon'})
lead6 = xr.open_dataset('Z:/Data/hindcast_2022_2023/hindcast_2022_2023_lead6.nc')['2m_temperature'].sel(time=target_date).rename({'latitude': 'lat', 'longitude': 'lon'})

# 观测数据
obs = xr.open_dataset('Z:/Data/hindcast_2022_2023/hindcast_2022_2023_lead1.nc')['2m_temperature_obs'].sel(time=target_date).rename({'latitude': 'lat', 'longitude': 'lon'})

# 3. 将6个lead数据放入列表
lead_data = [lead1, lead2, lead3, lead4, lead5, lead6]

# 4. 计算气温距平
def calculate_temperature_anomaly(data, climatology, target_date):
    """计算气温距平"""
    # 获取目标日期的周数 (2022-07-02是第27周)
    target_week = 27
    
    # 获取对应周的气候态值
    clim_week = climatology.sel(week=target_week)
    
    # 计算距平: 观测值 - 气候态
    anomaly = data - clim_week
    
    return anomaly

# 选取中国区域并创建陆地掩膜
def process_china_anomaly_data(data, climatology, china_shp, target_date):
    """处理中国区域数据并计算距平"""
    # 计算距平
    anomaly_data = calculate_temperature_anomaly(data, climatology, target_date)
    
    # 选取中国区域
    china_anomaly = anomaly_data.sel(
        lon=slice(70, 140),
        lat=slice(55, 15)
    )
    
    # 使用salem创建掩膜，只显示中国陆地
    ds_t = salem.DataArrayAccessor(china_anomaly)
    masked_data = ds_t.roi(shape=china_shp)
    
    return masked_data

# 处理每个lead的数据
china_lead_anomaly_data = []
for i, lead in enumerate(lead_data):
    print(f"Processing Lead {i+1} temperature anomaly...")
    china_anomaly = process_china_anomaly_data(lead, t2m_clim, china_shp, target_date)
    china_lead_anomaly_data.append(china_anomaly)

# 处理观测数据
print("Processing Observation temperature anomaly...")
china_obs_anomaly = process_china_anomaly_data(obs, t2m_clim, china_shp, target_date)

# 5. 设定色标和范围（适用于气温距平）
# 气温距平范围，单位：K
vmin, vmax = -5, 5  # 气温距平通常在-5到5K范围内
unit_label = 'K'
title_prefix = 'CAS-Canglong'
# 使用适合距平的颜色图（蓝-白-红）
data_cmap = cmaps.BlueWhiteOrangeRed_r  # 蓝-白-红颜色图，适合距平

# 6. 创建图形和投影（7个子图：6个lead + 1个观测）
fig = plt.figure(figsize=(49, 28))  # 调整为7个子图的大小
axes = []
for i in range(7):
    ax = fig.add_subplot(2, 4, i+1, projection=ccrs.LambertConformal(
        central_longitude=105,
        central_latitude=40,
        standard_parallels=(25.0, 47.0)
    ))
    axes.append(ax)

levels = np.linspace(vmin, vmax, 21)  # 21个分级
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# 7. 绘制6个lead时间的气温距平数据
mappable = None
for t in range(6):
    ax = axes[t]
    current_data = china_lead_anomaly_data[t]
    
    # 绘图
    mappable = plot.one_map_china(
        current_data, 
        ax, 
        cmap=data_cmap, 
        levels=levels,
        norm=norm,
        mask_ocean=False, 
        add_coastlines=True, 
        add_land=False, 
        add_river=True, 
        add_lake=True, 
        add_stock=False, 
        add_gridlines=True, 
        colorbar=False, 
        plotfunc="pcolormesh"
    )
    
    # 添加小地图（九段线）
    ax2 = fig.add_axes([0.192 + (t % 4) * 0.195, 0.0500 + (1 - t // 4) * 0.4800, 0.05, 0.075], 
                      projection=ccrs.LambertConformal(
                          central_longitude=105,
                          central_latitude=40,
                          standard_parallels=(25.0, 47.0)
                      ))
    plot.sub_china_map(current_data, ax2, cmap=data_cmap, add_coastlines=False, add_land=False)
    
    # 时间标签
    start_date = '20220702'
    lead_start = np.datetime64('2022-07-02') + np.timedelta64(t*7, 'D')
    lead_end = lead_start + np.timedelta64(6, 'D')
    start_str = str(lead_start).replace('-', '')
    end_str = str(lead_end).replace('-', '')
    
    ax.set_title(f'{title_prefix} Lead{t+1} for {start_str}-{end_str}', fontsize=20, fontfamily='Arial')

# 8. 绘制观测气温距平数据
ax = axes[6]
current_data = china_obs_anomaly

# 绘图
mappable = plot.one_map_china(
    current_data, 
    ax, 
    cmap=data_cmap, 
    levels=levels,
    norm=norm,
    mask_ocean=False, 
    add_coastlines=True, 
    add_land=False, 
    add_river=True, 
    add_lake=True, 
    add_stock=False, 
    add_gridlines=True, 
    colorbar=False, 
    plotfunc="pcolormesh"
)

# 添加小地图（九段线）
ax2 = fig.add_axes([0.192 + (6 % 4) * 0.195, 0.0500 + (1 - 6 // 4) * 0.4800, 0.05, 0.075], 
                  projection=ccrs.LambertConformal(
                      central_longitude=105,
                      central_latitude=40,
                      standard_parallels=(25.0, 47.0)
                  ))
plot.sub_china_map(current_data, ax2, cmap=data_cmap, add_coastlines=False, add_land=False)

# 设置标题
ax.set_title('Observation for 20220702', fontsize=20, fontfamily='Arial')

# 9. 使用mpu添加色标
cbar_ax = fig.add_axes([0.88, 0.15, 0.01, 0.7])  # 调整色标位置和大小
cbar = fig.colorbar(mappable, cax=cbar_ax)
cbar.set_label('Temperature Anomaly (K)', fontsize=20)
cbar.ax.tick_params(labelsize=20)  # 设置刻度标签字体大小

# 10. 调整布局
plt.subplots_adjust(left=0.025, right=0.85, top=0.9, bottom=0.05, wspace=0.2, hspace=0.3)
mpu.set_map_layout(axes, width=80)

# 11. 保存图片
output_path = 'figures/temperature_anomaly_20220702.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {output_path}")

# 也保存为SVG格式
output_path_svg = 'figures/temperature_anomaly_20220702.svg'
plt.savefig(output_path_svg, format='svg', bbox_inches='tight')
print(f"Figure saved to: {output_path_svg}")

plt.show()

# 关闭气候态数据集（如果还未关闭）
try:
    ds_clim.close()
except:
    pass

print("Temperature anomaly processing completed!")