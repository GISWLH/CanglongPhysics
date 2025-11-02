"""
Plot soil water time series and scatter plot from saved data
Usage: python plot_soil_water_line_scatter.py --lead 1
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import pickle
import argparse
import warnings
warnings.filterwarnings('ignore')

# Set up Arial font
font_path = "/usr/share/fonts/arial/ARIAL.TTF"
try:
    font_manager.fontManager.addfont(font_path)
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams['font.family'] = font_name
except:
    plt.rcParams['font.family'] = 'Arial'

# Set Nature style parameters
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
    'figure.figsize': (12, 5),
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


def plot_soil_water_line_scatter(lead_time):
    """
    Plot soil water time series and scatter plot for a specific lead time

    Parameters:
    -----------
    lead_time : int
        Lead time (1-6)
    """
    print(f"="*70)
    print(f"Plotting soil water time series and scatter for Lead {lead_time}")
    print(f"="*70)

    # Load data
    data_path = f'figures/soil_water_tcc_data_lead{lead_time}.pkl'
    print(f"\n1. Loading data from: {data_path}")

    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"\nError: Data file not found!")
        print(f"Please run: python generate_soil_water_tcc_data.py --lead {lead_time}")
        return

    print(f"   Data loaded successfully")
    print(f"   Lead time: {data['lead_time']}")
    print(f"   Time series TCC: {data['timeseries_tcc']:.4f}")

    # Extract data (now masked to land only)
    timeseries_tcc = data['timeseries_tcc']
    time_index = data['time_index']
    obs_timeseries = data['obs_timeseries']
    forecast_timeseries = data['forecast_timeseries']
    land_mask = data['land_mask']  # Land mask for reference
    
    # 系统性偏差校正：根据平均值差异进行校正
    print(f"   - Original forecast mean: {np.mean(forecast_timeseries):.6f}")
    print(f"   - Observation mean: {np.mean(obs_timeseries):.6f}")
    
    # 计算系统性偏差（平均值差异）
    systematic_bias = np.mean(obs_timeseries) - np.mean(forecast_timeseries)
    print(f"   - Systematic bias (obs - forecast): {systematic_bias:.6f}")
    
    # 应用系统性偏差校正：所有预报值都加上偏差值
    corrected_forecast = forecast_timeseries + systematic_bias
    print(f"   - Corrected forecast mean: {np.mean(corrected_forecast):.6f}")
    
    # 重新计算TCC（校正后的）
    timeseries_tcc_corrected = np.corrcoef(obs_timeseries, corrected_forecast)[0, 1]
    
    # 使用校正后的数据作为展示数据
    adjusted_forecast = corrected_forecast
    timeseries_tcc_adjusted = timeseries_tcc_corrected
    
    print(f"   - Original TCC: {timeseries_tcc:.4f}")
    print(f"   - Corrected TCC: {timeseries_tcc_corrected:.4f}")
    print(f"   - Land points used: {np.sum(land_mask)} out of {land_mask.size}")

    # Create figure with 2 subplots
    print(f"\n2. Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Time series (Method 2)
    print(f"   - Plotting soil water time series...")
    ax1 = axes[0]
    ax1.set_prop_cycle(plt.cycler('color', plt.cm.Set2.colors))
    ax1.plot(time_index, obs_timeseries, '-', linewidth=2,
             label='Observation', alpha=0.8)
    ax1.plot(time_index, corrected_forecast, '-', linewidth=1.5,
             label='CAS-Canglong (bias-corrected)', alpha=0.8)  # Removed "(adjusted)" for soil water
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_title(f'Method 2: Global Mean Soil Water Time Series (Lead {lead_time})\n' +
                  '(Spatial mean then temporal correlation)',
                  fontweight='bold', fontsize=11)
    ax1.set_xlabel('Time Index', fontsize=10)
    ax1.set_ylabel('Soil Water Anomaly (m³/m³)', fontsize=10)
    ax1.legend(loc='lower left', frameon=True, fancybox=False,
               edgecolor='#454545', framealpha=1)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add text box with TCC in top-left corner
    ax1.text(0.02, 0.98, f'TACC = {timeseries_tcc_corrected:.4f}',  # Use original TCC for soil water
             transform=ax1.transAxes, va='top', ha='left',
             fontweight='bold', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.9, edgecolor='black', linewidth=1))

    # Plot 2: Scatter plot showing correlation
    print(f"   - Plotting soil water scatter plot...")
    ax2 = axes[1]

    # Scatter plot with fixed color (use bias-corrected forecast for soil water)
    scatter = ax2.scatter(obs_timeseries, corrected_forecast,
                         alpha=0.6, s=50, c='#9fc5e8',
                         edgecolors='black', linewidth=0.5)

    # Perfect correlation line with extended range to avoid compression
    data_min = min(obs_timeseries.min(), corrected_forecast.min())
    data_max = max(obs_timeseries.max(), corrected_forecast.max())
    
    # Extend the range by 10% to avoid compression
    data_range = data_max - data_min
    lim_min = data_min - 0.1 * data_range
    lim_max = data_max + 0.1 * data_range
    
    # Plot 1:1 line with extended range
    ax2.plot([lim_min, lim_max], [lim_min, lim_max], 'k--',
             linewidth=2, alpha=0.5, label='1:1 line')

    # Fit line with confidence interval (use bias-corrected forecast for soil water)
    # Calculate linear regression
    slope, intercept = np.polyfit(obs_timeseries, corrected_forecast, 1)
    
    # Calculate predicted values and residuals
    y_pred = slope * obs_timeseries + intercept
    residuals = corrected_forecast - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((corrected_forecast - np.mean(corrected_forecast)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Standard error of the estimate
    n = len(obs_timeseries)
    x_mean = np.mean(obs_timeseries)
    sxx = np.sum((obs_timeseries - x_mean) ** 2)
    stderr = np.sqrt(ss_res / max(n - 2, 1))
    
    # Confidence interval similar to ggplot's geom_smooth (wider at edges)
    x_vals = np.linspace(lim_min, lim_max, 200)
    y_vals = slope * x_vals + intercept
    if sxx > 0:
        se_line = stderr * np.sqrt(1 / max(n, 1) + ((x_vals - x_mean) ** 2) / sxx)
    else:
        se_line = np.full_like(x_vals, stderr)
    t_val = 1.96  # Approximate 95% confidence
    ci = t_val * se_line
    
    # Plot fit line
    ax2.plot(x_vals, y_vals, color='#3c78d8', linewidth=2, alpha=0.7,
             label='Fitted')
    
    # Plot confidence interval - wider at edges, narrower near mean
    ax2.fill_between(x_vals, y_vals - ci, y_vals + ci, 
                     color='#3c78d8', alpha=0.1, label='95% CI')

    ax2.set_title(f'Method 2: Soil Water Scatter Plot (Lead {lead_time})\n' +
                  '(Bias-corrected, shows correlation strength)',
                  fontweight='bold', fontsize=11)
    ax2.set_xlabel('Observed Soil Water Anomaly (m³/m³)', fontsize=10)
    ax2.set_ylabel('Forecast Soil Water Anomaly (m³/m³)', fontsize=10)
    ax2.legend(loc='best', frameon=True, fancybox=False,
               edgecolor='#454545', framealpha=1, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    # Set equal aspect ratio
    ax2.set_aspect('equal', adjustable='box')
    
    # Set axis limits to prevent compression
    ax2.set_xlim(lim_min, lim_max)
    ax2.set_ylim(lim_min, lim_max)

    # Overall title
    fig.suptitle(f'Soil Water Time Series and Scatter Plot: CAS-Canglong Lead {lead_time} Week (2022-2023)\nBias-corrected for systematic error',
                 fontsize=14, fontweight='bold', y=1.00)

    plt.tight_layout()

    # Save figure
    output_png = f'figures/soil_water_tcc_line_scatter_lead{lead_time}.png'
    output_svg = f'figures/soil_water_tcc_line_scatter_lead{lead_time}.svg'

    print(f"\n3. Saving figures...")
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    print(f"   PNG saved: {output_png}")

    plt.savefig(output_svg, bbox_inches='tight')
    print(f"   SVG saved: {output_svg}")


    print("\n[OK] Soil water time series and scatter plotting completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot soil water time series and scatter from saved data')
    parser.add_argument('--lead', type=int, required=True, choices=[1,2,3,4,5,6],
                        help='Lead time (1-6)')

    args = parser.parse_args()

    plot_soil_water_line_scatter(args.lead)