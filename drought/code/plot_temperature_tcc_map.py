"""
Plot temperature TCC map from saved data
Usage: python plot_temperature_tcc_map.py --lead 1
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import pickle
import argparse
import warnings
import xarray as xr
import cartopy.crs as ccrs
from utils import plot
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
    'figure.figsize': (8, 6),
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


def plot_temperature_tcc_map(lead_time):
    """
    Plot temperature TCC map for a specific lead time

    Parameters:
    -----------
    lead_time : int
        Lead time (1-6)
    """
    print(f"="*70)
    print(f"Plotting temperature TCC map for Lead {lead_time}")
    print(f"="*70)

    # Load data
    data_path = f'figures/temperature_tcc_data_lead{lead_time}.pkl'
    print(f"\n1. Loading data from: {data_path}")

    try:
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f"\nError: Data file not found!")
        print(f"Please run: python generate_temperature_tcc_data.py --lead {lead_time}")
        return

    print(f"   Data loaded successfully")
    print(f"   Lead time: {data['lead_time']}")
    print(f"   Gridpoint TCC: {data['gridpoint_tcc']:.4f}")

    # Extract data
    tcc_map = data['tcc_map']
    gridpoint_tcc = data['gridpoint_tcc']

    # Create figure with geographic projection
    print(f"\n2. Creating visualization...")
    print(f"   - Plotting temperature TCC map with geographic projection...")
    
    # Create DataArray for plotting
    lat = np.linspace(90, -90, tcc_map.shape[0])
    lon = np.linspace(0, 359.75, tcc_map.shape[1])
    dataarray = xr.DataArray(
        tcc_map,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        name="tcc"
    )
    
    # Create figure with geographic projection
    fig = plt.figure(figsize=(8, 6))
    proj = ccrs.Robinson(central_longitude=180)
    ax = fig.add_subplot(111, projection=proj)
    
    # Plot using one_map_flat
    levels = np.linspace(-1, 1, num=21)
    plot.one_map_flat(dataarray, ax, levels=levels, cmap="RdBu_r", mask_ocean=False, 
                      add_coastlines=True, add_land=False, plotfunc="pcolormesh", colorbar=True)
    
    ax.set_title(f'Temperature TCC Map (Lead {lead_time})\n' +
                 '(Temporal correlation at each point)',
                 fontweight='bold', fontsize=11)
    
    # Add text box with mean TCC
    ax.text(0.02, 0.98, f'Mean TCC = {gridpoint_tcc:.4f}',
             transform=ax.transAxes, va='top', ha='left',
             fontweight='bold', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='white',
                      alpha=0.9, edgecolor='black', linewidth=1))

    # Save figure
    output_png = f'figures/temperature_tcc_map_lead{lead_time}.png'
    output_svg = f'figures/temperature_tcc_map_lead{lead_time}.svg'

    print(f"\n3. Saving figures...")
    plt.savefig(output_png, dpi=600, bbox_inches='tight')
    print(f"   PNG saved: {output_png}")

    plt.savefig(output_svg, bbox_inches='tight')
    print(f"   SVG saved: {output_svg}")

    plt.show()

    print("\n[OK] Temperature TCC map plotting completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot temperature TCC map from saved data')
    parser.add_argument('--lead', type=int, required=True, choices=[1,2,3,4,5,6],
                        help='Lead time (1-6)')

    args = parser.parse_args()

    plot_temperature_tcc_map(args.lead)