"""
CAS-Canglong SST16 ENSO Forecast Plot
=====================================
Read `ENSO_all.csv` and `ENSO_pre_temp.csv`, then draw a report-style ENSO forecast figure.

Usage:
    conda activate torch
    python analysis/operation/SSTmodel/plot_enso_forecast.py
"""

from datetime import datetime
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None


SCRIPT_DIR = Path(__file__).resolve().parent
ENSO_OBS_CSV = SCRIPT_DIR / 'ENSO_all.csv'
ENSO_PRE_CSV = SCRIPT_DIR / 'ENSO_pre_temp.csv'
FORECAST_START = '202603'
FC_DATE = datetime.strptime(FORECAST_START, '%Y%m')
PRED_MONTHS = 12
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

COLOR_OBS = '#20262E'
COLOR_MEAN = '#005B96'
COLOR_MEMBER = '#63B3ED'
COLOR_BAND_OUTER = '#D9ECFF'
COLOR_BAND_INNER = '#93CFFF'
COLOR_WARM = '#D55E00'
COLOR_COOL = '#0072B2'
COLOR_ZERO = '#7A7A7A'
COLOR_FORECAST_BG = '#F4F8FC'
COLOR_GRID = '#D7E0EA'
COLOR_TEXT = '#1F2937'
COLOR_SUBTEXT = '#64748B'


font_candidates = [
    '/usr/share/fonts/arial/ARIAL.TTF',
    '/usr/share/fonts/truetype/msttcorefonts/Arial.ttf',
    '/usr/share/fonts/truetype/msttcorefonts/arial.ttf',
]

font_family = 'DejaVu Sans'
for font_path in font_candidates:
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        font_family = 'Arial'
        break

mpl.rcParams.update({
    'font.family': font_family,
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 600,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.0,
    'axes.edgecolor': '#64748B',
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.color': '#475569',
    'ytick.color': '#475569',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
})


def classify_enso_phase(value):
    if value >= 0.5:
        return 'Warm-event threshold exceeded'
    if value <= -0.5:
        return 'Cold-event threshold exceeded'
    return 'Neutral-range mean'



def build_month_labels(dates):
    labels = []
    for index, current_date in enumerate(dates):
        if index == 0 or current_date.month == 1:
            labels.append(f'{current_date:%b}\n{current_date:%Y}')
        else:
            labels.append(f'{current_date:%b}')
    return labels


def smooth_curve(x, y, points_per_step=40):
    if len(x) < 3:
        return x, y

    x_smooth = np.linspace(x[0], x[-1], int((x[-1] - x[0]) * points_per_step) + 1)
    if PchipInterpolator is not None:
        y_smooth = PchipInterpolator(x, y)(x_smooth)
    else:
        y_smooth = np.interp(x_smooth, x, y)
    return x_smooth, y_smooth


df_obs_raw = pd.read_csv(ENSO_OBS_CSV)
df_pre = pd.read_csv(ENSO_PRE_CSV)

clim_rows = df_obs_raw[(df_obs_raw['Year'] >= 1991) & (df_obs_raw['Year'] <= 2020)]
enso_clim = {}
for month_index, month_name in enumerate(MONTH_NAMES, 1):
    enso_clim[month_index] = clim_rows[month_name].astype(float).mean()

obs_dates = []
obs_ssta = []
for month_offset in [3, 2, 1, 0]:
    current_date = FC_DATE - relativedelta(months=month_offset)
    row = df_obs_raw[df_obs_raw['Year'] == current_date.year]
    if len(row) == 0:
        continue
    value = row[MONTH_NAMES[current_date.month - 1]].values[0]
    if pd.notna(value):
        obs_dates.append(current_date)
        obs_ssta.append(float(value) - enso_clim[current_date.month])

pred_dates = [datetime.strptime(date_text, '%Y-%m') for date_text in df_pre['Date']]
pred_ssta = df_pre['SSTA'].to_numpy(dtype=float)
pred_std = df_pre['SSTA_STD'].to_numpy(dtype=float)
member_cols = sorted([col for col in df_pre.columns if col.startswith('SSTA_') and col[5:].isdigit()])
member_ssta = df_pre[member_cols].to_numpy(dtype=float) if member_cols else None

all_dates = sorted(set(obs_dates + pred_dates))
all_labels = build_month_labels(all_dates)
x_map = {current_date: index for index, current_date in enumerate(all_dates)}

x_obs = np.array([x_map[current_date] for current_date in obs_dates])
y_obs = np.array(obs_ssta)
x_pred = np.array([x_map[current_date] for current_date in pred_dates])
y_pred = pred_ssta

obs_anchor_date = obs_dates[-1]
obs_anchor_ssta = obs_ssta[-1]
x_anchor = x_map[obs_anchor_date]
x_pred_fan = x_pred[1:]
y_pred_fan = y_pred[1:]

band_x = np.concatenate(([x_anchor], x_pred_fan))
if member_ssta is not None:
    q10, q25, q75, q90 = np.percentile(member_ssta[1:], [10, 25, 75, 90], axis=1)
    band_q10 = np.concatenate(([obs_anchor_ssta], q10))
    band_q25 = np.concatenate(([obs_anchor_ssta], q25))
    band_q75 = np.concatenate(([obs_anchor_ssta], q75))
    band_q90 = np.concatenate(([obs_anchor_ssta], q90))
else:
    band_q10 = np.concatenate(([obs_anchor_ssta], y_pred_fan - pred_std[1:]))
    band_q25 = np.concatenate(([obs_anchor_ssta], y_pred_fan - 0.5 * pred_std[1:]))
    band_q75 = np.concatenate(([obs_anchor_ssta], y_pred_fan + 0.5 * pred_std[1:]))
    band_q90 = np.concatenate(([obs_anchor_ssta], y_pred_fan + pred_std[1:]))

mean_x = np.concatenate(([x_anchor], x_pred_fan))
mean_y = np.concatenate(([obs_anchor_ssta], y_pred_fan))
forecast_x = x_map[FC_DATE]
pred_end = FC_DATE + relativedelta(months=PRED_MONTHS - 1)

fig, ax = plt.subplots(figsize=(11.2, 6.2))
fig.subplots_adjust(left=0.08, right=0.96, bottom=0.18, top=0.78)

ax.axvspan(forecast_x - 0.5, len(all_dates) - 0.5, color=COLOR_FORECAST_BG, zorder=0)
ax.axhspan(0.5, 2.0, color=COLOR_WARM, alpha=0.04, zorder=0)
ax.axhspan(-2.0, -0.5, color=COLOR_COOL, alpha=0.04, zorder=0)

ax.axhline(0.0, color=COLOR_ZERO, linewidth=1.0, zorder=1)
ax.axhline(0.5, color=COLOR_WARM, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.9, zorder=1)
ax.axhline(-0.5, color=COLOR_COOL, linestyle=(0, (4, 3)), linewidth=1.0, alpha=0.9, zorder=1)
ax.axvline(forecast_x - 0.5, color='#94A3B8', linestyle=(0, (3, 3)), linewidth=1.1, zorder=1)

ax.grid(axis='y', color=COLOR_GRID, linewidth=0.8)
ax.grid(axis='x', visible=False)
ax.set_axisbelow(True)

smooth_band_x, smooth_q10 = smooth_curve(band_x, band_q10)
_, smooth_q25 = smooth_curve(band_x, band_q25)
_, smooth_q75 = smooth_curve(band_x, band_q75)
_, smooth_q90 = smooth_curve(band_x, band_q90)

smooth_q25 = np.maximum(smooth_q25, smooth_q10)
smooth_q75 = np.minimum(smooth_q75, smooth_q90)

ax.fill_between(smooth_band_x, smooth_q10, smooth_q90, color=COLOR_BAND_OUTER, alpha=0.2, zorder=2)
ax.fill_between(smooth_band_x, smooth_q25, smooth_q75, color=COLOR_BAND_INNER, alpha=0.2, zorder=3)

if member_ssta is not None:
    for member_index in range(member_ssta.shape[1]):
        member_y = np.concatenate(([obs_anchor_ssta], member_ssta[1:, member_index]))
        member_x = np.concatenate(([x_anchor], x_pred_fan))
        ax.plot(
            member_x,
            member_y,
            color=COLOR_MEMBER,
            linewidth=0.9,
            alpha=0.23,
            zorder=4,
        )

ax.plot(
    x_obs,
    y_obs,
    color=COLOR_OBS,
    marker='o',
    markersize=5.8,
    linewidth=2.0,
    markerfacecolor='white',
    markeredgecolor=COLOR_OBS,
    markeredgewidth=1.4,
    zorder=6,
)

ax.plot(
    mean_x,
    mean_y,
    color=COLOR_MEAN,
    marker='o',
    markersize=6.0,
    linewidth=2.6,
    markerfacecolor=COLOR_MEAN,
    markeredgecolor='white',
    markeredgewidth=0.8,
    zorder=7,
)

final_x = mean_x[-1]
final_y = mean_y[-1]
ax.scatter(final_x, final_y, s=78, color=COLOR_MEAN, edgecolor='white', linewidth=1.0, zorder=8)

ax.text(
    0.14,
    0.96,
    'Observed',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=9,
    color=COLOR_SUBTEXT,
    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', edgecolor='none', alpha=0.95),
)
ax.text(
    0.60,
    0.96,
    'Forecast',
    transform=ax.transAxes,
    ha='center',
    va='center',
    fontsize=9,
    color=COLOR_SUBTEXT,
    bbox=dict(boxstyle='round,pad=0.25', facecolor='#EAF2FB', edgecolor='none', alpha=1.0),
)

ax.text(len(all_dates) - 0.35, 0.54, 'El Niño threshold', ha='right', va='bottom', fontsize=8.5, color=COLOR_WARM)

ax.set_xticks(np.arange(len(all_dates)))
ax.set_xticklabels(all_labels)
ax.tick_params(axis='x', length=0, pad=8)
ax.tick_params(axis='y', which='major', length=6)
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.set_xlabel('Valid month', labelpad=10)
ax.set_ylabel('Nino 3.4 SSTA (°C)', labelpad=8)
ax.set_ylim(-2.0, 2.0)
ax.set_xlim(-0.5, len(all_dates) - 0.5)

legend_handles = [
    Line2D([0], [0], color=COLOR_OBS, marker='o', markersize=5.8, linewidth=2.0,
           markerfacecolor='white', markeredgecolor=COLOR_OBS, markeredgewidth=1.3,
           label='Observed'),
    Patch(facecolor=COLOR_BAND_OUTER, edgecolor='none', label='Ensemble spread (10–90%)'),
    Patch(facecolor=COLOR_BAND_INNER, edgecolor='none', label='Ensemble spread (25–75%)'),
    Line2D([0], [0], color=COLOR_MEMBER, linewidth=1.1, alpha=0.35, label='Ensemble members'),
    Line2D([0], [0], color=COLOR_MEAN, marker='o', markersize=6.0, linewidth=2.6,
           markerfacecolor=COLOR_MEAN, markeredgecolor='white', markeredgewidth=0.8,
           label='CAS-Canglong mean'),
]
fig.legend(
    handles=legend_handles,
    loc='upper left',
    bbox_to_anchor=(0.08, 0.855),
    frameon=False,
    ncol=3,
    handlelength=2.6,
    columnspacing=1.2,
)

fig.suptitle(
    'Nino 3.4 SST anomaly forecast',
    x=0.08,
    y=0.948,
    ha='left',
    fontsize=15,
    fontweight='bold',
    color=COLOR_TEXT,
)
fig.text(
    0.08,
    0.895,
    f'CAS-Canglong SST16 ensemble forecast initialized {FC_DATE:%B %Y}; valid through {pred_end:%B %Y}; '
    'anomalies relative to 1991–2020 climatology',
    ha='left',
    fontsize=10,
    color=COLOR_SUBTEXT,
)
save_path = SCRIPT_DIR / f'enso_prediction_{FORECAST_START}.png'
fig.savefig(save_path, dpi=600)
fig.savefig(save_path.with_suffix('.svg'))
print(f'Figure saved: {save_path}')
plt.close(fig)
print('Done.')
