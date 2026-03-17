"""
Compute FuXi-S2S MJO bivariate COR confidence bands.

Strategy:
  - Load cached EOF vectors (0.25°) from mjo_cache_v35.npz
  - Downsample to 1.5° by taking every 6th longitude point
  - Load FuXi-S2S OLR/U850/U200 forecasts from Infer/eval/fuxi_s2s_target_week.nc
  - Compute tropical-average anomalies, project onto EOF → RMM indices per sample
  - Group by year (2017-2021) → compute bivariate COR per year at each lead
  - Output: per-year COR table + mean/p10/p90 confidence columns added to mjo.csv

Output:
  analysis/MJO/fuxi_mjo_ci.csv        - per-year COR values (1 row per lead×year)
  analysis/MJO/mjo.csv                - updated with FuXi CI columns
  analysis/MJO/mjo_skill_comparison.png/.svg - re-rendered with CI shading
"""

import numpy as np
import netCDF4 as nc4
import pandas as pd
import os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

# ─── Paths ─────────────────────────────────────────────────────────────────
CACHE_PATH  = os.path.join(ROOT, 'analysis/MJO/mjo_cache_v35.npz')
FUXI_NC     = os.path.join(ROOT, 'Infer/eval/fuxi_s2s_target_week.nc')
MJO_CSV     = os.path.join(ROOT, 'analysis/MJO/mjo.csv')
CI_CSV      = os.path.join(ROOT, 'analysis/MJO/fuxi_mjo_ci.csv')

N_LEADS = 6

# ─── FuXi grid constants ���───────────────────────────────────────────────────
# lat: 90, 88.5, ..., -90  (step -1.5°, 121 points)
# lon:  0,  1.5, ..., 358.5 (step +1.5°, 240 points)
FX_H, FX_W = 121, 240
FX_LATS = 90.0 - np.arange(FX_H) * 1.5      # (121,)

# Tropical band 15°N – 15°S on FuXi grid
FX_LAT_N_IDX = int((90 - 15) / 1.5)    # 50
FX_LAT_S_IDX = int((90 + 15) / 1.5)    # 70
FX_TROP_LATS = FX_LATS[FX_LAT_N_IDX : FX_LAT_S_IDX + 1]   # 21 points (15 to -15)
FX_TROP_COS  = np.cos(np.deg2rad(FX_TROP_LATS))              # (21,)


def tropical_avg_fx(field_2d):
    """Cos-weighted meridional average over tropical band for 1.5° grid.
    Input: (121, 240) -> Output: (240,)
    """
    trop = field_2d[FX_LAT_N_IDX : FX_LAT_S_IDX + 1, :]  # (21, 240)
    w    = FX_TROP_COS[:, None]                             # (21, 1)
    return (trop * w).sum(axis=0) / w.sum()


def bivariate_cor_samples(obs_rmm1, obs_rmm2, pred_rmm1, pred_rmm2, mask):
    """Bivariate COR for a subset of samples given by boolean mask.

    obs_rmm1/2: (T,)
    pred_rmm1/2: (T, L) or (T,) for a single lead
    mask: (T,) boolean
    Returns: scalar COR
    """
    a1, a2 = obs_rmm1[mask], obs_rmm2[mask]
    b1, b2 = pred_rmm1[mask], pred_rmm2[mask]
    num  = np.sum(a1 * b1 + a2 * b2)
    den  = np.sqrt(np.sum(a1**2 + a2**2) * np.sum(b1**2 + b2**2))
    return num / den if den > 1e-12 else np.nan


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    # 1. Load cached EOF / seasonal cycle (0.25° = 1440 lons)
    print('Loading cache...')
    cache = np.load(CACHE_PATH, allow_pickle=True)
    eof1_025 = cache['eof1'].astype(np.float64)    # (4320,)
    eof2_025 = cache['eof2'].astype(np.float64)
    sea_olr_025  = cache['seasonal_olr'].astype(np.float64)   # (52, 1440)
    sea_u850_025 = cache['seasonal_u850'].astype(np.float64)
    sea_u200_025 = cache['seasonal_u200'].astype(np.float64)
    fstd_olr  = float(cache['field_std_olr'])
    fstd_u850 = float(cache['field_std_u850'])
    fstd_u200 = float(cache['field_std_u200'])
    print(f'  field_std: OLR={fstd_olr:.4f}, U850={fstd_u850:.4f}, U200={fstd_u200:.4f}')

    # 2. Downsample from 0.25° (1440) to 1.5° (240)
    #    0.25° indices of 1.5° longitudes: 0, 6, 12, ..., 1434
    step = 6
    sea_olr  = sea_olr_025[:, ::step]    # (52, 240)
    sea_u850 = sea_u850_025[:, ::step]
    sea_u200 = sea_u200_025[:, ::step]

    W_FX = FX_W  # 240
    # EOF is [olr(1440) | u850(1440) | u200(1440)]
    eof1 = np.concatenate([eof1_025[:1440:step],
                            eof1_025[1440:2880:step],
                            eof1_025[2880::step]])    # (720,)
    eof2 = np.concatenate([eof2_025[:1440:step],
                            eof2_025[1440:2880:step],
                            eof2_025[2880::step]])
    print(f'  EOF shapes after downsampling: {eof1.shape}')

    # 3. Load FuXi forecasts
    print(f'\nLoading FuXi forecasts from {FUXI_NC} ...')
    ds = nc4.Dataset(FUXI_NC, 'r')
    T     = ds.dimensions['time'].size          # 254
    woys  = ds.variables['woy'][:].astype(np.int32)   # (254,)
    years = ds.variables['year'][:].astype(np.int32)  # (254,)
    print(f'  T={T}, years: {np.unique(years)}, woy range: {woys.min()}–{woys.max()}')

    # Read tropical averages for obs (no need to reproject obs – use Canglong's)
    # Actually, we need FuXi's obs to match its samples – but for a fair comparison
    # we reuse the shared ERA5 obs RMM from the cache (which has 260 samples, 2017-2021)
    # However, FuXi has only 254 samples (some weeks missing).
    # Strategy: compute obs RMM directly from FuXi NC obs fields.
    print('  Computing obs tropical averages...')
    obs_olr_trop  = np.zeros((T, W_FX), dtype=np.float64)
    obs_u850_trop = np.zeros((T, W_FX), dtype=np.float64)
    obs_u200_trop = np.zeros((T, W_FX), dtype=np.float64)
    for i in range(T):
        obs_olr_trop[i]  = tropical_avg_fx(ds.variables['obs_olr'][i])
        obs_u850_trop[i] = tropical_avg_fx(ds.variables['obs_u850'][i])
        obs_u200_trop[i] = tropical_avg_fx(ds.variables['obs_u200'][i])

    # Read pred tropical averages (T, N_LEADS, W_FX)
    pred_olr_trop  = np.zeros((T, N_LEADS, W_FX), dtype=np.float64)
    pred_u850_trop = np.zeros((T, N_LEADS, W_FX), dtype=np.float64)
    pred_u200_trop = np.zeros((T, N_LEADS, W_FX), dtype=np.float64)
    for l in range(N_LEADS):
        lid = l + 1
        print(f'  Reading pred lead {lid}...')
        for i in range(T):
            pred_olr_trop[i, l]  = tropical_avg_fx(ds.variables[f'pred_olr_lead{lid}'][i])
            pred_u850_trop[i, l] = tropical_avg_fx(ds.variables[f'pred_u850_lead{lid}'][i])
            pred_u200_trop[i, l] = tropical_avg_fx(ds.variables[f'pred_u200_lead{lid}'][i])
    ds.close()

    # 4. Compute RMM indices
    print('\nComputing RMM indices...')
    obs_rmm1 = np.zeros(T, dtype=np.float64)
    obs_rmm2 = np.zeros(T, dtype=np.float64)
    for i in range(T):
        woy = int(woys[i])
        olr_a  = (obs_olr_trop[i]  - sea_olr[woy])  / fstd_olr
        u850_a = (obs_u850_trop[i] - sea_u850[woy]) / fstd_u850
        u200_a = (obs_u200_trop[i] - sea_u200[woy]) / fstd_u200
        combined = np.concatenate([olr_a, u850_a, u200_a])
        obs_rmm1[i] = combined @ eof1
        obs_rmm2[i] = combined @ eof2

    pred_rmm1 = np.zeros((T, N_LEADS), dtype=np.float64)
    pred_rmm2 = np.zeros((T, N_LEADS), dtype=np.float64)
    for i in range(T):
        woy = int(woys[i])
        for l in range(N_LEADS):
            olr_a  = (pred_olr_trop[i, l]  - sea_olr[woy])  / fstd_olr
            u850_a = (pred_u850_trop[i, l] - sea_u850[woy]) / fstd_u850
            u200_a = (pred_u200_trop[i, l] - sea_u200[woy]) / fstd_u200
            combined = np.concatenate([olr_a, u850_a, u200_a])
            pred_rmm1[i, l] = combined @ eof1
            pred_rmm2[i, l] = combined @ eof2

    # 5. Overall bivariate COR (all samples)
    print('\nOverall FuXi-S2S COR (all samples):')
    cor_all = np.zeros(N_LEADS)
    for l in range(N_LEADS):
        num  = np.sum(obs_rmm1 * pred_rmm1[:, l] + obs_rmm2 * pred_rmm2[:, l])
        den  = np.sqrt(np.sum(obs_rmm1**2 + obs_rmm2**2) *
                       np.sum(pred_rmm1[:, l]**2 + pred_rmm2[:, l]**2))
        cor_all[l] = num / den if den > 1e-12 else np.nan
        print(f'  Lead {l+1}: COR = {cor_all[l]:.4f}')

    # 6. Per-year COR
    print('\nPer-year FuXi-S2S COR:')
    uniq_years = np.unique(years)
    cor_yearly = {}  # year → (N_LEADS,)
    for yr in uniq_years:
        mask = years == yr
        n = mask.sum()
        cor_yr = np.zeros(N_LEADS)
        for l in range(N_LEADS):
            cor_yr[l] = bivariate_cor_samples(
                obs_rmm1, obs_rmm2,
                pred_rmm1[:, l], pred_rmm2[:, l],
                mask)
        cor_yearly[int(yr)] = cor_yr
        print(f'  {yr} (n={n}): ' + ', '.join(f'{c:.3f}' for c in cor_yr))

    # 7. Confidence statistics
    cor_matrix = np.array([cor_yearly[yr] for yr in sorted(cor_yearly)])  # (5, 6)
    cor_mean = cor_matrix.mean(axis=0)
    cor_p10  = np.percentile(cor_matrix, 10, axis=0)
    cor_p90  = np.percentile(cor_matrix, 90, axis=0)
    cor_std  = cor_matrix.std(axis=0)

    print('\nConfidence statistics:')
    for l in range(N_LEADS):
        print(f'  Lead {l+1}: mean={cor_mean[l]:.4f}, '
              f'p10={cor_p10[l]:.4f}, p90={cor_p90[l]:.4f}')

    # 8. Save per-year COR CSV
    rows = []
    for yr in sorted(cor_yearly):
        for l in range(N_LEADS):
            rows.append({'year': yr, 'lead_week': l+1, 'COR': round(cor_yearly[yr][l], 6)})
    pd.DataFrame(rows).to_csv(CI_CSV, index=False)
    print(f'\nSaved per-year COR: {CI_CSV}')

    # 9. Add CI columns to mjo.csv
    df = pd.read_csv(MJO_CSV)

    # FuXi weekly days in mjo.csv: days 7, 14, 21, 28, 35, 42
    # Lead 1 = day 7, Lead 2 = day 14, ..., Lead 6 = day 42
    fuxi_days = [7, 14, 21, 28, 35, 42]

    # Initialize new columns with NaN
    for col in ['FuXi_COR_p10', 'FuXi_COR_p90']:
        df[col] = np.nan

    for l, day in enumerate(fuxi_days):
        row_mask = df['day'] == day
        if row_mask.any():
            df.loc[row_mask, 'FuXi_COR_p10'] = round(cor_p10[l], 6)
            df.loc[row_mask, 'FuXi_COR_p90'] = round(cor_p90[l], 6)

    df.to_csv(MJO_CSV, index=False)
    print(f'Updated {MJO_CSV} with FuXi CI columns')
    print(df[['day', 'FuXi-S2S_COR', 'FuXi_COR_p10', 'FuXi_COR_p90']].dropna().to_string(index=False))

    # 10. Re-render the comparison plot
    render_plot(df)


def render_plot(df):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = '/usr/share/fonts/arial/ARIAL.TTF'
    font_manager.fontManager.addfont(font_path)
    plt.style.use('seaborn-v0_8-talk')
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 10, 'axes.titlesize': 11,
        'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
        'legend.fontsize': 9, 'figure.dpi': 600,
        'lines.linewidth': 1.5, 'axes.linewidth': 1.0,
        'axes.spines.left': True, 'axes.spines.bottom': True,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.edgecolor': '#454545',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.minor.size': 3, 'ytick.minor.size': 3,
        'xtick.major.width': 1.0, 'ytick.major.width': 1.0,
        'xtick.color': '#454545', 'ytick.color': '#454545',
        'savefig.bbox': 'tight', 'savefig.transparent': False,
    })
    mpl.rcParams['svg.fonttype'] = 'none'

    # Colors
    c_fw_o     = '#D62728'
    c_fw_m     = '#1F77B4'
    c_ecmwf1   = '#444444'
    c_fuxi     = '#FF7F0E'
    c_ecmwf2   = '#8C8C8C'
    c_canglong = '#2CA02C'

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(df['day'], df['FengWu-W2S_o_ACC'], color=c_fw_o,   lw=1.6, label='FengWu-W2S$_o$')
    ax.plot(df['day'], df['FengWu-W2S_m_ACC'], color=c_fw_m,   lw=1.6, label='FengWu-W2S$_m$')
    ax.plot(df['day'], df['ECMWF_fengwu_ACC'], color=c_ecmwf1, lw=1.4, label='ECMWF (FengWu paper)')
    ax.plot(df['day'], df['FuXi-S2S_COR'],     color=c_fuxi,   lw=1.6, label='FuXi-S2S')
    ax.plot(df['day'], df['ECMWF_fuxi_COR'],   color=c_ecmwf2, lw=1.4, ls='--', label='ECMWF (FuXi paper)')

    # FuXi confidence band (p10-p90 across years)
    ci_df = df[['day', 'FuXi_COR_p10', 'FuXi_COR_p90']].dropna()
    ax.fill_between(ci_df['day'], ci_df['FuXi_COR_p10'], ci_df['FuXi_COR_p90'],
                    color=c_fuxi, alpha=0.15, label='FuXi-S2S (p10–p90)')

    # Canglong
    cl = df[['day', 'Canglong_V35_COR']].dropna()
    ax.plot(cl['day'], cl['Canglong_V35_COR'],
            color=c_canglong, lw=1.8, ls='--', marker='o',
            markersize=5, markeredgewidth=1.2, markerfacecolor='white',
            markeredgecolor=c_canglong, label='Canglong V3.5 (+2 week shift)', zorder=6)

    # Reference line
    ax.axhline(0.5, color='black', lw=1.0, ls=':', alpha=0.7)
    ax.text(1.0, 0.505, 'COR = 0.5', ha='left', va='bottom', fontsize=8,
            color='#333333', transform=ax.get_yaxis_transform())

    ax.set_xlim(0, 56)
    ax.set_ylim(0.25, 1.02)
    ax.set_xticks([0, 7, 14, 21, 28, 35, 42, 49, 56])
    ax.set_xticklabels(['0\n(W0)', '7\n(W1)', '14\n(W2)', '21\n(W3)', '28\n(W4)',
                        '35\n(W5)', '42\n(W6)', '49\n(W7)', '56\n(W8)'])
    ax.set_xlabel('Lead time (days / week)')
    ax.set_ylabel('MJO Bivariate Correlation')
    ax.set_title('MJO Prediction Skill: AI Models vs CAS-Canglong')

    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(7))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))

    ax.legend(loc='upper right', framealpha=0.92, edgecolor='#cccccc',
              ncol=1, handlelength=2.2, handletextpad=0.5,
              borderpad=0.6, labelspacing=0.4)

    out_dir = os.path.join(ROOT, 'analysis/MJO')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'mjo_skill_comparison.png'), dpi=600, bbox_inches='tight')
    plt.savefig(os.path.join(out_dir, 'mjo_skill_comparison.svg'), bbox_inches='tight')
    print('Saved: mjo_skill_comparison.png / .svg')


if __name__ == '__main__':
    main()
