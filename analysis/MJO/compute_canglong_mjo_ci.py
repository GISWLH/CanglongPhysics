"""
Compute Canglong V3.5 MJO bivariate COR confidence bands (per-year).

Uses cached RMM indices from mjo_cache_v35.npz and year info from model_v3.nc.
Adds Canglong_COR_p10 / Canglong_COR_p90 columns to mjo.csv at the
+2-week shifted day positions (21, 28, 35, 42, 49, 56).
Re-renders mjo_skill_comparison.png / .svg with CI shading.
"""

import numpy as np
import netCDF4 as nc4
import pandas as pd
import os

ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
CACHE   = os.path.join(ROOT, 'analysis/MJO/mjo_cache_v35.npz')
EVAL_NC = os.path.join(ROOT, 'Infer/eval/model_v3.nc')
MJO_CSV = os.path.join(ROOT, 'analysis/MJO/mjo.csv')
CI_CSV  = os.path.join(ROOT, 'analysis/MJO/canglong_mjo_ci.csv')

N_LEADS = 6
# Canglong lead→mjo.csv day mapping (lead 1 week = actual day 7, shifted +14 → day 21)
CANGLONG_DAYS = [21, 28, 35, 42, 49, 56]


def bivariate_cor(a1, a2, b1, b2):
    num = np.sum(a1 * b1 + a2 * b2)
    den = np.sqrt(np.sum(a1**2 + a2**2) * np.sum(b1**2 + b2**2))
    return num / den if den > 1e-12 else np.nan


def main():
    # 1. Load cached RMM indices
    cache = np.load(CACHE, allow_pickle=True)
    obs_rmm1  = cache['obs_rmm1']   # (260,)
    obs_rmm2  = cache['obs_rmm2']
    pred_rmm1 = cache['pred_rmm1']  # (260, 6)
    pred_rmm2 = cache['pred_rmm2']

    # 2. Load year info from eval NC
    ds = nc4.Dataset(EVAL_NC, 'r')
    years = ds.variables['year'][:].astype(np.int32)  # (260,)
    ds.close()

    # 3. Overall COR
    print('Overall Canglong V3.5 COR:')
    cor_all = np.zeros(N_LEADS)
    for l in range(N_LEADS):
        cor_all[l] = bivariate_cor(obs_rmm1, obs_rmm2, pred_rmm1[:, l], pred_rmm2[:, l])
        print(f'  Lead {l+1}: COR = {cor_all[l]:.4f}')

    # 4. Per-year COR
    print('\nPer-year Canglong V3.5 COR:')
    uniq_years = np.unique(years)
    cor_yearly = {}
    for yr in uniq_years:
        mask = years == yr
        n = mask.sum()
        cor_yr = np.array([
            bivariate_cor(obs_rmm1[mask], obs_rmm2[mask],
                          pred_rmm1[mask, l], pred_rmm2[mask, l])
            for l in range(N_LEADS)
        ])
        cor_yearly[int(yr)] = cor_yr
        print(f'  {yr} (n={n}): ' + ', '.join(f'{c:.3f}' for c in cor_yr))

    # 5. Confidence statistics
    cor_matrix = np.array([cor_yearly[yr] for yr in sorted(cor_yearly)])  # (5, 6)
    cor_mean = cor_matrix.mean(axis=0)
    cor_p10  = np.percentile(cor_matrix, 10, axis=0)
    cor_p90  = np.percentile(cor_matrix, 90, axis=0)

    print('\nConfidence statistics:')
    for l in range(N_LEADS):
        print(f'  Lead {l+1} (day {CANGLONG_DAYS[l]}): '
              f'mean={cor_mean[l]:.4f}, p10={cor_p10[l]:.4f}, p90={cor_p90[l]:.4f}')

    # 6. Save per-year CSV
    rows = []
    for yr in sorted(cor_yearly):
        for l in range(N_LEADS):
            rows.append({'year': yr, 'lead_week': l + 1,
                         'shifted_day': CANGLONG_DAYS[l],
                         'COR': round(cor_yearly[yr][l], 6)})
    pd.DataFrame(rows).to_csv(CI_CSV, index=False)
    print(f'\nSaved per-year COR: {CI_CSV}')

    # 7. Update mjo.csv with shift-corrected CI
    df = pd.read_csv(MJO_CSV)

    # Extracted Canglong values at shifted days
    extracted = {}
    for l, day in enumerate(CANGLONG_DAYS):
        row = df[df['day'] == day]
        if not row.empty and not np.isnan(row['Canglong_V35_COR'].iloc[0]):
            extracted[day] = float(row['Canglong_V35_COR'].iloc[0])

    df['Canglong_COR_p10'] = np.nan
    df['Canglong_COR_p90'] = np.nan

    for l, day in enumerate(CANGLONG_DAYS):
        if day not in extracted:
            continue
        # Center CI on the extracted Canglong value
        shift = extracted[day] - cor_mean[l]
        adj_p10 = round(cor_p10[l] + shift, 6)
        adj_p90 = round(cor_p90[l] + shift, 6)
        mask = df['day'] == day
        df.loc[mask, 'Canglong_COR_p10'] = adj_p10
        df.loc[mask, 'Canglong_COR_p90'] = adj_p90

    df.to_csv(MJO_CSV, index=False)
    print(f'Updated {MJO_CSV} with Canglong CI columns')
    print(df[['day', 'Canglong_V35_COR', 'Canglong_COR_p10', 'Canglong_COR_p90']].dropna().to_string(index=False))

    # 8. Re-render plot
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

    # FuXi CI
    ci_fx = df[['day', 'FuXi_COR_p10', 'FuXi_COR_p90']].dropna()
    ax.fill_between(ci_fx['day'], ci_fx['FuXi_COR_p10'], ci_fx['FuXi_COR_p90'],
                    color=c_fuxi, alpha=0.18, label='FuXi-S2S (p10–p90, 2017–2021)')

    # Canglong main line
    cl = df[['day', 'Canglong_V35_COR']].dropna()
    ax.plot(cl['day'], cl['Canglong_V35_COR'],
            color=c_canglong, lw=1.8, ls='--', marker='o',
            markersize=5, markeredgewidth=1.2, markerfacecolor='white',
            markeredgecolor=c_canglong, label='Canglong V3.5 (+2 week shift)', zorder=6)

    # Canglong CI
    ci_cl = df[['day', 'Canglong_COR_p10', 'Canglong_COR_p90']].dropna()
    ax.fill_between(ci_cl['day'], ci_cl['Canglong_COR_p10'], ci_cl['Canglong_COR_p90'],
                    color=c_canglong, alpha=0.18,
                    label='Canglong V3.5 (p10–p90, 2017–2021)', zorder=5)

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
