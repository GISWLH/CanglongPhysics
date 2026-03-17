"""
Compute MJO prediction skill (bivariate COR) for CAS-Canglong V3.5.

Reads from target-week-centric eval NC file (no GPU needed).

Steps:
1. Compute combined EOF of tropical-average OLR, U850, U200 from ERA5 2002-2016
2. Extract tropical-average MJO variables from eval NC (obs + pred per lead)
3. Project anomalies onto EOF -> RMM1, RMM2
4. Compute bivariate COR for leads 1-6 weeks
5. Plot COR vs lead time

Usage:
    cd /home/lhwang/Desktop/CanglongPhysics
    /home/lhwang/anaconda3/envs/torch/bin/python analysis/MJO/compute_mjo_skill.py

Output:
    analysis/MJO/mjo_cor_v35.png    - COR vs lead time plot
    analysis/MJO/mjo_results_v35.csv - COR values
    analysis/MJO/mjo_cache_v35.npz  - cached RMM indices (skip EOF on re-run)
"""

import numpy as np
import os
import sys
import json
import numcodecs
import netCDF4 as nc4
from scipy.linalg import svd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, ROOT)

# ─── Paths ────────────────────────────────────────────────────────
STORE_PATH = '/data/lhwang/ERA5_raw/weekly_data_zarr/ERA5_1982_2023_weekly.zarr'
WOY_PATH   = os.path.join(ROOT, 'Infer/eval/woy_map.npy')
EVAL_NC    = os.path.join(ROOT, 'Infer/eval/model_v3.nc')
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(OUT_DIR, 'mjo_cache_v35.npz')

# ─── Constants ────────────────────────────────────────────────────
H, W = 721, 1440
N_LEADS = 6
LATS = np.linspace(90, -90, H)

# Tropical band 15N-15S
LAT_N_IDX = int((90 - 15) / 0.25)   # 300
LAT_S_IDX = int((90 + 15) / 0.25)   # 420
TROP_LATS = LATS[LAT_N_IDX:LAT_S_IDX + 1]  # 121 points
TROP_COS  = np.cos(np.deg2rad(TROP_LATS))   # cos weights

# EOF reference period
EOF_START_YEAR = 2002
EOF_END_YEAR   = 2016

# Zarr variable indices (for EOF reference period reading)
OLR_SURF_IDX  = 1     # surface: avg_tnlwrf (OLR)
U_UPPER_IDX   = 3     # upper_air: u component
LEVEL_200_IDX = 0     # 200 hPa
LEVEL_850_IDX = 4     # 850 hPa


# ─── Zarr reader (for reference period EOF only) ──────────────────
def _load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def _build_blosc(codecs):
    for codec in codecs:
        if codec.get('name') == 'blosc':
            cfg = codec.get('configuration', {})
            shuffle = cfg.get('shuffle', 1)
            if shuffle == 'shuffle':    shuffle = 1
            elif shuffle == 'bitshuffle': shuffle = 2
            elif shuffle == 'noshuffle':  shuffle = 0
            return numcodecs.Blosc(
                cname=cfg.get('cname', 'lz4'),
                clevel=cfg.get('clevel', 5),
                shuffle=shuffle,
                blocksize=cfg.get('blocksize', 0),
            )
    return None

class ZarrArray:
    def __init__(self, store_path, name):
        meta = _load_json(os.path.join(store_path, name, 'zarr.json'))
        self.shape = tuple(meta['shape'])
        self.chunk_shape = tuple(meta['chunk_grid']['configuration']['chunk_shape'])
        self.dtype = np.dtype(meta['data_type'])
        endian = 'little'
        for c in meta.get('codecs', []):
            if c.get('name') == 'bytes':
                endian = c.get('configuration', {}).get('endian', 'little')
        self.dtype = self.dtype.newbyteorder('<' if endian == 'little' else '>')
        self.compressor = _build_blosc(meta.get('codecs', []))
        self.array_dir = os.path.join(store_path, name)
        self.chunk_tail = ['0'] * (len(self.shape) - 1)

    def read_time(self, t_idx):
        chunk_path = os.path.join(self.array_dir, 'c', str(t_idx), *self.chunk_tail)
        with open(chunk_path, 'rb') as f:
            raw = f.read()
        if self.compressor:
            raw = self.compressor.decode(raw)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.chunk_shape)[0]

def read_time_array(store_path):
    meta = _load_json(os.path.join(store_path, 'time', 'zarr.json'))
    dtype = np.dtype(meta['data_type'])
    endian = 'little'
    for c in meta.get('codecs', []):
        if c.get('name') == 'bytes':
            endian = c.get('configuration', {}).get('endian', 'little')
    dtype = dtype.newbyteorder('<' if endian == 'little' else '>')
    compressor = _build_blosc(meta.get('codecs', []))
    with open(os.path.join(store_path, 'time', 'c', '0'), 'rb') as f:
        raw = f.read()
    if compressor:
        raw = compressor.decode(raw)
    return np.frombuffer(raw, dtype=dtype)


# ─── Tropical averaging ──────────────────────────────────────────
def tropical_avg(field_2d):
    """Cos-weighted meridional average over 15N-15S.
    Input: (721, 1440) -> Output: (1440,)
    """
    trop = field_2d[LAT_N_IDX:LAT_S_IDX + 1, :]  # (121, 1440)
    weights = TROP_COS[:, None]  # (121, 1)
    return (trop * weights).sum(axis=0) / weights.sum()


# ─── Part 1: EOF computation from reference period ───────────────
def compute_eof_and_clim(store_path, woy_map):
    """Compute RMM EOF patterns from ERA5 reference period (2002-2016).

    Steps:
    1. Read tropical-avg OLR, U850, U200 for each week
    2. Compute seasonal cycle (52-week mean per longitude)
    3. Remove seasonal cycle -> anomaly
    4. Normalize each field by its std (scalar)
    5. Concatenate -> (N, 3*1440) matrix
    6. SVD -> EOF1, EOF2

    Returns:
        eof1, eof2: (4320,) EOF patterns
        seasonal_cycle: dict with olr/u850/u200 each (52, 1440)
        field_std: dict with olr/u850/u200 each scalar
    """
    print('Computing EOF from ERA5 reference period (2002-2016)...')

    # Time indexing
    time_days = read_time_array(store_path)
    base = np.datetime64('1940-01-01')
    dates = base + time_days.astype('timedelta64[D]')
    years = dates.astype('datetime64[Y]').astype(int) + 1970

    surface_arr = ZarrArray(store_path, 'surface')
    upper_arr   = ZarrArray(store_path, 'upper_air')

    # Collect tropical profiles for reference period
    olr_list, u850_list, u200_list = [], [], []
    woy_list = []

    for year in range(EOF_START_YEAR, EOF_END_YEAR + 1):
        idx_year = np.where(years == year)[0]
        for gi in idx_year:
            gi = int(gi)
            s = surface_arr.read_time(gi)
            u = upper_arr.read_time(gi)
            olr  = tropical_avg(s[OLR_SURF_IDX])
            u850 = tropical_avg(u[U_UPPER_IDX, LEVEL_850_IDX])
            u200 = tropical_avg(u[U_UPPER_IDX, LEVEL_200_IDX])
            olr_list.append(olr)
            u850_list.append(u850)
            u200_list.append(u200)
            woy_list.append(int(woy_map[gi]))

    olr_all  = np.array(olr_list, dtype=np.float64)
    u850_all = np.array(u850_list, dtype=np.float64)
    u200_all = np.array(u200_list, dtype=np.float64)
    woy_arr  = np.array(woy_list)
    N_ref = len(woy_arr)
    print(f'  Reference samples: {N_ref} weeks ({EOF_START_YEAR}-{EOF_END_YEAR})')

    # Seasonal cycle: mean per week-of-year per longitude
    seasonal = {
        'olr':  np.zeros((52, W), dtype=np.float64),
        'u850': np.zeros((52, W), dtype=np.float64),
        'u200': np.zeros((52, W), dtype=np.float64),
    }
    count = np.zeros(52, dtype=np.float64)
    for i in range(N_ref):
        w = woy_arr[i]
        seasonal['olr'][w]  += olr_all[i]
        seasonal['u850'][w] += u850_all[i]
        seasonal['u200'][w] += u200_all[i]
        count[w] += 1
    for w in range(52):
        if count[w] > 0:
            seasonal['olr'][w]  /= count[w]
            seasonal['u850'][w] /= count[w]
            seasonal['u200'][w] /= count[w]

    # Remove seasonal cycle -> anomalies
    olr_anom  = np.zeros_like(olr_all)
    u850_anom = np.zeros_like(u850_all)
    u200_anom = np.zeros_like(u200_all)
    for i in range(N_ref):
        w = woy_arr[i]
        olr_anom[i]  = olr_all[i]  - seasonal['olr'][w]
        u850_anom[i] = u850_all[i] - seasonal['u850'][w]
        u200_anom[i] = u200_all[i] - seasonal['u200'][w]

    # Normalize each field by its std
    field_std = {
        'olr':  float(np.std(olr_anom)),
        'u850': float(np.std(u850_anom)),
        'u200': float(np.std(u200_anom)),
    }
    print(f'  Field std: OLR={field_std["olr"]:.4f}, U850={field_std["u850"]:.4f}, U200={field_std["u200"]:.4f}')

    olr_norm  = olr_anom  / field_std['olr']
    u850_norm = u850_anom / field_std['u850']
    u200_norm = u200_anom / field_std['u200']

    # Combined matrix: (N_ref, 3*1440)
    X = np.concatenate([olr_norm, u850_norm, u200_norm], axis=1)
    print(f'  Combined matrix shape: {X.shape}')

    # SVD -> EOF
    U, S, Vt = svd(X, full_matrices=False)
    eof1 = Vt[0].astype(np.float64)
    eof2 = Vt[1].astype(np.float64)

    var_explained = S**2 / np.sum(S**2)
    print(f'  EOF1 variance: {var_explained[0]*100:.1f}%, EOF2: {var_explained[1]*100:.1f}%')

    eof1_olr  = eof1[:W]
    eof1_u850 = eof1[W:2*W]
    eof1_u200 = eof1[2*W:]
    print(f'  EOF1 norms: OLR={np.linalg.norm(eof1_olr):.3f}, '
          f'U850={np.linalg.norm(eof1_u850):.3f}, U200={np.linalg.norm(eof1_u200):.3f}')

    seasonal_f32 = {k: v.astype(np.float32) for k, v in seasonal.items()}
    return eof1, eof2, seasonal_f32, field_std


# ─── Part 2: Extract MJO profiles from eval NC ──────────────────
def extract_mjo_from_eval_nc(eval_nc_path):
    """Read OLR, U850, U200 tropical averages from target-week-centric eval NC.

    Returns:
        obs_profiles:  dict with olr/u850/u200 each (T, 1440)
        pred_profiles: dict with olr/u850/u200 each (T, N_LEADS, 1440)
        target_woys:   (T,) week-of-year for each target week
    """
    print(f'\nExtracting MJO profiles from {eval_nc_path} (no GPU needed)')
    ds = nc4.Dataset(eval_nc_path, 'r')
    T = ds.dimensions['time'].size
    print(f'  Target weeks: {T}')

    target_woys = ds.variables['woy'][:].astype(np.int32)  # (T,)

    obs_profiles  = {v: np.zeros((T, W), dtype=np.float32) for v in ['olr', 'u850', 'u200']}
    pred_profiles = {v: np.zeros((T, N_LEADS, W), dtype=np.float32) for v in ['olr', 'u850', 'u200']}

    # Map NC variable names to MJO variable names
    nc_var_map = {'olr': 'olr', 'u850': 'u850', 'u200': 'u200'}

    for mjo_var, nc_var in nc_var_map.items():
        print(f'  Reading obs_{nc_var}...')
        for i in range(T):
            field_2d = ds.variables[f'obs_{nc_var}'][i, :, :]  # (721, 1440)
            obs_profiles[mjo_var][i] = tropical_avg(field_2d)

        for lead in range(N_LEADS):
            lead_num = lead + 1
            print(f'  Reading pred_{nc_var}_lead{lead_num}...')
            for i in range(T):
                field_2d = ds.variables[f'pred_{nc_var}_lead{lead_num}'][i, :, :]
                pred_profiles[mjo_var][i, lead] = tropical_avg(field_2d)

    ds.close()
    print(f'  Extracted {T} target weeks x {N_LEADS} leads')
    return obs_profiles, pred_profiles, target_woys


# ─── Part 3: RMM computation ─────────────────────────────────────
def compute_rmm_obs(obs_profiles, target_woys, seasonal, field_std, eof1, eof2):
    """Compute RMM1, RMM2 for obs (T, 1440) -> (T,).

    Returns: rmm1 (T,), rmm2 (T,)
    """
    T = obs_profiles['olr'].shape[0]
    rmm1 = np.zeros(T, dtype=np.float64)
    rmm2 = np.zeros(T, dtype=np.float64)

    for i in range(T):
        woy = int(target_woys[i])
        olr_anom  = obs_profiles['olr'][i].astype(np.float64)  - seasonal['olr'][woy]
        u850_anom = obs_profiles['u850'][i].astype(np.float64) - seasonal['u850'][woy]
        u200_anom = obs_profiles['u200'][i].astype(np.float64) - seasonal['u200'][woy]

        olr_norm  = olr_anom  / field_std['olr']
        u850_norm = u850_anom / field_std['u850']
        u200_norm = u200_anom / field_std['u200']

        combined = np.concatenate([olr_norm, u850_norm, u200_norm])
        rmm1[i] = combined @ eof1
        rmm2[i] = combined @ eof2

    return rmm1, rmm2


def compute_rmm_pred(pred_profiles, target_woys, seasonal, field_std, eof1, eof2):
    """Compute RMM1, RMM2 for pred (T, N_LEADS, 1440) -> (T, N_LEADS).

    Returns: rmm1 (T, N_LEADS), rmm2 (T, N_LEADS)
    """
    T = pred_profiles['olr'].shape[0]
    L = pred_profiles['olr'].shape[1]
    rmm1 = np.zeros((T, L), dtype=np.float64)
    rmm2 = np.zeros((T, L), dtype=np.float64)

    for i in range(T):
        woy = int(target_woys[i])
        for l in range(L):
            olr_anom  = pred_profiles['olr'][i, l].astype(np.float64)  - seasonal['olr'][woy]
            u850_anom = pred_profiles['u850'][i, l].astype(np.float64) - seasonal['u850'][woy]
            u200_anom = pred_profiles['u200'][i, l].astype(np.float64) - seasonal['u200'][woy]

            olr_norm  = olr_anom  / field_std['olr']
            u850_norm = u850_anom / field_std['u850']
            u200_norm = u200_anom / field_std['u200']

            combined = np.concatenate([olr_norm, u850_norm, u200_norm])
            rmm1[i, l] = combined @ eof1
            rmm2[i, l] = combined @ eof2

    return rmm1, rmm2


# ─── Part 4: Bivariate COR ───────────────────────────────────────
def bivariate_cor(obs_rmm1, obs_rmm2, pred_rmm1, pred_rmm2):
    """Bivariate correlation coefficient for MJO at each lead.

    COR(tau) = sum[a1*b1 + a2*b2] / sqrt(sum[a1^2+a2^2] * sum[b1^2+b2^2])

    obs_rmm1, obs_rmm2: (T,)
    pred_rmm1, pred_rmm2: (T, 6)
    Returns: cor (6,)
    """
    L = pred_rmm1.shape[1]
    cor = np.zeros(L)
    for l in range(L):
        a1 = obs_rmm1
        a2 = obs_rmm2
        b1 = pred_rmm1[:, l]
        b2 = pred_rmm2[:, l]

        numerator   = np.sum(a1 * b1 + a2 * b2)
        denom_obs   = np.sum(a1**2 + a2**2)
        denom_pred  = np.sum(b1**2 + b2**2)
        denominator = np.sqrt(denom_obs * denom_pred)

        cor[l] = numerator / denominator if denominator > 1e-12 else np.nan
    return cor


# ─── Part 5: Plotting ────────────────────────────────────────────
def plot_cor(cor_values, out_path):
    """Plot COR vs lead time (Nature style)."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = "/usr/share/fonts/arial/ARIAL.TTF"
    font_manager.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = 'Arial'
    mpl.rcParams['svg.fonttype'] = 'none'

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
        'figure.figsize': (5, 3.5),
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
        'savefig.transparent': False,
    })

    leads = np.arange(1, N_LEADS + 1)

    fig, ax = plt.subplots()

    # COR line
    ax.plot(leads, cor_values, 'o-', color='#2166AC', markersize=5,
            label='CAS-Canglong V3.5', zorder=3)

    # Skill threshold
    ax.axhline(y=0.5, color='#B2182B', linestyle='--', linewidth=1.0,
               label='COR = 0.5 (skill threshold)', zorder=2)

    # Find where COR crosses 0.5
    above = cor_values >= 0.5
    if np.any(above) and not np.all(above):
        for i in range(len(cor_values) - 1):
            if cor_values[i] >= 0.5 and cor_values[i+1] < 0.5:
                cross = leads[i] + (0.5 - cor_values[i]) / (cor_values[i+1] - cor_values[i])
                ax.axvline(x=cross, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
                ax.text(cross + 0.1, 0.52, f'{cross:.1f} weeks',
                        fontsize=8, color='gray', va='bottom')
                break

    ax.set_xlabel('Lead Time (weeks)')
    ax.set_ylabel('Bivariate Correlation (COR)')
    ax.set_title('MJO Prediction Skill (2017-2021)')
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(leads)
    ax.legend(loc='upper right', frameon=True, edgecolor='#cccccc')
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.savefig(out_path, dpi=600)
    fig.savefig(out_path.replace('.png', '.svg'))
    plt.close(fig)
    print(f'  Plot saved: {out_path}')


# ─── Main ─────────────────────────────────────────────────────────
def main():
    woy_map = np.load(WOY_PATH)

    # Check cache
    if os.path.exists(CACHE_PATH):
        print(f'Loading cached MJO data from {CACHE_PATH}')
        cache = np.load(CACHE_PATH, allow_pickle=True)
        obs_rmm1    = cache['obs_rmm1']
        obs_rmm2    = cache['obs_rmm2']
        pred_rmm1   = cache['pred_rmm1']
        pred_rmm2   = cache['pred_rmm2']
    else:
        # Step 1: EOF from reference period (needs Zarr)
        eof1, eof2, seasonal, field_std = compute_eof_and_clim(STORE_PATH, woy_map)

        # Step 2: Extract MJO profiles from eval NC (no GPU)
        obs_profiles, pred_profiles, target_woys = extract_mjo_from_eval_nc(EVAL_NC)

        # Step 3: RMM indices
        print('\nComputing RMM indices...')
        obs_rmm1, obs_rmm2 = compute_rmm_obs(
            obs_profiles, target_woys, seasonal, field_std, eof1, eof2)
        pred_rmm1, pred_rmm2 = compute_rmm_pred(
            pred_profiles, target_woys, seasonal, field_std, eof1, eof2)

        # Cache
        np.savez(CACHE_PATH,
                 obs_rmm1=obs_rmm1, obs_rmm2=obs_rmm2,
                 pred_rmm1=pred_rmm1, pred_rmm2=pred_rmm2,
                 eof1=eof1, eof2=eof2,
                 seasonal_olr=seasonal['olr'],
                 seasonal_u850=seasonal['u850'],
                 seasonal_u200=seasonal['u200'],
                 field_std_olr=field_std['olr'],
                 field_std_u850=field_std['u850'],
                 field_std_u200=field_std['u200'])
        print(f'  Cached to {CACHE_PATH}')

    # Step 4: Bivariate COR
    print('\nBivariate Correlation Coefficient (COR):')
    cor = bivariate_cor(obs_rmm1, obs_rmm2, pred_rmm1, pred_rmm2)
    for l in range(N_LEADS):
        skill = 'skillful' if cor[l] >= 0.5 else ''
        print(f'  Lead {l+1} week: COR = {cor[l]:.4f}  {skill}')

    # Save CSV
    csv_path = os.path.join(OUT_DIR, 'mjo_results_v35.csv')
    with open(csv_path, 'w') as f:
        f.write('lead_week,COR,skillful\n')
        for l in range(N_LEADS):
            f.write(f'{l+1},{cor[l]:.6f},{"yes" if cor[l] >= 0.5 else "no"}\n')
    print(f'  Results saved: {csv_path}')

    # Step 5: Plot
    plot_path = os.path.join(OUT_DIR, 'mjo_cor_v35.png')
    plot_cor(cor, plot_path)

    # Summary
    print(f'\n=== MJO Prediction Skill Summary ===')
    above_05 = np.sum(cor >= 0.5)
    if above_05 > 0:
        for i in range(len(cor) - 1):
            if cor[i] >= 0.5 and cor[i+1] < 0.5:
                cross = (i + 1) + (0.5 - cor[i]) / (cor[i+1] - cor[i])
                print(f'  Skillful prediction limit: ~{cross:.1f} weeks (COR > 0.5)')
                break
        else:
            if cor[-1] >= 0.5:
                print(f'  Skillful prediction: all 6 weeks (COR > 0.5)')
    else:
        print(f'  No skillful prediction at any lead time')

    print(f'  COR values: {[f"{c:.4f}" for c in cor]}')
    print('Done.')


if __name__ == '__main__':
    main()
