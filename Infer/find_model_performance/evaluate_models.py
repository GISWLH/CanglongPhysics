"""
Systematic evaluation of V3 model checkpoints on 2023 full-year test set.
Evaluates S2S key variables: precipitation, OLR, t2m, d2m, u200, u850.
Outputs CSV + Nature-style comparison figure.

Usage:
    conda activate torch
    CUDA_VISIBLE_DEVICES=0 python evaluate_models.py
"""

import torch
import numpy as np
import pandas as pd
import sys, os, json, numcodecs

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from canglong import CanglongV2_5
from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays

# ── Variable indices (CLAUDE.md order) ────────────────────────────────
SURFACE_VARS = ['avg_tnswrf','avg_tnlwrf','tciw','tcc','lsrr','crr','blh',
                'u10','v10','d2m','t2m','avg_iews','avg_inss','slhf','sshf',
                'avg_snswrf','avg_snlwrf','ssr','str','sp','msl','siconc',
                'sst','ro','stl','swvl']
UPPER_VARS = ['o3','z','t','u','v','w','q','cc','ciwc','clwc']
LEVELS = [200, 300, 500, 700, 850]

IDX = {v: i for i, v in enumerate(SURFACE_VARS)}
UIDX = {v: i for i, v in enumerate(UPPER_VARS)}
LIDX = {l: i for i, l in enumerate(LEVELS)}

# ── Zarr reader (from infer_v3.ipynb) ─────────────────────────────────
def _load_json(path):
    with open(path) as f:
        return json.load(f)

def _build_blosc(codecs):
    for c in codecs:
        if c.get("name") == "blosc":
            cfg = c.get("configuration", {})
            shuffle = cfg.get("shuffle", 1)
            if shuffle == "shuffle": shuffle = 1
            elif shuffle == "bitshuffle": shuffle = 2
            elif shuffle == "noshuffle": shuffle = 0
            return numcodecs.Blosc(cname=cfg.get("cname","lz4"),
                                   clevel=cfg.get("clevel",5),
                                   shuffle=shuffle,
                                   blocksize=cfg.get("blocksize",0))
    return None

class ZarrArray:
    def __init__(self, store_path, name):
        meta = _load_json(os.path.join(store_path, name, "zarr.json"))
        self.shape = tuple(meta["shape"])
        self.chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
        self.dtype = np.dtype(meta["data_type"])
        for c in meta.get("codecs", []):
            if c.get("name") == "bytes":
                endian = c.get("configuration", {}).get("endian", "little")
                self.dtype = self.dtype.newbyteorder("<" if endian == "little" else ">")
        self.compressor = _build_blosc(meta.get("codecs", []))
        self.array_dir = os.path.join(store_path, name)
        self.chunk_tail = ["0"] * (len(self.shape) - 1)

    def read_time(self, t):
        path = os.path.join(self.array_dir, "c", str(t), *self.chunk_tail)
        with open(path, "rb") as f:
            raw = f.read()
        if self.compressor:
            raw = self.compressor.decode(raw)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.chunk_shape)[0]

def read_time_array(store_path):
    meta = _load_json(os.path.join(store_path, "time", "zarr.json"))
    dtype = np.dtype(meta["data_type"])
    for c in meta.get("codecs", []):
        if c.get("name") == "bytes":
            endian = c.get("configuration", {}).get("endian", "little")
            dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    compressor = _build_blosc(meta.get("codecs", []))
    with open(os.path.join(store_path, "time", "c", "0"), "rb") as f:
        raw = f.read()
    if compressor:
        raw = compressor.decode(raw)
    return np.frombuffer(raw, dtype=dtype)

# ── Metrics ───────────────────────────────────────────────────────────
def pcc(pred, target):
    """Spatial Pearson correlation coefficient."""
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    pa = p - p.mean(1, keepdim=True)
    ta = t - t.mean(1, keepdim=True)
    num = (pa * ta).sum(1)
    den = torch.clamp(torch.sqrt(pa.pow(2).sum(1) * ta.pow(2).sum(1)), min=1e-12)
    return num / den

def rmse(pred, target):
    d = (pred - target).reshape(pred.shape[0], -1)
    return torch.sqrt(d.pow(2).mean(1))

def acc(pred, target, clim):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    c = clim.reshape(1, -1).to(p.device)
    pa = p - c
    ta = t - c
    num = (pa * ta).sum(1)
    den = torch.clamp(torch.sqrt(pa.pow(2).sum(1) * ta.pow(2).sum(1)), min=1e-12)
    return num / den

# ── Evaluate one model ────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model, surface_arr, upper_arr, window_start, n_weeks,
                   surface_mean, surface_std, upper_mean, upper_std, device):
    """
    Run model on full year (n_weeks - 2 valid samples) and return per-variable metrics.
    """
    model.eval()

    # climatology for ACC (mean over all training data)
    surf_mean_2d = surface_mean[0, :, 0, :, :]  # (26, 721, 1440)
    precip_clim = surf_mean_2d[IDX['lsrr']] + surf_mean_2d[IDX['crr']]
    t2m_clim = surf_mean_2d[IDX['t2m']]
    olr_clim = surf_mean_2d[IDX['avg_tnlwrf']]

    records = {k: [] for k in [
        'precip_pcc','precip_acc','precip_rmse',
        'olr_pcc','olr_rmse',
        't2m_pcc','t2m_acc','t2m_rmse',
        'd2m_pcc','d2m_rmse',
        'u200_pcc','u200_rmse',
        'u850_pcc','u850_rmse',
        'surface_rmse','upper_rmse'
    ]}

    n_samples = n_weeks - 2
    for i in range(n_samples):
        t_abs = window_start + i
        s0 = surface_arr.read_time(t_abs)
        s1 = surface_arr.read_time(t_abs + 1)
        s2 = surface_arr.read_time(t_abs + 2)
        u0 = upper_arr.read_time(t_abs)
        u1 = upper_arr.read_time(t_abs + 1)
        u2 = upper_arr.read_time(t_abs + 2)

        inp_s = torch.from_numpy(np.stack([s0, s1], axis=1)[None]).float().to(device)
        inp_u = torch.from_numpy(np.stack([u0, u1], axis=2)[None]).float().to(device)
        tgt_s = torch.from_numpy(s2[None, :, None]).float().to(device)
        tgt_u = torch.from_numpy(u2[None, :, :, None]).float().to(device)

        # Normalize inputs
        inp_s = (inp_s - surface_mean) / surface_std
        inp_u = (inp_u - upper_mean) / upper_std

        out_s, out_u = model(inp_s, inp_u)

        # Normalized targets
        tgt_s_n = (tgt_s - surface_mean) / surface_std
        tgt_u_n = (tgt_u - upper_mean) / upper_std

        # Physical-space outputs
        out_s_phys = out_s * surface_std + surface_mean
        out_u_phys = out_u * upper_std + upper_mean

        # ── Precipitation ──
        pred_p = out_s_phys[:, IDX['lsrr'], 0] + out_s_phys[:, IDX['crr'], 0]
        true_p = tgt_s[:, IDX['lsrr'], 0] + tgt_s[:, IDX['crr'], 0]
        pred_p_n = out_s[:, IDX['lsrr'], 0] + out_s[:, IDX['crr'], 0]
        true_p_n = tgt_s_n[:, IDX['lsrr'], 0] + tgt_s_n[:, IDX['crr'], 0]
        records['precip_pcc'].append(pcc(pred_p, true_p).item())
        records['precip_acc'].append(acc(pred_p, true_p, precip_clim).item())
        records['precip_rmse'].append(rmse(pred_p_n, true_p_n).item())

        # ── OLR ──
        records['olr_pcc'].append(pcc(out_s_phys[:, IDX['avg_tnlwrf'], 0],
                                       tgt_s[:, IDX['avg_tnlwrf'], 0]).item())
        records['olr_rmse'].append(rmse(out_s[:, IDX['avg_tnlwrf'], 0],
                                         tgt_s_n[:, IDX['avg_tnlwrf'], 0]).item())

        # ── t2m ──
        records['t2m_pcc'].append(pcc(out_s_phys[:, IDX['t2m'], 0],
                                       tgt_s[:, IDX['t2m'], 0]).item())
        records['t2m_acc'].append(acc(out_s_phys[:, IDX['t2m'], 0],
                                       tgt_s[:, IDX['t2m'], 0], t2m_clim).item())
        records['t2m_rmse'].append(rmse(out_s[:, IDX['t2m'], 0],
                                         tgt_s_n[:, IDX['t2m'], 0]).item())

        # ── d2m ──
        records['d2m_pcc'].append(pcc(out_s_phys[:, IDX['d2m'], 0],
                                       tgt_s[:, IDX['d2m'], 0]).item())
        records['d2m_rmse'].append(rmse(out_s[:, IDX['d2m'], 0],
                                         tgt_s_n[:, IDX['d2m'], 0]).item())

        # ── u200 ──
        records['u200_pcc'].append(pcc(out_u_phys[:, UIDX['u'], LIDX[200], 0],
                                        tgt_u[:, UIDX['u'], LIDX[200], 0]).item())
        records['u200_rmse'].append(rmse(out_u[:, UIDX['u'], LIDX[200], 0],
                                          tgt_u_n[:, UIDX['u'], LIDX[200], 0]).item())

        # ── u850 ──
        records['u850_pcc'].append(pcc(out_u_phys[:, UIDX['u'], LIDX[850], 0],
                                        tgt_u[:, UIDX['u'], LIDX[850], 0]).item())
        records['u850_rmse'].append(rmse(out_u[:, UIDX['u'], LIDX[850], 0],
                                          tgt_u_n[:, UIDX['u'], LIDX[850], 0]).item())

        # ── Aggregate surface / upper ──
        records['surface_rmse'].append(rmse(out_s, tgt_s_n).item())
        records['upper_rmse'].append(rmse(out_u, tgt_u_n).item())

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{n_samples}] precip_pcc={records['precip_pcc'][-1]:.4f} "
                  f"t2m_pcc={records['t2m_pcc'][-1]:.4f}")

    return {k: float(np.mean(v)) for k, v in records.items()}

# ── Main ──────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda:0")
    print(f"Using device: {device}")

    # ── Data ──
    store_path = "/data/lhwang/ERA5_raw/weekly_data_zarr/ERA5_1982_2023_weekly.zarr"
    time_days = read_time_array(store_path)
    base = np.datetime64("1940-01-01")
    dates = base + time_days.astype("timedelta64[D]")
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    idx_2023 = np.where(years == 2023)[0]
    window_start = int(idx_2023[0])
    n_weeks = len(idx_2023)
    print(f"2023 test set: start={dates[window_start]}, weeks={n_weeks}, samples={n_weeks-2}")

    surface_arr = ZarrArray(store_path, "surface")
    upper_arr = ZarrArray(store_path, "upper_air")

    # ── Normalization ──
    json_path = os.path.join(os.path.dirname(__file__), '..', '..', 'code_v2',
                             'ERA5_1940_2023_mean_std_v2.json')
    sm, ss, um, us = load_normalization_arrays(json_path)
    surface_mean = torch.from_numpy(sm).float().to(device)
    surface_std  = torch.from_numpy(ss).float().to(device)
    upper_mean   = torch.from_numpy(um).float().to(device)
    upper_std    = torch.from_numpy(us).float().to(device)

    # ── Models to evaluate ──
    model_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'model')
    models = {
        'v3.5_init (ep50)':        'model_v3_5_epoch50.pth',
        'v3.5_best (ep82)':        'model_v3_5_best.pth',
        'v3.5_cont_best (ep60)':   'model_v3_5_continue_record_best.pth',
        'v3.5_ft2_best (ep10)':    'model_v3_5_continue_record_ft2_best.pth',
        'v3.5_ft2_ep20':           'model_v3_5_continue_record_ft2_epoch20.pth',
        'v3.5_mse_best (ep22)':    'model_v3_5_continue_mse_only_best.pth',
        'v3.5_mse_ep50':           'model_v3_5_continue_mse_only_epoch50.pth',
        'v3.5_mse_ep100':          'model_v3_5_continue_mse_only_epoch100.pth',
    }

    # ── Evaluate loop ──
    all_results = []
    for label, ckpt in models.items():
        path = os.path.join(model_dir, ckpt)
        if not os.path.exists(path):
            print(f"SKIP {label}: {ckpt} not found")
            continue
        print(f"\n{'='*60}")
        print(f"Evaluating: {label}  ({ckpt})")
        print(f"{'='*60}")

        model = CanglongV2_5()
        state = torch.load(path, map_location='cpu')
        model.load_state_dict(state, strict=False)
        model.to(device)

        metrics = evaluate_model(model, surface_arr, upper_arr, window_start, n_weeks,
                                 surface_mean, surface_std, upper_mean, upper_std, device)
        metrics['model'] = label
        all_results.append(metrics)

        # Free GPU memory
        del model, state
        torch.cuda.empty_cache()

        # Print summary
        print(f"  precip  PCC={metrics['precip_pcc']:.4f}  ACC={metrics['precip_acc']:.4f}  RMSE={metrics['precip_rmse']:.4f}")
        print(f"  OLR     PCC={metrics['olr_pcc']:.4f}  RMSE={metrics['olr_rmse']:.4f}")
        print(f"  t2m     PCC={metrics['t2m_pcc']:.4f}  ACC={metrics['t2m_acc']:.4f}  RMSE={metrics['t2m_rmse']:.4f}")
        print(f"  d2m     PCC={metrics['d2m_pcc']:.4f}  RMSE={metrics['d2m_rmse']:.4f}")
        print(f"  u200    PCC={metrics['u200_pcc']:.4f}  RMSE={metrics['u200_rmse']:.4f}")
        print(f"  u850    PCC={metrics['u850_pcc']:.4f}  RMSE={metrics['u850_rmse']:.4f}")
        print(f"  surface RMSE={metrics['surface_rmse']:.4f}  upper RMSE={metrics['upper_rmse']:.4f}")

    # ── Save CSV ──
    df = pd.DataFrame(all_results)
    cols = ['model'] + [c for c in df.columns if c != 'model']
    df = df[cols]
    csv_path = os.path.join(os.path.dirname(__file__), 'model_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.6f')
    print(f"\nResults saved to {csv_path}")

    # ── Plot ──
    plot_results(df)


def plot_results(df):
    """Generate Nature-style multi-panel comparison figure."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    font_path = "/usr/share/fonts/arial/ARIAL.TTF"
    try:
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=font_path).get_name()
    except FileNotFoundError:
        plt.rcParams['font.family'] = 'Arial'

    mpl.rcParams['svg.fonttype'] = 'none'
    plt.style.use('seaborn-v0_8-talk')
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 9,
        'axes.titlesize': 10, 'axes.labelsize': 9,
        'xtick.labelsize': 7, 'ytick.labelsize': 8,
        'legend.fontsize': 7, 'figure.dpi': 600,
        'axes.linewidth': 0.8,
        'axes.spines.left': True, 'axes.spines.bottom': True,
        'axes.spines.top': True, 'axes.spines.right': True,
        'axes.edgecolor': '#454545',
        'xtick.direction': 'in', 'ytick.direction': 'in',
        'xtick.major.size': 5, 'ytick.major.size': 5,
        'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'savefig.bbox': 'tight', 'savefig.transparent': False,
    })

    short_labels = [m.split('(')[0].strip() for m in df['model']]
    x = np.arange(len(short_labels))
    width = 0.6

    # 6 key variables, 2 rows: PCC (top) + RMSE (bottom)
    pcc_cols = ['precip_pcc', 'olr_pcc', 't2m_pcc', 'd2m_pcc', 'u200_pcc', 'u850_pcc']
    rmse_cols = ['precip_rmse', 'olr_rmse', 't2m_rmse', 'd2m_rmse', 'u200_rmse', 'u850_rmse']
    var_labels = ['Precip', 'OLR', 'T2m', 'D2m', 'U200', 'U850']

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # ── PCC panel ──
    ax = axes[0]
    for i, (col, color, vlbl) in enumerate(zip(pcc_cols, colors, var_labels)):
        offset = (i - len(pcc_cols)/2 + 0.5) * (width / len(pcc_cols))
        bars = ax.bar(x + offset, df[col], width / len(pcc_cols), label=vlbl,
                      color=color, edgecolor='white', linewidth=0.3)
    ax.set_ylabel('PCC (higher is better)')
    ax.set_title('Spatial Pearson Correlation Coefficient by Variable', fontweight='bold')
    ax.legend(ncol=6, loc='lower left', frameon=True, edgecolor='#cccccc')
    ax.set_ylim(bottom=max(0, df[pcc_cols].min().min() - 0.05))

    # ── RMSE panel ──
    ax = axes[1]
    for i, (col, color, vlbl) in enumerate(zip(rmse_cols, colors, var_labels)):
        offset = (i - len(rmse_cols)/2 + 0.5) * (width / len(rmse_cols))
        ax.bar(x + offset, df[col], width / len(rmse_cols), label=vlbl,
               color=color, edgecolor='white', linewidth=0.3)
    ax.set_ylabel('RMSE (normalized, lower is better)')
    ax.set_title('Normalized RMSE by Variable', fontweight='bold')
    ax.legend(ncol=6, loc='upper left', frameon=True, edgecolor='#cccccc')
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=25, ha='right')
    ax.set_xlabel('Model Checkpoint')

    plt.tight_layout()
    fig_path = os.path.join(os.path.dirname(__file__), 'model_comparison.png')
    plt.savefig(fig_path, dpi=600)
    plt.close()
    print(f"Figure saved to {fig_path}")

    # ── Composite score summary ──
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    composite = df[['precip_pcc','t2m_pcc','d2m_pcc','u200_pcc','u850_pcc','olr_pcc']].mean(axis=1)
    bars = ax2.barh(short_labels, composite, color='#4C72B0', edgecolor='white', height=0.6)
    ax2.set_xlabel('Mean PCC across 6 key variables (higher is better)')
    ax2.set_title('Composite S2S Skill Score', fontweight='bold')
    for bar, val in zip(bars, composite):
        ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.4f}', va='center', fontsize=8)
    ax2.set_xlim(right=composite.max() + 0.03)
    plt.tight_layout()
    fig2_path = os.path.join(os.path.dirname(__file__), 'composite_score.png')
    plt.savefig(fig2_path, dpi=600)
    plt.close()
    print(f"Composite figure saved to {fig2_path}")


if __name__ == '__main__':
    main()
