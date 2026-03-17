"""
Inference script for CAS-Canglong V0 (Lite) on the 2017-2021 test set.
V0 uses 16 surface variables, 7 upper-air variables, 4 pressure levels.

Usage:
    cd /home/lhwang/Desktop/CanglongPhysics/canglong
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    /home/lhwang/anaconda3/envs/torch/bin/python ../Infer/infer_v0_2017_2021.py

NOTE: Must run from canglong/ directory so model_v0.py can find
      ../constant_masks/input_tensor.pt at module import time.
"""

import torch
import numpy as np
import os
import sys
import json
import numcodecs
import csv
import __main__
from torch.utils.data import Dataset, DataLoader

# ── paths ──────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

# Monkey-patch embed/recovery to old versions (required for V0 pickle)
import canglong.embed_old as embed_old
sys.modules['canglong.embed'] = embed_old
import canglong.recovery_old as recovery_old
sys.modules['canglong.recovery'] = recovery_old

from canglong.model_v0 import (
    Canglong, Encoder, Decoder, BasicLayer,
    EarthSpecificBlock, EarthAttention3D, UpSample, DownSample, Mlp
)
from canglong.helper import (
    GroupNorm, Swish, ResidualBlock,
    UpSampleBlock, DownSampleBlock, NonLocalBlock
)

# Register all classes to __main__ for pickle unpickling
for cls in [Canglong, Encoder, Decoder, BasicLayer,
            EarthSpecificBlock, EarthAttention3D, UpSample, DownSample, Mlp,
            GroupNorm, Swish, ResidualBlock, UpSampleBlock, DownSampleBlock, NonLocalBlock]:
    setattr(__main__, cls.__name__, cls)

STORE_PATH = '/data/lhwang/ERA5_raw/weekly_data_zarr/ERA5_1982_2023_weekly.zarr'
MODEL_PATH = '/home/lhwang/Desktop/model/weather_model_epoch_500.pt'
OUT_DIR    = os.path.join(ROOT, 'Infer/results_v0_2017_2021')
os.makedirs(OUT_DIR, exist_ok=True)

# ── V0 variable mapping ───────────────────────────────────────────
# V0 surface: 16 vars → indices in Zarr's 26-var surface array
# V0 order: lsrr, crr, tciw, tcc, tsrc, u10, v10, d2m, t2m, slhf, sshf, sp, swvl, msl, siconc, sst
# Note: tsrc (top_net_solar_radiation_clear_sky) not in Zarr;
#       using avg_tnswrf (idx 0) as closest available substitute.
V0_SURFACE_ZARR_IDX = [4, 5, 2, 3, 0, 7, 8, 9, 10, 13, 14, 19, 25, 20, 21, 22]

V0_SURFACE_NAMES = [
    'lsrr', 'crr', 'tciw', 'tcc', 'tsrc(avg_tnswrf)',
    'u10', 'v10', 'd2m', 't2m', 'slhf', 'sshf',
    'sp', 'swvl', 'msl', 'siconc', 'sst'
]

# V0 upper air: 7 vars → indices in Zarr's 10-var upper array
# V0 order: z, w, u, v, cc, t, q
V0_UPPER_ZARR_IDX = [1, 5, 3, 4, 7, 2, 6]

# V0 pressure levels: [300, 500, 700, 850] → Zarr level indices [1, 2, 3, 4]
# (Zarr levels: [200, 300, 500, 700, 850])
V0_LEVEL_ZARR_IDX = [1, 2, 3, 4]

# V0 scalar normalization (from run.py ordered_var_stats)
V0_SURFACE_STATS = {
    'lsrr':  {'mean': 1.10E-05, 'std': 2.55E-05},
    'crr':   {'mean': 1.29E-05, 'std': 2.97E-05},
    'tciw':  {'mean': 0.022627383, 'std': 0.023428712},
    'tcc':   {'mean': 0.673692584, 'std': 0.235167906},
    'tsrc':  {'mean': 856148, 'std': 534222.125},
    'u10':   {'mean': -0.068418466, 'std': 4.427545547},
    'v10':   {'mean': 0.197138891, 'std': 3.09530735},
    'd2m':   {'mean': 274.2094421, 'std': 20.45770073},
    't2m':   {'mean': 278.7841187, 'std': 21.03286934},
    'slhf':  {'mean': -5410301.5, 'std': 5349063.5},
    'sshf':  {'mean': -971651.375, 'std': 2276764.75},
    'sp':    {'mean': 96651.14063, 'std': 9569.695313},
    'swvl':  {'mean': 0.34216917, 'std': 0.5484813},
    'msl':   {'mean': 100972.3438, 'std': 1191.102417},
    'siconc':{'mean': 0.785884917, 'std': 0.914535105},
    'sst':   {'mean': 189.7337189, 'std': 136.1803131},
}

V0_UPPER_STATS = {
    'z':  {'300': {'mean': 13763.50879, 'std': 1403.990112},
           '500': {'mean': 28954.94531, 'std': 2085.838867},
           '700': {'mean': 54156.85547, 'std': 3300.384277},
           '850': {'mean': 89503.79688, 'std': 5027.79541}},
    'w':  {'300': {'mean': 0.011849277, 'std': 0.126232564},
           '500': {'mean': 0.002759292, 'std': 0.097579598},
           '700': {'mean': 0.000348145, 'std': 0.072489716},
           '850': {'mean': 0.000108061, 'std': 0.049831692}},
    'u':  {'300': {'mean': 1.374536991, 'std': 6.700420856},
           '500': {'mean': 3.290786982, 'std': 7.666454315},
           '700': {'mean': 6.491596222, 'std': 9.875613213},
           '850': {'mean': 11.66026878, 'std': 14.00845909}},
    'v':  {'300': {'mean': 0.146550566, 'std': 3.75399971},
           '500': {'mean': 0.022800878, 'std': 4.179731846},
           '700': {'mean': -0.025720235, 'std': 5.324173927},
           '850': {'mean': -0.027837994, 'std': 7.523460865}},
    'cc': {'300': {'mean': 0.152513072, 'std': 0.15887706},
           '500': {'mean': 0.106524825, 'std': 0.144112185},
           '700': {'mean': 0.105878539, 'std': 0.112193666},
           '850': {'mean': 0.108120449, 'std': 0.108371623}},
    't':  {'300': {'mean': 274.8048401, 'std': 15.28209305},
           '500': {'mean': 267.6254578, 'std': 14.55300999},
           '700': {'mean': 253.1627655, 'std': 12.77071381},
           '850': {'mean': 229.0860138, 'std': 10.5536499}},
    'q':  {'300': {'mean': 0.004610791, 'std': 0.003879665},
           '500': {'mean': 0.002473272, 'std': 0.002312181},
           '700': {'mean': 0.000875093, 'std': 0.000944978},
           '850': {'mean': 0.000130984, 'std': 0.000145811}},
}

V0_UPPER_NAMES = ['z', 'w', 'u', 'v', 'cc', 't', 'q']
V0_SURFACE_SHORT = ['lsrr','crr','tciw','tcc','tsrc','u10','v10','d2m','t2m','slhf','sshf','sp','swvl','msl','siconc','sst']
V0_LEVELS = [300, 500, 700, 850]

# Build normalization arrays: surface (16, 1, 1, 1), upper (7, 4, 1, 1)
def build_v0_norm_arrays():
    s_mean = np.array([V0_SURFACE_STATS[v]['mean'] for v in V0_SURFACE_SHORT]).reshape(16, 1, 1, 1)
    s_std  = np.array([V0_SURFACE_STATS[v]['std']  for v in V0_SURFACE_SHORT]).reshape(16, 1, 1, 1)

    u_mean = np.zeros((7, 4, 1, 1))
    u_std  = np.zeros((7, 4, 1, 1))
    for vi, vn in enumerate(V0_UPPER_NAMES):
        for li, lv in enumerate(V0_LEVELS):
            u_mean[vi, li, 0, 0] = V0_UPPER_STATS[vn][str(lv)]['mean']
            u_std[vi, li, 0, 0]  = V0_UPPER_STATS[vn][str(lv)]['std']

    return s_mean, s_std, u_mean, u_std

# ── Evaluation variable indices in V0 space ────────────────────────
# V0 surface indices: lsrr=0, crr=1, d2m=7, t2m=8
V0_LSRR, V0_CRR, V0_D2M, V0_T2M = 0, 1, 7, 8
# V0 upper: u=2 (index in V0's 7 vars)
V0_U_IDX = 2
# V0 levels: [300, 500, 700, 850] → idx 0 for 300hPa (no 200hPa in V0)
# We cannot compute u200 for V0 (no 200hPa level)
# u850 is at level idx 3

# ── Zarr reader ────────────────────────────────────────────────────
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

# ── Dataset ────────────────────────────────────────────────────────
class V0ZarrDataset(Dataset):
    """Extract V0's 16/7 variable subset from the 26/10 Zarr store."""
    def __init__(self, surface_arr, upper_arr, global_indices):
        self.surface_arr = surface_arr
        self.upper_arr = upper_arr
        self.indices = global_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        # Read full 26-var surface and 10-var upper from Zarr
        s0_full = self.surface_arr.read_time(idx)      # (26, 721, 1440)
        s1_full = self.surface_arr.read_time(idx + 1)
        s2_full = self.surface_arr.read_time(idx + 2)

        u0_full = self.upper_arr.read_time(idx)         # (10, 5, 721, 1440)
        u1_full = self.upper_arr.read_time(idx + 1)
        u2_full = self.upper_arr.read_time(idx + 2)

        # Extract V0 subset
        s0 = s0_full[V0_SURFACE_ZARR_IDX]               # (16, 721, 1440)
        s1 = s1_full[V0_SURFACE_ZARR_IDX]
        s2 = s2_full[V0_SURFACE_ZARR_IDX]

        u0 = u0_full[np.ix_(V0_UPPER_ZARR_IDX, V0_LEVEL_ZARR_IDX)]  # (7, 4, 721, 1440)
        u1 = u1_full[np.ix_(V0_UPPER_ZARR_IDX, V0_LEVEL_ZARR_IDX)]
        u2 = u2_full[np.ix_(V0_UPPER_ZARR_IDX, V0_LEVEL_ZARR_IDX)]

        input_surface  = np.stack([s0, s1], axis=1)      # (16, 2, 721, 1440)
        input_upper    = np.stack([u0, u1], axis=2)       # (7, 4, 2, 721, 1440)
        target_surface = s2[:, None, :, :]                # (16, 1, 721, 1440)
        target_upper   = u2[:, :, None, :, :]             # (7, 4, 1, 721, 1440)

        return input_surface, input_upper, target_surface, target_upper, idx

# ── metrics ────────────────────────────────────────────────────────
def pcc(pred, target):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    pa = p - p.mean(1, keepdim=True)
    ta = t - t.mean(1, keepdim=True)
    num = (pa * ta).sum(1)
    den = torch.clamp(torch.sqrt(pa.pow(2).sum(1) * ta.pow(2).sum(1)), min=1e-12)
    return num / den

def acc(pred, target, clim):
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    c = clim.reshape(1, -1).to(p.device)
    pa = p - c
    ta = t - c
    num = (pa * ta).sum(1)
    den = torch.clamp(torch.sqrt(pa.pow(2).sum(1) * ta.pow(2).sum(1)), min=1e-12)
    return num / den

def rmse(pred, target):
    d = pred - target
    return torch.sqrt(d.pow(2).reshape(d.shape[0], -1).mean(1))

# ── main ───────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Time indexing
    time_days = read_time_array(STORE_PATH)
    base = np.datetime64('1940-01-01')
    dates = base + time_days.astype('timedelta64[D]')
    years = dates.astype('datetime64[Y]').astype(int) + 1970

    # Build sample indices for 2017-2021
    global_indices = []
    sample_years = []
    sample_week_nums = []
    for year in range(2017, 2022):
        idx_year = np.where(years == year)[0]
        for k in range(len(idx_year) - 2):
            gi = int(idx_year[k])
            global_indices.append(gi)
            sample_years.append(year)
            sample_week_nums.append(k + 1)
    print(f'Total samples: {len(global_indices)} (5 years x 50 weeks)')

    # Data readers
    surface_arr = ZarrArray(STORE_PATH, 'surface')
    upper_arr   = ZarrArray(STORE_PATH, 'upper_air')

    # V0 normalization (scalar per variable)
    s_mean_np, s_std_np, u_mean_np, u_std_np = build_v0_norm_arrays()
    # Surface: (16,1,1,1) → (1,16,1,1,1) to broadcast with (B,16,time,721,1440)
    s_mean = torch.from_numpy(s_mean_np).float().unsqueeze(0).to(device)  # (1,16,1,1,1)
    s_std  = torch.from_numpy(s_std_np).float().unsqueeze(0).to(device)
    # Upper: (7,4,1,1) → (1,7,4,1,1,1) to broadcast with (B,7,4,time,721,1440)
    u_mean = torch.from_numpy(u_mean_np).float().unsqueeze(0).unsqueeze(3).to(device)  # (1,7,4,1,1,1)
    u_std  = torch.from_numpy(u_std_np).float().unsqueeze(0).unsqueeze(3).to(device)

    print(f's_mean shape: {s_mean.shape}, u_mean shape: {u_mean.shape}')

    # Climatology for ACC (use V0 normalization means as long-term average)
    precip_clim = torch.tensor(
        V0_SURFACE_STATS['lsrr']['mean'] + V0_SURFACE_STATS['crr']['mean']
    ).float().to(device)
    t2m_clim = torch.tensor(V0_SURFACE_STATS['t2m']['mean']).float().to(device)
    d2m_clim = torch.tensor(V0_SURFACE_STATS['d2m']['mean']).float().to(device)
    # u850 climatology (V0 upper: u is idx 2, 850hPa is level idx 3)
    u850_clim = torch.tensor(V0_UPPER_STATS['u']['850']['mean']).float().to(device)

    # Load V0 model
    print(f'Loading model: {MODEL_PATH}')
    dp_model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = dp_model.module  # unwrap DataParallel
    model.input_constant = model.input_constant.to(device)
    model.to(device).eval()
    print('Model loaded (V0, unwrapped from DataParallel).')

    # DataLoader
    dataset = V0ZarrDataset(surface_arr, upper_arr, global_indices)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    # ── Inference loop ─────────────────────────────────────────────
    records = []
    with torch.no_grad():
        for i, (inp_s, inp_u, tgt_s, tgt_u, gi) in enumerate(loader):
            inp_s = inp_s.float().to(device)  # (1, 16, 2, 721, 1440)
            inp_u = inp_u.float().to(device)  # (1, 7, 4, 2, 721, 1440)
            tgt_s = tgt_s.float().to(device)  # (1, 16, 1, 721, 1440)
            tgt_u = tgt_u.float().to(device)  # (1, 7, 4, 1, 721, 1440)

            # Normalize input
            inp_s_n = (inp_s - s_mean) / s_std
            inp_u_n = (inp_u - u_mean) / u_std
            # Replace NaN with 0 (as in run.py)
            inp_s_n = torch.nan_to_num(inp_s_n, nan=0.0)
            inp_u_n = torch.nan_to_num(inp_u_n, nan=0.0)

            # Forward: V0 outputs (B,16,2,721,1440) and (B,7,4,2,721,1440)
            out_s_raw, out_u_raw = model(inp_s_n, inp_u_n)

            # Take first time step as prediction
            out_s_n = out_s_raw[:, :, 0:1, :, :]    # (1, 16, 1, 721, 1440)
            out_u_n = out_u_raw[:, :, :, 0:1, :, :]  # (1, 7, 4, 1, 721, 1440)

            # Denormalize output
            out_s = out_s_n * s_std + s_mean
            out_u = out_u_n * u_std + u_mean

            # Normalize target
            tgt_s_n = (tgt_s - s_mean) / s_std
            tgt_u_n = (tgt_u - u_mean) / u_std

            # ── per-variable metrics ───────────────────────────────
            m = {}

            # Precipitation
            pred_precip = out_s[:, V0_LSRR, 0, :, :] + out_s[:, V0_CRR, 0, :, :]
            true_precip = tgt_s[:, V0_LSRR, 0, :, :] + tgt_s[:, V0_CRR, 0, :, :]
            pred_precip_n = out_s_n[:, V0_LSRR, 0, :, :] + out_s_n[:, V0_CRR, 0, :, :]
            true_precip_n = tgt_s_n[:, V0_LSRR, 0, :, :] + tgt_s_n[:, V0_CRR, 0, :, :]
            m['precip_pcc']  = float(pcc(pred_precip, true_precip)[0])
            m['precip_acc']  = float(acc(pred_precip, true_precip, precip_clim)[0])
            m['precip_rmse'] = float(rmse(pred_precip_n, true_precip_n)[0])

            # T2M
            m['t2m_pcc']  = float(pcc(out_s[:, V0_T2M, 0], tgt_s[:, V0_T2M, 0])[0])
            m['t2m_acc']  = float(acc(out_s[:, V0_T2M, 0], tgt_s[:, V0_T2M, 0], t2m_clim)[0])
            m['t2m_rmse'] = float(rmse(out_s_n[:, V0_T2M, 0], tgt_s_n[:, V0_T2M, 0])[0])

            # D2M
            m['d2m_pcc']  = float(pcc(out_s[:, V0_D2M, 0], tgt_s[:, V0_D2M, 0])[0])
            m['d2m_acc']  = float(acc(out_s[:, V0_D2M, 0], tgt_s[:, V0_D2M, 0], d2m_clim)[0])
            m['d2m_rmse'] = float(rmse(out_s_n[:, V0_D2M, 0], tgt_s_n[:, V0_D2M, 0])[0])

            # U850 (V0: u is var idx 2, 850hPa is level idx 3)
            m['u850_pcc']  = float(pcc(out_u[:, V0_U_IDX, 3, 0], tgt_u[:, V0_U_IDX, 3, 0])[0])
            m['u850_acc']  = float(acc(out_u[:, V0_U_IDX, 3, 0], tgt_u[:, V0_U_IDX, 3, 0], u850_clim)[0])
            m['u850_rmse'] = float(rmse(out_u_n[:, V0_U_IDX, 3, 0], tgt_u_n[:, V0_U_IDX, 3, 0])[0])

            # Aggregate RMSE
            m['surface_rmse']   = float(rmse(out_s_n, tgt_s_n)[0])
            m['upper_air_rmse'] = float(rmse(out_u_n, tgt_u_n)[0])

            m['year'] = sample_years[i]
            m['week'] = sample_week_nums[i]
            m['global_idx'] = int(gi[0])
            records.append(m)

            # Free GPU memory
            del inp_s, inp_u, tgt_s, tgt_u
            del inp_s_n, inp_u_n, out_s_raw, out_u_raw, out_s_n, out_u_n, out_s, out_u
            del tgt_s_n, tgt_u_n
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()

            if (i + 1) % 10 == 0 or i == 0:
                print(f'[{i+1:3d}/{len(dataset)}] year={m["year"]} week={m["week"]:02d} '
                      f'precip_pcc={m["precip_pcc"]:.4f} t2m_pcc={m["t2m_pcc"]:.4f} '
                      f'd2m_pcc={m["d2m_pcc"]:.4f} u850_pcc={m["u850_pcc"]:.4f}')

    # ── Save per-sample CSV ────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, 'per_sample_metrics.csv')
    fields = ['year', 'week', 'global_idx',
              'precip_pcc', 'precip_acc', 'precip_rmse',
              't2m_pcc', 't2m_acc', 't2m_rmse',
              'd2m_pcc', 'd2m_acc', 'd2m_rmse',
              'u850_pcc', 'u850_acc', 'u850_rmse',
              'surface_rmse', 'upper_air_rmse']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fields})
    print(f'\nPer-sample metrics saved to {csv_path}')

    # ── Per-year summary ───────────────────────────────────────────
    print('\n' + '='*80)
    print('Per-year summary (V0 Lite):')
    print('='*80)
    var_keys = ['precip', 't2m', 'd2m', 'u850']
    header = f'{"Year":>6} | ' + ' | '.join(f'{v}_pcc' for v in var_keys) + ' | surface_rmse | upper_rmse'
    print(header)
    print('-' * len(header))

    year_summaries = []
    for year in range(2017, 2022):
        yr = [r for r in records if r['year'] == year]
        summary = {'year': year, 'n_samples': len(yr)}
        for vk in var_keys:
            for metric in ['pcc', 'acc', 'rmse']:
                key = f'{vk}_{metric}'
                summary[key] = np.mean([r[key] for r in yr])
        summary['surface_rmse'] = np.mean([r['surface_rmse'] for r in yr])
        summary['upper_air_rmse'] = np.mean([r['upper_air_rmse'] for r in yr])
        year_summaries.append(summary)
        vals = ' | '.join(f'{summary[f"{v}_pcc"]:>10.4f}' for v in var_keys)
        print(f'{year:>6} | {vals} | {summary["surface_rmse"]:>12.4f} | {summary["upper_air_rmse"]:>10.4f}')

    # Overall average
    overall = {f'{vk}_{mt}': np.mean([r[f'{vk}_{mt}'] for r in records])
               for vk in var_keys for mt in ['pcc', 'acc', 'rmse']}
    overall['surface_rmse'] = np.mean([r['surface_rmse'] for r in records])
    overall['upper_air_rmse'] = np.mean([r['upper_air_rmse'] for r in records])
    vals = ' | '.join(f'{overall[f"{v}_pcc"]:>10.4f}' for v in var_keys)
    print(f'{"ALL":>6} | {vals} | {overall["surface_rmse"]:>12.4f} | {overall["upper_air_rmse"]:>10.4f}')

    # Save year summary
    summary_path = os.path.join(OUT_DIR, 'year_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        sfields = ['year', 'n_samples'] + [f'{v}_{m}' for v in var_keys for m in ['pcc', 'acc', 'rmse']] + ['surface_rmse', 'upper_air_rmse']
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        for s in year_summaries:
            w.writerow({k: s[k] for k in sfields})
    print(f'Year summary saved to {summary_path}')

    print('\nDone.')

if __name__ == '__main__':
    main()
