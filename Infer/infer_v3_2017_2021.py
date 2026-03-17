"""
Inference script for CAS-Canglong V3.5 on the 2017-2021 test set.
Evaluates the best model (model_v3_5_continue_record_ft2_best.pth)
on 5 years of weekly data, computing per-sample and per-year metrics
for S2S/MJO key variables.

Usage:
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    /home/lhwang/anaconda3/envs/torch/bin/python Infer/infer_2017_2021.py
"""

import torch
import numpy as np
import os
import sys
import json
import numcodecs
import csv
from torch.utils.data import Dataset, DataLoader

# ── paths ──────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code_v2'))

from canglong import CanglongV2_5
from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays

STORE_PATH = '/data/lhwang/ERA5_raw/weekly_data_zarr/ERA5_1982_2023_weekly.zarr'
NORM_JSON  = os.path.join(ROOT, 'code_v2/ERA5_1940_2023_mean_std_v2.json')
MODEL_PATH = os.path.join(ROOT, 'model/model_v3_5_continue_record_ft2_best.pth')
OUT_DIR    = os.path.join(ROOT, 'Infer/results_2017_2021')
os.makedirs(OUT_DIR, exist_ok=True)

# ── variable indices ───────────────────────────────────────────────
SURFACE_VARS = [
    'avg_tnswrf','avg_tnlwrf','tciw','tcc','lsrr','crr','blh',
    'u10','v10','d2m','t2m','avg_iews','avg_inss','slhf','sshf',
    'avg_snswrf','avg_snlwrf','ssr','str','sp','msl','siconc',
    'sst','ro','stl','swvl',
]
UPPER_VARS = ['o3','z','t','u','v','w','q','cc','ciwc','clwc']
PRESSURE_LEVELS = [200, 300, 500, 700, 850]

IDX = {v: i for i, v in enumerate(SURFACE_VARS)}
LSRR, CRR = IDX['lsrr'], IDX['crr']
OLR, T2M, D2M = IDX['avg_tnlwrf'], IDX['t2m'], IDX['d2m']
U_IDX = UPPER_VARS.index('u')
P200, P850 = PRESSURE_LEVELS.index(200), PRESSURE_LEVELS.index(850)

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
class ZarrWindowDataset(Dataset):
    """Each sample: (input_surface[t,t+1], input_upper[t,t+1]) -> (target_surface[t+2], target_upper[t+2])"""
    def __init__(self, surface_arr, upper_arr, global_indices):
        self.surface_arr = surface_arr
        self.upper_arr = upper_arr
        # global_indices: list of starting global time indices
        # each sample uses [idx, idx+1] as input and idx+2 as target
        self.indices = global_indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        s0 = self.surface_arr.read_time(idx)
        s1 = self.surface_arr.read_time(idx + 1)
        s2 = self.surface_arr.read_time(idx + 2)
        u0 = self.upper_arr.read_time(idx)
        u1 = self.upper_arr.read_time(idx + 1)
        u2 = self.upper_arr.read_time(idx + 2)
        input_surface = np.stack([s0, s1], axis=1)          # (26, 2, 721, 1440)
        input_upper   = np.stack([u0, u1], axis=2)          # (10, 5, 2, 721, 1440)
        target_surface = s2[:, None, :, :]                   # (26, 1, 721, 1440)
        target_upper   = u2[:, :, None, :, :]                # (10, 5, 1, 721, 1440)
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
    # Each year has 52 weeks; a sample needs 3 consecutive weeks
    global_indices = []
    sample_years = []
    sample_week_nums = []
    for year in range(2017, 2022):
        idx_year = np.where(years == year)[0]
        for k in range(len(idx_year) - 2):  # 50 samples per year
            gi = int(idx_year[k])
            global_indices.append(gi)
            sample_years.append(year)
            sample_week_nums.append(k + 1)  # target is week k+3
    print(f'Total samples: {len(global_indices)} (5 years x 50 weeks)')

    # Data readers
    surface_arr = ZarrArray(STORE_PATH, 'surface')
    upper_arr   = ZarrArray(STORE_PATH, 'upper_air')

    # Normalization
    s_mean_np, s_std_np, u_mean_np, u_std_np = load_normalization_arrays(NORM_JSON)
    s_mean = torch.from_numpy(s_mean_np).float().to(device)
    s_std  = torch.from_numpy(s_std_np).float().to(device)
    u_mean = torch.from_numpy(u_mean_np).float().to(device)
    u_std  = torch.from_numpy(u_std_np).float().to(device)

    # Climatology for ACC (mean of normalization = long-term mean)
    s_mean_2d = s_mean[0, :, 0, :, :]  # (26, 721, 1440)
    precip_clim = s_mean_2d[LSRR] + s_mean_2d[CRR]
    olr_clim    = s_mean_2d[OLR]
    t2m_clim    = s_mean_2d[T2M]
    d2m_clim    = s_mean_2d[D2M]
    u_mean_2d   = u_mean[0, :, :, 0, :, :]  # (10, 5, 721, 1440)
    u200_clim   = u_mean_2d[U_IDX, P200]
    u850_clim   = u_mean_2d[U_IDX, P850]

    # Model
    print(f'Loading model: {MODEL_PATH}')
    model = CanglongV2_5()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print('Model loaded.')

    # DataLoader
    dataset = ZarrWindowDataset(surface_arr, upper_arr, global_indices)
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

    # Helper extractors (defined once outside loop)
    def _extract_s(arr, idx):
        return arr[:, idx, 0, :, :]
    def _extract_u(arr, var, lev):
        return arr[:, var, lev, 0, :, :]

    # ── Inference loop ─────────────────────────────────────────────
    records = []
    with torch.no_grad():
        for i, (inp_s, inp_u, tgt_s, tgt_u, gi) in enumerate(loader):
            inp_s = inp_s.float().to(device)
            inp_u = inp_u.float().to(device)
            tgt_s = tgt_s.float().to(device)
            tgt_u = tgt_u.float().to(device)

            # Normalize input
            inp_s_n = (inp_s - s_mean) / s_std
            inp_u_n = (inp_u - u_mean) / u_std

            # Forward
            out_s_n, out_u_n = model(inp_s_n, inp_u_n)

            # Denormalize output
            out_s = out_s_n * s_std + s_mean
            out_u = out_u_n * u_std + u_mean

            # Normalize target (for RMSE in standardized space)
            tgt_s_n = (tgt_s - s_mean) / s_std
            tgt_u_n = (tgt_u - u_mean) / u_std

            # Precipitation (physical space for PCC/ACC, normalized for RMSE)
            pred_precip = _extract_s(out_s, LSRR) + _extract_s(out_s, CRR)
            true_precip = _extract_s(tgt_s, LSRR) + _extract_s(tgt_s, CRR)
            pred_precip_n = _extract_s(out_s_n, LSRR) + _extract_s(out_s_n, CRR)
            true_precip_n = _extract_s(tgt_s_n, LSRR) + _extract_s(tgt_s_n, CRR)

            m = {}
            m['precip_pcc']  = float(pcc(pred_precip, true_precip)[0])
            m['precip_acc']  = float(acc(pred_precip, true_precip, precip_clim)[0])
            m['precip_rmse'] = float(rmse(pred_precip_n, true_precip_n)[0])

            # OLR
            m['olr_pcc']  = float(pcc(_extract_s(out_s, OLR), _extract_s(tgt_s, OLR))[0])
            m['olr_acc']  = float(acc(_extract_s(out_s, OLR), _extract_s(tgt_s, OLR), olr_clim)[0])
            m['olr_rmse'] = float(rmse(_extract_s(out_s_n, OLR), _extract_s(tgt_s_n, OLR))[0])

            # T2M
            m['t2m_pcc']  = float(pcc(_extract_s(out_s, T2M), _extract_s(tgt_s, T2M))[0])
            m['t2m_acc']  = float(acc(_extract_s(out_s, T2M), _extract_s(tgt_s, T2M), t2m_clim)[0])
            m['t2m_rmse'] = float(rmse(_extract_s(out_s_n, T2M), _extract_s(tgt_s_n, T2M))[0])

            # D2M
            m['d2m_pcc']  = float(pcc(_extract_s(out_s, D2M), _extract_s(tgt_s, D2M))[0])
            m['d2m_acc']  = float(acc(_extract_s(out_s, D2M), _extract_s(tgt_s, D2M), d2m_clim)[0])
            m['d2m_rmse'] = float(rmse(_extract_s(out_s_n, D2M), _extract_s(tgt_s_n, D2M))[0])

            # U200
            m['u200_pcc']  = float(pcc(_extract_u(out_u, U_IDX, P200), _extract_u(tgt_u, U_IDX, P200))[0])
            m['u200_acc']  = float(acc(_extract_u(out_u, U_IDX, P200), _extract_u(tgt_u, U_IDX, P200), u200_clim)[0])
            m['u200_rmse'] = float(rmse(_extract_u(out_u_n, U_IDX, P200), _extract_u(tgt_u_n, U_IDX, P200))[0])

            # U850
            m['u850_pcc']  = float(pcc(_extract_u(out_u, U_IDX, P850), _extract_u(tgt_u, U_IDX, P850))[0])
            m['u850_acc']  = float(acc(_extract_u(out_u, U_IDX, P850), _extract_u(tgt_u, U_IDX, P850), u850_clim)[0])
            m['u850_rmse'] = float(rmse(_extract_u(out_u_n, U_IDX, P850), _extract_u(tgt_u_n, U_IDX, P850))[0])

            # Aggregate RMSE
            m['surface_rmse']   = float(rmse(out_s_n, tgt_s_n)[0])
            m['upper_air_rmse'] = float(rmse(out_u_n, tgt_u_n)[0])

            m['year'] = sample_years[i]
            m['week'] = sample_week_nums[i]
            m['global_idx'] = int(gi[0])
            records.append(m)

            # Free GPU tensors explicitly
            del inp_s, inp_u, tgt_s, tgt_u
            del inp_s_n, inp_u_n, out_s_n, out_u_n, out_s, out_u
            del tgt_s_n, tgt_u_n
            if (i + 1) % 20 == 0:
                torch.cuda.empty_cache()

            if (i + 1) % 10 == 0 or i == 0:
                print(f'[{i+1:3d}/{len(dataset)}] year={m["year"]} week={m["week"]:02d} '
                      f'precip_pcc={m["precip_pcc"]:.4f} t2m_pcc={m["t2m_pcc"]:.4f} '
                      f'olr_pcc={m["olr_pcc"]:.4f} u200_pcc={m["u200_pcc"]:.4f}')

    # ── Save per-sample CSV ────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, 'per_sample_metrics.csv')
    fields = ['year','week','global_idx',
              'precip_pcc','precip_acc','precip_rmse',
              'olr_pcc','olr_acc','olr_rmse',
              't2m_pcc','t2m_acc','t2m_rmse',
              'd2m_pcc','d2m_acc','d2m_rmse',
              'u200_pcc','u200_acc','u200_rmse',
              'u850_pcc','u850_acc','u850_rmse',
              'surface_rmse','upper_air_rmse']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r[k] for k in fields})
    print(f'\nPer-sample metrics saved to {csv_path}')

    # ── Per-year summary ───────────────────────────────────────────
    print('\n' + '='*80)
    print('Per-year summary:')
    print('='*80)
    var_keys = ['precip','olr','t2m','d2m','u200','u850']
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
               for vk in var_keys for mt in ['pcc','acc','rmse']}
    overall['surface_rmse'] = np.mean([r['surface_rmse'] for r in records])
    overall['upper_air_rmse'] = np.mean([r['upper_air_rmse'] for r in records])
    vals = ' | '.join(f'{overall[f"{v}_pcc"]:>10.4f}' for v in var_keys)
    print(f'{"ALL":>6} | {vals} | {overall["surface_rmse"]:>12.4f} | {overall["upper_air_rmse"]:>10.4f}')

    # Save year summary
    summary_path = os.path.join(OUT_DIR, 'year_summary.csv')
    with open(summary_path, 'w', newline='') as f:
        sfields = ['year','n_samples'] + [f'{v}_{m}' for v in var_keys for m in ['pcc','acc','rmse']] + ['surface_rmse','upper_air_rmse']
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        for s in year_summaries:
            w.writerow({k: s[k] for k in sfields})
    print(f'Year summary saved to {summary_path}')

    print('\nDone.')

if __name__ == '__main__':
    main()
