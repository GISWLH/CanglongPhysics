"""
Generate standardized evaluation dataset for CAS-Canglong V0 (Lite).

Target-week-centric format:
  - time axis = ALL continuous weeks in 2017-2021
  - obs_{var}(time, lat, lon)          : ERA5 ground truth, stored once
  - pred_{var}_lead{1..6}(time, lat, lon) : model predictions per lead

pred_olr is NaN because V0 does not predict OLR (avg_tnlwrf).

Output: Infer/eval/model_v0.nc

Usage (MUST run from canglong/ directory):
    cd /home/lhwang/Desktop/CanglongPhysics/canglong
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/lhwang/anaconda3/envs/torch/bin/python ../Infer/gen_eval_v0.py
"""

import torch
import numpy as np
import os
import sys
import json
import numcodecs
import netCDF4 as nc4
import __main__

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code_v2'))

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

for cls in [Canglong, Encoder, Decoder, BasicLayer,
            EarthSpecificBlock, EarthAttention3D, UpSample, DownSample, Mlp,
            GroupNorm, Swish, ResidualBlock, UpSampleBlock, DownSampleBlock, NonLocalBlock]:
    setattr(__main__, cls.__name__, cls)

STORE_PATH = '/data/lhwang/ERA5_raw/weekly_data_zarr/ERA5_1982_2023_weekly.zarr'
MODEL_PATH = '/home/lhwang/Desktop/model/weather_model_epoch_500.pt'
WOY_PATH   = os.path.join(ROOT, 'Infer/eval/woy_map.npy')
OUT_DIR    = os.path.join(ROOT, 'Infer/eval')
os.makedirs(OUT_DIR, exist_ok=True)

N_LEADS = 6
H, W = 721, 1440
EVAL_VARS = ['tp', 't2m', 'olr', 'z500']

# ── V0 variable mapping ─────────────────────────────────────────
V0_SURFACE_ZARR_IDX = [4, 5, 2, 3, 0, 7, 8, 9, 10, 13, 14, 19, 25, 20, 21, 22]
V0_UPPER_ZARR_IDX = [1, 5, 3, 4, 7, 2, 6]
V0_LEVEL_ZARR_IDX = [1, 2, 3, 4]

V0_LSRR, V0_CRR = 0, 1
V0_T2M = 8
V0_Z_IDX = 0
V0_P500_LEVEL = 1

# Zarr indices for obs
ZARR_LSRR, ZARR_CRR = 4, 5
ZARR_T2M = 10
ZARR_OLR = 1
ZARR_Z_IDX = 1
ZARR_P500_IDX = 2

# V0 scalar normalization
V0_SURFACE_SHORT = ['lsrr','crr','tciw','tcc','tsrc','u10','v10','d2m','t2m',
                     'slhf','sshf','sp','swvl','msl','siconc','sst']
V0_UPPER_NAMES = ['z', 'w', 'u', 'v', 'cc', 't', 'q']
V0_LEVELS = [300, 500, 700, 850]

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

# ── Zarr reader ──────────────────────────────────────────────────
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

# ── Main ─────────────────────────────────────────────────────────
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Time indexing
    time_days = read_time_array(STORE_PATH)
    base = np.datetime64('1940-01-01')
    dates = base + time_days.astype('timedelta64[D]')
    years = dates.astype('datetime64[Y]').astype(int) + 1970

    woy_map = np.load(WOY_PATH)

    # Target weeks: ALL weeks in 2017-2021 (continuous)
    target_mask = (years >= 2017) & (years <= 2021)
    target_gidx = np.where(target_mask)[0]
    T = len(target_gidx)
    gi_to_tidx = {int(gi): i for i, gi in enumerate(target_gidx)}

    target_years = years[target_gidx]
    target_woys = np.array([int(woy_map[gi]) for gi in target_gidx])
    print(f'Target weeks: {T} (2017-2021, continuous)')

    # Init points
    gi_min = int(target_gidx[0])
    gi_max = int(target_gidx[-1])
    init_start = gi_min - 7
    init_end = gi_max - 2
    init_points = list(range(init_start, init_end + 1))
    N_init = len(init_points)
    print(f'Init points: {N_init} (gi {init_start}..{init_end})')

    # Data readers
    surface_arr = ZarrArray(STORE_PATH, 'surface')
    upper_arr   = ZarrArray(STORE_PATH, 'upper_air')

    # V0 normalization
    s_mean_np, s_std_np, u_mean_np, u_std_np = build_v0_norm_arrays()
    s_mean = torch.from_numpy(s_mean_np).float().unsqueeze(0).to(device)
    s_std  = torch.from_numpy(s_std_np).float().unsqueeze(0).to(device)
    u_mean = torch.from_numpy(u_mean_np).float().unsqueeze(0).unsqueeze(3).to(device)
    u_std  = torch.from_numpy(u_std_np).float().unsqueeze(0).unsqueeze(3).to(device)

    # Load V0 model
    print(f'Loading model: {MODEL_PATH}')
    dp_model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = dp_model.module
    model.input_constant = model.input_constant.to(device)
    model.to(device).eval()
    print('Model loaded (V0 Lite).')

    # ── Create output NetCDF4 ────────────────────────────────────
    out_path = os.path.join(OUT_DIR, 'model_v0.nc')
    ds = nc4.Dataset(out_path, 'w', format='NETCDF4')

    ds.createDimension('time', T)
    ds.createDimension('lat', H)
    ds.createDimension('lon', W)

    # Time coordinate (CF-convention)
    time_v = ds.createVariable('time', 'f8', ('time',))
    time_v.units = 'days since 1940-01-01'
    time_v.calendar = 'standard'
    time_v[:] = time_days[target_gidx]

    lat_v = ds.createVariable('lat', 'f4', ('lat',))
    lat_v[:] = np.linspace(90, -90, H)
    lat_v.units = 'degrees_north'
    lon_v = ds.createVariable('lon', 'f4', ('lon',))
    lon_v[:] = np.linspace(0, 360 - 0.25, W)
    lon_v.units = 'degrees_east'

    year_v = ds.createVariable('year', 'i4', ('time',))
    year_v[:] = target_years
    year_v.long_name = 'year of target week'
    woy_v = ds.createVariable('woy', 'i4', ('time',))
    woy_v[:] = target_woys
    woy_v.long_name = 'week-of-year (0-indexed, for climatology lookup)'
    gidx_v = ds.createVariable('global_idx', 'i4', ('time',))
    gidx_v[:] = target_gidx
    gidx_v.long_name = 'Zarr global time index'

    # Obs variables
    obs_nc = {}
    for var in EVAL_VARS:
        obs_nc[var] = ds.createVariable(
            f'obs_{var}', 'f4', ('time', 'lat', 'lon'),
            zlib=True, complevel=4, chunksizes=(1, H, W))

    # Pred variables
    pred_nc = {}
    for var in EVAL_VARS:
        for lead in range(1, N_LEADS + 1):
            vname = f'pred_{var}_lead{lead}'
            pred_nc[(var, lead)] = ds.createVariable(
                vname, 'f4', ('time', 'lat', 'lon'),
                zlib=True, complevel=4, chunksizes=(1, H, W))

    var_info = {
        'tp':   ('Total precipitation (lsrr+crr)', 'kg/m2/s'),
        't2m':  ('2m temperature', 'K'),
        'olr':  ('Top net long wave radiation (OLR)', 'W/m2'),
        'z500': ('Geopotential at 500hPa', 'm2/s2'),
    }
    for var in EVAL_VARS:
        lname, units = var_info[var]
        obs_nc[var].long_name = f'{lname} (ERA5 obs)'
        obs_nc[var].units = units
        for lead in range(1, N_LEADS + 1):
            pred_nc[(var, lead)].long_name = f'{lname} (pred, lead {lead}w)'
            pred_nc[(var, lead)].units = units

    ds.model_name = 'CAS-Canglong V0 Lite'
    ds.model_version = 'V0 (Canglong)'
    ds.test_period = '2017-2021'
    ds.n_target_weeks = T
    ds.n_leads = N_LEADS
    ds.format = 'target-week-centric: obs once, pred per lead'
    ds.note_olr = 'pred_olr_lead* is NaN because V0 does not predict OLR'

    print(f'Output: {out_path}')

    # ── Phase 1: Write obs (CPU only) ────────────────────────────
    print(f'Phase 1: Writing observations ({T} target weeks)...')
    for i in range(T):
        gi = int(target_gidx[i])
        raw_s = surface_arr.read_time(gi)
        raw_u = upper_arr.read_time(gi)

        obs_nc['tp'][i]   = (raw_s[ZARR_LSRR] + raw_s[ZARR_CRR]).astype(np.float32)
        obs_nc['t2m'][i]  = raw_s[ZARR_T2M].astype(np.float32)
        obs_nc['olr'][i]  = raw_s[ZARR_OLR].astype(np.float32)
        obs_nc['z500'][i] = raw_u[ZARR_Z_IDX, ZARR_P500_IDX].astype(np.float32)

        if (i + 1) % 50 == 0 or i == 0:
            print(f'  [{i+1:3d}/{T}] year={target_years[i]} woy={target_woys[i]:02d}')
    ds.sync()
    print('  Observations written.')

    # ── Phase 2: Inference (GPU) ─────────────────────────────────
    print(f'Phase 2: Inference ({N_init} init points, {N_LEADS} leads each)...')
    nan_field = np.full((H, W), np.nan, dtype=np.float32)

    with torch.no_grad():
        for idx, gi in enumerate(init_points):
            # Read full Zarr, then subset to V0 space
            raw_s0_full = surface_arr.read_time(gi)
            raw_s1_full = surface_arr.read_time(gi + 1)
            raw_u0_full = upper_arr.read_time(gi)
            raw_u1_full = upper_arr.read_time(gi + 1)

            cur_s0 = raw_s0_full[V0_SURFACE_ZARR_IDX]
            cur_s1 = raw_s1_full[V0_SURFACE_ZARR_IDX]
            cur_u0 = raw_u0_full[np.ix_(V0_UPPER_ZARR_IDX, V0_LEVEL_ZARR_IDX)]
            cur_u1 = raw_u1_full[np.ix_(V0_UPPER_ZARR_IDX, V0_LEVEL_ZARR_IDX)]

            for lead_idx in range(N_LEADS):
                lead = lead_idx + 1
                target_gi = gi + 2 + lead_idx

                inp_s = torch.from_numpy(
                    np.stack([cur_s0, cur_s1], axis=1)[None]).float().to(device)
                inp_u = torch.from_numpy(
                    np.stack([cur_u0, cur_u1], axis=2)[None]).float().to(device)

                inp_s_n = torch.nan_to_num((inp_s - s_mean) / s_std, nan=0.0)
                inp_u_n = torch.nan_to_num((inp_u - u_mean) / u_std, nan=0.0)

                out_s_raw, out_u_raw = model(inp_s_n, inp_u_n)
                out_s_n = out_s_raw[:, :, 0:1, :, :]
                out_u_n = out_u_raw[:, :, :, 0:1, :, :]

                pred_s = (out_s_n * s_std + s_mean)[0, :, 0].cpu().numpy()
                pred_u = (out_u_n * u_std + u_mean)[0, :, :, 0].cpu().numpy()

                # Scatter to target week if within 2017-2021
                if target_gi in gi_to_tidx:
                    tidx = gi_to_tidx[target_gi]
                    pred_nc[('tp', lead)][tidx]   = (pred_s[V0_LSRR] + pred_s[V0_CRR]).astype(np.float32)
                    pred_nc[('t2m', lead)][tidx]  = pred_s[V0_T2M].astype(np.float32)
                    pred_nc[('olr', lead)][tidx]   = nan_field
                    pred_nc[('z500', lead)][tidx] = pred_u[V0_Z_IDX, V0_P500_LEVEL].astype(np.float32)

                # Autoregressive update (V0 space)
                cur_s0, cur_s1 = cur_s1, pred_s
                cur_u0, cur_u1 = cur_u1, pred_u

                del inp_s, inp_u, inp_s_n, inp_u_n, out_s_raw, out_u_raw, out_s_n, out_u_n

            if (idx + 1) % 5 == 0:
                torch.cuda.empty_cache()
            if (idx + 1) % 10 == 0 or idx == 0:
                print(f'  [{idx+1:3d}/{N_init}] gi={gi}')

    ds.close()
    file_size = os.path.getsize(out_path) / 1e9
    print(f'\nSaved: {out_path} ({file_size:.2f} GB)')
    print('Done.')


if __name__ == '__main__':
    main()
