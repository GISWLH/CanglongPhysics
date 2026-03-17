"""
Generate standardized evaluation dataset for CAS-Canglong V3.5.

Target-week-centric format:
  - time axis = ALL continuous weeks in 2017-2021
  - obs_{var}(time, lat, lon)          : ERA5 ground truth, stored once
  - pred_{var}_lead{1..6}(time, lat, lon) : model predictions per lead

For each target week t:
  - obs is the ERA5 value at week t
  - pred_lead1 is the 1-week-ahead forecast (init at t-2, 1 autoregressive step)
  - pred_lead6 is the 6-week-ahead forecast (init at t-7, 6 autoregressive steps)

Output: Infer/eval/model_v3.nc

Usage:
    cd /home/lhwang/Desktop/CanglongPhysics
    PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    /home/lhwang/anaconda3/envs/torch/bin/python Infer/gen_eval_v3.py
"""

import torch
import numpy as np
import os
import sys
import json
import numcodecs
import netCDF4 as nc4

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'code_v2'))

from canglong import CanglongV2_5
from convert_dict_to_pytorch_arrays_v2 import load_normalization_arrays

STORE_PATH = '/data/lhwang/ERA5_raw/weekly_data_zarr/ERA5_1982_2023_weekly.zarr'
NORM_JSON  = os.path.join(ROOT, 'code_v2/ERA5_1940_2023_mean_std_v2.json')
MODEL_PATH = os.path.join(ROOT, 'model/model_v3_5_continue_record_ft2_best.pth')
WOY_PATH   = os.path.join(ROOT, 'Infer/eval/woy_map.npy')
OUT_DIR    = os.path.join(ROOT, 'Infer/eval')
os.makedirs(OUT_DIR, exist_ok=True)

N_LEADS = 6
H, W = 721, 1440

# Zarr variable indices
LSRR_IDX, CRR_IDX = 4, 5
T2M_IDX = 10
OLR_IDX = 1
Z_IDX = 1
P500_IDX = 2
U_IDX = 3
LEVEL_200 = 0
LEVEL_850 = 4

EVAL_VARS = ['tp', 't2m', 'olr', 'z500', 'u850', 'u200']

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

# ── Extract eval variables from raw arrays ───────────────────────
def extract_eval_surface(s):
    return {
        'tp':  (s[LSRR_IDX] + s[CRR_IDX]).astype(np.float32),
        't2m': s[T2M_IDX].astype(np.float32),
        'olr': s[OLR_IDX].astype(np.float32),
    }

def extract_eval_upper(u):
    return {
        'z500': u[Z_IDX, P500_IDX].astype(np.float32),
        'u850': u[U_IDX, LEVEL_850].astype(np.float32),
        'u200': u[U_IDX, LEVEL_200].astype(np.float32),
    }

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

    # Init points: cover all target weeks for all leads
    # For lead L (1-6), init at gi = target - L - 1
    # lead 6 of first target needs init at target_min - 7
    # lead 1 of last target needs init at target_max - 2
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

    # Normalization (V3.5: spatially varying)
    s_mean_np, s_std_np, u_mean_np, u_std_np = load_normalization_arrays(NORM_JSON)
    s_mean = torch.from_numpy(s_mean_np).float().to(device)
    s_std  = torch.from_numpy(s_std_np).float().to(device)
    u_mean = torch.from_numpy(u_mean_np).float().to(device)
    u_std  = torch.from_numpy(u_std_np).float().to(device)

    # Load model
    print(f'Loading model: {MODEL_PATH}')
    model = CanglongV2_5()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print('Model loaded (V3.5 CanglongV2_5).')

    # ── Create output NetCDF4 ────────────────────────────────────
    out_path = os.path.join(OUT_DIR, 'model_v3.nc')
    ds = nc4.Dataset(out_path, 'w', format='NETCDF4')

    ds.createDimension('time', T)
    ds.createDimension('lat', H)
    ds.createDimension('lon', W)

    # Time coordinate (CF-convention: decodable by xarray)
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

    # Auxiliary coordinates
    year_v = ds.createVariable('year', 'i4', ('time',))
    year_v[:] = target_years
    year_v.long_name = 'year of target week'
    woy_v = ds.createVariable('woy', 'i4', ('time',))
    woy_v[:] = target_woys
    woy_v.long_name = 'week-of-year (0-indexed, for climatology lookup)'
    gidx_v = ds.createVariable('global_idx', 'i4', ('time',))
    gidx_v[:] = target_gidx
    gidx_v.long_name = 'Zarr global time index'

    # Obs variables: obs_{var}(time, lat, lon)
    obs_nc = {}
    for var in EVAL_VARS:
        obs_nc[var] = ds.createVariable(
            f'obs_{var}', 'f4', ('time', 'lat', 'lon'),
            zlib=True, complevel=4, chunksizes=(1, H, W))

    # Pred variables: pred_{var}_lead{1..6}(time, lat, lon)
    pred_nc = {}
    for var in EVAL_VARS:
        for lead in range(1, N_LEADS + 1):
            vname = f'pred_{var}_lead{lead}'
            pred_nc[(var, lead)] = ds.createVariable(
                vname, 'f4', ('time', 'lat', 'lon'),
                zlib=True, complevel=4, chunksizes=(1, H, W))

    # Variable metadata
    var_info = {
        'tp':   ('Total precipitation (lsrr+crr)', 'kg/m2/s'),
        't2m':  ('2m temperature', 'K'),
        'olr':  ('Top net long wave radiation (OLR)', 'W/m2'),
        'z500': ('Geopotential at 500hPa', 'm2/s2'),
        'u850': ('Zonal wind at 850hPa', 'm/s'),
        'u200': ('Zonal wind at 200hPa', 'm/s'),
    }
    for var in EVAL_VARS:
        lname, units = var_info[var]
        obs_nc[var].long_name = f'{lname} (ERA5 obs)'
        obs_nc[var].units = units
        for lead in range(1, N_LEADS + 1):
            pred_nc[(var, lead)].long_name = f'{lname} (pred, lead {lead}w)'
            pred_nc[(var, lead)].units = units

    # Global attributes
    ds.model_name = 'CAS-Canglong V3.5'
    ds.model_version = 'V3.5 (CanglongV2_5)'
    ds.test_period = '2017-2021'
    ds.n_target_weeks = T
    ds.n_leads = N_LEADS
    ds.format = 'target-week-centric: obs once, pred per lead'
    ds.description = ('For each target week t: obs = ERA5 at t; '
                      'pred_lead{L} = L-week-ahead autoregressive forecast for t')

    print(f'Output: {out_path}')

    # ── Phase 1: Write obs (CPU only) ────────────────────────────
    print(f'Phase 1: Writing observations ({T} target weeks)...')
    for i in range(T):
        gi = int(target_gidx[i])
        raw_s = surface_arr.read_time(gi)
        raw_u = upper_arr.read_time(gi)
        obs_eval = {**extract_eval_surface(raw_s), **extract_eval_upper(raw_u)}
        for var in EVAL_VARS:
            obs_nc[var][i] = obs_eval[var]
        if (i + 1) % 50 == 0 or i == 0:
            print(f'  [{i+1:3d}/{T}] year={target_years[i]} woy={target_woys[i]:02d}')
    ds.sync()
    print('  Observations written.')

    # ── Phase 2: Inference (GPU) ─────────────────────────────────
    # Each init point gi produces predictions for target weeks gi+2..gi+7
    # (lead 1..6). We scatter each prediction to the correct target slot.
    print(f'Phase 2: Inference ({N_init} init points, {N_LEADS} leads each)...')
    with torch.no_grad():
        for idx, gi in enumerate(init_points):
            raw_s0 = surface_arr.read_time(gi)
            raw_s1 = surface_arr.read_time(gi + 1)
            raw_u0 = upper_arr.read_time(gi)
            raw_u1 = upper_arr.read_time(gi + 1)

            cur_s0, cur_s1 = raw_s0, raw_s1
            cur_u0, cur_u1 = raw_u0, raw_u1

            for lead_idx in range(N_LEADS):
                lead = lead_idx + 1
                target_gi = gi + 2 + lead_idx

                inp_s = torch.from_numpy(
                    np.stack([cur_s0, cur_s1], axis=1)[None]).float().to(device)
                inp_u = torch.from_numpy(
                    np.stack([cur_u0, cur_u1], axis=2)[None]).float().to(device)

                inp_s_n = torch.nan_to_num((inp_s - s_mean) / s_std, nan=0.0)
                inp_u_n = torch.nan_to_num((inp_u - u_mean) / u_std, nan=0.0)

                out_s_n, out_u_n = model(inp_s_n, inp_u_n)

                pred_s = (out_s_n * s_std + s_mean)[0, :, 0].cpu().numpy()
                pred_u = (out_u_n * u_std + u_mean)[0, :, :, 0].cpu().numpy()

                # Scatter to target week if within 2017-2021
                if target_gi in gi_to_tidx:
                    tidx = gi_to_tidx[target_gi]
                    pred_eval = {**extract_eval_surface(pred_s),
                                 **extract_eval_upper(pred_u)}
                    for var in EVAL_VARS:
                        pred_nc[(var, lead)][tidx] = pred_eval[var]

                # Autoregressive update
                cur_s0, cur_s1 = cur_s1, pred_s
                cur_u0, cur_u1 = cur_u1, pred_u

                del inp_s, inp_u, inp_s_n, inp_u_n, out_s_n, out_u_n

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
