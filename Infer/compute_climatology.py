"""
Compute 52-week climatology from 2002-2016 ERA5 Zarr data.
Variables: TP (lsrr+crr), T2M, OLR (avg_tnlwrf), Z500, U850, U200.

Output: Infer/eval/climatology_2002_2016.nc + woy_map.npy

Usage:
    cd /home/lhwang/Desktop/CanglongPhysics
    /home/lhwang/anaconda3/envs/torch/bin/python Infer/compute_climatology.py
"""

import numpy as np
import os
import json
import numcodecs
import xarray as xr

STORE_PATH = '/data/lhwang/ERA5_raw/weekly_data_zarr/ERA5_1982_2023_weekly.zarr'
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval')
os.makedirs(OUT_DIR, exist_ok=True)

N_WEEKS = 52
H, W = 721, 1440

# Zarr variable indices
LSRR_IDX, CRR_IDX = 4, 5   # surface
T2M_IDX = 10                 # surface
OLR_IDX = 1                  # surface (avg_tnlwrf)
Z_IDX = 1                    # upper air (z = geopotential)
P500_IDX = 2                 # upper air pressure level (500hPa)
U_IDX = 3                    # upper air (u = zonal wind)
LEVEL_850_IDX = 4            # 850hPa
LEVEL_200_IDX = 0            # 200hPa

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
    print('Computing 52-week climatology from 2002-2016 ERA5 data...')

    # Time indexing
    time_days = read_time_array(STORE_PATH)
    base = np.datetime64('1940-01-01')
    dates = base + time_days.astype('timedelta64[D]')
    years = dates.astype('datetime64[Y]').astype(int) + 1970

    # Build week-of-year map for all data (0-indexed: 0-51)
    woy_map = np.full(len(dates), -1, dtype=np.int32)
    for year in range(1982, 2024):
        idx_year = np.where(years == year)[0]
        for k, gi in enumerate(idx_year):
            if k < N_WEEKS:
                woy_map[gi] = k

    woy_path = os.path.join(OUT_DIR, 'woy_map.npy')
    np.save(woy_path, woy_map)
    print(f'Week-of-year map saved: {woy_path} (shape={woy_map.shape})')

    # Initialize accumulators (float64 for numerical stability)
    tp_sum   = np.zeros((N_WEEKS, H, W), dtype=np.float64)
    t2m_sum  = np.zeros((N_WEEKS, H, W), dtype=np.float64)
    olr_sum  = np.zeros((N_WEEKS, H, W), dtype=np.float64)
    z500_sum = np.zeros((N_WEEKS, H, W), dtype=np.float64)
    u850_sum = np.zeros((N_WEEKS, H, W), dtype=np.float64)
    u200_sum = np.zeros((N_WEEKS, H, W), dtype=np.float64)
    count    = np.zeros(N_WEEKS, dtype=np.int32)

    # Data readers
    surface_arr = ZarrArray(STORE_PATH, 'surface')
    upper_arr   = ZarrArray(STORE_PATH, 'upper_air')
    print(f'Surface shape: {surface_arr.shape}, Upper shape: {upper_arr.shape}')

    # Accumulate over 2002-2016 (15 years)
    total_count = 0
    for year in range(2002, 2017):
        idx_year = np.where(years == year)[0]
        print(f'  Year {year}: {len(idx_year)} weeks (global indices {idx_year[0]}-{idx_year[-1]})')

        for k, gi in enumerate(idx_year):
            gi = int(gi)
            woy = k  # week-of-year (0-indexed)
            if woy >= N_WEEKS:
                continue

            s = surface_arr.read_time(gi)  # (26, 721, 1440)
            u = upper_arr.read_time(gi)    # (10, 5, 721, 1440)

            tp_sum[woy]   += (s[LSRR_IDX] + s[CRR_IDX]).astype(np.float64)
            t2m_sum[woy]  += s[T2M_IDX].astype(np.float64)
            olr_sum[woy]  += s[OLR_IDX].astype(np.float64)
            z500_sum[woy] += u[Z_IDX, P500_IDX].astype(np.float64)
            u850_sum[woy] += u[U_IDX, LEVEL_850_IDX].astype(np.float64)
            u200_sum[woy] += u[U_IDX, LEVEL_200_IDX].astype(np.float64)
            count[woy] += 1
            total_count += 1

        print(f'    Cumulative: {total_count} chunks processed')

    # Compute mean
    print(f'\nTotal chunks: {total_count}')
    print(f'Counts per week: min={count.min()}, max={count.max()}, mean={count.mean():.1f}')

    for w in range(N_WEEKS):
        if count[w] > 0:
            tp_sum[w]   /= count[w]
            t2m_sum[w]  /= count[w]
            olr_sum[w]  /= count[w]
            z500_sum[w] /= count[w]
            u850_sum[w] /= count[w]
            u200_sum[w] /= count[w]

    # Save as NetCDF
    lat = np.linspace(90, -90, H)
    lon = np.linspace(0, 360 - 0.25, W)

    ds = xr.Dataset(
        data_vars={
            'tp_clim':   (['week', 'lat', 'lon'], tp_sum.astype(np.float32)),
            't2m_clim':  (['week', 'lat', 'lon'], t2m_sum.astype(np.float32)),
            'olr_clim':  (['week', 'lat', 'lon'], olr_sum.astype(np.float32)),
            'z500_clim': (['week', 'lat', 'lon'], z500_sum.astype(np.float32)),
            'u850_clim': (['week', 'lat', 'lon'], u850_sum.astype(np.float32)),
            'u200_clim': (['week', 'lat', 'lon'], u200_sum.astype(np.float32)),
            'count':     (['week'], count),
        },
        coords={
            'week': np.arange(N_WEEKS),  # 0-indexed
            'lat': lat,
            'lon': lon,
        },
        attrs={
            'description': 'Weekly climatology for TCC computation',
            'clim_period': '2002-2016 (15 years)',
            'tp': 'Total precipitation (lsrr+crr), kg/m2/s',
            't2m': '2m temperature, K',
            'olr': 'Mean top net longwave radiation flux (avg_tnlwrf), W/m2',
            'z500': 'Geopotential at 500hPa, m2/s2',
            'u850': 'Zonal wind at 850hPa, m/s',
            'u200': 'Zonal wind at 200hPa, m/s',
            'week_convention': '0-indexed week of year (0=first week of Jan)',
        }
    )

    out_path = os.path.join(OUT_DIR, 'climatology_2002_2016.nc')
    ds.to_netcdf(out_path, encoding={
        'tp_clim':   {'zlib': True, 'complevel': 4},
        't2m_clim':  {'zlib': True, 'complevel': 4},
        'olr_clim':  {'zlib': True, 'complevel': 4},
        'z500_clim': {'zlib': True, 'complevel': 4},
        'u850_clim': {'zlib': True, 'complevel': 4},
        'u200_clim': {'zlib': True, 'complevel': 4},
    })

    file_size = os.path.getsize(out_path) / 1e6
    print(f'\nSaved: {out_path} ({file_size:.1f} MB)')

    # Print statistics
    print('\nClimatology statistics:')
    for name, data in [('TP', tp_sum), ('T2M', t2m_sum), ('OLR', olr_sum), ('Z500', z500_sum),
                        ('U850', u850_sum), ('U200', u200_sum)]:
        print(f'  {name}: min={data.min():.6f}, max={data.max():.6f}, mean={data.mean():.6f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
