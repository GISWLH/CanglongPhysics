"""
Compute TCC (Temporal Correlation Coefficient) maps for CAS-Canglong V0 (Lite).

This version reads the pre-generated evaluation dataset
`Infer/eval/model_v0.nc` and does not rerun model inference.

Variables: TP, T2M, OLR, Z500.
V0 does not predict OLR, so OLR TCC remains NaN.

TCC = per-grid-point Pearson correlation of predicted vs observed anomalies
      across target weeks in 2017-2021.
Anomaly = value - weekly climatology (2002-2016).

Input:
    - Infer/eval/model_v0.nc
    - Infer/eval/climatology_2002_2016.nc

Output:
    - Infer/eval/tcc_v0.nc

Usage:
    cd /home/lhwang/Desktop/CanglongPhysics
    /home/lhwang/anaconda3/envs/torch/bin/python Infer/compute_tcc_v0.py
"""

import os

import numpy as np
import xarray as xr

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

EVAL_PATH = os.path.join(ROOT, 'Infer/eval/model_v0.nc')
CLIM_PATH = os.path.join(ROOT, 'Infer/eval/climatology_2002_2016.nc')
OUT_DIR = os.path.join(ROOT, 'Infer/eval')
OUT_PATH = os.path.join(OUT_DIR, 'tcc_v0.nc')

N_LEADS = 6
VAR_NAMES = ['tp', 't2m', 'olr', 'z500']
CLIM_VARS = {
    'tp': 'tp_clim',
    't2m': 't2m_clim',
    'olr': 'olr_clim',
    'z500': 'z500_clim',
}


def compute_tcc_map(pred_anom, obs_anom):
    n_samples = pred_anom.shape[0]

    sum_p = pred_anom.sum(axis=0, dtype=np.float64)
    sum_o = obs_anom.sum(axis=0, dtype=np.float64)
    sum_p2 = np.einsum('thw,thw->hw', pred_anom, pred_anom, dtype=np.float64, optimize=True)
    sum_o2 = np.einsum('thw,thw->hw', obs_anom, obs_anom, dtype=np.float64, optimize=True)
    sum_po = np.einsum('thw,thw->hw', pred_anom, obs_anom, dtype=np.float64, optimize=True)

    num = n_samples * sum_po - sum_p * sum_o
    den_p = n_samples * sum_p2 - sum_p ** 2
    den_o = n_samples * sum_o2 - sum_o ** 2
    den = np.sqrt(np.maximum(den_p * den_o, 0.0))

    tcc = np.full(sum_p.shape, np.nan, dtype=np.float32)
    valid = den > 1e-30
    tcc[valid] = (num[valid] / den[valid]).astype(np.float32)
    return tcc, n_samples


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(EVAL_PATH):
        raise FileNotFoundError(
            f'Missing eval dataset: {EVAL_PATH}\n'
            'Run Infer/gen_eval_v0.py first to generate Infer/eval/model_v0.nc.'
        )
    if not os.path.exists(CLIM_PATH):
        raise FileNotFoundError(
            f'Missing climatology dataset: {CLIM_PATH}\n'
            'Run Infer/compute_climatology.py first.'
        )

    print(f'Loading eval dataset: {EVAL_PATH}')
    eval_ds = xr.open_dataset(EVAL_PATH)
    print(f'Loading climatology: {CLIM_PATH}')
    clim_ds = xr.open_dataset(CLIM_PATH)

    woy = eval_ds['woy'].values.astype(np.int64)
    lat = eval_ds['lat'].values
    lon = eval_ds['lon'].values
    time = eval_ds['time'].values

    n_time = time.shape[0]
    print(f'Target weeks available: {n_time}')

    tcc = np.full((len(VAR_NAMES), N_LEADS, lat.shape[0], lon.shape[0]), np.nan, dtype=np.float32)
    n_samples_per_lead = np.zeros((len(VAR_NAMES), N_LEADS), dtype=np.int32)

    for vi, var_name in enumerate(VAR_NAMES):
        clim_name = CLIM_VARS[var_name]
        print(f'\nPreparing anomalies for {var_name}...')

        clim = clim_ds[clim_name].values.astype(np.float32, copy=False)
        clim_by_time = clim[woy]

        obs = eval_ds[f'obs_{var_name}'].values.astype(np.float32, copy=False)
        obs_anom = obs.copy()
        obs_anom -= clim_by_time

        if not np.isfinite(obs_anom).all():
            raise ValueError(f'Found non-finite values in obs_{var_name} anomalies.')

        for lead in range(1, N_LEADS + 1):
            print(f'  Lead {lead}: reading pred_{var_name}_lead{lead}')
            pred = eval_ds[f'pred_{var_name}_lead{lead}'].values.astype(np.float32, copy=False)

            if not np.isfinite(pred).any():
                print(f'    pred_{var_name}_lead{lead} is all-NaN; TCC remains NaN.')
                continue
            if not np.isfinite(pred).all():
                raise ValueError(f'Found mixed finite/non-finite values in pred_{var_name}_lead{lead}.')

            pred_anom = pred.copy()
            pred_anom -= clim_by_time

            if not np.isfinite(pred_anom).all():
                raise ValueError(f'Found non-finite values in pred_{var_name}_lead{lead} anomalies.')

            tcc_map, n_samples = compute_tcc_map(pred_anom, obs_anom)
            tcc[vi, lead - 1] = tcc_map
            n_samples_per_lead[vi, lead - 1] = n_samples

            del pred, pred_anom, tcc_map

        del clim, clim_by_time, obs, obs_anom

    eval_ds.close()
    clim_ds.close()

    data_vars = {}
    for vi, var_name in enumerate(VAR_NAMES):
        data_vars[f'{var_name}_tcc'] = (['lead', 'lat', 'lon'], tcc[vi])

    ds_out = xr.Dataset(
        data_vars=data_vars,
        coords={
            'lead': np.arange(1, N_LEADS + 1),
            'lat': lat,
            'lon': lon,
        },
        attrs={
            'model_name': 'CAS-Canglong V0 Lite',
            'model_version': 'V0 (Canglong)',
            'test_period': '2017-2021',
            'eval_source': 'Infer/eval/model_v0.nc',
            'clim_period': '2002-2016',
            'n_target_weeks': int(n_time),
            'description': 'Temporal Correlation Coefficient (TCC) maps',
            'tcc_definition': 'Per-grid-point Pearson correlation of anomaly time series',
            'note': 'Computed from pre-generated eval NetCDF without rerunning model inference',
            'note_olr': 'OLR TCC is NaN because V0 does not predict OLR',
        }
    )
    ds_out['n_samples'] = (
        ['variable', 'lead'],
        n_samples_per_lead,
    )
    ds_out = ds_out.assign_coords(variable=np.array(VAR_NAMES, dtype='U8'))
    ds_out['n_samples'].attrs['long_name'] = 'Number of target weeks used for each variable/lead TCC'

    encoding = {
        **{f'{var_name}_tcc': {'zlib': True, 'complevel': 4} for var_name in VAR_NAMES},
        'n_samples': {'zlib': True, 'complevel': 4},
    }
    ds_out.to_netcdf(OUT_PATH, encoding=encoding)

    file_size = os.path.getsize(OUT_PATH) / 1e6
    print(f'\nSaved: {OUT_PATH} ({file_size:.1f} MB)')

    print('\nGlobal mean TCC (V0 Lite, from model_v0.nc):')
    print(f'{"Lead":>6} | {"TP":>8} {"T2M":>8} {"OLR":>8} {"Z500":>8}')
    print('-' * 46)
    for lead_idx in range(N_LEADS):
        vals = []
        for vi in range(len(VAR_NAMES)):
            arr = tcc[vi, lead_idx]
            finite = np.isfinite(arr)
            vals.append(np.nan if not finite.any() else float(arr[finite].mean()))
        print(f'  {lead_idx+1:>4} | {vals[0]:>8.4f} {vals[1]:>8.4f} {"NaN":>8} {vals[3]:>8.4f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
