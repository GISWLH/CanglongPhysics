"""
CAS-Canglong SST16 ENSO Forecast Inference
==========================================
GPU inference only: save `ENSO_pre_temp.csv` with 10 ensemble members.

Usage:
    conda activate torch
    PYTHONPATH=/home/lhwang/Desktop/CanglongPhysics \
    python analysis/operation/SSTmodel/run_enso_forecast.py

Then plot from two CSVs with:
    python analysis/operation/SSTmodel/plot_enso_forecast.py
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import xarray as xr
from dateutil.relativedelta import relativedelta


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(ROOT_DIR))

from canglong import CanglongSST16
from canglong.model_v1 import UpSample, DownSample, EarthAttention3D, EarthSpecificBlock, BasicLayer, Mlp


Canglong = CanglongSST16

ENSO_OBS_CSV = SCRIPT_DIR / 'ENSO_all.csv'
ENSO_PRE_CSV = SCRIPT_DIR / 'ENSO_pre_temp.csv'
FORECAST_START = '202603'
MONTHLY_MAP_NC = SCRIPT_DIR / f'sst_forecast_monthly_{FORECAST_START}.nc'
SEASONAL_MAP_NC = SCRIPT_DIR / f'sst_forecast_seasonal_{FORECAST_START}.nc'
FC_DATE = datetime.strptime(FORECAST_START, '%Y%m')
INPUT_START = FC_DATE - relativedelta(months=16)
INPUT_END = FC_DATE - relativedelta(months=1)
PRED_MONTHS = 12
MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
NINO34_LAT = slice(340, 380)
NINO34_LON = slice(760, 960)

N_ENSEMBLE = 10
GLOBAL_PERTURB_STD_NORM = 0.16
ENSO_PERTURB_STD_NORM = 0.16
ENSO_MEMBER_SST_BIAS_STD_NORM = 0.28
ENSO_MEMBER_WIND_BIAS_STD_NORM = 0.08
ENSO_BIAS_STD_NORM = 0.04
GLOBAL_PERTURB_GRID = (60, 120)
ENSO_PERTURB_GRID = (36, 72)
PERTURB_SEED = 202603
PERTURB_LAST_N_MONTHS = 12
MONTH_PERTURB_WEIGHTS = [0.35, 0.42, 0.50, 0.58, 0.66, 0.74, 0.82, 0.88, 0.92, 0.96, 0.98, 1.00]
GLOBAL_CHANNEL_WEIGHTS = torch.tensor([1.2, 0.5, 0.4, 0.7, 0.7, 0.3, 0.3, 0.4], dtype=torch.float32).view(8, 1, 1)
ENSO_CHANNEL_WEIGHTS = torch.tensor([1.2, 0.1, 0.1, 0.7, 0.7, 0.0, 0.0, 0.1], dtype=torch.float32).view(8, 1, 1)
SPREAD_SCALE_BY_LEAD = np.array([2.2, 2.0, 1.8, 1.6, 1.35, 1.15, 1.00, 0.88, 0.75, 0.62, 0.50, 0.42], dtype=np.float32)

FORECAST_DATES = [FC_DATE + relativedelta(months=lead) for lead in range(PRED_MONTHS)]
FORECAST_MONTH_LABELS = [f'{current_date:%Y-%m}' for current_date in FORECAST_DATES]
SEASON_ORDER = ['MAM', 'JJA', 'SON', 'DJ']
SEASON_DEFS = {
    'MAM': ['2026-03', '2026-04', '2026-05'],
    'JJA': ['2026-06', '2026-07', '2026-08'],
    'SON': ['2026-09', '2026-10', '2026-11'],
    'DJ': ['2026-12', '2027-01'],
}

monthly_avg_temp = np.load('/data/lhwang/monthly_avg_temp.npy')
ocean_mask = np.load('/data/lhwang/ocean_mask.npy')

mean_all = torch.tensor([[[[2.8679e+02]],
                          [[1.0096e+05]],
                          [[-5.3626e+06]],
                          [[-5.1725e-02]],
                          [[1.8698e-01]],
                          [[5.4089e+04]],
                          [[1.3745e+04]],
                          [[1.1180e+07]]]])
std_all = torch.tensor([[[[1.1627e+01]],
                         [[1.0610e+03]],
                         [[4.9920e+06]],
                         [[3.8811e+00]],
                         [[2.4887e+00]],
                         [[3.2341e+03]],
                         [[1.3297e+03]],
                         [[7.8841e+06]]]])

fc_tag = f'{INPUT_START:%Y%m}-{INPUT_END:%Y%m}'
surf_file = Path('/data/lhwang/SST') / f'ERA5_surface_monthly_{fc_tag}.nc'
pres_file = Path('/data/lhwang/SST') / f'ERA5_pressure_monthly_{fc_tag}.nc'

print(f'Forecast start : {FC_DATE:%Y-%m}')
print(f'Input window   : {INPUT_START:%Y-%m} ~ {INPUT_END:%Y-%m} (16 months)')
print(f'Ensemble size  : {N_ENSEMBLE}')
print(
    f'Perturbation   : global={GLOBAL_PERTURB_STD_NORM:.2f}, '
    f'enso_noise={ENSO_PERTURB_STD_NORM:.2f}, '
    f'enso_sst_bias={ENSO_MEMBER_SST_BIAS_STD_NORM:.2f}, '
    f'enso_wind_bias={ENSO_MEMBER_WIND_BIAS_STD_NORM:.2f}, '
    f'months={PERTURB_LAST_N_MONTHS}, seed={PERTURB_SEED}'
)

ds_surf = xr.open_dataset(surf_file)
ds_pres = xr.open_dataset(pres_file)
plevs = ds_pres['pressure_level'].values
print(f'Surface vars={list(ds_surf.data_vars)}, Pressure levels={plevs}')

lat_name = 'latitude' if 'latitude' in ds_surf.coords else 'lat'
lon_name = 'longitude' if 'longitude' in ds_surf.coords else 'lon'
lat_values = ds_surf[lat_name].values.astype(np.float32)
lon_values = ds_surf[lon_name].values.astype(np.float32)

sst_data = ds_surf['sst'].values
msl_data = ds_surf['msl'].values
slhf_data = ds_surf['slhf'].values
u10_data = ds_surf['u10'].values
v10_data = ds_surf['v10'].values
ssr_data = ds_surf['ssr'].values

z_all = ds_pres['z'].values
idx_500 = int(np.where(plevs == 500)[0][0])
idx_850 = int(np.where(plevs == 850)[0][0])
z500_data = z_all[:, idx_500, :, :]
z850_data = z_all[:, idx_850, :, :]

combined = np.stack([
    sst_data,
    msl_data,
    slhf_data,
    u10_data,
    v10_data,
    z500_data,
    z850_data,
    ssr_data,
], axis=1)
input_tensor = torch.from_numpy(combined).float()
normalized_input = torch.nan_to_num((input_tensor - mean_all) / std_all, nan=0.0)
print(f'Input tensor: {normalized_input.shape}')

ds_surf.close()
ds_pres.close()

height, width = normalized_input.shape[2:]
lat_idx = torch.arange(height, dtype=torch.float32).view(height, 1)
lon_idx = torch.arange(width, dtype=torch.float32).view(1, width)
enso_lat_mask = torch.exp(-0.5 * ((lat_idx - 360.0) / 48.0) ** 2)
enso_lon_mask = torch.exp(-0.5 * ((lon_idx - 860.0) / 140.0) ** 2)
ENSO_MASK = (enso_lat_mask * enso_lon_mask).view(1, height, width)

print('\n--- SSTA Offset Calibration ---')
enso_csv = pd.read_csv(ENSO_OBS_CSV)
clim_rows = enso_csv[(enso_csv['Year'] >= 1991) & (enso_csv['Year'] <= 2020)]
enso_clim = {}
for month_index, month_name in enumerate(MONTH_NAMES, 1):
    enso_clim[month_index] = clim_rows[month_name].astype(float).mean()

model_clim_nino34 = {}
for month_index in range(12):
    model_clim_nino34[month_index] = float(
        np.nanmean(monthly_avg_temp[month_index, NINO34_LAT, NINO34_LON]) - 273.15
    )

offsets = []
for offset_index in range(16):
    current_date = INPUT_START + relativedelta(months=offset_index)
    month_value = current_date.month
    era5_sst_c = float(np.nanmean(sst_data[offset_index, NINO34_LAT, NINO34_LON])) - 273.15
    model_ssta = era5_sst_c - model_clim_nino34[month_value - 1]
    row = enso_csv[enso_csv['Year'] == current_date.year]
    csv_val = float(row[MONTH_NAMES[month_value - 1]].values[0]) if len(row) > 0 else np.nan
    csv_ssta = (csv_val - enso_clim[month_value]) if not np.isnan(csv_val) else np.nan
    diff = csv_ssta - model_ssta if not np.isnan(csv_ssta) else np.nan
    if not np.isnan(diff):
        offsets.append(diff)

fixed_offset = float(np.mean(offsets))
print(f'Climatology offset = {fixed_offset:+.4f} C  (N={len(offsets)})')

row0 = enso_csv[enso_csv['Year'] == FC_DATE.year]
_anchor_val = row0[MONTH_NAMES[FC_DATE.month - 1]].values[0]
if _anchor_val == '' or (isinstance(_anchor_val, float) and np.isnan(_anchor_val)):
    _fallback_date = FC_DATE - relativedelta(months=1)
    _fallback_row = enso_csv[enso_csv['Year'] == _fallback_date.year]
    obs_init_val = float(_fallback_row[MONTH_NAMES[_fallback_date.month - 1]].values[0])
    _anchor_month = _fallback_date.month
    print(f'[anchor] {MONTH_NAMES[FC_DATE.month - 1]} {FC_DATE.year} not available, '
          f'falling back to {MONTH_NAMES[_fallback_date.month - 1]} {_fallback_date.year} = {obs_init_val}')
else:
    obs_init_val = float(_anchor_val)
    _anchor_month = FC_DATE.month
obs_init_ssta = obs_init_val - enso_clim[_anchor_month]


def make_global_noise(generator, n_channels, target_hw):
    coarse_noise = torch.randn((1, n_channels, GLOBAL_PERTURB_GRID[0], GLOBAL_PERTURB_GRID[1]), generator=generator)
    smooth_noise = F.interpolate(coarse_noise, size=target_hw, mode='bicubic', align_corners=False)
    return smooth_noise.squeeze(0)


def make_enso_noise(generator, n_channels, target_hw):
    coarse_noise = torch.randn((1, n_channels, ENSO_PERTURB_GRID[0], ENSO_PERTURB_GRID[1]), generator=generator)
    smooth_noise = F.interpolate(coarse_noise, size=target_hw, mode='bicubic', align_corners=False)
    return smooth_noise.squeeze(0)


def draw_member_phase(member_index):
    generator = torch.Generator()
    generator.manual_seed(PERTURB_SEED + member_index * 1000 + 17)
    phase = torch.randn((1,), generator=generator).item()
    return float(np.clip(phase, -1.5, 1.5))


def weighted_mean_months(field_stack, month_labels, target_months):
    index_lookup = {month_label: idx for idx, month_label in enumerate(month_labels)}
    missing = [month_label for month_label in target_months if month_label not in index_lookup]
    if missing:
        raise KeyError(f'Missing months for weighted mean: {missing}')

    target_indices = [index_lookup[month_label] for month_label in target_months]
    weights = np.array(
        [pd.Timestamp(f'{month_label}-01').days_in_month for month_label in target_months],
        dtype=np.float32,
    )
    selected = field_stack[target_indices].astype(np.float32)
    weighted = np.tensordot(weights, selected, axes=(0, 0)) / float(weights.sum())
    return weighted.astype(np.float32)


def build_monthly_dataset(monthly_sst_mean, monthly_sst_std, monthly_ssta_mean, monthly_ssta_std):
    valid_times = np.array(FORECAST_DATES, dtype='datetime64[ns]')
    ds = xr.Dataset(
        data_vars={
            'sst': (('month', 'lat', 'lon'), monthly_sst_mean.astype(np.float32)),
            'sst_std': (('month', 'lat', 'lon'), monthly_sst_std.astype(np.float32)),
            'ssta': (('month', 'lat', 'lon'), monthly_ssta_mean.astype(np.float32)),
            'ssta_std': (('month', 'lat', 'lon'), monthly_ssta_std.astype(np.float32)),
        },
        coords={
            'month': np.asarray(FORECAST_MONTH_LABELS, dtype=object),
            'valid_time': ('month', valid_times),
            'lat': lat_values,
            'lon': lon_values,
        },
        attrs={
            'forecast_start': f'{FC_DATE:%Y-%m}',
            'input_window': f'{INPUT_START:%Y-%m} to {INPUT_END:%Y-%m}',
            'ensemble_size': int(N_ENSEMBLE),
            'model_name': 'CAS-Canglong SST16',
            'anomaly_reference': '/data/lhwang/monthly_avg_temp.npy',
            'land_mask_reference': '/data/lhwang/ocean_mask.npy (True over land)',
            'nino34_scalar_bias_correction_applied_to_maps': 'false',
            'description': 'Monthly ensemble-mean SST and SSTA forecast fields',
        },
    )
    ds['sst'].attrs.update({'units': 'K', 'long_name': 'Sea surface temperature ensemble mean'})
    ds['sst_std'].attrs.update({'units': 'K', 'long_name': 'Sea surface temperature ensemble spread'})
    ds['ssta'].attrs.update({'units': 'degC', 'long_name': 'Sea surface temperature anomaly ensemble mean'})
    ds['ssta_std'].attrs.update({'units': 'degC', 'long_name': 'Sea surface temperature anomaly ensemble spread'})
    return ds


def build_seasonal_dataset(seasonal_sst_mean, seasonal_sst_std, seasonal_ssta_mean, seasonal_ssta_std):
    season_start = np.array([SEASON_DEFS[label][0] for label in SEASON_ORDER], dtype=object)
    season_end = np.array([SEASON_DEFS[label][-1] for label in SEASON_ORDER], dtype=object)
    ds = xr.Dataset(
        data_vars={
            'sst': (('season', 'lat', 'lon'), seasonal_sst_mean.astype(np.float32)),
            'sst_std': (('season', 'lat', 'lon'), seasonal_sst_std.astype(np.float32)),
            'ssta': (('season', 'lat', 'lon'), seasonal_ssta_mean.astype(np.float32)),
            'ssta_std': (('season', 'lat', 'lon'), seasonal_ssta_std.astype(np.float32)),
        },
        coords={
            'season': np.asarray(SEASON_ORDER, dtype=object),
            'season_start': ('season', season_start),
            'season_end': ('season', season_end),
            'lat': lat_values,
            'lon': lon_values,
        },
        attrs={
            'forecast_start': f'{FC_DATE:%Y-%m}',
            'ensemble_size': int(N_ENSEMBLE),
            'model_name': 'CAS-Canglong SST16',
            'anomaly_reference': '/data/lhwang/monthly_avg_temp.npy',
            'land_mask_reference': '/data/lhwang/ocean_mask.npy (True over land)',
            'description': 'Seasonal ensemble-mean SST and SSTA forecast fields (weighted by days in month)',
            'season_months_MAM': ','.join(SEASON_DEFS['MAM']),
            'season_months_JJA': ','.join(SEASON_DEFS['JJA']),
            'season_months_SON': ','.join(SEASON_DEFS['SON']),
            'season_months_DJ': ','.join(SEASON_DEFS['DJ']),
        },
    )
    ds['sst'].attrs.update({'units': 'K', 'long_name': 'Seasonal sea surface temperature ensemble mean'})
    ds['sst_std'].attrs.update({'units': 'K', 'long_name': 'Seasonal sea surface temperature ensemble spread'})
    ds['ssta'].attrs.update({'units': 'degC', 'long_name': 'Seasonal sea surface temperature anomaly ensemble mean'})
    ds['ssta_std'].attrs.update({'units': 'degC', 'long_name': 'Seasonal sea surface temperature anomaly ensemble spread'})
    return ds


def make_member_perturbation(member_index, local_index, n_channels, target_hw, member_phase):
    generator = torch.Generator()
    generator.manual_seed(PERTURB_SEED + member_index * 100 + local_index)

    global_noise = make_global_noise(generator, n_channels, target_hw)
    enso_noise = make_enso_noise(generator, n_channels, target_hw)
    enso_bias = torch.randn((n_channels, 1, 1), generator=generator)

    perturbation = GLOBAL_PERTURB_STD_NORM * global_noise * GLOBAL_CHANNEL_WEIGHTS
    perturbation = perturbation + ENSO_PERTURB_STD_NORM * enso_noise * ENSO_CHANNEL_WEIGHTS * ENSO_MASK
    perturbation = perturbation + ENSO_BIAS_STD_NORM * enso_bias * ENSO_CHANNEL_WEIGHTS * ENSO_MASK

    phase_mask = ENSO_MASK.squeeze(0)
    perturbation[0] = perturbation[0] + ENSO_MEMBER_SST_BIAS_STD_NORM * member_phase * phase_mask
    perturbation[3] = perturbation[3] + ENSO_MEMBER_WIND_BIAS_STD_NORM * member_phase * phase_mask
    perturbation[4] = perturbation[4] - 0.5 * ENSO_MEMBER_WIND_BIAS_STD_NORM * member_phase * phase_mask
    return perturbation


def run_one_member(member_index):
    member_input = normalized_input.clone()
    n_months = min(PERTURB_LAST_N_MONTHS, member_input.shape[0])
    month_weights = MONTH_PERTURB_WEIGHTS[-n_months:]
    member_phase = draw_member_phase(member_index)

    for local_index in range(n_months):
        month_slot = member_input.shape[0] - n_months + local_index
        perturbation = make_member_perturbation(
            member_index=member_index,
            local_index=local_index,
            n_channels=member_input.shape[1],
            target_hw=member_input.shape[2:],
            member_phase=member_phase,
        )
        member_input[month_slot] = member_input[month_slot] + month_weights[local_index] * perturbation

    batch = member_input.permute(1, 0, 2, 3).unsqueeze(0)

    with torch.no_grad():
        output = the_model(batch.to(device))
        output = output.squeeze() * std_all[0, 0, 0, 0].item() + mean_all[0, 0, 0, 0].item()
        output = output.cpu().numpy().astype(np.float32)

    pred_month_idx = np.array([(FC_DATE.month - 1 + lead) % 12 for lead in range(16)])
    pred_ssta_map = output - monthly_avg_temp[pred_month_idx, :, :]
    pred_ssta_map[:, ocean_mask] = np.nan
    pred_nino34_raw = np.nanmean(pred_ssta_map[:, NINO34_LAT, NINO34_LON], axis=(1, 2))
    return output, pred_ssta_map.astype(np.float32), pred_nino34_raw, member_phase


device = 'cuda:0'
print(f'\nLoading model on {device} ...')
the_model = torch.load(
    '/home/lhwang/Desktop/weather/model/canglong16_0005_600ep_base.pth',
    weights_only=False,
)
the_model = the_model.module
the_model.to(device)
the_model.eval()

print('\n--- Ensemble inference ---')
member_predictions_raw = []
member_phases = []
monthly_sst_sum = None
monthly_sst_sumsq = None
monthly_ssta_sum = None
monthly_ssta_sumsq = None
seasonal_sst_sum = None
seasonal_sst_sumsq = None
seasonal_ssta_sum = None
seasonal_ssta_sumsq = None

ocean_points = (~ocean_mask)[None, :, :]
for member_index in range(N_ENSEMBLE):
    pred_sst_map, pred_ssta_map, pred_nino34_raw, member_phase = run_one_member(member_index)
    pred_sst_map = pred_sst_map[:PRED_MONTHS].astype(np.float32)
    pred_ssta_map = pred_ssta_map[:PRED_MONTHS].astype(np.float32)
    pred_sst_masked = np.where(ocean_points, pred_sst_map, 0.0)
    pred_ssta_masked = np.where(ocean_points, pred_ssta_map, 0.0)

    if monthly_sst_sum is None:
        monthly_shape = pred_sst_map.shape
        seasonal_shape = (len(SEASON_ORDER), monthly_shape[1], monthly_shape[2])
        monthly_sst_sum = np.zeros(monthly_shape, dtype=np.float32)
        monthly_sst_sumsq = np.zeros(monthly_shape, dtype=np.float32)
        monthly_ssta_sum = np.zeros(monthly_shape, dtype=np.float32)
        monthly_ssta_sumsq = np.zeros(monthly_shape, dtype=np.float32)
        seasonal_sst_sum = np.zeros(seasonal_shape, dtype=np.float32)
        seasonal_sst_sumsq = np.zeros(seasonal_shape, dtype=np.float32)
        seasonal_ssta_sum = np.zeros(seasonal_shape, dtype=np.float32)
        seasonal_ssta_sumsq = np.zeros(seasonal_shape, dtype=np.float32)

    monthly_sst_sum += pred_sst_masked
    monthly_sst_sumsq += pred_sst_masked * pred_sst_masked
    monthly_ssta_sum += pred_ssta_masked
    monthly_ssta_sumsq += pred_ssta_masked * pred_ssta_masked

    seasonal_sst_member = np.stack(
        [weighted_mean_months(pred_sst_map, FORECAST_MONTH_LABELS, SEASON_DEFS[label]) for label in SEASON_ORDER],
        axis=0,
    )
    seasonal_ssta_member = np.stack(
        [weighted_mean_months(pred_ssta_map, FORECAST_MONTH_LABELS, SEASON_DEFS[label]) for label in SEASON_ORDER],
        axis=0,
    )
    seasonal_sst_member = np.where((~ocean_mask)[None, :, :], seasonal_sst_member, 0.0)
    seasonal_ssta_member = np.where((~ocean_mask)[None, :, :], seasonal_ssta_member, 0.0)

    seasonal_sst_sum += seasonal_sst_member
    seasonal_sst_sumsq += seasonal_sst_member * seasonal_sst_member
    seasonal_ssta_sum += seasonal_ssta_member
    seasonal_ssta_sumsq += seasonal_ssta_member * seasonal_ssta_member

    member_predictions_raw.append(pred_nino34_raw[:PRED_MONTHS])
    member_phases.append(member_phase)
    print(
        f'Member {member_index + 1:02d}: '
        f'phase={member_phase:+.2f}, '
        f'raw lead0={pred_nino34_raw[0]:+.3f} C, '
        f'raw lead12={pred_nino34_raw[PRED_MONTHS - 1]:+.3f} C'
    )

member_predictions_raw = np.stack(member_predictions_raw, axis=0)
member_predictions_clim = member_predictions_raw + fixed_offset
common_anchor_correction = obs_init_ssta - np.mean(member_predictions_clim[:, 0])
member_predictions = member_predictions_clim + common_anchor_correction

raw_ensemble_mean = np.mean(member_predictions, axis=0)
raw_ensemble_std = np.std(member_predictions, axis=0)
centered_members = member_predictions - raw_ensemble_mean[None, :]
member_predictions = raw_ensemble_mean[None, :] + centered_members * SPREAD_SCALE_BY_LEAD[None, :]

ensemble_mean = np.mean(member_predictions, axis=0)
ensemble_std = np.std(member_predictions, axis=0)

monthly_sst_mean = monthly_sst_sum / float(N_ENSEMBLE)
monthly_sst_std = np.sqrt(np.maximum(monthly_sst_sumsq / float(N_ENSEMBLE) - monthly_sst_mean ** 2, 0.0))
monthly_ssta_mean = monthly_ssta_sum / float(N_ENSEMBLE)
monthly_ssta_std = np.sqrt(np.maximum(monthly_ssta_sumsq / float(N_ENSEMBLE) - monthly_ssta_mean ** 2, 0.0))
monthly_sst_mean[:, ocean_mask] = np.nan
monthly_sst_std[:, ocean_mask] = np.nan
monthly_ssta_mean[:, ocean_mask] = np.nan
monthly_ssta_std[:, ocean_mask] = np.nan

seasonal_sst_mean = seasonal_sst_sum / float(N_ENSEMBLE)
seasonal_sst_std = np.sqrt(np.maximum(seasonal_sst_sumsq / float(N_ENSEMBLE) - seasonal_sst_mean ** 2, 0.0))
seasonal_ssta_mean = seasonal_ssta_sum / float(N_ENSEMBLE)
seasonal_ssta_std = np.sqrt(np.maximum(seasonal_ssta_sumsq / float(N_ENSEMBLE) - seasonal_ssta_mean ** 2, 0.0))
seasonal_sst_mean[:, ocean_mask] = np.nan
seasonal_sst_std[:, ocean_mask] = np.nan
seasonal_ssta_mean[:, ocean_mask] = np.nan
seasonal_ssta_std[:, ocean_mask] = np.nan

print(f'\nClimatology offset : {fixed_offset:+.4f} C')
print(f'Anchor correction  : {common_anchor_correction:+.4f} C (shared by all members)')
print(f'Spread shaping     : lead1 x{SPREAD_SCALE_BY_LEAD[0]:.2f}, lead12 x{SPREAD_SCALE_BY_LEAD[PRED_MONTHS - 1]:.2f}')
print(f'Ensemble spread    : lead1 std={raw_ensemble_std[0]:.4f} -> {ensemble_std[0]:.4f} C, lead12 std={raw_ensemble_std[PRED_MONTHS - 1]:.4f} -> {ensemble_std[PRED_MONTHS - 1]:.4f} C')

pred_records = []
for lead in range(PRED_MONTHS):
    current_date = FC_DATE + relativedelta(months=lead)
    record = {
        'Date': f'{current_date:%Y-%m}',
        'Year': current_date.year,
        'Month': current_date.month,
        'SSTA': round(float(ensemble_mean[lead]), 3),
        'SSTA_STD': round(float(ensemble_std[lead]), 3),
    }
    for member_index in range(N_ENSEMBLE):
        record[f'SSTA_{member_index + 1:02d}'] = round(float(member_predictions[member_index, lead]), 3)
    pred_records.append(record)

df_pred = pd.DataFrame(pred_records)
df_pred.to_csv(ENSO_PRE_CSV, index=False)
print(f'\nPrediction saved: {ENSO_PRE_CSV}')
print(df_pred.to_string(index=False))

monthly_ds = build_monthly_dataset(monthly_sst_mean, monthly_sst_std, monthly_ssta_mean, monthly_ssta_std)
monthly_ds.to_netcdf(MONTHLY_MAP_NC)
print(f'\nMonthly forecast grid saved: {MONTHLY_MAP_NC}')

seasonal_ds = build_seasonal_dataset(seasonal_sst_mean, seasonal_sst_std, seasonal_ssta_mean, seasonal_ssta_std)
seasonal_ds.to_netcdf(SEASONAL_MAP_NC)
print(f'Seasonal forecast grid saved: {SEASONAL_MAP_NC}')

print('\nDone.')
