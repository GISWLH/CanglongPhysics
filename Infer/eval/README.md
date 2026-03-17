# Standardized Evaluation Dataset

## Overview

This directory contains standardized evaluation datasets in **target-week-centric** NetCDF format. The format is designed for offline computation of any evaluation metric (PCC, ACC, RMSE, TCC, etc.) without re-running GPU inference.

The key design principle: **for each target week, store the observation once and all lead-time predictions as separate variables**. This eliminates redundant observation storage and provides a continuous time axis, making it straightforward to compare any forecasting model (AI or NWP) on the same footing.

## File Inventory

| File | Description | Size |
|------|-------------|------|
| `model_v3.nc` | CAS-Canglong V3.5 predictions + ERA5 obs | ~25 GB |
| `model_v0.nc` | CAS-Canglong V0 (Lite) predictions + ERA5 obs | ~13 GB |
| `climatology_2002_2016.nc` | 52-week climatology (2002-2016, 15 years) | small |
| `woy_map.npy` | Zarr global index -> week-of-year mapping | small |
| `tcc_v35.nc` | Pre-computed TCC for V3.5 | small |
| `tcc_v0.nc` | Pre-computed TCC for V0 | small |

## Data Format Specification

### Dimensions

```
time: N  (all continuous target weeks in 2017-2021, ~260)
lat:  721  (0.25 deg, 90N to 90S)
lon:  1440  (0.25 deg, 0 to 359.75E)
```

**No `lead` dimension.** Each lead time is stored as a separate variable.

### Coordinates

| Name | Dims | Type | Description |
|------|------|------|-------------|
| `time` | (time,) | float64 | Days since 1940-01-01 (CF-convention, auto-decoded by xarray) |
| `lat` | (lat,) | float32 | Latitude, 90 to -90 |
| `lon` | (lon,) | float32 | Longitude, 0 to 359.75 |
| `year` | (time,) | int32 | Year of target week |
| `woy` | (time,) | int32 | Week-of-year (0-indexed, for climatology lookup) |
| `global_idx` | (time,) | int32 | Zarr global time index of target week |

### Data Variables

For each evaluation variable `{var}`:

| Variable | Dims | Description |
|----------|------|-------------|
| `obs_{var}` | (time, lat, lon) | ERA5 ground truth at target week |
| `pred_{var}_lead1` | (time, lat, lon) | 1-week-ahead forecast for target week |
| `pred_{var}_lead2` | (time, lat, lon) | 2-week-ahead forecast for target week |
| `pred_{var}_lead3` | (time, lat, lon) | 3-week-ahead forecast for target week |
| `pred_{var}_lead4` | (time, lat, lon) | 4-week-ahead forecast for target week |
| `pred_{var}_lead5` | (time, lat, lon) | 5-week-ahead forecast for target week |
| `pred_{var}_lead6` | (time, lat, lon) | 6-week-ahead forecast for target week |

### Evaluation Variables

| Variable | Full Name | Units | V0 | V3.5 |
|----------|-----------|-------|----|------|
| `tp` | Total precipitation (lsrr + crr) | kg/m2/s | yes | yes |
| `t2m` | 2m temperature | K | yes | yes |
| `olr` | Top net long-wave radiation | W/m2 | pred=NaN | yes |
| `z500` | Geopotential at 500 hPa | m2/s2 | yes | yes |
| `u850` | Zonal wind at 850 hPa | m/s | - | yes |
| `u200` | Zonal wind at 200 hPa | m/s | - | yes |

**Total variables**: V3.5 has 42 (6 obs + 36 pred), V0 has 28 (4 obs + 24 pred).

## Semantic Meaning of `pred_{var}_lead{L}`

For a target week `t`, `pred_{var}_lead{L}` is the L-week-ahead autoregressive forecast:

| Lead | Init week | Obs input | Autoregressive steps | Target |
|------|-----------|-----------|---------------------|--------|
| 1 | t-2 | [t-2, t-1] | 1 step | t |
| 2 | t-3 | [t-3, t-2] | 2 steps | t |
| 3 | t-4 | [t-4, t-3] | 3 steps | t |
| 4 | t-5 | [t-5, t-4] | 4 steps | t |
| 5 | t-6 | [t-6, t-5] | 5 steps | t |
| 6 | t-7 | [t-7, t-6] | 6 steps | t |

### Autoregressive Chain Example

For target week 2017-01-01 (week 0), `pred_tp_lead3` means:

```
Init: use obs [2016-W49, 2016-W50] as input
Step 1: model([W49, W50]) -> pred_W51
Step 2: model([W50, pred_W51]) -> pred_W52
Step 3: model([pred_W51, pred_W52]) -> pred_2017W00  <-- this is pred_tp_lead3
```

Only lead 1 uses pure observations as input. From lead 2 onward, at least one input is a model prediction, so errors accumulate with increasing lead time.

## Climatology

`climatology_2002_2016.nc` provides 52-week climatological means (2002-2016, 15 years):

| Variable | Dims | Description |
|----------|------|-------------|
| `tp_clim` | (week, lat, lon) | Precipitation climatology |
| `t2m_clim` | (week, lat, lon) | 2m temperature climatology |
| `olr_clim` | (week, lat, lon) | OLR climatology |
| `z500_clim` | (week, lat, lon) | Z500 climatology |

`week` is 0-indexed week-of-year (0-51). Use the `woy` coordinate in model files to look up the corresponding climatology.

## Usage Examples

### Load and Inspect

```python
import xarray as xr

ds = xr.open_dataset('Infer/eval/model_v3.nc')
print(ds)
# time is auto-decoded to datetime64 via CF-convention
# ds.time.values -> array(['2017-01-01', '2017-01-08', ...], dtype='datetime64')
```

### Spatial PCC (single sample)

```python
import numpy as np

pred = ds['pred_t2m_lead1'].isel(time=0).values.ravel()
obs = ds['obs_t2m'].isel(time=0).values.ravel()
pcc = np.corrcoef(pred, obs)[0, 1]
```

### Global-mean RMSE by Lead

```python
for lead in range(1, 7):
    pred = ds[f'pred_t2m_lead{lead}'].values
    obs = ds['obs_t2m'].values
    rmse = np.sqrt(np.nanmean((pred - obs) ** 2, axis=(1, 2)))
    print(f'Lead {lead}: t2m RMSE = {rmse.mean():.4f} K')
```

### TCC (temporal correlation at each grid point)

```python
clim = xr.open_dataset('Infer/eval/climatology_2002_2016.nc')

# Build anomalies
woy = ds['woy'].values
obs_anom = ds['obs_t2m'].values.copy()
pred_anom = ds['pred_t2m_lead1'].values.copy()
for i in range(len(woy)):
    obs_anom[i] -= clim['t2m_clim'].values[woy[i]]
    pred_anom[i] -= clim['t2m_clim'].values[woy[i]]

# Pearson correlation along time axis
from scipy.stats import pearsonr
tcc = np.zeros((721, 1440))
for j in range(721):
    for k in range(1440):
        tcc[j, k] = pearsonr(obs_anom[:, j, k], pred_anom[:, j, k])[0]
```

### Cross-Model Comparison

```python
v0 = xr.open_dataset('Infer/eval/model_v0.nc')
v3 = xr.open_dataset('Infer/eval/model_v3.nc')

for lead in range(1, 7):
    for name, d in [('V0', v0), ('V3.5', v3)]:
        pred = d[f'pred_t2m_lead{lead}'].values
        obs = d['obs_t2m'].values
        pccs = [np.corrcoef(pred[i].ravel(), obs[i].ravel())[0, 1]
                for i in range(pred.shape[0])]
        print(f'{name} lead{lead}: t2m PCC = {np.mean(pccs):.4f}')
```

## Adding a New Model

To add a new model (AI or NWP) to this evaluation framework:

1. Create a generation script following the pattern in `Infer/gen_eval_v3.py`
2. Output file: `Infer/eval/model_{name}.nc`
3. Required structure:
   - Same dimensions: `(time, lat, lon)` with identical target weeks
   - Same coordinate variables: `time`, `lat`, `lon`, `year`, `woy`, `global_idx`
   - Same observation variables: `obs_{var}` (should be identical across all model files)
   - Prediction variables: `pred_{var}_lead{1..6}` with the model's forecasts
4. Evaluation variables must use the same physical units as listed above

This format is model-agnostic. Any forecasting system that produces 1-6 week ahead predictions on the 0.25-degree global grid can be stored and compared using the same tooling.

## Generation Scripts

```bash
cd /home/lhwang/Desktop/CanglongPhysics

# Climatology (CPU, no GPU needed)
/home/lhwang/anaconda3/envs/torch/bin/python Infer/compute_climatology.py

# V3.5 eval dataset (GPU)
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/lhwang/anaconda3/envs/torch/bin/python Infer/gen_eval_v3.py

# V0 eval dataset (GPU, must run from canglong/ directory)
cd /home/lhwang/Desktop/CanglongPhysics/canglong
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/lhwang/anaconda3/envs/torch/bin/python ../Infer/gen_eval_v0.py
```
