# Evaluation Dataset Audit

Audit date: 2026-03-10
Last in-place data fix: 2026-03-10

This note audits the local target-week-centric evaluation datasets currently present in `Infer/eval`:

- `model_v3.nc`
- `bom_s2s_target_week.nc`
- `cma_s2s_target_week.nc`
- `ecmwf_s2s_target_week.nc`
- `fuxi_s2s_target_week.nc`
- `gefs_s2s_target_week.nc`
- `ifs_s2s_target_week.nc`
- `ncep_s2s_target_week.nc`

The audit script used for repeat checks is `analysis/NWP_AI/audit_eval_datasets.py`.
The in-place patch script used to modify the local NetCDF files is `analysis/NWP_AI/fix_eval_file_inconsistencies.py`.

## 1. Executive Summary

The 8 datasets still cannot be evaluated together directly.

The main reasons are:

- time windows are different
- variable coverage is different
- lead coverage is different
- grids are different
- one file still lacks observations and some files only contain lead `1-2`

Two physics-level inconsistencies were found in the local files and have now been patched in place:

1. `FuXi-S2S pred_tp_*` was about `3600x` too large relative to `obs_tp`
   - fix applied directly to `fuxi_s2s_target_week.nc`
   - operation: `pred_tp_lead1-6 = pred_tp_lead1-6 / 3600`
   - after patch, `pred_tp_*` is back on the same scale as `obs_tp`

2. `GEFS pred_olr_*` had the opposite sign of `obs_olr`
   - fix applied directly to `gefs_s2s_target_week.nc`
   - operation: `pred_olr_lead1-2 = -pred_olr_lead1-2`
   - after patch, `pred_olr_*` matches the sign convention used by the stored `obs_olr`

The remaining major blocker is coverage, not unit mismatch:

- `IFS` now has `obs_tp/obs_t2m/obs_z500/obs_u850/obs_u200`, but only for `2022-02-05 -> 2023-12-24`
- the local weekly ERA5 zarr currently ends at `2023-12-24`, so `2024-01-01 -> 2024-12-23` remains `NaN`
- `IFS` still has no time overlap with the 2017-2021 model group, so it is still excluded from the current multi-model comparison sets

## 2. Dataset Matrix

| Dataset | File | Vars present | Leads present | Time range | Target weeks | Grid |
|---|---|---|---|---|---:|---|
| CAS-Canglong | `model_v3.nc` | `tp,t2m,olr,z500,u850,u200` | `1-6` | `2017-01-01 -> 2021-12-24` | 260 | `721x1440`, regular `0.25 deg` |
| BOM | `bom_s2s_target_week.nc` | `tp,t2m` | `1-6` | `2017-02-12 -> 2020-09-30` | 190 | `72x144`, lon regular `2.5 deg`, lat non-uniform |
| CMA | `cma_s2s_target_week.nc` | `tp,t2m` | `1-6` | `2017-02-12 -> 2021-12-24` | 254 | `121x240`, regular `1.5 deg` |
| ECMWF | `ecmwf_s2s_target_week.nc` | `tp,t2m` | `1-6` | `2017-02-12 -> 2021-12-24` | 254 | `121x240`, regular `1.5 deg` |
| FuXi-S2S | `fuxi_s2s_target_week.nc` | `tp,t2m,olr,z500,u850,u200` | `1-6` | `2017-02-12 -> 2021-12-24` | 254 | `121x240`, regular `1.5 deg` |
| GEFS | `gefs_s2s_target_week.nc` | `tp,t2m,olr,z500,u850,u200` | `1-2` | `2017-01-08 -> 2019-12-24` | 155 | `361x720`, regular `0.5 deg` |
| IFS | `ifs_s2s_target_week.nc` | `tp,t2m,z500,u850,u200` | `1-2` | `2022-02-05 -> 2024-12-23` | 151 | `721x1440`, regular `0.25 deg`, obs valid only through `2023-12-24` |
| NCEP | `ncep_s2s_target_week.nc` | `tp,t2m` | `1-6` | `2019-02-12 -> 2021-01-01` | 99 | `121x240`, regular `1.5 deg` |

## 3. Coordinate and Orientation Audit

### 3.1 What is consistent

All 8 files use:

- latitude descending from north to south
- longitude ascending
- longitude convention `0E -> 360E`, not `-180 -> 180`

This means there is no east-west reversal problem in the current local files.

### 3.2 What is not consistent

The spatial grids are not unified.

- CAS-Canglong and IFS use the same `0.25 deg` global grid
- CMA, ECMWF, FuXi-S2S, and NCEP use the same `1.5 deg` global grid
- GEFS uses a regular `0.5 deg` global grid
- BOM is not a regular `2.5 x 2.5` lat-lon grid

Important BOM detail:

- BOM longitude is regular `0.0 -> 357.5` with `2.5 deg` spacing
- BOM latitude is `88.0995 -> -88.0995` with non-uniform spacing
- therefore BOM should not be handled as a simple regular lat grid

### 3.3 Observation consistency on the shared 1.5-degree grid

For the overlapping period and the common `121x240` grid:

- `CMA`, `ECMWF`, `FuXi-S2S`, and `NCEP` have exactly identical `obs_tp`
- `CMA`, `ECMWF`, `FuXi-S2S`, and `NCEP` have exactly identical `obs_t2m`

This is good news: within the shared `1.5 deg` group, the stored observation fields are already consistent for `tp` and `t2m`.

## 4. Time-Axis Audit

### 4.1 These files are not uniformly 7 days apart across year boundaries

That is expected and consistent with the current project rule:

- 52 weeks per year
- no cross-year weeks
- leftover days at the end of the year are discarded

So the local files show:

- normal within-year spacing: `7 days`
- year-end jumps: `8 days` or `9 days`

Example from `model_v3.nc`:

- `2017-12-24 -> 2018-01-01` is `8 days`
- `2020-12-23 -> 2021-01-01` is `9 days`

This is not a bug. It is the expected natural-week convention used in this repository.

### 4.2 Practical consequence

Do not assume:

```text
next_target_week_time = current_time + 7 days
```

Instead use:

- the actual `time` coordinate
- `year`
- `woy`

## 5. Variable Coverage Audit

### 5.1 Full 6-variable datasets

- CAS-Canglong: `tp,t2m,olr,z500,u850,u200`
- FuXi-S2S: `tp,t2m,olr,z500,u850,u200`
- GEFS: `tp,t2m,olr,z500,u850,u200`, but only `lead1-2`

### 5.2 Temperature and precipitation only

- BOM: `tp,t2m`
- CMA: `tp,t2m`
- ECMWF: `tp,t2m`
- NCEP: `tp,t2m`

### 5.3 IFS

- IFS contains `tp,t2m,z500,u850,u200`
- IFS omits `olr`
- IFS has only `lead1-2`
- IFS now contains `obs_tp, obs_t2m, obs_z500, obs_u850, obs_u200`
- those obs fields are valid only for `2022-02-05 -> 2023-12-24`
- `2024-01-01 -> 2024-12-23` remains `NaN` because the local ERA5 weekly zarr ends at `2023-12-24`

## 6. Unit Audit

### 6.1 Units that are already consistent across most files

For the local files, the declared units are mostly consistent:

- `tp`: `kg/m2/s`
- `t2m`: `K`
- `olr`: `W/m2`
- `z500`: `m2/s2`
- `u850`: `m/s`
- `u200`: `m/s`

The following datasets look physically consistent for `tp` scale:

- CAS-Canglong
- BOM
- CMA
- ECMWF
- GEFS
- IFS
- NCEP

### 6.2 FuXi precipitation scale mismatch: fixed in place

Before the patch:

- `obs_tp` sample mean is about `2.34e-05 kg/m2/s`
- `pred_tp_lead1` sample mean is about `8.58e-02`

This is about `3600x` too large.

Patch applied:

- `pred_tp_lead1-6 = pred_tp_lead1-6 / 3600`

After the patch:

- `pred_tp_lead1` sample mean is about `2.38e-05 kg/m2/s`
- `obs_tp` sample mean is about `2.34e-05 kg/m2/s`

So this issue is no longer a current blocker for evaluation.

### 6.3 GEFS OLR sign mismatch: fixed in place

Before the patch:

- `obs_olr` is negative
- `pred_olr_lead1` is positive

Patch applied:

- `pred_olr_lead1-2 = -pred_olr_lead1-2`

After the patch:

- `pred_olr_lead1` is negative
- `pred_olr_lead1` now matches the sign and scale of `obs_olr`

So this issue is no longer a current blocker for evaluation.

### 6.4 Important semantic note for `olr`

In CAS-Canglong and FuXi local files, the stored OLR-like field behaves like ERA5 top net longwave flux:

- sample means are around `-225 W/m2`
- values are negative

So if the downstream analysis expects positive outgoing longwave radiation, the sign convention must be standardized first.

### 6.5 Important semantic note for `z500`

`z500` is stored as geopotential in `m2/s2`, not geopotential height in `gpm`.

This is okay for comparison if all files keep the same convention.

But if any plotting or metric code expects `gpm`, convert before use:

```text
z500_gpm = z500 / 9.80665
```

## 7. Other Data-Level Caveats

### 7.1 CAS precipitation can be slightly negative

From the local sample:

- `CAS pred_tp_lead1` minimum is about `-6.7e-06 kg/m2/s`

That is a model-output property, not necessarily a file-format error.

Recommendation:

- keep raw values for raw skill audit
- clip to `0` only if producing physically constrained precipitation products or accumulations

### 7.2 Auxiliary fields are not stored as xarray coordinates

In the current files:

- `year`
- `woy`
- `global_idx`

are stored as data variables, not formal coordinates.

This does not block evaluation, but it means xarray will not automatically align datasets on those fields as coordinates.

## 8. What Can Be Compared Right Now

### 8.1 Six models, `tp+t2m`, leads `1-6`

Feasible set:

- CAS-Canglong
- BOM
- CMA
- ECMWF
- FuXi-S2S
- NCEP

Common target-week overlap:

- `2019-02-12 -> 2020-09-30`
- `86` target weeks

Required preprocessing before running:

- regrid all models to one common grid

### 8.2 Two models, full 6 variables, leads `1-6`

Feasible set:

- CAS-Canglong
- FuXi-S2S

Common target-week overlap:

- `2017-02-12 -> 2021-12-24`
- `254` target weeks

Required preprocessing before running:

- choose one common grid: `0.25 deg` or `1.5 deg`

### 8.3 Three models, full 6 variables, leads `1-2`

Feasible set:

- CAS-Canglong
- FuXi-S2S
- GEFS

Common target-week overlap:

- `2017-02-12 -> 2019-12-24`
- `150` target weeks

Required preprocessing before running:

- regrid to a common grid

### 8.4 IFS

IFS is now directly evaluable as a standalone model-vs-ERA5 file for the covered period, but it is still not usable in the current multi-model comparison groups.

Reasons:

- obs exist only for `2022-02-05 -> 2023-12-24`
- time range is still `2022-02-05 -> 2024-12-23`
- there is still no overlap with the current 2017-2021 local comparison files

To evaluate all 151 IFS target weeks, we still need:

- a weekly ERA5 observation source extended through `2024-12-23`

To compare IFS against the current 2017-2021 model archive, we would additionally need:

- another model archive that also covers `2022-2024`

## 9. Recommended Pre-Evaluation Checklist

Before any formal multi-model score table, do these steps first:

1. Confirm the local patch state
   - FuXi `pred_tp_*` should already be divided by `3600`
   - GEFS `pred_olr_*` should already be sign-corrected
   - rerun `analysis/NWP_AI/audit_eval_datasets.py` before scoring if any file is replaced

2. Freeze one shared physical convention
   - decide whether `olr` should be stored as negative net longwave flux or positive outgoing flux
   - decide whether `z500` stays in `m2/s2` or is converted to `gpm`

3. Freeze one comparison subset
   - not all 8 files can be compared in one table
   - choose by variable set, lead set, and time overlap first

4. Freeze one common grid
   - BOM cannot be treated as regular `2.5 x 2.5`
   - `1.5 deg` is the easiest shared grid for `CMA/ECMWF/FuXi/NCEP`
   - `0.25 deg` is shared by `CAS-Canglong/IFS`

5. Use the real target-week axis
   - do not assume fixed `7-day` spacing at year boundaries

6. Only then run evaluation metrics
   - RMSE
   - ACC
   - TCC
   - regional diagnostics

## 10. Bottom Line

The local evaluation archive is already in a useful target-week-centric shape, but it is not yet plug-and-play across all models.

The minimum mandatory constraints before multi-model evaluation are now:

- explicit choice of common time window
- explicit choice of common grid
- explicit choice of common variable subset and lead subset
- excluding IFS from the current shared comparison sets, unless a `2022-2024` multi-model overlap set is built
