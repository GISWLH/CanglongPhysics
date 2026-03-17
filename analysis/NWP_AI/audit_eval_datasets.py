from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "Infer" / "eval"

DATASETS = {
    "CAS-Canglong": "model_v3.nc",
    "BOM": "bom_s2s_target_week.nc",
    "CMA": "cma_s2s_target_week.nc",
    "ECMWF": "ecmwf_s2s_target_week.nc",
    "FuXi-S2S": "fuxi_s2s_target_week.nc",
    "GEFS": "gefs_s2s_target_week.nc",
    "IFS": "ifs_s2s_target_week.nc",
    "NCEP": "ncep_s2s_target_week.nc",
}

BASE_VARS = ["tp", "t2m", "olr", "z500", "u850", "u200"]


def dataset_var_summary(ds: xr.Dataset) -> tuple[list[str], dict[str, list[int]]]:
    obs_vars = sorted(v[4:] for v in ds.data_vars if v.startswith("obs_"))
    pred_leads: dict[str, list[int]] = {}
    for v in ds.data_vars:
        if not v.startswith("pred_"):
            continue
        m = re.match(r"pred_(.+)_lead(\d+)$", v)
        if m is None:
            continue
        base = m.group(1)
        lead = int(m.group(2))
        pred_leads.setdefault(base, []).append(lead)
    for base in pred_leads:
        pred_leads[base] = sorted(pred_leads[base])
    return obs_vars, pred_leads


def sample_stats(da: xr.DataArray) -> tuple[float, float, float] | None:
    if "time" in da.dims and da.sizes.get("time", 0) > 0:
        arr = da.isel(time=0).values
    else:
        arr = da.values
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return None
    return float(vals.min()), float(vals.max()), float(vals.mean())


def unique_step(arr: np.ndarray) -> np.ndarray:
    diffs = np.diff(arr)
    return np.unique(np.round(np.abs(diffs), 6))


def time_gap_summary(times: np.ndarray) -> list[str]:
    if len(times) < 2:
        return []
    deltas = np.diff(times).astype("timedelta64[D]").astype(int)
    idx = np.where(deltas != 7)[0]
    notes = []
    for i in idx:
        notes.append(
            f"{str(times[i])[:10]} -> {str(times[i + 1])[:10]} ({int(deltas[i])} days)"
        )
    return notes


def ratio(a: float, b: float) -> float | None:
    if b == 0 or math.isnan(a) or math.isnan(b):
        return None
    return a / b


def print_dataset_summary(label: str, path: Path) -> None:
    ds = xr.open_dataset(path, decode_times=True)
    obs_vars, pred_leads = dataset_var_summary(ds)
    lat = ds["lat"].values
    lon = ds["lon"].values
    gaps = time_gap_summary(ds["time"].values)

    print(f"\n=== {label} ===")
    print(f"file: {path.name}")
    print(f"dims: {dict(ds.sizes)}")
    print(
        "time:",
        f"{str(ds['time'].values[0])[:10]} -> {str(ds['time'].values[-1])[:10]}",
        f"(n={ds.sizes['time']})",
    )
    print(
        "lat:",
        f"{float(lat[0])} -> {float(lat[-1])}",
        f"steps={unique_step(lat)[:5]}",
        f"descending={bool(np.all(np.diff(lat) < 0))}",
    )
    print(
        "lon:",
        f"{float(lon[0])} -> {float(lon[-1])}",
        f"steps={unique_step(lon)[:5]}",
        f"ascending={bool(np.all(np.diff(lon) > 0))}",
    )
    print("obs vars:", obs_vars)
    print("pred leads:", pred_leads)
    if gaps:
        print("non-7-day gaps:", "; ".join(gaps))
    else:
        print("non-7-day gaps: none")
    if ds.attrs.get("obs_valid_time_range"):
        print("obs_valid_time_range:", ds.attrs["obs_valid_time_range"])
    if ds.attrs.get("obs_missing_time_range"):
        print("obs_missing_time_range:", ds.attrs["obs_missing_time_range"])

    warnings: list[str] = []
    if not obs_vars:
        warnings.append("no obs variables")
    if ds.attrs.get("obs_missing_time_range"):
        warnings.append(
            f"obs missing for part of the file time axis ({ds.attrs['obs_missing_time_range']})"
        )
    if any(max(leads) < 6 for leads in pred_leads.values()):
        warnings.append("lead coverage shorter than 1-6 for at least one variable")
    if unique_step(lat).size > 1:
        warnings.append("latitude grid is non-uniform")

    for base in BASE_VARS:
        pred_name = f"pred_{base}_lead1"
        obs_name = f"obs_{base}"
        if pred_name not in ds.data_vars:
            continue

        pred_stats = sample_stats(ds[pred_name])
        obs_stats = sample_stats(ds[obs_name]) if obs_name in ds.data_vars else None
        units = ds[pred_name].attrs.get("units", "")
        print(f"{pred_name}: units={units!r} sample={pred_stats}")
        if obs_stats is not None:
            print(f"{obs_name}: units={ds[obs_name].attrs.get('units', '')!r} sample={obs_stats}")

        if base == "tp" and pred_stats and obs_stats:
            pred_mean = pred_stats[2]
            obs_mean = obs_stats[2]
            r = ratio(abs(pred_mean), abs(obs_mean))
            r_div_3600 = ratio(abs(pred_mean) / 3600.0, abs(obs_mean))
            if r is not None and r > 100:
                warnings.append(
                    "pred_tp scale mismatch vs obs_tp; divide by 3600 looks plausible "
                    f"(ratio={r:.2f}, ratio_after_div3600={r_div_3600:.3f})"
                )

        if base == "olr" and pred_stats and obs_stats:
            if math.copysign(1.0, pred_stats[2]) != math.copysign(1.0, obs_stats[2]):
                warnings.append("pred_olr sign differs from obs_olr")

    if any(v in ds.data_vars for v in ("pred_z500_lead1", "obs_z500")):
        warnings.append("z500 is stored as geopotential (m2/s2), not geopotential height (gpm)")

    if warnings:
        print("warnings:")
        for note in warnings:
            print(" -", note)


def compare_shared_obs() -> None:
    pairs = [
        ("CMA", "ECMWF"),
        ("CMA", "FuXi-S2S"),
        ("ECMWF", "FuXi-S2S"),
        ("CMA", "NCEP"),
        ("ECMWF", "NCEP"),
        ("FuXi-S2S", "NCEP"),
    ]
    print("\n=== Shared 1.5-degree obs consistency ===")
    opened = {
        label: xr.open_dataset(EVAL_DIR / fname, decode_times=True)
        for label, fname in DATASETS.items()
        if label in {"CMA", "ECMWF", "FuXi-S2S", "NCEP"}
    }
    for a, b in pairs:
        dsa = opened[a]
        dsb = opened[b]
        same_lat = np.array_equal(dsa["lat"].values, dsb["lat"].values)
        same_lon = np.array_equal(dsa["lon"].values, dsb["lon"].values)
        common = np.intersect1d(dsa["time"].values, dsb["time"].values)
        print(f"{a} vs {b}: same_lat={same_lat} same_lon={same_lon} common_times={len(common)}")
        for var in ("obs_tp", "obs_t2m"):
            if var not in dsa.data_vars or var not in dsb.data_vars:
                continue
            xa = dsa[var].sel(time=common)
            xb = dsb[var].sel(time=common)
            diff = (xa - xb).values
            print(
                f"  {var}: max_abs_diff={float(np.nanmax(np.abs(diff))):.6g} "
                f"mean_abs_diff={float(np.nanmean(np.abs(diff))):.6g}"
            )


def compare_time_overlaps() -> None:
    print("\n=== Pairwise time overlaps ===")
    time_sets = {}
    for label, fname in DATASETS.items():
        ds = xr.open_dataset(EVAL_DIR / fname, decode_times=True)
        time_sets[label] = {str(t)[:10] for t in ds["time"].values}

    labels = list(DATASETS)
    for i, a in enumerate(labels):
        for b in labels[i + 1 :]:
            common = sorted(time_sets[a] & time_sets[b])
            if common:
                print(f"{a} vs {b}: n={len(common)} {common[0]} -> {common[-1]}")
            else:
                print(f"{a} vs {b}: n=0")


def main() -> None:
    for label, fname in DATASETS.items():
        print_dataset_summary(label, EVAL_DIR / fname)
    compare_shared_obs()
    compare_time_overlaps()


if __name__ == "__main__":
    main()
