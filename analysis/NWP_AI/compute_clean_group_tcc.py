from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "Infer" / "eval"
OUT_DIR = ROOT / "analysis" / "NWP_AI"

MODEL_FILES = {
    "CAS-Canglong": EVAL_DIR / "model_v3.nc",
    "CMA": EVAL_DIR / "cma_s2s_target_week.nc",
    "ECMWF": EVAL_DIR / "ecmwf_s2s_target_week.nc",
    "FuXi-S2S": EVAL_DIR / "fuxi_s2s_target_week.nc",
}

CLIM_FILE = EVAL_DIR / "climatology_2002_2016.nc"

OUT_CSV = OUT_DIR / "clean_group_tcc_tp_t2m_weighted.csv"
OUT_MAP_NC = OUT_DIR / "clean_group_tcc_tp_t2m_maps.nc"

VARS = ["tp", "t2m"]
N_LEADS = 6
REF_MODEL = "FuXi-S2S"
SAMPLE_GRID_DEG = 2.5
SAMPLE_SIZE = 50
SAMPLE_SEED = 20260310


def compute_tcc_map(pred_anom: np.ndarray, obs_anom: np.ndarray) -> tuple[np.ndarray, int]:
    n_samples = pred_anom.shape[0]

    sum_p = pred_anom.sum(axis=0, dtype=np.float64)
    sum_o = obs_anom.sum(axis=0, dtype=np.float64)
    sum_p2 = np.einsum("thw,thw->hw", pred_anom, pred_anom, dtype=np.float64, optimize=True)
    sum_o2 = np.einsum("thw,thw->hw", obs_anom, obs_anom, dtype=np.float64, optimize=True)
    sum_po = np.einsum("thw,thw->hw", pred_anom, obs_anom, dtype=np.float64, optimize=True)

    num = n_samples * sum_po - sum_p * sum_o
    den_p = n_samples * sum_p2 - sum_p**2
    den_o = n_samples * sum_o2 - sum_o**2
    den = np.sqrt(np.maximum(den_p * den_o, 0.0))

    tcc = np.full(sum_p.shape, np.nan, dtype=np.float32)
    valid = den > 1e-30
    tcc[valid] = (num[valid] / den[valid]).astype(np.float32)
    return tcc, n_samples


def cosine_weighted_mean(field: np.ndarray, lat: np.ndarray) -> float:
    weights_lat = np.cos(np.deg2rad(lat)).astype(np.float64)
    weights_2d = weights_lat[:, None] * np.ones((1, field.shape[1]), dtype=np.float64) / field.shape[1]
    valid = np.isfinite(field)
    if not valid.any():
        return float("nan")
    return float(np.average(field[valid], weights=weights_2d[valid]))


def build_regular_lat_lon(step_deg: float) -> tuple[np.ndarray, np.ndarray]:
    lat = np.arange(90.0, -90.0 - 0.5 * step_deg, -step_deg, dtype=np.float32)
    lon = np.arange(0.0, 360.0, step_deg, dtype=np.float32)
    return lat, lon


def interp_time_series_to_grid(
    field: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    da = xr.DataArray(
        field,
        coords={"time": np.arange(field.shape[0]), "lat": src_lat, "lon": src_lon},
        dims=("time", "lat", "lon"),
    )
    out = da.interp(lat=target_lat, lon=target_lon, method="linear")
    return out.values.astype(np.float32, copy=False)


def interp_weekly_clim_to_grid(
    clim: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> np.ndarray:
    da = xr.DataArray(
        clim,
        coords={"week": np.arange(clim.shape[0]), "lat": src_lat, "lon": src_lon},
        dims=("week", "lat", "lon"),
    )
    out = da.interp(lat=target_lat, lon=target_lon, method="linear")
    return out.values.astype(np.float32, copy=False)


def select_random_sample_points(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SAMPLE_SEED)
    flat_idx = rng.choice(lat.size * lon.size, size=SAMPLE_SIZE, replace=False)
    return flat_idx // lon.size, flat_idx % lon.size


def load_common_times() -> np.ndarray:
    time_sets = []
    for path in MODEL_FILES.values():
        with xr.open_dataset(path, decode_times=True) as ds:
            time_sets.append(set(ds["time"].values))
    common = sorted(set.intersection(*time_sets))
    return np.array(common)


def build_common_grid_and_clim() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    with xr.open_dataset(MODEL_FILES[REF_MODEL], decode_times=True) as ref_ds:
        lat = ref_ds["lat"].values.copy()
        lon = ref_ds["lon"].values.copy()

    with xr.open_dataset(CLIM_FILE) as clim_ds:
        clim_vars = {
            "tp": clim_ds["tp_clim"].isel(lat=slice(None, None, 6), lon=slice(None, None, 6)).values.astype(np.float32),
            "t2m": clim_ds["t2m_clim"].isel(lat=slice(None, None, 6), lon=slice(None, None, 6)).values.astype(np.float32),
        }
        clim_lat = clim_ds["lat"].values[::6]
        clim_lon = clim_ds["lon"].values[::6]

    if not np.array_equal(lat, clim_lat):
        raise ValueError("1.5-degree latitude grid does not align with 0.25-degree climatology stride.")
    if not np.array_equal(lon, clim_lon):
        raise ValueError("1.5-degree longitude grid does not align with 0.25-degree climatology stride.")

    return lat, lon, load_common_times(), clim_vars


def extract_common_field(ds: xr.Dataset, var_name: str, common_times: np.ndarray, is_cas: bool) -> np.ndarray:
    da = ds[var_name].sel(time=common_times)
    if is_cas:
        da = da.isel(lat=slice(None, None, 6), lon=slice(None, None, 6))
    return da.values.astype(np.float32, copy=False)


def verify_cas_alignment(common_times: np.ndarray) -> None:
    with xr.open_dataset(MODEL_FILES["CAS-Canglong"], decode_times=True) as cas_ds, xr.open_dataset(
        MODEL_FILES[REF_MODEL], decode_times=True
    ) as ref_ds:
        if not np.array_equal(cas_ds["lat"].values[::6], ref_ds["lat"].values):
            raise ValueError("CAS 0.25-degree latitude does not align with 1.5-degree reference grid.")
        if not np.array_equal(cas_ds["lon"].values[::6], ref_ds["lon"].values):
            raise ValueError("CAS 0.25-degree longitude does not align with 1.5-degree reference grid.")

        for base in VARS:
            cas_obs = (
                cas_ds[f"obs_{base}"]
                .sel(time=common_times)
                .isel(lat=slice(None, None, 6), lon=slice(None, None, 6))
                .values.astype(np.float32)
            )
            ref_obs = ref_ds[f"obs_{base}"].sel(time=common_times).values.astype(np.float32)
            if not np.array_equal(cas_obs, ref_obs):
                diff = np.nanmax(np.abs(cas_obs - ref_obs))
                raise ValueError(f"CAS and reference obs_{base} differ on common 1.5-degree grid, max diff={diff}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lat, lon, common_times, clim_vars = build_common_grid_and_clim()
    verify_cas_alignment(common_times)
    sample_lat, sample_lon = build_regular_lat_lon(SAMPLE_GRID_DEG)
    sample_i, sample_j = select_random_sample_points(sample_lat, sample_lon)

    common_start = str(common_times[0])[:10]
    common_end = str(common_times[-1])[:10]
    print(f"Common clean-group period: {common_start} -> {common_end} (n={len(common_times)})")

    with xr.open_dataset(MODEL_FILES[REF_MODEL], decode_times=True) as ref_ds:
        woy = ref_ds["woy"].sel(time=common_times).values.astype(np.int64)
        ref_obs = {
            base: ref_ds[f"obs_{base}"].sel(time=common_times).values.astype(np.float32)
            for base in VARS
        }

    sample_clim_vars = {
        base: interp_weekly_clim_to_grid(clim_vars[base], lat, lon, sample_lat, sample_lon)
        for base in VARS
    }
    sample_obs_anom = {
        base: interp_time_series_to_grid(ref_obs[base], lat, lon, sample_lat, sample_lon) - sample_clim_vars[base][woy]
        for base in VARS
    }

    model_maps = {}
    rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for model_name, path in MODEL_FILES.items():
        is_cas = model_name == "CAS-Canglong"
        print(f"\nProcessing {model_name} from {path.name}")
        with xr.open_dataset(path, decode_times=True) as ds:
            model_data = np.full((len(VARS), N_LEADS, len(lat), len(lon)), np.nan, dtype=np.float32)

            for vi, base in enumerate(VARS):
                clim_by_time = clim_vars[base][woy]
                obs = extract_common_field(ds, f"obs_{base}", common_times, is_cas)
                obs_anom = obs - clim_by_time
                sample_clim_by_time = sample_clim_vars[base][woy]
                sample_obs_base_anom = sample_obs_anom[base]

                for lead in range(1, N_LEADS + 1):
                    pred = extract_common_field(ds, f"pred_{base}_lead{lead}", common_times, is_cas)
                    pred_anom = pred - clim_by_time
                    tcc_map, n_samples = compute_tcc_map(pred_anom, obs_anom)
                    weighted_tcc = cosine_weighted_mean(tcc_map, lat)
                    pred_sample = interp_time_series_to_grid(pred, lat, lon, sample_lat, sample_lon)
                    pred_sample_anom = pred_sample - sample_clim_by_time
                    sample_tcc_map, _ = compute_tcc_map(pred_sample_anom, sample_obs_base_anom)
                    sample_values = sample_tcc_map[sample_i, sample_j].astype(np.float32)
                    valid_sample_mask = np.isfinite(sample_values)
                    sample_valid_count = int(valid_sample_mask.sum())
                    sample_mean = float(np.nanmean(sample_values)) if sample_valid_count else float("nan")
                    sample_std = (
                        float(np.nanstd(sample_values, ddof=1)) if sample_valid_count > 1 else float("nan")
                    )
                    model_data[vi, lead - 1] = tcc_map
                    rows.append(
                        {
                            "model": model_name,
                            "variable": base,
                            "lead": lead,
                            "tcc_weighted": weighted_tcc,
                            "n_target_weeks": n_samples,
                            "time_start": common_start,
                            "time_end": common_end,
                            "grid": "121x240 1.5deg",
                            "sample_grid": f"{len(sample_lat)}x{len(sample_lon)} regular {SAMPLE_GRID_DEG:.1f}deg",
                            "sample_seed": SAMPLE_SEED,
                            "sample_size": SAMPLE_SIZE,
                            "sample_valid_count": sample_valid_count,
                            "sample_mean": sample_mean,
                            "sample_std": sample_std,
                            "tcc_definition": "gridpoint anomaly TCC, then cosine-lat weighted spatial mean",
                            "sample_definition": (
                                "TCC computed on a regular 2.5-degree grid after interpolating anomaly time series; "
                                "50 global cells sampled with a fixed random seed"
                            ),
                        }
                    )
                    for sample_id, (ii, jj, sample_val) in enumerate(zip(sample_i, sample_j, sample_values), start=1):
                        sample_rows.append(
                            {
                                "model": model_name,
                                "variable": base,
                                "lead": lead,
                                "sample_id": sample_id,
                                "sample_lat": float(sample_lat[ii]),
                                "sample_lon": float(sample_lon[jj]),
                                "sample_tcc": float(sample_val),
                                "sample_grid": f"{len(sample_lat)}x{len(sample_lon)} regular {SAMPLE_GRID_DEG:.1f}deg",
                                "sample_seed": SAMPLE_SEED,
                                "time_start": common_start,
                                "time_end": common_end,
                            }
                        )
                    print(
                        f"  {base} lead{lead}: weighted TCC = {weighted_tcc:.4f}, "
                        f"sample mean/std = {sample_mean:.4f}/{sample_std:.4f}"
                    )

            model_maps[model_name] = model_data

    summary_df = pd.DataFrame(rows).sort_values(["variable", "lead", "model"]).reset_index(drop=True)
    sample_df = pd.DataFrame(sample_rows).sort_values(["variable", "lead", "model", "sample_id"]).reset_index(drop=True)

    agg_rows: list[dict[str, object]] = []
    for (model, lead), group in summary_df.groupby(["model", "lead"], sort=False):
        group = group.sort_values("variable").reset_index(drop=True)
        sample_group = sample_df[(sample_df["model"] == model) & (sample_df["lead"] == lead)]
        sample_by_id = (
            sample_group.groupby("sample_id", as_index=False)
            .agg(
                sample_lat=("sample_lat", "first"),
                sample_lon=("sample_lon", "first"),
                sample_tcc=("sample_tcc", "mean"),
            )
            .sort_values("sample_id")
            .reset_index(drop=True)
        )
        sample_values = sample_by_id["sample_tcc"].to_numpy(dtype=np.float64)
        valid_sample_mask = np.isfinite(sample_values)
        sample_valid_count = int(valid_sample_mask.sum())
        agg_rows.append(
            {
                "model": model,
                "variable": "all",
                "lead": lead,
                "tcc_weighted": float(group["tcc_weighted"].mean()),
                "n_target_weeks": int(group["n_target_weeks"].iloc[0]),
                "time_start": group["time_start"].iloc[0],
                "time_end": group["time_end"].iloc[0],
                "grid": group["grid"].iloc[0],
                "sample_grid": group["sample_grid"].iloc[0],
                "sample_seed": int(group["sample_seed"].iloc[0]),
                "sample_size": int(group["sample_size"].iloc[0]),
                "sample_valid_count": sample_valid_count,
                "sample_mean": float(np.nanmean(sample_values)) if sample_valid_count else float("nan"),
                "sample_std": float(np.nanstd(sample_values, ddof=1)) if sample_valid_count > 1 else float("nan"),
                "tcc_definition": group["tcc_definition"].iloc[0],
                "sample_definition": (
                    "Per-sample mean across tp and t2m on the fixed 2.5-degree random sample set; "
                    "summary row added for plotting convenience"
                ),
            }
        )
        for _, row in sample_by_id.iterrows():
            sample_rows.append(
                {
                    "model": model,
                    "variable": "all",
                    "lead": lead,
                    "sample_id": int(row["sample_id"]),
                    "sample_lat": float(row["sample_lat"]),
                    "sample_lon": float(row["sample_lon"]),
                    "sample_tcc": float(row["sample_tcc"]),
                    "sample_grid": group["sample_grid"].iloc[0],
                    "sample_seed": int(group["sample_seed"].iloc[0]),
                    "time_start": group["time_start"].iloc[0],
                    "time_end": group["time_end"].iloc[0],
                }
            )

    summary_df = (
        pd.concat([summary_df, pd.DataFrame(agg_rows)], ignore_index=True)
        .sort_values(["variable", "lead", "model"])
        .reset_index(drop=True)
    )
    sample_df = pd.DataFrame(sample_rows).sort_values(["variable", "lead", "model", "sample_id"]).reset_index(drop=True)

    summary_out = summary_df.copy()
    summary_out["row_type"] = "summary"
    summary_out["sample_id"] = np.nan
    summary_out["sample_lat"] = np.nan
    summary_out["sample_lon"] = np.nan
    summary_out["sample_tcc"] = np.nan

    sample_metric_cols = [
        "model",
        "variable",
        "lead",
        "tcc_weighted",
        "n_target_weeks",
        "grid",
        "sample_size",
        "sample_valid_count",
        "sample_mean",
        "sample_std",
        "tcc_definition",
        "sample_definition",
    ]
    sample_out = sample_df.merge(summary_df[sample_metric_cols], on=["model", "variable", "lead"], how="left")
    sample_out["row_type"] = "sample"

    combined_columns = [
        "row_type",
        "model",
        "variable",
        "lead",
        "tcc_weighted",
        "sample_tcc",
        "sample_id",
        "sample_lat",
        "sample_lon",
        "n_target_weeks",
        "time_start",
        "time_end",
        "grid",
        "sample_grid",
        "sample_seed",
        "sample_size",
        "sample_valid_count",
        "sample_mean",
        "sample_std",
        "tcc_definition",
        "sample_definition",
    ]
    combined_df = (
        pd.concat([summary_out[combined_columns], sample_out[combined_columns]], ignore_index=True)
        .sort_values(["row_type", "variable", "lead", "model", "sample_id"], na_position="first")
        .reset_index(drop=True)
    )

    combined_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved combined CSV: {OUT_CSV}")

    tcc_stack = np.stack([model_maps[name] for name in MODEL_FILES], axis=0)
    ds_out = xr.Dataset(
        data_vars={
            "tcc_map": (["model", "variable", "lead", "lat", "lon"], tcc_stack),
        },
        coords={
            "model": np.array(list(MODEL_FILES.keys()), dtype="U32"),
            "variable": np.array(VARS, dtype="U8"),
            "lead": np.arange(1, N_LEADS + 1, dtype=np.int32),
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "description": "Clean-group TCC maps on the shared 1.5-degree grid",
            "models": "CAS-Canglong,CMA,ECMWF,FuXi-S2S",
            "variables": "tp,t2m",
            "lead_range": "1-6",
            "time_range": f"{common_start} to {common_end}",
            "n_target_weeks": int(len(common_times)),
            "grid_note": "CAS 0.25-degree fields are sampled every 6 cells onto the exact shared 1.5-degree grid",
            "clim_source": str(CLIM_FILE),
            "clim_note": "2002-2016 weekly climatology sampled from 0.25 degree to the exact shared 1.5 degree grid",
            "tcc_definition": "Per-grid-point Pearson correlation of anomaly time series, then cosine-lat weighted spatial mean for scalar summaries",
            "sample_grid_note": (
                "A separate CSV stores 50 fixed random samples from a regular 2.5-degree TCC map. "
                "The 2.5-degree TCC map is computed after interpolating anomaly time series from the shared 1.5-degree grid."
            ),
            "sample_seed": SAMPLE_SEED,
        },
    )
    ds_out.to_netcdf(OUT_MAP_NC, encoding={"tcc_map": {"zlib": True, "complevel": 4}})
    print(f"Saved TCC map NetCDF: {OUT_MAP_NC}")

    print("\nWeighted TCC summary:")
    for base in VARS:
        sub = summary_df[summary_df["variable"] == base]
        print(f"\n[{base}]")
        pivot = sub.pivot(index="lead", columns="model", values="tcc_weighted")
        print(pivot.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
