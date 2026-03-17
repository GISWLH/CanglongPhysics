from __future__ import annotations

import json
import os
from pathlib import Path

import numcodecs
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "Infer" / "eval"
OUT_DIR = ROOT / "analysis" / "NWP_AI"

STORE_PATH = "/data/lhwang/ERA5_raw/weekly_data_zarr/ERA5_1982_2023_weekly.zarr"
CLIM_FILE = EVAL_DIR / "climatology_2002_2016.nc"
TARGET_GRID_FILE = EVAL_DIR / "bom_s2s_target_week.nc"
OUT_CSV = OUT_DIR / "one_year_equal_length_all_models_tcc.csv"

MODEL_SPECS = {
    "CAS-Canglong": {"path": EVAL_DIR / "model_v3.nc", "window_year": 2021},
    "BOM": {"path": EVAL_DIR / "bom_s2s_target_week.nc", "window_year": 2019},
    "CMA": {"path": EVAL_DIR / "cma_s2s_target_week.nc", "window_year": 2021},
    "ECMWF": {"path": EVAL_DIR / "ecmwf_s2s_target_week.nc", "window_year": 2021},
    "FuXi-S2S": {"path": EVAL_DIR / "fuxi_s2s_target_week.nc", "window_year": 2021},
    "GEFS": {"path": EVAL_DIR / "gefs_s2s_target_week.nc", "window_year": 2019},
    "IFS": {"path": EVAL_DIR / "ifs_s2s_target_week.nc", "window_year": 2023},
    "NCEP": {"path": EVAL_DIR / "ncep_s2s_target_week.nc", "window_year": 2020},
}

VARS = ["tp", "t2m", "olr", "z500", "u850", "u200"]
LEADS = range(1, 7)
SAMPLE_SIZE = 50
SAMPLE_SEED = 20260311
CLIM_VARS = {
    "tp": "tp_clim",
    "t2m": "t2m_clim",
    "olr": "olr_clim",
    "z500": "z500_clim",
    "u850": "u850_clim",
    "u200": "u200_clim",
}

ERA5_LAT = np.linspace(90, -90, 721, dtype=np.float32)
ERA5_LON = np.linspace(0, 360 - 0.25, 1440, dtype=np.float32)

LSRR_IDX, CRR_IDX = 4, 5
T2M_IDX = 10
OLR_IDX = 1
Z_IDX = 1
U_IDX = 3
LEVEL_200_IDX = 0
LEVEL_500_IDX = 2
LEVEL_850_IDX = 4


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _build_blosc(codecs: list[dict]) -> numcodecs.Blosc | None:
    for codec in codecs:
        if codec.get("name") != "blosc":
            continue
        cfg = codec.get("configuration", {})
        shuffle = cfg.get("shuffle", 1)
        if shuffle == "shuffle":
            shuffle = 1
        elif shuffle == "bitshuffle":
            shuffle = 2
        elif shuffle == "noshuffle":
            shuffle = 0
        return numcodecs.Blosc(
            cname=cfg.get("cname", "lz4"),
            clevel=cfg.get("clevel", 5),
            shuffle=shuffle,
            blocksize=cfg.get("blocksize", 0),
        )
    return None


class ZarrArray:
    def __init__(self, store_path: str, name: str):
        meta = _load_json(os.path.join(store_path, name, "zarr.json"))
        self.shape = tuple(meta["shape"])
        self.chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
        self.dtype = np.dtype(meta["data_type"])
        endian = "little"
        for codec in meta.get("codecs", []):
            if codec.get("name") == "bytes":
                endian = codec.get("configuration", {}).get("endian", "little")
        self.dtype = self.dtype.newbyteorder("<" if endian == "little" else ">")
        self.compressor = _build_blosc(meta.get("codecs", []))
        self.array_dir = os.path.join(store_path, name)
        self.chunk_tail = ["0"] * (len(self.shape) - 1)

    def read_time(self, t_idx: int) -> np.ndarray:
        chunk_path = os.path.join(self.array_dir, "c", str(t_idx), *self.chunk_tail)
        with open(chunk_path, "rb") as f:
            raw = f.read()
        if self.compressor:
            raw = self.compressor.decode(raw)
        return np.frombuffer(raw, dtype=self.dtype).reshape(self.chunk_shape)[0]


def read_time_array(store_path: str) -> np.ndarray:
    meta = _load_json(os.path.join(store_path, "time", "zarr.json"))
    dtype = np.dtype(meta["data_type"])
    endian = "little"
    for codec in meta.get("codecs", []):
        if codec.get("name") == "bytes":
            endian = codec.get("configuration", {}).get("endian", "little")
    dtype = dtype.newbyteorder("<" if endian == "little" else ">")
    compressor = _build_blosc(meta.get("codecs", []))
    with open(os.path.join(store_path, "time", "c", "0"), "rb") as f:
        raw = f.read()
    if compressor:
        raw = compressor.decode(raw)
    return np.frombuffer(raw, dtype=dtype)


def project_multivar_era5_to_target(fields: dict[str, np.ndarray], target_lat: np.ndarray, target_lon: np.ndarray) -> dict[str, np.ndarray]:
    stack = np.stack([fields[var] for var in VARS], axis=0).astype(np.float32, copy=False)
    da = xr.DataArray(
        stack,
        coords={"variable": np.array(VARS, dtype="U8"), "lat": ERA5_LAT, "lon": ERA5_LON},
        dims=("variable", "lat", "lon"),
    )
    if np.array_equal(target_lat, ERA5_LAT) and np.array_equal(target_lon, ERA5_LON):
        out = da
    else:
        out = da.interp(lat=target_lat, lon=target_lon, method="linear")
    arr = out.values.astype(np.float32, copy=False)
    return {var: arr[i] for i, var in enumerate(VARS)}


def extract_obs_fields(surface: np.ndarray, upper: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "tp": (surface[LSRR_IDX] + surface[CRR_IDX]).astype(np.float32),
        "t2m": surface[T2M_IDX].astype(np.float32),
        "olr": surface[OLR_IDX].astype(np.float32),
        "z500": upper[Z_IDX, LEVEL_500_IDX].astype(np.float32),
        "u850": upper[U_IDX, LEVEL_850_IDX].astype(np.float32),
        "u200": upper[U_IDX, LEVEL_200_IDX].astype(np.float32),
    }


def compute_tcc_map(pred_anom: np.ndarray, obs_anom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(pred_anom) & np.isfinite(obs_anom)
    n = valid.sum(axis=0, dtype=np.int32)

    pred_masked = np.where(valid, pred_anom, 0.0).astype(np.float64, copy=False)
    obs_masked = np.where(valid, obs_anom, 0.0).astype(np.float64, copy=False)

    sum_p = pred_masked.sum(axis=0, dtype=np.float64)
    sum_o = obs_masked.sum(axis=0, dtype=np.float64)
    sum_p2 = np.einsum("thw,thw->hw", pred_masked, pred_masked, dtype=np.float64, optimize=True)
    sum_o2 = np.einsum("thw,thw->hw", obs_masked, obs_masked, dtype=np.float64, optimize=True)
    sum_po = np.einsum("thw,thw->hw", pred_masked, obs_masked, dtype=np.float64, optimize=True)

    num = n * sum_po - sum_p * sum_o
    den_p = n * sum_p2 - sum_p**2
    den_o = n * sum_o2 - sum_o**2
    den = np.sqrt(np.maximum(den_p * den_o, 0.0))

    tcc = np.full(sum_p.shape, np.nan, dtype=np.float32)
    ok = (n >= 2) & (den > 1e-30)
    tcc[ok] = (num[ok] / den[ok]).astype(np.float32)
    return tcc, n


def cosine_weighted_mean(field: np.ndarray, lat: np.ndarray) -> float:
    weights_lat = np.cos(np.deg2rad(lat)).astype(np.float64)
    weights_2d = weights_lat[:, None] * np.ones((1, field.shape[1]), dtype=np.float64) / field.shape[1]
    valid = np.isfinite(field)
    if not valid.any():
        return float("nan")
    return float(np.average(field[valid], weights=weights_2d[valid]))


def select_random_sample_points(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SAMPLE_SEED)
    flat_idx = rng.choice(lat.size * lon.size, size=SAMPLE_SIZE, replace=False)
    return flat_idx // lon.size, flat_idx % lon.size


def load_target_grid() -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(TARGET_GRID_FILE) as ds:
        lat = ds["lat"].values.astype(np.float32).copy()
        lon = ds["lon"].values.astype(np.float32).copy()
    return lat, lon


def load_regridded_climatology(target_lat: np.ndarray, target_lon: np.ndarray) -> dict[str, np.ndarray]:
    clim = {}
    with xr.open_dataset(CLIM_FILE) as ds:
        for var in VARS:
            da = ds[CLIM_VARS[var]]
            if np.array_equal(da["lat"].values, target_lat) and np.array_equal(da["lon"].values, target_lon):
                out = da
            else:
                out = da.interp(lat=target_lat, lon=target_lon, method="linear")
            clim[var] = out.values.astype(np.float32, copy=False)
    return clim


def load_model_window(path: Path, window_year: int) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(path, decode_times=True) as ds:
        sub = ds.sel(time=slice(f"{window_year}-01-01", f"{window_year}-12-31"))
        times = sub["time"].values.copy()
        woy = sub["woy"].values.astype(np.int64).copy()
    if len(times) != 52:
        raise ValueError(f"{path.name} year {window_year} does not contain 52 target weeks, got {len(times)}")
    return times, woy


def load_common_obs_for_times(
    times: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    surface_arr: ZarrArray,
    upper_arr: ZarrArray,
    date_to_idx: dict[int, int],
) -> dict[str, np.ndarray]:
    times_d = times.astype("datetime64[D]").astype(np.int64)
    out = {var: np.full((len(times), len(target_lat), len(target_lon)), np.nan, dtype=np.float32) for var in VARS}

    for i, day in enumerate(times_d):
        if int(day) not in date_to_idx:
            raise KeyError(f"ERA5 weekly zarr does not contain target date {str(times[i])[:10]}")
        t_idx = date_to_idx[int(day)]
        surface = surface_arr.read_time(t_idx)
        upper = upper_arr.read_time(t_idx)
        projected = project_multivar_era5_to_target(extract_obs_fields(surface, upper), target_lat, target_lon)
        for var in VARS:
            out[var][i] = projected[var]
        if (i + 1) % 10 == 0 or i == len(times) - 1:
            print(f"    ERA5 obs cached {i + 1}/{len(times)}")
    return out


def regrid_model_field(da: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    src_lat = da["lat"].values
    src_lon = da["lon"].values
    if np.array_equal(src_lat, target_lat) and np.array_equal(src_lon, target_lon):
        out = da
    else:
        out = da.interp(lat=target_lat, lon=target_lon, method="linear")
    return out.values.astype(np.float32, copy=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    target_lat, target_lon = load_target_grid()
    sample_i, sample_j = select_random_sample_points(target_lat, target_lon)
    clim = load_regridded_climatology(target_lat, target_lon)

    print("Loading ERA5 zarr time index...")
    time_days = read_time_array(STORE_PATH)
    zarr_dates = (np.datetime64("1940-01-01") + time_days.astype("timedelta64[D]")).astype("datetime64[D]")
    date_to_idx = {int(day): idx for idx, day in enumerate(zarr_dates.astype(np.int64))}
    surface_arr = ZarrArray(STORE_PATH, "surface")
    upper_arr = ZarrArray(STORE_PATH, "upper_air")

    windows: dict[str, dict[str, object]] = {}
    obs_cache: dict[int, dict[str, np.ndarray]] = {}
    for model, spec in MODEL_SPECS.items():
        times, woy = load_model_window(spec["path"], spec["window_year"])
        windows[model] = {"times": times, "woy": woy}

    for window_year in sorted({spec["window_year"] for spec in MODEL_SPECS.values()}):
        times = next(info["times"] for name, info in windows.items() if MODEL_SPECS[name]["window_year"] == window_year)
        print(f"Caching common ERA5 obs for window year {window_year}...")
        obs_cache[window_year] = load_common_obs_for_times(
            times=times,
            target_lat=target_lat,
            target_lon=target_lon,
            surface_arr=surface_arr,
            upper_arr=upper_arr,
            date_to_idx=date_to_idx,
        )

    rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []

    for model, spec in MODEL_SPECS.items():
        path = spec["path"]
        window_year = int(spec["window_year"])
        times = windows[model]["times"]
        woy = windows[model]["woy"]
        obs_year = obs_cache[window_year]
        window_start = str(times[0])[:10]
        window_end = str(times[-1])[:10]

        print(f"\nProcessing {model} ({window_year})...")
        with xr.open_dataset(path, decode_times=True) as ds:
            source_grid = f"{ds.sizes['lat']}x{ds.sizes['lon']}"
            for var in VARS:
                clim_by_time = clim[var][woy]
                obs = obs_year[var]
                obs_anom = obs - clim_by_time

                for lead in LEADS:
                    pred_name = f"pred_{var}_lead{lead}"
                    row = {
                        "model": model,
                        "row_type": "summary",
                        "window_year": window_year,
                        "window_start": window_start,
                        "window_end": window_end,
                        "n_target_weeks": int(len(times)),
                        "variable": var,
                        "lead": lead,
                        "source_file": path.name,
                        "source_grid": source_grid,
                        "eval_grid": "BOM native 72x144",
                        "obs_source": "ERA5 weekly zarr on BOM grid",
                        "clim_source": "2002-2016 weekly climatology on BOM grid",
                        "tcc_definition": "gridpoint anomaly TCC, then cosine-lat weighted spatial mean",
                        "sample_definition": "TCC values at 50 fixed random cells on the BOM target grid",
                        "status": "ok",
                        "tcc_weighted": np.nan,
                        "n_valid_weeks_min": np.nan,
                        "n_valid_weeks_max": np.nan,
                        "sample_grid": "BOM native 72x144 random 50 cells",
                        "sample_seed": SAMPLE_SEED,
                        "sample_size": SAMPLE_SIZE,
                        "sample_valid_count": 0,
                        "sample_mean": np.nan,
                        "sample_std": np.nan,
                        "sample_id": np.nan,
                        "sample_lat": np.nan,
                        "sample_lon": np.nan,
                        "sample_tcc": np.nan,
                        "n_variables_aggregated": 0,
                        "aggregated_variables": "",
                    }

                    if pred_name not in ds.data_vars:
                        row["status"] = "missing_prediction"
                        rows.append(row)
                        for sample_id, (ii, jj) in enumerate(zip(sample_i, sample_j), start=1):
                            sample_rows.append(
                                {
                                    **row,
                                    "row_type": "sample",
                                    "sample_id": sample_id,
                                    "sample_lat": float(target_lat[ii]),
                                    "sample_lon": float(target_lon[jj]),
                                    "sample_tcc": np.nan,
                                }
                            )
                        continue

                    pred = regrid_model_field(ds[pred_name].sel(time=times), target_lat, target_lon)
                    pred_anom = pred - clim_by_time
                    tcc_map, valid_count = compute_tcc_map(pred_anom, obs_anom)
                    row["tcc_weighted"] = cosine_weighted_mean(tcc_map, target_lat)
                    row["n_valid_weeks_min"] = int(np.nanmin(valid_count))
                    row["n_valid_weeks_max"] = int(np.nanmax(valid_count))
                    sample_values = tcc_map[sample_i, sample_j].astype(np.float32)
                    valid_sample_mask = np.isfinite(sample_values)
                    row["sample_valid_count"] = int(valid_sample_mask.sum())
                    row["sample_mean"] = float(np.nanmean(sample_values)) if row["sample_valid_count"] else float("nan")
                    row["sample_std"] = (
                        float(np.nanstd(sample_values, ddof=1)) if row["sample_valid_count"] > 1 else float("nan")
                    )
                    row["n_variables_aggregated"] = 1
                    row["aggregated_variables"] = var
                    rows.append(row)
                    for sample_id, (ii, jj, sample_val) in enumerate(zip(sample_i, sample_j, sample_values), start=1):
                        sample_rows.append(
                            {
                                **row,
                                "row_type": "sample",
                                "sample_id": sample_id,
                                "sample_lat": float(target_lat[ii]),
                                "sample_lon": float(target_lon[jj]),
                                "sample_tcc": float(sample_val),
                            }
                        )
                    print(
                        f"  {var} lead{lead}: {row['status']}, "
                        f"TCC={row['tcc_weighted']:.4f}" if np.isfinite(row["tcc_weighted"]) else f"  {var} lead{lead}: {row['status']}"
                    )

    summary_df = pd.DataFrame(rows).sort_values(["variable", "lead", "model"]).reset_index(drop=True)
    sample_df = pd.DataFrame(sample_rows).sort_values(["variable", "lead", "model", "sample_id"]).reset_index(drop=True)

    agg_rows: list[dict[str, object]] = []
    agg_sample_rows: list[dict[str, object]] = []
    for (model, lead), group in summary_df.groupby(["model", "lead"], sort=False):
        group = group.sort_values("variable").reset_index(drop=True)
        ok_group = group[group["status"] == "ok"].copy()
        template = group.iloc[0].to_dict()
        aggregated_variables = ",".join(ok_group["variable"].tolist())
        n_agg = int(len(ok_group))

        agg_row = {
            **template,
            "row_type": "summary",
            "variable": "all",
            "status": "ok" if n_agg == len(VARS) else ("partial" if n_agg > 0 else "missing_prediction"),
            "tcc_weighted": float(ok_group["tcc_weighted"].mean()) if n_agg > 0 else float("nan"),
            "n_valid_weeks_min": float(ok_group["n_valid_weeks_min"].min()) if n_agg > 0 else np.nan,
            "n_valid_weeks_max": float(ok_group["n_valid_weeks_max"].max()) if n_agg > 0 else np.nan,
            "sample_valid_count": 0,
            "sample_mean": np.nan,
            "sample_std": np.nan,
            "sample_id": np.nan,
            "sample_lat": np.nan,
            "sample_lon": np.nan,
            "sample_tcc": np.nan,
            "n_variables_aggregated": n_agg,
            "aggregated_variables": aggregated_variables,
            "sample_definition": "Per-sample mean across available variables at the fixed random BOM-grid sample set; summary row added for plotting convenience",
        }

        if n_agg > 0:
            sample_group = sample_df[
                (sample_df["model"] == model)
                & (sample_df["lead"] == lead)
                & (sample_df["variable"].isin(ok_group["variable"]))
            ]
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
            agg_row["sample_valid_count"] = int(valid_sample_mask.sum())
            agg_row["sample_mean"] = (
                float(np.nanmean(sample_values)) if agg_row["sample_valid_count"] else float("nan")
            )
            agg_row["sample_std"] = (
                float(np.nanstd(sample_values, ddof=1)) if agg_row["sample_valid_count"] > 1 else float("nan")
            )
            for _, sample_rec in sample_by_id.iterrows():
                agg_sample_rows.append(
                    {
                        **agg_row,
                        "row_type": "sample",
                        "sample_id": int(sample_rec["sample_id"]),
                        "sample_lat": float(sample_rec["sample_lat"]),
                        "sample_lon": float(sample_rec["sample_lon"]),
                        "sample_tcc": float(sample_rec["sample_tcc"]),
                    }
                )
        else:
            for sample_id, (ii, jj) in enumerate(zip(sample_i, sample_j), start=1):
                agg_sample_rows.append(
                    {
                        **agg_row,
                        "row_type": "sample",
                        "sample_id": sample_id,
                        "sample_lat": float(target_lat[ii]),
                        "sample_lon": float(target_lon[jj]),
                        "sample_tcc": np.nan,
                    }
                )

        agg_rows.append(agg_row)

    combined_df = (
        pd.concat([summary_df, pd.DataFrame(agg_rows), sample_df, pd.DataFrame(agg_sample_rows)], ignore_index=True)
        .sort_values(["row_type", "variable", "lead", "model", "sample_id"], na_position="first")
        .reset_index(drop=True)
    )
    combined_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
