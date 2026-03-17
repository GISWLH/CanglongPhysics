from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc4
import numcodecs
import numpy as np
import xarray as xr


ROOT = Path(__file__).resolve().parents[2]
IFS_PATH = ROOT / "Infer" / "eval" / "ifs_s2s_target_week.nc"
STORE_PATH = "/data/lhwang/ERA5_raw/weekly_data_zarr/ERA5_1982_2023_weekly.zarr"
UTC_NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

ERA5_LAT = np.linspace(90, -90, 721, dtype=np.float32)
ERA5_LON = np.linspace(0, 360 - 0.25, 1440, dtype=np.float32)

LSRR_IDX, CRR_IDX = 4, 5
T2M_IDX = 10
Z_IDX = 1
P500_IDX = 2
U_IDX = 3
LEVEL_200_IDX = 0
LEVEL_850_IDX = 4

OBS_INFO = {
    "tp": ("Total precipitation (lsrr+crr) (ERA5 obs)", "kg/m2/s"),
    "t2m": ("2m temperature (ERA5 obs)", "K"),
    "z500": ("Geopotential at 500hPa (ERA5 obs)", "m2/s2"),
    "u850": ("Zonal wind at 850hPa (ERA5 obs)", "m/s"),
    "u200": ("Zonal wind at 200hPa (ERA5 obs)", "m/s"),
}


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


def extract_obs_fields(surface: np.ndarray, upper: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "tp": (surface[LSRR_IDX] + surface[CRR_IDX]).astype(np.float32),
        "t2m": surface[T2M_IDX].astype(np.float32),
        "z500": upper[Z_IDX, P500_IDX].astype(np.float32),
        "u850": upper[U_IDX, LEVEL_850_IDX].astype(np.float32),
        "u200": upper[U_IDX, LEVEL_200_IDX].astype(np.float32),
    }


def project_field(field: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray) -> np.ndarray:
    if np.array_equal(target_lat, ERA5_LAT) and np.array_equal(target_lon, ERA5_LON):
        return field.astype(np.float32, copy=False)

    da = xr.DataArray(
        field,
        coords={"lat": ERA5_LAT, "lon": ERA5_LON},
        dims=("lat", "lon"),
    )
    out = da.interp(lat=target_lat, lon=target_lon, method="linear")
    return out.values.astype(np.float32)


def create_or_get_obs_var(ds: nc4.Dataset, name: str) -> nc4.Variable:
    if name in ds.variables:
        var = ds.variables[name]
    else:
        var = ds.createVariable(
            name,
            "f4",
            ("time", "lat", "lon"),
            zlib=True,
            complevel=4,
            chunksizes=(1, len(ds.dimensions["lat"]), len(ds.dimensions["lon"])),
            fill_value=np.float32(np.nan),
        )
    base = name.replace("obs_", "")
    long_name, units = OBS_INFO[base]
    var.long_name = long_name
    var.units = units
    return var


def append_history(ds: nc4.Dataset, note: str) -> None:
    history = getattr(ds, "history", "")
    entry = f"{UTC_NOW} {note}"
    ds.history = f"{history}\n{entry}".strip() if history else entry


def main() -> None:
    with xr.open_dataset(IFS_PATH, decode_times=True) as target:
        target_times = target["time"].values.copy()
        target_lat = target["lat"].values.astype(np.float32).copy()
        target_lon = target["lon"].values.astype(np.float32).copy()
        global_idx = target["global_idx"].values.astype(np.int64).copy()

    time_days = read_time_array(STORE_PATH)
    zarr_dates = np.datetime64("1940-01-01") + time_days.astype("timedelta64[D]")
    max_gi = len(zarr_dates) - 1
    valid_mask = (global_idx >= 0) & (global_idx <= max_gi)

    if valid_mask.any():
        expected_dates = zarr_dates[global_idx[valid_mask]]
        if not np.array_equal(expected_dates, target_times[valid_mask]):
            mismatch = np.where(expected_dates != target_times[valid_mask])[0][0]
            raise ValueError(
                "IFS target time does not match ERA5 zarr date at first valid index: "
                f"target={str(target_times[valid_mask][mismatch])[:10]} "
                f"zarr={str(expected_dates[mismatch])[:10]}"
            )

    covered_times = target_times[valid_mask]
    uncovered_times = target_times[~valid_mask]
    print(
        "IFS obs coverage from ERA5 zarr:",
        str(covered_times[0])[:10],
        "->",
        str(covered_times[-1])[:10],
        f"(n={covered_times.size})",
    )
    if uncovered_times.size:
        print(
            "IFS obs unavailable beyond zarr coverage:",
            str(uncovered_times[0])[:10],
            "->",
            str(uncovered_times[-1])[:10],
            f"(n={uncovered_times.size})",
        )

    surface_arr = ZarrArray(STORE_PATH, "surface")
    upper_arr = ZarrArray(STORE_PATH, "upper_air")

    with nc4.Dataset(IFS_PATH, "r+") as ds:
        ds.set_auto_mask(False)
        obs_vars = {name: create_or_get_obs_var(ds, f"obs_{name}") for name in OBS_INFO}

        if uncovered_times.size:
            first_uncovered = int(np.where(~valid_mask)[0][0])
            for var in obs_vars.values():
                var[first_uncovered:] = np.float32(np.nan)

        for tidx, gi in enumerate(global_idx):
            if gi < 0 or gi > max_gi:
                continue
            raw_surface = surface_arr.read_time(int(gi))
            raw_upper = upper_arr.read_time(int(gi))
            obs = extract_obs_fields(raw_surface, raw_upper)
            for name, field in obs.items():
                obs_vars[name][tidx] = project_field(field, target_lat, target_lon)
            if (tidx + 1) % 10 == 0 or tidx == 0:
                print(f"  wrote obs for {tidx + 1}/{covered_times.size} covered target weeks")
            if (tidx + 1) % 20 == 0:
                ds.sync()

        ds.available_obs_vars = "tp,t2m,z500,u850,u200"
        ds.obs_source = STORE_PATH
        ds.obs_grid_note = (
            "ERA5 weekly zarr projected to the native target grid; "
            "for IFS this is an exact 0.25 degree match"
        )
        ds.obs_valid_time_range = f"{str(covered_times[0])[:10]} to {str(covered_times[-1])[:10]}"
        if uncovered_times.size:
            ds.obs_missing_time_range = (
                f"{str(uncovered_times[0])[:10]} to {str(uncovered_times[-1])[:10]}"
            )
            ds.obs_missing_reason = (
                "Local ERA5 weekly zarr currently ends at "
                f"{str(zarr_dates[-1])[:10]}; later obs remain NaN"
            )
        ds.description = (
            "Target-week-centric evaluation dataset from local IFS NetCDF files. "
            "The 14 daily lead days are approximated as two 7-day blocks, each snapped "
            "to the nearest natural week, so the output keeps the same natural-week date "
            "axis as other S2S products. Older 0.4 degree files are regridded to 0.25 degree. "
            "ERA5 observations for tp,t2m,z500,u850,u200 are extracted from the weekly zarr "
            "archive and projected to the file grid. The source variable olr is omitted because "
            "the local IFS source does not provide a stable finite OLR field across the full file."
        )
        append_history(
            ds,
            "Added obs_tp, obs_t2m, obs_z500, obs_u850, obs_u200 from ERA5 weekly zarr; "
            f"valid through {str(covered_times[-1])[:10]}",
        )
        ds.sync()

    print(f"Updated {IFS_PATH}")


if __name__ == "__main__":
    main()
