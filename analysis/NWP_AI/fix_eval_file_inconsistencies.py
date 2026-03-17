from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import netCDF4 as nc
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = ROOT / "Infer" / "eval"
UTC_NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

FUXI_FILE = EVAL_DIR / "fuxi_s2s_target_week.nc"
GEFS_FILE = EVAL_DIR / "gefs_s2s_target_week.nc"


def blockwise_mean(var: nc.Variable, block_size: int = 16) -> float:
    n_time = var.shape[0]
    total = 0.0
    count = 0
    for start in range(0, n_time, block_size):
        stop = min(start + block_size, n_time)
        data = np.asarray(var[start:stop], dtype=np.float64)
        finite = np.isfinite(data)
        if not finite.any():
            continue
        total += float(data[finite].sum())
        count += int(finite.sum())
    return total / count if count else float("nan")


def apply_scale(
    ds: nc.Dataset,
    var_name: str,
    factor: float,
    operation: str,
    note: str,
    block_size: int = 16,
) -> None:
    var = ds.variables[var_name]
    if getattr(var, "inconsistency_fix_note", "") == note:
        print(f"skip {var_name}: already patched")
        return

    before_mean = blockwise_mean(var, block_size=block_size)
    n_time = var.shape[0]

    for start in range(0, n_time, block_size):
        stop = min(start + block_size, n_time)
        data = np.asarray(var[start:stop], dtype=np.float32)
        if operation == "divide":
            data = data / factor
        elif operation == "multiply":
            data = data * factor
        else:
            raise ValueError(f"unsupported operation: {operation}")
        var[start:stop] = data.astype(var.dtype)

    after_mean = blockwise_mean(var, block_size=block_size)
    var.inconsistency_fix_note = note
    var.inconsistency_fix_time_utc = UTC_NOW
    print(
        f"{var_name}: {operation} by {factor}, "
        f"mean {before_mean:.12g} -> {after_mean:.12g}"
    )


def append_history(ds: nc.Dataset, note: str) -> None:
    history = getattr(ds, "history", "")
    new_entry = f"{UTC_NOW} {note}"
    ds.history = f"{history}\n{new_entry}".strip() if history else new_entry


def patch_fuxi() -> None:
    note = "pred_tp_lead1-6 divided by 3600 to match kg/m2/s target-week convention"
    print(f"\nPatching {FUXI_FILE.name}")
    with nc.Dataset(FUXI_FILE, "r+") as ds:
        ds.set_auto_mask(False)
        for lead in range(1, 7):
            apply_scale(
                ds=ds,
                var_name=f"pred_tp_lead{lead}",
                factor=3600.0,
                operation="divide",
                note=note,
            )
        append_history(ds, f"Applied FuXi precipitation scale fix: {note}")


def patch_gefs() -> None:
    note = "pred_olr_lead1-2 multiplied by -1 to match stored obs_olr sign convention"
    print(f"\nPatching {GEFS_FILE.name}")
    with nc.Dataset(GEFS_FILE, "r+") as ds:
        ds.set_auto_mask(False)
        for lead in (1, 2):
            apply_scale(
                ds=ds,
                var_name=f"pred_olr_lead{lead}",
                factor=-1.0,
                operation="multiply",
                note=note,
            )
        append_history(ds, f"Applied GEFS OLR sign fix: {note}")


def main() -> None:
    patch_fuxi()
    patch_gefs()


if __name__ == "__main__":
    main()
