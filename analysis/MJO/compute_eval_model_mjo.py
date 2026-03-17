"""
Compute weekly MJO skill for local eval models and plot them together.

This script keeps the old paper-curve workflow untouched. It recomputes
weekly MJO bivariate COR directly from the local target-week-centric
`Infer/eval/*.nc` files so the comparison uses one consistent pipeline.

Models:
  - CAS-Canglong V3.5: full RMM (OLR + U850 + U200), 2017-2021 cache
  - FuXi-S2S:          full RMM, 2017-2021
  - GEFS:              full RMM, 2017-2019, lead 1-2 only
  - IFS:               wind-only proxy (U850 + U200), lead 1-2 only

IFS note:
  The local `ifs_s2s_target_week.nc` does not contain `pred_olr_*`, so a
  standard 3-field RMM cannot be computed from this file alone. For IFS we
  therefore project only the U850/U200 components onto the cached EOFs and
  label the result explicitly as a wind-only proxy.

Outputs:
  - analysis/MJO/mjo_eval_models.csv
  - analysis/MJO/mjo_eval_models_yearly.csv
  - analysis/MJO/mjo_eval_models.png
  - analysis/MJO/mjo_eval_models.svg
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc4
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis" / "MJO"
CACHE_PATH = OUT_DIR / "mjo_cache_v35.npz"

SUMMARY_CSV = OUT_DIR / "mjo_eval_models.csv"
YEARLY_CSV = OUT_DIR / "mjo_eval_models_yearly.csv"
PLOT_PNG = OUT_DIR / "mjo_eval_models.png"
PLOT_SVG = OUT_DIR / "mjo_eval_models.svg"


MODEL_CONFIGS = [
    {
        "model": "CAS-Canglong V3.5",
        "path": ROOT / "Infer" / "eval" / "model_v3.nc",
        "mode": "full",
        "group": "AI",
        "color": "#2CA02C",
        "linestyle": "-",
        "marker": "o",
        "source": "cache",
    },
    {
        "model": "FuXi-S2S",
        "path": ROOT / "Infer" / "eval" / "fuxi_s2s_target_week.nc",
        "mode": "full",
        "group": "AI",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "s",
        "source": "eval",
    },
    {
        "model": "GEFS",
        "path": ROOT / "Infer" / "eval" / "gefs_s2s_target_week.nc",
        "mode": "full",
        "group": "NWP",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "^",
        "source": "eval",
    },
    {
        "model": "IFS",
        "path": ROOT / "Infer" / "eval" / "ifs_s2s_target_week.nc",
        "mode": "wind_only",
        "group": "NWP",
        "color": "#666666",
        "linestyle": ":",
        "marker": "D",
        "source": "eval",
    },
]


FULL_FIELDS = ["olr", "u850", "u200"]
WIND_ONLY_FIELDS = ["u850", "u200"]
LEAD_RE = re.compile(r"^pred_u850_lead(\d+)$")
ARIAL_PATH = "/usr/share/fonts/arial/ARIAL.TTF"


def load_reference(cache_path: Path) -> dict[str, object]:
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing {cache_path}. Run analysis/MJO/compute_mjo_skill.py first."
        )

    cache = np.load(cache_path, allow_pickle=True)
    return {
        "obs_rmm1": cache["obs_rmm1"].astype(np.float64),
        "obs_rmm2": cache["obs_rmm2"].astype(np.float64),
        "pred_rmm1": cache["pred_rmm1"].astype(np.float64),
        "pred_rmm2": cache["pred_rmm2"].astype(np.float64),
        "eof1": cache["eof1"].astype(np.float64),
        "eof2": cache["eof2"].astype(np.float64),
        "seasonal": {
            "olr": cache["seasonal_olr"].astype(np.float64),
            "u850": cache["seasonal_u850"].astype(np.float64),
            "u200": cache["seasonal_u200"].astype(np.float64),
        },
        "field_std": {
            "olr": float(cache["field_std_olr"]),
            "u850": float(cache["field_std_u850"]),
            "u200": float(cache["field_std_u200"]),
        },
    }


def available_leads(ds: nc4.Dataset) -> list[int]:
    leads = []
    for name in ds.variables:
        match = LEAD_RE.match(name)
        if match:
            leads.append(int(match.group(1)))
    return sorted(leads)


def get_lon_index(lon_values: np.ndarray) -> np.ndarray:
    lon = np.mod(np.asarray(lon_values, dtype=np.float64), 360.0)
    idx = np.rint(lon / 0.25).astype(np.int64) % 1440
    max_err = np.max(np.abs(idx * 0.25 - lon))
    if max_err > 1e-6:
        raise ValueError(f"Longitude grid is not aligned to the 0.25 degree reference (max err {max_err})")
    return idx


def get_tropical_slice_and_weights(lat_values: np.ndarray) -> tuple[slice, np.ndarray]:
    lat = np.asarray(lat_values, dtype=np.float64)
    trop_idx = np.where(np.abs(lat) <= 15.0 + 1e-6)[0]
    if trop_idx.size == 0:
        raise ValueError("Could not find tropical latitudes within 15N-15S")
    lat_slice = slice(int(trop_idx[0]), int(trop_idx[-1]) + 1)
    weights = np.cos(np.deg2rad(lat[lat_slice])).astype(np.float64)
    return lat_slice, weights


def tropical_average(data: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = data * weights[None, :, None]
    valid_weights = np.isfinite(data) * weights[None, :, None]
    weight_sum = valid_weights.sum(axis=1)
    return np.nansum(weighted, axis=1) / np.where(weight_sum > 0.0, weight_sum, np.nan)


def extract_profiles(
    ds: nc4.Dataset,
    fields: list[str],
    leads: list[int],
    lat_slice: slice,
    weights: np.ndarray,
    block_size: int = 8,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    time_size = ds.dimensions["time"].size
    lon_size = ds.dimensions["lon"].size
    years = ds.variables["year"][:].astype(np.int32)
    woys = ds.variables["woy"][:].astype(np.int32)

    obs_profiles = {
        field: np.full((time_size, lon_size), np.nan, dtype=np.float64)
        for field in fields
    }
    pred_profiles = {
        field: np.full((time_size, len(leads), lon_size), np.nan, dtype=np.float64)
        for field in fields
    }

    for field in fields:
        obs_var = ds.variables[f"obs_{field}"]
        for start in range(0, time_size, block_size):
            stop = min(start + block_size, time_size)
            block = np.asarray(obs_var[start:stop, lat_slice, :], dtype=np.float64)
            obs_profiles[field][start:stop] = tropical_average(block, weights)

        for lead_index, lead in enumerate(leads):
            pred_var = ds.variables[f"pred_{field}_lead{lead}"]
            for start in range(0, time_size, block_size):
                stop = min(start + block_size, time_size)
                block = np.asarray(pred_var[start:stop, lat_slice, :], dtype=np.float64)
                pred_profiles[field][start:stop, lead_index] = tropical_average(block, weights)

    return obs_profiles, pred_profiles, years, woys


def prepare_reference(
    ref: dict[str, object],
    lon_index: np.ndarray,
    mode: str,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, float], np.ndarray, np.ndarray]:
    fields = FULL_FIELDS if mode == "full" else WIND_ONLY_FIELDS
    seasonal_all = ref["seasonal"]
    field_std_all = ref["field_std"]
    eof1_full = ref["eof1"]
    eof2_full = ref["eof2"]

    seasonal = {field: seasonal_all[field][:, lon_index] for field in fields}
    field_std = {field: field_std_all[field] for field in fields}

    components1 = []
    components2 = []
    for field in fields:
        if field == "olr":
            start = 0
        elif field == "u850":
            start = 1440
        elif field == "u200":
            start = 2880
        else:
            raise ValueError(f"Unsupported field {field}")
        stop = start + 1440
        components1.append(eof1_full[start:stop][lon_index])
        components2.append(eof2_full[start:stop][lon_index])

    eof1 = np.concatenate(components1)
    eof2 = np.concatenate(components2)
    return fields, seasonal, field_std, eof1, eof2


def compute_rmm(
    profiles: dict[str, np.ndarray],
    woys: np.ndarray,
    fields: list[str],
    seasonal: dict[str, np.ndarray],
    field_std: dict[str, float],
    eof1: np.ndarray,
    eof2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sample_shape = next(iter(profiles.values())).shape
    time_size = sample_shape[0]
    lead_axis = len(sample_shape) == 3
    n_leads = sample_shape[1] if lead_axis else None

    if lead_axis:
        rmm1 = np.full((time_size, n_leads), np.nan, dtype=np.float64)
        rmm2 = np.full((time_size, n_leads), np.nan, dtype=np.float64)
    else:
        rmm1 = np.full(time_size, np.nan, dtype=np.float64)
        rmm2 = np.full(time_size, np.nan, dtype=np.float64)

    for t in range(time_size):
        woy = int(woys[t])
        if lead_axis:
            for lead_index in range(n_leads):
                parts = []
                valid = True
                for field in fields:
                    values = profiles[field][t, lead_index]
                    if not np.all(np.isfinite(values)):
                        valid = False
                        break
                    parts.append((values - seasonal[field][woy]) / field_std[field])
                if not valid:
                    continue
                combined = np.concatenate(parts)
                rmm1[t, lead_index] = combined @ eof1
                rmm2[t, lead_index] = combined @ eof2
        else:
            parts = []
            valid = True
            for field in fields:
                values = profiles[field][t]
                if not np.all(np.isfinite(values)):
                    valid = False
                    break
                parts.append((values - seasonal[field][woy]) / field_std[field])
            if not valid:
                continue
            combined = np.concatenate(parts)
            rmm1[t] = combined @ eof1
            rmm2[t] = combined @ eof2

    return rmm1, rmm2


def bivariate_cor(
    obs_rmm1: np.ndarray,
    obs_rmm2: np.ndarray,
    pred_rmm1: np.ndarray,
    pred_rmm2: np.ndarray,
) -> tuple[float, int]:
    mask = (
        np.isfinite(obs_rmm1)
        & np.isfinite(obs_rmm2)
        & np.isfinite(pred_rmm1)
        & np.isfinite(pred_rmm2)
    )
    n_valid = int(mask.sum())
    if n_valid < 2:
        return np.nan, n_valid

    a1 = obs_rmm1[mask]
    a2 = obs_rmm2[mask]
    b1 = pred_rmm1[mask]
    b2 = pred_rmm2[mask]

    numerator = np.sum(a1 * b1 + a2 * b2)
    denominator = np.sqrt(
        np.sum(a1**2 + a2**2) * np.sum(b1**2 + b2**2)
    )
    if denominator <= 1e-12:
        return np.nan, n_valid
    return float(numerator / denominator), n_valid


def summarize_model(
    model: str,
    mode: str,
    years: np.ndarray,
    leads: list[int],
    obs_rmm1: np.ndarray,
    obs_rmm2: np.ndarray,
    pred_rmm1: np.ndarray,
    pred_rmm2: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    method = "full_rmm" if mode == "full" else "wind_only_proxy"
    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []

    for lead_index, lead in enumerate(leads):
        cor, n_valid = bivariate_cor(
            obs_rmm1,
            obs_rmm2,
            pred_rmm1[:, lead_index],
            pred_rmm2[:, lead_index],
        )
        summary_rows.append(
            {
                "model": model,
                "method": method,
                "lead_week": int(lead),
                "lead_day": int(lead * 7),
                "cor": cor,
                "n_samples": n_valid,
            }
        )

    for year in sorted(np.unique(years).tolist()):
        year_mask = years == year
        for lead_index, lead in enumerate(leads):
            cor, n_valid = bivariate_cor(
                obs_rmm1[year_mask],
                obs_rmm2[year_mask],
                pred_rmm1[year_mask, lead_index],
                pred_rmm2[year_mask, lead_index],
            )
            yearly_rows.append(
                {
                    "model": model,
                    "method": method,
                    "year": int(year),
                    "lead_week": int(lead),
                    "lead_day": int(lead * 7),
                    "cor": cor,
                    "n_samples": n_valid,
                }
            )

    return summary_rows, yearly_rows


def compute_canglong(ref: dict[str, object], eval_path: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    print(f"\n[{MODEL_CONFIGS[0]['model']}] loading cached RMM indices")
    with nc4.Dataset(eval_path, "r") as ds:
        years = ds.variables["year"][:].astype(np.int32)

    obs_rmm1 = ref["obs_rmm1"]
    obs_rmm2 = ref["obs_rmm2"]
    pred_rmm1 = ref["pred_rmm1"]
    pred_rmm2 = ref["pred_rmm2"]
    leads = list(range(1, pred_rmm1.shape[1] + 1))

    return summarize_model(
        model="CAS-Canglong V3.5",
        mode="full",
        years=years,
        leads=leads,
        obs_rmm1=obs_rmm1,
        obs_rmm2=obs_rmm2,
        pred_rmm1=pred_rmm1,
        pred_rmm2=pred_rmm2,
    )


def compute_eval_model(config: dict[str, object], ref: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = str(config["model"])
    path = Path(config["path"])
    mode = str(config["mode"])

    print(f"\n[{model}] reading {path.name}")
    with nc4.Dataset(path, "r") as ds:
        leads = available_leads(ds)
        lat_slice, weights = get_tropical_slice_and_weights(ds.variables["lat"][:])
        lon_index = get_lon_index(ds.variables["lon"][:])
        fields, seasonal, field_std, eof1, eof2 = prepare_reference(ref, lon_index, mode)

        print(
            f"  fields={fields}, leads={leads}, grid={ds.dimensions['lat'].size}x{ds.dimensions['lon'].size}"
        )
        obs_profiles, pred_profiles, years, woys = extract_profiles(
            ds=ds,
            fields=fields,
            leads=leads,
            lat_slice=lat_slice,
            weights=weights,
        )

    obs_rmm1, obs_rmm2 = compute_rmm(
        profiles=obs_profiles,
        woys=woys,
        fields=fields,
        seasonal=seasonal,
        field_std=field_std,
        eof1=eof1,
        eof2=eof2,
    )
    pred_rmm1, pred_rmm2 = compute_rmm(
        profiles=pred_profiles,
        woys=woys,
        fields=fields,
        seasonal=seasonal,
        field_std=field_std,
        eof1=eof1,
        eof2=eof2,
    )

    summary_rows, yearly_rows = summarize_model(
        model=model,
        mode=mode,
        years=years,
        leads=leads,
        obs_rmm1=obs_rmm1,
        obs_rmm2=obs_rmm2,
        pred_rmm1=pred_rmm1,
        pred_rmm2=pred_rmm2,
    )

    for row in summary_rows:
        cor_text = "nan" if not np.isfinite(row["cor"]) else f"{row['cor']:.4f}"
        print(
            f"  lead {row['lead_week']} (day {row['lead_day']}): COR={cor_text}, n={row['n_samples']}"
        )
    return summary_rows, yearly_rows


def year_label_from_yearly(yearly_df: pd.DataFrame, model: str) -> str:
    valid = yearly_df[
        (yearly_df["model"] == model)
        & yearly_df["n_samples"].gt(0)
        & yearly_df["cor"].notna()
    ]["year"].drop_duplicates().sort_values()
    if valid.empty:
        return "no valid years"
    years = valid.astype(int).tolist()
    return str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"


def configure_matplotlib() -> None:
    if os.path.exists(ARIAL_PATH):
        font_manager.fontManager.addfont(ARIAL_PATH)

    plt.style.use("seaborn-v0_8-talk")
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 600,
            "lines.linewidth": 1.8,
            "axes.linewidth": 1.0,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.edgecolor": "#454545",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.color": "#454545",
            "ytick.color": "#454545",
            "savefig.bbox": "tight",
            "savefig.transparent": False,
        }
    )
    mpl.rcParams["svg.fonttype"] = "none"


def render_plot(summary_df: pd.DataFrame, yearly_df: pd.DataFrame) -> None:
    configure_matplotlib()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    for config in MODEL_CONFIGS:
        model = str(config["model"])
        sub = (
            summary_df[summary_df["model"] == model]
            .sort_values("lead_day")
            .copy()
        )
        if sub.empty:
            continue

        label = model
        year_label = year_label_from_yearly(yearly_df, model)
        if model == "IFS":
            label = f"{model} wind proxy ({year_label})"
        else:
            label = f"{model} ({year_label})"

        ax.plot(
            sub["lead_day"],
            sub["cor"],
            color=config["color"],
            linestyle=config["linestyle"],
            marker=config["marker"],
            markersize=5.5,
            markeredgewidth=1.0,
            markerfacecolor="white" if config["group"] == "NWP" else config["color"],
            markeredgecolor=config["color"],
            label=label,
            zorder=5 if config["group"] == "AI" else 4,
        )

    ax.axhline(0.5, color="black", lw=1.0, ls=":", alpha=0.75)
    ax.text(
        0.99,
        0.505,
        "COR = 0.5",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
        transform=ax.get_yaxis_transform(),
    )

    ax.set_xlim(6, 43)
    ax.set_ylim(0.2, 1.02)
    ax.set_xticks([7, 14, 21, 28, 35, 42])
    ax.set_xticklabels(["7\n(W1)", "14\n(W2)", "21\n(W3)", "28\n(W4)", "35\n(W5)", "42\n(W6)"])
    ax.set_xlabel("Lead time (days / week)")
    ax.set_ylabel("MJO Bivariate Correlation")
    ax.set_title("Weekly MJO Skill from Local Eval Files")

    ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(7))
    ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))

    ax.legend(
        loc="upper right",
        framealpha=0.93,
        edgecolor="#cccccc",
        handlelength=2.4,
        borderpad=0.6,
        labelspacing=0.4,
    )

    fig.text(
        0.012,
        0.01,
        "IFS uses a U850/U200 wind-only proxy because the local eval file lacks pred_olr. "
        "GEFS and IFS are available only through day 14.",
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )

    plt.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
    fig.savefig(PLOT_PNG, dpi=600, bbox_inches="tight")
    fig.savefig(PLOT_SVG, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved plot: {PLOT_PNG}")
    print(f"Saved plot: {PLOT_SVG}")


def main() -> None:
    ref = load_reference(CACHE_PATH)

    summary_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []

    canglong_summary, canglong_yearly = compute_canglong(ref, MODEL_CONFIGS[0]["path"])
    summary_rows.extend(canglong_summary)
    yearly_rows.extend(canglong_yearly)

    for config in MODEL_CONFIGS[1:]:
        summary, yearly = compute_eval_model(config, ref)
        summary_rows.extend(summary)
        yearly_rows.extend(yearly)

    summary_df = pd.DataFrame(summary_rows).sort_values(["model", "lead_week"]).reset_index(drop=True)
    yearly_df = pd.DataFrame(yearly_rows).sort_values(["model", "year", "lead_week"]).reset_index(drop=True)

    summary_df.to_csv(SUMMARY_CSV, index=False)
    yearly_df.to_csv(YEARLY_CSV, index=False)
    print(f"\nSaved summary CSV: {SUMMARY_CSV}")
    print(f"Saved yearly CSV: {YEARLY_CSV}")
    print("\nSummary:")
    print(summary_df.to_string(index=False))

    render_plot(summary_df, yearly_df)


if __name__ == "__main__":
    main()
