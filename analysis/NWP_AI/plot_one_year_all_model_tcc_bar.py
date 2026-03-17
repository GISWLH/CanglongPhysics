from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "analysis" / "NWP_AI" / "one_year_equal_length_all_models_tcc.csv"
OUT_PNG = ROOT / "analysis" / "NWP_AI" / "one_year_equal_length_all_models_tcc_bar.png"

MODELS = ["CAS-Canglong", "BOM", "CMA", "ECMWF", "FuXi-S2S", "GEFS", "IFS", "NCEP"]
VARS = ["tp", "t2m", "olr", "z500", "u850", "u200"]
VAR_TITLES = {
    "tp": "TP",
    "t2m": "T2M",
    "olr": "OLR",
    "z500": "Z500",
    "u850": "U850",
    "u200": "U200",
}
LEADS = [1, 2, 3, 4, 5, 6]
LEAD_COLORS = ["#0B3954", "#087E8B", "#BFD7EA", "#FF5A5F", "#C81D25", "#7F1D1D"]


def setup_matplotlib() -> None:
    font_path = "/usr/share/fonts/arial/ARIAL.TTF"
    try:
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(fname=font_path).get_name()
    except Exception:
        font_name = "Arial"

    plt.rcParams.update(
        {
            "font.family": font_name,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {IN_CSV}")

    setup_matplotlib()
    df = pd.read_csv(IN_CSV)
    if "row_type" in df.columns:
        df = df[df["row_type"] == "summary"].copy()

    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharey=False)
    axes = axes.ravel()

    x = np.arange(len(MODELS), dtype=np.float64)
    width = 0.12
    offsets = (np.arange(len(LEADS)) - (len(LEADS) - 1) / 2.0) * width

    for ax, var in zip(axes, VARS):
        sub = df[df["variable"] == var].copy()
        bar_max = 0.0

        for lead_idx, lead in enumerate(LEADS):
            lead_sub = sub[sub["lead"] == lead].set_index("model")
            vals = np.array(
                [
                    lead_sub.loc[m, "tcc_weighted"] if m in lead_sub.index else np.nan
                    for m in MODELS
                ],
                dtype=np.float64,
            )
            xpos = x + offsets[lead_idx]
            finite = np.isfinite(vals)
            if finite.any():
                ax.bar(
                    xpos[finite],
                    vals[finite],
                    width=width,
                    color=LEAD_COLORS[lead_idx],
                    edgecolor="white",
                    linewidth=0.6,
                    label=f"Lead {lead}",
                    zorder=3,
                )
                bar_max = max(bar_max, float(np.nanmax(vals)))

            missing = ~finite
            if missing.any():
                ax.bar(
                    xpos[missing],
                    np.full(missing.sum(), 0.015),
                    width=width,
                    facecolor="none",
                    edgecolor="#C7C7C7",
                    linewidth=0.8,
                    linestyle="--",
                    zorder=2,
                )

        ax.set_title(VAR_TITLES[var], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        ax.set_ylabel("Weighted TCC")
        ax.set_ylim(0.0, max(0.18, min(1.02, bar_max + 0.12)))

        avail_models = sorted(sub.loc[sub["status"] == "ok", "model"].unique().tolist())
        if len(avail_models) < len(MODELS):
            missing_models = [m for m in MODELS if m not in avail_models]
            ax.text(
                0.01,
                0.98,
                "Missing: " + ", ".join(missing_models),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#666666",
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#DDDDDD", "alpha": 0.9},
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=LEAD_COLORS[i], edgecolor="white")
        for i in range(len(LEADS))
    ]
    labels = [f"Lead {lead}" for lead in LEADS]
    fig.legend(handles, labels, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 0.98), frameon=False)

    fig.suptitle(
        "Equal-Length One-Year Benchmark TCC Comparison\n"
        "Anomaly-first, gridpoint temporal correlation, cosine-lat weighted spatial mean",
        y=0.995,
        fontsize=14,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Blank dashed bars indicate missing variable/lead in the source model file. "
        "All statistics are computed on the BOM 72x144 target grid with common ERA5 obs/climatology.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#444444",
    )

    plt.tight_layout(rect=[0.02, 0.04, 0.98, 0.94])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
