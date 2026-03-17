from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
IN_CSV = ROOT / "analysis" / "NWP_AI" / "clean_group_tcc_tp_t2m_weighted.csv"
OUT_PNG = ROOT / "analysis" / "NWP_AI" / "clean_group_tcc_tp_t2m_lead2_6_box.png"

MODELS = ["CAS-Canglong", "CMA", "ECMWF", "FuXi-S2S"]
VARS = ["tp", "t2m"]
VAR_LABELS = {"tp": "TP", "t2m": "T2M"}
LEADS = [2, 3, 4, 5, 6]
MODEL_COLORS = {
    "CAS-Canglong": "#0072B2",
    "CMA": "#E69F00",
    "ECMWF": "#009E73",
    "FuXi-S2S": "#D55E00",
}


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


def draw_single_box(
    ax: plt.Axes,
    values: np.ndarray,
    pos: float,
    width: float,
    color: str,
) -> None:
    bp = ax.boxplot(
        [values],
        positions=[pos],
        widths=width,
        patch_artist=True,
        manage_ticks=False,
        showfliers=True,
        boxprops={"facecolor": color, "edgecolor": color, "linewidth": 1.0, "alpha": 0.82},
        whiskerprops={"color": color, "linewidth": 1.0},
        capprops={"color": color, "linewidth": 1.0},
        medianprops={"color": "white", "linewidth": 1.4},
        flierprops={
            "marker": "o",
            "markersize": 2.2,
            "markerfacecolor": color,
            "markeredgecolor": color,
            "alpha": 0.45,
        },
    )
    for artist in bp["boxes"] + bp["whiskers"] + bp["caps"] + bp["medians"] + bp["fliers"]:
        artist.set_zorder(3)


def main() -> None:
    if not IN_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {IN_CSV}")

    setup_matplotlib()
    df = pd.read_csv(IN_CSV)
    sample_df = df[
        (df["row_type"] == "sample")
        & (df["variable"].isin(VARS))
        & (df["lead"].isin(LEADS))
        & (df["model"].isin(MODELS))
    ].copy()
    summary_df = df[
        (df["row_type"] == "summary")
        & (df["variable"].isin(VARS))
        & (df["lead"].isin(LEADS))
        & (df["model"].isin(MODELS))
    ].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), sharey=False)
    x = np.arange(len(LEADS), dtype=np.float64)
    width = 0.16
    offsets = (np.arange(len(MODELS)) - (len(MODELS) - 1) / 2.0) * width

    for ax, var in zip(axes, VARS):
        panel_sample = sample_df[sample_df["variable"] == var]
        panel_summary = summary_df[summary_df["variable"] == var]

        y_min = np.inf
        y_max = -np.inf

        for model_idx, model in enumerate(MODELS):
            model_sample = panel_sample[panel_sample["model"] == model]
            model_summary = panel_summary[panel_summary["model"] == model].set_index("lead").reindex(LEADS)
            positions = x + offsets[model_idx]

            for lead_idx, lead in enumerate(LEADS):
                vals = (
                    model_sample.loc[model_sample["lead"] == lead, "sample_tcc"]
                    .dropna()
                    .to_numpy(dtype=np.float64)
                )
                if vals.size == 0:
                    continue

                draw_single_box(ax, vals, positions[lead_idx], width * 0.92, MODEL_COLORS[model])
                y_min = min(y_min, float(np.nanmin(vals)))
                y_max = max(y_max, float(np.nanmax(vals)))

            summary_vals = model_summary["tcc_weighted"].to_numpy(dtype=np.float64)
            finite = np.isfinite(summary_vals)
            if finite.any():
                ax.scatter(
                    positions[finite],
                    summary_vals[finite],
                    marker="D",
                    s=26,
                    facecolor="white",
                    edgecolor="#111111",
                    linewidth=0.9,
                    zorder=4,
                )
                y_min = min(y_min, float(np.nanmin(summary_vals[finite])))
                y_max = max(y_max, float(np.nanmax(summary_vals[finite])))

        ax.set_title(VAR_LABELS[var], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Lead {lead}" for lead in LEADS])
        ax.set_ylabel("TCC")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        ax.axhline(0.0, color="#666666", linewidth=0.8)
        if np.isfinite(y_min) and np.isfinite(y_max):
            bottom = min(-0.08, y_min - 0.05)
            top = min(1.0, y_max + 0.08)
            if top <= bottom:
                top = bottom + 0.2
            ax.set_ylim(bottom, top)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor=MODEL_COLORS[m], label=m, alpha=0.82) for m in MODELS]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="D",
            color="none",
            markerfacecolor="white",
            markeredgecolor="#111111",
            markeredgewidth=0.9,
            markersize=6,
            label="Weighted mean",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.suptitle(
        "Clean-Group TCC Boxplot for TP and T2M\n"
        "Lead 2-6 weeks, boxes from 50 random 2.5-degree samples, diamond = weighted mean",
        y=1.08,
        fontsize=13,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PNG, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
