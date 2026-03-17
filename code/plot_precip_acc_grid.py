#!/usr/bin/env python3
"""
Plot 2x2 precipitation ACC panels directly from verification CSV files.
"""

from pathlib import Path
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import pandas as pd


def setup_font():
    font_name = None
    noto_path = "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc"
    if Path(noto_path).exists():
        font_manager.fontManager.addfont(noto_path)
        font_name = font_manager.FontProperties(fname=noto_path).get_name()
    else:
        arial_path = "/usr/share/fonts/arial/ARIAL.TTF"
        if Path(arial_path).exists():
            font_manager.fontManager.addfont(arial_path)
            font_name = font_manager.FontProperties(fname=arial_path).get_name()
    if font_name:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [font_name]


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "Model" not in df.columns or "Lead_Time" not in df.columns:
        raise ValueError(f"Missing required columns in {csv_path}")
    return df


def title_from_csv(csv_path):
    stem = csv_path.stem
    if "to_" not in stem:
        raise ValueError(f"Cannot parse end date from {csv_path}")
    end_date_str = stem.split("to_")[-1]
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    report_date = end_date + timedelta(days=8)
    return report_date.strftime("%Y-%m-%d")


def plot_panel(ax, df, title, colors, metric_key, y_ticks, y_limits):
    for model in ["CAS-Canglong", "ECMWF"]:
        model_df = df[df["Model"] == model].sort_values("Lead_Time")
        if model_df.empty:
            continue
        label = model
        line_width = 5.0
        marker_size = 12
        marker_edge_width = 2.0
        if model == "CAS-Canglong":
            label = r"$\bf{CAS-Canglong}$"
            line_width = 7.0
            marker_size = 16
            marker_edge_width = 2.5
        ax.plot(
            model_df["Lead_Time"],
            model_df[metric_key],
            "o-",
            color=colors[model],
            markersize=marker_size,
            linewidth=line_width,
            markeredgewidth=marker_edge_width,
            label=label,
        )
    ax.set_title(title, fontsize=34)
    ax.set_xticks([1, 3, 5])
    ax.set_yticks(y_ticks)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=34)
    ax.set_ylim(*y_limits)
    ax.grid(False)


def main():
    setup_font()

    figures_dir = Path("/home/lhwang/Desktop/CanglongPhysics/figures/hindcast_china")
    csv_files = [
        figures_dir / "verification_2025-06-18_to_2025-06-24.csv",
        figures_dir / "verification_2025-06-25_to_2025-07-01.csv",
        figures_dir / "verification_NAS_2025-07-16_to_2025-07-22.csv",
        figures_dir / "verification_NAS_2025-07-23_to_2025-07-29.csv",
    ]

    for csv_path in csv_files:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

    colors = {"CAS-Canglong": "#1f77b4", "ECMWF": "#d62728"}

    def plot_grid(metric_key, output_name, y_ticks, y_limits):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
        axes = axes.ravel()

        for ax, csv_path in zip(axes, csv_files):
            df = load_csv(csv_path)
            if metric_key not in df.columns:
                raise ValueError(f"Missing column {metric_key} in {csv_path}")
            title = title_from_csv(csv_path)
            plot_panel(ax, df, title, colors, metric_key, y_ticks, y_limits)

        axes[1].legend(
            loc="upper right",
            frameon=False,
            fontsize=18,
        )

        plt.tight_layout()
        output_path = figures_dir / output_name
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_path}")

    plot_grid(
        "Precip_ACC",
        "precipitation_ACC_2x2_grid.png",
        y_ticks=[0.2, 0.5, 0.8],
        y_limits=(0.2, 0.8),
    )
    plot_grid(
        "Temp_ACC",
        "temperature_ACC_2x2_grid.png",
        y_ticks=[0.8, 0.9, 1.0],
        y_limits=(0.8, 1.0),
    )
    plot_grid(
        "SPEI_Agreement",
        "SPEI_agreement_2x2_grid.png",
        y_ticks=[0.4, 0.6, 0.8],
        y_limits=(0.4, 0.9),
    )


if __name__ == "__main__":
    main()
