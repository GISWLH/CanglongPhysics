"""
Plot selected MJO skill curves in one panel.

Data source:
  - analysis/MJO/mjo.csv
"""

import os

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import blended_transform_factory

try:
    from scipy.interpolate import PchipInterpolator
except ImportError:  # pragma: no cover
    PchipInterpolator = None


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MJO_CSV = os.path.join(ROOT, "analysis/MJO/mjo.csv")
OUT_PNG = os.path.join(ROOT, "analysis/MJO/mjo_skill_comparison.png")
OUT_SVG = os.path.join(ROOT, "analysis/MJO/mjo_skill_comparison.svg")


def setup_style():
    for font_path in [
        "/usr/share/fonts/arial/ARIAL.TTF",
        "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf",
    ]:
        if os.path.exists(font_path):
            font_manager.fontManager.addfont(font_path)

    sns.set_theme(style="white", context="paper")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.labelsize": 11.5,
        "axes.titlesize": 12.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.2,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "lines.linewidth": 2.2,
        "axes.linewidth": 1.0,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#202124",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 4.8,
        "ytick.major.size": 4.8,
        "xtick.minor.size": 2.6,
        "ytick.minor.size": 2.6,
        "xtick.major.width": 0.95,
        "ytick.major.width": 0.95,
        "xtick.minor.width": 0.75,
        "ytick.minor.width": 0.75,
        "xtick.color": "#4B4B4B",
        "ytick.color": "#4B4B4B",
        "grid.color": "#D7DCE2",
        "grid.linewidth": 0.85,
        "grid.alpha": 0.9,
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    })
    mpl.rcParams["svg.fonttype"] = "none"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42


def extend_from_origin(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size == 0:
        return x, y
    if np.isclose(x[0], 0.0):
        return x, y
    return np.concatenate([[0.0], x]), np.concatenate([[1.0], y])


def smooth_curve(x, y, n_points=320):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size <= 2:
        return x, y

    x_dense = np.linspace(x.min(), x.max(), n_points)
    if PchipInterpolator is not None:
        y_dense = PchipInterpolator(x, y)(x_dense)
    else:
        y_dense = np.interp(x_dense, x, y)
    return x_dense, y_dense


def add_line_glow(line, width_pad=2.6, alpha=0.95):
    line.set_path_effects([
        pe.Stroke(linewidth=line.get_linewidth() + width_pad, foreground="white", alpha=alpha),
        pe.Normal(),
    ])


def plot_reference_series(
    ax,
    x,
    y,
    color,
    marker,
    linestyle,
    label,
    zorder,
    extend_origin=True,
    line_width=2.5,
):
    if extend_origin:
        x_plot, y_plot = extend_from_origin(x, y)
    else:
        x_plot = np.asarray(x, dtype=np.float64)
        y_plot = np.asarray(y, dtype=np.float64)

    mask = np.isfinite(x_plot) & np.isfinite(y_plot)
    x_plot = x_plot[mask]
    y_plot = y_plot[mask]
    if x_plot.size == 0:
        return

    x_s, y_s = smooth_curve(x_plot, y_plot)
    line, = ax.plot(
        x_s,
        np.clip(y_s, 0.0, 1.02),
        color=color,
        lw=line_width,
        ls=linestyle,
        label=label,
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=zorder,
    )
    add_line_glow(line, width_pad=2.8)

    week_mask = np.isclose(np.mod(x_plot, 7.0), 0.0)
    if not np.any(week_mask):
        week_mask = np.arange(x_plot.size) % max(1, x_plot.size // 6) == 0
    ax.scatter(
        x_plot[week_mask],
        y_plot[week_mask],
        s=42,
        marker=marker,
        facecolor="white",
        edgecolor=color,
        linewidth=1.35,
        zorder=zorder + 0.4,
    )


def plot_weekly_series(
    ax,
    df,
    y_col,
    std_col,
    color,
    marker,
    linestyle,
    label,
    zorder,
    extend_origin=True,
    line_width=2.4,
    band_alpha=0.14,
):
    cols = ["day", y_col]
    if std_col is not None and std_col in df.columns:
        cols.append(std_col)

    sub = df[cols].dropna(subset=[y_col]).copy()
    if sub.empty:
        return

    if std_col is not None and std_col in sub.columns:
        ci = sub.dropna(subset=[std_col])
        if not ci.empty:
            ci_x = ci["day"].values
            ci_lo = (ci[y_col] - ci[std_col]).values
            ci_hi = (ci[y_col] + ci[std_col]).values
            if extend_origin:
                ci_x, ci_lo = extend_from_origin(ci_x, ci_lo)
                _, ci_hi = extend_from_origin(ci["day"].values, (ci[y_col] + ci[std_col]).values)
            ci_x_s, ci_lo_s = smooth_curve(ci_x, ci_lo, n_points=280)
            _, ci_hi_s = smooth_curve(ci_x, ci_hi, n_points=280)
            ax.fill_between(
                ci_x_s,
                np.clip(ci_lo_s, 0.0, 1.02),
                np.clip(ci_hi_s, 0.0, 1.02),
                color=color,
                alpha=band_alpha,
                linewidth=0.0,
                zorder=zorder - 1,
            )

    if extend_origin:
        x_plot, y_plot = extend_from_origin(sub["day"].values, sub[y_col].values)
    else:
        x_plot = np.asarray(sub["day"].values, dtype=np.float64)
        y_plot = np.asarray(sub[y_col].values, dtype=np.float64)

    x_s, y_s = smooth_curve(x_plot, y_plot, n_points=280)
    line, = ax.plot(
        x_s,
        np.clip(y_s, 0.0, 1.02),
        color=color,
        lw=line_width,
        ls=linestyle,
        label=label,
        solid_capstyle="round",
        dash_capstyle="round",
        zorder=zorder,
    )
    add_line_glow(line, width_pad=3.0 if label == "Canglong" else 2.6)

    ax.scatter(
        x_plot,
        y_plot,
        s=48 if label != "Canglong" else 54,
        marker=marker,
        facecolor="white",
        edgecolor=color,
        linewidth=1.4,
        zorder=zorder + 0.5,
    )


def style_axis(ax):
    week_edges = [0, 7, 14, 21, 28, 35, 42]
    for idx, start in enumerate(week_edges[:-1]):
        if idx % 2 == 0:
            ax.axvspan(start, week_edges[idx + 1], color="#F5F7FB", alpha=0.95, zorder=0)

    for x0 in week_edges[1:-1]:
        ax.axvline(x0, color="#E3E8EF", lw=0.9, zorder=0)

    ax.axhline(0.5, color="#A55D39", lw=1.15, ls=(0, (3.0, 2.4)), alpha=0.85, zorder=1)
    threshold_transform = blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(
        1.01,
        0.5,
        "Skill threshold = 0.5",
        transform=threshold_transform,
        ha="left",
        va="center",
        fontsize=9.2,
        color="#8F5B3A",
    )

    ax.set_xlim(0, 42)
    ax.set_ylim(0.22, 1.02)
    ax.set_xticks(week_edges)
    ax.set_xlabel("Lead time (days)")
    ax.set_ylabel("MJO bivariate correlation")

    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax.grid(axis="y", which="major")
    ax.grid(axis="y", which="minor", color="#EEF2F6", linewidth=0.65)
    ax.tick_params(axis="both", which="major", pad=4.5)

    secax = ax.secondary_xaxis("top")
    secax.set_xticks(week_edges, labels=[f"W{i}" for i in range(len(week_edges))])
    secax.tick_params(axis="x", length=0, pad=6, labelsize=9.3, colors="#6B7280")
    secax.spines["top"].set_visible(False)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#444444")


def main():
    setup_style()
    df = pd.read_csv(MJO_CSV)

    colors = {
        "fengwu": "#355CDE",
        "fuxi": "#F08C00",
        "ecmwf": "#8A8F98",
        "canglong": "#2B9348",
        "gefs": "#17A2B8",
        "ifs": "#9A6B53",
    }

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    style_axis(ax)

    plot_reference_series(
        ax=ax,
        x=df["day"].values,
        y=df["FengWu-W2S_m_ACC"].values,
        color=colors["fengwu"],
        marker="o",
        linestyle="-",
        label="FengWu-W2S$_m$",
        zorder=6,
        line_width=2.8,
    )
    plot_reference_series(
        ax=ax,
        x=df["day"].values,
        y=df["FuXi-S2S_COR"].values,
        color=colors["fuxi"],
        marker="s",
        linestyle="-",
        label="FuXi-S2S",
        zorder=5,
        line_width=2.6,
    )
    plot_reference_series(
        ax=ax,
        x=df["day"].values,
        y=df["ECMWF_fuxi_COR"].values,
        color=colors["ecmwf"],
        marker="^",
        linestyle=(0, (4.6, 2.2)),
        label="ECMWF",
        zorder=4,
        line_width=2.3,
    )
    plot_weekly_series(
        ax=ax,
        df=df,
        y_col="GEFS_COR",
        std_col="GEFS_COR_std",
        color=colors["gefs"],
        marker="^",
        linestyle=(0, (4.0, 2.0)),
        label="GEFS",
        zorder=7,
        line_width=2.3,
        band_alpha=0.12,
    )
    plot_weekly_series(
        ax=ax,
        df=df,
        y_col="IFS_proxy_COR",
        std_col="IFS_proxy_std",
        color=colors["ifs"],
        marker="D",
        linestyle=(0, (1.0, 1.8)),
        label="IFS",
        zorder=8,
        line_width=2.2,
        band_alpha=0.11,
    )
    plot_weekly_series(
        ax=ax,
        df=df,
        y_col="Canglong_V35_COR",
        std_col="Canglong_COR_std",
        color=colors["canglong"],
        marker="o",
        linestyle="-",
        label="Canglong",
        zorder=9,
        line_width=2.8,
        band_alpha=0.16,
    )

    legend = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=3,
        frameon=False,
        columnspacing=1.3,
        handlelength=2.5,
        handletextpad=0.6,
    )
    for handle in legend.legend_handles:
        if hasattr(handle, "set_linewidth"):
            handle.set_linewidth(2.6)

    plt.savefig(OUT_PNG)
    plt.savefig(OUT_SVG)
    print("Saved: mjo_skill_comparison.png / .svg")


if __name__ == "__main__":
    main()
