from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from typing import Optional, List, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_BASE = "/8202302003"

def _resolve_path(*parts: str) -> str:
    p = os.path.join(*parts)
    if os.path.exists(p):
        return p
    alt = os.path.join(SCRIPT_DIR, os.path.basename(p))
    return alt if os.path.exists(alt) else p

def get_data_dir() -> str:
    for base in [SERVER_BASE, SCRIPT_DIR]:
        p_est = os.path.join(base, "fan_vote_estimates.csv")
        if os.path.exists(p_est):
            return base
    return SCRIPT_DIR

DATA_DIR = get_data_dir()
OUT_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 300
FONT_SIZE = 10
TITLE_SIZE = 11
COLORS = {
    "mean": "#2E86AB",
    "interval": "#A23B72",
    "consistent": "#28A745",
    "inconsistent": "#DC3545",
    "rank": "#6C5CE7",
    "percent": "#00B894",
    "grid": "#E0E0E0",
    "text": "#333333",
}
CB_PALETTE = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#FBAFE4", "#949494", "#ECE133"]

def apply_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.linewidth": 1.0,
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": COLORS["text"],
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.facecolor": "white",
        "figure.dpi": 100,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.7,
    })

def load_estimates() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "fan_vote_estimates.csv")
    df = pd.read_csv(path)
    df["fan_share_width"] = df["fan_share_ub"] - df["fan_share_lb"]
    return df

def load_bounds() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "fan_vote_bounds.csv")
    return pd.read_csv(path)

def load_consistency() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "consistency_report.csv")
    return pd.read_csv(path)


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.05, y: float = 1.18):
    ax.text(
        x, y, f"({label})",
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def fig_estimates_timeseries(estimates: pd.DataFrame, seasons: Optional[List[int]] = None, max_contestants: int = 8, axes=None):
    if seasons is None:
        seasons = sorted(estimates["season"].unique())[:4]
    est = estimates[estimates["season"].isin(seasons)].copy()
    if axes is not None:
        fig = axes[0].get_figure()
        axes_flat = axes if len(axes) == 4 else axes.ravel()
    else:
        fig, axes_flat = plt.subplots(2, 2, figsize=(7, 5.5), sharex=False, sharey=True)
        axes_flat = axes_flat.ravel()
    for idx, season in enumerate(seasons):
        if idx >= 4:
            break
        ax = axes_flat[idx]
        sub = est[est["season"] == season].sort_values(["week", "celebrity_name"])
        contestants = sub["celebrity_name"].unique().tolist()
        if len(contestants) > max_contestants:
            weeks_per = sub.groupby("celebrity_name")["week"].count()
            contestants = weeks_per.nlargest(max_contestants).index.tolist()
            sub = sub[sub["celebrity_name"].isin(contestants)]
        colors = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(contestants))]
        for i, name in enumerate(contestants):
            csub = sub[sub["celebrity_name"] == name].sort_values("week")
            if csub.empty:
                continue
            w = csub["week"].values
            mu = csub["fan_share_mean"].values
            std = csub["fan_share_std"].values
            ax.plot(w, mu, "-o", color=colors[i], label=name[:14], markersize=3, linewidth=1.5)
            ax.fill_between(w, mu - std, mu + std, color=colors[i], alpha=0.2)
        ax.set_xlabel("Week")
        ax.set_ylabel("Fan vote share")
        ax.set_title(f"Season {season}")
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.legend(loc="upper right", fontsize=6, ncol=1)
        ax.grid(True, alpha=0.5)
        ax.set_axisbelow(True)
    for j in range(len(seasons), 4):
        axes_flat[j].set_visible(False)
    if axes is None:
        fig.suptitle("Estimated fan vote share over time (mean ± 1 SD)", fontsize=TITLE_SIZE, y=1.02)
        plt.tight_layout()
    return fig


def fig_feasible_width(estimates: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
    else:
        fig = ax.get_figure()
    for method, label in [("rank", "Rank-based"), ("percent", "Percentage-based")]:
        sub = estimates[estimates["voting_method"] == method]["fan_share_width"]
        if sub.empty:
            continue
        ax.hist(sub.clip(0, 1), bins=40, alpha=0.7, label=label, color=COLORS[method], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Feasible interval width (upper − lower bound)")
    ax.set_ylabel("Count")
    ax.set_title("Uncertainty: width of feasible fan-share region")
    ax.legend()
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5, axis="y")
    if ax is None:
        plt.tight_layout()
    return fig


def fig_interval_one_week(bounds: pd.DataFrame, estimates: pd.DataFrame, season: int = 1, week: int = 2, ax=None):
    sub_b = bounds[(bounds["season"] == season) & (bounds["week"] == week)].sort_values("fan_share_lb")
    sub_e = estimates[(estimates["season"] == season) & (estimates["week"] == week)].set_index("celebrity_name")
    if sub_b.empty:
        return None
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5, max(3, 0.35 * len(sub_b))))
    else:
        fig = ax.get_figure()
    y_pos = np.arange(len(sub_b))
    names = sub_b["celebrity_name"].tolist()
    lb = sub_b["fan_share_lb"].values
    ub = sub_b["fan_share_ub"].values
    for i, (l, u) in enumerate(zip(lb, ub)):
        ax.barh(i, u - l, left=l, height=0.5, color=COLORS["interval"], alpha=0.5, edgecolor=COLORS["interval"], linewidth=0.8)
    means = []
    for n in names:
        if n in sub_e.index:
            means.append(sub_e.loc[n, "fan_share_mean"])
        else:
            means.append((sub_b[sub_b["celebrity_name"] == n]["fan_share_lb"].values[0] + sub_b[sub_b["celebrity_name"] == n]["fan_share_ub"].values[0]) / 2)
    ax.scatter(means, y_pos, color=COLORS["mean"], s=28, zorder=3, label="Point estimate")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n[:20] for n in names], fontsize=8)
    ax.set_xlabel("Fan vote share")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_title(f"Season {season}, Week {week}: feasible region (bar) and point estimate (dot)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.5, axis="x")
    ax.set_axisbelow(True)
    if created:
        plt.tight_layout()
    return fig


def fig_consistency_by_season(consistency: pd.DataFrame, ax=None):
    agg = consistency.groupby("season").agg(
        consistent=("consistent", "sum"),
        total=("consistent", "count"),
    ).reset_index()
    agg["inconsistent"] = agg["total"] - agg["consistent"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.2))
    else:
        fig = ax.get_figure()
    x = agg["season"].values
    w = 0.5
    ax.bar(x - w/2, agg["consistent"], width=w, label="Consistent", color=COLORS["consistent"], edgecolor="white", linewidth=0.5)
    ax.bar(x - w/2, agg["inconsistent"], width=w, bottom=agg["consistent"], label="Inconsistent", color=COLORS["inconsistent"], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Season")
    ax.set_ylabel("Number of weeks")
    ax.set_title("Model consistency: weeks matching elimination rule")
    ax.legend(loc="upper right")
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5, axis="y")
    if ax is None:
        plt.tight_layout()
    return fig


def fig_consistency_by_method(consistency: pd.DataFrame, ax=None):
    agg = consistency.groupby("voting_method").agg(
        consistent=("consistent", "sum"),
        total=("consistent", "count"),
    ).reset_index()
    agg["pct"] = 100 * agg["consistent"] / agg["total"]
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.8, 3))
    else:
        fig = ax.get_figure()
    x = np.arange(len(agg))
    bars = ax.bar(x, agg["pct"], color=[COLORS["rank"], COLORS["percent"]][:len(agg)], edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["voting_method"].str.capitalize())
    ax.set_ylabel("Consistent weeks (%)")
    ax.set_title("Consistency rate by voting method")
    ax.set_ylim(0, 105)
    for b, v in zip(bars, agg["pct"]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{v:.1f}%", ha="center", fontsize=9)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5, axis="y")
    if ax is None:
        plt.tight_layout()
    return fig


def fig_smoothed_vs_raw(estimates: pd.DataFrame, season: int = 1, ax=None):
    sub = estimates[estimates["season"] == season].copy()
    if sub.empty or "fan_share_mean_raw" not in sub.columns:
        return None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    else:
        fig = ax.get_figure()
    contestants = sub["celebrity_name"].unique()[:6]
    for i, name in enumerate(contestants):
        csub = sub[sub["celebrity_name"] == name].sort_values("week")
        if csub.empty:
            continue
        ax.plot(csub["week"], csub["fan_share_mean_raw"], "o--", color=CB_PALETTE[i % len(CB_PALETTE)], alpha=0.7, markersize=4, label=f"{name[:12]} (raw)")
        ax.plot(csub["week"], csub["fan_share_mean"], "s-", color=CB_PALETTE[i % len(CB_PALETTE)], markersize=4, linewidth=1.5, label=f"{name[:12]} (smoothed)")
    ax.set_xlabel("Week")
    ax.set_ylabel("Fan vote share")
    ax.set_title(f"Season {season}: raw vs temporally smoothed estimate")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(True, alpha=0.5)
    if ax is None:
        plt.tight_layout()
    return fig


def fig_uncertainty_heatmap(estimates: pd.DataFrame, season: int = 1, ax=None, max_contestants: int = 20):
    sub = estimates[(estimates["season"] == season)].copy()
    if sub.empty:
        return None
    if "fan_share_width" not in sub.columns:
        sub["fan_share_width"] = sub["fan_share_ub"] - sub["fan_share_lb"]
    pivot = sub.pivot_table(index="celebrity_name", columns="week", values="fan_share_width", aggfunc="first")
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    pivot = pivot.fillna(np.nan)
    if pivot.size == 0:
        return None
    if len(pivot) > max_contestants:
        pivot = pivot.iloc[:max_contestants]
    pivot.index = [s[:18] + ("…" if len(s) > 18 else "") for s in pivot.index]
    vmax = float(np.nanmax(pivot.values)) if np.any(np.isfinite(pivot.values)) else 1.0
    vmax = min(1.0, vmax * 1.2) if vmax > 0 else 0.1
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6, max(3.5, 0.28 * len(pivot))))
    else:
        fig = ax.get_figure()
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(int))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Week")
    ax.set_ylabel("Contestant")
    ax.set_title(f"Uncertainty (interval width) — Season {season}\n(contestant × week)")
    plt.colorbar(im, ax=ax, label="Width (ub − lb)", shrink=0.7)
    if created:
        plt.tight_layout()
    return fig


def fig_width_by_week(estimates: pd.DataFrame, ax=None, by_season: bool = False):
    df = estimates.copy()
    df["fan_share_width"] = df["fan_share_ub"] - df["fan_share_lb"]
    df["fan_share_width"] = df["fan_share_width"].clip(0, 1)
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(6, 3.5))
    else:
        fig = ax.get_figure()
    if by_season:
        seasons = sorted(df["season"].unique())[:8]
        for s in seasons:
            sub = df[df["season"] == s]
            if sub.empty:
                continue
            ax.scatter(sub["week"], sub["fan_share_width"], alpha=0.4, s=12, label=f"S{s}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Interval width (ub − lb)")
        ax.set_title("Uncertainty by week (per season)")
    else:
        weeks = sorted(df["week"].unique())
        data = [df[df["week"] == w]["fan_share_width"].dropna().values for w in weeks]
        bp = ax.boxplot(data, positions=weeks, widths=0.5, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(COLORS["interval"])
            patch.set_alpha(0.7)
        ax.set_xlabel("Week")
        ax.set_ylabel("Interval width (ub − lb)")
        ax.set_title("Uncertainty: interval width distribution by week")
        ax.set_xticks(weeks)
        ax.set_xticklabels(weeks)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5, axis="y")
    if by_season:
        ax.legend(loc="upper right", fontsize=8)
    if created:
        plt.tight_layout()
    return fig


def fig_overview_heatmap(estimates: pd.DataFrame, axes=None):
    by_season = estimates.groupby("season").agg(
        mean_share=("fan_share_mean", "mean"),
        mean_width=("fan_share_width", "mean"),
        n_obs=("fan_share_mean", "count"),
    ).reset_index()
    if axes is not None:
        ax1, ax2 = axes
        fig = ax1.get_figure()
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3))
        axes = (ax1, ax2)
    ax1.bar(by_season["season"], by_season["mean_share"], color=COLORS["mean"], alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Season")
    ax1.set_ylabel("Mean fan vote share")
    ax1.set_title("Mean share by season", fontsize=TITLE_SIZE - 1)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.bar(by_season["season"], by_season["mean_width"], color=COLORS["interval"], alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Season")
    ax2.set_ylabel("Mean feasible width")
    ax2.set_title("Mean interval width by season", fontsize=TITLE_SIZE - 1)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.5, axis="y")
    if axes is None:
        plt.tight_layout(pad=1.0, wspace=0.35)
    return fig


def fig_combined(estimates: pd.DataFrame, bounds: pd.DataFrame, consistency: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(5, 3, figure=fig, hspace=0.38, wspace=0.3, left=0.08, right=0.97, top=0.96, bottom=0.04)

    ax_a1 = fig.add_subplot(gs[0, 0])
    ax_a2 = fig.add_subplot(gs[0, 1])
    ax_a3 = fig.add_subplot(gs[1, 0])
    ax_a4 = fig.add_subplot(gs[1, 1])
    fig_estimates_timeseries(estimates, seasons=[1, 2, 3, 4], axes=[ax_a1, ax_a2, ax_a3, ax_a4])
    add_panel_label(ax_a1, "a")

    ax_b = fig.add_subplot(gs[0, 2])
    fig_feasible_width(estimates, ax=ax_b)
    add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[1, 2])
    fig_interval_one_week(bounds, estimates, season=1, week=2, ax=ax_c)
    add_panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[2, 0])
    fig_consistency_by_season(consistency, ax=ax_d)
    add_panel_label(ax_d, "d")

    ax_e = fig.add_subplot(gs[2, 1])
    fig_consistency_by_method(consistency, ax=ax_e)
    add_panel_label(ax_e, "e")

    ax_f = fig.add_subplot(gs[2, 2])
    fig_smoothed_vs_raw(estimates, season=1, ax=ax_f)
    add_panel_label(ax_f, "f")

    ax_g1 = fig.add_subplot(gs[3, 0])
    ax_g2 = fig.add_subplot(gs[3, 1])
    fig_overview_heatmap(estimates, axes=(ax_g1, ax_g2))
    add_panel_label(ax_g1, "g")

    ax_h = fig.add_subplot(gs[3, 2])
    fig_uncertainty_heatmap(estimates, season=1, ax=ax_h, max_contestants=14)
    add_panel_label(ax_h, "h")

    ax_i = fig.add_subplot(gs[4, :])
    fig_width_by_week(estimates, ax=ax_i, by_season=False)
    add_panel_label(ax_i, "i")

    return fig


def save_fig(fig: plt.Figure, name: str):
    for ext in ["pdf", "png"]:
        path = os.path.join(OUT_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main():
    apply_style()
    estimates = load_estimates()
    bounds = load_bounds()
    consistency = load_consistency()

    fig1 = fig_estimates_timeseries(estimates, seasons=[1, 2, 3, 4])
    save_fig(fig1, "fig1_fan_share_timeseries")

    fig2 = fig_feasible_width(estimates)
    save_fig(fig2, "fig2_feasible_width_by_method")

    fig3 = fig_interval_one_week(bounds, estimates, season=1, week=2)
    if fig3 is not None:
        save_fig(fig3, "fig3_interval_one_week")

    fig4 = fig_consistency_by_season(consistency)
    save_fig(fig4, "fig4_consistency_by_season")

    fig5 = fig_consistency_by_method(consistency)
    save_fig(fig5, "fig5_consistency_by_method")

    fig6 = fig_smoothed_vs_raw(estimates, season=1)
    if fig6 is not None:
        save_fig(fig6, "fig6_smoothed_vs_raw")

    fig7 = fig_overview_heatmap(estimates)
    save_fig(fig7, "fig7_overview_by_season")

    fig8 = fig_uncertainty_heatmap(estimates, season=1)
    if fig8 is not None:
        save_fig(fig8, "fig8_uncertainty_heatmap_season1")
    fig8b = fig_uncertainty_heatmap(estimates, season=3)
    if fig8b is not None:
        save_fig(fig8b, "fig8_uncertainty_heatmap_season3")

    fig9 = fig_width_by_week(estimates, by_season=False)
    save_fig(fig9, "fig9_width_by_week")
    fig9b = fig_width_by_week(estimates, by_season=True)
    save_fig(fig9b, "fig9_width_by_week_by_season")

    fig_combined_all = fig_combined(estimates, bounds, consistency)
    save_fig(fig_combined_all, "fig_combined_all")

    print("Figures saved to:", OUT_DIR)
    for f in os.listdir(OUT_DIR):
        if f.endswith((".png", ".pdf")):
            print("  ", f)


if __name__ == "__main__":
    main()
