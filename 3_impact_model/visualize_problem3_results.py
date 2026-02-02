from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SERVER_BASE = "/8202302003"
if os.path.exists(SERVER_BASE):
    OUT_DIR = os.path.join(SERVER_BASE, "问题三", "outputs")
else:
    OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
FIG_DIR = os.path.join(os.path.dirname(OUT_DIR), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

DPI = 300
FONT_SIZE = 10
TITLE_SIZE = 11
LABEL_PAD = 0.02

READABLE_VAR = {
    "intercept": "Intercept",
    "age": "Age",
    "week_norm": "Week",
    "has_bonus": "Bonus",
    "lag_judge": "Lag (judge)",
    "lag_fan": "Lag (fan)",
}


def _variable_to_readable(var: str) -> str:
    s = str(var).strip()
    if s in READABLE_VAR:
        return READABLE_VAR[s]
    if s.startswith("ind_"):
        return "Industry: " + s[4:].replace("_", " ")
    if s.startswith("reg_"):
        return "Region: " + s[4:].replace("_", " ")
    if s.startswith("sea_"):
        return "Season " + s[4:]
    if s.startswith("pro_"):
        name = s[4:].strip()
        if " (" in name:
            name = name.split(" (")[0].strip()
        return name if len(name) <= 20 else name[:17] + "..."
    return s.replace("_", " ")


def _read_csv(name: str) -> Optional[pd.DataFrame]:
    p = os.path.join(OUT_DIR, name)
    if os.path.isfile(p):
        return pd.read_csv(p, encoding="utf-8-sig")
    return None


def plot_butterfly_coefficient_contrast(ax: plt.Axes) -> None:
    df = _read_csv("key_coefficient_comparison.csv")
    if df is None:
        df = _read_csv("coefficient_comparison_judge_vs_fan.csv")
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No key_coefficient_comparison.csv", ha="center", va="center", transform=ax.transAxes)
        return
    key_vars = ["age", "week_norm", "has_bonus", "lag_judge", "lag_fan"]
    sub = df[df["variable"].isin(key_vars)].copy()
    if sub.empty:
        sub = df.head(8)
    sub = sub.sort_values("variable")
    y_pos = np.arange(len(sub))
    coef_j = sub["coef_judge"].fillna(0).values
    coef_f = sub["coef_fan"].fillna(0).values
    ax.barh(y_pos, -np.asarray(coef_j, dtype=float), height=0.38, color="#2E86AB", label="Judge score", align="center", edgecolor="none")
    ax.barh(y_pos, np.asarray(coef_f, dtype=float), height=0.38, color="#E94F37", label="Fan vote", align="center", edgecolor="none")
    ax.axvline(0, color="#333333", linewidth=1.0)
    labels = sub["variable"].apply(_variable_to_readable)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE)
    ax.set_xlabel("Coefficient (left: judge, right: fan)", fontsize=FONT_SIZE)
    ax.set_title("Judge vs fan: same direction?", fontsize=TITLE_SIZE, fontweight="medium")
    ax.legend(loc="upper right", fontsize=FONT_SIZE - 1, frameon=True, fancybox=False, edgecolor="#ccc")
    xabs = max(np.abs(coef_j).max(), np.abs(coef_f).max(), 0.2)
    ax.set_xlim(-xabs - 0.1, xabs + 0.1)
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_kingmaker_quadrant(ax: plt.Axes) -> None:
    df = _read_csv("pro_dancer_random_effects.csv")
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No pro_dancer_random_effects.csv", ha="center", va="center", transform=ax.transAxes)
        return
    x = df["u_pro_judge"].values
    y = df["u_pro_fan"].values
    names = df["ballroom_partner"].astype(str).str.replace("pro_", "", regex=False)
    names = names.apply(lambda s: s.split(" (")[0].strip() if " (" in s else s.strip())
    ax.scatter(x, y, s=36, c="#2E86AB", alpha=0.82, edgecolors="white", linewidths=0.6, zorder=3)
    dist = np.sqrt(np.asarray(x, dtype=float) ** 2 + np.asarray(y, dtype=float) ** 2)
    n_label = min(10, len(names))
    idx_label = np.argsort(dist)[::-1][:n_label]
    texts = []
    for i in idx_label:
        short_name = (names.iloc[i][:14] + "…") if len(names.iloc[i]) > 14 else names.iloc[i]
        t = ax.annotate(short_name, (float(x[i]), float(y[i])), fontsize=6, xytext=(6, 6), textcoords="offset points",
                        alpha=0.95, annotation_clip=True)
        texts.append(t)
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray", lw=0.5),
                    expand_points=(1.2, 1.2), force_points=(0.3, 0.3))
    except ImportError:
        pass
    ax.axhline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$u_{pro}^J$ (judge score)", fontsize=FONT_SIZE)
    ax.set_ylabel(r"$u_{pro}^F$ (fan vote)", fontsize=FONT_SIZE)
    ax.set_title("Pro dancer 'Kingmaker' effect", fontsize=TITLE_SIZE, fontweight="medium")
    ax.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.text(0.95, 0.95, "Dual stars", transform=ax.transAxes, fontsize=7, ha="right", va="top", style="italic", color="#555")
    ax.text(0.05, 0.95, "Traffic", transform=ax.transAxes, fontsize=7, ha="left", va="top", style="italic", color="#555")
    ax.text(0.95, 0.05, "Technical", transform=ax.transAxes, fontsize=7, ha="right", va="bottom", style="italic", color="#555")


def plot_drivers_of_survival(ax: plt.Axes) -> None:
    df = _read_csv("outcome_weeks_survived_coefs.csv")
    if df is None or df.empty:
        df = _read_csv("outcome_placement_weeks_key_coefs.csv")
        if df is not None and "coef_weeks_survived" in df.columns:
            df = df.rename(columns={"coef_weeks_survived": "coef"})
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No outcome_weeks_survived_coefs.csv", ha="center", va="center", transform=ax.transAxes)
        return
    if "coef" not in df.columns:
        return
    sub = df[df["variable"].astype(str) != "intercept"].copy()
    sub = sub.sort_values("coef", ascending=True)
    if len(sub) > 14:
        sub = sub.reindex(sub["coef"].abs().sort_values(ascending=False).index).head(14).reset_index(drop=True)
    sub = sub.sort_values("coef", ascending=True)
    y_pos = np.arange(len(sub))
    coef = sub["coef"].values
    colors = ["#28A745" if c >= 0 else "#DC3545" for c in coef]
    ax.barh(y_pos, coef, height=0.62, color=colors, edgecolor="none")
    ax.axvline(0, color="#333333", linewidth=1.0)
    labels = sub["variable"].astype(str).apply(_variable_to_readable)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE)
    ax.set_xlabel("Coefficient (weeks survived)", fontsize=FONT_SIZE)
    ax.set_title("Drivers of survival", fontsize=TITLE_SIZE, fontweight="medium")
    ax.grid(axis="x", alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def build_combined_figure() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.facecolor": "#FAFAFA",
        "figure.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
    })
    fig = plt.figure(figsize=(15, 5.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.35, 1], wspace=0.38, left=0.06, right=0.97, top=0.86, bottom=0.2)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])]
    plot_butterfly_coefficient_contrast(axes[0])
    plot_kingmaker_quadrant(axes[1])
    plot_drivers_of_survival(axes[2])
    for i, ax in enumerate(axes):
        label = "(" + chr(97 + i) + ")"
        ax.text(LABEL_PAD, 1 - LABEL_PAD, label, transform=ax.transAxes, fontsize=14, fontweight="bold",
                va="top", ha="left", bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="gray", linewidth=0.5))
    out_path = os.path.join(FIG_DIR, "problem3_impact_combined.pdf")
    png_path = os.path.join(FIG_DIR, "problem3_impact_combined.png")
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path, "and", png_path)


if __name__ == "__main__":
    build_combined_figure()
