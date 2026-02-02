from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import List, Dict, Tuple, Optional, Literal

SERVER_BASE = "/8202302003"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if os.path.exists(SERVER_BASE):
    DATA_CLEAN_DIR = os.path.join(SERVER_BASE, "数据清洗")
    Q1_DIR = os.path.join(SERVER_BASE, "问题一")
    OUT_DIR = os.path.join(SERVER_BASE, "问题二", "outputs")
else:
    DATA_CLEAN_DIR = os.path.join(PROJECT_ROOT, "数据清洗")
    Q1_DIR = os.path.join(PROJECT_ROOT, "问题一")
    OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

os.makedirs(OUT_DIR, exist_ok=True)


def _path(name: str, *subdirs: str) -> str:
    if os.path.exists(SERVER_BASE):
        flat = os.path.join(SERVER_BASE, name)
        if os.path.exists(flat):
            return flat
        for sub in ("数据清洗", "问题一"):
            p = os.path.join(SERVER_BASE, sub, name)
            if os.path.exists(p):
                return p
    for base in [Q1_DIR, DATA_CLEAN_DIR, SCRIPT_DIR, PROJECT_ROOT]:
        p = os.path.join(base, *subdirs, name) if subdirs else os.path.join(base, name)
        if os.path.exists(p):
            return p
    return os.path.join(SCRIPT_DIR, name)


def load_dwts_long() -> pd.DataFrame:
    path = _path("dwts_long.csv")
    return pd.read_csv(path)


def load_dwts_cleaned() -> pd.DataFrame:
    path = _path("dwts_cleaned.csv")
    return pd.read_csv(path)


def load_fan_vote_estimates() -> pd.DataFrame:
    path = _path("fan_vote_estimates.csv")
    return pd.read_csv(path)


def _rank_desc(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    return rankdata(-x, method="min").astype(float)


def elimination_rank(judge_scores: np.ndarray, fan_votes: np.ndarray) -> int:
    jr = _rank_desc(judge_scores)
    fr = _rank_desc(fan_votes)
    combined = jr + fr
    return int(np.argmax(combined))


def elimination_percent(judge_scores: np.ndarray, fan_votes: np.ndarray) -> int:
    j = np.asarray(judge_scores, dtype=float).ravel()
    f = np.asarray(fan_votes, dtype=float).ravel()
    js, fs = np.nansum(j), np.nansum(f)
    if js <= 0 or fs <= 0:
        return 0
    cp = j / js + f / fs
    return int(np.argmin(cp))


def get_worst_k_rank(judge_scores: np.ndarray, fan_votes: np.ndarray, k: int) -> List[int]:
    jr = _rank_desc(judge_scores)
    fr = _rank_desc(fan_votes)
    combined = jr + fr
    idx = np.argsort(-combined)[:k]
    return [int(i) for i in idx]


def get_worst_k_percent(judge_scores: np.ndarray, fan_votes: np.ndarray, k: int) -> List[int]:
    j = np.asarray(judge_scores, dtype=float).ravel()
    f = np.asarray(fan_votes, dtype=float).ravel()
    js, fs = np.nansum(j), np.nansum(f)
    if js <= 0 or fs <= 0:
        return list(range(min(k, len(j))))
    cp = j / js + f / fs
    idx = np.argsort(cp)[:k]
    return [int(i) for i in idx]


def get_bottom_two_rank(judge_scores: np.ndarray, fan_votes: np.ndarray) -> Tuple[int, int]:
    jr = _rank_desc(judge_scores)
    fr = _rank_desc(fan_votes)
    combined = jr + fr
    idx = np.argsort(-combined)[:2]
    return int(idx[0]), int(idx[1])


def get_bottom_two_percent(judge_scores: np.ndarray, fan_votes: np.ndarray) -> Tuple[int, int]:
    j = np.asarray(judge_scores, dtype=float).ravel()
    f = np.asarray(fan_votes, dtype=float).ravel()
    js, fs = np.nansum(j), np.nansum(f)
    if js <= 0 or fs <= 0:
        return 0, 1
    cp = j / js + f / fs
    idx = np.argsort(cp)[:2]
    return int(idx[0]), int(idx[1])


def p_judges_save_a(score_a: float, score_b: float) -> float:
    if score_a > score_b:
        return 1.0
    if score_a == score_b:
        return 0.5
    return 0.0


def p_judges_save_a_logit(score_a: float, score_b: float, beta: float = 5.0) -> float:
    if beta <= 0:
        return 0.5
    diff = score_a - score_b
    return 1.0 / (1.0 + np.exp(-beta * diff))


def apply_judges_save(
    judge_scores: np.ndarray,
    bottom_i: int,
    bottom_j: int,
    rng: Optional[np.random.Generator] = None,
) -> int:
    if rng is None:
        rng = np.random.default_rng()
    ji, jj = float(judge_scores[bottom_i]), float(judge_scores[bottom_j])
    p_save_i = p_judges_save_a(ji, jj)
    if p_save_i >= 1.0:
        return bottom_j
    if p_save_i <= 0.0:
        return bottom_i
    return bottom_j if rng.random() < p_save_i else bottom_i


MethodType = Literal["rank", "percent", "rank_save", "percent_save"]


def evaluate_outcome(
    judge_scores: np.ndarray,
    fan_votes: np.ndarray,
    method: MethodType,
    rng: Optional[np.random.Generator] = None,
) -> int:
    n = len(judge_scores)
    if n <= 1:
        return 0
    if method == "rank":
        return elimination_rank(judge_scores, fan_votes)
    if method == "percent":
        return elimination_percent(judge_scores, fan_votes)
    if method == "rank_save":
        i, j = get_bottom_two_rank(judge_scores, fan_votes)
        return apply_judges_save(judge_scores, i, j, rng)
    if method == "percent_save":
        i, j = get_bottom_two_percent(judge_scores, fan_votes)
        return apply_judges_save(judge_scores, i, j, rng)
    return elimination_rank(judge_scores, fan_votes)


def get_contestants_still_in(season: int, week: int, df_wide: pd.DataFrame) -> List[str]:
    in_season = df_wide[df_wide["season"] == season]
    out = []
    for _, row in in_season.iterrows():
        last = row.get("last_week_competed")
        if pd.isna(last) or int(last) < week:
            continue
        elim_week = row.get("eliminated_week")
        if pd.notna(elim_week) and int(elim_week) < week:
            continue
        out.append(row["celebrity_name"])
    return out


def get_judge_scores_week(
    season: int, week: int, contestants: List[str], df_long: pd.DataFrame
) -> Tuple[List[str], np.ndarray]:
    grp = df_long[(df_long["season"] == season) & (df_long["week"] == week)]
    names, scores = [], []
    for name in contestants:
        row = grp[grp["celebrity_name"] == name]
        if row.empty:
            continue
        s = row["total_judge_score"].iloc[0]
        if pd.isna(s) or float(s) < 0:
            continue
        names.append(name)
        scores.append(float(s))
    return names, np.array(scores) if scores else np.array([])


def get_fan_shares_week(
    season: int, week: int, contestants: List[str], df_est: pd.DataFrame
) -> np.ndarray:
    sub = df_est[(df_est["season"] == season) & (df_est["week"] == week)]
    out = []
    for name in contestants:
        r = sub[sub["celebrity_name"] == name]
        if r.empty:
            out.append(np.nan)
        else:
            out.append(float(r["fan_share_mean"].iloc[0]))
    arr = np.array(out, dtype=float)
    if np.any(np.isnan(arr)) or np.nansum(arr) <= 0:
        arr = np.ones(len(contestants)) / len(contestants)
    else:
        arr = arr / np.nansum(arr)
    return arr


def sample_fan_estimates_from_bounds(df_est: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df_est.copy()
    if "fan_share_lb" not in df.columns or "fan_share_ub" not in df.columns:
        df["fan_share_lb"] = (df["fan_share_mean"] - 0.01).clip(1e-6, None)
        df["fan_share_ub"] = (df["fan_share_mean"] + 0.01).clip(None, 1 - 1e-6)
    lb = df["fan_share_lb"].values.astype(float)
    ub = df["fan_share_ub"].values.astype(float)
    lb, ub = np.minimum(lb, ub), np.maximum(lb, ub)
    u = rng.uniform(0, 1, size=len(df))
    df["fan_share_mean"] = (lb + u * (ub - lb)).clip(1e-6, 1 - 1e-6)
    g = df.groupby(["season", "week"])["fan_share_mean"]
    total = g.transform("sum")
    df["fan_share_mean"] = (df["fan_share_mean"] / total).clip(1e-6, 1 - 1e-6)
    total2 = df.groupby(["season", "week"])["fan_share_mean"].transform("sum")
    df["fan_share_mean"] = df["fan_share_mean"] / total2
    return df


def get_eliminated_this_week(season: int, week: int, df_wide: pd.DataFrame) -> List[str]:
    cand = df_wide[
        (df_wide["season"] == season)
        & (df_wide["result_type"] == "eliminated")
        & (df_wide["eliminated_week"].notna())
    ]
    if cand.empty:
        return []
    elim = cand[cand["eliminated_week"].astype(int) == week]
    return elim["celebrity_name"].tolist()


def build_weekly_contexts(
    df_long: pd.DataFrame, df_wide: pd.DataFrame, df_est: pd.DataFrame
) -> List[Dict]:
    seasons = sorted(df_long["season"].unique())
    contexts = []
    for season in seasons:
        max_week = int(df_long[df_long["season"] == season]["week"].max())
        for week in range(1, max_week + 1):
            current = get_contestants_still_in(season, week, df_wide)
            if not current:
                continue
            contestants, judge_totals = get_judge_scores_week(
                season, week, current, df_long
            )
            if len(contestants) == 0 or len(judge_totals) == 0:
                continue
            if np.nansum(judge_totals) <= 0:
                continue
            eliminated = get_eliminated_this_week(season, week, df_wide)
            if not eliminated:
                continue
            idx_elim = []
            for name in eliminated:
                if name in contestants:
                    idx_elim.append(contestants.index(name))
            if len(idx_elim) != len(eliminated):
                continue
            fan_shares = get_fan_shares_week(season, week, contestants, df_est)
            if len(fan_shares) != len(contestants):
                fan_shares = np.ones(len(contestants)) / len(contestants)
            if "voting_method" in df_wide.columns:
                vm = df_wide[df_wide["season"] == season]["voting_method"].drop_duplicates()
                voting_method = str(vm.iloc[0]).strip().lower() if len(vm) else "percent"
            else:
                voting_method = "rank" if season in (1, 2) or (28 <= season <= 34) else "percent"
            if voting_method not in ("rank", "percent"):
                voting_method = "percent"
            contexts.append({
                "season": season,
                "week": week,
                "contestants": contestants,
                "judge_totals": judge_totals,
                "fan_shares": fan_shares,
                "eliminated_names": eliminated,
                "idx_elim": idx_elim,
                "voting_method": voting_method,
            })
    return contexts


def run_global_cross_validation(contexts: List[Dict]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for ctx in contexts:
        J = ctx["judge_totals"]
        F = ctx["fan_shares"]
        names = ctx["contestants"]
        k = len(ctx["idx_elim"])
        k = max(1, min(k, len(names)))
        if k == 1:
            elim_rank = evaluate_outcome(J, F, "rank", rng)
            elim_pct = evaluate_outcome(J, F, "percent", rng)
            set_rank = {elim_rank}
            set_pct = {elim_pct}
            elim_rank_str = names[elim_rank]
            elim_pct_str = names[elim_pct]
        else:
            set_rank = set(get_worst_k_rank(J, F, k))
            set_pct = set(get_worst_k_percent(J, F, k))
            elim_rank_str = ";".join(sorted(names[i] for i in set_rank))
            elim_pct_str = ";".join(sorted(names[i] for i in set_pct))
        flip = set_rank != set_pct
        rows.append({
            "season": ctx["season"],
            "week": ctx["week"],
            "n_elim": k,
            "elim_rank": elim_rank_str,
            "elim_percent": elim_pct_str,
            "flip": flip,
        })
    df = pd.DataFrame(rows)
    return df


def flip_rate_by_season(flip_df: pd.DataFrame) -> pd.DataFrame:
    agg = flip_df.groupby("season").agg(
        total_weeks=("flip", "count"),
        flip_count=("flip", "sum"),
    ).reset_index()
    agg["flip_rate"] = agg["flip_count"] / agg["total_weeks"]
    return agg


def variance_ratio_per_week(contexts: List[Dict]) -> pd.DataFrame:
    rows = []
    for ctx in contexts:
        J = ctx["judge_totals"]
        F = ctx["fan_shares"]
        n = len(J)
        if n < 2:
            continue
        j_sum, f_sum = np.nansum(J), np.nansum(F)
        if j_sum <= 0 or f_sum <= 0:
            continue
        j_pct = J / j_sum
        f_pct = F / f_sum
        sigma_j = np.std(j_pct)
        sigma_f = np.std(f_pct)
        if sigma_j < 1e-10:
            sigma_j = 1e-10
        ratio = sigma_f / sigma_j
        rows.append({
            "season": ctx["season"],
            "week": ctx["week"],
            "sigma_j": sigma_j,
            "sigma_f": sigma_f,
            "sigma_f_over_sigma_j": ratio,
        })
    return pd.DataFrame(rows)


def simulate_season_with_method(
    season: int,
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    df_est: pd.DataFrame,
    method: MethodType,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, int]:
    if rng is None:
        rng = np.random.default_rng(42)
    max_week = int(df_long[df_long["season"] == season]["week"].max())
    alive = set(get_contestants_still_in(season, 1, df_wide))
    last_week: Dict[str, int] = {c: 1 for c in alive}
    for week in range(1, max_week + 1):
        current = [c for c in alive if c in get_contestants_still_in(season, week, df_wide)]
        if len(current) <= 1:
            for c in current:
                last_week[c] = week
            break
        contestants, judge_totals = get_judge_scores_week(season, week, current, df_long)
        if len(contestants) < 2:
            for c in contestants:
                last_week[c] = week
            break
        fan_shares = get_fan_shares_week(season, week, contestants, df_est)
        if len(fan_shares) != len(contestants):
            fan_shares = np.ones(len(contestants)) / len(contestants)
        idx_elim = evaluate_outcome(judge_totals, fan_shares, method, rng)
        eliminated_name = contestants[idx_elim]
        last_week[eliminated_name] = week
        alive.discard(eliminated_name)
        for c in contestants:
            if c != eliminated_name:
                last_week[c] = week
        if len(alive) <= 1:
            break
    return last_week


def get_actual_last_week(season: int, df_wide: pd.DataFrame) -> Dict[str, int]:
    sub = df_wide[df_wide["season"] == season][["celebrity_name", "last_week_competed"]].drop_duplicates()
    return dict(zip(sub["celebrity_name"], sub["last_week_competed"].astype(int)))


def get_last_observed_week(season: int, df_long: pd.DataFrame) -> Dict[str, int]:
    sub = df_long[(df_long["season"] == season) & (df_long["total_judge_score"].notna())]
    if sub.empty:
        return {}
    agg = sub.groupby("celebrity_name")["week"].max()
    return agg.to_dict()


def get_judge_scores_week_imputed(
    season: int,
    week: int,
    contestants: List[str],
    df_long: pd.DataFrame,
    last_observed: Dict[str, int],
) -> Tuple[List[str], np.ndarray]:
    names, scores = [], []
    for name in contestants:
        eff_week = min(week, last_observed.get(name, week))
        grp = df_long[(df_long["season"] == season) & (df_long["week"] == eff_week) & (df_long["celebrity_name"] == name)]
        if grp.empty:
            continue
        s = grp["total_judge_score"].iloc[0]
        if pd.isna(s) or float(s) < 0:
            continue
        names.append(name)
        scores.append(float(s))
    return names, np.array(scores) if scores else np.array([])


def get_fan_shares_week_imputed(
    season: int,
    week: int,
    contestants: List[str],
    df_est: pd.DataFrame,
    last_observed: Dict[str, int],
) -> np.ndarray:
    out = []
    for name in contestants:
        eff_week = min(week, last_observed.get(name, week))
        sub = df_est[(df_est["season"] == season) & (df_est["week"] == eff_week) & (df_est["celebrity_name"] == name)]
        if sub.empty:
            out.append(np.nan)
        else:
            out.append(float(sub["fan_share_mean"].iloc[0]))
    arr = np.array(out, dtype=float)
    if np.any(np.isnan(arr)) or np.nansum(arr) <= 0:
        arr = np.ones(len(contestants)) / len(contestants)
    else:
        arr = arr / np.nansum(arr)
    return arr


def simulate_season_with_save_imputation(
    season: int,
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    df_est: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, int]:
    if rng is None:
        rng = np.random.default_rng(42)
    max_week = int(df_long[df_long["season"] == season]["week"].max())
    alive = set(get_contestants_still_in(season, 1, df_wide))
    last_observed = get_last_observed_week(season, df_long)
    for c in alive:
        if c not in last_observed:
            last_observed[c] = 1
    method: MethodType = "rank_save" if (season in (1, 2) or (28 <= season <= 34)) else "percent_save"
    last_week: Dict[str, int] = {c: 1 for c in alive}
    for week in range(1, max_week + 1):
        current = [c for c in alive if last_observed.get(c, 0) >= week]
        if len(current) <= 1:
            for c in current:
                last_week[c] = week
            break
        contestants, judge_totals = get_judge_scores_week_imputed(season, week, current, df_long, last_observed)
        if len(contestants) < 2:
            for c in contestants:
                last_week[c] = week
            break
        fan_shares = get_fan_shares_week_imputed(season, week, contestants, df_est, last_observed)
        if len(fan_shares) != len(contestants):
            fan_shares = np.ones(len(contestants)) / len(contestants)
        idx_elim = evaluate_outcome(judge_totals, fan_shares, method, rng)
        eliminated_name = contestants[idx_elim]
        last_week[eliminated_name] = week
        alive.discard(eliminated_name)
        for c in contestants:
            if c != eliminated_name:
                last_week[c] = week
        if len(alive) <= 1:
            break
    return last_week


def placement_from_last_week(last_week: Dict[str, int], max_week: int) -> Dict[str, int]:
    order = sorted(last_week.items(), key=lambda x: -x[1])
    return {name: r for r, (name, _) in enumerate(order, start=1)}


def get_champion(last_week: Dict[str, int], max_week: int) -> Optional[str]:
    if not last_week:
        return None
    best = max(last_week.items(), key=lambda x: x[1])
    return best[0]


def survival_timeline_for_celebrity(
    name: str,
    season: int,
    actual_last: Dict[str, int],
    sim_rank: Dict[str, int],
    sim_percent: Dict[str, int],
    max_week: int,
) -> pd.DataFrame:
    rows = []
    for w in range(1, max_week + 1):
        rows.append({
            "week": w,
            "actual_survived": 1 if actual_last.get(name, 0) >= w else 0,
            "sim_rank_survived": 1 if sim_rank.get(name, 0) >= w else 0,
            "sim_percent_survived": 1 if sim_percent.get(name, 0) >= w else 0,
        })
    return pd.DataFrame(rows)


def save_impact_table(
    contexts: List[Dict],
    df_wide: pd.DataFrame,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(42)
    rows = []
    for ctx in contexts:
        J = ctx["judge_totals"]
        F = ctx["fan_shares"]
        names = ctx["contestants"]
        if len(names) < 2:
            continue
        if len(ctx["idx_elim"]) != 1:
            continue
        actual_idx = ctx["idx_elim"][0]
        actual_name = names[actual_idx]
        use_rank = ctx.get("voting_method", "percent") == "rank"
        if use_rank:
            i, j = get_bottom_two_rank(J, F)
        else:
            i, j = get_bottom_two_percent(J, F)
        if i == j:
            continue
        if actual_idx not in (i, j):
            continue
        other_idx = j if actual_idx == i else i
        other_name = names[other_idx]
        j_actual = J[actual_idx]
        j_other = J[other_idx]
        would_be_saved = j_actual > j_other
        rows.append({
            "season": ctx["season"],
            "week": ctx["week"],
            "voting_method": ctx.get("voting_method", "percent"),
            "eliminated_name": actual_name,
            "bottom_two_other": other_name,
            "judge_score_eliminated": j_actual,
            "judge_score_other": j_other,
            "would_have_been_saved": would_be_saved,
        })
    df = pd.DataFrame(rows)
    return df


def _apply_plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.linewidth": 1.0,
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "grid.color": "#E0E0E0",
        "grid.alpha": 0.7,
    })


def _add_panel_label(ax, label: str, x: float = -0.05, y: float = 1.18):
    ax.text(x, y, f"({label})", transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="left", clip_on=False)


_PLOT_COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "actual": "#029E73",
    "simulated": "#0173B2",
    "bar_edge": "white",
}
_PALETTE = ["#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#949494"]
_DPI = 300


def _plot_flip_rate(flip_by_season: pd.DataFrame, ax=None):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.2))
    else:
        fig = ax.get_figure()
    bars = ax.bar(flip_by_season["season"], flip_by_season["flip_rate"], color=_PLOT_COLORS["primary"],
                  alpha=0.85, edgecolor=_PLOT_COLORS["bar_edge"], linewidth=0.8)
    ax.set_xlabel("Season")
    ax.set_ylabel("Flip rate")
    ax.set_title("Rank vs Percent: weeks with different elimination", fontsize=10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5, axis="y")
    if ax is None:
        plt.tight_layout()
    return fig


def _plot_variance_ratio(var_df: pd.DataFrame, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.2))
    else:
        fig = ax.get_figure()
    by_season = var_df.groupby("season")["sigma_f_over_sigma_j"].mean().reset_index()
    ax.bar(by_season["season"], by_season["sigma_f_over_sigma_j"], color=_PLOT_COLORS["secondary"],
           alpha=0.85, edgecolor=_PLOT_COLORS["bar_edge"], linewidth=0.8)
    ax.set_xlabel("Season")
    ax.set_ylabel(r"$\sigma_F / \sigma_J$")
    ax.set_title("Fan vs judge variance ratio (Percentage mode)", fontsize=10)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5, axis="y")
    if ax is None:
        plt.tight_layout()
    return fig


def _plot_survival_timeline(
    tl_df: pd.DataFrame,
    title: str,
    actual_label: str,
    sim_label: str,
    actual_key: str,
    sim_key: str,
    ax=None,
):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3))
    else:
        fig = ax.get_figure()
    w = tl_df["week"].values
    ax.plot(w, tl_df[actual_key], "o-", label=actual_label, color=_PLOT_COLORS["actual"], markersize=5, linewidth=2)
    ax.plot(w, tl_df[sim_key], "s--", label=sim_label, color=_PLOT_COLORS["simulated"], markersize=4, linewidth=1.5)
    ax.set_xlabel("Week")
    ax.set_ylabel("Survived (1=yes)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_ylim(-0.15, 1.2)
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.5)
    if ax is None:
        plt.tight_layout()
    return fig


def _fig_combined(
    flip_by_season: pd.DataFrame,
    var_df: pd.DataFrame,
    jerry_rice, actual_s2, sim_rank_s2, sim_percent_s2, max_week_s2,
    billy_ray, actual_s4, sim_rank_s4, sim_percent_s4, max_week_s4,
    bristol_palin, actual_s11, sim_rank_s11, sim_percent_s11, max_week_s11,
    bobby_bones, actual_s27, sim_rank_s27, sim_percent_s27, max_week_s27,
):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.3, left=0.08, right=0.97, top=0.94, bottom=0.06)

    ax_a = fig.add_subplot(gs[0, 0])
    _plot_flip_rate(flip_by_season, ax=ax_a)
    _add_panel_label(ax_a, "a")

    ax_b = fig.add_subplot(gs[0, 1])
    _plot_variance_ratio(var_df, ax=ax_b)
    _add_panel_label(ax_b, "b")

    ax_c = fig.add_subplot(gs[0, 2])
    if bobby_bones:
        tl = survival_timeline_for_celebrity(bobby_bones, 27, actual_s27, sim_rank_s27, sim_percent_s27, max_week_s27)
        _plot_survival_timeline(tl, f"{bobby_bones} (S27)", "Actual (Percent)", "Simulated (Rank)",
                               "actual_survived", "sim_rank_survived", ax=ax_c)
    _add_panel_label(ax_c, "c")

    ax_d = fig.add_subplot(gs[1, 0])
    if jerry_rice:
        tl_j = survival_timeline_for_celebrity(jerry_rice, 2, actual_s2, sim_rank_s2, sim_percent_s2, max_week_s2)
        _plot_survival_timeline(tl_j, f"{jerry_rice} (S2)", "Actual (Rank)", "Simulated (Percent)",
                               "actual_survived", "sim_percent_survived", ax=ax_d)
    _add_panel_label(ax_d, "d")

    ax_e = fig.add_subplot(gs[1, 1])
    if billy_ray:
        tl_br = survival_timeline_for_celebrity(billy_ray, 4, actual_s4, sim_rank_s4, sim_percent_s4, max_week_s4)
        _plot_survival_timeline(tl_br, f"{billy_ray} (S4)", "Actual (Percent)", "Simulated (Rank)",
                               "actual_survived", "sim_rank_survived", ax=ax_e)
    _add_panel_label(ax_e, "e")

    ax_f = fig.add_subplot(gs[1, 2])
    if bristol_palin:
        tl_bp = survival_timeline_for_celebrity(bristol_palin, 11, actual_s11, sim_rank_s11, sim_percent_s11, max_week_s11)
        _plot_survival_timeline(tl_bp, f"{bristol_palin} (S11)", "Actual (Percent)", "Simulated (Rank)",
                               "actual_survived", "sim_rank_survived", ax=ax_f)
    _add_panel_label(ax_f, "f")

    return fig


def main():
    print("Loading data...")
    df_long = load_dwts_long()
    df_wide = load_dwts_cleaned()
    df_est = load_fan_vote_estimates()

    print("Building weekly contexts...")
    contexts = build_weekly_contexts(df_long, df_wide, df_est)
    print("Contexts:", len(contexts))

    print("Step A: Global cross-validation (Rank vs Percent)...")
    flip_df = run_global_cross_validation(contexts)
    flip_by_season = flip_rate_by_season(flip_df)
    flip_df.to_csv(os.path.join(OUT_DIR, "flip_detail.csv"), index=False, encoding="utf-8-sig")
    flip_by_season.to_csv(os.path.join(OUT_DIR, "flip_rate_by_season.csv"), index=False, encoding="utf-8-sig")

    print("Variance ratio (sigma_F / sigma_J)...")
    var_df = variance_ratio_per_week(contexts)
    var_df.to_csv(os.path.join(OUT_DIR, "variance_ratio_by_week.csv"), index=False, encoding="utf-8-sig")

    rng = np.random.default_rng(42)
    max_week_s2 = int(df_long[df_long["season"] == 2]["week"].max())
    max_week_s4 = int(df_long[df_long["season"] == 4]["week"].max())
    max_week_s11 = int(df_long[df_long["season"] == 11]["week"].max())
    max_week_s27 = int(df_long[df_long["season"] == 27]["week"].max())
    actual_s2 = get_actual_last_week(2, df_wide)
    actual_s4 = get_actual_last_week(4, df_wide)
    actual_s11 = get_actual_last_week(11, df_wide)
    actual_s27 = get_actual_last_week(27, df_wide)

    sim_rank_s2 = simulate_season_with_method(2, df_long, df_wide, df_est, "rank", rng)
    sim_percent_s2 = simulate_season_with_method(2, df_long, df_wide, df_est, "percent", rng)
    sim_rank_s4 = simulate_season_with_method(4, df_long, df_wide, df_est, "rank", rng)
    sim_percent_s4 = simulate_season_with_method(4, df_long, df_wide, df_est, "percent", rng)
    sim_rank_s11 = simulate_season_with_method(11, df_long, df_wide, df_est, "rank", rng)
    sim_percent_s11 = simulate_season_with_method(11, df_long, df_wide, df_est, "percent", rng)
    sim_rank_s27 = simulate_season_with_method(27, df_long, df_wide, df_est, "rank", rng)
    sim_percent_s27 = simulate_season_with_method(27, df_long, df_wide, df_est, "percent", rng)

    jerry_rice = "Jerry Rice" if "Jerry Rice" in actual_s2 else ([c for c in actual_s2 if "Jerry" in c or "Rice" in c][0] if [c for c in actual_s2 if "Jerry" in c or "Rice" in c] else None)
    billy_ray = [c for c in actual_s4 if "Billy" in c][0] if [c for c in actual_s4 if "Billy" in c] else None
    bristol_palin = [c for c in actual_s11 if "Bristol" in c][0] if [c for c in actual_s11 if "Bristol" in c] else None
    bobby_bones = "Bobby Bones" if "Bobby Bones" in actual_s27 else ([c for c in actual_s27 if "Bobby" in c or "Bones" in c][0] if [c for c in actual_s27 if "Bobby" in c or "Bones" in c] else None)

    case_studies = []
    if jerry_rice:
        tl_j = survival_timeline_for_celebrity(
            jerry_rice, 2, actual_s2, sim_rank_s2, sim_percent_s2, max_week_s2
        )
        tl_j["celebrity_name"] = jerry_rice
        tl_j["season"] = 2
        case_studies.append(tl_j)
    if billy_ray:
        tl_br = survival_timeline_for_celebrity(
            billy_ray, 4, actual_s4, sim_rank_s4, sim_percent_s4, max_week_s4
        )
        tl_br["celebrity_name"] = billy_ray
        tl_br["season"] = 4
        case_studies.append(tl_br)
    if bristol_palin:
        tl_bp = survival_timeline_for_celebrity(
            bristol_palin, 11, actual_s11, sim_rank_s11, sim_percent_s11, max_week_s11
        )
        tl_bp["celebrity_name"] = bristol_palin
        tl_bp["season"] = 11
        case_studies.append(tl_bp)
    if bobby_bones:
        tl_b = survival_timeline_for_celebrity(
            bobby_bones, 27, actual_s27, sim_rank_s27, sim_percent_s27, max_week_s27
        )
        tl_b["celebrity_name"] = bobby_bones
        tl_b["season"] = 27
        case_studies.append(tl_b)
    if case_studies:
        pd.concat(case_studies, ignore_index=True).to_csv(
            os.path.join(OUT_DIR, "controversy_survival_timeline.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    print("Save impact (would-have-been-saved count)...")
    save_df = save_impact_table(contexts, df_wide, rng)
    save_df.to_csv(os.path.join(OUT_DIR, "save_impact_detail.csv"), index=False, encoding="utf-8-sig")
    save_summary = save_df.groupby("season").agg(
        total_bottom_two_weeks=("would_have_been_saved", "count"),
        would_have_been_saved=("would_have_been_saved", "sum"),
    ).reset_index()
    save_summary.to_csv(os.path.join(OUT_DIR, "save_impact_by_season.csv"), index=False, encoding="utf-8-sig")

    print("Save season-level impact (under last-observed imputation)...")
    season_impact_rows = []
    for season in sorted(df_long["season"].unique()):
        max_week = int(df_long[df_long["season"] == season]["week"].max())
        method_no_save: MethodType = "percent" if (3 <= season <= 27) else "rank"
        last_no = simulate_season_with_method(season, df_long, df_wide, df_est, method_no_save, rng)
        last_save = simulate_season_with_save_imputation(season, df_long, df_wide, df_est, rng)
        champ_no = get_champion(last_no, max_week)
        champ_save = get_champion(last_save, max_week)
        place_no = placement_from_last_week(last_no, max_week)
        place_save = placement_from_last_week(last_save, max_week)
        top3_no = set(c for c, p in place_no.items() if p <= 3)
        top3_save = set(c for c, p in place_save.items() if p <= 3)
        season_impact_rows.append({
            "season": season,
            "champion_no_save": champ_no or "",
            "champion_with_save": champ_save or "",
            "champion_changed": 1 if (champ_no and champ_save and champ_no != champ_save) else 0,
            "finalists_top3_changed": 1 if top3_no != top3_save else 0,
        })
    pd.DataFrame(season_impact_rows).to_csv(
        os.path.join(OUT_DIR, "save_season_impact_imputed.csv"), index=False, encoding="utf-8-sig"
    )
    n_champ_changed = sum(r["champion_changed"] for r in season_impact_rows)
    n_final_changed = sum(r["finalists_top3_changed"] for r in season_impact_rows)
    print("  Champion changed (under imputation): {} seasons; finalists (top3) changed: {} seasons.".format(
        n_champ_changed, n_final_changed
    ))

    B_robust = 30
    if "fan_share_lb" in df_est.columns and "fan_share_ub" in df_est.columns:
        print("Robustness: propagating Q1 fan vote uncertainty (B={} samples from [lb,ub])...".format(B_robust))
        rng_robust = np.random.default_rng(43)
        flip_rates_by_season: Dict[int, List[float]] = {}
        would_saved_list: List[int] = []
        for b in range(B_robust):
            df_est_b = sample_fan_estimates_from_bounds(df_est, rng_robust)
            ctx_b = build_weekly_contexts(df_long, df_wide, df_est_b)
            flip_df_b = run_global_cross_validation(ctx_b)
            flip_by_b = flip_rate_by_season(flip_df_b)
            for _, row in flip_by_b.iterrows():
                s = int(row["season"])
                if s not in flip_rates_by_season:
                    flip_rates_by_season[s] = []
                flip_rates_by_season[s].append(float(row["flip_rate"]))
            save_df_b = save_impact_table(ctx_b, df_wide, rng_robust)
            would_saved_list.append(int(save_df_b["would_have_been_saved"].sum()))
        rows_robust = []
        for s in sorted(flip_rates_by_season.keys()):
            vals = flip_rates_by_season[s]
            rows_robust.append({
                "season": s,
                "flip_rate_mean": float(np.mean(vals)),
                "flip_rate_std": float(np.std(vals)) if len(vals) > 1 else 0.0,
            })
        pd.DataFrame(rows_robust).to_csv(
            os.path.join(OUT_DIR, "robustness_flip_rate_by_season.csv"), index=False, encoding="utf-8-sig"
        )
        with open(os.path.join(OUT_DIR, "robustness_save_summary.txt"), "w", encoding="utf-8") as f:
            f.write("Q1 fan vote uncertainty propagation (B={} samples from [lb,ub])\n".format(B_robust))
            f.write("Would-have-been-saved count: mean = {:.2f}, std = {:.2f}\n".format(
                np.mean(would_saved_list), np.std(would_saved_list) if len(would_saved_list) > 1 else 0.0
            ))
            f.write("Flip rate per season: see robustness_flip_rate_by_season.csv (mean +/- std).\n")
        print("  Robustness outputs: robustness_flip_rate_by_season.csv, robustness_save_summary.txt")
    else:
        print("Robustness skipped: fan_share_lb/ub not in fan_vote_estimates.")

    with open(os.path.join(OUT_DIR, "methodology_note.txt"), "w", encoding="utf-8") as f:
        f.write("Problem 2 — Methodology note\n")
        f.write("=" * 50 + "\n\n")
        f.write("1. Counterfactual (rule switch): Conducted under the controlled assumption that\n")
        f.write("   'underlying fan preferences do not change with the scoring rule.' Same estimated\n")
        f.write("   fan votes F (from Q1) are used; only the combination rule (Rank / Percent) is switched.\n")
        f.write("   Robustness: conclusions can be checked by sampling F from [lb, ub] and re-running.\n\n")
        f.write("2. Judges' Save impact: Bottom Two each week is determined by the actual rule for that\n")
        f.write("   season: Rank-based (S1-2, S28-34) vs Percentage-based (S3-27). Then who would be saved\n")
        f.write("   is decided by higher judge score. See save_impact_detail.csv column 'voting_method'.\n")
        f.write("   Baseline: deterministic (higher J saved). Optional: logit P(save i)=sigmoid(beta*(Ji-Jj));\n")
        f.write("   sensitivity on beta can be run for MCM-style robustness (code: p_judges_save_a_logit).\n")
        f.write("   This is a single-week static statistic (would this person have been saved this week?).\n")
        f.write("   Season-level impact: We also run a full-season replay WITH Save under 'last-observed\n")
        f.write("   imputation' (saved contestant keeps their last week J/F in subsequent weeks). See\n")
        f.write("   save_season_impact_imputed.csv for champion_changed and finalists_top3_changed.\n\n")
        f.write("3. simulate_season_with_method: Uses alive ∩ get_contestants_still_in(season, week, df_wide),\n")
        f.write("   i.e. within-observed-weeks counterfactual. A person historically eliminated in week 5\n")
        f.write("   does not appear in week 6+ (no J/F for them). Not a full season replay.\n\n")
        f.write("4. Q1 fan vote uncertainty: We propagate F uncertainty by sampling from [lb, ub] (B runs)\n")
        f.write("   and re-running flip/save; see robustness_flip_rate_by_season.csv and\n")
        f.write("   robustness_save_summary.txt. Conclusions are stable if std is small.\n")

    avg_ratio = float(var_df["sigma_f_over_sigma_j"].mean()) if len(var_df) else 0.0
    total_save_weeks = len(save_df)
    would_saved = int(save_df["would_have_been_saved"].sum()) if total_save_weeks else 0
    with open(os.path.join(OUT_DIR, "recommendation_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Problem 2 — Summary for recommendations\n")
        f.write("=" * 50 + "\n\n")
        f.write("Fan vs judge influence (Percentage mode):\n")
        f.write("  Average sigma_F / sigma_J = {:.4f}\n".format(avg_ratio))
        f.write("  (>1: fan variance dominates; higher = more fan-biased.)\n\n")
        f.write("Judges' Save impact (single-week):\n")
        f.write("  Total bottom-two weeks considered: {}\n".format(total_save_weeks))
        f.write("  Times eliminated contestant had higher judge score (would have been saved): {}\n".format(would_saved))
        if total_save_weeks > 0:
            f.write("  Rate: {:.2%}\n".format(would_saved / total_save_weeks))
        f.write("\nFlip rate (Rank vs Percent): see flip_rate_by_season.csv.\n")
        f.write("Controversy timelines: see controversy_survival_timeline.csv.\n")
        f.write("Save season-level impact (under imputation): see save_season_impact_imputed.csv.\n")
        f.write("Fan vote uncertainty propagation: see robustness_flip_rate_by_season.csv, robustness_save_summary.txt.\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _apply_plot_style()

        def _save_fig(fig, name: str):
            for ext in ["png", "pdf"]:
                path = os.path.join(OUT_DIR, f"{name}.{ext}")
                fig.savefig(path, dpi=_DPI if ext == "png" else None, bbox_inches="tight")
            plt.close(fig)

        fig1 = _plot_flip_rate(flip_by_season)
        _save_fig(fig1, "fig_flip_rate_by_season")

        fig2 = _plot_variance_ratio(var_df)
        _save_fig(fig2, "fig_variance_ratio")

        if bobby_bones:
            tl = survival_timeline_for_celebrity(bobby_bones, 27, actual_s27, sim_rank_s27, sim_percent_s27, max_week_s27)
            fig = _plot_survival_timeline(tl, f"{bobby_bones} (S27)", "Actual (Percent)", "Simulated (Rank)",
                                          "actual_survived", "sim_rank_survived")
            _save_fig(fig, "fig_controversy_timeline_bobby_bones")
        if jerry_rice:
            tl_j = survival_timeline_for_celebrity(jerry_rice, 2, actual_s2, sim_rank_s2, sim_percent_s2, max_week_s2)
            fig = _plot_survival_timeline(tl_j, f"{jerry_rice} (S2)", "Actual (Rank)", "Simulated (Percent)",
                                          "actual_survived", "sim_percent_survived")
            _save_fig(fig, "fig_controversy_timeline_jerry_rice")
        if billy_ray:
            tl_br = survival_timeline_for_celebrity(billy_ray, 4, actual_s4, sim_rank_s4, sim_percent_s4, max_week_s4)
            fig = _plot_survival_timeline(tl_br, f"{billy_ray} (S4)", "Actual (Percent)", "Simulated (Rank)",
                                          "actual_survived", "sim_rank_survived")
            _save_fig(fig, "fig_controversy_timeline_billy_ray_cyrus")
        if bristol_palin:
            tl_bp = survival_timeline_for_celebrity(bristol_palin, 11, actual_s11, sim_rank_s11, sim_percent_s11, max_week_s11)
            fig = _plot_survival_timeline(tl_bp, f"{bristol_palin} (S11)", "Actual (Percent)", "Simulated (Rank)",
                                          "actual_survived", "sim_rank_survived")
            _save_fig(fig, "fig_controversy_timeline_bristol_palin")

        fig_combined = _fig_combined(
            flip_by_season, var_df,
            jerry_rice, actual_s2, sim_rank_s2, sim_percent_s2, max_week_s2,
            billy_ray, actual_s4, sim_rank_s4, sim_percent_s4, max_week_s4,
            bristol_palin, actual_s11, sim_rank_s11, sim_percent_s11, max_week_s11,
            bobby_bones, actual_s27, sim_rank_s27, sim_percent_s27, max_week_s27,
        )
        _save_fig(fig_combined, "fig_combined_all")

        print("Charts saved to", OUT_DIR)
    except Exception as e:
        print("Plotting skipped:", e)

    print("Done. Outputs in", OUT_DIR)
    return


if __name__ == "__main__":
    main()
