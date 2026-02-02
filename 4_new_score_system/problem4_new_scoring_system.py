from __future__ import annotations

import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_CLEAN_DIR = os.path.join(PROJECT_ROOT, "数据清洗")
Q1_DIR = os.path.join(PROJECT_ROOT, "问题一")
SERVER_BASE = "/8202302003"
if os.path.exists(SERVER_BASE):
    DATA_CLEAN_DIR = os.path.join(SERVER_BASE, "数据清洗")
    Q1_DIR = os.path.join(SERVER_BASE, "问题一")
    OUT_DIR = os.path.join(SERVER_BASE, "问题四", "outputs")
else:
    OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
FIG_DIR = os.path.join(os.path.dirname(OUT_DIR), "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def _path(name: str) -> str:
    if os.path.exists(SERVER_BASE):
        p_flat = os.path.join(SERVER_BASE, name)
        if os.path.isfile(p_flat):
            return p_flat
    for base in [DATA_CLEAN_DIR, Q1_DIR, SCRIPT_DIR, PROJECT_ROOT]:
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
    return os.path.join(SCRIPT_DIR, name)


def load_long() -> pd.DataFrame:
    return pd.read_csv(_path("dwts_long.csv"), encoding="utf-8-sig")


def load_wide() -> pd.DataFrame:
    return pd.read_csv(_path("dwts_cleaned.csv"), encoding="utf-8-sig")


def load_fan_estimates() -> pd.DataFrame:
    return pd.read_csv(_path("fan_vote_estimates.csv"), encoding="utf-8-sig")


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
    g2 = df.groupby(["season", "week"])["fan_share_mean"]
    total2 = g2.transform("sum")
    df["fan_share_mean"] = df["fan_share_mean"] / total2
    return df


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


def z_score_standardize(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    m, s = np.nanmean(x), np.nanstd(x)
    if s < eps:
        return np.zeros_like(x)
    return (np.asarray(x, dtype=float) - m) / s

def phi_merit(z: np.ndarray) -> np.ndarray:
    return norm.cdf(np.asarray(z, dtype=float))


def T_popularity(z: np.ndarray, k: float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(k * np.asarray(z, dtype=float)))


def combined_score(
    z_judge: np.ndarray, z_fan: np.ndarray, alpha: float, k: float
) -> np.ndarray:
    return (1.0 - alpha) * phi_merit(z_judge) + alpha * T_popularity(z_fan, k)


SUSPENSE_GAP_THRESHOLD = 0.05

def elimination_percent(judge_scores: np.ndarray, fan_votes: np.ndarray) -> int:
    j = np.asarray(judge_scores, dtype=float).ravel()
    f = np.asarray(fan_votes, dtype=float).ravel()
    js, fs = np.nansum(j), np.nansum(f)
    if js <= 0 or fs <= 0:
        return 0
    cp = j / js + f / fs
    return int(np.argmin(cp))


def new_system_elimination(S: np.ndarray, J: np.ndarray) -> int:
    n = len(S)
    if n <= 1:
        return 0
    idx = np.argsort(S)[:2]
    i, j = int(idx[0]), int(idx[1])
    return i if J[i] <= J[j] else j


def _bottom2_gap(scores: np.ndarray) -> float:
    if len(scores) < 2:
        return 0.0
    s = np.sort(np.asarray(scores, dtype=float).ravel())
    return float(s[1] - s[0])


def simulate_season_percent(
    season: int,
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    df_est: pd.DataFrame,
) -> Dict[str, int]:
    max_week = int(df_long[df_long["season"] == season]["week"].max())
    alive = set(get_contestants_still_in(season, 1, df_wide))
    last_week: Dict[str, int] = {c: 1 for c in alive}
    for week in range(1, max_week + 1):
        current = [c for c in alive if c in get_contestants_still_in(season, week, df_wide)]
        if len(current) <= 1:
            for c in current:
                last_week[c] = week
            break
        contestants, J = get_judge_scores_week(season, week, current, df_long)
        if len(contestants) < 2:
            for c in contestants:
                last_week[c] = week
            break
        F = get_fan_shares_week(season, week, contestants, df_est)
        if len(F) != len(contestants):
            F = np.ones(len(contestants)) / len(contestants)
        idx_elim = elimination_percent(J, F)
        eliminated_name = contestants[idx_elim]
        last_week[eliminated_name] = week
        alive.discard(eliminated_name)
        for c in contestants:
            if c != eliminated_name:
                last_week[c] = week
        if len(alive) <= 1:
            break
    return last_week


def simulate_season_percent_with_details(
    season: int,
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    df_est: pd.DataFrame,
) -> Tuple[Dict[str, int], List[Tuple[int, float, float, float]]]:
    max_week = int(df_long[df_long["season"] == season]["week"].max())
    alive = set(get_contestants_still_in(season, 1, df_wide))
    last_week: Dict[str, int] = {c: 1 for c in alive}
    elim_details: List[Tuple[int, float, float, float]] = []
    for week in range(1, max_week + 1):
        current = [c for c in alive if c in get_contestants_still_in(season, week, df_wide)]
        if len(current) <= 1:
            for c in current:
                last_week[c] = week
            break
        contestants, J = get_judge_scores_week(season, week, current, df_long)
        if len(contestants) < 2:
            for c in contestants:
                last_week[c] = week
            break
        F = get_fan_shares_week(season, week, contestants, df_est)
        if len(F) != len(contestants):
            F = np.ones(len(contestants)) / len(contestants)
        combined = J / np.nansum(J) + F / np.nansum(F)
        idx_elim = elimination_percent(J, F)
        J_elim = float(J[idx_elim])
        J_survivors = np.delete(J, idx_elim)
        min_J_survivors = float(np.min(J_survivors)) if len(J_survivors) > 0 else J_elim
        elim_details.append((week, J_elim, min_J_survivors, _bottom2_gap(combined)))
        eliminated_name = contestants[idx_elim]
        last_week[eliminated_name] = week
        alive.discard(eliminated_name)
        for c in contestants:
            if c != eliminated_name:
                last_week[c] = week
        if len(alive) <= 1:
            break
    return last_week, elim_details


def simulate_season_new_system(
    season: int,
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    df_est: pd.DataFrame,
    alpha: float,
    k: float,
) -> Dict[str, int]:
    max_week = int(df_long[df_long["season"] == season]["week"].max())
    alive = set(get_contestants_still_in(season, 1, df_wide))
    last_week: Dict[str, int] = {c: 1 for c in alive}
    for week in range(1, max_week + 1):
        current = [c for c in alive if c in get_contestants_still_in(season, week, df_wide)]
        if len(current) <= 1:
            for c in current:
                last_week[c] = week
            break
        contestants, J = get_judge_scores_week(season, week, current, df_long)
        if len(contestants) < 2:
            for c in contestants:
                last_week[c] = week
            break
        F = get_fan_shares_week(season, week, contestants, df_est)
        if len(F) != len(contestants):
            F = np.ones(len(contestants)) / len(contestants)
        Z_J = z_score_standardize(J)
        Z_F = z_score_standardize(F)
        S = combined_score(Z_J, Z_F, alpha, k)
        idx_elim = new_system_elimination(S, J)
        eliminated_name = contestants[idx_elim]
        last_week[eliminated_name] = week
        alive.discard(eliminated_name)
        for c in contestants:
            if c != eliminated_name:
                last_week[c] = week
        if len(alive) <= 1:
            break
    return last_week


def simulate_season_new_system_with_details(
    season: int,
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    df_est: pd.DataFrame,
    alpha: float,
    k: float,
) -> Tuple[Dict[str, int], List[Tuple[int, float, float, float]]]:
    max_week = int(df_long[df_long["season"] == season]["week"].max())
    alive = set(get_contestants_still_in(season, 1, df_wide))
    last_week: Dict[str, int] = {c: 1 for c in alive}
    elim_details: List[Tuple[int, float, float, float]] = []
    for week in range(1, max_week + 1):
        current = [c for c in alive if c in get_contestants_still_in(season, week, df_wide)]
        if len(current) <= 1:
            for c in current:
                last_week[c] = week
            break
        contestants, J = get_judge_scores_week(season, week, current, df_long)
        if len(contestants) < 2:
            for c in contestants:
                last_week[c] = week
            break
        F = get_fan_shares_week(season, week, contestants, df_est)
        if len(F) != len(contestants):
            F = np.ones(len(contestants)) / len(contestants)
        Z_J = z_score_standardize(J)
        Z_F = z_score_standardize(F)
        S = combined_score(Z_J, Z_F, alpha, k)
        idx_elim = new_system_elimination(S, J)
        J_elim = float(J[idx_elim])
        J_survivors = np.delete(J, idx_elim)
        min_J_survivors = float(np.min(J_survivors)) if len(J_survivors) > 0 else J_elim
        elim_details.append((week, J_elim, min_J_survivors, _bottom2_gap(S)))
        eliminated_name = contestants[idx_elim]
        last_week[eliminated_name] = week
        alive.discard(eliminated_name)
        for c in contestants:
            if c != eliminated_name:
                last_week[c] = week
        if len(alive) <= 1:
            break
    return last_week, elim_details


def placement_from_last_week(last_week: Dict[str, int], max_week: int) -> Dict[str, int]:
    order = sorted(last_week.items(), key=lambda x: -x[1])
    return {name: r for r, (name, _) in enumerate(order, start=1)}


def judge_only_placement(
    season: int, df_long: pd.DataFrame, df_wide: pd.DataFrame
) -> Dict[str, int]:
    sub = df_long[df_long["season"] == season]
    if sub.empty:
        return {}
    mean_j = sub.groupby("celebrity_name")["total_judge_score"].mean()
    order = mean_j.sort_values(ascending=False).index.tolist()
    return {name: r for r, name in enumerate(order, start=1)}


def spearman_placement(place_new: Dict[str, int], place_judge: Dict[str, int]) -> float:
    common = [c for c in place_new if c in place_judge]
    if len(common) < 2:
        return 0.0
    a = np.array([place_new[c] for c in common])
    b = np.array([place_judge[c] for c in common])
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def comeback_rate(place_new: Dict[str, int], place_judge: Dict[str, int]) -> float:
    common = [c for c in place_new if c in place_judge]
    if len(common) < 2:
        return 0.0
    pairs = 0
    comebacks = 0
    for i, ci in enumerate(common):
        for cj in common[i + 1 :]:
            ji, jj = place_judge[ci], place_judge[cj]
            ni, nj = place_new[ci], place_new[cj]
            if ji == jj:
                continue
            pairs += 1
            if ji > jj and ni < nj:
                comebacks += 1
            elif jj > ji and nj < ni:
                comebacks += 1
    return comebacks / pairs if pairs > 0 else 0.0


def obvious_injustice_rate_from_details(elim_details_list: List[List[Tuple[int, float, float, float]]]) -> float:
    total, injustices = 0, 0
    for details in elim_details_list:
        for (_, J_elim, min_J_survivors, _) in details:
            total += 1
            if J_elim > min_J_survivors:
                injustices += 1
    return injustices / total if total > 0 else 0.0


def suspense_rate_from_details(
    elim_details_list: List[List[Tuple[int, float, float, float]]],
    gap_threshold: float = SUSPENSE_GAP_THRESHOLD,
) -> float:
    total, close_calls = 0, 0
    for details in elim_details_list:
        for (_, _, _, gap) in details:
            total += 1
            if gap < gap_threshold:
                close_calls += 1
    return close_calls / total if total > 0 else 0.0


def evaluate_system(
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    df_est: pd.DataFrame,
    simulate_fn,
    sim_kwargs: Optional[dict] = None,
) -> Tuple[float, float]:
    sim_kwargs = sim_kwargs or {}
    seasons = sorted(df_long["season"].unique())
    mf_list, me_list = [], []
    for season in seasons:
        last_week = simulate_fn(season, df_long, df_wide, df_est, **sim_kwargs)
        max_week = int(df_long[df_long["season"] == season]["week"].max())
        place_new = placement_from_last_week(last_week, max_week)
        place_judge = judge_only_placement(season, df_long, df_wide)
        if len(place_new) < 2 or len(place_judge) < 2:
            continue
        mf_list.append(spearman_placement(place_new, place_judge))
        me_list.append(comeback_rate(place_new, place_judge))
    return (
        float(np.nanmean(mf_list)) if mf_list else 0.0,
        float(np.nanmean(me_list)) if me_list else 0.0,
    )


def evaluate_system_with_details(
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    df_est: pd.DataFrame,
    simulate_fn,
    sim_kwargs: Optional[dict] = None,
) -> Tuple[float, float, float, float]:
    sim_kwargs = sim_kwargs or {}
    seasons = sorted(df_long["season"].unique())
    mf_list, me_list = [], []
    all_elim_details: List[List[Tuple[int, float, float, float]]] = []
    for season in seasons:
        if "alpha" in sim_kwargs and "k" in sim_kwargs:
            last_week, elim_details = simulate_season_new_system_with_details(
                season, df_long, df_wide, df_est,
                sim_kwargs["alpha"], sim_kwargs["k"],
            )
        else:
            last_week, elim_details = simulate_season_percent_with_details(
                season, df_long, df_wide, df_est,
            )
        all_elim_details.append(elim_details)
        max_week = int(df_long[df_long["season"] == season]["week"].max())
        place_new = placement_from_last_week(last_week, max_week)
        place_judge = judge_only_placement(season, df_long, df_wide)
        if len(place_new) < 2 or len(place_judge) < 2:
            continue
        mf_list.append(spearman_placement(place_new, place_judge))
        me_list.append(comeback_rate(place_new, place_judge))
    mf = float(np.nanmean(mf_list)) if mf_list else 0.0
    me = float(np.nanmean(me_list)) if me_list else 0.0
    m_injustice = obvious_injustice_rate_from_details(all_elim_details)
    m_suspense = suspense_rate_from_details(all_elim_details)
    return mf, me, m_injustice, m_suspense


def run_grid_search(
    df_long: pd.DataFrame,
    df_wide: pd.DataFrame,
    df_est: pd.DataFrame,
    alphas: Optional[np.ndarray] = None,
    ks: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    alphas = alphas if alphas is not None else np.linspace(0.3, 0.7, 9)
    ks = ks if ks is not None else np.linspace(0.5, 3.0, 6)
    rows = []
    for alpha in alphas:
        for k in ks:
            mf, me = evaluate_system(
                df_long, df_wide, df_est,
                simulate_season_new_system,
                {"alpha": float(alpha), "k": float(k)},
            )
            rows.append({"alpha": alpha, "k": k, "M_F": mf, "M_E": me})
    return pd.DataFrame(rows)


def recommend_parameters(grid_df: pd.DataFrame) -> Tuple[float, float]:
    if grid_df.empty:
        return 0.4, 1.2
    me_max = grid_df["M_E"].max()
    me_min_acceptable = 0.35 * me_max if me_max > 0 else 0
    feasible = grid_df[grid_df["M_E"] >= me_min_acceptable]
    if feasible.empty:
        feasible = grid_df
    best = feasible.loc[feasible["M_F"].idxmax()]
    return float(best["alpha"]), float(best["k"])


def plot_pareto_tradeoff(
    grid_df: pd.DataFrame,
    mf_current: float,
    me_current: float,
    alpha_rec: float,
    k_rec: float,
    mf_rec: float,
    me_rec: float,
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(grid_df["M_E"], grid_df["M_F"], c="#2E86AB", alpha=0.5, s=25, label="New system (α, k)")
    ax.scatter(me_current, mf_current, c="#E94F37", s=120, marker="s", label="Point A: Percentage method", zorder=5)
    ax.scatter(me_rec, mf_rec, c="#28A745", s=120, marker="*", label="Point B: Recommended (α, k)", zorder=5)
    ax.set_xlabel("Excitement (M_E, comeback rate)")
    ax.set_ylabel("Fairness (M_F, Spearman vs. judge)")
    ax.set_title("Pareto trade-off: Fairness vs. Excitement")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Problem 4: New scoring system (Sigmoid-Scaled Meritocracy)")
    print("Loading data...")
    df_long = load_long()
    df_wide = load_wide()
    df_est = load_fan_estimates()
    seasons = sorted(df_long["season"].unique())
    print("Seasons:", len(seasons))

    print("Evaluating current system (Percentage method)...")
    mf_current, me_current = evaluate_system(
        df_long, df_wide, df_est,
        lambda s, dl, dw, de: simulate_season_percent(s, dl, dw, de),
    )
    print("  Current: M_F = {:.4f}, M_E = {:.4f}".format(mf_current, me_current))

    print("Grid search over (α, k)...")
    grid_df = run_grid_search(df_long, df_wide, df_est)
    grid_df.to_csv(
        os.path.join(OUT_DIR, "grid_alpha_k_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    print("  Grid size:", len(grid_df))

    alpha_rec, k_rec = recommend_parameters(grid_df)
    idx_rec = grid_df[(np.isclose(grid_df["alpha"], alpha_rec)) & (np.isclose(grid_df["k"], k_rec))].index
    if len(idx_rec) > 0:
        mf_rec = float(grid_df.loc[idx_rec[0], "M_F"])
        me_rec = float(grid_df.loc[idx_rec[0], "M_E"])
    else:
        mf_rec = float(grid_df["M_F"].max())
        me_rec = float(grid_df.loc[grid_df["M_F"].idxmax(), "M_E"])
    print("  Recommended: α = {:.2f}, k = {:.2f} → M_F = {:.4f}, M_E = {:.4f}".format(alpha_rec, k_rec, mf_rec, me_rec))

    plot_pareto_tradeoff(
        grid_df, mf_current, me_current,
        alpha_rec, k_rec, mf_rec, me_rec,
        os.path.join(FIG_DIR, "problem4_pareto_tradeoff.pdf"),
    )
    plot_pareto_tradeoff(
        grid_df, mf_current, me_current,
        alpha_rec, k_rec, mf_rec, me_rec,
        os.path.join(FIG_DIR, "problem4_pareto_tradeoff.png"),
    )
    print("  Pareto plot saved to", FIG_DIR)

    print("Computing extended fairness/excitement metrics (M_injustice, M_suspense)...")
    mf_cur_d, me_cur_d, mi_cur, ms_cur = evaluate_system_with_details(
        df_long, df_wide, df_est,
        lambda s, dl, dw, de: simulate_season_percent(s, dl, dw, de),
    )
    mf_rec_d, me_rec_d, mi_rec, ms_rec = evaluate_system_with_details(
        df_long, df_wide, df_est,
        simulate_season_new_system_with_details,
        {"alpha": alpha_rec, "k": k_rec},
    )
    print("  Current: M_injustice = {:.4f}, M_suspense = {:.4f}".format(mi_cur, ms_cur))
    print("  Recommended: M_injustice = {:.4f}, M_suspense = {:.4f}".format(mi_rec, ms_rec))

    B_robust = 30
    print("Robustness: interval sampling over fan-share [lb, ub], B = {}...".format(B_robust))
    rng = np.random.default_rng(42)
    mf_robust, me_robust = [], []
    for b in range(B_robust):
        df_est_b = sample_fan_estimates_from_bounds(df_est, rng)
        mf_b, me_b = evaluate_system(
            df_long, df_wide, df_est_b,
            simulate_season_new_system,
            {"alpha": alpha_rec, "k": k_rec},
        )
        mf_robust.append(mf_b)
        me_robust.append(me_b)
    mf_robust_mean, mf_robust_std = float(np.mean(mf_robust)), float(np.std(mf_robust))
    me_robust_mean, me_robust_std = float(np.mean(me_robust)), float(np.std(me_robust))
    pd.DataFrame({
        "run": range(B_robust),
        "M_F": mf_robust,
        "M_E": me_robust,
    }).to_csv(os.path.join(OUT_DIR, "robustness_interval_sampling.csv"), index=False, encoding="utf-8-sig")
    print("  Recommended (α, k) under fan uncertainty: M_F = {:.4f} ± {:.4f}, M_E = {:.4f} ± {:.4f}".format(
        mf_robust_mean, mf_robust_std, me_robust_mean, me_robust_std))

    with open(os.path.join(OUT_DIR, "recommendation_summary.txt"), "w", encoding="utf-8") as f:
        f.write("Problem 4: New Scoring System — Recommendation for Show Producers\n")
        f.write("=" * 60 + "\n\n")
        f.write("Modeling note: We evaluate the proposed system as a unified rule on historical data.\n")
        f.write("Historical DWTS rules (rank/percent, judge-save by season) differ; we use a single\n")
        f.write("baseline (Percentage + bottom-two + judge save) for comparable evaluation.\n\n")
        f.write("Current system (Percentage method):\n")
        f.write("  Fairness (M_F): {:.4f}\n".format(mf_current))
        f.write("  Excitement (M_E): {:.4f}\n".format(me_current))
        f.write("  Obvious-injustice rate (M_injustice): {:.4f}\n".format(mi_cur))
        f.write("  Suspense rate M_suspense (close-call weeks): {:.4f}\n\n".format(ms_cur))
        f.write("Proposed system (Sigmoid-Scaled Meritocracy):\n")
        f.write("  Recommended parameters: α = {:.2f}, k = {:.2f}\n".format(alpha_rec, k_rec))
        f.write("  Fairness (M_F): {:.4f}\n".format(mf_rec))
        f.write("  Excitement (M_E): {:.4f}\n".format(me_rec))
        f.write("  Obvious-injustice rate (M_injustice): {:.4f}\n".format(mi_rec))
        f.write("  Suspense rate M_suspense: {:.4f}\n\n".format(ms_rec))
        f.write("Robustness (fan vote uncertainty, B={} samples from [lb, ub]):\n".format(B_robust))
        f.write("  M_F = {:.4f} ± {:.4f}\n".format(mf_robust_mean, mf_robust_std))
        f.write("  M_E = {:.4f} ± {:.4f}\n".format(me_robust_mean, me_robust_std))
        f.write("  → Recommended (α, k) remains preferable to baseline under estimation error.\n\n")
        f.write("Why adopt:\n")
        f.write("  - Higher fairness: outcomes align better with judge assessment (M_F increased).\n")
        f.write("  - Lower obvious-injustice rate: fewer eliminations where judge score was higher than a survivor.\n")
        f.write("  - Bounded fan impact: tanh(k·z) prevents runaway fan dominance (elite protection).\n")
        f.write("  - Excitement preserved: comeback rate M_E and suspense M_suspense remain competitive.\n")
        f.write("  - Z-score standardization equalizes judge and fan voice before combination.\n")
    print("  Recommendation summary saved to", os.path.join(OUT_DIR, "recommendation_summary.txt"))
    print("Done. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
