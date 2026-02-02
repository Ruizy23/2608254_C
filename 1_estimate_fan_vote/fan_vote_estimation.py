import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Dict, Union
from scipy.optimize import linprog

BASE_PATH = "/8202302003"
DWTS_LONG = os.path.join(BASE_PATH, "dwts_long.csv")
DWTS_CLEANED = os.path.join(BASE_PATH, "dwts_cleaned.csv")
OUT_ESTIMATES = os.path.join(BASE_PATH, "fan_vote_estimates.csv")
OUT_BOUNDS = os.path.join(BASE_PATH, "fan_vote_bounds.csv")
OUT_CONSISTENCY = os.path.join(BASE_PATH, "consistency_report.csv")


def _data_path(path: str, base: Optional[str] = None) -> str:
    base = base or BASE_PATH
    if os.path.exists(path):
        return path
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.basename(path))
    return alt if os.path.exists(alt) else path


def load_cleaned_data(base_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = base_path or BASE_PATH
    long_path = _data_path(DWTS_LONG, base)
    wide_path = _data_path(DWTS_CLEANED, base)
    df_long = pd.read_csv(long_path)
    df_wide = pd.read_csv(wide_path)
    return df_long, df_wide


def get_rule_by_season(season: int) -> str:
    if season in (1, 2):
        return "rank"
    if 3 <= season <= 27:
        return "percent"
    if 28 <= season <= 34:
        return "rank"
    return "rank"


def uses_judges_save(season: int) -> bool:
    return season >= 28


def _get_bottom_k_indices(combined_rank: np.ndarray, k: int) -> set:
    combined = np.asarray(combined_rank, dtype=float).ravel()
    idx = np.argsort(-combined)[:k]
    return set(idx.tolist())


def calculate_elimination_rank_method(
    judge_scores: np.ndarray, fan_votes: np.ndarray,
    return_bottom_two: bool = False,
) -> Union[int, Tuple[int, set]]:
    j_rank = _rank_desc(np.asarray(judge_scores, dtype=float))
    f_rank = _rank_desc(np.asarray(fan_votes, dtype=float))
    combined = j_rank + f_rank
    worst = int(np.argmax(combined))
    if return_bottom_two:
        bottom_two = _get_bottom_k_indices(combined, 2)
        return worst, bottom_two
    return worst


def calculate_elimination_percent_method(
    judge_scores: np.ndarray, fan_votes: np.ndarray
) -> int:
    j = np.asarray(judge_scores, dtype=float).ravel()
    f = np.asarray(fan_votes, dtype=float).ravel()
    j_sum = np.nansum(j)
    f_sum = np.nansum(f)
    if j_sum <= 0 or f_sum <= 0:
        return 0
    j_pct = j / j_sum
    f_pct = f / f_sum
    combined = j_pct + f_pct
    return int(np.argmin(combined))


def _get_contestants_still_in_competition(
    season: int, week: int, df_long: pd.DataFrame, df_wide: pd.DataFrame
) -> List[str]:
    in_season = df_wide[df_wide["season"] == season]
    still_in = []
    for _, row in in_season.iterrows():
        last = row.get("last_week_competed")
        if pd.isna(last):
            continue
        if int(last) < week:
            continue
        elim_week = row.get("eliminated_week")
        if pd.notna(elim_week) and int(elim_week) < week:
            continue
        still_in.append(row["celebrity_name"])
    return still_in


def _get_judge_scores_this_week(
    season: int, week: int, contestants: List[str], df_long: pd.DataFrame
) -> Tuple[List[str], np.ndarray]:
    grp = df_long[(df_long["season"] == season) & (df_long["week"] == week)]
    order = []
    scores = []
    for name in contestants:
        row = grp[grp["celebrity_name"] == name]
        if row.empty:
            continue
        s = row["total_judge_score"].iloc[0]
        if pd.isna(s) or float(s) < 0:
            continue
        order.append(name)
        scores.append(float(s))
    return order, np.array(scores) if scores else np.array([])


def _get_eliminated_this_week(
    season: int, week: int, df_wide: pd.DataFrame
) -> List[str]:
    candidates = df_wide[
        (df_wide["season"] == season)
        & (df_wide["result_type"] == "eliminated")
        & (df_wide["eliminated_week"].notna())
    ]
    if candidates.empty:
        return []
    elim = candidates[candidates["eliminated_week"].astype(int) == week]
    return elim["celebrity_name"].tolist()


def build_contexts_week_by_week_simulation(
    df_long: pd.DataFrame, df_wide: pd.DataFrame
) -> List[Dict]:
    seasons = sorted(df_long["season"].unique())
    contexts = []
    for season in seasons:
        max_week = int(df_long[df_long["season"] == season]["week"].max())
        for week in range(1, max_week + 1):
            current_contestants = _get_contestants_still_in_competition(
                season, week, df_long, df_wide
            )
            if not current_contestants:
                continue
            contestants, judge_totals = _get_judge_scores_this_week(
                season, week, current_contestants, df_long
            )
            if len(contestants) == 0 or len(judge_totals) == 0:
                continue
            total_judge = np.nansum(judge_totals)
            if total_judge <= 0 or np.any(np.isnan(judge_totals)):
                continue
            eliminated_names = _get_eliminated_this_week(season, week, df_wide)
            if len(eliminated_names) == 0:
                continue
            idx_elim_list = []
            for name in eliminated_names:
                if name in contestants:
                    idx_elim_list.append(contestants.index(name))
            if len(idx_elim_list) != len(eliminated_names):
                continue
            n = len(contestants)
            judge_pct = judge_totals / total_judge
            judge_rank_1based = _rank_desc(judge_totals)
            rule = get_rule_by_season(season)
            judges_save = uses_judges_save(season)
            contexts.append({
                "season": season,
                "week": week,
                "contestants": contestants,
                "judge_totals": judge_totals.copy(),
                "judge_pct": judge_pct.copy(),
                "judge_rank_1based": judge_rank_1based.copy(),
                "idx_elim": idx_elim_list,
                "eliminated_names": eliminated_names.copy(),
                "voting_method": rule,
                "judges_save": judges_save,
                "n": n,
            })
    return contexts


def get_weekly_elimination_context(
    df_long: pd.DataFrame, df_wide: pd.DataFrame
) -> List[Dict]:
    return build_contexts_week_by_week_simulation(df_long, df_wide)


def _rank_desc(x: np.ndarray) -> np.ndarray:
    order = np.argsort(-np.asarray(x, dtype=float))
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1, dtype=float)
    return r


def layer1_feasible_percent(
    judge_pct: np.ndarray, idx_elim: List[int], n: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    j = np.asarray(judge_pct, dtype=float).ravel()
    elim_set = set(idx_elim)
    if len(j) != n or any(e < 0 or e >= n for e in elim_set):
        return None, None, "dim mismatch"
    rows = []
    rhs = []
    for E in elim_set:
        for i in range(n):
            if i in elim_set:
                continue
            row = np.zeros(n)
            row[i] = -1
            row[E] = 1
            rows.append(row)
            rhs.append(-float(j[E] - j[i]))
    A_eq = np.ones((1, n))
    b_eq = np.array([1.0])
    lb_all = np.zeros(n)
    ub_all = np.ones(n)
    if not rows:
        A_ub = np.zeros((1, n))
        b_ub = np.array([0.0])
    else:
        A_ub = np.array(rows)
        b_ub = np.array(rhs)

    for k in range(n):
        c = np.zeros(n)
        c[k] = 1
        res_min = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
        if res_min.success:
            lb_all[k] = float(res_min.x[k])
        res_max = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1), method="highs")
        if res_max.success:
            ub_all[k] = float(res_max.x[k])

    return lb_all, ub_all, None


def layer1_feasible_rank(
    judge_rank_1based: np.ndarray,
    idx_elim: List[int],
    n: int,
    judges_save: bool = False,
    n_samples: int = 5000,
    prior_mean_prev_week: Optional[np.ndarray] = None,
    prior_strength: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    jr = np.asarray(judge_rank_1based, dtype=float).ravel()
    elim_set = set(idx_elim)
    if len(jr) != n or any(e < 0 or e >= n for e in elim_set):
        return np.zeros((0, n)), np.zeros(n), np.ones(n)
    judge_scores_fake = np.max(jr) + 1 - jr
    k_elim = len(elim_set)

    if prior_mean_prev_week is not None and len(prior_mean_prev_week) == n:
        alpha = np.maximum(
            prior_strength * np.asarray(prior_mean_prev_week, dtype=float).ravel(),
            1e-6,
        )
    else:
        alpha = np.ones(n)

    def is_feasible(f: np.ndarray) -> bool:
        _, bottom_two = calculate_elimination_rank_method(
            judge_scores_fake, f, return_bottom_two=True
        )
        bottom_k = _get_bottom_k_indices(
            _rank_desc(judge_scores_fake) + _rank_desc(f), k_elim
        )
        if judges_save and k_elim == 1:
            return idx_elim[0] in bottom_two
        return elim_set == bottom_k

    samples = []
    for _ in range(n_samples):
        f = np.random.dirichlet(alpha)
        if is_feasible(f):
            samples.append(f)
    if not samples:
        for _ in range(n_samples * 2):
            f = np.random.dirichlet(alpha)
            if is_feasible(f):
                samples.append(f)
    arr = np.array(samples) if samples else np.zeros((0, n))
    lb = np.min(arr, axis=0) if len(arr) > 0 else np.zeros(n)
    ub = np.max(arr, axis=0) if len(arr) > 0 else np.ones(n)
    return arr, lb, ub


def layer2_posterior_from_feasible(
    lb: np.ndarray, ub: np.ndarray,
    feasible_samples: Optional[np.ndarray] = None,
    prior_alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:

    n = len(lb)
    if feasible_samples is not None and len(feasible_samples) > 0:
        mean_est = np.mean(feasible_samples, axis=0)
        std_est = np.std(feasible_samples, axis=0)
        return mean_est, std_est
    mean_est = 0.5 * (lb + ub)
    width = np.maximum(ub - lb, 1e-6)
    std_est = width / 4.0
    return mean_est, std_est


def run_layer1_layer2_for_context(
    ctx: Dict,
    rank_mc_samples: int = 50000,
    prior_mean_prev_week: Optional[np.ndarray] = None,
    prior_strength: float = 5.0,
) -> Dict:
    n = ctx["n"]
    idx_elim_list = ctx["idx_elim"] if isinstance(ctx["idx_elim"], list) else [ctx["idx_elim"]]
    judge_pct = ctx["judge_pct"]
    judge_rank = ctx["judge_rank_1based"]
    method = ctx["voting_method"]
    judges_save = ctx.get("judges_save", False)

    if method == "percent":
        lb, ub, err = layer1_feasible_percent(judge_pct, idx_elim_list, n)
        if err is not None:
            return {"error": err, "context": ctx}
        feasible_samples = None
    else:
        feasible_samples, lb, ub = layer1_feasible_rank(
            judge_rank, idx_elim_list, n,
            judges_save=judges_save,
            n_samples=rank_mc_samples,
            prior_mean_prev_week=prior_mean_prev_week,
            prior_strength=prior_strength,
        )
        if len(feasible_samples) == 0:
            return {"error": "no feasible rank sample", "context": ctx}
        err = None

    mean_est, std_est = layer2_posterior_from_feasible(lb, ub, feasible_samples)
    return {
        "season": ctx["season"],
        "week": ctx["week"],
        "contestants": ctx["contestants"],
        "eliminated_names": ctx["eliminated_names"],
        "voting_method": method,
        "fan_share_mean": mean_est,
        "fan_share_std": std_est,
        "fan_share_lb": lb,
        "fan_share_ub": ub,
        "context": ctx,
        "error": None,
    }


def check_consistency(res: Dict) -> Tuple[bool, str]:
    if res.get("error"):
        return False, res["error"]
    method = res["voting_method"]
    mean_f = np.asarray(res["fan_share_mean"])
    ctx = res.get("context")
    if ctx is None:
        return True, "no context to check"
    idx_elim_list = ctx["idx_elim"] if isinstance(ctx["idx_elim"], list) else [ctx["idx_elim"]]
    elim_set = set(idx_elim_list)
    judge_totals = ctx["judge_totals"]
    judges_save = ctx.get("judges_save", False)
    k = len(elim_set)

    if method == "percent":
        combined = ctx["judge_pct"] + mean_f
        pred_bottom_k = set(np.argsort(combined)[:k].tolist())
        ok = elim_set == pred_bottom_k
        return ok, "consistent" if ok else "pred_bottom_k_{}".format(pred_bottom_k)
    else:
        _, bottom_two = calculate_elimination_rank_method(
            judge_totals, mean_f, return_bottom_two=True
        )
        combined_rank = _rank_desc(judge_totals) + _rank_desc(mean_f)
        pred_bottom_k = _get_bottom_k_indices(combined_rank, k)
        if judges_save and k == 1:
            ok = idx_elim_list[0] in bottom_two
        else:
            ok = elim_set == pred_bottom_k
        return ok, "consistent" if ok else "pred_bottom_k_{}".format(pred_bottom_k)


def build_estimates_table(contexts: List[Dict], results: List[Dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        if res.get("error"):
            continue
        eliminated_names = res.get("eliminated_names", [])
        if isinstance(eliminated_names, str):
            eliminated_names = [eliminated_names]
        for i, name in enumerate(res["contestants"]):
            rows.append({
                "season": res["season"],
                "week": res["week"],
                "celebrity_name": name,
                "voting_method": res["voting_method"],
                "fan_share_mean": res["fan_share_mean"][i],
                "fan_share_std": res["fan_share_std"][i],
                "fan_share_lb": res["fan_share_lb"][i],
                "fan_share_ub": res["fan_share_ub"][i],
                "is_eliminated_this_week": 1 if name in eliminated_names else 0,
            })
    return pd.DataFrame(rows)


def build_consistency_report(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        ok, msg = check_consistency(res)
        rows.append({
            "season": res.get("season"),
            "week": res.get("week"),
            "voting_method": res.get("voting_method"),
            "consistent": ok,
            "message": msg,
            "error": res.get("error"),
        })
    return pd.DataFrame(rows)


def apply_temporal_smoothing(
    estimates_df: pd.DataFrame, alpha: float = 0.3
) -> pd.DataFrame:
    if estimates_df.empty or "fan_share_mean" not in estimates_df.columns:
        return estimates_df
    df = estimates_df.copy()
    df["fan_share_mean_raw"] = df["fan_share_mean"]
    out = []
    for (season, celebrity_name), grp in df.groupby(["season", "celebrity_name"]):
        grp = grp.sort_values("week").reset_index(drop=True)
        raw = grp["fan_share_mean_raw"].values.astype(float)
        smoothed = np.empty_like(raw)
        smoothed[0] = raw[0]
        for t in range(1, len(raw)):
            smoothed[t] = (1 - alpha) * raw[t] + alpha * smoothed[t - 1]
        grp = grp.copy()
        grp["fan_share_mean"] = smoothed
        out.append(grp)
    if not out:
        return df
    return pd.concat(out, ignore_index=True)


def run_estimation(
    base_path: Optional[str] = None,
    rank_mc_samples: int = 50000,
    prior_strength: float = 5.0,
    use_bayesian_filter: bool = True,
    temporal_smoothing_alpha: Optional[float] = 0.3,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = base_path or BASE_PATH
    df_long, df_wide = load_cleaned_data(base)
    contexts = get_weekly_elimination_context(df_long, df_wide)

    results = []
    last_posterior_by_season: Dict[int, Dict] = {}

    for ctx in contexts:
        prior_mean = None
        if use_bayesian_filter and ctx["voting_method"] == "rank":
            season = ctx["season"]
            if season in last_posterior_by_season:
                prev = last_posterior_by_season[season]
                prev_names = prev["contestants"]
                prev_mean = prev["mean"]
                prior_mean = np.zeros(ctx["n"])
                for i, name in enumerate(ctx["contestants"]):
                    if name in prev_names:
                        j = prev_names.index(name)
                        prior_mean[i] = prev_mean[j]
                s = prior_mean.sum()
                if s > 1e-9:
                    prior_mean = prior_mean / s
                else:
                    prior_mean = None

        res = run_layer1_layer2_for_context(
            ctx,
            rank_mc_samples=rank_mc_samples,
            prior_mean_prev_week=prior_mean,
            prior_strength=prior_strength,
        )
        res["context"] = ctx
        results.append(res)
        if not res.get("error") and res.get("fan_share_mean") is not None:
            last_posterior_by_season[ctx["season"]] = {
                "contestants": ctx["contestants"],
                "mean": np.asarray(res["fan_share_mean"]).copy(),
            }

    estimates_df = build_estimates_table(contexts, results)
    if temporal_smoothing_alpha is not None:
        estimates_df = apply_temporal_smoothing(estimates_df, alpha=temporal_smoothing_alpha)
    consistency_df = build_consistency_report(results)
    bounds_df = estimates_df[["season", "week", "celebrity_name", "fan_share_lb", "fan_share_ub"]].copy()

    return estimates_df, bounds_df, consistency_df


def main():
    base = BASE_PATH
    if not os.path.exists(_data_path(DWTS_LONG, base)):
        base = os.path.dirname(os.path.abspath(__file__))
    estimates_df, bounds_df, consistency_df = run_estimation(base_path=base)

    out_dir = base
    for p in [OUT_ESTIMATES, OUT_BOUNDS, OUT_CONSISTENCY]:
        d = os.path.dirname(p)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
    estimates_df.to_csv(
        os.path.join(out_dir, os.path.basename(OUT_ESTIMATES)), index=False, encoding="utf-8-sig"
    )
    bounds_df.to_csv(
        os.path.join(out_dir, os.path.basename(OUT_BOUNDS)), index=False, encoding="utf-8-sig"
    )
    consistency_df.to_csv(
        os.path.join(out_dir, os.path.basename(OUT_CONSISTENCY)), index=False, encoding="utf-8-sig"
    )

    n_weeks = consistency_df.shape[0]
    n_consistent = (
        consistency_df["consistent"].sum() if "consistent" in consistency_df.columns else 0
    )
    print("Fan vote estimation done. Weeks:", n_weeks, "Consistent:", n_consistent)
    print("Outputs:", OUT_ESTIMATES, OUT_BOUNDS, OUT_CONSISTENCY)
    return estimates_df, bounds_df, consistency_df


if __name__ == "__main__":
    main()
