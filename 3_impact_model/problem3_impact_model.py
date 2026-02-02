from __future__ import annotations

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from typing import Optional, Tuple
from scipy.special import logit

SERVER_BASE = "/8202302003"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_CLEAN_DIR = os.path.join(PROJECT_ROOT, "数据清洗")
Q1_DIR = os.path.join(PROJECT_ROOT, "问题一")
if os.path.exists(SERVER_BASE):
    DATA_CLEAN_DIR = os.path.join(SERVER_BASE, "数据清洗")
    Q1_DIR = os.path.join(SERVER_BASE, "问题一")
    OUT_DIR = os.path.join(SERVER_BASE, "问题三", "outputs")
else:
    OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(step: int) -> str:
    return os.path.join(CACHE_DIR, f"step{step}.pkl")


def _load_cache(step: int):
    p = _cache_path(step)
    if os.path.isfile(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(step: int, data: dict) -> None:
    with open(_cache_path(step), "wb") as f:
        pickle.dump(data, f)


def _path(name: str) -> str:
    for base in [Q1_DIR, DATA_CLEAN_DIR, SCRIPT_DIR, PROJECT_ROOT]:
        p = os.path.join(base, name)
        if os.path.exists(p):
            return p
    return os.path.join(SCRIPT_DIR, name)


def load_long() -> pd.DataFrame:
    return pd.read_csv(_path("dwts_long.csv"))


def load_fan_estimates() -> pd.DataFrame:
    return pd.read_csv(_path("fan_vote_estimates.csv"))


def build_analysis_df(df_long: pd.DataFrame, df_fan: pd.DataFrame) -> pd.DataFrame:
    fan_cols = ["season", "week", "celebrity_name", "fan_share_mean"]
    if "fan_share_lb" in df_fan.columns:
        fan_cols.extend(["fan_share_lb", "fan_share_ub"])
    if "fan_share_std" in df_fan.columns:
        fan_cols.append("fan_share_std")
    fan_cols = [c for c in fan_cols if c in df_fan.columns]
    df = df_long.copy()
    df = df[df["percentage_score_capped"].notna() & (df["percentage_score_capped"] > 0) & (df["percentage_score_capped"] < 1)].copy()
    df = df.merge(df_fan[fan_cols], on=["season", "week", "celebrity_name"], how="left")
    df = df[df["fan_share_mean"].notna()].copy()
    if "fan_share_lb" not in df.columns:
        df["fan_share_lb"] = (df["fan_share_mean"] - 0.01).clip(1e-6, None)
    if "fan_share_ub" not in df.columns:
        df["fan_share_ub"] = (df["fan_share_mean"] + 0.01).clip(None, 1 - 1e-6)
    df["fan_share_lb"] = df["fan_share_lb"].clip(1e-6, 1 - 1e-6)
    df["fan_share_ub"] = df["fan_share_ub"].clip(1e-6, 1 - 1e-6)
    df["fan_share_clip"] = df["fan_share_mean"].clip(1e-6, 1 - 1e-6)
    df["Y_judge"] = df["percentage_score_capped"].astype(float)
    df["Y_fan"] = df["fan_share_clip"].astype(float)
    df["E_elim"] = (df["eliminated_week"].notna() & (df["eliminated_week"].astype(float) == df["week"])).astype(int)
    df = df.sort_values(["celebrity_name", "season", "week"])
    df["lag_judge"] = df.groupby(["celebrity_name", "season"])["Y_judge"].shift(1)
    df["lag_fan"] = df.groupby(["celebrity_name", "season"])["Y_fan"].shift(1)
    df["has_bonus"] = (df["has_bonus_this_week"].fillna(False)).astype(int)
    df["week_norm"] = (df["week"].astype(float) - 1) / df["season_total_weeks"].clip(1).astype(float)
    df["pro_id"] = pd.Categorical(df["ballroom_partner"]).codes
    df["season_id"] = pd.Categorical(df["season"]).codes
    df["age"] = pd.to_numeric(df["celebrity_age_during_season"], errors="coerce").fillna(df["celebrity_age_during_season"].median())
    df = df.dropna(subset=["age", "Y_judge", "Y_fan"])
    return df.reset_index(drop=True)

def build_contestant_season_df(df_long: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "celebrity_name", "season", "ballroom_partner", "celebrity_industry",
        "celebrity_homecountry_region", "celebrity_age_during_season",
        "placement", "last_week_competed", "season_total_weeks",
    ]
    available = [c for c in keep_cols if c in df_long.columns]
    sub = df_long[available].drop_duplicates(subset=["celebrity_name", "season"], keep="first")
    sub = sub.rename(columns={"last_week_competed": "weeks_survived"})
    sub["age"] = pd.to_numeric(sub["celebrity_age_during_season"], errors="coerce")
    sub = sub.dropna(subset=["placement", "weeks_survived", "age"])
    sub["placement"] = pd.to_numeric(sub["placement"], errors="coerce")
    sub["weeks_survived"] = pd.to_numeric(sub["weeks_survived"], errors="coerce")
    sub = sub.dropna(subset=["placement", "weeks_survived"])
    return sub.reset_index(drop=True)


def build_design_contestant_season(cs_df: pd.DataFrame) -> pd.DataFrame:
    df = cs_df.copy()
    industry = df["celebrity_industry"].fillna("Other").astype(str)
    top_ind = industry.value_counts().head(12).index.tolist()
    df["industry_cat"] = industry.where(industry.isin(top_ind), "Other")
    region = df["celebrity_homecountry_region"].fillna("Other").astype(str)
    top_reg = region.value_counts().head(8).index.tolist()
    df["region_cat"] = region.where(region.isin(top_reg), "Other")
    ind_dum = pd.get_dummies(df["industry_cat"], prefix="ind", drop_first=True)
    reg_dum = pd.get_dummies(df["region_cat"], prefix="reg", drop_first=True)
    pro_dum = pd.get_dummies(df["ballroom_partner"].astype(str), prefix="pro", drop_first=True)
    sea_dum = pd.get_dummies(df["season"].astype(str), prefix="sea", drop_first=True)
    X = pd.DataFrame({
        "intercept": 1.0,
        "age": (df["age"] - df["age"].mean()) / (df["age"].std() + 1e-8),
    }, index=df.index)
    X = pd.concat([X, ind_dum, reg_dum, pro_dum, sea_dum], axis=1)
    X = X.astype(np.float64)
    return X

def build_design_lmm(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    X = pd.DataFrame({
        "intercept": 1.0,
        "age": (df["age"] - df["age"].mean()) / (df["age"].std() + 1e-8),
        "week_norm": df["week_norm"].astype(np.float64),
        "has_bonus": df["has_bonus"].astype(np.float64),
        "lag_judge": df["lag_judge"].fillna(df["Y_judge"].mean()).astype(np.float64),
        "lag_fan": df["lag_fan"].fillna(df["Y_fan"].mean()).astype(np.float64),
    }, index=df.index)
    X = X.astype(np.float64)
    valid = X.notna().all(axis=1)
    X = X[valid]
    groups = df.loc[valid, "ballroom_partner"]
    return X, groups


def build_design(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    industry = df["celebrity_industry"].fillna("Other").astype(str)
    top_industry = industry.value_counts().head(12).index.tolist()
    df["industry_cat"] = industry.where(industry.isin(top_industry), "Other")
    region = df["celebrity_homecountry_region"].fillna("Other").astype(str)
    top_region = region.value_counts().head(8).index.tolist()
    df["region_cat"] = region.where(region.isin(top_region), "Other")
    ind_dum = pd.get_dummies(df["industry_cat"], prefix="ind", drop_first=True)
    reg_dum = pd.get_dummies(df["region_cat"], prefix="reg", drop_first=True)
    sea_dum = pd.get_dummies(df["season"], prefix="sea", drop_first=True)
    X = pd.DataFrame({
        "intercept": 1.0,
        "age": (df["age"] - df["age"].mean()) / (df["age"].std() + 1e-8),
        "week_norm": df["week_norm"].astype(np.float64),
        "has_bonus": df["has_bonus"].astype(np.float64),
        "lag_judge": df["lag_judge"].fillna(df["Y_judge"].mean()).astype(np.float64),
        "lag_fan": df["lag_fan"].fillna(df["Y_fan"].mean()).astype(np.float64),
    }, index=df.index)
    X = pd.concat([X, ind_dum, reg_dum, sea_dum], axis=1)
    X = X.astype(np.float64)
    valid = X.notna().all(axis=1)
    X = X[valid]
    groups = df.loc[valid, "ballroom_partner"]
    return X, groups


def get_y_and_valid(df: pd.DataFrame, X: pd.DataFrame, groups: pd.Series, target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sub = df.loc[X.index]
    y = np.asarray(sub[target].values, dtype=np.float64)
    ok = np.isfinite(y)
    if target == "Y_fan":
        ok &= (y > 0) & (y < 1)
    y = y[ok]
    X_mat = np.asarray(X.loc[X.index[ok]].values, dtype=np.float64)
    g = groups.loc[groups.index[ok]].values
    return y, X_mat, g


def _filter_small_groups(y: np.ndarray, X_mat: np.ndarray, g: np.ndarray, min_per_group: int = 2):
    import pandas as pd
    gg = pd.Series(g)
    cnt = gg.groupby(gg).transform("count")
    keep = (cnt >= min_per_group).values
    return y[keep], X_mat[keep], g[keep]


def _drop_constant_columns(X_mat: np.ndarray, col_names: list, tol: float = 1e-10) -> Tuple[np.ndarray, list]:
    var = np.var(X_mat, axis=0)
    keep = var > tol
    if keep.all():
        return X_mat, col_names
    return X_mat[:, keep], [c for c, k in zip(col_names, keep) if k]


def fit_judge_lmm(df: pd.DataFrame):
    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.regression.linear_model import OLS
    X, groups = build_design_lmm(df)
    y, X_mat, g = get_y_and_valid(df, X, groups, "Y_judge")
    y, X_mat, g = _filter_small_groups(y, X_mat, g, min_per_group=2)
    col_names = X.columns.tolist()
    X_mat, col_names = _drop_constant_columns(X_mat, col_names)
    if X_mat.size == 0 or X_mat.shape[0] < X_mat.shape[1] + 5:
        raise ValueError("Too few observations after filtering for LMM.")
    model = MixedLM(y, X_mat, groups=g)
    try:
        result = model.fit(method="powell", maxiter=500)
        return result, col_names, g
    except (np.linalg.LinAlgError, Exception):
        exog = pd.DataFrame(X_mat, columns=col_names)
        result = OLS(y, exog).fit()
        result.random_effects = {}
        return result, col_names, g


def fit_fan_lmm(df: pd.DataFrame):
    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.regression.linear_model import OLS
    X, groups = build_design_lmm(df)
    y, X_mat, g = get_y_and_valid(df, X, groups, "Y_fan")
    y, X_mat, g = _filter_small_groups(y, X_mat, g, min_per_group=2)
    col_names = X.columns.tolist()
    X_mat, col_names = _drop_constant_columns(X_mat, col_names)
    y_logit = logit(y)
    if X_mat.size == 0 or X_mat.shape[0] < X_mat.shape[1] + 5:
        raise ValueError("Too few observations after filtering for LMM.")
    model = MixedLM(y_logit, X_mat, groups=g)
    try:
        result = model.fit(method="powell", maxiter=500)
        return result, col_names, g
    except (np.linalg.LinAlgError, Exception):
        exog = pd.DataFrame(X_mat, columns=col_names)
        result = OLS(y_logit, exog).fit()
        result.random_effects = {}
        return result, col_names, g

def sample_fan_shares_from_interval(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    lb = df["fan_share_lb"].values.astype(float)
    ub = df["fan_share_ub"].values.astype(float)
    lb, ub = np.minimum(lb, ub), np.maximum(lb, ub)
    u = rng.uniform(0, 1, size=len(df))
    df["Y_fan"] = (lb + u * (ub - lb)).clip(1e-6, 1 - 1e-6)
    df = df.sort_values(["celebrity_name", "season", "week"])
    df["lag_fan"] = df.groupby(["celebrity_name", "season"])["Y_fan"].shift(1)
    df["lag_fan"] = df["lag_fan"].fillna(df["Y_fan"].mean())
    return df


def fit_fan_lmm_with_sampled_y(df: pd.DataFrame) -> Optional[pd.Series]:
    from statsmodels.regression.mixed_linear_model import MixedLM
    from statsmodels.regression.linear_model import OLS
    X, groups = build_design_lmm(df)
    y, X_mat, g = get_y_and_valid(df, X, groups, "Y_fan")
    y, X_mat, g = _filter_small_groups(y, X_mat, g, min_per_group=2)
    col_names = X.columns.tolist()
    X_mat, col_names = _drop_constant_columns(X_mat, col_names)
    if len(y) < 10 or X_mat.shape[0] < X_mat.shape[1] + 5:
        return None
    y_logit = logit(y)
    model = MixedLM(y_logit, X_mat, groups=g)
    try:
        result = model.fit(method="powell", maxiter=500)
        return result.params
    except (np.linalg.LinAlgError, Exception):
        try:
            exog = pd.DataFrame(X_mat, columns=col_names)
            result = OLS(y_logit, exog).fit()
            return result.params
        except Exception:
            return None


def fan_coefficient_intervals(
    df_base: pd.DataFrame,
    B: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    import warnings
    try:
        from statsmodels.tools.sm_exceptions import ConvergenceWarning as SM_ConvWarning
    except ImportError:
        SM_ConvWarning = UserWarning
    rng = np.random.default_rng(seed)
    param_list = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=SM_ConvWarning)
        warnings.filterwarnings("ignore", message=".*[Bb]oundary.*", module="statsmodels")
        for i in range(B):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Interval sampling {i + 1}/{B}...")
            df_b = sample_fan_shares_from_interval(df_base, rng)
            params = fit_fan_lmm_with_sampled_y(df_b)
            if params is not None:
                param_list.append(params)
    if not param_list:
        return pd.DataFrame()
    all_params = pd.DataFrame(param_list)
    summary = pd.DataFrame({
        "variable": all_params.columns.tolist(),
        "coef_mean": all_params.mean().values,
        "coef_std": all_params.std().values,
        "coef_q025": all_params.quantile(0.025).values,
        "coef_q975": all_params.quantile(0.975).values,
    })
    return summary


def fit_placement_ols(cs_df: pd.DataFrame):
    from statsmodels.regression.linear_model import OLS
    X = build_design_contestant_season(cs_df)
    y_raw = np.asarray(cs_df.loc[X.index, "placement"].values, dtype=np.float64)
    valid = X.notna().all(axis=1) & np.isfinite(y_raw)
    X = X.loc[valid].copy()
    y = y_raw[valid]
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    still = X.notna().all(axis=1)
    X = X.loc[still].astype(np.float64)
    y = np.asarray(y[still], dtype=np.float64)
    model = OLS(y, X)
    result = model.fit()
    return result, X.columns.tolist()


def fit_weeks_survived_ols(cs_df: pd.DataFrame):
    from statsmodels.regression.linear_model import OLS
    X = build_design_contestant_season(cs_df)
    y_raw = np.asarray(cs_df.loc[X.index, "weeks_survived"].values, dtype=np.float64)
    valid = X.notna().all(axis=1) & np.isfinite(y_raw)
    X = X.loc[valid].copy()
    y = y_raw[valid]
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    still = X.notna().all(axis=1)
    X = X.loc[still].astype(np.float64)
    y = np.asarray(y[still], dtype=np.float64)
    model = OLS(y, X)
    result = model.fit()
    return result, X.columns.tolist()


def _logit_design_full_rank(X_full: pd.DataFrame, y: np.ndarray, tol: float = 1e-10) -> pd.DataFrame:
    var = X_full.var(axis=0)
    keep_cols = (var > tol) | (X_full.columns == "intercept")
    X = X_full.loc[:, keep_cols].astype(np.float64)
    while True:
        r = np.linalg.matrix_rank(X.values, tol=1e-8)
        if r >= X.shape[1]:
            return X
        if X.shape[1] <= 1:
            return X
        X = X.iloc[:, :-1]


def fit_elimination_logit(df: pd.DataFrame):
    from statsmodels.discrete.discrete_model import Logit
    X, groups = build_design(df)
    df_sub = df.loc[X.index]
    y_raw = pd.Series(np.asarray(df_sub["E_elim"].values, dtype=np.float64), index=X.index)
    pro_dum = pd.get_dummies(df_sub["ballroom_partner"].astype(str), prefix="pro", drop_first=True)
    X_full = pd.concat([X, pro_dum], axis=1)
    valid = X_full.notna().all(axis=1) & np.isfinite(y_raw.values)
    X_full = X_full.loc[valid].copy()
    y = y_raw.loc[valid].values.astype(np.float64)
    for c in X_full.columns:
        X_full[c] = pd.to_numeric(X_full[c], errors="coerce")
    still_valid = X_full.notna().all(axis=1)
    X_full = X_full.loc[still_valid].astype(np.float64)
    y = y[still_valid]
    X_full = _logit_design_full_rank(X_full, y)
    model = Logit(y, X_full)
    try:
        result = model.fit(disp=0, maxiter=200)
        return result, X_full.columns.tolist()
    except np.linalg.LinAlgError:
        X_reduced = X_full[[c for c in X_full.columns if not c.startswith("pro_")]]
        if X_reduced.shape[1] < 2:
            X_reduced = X_full.iloc[:, : min(10, X_full.shape[1])]
        X_reduced = _logit_design_full_rank(X_reduced, y)
        model = Logit(y, X_reduced)
        result = model.fit(disp=0, maxiter=200)
        return result, X_reduced.columns.tolist()


def _get_param(res, name: str, name_list: list) -> float:
    p = res.params
    if hasattr(p, "get"):
        return p.get(name, np.nan)
    if isinstance(p, np.ndarray) and name_list and name in name_list:
        return float(p[name_list.index(name)])
    return np.nan


def compare_coefficients(res_judge, res_fan, fe_names: list, fe_names_judge: list = None, fe_names_fan: list = None) -> pd.DataFrame:
    idx_j = getattr(res_judge.params, "index", None)
    idx_f = getattr(res_fan.params, "index", None)
    if idx_j is not None and idx_f is not None:
        common = [n for n in fe_names if n in idx_j and n in idx_f]
    else:
        common = list(fe_names)
    name_list_j = fe_names_judge if fe_names_judge is not None else common
    name_list_f = fe_names_fan if fe_names_fan is not None else common
    rows = []
    for n in common:
        bj = _get_param(res_judge, n, name_list_j)
        bf = _get_param(res_fan, n, name_list_f)
        rows.append({
            "variable": n,
            "coef_judge": bj,
            "coef_fan": bf,
            "same_sign": (np.sign(bj) == np.sign(bf)) if np.isfinite(bj) and np.isfinite(bf) else None,
            "abs_coef_judge": np.abs(bj) if np.isfinite(bj) else np.nan,
            "abs_coef_fan": np.abs(bf) if np.isfinite(bf) else np.nan,
        })
    return pd.DataFrame(rows)


def pro_random_effects(res_judge, res_fan) -> pd.DataFrame:
    re_j = getattr(res_judge, "random_effects", None) or {}
    re_f = getattr(res_fan, "random_effects", None) or {}
    u_judge = {k: float(np.asarray(v).flat[0]) for k, v in re_j.items() if getattr(v, "size", 0) > 0}
    u_fan = {k: float(np.asarray(v).flat[0]) for k, v in re_f.items() if getattr(v, "size", 0) > 0}
    all_partners = sorted(set(u_judge) | set(u_fan))
    rows = []
    for p in all_partners:
        uj = u_judge.get(p, np.nan)
        uf = u_fan.get(p, np.nan)
        rows.append({
            "ballroom_partner": p,
            "u_pro_judge": uj,
            "u_pro_fan": uf,
            "split_style": "judge_fav" if (np.nan_to_num(uj) > np.nan_to_num(uf)) else "fan_fav",
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="DWTS 问题三：影响模型")
    parser.add_argument(
        "--from-step",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help="从第几步开始运行（1=Model1, 2=Model2, 3=Model2b, 4=Model3, 5=Model4, 6=Model5）；之前的步骤从 cache 加载",
    )
    args = parser.parse_args()
    from_step = args.from_step

    print("Loading data...")
    df_long = load_long()
    df_fan = load_fan_estimates()
    df = build_analysis_df(df_long, df_fan)
    cs_df = build_contestant_season_df(df_long)
    print("Analysis rows (week-level):", len(df), "| Contestant-season rows:", len(cs_df))

    if from_step <= 1:
        print("Fitting Model 1 (Judge score LMM)...")
        res_judge, fe_names_judge, groups_judge = fit_judge_lmm(df)
        _save_cache(1, {"res_judge": res_judge, "fe_names_judge": fe_names_judge, "groups_judge": groups_judge})
        print(res_judge.summary())
    else:
        c1 = _load_cache(1)
        if c1 is None:
            raise FileNotFoundError("Cache step1 不存在，请先完整运行一次或使用 --from-step 1")
        res_judge, fe_names_judge, groups_judge = c1["res_judge"], c1["fe_names_judge"], c1["groups_judge"]
        print("Model 1: 从 cache 加载")

    if from_step <= 2:
        print("Fitting Model 2 (Fan vote LMM, point estimate)...")
        res_fan, fe_names_fan, groups_fan = fit_fan_lmm(df)
        _save_cache(2, {"res_fan": res_fan, "fe_names_fan": fe_names_fan, "groups_fan": groups_fan})
        print(res_fan.summary())
    else:
        c2 = _load_cache(2)
        if c2 is None:
            raise FileNotFoundError("Cache step2 不存在，请先运行到 step 2 或使用 --from-step 1")
        res_fan, fe_names_fan, groups_fan = c2["res_fan"], c2["fe_names_fan"], c2["groups_fan"]
        print("Model 2: 从 cache 加载")

    if from_step <= 3:
        print("Model 2b: Interval sampling (fan_share in [lb, ub]) → multiple fits → coefficient intervals...")
        B_sampling = 50
        fan_coef_interval = fan_coefficient_intervals(df, B=B_sampling, seed=42, verbose=True)
        _save_cache(3, {"fan_coef_interval": fan_coef_interval})
        if not fan_coef_interval.empty:
            fan_coef_interval.to_csv(
                os.path.join(OUT_DIR, "fan_model_coefficient_intervals.csv"),
                index=False,
                encoding="utf-8-sig",
            )
            print("Fan coefficient intervals (measurement-error aware) saved: fan_model_coefficient_intervals.csv")
            key_fan_vars = ["age", "week_norm", "has_bonus", "lag_judge", "lag_fan"]
            key_fan_interval = fan_coef_interval[fan_coef_interval["variable"].isin(key_fan_vars)]
            key_fan_interval.to_csv(
                os.path.join(OUT_DIR, "fan_model_key_coef_intervals.csv"),
                index=False,
                encoding="utf-8-sig",
            )
    else:
        c3 = _load_cache(3)
        if c3 is None:
            raise FileNotFoundError("Cache step3 不存在")
        fan_coef_interval = c3["fan_coef_interval"]
        print("Model 2b: 从 cache 加载")

    if from_step <= 4:
        print("Fitting Model 3 (Elimination logit)...")
        res_elim, elim_names = fit_elimination_logit(df)
        _save_cache(4, {"res_elim": res_elim, "elim_names": elim_names})
        print(res_elim.summary())
    else:
        c4 = _load_cache(4)
        if c4 is None:
            raise FileNotFoundError("Cache step4 不存在")
        res_elim, elim_names = c4["res_elim"], c4["elim_names"]
        print("Model 3: 从 cache 加载")

    if from_step <= 5:
        print("Fitting Model 4 (Final placement OLS)...")
        res_placement, placement_names = fit_placement_ols(cs_df)
        _save_cache(5, {"res_placement": res_placement, "placement_names": placement_names})
        print(res_placement.summary())
    else:
        c5 = _load_cache(5)
        if c5 is None:
            raise FileNotFoundError("Cache step5 不存在")
        res_placement, placement_names = c5["res_placement"], c5["placement_names"]
        print("Model 4: 从 cache 加载")

    if from_step <= 6:
        print("Fitting Model 5 (Weeks survived OLS)...")
        res_weeks, weeks_names = fit_weeks_survived_ols(cs_df)
        _save_cache(6, {"res_weeks": res_weeks, "weeks_names": weeks_names})
        print(res_weeks.summary())
    else:
        c6 = _load_cache(6)
        if c6 is None:
            raise FileNotFoundError("Cache step6 不存在")
        res_weeks, weeks_names = c6["res_weeks"], c6["weeks_names"]
        print("Model 5: 从 cache 加载")

    idx_j = getattr(res_judge.params, "index", None)
    idx_f = getattr(res_fan.params, "index", None)
    if idx_j is not None and idx_f is not None:
        fe_common = [n for n in fe_names_judge if n in idx_j and n in idx_f]
    else:
        fe_common = list(fe_names_judge)
    cmp = compare_coefficients(res_judge, res_fan, fe_common, fe_names_judge, fe_names_fan)
    cmp.to_csv(os.path.join(OUT_DIR, "coefficient_comparison_judge_vs_fan.csv"), index=False, encoding="utf-8-sig")
    print("Coefficient comparison (judge vs fan) saved.")

    try:
        pro_re = pro_random_effects(res_judge, res_fan)
        pro_re.to_csv(os.path.join(OUT_DIR, "pro_dancer_random_effects.csv"), index=False, encoding="utf-8-sig")
        print("Pro dancer random effects (u_pro judge vs fan) saved.")
    except Exception as e:
        print("Pro random effects extraction skipped:", e)

    with open(os.path.join(OUT_DIR, "model_summary_judge.txt"), "w", encoding="utf-8") as f:
        f.write(res_judge.summary().as_text())
    with open(os.path.join(OUT_DIR, "model_summary_fan.txt"), "w", encoding="utf-8") as f:
        f.write(res_fan.summary().as_text())
    with open(os.path.join(OUT_DIR, "model_summary_elimination.txt"), "w", encoding="utf-8") as f:
        f.write(res_elim.summary().as_text())
    with open(os.path.join(OUT_DIR, "model_summary_placement.txt"), "w", encoding="utf-8") as f:
        f.write(res_placement.summary().as_text())
    with open(os.path.join(OUT_DIR, "model_summary_weeks_survived.txt"), "w", encoding="utf-8") as f:
        f.write(res_weeks.summary().as_text())

    key_outcome_vars = ["intercept", "age"]
    outcome_cmp = pd.DataFrame({"variable": key_outcome_vars})
    outcome_cmp["coef_placement"] = outcome_cmp["variable"].map(res_placement.params.get)
    outcome_cmp["coef_weeks_survived"] = outcome_cmp["variable"].map(res_weeks.params.get)
    outcome_cmp.to_csv(os.path.join(OUT_DIR, "outcome_placement_weeks_key_coefs.csv"), index=False, encoding="utf-8-sig")
    weeks_coefs = pd.DataFrame({"variable": res_weeks.params.index.astype(str), "coef": res_weeks.params.values})
    weeks_coefs.to_csv(os.path.join(OUT_DIR, "outcome_weeks_survived_coefs.csv"), index=False, encoding="utf-8-sig")

    key_vars = ["age", "week_norm", "has_bonus", "lag_judge", "lag_fan"]
    key_cmp = cmp[cmp["variable"].isin(key_vars)]
    key_cmp.to_csv(os.path.join(OUT_DIR, "key_coefficient_comparison.csv"), index=False, encoding="utf-8-sig")
    print("Done. Outputs in", OUT_DIR)
    return df, cs_df, res_judge, res_fan, res_elim, res_placement, res_weeks


if __name__ == "__main__":
    main()
