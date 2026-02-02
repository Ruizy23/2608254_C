import os
import re
import pandas as pd
import numpy as np
from typing import Optional, Tuple

BASE_PATH = "/8202302003"
RAW_CSV = os.path.join(BASE_PATH, "2026_MCM_Problem_C_Data.csv")
OUT_CLEANED_CSV = os.path.join(BASE_PATH, "dwts_cleaned.csv")
OUT_LONG_CSV = os.path.join(BASE_PATH, "dwts_long.csv")
OUT_META_CSV = os.path.join(BASE_PATH, "dwts_season_meta.csv")


def parse_results(results: str) -> dict:

    if pd.isna(results):
        return {"result_type": "unknown", "eliminated_week": None, "is_finalist": False}

    s = str(results).strip()
    out = {"result_type": "finalist", "eliminated_week": None, "is_finalist": False}

    m = re.match(r"Eliminated Week (\d+)", s, re.I)
    if m:
        out["result_type"] = "eliminated"
        out["eliminated_week"] = int(m.group(1))
        return out

    if "Withdrew" in s:
        out["result_type"] = "withdrew"
        return out

    if re.search(r"(\d)(?:st|nd|rd|th) Place", s):
        out["is_finalist"] = True
    return out


def get_judge_score_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if re.match(r"week\d+_judge\d+_score", c)]
    def key(c):
        m = re.match(r"week(\d+)_judge(\d+)_score", c)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return sorted(cols, key=key)


def week_from_col(col: str) -> int:
    m = re.match(r"week(\d+)_", col)
    return int(m.group(1)) if m else 0


def total_judge_score_and_n_judges(row: pd.Series, week_cols: list) -> dict:
    by_week = {}
    for col in week_cols:
        w = week_from_col(col)
        if w not in by_week:
            by_week[w] = []
        val = row[col]
        if pd.notna(val):
            try:
                by_week[w].append(float(val))
            except (TypeError, ValueError):
                pass
    result = {}
    for w, scores in by_week.items():
        if scores:
            result[w] = (sum(scores), len(scores))
        else:
            result[w] = (np.nan, 0)
    return result


def infer_last_week_competed(row: pd.Series, week_cols: list, parsed: dict) -> int:
    eliminated_week = parsed.get("eliminated_week")
    if eliminated_week is not None:
        return eliminated_week

    last = 0
    weeks = sorted(set(week_from_col(c) for c in week_cols))
    for w in weeks:
        cols_w = [c for c in week_cols if week_from_col(c) == w]
        vals = [row[c] for c in cols_w]
        numeric = []
        for v in vals:
            if pd.isna(v):
                continue
            try:
                numeric.append(float(v))
            except (TypeError, ValueError):
                pass
        if numeric and any(x != 0 for x in numeric):
            last = w
    return last if last > 0 else max(weeks)


def load_raw(base_path: str = BASE_PATH) -> pd.DataFrame:
    raw_path = os.path.join(base_path, "2026_MCM_Problem_C_Data.csv")
    df = pd.read_csv(raw_path, na_values=["N/A", ""], keep_default_na=True)
    return df


def add_voting_method(season: int) -> str:
    if season in (1, 2):
        return "rank"
    if 3 <= season <= 27:
        return "percent"
    if 28 <= season <= 34:
        return "rank"
    return "unknown"


def clean_data(base_path: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_path = base_path or BASE_PATH
    df = load_raw(base_path)

    parsed_list = [parse_results(r) for r in df["results"]]
    df["result_type"] = [p["result_type"] for p in parsed_list]
    df["eliminated_week"] = [p["eliminated_week"] for p in parsed_list]
    df["is_finalist"] = [p["is_finalist"] for p in parsed_list]

    judge_cols = get_judge_score_columns(df)
    weeks_sorted = sorted(set(week_from_col(c) for c in judge_cols))

    last_weeks = []
    for i, row in df.iterrows():
        last_weeks.append(infer_last_week_competed(row, judge_cols, parsed_list[i]))

    df["last_week_competed"] = last_weeks

    for w in weeks_sorted:
        cols_w = [c for c in judge_cols if week_from_col(c) == w]
        def sum_judges(r):
            vals = [r[c] for c in cols_w]
            nums = []
            for v in vals:
                if pd.notna(v):
                    try:
                        nums.append(float(v))
                    except (TypeError, ValueError):
                        pass
            if not nums:
                return np.nan, 0
            return sum(nums), len(nums)

        totals = df.apply(lambda r: sum_judges(r), axis=1)
        df[f"week{w}_total_judge_score"] = [t[0] for t in totals]
        df[f"week{w}_n_judges"] = [t[1] for t in totals]
        df[f"week{w}_competed"] = (df["last_week_competed"] >= w).astype(int)

    last_active_list = []
    for _, row in df.iterrows():
        last_active = 0
        for w in weeks_sorted:
            total = row.get(f"week{w}_total_judge_score")
            if pd.notna(total) and float(total) > 0:
                last_active = w
        last_active_list.append(last_active)
    df["last_active_week"] = last_active_list
    withdrew_mask = df["result_type"] == "withdrew"
    df.loc[withdrew_mask, "last_week_competed"] = df.loc[withdrew_mask, "last_active_week"]
    for w in weeks_sorted:
        df[f"week{w}_competed"] = (df["last_week_competed"] >= w).astype(int)
    df["season_total_weeks"] = df.groupby("season")["last_week_competed"].transform("max")

    df["voting_method"] = df["season"].map(add_voting_method)
    df["is_all_star_season"] = (df["season"] == 15).astype(int)

    if "celebrity_homecountry/region" in df.columns:
        df = df.rename(columns={"celebrity_homecountry/region": "celebrity_homecountry_region"})

    season_meta = (
        df.groupby("season")
        .agg(
            n_contestants=("celebrity_name", "count"),
            max_week_competed=("last_week_competed", "max"),
        )
        .reset_index()
    )
    season_meta["voting_method"] = season_meta["season"].map(add_voting_method)
    season_meta["is_all_star"] = (season_meta["season"] == 15).astype(int)

    id_cols = [
        "celebrity_name", "ballroom_partner", "celebrity_industry",
        "celebrity_homestate", "celebrity_homecountry_region",
        "celebrity_age_during_season", "season", "results", "placement",
        "result_type", "eliminated_week", "is_finalist", "last_week_competed",
        "last_active_week", "season_total_weeks", "voting_method", "is_all_star_season",
    ]
    id_cols = [c for c in id_cols if c in df.columns]
    long_rows = []
    for _, row in df.iterrows():
        last = int(row["last_week_competed"])
        for w in weeks_sorted:
            if w > last:
                break
            competed = int(row.get(f"week{w}_competed", 0))
            if competed == 0:
                continue
            total_score = row.get(f"week{w}_total_judge_score")
            n_judges = row.get(f"week{w}_n_judges", 0)
            post_elim = 0
            if row["result_type"] == "eliminated" and row["eliminated_week"] is not None:
                if w < row["eliminated_week"]:
                    post_elim = 0
                elif w > row["eliminated_week"]:
                    post_elim = 1
                else:
                    post_elim = 0
            n_j = int(n_judges) if pd.notna(n_judges) else 0
            max_possible = n_j * 10.0 if n_j > 0 else np.nan
            has_bonus = (
                pd.notna(total_score) and pd.notna(max_possible) and float(total_score) > max_possible
            )
            if pd.notna(total_score) and pd.notna(max_possible) and max_possible > 0:
                pct_raw = float(total_score) / max_possible
                pct_capped = min(1.0, pct_raw)
            else:
                pct_raw = np.nan
                pct_capped = np.nan
            long_rows.append({
                **{k: row[k] for k in id_cols},
                "week": w,
                "total_judge_score": total_score,
                "n_judges": n_judges,
                "max_possible_judge_score": max_possible,
                "percentage_score": pct_raw,
                "percentage_score_capped": pct_capped,
                "has_bonus_this_week": has_bonus,
                "post_elimination_placeholder": post_elim,
            })
    long_df = pd.DataFrame(long_rows)
    return df, long_df, season_meta


def main():
    print("Data cleaning for DWTS 2026 MCM Problem C")
    print("BASE_PATH:", BASE_PATH)

    if not os.path.exists(os.path.join(BASE_PATH, "2026_MCM_Problem_C_Data.csv")):
        alt_path = os.path.dirname(os.path.abspath(__file__))
        print("Raw CSV not under BASE_PATH, trying:", alt_path)
        cleaned_wide, long_df, season_meta = clean_data(base_path=alt_path)
        out_clean = os.path.join(alt_path, "dwts_cleaned.csv")
        out_long = os.path.join(alt_path, "dwts_long.csv")
        out_meta = os.path.join(alt_path, "dwts_season_meta.csv")
    else:
        cleaned_wide, long_df, season_meta = clean_data()
        out_clean = OUT_CLEANED_CSV
        out_long = OUT_LONG_CSV
        out_meta = OUT_META_CSV

    for p in [out_clean, out_long, out_meta]:
        d = os.path.dirname(p)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)

    cleaned_wide.to_csv(out_clean, index=False, encoding="utf-8-sig")
    long_df.to_csv(out_long, index=False, encoding="utf-8-sig")
    season_meta.to_csv(out_meta, index=False, encoding="utf-8-sig")

    print("Cleaned wide shape:", cleaned_wide.shape)
    print("Long format shape:", long_df.shape)
    print("Season meta shape:", season_meta.shape)
    print("Saved:")
    print("  ", out_clean)
    print("  ", out_long)
    print("  ", out_meta)
    return cleaned_wide, long_df, season_meta


if __name__ == "__main__":
    main()
