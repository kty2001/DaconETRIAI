"""
야간 심박수 변동성 피처 빌더

wHr.parquet에서 수면 중(00-06h) 심박수 분포 및 고주파 변동성 지표 계산.
parquet_features_v2의 hr_mean_sleep, hr_std_sleep, hr_sleep_rmssd를 보완.

신규 피처 (7개):
  hr_sleep_iqr        : 수면 중 분별 HR IQR (Q75-Q25, robust variability)
  hr_sleep_p10        : 수면 중 분별 HR 10th percentile (심박 저점)
  hr_sleep_p90        : 수면 중 분별 HR 90th percentile (각성/불규칙 상한)
  hr_sleep_within_std : 분 내 초 단위 HR std 평균 (고주파 HRV 근사)
  hr_sleep_count      : 수면 중 기록 분 수 (워치 착용/수면 시간 밀도)
  hr_night_dip        : morning_mean - sleep_mean (야간 심박 강하량)
  hr_presleep_delta   : presleep_mean - sleep_mean (취침→수면 전환 강하량)

date = lifelog_date 기준 (parquet_features_v2와 동일 조인 키)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "ch2025_data_items"


def build_whr_variability() -> pd.DataFrame:
    df = pd.read_parquet(DATA / "ch2025_wHr.parquet")
    df["hour"] = df["timestamp"].dt.hour
    df["date"] = pd.to_datetime(df["timestamp"].dt.date)

    df["hr_min_mean"] = df["heart_rate"].apply(
        lambda x: float(np.mean(x)) if len(x) > 0 else np.nan
    )
    df["hr_min_within_std"] = df["heart_rate"].apply(
        lambda x: float(np.std(x)) if len(x) > 1 else np.nan
    )

    sleep_df   = df[df["hour"] < 6].copy()
    morning_df = df[(df["hour"] >= 6) & (df["hour"] < 12)].copy()
    pre_df     = df[df["hour"] >= 22].copy()

    rows = []
    for (sid, date), grp in sleep_df.groupby(["subject_id", "date"]):
        vals = grp["hr_min_mean"].dropna().values
        n    = len(vals)

        if n >= 2:
            iqr  = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            p10  = float(np.percentile(vals, 10))
            p90  = float(np.percentile(vals, 90))
        elif n == 1:
            iqr  = 0.0
            p10  = float(vals[0])
            p90  = float(vals[0])
        else:
            iqr  = np.nan
            p10  = np.nan
            p90  = np.nan

        within_std = grp["hr_min_within_std"].dropna().mean() if n > 0 else np.nan
        sleep_mean = float(np.nanmean(vals)) if n > 0 else np.nan

        rows.append({
            "subject_id":          sid,
            "date":                date,
            "hr_sleep_iqr":        iqr,
            "hr_sleep_p10":        p10,
            "hr_sleep_p90":        p90,
            "hr_sleep_within_std": within_std,
            "hr_sleep_count":      n,
            "_sleep_mean":         sleep_mean,
        })

    result = pd.DataFrame(rows)

    morning_mean = (
        morning_df.groupby(["subject_id", "date"])["hr_min_mean"]
        .mean()
        .reset_index()
        .rename(columns={"hr_min_mean": "_morning_mean"})
    )
    pre_mean = (
        pre_df.groupby(["subject_id", "date"])["hr_min_mean"]
        .mean()
        .reset_index()
        .rename(columns={"hr_min_mean": "_presleep_mean"})
    )

    result = result.merge(morning_mean, on=["subject_id", "date"], how="left")
    result = result.merge(pre_mean,    on=["subject_id", "date"], how="left")

    result["hr_night_dip"]      = result["_morning_mean"]  - result["_sleep_mean"]
    result["hr_presleep_delta"] = result["_presleep_mean"] - result["_sleep_mean"]

    drop_cols = [c for c in result.columns if c.startswith("_")]
    result = result.drop(columns=drop_cols)

    print(f"  wHr variability: {len(result)} rows, "
          f"{len([c for c in result.columns if c not in ('subject_id','date')])} features")
    return result


if __name__ == "__main__":
    feat = build_whr_variability()
    print(feat.describe())
    print(feat.isnull().mean().to_string())
