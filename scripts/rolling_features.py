"""
피험자별 일별 피처 이동평균 / delta 피처 빌더

parquet_features_v2.build_all() 결과를 입력으로 받아,
주요 피처에 대해 피험자 내 시계열 이동평균과 변화량을 계산한다.

신규 피처 (기준 피처 1개당 3개):
  {col}_ma3    : 전날 기준 3일 이동평균 (shift=1, min_periods=1)
  {col}_ma7    : 전날 기준 7일 이동평균
  {col}_delta3 : 당일값 - ma3  (기저선 대비 변화량)

기준 피처 (존재하는 것만 사용):
  pedo_step_sum, pedo_calories_sum, pedo_speed_mean
  hr_mean, hr_std, hr_mean_sleep, hr_std_sleep, hr_sleep_rmssd
  act_active_ratio, act_still_ratio, act_ratio_presleep, act_ratio_sleep
  screen_on_ratio, screen_ratio_presleep, screen_ratio_evening
  light_mean, light_mean_presleep, light_mean_sleep, light_mean_evening
"""

import numpy as np
import pandas as pd

BASE_COLS = [
    "pedo_step_sum", "pedo_calories_sum", "pedo_speed_mean",
    "hr_mean", "hr_std", "hr_mean_sleep", "hr_std_sleep", "hr_sleep_rmssd",
    "act_active_ratio", "act_still_ratio", "act_ratio_presleep", "act_ratio_sleep",
    "screen_on_ratio", "screen_ratio_presleep", "screen_ratio_evening",
    "light_mean", "light_mean_presleep", "light_mean_sleep", "light_mean_evening",
]


def build_rolling_features(parquet_feat: pd.DataFrame) -> pd.DataFrame:
    """
    parquet_feat: build_all() 반환값 (subject_id, date, ...)
    date 컬럼은 str 또는 datetime 모두 허용.
    반환: (subject_id, date, rolling 피처들)
    """
    avail = [c for c in BASE_COLS if c in parquet_feat.columns]

    df = parquet_feat[["subject_id", "date"] + avail].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["subject_id", "date"]).reset_index(drop=True)

    groups = []
    for sid, grp in df.groupby("subject_id", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        for col in avail:
            shifted = grp[col].shift(1)
            grp[f"{col}_ma3"]    = shifted.rolling(3, min_periods=1).mean()
            grp[f"{col}_ma7"]    = shifted.rolling(7, min_periods=1).mean()
            grp[f"{col}_delta3"] = grp[col] - grp[f"{col}_ma3"]
        groups.append(grp)

    result = pd.concat(groups, ignore_index=True)
    result["date"] = result["date"].dt.date.astype(str)

    feat_cols = [
        c for c in result.columns
        if c.endswith("_ma3") or c.endswith("_ma7") or c.endswith("_delta3")
    ]
    n_base = len(avail)
    n_feat = len(feat_cols)
    print(f"  rolling: {n_base}개 기준 피처 -> {n_feat}개 rolling 피처")
    return result[["subject_id", "date"] + feat_cols]


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from parquet_features_v2 import build_all

    print("=== parquet v2 집계 중 ===")
    pf = build_all()
    print("\n=== rolling 피처 계산 ===")
    rf = build_rolling_features(pf)
    print(rf.describe().to_string())
    print("\n결측률:")
    print(rf.isnull().mean().sort_values(ascending=False).head(20).to_string())
