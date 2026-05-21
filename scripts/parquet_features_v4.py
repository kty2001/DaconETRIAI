"""
parquet 피처 v4: v3 기반 + 수면 확장 구간(00-09h) wHr + mACStatus 충전 패턴
신규 피처:
  wHr(00-09h): hr_extsleep_mean, hr_extsleep_std, hr_extsleep_rmssd, hr_extsleep_spike_ratio
  mACStatus:   ac_sleep_ratio, ac_presleep_ratio, ac_morning_ratio, ac_daily_ratio
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from parquet_features_v3 import build_all as _build_v3
from parquet_features_v2 import _load

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"


def build_whr_extsleep() -> pd.DataFrame:
    """수면 확장 구간(00-09h) 심박 피처: 수면 종료 직전까지 포함"""
    df = _load("ch2025_wHr.parquet")
    df["hr_mean_row"] = df["heart_rate"].apply(
        lambda x: float(np.mean(x)) if len(x) > 0 else np.nan
    )

    ext_sleep = df[df["hour"] < 9].copy()

    stats = (
        ext_sleep.groupby(["subject_id", "date"])["hr_mean_row"]
        .agg(hr_extsleep_mean="mean", hr_extsleep_std="std")
        .reset_index()
    )

    rmssd_rows = []
    for (sid, date), grp in ext_sleep.groupby(["subject_id", "date"]):
        vals = grp.sort_values("hour")["hr_mean_row"].dropna().values
        rmssd = float(np.mean(np.abs(np.diff(vals)))) if len(vals) >= 2 else np.nan
        rmssd_rows.append({"subject_id": sid, "date": date, "hr_extsleep_rmssd": rmssd})
    rmssd_df = pd.DataFrame(rmssd_rows)

    result = stats.merge(rmssd_df, on=["subject_id", "date"], how="left")

    # 스파이크 비율: HR > 피험자 수면구간 평균 + 1std 인 시간 비율 (수면 중 HR 교란)
    subj_agg = (
        ext_sleep.groupby("subject_id")["hr_mean_row"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "subj_ext_mean", "std": "subj_ext_std"})
    )
    ext_sleep = ext_sleep.merge(subj_agg, on="subject_id", how="left")
    ext_sleep["is_spike"] = (
        ext_sleep["hr_mean_row"] > (ext_sleep["subj_ext_mean"] + ext_sleep["subj_ext_std"])
    ).astype(float)
    spike_ratio = (
        ext_sleep.groupby(["subject_id", "date"])["is_spike"]
        .mean().rename("hr_extsleep_spike_ratio").reset_index()
    )

    return result.merge(spike_ratio, on=["subject_id", "date"], how="left")


def build_macstatus() -> pd.DataFrame:
    """스마트폰 충전 패턴 피처 (수면 스케줄 규칙성 프록시)"""
    df = pd.read_parquet(PARQUET_DIR / "ch2025_mACStatus.parquet")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour

    sleep_zone    = df[df["hour"] < 6]
    presleep_zone = df[df["hour"] >= 22]
    morning_zone  = df[(df["hour"] >= 6) & (df["hour"] < 9)]

    daily_ratio = (
        df.groupby(["subject_id", "date"])["m_charging"]
        .mean().rename("ac_daily_ratio").reset_index()
    )
    sleep_ratio = (
        sleep_zone.groupby(["subject_id", "date"])["m_charging"]
        .mean().rename("ac_sleep_ratio").reset_index()
    )
    presleep_ratio = (
        presleep_zone.groupby(["subject_id", "date"])["m_charging"]
        .mean().rename("ac_presleep_ratio").reset_index()
    )
    morning_ratio = (
        morning_zone.groupby(["subject_id", "date"])["m_charging"]
        .mean().rename("ac_morning_ratio").reset_index()
    )

    result = daily_ratio
    for extra in [sleep_ratio, presleep_ratio, morning_ratio]:
        result = result.merge(extra, on=["subject_id", "date"], how="left")
    return result


def build_all() -> pd.DataFrame:
    """v3 피처 전체 + v4 신규 피처"""
    feat = _build_v3()

    extras = [
        ("wHr 수면확장(00-09h)", build_whr_extsleep),
        ("mACStatus",           build_macstatus),
    ]
    for name, fn in extras:
        print(f"  {name} 집계 중...")
        extra_feat = fn()
        feat = feat.merge(extra_feat, on=["subject_id", "date"], how="left")

    drop_cols = [c for c in feat.columns if c.startswith("level_")]
    feat = feat.drop(columns=drop_cols, errors="ignore")

    print(f"  완료: {len(feat)}행, {len(feat.columns)}컬럼")
    return feat


if __name__ == "__main__":
    print("=== parquet 피처 v4 집계 ===")
    feat = build_all()
    v4_kw = ["hr_extsleep", "ac_sleep_ratio", "ac_presleep", "ac_morning", "ac_daily"]
    new_cols = [c for c in feat.columns if any(kw in c for kw in v4_kw)]
    print(f"\n신규 피처 ({len(new_cols)}개):")
    for c in new_cols:
        print(f"  {c}")
    print(f"\n전체 parquet 피처 수: {len(feat.columns) - 2}개")
