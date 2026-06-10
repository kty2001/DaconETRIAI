"""
parquet v6: v4 + mACStatus 보완 + wHr 수면 HRV (당일 밤 정렬)

v4 대비 신규:
  [mACStatus]
    ac_afternoon   : 오후(12-18h) 충전 비율
    ac_evening     : 저녁(18-22h) 충전 비율
    ac_transitions : 일별 충전/해제 전환 횟수 (폰 활동성)

  [wHr sleep HRV - 날짜 정렬 수정]
    hr_sleep_rmssd_curr : 당일 밤 수면 RMSSD (second-level HR 배열)
    hr_sleep_pnn50_curr : 당일 밤 수면 pNN50

    v2/v4의 hr_sleep_* 는 timestamp.date=sleep_date 기준이므로
    lifelog_date merge 시 전날 밤 수면 HR을 사용.
    이 모듈은 날짜를 -1일 시프트해 당일 밤 수면 HR을 lifelog_date에 정렬.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from parquet_features_v4 import build_all as _build_v4

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"


def _intra_rmssd(hr_arr):
    arr = np.asarray(hr_arr, dtype=float)
    arr = arr[(arr > 30) & (arr < 200)]
    if len(arr) < 3:
        return np.nan
    rr = 60000.0 / arr
    return float(np.sqrt(np.mean(np.diff(rr) ** 2)))


def _intra_pnn50(hr_arr):
    arr = np.asarray(hr_arr, dtype=float)
    arr = arr[(arr > 30) & (arr < 200)]
    if len(arr) < 3:
        return np.nan
    rr = 60000.0 / arr
    return float(np.mean(np.abs(np.diff(rr)) > 50))


def build_whr_sleep_hrv_curr() -> pd.DataFrame:
    """
    당일 밤 수면(00-06h) RMSSD / pNN50.
    00-06h timestamp.date() = sleep_date.
    -1일 시프트 -> lifelog_date 기준으로 merge 가능.
    """
    df = pd.read_parquet(PARQUET_DIR / "ch2025_wHr.parquet")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour

    sleep_df = df[df["hour"] < 6].copy()
    sleep_df["intra_rmssd"] = sleep_df["heart_rate"].apply(_intra_rmssd)
    sleep_df["intra_pnn50"] = sleep_df["heart_rate"].apply(_intra_pnn50)

    daily = sleep_df.groupby(["subject_id", "date"]).agg(
        hr_sleep_rmssd_curr=("intra_rmssd", "mean"),
        hr_sleep_pnn50_curr=("intra_pnn50", "mean"),
    ).reset_index()

    # sleep_date -> lifelog_date (sleep_date - 1)
    daily["date"] = (
        pd.to_datetime(daily["date"]) - pd.Timedelta(days=1)
    ).dt.strftime("%Y-%m-%d")

    return daily


def build_macstatus_extra() -> pd.DataFrame:
    """mACStatus 보완: 오후/저녁 충전 비율 + 충전 전환 횟수"""
    df = pd.read_parquet(PARQUET_DIR / "ch2025_mACStatus.parquet")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour

    afternoon = (
        df[(df["hour"] >= 12) & (df["hour"] < 18)]
        .groupby(["subject_id", "date"])["m_charging"]
        .mean()
        .rename("ac_afternoon")
    )
    evening = (
        df[(df["hour"] >= 18) & (df["hour"] < 22)]
        .groupby(["subject_id", "date"])["m_charging"]
        .mean()
        .rename("ac_evening")
    )
    transitions = (
        df.sort_values(["subject_id", "date", "timestamp"])
        .groupby(["subject_id", "date"])["m_charging"]
        .apply(lambda x: int((x.diff().abs() > 0).sum()))
        .rename("ac_transitions")
    )

    result = afternoon.reset_index()
    for s in [evening.reset_index(), transitions.reset_index()]:
        result = result.merge(s, on=["subject_id", "date"], how="outer")
    return result


def build_all() -> pd.DataFrame:
    """v4 기반 + v6 신규 피처"""
    feat = _build_v4()

    print("  mACStatus 보완(afternoon/evening/transitions)...")
    ac_extra = build_macstatus_extra()
    feat = feat.merge(ac_extra, on=["subject_id", "date"], how="left")

    print("  wHr 수면 HRV (당일 밤 date-shifted)...")
    hrv_curr = build_whr_sleep_hrv_curr()
    feat = feat.merge(hrv_curr, on=["subject_id", "date"], how="left")

    drop_cols = [c for c in feat.columns if c.startswith("level_")]
    feat = feat.drop(columns=drop_cols, errors="ignore")

    print(f"  v6 완료: {len(feat)}행, {len(feat.columns)}컬럼")
    return feat


if __name__ == "__main__":
    print("=== parquet v6 집계 ===")
    feat = build_all()
    new_kw = ["ac_afternoon", "ac_evening", "ac_transitions",
              "hr_sleep_rmssd_curr", "hr_sleep_pnn50_curr"]
    new_cols = [c for c in feat.columns if any(k in c for k in new_kw)]
    print(f"\n신규 피처 ({len(new_cols)}개):", new_cols)
    if new_cols:
        print(feat[new_cols].describe())
