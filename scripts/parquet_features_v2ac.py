"""
parquet v2ac: v2 + mACStatus 충전 패턴 피처

v2 기반으로 최소한의 신규 피처만 추가:
  ac_daily_ratio   : 일 전체 충전 비율
  ac_morning_ratio : 아침(06-09h) 충전 비율
  ac_afternoon     : 오후(12-18h) 충전 비율
  ac_evening       : 저녁(18-22h) 충전 비율
  ac_presleep_ratio: 취침 전(22-24h) 충전 비율
  ac_sleep_ratio   : 수면 중(00-06h) 충전 비율
  ac_transitions   : 일별 충전/해제 전환 횟수
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from parquet_features_v2 import build_all as _build_v2

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"


def build_macstatus() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_DIR / "ch2025_mACStatus.parquet")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour

    daily = (
        df.groupby(["subject_id", "date"])["m_charging"]
        .mean().rename("ac_daily_ratio").reset_index()
    )

    zones = {
        "ac_morning_ratio": (df["hour"] >= 6) & (df["hour"] < 9),
        "ac_afternoon":     (df["hour"] >= 12) & (df["hour"] < 18),
        "ac_evening":       (df["hour"] >= 18) & (df["hour"] < 22),
        "ac_presleep_ratio": df["hour"] >= 22,
        "ac_sleep_ratio":   df["hour"] < 6,
    }
    for col, mask in zones.items():
        sub = (
            df[mask].groupby(["subject_id", "date"])["m_charging"]
            .mean().rename(col).reset_index()
        )
        daily = daily.merge(sub, on=["subject_id", "date"], how="left")

    transitions = (
        df.sort_values(["subject_id", "date", "timestamp"])
        .groupby(["subject_id", "date"])["m_charging"]
        .apply(lambda x: int((x.diff().abs() > 0).sum()))
        .rename("ac_transitions")
        .reset_index()
    )
    daily = daily.merge(transitions, on=["subject_id", "date"], how="left")
    return daily


def build_all() -> pd.DataFrame:
    feat = _build_v2()
    print("  mACStatus 집계 중...")
    ac = build_macstatus()
    feat = feat.merge(ac, on=["subject_id", "date"], how="left")
    drop_cols = [c for c in feat.columns if c.startswith("level_")]
    feat = feat.drop(columns=drop_cols, errors="ignore")
    print(f"  v2ac 완료: {len(feat)}행, {len(feat.columns)}컬럼")
    return feat


if __name__ == "__main__":
    feat = build_all()
    new_cols = [c for c in feat.columns if c.startswith("ac_")]
    print("mACStatus 피처:", new_cols)
    print(feat[new_cols].describe())
