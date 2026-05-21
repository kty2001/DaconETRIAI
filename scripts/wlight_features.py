"""
wLight 피처 빌더 (ch2025_wLight.parquet)
w_light: 워치 광센서 (lux), 0% NaN, float64

피처 구성:
  시간대별  : wlight_{zone}_mean/std (morning~sleep)
  수면 특화 : wlight_sleep_dark_ratio (< 10 lux 비율),
              wlight_sleep_bright_ratio (>= 100 lux 비율),
              wlight_presleep_to_sleep_drop (취침 전 → 수면 밝기 감소)
  일별 요약 : wlight_daily_mean, wlight_daily_max, wlight_daytime_mean
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"

DARK_THRESH   = 10.0    # lux: dark (good for sleep)
BRIGHT_THRESH = 100.0   # lux: bright (melatonin disruptor)

ZONES = ["morning", "afternoon", "evening", "presleep", "sleep"]


def _zone(hour: pd.Series) -> pd.Series:
    conditions = [
        (hour >= 6)  & (hour < 12),
        (hour >= 12) & (hour < 18),
        (hour >= 18) & (hour < 22),
        (hour >= 22),
        (hour < 6),
    ]
    return np.select(conditions, ZONES, default="sleep")


def build_wlight() -> pd.DataFrame:
    """
    반환: (subject_id, date) 기준 wLight 피처 DataFrame
          date = lifelog_date (parquet_features_v2와 동일 키)
    """
    df = pd.read_parquet(PARQUET_DIR / "ch2025_wLight.parquet")
    df["date"]  = df["timestamp"].dt.date.astype(str)
    df["hour"]  = df["timestamp"].dt.hour
    df["zone"]  = _zone(df["hour"])

    df["is_dark"]   = (df["w_light"] < DARK_THRESH).astype(float)
    df["is_bright"]  = (df["w_light"] >= BRIGHT_THRESH).astype(float)

    rows = []
    for (sid, date), grp in df.groupby(["subject_id", "date"]):
        feat = {"subject_id": sid, "date": date}

        # 시간대별 평균/표준편차
        for zone in ZONES:
            z = grp[grp["zone"] == zone]["w_light"]
            feat[f"wlight_{zone}_mean"] = z.mean() if len(z) > 0 else np.nan
            feat[f"wlight_{zone}_std"]  = z.std()  if len(z) > 1 else np.nan

        # 수면 중 어두움/밝음 비율
        sleep_z = grp[grp["zone"] == "sleep"]
        if len(sleep_z) > 0:
            feat["wlight_sleep_dark_ratio"]   = sleep_z["is_dark"].mean()
            feat["wlight_sleep_bright_ratio"] = sleep_z["is_bright"].mean()
        else:
            feat["wlight_sleep_dark_ratio"]   = np.nan
            feat["wlight_sleep_bright_ratio"] = np.nan

        # 취침 전 → 수면 밝기 감소 (pre-sleep blue light vs actual sleep darkness)
        presleep_mean = feat.get("wlight_presleep_mean", np.nan)
        sleep_mean    = feat.get("wlight_sleep_mean",    np.nan)
        if not (np.isnan(presleep_mean) or np.isnan(sleep_mean)):
            feat["wlight_presleep_to_sleep_drop"] = presleep_mean - sleep_mean
        else:
            feat["wlight_presleep_to_sleep_drop"] = np.nan

        # 일별 요약
        feat["wlight_daily_mean"]   = grp["w_light"].mean()
        feat["wlight_daily_max"]    = grp["w_light"].max()
        daytime = grp[grp["zone"].isin(["morning", "afternoon"])]["w_light"]
        feat["wlight_daytime_mean"] = daytime.mean() if len(daytime) > 0 else np.nan

        rows.append(feat)

    result = pd.DataFrame(rows)
    print(f"  wLight 피처 완료: {result.shape[1]-2}개 피처 x {len(result)}행")
    return result


if __name__ == "__main__":
    feat = build_wlight()
    print(feat.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    nan_rates = feat.drop(columns=["subject_id", "date"]).isna().mean()
    print("\nNaN 비율:")
    print(nan_rates.round(3).to_string())
