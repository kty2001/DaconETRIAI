"""
parquet 파일을 subject_id + date 기준으로 집계하여 피처 테이블 생성
대상: wPedo, mActivity, mScreenStatus, wHr, mLight
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"


def _load(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_DIR / filename)
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour
    return df


def build_wpedo() -> pd.DataFrame:
    df = _load("ch2025_wPedo.parquet")
    agg = df.groupby(["subject_id", "date"]).agg(
        pedo_step_sum=("step", "sum"),
        pedo_run_sum=("running_step", "sum"),
        pedo_walk_sum=("walking_step", "sum"),
        pedo_calories_sum=("burned_calories", "sum"),
        pedo_distance_sum=("distance", "sum"),
        pedo_speed_mean=("speed", "mean"),
    ).reset_index()
    return agg


def build_mactivity() -> pd.DataFrame:
    df = _load("ch2025_mActivity.parquet")
    total = df.groupby(["subject_id", "date"]).size().rename("act_total")
    # 3=정지, 4=걷기, 7=달리기
    active = (
        df[df["m_activity"].isin([4, 7])]
        .groupby(["subject_id", "date"]).size().rename("act_active_cnt")
    )
    still = (
        df[df["m_activity"] == 3]
        .groupby(["subject_id", "date"]).size().rename("act_still_cnt")
    )
    agg = pd.concat([total, active, still], axis=1).fillna(0).reset_index()
    agg["act_active_ratio"] = agg["act_active_cnt"] / agg["act_total"].clip(lower=1)
    agg["act_still_ratio"] = agg["act_still_cnt"] / agg["act_total"].clip(lower=1)
    return agg[["subject_id", "date", "act_active_ratio", "act_still_ratio", "act_active_cnt"]]


def build_mscreen() -> pd.DataFrame:
    df = _load("ch2025_mScreenStatus.parquet")
    daily = df.groupby(["subject_id", "date"]).agg(
        screen_on_ratio=("m_screen_use", "mean"),
        screen_on_count=("m_screen_use", "sum"),
    ).reset_index()
    # 취침 전 2시간 (22~24시) 화면 사용률
    late = (
        df[df["hour"] >= 22]
        .groupby(["subject_id", "date"])
        .agg(screen_late_ratio=("m_screen_use", "mean"))
        .reset_index()
    )
    return daily.merge(late, on=["subject_id", "date"], how="left")


def build_whr() -> pd.DataFrame:
    df = _load("ch2025_wHr.parquet")
    # 분당 배열의 평균값을 분 단위 심박수로 사용
    df["hr_min"] = df["heart_rate"].apply(
        lambda x: float(np.mean(x)) if len(x) > 0 else np.nan
    )
    daily = df.groupby(["subject_id", "date"]).agg(
        hr_mean=("hr_min", "mean"),
        hr_std=("hr_min", "std"),
        hr_min_val=("hr_min", "min"),
        hr_max_val=("hr_min", "max"),
    ).reset_index()
    # 새벽 (0~6시): 수면 중 심박 패턴
    night = (
        df[df["hour"] < 6]
        .groupby(["subject_id", "date"])
        .agg(
            hr_night_mean=("hr_min", "mean"),
            hr_night_std=("hr_min", "std"),
        )
        .reset_index()
    )
    return daily.merge(night, on=["subject_id", "date"], how="left")


def build_mlight() -> pd.DataFrame:
    df = _load("ch2025_mLight.parquet")
    daily = df.groupby(["subject_id", "date"]).agg(
        light_mean=("m_light", "mean"),
        light_max=("m_light", "max"),
    ).reset_index()
    # 야간 (22시 이후) 조도: 취침 환경
    night = (
        df[df["hour"] >= 22]
        .groupby(["subject_id", "date"])
        .agg(light_night_mean=("m_light", "mean"))
        .reset_index()
    )
    return daily.merge(night, on=["subject_id", "date"], how="left")


def build_all() -> pd.DataFrame:
    builders = [
        ("wPedo", build_wpedo),
        ("mActivity", build_mactivity),
        ("mScreenStatus", build_mscreen),
        ("wHr", build_whr),
        ("mLight", build_mlight),
    ]
    feat = None
    for name, fn in builders:
        print(f"  {name} 집계 중...")
        df = fn()
        feat = df if feat is None else feat.merge(df, on=["subject_id", "date"], how="outer")

    print(f"  완료: {len(feat)}행, {len(feat.columns)}컬럼")
    return feat


if __name__ == "__main__":
    print("=== parquet 피처 집계 ===")
    feat = build_all()
    print(feat.head(3).to_string())
    print("\n컬럼 목록:")
    print([c for c in feat.columns if c not in ("subject_id", "date")])
