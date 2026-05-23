"""
요일 효과 편차 피처 (Day-of-Week Deviation Features)
- 개인별로 요일(0=월~6=일)별 평균을 계산
- 편차 = 당일 값 - 개인 요일 평균 (평소 요일 대비 오늘 얼마나 다른가)
- 타깃 센서: 걸음/심박/조도/스크린/활동량 관련 핵심 피처

사용법:
    from dow_deviation_features import add_dow_deviations
    feat_with_dow = add_dow_deviations(feat_df, ref_df)
    # feat_df: 피처 df (subject_id, date 포함)
    # ref_df : 전체 참조 df (subject_id, date 포함, 동일 또는 전체 데이터)
"""

import numpy as np
import pandas as pd

# 요일 편차를 계산할 핵심 피처 목록
DOW_BASE_FEATURES = [
    # 보행/활동
    "pedo_step_sum",
    "pedo_calories_sum",
    "act_active_ratio",
    "act_active_cnt",
    # 심박
    "hr_mean",
    "hr_std",
    "hr_sleep_rmssd",
    # 조도 (손목 / 폰)
    "wlight_daily_mean",
    "wlight_sleep_dark_ratio",
    "wlight_presleep_to_sleep_drop",
    "light_mean",
    # 스크린
    "screen_on_ratio",
    "screen_on_count",
    # GPS
    "gps_speed_mean",
    "gps_moving_ratio",
    "gps_place_entropy",
    "gps_n_places",
    "gps_home_ratio",
    # WiFi
    "wifi_entropy",
    "wifi_n_unique_daily",
    "wifi_home_ratio",
    # BLE
    "ble_n_unique_daily",
    "ble_devices_per_scan_mean",
]


def add_dow_deviations(feat_df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    """
    feat_df: 편차를 추가할 대상 DataFrame (subject_id, date 열 필수)
    ref_df : 요일 평균 계산 참조용 DataFrame (학습 fold 데이터 권장)
             같은 피처 컬럼을 가져야 함

    반환: feat_df에 {feature}_dow_dev, {feature}_dow_mean 컬럼 추가된 DataFrame
    """
    feat_df = feat_df.copy()
    ref_df  = ref_df.copy()

    # date -> 요일 (0=월, 6=일)
    feat_df["_dow"] = pd.to_datetime(feat_df["date"]).dt.dayofweek
    ref_df["_dow"]  = pd.to_datetime(ref_df["date"]).dt.dayofweek

    base_feats = [f for f in DOW_BASE_FEATURES if f in feat_df.columns and f in ref_df.columns]

    # 참조 데이터에서 subject×weekday 평균 계산
    dow_means = (
        ref_df.groupby(["subject_id", "_dow"])[base_feats]
        .mean()
        .reset_index()
    )

    # feat_df에 join
    feat_with_dow = feat_df.merge(
        dow_means.rename(columns={f: f"__dm_{f}" for f in base_feats}),
        on=["subject_id", "_dow"],
        how="left",
    )

    for f in base_feats:
        dm_col = f"__dm_{f}"
        if dm_col in feat_with_dow.columns:
            feat_with_dow[f"{f}_dow_dev"]  = feat_with_dow[f] - feat_with_dow[dm_col]
            feat_with_dow[f"{f}_dow_mean"] = feat_with_dow[dm_col]
            feat_with_dow = feat_with_dow.drop(columns=[dm_col])

    feat_with_dow = feat_with_dow.drop(columns=["_dow"])
    return feat_with_dow
