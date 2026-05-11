"""
lag1/lag2/lag7 + roll3/roll7/roll14 피처 생성
- lag1: 전날 레이블 (date_diff <= 2인 경우만, 아니면 NaN)
- lag2: 2일 전 레이블 (date_diff <= 2 연속 조건, 아니면 NaN)
- lag7: 7일 전 레이블 (target-7일 ± 2일 이내 가장 가까운 값, 주간 패턴)
- roll3: 최근 3개 이전 값의 평균 (30일 이내)
- roll7: 최근 7개 이전 값의 평균 (30일 이내)
- roll14: 최근 14개 이전 값의 평균 (60일 이내, 2주 패턴)

반환: (subject_id, date) 기준 DataFrame
date 컬럼 = lifelog_date와 join할 날짜
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
MAX_LAG_DAYS = 2       # 날짜 간격이 이 이상이면 lag를 NaN 처리
ROLL_WINDOW_DAYS = 30  # roll3/roll7 계산 시 최대 30일 이내 값만 사용
ROLL14_WINDOW_DAYS = 60  # roll14 계산 시 최대 60일 이내 값만 사용


def build_label_features(
    train: pd.DataFrame,
    query: pd.DataFrame,
) -> pd.DataFrame:
    """
    train: 학습 레이블 (subject_id, sleep_date, Q1~S4 포함)
    query: 피처를 붙일 대상 (subject_id, lifelog_date 포함)
           train 또는 test 모두 가능
    반환: query 행 순서와 동일한 DataFrame (subject_id, date + lag/roll 컬럼)
    """
    train = train.copy()
    train["sleep_date"] = pd.to_datetime(train["sleep_date"])

    result_rows = []

    for _, row in query.iterrows():
        sid = row["subject_id"]
        # lifelog_date는 sleep_date - 1일이므로
        # 해당 sleep_date = lifelog_date + 1일
        target_sleep_date = pd.to_datetime(row["lifelog_date"]) + pd.Timedelta(days=1)

        # 해당 subject의 과거 레이블만 사용 (target_sleep_date 이전)
        hist = train[
            (train["subject_id"] == sid) &
            (train["sleep_date"] < target_sleep_date)
        ].sort_values("sleep_date")

        feat = {"subject_id": sid, "date": str(row["lifelog_date"])[:10]}

        for t in TARGETS:
            # lag1: 가장 최근 값
            if len(hist) >= 1:
                last = hist.iloc[-1]
                diff1 = (target_sleep_date - last["sleep_date"]).days
                feat[f"lag1_{t}"] = last[t] if diff1 <= MAX_LAG_DAYS else np.nan
            else:
                feat[f"lag1_{t}"] = np.nan

            # lag2: 두 번째로 최근 값 (lag1이 유효한 경우에만)
            if len(hist) >= 2 and not np.isnan(feat[f"lag1_{t}"]):
                prev = hist.iloc[-2]
                diff2 = (hist.iloc[-1]["sleep_date"] - prev["sleep_date"]).days
                feat[f"lag2_{t}"] = prev[t] if diff2 <= MAX_LAG_DAYS else np.nan
            else:
                feat[f"lag2_{t}"] = np.nan

            # lag7: target-7일 ± 2일 이내 가장 가까운 값 (주간 패턴)
            target_7 = target_sleep_date - pd.Timedelta(days=7)
            hist_near7 = hist[
                abs((hist["sleep_date"] - target_7).dt.days) <= MAX_LAG_DAYS
            ]
            if len(hist_near7) >= 1:
                closest_idx = (hist_near7["sleep_date"] - target_7).abs().idxmin()
                feat[f"lag7_{t}"] = hist_near7.loc[closest_idx, t]
            else:
                feat[f"lag7_{t}"] = np.nan

            # roll3/roll7: 30일 이내 최근 3/7개 평균
            cutoff = target_sleep_date - pd.Timedelta(days=ROLL_WINDOW_DAYS)
            recent = hist[hist["sleep_date"] >= cutoff][t].dropna().values

            feat[f"roll3_{t}"] = float(np.mean(recent[-3:])) if len(recent) >= 1 else np.nan
            feat[f"roll7_{t}"] = float(np.mean(recent[-7:])) if len(recent) >= 1 else np.nan

            # roll14: 60일 이내 최근 14개 평균 (2주 패턴)
            cutoff14 = target_sleep_date - pd.Timedelta(days=ROLL14_WINDOW_DAYS)
            recent14 = hist[hist["sleep_date"] >= cutoff14][t].dropna().values
            feat[f"roll14_{t}"] = float(np.mean(recent14[-14:])) if len(recent14) >= 1 else np.nan

        result_rows.append(feat)

    return pd.DataFrame(result_rows)


if __name__ == "__main__":
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== train 기준 label 피처 (앞 5행) ===")
    feat_train = build_label_features(train, train)
    print(feat_train.head())
    print(f"shape: {feat_train.shape}")

    print("\n=== test 기준 label 피처 (앞 5행) ===")
    feat_test = build_label_features(train, sample)
    print(feat_test.head())
    nan_rate = feat_test[[c for c in feat_test.columns if c.startswith("lag1")]].isna().mean()
    print("\nlag1 NaN 비율:")
    print(nan_rate.to_string())
