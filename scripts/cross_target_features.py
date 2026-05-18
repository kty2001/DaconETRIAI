"""
교차 타깃 피처 생성 및 저장
- n_pos_lag1: 전날 양성 타깃 수 (0~7)
- n_pos_roll7: 최근 7일 평균 양성 타깃 수
- xcorr_{t1}_{t2}_14: 타깃 쌍 rolling Pearson correlation (14개 이내, 60일 이내)
- xcorr_{t1}_{t2}_28: 타깃 쌍 rolling Pearson correlation (28개 이내, 90일 이내)
- lag1_diff_{t1}_{t2}: 전날 타깃 쌍 차이 (t1 - t2 ∈ {-1, 0, 1})
- momentum_{t1}_{t2}: 전날 vs 이틀 전 변화 방향 일치 ((Δt1)*(Δt2))
저장: data/cross_target_features.csv
"""

import numpy as np
import pandas as pd
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

TARGETS      = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
TARGET_PAIRS = list(combinations(TARGETS, 2))  # 21 쌍
MAX_LAG_DAYS = 2


def build_cross_target_features(train: pd.DataFrame, query: pd.DataFrame) -> pd.DataFrame:
    """
    train: 학습 레이블 (subject_id, sleep_date, Q1~S4)
    query: 피처 계산 대상 (subject_id, lifelog_date)
    반환: (subject_id, date) 기준 교차 타깃 피처 DataFrame
    """
    train = train.copy()
    train["sleep_date"] = pd.to_datetime(train["sleep_date"])

    result_rows = []

    for _, row in query.iterrows():
        sid = row["subject_id"]
        target_sleep_date = pd.to_datetime(row["lifelog_date"]) + pd.Timedelta(days=1)

        hist = train[
            (train["subject_id"] == sid) &
            (train["sleep_date"] < target_sleep_date)
        ].sort_values("sleep_date")

        feat = {"subject_id": sid, "date": str(row["lifelog_date"])[:10]}

        # ── lag1 / lag2 유효값 추출 ──────────────────────────────────────────
        lag1 = {}
        lag2 = {}

        if len(hist) >= 1:
            last  = hist.iloc[-1]
            diff1 = (target_sleep_date - last["sleep_date"]).days
            v1    = diff1 <= MAX_LAG_DAYS
            for t in TARGETS:
                lag1[t] = float(last[t]) if v1 else np.nan
        else:
            for t in TARGETS:
                lag1[t] = np.nan

        lag1_any_valid = not all(np.isnan(v) for v in lag1.values())
        if len(hist) >= 2 and lag1_any_valid:
            prev  = hist.iloc[-2]
            diff2 = (hist.iloc[-1]["sleep_date"] - prev["sleep_date"]).days
            v2    = diff2 <= MAX_LAG_DAYS
            for t in TARGETS:
                lag2[t] = float(prev[t]) if v2 else np.nan
        else:
            for t in TARGETS:
                lag2[t] = np.nan

        # ── 전날 양성 타깃 수 ────────────────────────────────────────────────
        valid1 = [v for v in lag1.values() if not np.isnan(v)]
        feat["n_pos_lag1"] = float(sum(valid1)) if valid1 else np.nan

        # ── 최근 7일 평균 양성 타깃 수 (30일 이내) ──────────────────────────
        cutoff30 = target_sleep_date - pd.Timedelta(days=30)
        recent7  = hist[hist["sleep_date"] >= cutoff30].iloc[-7:]
        feat["n_pos_roll7"] = float(recent7[TARGETS].sum(axis=1).mean()) if len(recent7) >= 1 else np.nan

        # ── 타깃 쌍별 피처 ──────────────────────────────────────────────────
        cutoff60 = target_sleep_date - pd.Timedelta(days=60)
        cutoff90 = target_sleep_date - pd.Timedelta(days=90)
        data60   = hist[hist["sleep_date"] >= cutoff60]
        data90   = hist[hist["sleep_date"] >= cutoff90]

        for t1, t2 in TARGET_PAIRS:
            pn = f"{t1}_{t2}"

            # rolling correlation 14개 이내 (60일)
            d14 = data60[[t1, t2]].dropna().iloc[-14:]
            if len(d14) >= 4:
                c = d14[t1].corr(d14[t2])
                feat[f"xcorr_{pn}_14"] = float(c) if not np.isnan(c) else np.nan
            else:
                feat[f"xcorr_{pn}_14"] = np.nan

            # rolling correlation 28개 이내 (90일)
            d28 = data90[[t1, t2]].dropna().iloc[-28:]
            if len(d28) >= 4:
                c = d28[t1].corr(d28[t2])
                feat[f"xcorr_{pn}_28"] = float(c) if not np.isnan(c) else np.nan
            else:
                feat[f"xcorr_{pn}_28"] = np.nan

            # lag1 차이 (t1 - t2 ∈ {-1, 0, 1})
            v1, v2 = lag1[t1], lag1[t2]
            feat[f"lag1_diff_{pn}"] = (v1 - v2) if (not np.isnan(v1) and not np.isnan(v2)) else np.nan

            # momentum: 두 타깃 변화 방향 일치 ((Δt1)*(Δt2))
            l1_t1, l2_t1 = lag1[t1], lag2[t1]
            l1_t2, l2_t2 = lag1[t2], lag2[t2]
            if not any(np.isnan(v) for v in [l1_t1, l2_t1, l1_t2, l2_t2]):
                feat[f"momentum_{pn}"] = float((l1_t1 - l2_t1) * (l1_t2 - l2_t2))
            else:
                feat[f"momentum_{pn}"] = np.nan

        result_rows.append(feat)

    return pd.DataFrame(result_rows)


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    all_query = pd.concat([
        train[["subject_id",  "lifelog_date"]],
        sample[["subject_id", "lifelog_date"]],
    ], ignore_index=True).drop_duplicates(subset=["subject_id", "lifelog_date"])

    print(f"교차 타깃 피처 계산 중... ({len(all_query)}행, 21 타깃 쌍)")
    xt_feat = build_cross_target_features(train, all_query)

    out_path = DATA / "cross_target_features.csv"
    xt_feat.to_csv(out_path, index=False)

    n_feats = xt_feat.shape[1] - 2
    print(f"저장 완료: {out_path}  (행: {len(xt_feat)}, 피처: {n_feats}개)")

    key_cols = ["n_pos_lag1", "n_pos_roll7",
                "xcorr_Q1_S1_14", "xcorr_Q1_S1_28",
                "lag1_diff_Q1_Q2", "momentum_Q1_S1"]
    print("\nNaN 비율 (주요 피처):")
    for c in key_cols:
        if c in xt_feat.columns:
            print(f"  {c}: {xt_feat[c].isna().mean():.1%}")


if __name__ == "__main__":
    main()
