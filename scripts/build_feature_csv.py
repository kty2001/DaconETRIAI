"""
모든 피처를 하나의 CSV로 저장
- ch2026_metrics_train.csv (레이블 포함)
- parquet 집계 피처 v2 (시간대별 + mAmbience)
- label 피처 (lag1/2/7, roll3/7/14/21/28, rollstd7/14)
출력: data/features_all_v2.csv
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from parquet_features_v2 import build_all as build_parquet_features
from label_features import build_label_features

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v2 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== label 피처 (lag/roll) 계산 중 ===")
    print("  train 행 계산 중...")
    label_feat_train = build_label_features(train, train)
    print("  test 행 계산 중...")
    label_feat_test = build_label_features(train, sample)
    print()

    # ── train 합치기 ──────────────────────────────────────────────
    train_merged = train.merge(
        parquet_feat,
        left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    train_merged = train_merged.merge(
        label_feat_train,
        left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    train_merged["split"] = "train"

    # ── test 합치기 ───────────────────────────────────────────────
    test_merged = sample.merge(
        parquet_feat,
        left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    test_merged = test_merged.merge(
        label_feat_test,
        left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    test_merged["split"] = "test"

    # ── 합치기 ────────────────────────────────────────────────────
    combined = pd.concat([train_merged, test_merged], ignore_index=True)

    out_path = DATA / "features_all_v2.csv"
    combined.to_csv(out_path, index=False)

    print(f"=== 저장 완료: {out_path} ===")
    print(f"  전체 행: {len(combined)} (train={len(train_merged)}, test={len(test_merged)})")
    print(f"  전체 컬럼: {len(combined.columns)}")

    label_cols = [c for c in combined.columns if any(
        c.startswith(p) for p in ("lag", "roll")
    )]
    parquet_cols = [c for c in combined.columns if c not in
                    list(train.columns) + label_cols + ["split"]]
    print(f"\n  피처 구성:")
    print(f"    parquet 피처: {len(parquet_cols)}개")
    print(f"    lag/roll 피처: {len(label_cols)}개")
    print(f"      (기존) lag1/2/7, roll3/7/14 × 7타깃 = {6*7}개")
    print(f"      (신규) roll21/28, rollstd7/14 × 7타깃 = {4*7}개")


if __name__ == "__main__":
    main()
