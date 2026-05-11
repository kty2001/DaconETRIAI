"""
베이스라인: subject별 target 평균으로 예측
각 subject의 train 데이터에서 Q1~S4 평균을 계산하고,
0.5 초과이면 1, 이하이면 0으로 예측
"""

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def compute_subject_means(train: pd.DataFrame) -> pd.DataFrame:
    """subject별 각 target의 평균값 계산"""
    return train.groupby("subject_id")[TARGETS].mean()


def predict(subject_means: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """submission sample에 예측값 채우기"""
    result = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()

    for target in TARGETS:
        result[target] = result["subject_id"].map(subject_means[target])

    # 0.5 초과 → 1, 이하 → 0
    result[TARGETS] = (result[TARGETS] > 0.5).astype(int)

    return result


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    subject_means = compute_subject_means(train)

    print("=== subject별 target 평균 ===")
    print(subject_means.round(3).to_string())
    print()

    result = predict(subject_means, sample)

    print("=== 예측 분포 ===")
    for t in TARGETS:
        ones = result[t].sum()
        total = len(result)
        print(f"  {t}: 1={ones} ({ones/total*100:.1f}%), 0={total-ones}")
    print()

    out_path = SUBMISSION_DIR / "baseline_subject_mean.csv"
    result.to_csv(out_path, index=False)
    print(f"저장 완료: {out_path}")


if __name__ == "__main__":
    main()
