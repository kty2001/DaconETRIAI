"""
LightGBM 모델: train.csv 피처만 사용 (parquet 미사용)
피처: subject 인코딩, subject별 target 평균, 날짜 피처
타깃 7개를 개별 이진 분류기로 학습
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 10,
    "random_state": 42,
}


def add_date_features(df: pd.DataFrame, date_col: str = "sleep_date") -> pd.DataFrame:
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def add_subject_mean_features(
    df: pd.DataFrame, train: pd.DataFrame, is_train: bool
) -> pd.DataFrame:
    """subject별 각 target 평균을 피처로 추가.
    train은 자신을 제외한 leave-one-out 평균, test는 전체 train 평균 사용.
    """
    subject_sum = train.groupby("subject_id")[TARGETS].sum()
    subject_count = train.groupby("subject_id")[TARGETS].count()

    if is_train:
        for t in TARGETS:
            s_sum = df["subject_id"].map(subject_sum[t])
            s_cnt = df["subject_id"].map(subject_count[t])
            df[f"subj_mean_{t}"] = (s_sum - df[t]) / (s_cnt - 1).clip(lower=1)
    else:
        subject_mean = subject_sum / subject_count
        for t in TARGETS:
            df[f"subj_mean_{t}"] = df["subject_id"].map(subject_mean[t])

    return df


def build_features(df: pd.DataFrame, train: pd.DataFrame, is_train: bool) -> pd.DataFrame:
    df = df.copy()
    df = add_date_features(df)
    df = add_subject_mean_features(df, train, is_train)

    le = LabelEncoder().fit(train["subject_id"])
    df["subject_enc"] = le.transform(df["subject_id"])

    feature_cols = (
        ["subject_enc", "day_of_week", "month", "day_of_month", "is_weekend", "week_of_year"]
        + [f"subj_mean_{t}" for t in TARGETS]
    )
    return df[feature_cols]


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    X_train = build_features(train, train, is_train=True)
    X_test = build_features(test, train, is_train=False)

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"{'타깃':<5} {'Fold F1':^40} {'평균 F1':>8}")
    print("-" * 58)

    for t in TARGETS:
        y = train[t].values
        oof_preds = np.zeros(len(train))
        test_preds = np.zeros(len(test))

        # 예측 대상 target의 subject mean 피처는 리케이지 방지를 위해 제거
        drop_col = f"subj_mean_{t}"
        X_tr = X_train.drop(columns=[drop_col])
        X_te = X_test.drop(columns=[drop_col])

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y)):
            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_tr.iloc[tr_idx], y[tr_idx])

            oof_preds[val_idx] = model.predict_proba(X_tr.iloc[val_idx])[:, 1]
            test_preds += model.predict_proba(X_te)[:, 1] / cv.n_splits

        fold_f1s = []
        for fold, (_, val_idx) in enumerate(cv.split(X_train, y)):
            f1 = f1_score(y[val_idx], (oof_preds[val_idx] > 0.5).astype(int))
            fold_f1s.append(f1)

        mean_f1 = np.mean(fold_f1s)
        fold_str = "  ".join(f"{f:.3f}" for f in fold_f1s)
        print(f"{t:<5} {fold_str}  {mean_f1:.3f}")

        result[t] = (test_preds > 0.5).astype(int)

    return result


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== LightGBM (CSV only) 학습 시작 ===\n")
    result = train_and_predict(train, sample)

    print("\n=== 예측 분포 ===")
    for t in TARGETS:
        ones = result[t].sum()
        total = len(result)
        print(f"  {t}: 1={ones} ({ones/total*100:.1f}%), 0={total-ones}")

    out_path = SUBMISSION_DIR / "lgbm_csv_only.csv"
    result.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
