"""
LightGBM + parquet 피처: lgbm_csv_only.py 피처에 parquet 집계 피처 추가
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from parquet_features import build_all as build_parquet_features

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


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["sleep_date"])
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def add_subject_mean_features(df: pd.DataFrame, train: pd.DataFrame, is_train: bool) -> pd.DataFrame:
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


def build_features(
    df: pd.DataFrame,
    train: pd.DataFrame,
    parquet_feat: pd.DataFrame,
    is_train: bool,
) -> pd.DataFrame:
    df = df.copy()
    df = add_date_features(df)
    df = add_subject_mean_features(df, train, is_train)

    le = LabelEncoder().fit(train["subject_id"])
    df["subject_enc"] = le.transform(df["subject_id"])

    # parquet 피처 join (lifelog_date == parquet date)
    df = df.merge(
        parquet_feat,
        left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    base_cols = (
        ["subject_enc", "day_of_week", "month", "day_of_month", "is_weekend", "week_of_year"]
        + [f"subj_mean_{t}" for t in TARGETS]
    )
    parquet_cols = [c for c in parquet_feat.columns if c not in ("subject_id", "date")]
    return df[base_cols + parquet_cols]


def train_and_predict(
    train: pd.DataFrame, test: pd.DataFrame, parquet_feat: pd.DataFrame
) -> pd.DataFrame:
    X_train = build_features(train, train, parquet_feat, is_train=True)
    X_test = build_features(test, train, parquet_feat, is_train=False)

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"{'타깃':<5} {'Fold F1':^40} {'평균 F1':>8}")
    print("-" * 58)

    feat_importance = pd.DataFrame(index=X_train.columns)

    for t in TARGETS:
        y = train[t].values
        oof_preds = np.zeros(len(train))
        test_preds = np.zeros(len(test))

        drop_col = f"subj_mean_{t}"
        X_tr = X_train.drop(columns=[drop_col])
        X_te = X_test.drop(columns=[drop_col])

        importance_sum = np.zeros(X_tr.shape[1])

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train, y)):
            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_tr.iloc[tr_idx], y[tr_idx])
            oof_preds[val_idx] = model.predict_proba(X_tr.iloc[val_idx])[:, 1]
            test_preds += model.predict_proba(X_te)[:, 1] / cv.n_splits
            importance_sum += model.feature_importances_ / cv.n_splits

        feat_importance[t] = pd.Series(importance_sum, index=X_tr.columns)

        fold_f1s = []
        for _, val_idx in cv.split(X_train, y):
            f1 = f1_score(y[val_idx], (oof_preds[val_idx] > 0.5).astype(int))
            fold_f1s.append(f1)

        mean_f1 = np.mean(fold_f1s)
        fold_str = "  ".join(f"{f:.3f}" for f in fold_f1s)
        print(f"{t:<5} {fold_str}  {mean_f1:.3f}")

        result[t] = (test_preds > 0.5).astype(int)

    print("\n=== 피처 중요도 (상위 10개, target별 평균) ===")
    feat_importance["mean"] = feat_importance.mean(axis=1)
    top10 = feat_importance["mean"].sort_values(ascending=False).head(10)
    for feat, val in top10.items():
        print(f"  {feat:<30} {val:.1f}")

    return result


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== LightGBM + parquet 학습 시작 ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    print("\n=== 예측 분포 ===")
    for t in TARGETS:
        ones = result[t].sum()
        total = len(result)
        print(f"  {t}: 1={ones} ({ones/total*100:.1f}%), 0={total-ones}")

    out_path = SUBMISSION_DIR / "lgbm_with_parquet.csv"
    result.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
