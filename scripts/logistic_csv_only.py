"""
Logistic Regression: train.csv 피처만 사용 (parquet 미사용)
피처 구성은 lgbm_csv_only.py와 동일
목적: 피처별 계수(coefficient)로 각 target에 대한 피처 중요도 파악
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


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


def print_feature_importance(coef_accum: dict, feature_cols: list):
    print("\n=== 피처 중요도 (계수 절댓값 평균, 5-Fold) ===")
    header = f"{'피처':<20}" + "".join(f"{t:>8}" for t in TARGETS)
    print(header)
    print("-" * (20 + 8 * len(TARGETS)))

    for feat in feature_cols:
        row = f"{feat:<20}"
        for t in TARGETS:
            row += f"{coef_accum[t][feat]:>8.3f}"
        print(row)


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    X_train_full = build_features(train, train, is_train=True)
    X_test_full = build_features(test, train, is_train=False)

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # target별 피처 계수 누적 (절댓값 평균)
    coef_accum = {t: {} for t in TARGETS}

    print(f"{'타깃':<5} {'Fold F1':^40} {'평균 F1':>8}")
    print("-" * 58)

    for t in TARGETS:
        y = train[t].values
        oof_preds = np.zeros(len(train))
        test_preds = np.zeros(len(test))

        drop_col = f"subj_mean_{t}"
        X_tr = X_train_full.drop(columns=[drop_col])
        X_te = X_test_full.drop(columns=[drop_col])
        feat_cols = X_tr.columns.tolist()

        fold_coefs = np.zeros(len(feat_cols))

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_full, y)):
            scaler = StandardScaler()
            X_tr_fold = scaler.fit_transform(X_tr.iloc[tr_idx])
            X_val_fold = scaler.transform(X_tr.iloc[val_idx])
            X_te_fold = scaler.transform(X_te)

            model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            model.fit(X_tr_fold, y[tr_idx])

            oof_preds[val_idx] = model.predict_proba(X_val_fold)[:, 1]
            test_preds += model.predict_proba(X_te_fold)[:, 1] / cv.n_splits
            fold_coefs += np.abs(model.coef_[0]) / cv.n_splits

        # fold 평균 계수 저장
        for feat, coef in zip(feat_cols, fold_coefs):
            coef_accum[t][feat] = coef
        # 제거된 자신의 피처는 0으로 표시
        coef_accum[t][drop_col] = 0.0

        fold_f1s = []
        for _, val_idx in cv.split(X_train_full, y):
            f1 = f1_score(y[val_idx], (oof_preds[val_idx] > 0.5).astype(int))
            fold_f1s.append(f1)

        mean_f1 = np.mean(fold_f1s)
        fold_str = "  ".join(f"{f:.3f}" for f in fold_f1s)
        print(f"{t:<5} {fold_str}  {mean_f1:.3f}")

        result[t] = (test_preds > 0.5).astype(int)

    all_feat_cols = X_train_full.columns.tolist()
    print_feature_importance(coef_accum, all_feat_cols)

    return result


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== Logistic Regression (CSV only) 학습 시작 ===\n")
    result = train_and_predict(train, sample)

    print("\n=== 예측 분포 ===")
    for t in TARGETS:
        ones = result[t].sum()
        total = len(result)
        print(f"  {t}: 1={ones} ({ones/total*100:.1f}%), 0={total-ones}")

    out_path = SUBMISSION_DIR / "logistic_csv_only.csv"
    result.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
