"""
LightGBM + GroupKFold CV (CSV only)
n_splits=10 → LOSO: fold마다 subject 1명 전체를 validation으로 사용
피처는 fold 내 학습 데이터만으로 계산해 leakage 방지
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
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


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["sleep_date"])
    df = df.copy()
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def add_subject_mean_features(
    df: pd.DataFrame, ref: pd.DataFrame, is_train: bool
) -> pd.DataFrame:
    """ref 데이터 기준으로 subject 평균 계산.
    is_train=True: LOO (자기 자신 제외)
    is_train=False: ref 전체 평균 (held-out subject는 NaN → LightGBM이 처리)
    """
    subject_sum = ref.groupby("subject_id")[TARGETS].sum()
    subject_count = ref.groupby("subject_id")[TARGETS].count()

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
    df: pd.DataFrame, ref: pd.DataFrame, is_train: bool, le: LabelEncoder
) -> pd.DataFrame:
    df = add_date_features(df)
    df = add_subject_mean_features(df, ref, is_train)
    df["subject_enc"] = df["subject_id"].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    feature_cols = (
        ["subject_enc", "day_of_week", "month", "day_of_month", "is_weekend", "week_of_year"]
        + [f"subj_mean_{t}" for t in TARGETS]
    )
    return df[feature_cols]


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder().fit(train["subject_id"])
    groups = train["subject_id"].values
    cv = GroupKFold(n_splits=10)

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()

    # 전체 train 피처 (test 예측용 최종 모델에 사용)
    X_train_all = build_features(train.copy(), train, is_train=True, le=le)
    X_test = build_features(test.copy(), train, is_train=False, le=le)

    print(f"{'타깃':<5}  {'held-out subject':^14}  {'F1':>6}  {'LogLoss':>8}")
    print("-" * 46)

    all_results = {t: {"f1": [], "logloss": [], "subject": []} for t in TARGETS}

    for t in TARGETS:
        y = train[t].values
        oof_preds = np.zeros(len(train))
        test_preds = np.zeros(len(test))

        drop_col = f"subj_mean_{t}"

        for fold, (tr_idx, val_idx) in enumerate(cv.split(X_train_all, y, groups)):
            held_out = train["subject_id"].iloc[val_idx].unique()[0]

            # fold 내에서 피처 재계산 (leakage 방지)
            train_fold = train.iloc[tr_idx].copy()
            val_fold = train.iloc[val_idx].copy()

            X_tr = build_features(train_fold, train_fold, is_train=True, le=le)
            X_val = build_features(val_fold, train_fold, is_train=False, le=le)

            X_tr = X_tr.drop(columns=[drop_col])
            X_val = X_val.drop(columns=[drop_col])

            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_tr, y[tr_idx])
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

            # test 예측: 전체 train으로 학습한 모델을 fold 수만큼 평균
            X_te_fold = X_test.drop(columns=[drop_col])
            test_preds += model.predict_proba(X_te_fold)[:, 1] / cv.n_splits

            f1 = f1_score(y[val_idx], (oof_preds[val_idx] > 0.5).astype(int))
            ll = log_loss(y[val_idx], oof_preds[val_idx])

            all_results[t]["f1"].append(f1)
            all_results[t]["logloss"].append(ll)
            all_results[t]["subject"].append(held_out)

            print(f"{t:<5}  {held_out:^14}  {f1:>6.3f}  {ll:>8.4f}")

        result[t] = (test_preds > 0.5).astype(int)

    print()
    print(f"{'타깃':<5}  {'평균 F1':>8}  {'평균 LogLoss':>12}")
    print("-" * 32)
    for t in TARGETS:
        mean_f1 = np.mean(all_results[t]["f1"])
        mean_ll = np.mean(all_results[t]["logloss"])
        print(f"{t:<5}  {mean_f1:>8.3f}  {mean_ll:>12.4f}")

    return result


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== LightGBM + GroupKFold (CSV only) ===\n")
    result = train_and_predict(train, sample)

    print("\n=== 예측 분포 ===")
    for t in TARGETS:
        ones = result[t].sum()
        total = len(result)
        print(f"  {t}: 1={ones} ({ones/total*100:.1f}%), 0={total-ones}")

    out_path = SUBMISSION_DIR / "lgbm_csv_group.csv"
    result.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
