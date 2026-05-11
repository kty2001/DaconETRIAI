"""
LightGBM + GroupKFold CV (parquet 피처 포함)
lgbm_csv_group.py 기반 + parquet 집계 피처 추가
출력: hard 0/1 (lgbm_parquet_group.csv)
      clip(0.1~0.9) 확률값 (lgbm_parquet_group_prob.csv)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
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
    df: pd.DataFrame,
    ref: pd.DataFrame,
    parquet_feat: pd.DataFrame,
    is_train: bool,
    le: LabelEncoder,
) -> pd.DataFrame:
    df = add_date_features(df)
    df = add_subject_mean_features(df, ref, is_train)
    df["subject_enc"] = df["subject_id"].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
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
    le = LabelEncoder().fit(train["subject_id"])
    groups = train["subject_id"].values
    cv = GroupKFold(n_splits=10)

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()

    print(f"{'타깃':<5}  {'held-out subject':^14}  {'F1':>6}  {'LogLoss':>8}")
    print("-" * 46)

    all_results = {t: {"f1": [], "logloss": []} for t in TARGETS}

    # dummy split으로 fold 인덱스 미리 추출
    dummy_X = np.zeros(len(train))
    fold_indices = list(cv.split(dummy_X, train[TARGETS[0]].values, groups))

    for t in TARGETS:
        y = train[t].values
        oof_preds = np.zeros(len(train))
        test_preds = np.zeros(len(test))

        drop_col = f"subj_mean_{t}"

        for fold, (tr_idx, val_idx) in enumerate(fold_indices):
            held_out = train["subject_id"].iloc[val_idx].unique()[0]

            train_fold = train.iloc[tr_idx].copy()
            val_fold = train.iloc[val_idx].copy()

            X_tr = build_features(train_fold, train_fold, parquet_feat, is_train=True, le=le)
            X_val = build_features(val_fold, train_fold, parquet_feat, is_train=False, le=le)
            X_tr = X_tr.drop(columns=[drop_col])
            X_val = X_val.drop(columns=[drop_col])

            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_tr, y[tr_idx])
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

            # test 예측: fold 수만큼 평균
            X_te_full = build_features(
                test.copy(), train, parquet_feat, is_train=False, le=le
            ).drop(columns=[drop_col])
            test_preds += model.predict_proba(X_te_full)[:, 1] / cv.n_splits

            f1 = f1_score(y[val_idx], (oof_preds[val_idx] > 0.5).astype(int))
            ll = log_loss(y[val_idx], oof_preds[val_idx])

            all_results[t]["f1"].append(f1)
            all_results[t]["logloss"].append(ll)

            print(f"{t:<5}  {held_out:^14}  {f1:>6.3f}  {ll:>8.4f}")

        result[t] = test_preds  # 확률값 보관 (저장 시 변환)

    print()
    print(f"{'타깃':<5}  {'평균 F1':>8}  {'평균 LogLoss':>12}  {'csv_group F1':>13}  {'차이':>6}")
    print("-" * 55)
    csv_f1 = {"Q1": 0.629, "Q2": 0.689, "Q3": 0.727, "S1": 0.796,
              "S2": 0.745, "S3": 0.372, "S4": 0.675}
    for t in TARGETS:
        mean_f1 = np.mean(all_results[t]["f1"])
        mean_ll = np.mean(all_results[t]["logloss"])
        diff = mean_f1 - csv_f1[t]
        sign = "+" if diff >= 0 else ""
        print(f"{t:<5}  {mean_f1:>8.3f}  {mean_ll:>12.4f}  {csv_f1[t]:>13.3f}  {sign}{diff:>5.3f}")

    return result


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== LightGBM + GroupKFold (parquet 포함) ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    # --- hard 0/1 버전 ---
    result_hard = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_hard[t] = (result[t] > 0.5).astype(int)

    print("\n=== 예측 분포 (hard 0/1) ===")
    for t in TARGETS:
        ones = result_hard[t].sum()
        total = len(result_hard)
        print(f"  {t}: 1={ones} ({ones/total*100:.1f}%), 0={total-ones}")

    hard_path = SUBMISSION_DIR / "lgbm_parquet_group.csv"
    result_hard.to_csv(hard_path, index=False)
    print(f"저장 완료: {hard_path}")

    # --- clip(0.1~0.9) 확률값 버전 ---
    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, mean={result_prob[t].mean():.3f}, max={result_prob[t].max():.3f}")

    prob_path = SUBMISSION_DIR / "lgbm_parquet_group_prob.csv"
    result_prob.to_csv(prob_path, index=False)
    print(f"저장 완료: {prob_path}")


if __name__ == "__main__":
    main()
