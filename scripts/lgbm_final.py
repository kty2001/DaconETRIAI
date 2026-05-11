"""
LightGBM 최종 모델: lag1/2 + roll3/7 + parquet v2 피처
GroupKFold(10) LOSO CV
출력: submission/lgbm_final_prob.csv (clip 0.1~0.9 확률값)
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
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features

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

# parquet group 기준 F1 (비교용)
BASELINE_F1 = {
    "Q1": 0.647, "Q2": 0.688, "Q3": 0.740,
    "S1": 0.811, "S2": 0.688, "S3": 0.494, "S4": 0.675,
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
    full_train: pd.DataFrame,
    parquet_feat: pd.DataFrame,
    label_feat: pd.DataFrame,
    is_train: bool,
    le: LabelEncoder,
) -> pd.DataFrame:
    df = add_date_features(df)
    df = add_subject_mean_features(df, ref, is_train)
    df["subject_enc"] = df["subject_id"].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )

    # parquet 피처 join
    df = df.merge(
        parquet_feat,
        left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    # label 피처 join (lag/roll)
    df = df.merge(
        label_feat,
        left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    base_cols = (
        ["subject_enc", "day_of_week", "month", "day_of_month", "is_weekend", "week_of_year"]
        + [f"subj_mean_{t}" for t in TARGETS]
    )
    parquet_cols = [c for c in parquet_feat.columns if c not in ("subject_id", "date")]
    label_cols = [c for c in label_feat.columns if c not in ("subject_id", "date")]

    all_cols = base_cols + parquet_cols + label_cols
    # 실제 존재하는 컬럼만 선택
    all_cols = [c for c in all_cols if c in df.columns]
    return df[all_cols]


def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    parquet_feat: pd.DataFrame,
) -> pd.DataFrame:
    le = LabelEncoder().fit(train["subject_id"])
    groups = train["subject_id"].values
    cv = GroupKFold(n_splits=10)

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()

    print(f"{'타깃':<5}  {'held-out subject':^14}  {'F1':>6}  {'LogLoss':>8}")
    print("-" * 46)

    all_results = {t: {"f1": [], "logloss": []} for t in TARGETS}

    dummy_X = np.zeros(len(train))
    fold_indices = list(cv.split(dummy_X, train[TARGETS[0]].values, groups))

    # test용 label 피처는 전체 train 기준으로 미리 계산
    label_feat_test = build_label_features(train, test)

    for t in TARGETS:
        y = train[t].values
        oof_preds = np.zeros(len(train))
        test_preds = np.zeros(len(test))
        drop_col = f"subj_mean_{t}"
        lag_drop = [f"lag1_{t}", f"lag2_{t}", f"roll3_{t}", f"roll7_{t}"]

        for fold, (tr_idx, val_idx) in enumerate(fold_indices):
            held_out = train["subject_id"].iloc[val_idx].unique()[0]

            train_fold = train.iloc[tr_idx].copy()
            val_fold = train.iloc[val_idx].copy()

            # label 피처 계산
            # train: fold 내 train 기준 (leakage 방지)
            # val: held-out subject 자신의 과거 데이터 사용 (날짜 순, 미래 행 제외)
            label_feat_tr = build_label_features(train_fold, train_fold)
            label_feat_val = build_label_features(val_fold, val_fold)

            X_tr = build_features(
                train_fold, train_fold, train, parquet_feat, label_feat_tr, True, le
            )
            X_val = build_features(
                val_fold, train_fold, train, parquet_feat, label_feat_val, False, le
            )

            # 타깃 자신의 subj_mean + lag/roll 제거
            drop_cols = [drop_col] + [c for c in lag_drop if c in X_tr.columns]
            X_tr = X_tr.drop(columns=drop_cols, errors="ignore")
            X_val = X_val.drop(columns=drop_cols, errors="ignore")

            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_tr, y[tr_idx])
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

            # test 예측: fold별 평균
            X_te = build_features(
                test.copy(), train, train, parquet_feat, label_feat_test, False, le
            ).drop(columns=drop_cols, errors="ignore")
            test_preds += model.predict_proba(X_te)[:, 1] / cv.n_splits

            f1 = f1_score(y[val_idx], (oof_preds[val_idx] > 0.5).astype(int))
            ll = log_loss(y[val_idx], oof_preds[val_idx])
            all_results[t]["f1"].append(f1)
            all_results[t]["logloss"].append(ll)
            print(f"{t:<5}  {held_out:^14}  {f1:>6.3f}  {ll:>8.4f}")

        result[t] = test_preds

    print()
    print(f"{'타깃':<5}  {'평균 F1':>8}  {'평균 LogLoss':>12}  {'기준 F1':>8}  {'차이':>6}")
    print("-" * 52)
    for t in TARGETS:
        mean_f1 = np.mean(all_results[t]["f1"])
        mean_ll = np.mean(all_results[t]["logloss"])
        base = BASELINE_F1[t]
        diff = mean_f1 - base
        sign = "+" if diff >= 0 else ""
        print(f"{t:<5}  {mean_f1:>8.3f}  {mean_ll:>12.4f}  {base:>8.3f}  {sign}{diff:>5.3f}")

    return result


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v2 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== LightGBM 최종 모델 (lag+roll+parquet v2+GroupKFold) ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    # clip(0.1~0.9) 확률값
    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        mn = result_prob[t].min()
        mean = result_prob[t].mean()
        mx = result_prob[t].max()
        print(f"  {t}: min={mn:.3f}, mean={mean:.3f}, max={mx:.3f}")

    out_path = SUBMISSION_DIR / "lgbm_final_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
