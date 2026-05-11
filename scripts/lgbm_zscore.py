"""
LightGBM + subject z-score 정규화
lgbm_final.py에 subject별 센서 피처 z-score 정규화 추가
- 센서 피처(parquet)를 subject 평균/표준편차 기준으로 정규화
- train fold 내 각 subject 통계로 train/val 정규화
- test는 전체 train 통계 사용
출력: submission/lgbm_zscore_prob.csv (clip 0.1~0.9)
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

BASELINE_F1 = {
    "Q1": 0.653, "Q2": 0.665, "Q3": 0.741,
    "S1": 0.793, "S2": 0.579, "S3": 0.537, "S4": 0.663,
}


# ── z-score 관련 헬퍼 ─────────────────────────────────────────────────────────

def get_sensor_cols(parquet_feat: pd.DataFrame) -> list:
    """z-score 적용 대상: parquet 센서 피처만 (날짜/레이블/lag/subj_mean 제외)"""
    return [c for c in parquet_feat.columns if c not in ("subject_id", "date")]


def compute_subj_stats(subject_ids: pd.Series, X: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    """
    subject별 각 센서 피처의 mean/std 계산
    반환: index=subject_id, columns=MultiIndex (col, stat)
    """
    avail = [c for c in sensor_cols if c in X.columns]
    tmp = X[avail].copy()
    tmp["subject_id"] = subject_ids.values
    stats = tmp.groupby("subject_id")[avail].agg(["mean", "std"])
    return stats


def apply_zscore(
    subject_ids: pd.Series,
    X: pd.DataFrame,
    stats: pd.DataFrame,
    sensor_cols: list,
) -> pd.DataFrame:
    """
    stats 기준으로 X에 z-score 정규화 적용
    stats에 없는 subject(새 subject)는 정규화 생략 (NaN 유지)
    std=0인 경우 정규화 생략 (해당 컬럼 그대로 유지)
    """
    X = X.copy()
    for col in sensor_cols:
        if col not in X.columns:
            continue
        if (col, "mean") not in stats.columns:
            continue
        means = subject_ids.map(stats[(col, "mean")])
        stds  = subject_ids.map(stats[(col, "std")])
        valid = stds.notna() & (stds > 0)
        # int 컬럼은 float으로 변환 후 z-score 적용
        X[col] = X[col].astype(float)
        X.loc[valid, col] = (X.loc[valid, col] - means[valid]) / stds[valid]
    return X


# ── 피처 빌더 ─────────────────────────────────────────────────────────────────

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["sleep_date"])
    df = df.copy()
    df["day_of_week"]   = dt.dt.dayofweek
    df["month"]         = dt.dt.month
    df["day_of_month"]  = dt.dt.day
    df["is_weekend"]    = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"]  = dt.dt.isocalendar().week.astype(int)
    return df


def add_subject_mean_features(
    df: pd.DataFrame, ref: pd.DataFrame, is_train: bool
) -> pd.DataFrame:
    subject_sum   = ref.groupby("subject_id")[TARGETS].sum()
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
    label_feat: pd.DataFrame,
    is_train: bool,
    le: LabelEncoder,
) -> pd.DataFrame:
    """피처 빌드. subject_id 컬럼 포함하여 반환 (z-score 적용 후 제거)"""
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

    df = df.merge(
        label_feat,
        left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    base_cols    = (
        ["subject_enc", "day_of_week", "month", "day_of_month", "is_weekend", "week_of_year"]
        + [f"subj_mean_{t}" for t in TARGETS]
    )
    parquet_cols = [c for c in parquet_feat.columns if c not in ("subject_id", "date")]
    label_cols   = [c for c in label_feat.columns   if c not in ("subject_id", "date")]

    # subject_id를 앞에 포함 (z-score용) — 모델 학습 전 제거
    all_cols = ["subject_id"] + base_cols + parquet_cols + label_cols
    all_cols = [c for c in all_cols if c in df.columns]
    return df[all_cols].reset_index(drop=True)


# ── 학습 & 예측 ───────────────────────────────────────────────────────────────

def train_and_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    parquet_feat: pd.DataFrame,
) -> pd.DataFrame:
    le     = LabelEncoder().fit(train["subject_id"])
    groups = train["subject_id"].values
    cv     = GroupKFold(n_splits=10)

    sensor_cols = get_sensor_cols(parquet_feat)

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()

    print(f"{'타깃':<5}  {'held-out subject':^14}  {'F1':>6}  {'LogLoss':>8}")
    print("-" * 46)

    all_results = {t: {"f1": [], "logloss": []} for t in TARGETS}

    dummy_X      = np.zeros(len(train))
    fold_indices = list(cv.split(dummy_X, train[TARGETS[0]].values, groups))

    label_feat_test = build_label_features(train, test)

    # test용 z-score 통계: 전체 train 기준으로 미리 계산
    label_feat_full = build_label_features(train, train)
    X_full = build_features(train, train, parquet_feat, label_feat_full, True, le)
    full_stats = compute_subj_stats(X_full["subject_id"], X_full, sensor_cols)

    for t in TARGETS:
        y        = train[t].values
        oof_preds  = np.zeros(len(train))
        test_preds = np.zeros(len(test))
        drop_col  = f"subj_mean_{t}"
        lag_drop  = [f"lag1_{t}", f"lag2_{t}", f"roll3_{t}", f"roll7_{t}"]

        for fold, (tr_idx, val_idx) in enumerate(fold_indices):
            held_out = train["subject_id"].iloc[val_idx].unique()[0]

            train_fold = train.iloc[tr_idx].copy()
            val_fold   = train.iloc[val_idx].copy()

            label_feat_tr  = build_label_features(train_fold, train_fold)
            label_feat_val = build_label_features(val_fold,   val_fold)

            X_tr  = build_features(train_fold, train_fold, parquet_feat, label_feat_tr,  True,  le)
            X_val = build_features(val_fold,   train_fold, parquet_feat, label_feat_val, False, le)

            # z-score: train fold 통계 → X_tr 정규화
            #          val fold 통계   → X_val 정규화 (val subject 자신의 분포 기준)
            tr_stats  = compute_subj_stats(X_tr["subject_id"],  X_tr,  sensor_cols)
            val_stats = compute_subj_stats(X_val["subject_id"], X_val, sensor_cols)

            sid_tr  = X_tr["subject_id"].reset_index(drop=True)
            sid_val = X_val["subject_id"].reset_index(drop=True)

            X_tr  = apply_zscore(sid_tr,  X_tr.drop(columns=["subject_id"]),  tr_stats,  sensor_cols)
            X_val = apply_zscore(sid_val, X_val.drop(columns=["subject_id"]), val_stats, sensor_cols)

            # 타깃 자신의 subj_mean + lag/roll 제거
            drop_cols = [drop_col] + [c for c in lag_drop if c in X_tr.columns]
            X_tr  = X_tr.drop(columns=drop_cols,  errors="ignore")
            X_val = X_val.drop(columns=drop_cols, errors="ignore")

            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_tr, y[tr_idx])
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

            # test 예측
            X_te = build_features(
                test.copy(), train, parquet_feat, label_feat_test, False, le
            )
            sid_te = X_te["subject_id"].reset_index(drop=True)
            X_te   = apply_zscore(sid_te, X_te.drop(columns=["subject_id"]), full_stats, sensor_cols)
            X_te   = X_te.drop(columns=drop_cols, errors="ignore")
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
        base    = BASELINE_F1[t]
        diff    = mean_f1 - base
        sign    = "+" if diff >= 0 else ""
        print(f"{t:<5}  {mean_f1:>8.3f}  {mean_ll:>12.4f}  {base:>8.3f}  {sign}{diff:>5.3f}")

    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v2 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== LightGBM + subject z-score 정규화 ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "lgbm_zscore_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
