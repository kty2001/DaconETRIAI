"""
GPS 모델 Feature Importance 분석
- 캐시된 extratrees_gps params 로드
- seed=42 단일 학습으로 타깃별/폴드별 importance 수집
- 평균 importance 기준 하위 피처 목록 출력 (slim 모델 구성용)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from gps_features import build_gps
from optuna_params_io import load_params

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]


def get_sensor_cols(parquet_feat):
    return [c for c in parquet_feat.columns if c not in ("subject_id", "date")]


def compute_subj_stats(subject_ids, X, sensor_cols):
    avail = [c for c in sensor_cols if c in X.columns]
    tmp = X[avail].copy()
    tmp["subject_id"] = subject_ids.values
    return tmp.groupby("subject_id")[avail].agg(["mean", "std"])


def apply_zscore(subject_ids, X, stats, sensor_cols):
    X = X.copy()
    for col in sensor_cols:
        if col not in X.columns or (col, "mean") not in stats.columns:
            continue
        means = subject_ids.map(stats[(col, "mean")])
        stds  = subject_ids.map(stats[(col, "std")])
        valid = stds.notna() & (stds > 0)
        X[col] = X[col].astype(float)
        X.loc[valid, col] = (X.loc[valid, col] - means[valid]) / stds[valid]
    return X


def add_date_features(df):
    dt = pd.to_datetime(df["sleep_date"])
    df = df.copy()
    df["day_of_week"]  = dt.dt.dayofweek
    df["month"]        = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def add_subject_mean_features(df, ref, is_train):
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


def build_features(df, ref, parquet_feat, label_feat, is_train, le):
    df = add_date_features(df)
    df = add_subject_mean_features(df, ref, is_train)
    df["subject_enc"] = df["subject_id"].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    df = df.merge(parquet_feat, left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")
    df = df.merge(label_feat, left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")
    base_cols    = (["subject_enc", "day_of_week", "month", "day_of_month",
                     "is_weekend", "week_of_year"]
                    + [f"subj_mean_{t}" for t in TARGETS])
    parquet_cols = [c for c in parquet_feat.columns if c not in ("subject_id", "date")]
    label_cols   = [c for c in label_feat.columns   if c not in ("subject_id", "date")]
    all_cols = ["subject_id"] + base_cols + parquet_cols + label_cols
    return df[[c for c in all_cols if c in df.columns]].reset_index(drop=True)


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v2 + GPS 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")

    cached_et = load_params("extratrees_gps")
    if not cached_et:
        print("ERROR: extratrees_gps params 캐시 없음")
        return
    best_et = {t: cached_et[t] for t in TARGETS}

    le          = LabelEncoder().fit(train["subject_id"])
    groups      = train["subject_id"].values
    cv          = GroupKFold(n_splits=10)
    sensor_cols = get_sensor_cols(parquet_feat)

    fold_indices = list(cv.split(np.zeros(len(train)), train[TARGETS[0]].values, groups))

    print("label 피처 계산 중...")
    fold_label_feats = []
    for tr_idx, val_idx in fold_indices:
        train_fold = train.iloc[tr_idx].copy()
        val_fold   = train.iloc[val_idx].copy()
        lf_tr  = build_label_features(train_fold, train_fold)
        lf_val = build_label_features(val_fold,   val_fold)
        fold_label_feats.append((tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val))
    print("  완료")

    # feature importance 수집
    print("\n=== Feature Importance 계산 (seed=42) ===")
    importance_dict = {t: [] for t in TARGETS}
    feature_names_ref = None

    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr, True, le)
        sid_tr = X_tr["subject_id"].reset_index(drop=True)
        tr_stats = compute_subj_stats(sid_tr, X_tr, sensor_cols)
        X_tr_z  = apply_zscore(sid_tr, X_tr.drop(columns=["subject_id"]), tr_stats, sensor_cols)

        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            X_tr_t = X_tr_z.drop(columns=drop_cols, errors="ignore")
            if feature_names_ref is None:
                feature_names_ref = {t: X_tr_t.columns.tolist()}
            elif t not in feature_names_ref:
                feature_names_ref[t] = X_tr_t.columns.tolist()

            tr_median   = X_tr_t.median()
            X_tr_filled = X_tr_t.fillna(tr_median)

            y = train[t].values
            model = ExtraTreesClassifier(**{**best_et[t], "random_state": 42})
            model.fit(X_tr_filled, y[tr_idx])
            importance_dict[t].append(model.feature_importances_)

    # 타깃별 평균 importance
    print("\n=== 타깃별 Feature Importance (상위 30) ===")
    all_imp = {}
    for t in TARGETS:
        imp = np.mean(importance_dict[t], axis=0)
        feat_names = feature_names_ref[t]
        imp_series = pd.Series(imp, index=feat_names).sort_values(ascending=False)
        all_imp[t] = imp_series

    # 전체 타깃 평균 importance (정규화)
    # 타깃마다 피처 집합이 약간 다르므로 공통 피처 기준으로 합산
    from functools import reduce
    common_feats = reduce(lambda a, b: a & b,
                          [set(all_imp[t].index) for t in TARGETS])
    combined = pd.DataFrame({t: all_imp[t][list(common_feats)] for t in TARGETS})
    combined["mean_imp"] = combined.mean(axis=1)
    combined = combined.sort_values("mean_imp", ascending=False)

    print(f"\n공통 피처 수: {len(common_feats)}")
    print("\n--- 상위 30개 ---")
    print(combined["mean_imp"].head(30).round(6).to_string())

    print("\n--- 하위 30개 (제거 후보) ---")
    print(combined["mean_imp"].tail(30).round(6).to_string())

    # 임계값 기준 제거 대상
    threshold = 0.001
    low_imp = combined[combined["mean_imp"] < threshold].index.tolist()
    print(f"\n임계값 {threshold} 미만 피처 수: {len(low_imp)}")
    print("제거 후보:", low_imp)

    # 누적 중요도 기준
    total = combined["mean_imp"].sum()
    combined["cumsum"] = combined["mean_imp"].cumsum() / total
    n_top_90 = (combined["cumsum"] <= 0.90).sum()
    n_top_95 = (combined["cumsum"] <= 0.95).sum()
    n_top_99 = (combined["cumsum"] <= 0.99).sum()
    print(f"\n상위 90% 중요도 커버: {n_top_90}개 피처")
    print(f"상위 95% 중요도 커버: {n_top_95}개 피처")
    print(f"상위 99% 중요도 커버: {n_top_99}개 피처")

    # 제거 대상 저장
    drop_file = ROOT / "submission" / "low_importance_features.txt"
    with open(drop_file, "w") as f:
        for feat in low_imp:
            f.write(feat + "\n")
    print(f"\n제거 후보 목록 저장: {drop_file}")


if __name__ == "__main__":
    main()
