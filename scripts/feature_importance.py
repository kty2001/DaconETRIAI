"""
LightGBM gain 기반 피처 중요도 분석
lgbm_optuna.py 동일 파이프라인 (LOSO CV, z-score) 사용
폴드별 importance를 평균내어 타깃별 top-20 출력
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "random_state": 42,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
}

TOP_N = 20


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
        if col not in X.columns:
            continue
        if (col, "mean") not in stats.columns:
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
    all_cols = [c for c in all_cols if c in df.columns]
    return df[all_cols].reset_index(drop=True)


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 집계 중 ===")
    parquet_feat = build_parquet_features()

    le          = LabelEncoder().fit(train["subject_id"])
    groups      = train["subject_id"].values
    cv          = GroupKFold(n_splits=10)
    sensor_cols = get_sensor_cols(parquet_feat)

    dummy_X      = np.zeros(len(train))
    fold_indices = list(cv.split(dummy_X, train[TARGETS[0]].values, groups))

    print("label 피처 사전 계산 중...")
    fold_label_feats = []
    for tr_idx, val_idx in fold_indices:
        train_fold = train.iloc[tr_idx].copy()
        val_fold   = train.iloc[val_idx].copy()
        lf_tr  = build_label_features(train_fold, train_fold)
        lf_val = build_label_features(val_fold, val_fold)
        fold_label_feats.append((tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val))
    print(f"  완료 ({len(fold_indices)} folds)\n")

    # 타깃별 폴드 평균 importance 집계
    importance_by_target = {}

    for t in TARGETS:
        drop_col = f"subj_mean_{t}"
        fold_importances = []
        feature_names = None

        for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
            X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
            X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)

            sid_tr  = X_tr["subject_id"].reset_index(drop=True)
            sid_val = X_val["subject_id"].reset_index(drop=True)

            tr_stats = compute_subj_stats(sid_tr, X_tr, sensor_cols)
            val_stats = compute_subj_stats(sid_val, X_val, sensor_cols)

            X_tr_z  = apply_zscore(sid_tr,  X_tr.drop(columns=["subject_id"]),  tr_stats,  sensor_cols)
            X_val_z = apply_zscore(sid_val, X_val.drop(columns=["subject_id"]), val_stats, sensor_cols)

            X_tr_z  = X_tr_z.drop(columns=[drop_col], errors="ignore")

            y_tr = train[t].values[tr_idx]

            model = lgb.LGBMClassifier(**PARAMS)
            model.fit(X_tr_z, y_tr)

            if feature_names is None:
                feature_names = X_tr_z.columns.tolist()

            fold_importances.append(model.feature_importances_)

        imp_arr = np.array(fold_importances)
        mean_imp = imp_arr.mean(axis=0)
        importance_by_target[t] = pd.Series(mean_imp, index=feature_names).sort_values(ascending=False)

    # ── 결과 출력 ─────────────────────────────────────────────────────────────
    print("=" * 65)
    print(f"  타깃별 피처 중요도 (gain 기준, 폴드 평균, Top-{TOP_N})")
    print("=" * 65)

    for t in TARGETS:
        imp = importance_by_target[t]
        total = imp.sum()
        print(f"\n[{t}] (전체 importance 합: {total:.1f})")
        print(f"  {'순위':>4}  {'피처명':<35}  {'중요도':>8}  {'비율':>6}")
        print(f"  {'-'*4}  {'-'*35}  {'-'*8}  {'-'*6}")
        for rank, (feat, val) in enumerate(imp.head(TOP_N).items(), 1):
            ratio = val / total * 100
            print(f"  {rank:>4}  {feat:<35}  {val:>8.1f}  {ratio:>5.1f}%")

    # ── 전체 타깃 평균 순위 ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  전체 타깃 평균 피처 중요도 (Top-30)")
    print("=" * 65)

    all_feats = set()
    for imp in importance_by_target.values():
        all_feats.update(imp.index.tolist())

    combined = pd.DataFrame({t: importance_by_target[t] for t in TARGETS}).fillna(0)
    combined["mean"] = combined.mean(axis=1)
    combined = combined.sort_values("mean", ascending=False)

    print(f"\n  {'순위':>4}  {'피처명':<35}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  {'평균':>7}")
    print(f"  {'-'*4}  {'-'*35}  " + "  ".join(["-"*6]*len(TARGETS)) + "  {'-'*7}")
    for rank, (feat, row) in enumerate(combined.head(30).iterrows(), 1):
        vals = "  ".join(f"{row[t]:>6.0f}" for t in TARGETS)
        print(f"  {rank:>4}  {feat:<35}  {vals}  {row['mean']:>7.1f}")


if __name__ == "__main__":
    main()
