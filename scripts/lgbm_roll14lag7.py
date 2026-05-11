"""
LightGBM + lag7/roll14 확장 피처
lgbm_multiseed.py 기반에서 lag7(7일 전 레이블) + roll14(2주 이동평균) 추가
- lag7: target-7일 ± 2일 이내 값 → 주간 패턴 포착
- roll14: 60일 이내 최근 14개 평균 → 2주 기준선
- 동일 타깃의 lag7/roll14도 drop (기존 lag1/2/roll3/7과 동일 방침)
출력: submission/lgbm_roll14lag7_prob.csv (clip 0.1~0.9)
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
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

LGBM_BASE_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "verbosity":         -1,
    "n_estimators":      200,
    "learning_rate":     0.05,
    "num_leaves":        15,
    "min_child_samples": 10,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
}

BASELINE_F1 = {
    "Q1": 0.649, "Q2": 0.662, "Q3": 0.712,
    "S1": 0.788, "S2": 0.611, "S3": 0.526, "S4": 0.611,
}


# ── z-score 헬퍼 ──────────────────────────────────────────────────────────────

def get_sensor_cols(parquet_feat: pd.DataFrame) -> list:
    return [c for c in parquet_feat.columns if c not in ("subject_id", "date")]


def compute_subj_stats(subject_ids: pd.Series, X: pd.DataFrame, sensor_cols: list) -> pd.DataFrame:
    avail = [c for c in sensor_cols if c in X.columns]
    tmp = X[avail].copy()
    tmp["subject_id"] = subject_ids.values
    return tmp.groupby("subject_id")[avail].agg(["mean", "std"])


def apply_zscore(
    subject_ids: pd.Series,
    X: pd.DataFrame,
    stats: pd.DataFrame,
    sensor_cols: list,
) -> pd.DataFrame:
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


# ── 피처 빌더 ─────────────────────────────────────────────────────────────────

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["sleep_date"])
    df = df.copy()
    df["day_of_week"]  = dt.dt.dayofweek
    df["month"]        = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def add_subject_mean_features(df: pd.DataFrame, ref: pd.DataFrame, is_train: bool) -> pd.DataFrame:
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

    all_cols = ["subject_id"] + base_cols + parquet_cols + label_cols
    all_cols = [c for c in all_cols if c in df.columns]
    return df[all_cols].reset_index(drop=True)


# ── 학습 & 예측 ───────────────────────────────────────────────────────────────

def train_and_predict(train: pd.DataFrame, test: pd.DataFrame, parquet_feat: pd.DataFrame) -> pd.DataFrame:
    le          = LabelEncoder().fit(train["subject_id"])
    groups      = train["subject_id"].values
    cv          = GroupKFold(n_splits=10)
    sensor_cols = get_sensor_cols(parquet_feat)
    n_seeds     = len(SEEDS)

    dummy_X      = np.zeros(len(train))
    fold_indices = list(cv.split(dummy_X, train[TARGETS[0]].values, groups))

    print("label 피처 사전 계산 중 (lag1/2/7 + roll3/7/14)...")
    label_feat_test = build_label_features(train, test)

    fold_label_feats = []
    for fold_idx, (tr_idx, val_idx) in enumerate(fold_indices):
        train_fold = train.iloc[tr_idx].copy()
        val_fold   = train.iloc[val_idx].copy()
        lf_tr  = build_label_features(train_fold, train_fold)
        lf_val = build_label_features(val_fold,   val_fold)
        fold_label_feats.append((tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val))
    print(f"  완료 ({len(fold_indices)} folds)\n")

    lf_full    = build_label_features(train, train)
    X_full     = build_features(train, train, parquet_feat, lf_full, True, le)
    full_stats = compute_subj_stats(X_full["subject_id"], X_full, sensor_cols)

    oof_accum  = {t: np.zeros(len(train)) for t in TARGETS}
    test_accum = {t: np.zeros(len(test))  for t in TARGETS}

    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균F1   평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 16))

    for seed_i, seed in enumerate(SEEDS):
        params = {**LGBM_BASE_PARAMS, "random_state": seed}
        seed_oof  = {t: np.zeros(len(train)) for t in TARGETS}
        seed_test = {t: np.zeros(len(test))  for t in TARGETS}

        for t in TARGETS:
            y        = train[t].values
            drop_col = f"subj_mean_{t}"
            # lag7/roll14도 동일 타깃은 제거 (주간 패턴도 동일 타깃 누수 방지)
            lag_drop = [
                f"lag1_{t}", f"lag2_{t}", f"lag7_{t}",
                f"roll3_{t}", f"roll7_{t}", f"roll14_{t}",
            ]

            for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
                X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
                X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)

                sid_tr  = X_tr["subject_id"].reset_index(drop=True)
                sid_val = X_val["subject_id"].reset_index(drop=True)

                tr_stats  = compute_subj_stats(sid_tr,  X_tr,  sensor_cols)
                val_stats = compute_subj_stats(sid_val, X_val, sensor_cols)

                X_tr  = apply_zscore(sid_tr,  X_tr.drop(columns=["subject_id"]),  tr_stats,  sensor_cols)
                X_val = apply_zscore(sid_val, X_val.drop(columns=["subject_id"]), val_stats, sensor_cols)

                drop_cols = [drop_col] + [c for c in lag_drop if c in X_tr.columns]
                X_tr  = X_tr.drop(columns=drop_cols, errors="ignore")
                X_val = X_val.drop(columns=drop_cols, errors="ignore")

                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y[tr_idx])
                seed_oof[t][val_idx] = model.predict_proba(X_val)[:, 1]

                X_te = build_features(
                    test.copy(), train, parquet_feat, label_feat_test, False, le
                )
                sid_te = X_te["subject_id"].reset_index(drop=True)
                X_te   = apply_zscore(sid_te, X_te.drop(columns=["subject_id"]), full_stats, sensor_cols)
                X_te   = X_te.drop(columns=drop_cols, errors="ignore")
                seed_test[t] += model.predict_proba(X_te)[:, 1] / cv.n_splits

        for t in TARGETS:
            oof_accum[t]  += seed_oof[t]  / n_seeds
            test_accum[t] += seed_test[t] / n_seeds

        f1s, lls = [], []
        for t in TARGETS:
            cur_oof = oof_accum[t] * n_seeds / (seed_i + 1)
            f1s.append(f1_score(train[t].values, (cur_oof > 0.5).astype(int)))
            lls.append(log_loss(train[t].values, cur_oof))
        f1_str = "  ".join(f"{f:>6.3f}" for f in f1s)
        print(f"{seed:>6}  {f1_str}  {np.mean(f1s):>6.3f}  {np.mean(lls):>6.4f}")

    print()
    print(f"{'타깃':<5}  {'평균 F1':>8}  {'평균 LogLoss':>12}  {'기준 F1':>8}  {'차이':>6}")
    print("-" * 52)
    for t in TARGETS:
        mean_f1 = f1_score(train[t].values, (oof_accum[t] > 0.5).astype(int))
        mean_ll = log_loss(train[t].values, oof_accum[t])
        base    = BASELINE_F1[t]
        diff    = mean_f1 - base
        sign    = "+" if diff >= 0 else ""
        print(f"{t:<5}  {mean_f1:>8.3f}  {mean_ll:>12.4f}  {base:>8.3f}  {sign}{diff:>5.3f}")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = test_accum[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v2 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print(f"=== LightGBM lag7/roll14 확장 + 멀티 시드 ({len(SEEDS)}개) ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "lgbm_roll14lag7_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
