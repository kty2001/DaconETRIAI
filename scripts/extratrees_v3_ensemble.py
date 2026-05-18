"""
ExtraTrees v3 단독 앙상블 (z-score 없음)
- parquet v3 피처 (117컬럼, v2 대비 +34)
- extratrees_v3 캐시 params 로드 (hgb_et_v4_optuna.py에서 탐색 완료)
- 10 seeds 멀티 시드 앙상블, NaN → fold별 train median imputation
- v2 기준(OOF 0.6462) 대비 성능 비교
출력: submission/extratrees_v3_ensemble_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v3 import build_all as build_parquet_features
from optuna_params_io import load_params

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

ET_KEY  = "extratrees_v3"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

# ET v2 OOF LL 비교 기준
ET_LL_V2 = {
    "Q1": 0.6998, "Q2": 0.6478, "Q3": 0.6443,
    "S1": 0.6161, "S2": 0.6161, "S3": 0.6124, "S4": 0.6870,
}

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]


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


def precompute_fold_features(train, test, parquet_feat, le, fold_label_feats, label_feat_test):
    fold_base = []
    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
        X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)
        fold_base.append((tr_idx, val_idx,
                          X_tr.drop(columns=["subject_id"]),
                          X_val.drop(columns=["subject_id"])))

    X_te = build_features(test.copy(), train, parquet_feat, label_feat_test, False, le)
    X_te = X_te.drop(columns=["subject_id"])

    fold_by_target = {}
    te_by_target   = {}
    for t in TARGETS:
        drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
        fold_by_target[t] = [
            (tr_idx, val_idx,
             X_tr.drop(columns=drop_cols, errors="ignore"),
             X_val.drop(columns=drop_cols, errors="ignore"))
            for tr_idx, val_idx, X_tr, X_val in fold_base
        ]
        te_by_target[t] = X_te.drop(columns=drop_cols, errors="ignore")

    return fold_by_target, te_by_target


def train_and_predict(train, test, parquet_feat):
    le      = LabelEncoder().fit(train["subject_id"])
    groups  = train["subject_id"].values
    cv      = GroupKFold(n_splits=10)
    n_seeds = len(SEEDS)

    fold_indices = list(cv.split(np.zeros(len(train)), train[TARGETS[0]].values, groups))

    print("label 피처 사전 계산 중...")
    label_feat_test = build_label_features(train, test)
    fold_label_feats = []
    for tr_idx, val_idx in fold_indices:
        train_fold = train.iloc[tr_idx].copy()
        val_fold   = train.iloc[val_idx].copy()
        lf_tr  = build_label_features(train_fold, train_fold)
        lf_val = build_label_features(val_fold,   val_fold)
        fold_label_feats.append((tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val))
    print("  완료")

    print("폴드 피처 사전 계산 중...")
    fold_by_target, te_by_target = precompute_fold_features(
        train, test, parquet_feat, le, fold_label_feats, label_feat_test,
    )
    n_feat = fold_by_target[TARGETS[0]][0][2].shape[1]
    print(f"  완료 (피처 수: {n_feat}개, v2 대비 +{n_feat - 149}개)\n")

    cached_et = load_params(ET_KEY)
    if not cached_et:
        raise RuntimeError(
            f"캐시에 {ET_KEY} params 없음. hgb_et_v4_optuna.py 먼저 실행하세요."
        )

    print(f"=== Phase 1: 캐시 params 로드 ({ET_KEY}) ===")
    for t in TARGETS:
        bp = cached_et[t]
        print(f"  {t}: n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
              f"max_feat={bp['max_features']:.3f}")
    print()

    print(f"=== Phase 2: ET v3 단독 앙상블 ({n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균F1   평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 16))

    et_oof  = {t: np.zeros(len(train)) for t in TARGETS}
    et_test = {t: np.zeros(len(test))  for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        seed_oof  = {t: np.zeros(len(train)) for t in TARGETS}
        seed_test = {t: np.zeros(len(test))  for t in TARGETS}

        for t in TARGETS:
            y = train[t].values
            et_params = {**cached_et[t], "random_state": seed}

            for tr_idx, val_idx, X_tr, X_val in fold_by_target[t]:
                tr_median = X_tr.median()
                et = ExtraTreesClassifier(**et_params)
                et.fit(X_tr.fillna(tr_median), y[tr_idx])
                seed_oof[t][val_idx]  = et.predict_proba(X_val.fillna(tr_median))[:, 1]
                seed_test[t]         += et.predict_proba(te_by_target[t].fillna(tr_median))[:, 1] / cv.n_splits

        for t in TARGETS:
            et_oof[t]  += seed_oof[t]  / n_seeds
            et_test[t] += seed_test[t] / n_seeds

        f1s, lls = [], []
        for t in TARGETS:
            cur = et_oof[t] * n_seeds / (seed_i + 1)
            f1s.append(f1_score(train[t].values, (cur > 0.5).astype(int)))
            lls.append(log_loss(train[t].values, cur))
        f1_str = "  ".join(f"{f:>6.3f}" for f in f1s)
        print(f"{seed:>6}  {f1_str}  {np.mean(f1s):>6.3f}  {np.mean(lls):>6.4f}")

    print()
    print(f"{'타깃':<5}  {'v3 F1':>6}  {'v3 LL':>7}  {'v2 LL':>7}  {'차이':>6}")
    print("-" * 42)
    for t in TARGETS:
        f1   = f1_score(train[t].values, (et_oof[t] > 0.5).astype(int))
        ll   = log_loss(train[t].values, et_oof[t])
        diff = ll - ET_LL_V2[t]
        sign = "+" if diff > 0 else ""
        print(f"{t:<5}  {f1:>6.3f}  {ll:>7.4f}  {ET_LL_V2[t]:>7.4f}  {sign}{diff:>5.4f}")

    avg_ll_v3 = np.mean([log_loss(train[t].values, et_oof[t]) for t in TARGETS])
    avg_f1_v3 = np.mean([f1_score(train[t].values, (et_oof[t] > 0.5).astype(int)) for t in TARGETS])
    avg_diff  = avg_ll_v3 - 0.6462
    sign = "+" if avg_diff > 0 else ""
    print(f"{'평균':<5}  {avg_f1_v3:>6.3f}  {avg_ll_v3:>7.4f}  {'0.6462':>7}  {sign}{avg_diff:>5.4f}")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = et_test[t]
    return result, et_oof


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v3 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== ExtraTrees v3 단독 앙상블 ===\n")
    result, et_oof = train_and_predict(train, sample, parquet_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== ET v3 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "extratrees_v3_ensemble_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
