"""
HGB + ExtraTrees 앙상블 (교차 타깃 피처 추가)
- data/cross_target_features.csv 사전 생성 필요 (cross_target_features.py)
- 캐시된 hgb_v2 + extratrees_v2 params 로드
- 10 seeds 멀티 시드 앙상블, 동일 가중치 평균
출력: submission/hgb_et_xt_ensemble_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from optuna_params_io import load_params

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

HGB_KEY = "hgb_v2"
ET_KEY  = "extratrees_v2"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

BASELINE_F1 = {
    "Q1": 0.649, "Q2": 0.662, "Q3": 0.712,
    "S1": 0.788, "S2": 0.611, "S3": 0.526, "S4": 0.611,
}
# hgb_et_ensemble OOF 기준 (교차 타깃 피처 없는 버전)
HGB_ET_LL = {"Q1": 0.6466, "Q2": 0.6357, "Q3": 0.6175,
             "S1": 0.5994, "S2": 0.6066, "S3": 0.6060, "S4": 0.6641}

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


def build_features(df, ref, parquet_feat, label_feat, xt_feat, is_train, le):
    df = add_date_features(df)
    df = add_subject_mean_features(df, ref, is_train)
    df["subject_enc"] = df["subject_id"].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    df = df.merge(parquet_feat, left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")
    df = df.merge(label_feat, left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")
    df = df.merge(xt_feat, left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")

    base_cols    = (["subject_enc", "day_of_week", "month", "day_of_month",
                     "is_weekend", "week_of_year"]
                    + [f"subj_mean_{t}" for t in TARGETS])
    parquet_cols = [c for c in parquet_feat.columns if c not in ("subject_id", "date")]
    label_cols   = [c for c in label_feat.columns   if c not in ("subject_id", "date")]
    xt_cols      = [c for c in xt_feat.columns      if c not in ("subject_id", "date")]
    all_cols = ["subject_id"] + base_cols + parquet_cols + label_cols + xt_cols
    return df[[c for c in all_cols if c in df.columns]].reset_index(drop=True)


def precompute_fold_features(train, test, parquet_feat, le, fold_label_feats,
                             label_feat_test, xt_feat):
    fold_base = []
    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  xt_feat, True,  le)
        X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, xt_feat, False, le)
        fold_base.append((tr_idx, val_idx,
                          X_tr.drop(columns=["subject_id"]),
                          X_val.drop(columns=["subject_id"])))

    X_te = build_features(test.copy(), train, parquet_feat, label_feat_test, xt_feat, False, le)
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


def train_and_predict(train, test, parquet_feat, xt_feat):
    le     = LabelEncoder().fit(train["subject_id"])
    groups = train["subject_id"].values
    cv     = GroupKFold(n_splits=10)

    fold_indices = list(cv.split(np.zeros(len(train)), train[TARGETS[0]].values, groups))

    print("label 피처 사전 계산 중...")
    label_feat_test  = build_label_features(train, test)
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
        train, test, parquet_feat, le, fold_label_feats, label_feat_test, xt_feat,
    )
    print(f"  완료 (피처 수: {fold_by_target[TARGETS[0]][0][2].shape[1]}개)\n")

    cached_hgb = load_params(HGB_KEY)
    cached_et  = load_params(ET_KEY)
    if not cached_hgb:
        raise RuntimeError(f"캐시에 {HGB_KEY} params 없음. hgb_ensemble.py 먼저 실행하세요.")
    if not cached_et:
        raise RuntimeError(f"캐시에 {ET_KEY} params 없음. extratrees_ensemble.py 먼저 실행하세요.")

    print(f"=== Phase 1: 캐시 params 로드 ===")
    print(f"  HGB ({HGB_KEY}): Q1 lr={cached_hgb['Q1']['learning_rate']:.4f}, iter={cached_hgb['Q1']['max_iter']}")
    print(f"  ET  ({ET_KEY}):  Q1 n_est={cached_et['Q1']['n_estimators']}, depth={cached_et['Q1']['max_depth']}\n")

    n_seeds = len(SEEDS)
    print(f"=== Phase 2: HGB+ET+XT 앙상블 ({n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균F1   평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 16))

    ens_oof  = {t: np.zeros(len(train)) for t in TARGETS}
    ens_test = {t: np.zeros(len(test))  for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        seed_oof  = {t: np.zeros(len(train)) for t in TARGETS}
        seed_test = {t: np.zeros(len(test))  for t in TARGETS}

        for t in TARGETS:
            y = train[t].values
            hgb_params = {**cached_hgb[t], "random_state": seed}
            et_params  = {**cached_et[t],  "random_state": seed}

            for tr_idx, val_idx, X_tr, X_val in fold_by_target[t]:
                X_te = te_by_target[t]

                # HGB — NaN 자체 처리
                hgb = HistGradientBoostingClassifier(**hgb_params)
                hgb.fit(X_tr.to_numpy(), y[tr_idx])
                hgb_val  = hgb.predict_proba(X_val.to_numpy())[:, 1]
                hgb_test = hgb.predict_proba(X_te.to_numpy())[:, 1]

                # ET — fold median imputation
                tr_median = X_tr.median()
                et = ExtraTreesClassifier(**et_params)
                et.fit(X_tr.fillna(tr_median).to_numpy(), y[tr_idx])
                et_val  = et.predict_proba(X_val.fillna(tr_median).to_numpy())[:, 1]
                et_test = et.predict_proba(X_te.fillna(tr_median).to_numpy())[:, 1]

                seed_oof[t][val_idx] = (hgb_val + et_val) / 2
                seed_test[t]        += (hgb_test + et_test) / 2 / cv.n_splits

        for t in TARGETS:
            ens_oof[t]  += seed_oof[t]  / n_seeds
            ens_test[t] += seed_test[t] / n_seeds

        f1s, lls = [], []
        for t in TARGETS:
            cur = ens_oof[t] * n_seeds / (seed_i + 1)
            f1s.append(f1_score(train[t].values, (cur > 0.5).astype(int)))
            lls.append(log_loss(train[t].values, cur))
        f1_str = "  ".join(f"{f:>6.3f}" for f in f1s)
        print(f"{seed:>6}  {f1_str}  {np.mean(f1s):>6.3f}  {np.mean(lls):>6.4f}")

    print()
    print(f"{'타깃':<5}  {'앙상블 F1':>8}  {'앙상블 LL':>9}  {'기준(HGB+ET)':>12}  {'개선':>7}")
    print("-" * 56)
    for t in TARGETS:
        f1 = f1_score(train[t].values, (ens_oof[t] > 0.5).astype(int))
        ll = log_loss(train[t].values, ens_oof[t])
        diff = ll - HGB_ET_LL.get(t, 0)
        print(f"{t:<5}  {f1:>8.3f}  {ll:>9.4f}  {HGB_ET_LL.get(t,0):>12.4f}  {diff:>+7.4f}")
    avg_ll = np.mean([log_loss(train[t].values, ens_oof[t]) for t in TARGETS])
    avg_f1 = np.mean([f1_score(train[t].values, (ens_oof[t] > 0.5).astype(int)) for t in TARGETS])
    avg_base = np.mean(list(HGB_ET_LL.values()))
    print(f"{'평균':<5}  {avg_f1:>8.3f}  {avg_ll:>9.4f}  {avg_base:>12.4f}  {avg_ll-avg_base:>+7.4f}")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = ens_test[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    xt_path = DATA / "cross_target_features.csv"
    if not xt_path.exists():
        raise FileNotFoundError(
            f"교차 타깃 피처 파일 없음: {xt_path}\n"
            "먼저 uv run scripts/cross_target_features.py 를 실행하세요."
        )
    xt_feat = pd.read_csv(xt_path)
    print(f"교차 타깃 피처 로드: {xt_feat.shape[0]}행 × {xt_feat.shape[1]-2}개 피처")

    print("=== parquet 피처 v2 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== HGB + ExtraTrees 앙상블 (교차 타깃 피처 포함) ===\n")
    result = train_and_predict(train, sample, parquet_feat, xt_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== HGB+ET+XT 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "hgb_et_xt_ensemble_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
