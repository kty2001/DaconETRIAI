"""
LightGBM + Optuna + mUsageStats 피처
lgbm_optuna.py 기반에서 mUsageStats (앱 사용 통계) 13개 피처 추가
- usage_ms_{zone}: 시간대별 총 앱 사용 시간 (ms)
- usage_apps_{zone}: 시간대별 사용 앱 수
- usage_ms_total, usage_presleep_ratio, usage_sleep_ratio
출력: submission/lgbm_usage_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS  = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS    = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]
N_TRIALS = 30

BASELINE_F1 = {
    "Q1": 0.649, "Q2": 0.662, "Q3": 0.712,
    "S1": 0.788, "S2": 0.611, "S3": 0.526, "S4": 0.611,
}

SEARCH_SPACE = {
    "num_leaves":        ("int",   8,    64),
    "learning_rate":     ("float", 0.01, 0.3,  {"log": True}),
    "n_estimators":      ("int",   100,  600),
    "min_child_samples": ("int",   5,    50),
    "subsample":         ("float", 0.5,  1.0),
    "colsample_bytree":  ("float", 0.5,  1.0),
    "reg_alpha":         ("float", 1e-8, 10.0, {"log": True}),
    "reg_lambda":        ("float", 1e-8, 10.0, {"log": True}),
}

FIXED_PARAMS = {
    "objective": "binary",
    "metric":    "binary_logloss",
    "verbosity": -1,
    "random_state": 42,
}


# ── z-score 헬퍼 ──────────────────────────────────────────────────────────────

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


# ── 피처 빌더 ─────────────────────────────────────────────────────────────────

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


# ── 폴드 피처 사전 계산 ───────────────────────────────────────────────────────

def precompute_fold_features(train, test, parquet_feat, le, sensor_cols, fold_label_feats, label_feat_test, full_stats):
    """폴드별 z-scored 피처를 한 번만 계산. Optuna trial에서 재사용."""
    fold_base = []
    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
        X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)

        sid_tr  = X_tr["subject_id"].reset_index(drop=True)
        sid_val = X_val["subject_id"].reset_index(drop=True)

        tr_stats  = compute_subj_stats(sid_tr,  X_tr,  sensor_cols)
        val_stats = compute_subj_stats(sid_val, X_val, sensor_cols)

        X_tr_z  = apply_zscore(sid_tr,  X_tr.drop(columns=["subject_id"]),  tr_stats,  sensor_cols)
        X_val_z = apply_zscore(sid_val, X_val.drop(columns=["subject_id"]), val_stats, sensor_cols)

        fold_base.append((tr_idx, val_idx, X_tr_z, X_val_z))

    # 테스트 피처 (full_stats 기준)
    X_te_raw = build_features(test.copy(), train, parquet_feat, label_feat_test, False, le)
    sid_te   = X_te_raw["subject_id"].reset_index(drop=True)
    X_te_z   = apply_zscore(sid_te, X_te_raw.drop(columns=["subject_id"]), full_stats, sensor_cols)

    # 타깃별 subj_mean 제거 적용
    fold_by_target = {}
    te_by_target   = {}
    for t in TARGETS:
        drop_col = f"subj_mean_{t}"
        fold_by_target[t] = [
            (tr_idx, val_idx,
             X_tr_z.drop(columns=[drop_col], errors="ignore"),
             X_val_z.drop(columns=[drop_col], errors="ignore"))
            for tr_idx, val_idx, X_tr_z, X_val_z in fold_base
        ]
        te_by_target[t] = X_te_z.drop(columns=[drop_col], errors="ignore")

    return fold_by_target, te_by_target


# ── Optuna objective ──────────────────────────────────────────────────────────

def make_objective(t, fold_features_t, y):
    def objective(trial):
        params = {**FIXED_PARAMS}
        for name, spec in SEARCH_SPACE.items():
            kind = spec[0]
            kwargs = spec[3] if len(spec) > 3 else {}
            if kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2], **kwargs)
            else:
                params[name] = trial.suggest_float(name, spec[1], spec[2], **kwargs)

        oof = np.zeros(len(y))
        for tr_idx, val_idx, X_tr, X_val in fold_features_t:
            model = lgb.LGBMClassifier(**params)
            model.fit(X_tr, y[tr_idx])
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
        return log_loss(y, oof)

    return objective


# ── 메인 학습 ─────────────────────────────────────────────────────────────────

def train_and_predict(train, test, parquet_feat):
    le          = LabelEncoder().fit(train["subject_id"])
    groups      = train["subject_id"].values
    cv          = GroupKFold(n_splits=10)
    sensor_cols = get_sensor_cols(parquet_feat)
    n_seeds     = len(SEEDS)

    dummy_X      = np.zeros(len(train))
    fold_indices = list(cv.split(dummy_X, train[TARGETS[0]].values, groups))

    print("label 피처 사전 계산 중...")
    label_feat_test = build_label_features(train, test)
    fold_label_feats = []
    for tr_idx, val_idx in fold_indices:
        train_fold = train.iloc[tr_idx].copy()
        val_fold   = train.iloc[val_idx].copy()
        lf_tr  = build_label_features(train_fold, train_fold)
        lf_val = build_label_features(val_fold,   val_fold)
        fold_label_feats.append((tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val))
    print(f"  완료 ({len(fold_indices)} folds)")

    lf_full    = build_label_features(train, train)
    X_full     = build_features(train, train, parquet_feat, lf_full, True, le)
    full_stats = compute_subj_stats(X_full["subject_id"], X_full, sensor_cols)

    print("폴드 피처 사전 계산 중...")
    fold_by_target, te_by_target = precompute_fold_features(
        train, test, parquet_feat, le, sensor_cols,
        fold_label_feats, label_feat_test, full_stats,
    )
    print("  완료\n")

    # ── Phase 1: Optuna 탐색 ─────────────────────────────────────────────────
    print(f"=== Phase 1: Optuna 탐색 (타깃별 {N_TRIALS} trials) ===")
    best_params = {}

    for t in TARGETS:
        y = train[t].values
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(t, fold_by_target[t], y), n_trials=N_TRIALS, show_progress_bar=False)

        bp = {**FIXED_PARAMS, **study.best_params}
        best_params[t] = bp
        print(f"  {t}: LogLoss={study.best_value:.4f} | "
              f"leaves={bp['num_leaves']}, lr={bp['learning_rate']:.4f}, "
              f"n_est={bp['n_estimators']}, child={bp['min_child_samples']}")

    print()

    # ── Phase 2: best_params + 멀티 시드로 최종 예측 ────────────────────────
    print(f"=== Phase 2: 최종 예측 (best_params + {n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균F1   평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 16))

    oof_accum  = {t: np.zeros(len(train)) for t in TARGETS}
    test_accum = {t: np.zeros(len(test))  for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        seed_oof  = {t: np.zeros(len(train)) for t in TARGETS}
        seed_test = {t: np.zeros(len(test))  for t in TARGETS}

        for t in TARGETS:
            y      = train[t].values
            params = {**best_params[t], "random_state": seed,
                      "subsample": best_params[t].get("subsample", 0.8),
                      "colsample_bytree": best_params[t].get("colsample_bytree", 0.8)}

            for tr_idx, val_idx, X_tr, X_val in fold_by_target[t]:
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y[tr_idx])
                seed_oof[t][val_idx] = model.predict_proba(X_val)[:, 1]
                seed_test[t] += model.predict_proba(te_by_target[t])[:, 1] / cv.n_splits

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

    print(f"=== LightGBM + mUsageStats + Optuna ({N_TRIALS} trials/target) + 멀티 시드 ({len(SEEDS)}개) ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "lgbm_usage_zscore_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
