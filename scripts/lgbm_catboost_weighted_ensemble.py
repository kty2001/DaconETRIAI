"""
LightGBM + CatBoost OOF 기반 가중 앙상블
- lgbm_v3 / catboost_v3 캐시 params 로드 (Optuna 생략)
- 10 seeds 예측으로 lgbm_oof / cat_oof 축적
- Phase 3: 타깃별 최적 α (LGBM 비중) OOF LogLoss 기준 그리드 탐색
- 균등 50:50 대신 타깃별 최적 비율 적용
출력: submission/lgbm_catboost_weighted_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from optuna_params_io import load_params, save_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

LGBM_KEY = "lgbm_v2"
CAT_KEY  = "catboost_v2"

TARGETS  = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS    = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]
N_TRIALS = 30

BASELINE_F1 = {
    "Q1": 0.649, "Q2": 0.662, "Q3": 0.712,
    "S1": 0.788, "S2": 0.611, "S3": 0.526, "S4": 0.611,
}

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]

LGBM_SEARCH = {
    "num_leaves":        ("int",   8,    64),
    "learning_rate":     ("float", 0.01, 0.3,  {"log": True}),
    "n_estimators":      ("int",   100,  600),
    "min_child_samples": ("int",   5,    50),
    "subsample":         ("float", 0.5,  1.0),
    "colsample_bytree":  ("float", 0.5,  1.0),
    "reg_alpha":         ("float", 1e-8, 10.0, {"log": True}),
    "reg_lambda":        ("float", 1e-8, 10.0, {"log": True}),
}
LGBM_FIXED = {
    "objective": "binary", "metric": "binary_logloss",
    "verbosity": -1, "random_state": 42,
}

CAT_SEARCH = {
    "iterations":          ("int",   100,  500),
    "depth":               ("int",   3,    8),
    "learning_rate":       ("float", 0.01, 0.3,  {"log": True}),
    "l2_leaf_reg":         ("float", 1e-8, 10.0, {"log": True}),
    "random_strength":     ("float", 1e-8, 10.0, {"log": True}),
    "bagging_temperature": ("float", 0.0,  1.0),
}
CAT_FIXED = {
    "loss_function": "Logloss", "eval_metric": "Logloss",
    "verbose": 0, "random_seed": 42, "task_type": "CPU",
}


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


def precompute_fold_features(train, test, parquet_feat, le, sensor_cols,
                             fold_label_feats, label_feat_test, full_stats):
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

    X_te_raw = build_features(test.copy(), train, parquet_feat, label_feat_test, False, le)
    sid_te   = X_te_raw["subject_id"].reset_index(drop=True)
    X_te_z   = apply_zscore(sid_te, X_te_raw.drop(columns=["subject_id"]), full_stats, sensor_cols)

    fold_by_target = {}
    te_by_target   = {}
    for t in TARGETS:
        drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
        fold_by_target[t] = [
            (tr_idx, val_idx,
             X_tr_z.drop(columns=drop_cols, errors="ignore"),
             X_val_z.drop(columns=drop_cols, errors="ignore"))
            for tr_idx, val_idx, X_tr_z, X_val_z in fold_base
        ]
        te_by_target[t] = X_te_z.drop(columns=drop_cols, errors="ignore")

    return fold_by_target, te_by_target


def make_objective(model_type, search_space, fixed_params, fold_features_t, y):
    def objective(trial):
        params = {**fixed_params}
        for name, spec in search_space.items():
            kind   = spec[0]
            kwargs = spec[3] if len(spec) > 3 else {}
            if kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2], **kwargs)
            else:
                params[name] = trial.suggest_float(name, spec[1], spec[2], **kwargs)

        oof = np.zeros(len(y))
        for tr_idx, val_idx, X_tr, X_val in fold_features_t:
            if model_type == "lgbm":
                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y[tr_idx])
            else:
                seed = params.pop("random_seed", 42)
                model = CatBoostClassifier(**params, random_seed=seed)
                model.fit(X_tr, y[tr_idx])
                params["random_seed"] = seed
            oof[val_idx] = model.predict_proba(X_val)[:, 1]
        return log_loss(y, oof)

    return objective


def find_best_alpha(lgbm_oof, cat_oof, y):
    """OOF LogLoss 기준 최적 LGBM 비중(α) 그리드 탐색"""
    alphas = np.linspace(0, 1, 21)
    best_ll, best_a = np.inf, 0.5
    for a in alphas:
        ll = log_loss(y, a * lgbm_oof + (1 - a) * cat_oof)
        if ll < best_ll:
            best_ll, best_a = ll, a
    return best_a, best_ll


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
    print("  완료")

    lf_full    = build_label_features(train, train)
    X_full     = build_features(train, train, parquet_feat, lf_full, True, le)
    full_stats = compute_subj_stats(X_full["subject_id"], X_full, sensor_cols)

    print("폴드 피처 사전 계산 중...")
    fold_by_target, te_by_target = precompute_fold_features(
        train, test, parquet_feat, le, sensor_cols,
        fold_label_feats, label_feat_test, full_stats,
    )
    print(f"  완료 (피처 수: {fold_by_target[TARGETS[0]][0][2].shape[1]}개)\n")

    # ── Phase 1: 캐시 로드 (Optuna 생략) ─────────────────────────────────────
    cached_lgbm = load_params(LGBM_KEY)
    cached_cat  = load_params(CAT_KEY)

    if cached_lgbm:
        print(f"=== Phase 1-A: 저장된 params 로드 ({LGBM_KEY}) ===")
        best_lgbm = {t: cached_lgbm[t] for t in TARGETS}
        for t in TARGETS:
            bp = best_lgbm[t]
            print(f"  {t}: leaves={bp['num_leaves']}, lr={bp['learning_rate']:.4f}, n_est={bp['n_estimators']}")
        print()
    else:
        print(f"=== Phase 1-A: LGBM Optuna ({N_TRIALS} trials/target) ===")
        best_lgbm = {}
        for t in TARGETS:
            y = train[t].values
            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(make_objective("lgbm", LGBM_SEARCH, LGBM_FIXED,
                                          fold_by_target[t], y),
                           n_trials=N_TRIALS, show_progress_bar=False)
            bp = {**LGBM_FIXED, **study.best_params}
            best_lgbm[t] = bp
            print(f"  {t}: LogLoss={study.best_value:.4f} | "
                  f"leaves={bp['num_leaves']}, lr={bp['learning_rate']:.4f}, "
                  f"n_est={bp['n_estimators']}")
        print()
        save_params(LGBM_KEY, best_lgbm)

    if cached_cat:
        print(f"=== Phase 1-B: 저장된 params 로드 ({CAT_KEY}) ===")
        best_cat = {t: cached_cat[t] for t in TARGETS}
        for t in TARGETS:
            bp = best_cat[t]
            print(f"  {t}: depth={bp['depth']}, lr={bp['learning_rate']:.4f}, iter={bp['iterations']}")
        print()
    else:
        print(f"=== Phase 1-B: CatBoost Optuna ({N_TRIALS} trials/target) ===")
        best_cat = {}
        for t in TARGETS:
            y = train[t].values
            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(make_objective("catboost", CAT_SEARCH, CAT_FIXED,
                                          fold_by_target[t], y),
                           n_trials=N_TRIALS, show_progress_bar=False)
            bp = {**CAT_FIXED, **study.best_params}
            best_cat[t] = bp
            print(f"  {t}: LogLoss={study.best_value:.4f} | "
                  f"depth={bp['depth']}, lr={bp['learning_rate']:.4f}, "
                  f"iter={bp['iterations']}")
        print()
        save_params(CAT_KEY, best_cat)

    # ── Phase 2: 멀티 시드 OOF 축적 ──────────────────────────────────────────
    print(f"=== Phase 2: 멀티 시드 OOF 축적 ({n_seeds} seeds) ===")

    lgbm_oof  = {t: np.zeros(len(train)) for t in TARGETS}
    lgbm_test = {t: np.zeros(len(test))  for t in TARGETS}
    cat_oof   = {t: np.zeros(len(train)) for t in TARGETS}
    cat_test  = {t: np.zeros(len(test))  for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        for t in TARGETS:
            y = train[t].values
            lgbm_params = {**best_lgbm[t], "random_state": seed}
            cat_params  = {**best_cat[t],  "random_seed":  seed}

            seed_lgbm_oof  = np.zeros(len(train))
            seed_cat_oof   = np.zeros(len(train))
            seed_lgbm_test = np.zeros(len(test))
            seed_cat_test  = np.zeros(len(test))

            for tr_idx, val_idx, X_tr, X_val in fold_by_target[t]:
                lm = lgb.LGBMClassifier(**lgbm_params)
                lm.fit(X_tr, y[tr_idx])
                seed_lgbm_oof[val_idx]  = lm.predict_proba(X_val)[:, 1]
                seed_lgbm_test         += lm.predict_proba(te_by_target[t])[:, 1] / cv.n_splits

                cm = CatBoostClassifier(**cat_params)
                cm.fit(X_tr, y[tr_idx])
                seed_cat_oof[val_idx]  = cm.predict_proba(X_val)[:, 1]
                seed_cat_test         += cm.predict_proba(te_by_target[t])[:, 1] / cv.n_splits

            lgbm_oof[t]  += seed_lgbm_oof  / n_seeds
            lgbm_test[t] += seed_lgbm_test / n_seeds
            cat_oof[t]   += seed_cat_oof   / n_seeds
            cat_test[t]  += seed_cat_test  / n_seeds

        print(f"  seed {seed} 완료 ({seed_i+1}/{n_seeds})")

    # ── Phase 3: 타깃별 최적 α 탐색 ──────────────────────────────────────────
    print(f"\n=== Phase 3: 타깃별 최적 α 탐색 (LGBM 비중, 0~1 그리드 21점) ===")
    print(f"{'타깃':<5}  {'최적 α':>6}  {'LGBM%':>6}  {'CAT%':>5}  {'가중 LL':>8}  {'균등 LL':>8}  {'개선':>6}")
    print("-" * 58)

    best_alpha = {}
    for t in TARGETS:
        y = train[t].values
        a, weighted_ll = find_best_alpha(lgbm_oof[t], cat_oof[t], y)
        equal_ll = log_loss(y, 0.5 * lgbm_oof[t] + 0.5 * cat_oof[t])
        best_alpha[t] = a
        print(f"{t:<5}  {a:>6.2f}  {a*100:>5.0f}%  {(1-a)*100:>4.0f}%  "
              f"{weighted_ll:>8.4f}  {equal_ll:>8.4f}  {equal_ll-weighted_ll:>+6.4f}")

    avg_weighted = np.mean([
        log_loss(train[t].values, best_alpha[t]*lgbm_oof[t] + (1-best_alpha[t])*cat_oof[t])
        for t in TARGETS
    ])
    avg_equal = np.mean([
        log_loss(train[t].values, 0.5*lgbm_oof[t] + 0.5*cat_oof[t])
        for t in TARGETS
    ])
    print(f"\n  균등 평균 OOF: {avg_equal:.4f}  →  가중 앙상블 OOF: {avg_weighted:.4f}  "
          f"(개선: {avg_equal-avg_weighted:+.4f})")

    # ── 최종 예측 ─────────────────────────────────────────────────────────────
    ens_test = {
        t: best_alpha[t] * lgbm_test[t] + (1 - best_alpha[t]) * cat_test[t]
        for t in TARGETS
    }

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = ens_test[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v2 집계 중 (mUsageStats 제거) ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== LGBM + CatBoost OOF 기반 가중 앙상블 ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 가중 앙상블 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "lgbm_catboost_weighted_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
