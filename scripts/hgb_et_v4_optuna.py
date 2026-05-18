"""
HGB + ExtraTrees Optuna 재튜닝 — parquet v3 피처 기준
- Phase 1: HGB Optuna (50 trials/target) → hgb_v3 저장
- Phase 2: ET  Optuna (50 trials/target) → extratrees_v3 저장
- Phase 3: HGB+ET 앙상블 (10 seeds) → OOF 비교 출력
출력: submission/hgb_et_v4_optuna_prob.csv (clip 0.1~0.9)

v2 기준선: HGB+ET OOF 0.6434
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
import optuna
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v3 import build_all as build_parquet_features
from optuna_params_io import load_params, save_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

HGB_KEY  = "hgb_v3"
ET_KEY   = "extratrees_v3"
N_TRIALS = 50

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

# v2 앙상블 기준선 (비교용)
ENS_LL_V2 = {
    "Q1": 0.6974, "Q2": 0.6509, "Q3": 0.6330,
    "S1": 0.6109, "S2": 0.6146, "S3": 0.6149, "S4": 0.6823,
}

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]

HGB_SEARCH = {
    "learning_rate":     ("float", 0.01, 0.3,  {"log": True}),
    "max_iter":          ("int",   100,  800),
    "max_leaf_nodes":    ("int",   15,   80),
    "min_samples_leaf":  ("int",   10,   60),
    "l2_regularization": ("float", 0.0,  10.0),
    "max_features":      ("float", 0.1,  1.0),
}
HGB_FIXED = {"random_state": 42, "early_stopping": False}

ET_SEARCH = {
    "n_estimators":      ("int",   100,  600),
    "max_depth":         ("int",   3,    20),
    "min_samples_split": ("int",   2,    20),
    "min_samples_leaf":  ("int",   1,    10),
    "max_features":      ("float", 0.05, 0.5),
}
ET_FIXED = {"criterion": "entropy", "n_jobs": -1, "random_state": 42}


# ── 피처 빌드 ─────────────────────────────────────────────────────────────────
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
    df = df.merge(parquet_feat,
                  left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"],
                  how="left").drop(columns=["date"], errors="ignore")
    df = df.merge(label_feat,
                  left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"],
                  how="left").drop(columns=["date"], errors="ignore")
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


# ── Optuna 목적함수 ────────────────────────────────────────────────────────────
def make_hgb_objective(fold_features_t, y):
    def objective(trial):
        params = {**HGB_FIXED}
        for name, spec in HGB_SEARCH.items():
            kind   = spec[0]
            kwargs = spec[3] if len(spec) > 3 else {}
            if kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2], **kwargs)
            elif kind == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2], **kwargs)
        oof = np.zeros(len(y))
        for tr_idx, val_idx, X_tr, X_val in fold_features_t:
            model = HistGradientBoostingClassifier(**params)
            model.fit(X_tr.to_numpy(), y[tr_idx])
            oof[val_idx] = model.predict_proba(X_val.to_numpy())[:, 1]
        return log_loss(y, oof)
    return objective


def make_et_objective(fold_features_t, y):
    def objective(trial):
        params = {**ET_FIXED}
        for name, spec in ET_SEARCH.items():
            kind   = spec[0]
            kwargs = spec[3] if len(spec) > 3 else {}
            if kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2], **kwargs)
            elif kind == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2], **kwargs)
        oof = np.zeros(len(y))
        for tr_idx, val_idx, X_tr, X_val in fold_features_t:
            tr_med = X_tr.median()
            model  = ExtraTreesClassifier(**params)
            model.fit(X_tr.fillna(tr_med).to_numpy(), y[tr_idx])
            oof[val_idx] = model.predict_proba(X_val.fillna(tr_med).to_numpy())[:, 1]
        return log_loss(y, oof)
    return objective


# ── 메인 ─────────────────────────────────────────────────────────────────────
def train_and_predict(train, test, parquet_feat):
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
        train, test, parquet_feat, le, fold_label_feats, label_feat_test,
    )
    n_feat = fold_by_target[TARGETS[0]][0][2].shape[1]
    print(f"  완료 (피처 수: {n_feat}개)\n")

    # ── Phase 1: HGB Optuna ────────────────────────────────────────────────────
    cached_hgb = load_params(HGB_KEY)
    if cached_hgb:
        print(f"=== Phase 1: HGB 캐시 로드 ({HGB_KEY}) ===")
        best_hgb = {t: cached_hgb[t] for t in TARGETS}
        for t in TARGETS:
            bp = best_hgb[t]
            print(f"  {t}: lr={bp['learning_rate']:.4f}, iter={bp['max_iter']}, "
                  f"leaves={bp['max_leaf_nodes']}")
    else:
        print(f"=== Phase 1: HGB Optuna ({N_TRIALS} trials/target) ===")
        best_hgb = {}
        for t in TARGETS:
            y = train[t].values
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(make_hgb_objective(fold_by_target[t], y),
                           n_trials=N_TRIALS, show_progress_bar=True)
            bp = {**HGB_FIXED, **study.best_params}
            best_hgb[t] = bp
            print(f"  {t}: LL={study.best_value:.4f} | "
                  f"lr={bp['learning_rate']:.4f}, iter={bp['max_iter']}, "
                  f"leaves={bp['max_leaf_nodes']}, min_samp={bp['min_samples_leaf']}")
        print()
        save_params(HGB_KEY, best_hgb)
        print(f"  → {HGB_KEY} 저장 완료\n")

    # ── Phase 2: ET Optuna ─────────────────────────────────────────────────────
    cached_et = load_params(ET_KEY)
    if cached_et:
        print(f"=== Phase 2: ET 캐시 로드 ({ET_KEY}) ===")
        best_et = {t: cached_et[t] for t in TARGETS}
        for t in TARGETS:
            bp = best_et[t]
            print(f"  {t}: n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
                  f"max_feat={bp['max_features']:.3f}")
    else:
        print(f"=== Phase 2: ET Optuna ({N_TRIALS} trials/target) ===")
        best_et = {}
        for t in TARGETS:
            y = train[t].values
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=42),
            )
            study.optimize(make_et_objective(fold_by_target[t], y),
                           n_trials=N_TRIALS, show_progress_bar=True)
            bp = {**ET_FIXED, **study.best_params}
            best_et[t] = bp
            print(f"  {t}: LL={study.best_value:.4f} | "
                  f"n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
                  f"max_feat={bp['max_features']:.3f}")
        print()
        save_params(ET_KEY, best_et)
        print(f"  → {ET_KEY} 저장 완료\n")

    # ── Phase 3: HGB+ET 앙상블 ────────────────────────────────────────────────
    n_seeds = len(SEEDS)
    print(f"=== Phase 3: HGB+ET 앙상블 ({n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균F1   평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 16))

    ens_oof  = {t: np.zeros(len(train)) for t in TARGETS}
    ens_test = {t: np.zeros(len(test))  for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        seed_oof  = {t: np.zeros(len(train)) for t in TARGETS}
        seed_test = {t: np.zeros(len(test))  for t in TARGETS}

        for t in TARGETS:
            y          = train[t].values
            hgb_params = {**best_hgb[t], "random_state": seed}
            et_params  = {**best_et[t],  "random_state": seed}

            for tr_idx, val_idx, X_tr, X_val in fold_by_target[t]:
                X_te = te_by_target[t]

                hgb = HistGradientBoostingClassifier(**hgb_params)
                hgb.fit(X_tr.to_numpy(), y[tr_idx])
                hgb_val  = hgb.predict_proba(X_val.to_numpy())[:, 1]
                hgb_test = hgb.predict_proba(X_te.to_numpy())[:, 1]

                tr_med = X_tr.median()
                et = ExtraTreesClassifier(**et_params)
                et.fit(X_tr.fillna(tr_med).to_numpy(), y[tr_idx])
                et_val  = et.predict_proba(X_val.fillna(tr_med).to_numpy())[:, 1]
                et_test = et.predict_proba(X_te.fillna(tr_med).to_numpy())[:, 1]

                seed_oof[t][val_idx]  = (hgb_val + et_val) / 2
                seed_test[t]         += (hgb_test + et_test) / 2 / cv.n_splits

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

    # ── 결과 요약 ─────────────────────────────────────────────────────────────
    print()
    print(f"{'타깃':<5}  {'v4opt F1':>8}  {'v4opt LL':>9}  {'v2 LL':>7}  {'차이':>6}")
    print("-" * 46)
    for t in TARGETS:
        f1   = f1_score(train[t].values, (ens_oof[t] > 0.5).astype(int))
        ll   = log_loss(train[t].values, ens_oof[t])
        diff = ll - ENS_LL_V2[t]
        sign = "+" if diff > 0 else ""
        print(f"{t:<5}  {f1:>8.3f}  {ll:>9.4f}  {ENS_LL_V2[t]:>7.4f}  {sign}{diff:>5.4f}")
    avg_ll   = np.mean([log_loss(train[t].values, ens_oof[t]) for t in TARGETS])
    avg_f1   = np.mean([f1_score(train[t].values, (ens_oof[t] > 0.5).astype(int))
                        for t in TARGETS])
    avg_diff = avg_ll - 0.6434
    sign = "+" if avg_diff > 0 else ""
    print(f"{'평균':<5}  {avg_f1:>8.3f}  {avg_ll:>9.4f}  {'0.6434':>7}  {sign}{avg_diff:>5.4f}")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = ens_test[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v3 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== HGB+ET Optuna 재튜닝 (parquet v3) ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "hgb_et_v4_optuna_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
