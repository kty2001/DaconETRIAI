"""
ExtraTrees + GPS Slim 80% 앙상블
- parquet v2 + GPS 피처, Feature Importance 상위 85% 커버 피처만 유지
- Phase 0: 중요도 계산 (extratrees_gps params 활용)
- Phase 1: Optuna 100 trials (extratrees_gps_slim80)
- Phase 2: 10 seeds 앙상블
출력: submission/extratrees_gps_slim80_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import optuna
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder
from functools import reduce

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from gps_features import build_gps
from optuna_params_io import load_params, save_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

ET_KEY_GPS  = "extratrees_gps"
ET_KEY_SLIM = "extratrees_gps_slim80"
N_TRIALS    = 100
IMP_COVERAGE = 0.80

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

ET_LL_GPS_SLIM = {
    "Q1": 0.6986, "Q2": 0.6446, "Q3": 0.6441,
    "S1": 0.6076, "S2": 0.6085, "S3": 0.6032, "S4": 0.6888,
}

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]

ET_SEARCH = {
    "n_estimators":      ("int",   200,  1500, {"step": 50}),
    "max_depth":         ("int",   3,    25),
    "min_samples_split": ("int",   2,    20),
    "min_samples_leaf":  ("int",   1,    10),
    "max_features":      ("float", 0.05, 0.5),
}
ET_FIXED = {
    "criterion":    "entropy",
    "n_jobs":       -1,
    "random_state": 42,
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


def compute_importance(train, parquet_feat, le, sensor_cols, fold_label_feats, best_gps):
    importance_dict = {t: [] for t in TARGETS}
    feature_names_ref = {}

    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr, True, le)
        sid_tr = X_tr["subject_id"].reset_index(drop=True)
        tr_stats = compute_subj_stats(sid_tr, X_tr, sensor_cols)
        X_tr_z  = apply_zscore(sid_tr, X_tr.drop(columns=["subject_id"]), tr_stats, sensor_cols)

        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            X_tr_t = X_tr_z.drop(columns=drop_cols, errors="ignore")
            if t not in feature_names_ref:
                feature_names_ref[t] = X_tr_t.columns.tolist()
            tr_median   = X_tr_t.median()
            X_tr_filled = X_tr_t.fillna(tr_median)
            y = train[t].values
            model = ExtraTreesClassifier(**{**best_gps[t], "random_state": 42})
            model.fit(X_tr_filled, y[tr_idx])
            importance_dict[t].append(model.feature_importances_)

    all_imp = {t: pd.Series(np.mean(importance_dict[t], axis=0),
                            index=feature_names_ref[t])
               for t in TARGETS}
    common_feats = list(reduce(lambda a, b: a & b,
                               [set(all_imp[t].index) for t in TARGETS]))
    combined = pd.DataFrame({t: all_imp[t][common_feats] for t in TARGETS})
    combined["mean_imp"] = combined.mean(axis=1)
    combined = combined.sort_values("mean_imp", ascending=False)
    total = combined["mean_imp"].sum()
    combined["cumsum"] = combined["mean_imp"].cumsum() / total
    keep = combined[combined["cumsum"] <= IMP_COVERAGE].index.tolist()

    print(f"  전체 공통 피처: {len(combined)}개")
    print(f"  상위 {IMP_COVERAGE*100:.0f}% 커버: {len(keep)}개 유지 / {len(combined)-len(keep)}개 제거")
    return set(keep)


def precompute_fold_features(train, test, parquet_feat, le, sensor_cols,
                             fold_label_feats, label_feat_test, full_stats, keep_feats):
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
        first_tr  = fold_base[0][2].drop(columns=drop_cols, errors="ignore")
        slim_cols = [c for c in first_tr.columns if c in keep_feats]

        fold_by_target[t] = []
        for tr_idx, val_idx, X_tr_z, X_val_z in fold_base:
            X_tr_t  = X_tr_z.drop(columns=drop_cols, errors="ignore")
            X_val_t = X_val_z.drop(columns=drop_cols, errors="ignore")
            fold_by_target[t].append(
                (tr_idx, val_idx,
                 X_tr_t[[c for c in slim_cols if c in X_tr_t.columns]],
                 X_val_t[[c for c in slim_cols if c in X_val_t.columns]])
            )
        X_te_t = X_te_z.drop(columns=drop_cols, errors="ignore")
        te_by_target[t] = X_te_t[[c for c in slim_cols if c in X_te_t.columns]]

    return fold_by_target, te_by_target


def make_et_objective(search_space, fixed_params, fold_features_t, y):
    def objective(trial):
        params = {**fixed_params}
        for name, spec in search_space.items():
            kind   = spec[0]
            kwargs = spec[3] if len(spec) > 3 else {}
            if kind == "int":
                params[name] = trial.suggest_int(name, spec[1], spec[2], **kwargs)
            elif kind == "float":
                params[name] = trial.suggest_float(name, spec[1], spec[2], **kwargs)
            elif kind == "categorical":
                params[name] = trial.suggest_categorical(name, spec[1])
        oof = np.zeros(len(y))
        for tr_idx, val_idx, X_tr, X_val in fold_features_t:
            tr_median    = X_tr.median()
            X_tr_filled  = X_tr.fillna(tr_median)
            X_val_filled = X_val.fillna(tr_median)
            model = ExtraTreesClassifier(**params)
            model.fit(X_tr_filled, y[tr_idx])
            oof[val_idx] = model.predict_proba(X_val_filled)[:, 1]
        return log_loss(y, oof)
    return objective


def train_and_predict(train, test, parquet_feat):
    le          = LabelEncoder().fit(train["subject_id"])
    groups      = train["subject_id"].values
    cv          = GroupKFold(n_splits=10)
    sensor_cols = get_sensor_cols(parquet_feat)
    n_seeds     = len(SEEDS)

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

    lf_full    = build_label_features(train, train)
    X_full     = build_features(train, train, parquet_feat, lf_full, True, le)
    full_stats = compute_subj_stats(X_full["subject_id"], X_full, sensor_cols)

    gps_params = load_params(ET_KEY_GPS)
    if not gps_params:
        print("ERROR: extratrees_gps params 캐시 없음")
        return None
    best_gps = {t: gps_params[t] for t in TARGETS}

    print("=== Phase 0: Feature Importance 계산 ===")
    keep_feats = compute_importance(train, parquet_feat, le, sensor_cols,
                                    fold_label_feats, best_gps)
    print()

    print("폴드 피처 사전 계산 중 (slim 80%)...")
    fold_by_target, te_by_target = precompute_fold_features(
        train, test, parquet_feat, le, sensor_cols,
        fold_label_feats, label_feat_test, full_stats, keep_feats,
    )
    n_feats = fold_by_target[TARGETS[0]][0][2].shape[1]
    print(f"  완료 (피처 수: {n_feats}개)\n")

    cached_et = load_params(ET_KEY_SLIM)
    if cached_et:
        print(f"=== Phase 1: 저장된 params 로드 ({ET_KEY_SLIM}) ===")
        best_et = {t: cached_et[t] for t in TARGETS}
        for t in TARGETS:
            bp = best_et[t]
            print(f"  {t}: n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
                  f"max_feat={bp['max_features']:.4f}")
        print()
    else:
        print(f"=== Phase 1: ExtraTrees Optuna slim80 ({N_TRIALS} trials/target) ===")
        best_et = {}
        for t in TARGETS:
            y = train[t].values
            study = optuna.create_study(direction="minimize",
                                        sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(make_et_objective(ET_SEARCH, ET_FIXED, fold_by_target[t], y),
                           n_trials=N_TRIALS, show_progress_bar=True)
            bp = {**ET_FIXED, **study.best_params}
            best_et[t] = bp
            ref_ll = ET_LL_GPS_SLIM.get(t, "-")
            print(f"  {t}: LL={study.best_value:.4f} (slim90: {ref_ll}) | "
                  f"n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
                  f"max_feat={bp['max_features']:.4f}")
        print()
        save_params(ET_KEY_SLIM, best_et)

    print(f"=== Phase 2: ExtraTrees slim80 앙상블 ({n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균F1   평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 16))

    et_oof  = {t: np.zeros(len(train)) for t in TARGETS}
    et_test = {t: np.zeros(len(test))  for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        seed_et_oof  = {t: np.zeros(len(train)) for t in TARGETS}
        seed_et_test = {t: np.zeros(len(test))  for t in TARGETS}

        for t in TARGETS:
            y = train[t].values
            et_params = {**best_et[t], "random_state": seed}
            for tr_idx, val_idx, X_tr, X_val in fold_by_target[t]:
                tr_median    = X_tr.median()
                X_tr_filled  = X_tr.fillna(tr_median)
                X_val_filled = X_val.fillna(tr_median)
                X_te_filled  = te_by_target[t].fillna(tr_median)
                em = ExtraTreesClassifier(**et_params)
                em.fit(X_tr_filled, y[tr_idx])
                seed_et_oof[t][val_idx]  = em.predict_proba(X_val_filled)[:, 1]
                seed_et_test[t]         += em.predict_proba(X_te_filled)[:, 1] / cv.n_splits

        for t in TARGETS:
            et_oof[t]  += seed_et_oof[t]  / n_seeds
            et_test[t] += seed_et_test[t] / n_seeds

        f1s, lls = [], []
        for t in TARGETS:
            cur = et_oof[t] * n_seeds / (seed_i + 1)
            f1s.append(f1_score(train[t].values, (cur > 0.5).astype(int)))
            lls.append(log_loss(train[t].values, cur))
        f1_str = "  ".join(f"{f:>6.3f}" for f in f1s)
        print(f"{seed:>6}  {f1_str}  {np.mean(f1s):>6.3f}  {np.mean(lls):>6.4f}")

    print()
    print(f"{'타깃':<5}  {'F1':>6}  {'OOF LL':>8}  {'slim90':>8}  {'개선':>7}")
    print("-" * 44)
    lls_final = []
    for t in TARGETS:
        f1 = f1_score(train[t].values, (et_oof[t] > 0.5).astype(int))
        ll = log_loss(train[t].values, et_oof[t])
        lls_final.append(ll)
        ref  = ET_LL_GPS_SLIM[t]
        diff = ref - ll
        sign = "[+]" if diff > 0 else "[-]"
        print(f"{t:<5}  {f1:>6.3f}  {ll:>8.4f}  {ref:>8.4f}  {diff:>+6.4f} {sign}")
    avg_ll  = np.mean(lls_final)
    avg_ref = np.mean(list(ET_LL_GPS_SLIM.values()))
    print(f"{'평균':<5}  {'':>6}  {avg_ll:>8.4f}  {avg_ref:>8.4f}  {avg_ref-avg_ll:>+6.4f}")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = et_test[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v2 + GPS 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    print(f"=== ExtraTrees GPS Slim 80% ===\n")
    result = train_and_predict(train, sample, parquet_feat)
    if result is None:
        return

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "extratrees_gps_slim80_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
