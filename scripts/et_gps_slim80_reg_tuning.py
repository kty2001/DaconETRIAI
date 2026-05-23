"""
ET GPS Slim 80% + LOSO logit bias 보정 REG 튜닝

기존 et_gps_slim80_calibrated.py (REG=0.5, Public 0.6027)에서
Phase 0~2 결과를 재활용해 다른 REG 값으로 보정을 비교.

REG 값: [0.1, 0.25, 1.0] 시도 (0.5는 기존 결과 참조)
  - REG 작을수록: 보정이 강해짐 (편향을 더 많이 보정)
  - REG 클수록:  보정이 약해짐 (원래 예측에 가깝게 유지)

Phase 0: Feature Importance (extratrees_gps 캐시 활용, slim 80%)
Phase 1: Optuna 파라미터 로드 (extratrees_gps_slim80 캐시 활용)
Phase 2: 10 seeds LOSO 앙상블 -> OOF/test 수집 (1회 실행)
Phase 3: REG 값별 logit bias 보정 및 제출 파일 저장

출력:
  submission/et_gps_slim80_reg01_prob.csv  (REG=0.10)
  submission/et_gps_slim80_reg025_prob.csv (REG=0.25)
  submission/et_gps_slim80_reg10_prob.csv  (REG=1.00)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize_scalar
from scipy.special import logit as logit_fn, expit as expit_fn
from pathlib import Path
from functools import reduce

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from gps_features import build_gps
from optuna_params_io import load_params

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

ET_KEY_GPS  = "extratrees_gps"
ET_KEY_SLIM = "extratrees_gps_slim80"
IMP_COVERAGE = 0.80

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

# REG=0.5 는 기존 et_gps_slim80_calibrated.py 결과 (Public 0.6027) 참조
REG_CANDIDATES = [0.1, 0.25, 1.0]
BIAS_BOUND = 2.0

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]

ET_FIXED = {
    "criterion":    "entropy",
    "n_jobs":       -1,
    "random_state": 42,
}


# ──────────────────────────────────────────────
# 피처 빌드 유틸
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# Phase 0: Feature Importance (slim 80%)
# ──────────────────────────────────────────────

def compute_importance(train, parquet_feat, le, sensor_cols, fold_label_feats, best_gps):
    importance_dict = {t: [] for t in TARGETS}
    feature_names_ref = {}

    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr   = build_features(train_fold, train_fold, parquet_feat, lf_tr, True, le)
        sid_tr = X_tr["subject_id"].reset_index(drop=True)
        tr_stats = compute_subj_stats(sid_tr, X_tr, sensor_cols)
        X_tr_z   = apply_zscore(sid_tr, X_tr.drop(columns=["subject_id"]), tr_stats, sensor_cols)

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


# ──────────────────────────────────────────────
# Phase 1~2 사전 계산
# ──────────────────────────────────────────────

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


# ──────────────────────────────────────────────
# logit bias 보정 함수
# ──────────────────────────────────────────────

def fit_logit_bias(pred_oof, y_true, reg, bound=BIAS_BOUND):
    eps = 1e-6
    p  = np.clip(pred_oof, eps, 1 - eps)
    lp = logit_fn(p)

    def obj(b):
        return log_loss(y_true, expit_fn(lp + b)) + reg * (b ** 2)

    res = minimize_scalar(obj, bounds=(-bound, bound), method="bounded")
    return float(res.x)


def apply_logit_bias(pred_raw, bias, eps=1e-6):
    p = np.clip(pred_raw, eps, 1 - eps)
    return expit_fn(logit_fn(p) + bias)


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS 피처 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    print("=== ET GPS Slim 80% REG 튜닝 ===\n")

    le          = LabelEncoder().fit(train["subject_id"])
    groups      = train["subject_id"].values
    cv          = GroupKFold(n_splits=10)
    sensor_cols = get_sensor_cols(parquet_feat)
    n_seeds     = len(SEEDS)

    fold_indices = list(cv.split(np.zeros(len(train)), train[TARGETS[0]].values, groups))

    print("label 피처 사전 계산 중...")
    label_feat_test = build_label_features(train, sample)
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
        print("ERROR: extratrees_gps params 캐시 없음. extratrees_gps_ensemble.py를 먼저 실행하세요.")
        return
    best_gps = {t: gps_params[t] for t in TARGETS}

    print("=== Phase 0: Feature Importance (slim 80%) ===")
    keep_feats = compute_importance(train, parquet_feat, le, sensor_cols,
                                    fold_label_feats, best_gps)
    print()

    print("폴드 피처 사전 계산 중 (slim 80%)...")
    fold_by_target, te_by_target = precompute_fold_features(
        train, sample, parquet_feat, le, sensor_cols,
        fold_label_feats, label_feat_test, full_stats, keep_feats,
    )
    n_feats = fold_by_target[TARGETS[0]][0][2].shape[1]
    print(f"  완료 (피처 수: {n_feats}개)\n")

    slim_params = load_params(ET_KEY_SLIM)
    if not slim_params:
        print(f"ERROR: {ET_KEY_SLIM} params 캐시 없음. extratrees_gps_slim80_ensemble.py를 먼저 실행하세요.")
        return
    best_et = {t: slim_params[t] for t in TARGETS}
    print(f"=== Phase 1: params 로드 ({ET_KEY_SLIM}) ===")
    for t in TARGETS:
        bp = best_et[t]
        print(f"  {t}: n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
              f"max_feat={bp['max_features']:.4f}")
    print()

    # ── Phase 2: 10 seeds 앙상블 (1회 실행) ─────────────────────
    print(f"=== Phase 2: LOSO 앙상블 ({n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 8))

    et_oof  = {t: np.zeros(len(train))  for t in TARGETS}
    et_test = {t: np.zeros(len(sample)) for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        seed_et_oof  = {t: np.zeros(len(train))  for t in TARGETS}
        seed_et_test = {t: np.zeros(len(sample)) for t in TARGETS}

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
                seed_et_oof[t][val_idx] = em.predict_proba(X_val_filled)[:, 1]
                seed_et_test[t]        += em.predict_proba(X_te_filled)[:, 1] / cv.n_splits

        for t in TARGETS:
            et_oof[t]  += seed_et_oof[t]  / n_seeds
            et_test[t] += seed_et_test[t] / n_seeds

        lls = [log_loss(train[t].values, et_oof[t] * n_seeds / (seed_i + 1))
               for t in TARGETS]
        ll_str = "  ".join(f"{ll:>6.4f}" for ll in lls)
        print(f"{seed:>6}  {ll_str}  {np.mean(lls):>6.4f}")

    print()
    lls_before = {}
    for t in TARGETS:
        lls_before[t] = log_loss(train[t].values, et_oof[t])
    avg_before = np.mean(list(lls_before.values()))
    print(f"Phase 2 OOF LL (보정 전): {avg_before:.4f}\n")

    subjects = sorted(train["subject_id"].unique())

    # ── Phase 3: REG 값별 보정 ────────────────────────────────
    print("=== Phase 3: REG 값별 logit bias 보정 ===")
    print()

    reg_file_map = {
        0.10:  "et_gps_slim80_reg01_prob.csv",
        0.25:  "et_gps_slim80_reg025_prob.csv",
        1.00:  "et_gps_slim80_reg10_prob.csv",
    }

    results_summary = {}

    for reg in REG_CANDIDATES:
        print(f"--- REG = {reg:.2f} ---")

        biases = {sid: {} for sid in subjects}
        for sid in subjects:
            mask = (train["subject_id"] == sid).values
            for t in TARGETS:
                pred = et_oof[t][mask]
                y    = train[t].values[mask]
                if len(np.unique(y)) < 2:
                    biases[sid][t] = 0.0
                    continue
                biases[sid][t] = fit_logit_bias(pred, y, reg=reg)

        # 편향 테이블 출력
        print(f"  {'피험자':<8}  " + "  ".join(f"{t:>7}" for t in TARGETS) + "  평균|편향|")
        print("  " + "-" * (10 + 10 * len(TARGETS)))
        for sid in subjects:
            vals = [biases[sid][t] for t in TARGETS]
            val_str = "  ".join(f"{v:>+7.4f}" for v in vals)
            print(f"  {sid:<8}  {val_str}  {np.mean(np.abs(vals)):>8.4f}")

        # OOF 보정 후 LL
        et_oof_cal = {t: np.zeros(len(train)) for t in TARGETS}
        for sid in subjects:
            mask = (train["subject_id"] == sid).values
            for t in TARGETS:
                et_oof_cal[t][mask] = apply_logit_bias(et_oof[t][mask], biases[sid][t])

        lls_after = {}
        for t in TARGETS:
            lls_after[t] = log_loss(train[t].values, et_oof_cal[t])
        avg_after = np.mean(list(lls_after.values()))

        print(f"\n  OOF LL: {avg_before:.4f} -> {avg_after:.4f} ({avg_before - avg_after:+.4f})")
        results_summary[reg] = avg_after

        # test 보정 및 저장
        et_test_cal = {t: et_test[t].copy() for t in TARGETS}
        for sid in subjects:
            mask_te = (sample["subject_id"] == sid).values
            if mask_te.sum() == 0:
                continue
            for t in TARGETS:
                et_test_cal[t][mask_te] = apply_logit_bias(et_test[t][mask_te], biases[sid][t])

        result = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
        for t in TARGETS:
            result[t] = et_test_cal[t].clip(0.1, 0.9)

        print(f"\n  최종 예측 분포 (clip 0.1~0.9):")
        for t in TARGETS:
            print(f"    {t}: min={result[t].min():.3f}, mean={result[t].mean():.3f}, max={result[t].max():.3f}")

        fname = reg_file_map[reg]
        out_path = SUBMISSION_DIR / fname
        result.to_csv(out_path, index=False)
        print(f"\n  저장 완료: {out_path}\n")

    # ── 최종 비교 ─────────────────────────────────────────────
    print("=== REG 값별 OOF LL 비교 (보정 후) ===")
    print(f"  {'REG':>6}  {'OOF LL':>8}  {'파일명'}")
    print("  " + "-" * 60)
    print(f"  {'0.50':>6}  {'(기존 0.6027 Public 참조)':>8}  et_gps_slim80_calibrated_prob.csv")
    for reg in REG_CANDIDATES:
        fname = reg_file_map[reg]
        print(f"  {reg:>6.2f}  {results_summary[reg]:>8.4f}  {fname}")
    print()
    print("주의: OOF LL은 보정 데이터와 평가 데이터가 동일해 과낙관적. Public 점수로 최종 판단.")


if __name__ == "__main__":
    main()
