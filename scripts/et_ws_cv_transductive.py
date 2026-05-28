"""
ET Within-Subject CV + Transductive Z-score + Optuna WS 튜닝 + 피험자별 로짓 편향 보정

LOSO (Leave-One-Subject-Out) 대신 Within-Subject 시간 기반 분할 사용:
  - 각 피험자의 sleep_date 순 정렬 후 앞 (1-VAL_RATIO) train, 뒤 VAL_RATIO val
  - CV 구조가 test 환경(동일 피험자 과거->미래 예측)과 일치
  - subject_id 피처: 모든 피험자가 train에 존재하여 cold-start 없음
  - label 피처: val fold 기준에서 train fold를 과거 참조 -> 정보 유출 없음

파이프라인:
  - Phase 0: Feature Importance (WS train fold, extratrees_gps 파라미터 기반)
  - Phase 1: Optuna WS 튜닝 (N_TRIALS 회, WS val split 기반, 캐시 없으면 실행)
  - Phase 2: 10 seeds WS 앙상블 (train fold 학습, val fold OOF 수집)
  - Phase 3: 피험자별 로짓 편향 fitting (WS val OOF 기반)
  - Phase 4: 전체 데이터 10 seeds 재학습 + test 예측 + 보정 적용

출력: submission/et_ws_cv_transductive_prob.csv
"""

import numpy as np
import pandas as pd
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize_scalar
from scipy.special import logit as logit_fn, expit as expit_fn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from gps_features import build_gps
from optuna_params_io import load_params, save_params

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

ET_KEY_GPS = "extratrees_gps"
ET_KEY_WS  = "extratrees_ws_cv"

IMP_COVERAGE = 0.80
VAL_RATIO    = 0.20
N_TRIALS     = 50
IMP_SEEDS    = [42, 123, 456]

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

CALIB_REG  = 0.5
BIAS_BOUND = 2.0

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]


# ──────────────────────────────────────────────
# WS 분할
# ──────────────────────────────────────────────

def make_ws_split(df, val_ratio=VAL_RATIO):
    """
    sleep_date 순 정렬 후 피험자별 뒤 val_ratio 를 val 로 분할.
    반환: (sorted_df, tr_pos, val_pos) - positional index 배열
    """
    sorted_df = df.sort_values(["subject_id", "sleep_date"]).reset_index(drop=True)
    tr_pos, val_pos = [], []
    for _, group in sorted_df.groupby("subject_id"):
        n = len(group)
        n_val = max(1, int(n * val_ratio))
        tr_pos.extend(group.index[:-n_val].tolist())
        val_pos.extend(group.index[-n_val:].tolist())
    return sorted_df, np.array(tr_pos), np.array(val_pos)


# ──────────────────────────────────────────────
# 피처 빌드 유틸
# ──────────────────────────────────────────────

def get_sensor_cols(parquet_feat):
    return [c for c in parquet_feat.columns if c not in ("subject_id", "date")]


def compute_transductive_stats(parquet_feat, sensor_cols):
    avail = [c for c in sensor_cols if c in parquet_feat.columns]
    return parquet_feat.groupby("subject_id")[avail].agg(["mean", "std"])


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
# Phase 0: 피처 중요도 (WS train fold 기반)
# ──────────────────────────────────────────────

def compute_importance_ws(train_fold, parquet_feat, le, sensor_cols,
                          label_feat_tr, transductive_stats, best_gps):
    X_tr   = build_features(train_fold, train_fold, parquet_feat, label_feat_tr, True, le)
    sid_tr = X_tr["subject_id"].reset_index(drop=True)
    X_tr_z = apply_zscore(sid_tr, X_tr.drop(columns=["subject_id"]),
                          transductive_stats, sensor_cols)

    importance_dict = {t: [] for t in TARGETS}
    feature_names_ref = {}

    for seed in IMP_SEEDS:
        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            X_t = X_tr_z.drop(columns=drop_cols, errors="ignore")
            if t not in feature_names_ref:
                feature_names_ref[t] = X_t.columns.tolist()
            X_t_filled = X_t.fillna(X_t.median())
            y = train_fold[t].values
            model = ExtraTreesClassifier(**{**best_gps[t], "random_state": seed})
            model.fit(X_t_filled, y)
            importance_dict[t].append(model.feature_importances_)

    all_imp = {t: pd.Series(np.mean(importance_dict[t], axis=0),
                            index=feature_names_ref[t])
               for t in TARGETS}
    # 공통 피처 집합으로 합산
    common = list(set.intersection(*[set(v.index) for v in all_imp.values()]))
    combined = pd.DataFrame({t: all_imp[t][common] for t in TARGETS})
    combined["mean_imp"] = combined.mean(axis=1)
    combined = combined.sort_values("mean_imp", ascending=False)
    total = combined["mean_imp"].sum()
    combined["cumsum"] = combined["mean_imp"].cumsum() / total
    keep = combined[combined["cumsum"] <= IMP_COVERAGE].index.tolist()

    print(f"  전체 공통 피처: {len(combined)}개")
    print(f"  상위 {IMP_COVERAGE*100:.0f}% 커버: {len(keep)}개 유지 / {len(combined)-len(keep)}개 제거")
    return set(keep)


# ──────────────────────────────────────────────
# Phase 1: Optuna WS 튜닝
# ──────────────────────────────────────────────

def run_optuna_ws(X_tr_by_target, y_tr_by_target,
                  X_val_by_target, y_val_by_target):
    best_ws = {}
    for t in TARGETS:
        X_tr  = X_tr_by_target[t]
        y_tr  = y_tr_by_target[t]
        X_val = X_val_by_target[t]
        y_val = y_val_by_target[t]
        tr_med = X_tr.median()

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
                "max_depth":    trial.suggest_int("max_depth",    3,   30),
                "max_features": trial.suggest_float("max_features", 0.03, 0.9),
                "criterion":    "entropy",
                "n_jobs":       -1,
                "random_state": 42,
            }
            m = ExtraTreesClassifier(**params)
            m.fit(X_tr.fillna(tr_med), y_tr)
            pred = m.predict_proba(X_val.fillna(tr_med))[:, 1]
            return log_loss(y_val, pred)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=42),
        )
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
        bp = study.best_params
        best_ws[t] = {k: v for k, v in bp.items()}
        best_ws[t]["criterion"]    = "entropy"
        best_ws[t]["n_jobs"]       = -1
        best_ws[t]["random_state"] = 42
        print(f"  {t}: n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
              f"max_feat={bp['max_features']:.4f}  val_LL={study.best_value:.4f}")
    return best_ws


# ──────────────────────────────────────────────
# 피험자별 로짓 편향 보정
# ──────────────────────────────────────────────

def fit_logit_bias(pred_oof, y_true, reg=CALIB_REG, bound=BIAS_BOUND):
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
# 메인 학습/예측 파이프라인
# ──────────────────────────────────────────────

def train_and_predict(train, test, parquet_feat):
    le          = LabelEncoder().fit(train["subject_id"])
    sensor_cols = get_sensor_cols(parquet_feat)
    n_seeds     = len(SEEDS)

    # Transductive Z-score 통계
    transductive_stats = compute_transductive_stats(parquet_feat, sensor_cols)
    print(f"  transductive 통계: {len(transductive_stats)}명, {len(sensor_cols)}개 센서")

    # WS 분할
    train_sorted, tr_pos, val_pos = make_ws_split(train)
    train_fold = train_sorted.iloc[tr_pos].reset_index(drop=True)
    val_fold   = train_sorted.iloc[val_pos].reset_index(drop=True)
    print(f"  WS 분할: train {len(train_fold)}행, val {len(val_fold)}행")

    # Label 피처 사전 계산
    print("label 피처 사전 계산 중...")
    lf_tr   = build_label_features(train_fold, train_fold)
    lf_val  = build_label_features(train_fold, val_fold)
    lf_full = build_label_features(train_sorted, train_sorted)
    lf_te   = build_label_features(train_sorted, test)
    print("  완료")

    # GPS 파라미터 (Phase 0 feature importance용)
    gps_params = load_params(ET_KEY_GPS)
    if not gps_params:
        print("ERROR: extratrees_gps params 캐시 없음.")
        return None
    best_gps = {t: gps_params[t] for t in TARGETS}

    # Phase 0: Feature Importance
    print("\n=== Phase 0: Feature Importance (WS train fold) ===")
    keep_feats = compute_importance_ws(
        train_fold, parquet_feat, le, sensor_cols,
        lf_tr, transductive_stats, best_gps,
    )
    print()

    # 피처 행렬 사전 계산 (Z-score + slim 적용)
    print("피처 행렬 계산 중 (slim 80% + transductive z-score)...")

    def make_slim(X_raw, ref_X_raw=None):
        """Z-score 적용 후 slim 피처 선택. ref_X_raw 는 None 이면 자기 자신 기준."""
        sid = X_raw["subject_id"].reset_index(drop=True)
        X_z = apply_zscore(sid, X_raw.drop(columns=["subject_id"]), transductive_stats, sensor_cols)
        return X_z

    X_tr_raw   = build_features(train_fold,   train_fold,   parquet_feat, lf_tr,   True,  le)
    X_val_raw  = build_features(val_fold,      train_fold,   parquet_feat, lf_val,  False, le)
    X_full_raw = build_features(train_sorted,  train_sorted, parquet_feat, lf_full, True,  le)
    X_te_raw   = build_features(test.copy(),   train_sorted, parquet_feat, lf_te,   False, le)

    X_tr_z   = make_slim(X_tr_raw)
    X_val_z  = make_slim(X_val_raw)
    X_full_z = make_slim(X_full_raw)
    X_te_z   = make_slim(X_te_raw)

    # 타깃별 slim 피처 선택 + median 채우기
    X_tr_by_t, X_val_by_t, X_full_by_t, X_te_by_t = {}, {}, {}, {}
    med_tr_by_t = {}

    for t in TARGETS:
        drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
        slim_cols = [c for c in X_tr_z.drop(columns=drop_cols, errors="ignore").columns
                     if c in keep_feats]

        def sel(X):
            X_t = X.drop(columns=drop_cols, errors="ignore")
            return X_t[[c for c in slim_cols if c in X_t.columns]]

        X_tr_t   = sel(X_tr_z)
        tr_med   = X_tr_t.median()
        med_tr_by_t[t]  = tr_med
        X_tr_by_t[t]    = X_tr_t.fillna(tr_med)
        X_val_by_t[t]   = sel(X_val_z).fillna(tr_med)
        X_full_by_t[t]  = sel(X_full_z).fillna(tr_med)
        X_te_by_t[t]    = sel(X_te_z).fillna(tr_med)

    n_feats = X_tr_by_t[TARGETS[0]].shape[1]
    print(f"  완료 (피처 수: {n_feats}개)\n")

    y_tr   = {t: train_fold[t].values   for t in TARGETS}
    y_val  = {t: val_fold[t].values     for t in TARGETS}
    y_full = {t: train_sorted[t].values for t in TARGETS}

    # Phase 1: Optuna WS 튜닝 or 캐시 로드
    ws_params = load_params(ET_KEY_WS)
    if ws_params:
        print(f"=== Phase 1: 캐시된 WS params 로드 ({ET_KEY_WS}) ===")
        best_ws = {t: ws_params[t] for t in TARGETS}
        for t in TARGETS:
            bp = best_ws[t]
            print(f"  {t}: n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
                  f"max_feat={bp['max_features']:.4f}")
    else:
        print(f"=== Phase 1: Optuna WS 튜닝 ({N_TRIALS} trials per target) ===")
        best_ws = run_optuna_ws(X_tr_by_t, y_tr, X_val_by_t, y_val)
        save_params(ET_KEY_WS, best_ws)
    print()

    # Phase 2: 10 seeds WS 앙상블 (val OOF 수집)
    print(f"=== Phase 2: WS 앙상블 ({n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 8))

    ws_val_oof  = {t: np.zeros(len(val_fold)) for t in TARGETS}
    ws_test_pred = {t: np.zeros(len(test))    for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        for t in TARGETS:
            params = {**best_ws[t], "random_state": seed}
            m = ExtraTreesClassifier(**params)
            m.fit(X_tr_by_t[t], y_tr[t])
            ws_val_oof[t]  += m.predict_proba(X_val_by_t[t])[:, 1] / n_seeds
            ws_test_pred[t] += m.predict_proba(X_te_by_t[t])[:, 1] / n_seeds

        lls = [log_loss(y_val[t], ws_val_oof[t] * n_seeds / (seed_i + 1))
               for t in TARGETS]
        ll_str = "  ".join(f"{ll:>6.4f}" for ll in lls)
        print(f"{seed:>6}  {ll_str}  {np.mean(lls):>6.4f}")

    print()
    print("=== Phase 2 WS val OOF LL ===")
    print(f"{'타깃':<5}  {'val LL':>8}")
    print("-" * 20)
    lls_val = {}
    for t in TARGETS:
        ll = log_loss(y_val[t], ws_val_oof[t])
        lls_val[t] = ll
        print(f"{t:<5}  {ll:>8.4f}")
    print(f"{'평균':<5}  {np.mean(list(lls_val.values())):>8.4f}")

    # Phase 3: 피험자별 로짓 편향 fitting (WS val 기반)
    print(f"\n=== Phase 3: 피험자별 로짓 편향 보정 (REG={CALIB_REG}, WS val 기반) ===")
    subjects = sorted(train["subject_id"].unique())
    biases = {sid: {} for sid in subjects}

    for sid in subjects:
        mask_val = (val_fold["subject_id"] == sid).values
        for t in TARGETS:
            pred = ws_val_oof[t][mask_val]
            y    = y_val[t][mask_val]
            if len(pred) == 0 or len(np.unique(y)) < 2:
                biases[sid][t] = 0.0
                continue
            biases[sid][t] = fit_logit_bias(pred, y)

    print(f"\n{'피험자':<8}  " + "  ".join(f"{t:>7}" for t in TARGETS) + "  평균|편향|")
    print("-" * (10 + 10 * len(TARGETS)))
    for sid in subjects:
        vals = [biases[sid][t] for t in TARGETS]
        val_str = "  ".join(f"{v:>+7.4f}" for v in vals)
        print(f"{sid:<8}  {val_str}  {np.mean(np.abs(vals)):>8.4f}")

    # Phase 4: 전체 데이터 재학습 + test 예측
    print(f"\n=== Phase 4: 전체 데이터 재학습 ({n_seeds} seeds) ===")
    full_test_pred = {t: np.zeros(len(test)) for t in TARGETS}

    for seed in SEEDS:
        for t in TARGETS:
            params = {**best_ws[t], "random_state": seed}
            m = ExtraTreesClassifier(**params)
            m.fit(X_full_by_t[t], y_full[t])
            full_test_pred[t] += m.predict_proba(X_te_by_t[t])[:, 1] / n_seeds

    # 보정 적용
    for sid in subjects:
        mask_te = (test["subject_id"] == sid).values
        if mask_te.sum() == 0:
            continue
        for t in TARGETS:
            full_test_pred[t][mask_te] = apply_logit_bias(
                full_test_pred[t][mask_te], biases[sid][t]
            )

    # WS val 보정후 LL 출력 (참고용)
    val_oof_cal = {t: np.zeros(len(val_fold)) for t in TARGETS}
    for sid in subjects:
        mask_val = (val_fold["subject_id"] == sid).values
        for t in TARGETS:
            val_oof_cal[t][mask_val] = apply_logit_bias(
                ws_val_oof[t][mask_val], biases[sid][t]
            )

    print("\n=== WS val OOF 보정 전후 비교 ===")
    print(f"{'타깃':<5}  {'보정전':>8}  {'보정후':>8}  {'개선':>8}")
    print("-" * 38)
    for t in TARGETS:
        ll_b = lls_val[t]
        ll_a = log_loss(y_val[t], val_oof_cal[t])
        diff = ll_b - ll_a
        sign = "[+]" if diff > 0 else "[-]"
        print(f"{t:<5}  {ll_b:>8.4f}  {ll_a:>8.4f}  {diff:>+7.4f} {sign}")
    print()
    print("주의: WS val OOF는 동일 피험자 과거 데이터 기반이므로 LOSO OOF 보다 낙관적일 수 있음.")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = full_test_pred[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS 피처 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    print("=== ET Within-Subject CV + Transductive Z-score + 피험자별 보정 ===\n")
    result = train_and_predict(train, sample, parquet_feat)
    if result is None:
        return

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 최종 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "et_ws_cv_transductive_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
