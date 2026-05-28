"""
ET GPS Slim 80% + Transductive Z-score + 피험자별 개인 모델 블렌딩 (타깃별 alpha)

personal_blend.py 에서 한 가지만 변경:
  Phase 3: 전역 단일 alpha -> 타깃별 독립 alpha 최적화
  각 타깃의 WS OOF val log-loss를 독립적으로 최소화해 alpha[t] 결정

  personal_blend 결과 (alpha=0.2056 전역):
    Q2: 0.6665 (global-only 0.6503보다 악화 -> Q2는 alpha 낮아야)
    S1: 0.5718 (global-only 0.5892보다 개선  -> S1은 alpha 높아야)
    => 타깃별 alpha가 전역 alpha보다 나을 수 있음

출력: submission/et_gps_slim80_pers_pertarget_prob.csv
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

TARGETS   = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS     = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]
WS_SEED   = 42
VAL_RATIO = 0.20
CALIB_REG = 0.5
BIAS_BOUND = 2.0

PERS_N_EST    = 50
PERS_DEPTH    = 3
PERS_MAX_FEAT = 0.3
PERS_MIN_LEAF = 2

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]


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


def make_ws_splits(train_df, val_ratio=VAL_RATIO):
    splits = {}
    for sid, grp in train_df.groupby("subject_id"):
        sorted_pos = grp.sort_values("sleep_date").index.tolist()
        n = len(sorted_pos)
        n_val = max(1, int(n * val_ratio))
        splits[sid] = {
            "val_pos":   sorted_pos[-n_val:],
            "train_pos": sorted_pos[:-n_val],
        }
    return splits


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


def blend_probs(global_p, pers_p, alpha):
    return alpha * pers_p + (1.0 - alpha) * global_p


def optimize_alpha_per_target(ws_oof_global, ws_oof_pers, train_reset, all_val_pos):
    """타깃별 독립 alpha 최적화 (0~0.6 범위 grid search + fine-tune)."""
    best_alphas = {}
    print(f"\n{'타깃':>4}  {'alpha':>7}  {'global LL':>9}  {'blend LL':>9}  {'개선':>7}")
    print("-" * 48)
    for t in TARGETS:
        y_v = train_reset.loc[all_val_pos, t].values
        g_v = np.array([ws_oof_global[t][p] for p in all_val_pos])
        p_v = np.array([ws_oof_pers[t][p]   for p in all_val_pos])

        def ll_t(alpha):
            blended = np.clip(alpha * p_v + (1.0 - alpha) * g_v, 1e-6, 1 - 1e-6)
            return log_loss(y_v, blended)

        alphas = np.linspace(0.0, 0.6, 31)
        grid_ll = [ll_t(a) for a in alphas]
        best_grid = alphas[int(np.argmin(grid_ll))]
        res = minimize_scalar(ll_t, bounds=(
            max(0.0, best_grid - 0.05),
            min(1.0, best_grid + 0.05)
        ), method="bounded")
        best_alphas[t] = float(res.x)

        global_ll = ll_t(0.0)
        blend_ll  = float(res.fun)
        print(f"  {t}  {best_alphas[t]:>7.4f}  {global_ll:>9.4f}  {blend_ll:>9.4f}  {global_ll - blend_ll:>+7.4f}")

    avg_global = np.mean([
        log_loss(train_reset.loc[all_val_pos, t].values,
                 np.clip([ws_oof_global[t][p] for p in all_val_pos], 1e-6, 1 - 1e-6))
        for t in TARGETS
    ])
    avg_blend = np.mean([
        log_loss(train_reset.loc[all_val_pos, t].values,
                 np.clip(blend_probs(
                     np.array([ws_oof_global[t][p] for p in all_val_pos]),
                     np.array([ws_oof_pers[t][p]   for p in all_val_pos]),
                     best_alphas[t]
                 ), 1e-6, 1 - 1e-6))
        for t in TARGETS
    ])
    print(f"\n  평균  {'':>7}  {avg_global:>9.4f}  {avg_blend:>9.4f}  {avg_global - avg_blend:>+7.4f}")
    return best_alphas, avg_blend


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS 피처 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    print("=== ET GPS Slim 80% + Transductive + Global+Personal (per-target alpha) ===\n")

    le          = LabelEncoder().fit(train["subject_id"])
    sensor_cols = get_sensor_cols(parquet_feat)
    subjects    = sorted(train["subject_id"].unique())

    transductive_stats = compute_transductive_stats(parquet_feat, sensor_cols)
    print(f"  transductive 통계: {len(transductive_stats)}명, {len(sensor_cols)}개 센서 컬럼")

    print("label 피처 사전 계산 중...")
    lf_all  = build_label_features(train, train)
    lf_test = build_label_features(train, sample)
    print("  완료")

    gps_params  = load_params(ET_KEY_GPS)
    slim_params = load_params(ET_KEY_SLIM)
    if not gps_params or not slim_params:
        print("ERROR: params 캐시 없음.")
        return
    best_gps  = {t: gps_params[t]  for t in TARGETS}
    best_slim = {t: slim_params[t] for t in TARGETS}

    # ── Phase 0: Feature Importance (LOSO, 1 seed) ──
    print("\n=== Phase 0: Feature Importance 계산 (slim 80%, LOSO 1 seed) ===")
    cv = GroupKFold(n_splits=10)
    groups = train["subject_id"].values
    fold_indices = list(cv.split(np.zeros(len(train)), train[TARGETS[0]].values, groups))

    importance_dict   = {t: [] for t in TARGETS}
    feature_names_ref = {}

    for tr_idx, val_idx in fold_indices:
        train_fold = train.iloc[tr_idx].copy()
        lf_tr = build_label_features(train_fold, train_fold)

        X_tr_raw = build_features(train_fold, train_fold, parquet_feat, lf_tr, True, le)
        sid_tr   = X_tr_raw["subject_id"].reset_index(drop=True)
        X_tr_z   = apply_zscore(sid_tr, X_tr_raw.drop(columns=["subject_id"]),
                                transductive_stats, sensor_cols)
        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            X_tr_t = X_tr_z.drop(columns=drop_cols, errors="ignore")
            if t not in feature_names_ref:
                feature_names_ref[t] = X_tr_t.columns.tolist()
            X_tr_filled = X_tr_t.fillna(X_tr_t.median())
            model = ExtraTreesClassifier(**{**best_gps[t], "random_state": 42})
            model.fit(X_tr_filled, train.iloc[tr_idx][t].values)
            importance_dict[t].append(model.feature_importances_)

    all_imp = {t: pd.Series(np.mean(importance_dict[t], axis=0),
                            index=feature_names_ref[t]) for t in TARGETS}
    common_feats = list(reduce(lambda a, b: a & b, [set(all_imp[t].index) for t in TARGETS]))
    combined = pd.DataFrame({t: all_imp[t][common_feats] for t in TARGETS})
    combined["mean_imp"] = combined.mean(axis=1)
    combined = combined.sort_values("mean_imp", ascending=False)
    total = combined["mean_imp"].sum()
    combined["cumsum"] = combined["mean_imp"].cumsum() / total
    keep_feats = set(combined[combined["cumsum"] <= IMP_COVERAGE].index.tolist())
    print(f"  전체 공통 피처: {len(combined)}개")
    print(f"  상위 {IMP_COVERAGE*100:.0f}% 커버: {len(keep_feats)}개 유지")

    # ── Phase 1: 전체 데이터 피처 계산 ──
    print("\n=== Phase 1: 전체 데이터 피처 계산 ===")
    X_all_raw = build_features(train, train, parquet_feat, lf_all, True, le)
    X_te_raw  = build_features(sample, train, parquet_feat, lf_test, False, le)

    train_reset = train.reset_index(drop=True)

    sid_all = X_all_raw["subject_id"].reset_index(drop=True)
    sid_te  = X_te_raw["subject_id"].reset_index(drop=True)

    X_all_z = apply_zscore(sid_all, X_all_raw.drop(columns=["subject_id"]),
                           transductive_stats, sensor_cols)
    X_te_z  = apply_zscore(sid_te,  X_te_raw.drop(columns=["subject_id"]),
                           transductive_stats, sensor_cols)

    # ── Phase 2: WS OOF 수집 (Global + Personal) ──
    print("\n=== Phase 2: WS OOF 수집 (Global + Personal 모델) ===")
    ws_splits = make_ws_splits(train_reset)

    ws_oof_global = {t: np.full(len(train_reset), np.nan) for t in TARGETS}
    ws_oof_pers   = {t: np.full(len(train_reset), np.nan) for t in TARGETS}

    for sid in subjects:
        val_pos   = ws_splits[sid]["val_pos"]
        all_non_val = sorted(set(train_reset.index) - set(val_pos))

        ws_tr_df  = train_reset.loc[all_non_val].reset_index(drop=True)
        ws_val_df = train_reset.loc[val_pos].reset_index(drop=True)

        lf_ws_tr  = build_label_features(ws_tr_df,  ws_tr_df)
        lf_ws_val = build_label_features(ws_val_df, ws_val_df)

        X_ws_tr_raw  = build_features(ws_tr_df,  ws_tr_df,  parquet_feat, lf_ws_tr,  True,  le)
        X_ws_val_raw = build_features(ws_val_df, ws_tr_df,  parquet_feat, lf_ws_val, False, le)

        sid_ws_tr  = X_ws_tr_raw["subject_id"].reset_index(drop=True)
        sid_ws_val = X_ws_val_raw["subject_id"].reset_index(drop=True)

        X_ws_tr_z  = apply_zscore(sid_ws_tr,  X_ws_tr_raw.drop(columns=["subject_id"]),
                                  transductive_stats, sensor_cols)
        X_ws_val_z = apply_zscore(sid_ws_val, X_ws_val_raw.drop(columns=["subject_id"]),
                                  transductive_stats, sensor_cols)

        sid_mask_in_tr = (ws_tr_df["subject_id"].values == sid)

        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            avail_tr  = [c for c in keep_feats if c in X_ws_tr_z.columns  and c not in drop_cols]
            avail_val = [c for c in keep_feats if c in X_ws_val_z.columns and c not in drop_cols]
            cols_t = [c for c in avail_tr if c in avail_val]

            X_tr_t  = X_ws_tr_z[cols_t]
            X_val_t = X_ws_val_z[cols_t]
            tr_med  = X_tr_t.median()
            X_tr_filled  = X_tr_t.fillna(tr_med)
            X_val_filled = X_val_t.fillna(tr_med)

            y_tr = ws_tr_df[t].values

            glb_model = ExtraTreesClassifier(**{**best_slim[t], "random_state": WS_SEED})
            glb_model.fit(X_tr_filled, y_tr)
            glb_preds = glb_model.predict_proba(X_val_filled)[:, 1]

            X_pers_tr = X_tr_t[sid_mask_in_tr].fillna(tr_med)
            y_pers_tr = y_tr[sid_mask_in_tr]
            pers_preds = np.full(len(val_pos), 0.5)
            if len(np.unique(y_pers_tr)) >= 2 and len(y_pers_tr) >= 4:
                pers_model = ExtraTreesClassifier(
                    n_estimators=PERS_N_EST,
                    max_depth=PERS_DEPTH,
                    max_features=PERS_MAX_FEAT,
                    min_samples_leaf=PERS_MIN_LEAF,
                    criterion="entropy",
                    random_state=WS_SEED,
                )
                pers_model.fit(X_pers_tr, y_pers_tr)
                pers_preds = pers_model.predict_proba(X_val_filled)[:, 1]

            for k, pos in enumerate(val_pos):
                ws_oof_global[t][pos] = glb_preds[k]
                ws_oof_pers[t][pos]   = pers_preds[k]

        n_val   = len(val_pos)
        n_dates = train_reset[train_reset["subject_id"] == sid].shape[0]
        n_pers  = int(sid_mask_in_tr.sum())
        print(f"  {sid}: {n_dates}일 (val={n_val}일, 개인train={n_pers}일)")

    # ── Phase 3: 타깃별 alpha 최적화 ──
    all_val_pos = [p for sid in subjects for p in ws_splits[sid]["val_pos"]]
    print("\n=== Phase 3: 타깃별 alpha 최적화 (WS OOF val 전체, 0~0.6 탐색) ===")
    best_alphas, avg_blend_ll = optimize_alpha_per_target(
        ws_oof_global, ws_oof_pers, train_reset, all_val_pos
    )

    # personal_blend 전역 alpha=0.2056 비교용
    global_alpha_ref = 0.2056
    avg_ref = np.mean([
        log_loss(train_reset.loc[all_val_pos, t].values,
                 np.clip(blend_probs(
                     np.array([ws_oof_global[t][p] for p in all_val_pos]),
                     np.array([ws_oof_pers[t][p]   for p in all_val_pos]),
                     global_alpha_ref
                 ), 1e-6, 1 - 1e-6))
        for t in TARGETS
    ])
    print(f"\n  비교: 전역 alpha={global_alpha_ref} WS OOF LL = {avg_ref:.4f}")
    print(f"  타깃별 alpha WS OOF LL = {avg_blend_ll:.4f}  "
          f"({'개선' if avg_blend_ll < avg_ref else '악화'} {abs(avg_ref - avg_blend_ll):.4f})")

    # ── Phase 4: 전체 데이터 Global 모델 (10 seeds) ──
    print(f"\n=== Phase 4: 전체 데이터 Global 모델 ({len(SEEDS)} seeds) ===")
    et_test_global = {t: np.zeros(len(sample)) for t in TARGETS}

    for seed in SEEDS:
        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            cols_t = [c for c in keep_feats if c in X_all_z.columns and c not in drop_cols]
            X_all_t  = X_all_z[cols_t]
            X_te_t   = X_te_z[[c for c in cols_t if c in X_te_z.columns]]
            all_med  = X_all_t.median()
            X_all_filled = X_all_t.fillna(all_med)
            X_te_filled  = X_te_t.fillna(all_med)
            model = ExtraTreesClassifier(**{**best_slim[t], "random_state": seed})
            model.fit(X_all_filled, train_reset[t].values)
            et_test_global[t] += model.predict_proba(X_te_filled)[:, 1] / len(SEEDS)
        print(f"  seed {seed} 완료")

    # ── Phase 5: 피험자별 Personal 모델 (10 seeds) ──
    print(f"\n=== Phase 5: 피험자별 Personal 모델 ({len(SEEDS)} seeds) ===")
    et_test_pers = {t: np.zeros(len(sample)) for t in TARGETS}

    for sid in subjects:
        mask_te  = (sample["subject_id"] == sid).values
        sid_mask = (train_reset["subject_id"] == sid).values
        n_pers   = int(sid_mask.sum())

        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            cols_t = [c for c in keep_feats if c in X_all_z.columns and c not in drop_cols]
            all_med = X_all_z[cols_t].median()

            X_pers = X_all_z[cols_t][sid_mask].fillna(all_med)
            y_pers = train_reset.loc[sid_mask, t].values
            X_te_sid = X_te_z[[c for c in cols_t if c in X_te_z.columns]][mask_te].fillna(all_med)

            if len(np.unique(y_pers)) < 2 or len(y_pers) < 4:
                et_test_pers[t][mask_te] = 0.5
                continue

            preds_sid = np.zeros(int(mask_te.sum()))
            for seed in SEEDS:
                pers_model = ExtraTreesClassifier(
                    n_estimators=PERS_N_EST,
                    max_depth=PERS_DEPTH,
                    max_features=PERS_MAX_FEAT,
                    min_samples_leaf=PERS_MIN_LEAF,
                    criterion="entropy",
                    random_state=seed,
                )
                pers_model.fit(X_pers, y_pers)
                preds_sid += pers_model.predict_proba(X_te_sid)[:, 1] / len(SEEDS)
            et_test_pers[t][mask_te] = preds_sid

        print(f"  {sid}: {n_pers}일 데이터로 개인 모델 완료")

    # ── Phase 6: 타깃별 alpha 블렌드 ──
    print("\n=== Phase 6: 타깃별 alpha 블렌드 ===")
    et_test_blend = {}
    for t in TARGETS:
        et_test_blend[t] = blend_probs(et_test_global[t], et_test_pers[t], best_alphas[t])
        print(f"  {t}: alpha={best_alphas[t]:.4f}")

    # ── Phase 7: WS OOF 블렌드 예측으로 피험자별 편향 보정 ──
    print(f"\n=== Phase 7: WS OOF 기반 편향 보정 (REG={CALIB_REG}, 타깃별 alpha 적용) ===")
    biases = {}
    for sid in subjects:
        val_pos = ws_splits[sid]["val_pos"]
        biases[sid] = {}
        for t in TARGETS:
            y_val = train_reset.loc[val_pos, t].values
            g_val = np.array([ws_oof_global[t][p] for p in val_pos])
            p_val = np.array([ws_oof_pers[t][p]   for p in val_pos])
            bl_val = np.clip(blend_probs(g_val, p_val, best_alphas[t]), 1e-6, 1 - 1e-6)
            if len(np.unique(y_val)) < 2:
                biases[sid][t] = 0.0
                continue
            biases[sid][t] = fit_logit_bias(bl_val, y_val)

    print(f"\n{'피험자':<8}  " + "  ".join(f"{t:>7}" for t in TARGETS) + "  평균|편향|")
    print("-" * (10 + 10 * len(TARGETS)))
    for sid in subjects:
        vals = [biases[sid][t] for t in TARGETS]
        val_str = "  ".join(f"{v:>+7.4f}" for v in vals)
        print(f"{sid:<8}  {val_str}  {np.mean(np.abs(vals)):>8.4f}")

    # ── Phase 8: 편향 적용 ──
    et_test_cal = {t: et_test_blend[t].copy() for t in TARGETS}
    for sid in subjects:
        mask_te = (sample["subject_id"] == sid).values
        if mask_te.sum() == 0:
            continue
        for t in TARGETS:
            et_test_cal[t][mask_te] = apply_logit_bias(
                et_test_blend[t][mask_te], biases[sid][t]
            )

    result_prob = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = et_test_cal[t].clip(0.1, 0.9)

    print("\n=== 최종 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "et_gps_slim80_pers_pertarget_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")
    print(f"WS OOF (타깃별 alpha): {avg_blend_ll:.4f}")
    print(f"WS OOF (전역 alpha=0.2056): {avg_ref:.4f}")
    print("참고: et_gps_slim80_personal_blend Public 0.5992 대비 비교 필요.")


if __name__ == "__main__":
    main()
