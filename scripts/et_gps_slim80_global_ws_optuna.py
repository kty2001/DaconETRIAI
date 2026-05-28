"""
ET GPS Slim 80% Global Model WS OOF 기반 Optuna 재탐색

현재 slim80 params는 LOSO(GroupKFold)로 최적화됨.
personal_blend 실제 평가 기준(WS OOF)으로 재탐색하면 다른 최적 params 발견 가능.

탐색 전략:
  - 목적함수: WS OOF avg LogLoss (LOSO 대신 피험자별 최근 20% holdout)
  - 개인 모델은 고정 (depth=2, max_feat=0.5, min_leaf=3 — grid search 최적)
  - 100 trials / target, 단일 seed (WS_SEED=42)로 속도 확보
  - 최적 params 저장 후 full pipeline 실행

출력:
  submission/optuna_params.json → extratrees_gps_slim80_ws 키로 저장
  submission/et_gps_slim80_global_ws_optuna_prob.csv
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
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
ET_KEY_OUT = "extratrees_gps_slim80_ws"
IMP_COVERAGE = 0.80

TARGETS    = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS      = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]
WS_SEED    = 42
VAL_RATIO  = 0.20
CALIB_REG  = 0.5
BIAS_BOUND = 2.0
N_TRIALS   = 50

# 개인 모델 — grid search 최적 파라미터 고정
PERS_N_EST    = 50
PERS_DEPTH    = 2
PERS_MAX_FEAT = 0.5
PERS_MIN_LEAF = 3

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]

ET_SEARCH = {
    "n_estimators":      ("int",   200,  2000, {"step": 50}),
    "max_depth":         ("int",   5,    30),
    "min_samples_split": ("int",   2,    10),
    "min_samples_leaf":  ("int",   1,    5),
    "max_features":      ("float", 0.05, 0.5),
}
ET_FIXED = {"criterion": "entropy", "n_jobs": -1}


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


def suggest_params(trial):
    params = {}
    for k, v in ET_SEARCH.items():
        if v[0] == "int":
            kw = v[3] if len(v) > 3 else {}
            params[k] = trial.suggest_int(k, v[1], v[2], **kw)
        else:
            params[k] = trial.suggest_float(k, v[1], v[2])
    return {**params, **ET_FIXED}


def compute_ws_oof_global(train_reset, subjects, ws_splits, parquet_feat,
                          transductive_stats, sensor_cols, lf_all, le,
                          keep_feats, params):
    ws_oof = {t: np.full(len(train_reset), np.nan) for t in TARGETS}

    for sid in subjects:
        val_pos     = ws_splits[sid]["val_pos"]
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

        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            avail_tr  = [c for c in keep_feats if c in X_ws_tr_z.columns  and c not in drop_cols]
            avail_val = [c for c in keep_feats if c in X_ws_val_z.columns and c not in drop_cols]
            cols_t = [c for c in avail_tr if c in avail_val]
            X_tr_t  = X_ws_tr_z[cols_t]
            X_val_t = X_ws_val_z[cols_t]
            tr_med  = X_tr_t.median()
            model = ExtraTreesClassifier(**{**params, "random_state": WS_SEED})
            model.fit(X_tr_t.fillna(tr_med), ws_tr_df[t].values)
            preds = model.predict_proba(X_val_t.fillna(tr_med))[:, 1]
            for k, pos in enumerate(val_pos):
                ws_oof[t][pos] = preds[k]

    return ws_oof


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS 피처 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    le          = LabelEncoder().fit(train["subject_id"])
    sensor_cols = get_sensor_cols(parquet_feat)
    subjects    = sorted(train["subject_id"].unique())
    transductive_stats = compute_transductive_stats(parquet_feat, sensor_cols)

    print("label 피처 사전 계산 중...")
    lf_all  = build_label_features(train, train)
    lf_test = build_label_features(train, sample)
    print("  완료")

    gps_params = load_params(ET_KEY_GPS)
    if not gps_params:
        print("ERROR: GPS params 캐시 없음.")
        return
    best_gps = {t: gps_params[t] for t in TARGETS}

    # ── Phase 0: Feature Importance ──
    print("\n=== Phase 0: Feature Importance (LOSO 1 seed) ===")
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
            model = ExtraTreesClassifier(**{**best_gps[t], "random_state": 42})
            model.fit(X_tr_t.fillna(X_tr_t.median()), train.iloc[tr_idx][t].values)
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
    print(f"  상위 {IMP_COVERAGE*100:.0f}% 커버: {len(keep_feats)}개 유지")

    train_reset = train.reset_index(drop=True)
    ws_splits   = make_ws_splits(train_reset)
    all_val_pos = [p for sid in subjects for p in ws_splits[sid]["val_pos"]]

    # 기준선: 현재 slim80 params WS OOF
    slim_params = load_params("extratrees_gps_slim80")
    if slim_params:
        print("\n=== 기준선: 현재 slim80 params WS OOF ===")
        baseline_ll = {}
        for t in TARGETS:
            ws_oof_base = compute_ws_oof_global(
                train_reset, subjects, ws_splits, parquet_feat,
                transductive_stats, sensor_cols, lf_all, le,
                keep_feats, slim_params[t]
            )
            y_v = train_reset.loc[all_val_pos, t].values
            g_v = np.array([ws_oof_base[t][p] for p in all_val_pos])
            baseline_ll[t] = log_loss(y_v, np.clip(g_v, 1e-6, 1 - 1e-6))
            print(f"  {t}: {baseline_ll[t]:.4f}")
        print(f"  평균: {np.mean(list(baseline_ll.values())):.4f}")

    # ── Phase 1: 전체 데이터 피처 계산 ──
    print("\n=== Phase 1: 전체 데이터 피처 계산 ===")
    X_all_raw = build_features(train, train, parquet_feat, lf_all, True, le)
    X_te_raw  = build_features(sample, train, parquet_feat, lf_test, False, le)

    sid_all = X_all_raw["subject_id"].reset_index(drop=True)
    sid_te  = X_te_raw["subject_id"].reset_index(drop=True)

    X_all_z = apply_zscore(sid_all, X_all_raw.drop(columns=["subject_id"]),
                           transductive_stats, sensor_cols)
    X_te_z  = apply_zscore(sid_te, X_te_raw.drop(columns=["subject_id"]),
                           transductive_stats, sensor_cols)
    X_all_z.index = train_reset.index

    # ── Phase 2: WS split feature matrix 사전 계산 (trial간 재사용) ──
    print("\n=== Phase 2: WS split feature matrix 사전 계산 ===")
    ws_cache = {}
    for i, sid in enumerate(subjects):
        val_pos     = ws_splits[sid]["val_pos"]
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

        ws_cache[sid] = {
            "X_tr":          X_ws_tr_z,
            "X_val":         X_ws_val_z,
            "y_tr":          {tt: ws_tr_df[tt].values  for tt in TARGETS},
            "y_val":         {tt: ws_val_df[tt].values for tt in TARGETS},
            "sid_mask_in_tr": np.where(ws_tr_df["subject_id"].values == sid)[0],
        }
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(subjects)} 완료", flush=True)
    print("  feature matrix 사전 계산 완료", flush=True)

    # 타깃별 컬럼 목록 사전 계산 (모든 피험자에서 공통)
    first_sid = subjects[0]
    ws_cols = {}
    for tt in TARGETS:
        drop_cols_tt = [f"subj_mean_{tt}"] + DROP_USAGE
        X_tr_sample = ws_cache[first_sid]["X_tr"]
        X_val_sample = ws_cache[first_sid]["X_val"]
        avail_tr  = [c for c in keep_feats if c in X_tr_sample.columns  and c not in drop_cols_tt]
        avail_val = [c for c in keep_feats if c in X_val_sample.columns and c not in drop_cols_tt]
        ws_cols[tt] = [c for c in avail_tr if c in avail_val]

    # 피험자별 타깃별 median 사전 계산
    ws_medians = {sid: {} for sid in subjects}
    for sid in subjects:
        for tt in TARGETS:
            cols_tt = ws_cols[tt]
            ws_medians[sid][tt] = ws_cache[sid]["X_tr"][cols_tt].median()

    # ── Phase 3: Optuna 탐색 (타깃별, 체크포인트 지원) ──
    print(f"\n=== Phase 3: Global Model Optuna ({N_TRIALS} trials/target) ===")

    # 이미 완료된 타깃 로드 (재실행 시 이어서 진행)
    existing = load_params(ET_KEY_OUT) or {}
    best_ws_params = dict(existing)
    skipped = [t for t in TARGETS if t in best_ws_params]
    if skipped:
        print(f"  체크포인트 복원: {skipped} 건너뜀")

    for t in TARGETS:
        if t in best_ws_params:
            print(f"  {t}: 이미 완료 (skip)")
            continue

        print(f"\n  타깃: {t}", flush=True)
        cols_t = ws_cols[t]

        def objective(trial, _t=t, _cols=cols_t):
            params = suggest_params(trial)
            all_y_true = []
            all_y_pred = []

            for sid in subjects:
                X_tr_t  = ws_cache[sid]["X_tr"][_cols]
                X_val_t = ws_cache[sid]["X_val"][_cols]
                tr_med  = ws_medians[sid][_t]

                model = ExtraTreesClassifier(**{**params, "random_state": WS_SEED})
                model.fit(X_tr_t.fillna(tr_med), ws_cache[sid]["y_tr"][_t])
                preds = model.predict_proba(X_val_t.fillna(tr_med))[:, 1]

                all_y_true.extend(ws_cache[sid]["y_val"][_t].tolist())
                all_y_pred.extend(np.clip(preds, 1e-6, 1 - 1e-6).tolist())

            return log_loss(all_y_true, all_y_pred, labels=[0, 1])

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

        best_ws_params[t] = {**study.best_params, **ET_FIXED}
        print(f"    best WS OOF LL: {study.best_value:.4f}  params: {study.best_params}", flush=True)

        # 타깃 완료 즉시 저장 (중단돼도 진행분 보존)
        save_params(ET_KEY_OUT, best_ws_params)
        print(f"    params 저장 완료 ({t})", flush=True)

    # ── Phase 3: 타깃별 WS OOF 기준선 vs 신규 비교 ──
    print("\n=== Phase 4: 타깃별 WS OOF 비교 (slim80 vs WS-optuna) ===")
    print(f"{'타깃':>4}  {'slim80':>8}  {'ws-opt':>8}  {'개선':>8}")
    print("-" * 36)
    for t in TARGETS:
        cols_t = ws_cols[t]
        all_y_true = []
        all_y_pred = []
        for sid in subjects:
            X_tr_t = ws_cache[sid]["X_tr"][cols_t]
            X_val_t = ws_cache[sid]["X_val"][cols_t]
            tr_med = ws_medians[sid][t]
            model = ExtraTreesClassifier(**{**best_ws_params[t], "random_state": WS_SEED})
            model.fit(X_tr_t.fillna(tr_med), ws_cache[sid]["y_tr"][t])
            preds = model.predict_proba(X_val_t.fillna(tr_med))[:, 1]
            all_y_true.extend(ws_cache[sid]["y_val"][t].tolist())
            all_y_pred.extend(np.clip(preds, 1e-6, 1 - 1e-6).tolist())
        new_ll = log_loss(all_y_true, all_y_pred, labels=[0, 1])
        base = baseline_ll.get(t, float("nan"))
        print(f"  {t}  {base:>8.4f}  {new_ll:>8.4f}  {base - new_ll:>+8.4f}")

    # ── Phase 5: 전체 데이터 Global 모델 ──
    print(f"\n=== Phase 5: 전체 데이터 Global 모델 ({len(SEEDS)} seeds, WS-optuna params) ===")
    et_test_global = {t: np.zeros(len(sample)) for t in TARGETS}
    for seed in SEEDS:
        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            cols_t_all = [c for c in keep_feats if c in X_all_z.columns and c not in drop_cols]
            X_all_t = X_all_z[cols_t_all]
            X_te_t  = X_te_z[[c for c in cols_t_all if c in X_te_z.columns]]
            all_med = X_all_t.median()
            model = ExtraTreesClassifier(**{**best_ws_params[t], "random_state": seed})
            model.fit(X_all_t.fillna(all_med), train_reset[t].values)
            et_test_global[t] += model.predict_proba(X_te_t.fillna(all_med))[:, 1] / len(SEEDS)
        print(f"  seed {seed} 완료")

    # ── Phase 6: WS OOF 수집 (캐시된 feature matrix 재사용) ──
    print(f"\n=== Phase 6: WS OOF 수집 (Global WS-opt + Personal depth=2) ===")
    ws_oof_global = {t: np.full(len(train_reset), np.nan) for t in TARGETS}
    ws_oof_pers   = {t: np.full(len(train_reset), np.nan) for t in TARGETS}

    for sid in subjects:
        val_pos = ws_splits[sid]["val_pos"]
        sid_mask_in_tr = ws_cache[sid]["sid_mask_in_tr"]

        for t in TARGETS:
            cols_t = ws_cols[t]
            X_tr_t  = ws_cache[sid]["X_tr"][cols_t]
            X_val_t = ws_cache[sid]["X_val"][cols_t]
            tr_med  = ws_medians[sid][t]
            X_tr_f  = X_tr_t.fillna(tr_med)
            X_val_f = X_val_t.fillna(tr_med)

            glb = ExtraTreesClassifier(**{**best_ws_params[t], "random_state": WS_SEED})
            glb.fit(X_tr_f, ws_cache[sid]["y_tr"][t])
            glb_preds = glb.predict_proba(X_val_f)[:, 1]

            X_pers = X_tr_f.iloc[sid_mask_in_tr]
            y_pers = ws_cache[sid]["y_tr"][t][sid_mask_in_tr]
            pers_preds = np.full(len(val_pos), 0.5)
            if len(np.unique(y_pers)) >= 2 and len(y_pers) >= 4:
                pers_m = ExtraTreesClassifier(
                    n_estimators=PERS_N_EST,
                    max_depth=PERS_DEPTH,
                    max_features=PERS_MAX_FEAT,
                    min_samples_leaf=PERS_MIN_LEAF,
                    criterion="entropy",
                    random_state=WS_SEED,
                )
                pers_m.fit(X_pers, y_pers)
                pers_preds = pers_m.predict_proba(X_val_f)[:, 1]

            for k, pos in enumerate(val_pos):
                ws_oof_global[t][pos] = glb_preds[k]
                ws_oof_pers[t][pos]   = pers_preds[k]

        print(f"  {sid} 완료")

    # ── Phase 7: alpha 최적화 ──
    print("\n=== Phase 7: 블렌드 alpha 최적화 ===")

    def blend_logloss(alpha):
        total_ll = 0.0
        for t in TARGETS:
            y_v = train_reset.loc[all_val_pos, t].values
            g_v = np.array([ws_oof_global[t][p] for p in all_val_pos])
            p_v = np.array([ws_oof_pers[t][p]   for p in all_val_pos])
            blended = np.clip(blend_probs(g_v, p_v, alpha), 1e-6, 1 - 1e-6)
            total_ll += log_loss(y_v, blended)
        return total_ll / len(TARGETS)

    alphas = np.linspace(0.0, 0.5, 21)
    grid_ll = [blend_logloss(a) for a in alphas]
    best_grid = alphas[int(np.argmin(grid_ll))]
    res = minimize_scalar(blend_logloss,
                          bounds=(max(0.0, best_grid - 0.05), min(1.0, best_grid + 0.05)),
                          method="bounded")
    best_alpha = float(res.x)
    best_ll    = float(res.fun)
    print(f"  alpha={best_alpha:.4f}, WS OOF blend LL={best_ll:.4f}")
    print(f"  참고: pers_grid_best WS OOF 0.6408 (동일 개인 모델)")

    # ── Phase 8: 피험자별 Personal 모델 (10 seeds) ──
    print(f"\n=== Phase 8: 피험자별 Personal 모델 ({len(SEEDS)} seeds) ===")
    et_test_pers = {t: np.zeros(len(sample)) for t in TARGETS}
    for sid in subjects:
        mask_te  = (sample["subject_id"] == sid).values
        sid_mask = (train_reset["subject_id"] == sid).values
        n_pers   = int(sid_mask.sum())
        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            cols_t = [c for c in keep_feats if c in X_all_z.columns and c not in drop_cols]
            all_med  = X_all_z[cols_t].median()
            X_pers   = X_all_z[cols_t][sid_mask].fillna(all_med)
            y_pers   = train_reset.loc[sid_mask, t].values
            X_te_sid = X_te_z[[c for c in cols_t if c in X_te_z.columns]][mask_te].fillna(all_med)
            if len(np.unique(y_pers)) < 2 or len(y_pers) < 4:
                et_test_pers[t][mask_te] = 0.5
                continue
            preds_sid = np.zeros(int(mask_te.sum()))
            for seed in SEEDS:
                m = ExtraTreesClassifier(
                    n_estimators=PERS_N_EST,
                    max_depth=PERS_DEPTH,
                    max_features=PERS_MAX_FEAT,
                    min_samples_leaf=PERS_MIN_LEAF,
                    criterion="entropy",
                    random_state=seed,
                )
                m.fit(X_pers, y_pers)
                preds_sid += m.predict_proba(X_te_sid)[:, 1] / len(SEEDS)
            et_test_pers[t][mask_te] = preds_sid
        print(f"  {sid}: {n_pers}일 완료")

    # ── Phase 9: 블렌드 + 편향 보정 ──
    print(f"\n=== Phase 9: 블렌드 (alpha={best_alpha:.4f}) + 편향 보정 ===")
    et_test_blend = {t: blend_probs(et_test_global[t], et_test_pers[t], best_alpha)
                     for t in TARGETS}

    biases = {}
    for sid in subjects:
        val_pos = ws_splits[sid]["val_pos"]
        biases[sid] = {}
        for t in TARGETS:
            y_val  = train_reset.loc[val_pos, t].values
            g_val  = np.array([ws_oof_global[t][p] for p in val_pos])
            p_val  = np.array([ws_oof_pers[t][p]   for p in val_pos])
            bl_val = np.clip(blend_probs(g_val, p_val, best_alpha), 1e-6, 1 - 1e-6)
            biases[sid][t] = 0.0 if len(np.unique(y_val)) < 2 else fit_logit_bias(bl_val, y_val)

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

    print("\n=== 최종 예측 분포 ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "et_gps_slim80_global_ws_optuna_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")
    print(f"WS OOF blend LL: {best_ll:.4f}  (alpha={best_alpha:.4f})")
    print(f"참고: pers_grid_best(depth=2,0.5,3) WS OOF 0.6408 대비 비교")


if __name__ == "__main__":
    main()
