"""
ET GPS Slim 80% Personal Model Parameter Grid Search

Global WS OOF은 한 번만 계산하고 개인 모델 파라미터(depth x max_feat x min_leaf)를 탐색.
최적 파라미터로 전체 파이프라인 실행 후 제출 파일 생성.

출력: submission/et_gps_slim80_pers_grid_best_prob.csv
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
import itertools

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

TARGETS    = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS      = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]
WS_SEED    = 42
VAL_RATIO  = 0.20
CALIB_REG  = 0.5
BIAS_BOUND = 2.0

PERS_N_EST = 50

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]

PARAM_GRID = list(itertools.product(
    [2, 3, 4],      # depth
    [0.2, 0.3, 0.5],  # max_features
    [2, 3],         # min_samples_leaf
))


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


def optimize_alpha(ws_oof_global, ws_oof_pers, all_val_pos, train_reset):
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
    return float(res.x), float(res.fun)


def compute_pers_oof(subjects, train_reset, ws_splits, X_all_z, keep_feats,
                     depth, max_feat, min_leaf):
    ws_oof_pers = {t: np.full(len(train_reset), np.nan) for t in TARGETS}
    for sid in subjects:
        val_pos     = ws_splits[sid]["val_pos"]
        all_non_val = sorted(set(train_reset.index) - set(val_pos))

        ws_tr_df  = train_reset.loc[all_non_val].reset_index(drop=True)
        ws_val_df = train_reset.loc[val_pos].reset_index(drop=True)

        lf_ws_tr  = build_label_features(ws_tr_df, ws_tr_df)
        lf_ws_val = build_label_features(ws_val_df, ws_val_df)

        from parquet_features_v2 import build_all as _bpf
        from gps_features import build_gps as _bgps
        # 피처는 이미 X_all_z에 있으므로 직접 슬라이싱
        sid_mask_tr  = train_reset.loc[all_non_val, "subject_id"].values
        sid_mask_val = train_reset.loc[val_pos, "subject_id"].values
        sid_pers_mask = (sid_mask_tr == sid)

        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            cols_t = [c for c in keep_feats if c in X_all_z.columns and c not in drop_cols]

            X_tr_t  = X_all_z.loc[all_non_val, cols_t]
            X_val_t = X_all_z.loc[val_pos,     cols_t]
            tr_med  = X_tr_t.median()
            X_tr_f  = X_tr_t.fillna(tr_med)
            X_val_f = X_val_t.fillna(tr_med)
            y_tr    = train_reset.loc[all_non_val, t].values

            X_pers = X_tr_f[sid_pers_mask]
            y_pers = y_tr[sid_pers_mask]
            pers_preds = np.full(len(val_pos), 0.5)
            if len(np.unique(y_pers)) >= 2 and len(y_pers) >= 4:
                m = ExtraTreesClassifier(
                    n_estimators=PERS_N_EST,
                    max_depth=depth,
                    max_features=max_feat,
                    min_samples_leaf=min_leaf,
                    criterion="entropy",
                    random_state=WS_SEED,
                )
                m.fit(X_pers, y_pers)
                pers_preds = m.predict_proba(X_val_f)[:, 1]

            for k, pos in enumerate(val_pos):
                ws_oof_pers[t][pos] = pers_preds[k]

    return ws_oof_pers


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

    gps_params  = load_params(ET_KEY_GPS)
    slim_params = load_params(ET_KEY_SLIM)
    if not gps_params or not slim_params:
        print("ERROR: params 캐시 없음.")
        return
    best_gps  = {t: gps_params[t]  for t in TARGETS}
    best_slim = {t: slim_params[t] for t in TARGETS}

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

    # ── Phase 1: 전체 데이터 피처 계산 ──
    print("\n=== Phase 1: 전체 데이터 피처 계산 ===")
    X_all_raw = build_features(train, train, parquet_feat, lf_all, True, le)
    X_te_raw  = build_features(sample, train, parquet_feat, lf_test, False, le)

    train_reset = train.reset_index(drop=True)

    sid_all = X_all_raw["subject_id"].reset_index(drop=True)
    sid_te  = X_te_raw["subject_id"].reset_index(drop=True)

    X_all_z = apply_zscore(sid_all, X_all_raw.drop(columns=["subject_id"]),
                           transductive_stats, sensor_cols)
    X_te_z  = apply_zscore(sid_te, X_te_raw.drop(columns=["subject_id"]),
                           transductive_stats, sensor_cols)
    X_all_z.index = train_reset.index

    ws_splits = make_ws_splits(train_reset)
    all_val_pos = [p for sid in subjects for p in ws_splits[sid]["val_pos"]]

    # ── Phase 2: Global WS OOF (한 번만 계산) ──
    print("\n=== Phase 2: Global WS OOF 계산 (1회) ===")
    ws_oof_global = {t: np.full(len(train_reset), np.nan) for t in TARGETS}

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
            glb = ExtraTreesClassifier(**{**best_slim[t], "random_state": WS_SEED})
            glb.fit(X_tr_t.fillna(tr_med), ws_tr_df[t].values)
            preds = glb.predict_proba(X_val_t.fillna(tr_med))[:, 1]
            for k, pos in enumerate(val_pos):
                ws_oof_global[t][pos] = preds[k]

    global_ll_0 = 0.0
    for t in TARGETS:
        y_v = train_reset.loc[all_val_pos, t].values
        g_v = np.array([ws_oof_global[t][p] for p in all_val_pos])
        global_ll_0 += log_loss(y_v, np.clip(g_v, 1e-6, 1 - 1e-6))
    global_ll_0 /= len(TARGETS)
    print(f"  Global only (alpha=0) WS OOF LL: {global_ll_0:.4f}")

    # ── Phase 3: 파라미터 Grid Search ──
    print(f"\n=== Phase 3: Personal Model Grid Search ({len(PARAM_GRID)}개 조합) ===")
    print(f"{'depth':>5}  {'max_f':>5}  {'m_leaf':>6}  {'alpha':>6}  {'WS OOF':>8}")
    print("-" * 42)

    grid_results = []

    for depth, max_feat, min_leaf in PARAM_GRID:
        ws_oof_pers = {t: np.full(len(train_reset), np.nan) for t in TARGETS}

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

            sid_mask_in_tr = (ws_tr_df["subject_id"].values == sid)

            for t in TARGETS:
                drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
                avail_tr  = [c for c in keep_feats if c in X_ws_tr_z.columns  and c not in drop_cols]
                avail_val = [c for c in keep_feats if c in X_ws_val_z.columns and c not in drop_cols]
                cols_t = [c for c in avail_tr if c in avail_val]
                X_tr_t  = X_ws_tr_z[cols_t]
                X_val_t = X_ws_val_z[cols_t]
                tr_med  = X_tr_t.median()
                X_tr_f  = X_tr_t.fillna(tr_med)
                X_val_f = X_val_t.fillna(tr_med)

                X_pers = X_tr_f[sid_mask_in_tr]
                y_pers = ws_tr_df.loc[sid_mask_in_tr, t].values
                pers_preds = np.full(len(val_pos), 0.5)
                if len(np.unique(y_pers)) >= 2 and len(y_pers) >= 4:
                    m = ExtraTreesClassifier(
                        n_estimators=PERS_N_EST,
                        max_depth=depth,
                        max_features=max_feat,
                        min_samples_leaf=min_leaf,
                        criterion="entropy",
                        random_state=WS_SEED,
                    )
                    m.fit(X_pers, y_pers)
                    pers_preds = m.predict_proba(X_val_f)[:, 1]

                for k, pos in enumerate(val_pos):
                    ws_oof_pers[t][pos] = pers_preds[k]

        best_alpha, best_ll = optimize_alpha(ws_oof_global, ws_oof_pers, all_val_pos, train_reset)
        grid_results.append({
            "depth": depth, "max_feat": max_feat, "min_leaf": min_leaf,
            "alpha": best_alpha, "ws_oof": best_ll,
            "ws_oof_pers": ws_oof_pers,
        })
        marker = " <-- baseline" if (depth == 3 and max_feat == 0.3 and min_leaf == 2) else ""
        print(f"  {depth:>3}  {max_feat:>5.1f}  {min_leaf:>6}  "
              f"{best_alpha:>6.4f}  {best_ll:>8.4f}{marker}")

    # ── Phase 4: Grid 결과 요약 ──
    print(f"\n=== Phase 4: Grid 결과 요약 ===")
    results_df = pd.DataFrame([
        {"depth": r["depth"], "max_feat": r["max_feat"], "min_leaf": r["min_leaf"],
         "alpha": r["alpha"], "ws_oof": r["ws_oof"]}
        for r in grid_results
    ]).sort_values("ws_oof")
    print(results_df.to_string(index=False))

    best_row = grid_results[int(np.argmin([r["ws_oof"] for r in grid_results]))]
    best_depth    = best_row["depth"]
    best_max_feat = best_row["max_feat"]
    best_min_leaf = best_row["min_leaf"]
    best_alpha    = best_row["alpha"]
    best_ws_oof   = best_row["ws_oof"]
    best_pers_oof = best_row["ws_oof_pers"]

    print(f"\n최적 파라미터: depth={best_depth}, max_feat={best_max_feat}, "
          f"min_leaf={best_min_leaf}, alpha={best_alpha:.4f}")
    print(f"최적 WS OOF LL: {best_ws_oof:.4f}  (baseline depth=3,0.3,2: "
          f"{next(r['ws_oof'] for r in grid_results if r['depth']==3 and r['max_feat']==0.3 and r['min_leaf']==2):.4f})")

    # ── Phase 5: 전체 데이터 Global 모델 ──
    print(f"\n=== Phase 5: 전체 데이터 Global 모델 ({len(SEEDS)} seeds) ===")
    et_test_global = {t: np.zeros(len(sample)) for t in TARGETS}
    for seed in SEEDS:
        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            cols_t = [c for c in keep_feats if c in X_all_z.columns and c not in drop_cols]
            X_all_t = X_all_z[cols_t]
            X_te_t  = X_te_z[[c for c in cols_t if c in X_te_z.columns]]
            all_med = X_all_t.median()
            model = ExtraTreesClassifier(**{**best_slim[t], "random_state": seed})
            model.fit(X_all_t.fillna(all_med), train_reset[t].values)
            et_test_global[t] += model.predict_proba(X_te_t.fillna(all_med))[:, 1] / len(SEEDS)
        print(f"  seed {seed} 완료")

    # ── Phase 6: 피험자별 Personal 모델 ──
    print(f"\n=== Phase 6: 피험자별 Personal 모델 ({len(SEEDS)} seeds, "
          f"depth={best_depth}, max_feat={best_max_feat}, min_leaf={best_min_leaf}) ===")
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
                    max_depth=best_depth,
                    max_features=best_max_feat,
                    min_samples_leaf=best_min_leaf,
                    criterion="entropy",
                    random_state=seed,
                )
                m.fit(X_pers, y_pers)
                preds_sid += m.predict_proba(X_te_sid)[:, 1] / len(SEEDS)
            et_test_pers[t][mask_te] = preds_sid
        print(f"  {sid}: {n_pers}일 완료")

    # ── Phase 7: 블렌드 ──
    print(f"\n=== Phase 7: 블렌드 (alpha={best_alpha:.4f}) ===")
    et_test_blend = {t: blend_probs(et_test_global[t], et_test_pers[t], best_alpha)
                     for t in TARGETS}

    # ── Phase 8: WS OOF 기반 편향 보정 ──
    print(f"\n=== Phase 8: 편향 보정 (REG={CALIB_REG}) ===")
    biases = {}
    for sid in subjects:
        val_pos = ws_splits[sid]["val_pos"]
        biases[sid] = {}
        for t in TARGETS:
            y_val  = train_reset.loc[val_pos, t].values
            g_val  = np.array([ws_oof_global[t][p] for p in val_pos])
            p_val  = np.array([best_pers_oof[t][p] for p in val_pos])
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

    out_path = SUBMISSION_DIR / "et_gps_slim80_pers_grid_best_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")
    print(f"WS OOF LL: {best_ws_oof:.4f}  (alpha={best_alpha:.4f})")
    print(f"파라미터: depth={best_depth}, max_feat={best_max_feat}, min_leaf={best_min_leaf}")
    print("참고: et_gps_slim80_personal_blend Public 0.5992 대비 비교 필요.")


if __name__ == "__main__":
    main()
