"""
ET GPS Slim 80% Cross-Target Stacking + Personal Blend

타깃 간 상관관계(수면 Q <-> 심리 S)를 2단계 스태킹으로 활용.

1단계: LOSO GroupKFold 10-fold OOF 예측 생성 (7 타깃 전체)
2단계: OOF 예측을 추가 피처로 사용하여 Personal Blend 재학습

데이터 유출 방지:
  훈련 ct 피처: LOSO OOF (피험자 단위 완전 분리)
  테스트 ct 피처: 1단계 전체 훈련 데이터로 학습한 모델의 예측

출력:
  submission/et_gps_slim80_cross_target_prob.csv
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

IMP_COVERAGE  = 0.80
TARGETS       = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS         = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]
WS_SEED       = 42
VAL_RATIO     = 0.20
CALIB_REG     = 0.5
BIAS_BOUND    = 2.0

# Personal model: pers_grid_best 최적값
PERS_N_EST    = 50
PERS_DEPTH    = 2
PERS_MAX_FEAT = 0.5
PERS_MIN_LEAF = 3

# 1단계 및 2단계 글로벌 모델: LOSO 최적 캐시 사용
STAGE1_KEY = "extratrees_gps"

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]

CT_COLS = [f"ct_{t}" for t in TARGETS]


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


def get_cols_for_target(t, keep_feats, X, extra_ct=True):
    drop_cols = [f"subj_mean_{t}"] + DROP_USAGE + ([f"ct_{t}"] if extra_ct else [])
    base = [c for c in keep_feats if c in X.columns and c not in drop_cols]
    # ct 피처: 다른 6개 타깃만 포함 (현재 타깃 제외)
    ct_other = [f"ct_{tt}" for tt in TARGETS if tt != t and f"ct_{tt}" in X.columns]
    return base + ct_other


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS feature build ===", flush=True)
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    le          = LabelEncoder().fit(train["subject_id"])
    sensor_cols = get_sensor_cols(parquet_feat)
    subjects    = sorted(train["subject_id"].unique())
    transductive_stats = compute_transductive_stats(parquet_feat, sensor_cols)

    print("label feature pre-build...", flush=True)
    lf_all  = build_label_features(train, train)
    lf_test = build_label_features(train, sample)
    print("  done", flush=True)

    gps_params = load_params(STAGE1_KEY)
    if not gps_params:
        print("ERROR: extratrees_gps params not found. Run et_gps_slim80 first.")
        return
    stage1_params = {t: gps_params[t] for t in TARGETS}

    # ── Phase 0: Feature Importance (ET LOSO 1 seed, slim 80%) ──
    print("\n=== Phase 0: Feature Importance (ET LOSO 1 seed) ===", flush=True)
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
            y_tr = train_fold[t].values
            if len(np.unique(y_tr)) < 2:
                importance_dict[t].append(np.zeros(len(feature_names_ref[t])))
                continue
            model = ExtraTreesClassifier(**{**stage1_params[t], "random_state": WS_SEED})
            model.fit(X_tr_t.fillna(X_tr_t.median()), y_tr)
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
    print(f"  top {IMP_COVERAGE*100:.0f}% coverage: {len(keep_feats)} features", flush=True)

    train_reset = train.reset_index(drop=True)
    ws_splits   = make_ws_splits(train_reset)
    all_val_pos = [p for sid in subjects for p in ws_splits[sid]["val_pos"]]

    # ── Phase 1: LOSO OOF for cross-target features ──
    print("\n=== Phase 1: LOSO OOF (cross-target features) ===", flush=True)
    ct_oof = np.zeros((len(train_reset), len(TARGETS)))

    for fold_i, (tr_idx, val_idx) in enumerate(fold_indices):
        train_fold = train_reset.iloc[tr_idx].copy()
        val_fold   = train_reset.iloc[val_idx].copy()

        lf_tr  = build_label_features(train_fold, train_fold)
        lf_val = build_label_features(val_fold, train_fold)

        X_tr_raw  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
        X_val_raw = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)

        sid_tr  = X_tr_raw["subject_id"].reset_index(drop=True)
        sid_val = X_val_raw["subject_id"].reset_index(drop=True)

        X_tr_z  = apply_zscore(sid_tr,  X_tr_raw.drop(columns=["subject_id"]),
                               transductive_stats, sensor_cols)
        X_val_z = apply_zscore(sid_val, X_val_raw.drop(columns=["subject_id"]),
                               transductive_stats, sensor_cols)

        for ti, t in enumerate(TARGETS):
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            cols_t = [c for c in keep_feats if c in X_tr_z.columns  and c not in drop_cols
                      and c in X_val_z.columns]
            X_tr_t  = X_tr_z[cols_t].fillna(X_tr_z[cols_t].median())
            X_val_t = X_val_z[cols_t].fillna(X_tr_z[cols_t].median())
            y_tr = train_fold[t].values
            if len(np.unique(y_tr)) < 2:
                ct_oof[val_idx, ti] = float(y_tr.mean())
                continue
            model = ExtraTreesClassifier(**{**stage1_params[t], "random_state": WS_SEED})
            model.fit(X_tr_t, y_tr)
            ct_oof[val_idx, ti] = model.predict_proba(X_val_t)[:, 1]

        print(f"  fold {fold_i + 1}/10 done", flush=True)

    # ct_oof -> DataFrame indexed like train_reset
    ct_train = pd.DataFrame(ct_oof, columns=CT_COLS, index=train_reset.index)
    print("  LOSO OOF done", flush=True)

    # ── Phase 2: Test ct features (full model) ──
    print("\n=== Phase 2: Test ct features (full model) ===", flush=True)
    X_all_raw = build_features(train, train, parquet_feat, lf_all, True, le)
    X_te_raw  = build_features(sample, train, parquet_feat, lf_test, False, le)

    sid_all = X_all_raw["subject_id"].reset_index(drop=True)
    sid_te  = X_te_raw["subject_id"].reset_index(drop=True)

    X_all_z = apply_zscore(sid_all, X_all_raw.drop(columns=["subject_id"]),
                           transductive_stats, sensor_cols)
    X_te_z  = apply_zscore(sid_te, X_te_raw.drop(columns=["subject_id"]),
                           transductive_stats, sensor_cols)
    X_all_z.index = train_reset.index

    ct_test = np.zeros((len(sample), len(TARGETS)))
    for ti, t in enumerate(TARGETS):
        drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
        cols_t = [c for c in keep_feats if c in X_all_z.columns and c not in drop_cols]
        X_all_t = X_all_z[cols_t]
        X_te_t  = X_te_z[[c for c in cols_t if c in X_te_z.columns]]
        all_med = X_all_t.median()
        preds_t = np.zeros(len(sample))
        for seed in SEEDS:
            m = ExtraTreesClassifier(**{**stage1_params[t], "random_state": seed})
            m.fit(X_all_t.fillna(all_med), train_reset[t].values)
            preds_t += m.predict_proba(X_te_t.fillna(all_med))[:, 1] / len(SEEDS)
        ct_test[:, ti] = preds_t
        print(f"  {t} test ct done", flush=True)

    ct_test_df = pd.DataFrame(ct_test, columns=CT_COLS)

    # ct 피처를 X_all_z, X_te_z에 추가
    X_all_z = pd.concat([X_all_z.reset_index(drop=True),
                         ct_train.reset_index(drop=True)], axis=1)
    X_all_z.index = train_reset.index
    X_te_z  = pd.concat([X_te_z.reset_index(drop=True), ct_test_df], axis=1)

    # ── Phase 3: WS split feature matrix pre-computation (ct 포함) ──
    print("\n=== Phase 3: WS split feature matrix pre-computation ===", flush=True)
    ws_cache  = {}
    ws_cols   = {}
    ws_medians = {}

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

        # ct 피처 추가 (LOSO OOF - 유출 없음)
        ct_tr  = ct_train.loc[all_non_val].reset_index(drop=True)
        ct_val = ct_train.loc[val_pos].reset_index(drop=True)
        X_ws_tr_z  = pd.concat([X_ws_tr_z.reset_index(drop=True),  ct_tr],  axis=1)
        X_ws_val_z = pd.concat([X_ws_val_z.reset_index(drop=True), ct_val], axis=1)

        ws_cache[sid] = {
            "X_tr":           X_ws_tr_z,
            "X_val":          X_ws_val_z,
            "y_tr":           {tt: ws_tr_df[tt].values  for tt in TARGETS},
            "y_val":          {tt: ws_val_df[tt].values for tt in TARGETS},
            "sid_mask_in_tr": np.where(ws_tr_df["subject_id"].values == sid)[0],
        }
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(subjects)} done", flush=True)

    print("  done", flush=True)

    # 타깃별 컬럼 목록 (slim + ct 다른 6개)
    first_sid = subjects[0]
    for tt in TARGETS:
        ws_cols[tt] = get_cols_for_target(
            tt, keep_feats, ws_cache[first_sid]["X_tr"], extra_ct=True
        )

    for sid in subjects:
        ws_medians[sid] = {}
        for tt in TARGETS:
            cols_tt = ws_cols[tt]
            ws_medians[sid][tt] = ws_cache[sid]["X_tr"][cols_tt].median()

    # ── Phase 4: WS OOF 수집 (Global + Personal) ──
    print("\n=== Phase 4: WS OOF (Global + Personal, ct features) ===", flush=True)
    ws_oof_global = {t: np.full(len(train_reset), np.nan) for t in TARGETS}
    ws_oof_pers   = {t: np.full(len(train_reset), np.nan) for t in TARGETS}

    for sid in subjects:
        val_pos        = ws_splits[sid]["val_pos"]
        sid_mask_in_tr = ws_cache[sid]["sid_mask_in_tr"]

        for t in TARGETS:
            cols_t  = ws_cols[t]
            X_tr_t  = ws_cache[sid]["X_tr"][cols_t]
            X_val_t = ws_cache[sid]["X_val"][cols_t]
            tr_med  = ws_medians[sid][t]
            X_tr_f  = X_tr_t.fillna(tr_med)
            X_val_f = X_val_t.fillna(tr_med)
            y_tr    = ws_cache[sid]["y_tr"][t]

            if len(np.unique(y_tr)) >= 2:
                glb = ExtraTreesClassifier(**{**stage1_params[t], "random_state": WS_SEED})
                glb.fit(X_tr_f, y_tr)
                glb_preds = glb.predict_proba(X_val_f)[:, 1]
            else:
                glb_preds = np.full(len(val_pos), float(y_tr.mean()))

            X_pers = X_tr_f.iloc[sid_mask_in_tr]
            y_pers = y_tr[sid_mask_in_tr]
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

        print(f"  {sid} done", flush=True)

    # ── Phase 5: WS OOF LL 출력 ──
    print("\n=== Phase 5: WS OOF LL (cross-target model) ===", flush=True)
    for t in TARGETS:
        y_v = train_reset.loc[all_val_pos, t].values
        g_v = np.array([ws_oof_global[t][p] for p in all_val_pos])
        ll = log_loss(y_v, np.clip(g_v, 1e-6, 1 - 1e-6))
        print(f"  {t}: {ll:.4f}", flush=True)

    # ── Phase 6: Alpha optimization ──
    print("\n=== Phase 6: Blend alpha optimization ===", flush=True)

    def blend_logloss(alpha):
        total_ll = 0.0
        for t in TARGETS:
            y_v = train_reset.loc[all_val_pos, t].values
            g_v = np.array([ws_oof_global[t][p] for p in all_val_pos])
            p_v = np.array([ws_oof_pers[t][p]   for p in all_val_pos])
            blended = np.clip(blend_probs(g_v, p_v, alpha), 1e-6, 1 - 1e-6)
            total_ll += log_loss(y_v, blended)
        return total_ll / len(TARGETS)

    alphas    = np.linspace(0.0, 0.5, 21)
    grid_ll   = [blend_logloss(a) for a in alphas]
    best_grid = alphas[int(np.argmin(grid_ll))]
    res = minimize_scalar(blend_logloss,
                          bounds=(max(0.0, best_grid - 0.05), min(1.0, best_grid + 0.05)),
                          method="bounded")
    best_alpha = float(res.x)
    best_ll    = float(res.fun)
    print(f"  alpha={best_alpha:.4f}, WS OOF blend LL={best_ll:.4f}", flush=True)

    # ── Phase 7: Global model (10 seeds) + Personal ET ──
    print(f"\n=== Phase 7: Global ET model ({len(SEEDS)} seeds, ct features) ===", flush=True)
    et_test_global = {t: np.zeros(len(sample)) for t in TARGETS}
    for seed in SEEDS:
        for t in TARGETS:
            cols_t  = get_cols_for_target(t, keep_feats, X_all_z, extra_ct=True)
            X_all_t = X_all_z[cols_t]
            X_te_t  = X_te_z[[c for c in cols_t if c in X_te_z.columns]]
            all_med = X_all_t.median()
            model = ExtraTreesClassifier(**{**stage1_params[t], "random_state": seed})
            model.fit(X_all_t.fillna(all_med), train_reset[t].values)
            et_test_global[t] += model.predict_proba(X_te_t.fillna(all_med))[:, 1] / len(SEEDS)
        print(f"  seed {seed} done", flush=True)

    print(f"\n=== Phase 7b: Personal ET ({len(SEEDS)} seeds) ===", flush=True)
    et_test_pers = {t: np.zeros(len(sample)) for t in TARGETS}
    for sid in subjects:
        mask_te  = (sample["subject_id"] == sid).values
        sid_mask = (train_reset["subject_id"] == sid).values
        n_pers   = int(sid_mask.sum())
        for t in TARGETS:
            cols_t  = get_cols_for_target(t, keep_feats, X_all_z, extra_ct=True)
            all_med = X_all_z[cols_t].median()
            X_pers  = X_all_z[cols_t][sid_mask].fillna(all_med)
            y_pers  = train_reset.loc[sid_mask, t].values
            X_te_s  = X_te_z[[c for c in cols_t if c in X_te_z.columns]][mask_te].fillna(all_med)
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
                preds_sid += m.predict_proba(X_te_s)[:, 1] / len(SEEDS)
            et_test_pers[t][mask_te] = preds_sid
        print(f"  {sid}: {n_pers} days done", flush=True)

    # ── Phase 8: Blend + Logit bias correction ──
    print(f"\n=== Phase 8: Blend (alpha={best_alpha:.4f}) + bias correction ===", flush=True)
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

    print("\n=== final prediction distribution ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "et_gps_slim80_cross_target_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(f"WS OOF blend LL: {best_ll:.4f}  (alpha={best_alpha:.4f})")


if __name__ == "__main__":
    main()
