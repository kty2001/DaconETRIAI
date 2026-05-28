"""
ET GPS Slim 80% + Transductive Z-score + 피험자별 가변 REG 보정

et_gps_slim80_transductive.py 에서 단 하나만 변경:
  Phase 3 logit bias 보정의 REG를 피험자별로 가변 적용.
  훈련 날짜가 많은 피험자는 bias 추정이 안정 -> REG 낮게 (더 많이 보정 허용).
  훈련 날짜가 적은 피험자는 bias 추정이 불안정 -> REG 높게 (보수적 보정).

  reg = BASE_REG * (BASE_N / n_dates)
  BASE_REG = 0.5, BASE_N = 45 (전체 피험자 평균 훈련 날짜 수)
  ex) id04 57일 -> REG = 0.5 * 45/57 = 0.395
      id03 33일 -> REG = 0.5 * 45/33 = 0.682

출력: submission/et_gps_slim80_trans_varreg_prob.csv
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

BASE_REG   = 0.5
BASE_N     = 45    # 전체 피험자 평균 훈련 날짜 수 (기준값)
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


def compute_importance(train, parquet_feat, le, sensor_cols, fold_label_feats,
                       best_gps, transductive_stats):
    importance_dict = {t: [] for t in TARGETS}
    feature_names_ref = {}

    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr   = build_features(train_fold, train_fold, parquet_feat, lf_tr, True, le)
        sid_tr = X_tr["subject_id"].reset_index(drop=True)
        X_tr_z = apply_zscore(sid_tr, X_tr.drop(columns=["subject_id"]),
                              transductive_stats, sensor_cols)

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
                             fold_label_feats, label_feat_test, transductive_stats, keep_feats):
    fold_base = []
    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
        X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)
        sid_tr  = X_tr["subject_id"].reset_index(drop=True)
        sid_val = X_val["subject_id"].reset_index(drop=True)
        X_tr_z  = apply_zscore(sid_tr,  X_tr.drop(columns=["subject_id"]),
                               transductive_stats, sensor_cols)
        X_val_z = apply_zscore(sid_val, X_val.drop(columns=["subject_id"]),
                               transductive_stats, sensor_cols)
        fold_base.append((tr_idx, val_idx, X_tr_z, X_val_z))

    X_te_raw = build_features(test.copy(), train, parquet_feat, label_feat_test, False, le)
    sid_te   = X_te_raw["subject_id"].reset_index(drop=True)
    X_te_z   = apply_zscore(sid_te, X_te_raw.drop(columns=["subject_id"]),
                            transductive_stats, sensor_cols)

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


def fit_logit_bias(pred_oof, y_true, reg, bound=BIAS_BOUND):
    eps = 1e-6
    p = np.clip(pred_oof, eps, 1 - eps)
    lp = logit_fn(p)

    def obj(b):
        return log_loss(y_true, expit_fn(lp + b)) + reg * (b ** 2)

    res = minimize_scalar(obj, bounds=(-bound, bound), method="bounded")
    return float(res.x)


def apply_logit_bias(pred_raw, bias, eps=1e-6):
    p = np.clip(pred_raw, eps, 1 - eps)
    return expit_fn(logit_fn(p) + bias)


def train_and_predict(train, test, parquet_feat):
    le          = LabelEncoder().fit(train["subject_id"])
    groups      = train["subject_id"].values
    cv          = GroupKFold(n_splits=10)
    sensor_cols = get_sensor_cols(parquet_feat)
    n_seeds     = len(SEEDS)

    fold_indices = list(cv.split(np.zeros(len(train)), train[TARGETS[0]].values, groups))

    transductive_stats = compute_transductive_stats(parquet_feat, sensor_cols)
    print(f"  transductive 통계: {len(transductive_stats)}명, {len(sensor_cols)}개 센서 컬럼")

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

    gps_params = load_params(ET_KEY_GPS)
    if not gps_params:
        print("ERROR: extratrees_gps params 캐시 없음. extratrees_gps_ensemble.py를 먼저 실행하세요.")
        return None
    best_gps = {t: gps_params[t] for t in TARGETS}

    print("=== Phase 0: Feature Importance 계산 (slim 80%) ===")
    keep_feats = compute_importance(train, parquet_feat, le, sensor_cols,
                                    fold_label_feats, best_gps, transductive_stats)
    print()

    print("폴드 피처 사전 계산 중 (slim 80% + transductive z-score)...")
    fold_by_target, te_by_target = precompute_fold_features(
        train, test, parquet_feat, le, sensor_cols,
        fold_label_feats, label_feat_test, transductive_stats, keep_feats,
    )
    n_feats = fold_by_target[TARGETS[0]][0][2].shape[1]
    print(f"  완료 (피처 수: {n_feats}개)\n")

    slim_params = load_params(ET_KEY_SLIM)
    if slim_params:
        print(f"=== Phase 1: 저장된 params 로드 ({ET_KEY_SLIM}) ===")
        best_et = {t: slim_params[t] for t in TARGETS}
        for t in TARGETS:
            bp = best_et[t]
            print(f"  {t}: n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
                  f"max_feat={bp['max_features']:.4f}")
        print()
    else:
        print(f"ERROR: {ET_KEY_SLIM} params 캐시 없음. extratrees_gps_slim80_ensemble.py를 먼저 실행하세요.")
        return None

    print(f"=== Phase 2: ExtraTrees slim80 앙상블 ({n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 8))

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

        lls = [log_loss(train[t].values, et_oof[t] * n_seeds / (seed_i + 1))
               for t in TARGETS]
        ll_str = "  ".join(f"{ll:>6.4f}" for ll in lls)
        print(f"{seed:>6}  {ll_str}  {np.mean(lls):>6.4f}")

    print()
    print("=== Phase 2 최종 OOF (보정 전) ===")
    print(f"{'타깃':<5}  {'OOF LL':>8}")
    print("-" * 20)
    lls_before = {}
    for t in TARGETS:
        ll = log_loss(train[t].values, et_oof[t])
        lls_before[t] = ll
        print(f"{t:<5}  {ll:>8.4f}")
    print(f"{'평균':<5}  {np.mean(list(lls_before.values())):>8.4f}")

    # Phase 3: 피험자별 가변 REG 로짓 편향 보정
    subjects = sorted(train["subject_id"].unique())
    n_dates_map = train.groupby("subject_id").size().to_dict()

    print(f"\n=== Phase 3: 피험자별 가변 REG 로짓 편향 보정 ===")
    print(f"  BASE_REG={BASE_REG}, BASE_N={BASE_N}")
    print(f"  reg = BASE_REG * (BASE_N / n_dates)")
    print()
    print(f"  {'피험자':<8}  {'훈련일수':>6}  {'REG':>6}")
    print(f"  {'-'*26}")
    for sid in subjects:
        n_dates = n_dates_map[sid]
        reg = BASE_REG * (BASE_N / n_dates)
        print(f"  {sid:<8}  {n_dates:>6}d  {reg:>6.3f}")

    biases = {sid: {} for sid in subjects}
    for sid in subjects:
        mask    = (train["subject_id"] == sid).values
        n_dates = int(mask.sum())
        reg     = BASE_REG * (BASE_N / n_dates)
        for t in TARGETS:
            pred = et_oof[t][mask]
            y    = train[t].values[mask]
            if len(np.unique(y)) < 2:
                biases[sid][t] = 0.0
                continue
            biases[sid][t] = fit_logit_bias(pred, y, reg=reg)

    print(f"\n{'피험자':<8}  " + "  ".join(f"{t:>7}" for t in TARGETS) + "  평균|편향|")
    print("-" * (10 + 10 * len(TARGETS)))
    for sid in subjects:
        vals = [biases[sid][t] for t in TARGETS]
        val_str = "  ".join(f"{v:>+7.4f}" for v in vals)
        print(f"{sid:<8}  {val_str}  {np.mean(np.abs(vals)):>8.4f}")

    et_oof_cal = {t: np.zeros(len(train)) for t in TARGETS}
    for sid in subjects:
        mask = (train["subject_id"] == sid).values
        for t in TARGETS:
            et_oof_cal[t][mask] = apply_logit_bias(et_oof[t][mask], biases[sid][t])

    print("\n=== OOF LL 보정 전후 비교 ===")
    print(f"{'타깃':<5}  {'보정전':>8}  {'보정후':>8}  {'개선':>8}")
    print("-" * 38)
    lls_after = {}
    for t in TARGETS:
        ll_b = lls_before[t]
        ll_a = log_loss(train[t].values, et_oof_cal[t])
        lls_after[t] = ll_a
        diff = ll_b - ll_a
        sign = "[+]" if diff > 0 else "[-]"
        print(f"{t:<5}  {ll_b:>8.4f}  {ll_a:>8.4f}  {diff:>+7.4f} {sign}")
    avg_b = np.mean(list(lls_before.values()))
    avg_a = np.mean(list(lls_after.values()))
    print(f"{'평균':<5}  {avg_b:>8.4f}  {avg_a:>8.4f}  {avg_b-avg_a:>+7.4f}")
    print()
    print("주의: OOF 보정후 LL은 OOF 데이터에 과적합된 값으로 과낙관적 해석 금지.")
    print("      et_gps_slim80_transductive_prob (REG=0.5 균일) 와 비교: OOF 차이 확인.")

    # Phase 4: test 예측에 보정 적용
    et_test_cal = {t: et_test[t].copy() for t in TARGETS}
    for sid in subjects:
        mask_te = (test["subject_id"] == sid).values
        if mask_te.sum() == 0:
            continue
        for t in TARGETS:
            et_test_cal[t][mask_te] = apply_logit_bias(et_test[t][mask_te], biases[sid][t])

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = et_test_cal[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS 피처 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    print("=== ET GPS Slim 80% + Transductive Z-score + 피험자별 가변 REG 보정 ===\n")
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

    out_path = SUBMISSION_DIR / "et_gps_slim80_trans_varreg_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
