"""
ExtraTrees 준지도 학습 앙상블
- parquet v2 피처 + subject z-score (ET v2 동일 환경)
- extratrees_v2 캐시 params 사용
- 핵심 개선: val/test subject의 subj_mean을 센서 유사도 가중 평균으로 대체
    현재: NaN → ET가 train median으로 처리
    개선: 센서 프로파일이 유사한 train subject의 레이블 가중 평균
출력: submission/extratrees_semisup_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from optuna_params_io import load_params

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

ET_KEY  = "extratrees_v2"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

ET_LL_V2 = {
    "Q1": 0.6998, "Q2": 0.6478, "Q3": 0.6443,
    "S1": 0.6161, "S2": 0.6161, "S3": 0.6124, "S4": 0.6870,
}

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]

# 유사도 계산용 핵심 센서 피처 (NaN 적고 신뢰도 높은 피처)
SIM_FEATURE_PATTERNS = [
    "hr_mean", "hr_std", "hr_min_val", "hr_max_val",
    "screen_ratio", "pedo_steps", "light_mean", "amb_",
]

TEMPERATURE = 8.0  # softmax 온도 (높을수록 가장 유사한 subject에 집중)


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


def compute_subject_profile(df, parquet_feat):
    """subject별 parquet 센서 피처 평균 프로파일"""
    sensor_cols = [c for c in parquet_feat.columns if c not in ("subject_id", "date")]
    sim_cols = [c for c in sensor_cols
                if any(p in c for p in SIM_FEATURE_PATTERNS)]
    if not sim_cols:
        sim_cols = sensor_cols

    merged = df[["subject_id", "lifelog_date"]].merge(
        parquet_feat, left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"], how="left"
    )
    profile = merged.groupby("subject_id")[sim_cols].mean()
    return profile


def compute_sim_weights(query_profile, ref_profile, temperature=TEMPERATURE):
    """쿼리 subject → 참조 subject 간 softmax 유사도 가중치"""
    common_cols = query_profile.columns.intersection(ref_profile.columns)
    common_cols = [c for c in common_cols
                   if query_profile[c].notna().any() and ref_profile[c].notna().any()]
    if not common_cols:
        n_ref = len(ref_profile)
        return pd.DataFrame(
            np.ones((len(query_profile), n_ref)) / n_ref,
            index=query_profile.index, columns=ref_profile.index
        )

    scaler = StandardScaler()
    ref_vals   = ref_profile[common_cols].fillna(ref_profile[common_cols].mean())
    query_vals = query_profile[common_cols].fillna(ref_profile[common_cols].mean())

    ref_scaled   = scaler.fit_transform(ref_vals)
    query_scaled = scaler.transform(query_vals)

    sim = cosine_similarity(query_scaled, ref_scaled)  # (n_query, n_ref)
    # softmax
    sim_scaled = sim * temperature
    sim_scaled -= sim_scaled.max(axis=1, keepdims=True)
    weights = np.exp(sim_scaled)
    weights /= weights.sum(axis=1, keepdims=True)

    return pd.DataFrame(weights, index=query_profile.index, columns=ref_profile.index)


def compute_sim_subj_mean(query_ids, sim_weights, ref_label_mean):
    """유사도 가중 subj_mean 계산"""
    result = {}
    for t in TARGETS:
        ref_means = ref_label_mean[t]  # Series: subject_id → mean
        ref_vec   = sim_weights.columns.map(ref_means).values.astype(float)
        # NaN인 참조 subject는 전체 평균으로 대체
        nan_mask = np.isnan(ref_vec)
        if nan_mask.any():
            ref_vec[nan_mask] = np.nanmean(ref_vec)
        weighted = sim_weights.values @ ref_vec  # (n_query,)
        result[t] = pd.Series(weighted, index=sim_weights.index)
    return result


def add_date_features(df):
    dt = pd.to_datetime(df["sleep_date"])
    df = df.copy()
    df["day_of_week"]  = dt.dt.dayofweek
    df["month"]        = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def add_subject_mean_features(df, ref, is_train,
                               sim_subj_mean=None):
    """
    is_train=True : LOO subj_mean (기존과 동일)
    is_train=False: sim_subj_mean이 있으면 유사도 가중 평균, 없으면 단순 평균
    """
    subject_sum   = ref.groupby("subject_id")[TARGETS].sum()
    subject_count = ref.groupby("subject_id")[TARGETS].count()

    if is_train:
        for t in TARGETS:
            s_sum = df["subject_id"].map(subject_sum[t])
            s_cnt = df["subject_id"].map(subject_count[t])
            df[f"subj_mean_{t}"] = (s_sum - df[t]) / (s_cnt - 1).clip(lower=1)
    else:
        if sim_subj_mean is not None:
            for t in TARGETS:
                df[f"subj_mean_{t}"] = df["subject_id"].map(sim_subj_mean[t])
        else:
            subject_mean = subject_sum / subject_count
            for t in TARGETS:
                df[f"subj_mean_{t}"] = df["subject_id"].map(subject_mean[t])
    return df


def build_features(df, ref, parquet_feat, label_feat, is_train, le,
                   sim_subj_mean=None):
    df = add_date_features(df)
    df = add_subject_mean_features(df, ref, is_train, sim_subj_mean)
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
        # ── val subject 유사도 subj_mean 계산 ──────────────────────────────
        train_profile = compute_subject_profile(train_fold, parquet_feat)
        val_profile   = compute_subject_profile(val_fold,   parquet_feat)

        val_subjects  = val_fold["subject_id"].unique()
        query_profile = val_profile.reindex(val_subjects)
        sim_w         = compute_sim_weights(query_profile, train_profile)

        ref_label_mean = (train_fold.groupby("subject_id")[TARGETS].sum()
                          / train_fold.groupby("subject_id")[TARGETS].count())
        val_sim_subj_mean = compute_sim_subj_mean(val_subjects, sim_w, ref_label_mean)
        # subject_id → 값 매핑으로 변환
        val_sim_map = {t: val_sim_subj_mean[t] for t in TARGETS}

        # ── 피처 빌드 ────────────────────────────────────────────────────────
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
        X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le,
                               sim_subj_mean=val_sim_map)

        sid_tr  = X_tr["subject_id"].reset_index(drop=True)
        sid_val = X_val["subject_id"].reset_index(drop=True)

        tr_stats  = compute_subj_stats(sid_tr,  X_tr,  sensor_cols)
        val_stats = compute_subj_stats(sid_val, X_val, sensor_cols)

        X_tr_z  = apply_zscore(sid_tr,  X_tr.drop(columns=["subject_id"]),  tr_stats,  sensor_cols)
        X_val_z = apply_zscore(sid_val, X_val.drop(columns=["subject_id"]), val_stats, sensor_cols)

        fold_base.append((tr_idx, val_idx, X_tr_z, X_val_z))

    # ── test subject 유사도 subj_mean 계산 ────────────────────────────────
    train_profile_full = compute_subject_profile(train, parquet_feat)
    test_profile_full  = compute_subject_profile(test,  parquet_feat)
    test_subjects      = test["subject_id"].unique()
    query_profile_te   = test_profile_full.reindex(test_subjects)
    sim_w_te           = compute_sim_weights(query_profile_te, train_profile_full)
    ref_label_mean_full = (train.groupby("subject_id")[TARGETS].sum()
                           / train.groupby("subject_id")[TARGETS].count())
    test_sim_subj_mean = compute_sim_subj_mean(test_subjects, sim_w_te, ref_label_mean_full)
    test_sim_map = {t: test_sim_subj_mean[t] for t in TARGETS}

    X_te_raw = build_features(test.copy(), train, parquet_feat, label_feat_test, False, le,
                               sim_subj_mean=test_sim_map)
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

    print("유사도 기반 subj_mean 및 폴드 피처 사전 계산 중...")
    fold_by_target, te_by_target = precompute_fold_features(
        train, test, parquet_feat, le, sensor_cols,
        fold_label_feats, label_feat_test, full_stats,
    )
    print(f"  완료 (피처 수: {fold_by_target[TARGETS[0]][0][2].shape[1]}개)\n")

    # 유사도 정보 출력
    train_profile_full = compute_subject_profile(train, parquet_feat)
    test_profile_full  = compute_subject_profile(test,  parquet_feat)
    test_subjects      = test["subject_id"].unique()
    query_profile_te   = test_profile_full.reindex(test_subjects)
    sim_w_te           = compute_sim_weights(query_profile_te, train_profile_full)
    print("=== test subject 유사도 top-2 ===")
    for subj in test_subjects:
        row  = sim_w_te.loc[subj].sort_values(ascending=False)
        top2 = [(s, f"{w:.3f}") for s, w in row.head(2).items()]
        print(f"  {subj}: {top2[0][0]}({top2[0][1]}) / {top2[1][0]}({top2[1][1]})")
    print()

    cached_et = load_params(ET_KEY)
    if not cached_et:
        raise RuntimeError(f"캐시에 {ET_KEY} params 없음. extratrees_ensemble.py 먼저 실행하세요.")

    print(f"=== Phase 1: 캐시 params 로드 ({ET_KEY}) ===")
    for t in TARGETS:
        bp = cached_et[t]
        print(f"  {t}: n_est={bp['n_estimators']}, depth={bp['max_depth']}, "
              f"max_feat={bp['max_features']:.3f}")
    print()

    print(f"=== Phase 2: ET 준지도 앙상블 ({n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균F1   평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 16))

    et_oof  = {t: np.zeros(len(train)) for t in TARGETS}
    et_test = {t: np.zeros(len(test))  for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        seed_oof  = {t: np.zeros(len(train)) for t in TARGETS}
        seed_test = {t: np.zeros(len(test))  for t in TARGETS}

        for t in TARGETS:
            y = train[t].values
            et_params = {**cached_et[t], "random_state": seed}

            for tr_idx, val_idx, X_tr, X_val in fold_by_target[t]:
                tr_median = X_tr.median()
                et = ExtraTreesClassifier(**et_params)
                et.fit(X_tr.fillna(tr_median), y[tr_idx])
                seed_oof[t][val_idx]  = et.predict_proba(X_val.fillna(tr_median))[:, 1]
                seed_test[t]         += et.predict_proba(te_by_target[t].fillna(tr_median))[:, 1] / cv.n_splits

        for t in TARGETS:
            et_oof[t]  += seed_oof[t]  / n_seeds
            et_test[t] += seed_test[t] / n_seeds

        f1s, lls = [], []
        for t in TARGETS:
            cur = et_oof[t] * n_seeds / (seed_i + 1)
            f1s.append(f1_score(train[t].values, (cur > 0.5).astype(int)))
            lls.append(log_loss(train[t].values, cur))
        f1_str = "  ".join(f"{f:>6.3f}" for f in f1s)
        print(f"{seed:>6}  {f1_str}  {np.mean(f1s):>6.3f}  {np.mean(lls):>6.4f}")

    print()
    print(f"{'타깃':<5}  {'준지도 F1':>8}  {'준지도 LL':>9}  {'v2 LL':>7}  {'차이':>6}")
    print("-" * 46)
    for t in TARGETS:
        f1   = f1_score(train[t].values, (et_oof[t] > 0.5).astype(int))
        ll   = log_loss(train[t].values, et_oof[t])
        diff = ll - ET_LL_V2[t]
        sign = "+" if diff > 0 else ""
        print(f"{t:<5}  {f1:>8.3f}  {ll:>9.4f}  {ET_LL_V2[t]:>7.4f}  {sign}{diff:>5.4f}")

    avg_ll = np.mean([log_loss(train[t].values, et_oof[t]) for t in TARGETS])
    avg_f1 = np.mean([f1_score(train[t].values, (et_oof[t] > 0.5).astype(int)) for t in TARGETS])
    diff   = avg_ll - 0.6462
    sign   = "+" if diff > 0 else ""
    print(f"{'평균':<5}  {avg_f1:>8.3f}  {avg_ll:>9.4f}  {'0.6462':>7}  {sign}{diff:>5.4f}")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = et_test[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v2 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== ExtraTrees 준지도 학습 앙상블 ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "extratrees_semisup_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
