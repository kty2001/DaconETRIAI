"""
2단계 스태킹
Stage 1: lgbm_multiseed와 동일 구조 → 7개 타깃 OOF 확률값 생성
Stage 2: Stage 1 OOF 예측값 7개를 피처로 추가 → 재학습
         타깃 간 상관관계(S1↔Q1, S3↔S4 등)를 2단계에서 학습
출력: submission/lgbm_stacking_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

LGBM_BASE_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "verbosity":         -1,
    "n_estimators":      200,
    "learning_rate":     0.05,
    "num_leaves":        15,
    "min_child_samples": 10,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
}

BASELINE_F1 = {
    "Q1": 0.649, "Q2": 0.666, "Q3": 0.726,
    "S1": 0.805, "S2": 0.636, "S3": 0.543, "S4": 0.624,
}


# ── z-score 헬퍼 ──────────────────────────────────────────────────────────────

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


# ── 피처 빌더 ─────────────────────────────────────────────────────────────────

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
    df = df.merge(
        parquet_feat,
        left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")
    df = df.merge(
        label_feat,
        left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"],
        how="left",
    ).drop(columns=["date"], errors="ignore")

    base_cols    = (
        ["subject_enc", "day_of_week", "month", "day_of_month", "is_weekend", "week_of_year"]
        + [f"subj_mean_{t}" for t in TARGETS]
    )
    parquet_cols = [c for c in parquet_feat.columns if c not in ("subject_id", "date")]
    label_cols   = [c for c in label_feat.columns   if c not in ("subject_id", "date")]

    all_cols = ["subject_id"] + base_cols + parquet_cols + label_cols
    all_cols = [c for c in all_cols if c in df.columns]
    return df[all_cols].reset_index(drop=True)


# ── fold 피처 사전 계산 (Stage1/2 공용) ──────────────────────────────────────

def precompute_fold_features(
    train, test, parquet_feat,
    fold_label_feats, label_feat_test,
    full_stats, sensor_cols, le,
):
    """
    fold별 X_tr_base, X_val_base, X_te_base 사전 계산 (z-score 적용, subject_id 제거)
    Stage 1/2 모두 이 결과를 재사용하여 중복 계산 방지
    """
    print("  fold 피처 사전 계산 중...")
    fold_feats = []
    for fi, (tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val) in enumerate(fold_label_feats):
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
        X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)
        X_te  = build_features(test.copy(), train,     parquet_feat, label_feat_test, False, le)

        sid_tr  = X_tr["subject_id"].reset_index(drop=True)
        sid_val = X_val["subject_id"].reset_index(drop=True)
        sid_te  = X_te["subject_id"].reset_index(drop=True)

        tr_stats  = compute_subj_stats(sid_tr,  X_tr,  sensor_cols)
        val_stats = compute_subj_stats(sid_val, X_val, sensor_cols)

        X_tr  = apply_zscore(sid_tr,  X_tr.drop(columns=["subject_id"]),  tr_stats,  sensor_cols)
        X_val = apply_zscore(sid_val, X_val.drop(columns=["subject_id"]), val_stats, sensor_cols)
        X_te  = apply_zscore(sid_te,  X_te.drop(columns=["subject_id"]),  full_stats, sensor_cols)

        fold_feats.append((tr_idx, val_idx, X_tr, X_val, X_te))
        print(f"    fold {fi+1}/10 완료", end="\r")
    print(f"  fold 피처 사전 계산 완료 ({len(fold_feats)} folds)      ")
    return fold_feats


# ── Stage 실행 공통 함수 ──────────────────────────────────────────────────────

def run_stage(
    stage_name, train, fold_feats,
    extra_train_feats,   # {t: array(len(train))} — Stage 2용 추가 피처 (None이면 Stage 1)
    extra_test_feats,    # {t: array(len(test))} — Stage 2용 추가 피처 (None이면 Stage 1)
):
    """
    LOSO × 멀티 시드 학습.
    extra_*_feats: Stage 2에서 Stage 1 예측값을 피처로 추가할 때 사용.
                  None이면 Stage 1 (추가 피처 없음).
    반환: (oof_accum, test_accum)
    """
    n_seeds = len(SEEDS)
    n_train = len(train)
    n_test  = fold_feats[0][4].shape[0]

    oof_accum  = {t: np.zeros(n_train) for t in TARGETS}
    test_accum = {t: np.zeros(n_test)  for t in TARGETS}

    hdr = f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균F1   평균LL"
    print(hdr)
    print("-" * len(hdr))

    for seed_i, seed in enumerate(SEEDS):
        params    = {**LGBM_BASE_PARAMS, "random_state": seed}
        seed_oof  = {t: np.zeros(n_train) for t in TARGETS}
        seed_test = {t: np.zeros(n_test)  for t in TARGETS}

        for t in TARGETS:
            y        = train[t].values
            drop_col = f"subj_mean_{t}"
            lag_drop = [f"lag1_{t}", f"lag2_{t}", f"roll3_{t}", f"roll7_{t}"]
            drop_cols = [drop_col] + lag_drop

            for tr_idx, val_idx, X_tr_base, X_val_base, X_te_base in fold_feats:
                X_tr  = X_tr_base.copy()
                X_val = X_val_base.copy()
                X_te  = X_te_base.copy()

                # Stage 2: 다른 타깃의 Stage 1 예측값 추가
                if extra_train_feats is not None:
                    for other_t in TARGETS:
                        X_tr[f"s1_{other_t}"]  = extra_train_feats[other_t][tr_idx]
                        X_val[f"s1_{other_t}"] = extra_train_feats[other_t][val_idx]
                        X_te[f"s1_{other_t}"]  = extra_test_feats[other_t]

                X_tr  = X_tr.drop(columns=drop_cols,  errors="ignore")
                X_val = X_val.drop(columns=drop_cols, errors="ignore")
                X_te  = X_te.drop(columns=drop_cols,  errors="ignore")

                model = lgb.LGBMClassifier(**params)
                model.fit(X_tr, y[tr_idx])
                seed_oof[t][val_idx]  = model.predict_proba(X_val)[:, 1]
                seed_test[t]         += model.predict_proba(X_te)[:, 1] / len(fold_feats)

        for t in TARGETS:
            oof_accum[t]  += seed_oof[t]  / n_seeds
            test_accum[t] += seed_test[t] / n_seeds

        # 시드별 중간 결과 출력
        cur_oof = {t: oof_accum[t] * n_seeds / (seed_i + 1) for t in TARGETS}
        f1s = [f1_score(train[t].values, (cur_oof[t] > 0.5).astype(int)) for t in TARGETS]
        lls = [log_loss(train[t].values, cur_oof[t]) for t in TARGETS]
        f1_str = "  ".join(f"{f:>6.3f}" for f in f1s)
        print(f"{seed:>6}  {f1_str}  {np.mean(f1s):>6.3f}  {np.mean(lls):>6.4f}")

    return oof_accum, test_accum


# ── 메인 ─────────────────────────────────────────────────────────────────────

def train_and_predict(train, test, parquet_feat):
    le          = LabelEncoder().fit(train["subject_id"])
    groups      = train["subject_id"].values
    cv          = GroupKFold(n_splits=10)
    sensor_cols = get_sensor_cols(parquet_feat)

    dummy_X      = np.zeros(len(train))
    fold_indices = list(cv.split(dummy_X, train[TARGETS[0]].values, groups))

    # ── 레이블 피처 & z-score 통계 사전 계산 ─────────────────────────────────
    print("=== 공통 피처 사전 계산 ===")
    label_feat_test = build_label_features(train, test)

    fold_label_feats = []
    for tr_idx, val_idx in fold_indices:
        train_fold = train.iloc[tr_idx].copy()
        val_fold   = train.iloc[val_idx].copy()
        lf_tr  = build_label_features(train_fold, train_fold)
        lf_val = build_label_features(val_fold,   val_fold)
        fold_label_feats.append((tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val))

    lf_full    = build_label_features(train, train)
    X_full     = build_features(train, train, parquet_feat, lf_full, True, le)
    full_stats = compute_subj_stats(X_full["subject_id"], X_full, sensor_cols)

    fold_feats = precompute_fold_features(
        train, test, parquet_feat,
        fold_label_feats, label_feat_test,
        full_stats, sensor_cols, le,
    )
    print()

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    print(f"=== Stage 1: 기본 모델 ({len(SEEDS)}개 시드) ===")
    s1_oof, s1_test = run_stage(
        "Stage1", train, fold_feats,
        extra_train_feats=None,
        extra_test_feats=None,
    )

    print()
    print("--- Stage 1 최종 OOF ---")
    print(f"{'타깃':<5}  {'F1':>6}  {'LogLoss':>8}")
    print("-" * 24)
    s1_f1s, s1_lls = [], []
    for t in TARGETS:
        f1 = f1_score(train[t].values, (s1_oof[t] > 0.5).astype(int))
        ll = log_loss(train[t].values, s1_oof[t])
        s1_f1s.append(f1); s1_lls.append(ll)
        print(f"{t:<5}  {f1:>6.3f}  {ll:>8.4f}")
    print(f"{'평균':<5}  {np.mean(s1_f1s):>6.3f}  {np.mean(s1_lls):>8.4f}")
    print()

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    print(f"=== Stage 2: 스태킹 모델 ({len(SEEDS)}개 시드) ===")
    s2_oof, s2_test = run_stage(
        "Stage2", train, fold_feats,
        extra_train_feats=s1_oof,
        extra_test_feats=s1_test,
    )

    print()
    print(f"{'타깃':<5}  {'S1 F1':>6}  {'S2 F1':>6}  {'변화':>6}  {'S2 LL':>8}  {'기준F1':>6}  {'차이':>6}")
    print("-" * 60)
    for t in TARGETS:
        f1_s1 = f1_score(train[t].values, (s1_oof[t] > 0.5).astype(int))
        f1_s2 = f1_score(train[t].values, (s2_oof[t] > 0.5).astype(int))
        ll_s2 = log_loss(train[t].values, s2_oof[t])
        base  = BASELINE_F1[t]
        diff  = f1_s2 - base
        sign  = "+" if diff >= 0 else ""
        chg   = f1_s2 - f1_s1
        chg_s = "+" if chg >= 0 else ""
        print(f"{t:<5}  {f1_s1:>6.3f}  {f1_s2:>6.3f}  {chg_s}{chg:>5.3f}  {ll_s2:>8.4f}  {base:>6.3f}  {sign}{diff:>5.3f}")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = s2_test[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v2 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    result = train_and_predict(train, sample, parquet_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "lgbm_stacking_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
