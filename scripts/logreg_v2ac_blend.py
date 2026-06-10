"""
피험자별 독립 Ridge Logistic Regression

각 피험자 S, 타깃 T: LogisticRegressionCV on S의 training rows만 사용
  - LOSO 패러다임 완전 탈피: 피험자 S의 모델은 S 데이터만 사용
  - Q1 고착 해결: intercept가 S의 기준 확률 직접 학습 (subj_mean NaN 문제 없음)
  - LogisticRegressionCV: 5-fold CV로 최적 C 자동 선택 [0.001, 0.01, 0.1]
  - 강한 L2 정규화 → 소표본(30~57행) 과적합 방지

피처:
  - sensor z-scores (parquet v2 + GPS, transductive within subject)
  - date features (day_of_week, month, is_weekend, day_of_month)
  - lag/roll label features (S의 과거 레이블)
  - 제외: subj_mean (intercept에 흡수), subject_enc, mUsageStats

출력:
  submission/per_subject_logreg_prob.csv           (독립 예측)
  submission/per_subject_logreg_et_blend_prob.csv  (ET+LGB+CB+XGB 앙상블과 blend)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2ac import build_all as build_parquet_features
from gps_features import build_gps

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS    = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
VAL_RATIO  = 0.20
C_GRID     = [0.001, 0.005, 0.01, 0.05, 0.1]
CV_FOLDS   = 5

ET_ENSEMBLE_FILE = "et_lgb_cb_xgb_ensemble_prob.csv"

DROP_COLS = [
    # subj_mean: intercept에 흡수
    "subj_mean_Q1", "subj_mean_Q2", "subj_mean_Q3",
    "subj_mean_S1", "subj_mean_S2", "subj_mean_S3", "subj_mean_S4",
    # mUsageStats
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
    # 식별자
    "subject_enc", "subject_id",
]


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


def add_date_features(df):
    dt = pd.to_datetime(df["sleep_date"])
    df = df.copy()
    df["day_of_week"]  = dt.dt.dayofweek
    df["month"]        = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def apply_subj_zscore(df_query, sid, parquet_combined):
    """피험자 S의 train+test 전체 통계로 센서 z-score 정규화."""
    subj_rows = parquet_combined[parquet_combined["subject_id"] == sid]
    sensor_cols = [c for c in parquet_combined.columns if c not in ("subject_id", "date")]
    stats = subj_rows[sensor_cols].agg(["mean", "std"])
    df = df_query.copy()
    for col in sensor_cols:
        if col not in df.columns:
            continue
        mu  = stats.loc["mean", col]
        sig = stats.loc["std", col]
        if pd.isna(mu) or pd.isna(sig) or sig < 1e-9:
            continue
        df[col] = (df[col].astype(float) - mu) / sig
    return df


def build_X(df_query, ref_train, parquet_combined, label_feat, sid):
    """피처 행렬 구성: date + sensor(z-scored) + lag/roll. subject_mean 제외."""
    df = add_date_features(df_query.copy())
    parquet_sid = parquet_combined[parquet_combined["subject_id"] == sid].drop(
        columns=["subject_id"], errors="ignore"
    )
    df = df.merge(
        parquet_sid, left_on="lifelog_date", right_on="date", how="left"
    ).drop(columns=["date"], errors="ignore")
    df = df.merge(label_feat, left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")
    # z-score within subject
    df = apply_subj_zscore(df, sid, parquet_combined)
    # 불필요 컬럼 제거
    drop = [c for c in DROP_COLS if c in df.columns]
    drop += [c for c in TARGETS if c in df.columns]
    drop += ["sleep_date", "lifelog_date"]
    X = df.drop(columns=drop, errors="ignore")
    return X.reset_index(drop=True)


def fit_predict_subject(X_tr, y_tr, X_te, target, n_folds=CV_FOLDS):
    """
    LogisticRegressionCV로 C 선택 후 예측.
    y에 두 클래스가 없으면 상수 예측 반환.
    """
    valid = ~np.isnan(y_tr)
    if valid.sum() < 4 or len(np.unique(y_tr[valid])) < 2:
        return np.full(len(X_te), float(np.nanmean(y_tr)) if valid.sum() > 0 else 0.5)

    X_tr_f = X_tr.fillna(0).values
    X_te_f = X_te.fillna(0).values
    y_tr_f = y_tr[valid].astype(int)
    X_tr_f = X_tr_f[valid]

    # fold 수를 클래스 수, 행 수에 맞게 조정
    actual_folds = min(n_folds, int(np.sum(y_tr_f == 0)), int(np.sum(y_tr_f == 1)))
    actual_folds = max(2, actual_folds)

    clf = LogisticRegressionCV(
        Cs=C_GRID,
        cv=actual_folds,
        scoring="neg_log_loss",
        max_iter=2000,
        solver="lbfgs",
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X_tr_f, y_tr_f)
    return clf.predict_proba(X_te_f)[:, 1]


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS 피처 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_combined = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    subjects    = sorted(train["subject_id"].unique())
    train_reset = train.reset_index(drop=True)
    ws_splits   = make_ws_splits(train_reset)

    print("=== Per-subject Logistic Regression ===\n")
    print(f"  피험자: {len(subjects)}명  타깃: {len(TARGETS)}개  C grid: {C_GRID}")
    print(f"  모델 수: {len(subjects) * len(TARGETS)}개 (CV_FOLDS={CV_FOLDS})\n")

    # ── WS OOF 수집 ──
    print("=== WS OOF 수집 (최적 C 리포트 포함) ===")
    ws_oof = {t: np.full(len(train_reset), np.nan) for t in TARGETS}

    for sid in subjects:
        val_pos   = ws_splits[sid]["val_pos"]
        non_val   = sorted(set(train_reset.index) - set(val_pos))

        ws_tr_df  = train_reset.loc[non_val].reset_index(drop=True)
        ws_val_df = train_reset.loc[val_pos].reset_index(drop=True)

        # label features: ws_tr 기준
        lf_tr  = build_label_features(ws_tr_df, ws_tr_df)
        lf_val = build_label_features(ws_val_df, ws_val_df)

        X_tr  = build_X(ws_tr_df[ws_tr_df["subject_id"] == sid],
                        ws_tr_df, parquet_combined, lf_tr, sid)
        X_val = build_X(ws_val_df[ws_val_df["subject_id"] == sid],
                        ws_tr_df, parquet_combined, lf_val, sid)

        # val_df 중 sid 행의 원래 index 추출
        sid_val_local = [i for i, p in enumerate(val_pos)
                         if train_reset.loc[p, "subject_id"] == sid]

        best_Cs = {}
        for t in TARGETS:
            y_tr  = ws_tr_df[ws_tr_df["subject_id"] == sid][t].values
            y_val = ws_val_df[ws_val_df["subject_id"] == sid][t].values

            preds = fit_predict_subject(X_tr, y_tr, X_val, t)

            valid = ~np.isnan(y_val)
            for k, pos in enumerate(val_pos):
                if train_reset.loc[pos, "subject_id"] == sid:
                    ws_oof[t][pos] = preds[k] if k < len(preds) else np.nan

            # best C 추적 (내부 LogisticRegressionCV에서 직접 꺼내기 어려우므로 skip)

        # OOF LL for this subject
        sid_val_pos = [p for p in val_pos if train_reset.loc[p, "subject_id"] == sid]
        if sid_val_pos:
            lls = []
            for t in TARGETS:
                y_v = train_reset.loc[sid_val_pos, t].values
                p_v = np.clip(np.array([ws_oof[t][p] for p in sid_val_pos]), 1e-6, 1 - 1e-6)
                if len(np.unique(y_v)) >= 2:
                    lls.append(log_loss(y_v, p_v))
            ll_str = f"{np.mean(lls):.4f}" if lls else "N/A"
            print(f"  {sid}: val={len(sid_val_pos)}행, 평균 WS OOF LL={ll_str}")

    # 전체 WS OOF LL
    all_val_pos = [p for sid in subjects for p in ws_splits[sid]["val_pos"]]
    total_ll = 0.0
    print("\n  타깃별 WS OOF LL:")
    for t in TARGETS:
        y_v  = train_reset.loc[all_val_pos, t].values
        p_v  = np.clip(np.array([ws_oof[t][p] for p in all_val_pos]), 1e-6, 1 - 1e-6)
        ll   = log_loss(y_v, p_v)
        total_ll += ll
        print(f"    {t}: {ll:.4f}")
    avg_ll = total_ll / len(TARGETS)
    print(f"    평균: {avg_ll:.4f}")
    print(f"\n  비교: et_gps_slim80_personal_blend WS OOF ~0.6493")

    # ── 최종 테스트 예측 (전체 training 사용) ──
    print("\n=== 최종 테스트 예측 (100% training) ===")
    lf_all  = build_label_features(train, train)
    lf_test = build_label_features(train, sample)

    test_preds = {t: np.full(len(sample), np.nan) for t in TARGETS}

    for sid in subjects:
        tr_df  = train_reset[train_reset["subject_id"] == sid]
        te_df  = sample[sample["subject_id"] == sid].reset_index(drop=True)
        mask   = (sample["subject_id"] == sid).values

        lf_tr_sid  = build_label_features(tr_df, tr_df)
        lf_te_sid  = build_label_features(tr_df, te_df)

        X_tr = build_X(tr_df, train_reset, parquet_combined, lf_tr_sid, sid)
        X_te = build_X(te_df, train_reset, parquet_combined, lf_te_sid, sid)

        for t in TARGETS:
            y_tr = tr_df[t].values
            preds = fit_predict_subject(X_tr, y_tr, X_te, t)
            test_preds[t][mask] = preds

        print(f"  {sid}: {len(tr_df)}행 학습 -> {int(mask.sum())}행 예측 완료")

    # 저장
    logreg_result = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        logreg_result[t] = np.clip(test_preds[t], 0.1, 0.9)

    print("\n  예측 분포 (clip 0.1~0.9):")
    for t in TARGETS:
        print(f"    {t}: mean={logreg_result[t].mean():.3f}, "
              f"min={logreg_result[t].min():.3f}, max={logreg_result[t].max():.3f}")

    standalone_path = SUBMISSION_DIR / "per_subject_logreg_prob.csv"
    logreg_result.to_csv(standalone_path, index=False)
    print(f"\n  저장: {standalone_path}")

    # ── ET 앙상블과 블렌드 ──
    et_path = SUBMISSION_DIR / ET_ENSEMBLE_FILE
    if not et_path.exists():
        print(f"  ET 파일 없음: {et_path}")
        return

    et_pred = pd.read_csv(et_path)

    # alpha 그리드: LogReg WS OOF vs ET 예측 (ET WS OOF 없으므로 alpha=0.1~0.4 범위 저장)
    print("\n=== ET 앙상블과 블렌드 ===")
    print(f"  LogReg WS OOF LL: {avg_ll:.4f}")
    print(f"  ET+LGB+CB+XGB Public: 0.5955")

    for alpha in [0.1, 0.2, 0.3, 0.4]:
        blend = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
        for t in TARGETS:
            lr_t = test_preds[t]
            et_t = et_pred[t].values if t in et_pred.columns else np.full(len(sample), 0.5)
            blend[t] = np.clip(alpha * lr_t + (1 - alpha) * et_t, 0.1, 0.9)
        path = SUBMISSION_DIR / f"logreg_v2ac_et_a{int(alpha*10)}_blend_prob.csv"
        blend.to_csv(path, index=False)
        print(f"  저장 (alpha={alpha}): {path}")

    print(f"\n=== 완료 ===")
    print(f"  LogReg 단독 WS OOF LL: {avg_ll:.4f}")
    print(f"  권장 제출: logreg_v2ac_et_a2_blend_prob.csv (LogReg 20% + ET 80%)")


if __name__ == "__main__":
    main()
