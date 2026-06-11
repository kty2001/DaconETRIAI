"""
Per-subject Logistic Regression with Forward Label Features (Lead)

테스트가 보간(interpolation)이므로 테스트 날짜 이후의
훈련 레이블도 피처로 활용.

현재 LR: lag1/lag7/roll 등 과거 레이블만 사용
이 스크립트: lag + lead1/lead7 추가 (양방향 시간 컨텍스트)

lead1_{target}: 해당 날짜 직후 훈련 날짜의 레이블 (15일 이내)
lead7_{target}: +7일 근처 훈련 날짜의 레이블 (21일 이내)
lead_dist: lead1까지의 거리 (일)

MP OOF 평가: 중간 20%를 val로 사용 (보간 시나리오와 일치)
WS OOF 평가: 마지막 20%를 val로 사용 (외삽, lead 피처 없음)

출력:
  submission/logreg_lead_a{alpha}_blend_prob.csv
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import log_loss
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from gps_features import build_gps

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS   = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
VAL_RATIO = 0.20
C_GRID    = [0.001, 0.005, 0.01, 0.05, 0.1]
LEAD_WINDOW_DAYS = 15
LEAD7_WINDOW_DAYS = 21

ET_ENSEMBLE_FILE = "et_lgb_cb_xgb_ensemble_prob.csv"

DROP_COLS = [
    "subj_mean_Q1", "subj_mean_Q2", "subj_mean_Q3",
    "subj_mean_S1", "subj_mean_S2", "subj_mean_S3", "subj_mean_S4",
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
    "subject_enc", "subject_id",
]


def build_lead_features(query_df, ref_df):
    """
    query_df: 예측 대상 날짜 (subject_id, sleep_date 포함)
    ref_df:   레이블이 있는 참조 데이터 (훈련 데이터)

    반환: query 순서와 동일한 DataFrame (subject_id, date + lead 컬럼)
    """
    rows = []
    for _, qrow in query_df.iterrows():
        sid = qrow["subject_id"]
        qdate = pd.Timestamp(qrow["sleep_date"])
        feat = {"subject_id": sid, "date": qrow["sleep_date"]}

        # 피험자의 참조 데이터 중 query 날짜 이후인 것
        ref_sid = ref_df[ref_df["subject_id"] == sid].copy()
        ref_sid["_date"] = pd.to_datetime(ref_sid["sleep_date"])
        future = ref_sid[ref_sid["_date"] > qdate].sort_values("_date")

        if len(future) == 0:
            for t in TARGETS:
                feat[f"lead1_{t}"] = np.nan
                feat[f"lead7_{t}"] = np.nan
            feat["lead_dist"] = np.nan
        else:
            # lead1: 가장 가까운 미래 훈련일 (LEAD_WINDOW_DAYS 이내)
            nearest = future.iloc[0]
            dist1 = (nearest["_date"] - qdate).days
            feat["lead_dist"] = float(dist1)

            for t in TARGETS:
                feat[f"lead1_{t}"] = float(nearest[t]) if dist1 <= LEAD_WINDOW_DAYS else np.nan

            # lead7: +7일 근처 훈련일 (LEAD7_WINDOW_DAYS 이내)
            target7 = qdate + pd.Timedelta(days=7)
            cands = future[
                (future["_date"] >= qdate + pd.Timedelta(days=4)) &
                (future["_date"] <= qdate + pd.Timedelta(days=LEAD7_WINDOW_DAYS))
            ]
            if len(cands) > 0:
                closest7 = cands.iloc[
                    np.argmin(np.abs((cands["_date"] - target7).dt.days.values))
                ]
                for t in TARGETS:
                    feat[f"lead7_{t}"] = float(closest7[t])
            else:
                for t in TARGETS:
                    feat[f"lead7_{t}"] = np.nan

        rows.append(feat)

    return pd.DataFrame(rows)


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


def make_mp_splits(train_df, val_ratio=VAL_RATIO):
    splits = {}
    for sid, grp in train_df.groupby("subject_id"):
        sorted_pos = grp.sort_values("sleep_date").index.tolist()
        n = len(sorted_pos)
        n_val = max(1, int(n * val_ratio))
        mid = n // 2
        half = n_val // 2
        v_start = max(0, mid - half)
        v_end = v_start + n_val
        if v_end > n:
            v_end = n
            v_start = max(0, n - n_val)
        val_pos   = sorted_pos[v_start:v_end]
        train_pos = sorted_pos[:v_start] + sorted_pos[v_end:]
        splits[sid] = {"val_pos": val_pos, "train_pos": train_pos}
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


def build_X(df_query, parquet_combined, label_feat, lead_feat, sid):
    df = add_date_features(df_query.copy())
    parquet_sid = parquet_combined[parquet_combined["subject_id"] == sid].drop(
        columns=["subject_id"], errors="ignore"
    )
    df = df.merge(
        parquet_sid, left_on="lifelog_date", right_on="date", how="left"
    ).drop(columns=["date"], errors="ignore")
    df = df.merge(
        label_feat, left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"], how="left"
    ).drop(columns=["date"], errors="ignore")
    # lead 피처 추가
    df = df.merge(
        lead_feat, left_on=["subject_id", "sleep_date"],
        right_on=["subject_id", "date"], how="left"
    ).drop(columns=["date"], errors="ignore")
    df = apply_subj_zscore(df, sid, parquet_combined)
    drop = [c for c in DROP_COLS if c in df.columns]
    drop += [c for c in TARGETS if c in df.columns]
    drop += ["sleep_date", "lifelog_date"]
    X = df.drop(columns=drop, errors="ignore")
    return X.reset_index(drop=True)


def fit_predict_subj(X_tr, y_tr, X_te):
    from sklearn.linear_model import LogisticRegression
    valid = ~np.isnan(y_tr)
    if valid.sum() < 4 or len(np.unique(y_tr[valid])) < 2:
        return np.full(len(X_te), float(np.nanmean(y_tr)) if valid.sum() > 0 else 0.5)
    actual_folds = min(5, int((y_tr[valid]==0).sum()), int((y_tr[valid]==1).sum()))
    actual_folds = max(2, actual_folds)
    X_tr_arr = X_tr.fillna(0).values[valid]
    y_tr_arr = y_tr[valid].astype(int)
    X_te_arr = X_te.fillna(0).values
    try:
        clf = LogisticRegressionCV(
            Cs=C_GRID, cv=actual_folds, scoring="neg_log_loss",
            max_iter=2000, solver="lbfgs", random_state=42
        )
        clf.fit(X_tr_arr, y_tr_arr)
    except Exception:
        clf = LogisticRegression(C=0.01, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_tr_arr, y_tr_arr)
    return clf.predict_proba(X_te_arr)[:, 1]


def _empty_lead():
    df = pd.DataFrame({"subject_id": pd.Series([], dtype=str), "date": pd.Series([], dtype=str)})
    for t in TARGETS:
        df[f"lead1_{t}"] = pd.Series([], dtype=float)
        df[f"lead7_{t}"] = pd.Series([], dtype=float)
    df["lead_dist"] = pd.Series([], dtype=float)
    return df


def compute_oof(train_df, splits, parquet_combined, use_lead=True):
    subjects = sorted(train_df["subject_id"].unique())
    all_val_pos = [p for sid in subjects for p in splits[sid]["val_pos"]]
    oof = {t: np.full(len(train_df), np.nan) for t in TARGETS}

    for sid in subjects:
        val_pos = splits[sid]["val_pos"]
        non_val = sorted(set(train_df.index) - set(val_pos))

        ws_tr_df  = train_df.loc[non_val].reset_index(drop=True)
        ws_val_df = train_df.loc[val_pos].reset_index(drop=True)

        lf_tr  = build_label_features(ws_tr_df, ws_tr_df)
        lf_val = build_label_features(ws_val_df, ws_val_df)

        if use_lead:
            lead_tr  = build_lead_features(ws_tr_df[ws_tr_df["subject_id"]==sid], ws_tr_df)
            lead_val = build_lead_features(ws_val_df[ws_val_df["subject_id"]==sid], ws_tr_df)
        else:
            lead_tr  = _empty_lead()
            lead_val = _empty_lead()

        tr_sid = ws_tr_df[ws_tr_df["subject_id"] == sid].reset_index(drop=True)
        va_sid = ws_val_df[ws_val_df["subject_id"] == sid].reset_index(drop=True)

        X_tr = build_X(tr_sid, parquet_combined, lf_tr, lead_tr, sid)
        X_va = build_X(va_sid, parquet_combined, lf_val, lead_val, sid)

        sid_val_pos = [p for p in val_pos if train_df.loc[p, "subject_id"] == sid]

        for t in TARGETS:
            y_tr = tr_sid[t].values
            preds = fit_predict_subj(X_tr, y_tr, X_va)
            for k, pos in enumerate(sid_val_pos):
                if k < len(preds):
                    oof[t][pos] = preds[k]

    total_ll = 0.0
    for t in TARGETS:
        y_v = train_df.loc[all_val_pos, t].values
        p_v = np.clip(np.array([oof[t][p] for p in all_val_pos]), 1e-6, 1-1e-6)
        total_ll += log_loss(y_v, p_v)
    return total_ll / len(TARGETS)


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS 피처 집계 중 ===")
    parquet_feat     = build_parquet_features()
    gps_feat         = build_gps()
    parquet_combined = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    train_reset = train.reset_index(drop=True)
    subjects    = sorted(train["subject_id"].unique())
    ws_splits   = make_ws_splits(train_reset)
    mp_splits   = make_mp_splits(train_reset)

    print("=== Lead 피처 커버리지 ===")
    for sid in subjects:
        te_sid = sample[sample["subject_id"]==sid]
        lf_te = build_lead_features(te_sid, train)
        n_lead = lf_te["lead1_Q1"].notna().sum()
        print(f"  {sid}: {n_lead}/{len(te_sid)} ({n_lead/len(te_sid)*100:.0f}%) lead1 유효")
    print()

    print("=== WS OOF vs MP OOF 비교 (기준: LR without lead) ===")
    ws_nolead = compute_oof(train_reset, ws_splits, parquet_combined, use_lead=False)
    mp_nolead = compute_oof(train_reset, mp_splits, parquet_combined, use_lead=False)
    print(f"  LR (no lead): WS OOF={ws_nolead:.4f}, MP OOF={mp_nolead:.4f}")

    print()
    print("=== WS OOF vs MP OOF (with lead features) ===")
    ws_lead = compute_oof(train_reset, ws_splits, parquet_combined, use_lead=True)
    mp_lead = compute_oof(train_reset, mp_splits, parquet_combined, use_lead=True)
    print(f"  LR (with lead): WS OOF={ws_lead:.4f}, MP OOF={mp_lead:.4f}")
    print(f"  WS 변화: {ws_lead-ws_nolead:+.4f}, MP 변화: {mp_lead-mp_nolead:+.4f}")

    # ET 앙상블 로드
    et_path = SUBMISSION_DIR / ET_ENSEMBLE_FILE
    if not et_path.exists():
        print(f"ET 파일 없음: {et_path}")
        return
    et_pred = pd.read_csv(et_path)

    print("\n=== 테스트 예측 (lead 피처 사용) ===")
    lf_tr_all = build_label_features(train, train)
    lf_te_all = build_label_features(train, sample)

    test_preds = {t: np.full(len(sample), np.nan) for t in TARGETS}
    for sid in subjects:
        tr_sid = train[train["subject_id"] == sid].reset_index(drop=True)
        te_sid = sample[sample["subject_id"] == sid].reset_index(drop=True)
        mask   = (sample["subject_id"] == sid).values

        lf_tr_s  = build_label_features(tr_sid, tr_sid)
        lf_te_s  = build_label_features(tr_sid, te_sid)
        lead_tr_s = build_lead_features(tr_sid, train)
        lead_te_s = build_lead_features(te_sid, train)

        X_tr = build_X(tr_sid, parquet_combined, lf_tr_s, lead_tr_s, sid)
        X_te = build_X(te_sid, parquet_combined, lf_te_s, lead_te_s, sid)

        for t in TARGETS:
            y_tr = tr_sid[t].values
            preds = fit_predict_subj(X_tr, y_tr, X_te)
            test_preds[t][mask] = preds

        lead_cov = lead_te_s["lead1_Q1"].notna().sum()
        print(f"  {sid}: lead_cov={lead_cov}/{mask.sum()} 완료")

    # 저장
    logreg_lead = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        logreg_lead[t] = np.clip(test_preds[t], 0.1, 0.9)
    logreg_lead.to_csv(SUBMISSION_DIR / "logreg_lead_prob.csv", index=False)
    print("  저장: logreg_lead_prob.csv")

    for alpha in [0.1, 0.2, 0.3]:
        blend = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
        for t in TARGETS:
            blend[t] = np.clip(
                alpha * test_preds[t] + (1-alpha) * et_pred[t].values, 0.1, 0.9
            )
        fname = f"logreg_lead_a{int(alpha*10)}_blend_prob.csv"
        blend.to_csv(SUBMISSION_DIR / fname, index=False)
        print(f"  저장: {fname}")

    print()
    print("=== 요약 ===")
    print(f"  LR no-lead: WS={ws_nolead:.4f}, MP={mp_nolead:.4f}")
    print(f"  LR + lead:  WS={ws_lead:.4f}, MP={mp_lead:.4f}")
    print(f"  비교: 기존 logreg_et_a2 Public=0.5940")


if __name__ == "__main__":
    main()
