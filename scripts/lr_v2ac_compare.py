"""
LR 피처 버전 비교: v2 vs v2ac (mACStatus 추가)
WS OOF, MP OOF 비교로 AC 피처의 LR 기여도 평가
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import log_loss
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from gps_features import build_gps

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
C_GRID = [0.001, 0.005, 0.01, 0.05, 0.1]
VAL_RATIO = 0.20
DROP_COLS = [
    "subj_mean_Q1","subj_mean_Q2","subj_mean_Q3",
    "subj_mean_S1","subj_mean_S2","subj_mean_S3","subj_mean_S4",
    "usage_ms_morning","usage_ms_afternoon","usage_ms_evening",
    "usage_ms_presleep","usage_ms_sleep","usage_ms_total",
    "usage_apps_morning","usage_apps_afternoon","usage_apps_evening",
    "usage_apps_presleep","usage_apps_sleep",
    "usage_presleep_ratio","usage_sleep_ratio",
    "subject_enc","subject_id",
]


def make_ws_splits(train_df):
    splits = {}
    for sid, grp in train_df.groupby("subject_id"):
        pos = grp.sort_values("sleep_date").index.tolist()
        n = len(pos)
        nv = max(1, int(n * VAL_RATIO))
        splits[sid] = {"val_pos": pos[-nv:]}
    return splits


def make_mp_splits(train_df):
    splits = {}
    for sid, grp in train_df.groupby("subject_id"):
        pos = grp.sort_values("sleep_date").index.tolist()
        n = len(pos)
        nv = max(1, int(n * VAL_RATIO))
        mid = n // 2; half = nv // 2
        vs = max(0, mid - half); ve = vs + nv
        if ve > n: ve = n; vs = max(0, n - nv)
        splits[sid] = {"val_pos": pos[vs:ve]}
    return splits


def add_date_feats(df):
    dt = pd.to_datetime(df["sleep_date"])
    df = df.copy()
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def subj_zscore(df_q, sid, pf):
    subj_rows = pf[pf["subject_id"] == sid]
    sc = [c for c in pf.columns if c not in ("subject_id", "date")]
    stats = subj_rows[sc].agg(["mean", "std"])
    df = df_q.copy()
    for col in sc:
        if col not in df.columns:
            continue
        mu = stats.loc["mean", col]
        sg = stats.loc["std", col]
        if pd.isna(mu) or pd.isna(sg) or sg < 1e-9:
            continue
        df[col] = (df[col].astype(float) - mu) / sg
    return df


def build_X(df_q, pf, lf, sid):
    df = add_date_feats(df_q.copy())
    pf_sid = pf[pf["subject_id"] == sid].drop(columns=["subject_id"], errors="ignore")
    df = df.merge(pf_sid, left_on="lifelog_date", right_on="date", how="left").drop(columns=["date"], errors="ignore")
    df = df.merge(lf, left_on=["subject_id", "lifelog_date"], right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")
    df = subj_zscore(df, sid, pf)
    drop = [c for c in DROP_COLS if c in df.columns]
    drop += [c for c in TARGETS if c in df.columns]
    drop += ["sleep_date", "lifelog_date"]
    return df.drop(columns=drop, errors="ignore").reset_index(drop=True)


def fit_predict(X_tr, y_tr, X_va):
    valid = ~np.isnan(y_tr)
    if valid.sum() < 4 or len(np.unique(y_tr[valid])) < 2:
        return np.full(len(X_va), float(np.nanmean(y_tr)) if valid.sum() > 0 else 0.5)
    nf = min(5, int((y_tr[valid] == 0).sum()), int((y_tr[valid] == 1).sum()))
    nf = max(2, nf)
    try:
        clf = LogisticRegressionCV(Cs=C_GRID, cv=nf, scoring="neg_log_loss", max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_tr.fillna(0).values[valid], y_tr[valid].astype(int))
    except Exception:
        clf = LogisticRegression(C=0.01, max_iter=2000, solver="lbfgs", random_state=42)
        clf.fit(X_tr.fillna(0).values[valid], y_tr[valid].astype(int))
    return clf.predict_proba(X_va.fillna(0).values)[:, 1]


def compute_oof(train_r, splits, pf, subjects):
    all_val = [p for sid in subjects for p in splits[sid]["val_pos"]]
    oof = {t: np.full(len(train_r), np.nan) for t in TARGETS}
    for sid in subjects:
        vp = splits[sid]["val_pos"]
        nv = sorted(set(train_r.index) - set(vp))
        tr_df = train_r.loc[nv].reset_index(drop=True)
        va_df = train_r.loc[vp].reset_index(drop=True)
        lf_tr = build_label_features(tr_df, tr_df)
        lf_va = build_label_features(va_df, va_df)
        tr_sid = tr_df[tr_df["subject_id"] == sid].reset_index(drop=True)
        va_sid = va_df[va_df["subject_id"] == sid].reset_index(drop=True)
        X_tr = build_X(tr_sid, pf, lf_tr, sid)
        X_va = build_X(va_sid, pf, lf_va, sid)
        sid_vp = [p for p in vp if train_r.loc[p, "subject_id"] == sid]
        for t in TARGETS:
            preds = fit_predict(X_tr, tr_sid[t].values, X_va)
            for k, pos in enumerate(sid_vp):
                if k < len(preds):
                    oof[t][pos] = preds[k]
    total_ll = 0.0
    for t in TARGETS:
        y_v = train_r.loc[all_val, t].values
        p_v = np.clip([oof[t][p] for p in all_val], 1e-6, 1 - 1e-6)
        total_ll += log_loss(y_v, p_v)
    return total_ll / len(TARGETS)


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    train_r = train.reset_index(drop=True)
    subjects = sorted(train["subject_id"].unique())
    ws_splits = make_ws_splits(train_r)
    mp_splits = make_mp_splits(train_r)

    gps_feat = build_gps()

    for label, pf_module in [("v2", "parquet_features_v2"), ("v2ac", "parquet_features_v2ac")]:
        import importlib
        mod = importlib.import_module(pf_module)
        print(f"\n=== LR {label} 피처 집계 중 ===")
        pf = mod.build_all().merge(gps_feat, on=["subject_id", "date"], how="left")
        ws_ll = compute_oof(train_r, ws_splits, pf, subjects)
        mp_ll = compute_oof(train_r, mp_splits, pf, subjects)
        print(f"  {label}: WS OOF={ws_ll:.4f}, MP OOF={mp_ll:.4f}")


if __name__ == "__main__":
    main()
