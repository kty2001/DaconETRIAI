"""
GP Temporal Blend

Per-subject Gaussian Process Regressor temporal interpolation.
For each (subject, target): fit GPR on (day_number -> binary_label).

Test dates are interleaved with training dates (same period, different specific days),
making this a genuine temporal interpolation task where GP excels.

output: submission/gp_temporal_blend_prob.csv
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import log_loss
from scipy.optimize import minimize_scalar
from scipy.special import expit as expit_fn, logit as logit_fn
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS    = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
VAL_RATIO  = 0.20
CALIB_REG  = 0.5
BIAS_BOUND = 2.0

# Length scale candidates (days); test dates interleave with train -> 7-45 appropriate
LENGTH_SCALE_CANDIDATES = [7, 14, 21, 30, 45, 60]
GP_NOISE = 0.25  # Bernoulli noise level at p=0.5

REF_DATE = pd.Timestamp("2024-01-01")


def dates_to_days(dates):
    return ((pd.to_datetime(dates) - REF_DATE).dt.days).values.astype(float)


def fit_gpr_predict(x_train, y_train, x_test, length_scale=30.0, noise=GP_NOISE):
    """
    GPR on binary labels. Normalizes x over train+test range.
    Returns approximate probabilities clipped to [0.05, 0.95].
    """
    unique = np.unique(y_train)
    if len(unique) < 2:
        base = float(unique[0]) if len(unique) > 0 else 0.5
        return np.full(len(x_test), 0.8 * base + 0.1)

    x_all   = np.concatenate([x_train, x_test])
    x_min   = x_all.min()
    x_range = max(x_all.max() - x_min, 1.0)
    x_tr_n  = (x_train - x_min) / x_range
    x_te_n  = (x_test  - x_min) / x_range
    ls_n    = length_scale / x_range

    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
        length_scale=ls_n, length_scale_bounds="fixed"
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel, alpha=noise, normalize_y=True, optimizer=None
    )
    gpr.fit(x_tr_n.reshape(-1, 1), y_train.astype(float))
    pred = gpr.predict(x_te_n.reshape(-1, 1))
    return np.clip(pred, 0.05, 0.95)


def make_ws_splits(train_df, val_ratio=VAL_RATIO):
    splits = {}
    for sid, grp in train_df.groupby("subject_id"):
        sorted_pos = grp.sort_values("sleep_date").index.tolist()
        n     = len(sorted_pos)
        n_val = max(1, int(n * val_ratio))
        splits[sid] = {
            "val_pos":   sorted_pos[-n_val:],
            "train_pos": sorted_pos[:-n_val],
        }
    return splits


def fit_logit_bias(pred_oof, y_true, reg=CALIB_REG, bound=BIAS_BOUND):
    eps = 1e-6
    p   = np.clip(pred_oof, eps, 1 - eps)
    lp  = logit_fn(p)
    def obj(b):
        return log_loss(y_true, expit_fn(lp + b)) + reg * (b ** 2)
    res = minimize_scalar(obj, bounds=(-bound, bound), method="bounded")
    return float(res.x)


def apply_logit_bias(pred_raw, bias, eps=1e-6):
    p = np.clip(pred_raw, eps, 1 - eps)
    return expit_fn(logit_fn(p) + bias)


def run_ws_oof(train, ws_splits, subjects, length_scale):
    """Per-subject WS OOF GP predictions."""
    oof = {t: np.full(len(train), np.nan) for t in TARGETS}
    for sid in subjects:
        val_pos  = ws_splits[sid]["val_pos"]
        val_set  = set(val_pos)
        sid_all  = train[train["subject_id"] == sid].index.tolist()
        sid_tr   = [i for i in sid_all if i not in val_set]
        sid_val  = val_pos

        if len(sid_tr) == 0 or len(sid_val) == 0:
            continue

        x_tr  = train.loc[sid_tr,  "day_num"].values
        x_val = train.loc[sid_val, "day_num"].values

        for t in TARGETS:
            y_tr  = train.loc[sid_tr,  t].values
            pred  = fit_gpr_predict(x_tr, y_tr, x_val, length_scale=length_scale)
            for k, p in enumerate(sid_val):
                oof[t][p] = pred[k]
    return oof


def oof_mean_ll(oof, train, subjects, ws_splits):
    all_val_pos = [p for sid in subjects for p in ws_splits[sid]["val_pos"]]
    total = 0.0
    for t in TARGETS:
        y_v   = train.loc[all_val_pos, t].values
        p_v   = np.clip([oof[t][p] for p in all_val_pos], 1e-6, 1 - 1e-6)
        total += log_loss(y_v, p_v)
    return total / len(TARGETS)


def run_interp_oof(train, subjects, length_scale):
    """
    Interpolation OOF: hold out later half of each subject's data,
    train on earlier half, predict on held-out portion.
    This simulates the actual test scenario where test dates interleave
    with later portions of the training period.
    """
    oof = {t: np.full(len(train), np.nan) for t in TARGETS}
    for sid in subjects:
        sid_rows = train[train["subject_id"] == sid].sort_values("sleep_date")
        n        = len(sid_rows)
        if n < 6:
            continue
        n_val   = max(1, n // 3)
        tr_idx  = sid_rows.index[:-n_val].tolist()
        val_idx = sid_rows.index[-n_val:].tolist()

        x_tr  = train.loc[tr_idx,  "day_num"].values
        x_val = train.loc[val_idx, "day_num"].values

        for t in TARGETS:
            y_tr = train.loc[tr_idx, t].values
            pred = fit_gpr_predict(x_tr, y_tr, x_val, length_scale=length_scale)
            for k, p in enumerate(val_idx):
                oof[t][p] = pred[k]
    return oof


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    train    = train.reset_index(drop=True)
    subjects = sorted(train["subject_id"].unique())

    train["day_num"]  = dates_to_days(train["sleep_date"])
    sample["day_num"] = dates_to_days(sample["sleep_date"])

    print("=== GP Temporal Blend ===", flush=True)
    print(f"train: {len(train)} rows, subjects: {len(subjects)}", flush=True)
    print(f"train date range: {train['sleep_date'].min()} - {train['sleep_date'].max()}")
    print(f"test  date range: {sample['sleep_date'].min()} - {sample['sleep_date'].max()}")

    month_tr = pd.to_datetime(train["sleep_date"]).dt.month.value_counts().sort_index()
    month_te = pd.to_datetime(sample["sleep_date"]).dt.month.value_counts().sort_index()
    print("  train months:", {int(k): int(v) for k, v in month_tr.items()})
    print("  test  months:", {int(k): int(v) for k, v in month_te.items()})

    ws_splits = make_ws_splits(train)

    # Phase 0: tune length_scale via WS OOF
    print("\n=== Phase 0: Length Scale Tuning (WS OOF) ===", flush=True)
    ls_scores = {}
    for ls in LENGTH_SCALE_CANDIDATES:
        oof_ls    = run_ws_oof(train, ws_splits, subjects, ls)
        ll        = oof_mean_ll(oof_ls, train, subjects, ws_splits)
        ls_scores[ls] = ll
        print(f"  ls={ls:3d} days: WS OOF LL={ll:.4f}", flush=True)

    best_ls    = min(ls_scores, key=ls_scores.get)
    best_ls_ll = ls_scores[best_ls]
    print(f"\n  best length_scale={best_ls} days (LL={best_ls_ll:.4f})", flush=True)

    # Phase 0b: interpolation OOF diagnostic (best ls)
    print("\n=== Phase 0b: Interpolation OOF Diagnostic ===", flush=True)
    oof_interp = run_interp_oof(train, subjects, best_ls)
    all_interp_pos = [p for sid in subjects
                      for p in train[train["subject_id"] == sid].sort_values("sleep_date").index[-max(1, len(train[train["subject_id"]==sid])//3):].tolist()]
    total_interp = 0.0
    n_valid_t = 0
    for t in TARGETS:
        y_v = train.loc[all_interp_pos, t].values
        p_v_raw = [oof_interp[t][p] for p in all_interp_pos]
        valid = [not np.isnan(x) for x in p_v_raw]
        if sum(valid) == 0:
            continue
        y_valid = y_v[[i for i, v in enumerate(valid) if v]]
        p_valid = np.clip([p_v_raw[i] for i, v in enumerate(valid) if v], 1e-6, 1 - 1e-6)
        ll_t = log_loss(y_valid, p_valid)
        total_interp += ll_t
        n_valid_t += 1
        print(f"  {t}: {ll_t:.4f}", flush=True)
    if n_valid_t > 0:
        print(f"  interp OOF mean LL: {total_interp / n_valid_t:.4f}", flush=True)

    # Phase 1: WS OOF with best ls (for bias correction values)
    print(f"\n=== Phase 1: WS OOF (ls={best_ls}) ===", flush=True)
    ws_oof = run_ws_oof(train, ws_splits, subjects, best_ls)

    all_val_pos = [p for sid in subjects for p in ws_splits[sid]["val_pos"]]
    print("  per-target WS OOF LL:")
    for sid in subjects:
        n_val = len(ws_splits[sid]["val_pos"])
        n_tr  = len(ws_splits[sid]["train_pos"])
        print(f"  {sid}: {len(train[train['subject_id']==sid])} days (val={n_val}, tr={n_tr})")

    total_ll = 0.0
    for t in TARGETS:
        y_v  = train.loc[all_val_pos, t].values
        p_v  = np.clip([ws_oof[t][p] for p in all_val_pos], 1e-6, 1 - 1e-6)
        ll_t = log_loss(y_v, p_v)
        total_ll += ll_t
        print(f"    {t}: {ll_t:.4f}", flush=True)
    ws_oof_ll = total_ll / len(TARGETS)
    print(f"  WS OOF mean LL: {ws_oof_ll:.4f}", flush=True)

    # Logit bias correction per (subject, target)
    biases = {}
    for sid in subjects:
        val_pos   = ws_splits[sid]["val_pos"]
        biases[sid] = {}
        for t in TARGETS:
            y_val = train.loc[val_pos, t].values
            p_val = np.array([ws_oof[t][p] for p in val_pos])
            valid = ~np.isnan(p_val)
            if valid.sum() < 2 or len(np.unique(y_val[valid])) < 2:
                biases[sid][t] = 0.0
            else:
                biases[sid][t] = fit_logit_bias(p_val[valid], y_val[valid])

    # Phase 2: test predictions
    print("\n=== Phase 2: Test Predictions ===", flush=True)
    gp_test = {t: np.zeros(len(sample)) for t in TARGETS}

    for sid in subjects:
        sid_tr  = train[train["subject_id"] == sid].index.tolist()
        mask_te = (sample["subject_id"] == sid).values

        if mask_te.sum() == 0:
            continue

        x_tr = train.loc[sid_tr, "day_num"].values
        x_te = sample.loc[mask_te, "day_num"].values

        for t in TARGETS:
            y_tr      = train.loc[sid_tr, t].values
            pred      = fit_gpr_predict(x_tr, y_tr, x_te, length_scale=best_ls)
            pred_cal  = apply_logit_bias(pred, biases[sid][t])
            gp_test[t][mask_te] = pred_cal

        print(f"  {sid}: tr={len(sid_tr)}, te={int(mask_te.sum())} done", flush=True)

    # Phase 3: final output
    result = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = np.clip(gp_test[t], 0.1, 0.9)

    print("\n=== final prediction distribution ===", flush=True)
    for t in TARGETS:
        print(f"  {t}: min={result[t].min():.3f}, "
              f"mean={result[t].mean():.3f}, "
              f"max={result[t].max():.3f}")

    out_path = SUBMISSION_DIR / "gp_temporal_blend_prob.csv"
    result.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(f"GP WS OOF LL: {ws_oof_ll:.4f} (length_scale={best_ls})")
    print(f"baseline ET seasonal: WS OOF 0.6504, Public 0.5987")
    print(f"best current Public: 0.5987 (ET+LGB+XGB seasonal 3-way)")


if __name__ == "__main__":
    main()
