"""
gp_mp_blend.py
GP Temporal: MP OOF(중간 구간 hold-out)로 length scale 튜닝
실제 테스트가 보간(interleaved) 구조이므로 WS OOF보다 MP OOF가 적합
"""

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.metrics import log_loss
from scipy.special import expit as expit_fn, logit as logit_fn
from scipy.optimize import minimize_scalar
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
VAL_RATIO = 0.20
GP_NOISE = 0.25
CALIB_REG = 0.5
BIAS_BOUND = 2.0
LENGTH_SCALE_CANDIDATES = [5, 7, 10, 14, 21, 30, 45, 60, 90]

REF_DATE = pd.Timestamp("2024-01-01")

BLEND_BASE_FILE = "logreg_v2ac_et_a2_blend_prob.csv"


def dates_to_days(dates):
    return ((pd.to_datetime(dates) - REF_DATE).dt.days).values.astype(float)


def fit_gpr_predict(x_train, y_train, x_test, length_scale=14.0):
    unique = np.unique(y_train)
    if len(unique) < 2:
        base = float(unique[0]) if len(unique) > 0 else 0.5
        return np.full(len(x_test), 0.8 * base + 0.1)
    x_all = np.concatenate([x_train, x_test])
    x_min = x_all.min()
    x_range = max(x_all.max() - x_min, 1.0)
    x_tr_n = (x_train - x_min) / x_range
    x_te_n = (x_test - x_min) / x_range
    ls_n = length_scale / x_range
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
        length_scale=ls_n, length_scale_bounds="fixed"
    )
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=GP_NOISE, normalize_y=True, optimizer=None)
    gpr.fit(x_tr_n.reshape(-1, 1), y_train.astype(float))
    return np.clip(gpr.predict(x_te_n.reshape(-1, 1)), 0.05, 0.95)


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
        mid = n // 2
        half = nv // 2
        vs = max(0, mid - half)
        ve = vs + nv
        if ve > n:
            ve = n
            vs = max(0, n - nv)
        splits[sid] = {"val_pos": pos[vs:ve]}
    return splits


def run_oof(train, splits, subjects, length_scale):
    oof = {t: np.full(len(train), np.nan) for t in TARGETS}
    for sid in subjects:
        val_pos = splits[sid]["val_pos"]
        val_set = set(val_pos)
        sid_all = train[train["subject_id"] == sid].index.tolist()
        sid_tr = [i for i in sid_all if i not in val_set]
        if len(sid_tr) == 0 or len(val_pos) == 0:
            continue
        x_tr = train.loc[sid_tr, "day_num"].values
        x_val = train.loc[val_pos, "day_num"].values
        for t in TARGETS:
            y_tr = train.loc[sid_tr, t].values
            pred = fit_gpr_predict(x_tr, y_tr, x_val, length_scale=length_scale)
            for k, p in enumerate(val_pos):
                oof[t][p] = pred[k]
    return oof


def oof_mean_ll(oof, train, all_val_pos):
    total = 0.0
    for t in TARGETS:
        y_v = train.loc[all_val_pos, t].values
        p_v = np.clip([oof[t][p] for p in all_val_pos], 1e-6, 1 - 1e-6)
        total += log_loss(y_v, p_v)
    return total / len(TARGETS)


def fit_logit_bias(pred_oof, y_true):
    eps = 1e-6
    p = np.clip(pred_oof, eps, 1 - eps)
    lp = logit_fn(p)
    def obj(b):
        return log_loss(y_true, expit_fn(lp + b)) + CALIB_REG * (b ** 2)
    res = minimize_scalar(obj, bounds=(-BIAS_BOUND, BIAS_BOUND), method="bounded")
    return float(res.x)


def apply_logit_bias(pred_raw, bias, eps=1e-6):
    p = np.clip(pred_raw, eps, 1 - eps)
    return expit_fn(logit_fn(p) + bias)


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")
    train = train.reset_index(drop=True)
    subjects = sorted(train["subject_id"].unique())

    train["day_num"] = dates_to_days(train["sleep_date"])
    sample["day_num"] = dates_to_days(sample["sleep_date"])

    ws_splits = make_ws_splits(train)
    mp_splits = make_mp_splits(train)
    ws_val_pos = [p for sid in subjects for p in ws_splits[sid]["val_pos"]]
    mp_val_pos = [p for sid in subjects for p in mp_splits[sid]["val_pos"]]

    print("=== Length Scale Tuning: WS vs MP OOF ===")
    print(f"{'ls':>5} | {'WS OOF':>8} | {'MP OOF':>8}")
    print("-" * 28)

    ws_scores = {}
    mp_scores = {}
    for ls in LENGTH_SCALE_CANDIDATES:
        ws_oof = run_oof(train, ws_splits, subjects, ls)
        mp_oof = run_oof(train, mp_splits, subjects, ls)
        ws_ll = oof_mean_ll(ws_oof, train, ws_val_pos)
        mp_ll = oof_mean_ll(mp_oof, train, mp_val_pos)
        ws_scores[ls] = ws_ll
        mp_scores[ls] = mp_ll
        print(f"  {ls:3d} | {ws_ll:.4f}   | {mp_ll:.4f}")

    best_ws = min(ws_scores, key=ws_scores.get)
    best_mp = min(mp_scores, key=mp_scores.get)
    print(f"\n  WS 최적 ls={best_ws} ({ws_scores[best_ws]:.4f})")
    print(f"  MP 최적 ls={best_mp} ({mp_scores[best_mp]:.4f})")

    # Per-target breakdown at MP-optimal ls
    print(f"\n=== MP OOF 타겟별 (ls={best_mp}) ===")
    mp_oof_best = run_oof(train, mp_splits, subjects, best_mp)
    for t in TARGETS:
        y_v = train.loc[mp_val_pos, t].values
        p_v = np.clip([mp_oof_best[t][p] for p in mp_val_pos], 1e-6, 1 - 1e-6)
        print(f"  {t}: {log_loss(y_v, p_v):.4f}")

    # Bias correction using MP OOF (interpolation-aligned)
    biases = {}
    for sid in subjects:
        val_pos = mp_splits[sid]["val_pos"]
        biases[sid] = {}
        for t in TARGETS:
            y_val = train.loc[val_pos, t].values
            p_val = np.array([mp_oof_best[t][p] for p in val_pos])
            valid = ~np.isnan(p_val)
            if valid.sum() < 2 or len(np.unique(y_val[valid])) < 2:
                biases[sid][t] = 0.0
            else:
                biases[sid][t] = fit_logit_bias(p_val[valid], y_val[valid])

    # Test predictions with MP-optimal ls
    print(f"\n=== 테스트 예측 (ls={best_mp}, MP-tuned) ===")
    gp_test = {t: np.zeros(len(sample)) for t in TARGETS}

    for sid in subjects:
        sid_tr = train[train["subject_id"] == sid].index.tolist()
        mask_te = (sample["subject_id"] == sid).values
        if mask_te.sum() == 0:
            continue
        x_tr = train.loc[sid_tr, "day_num"].values
        x_te = sample.loc[mask_te, "day_num"].values
        for t in TARGETS:
            y_tr = train.loc[sid_tr, t].values
            pred = fit_gpr_predict(x_tr, y_tr, x_te, length_scale=best_mp)
            pred_cal = apply_logit_bias(pred, biases[sid][t])
            gp_test[t][mask_te] = pred_cal
        print(f"  {sid}: done")

    # Save standalone GP prediction
    gp_result = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        gp_result[t] = np.clip(gp_test[t], 0.1, 0.9)
    gp_path = SUBMISSION_DIR / "gp_mp_prob.csv"
    gp_result.to_csv(gp_path, index=False)
    print(f"\n  GP 단독 저장: {gp_path.name}")

    # Blend with current best
    base_path = SUBMISSION_DIR / BLEND_BASE_FILE
    if not base_path.exists():
        print(f"  베이스 파일 없음: {base_path}")
        return
    base_pred = pd.read_csv(base_path)

    print(f"\n=== GP(MP-tuned) + logreg_v2ac_et_a2 블렌드 ===")
    for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        blend = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
        for t in TARGETS:
            gp_t = gp_test[t]
            base_t = base_pred[t].values if t in base_pred.columns else np.full(len(sample), 0.5)
            blend[t] = np.clip(alpha * gp_t + (1 - alpha) * base_t, 0.1, 0.9)
        tag = f"{int(alpha*100):02d}"
        path = SUBMISSION_DIR / f"gp_mp_logreg_a{tag}_blend_prob.csv"
        blend.to_csv(path, index=False)
        print(f"  alpha={alpha:.2f}: {path.name}")

    print("\n=== 완료 ===")
    print(f"  WS 최적 ls={best_ws}, MP 최적 ls={best_mp}")
    print(f"  현재 최고: gp_logreg_a1_blend_prob 0.5928")
    print(f"  권장 제출: gp_mp_logreg_a10_blend_prob (MP-tuned alpha=0.10)")


if __name__ == "__main__":
    main()
