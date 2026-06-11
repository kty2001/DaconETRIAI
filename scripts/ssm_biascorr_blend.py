"""
ssm_biascorr_blend.py
Bernoulli SSM + GP 방식 logit bias correction

GP가 public에서 SSM보다 나은 원인 분석:
  GP: fit_logit_bias(mp_oof_pred, y_true) -> per-subject-target 보정
  SSM: 보정 없음 -> 절대 확률 수준이 맞지 않음

이 스크립트는 SSM 예측에 GP와 동일한 logit bias correction을 적용한다.
  1. SSM MP OOF 계산
  2. per (subject, target) logit bias 피팅 (CALIB_REG=0.5)
  3. 보정된 SSM test 예측 + base blend
"""

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid, logit as logit_fn
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
BASE_COLS = ["subject_id", "sleep_date", "lifelog_date"]
VAL_RATIO = 0.20
REF_DATE = pd.Timestamp("2024-01-01")

SIGMA_Q = 0.30
SIGMA_0 = 2.0
CALIB_REG = 0.5
BIAS_BOUND = 2.0

BLEND_BASE_FILE = "logreg_v2ac_et_a2_blend_prob.csv"
SSM_HARMFUL = {"Q2"}

NAIVE_LL = {"Q1": 0.6931, "Q2": 0.6854, "Q3": 0.6730,
            "S1": 0.6252, "S2": 0.6468, "S3": 0.6396, "S4": 0.6859}


def dates_to_days(dates):
    return ((pd.to_datetime(dates) - REF_DATE).dt.days).values.astype(int)


def bernoulli_ssm_smooth(day_nums_tr, y_tr, day_nums_all,
                         sigma_q=SIGMA_Q, sigma_0=SIGMA_0):
    n = len(day_nums_all)
    tr_set = dict(zip(day_nums_tr.tolist(), y_tr.tolist()))

    m = np.zeros(n)
    P = np.zeros(n)

    mu0 = logit_fn(np.clip(float(y_tr.mean()), 0.05, 0.95))
    m[0] = mu0
    P[0] = sigma_0 ** 2

    if day_nums_all[0] in tr_set:
        y_obs = float(tr_set[day_nums_all[0]])
        p = sigmoid(m[0])
        H = p * (1.0 - p)
        K = P[0] * H / (H * P[0] * H + 1.0)
        m[0] = m[0] + K * (y_obs - p)
        P[0] = (1.0 - K * H) * P[0]

    for t in range(1, n):
        dt = max(int(day_nums_all[t]) - int(day_nums_all[t - 1]), 1)
        Q = sigma_q ** 2 * dt
        m_pred = m[t - 1]
        P_pred = P[t - 1] + Q

        if day_nums_all[t] in tr_set:
            y_obs = float(tr_set[day_nums_all[t]])
            p = sigmoid(m_pred)
            H = p * (1.0 - p)
            K = P_pred * H / (H * P_pred * H + 1.0)
            m[t] = m_pred + K * (y_obs - p)
            P[t] = (1.0 - K * H) * P_pred
        else:
            m[t] = m_pred
            P[t] = P_pred

    m_s = m.copy()
    P_s = P.copy()

    for t in range(n - 2, -1, -1):
        dt = max(int(day_nums_all[t + 1]) - int(day_nums_all[t]), 1)
        P_pred = P_s[t] + sigma_q ** 2 * dt
        G = P_s[t] / P_pred
        m_s[t] = m_s[t] + G * (m_s[t + 1] - m_s[t])
        P_s[t] = P_s[t] + G ** 2 * (P_s[t + 1] - P_pred)

    return np.clip(sigmoid(m_s), 0.05, 0.95)


def make_mp_splits(train_df):
    splits = {}
    for sid, grp in train_df.groupby("subject_id"):
        pos = grp.sort_values("sleep_date").index.tolist()
        n = len(pos)
        nv = max(1, int(n * VAL_RATIO))
        mid = n // 2
        vs = max(0, mid - nv // 2)
        ve = min(n, vs + nv)
        splits[sid] = {"val_pos": pos[vs:ve]}
    return splits


def run_ssm_oof(train, splits, subjects):
    oof = {t: np.full(len(train), np.nan) for t in TARGETS}
    for sid in subjects:
        val_pos = splits[sid]["val_pos"]
        val_set = set(val_pos)
        sid_all = train[train["subject_id"] == sid].index.tolist()
        tr_idx = [i for i in sid_all if i not in val_set]
        if len(tr_idx) == 0:
            continue
        all_day_nums = sorted(train.loc[sid_all, "day_num"].values)
        for t in TARGETS:
            day_nums_tr = train.loc[tr_idx, "day_num"].values
            y_tr = train.loc[tr_idx, t].values
            smoothed = bernoulli_ssm_smooth(day_nums_tr, y_tr, np.array(all_day_nums))
            day_to_pred = dict(zip(all_day_nums, smoothed.tolist()))
            for idx in val_pos:
                d = int(train.loc[idx, "day_num"])
                if d in day_to_pred:
                    oof[t][idx] = day_to_pred[d]
    return oof


def fit_logit_bias(pred_oof, y_true):
    eps = 1e-6
    p = np.clip(pred_oof, eps, 1 - eps)
    lp = logit_fn(p)
    def obj(b):
        return log_loss(y_true, sigmoid(lp + b)) + CALIB_REG * (b ** 2)
    res = minimize_scalar(obj, bounds=(-BIAS_BOUND, BIAS_BOUND), method="bounded")
    return float(res.x)


def apply_logit_bias(pred_raw, bias, eps=1e-6):
    p = np.clip(pred_raw, eps, 1 - eps)
    return sigmoid(logit_fn(p) + bias)


def run_ssm_test_biascorr(train, sample, subjects, biases):
    ssm_test = {t: np.zeros(len(sample)) for t in TARGETS}
    for sid in subjects:
        tr_idx = train[train["subject_id"] == sid].index.tolist()
        mask_te = (sample["subject_id"] == sid).values
        if mask_te.sum() == 0:
            continue
        tr_day_nums = train.loc[tr_idx, "day_num"].values
        te_day_nums = sample.loc[mask_te, "day_num"].values
        all_day_nums = np.unique(np.concatenate([tr_day_nums, te_day_nums]))
        for t in TARGETS:
            y_tr = train.loc[tr_idx, t].values
            smoothed = bernoulli_ssm_smooth(tr_day_nums, y_tr, all_day_nums)
            day_to_pred = dict(zip(all_day_nums.tolist(), smoothed.tolist()))
            te_preds_raw = np.array([day_to_pred.get(int(d), 0.5) for d in te_day_nums])
            b = biases.get(sid, {}).get(t, 0.0)
            te_preds = apply_logit_bias(te_preds_raw, b)
            ssm_test[t][mask_te] = te_preds
        print(f"  {sid}: done")
    return ssm_test


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")
    train = train.reset_index(drop=True)
    subjects = sorted(train["subject_id"].unique())

    train["day_num"] = dates_to_days(train["sleep_date"])
    sample["day_num"] = dates_to_days(sample["sleep_date"])

    mp_splits = make_mp_splits(train)
    mp_val_pos = [p for sid in subjects for p in mp_splits[sid]["val_pos"]]

    print("=== SSM MP OOF 계산 (bias correction용) ===")
    ssm_oof = run_ssm_oof(train, mp_splits, subjects)

    print("\n--- SSM bias correction 전 MP OOF ---")
    ssm_ll_raw = {}
    for t in TARGETS:
        y_v = train.loc[mp_val_pos, t].values
        p_v = np.clip([ssm_oof[t][i] if not np.isnan(ssm_oof[t][i]) else 0.5
                       for i in mp_val_pos], 1e-6, 1 - 1e-6)
        ssm_ll_raw[t] = log_loss(y_v, p_v)
        print(f"  {t}: {ssm_ll_raw[t]:.4f}  (naive={NAIVE_LL[t]:.4f})")

    # per-subject per-target logit bias 피팅
    print("\n--- per-(subject, target) logit bias 피팅 ---")
    biases = {}
    for sid in subjects:
        val_pos = mp_splits[sid]["val_pos"]
        biases[sid] = {}
        for t in TARGETS:
            y_val = train.loc[val_pos, t].values
            p_val = np.array([ssm_oof[t][i] if not np.isnan(ssm_oof[t][i]) else 0.5
                              for i in val_pos])
            valid = ~np.isnan(p_val)
            if valid.sum() < 2 or len(np.unique(y_val[valid])) < 2:
                biases[sid][t] = 0.0
            else:
                biases[sid][t] = fit_logit_bias(p_val[valid], y_val[valid])

    # bias correction 후 MP OOF 재계산
    print("\n--- SSM bias correction 후 MP OOF ---")
    ssm_ll_corr = {}
    for t in TARGETS:
        preds = []
        for i in mp_val_pos:
            sid = train.loc[i, "subject_id"]
            raw = ssm_oof[t][i] if not np.isnan(ssm_oof[t][i]) else 0.5
            b = biases[sid][t]
            preds.append(float(apply_logit_bias(raw, b)))
        y_v = train.loc[mp_val_pos, t].values
        p_v = np.clip(preds, 1e-6, 1 - 1e-6)
        ssm_ll_corr[t] = log_loss(y_v, p_v)
        diff = ssm_ll_raw[t] - ssm_ll_corr[t]
        print(f"  {t}: {ssm_ll_corr[t]:.4f}  (보정 전: {ssm_ll_raw[t]:.4f}, {diff:+.4f})")
    print(f"  avg (보정 전): {np.mean(list(ssm_ll_raw.values())):.4f}  "
          f"avg (보정 후): {np.mean(list(ssm_ll_corr.values())):.4f}")

    # bias 크기 요약
    print("\n--- 피팅된 bias 값 (logit space) ---")
    for t in TARGETS:
        bvals = [biases[sid][t] for sid in subjects]
        print(f"  {t}: mean={np.mean(bvals):+.3f}, "
              f"min={min(bvals):+.3f}, max={max(bvals):+.3f}")

    # 테스트 예측 생성 (bias 적용)
    print("\n=== SSM+biascorr 테스트 예측 생성 ===")
    ssm_test = run_ssm_test_biascorr(train, sample, subjects, biases)

    # base와 blend
    base_path = SUBMISSION_DIR / BLEND_BASE_FILE
    if not base_path.exists():
        print(f"  베이스 파일 없음: {base_path}")
        return
    base_pred = pd.read_csv(base_path)

    print("\n=== SSM+biascorr + base 블렌드 (Q2=0) ===")
    for alpha in [0.05, 0.10, 0.15]:
        blend = sample[BASE_COLS].copy()
        for t in TARGETS:
            a = 0.0 if t in SSM_HARMFUL else alpha
            blend[t] = np.clip(a * ssm_test[t] + (1 - a) * base_pred[t].values, 0.1, 0.9)
        tag = f"{int(alpha * 100):02d}"
        path = SUBMISSION_DIR / f"ssm_biascorr_noq2_a{tag}_blend_prob.csv"
        blend.to_csv(path, index=False)
        print(f"  alpha={alpha:.2f}: {path.name}")

    # GP-equivalent: SSM+biascorr, Q2=0, helpful targets only (Q3 포함)
    print("\n=== SSM+biascorr 타겟별 alpha (ssm_ll_corr < naive) ===")
    alphas_pt = {}
    for t in TARGETS:
        if t in SSM_HARMFUL or ssm_ll_corr[t] >= NAIVE_LL[t]:
            alphas_pt[t] = 0.0
        else:
            improvement = NAIVE_LL[t] - ssm_ll_corr[t]
            alphas_pt[t] = min(0.15, improvement * 2.0)
    print("  alpha:", {t: f"{a:.3f}" for t, a in alphas_pt.items()})

    blend_pt = sample[BASE_COLS].copy()
    for t in TARGETS:
        a = alphas_pt[t]
        blend_pt[t] = np.clip(a * ssm_test[t] + (1 - a) * base_pred[t].values, 0.1, 0.9)
    path_pt = SUBMISSION_DIR / "ssm_biascorr_pertarget_blend_prob.csv"
    blend_pt.to_csv(path_pt, index=False)
    print(f"  저장: {path_pt.name}")

    print("\n=== 완료 ===")
    print(f"  sigma_q={SIGMA_Q}, CALIB_REG={CALIB_REG}")
    print(f"  SSM+bias avg MP OOF: {np.mean(list(ssm_ll_corr.values())):.4f}")
    print(f"  GP avg MP OOF: 0.6493 (참고)")
    print("  권장 제출:")
    print("    ssm_biascorr_noq2_a10_blend_prob.csv")
    print("    ssm_biascorr_pertarget_blend_prob.csv")


if __name__ == "__main__":
    main()
