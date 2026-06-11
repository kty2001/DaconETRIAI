"""
ssm_blend.py
Bernoulli State Space Model (RTS Smoother) 기반 temporal blend

모델: l_t = logit(p_t) 가 Random Walk를 따름
  - Transition: l_t = l_{t-1} + N(0, sigma_q^2 * dt)
  - Observation: y_t ~ Bernoulli(sigmoid(l_t))
  - Forward: Extended Kalman Filter
  - Backward: RTS (Rauch-Tung-Striebel) Smoother

GP 대비 장점:
  - 시간 간격(dt)을 명시적으로 처리 (더 먼 날짜 = 더 큰 불확실성)
  - RTS 스무딩으로 테스트 날짜 양쪽의 훈련 라벨 모두 활용
  - Bernoulli 관측 모델 (GPR의 회귀 근사보다 정확)

MP OOF 타겟별 성능 (sigma_q=0.30):
  Q1: 0.6691 (GP 0.6803, SSM 도움)
  Q2: 0.7972 (GP 0.8530, SSM도 해로움 -> alpha=0)
  Q3: 0.6645 (GP 0.6849, SSM 도움 -- GP는 해로웠으나 SSM은 도움)
  S1: 0.5667 (GP 0.5734, SSM 도움)
  S2: 0.5434 (GP 0.5419, 거의 동일)
  S3: 0.5804 (GP 0.5805, 거의 동일)
  S4: 0.6222 (GP 0.6312, SSM 도움)
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

SIGMA_Q = 0.30   # process noise per day (MP OOF 최적)
SIGMA_0 = 2.0    # initial state uncertainty

BLEND_BASE_FILE = "logreg_v2ac_et_a2_blend_prob.csv"

# SSM도 Q2는 해로움 (SSM Q2 MP OOF 0.7972 > naive 0.6854)
SSM_HARMFUL = {"Q2"}   # alpha=0 강제


def dates_to_days(dates):
    return ((pd.to_datetime(dates) - REF_DATE).dt.days).values.astype(int)


def bernoulli_ssm_smooth(day_nums_tr, y_tr, day_nums_all,
                         sigma_q=SIGMA_Q, sigma_0=SIGMA_0):
    """
    EKF forward pass + RTS backward pass.
    day_nums_tr: 훈련 날짜 (정수)
    y_tr: 0/1 라벨
    day_nums_all: 예측 대상 전체 날짜 (정렬)
    반환: day_nums_all 위치의 smoothed probability
    """
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

    # RTS smoother (backward pass)
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
    """MP OOF SSM 예측 계산"""
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


def run_ssm_test(train, sample, subjects):
    """테스트 예측 생성 (전체 훈련 데이터 사용)"""
    ssm_test = {t: np.zeros(len(sample)) for t in TARGETS}

    for sid in subjects:
        tr_idx = train[train["subject_id"] == sid].index.tolist()
        mask_te = (sample["subject_id"] == sid).values

        if mask_te.sum() == 0:
            continue

        tr_day_nums = train.loc[tr_idx, "day_num"].values
        te_day_nums = sample.loc[mask_te, "day_num"].values

        # train + test 날짜 합쳐서 정렬
        all_day_nums = np.unique(np.concatenate([tr_day_nums, te_day_nums]))

        for t in TARGETS:
            y_tr = train.loc[tr_idx, t].values
            smoothed = bernoulli_ssm_smooth(tr_day_nums, y_tr, all_day_nums)
            day_to_pred = dict(zip(all_day_nums.tolist(), smoothed.tolist()))
            te_preds = np.array([day_to_pred.get(int(d), 0.5) for d in te_day_nums])
            ssm_test[t][mask_te] = te_preds

        print(f"  {sid}: done")

    return ssm_test


def fit_blend_alpha(ssm_oof, base_oof, y_train, val_pos, target,
                    alpha_bound=0.40, reg=0.5):
    """MP OOF로 SSM blend alpha 최적화"""
    y_v = y_train.loc[val_pos, target].values
    ssm_v = np.array([ssm_oof[target][i] for i in val_pos])
    base_v = np.array([base_oof[target][i] for i in val_pos])

    valid = ~(np.isnan(ssm_v) | np.isnan(base_v))
    if valid.sum() < 2:
        return 0.0

    def obj(a):
        blend = np.clip(a * ssm_v[valid] + (1 - a) * base_v[valid], 1e-6, 1 - 1e-6)
        return log_loss(y_v[valid], blend) + reg * a ** 2

    res = minimize_scalar(obj, bounds=(0.0, alpha_bound), method="bounded")
    return float(res.x)


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")
    train = train.reset_index(drop=True)
    subjects = sorted(train["subject_id"].unique())

    train["day_num"] = dates_to_days(train["sleep_date"])
    sample["day_num"] = dates_to_days(sample["sleep_date"])

    mp_splits = make_mp_splits(train)
    mp_val_pos = [p for sid in subjects for p in mp_splits[sid]["val_pos"]]

    # MP OOF 평가
    print("=== SSM MP OOF 계산 (sigma_q={:.2f}) ===".format(SIGMA_Q))
    ssm_oof = run_ssm_oof(train, mp_splits, subjects)

    print("\n타겟별 MP OOF (SSM vs Naive):")
    naive_ll = {"Q1": 0.6931, "Q2": 0.6854, "Q3": 0.6730,
                "S1": 0.6252, "S2": 0.6468, "S3": 0.6396, "S4": 0.6859}
    gp_ll    = {"Q1": 0.6803, "Q2": 0.8530, "Q3": 0.6849,
                "S1": 0.5734, "S2": 0.5419, "S3": 0.5805, "S4": 0.6312}
    ssm_ll   = {}
    for t in TARGETS:
        y_v = train.loc[mp_val_pos, t].values
        p_v = np.clip([ssm_oof[t][i] if not np.isnan(ssm_oof[t][i]) else 0.5
                       for i in mp_val_pos], 1e-6, 1 - 1e-6)
        ssm_ll[t] = log_loss(y_v, p_v)
        verdict = "도움" if ssm_ll[t] < naive_ll[t] else "해로움"
        print(f"  {t}: SSM={ssm_ll[t]:.4f}  Naive={naive_ll[t]:.4f}  "
              f"GP={gp_ll[t]:.4f}  -> {verdict}")
    print(f"  avg SSM: {np.mean(list(ssm_ll.values())):.4f}  "
          f"avg GP: {np.mean(list(gp_ll.values())):.4f}")

    # 테스트 예측 생성
    print("\n=== SSM 테스트 예측 생성 ===")
    ssm_test = run_ssm_test(train, sample, subjects)

    # SSM 단독 저장
    ssm_result = sample[BASE_COLS].copy()
    for t in TARGETS:
        ssm_result[t] = np.clip(ssm_test[t], 0.1, 0.9)
    ssm_path = SUBMISSION_DIR / "ssm_prob.csv"
    ssm_result.to_csv(ssm_path, index=False)
    print(f"\n  SSM 단독 저장: {ssm_path.name}")

    # 기존 base와 blend
    base_path = SUBMISSION_DIR / BLEND_BASE_FILE
    if not base_path.exists():
        print(f"  베이스 파일 없음: {base_path}")
        return
    base_pred = pd.read_csv(base_path)

    # 고정 alpha 블렌드 (0.05, 0.10, 0.15, 0.20) - Q2=0 제외
    print("\n=== SSM + logreg_v2ac_et_a2 고정 alpha 블렌드 ===")
    for alpha in [0.05, 0.10, 0.15, 0.20]:
        blend = sample[BASE_COLS].copy()
        for t in TARGETS:
            a = 0.0 if t in SSM_HARMFUL else alpha
            ssm_t = ssm_test[t]
            base_t = base_pred[t].values
            blend[t] = np.clip(a * ssm_t + (1 - a) * base_t, 0.1, 0.9)
        tag = f"{int(alpha * 100):02d}"
        path = SUBMISSION_DIR / f"ssm_noq2_a{tag}_blend_prob.csv"
        blend.to_csv(path, index=False)
        print(f"  alpha={alpha:.2f} (Q2=0): {path.name}")

    # MP OOF 기반 타겟별 alpha 최적화
    # base의 MP OOF 예측이 없으므로 base의 test 예측을 대리 사용 (근사)
    # 더 정확하게는 base의 WS/MP OOF 예측이 필요하지만,
    # alpha 범위를 [0, 0.20]으로 제한하여 과최적화 방지
    print("\n=== 타겟별 alpha 최적화 (SSM MP OOF 성능 기준) ===")
    # 간단히 SSM MP OOF vs Naive 비교로 alpha 결정
    # SSM이 naive보다 좋은 타겟만 alpha > 0
    alphas_pertarget = {}
    for t in TARGETS:
        if t in SSM_HARMFUL or ssm_ll[t] >= naive_ll[t]:
            alphas_pertarget[t] = 0.0
        else:
            # SSM 개선 정도에 비례하여 alpha 설정 (상한 0.15)
            improvement = naive_ll[t] - ssm_ll[t]
            alphas_pertarget[t] = min(0.15, improvement * 2.0)

    print("  타겟별 alpha:", {t: f"{a:.3f}" for t, a in alphas_pertarget.items()})

    blend_pt = sample[BASE_COLS].copy()
    for t in TARGETS:
        a = alphas_pertarget[t]
        blend_pt[t] = np.clip(a * ssm_test[t] + (1 - a) * base_pred[t].values, 0.1, 0.9)
    path_pt = SUBMISSION_DIR / "ssm_pertarget_blend_prob.csv"
    blend_pt.to_csv(path_pt, index=False)
    print(f"  저장: {path_pt.name}")

    print("\n=== 완료 ===")
    print(f"  sigma_q={SIGMA_Q}, Q2=0 (SSM 해로움)")
    print(f"  SSM avg MP OOF: {np.mean(list(ssm_ll.values())):.4f}  "
          f"(GP avg: {np.mean(list(gp_ll.values())):.4f})")
    print("  권장 제출 순서:")
    print("    1. ssm_noq2_a10_blend_prob  (alpha=0.10, Q2=0)")
    print("    2. ssm_pertarget_blend_prob (타겟별 최적화)")
    print("    3. ssm_noq2_a05_blend_prob  (alpha=0.05, Q2=0)")
    print("    4. ssm_noq2_a15_blend_prob  (alpha=0.15, Q2=0)")


if __name__ == "__main__":
    main()
