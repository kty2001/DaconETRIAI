"""
gp_pertarget_blend.py
GP alpha를 타겟별로 다르게 적용

GP MP OOF per target (gp_mp_blend.py 실행 결과):
  Q1: 0.6803, Q2: 0.8530, Q3: 0.6849
  S1: 0.5734, S2: 0.5419, S3: 0.5805, S4: 0.6312

Naive LL per target:
  Q1: 0.6931, Q2: 0.6854, Q3: 0.6730
  S1: 0.6252, S2: 0.6468, S3: 0.6396, S4: 0.6859

Q2: GP OOF 0.8530 >> naive 0.6854 (0.1676 악화) → alpha=0
Q3: GP OOF 0.6849 > naive 0.6730 (0.0119 악화) → alpha=0
나머지: GP가 naive보다 우수 → alpha=0.10 유지
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUBMISSION = ROOT / "submission"
DATA = ROOT / "data"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
BASE_COLS = ["subject_id", "sleep_date", "lifelog_date"]

GP_MP_OOF = {"Q1": 0.6803, "Q2": 0.8530, "Q3": 0.6849,
             "S1": 0.5734, "S2": 0.5419, "S3": 0.5805, "S4": 0.6312}
NAIVE_LL  = {"Q1": 0.6931, "Q2": 0.6854, "Q3": 0.6730,
             "S1": 0.6252, "S2": 0.6468, "S3": 0.6396, "S4": 0.6859}

BASE_FILE = "logreg_v2ac_et_a2_blend_prob.csv"
GP_FILE   = "gp_mp_prob.csv"


def make_blend(gp_df, base_df, sample_df, alphas):
    blend = sample_df[BASE_COLS].copy()
    for t in TARGETS:
        alpha = alphas[t]
        blend[t] = np.clip(alpha * gp_df[t].values + (1 - alpha) * base_df[t].values, 0.1, 0.9)
    return blend


def main():
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")
    gp = pd.read_csv(SUBMISSION / GP_FILE)
    base = pd.read_csv(SUBMISSION / BASE_FILE)

    print("=== GP 타겟별 기여 분석 ===")
    print(f"{'타겟':>4} | {'GP MP OOF':>10} | {'Naive LL':>9} | {'차이':>8} | {'판정':>8}")
    print("-" * 55)
    for t in TARGETS:
        diff = GP_MP_OOF[t] - NAIVE_LL[t]
        verdict = "GP 해로움" if diff > 0 else "GP 도움"
        print(f"  {t}  | {GP_MP_OOF[t]:>10.4f} | {NAIVE_LL[t]:>9.4f} | {diff:>+8.4f} | {verdict}")

    # Case 1: Q2만 제거 (Q2=0, 나머지=0.10)
    alpha_q2only = {t: 0.10 for t in TARGETS}
    alpha_q2only["Q2"] = 0.00
    blend1 = make_blend(gp, base, sample, alpha_q2only)
    path1 = SUBMISSION / "gp_noq2_blend_prob.csv"
    blend1.to_csv(path1, index=False)
    print(f"\n  저장: {path1.name}  (Q2=0, 나머지=0.10)")

    # Case 2: Q2+Q3 제거 (Q2=Q3=0, 나머지=0.10)
    alpha_q2q3 = {t: 0.10 for t in TARGETS}
    alpha_q2q3["Q2"] = 0.00
    alpha_q2q3["Q3"] = 0.00
    blend2 = make_blend(gp, base, sample, alpha_q2q3)
    path2 = SUBMISSION / "gp_noq2q3_blend_prob.csv"
    blend2.to_csv(path2, index=False)
    print(f"  저장: {path2.name}  (Q2=Q3=0, 나머지=0.10)")

    # Case 3: Q3은 0.05로 약하게 (Q2=0, Q3=0.05, 나머지=0.10)
    alpha_q3half = {t: 0.10 for t in TARGETS}
    alpha_q3half["Q2"] = 0.00
    alpha_q3half["Q3"] = 0.05
    blend3 = make_blend(gp, base, sample, alpha_q3half)
    path3 = SUBMISSION / "gp_noq2_halfq3_blend_prob.csv"
    blend3.to_csv(path3, index=False)
    print(f"  저장: {path3.name}  (Q2=0, Q3=0.05, 나머지=0.10)")

    # Case 4: GP OOF < naive인 타겟에만 적용 (Q1, S1-S4만 0.10)
    alpha_helpful = {t: 0.00 for t in TARGETS}
    for t in TARGETS:
        if GP_MP_OOF[t] < NAIVE_LL[t]:
            alpha_helpful[t] = 0.10
    blend4 = make_blend(gp, base, sample, alpha_helpful)
    path4 = SUBMISSION / "gp_helpful_only_blend_prob.csv"
    blend4.to_csv(path4, index=False)
    helpful = [t for t in TARGETS if GP_MP_OOF[t] < NAIVE_LL[t]]
    print(f"  저장: {path4.name}  (GP 도움 타겟만: {helpful})")

    print("\n=== 권장 제출 순서 ===")
    print("  1. gp_noq2_blend_prob        - Q2만 GP 제거, 가장 확실한 개선")
    print("  2. gp_noq2q3_blend_prob       - Q2+Q3 GP 제거")
    print("  3. gp_noq2_halfq3_blend_prob  - Q3은 50% 적용")
    print("  4. gp_helpful_only_blend_prob - GP 도움 타겟만 적용")


if __name__ == "__main__":
    main()
