---
name: Public Score 현황
description: 대회 제출 Public Score 실제 기록 (오기 방지)
type: project
---

현재 최고 Public Score: **0.6061**
파일명: `extratrees_ensemble_prob.csv` (ET v2 단독, z-score 있음)

**Why:** 직접 제출 결과 기준. 이전 대화에서 0.6088로 잘못 기억함.

**How to apply:** ET v2 기반 실험이 기준선. 다른 모델/피처 변경 시 이 점수 대비 개선 여부 판단.

제출 결과 요약 (최신순):
- extratrees_ensemble_prob (ET v2): **0.6061** ← 최고
- extratrees_v3_ensemble_prob (ET v3, z-score 있음): 악화
- mlp_hgb_et_ensemble_prob (MLP+HGB+ET v2): 미세 악화
- hgb_et_ensemble_prob (HGB+ET v2): 상당히 악화
- extratrees_v3_ensemble_prob (ET v3, z-score 없음): 악화
