# ETRI 라이프로그 데이터를 활용한 수면 및 심리 지표 예측 대회

## 대회 개요

700일 분량의 시계열 라이프로그 데이터와 450일분의 레이블을 이용해 **7가지 수면/심리 지표(Q1~Q3, S1~S4)** 를 이진 분류(0/1)로 예측하는 과제.

| 지표 | 분류 | 설명 | 1의 의미 |
|------|------|------|----------|
| Q1 | 설문 | 수면의 질 (기상 직후) | 평소보다 좋음 |
| Q2 | 설문 | 피로도 (취침 직전) | 평소보다 피로가 적음 |
| Q3 | 설문 | 스트레스 (취침 직전) | 평소보다 스트레스 낮음 |
| S1 | 센서 | 총 수면 시간 (TST) | 권장 기준 충족 |
| S2 | 센서 | 수면 효율 (SE) | 권장 기준 충족 |
| S3 | 센서 | 수면 지연 시간 (SOL) | 권장 기준 충족 |
| S4 | 센서 | 수면 중 각성 시간 (WASO) | 권장 기준 충족 |

---

## 프로젝트 구조

```
DaconETRIAI/
├── data/
│   ├── ch2026_metrics_train.csv          # 학습 레이블 (450행)
│   ├── ch2026_submission_sample.csv      # 제출 양식 (250행)
│   ├── ch2026_metrics_description.pdf    # 지표 설명서
│   ├── features_all.csv                  # 통합 피처 데이터셋
│   └── ch2025_data_items/                # 라이프로그 parquet (12종)
├── scripts/
│   ├── baseline_subject_mean.py          # 베이스라인: subject 평균 분류
│   ├── lgbm_csv_only.py                  # LightGBM (CSV 피처만)
│   ├── logistic_csv_only.py              # Logistic Regression (피처 중요도)
│   ├── parquet_features.py               # parquet 집계 v1
│   ├── parquet_features_v2.py            # parquet 집계 v2 + mUsageStats
│   ├── label_features.py                 # lag1/2/7 + roll3/7/14 피처 빌더
│   ├── build_feature_csv.py              # 전체 피처 통합 CSV 생성
│   ├── lgbm_final.py                     # LightGBM (모든 피처 + LOSO)
│   ├── lgbm_zscore.py                    # + subject z-score 정규화
│   ├── lgbm_multiseed.py                 # + 멀티 시드(10개)
│   ├── lgbm_roll14lag7.py                # + lag7/roll14 피처
│   ├── lgbm_lag_full.py                  # + 동일 타깃 lag 허용
│   ├── lgbm_optuna.py                    # + Optuna 하이퍼파라미터 튜닝 (현재 최고)
│   ├── lgbm_calibrated.py                # Platt Scaling 시도 (실패)
│   ├── lgbm_usage.py                     # mUsageStats 피처 시도
│   ├── feature_importance.py             # LightGBM gain 중요도 분석
│   ├── permutation_importance.py         # Permutation Importance 분석
│   ├── shap_importance.py                # SHAP 분석
│   ├── lgbm_feat_select.py               # Permutation 음수 피처 제거 시도 (실패)
│   ├── lgbm_usage_select.py              # mUsageStats 상위 선별 시도 (실패)
│   ├── lgbm_xgb_ensemble.py              # LGBM+XGBoost 앙상블 (실패)
│   ├── xgb_optuna.py                     # XGBoost 단독 (실패)
│   ├── xgb_shap_select.py                # XGBoost + SHAP 상위 35개 피처 (실패)
│   └── lgbm_catboost_ensemble.py         # LGBM+CatBoost 앙상블 (현재 최고)
├── submission/
│   ├── lgbm_catboost_ensemble_prob.csv   # 현재 최고 제출 파일 (0.6170)
│   ├── feature_result.md                 # 3가지 방법론 피처 중요도 전체 수치
│   └── submission_result.md
└── README.md
```

---

## 데이터 구조 요약

### 레이블 데이터

- **대상자**: id01 ~ id10, 총 10명
- **학습(train)**: 450행, 2024-06-04 ~ 2024-11-15 (인당 33~57일)
- **예측(submission)**: 250행, 2024-07-07 ~ 2024-11-20 (인당 19~32일)
- **train ↔ submission 날짜 겹침: 0개** (완전히 별개 구간)
- **날짜 연결 구조**: `lifelog_date = sleep_date - 1일` (전날 행동 → 당일 수면 예측)

### 라이프로그 parquet (12종)

| 구분 | 파일 | 측정값 | 사용 여부 |
|------|------|--------|-----------|
| 스마트워치 | wPedo | 걸음수/칼로리/거리/속도 | ✅ |
| 스마트폰 | mActivity | 활동 유형 코드 | ✅ |
| 스마트폰 | mScreenStatus | 화면 켜짐 여부 | ✅ |
| 스마트워치 | wHr | 심박수 배열 | ✅ |
| 스마트폰 | mLight | 조도 (lux) | ✅ |
| 스마트폰 | mAmbience | 주변 소리 분류 + 확률 | ✅ |
| 스마트폰 | mUsageStats | 앱별 사용 시간 | ⚠️ 추가했으나 성능 저하 |
| 스마트폰 | mGps | GPS 좌표/속도/고도 | ❌ wPedo와 중복 |
| 스마트폰 | mBle | 주변 BLE 기기 RSSI | ❌ 밀도 낮음 |
| 스마트폰 | mWifi | 주변 WiFi AP RSSI | ❌ 신호 약함 |
| 스마트워치 | wLight | 워치 조도 | ❌ mLight와 중복 |

---

## 평가 산식

**Average Log-loss** (낮을수록 좋음)

$$\text{score} = \frac{1}{K} \sum_{k=1}^{K} \left( -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log p_i + (1 - y_i) \log (1 - p_i) \right] \right)$$

| 구분 | 비율 | 설명 |
|------|------|------|
| Public score | 44% | 리더보드 즉시 반영 |
| Private score | 100% | 대회 종료 후 최종 순위 |

> 확률값(0~1)으로 제출해야 유리. hard 0/1 제출 시 log-loss 극단적으로 증가.

---

## 대회 제출 점수

| 모델 | Public Score | 비고 |
|------|:---:|------|
| submission_sample (전부 0) | 21.1269 | 하한 기준선 |
| baseline_subject_mean | 13.0531 | |
| logistic_csv_only | 12.2000 | |
| lgbm_csv_only | 12.0663 | hard 0/1 제출 |
| lgbm_zscore_prob | 0.6950 | 확률 제출 전환 후 첫 제출 |
| lgbm_multiseed_prob | 0.6924 | |
| lgbm_roll14lag7_prob | 0.6890 | lag7/roll14 추가 |
| lgbm_lag_full_prob | 0.6822 | 동일 타깃 lag 허용 |
| lgbm_usage_prob | 0.6239 | mUsageStats 추가 (역효과) |
| lgbm_usage_zscore_prob | 0.6255 | mUsageStats z-score (역효과) |
| lgbm_optuna_prob | 0.6178 | Optuna 30 trials (이전 최고) |
| lgbm_optuna100_prob | 0.6228 | Optuna 100 trials (역효과) |
| lgbm_feat_select_prob | 0.6281 | Permutation 음수 피처 79개 제거 (역효과) |
| lgbm_usage_select_prob | 0.6215 | mUsageStats 상위 3개 선별 (역효과) |
| lgbm_xgb_ensemble_prob | 0.6218 | LGBM + XGBoost 앙상블 (역효과) |
| xgb_optuna_prob | 0.6258 | XGBoost 단독 (역효과) |
| xgb_shap_select_prob | 0.6343 | XGBoost + SHAP 상위 35개 피처 (역효과) |
| **lgbm_catboost_ensemble_prob** | **0.6170** | **현재 최고 — LGBM + CatBoost 앙상블** |

> lgbm_csv_only(12.07) → lgbm_catboost_ensemble(0.617): 확률 제출 전환 + 피처 + 튜닝 + 앙상블로 **95% 개선**

---

## 모델 발전 과정

### 1단계: 베이스라인 구축

| 모델 | CV 전략 | 특징 |
|------|---------|------|
| baseline_subject_mean | — | subject별 평균으로 0/1 분류 |
| lgbm_csv_only | StratifiedKFold(5) | 날짜 피처 + subject 인코딩 |
| logistic_csv_only | StratifiedKFold(5) | 피처 중요도 분석용 |

**주요 발견**: `subj_mean_S4`, `subj_mean_Q3`이 예측력 강함

### 2단계: CV 전략 개선

| 전략 | 문제점 |
|------|--------|
| StratifiedKFold(5) | 동일 subject가 train/val 혼재 → 낙관적 평가 |
| **GroupKFold(10) LOSO** | subject 단위 완전 분리 → 실제 성능 근사 |

### 3단계: 피처 엔지니어링

| 피처 그룹 | 내용 | 효과 |
|-----------|------|------|
| parquet v1 | wPedo/mActivity/mScreen/wHr/mLight 일별 집계 | S3 +0.122 |
| parquet v2 | 시간대 5구간 분리 + mAmbience | Q3/S1 소폭 개선 |
| lag1/2 + roll3/7 | 전날·2일 전 레이블, 3/7개 이동 평균 | S3 개선 |
| lag7 + roll14 | 7일 전 레이블(주간 패턴), 14개 이동 평균 | S2/S3 추가 개선 |

**피처 수**: base 6 + subj_mean 7 + parquet 81(mUsageStats 포함) + lag/roll 42 = 총 **136개** (타깃별 135개)

### 4단계: Subject Z-score 정규화

각 센서 피처를 subject별 평균/표준편차로 정규화.

```
기존: hr_mean = 80.0 (id03과 id08 동일 취급)
개선: (80 - subject_mean) / subject_std
     → id03: +2.0 (비정상적으로 높은 날)
     → id08: -1.0 (평소보다 낮은 날)
```

**효과**: S2 F1 0.579 → 0.611, 평균 LogLoss 1.181 → 1.151

### 5단계: 멀티 시드 평균

10개 시드 + subsample/colsample 추가로 모델 다양성 확보.

**효과**: 평균 LogLoss 1.151 → 1.086 (5.6% 개선)

### 6단계: 동일 타깃 lag 허용

기존 코드는 Q1 예측 시 lag1_Q1, roll7_Q1 등 동일 타깃 lag를 제거했으나, 전날의 Q1은 오늘의 Q1을 예측하는 가장 강한 신호임을 확인.

**효과**: 평균 LogLoss 1.086 → 1.042 (4% 개선), Public Score 0.6924 → 0.6822

### 7단계: Optuna 하이퍼파라미터 튜닝

타깃별로 독립적인 최적 파라미터 탐색 (기존: 모든 타깃 동일 파라미터).

```python
# 탐색 공간
num_leaves: 8~64, learning_rate: 0.01~0.3, n_estimators: 100~600
min_child_samples: 5~50, subsample: 0.5~1.0, colsample_bytree: 0.5~1.0
reg_alpha/reg_lambda: 1e-8~10.0
```

| 구현 | 내용 |
|------|------|
| Phase 1 | 타깃별 Optuna (30 trials) → best_params 탐색 |
| Phase 2 | best_params + 10 seeds로 최종 예측 |
| 피처 최적화 | 폴드별 피처를 1회 사전 계산해 trial 속도 최대화 |

**효과**: OOF LogLoss 1.042 → 0.671, Public Score 0.6822 → 0.6178

### 8단계: 피처 중요도 분석 (3가지 방법론)

피처 선택을 위해 3가지 방법론으로 135개 피처의 실제 기여도를 측정.

| 방법론 | 도구 | 핵심 발견 |
|--------|------|-----------|
| **LightGBM gain** | `feature_importances_` | `light_mean_presleep` 1위, lag/roll 하위권 → gain이 실제 기여 과대/과소평가 |
| **Permutation Importance** | OOF LogLoss 변화량 | 135개 중 79개가 음수(노이즈). `roll3_Q2` Q2 1위로 gain과 불일치 |
| **SHAP** | TreeExplainer | `subj_mean_*` 다수 타깃 1위 — 개인 기준선이 가장 강한 예측 신호 |

**3가지 방법론의 핵심 공통 결론**

| 결론 | 내용 |
|------|------|
| gain ≠ 실제 중요도 | gain 1위 `light_mean_presleep`이 Permutation에서 음수 — gain은 신뢰도 낮음 |
| subj_mean_* 과소평가 | gain 중위권 → SHAP 다수 타깃 1위. 개인 기준선이 실제 예측의 핵심 |
| lag/roll 과소평가 | gain 하위권 → Permutation/SHAP 상위권. 실제로는 중요한 피처 |
| 노이즈 피처 다수 | Permutation 음수 피처 79개: pedo 절대값, act_ratio_*, amb_* 대부분 |

> 상세 수치 전체: `submission/feature_result.md`

### 9단계: 피처 선택 (Feature Selection)

Permutation Importance 음수 피처 79개 제거 후 성능 검증 (135개 → 56개).

| 타깃 | optuna 30t (135개) | feat_select (56개) | 변화 |
|------|:-----------------:|:------------------:|:----:|
| Q1 | 0.7200 | 0.7198 | ≈ |
| Q2 | 0.6530 | 0.6437 | 개선 |
| Q3 | 0.6466 | 0.6486 | 소폭 악화 |
| S1 | 0.6394 | 0.6431 | 소폭 악화 |
| S2 | 0.6511 | 0.6603 | 악화 |
| S3 | 0.6462 | 0.6763 | 크게 악화 |
| S4 | 0.7042 | 0.7042 | ≈ |

**Public Score**: 0.6178 → 0.6281 (역효과)

원인: Permutation Importance는 LOSO(10개 fold, 각 ~45행)에서 분산이 크기 때문에 "음수"로 측정된 피처 중 실제로 기여하는 피처가 포함됨. S3에서 `light_mean_presleep`, `roll14_S3`, `act_ratio_afternoon` 등이 제거되어 S3 성능 급락.

### 10단계: XGBoost 도입 (앙상블 및 단독) — 실패

LGBM + XGBoost 앙상블 및 XGBoost 단독 예측을 시도.

```python
# XGBoost 탐색 공간
n_estimators: 100~600, max_depth: 3~8, learning_rate: 0.01~0.3
min_child_weight: 1~20, subsample: 0.5~1.0, colsample_bytree: 0.5~1.0
reg_alpha/reg_lambda: 1e-8~10.0, gamma: 1e-8~1.0
```

| 타깃 | LGBM OOF | XGB 단독 OOF | 앙상블 OOF |
|------|:--------:|:-----------:|:---------:|
| Q1 | 0.7202 | 0.6948 | 0.7027 |
| Q2 | 0.6644 | 0.6582 | 0.6590 |
| Q3 | 0.6575 | 0.6504 | 0.6529 |
| S1 | 0.6417 | 0.6190 | 0.6271 |
| S2 | 0.6497 | 0.6174 | 0.6300 |
| S3 | 0.6665 | 0.6368 | 0.6382 |
| S4 | 0.7098 | 0.6991 | 0.6999 |
| **Avg** | 0.6728 | **0.6537** | 0.6585 |

**OOF 기준**: XGB 단독 0.6537 > 앙상블 0.6585 > LGBM 0.6728 → OOF는 모두 개선

**Public Score**: 앙상블 0.6218 / XGB 단독 0.6258 / XGB SHAP35 0.6343 — 모두 0.6178 대비 역효과

**원인**: XGBoost가 LOSO 훈련 fold(~45행)에 과적합. 피처 수를 35개로 줄여도 동일. OOF 개선이 Public 개선으로 이어지지 않는 LOSO 특수성이 XGBoost에서 더 강하게 나타남.

### 11단계: CatBoost 앙상블 — 신규 최고

LGBM + CatBoost 각각 타깃별 Optuna 30 trials → best_params로 10 seeds 예측 → 단순 평균 앙상블.

```python
# CatBoost 탐색 공간
iterations: 100~500, depth: 3~8, learning_rate: 0.01~0.3
l2_leaf_reg: 1e-8~10.0, random_strength: 1e-8~10.0, bagging_temperature: 0~1.0
```

| 타깃 | LGBM OOF | CatBoost OOF | 앙상블 OOF |
|------|:--------:|:-----------:|:---------:|
| Q1 | 0.7208 | 0.6956 | 0.7055 |
| Q2 | 0.6635 | 0.6780 | 0.6667 |
| Q3 | 0.6570 | 0.6608 | 0.6566 |
| S1 | 0.6427 | **0.6273** | 0.6322 |
| S2 | 0.6496 | **0.6307** | 0.6374 |
| S3 | 0.6568 | **0.6346** | 0.6399 |
| S4 | 0.7101 | **0.6902** | 0.6976 |
| **Avg** | 0.6715 | **0.6596** | 0.6623 |

**핵심**: CatBoost 최적 depth=4 (XGB는 5~7) — 더 얕은 대칭 트리로 과적합 억제. S1~S4 센서 타깃에서 LGBM 대비 강세.

**Public Score**: **0.6170 (신규 최고)** — CatBoost의 낮은 과적합 덕분에 처음으로 앙상블 효과 실현.

---

## 모델별 GroupKFold CV 결과

### F1 비교

| 타깃 | multiseed | roll14lag7 | lag_full | **optuna** |
|------|:---:|:---:|:---:|:---:|
| Q1 | 0.649 | 0.648 | 0.653 | **0.642** |
| Q2 | 0.666 | 0.673 | 0.701 | **0.713** |
| Q3 | **0.726** | 0.716 | 0.712 | 0.727 |
| S1 | 0.805 | 0.805 | 0.806 | **0.811** |
| S2 | 0.636 | 0.710 | 0.701 | **0.783** |
| S3 | 0.543 | 0.619 | 0.603 | **0.733** |
| S4 | 0.624 | 0.629 | 0.637 | **0.693** |

### 평균 LogLoss 비교

| 모델 | 평균 LogLoss | Public Score |
|------|:---:|:---:|
| lgbm_multiseed | 1.086 | 0.6924 |
| lgbm_roll14lag7 | 1.064 | 0.6890 |
| lgbm_lag_full | 1.042 | 0.6822 |
| lgbm_optuna | 0.671 | 0.6178 |
| lgbm_optuna100 | 0.6658 | 0.6228 (❌ OOF 노이즈 과적합) |
| lgbm_feat_select | 0.6709 | 0.6281 (❌ S3 급락) |
| lgbm_usage_select | 0.6721 | 0.6215 (❌ mUsageStats 과적합) |
| lgbm_stacking | 0.915 | 0.7163 (❌ 역효과) |
| xgb_optuna | 0.6537 | 0.6258 (❌ OOF↑ 불구 Public 악화) |
| xgb_shap_select | 0.6649 | 0.6343 (❌ 피처 줄여도 XGB 과적합 지속) |
| lgbm_xgb_ensemble | 0.6585 | 0.6218 (❌ OOF↑ 불구 Public 악화) |
| **lgbm_catboost_ensemble** | **0.6623** | **0.6170 (✅ 현재 최고)** |

---

## 피처 엔지니어링 상세

### 시간대 구분 (parquet v2)

| 시간대 | 범위 | 의미 |
|--------|------|------|
| morning | 06~12시 | 오전 활동 |
| afternoon | 12~18시 | 오후 활동 |
| evening | 18~22시 | 저녁 활동 |
| presleep | 22~24시 | 취침 전 |
| sleep | 00~06시 | 수면 중 |

### lag/roll 피처 조건

| 피처 | 조건 | 포착 패턴 |
|------|------|-----------|
| lag1_{t} | 날짜 간격 ≤ 2일 | 전날 상태 |
| lag2_{t} | lag1 유효 + 연속 날짜 | 이틀 전 상태 |
| lag7_{t} | target-7일 ± 2일 이내 | 주간 반복 패턴 |
| roll3_{t} | 30일 이내 직전 3개 평균 | 단기 기준선 |
| roll7_{t} | 30일 이내 직전 7개 평균 | 주간 기준선 |
| roll14_{t} | 60일 이내 직전 14개 평균 | 2주 기준선 |

### mUsageStats 피처 (13개, 현재 미사용)

| 피처 | 내용 |
|------|------|
| usage_ms_{zone} × 5 | 시간대별 총 앱 사용 시간 |
| usage_apps_{zone} × 5 | 시간대별 사용 앱 수 |
| usage_ms_total | 하루 전체 앱 사용 시간 |
| usage_presleep_ratio | 취침 전 사용 비율 |
| usage_sleep_ratio | 수면 중 사용 비율 |

> 절대값 및 subject z-score 모두 시도했으나 성능 저하 → 현재 모델에서 제외

---

## 피처 중요도 분석

> 3가지 방법론으로 135개 피처 분석. 상세 수치 전체: `submission/feature_result.md`
> - `scripts/feature_importance.py` : LightGBM gain
> - `scripts/permutation_importance.py` : Permutation Importance
> - `scripts/shap_importance.py` : SHAP (TreeExplainer)

### 전체 타깃 평균 Top-20 (LightGBM gain 기준)

| 순위 | 피처 | 설명 | 평균 importance |
|------|------|------|:---:|
| 1 | `light_mean_presleep` | 취침 전(22~24시) 조도 평균 | 95.2 |
| 2 | `hr_min_val` | 하루 최저 심박수 | 77.7 |
| 3 | `light_mean_morning` | 오전(06~12시) 조도 평균 | 75.3 |
| 4 | `screen_ratio_presleep` | 취침 전 스마트폰 화면 켜짐 비율 | 72.9 |
| 5 | `light_mean_evening` | 저녁(18~22시) 조도 평균 | 70.3 |
| 6 | `screen_ratio_afternoon` | 오후(12~18시) 화면 켜짐 비율 | 66.6 |
| 7 | `hr_mean_morning` | 오전 심박수 평균 | 64.6 |
| 8 | `hr_max_val` | 하루 최고 심박수 | 61.4 |
| 9 | `hr_std_evening` | 저녁 심박수 표준편차 | 61.3 |
| 10 | `screen_ratio_sleep` | 수면 중(00~06시) 화면 켜짐 비율 | 61.1 |
| 11 | `usage_apps_afternoon` | 오후 사용 앱 수 | 60.2 |
| 12 | `hr_std` | 하루 심박수 표준편차 | 60.0 |
| 13 | `usage_apps_evening` | 저녁 사용 앱 수 | 59.4 |
| 14 | `hr_mean_evening` | 저녁 심박수 평균 | 58.9 |
| 15 | `screen_ratio_evening` | 저녁 화면 켜짐 비율 | 57.3 |
| 16 | `usage_ms_afternoon` | 오후 총 앱 사용 시간 | 56.6 |
| 17 | `usage_apps_morning` | 오전 사용 앱 수 | 56.4 |
| 18 | `usage_ms_evening` | 저녁 총 앱 사용 시간 | 54.5 |
| 19 | `act_ratio_evening` | 저녁 활동 비율 | 54.4 |
| 20 | `hr_std_morning` | 오전 심박수 표준편차 | 54.2 |

### 타깃별 Top-10

| 순위 | Q1 (수면의 질) | Q2 (피로도) | Q3 (스트레스) | S1 (수면 시간) | S2 (수면 효율) | S3 (수면 지연) | S4 (각성 시간) |
|------|--------------|------------|--------------|--------------|--------------|--------------|--------------|
| 1 | `hr_mean` | `screen_ratio_presleep` | `light_mean_morning` | `subj_mean_Q1` | `light_mean_presleep` | `light_mean_presleep` | `light_mean_presleep` |
| 2 | `subj_mean_S1` | `hr_min_val` | `screen_ratio_presleep` | `hr_min_val` | `usage_apps_afternoon` | `hr_min_val` | `usage_apps_afternoon` |
| 3 | `light_mean_presleep` | `usage_apps_morning` | `roll3_Q1` | `subj_mean_S2` | `hr_max_val` | `screen_ratio_afternoon` | `hr_std_morning` |
| 4 | `light_mean_morning` | `subj_mean_Q3` | `hr_mean_evening` | `act_ratio_evening` | `usage_apps_evening` | `light_mean_evening` | `hr_std_evening` |
| 5 | `usage_apps_morning` | `hr_std_evening` | `day_of_month` | `screen_ratio_sleep` | `light_mean_evening` | `hr_mean_morning` | `light_mean_morning` |
| 6 | `screen_ratio_afternoon` | `screen_ratio_evening` | `hr_std` | `light_mean_evening` | `usage_ms_evening` | `act_ratio_afternoon` | `hr_min_val` |
| 7 | `hr_mean_evening` | `usage_apps_presleep` | `light_mean_presleep` | `hr_mean_morning` | `usage_ms_afternoon` | `screen_ratio_presleep` | `light_mean_sleep` |
| 8 | `usage_presleep_ratio` | `light_mean_morning` | `light_mean_afternoon` | `light_mean_presleep` | `usage_ms_total` | `usage_ms_presleep` | `usage_ms_afternoon` |
| 9 | `screen_ratio_sleep` | `hr_std` | `hr_std_presleep` | `day_of_week` | `roll14_S3` | `week_of_year` | `usage_apps_evening` |
| 10 | `hr_mean_morning` | `light_mean_presleep` | `light_mean_sleep` | `usage_presleep_ratio` | `light_max` | `hr_mean_afternoon` | `hr_std` |

### 3가지 방법론 비교 요약

| 피처 | gain 순위 | Permutation | SHAP 순위 | 해석 |
|------|:---------:|:-----------:|:---------:|------|
| `light_mean_presleep` | 1위 | 음수 (-0.0014) | 일부만 상위 | gain 과대평가 가능성 |
| `subj_mean_*` | 중위권 | ≈0 | **다수 타깃 1위** | 개인 기준선이 실제 예측 핵심 |
| `roll3_Q2` | 하위권 | Q2 **1위** | Q2 **2위** | gain이 lag/roll 과소평가 |
| `hr_min_val` | 2위 | 양수 (+0.0009) | 상위권 | 3가지 일관 → 신뢰도 높음 |
| `light_mean_morning` | 3위 | 양수 (+0.0004) | Q3 2위 | 3가지 일관 → 신뢰도 높음 |
| `screen_ratio_presleep` | 4위 | 양수 (+0.0007) | Q2 상위 | 3가지 일관 → 신뢰도 높음 |

### 주요 발견

| 구분 | 내용 |
|------|------|
| **조도(light) 피처** | gain top-5 중 3개. 단 Permutation에서 `light_mean_presleep`·`light_mean_evening` 등 음수 — 선별 필요 |
| **심박수(hr) 피처** | `hr_min_val`, `hr_max_val`, `hr_std`, `hr_std_morning` 등 3가지 방법론 모두 일관되게 중요 |
| **subj_mean_* 피처** | SHAP에서 다수 타깃 1위 — LOSO OOF에서 개인 기준선이 가장 강한 예측 신호 |
| **lag/roll 피처** | gain 하위 → Permutation/SHAP 상위. gain이 실제 기여를 과소평가. 제거 시 S3 급락 확인 |
| **mUsageStats 개별 유효성** | 전체 추가 시 과적합이나 개별 피처(`usage_apps_morning`, `usage_ms_total` 등)는 유효 신호 |
| **Permutation 음수 주의** | LOSO fold당 ~45행 → 분산 크고 음수 추정 신뢰도 낮음. 기계적 제거는 역효과 |

---

## 시도 결과 요약 (실패 포함)

| 시도 | 결과 | 원인 분석 |
|------|------|-----------|
| **2단계 스태킹** | ❌ 0.6924→0.7163 | 10명 소규모 데이터에서 Stage 2가 OOF 패턴 과적합 |
| **Subject 가중치** | ❌ 0.6822→0.7242 | 전체 train으로 학습 시 LOSO 특성 손실, 과적합 |
| **Platt Scaling** | ❌ 사용 불가 | LOSO OOF LogLoss(>1.0)가 클래스 사전확률보다 나빠 예측값이 모두 클래스 평균으로 수렴 |
| **mUsageStats 추가** | ❌ 0.6178→0.6239 | 13개 피처 추가로 450행 학습 데이터에 과적합 |
| **mUsageStats z-score** | ❌ 0.6239→0.6255 | subject 내 정규화 후에도 동일하게 과적합 |
| **Optuna 100 trials** | ❌ 0.6178→0.6228 | 노이즈 낀 LOSO OOF(10명)를 더 깊게 최적화 → 노이즈 과적합 |
| **피처 선택 (Permutation 음수 79개 제거)** | ❌ 0.6178→0.6281 | fold당 ~45행의 높은 분산 → 음수 추정 신뢰도 부족. 유효 피처 포함 제거로 S3 급락 |
| **mUsageStats 상위 3개 선별** | ❌ 0.6178→0.6215 | 선별해도 과적합 — mUsageStats는 전면 제외가 최선 |
| **XGBoost 단독** | ❌ 역효과 | OOF 0.6537(LGBM 0.671보다 좋음)이나 Public 악화. LOSO fold 과적합이 LGBM보다 심함 |
| **LGBM+XGBoost 앙상블** | ❌ 역효과 | OOF 0.6585로 개선되나 Public 악화. XGBoost 과적합이 LGBM 예측을 희석 |
| **XGBoost SHAP35 피처 선택** | ❌ 0.6178→0.6343 | 피처 35개로 줄여도 XGBoost 과적합 패턴 지속 |
| **LGBM+CatBoost 앙상블** | ✅ 0.6178→**0.6170** | CatBoost 대칭 트리의 낮은 과적합으로 처음으로 앙상블 효과 실현 |

---

## 주요 발견 및 교훈

| 발견 | 내용 |
|------|------|
| **확률 제출 효과 압도적** | hard 0/1(12.07) → 확률(0.695): 94% 개선. log-loss metric에선 필수 |
| **LOSO lag 처리** | val fold에서 lag를 train_fold로 계산 시 전부 NaN → val 자신의 데이터로 계산해야 함 |
| **동일 타깃 lag 허용 효과** | 기존에 제거하던 lag1_Q1 등을 허용하면 성능 향상. 전날의 레이블이 가장 강한 예측 신호 |
| **Optuna 효과 극적** | 30 trials만으로 OOF LogLoss 1.042→0.671, Public Score 0.6822→0.6178 |
| **피처 과다 위험** | 450행 데이터에서 13개 피처 추가(mUsageStats)가 과적합 유발 |
| **LOSO OOF 특수성** | LOSO OOF는 완전히 새로운 subject 예측 → LogLoss가 클래스 사전확률보다 높을 수 있음 |
| **subj_mean leakage** | 예측 타깃 t의 subj_mean_t 사용 시 F1=1.0 → 반드시 제거 |
| **gain ≠ 실제 중요도** | gain 1위 피처가 Permutation에서 음수인 경우 다수. Permutation·SHAP 병용 필수 |
| **Permutation 분산 주의** | LOSO fold당 ~45행 → Permutation 추정 분산 매우 큼. 음수라도 기계적 제거는 금물 |
| **subj_mean_* 실제 중요도** | gain 중위권이나 SHAP에서 다수 타깃 1위 — 개인 기준선이 수면/심리 예측의 핵심 신호 |
| **lag/roll 실제 중요도** | gain 하위권이나 Permutation/SHAP 상위권 — 제거 시 S3 급락으로 중요성 재확인 |
| **OOF ≠ Public (XGBoost 교훈)** | XGB OOF 0.6537 > LGBM OOF 0.671이나 Public은 반대. LOSO OOF가 XGBoost를 과낙관적으로 평가 |
| **CatBoost 앙상블 효과** | CatBoost의 대칭 트리(depth=4)는 XGBoost(depth=5~7)보다 과적합 적어 첫 앙상블 성공. 특히 S1~S4에서 LGBM 보완 |
| **앙상블 성공 조건** | 파트너 모델의 Public 과적합이 LGBM보다 낮아야 앙상블 효과 발생. OOF 개선폭보다 모델 편향의 일반화 능력이 더 중요 |

---

## 향후 개선 방향

> 현재 최고 점수: **0.6170** / 리더보드 1위: 0.56119 / 상위권 목표: ~0.5 이하

| 순위 | 방법 | 기대 효과 | 상태 |
|------|------|:---------:|------|
| 1 | **CatBoost 앙상블** | LGBM과 다른 편향 → 앙상블 효과 확인 | ✅ 0.6170 (현재 최고) |
| 2 | **교차 타깃 lag 피처** | 어제 S1→오늘 Q1 등 타깃 간 상관 활용 | 미시도 |
| 3 | **lag/roll 범위 확장** | roll21·roll28·lag3~6 추가, 더 긴 패턴 포착 | 미시도 |
| 4 | **Subject 트렌드 피처** | subj_mean 외 slope(기울기)·std 추가 | 미시도 |
| 5 | **OOF 기반 가중 앙상블** | 타깃별 LGBM·CatBoost 최적 비율 탐색 | 미시도 |
| 6 | **3모델 앙상블 (LGBM+CAT+기타)** | 다양성 확대 | 미시도 |

**실패로 확인된 방향**

| 방법 | 판정 | 원인 |
|------|------|------|
| ~~2단계 스태킹~~ | ❌ 0.6924→0.7163 | 소규모 데이터 과적합 |
| ~~Subject 가중치~~ | ❌ 0.6822→0.7242 | 역효과 |
| ~~Platt Scaling~~ | ❌ 사용 불가 | LOSO OOF 특성상 적용 불가 |
| ~~mUsageStats 전체/선별 추가~~ | ❌ 0.6178→0.6239~0.6215 | 규모 무관 과적합 — 전면 제외 유지 |
| ~~Optuna 100 trials~~ | ❌ 0.6178→0.6228 | LOSO OOF 노이즈 과적합 |
| ~~Permutation 음수 피처 제거~~ | ❌ 0.6178→0.6281 | 분산 큰 LOSO에서 유효 피처까지 제거됨 |
| ~~XGBoost 단독/앙상블/SHAP선택~~ | ❌ 0.6258~0.6343 | LOSO fold 과적합 — 피처 수 무관 |
| ~~LGBM+XGBoost 앙상블~~ | ❌ 0.6178→0.6218 | XGBoost 과적합이 LGBM 예측을 희석 |

---

## 환경 설정

```bash
uv pip install pandas pyarrow lightgbm scikit-learn optuna shap xgboost catboost
```

### 실행 순서

```bash
# 현재 최고 성능 모델 (LGBM + CatBoost 앙상블)
uv run python scripts/lgbm_catboost_ensemble.py

# 단일 모델 베이스라인 (LGBM Optuna)
uv run python scripts/lgbm_optuna.py

# 피처 중요도 분석
uv run python scripts/feature_importance.py      # LightGBM gain
uv run python scripts/permutation_importance.py  # Permutation Importance
uv run python scripts/shap_importance.py         # SHAP
```
