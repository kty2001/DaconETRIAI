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
│   ├── cross_target_features.csv         # 교차 타깃 피처 (700행, 86피처) ← NEW
│   └── ch2025_data_items/                # 라이프로그 parquet (12종)
├── data/
│   ├── features_all_v2.csv               # 통합 피처 v2 (700행, 162열)
│   └── features_all_v3.csv               # 통합 피처 v3 — 트렌드 피처 포함 (현재 미사용)
├── scripts/
│   ├── parquet_features_v2.py            # parquet 집계 v2 + mUsageStats
│   ├── parquet_features_v3.py            # parquet 집계 v3 — 수면 특화 심층 피처 34개 추가 (117컬럼)
│   ├── label_features.py                 # lag1/2/7 + roll3/7/14/21/28 + rollstd7/14 + trend_short/l1r7/long 빌더
│   ├── optuna_params_io.py               # Optuna best_params JSON 저장/로드 유틸
│   ├── lgbm_optuna.py                    # LightGBM Optuna 하이퍼파라미터 튜닝 (LOSO)
│   ├── lgbm_catboost_ensemble.py         # LGBM+CatBoost 앙상블 v1
│   ├── lgbm_catboost_ensemble_v2.py      # LGBM+CatBoost 앙상블 v2 (roll21/28+rollstd 피처)
│   ├── lgbm_catboost_ensemble_v3.py      # LGBM+CatBoost 앙상블 v3 — 트렌드 피처 (실패)
│   ├── lgbm_catboost_weighted_ensemble.py# OOF 가중 앙상블 시도 (실패 — NaN 문제)
│   ├── lgbm_catboost_et_ensemble.py      # LGBM+CatBoost+ET 3모델 시도 (실패 — NaN 문제)
│   ├── catboost_optuna.py                # CatBoost 단독 Optuna 튜닝
│   ├── extratrees_ensemble.py            # ExtraTrees 단독 앙상블 (현재 최고 공개점수)
│   ├── extratrees_clip005_ensemble.py    # ET clip [0.05, 0.95] + 30 seeds (Public 0.6068, 악화)
│   ├── extratrees_v3_ensemble.py         # ExtraTrees v3 피처 단독 앙상블 (z-score 없음)
│   ├── extratrees_extreme_optuna.py      # ET 극한 최적화 (300 trials, max_feat 0.05~0.15)
│   ├── extratrees_semisup_ensemble.py    # 센서 유사도 기반 준지도 ET 앙상블 (효과 없음)
│   ├── hgb_ensemble.py                   # HistGradientBoosting 단독 앙상블
│   ├── hgb_et_ensemble.py                # HGB+ET 앙상블
│   ├── hgb_et_v4_ensemble.py             # HGB+ET v4 (parquet v3 피처, hgb_v2/et_v2 파라미터)
│   ├── hgb_et_v4_optuna.py               # HGB/ET v3 파라미터 Optuna 재튜닝 (50 trials/target)
│   ├── cross_target_features.py          # 교차 타깃 피처 생성 → data/cross_target_features.csv
│   ├── hgb_et_xt_ensemble.py             # HGB+ET + 교차 타깃 피처 앙상블 (역효과)
│   ├── multitask_mlp_ensemble.py         # Multi-task MLP 단독 앙상블 (OOF 0.6398)
│   ├── mlp_hgb_et_ensemble.py            # MLP+HGB+ET 3모델 균등 앙상블 (OOF 0.6383)
│   ├── mlp_hgb_et_weighted_ensemble.py   # MLP+HGB+ET per-target 가중치 앙상블 (Public 0.6132, 악화)
│   ├── mlp_hgb_et_v4_ensemble.py         # MLP+HGB+ET v4 앙상블 (parquet v3 + hgb_v3 + et_v3)
│   ├── extratrees_optuna_v4.py           # ET max_features 0.1~0.3 집중 재탐색, 200 trials, 10 seeds
│   ├── gps_features.py                   # mGps.parquet → 19개 GPS 피처 빌더 (속도/장소/홈/회전반경)
│   ├── extratrees_gps_ensemble.py        # ET + GPS 피처 앙상블 (100 trials, 10 seeds)
│   ├── parquet_features_v4.py            # parquet 집계 v4 — 수면확장 wHr(00-09h) + mACStatus (125컬럼)
│   ├── anchor_et_ensemble.py             # 앵커 ET 앙상블 — Stage1 OOF를 피처로 추가한 2단계 학습
│   ├── wlight_features.py                # wLight.parquet → 16개 손목 조도 피처 빌더
│   ├── mble_features.py                  # mBle.parquet → 25개 BLE 스캔 피처 빌더
│   ├── feature_importance_gps.py         # ET GPS 피처 중요도 계산 스크립트 (분석용)
│   ├── extratrees_gps_slim_ensemble.py   # ET GPS Slim 90% (IMP_COVERAGE=0.90)
│   ├── extratrees_gps_slim85_ensemble.py # ET GPS Slim 85% (현재 최고 Public 0.6055)
│   ├── extratrees_gps_slim80_ensemble.py # ET GPS Slim 80% (OOF 기준 최고 0.6403)
│   ├── extratrees_gps_slim75_ensemble.py # ET GPS Slim 75% (OOF 0.6398, Public 악화 예상)
│   ├── extratrees_v3gps_slim_ensemble.py # v3 피처+GPS Slim 90% (역효과)
│   ├── extratrees_gps_pertarget_slim_ensemble.py # 타깃별 개별 slim 85%
│   ├── gru_ensemble.py                   # GRU 시계열 앙상블 (10 seeds, window=14) — 데이터 부족으로 부적합
│   └── catboost_gps_slim85_ensemble.py   # CatBoost GPS Slim 85% (실행 중)
├── submission/
│   ├── extratrees_ensemble_prob.csv      # ET 단독 앙상블 (Public 0.6061)
│   ├── mlp_hgb_et_ensemble_prob.csv      # MLP+HGB+ET 3모델 (OOF 0.6383, OOF 현재 최고)
│   ├── extratrees_v3_ensemble_prob.csv   # ET v3 단독 (Public 0.6094)
│   ├── hgb_et_xt_ensemble_prob.csv       # HGB+ET+교차타깃 (OOF 0.6453, 역효과)
│   ├── hgb_v2_ensemble_prob.csv          # HGB v2 단독 (OOF 0.6466, 미제출)
│   ├── hgb_et_ensemble_prob.csv          # HGB+ET 앙상블 (OOF 0.6434, Public 0.6103)
│   ├── multitask_mlp_ensemble_prob.csv   # MLP 단독 (OOF 0.6398, 미제출)
│   ├── hgb_et_v4_ensemble_prob.csv       # HGB+ET v4 (OOF 0.6438, 미제출)
│   ├── mlp_hgb_et_v4_ensemble_prob.csv   # MLP+HGB+ET v4 (OOF 0.6395, 미제출)
│   ├── extratrees_clip005_prob.csv       # ET clip [0.05, 0.95] 30 seeds (Public 0.6068, 악화)
│   ├── mlp_hgb_et_weighted_prob.csv      # MLP+HGB+ET 가중치 (OOF 0.6329, Public 0.6132, 악화)
│   ├── extratrees_optuna_v4_prob.csv     # ET v4 max_feat 0.1~0.3 집중 (Public 0.6051)
│   ├── extratrees_gps_prob.csv          # ET + GPS 피처 앙상블 (Public 0.6044)
│   ├── anchor_et_prob.csv               # 앵커 ET Stage2 (OOF 0.6490 — 역효과, 미제출 권장)
│   ├── extratrees_v4_ensemble_prob.csv  # ET v4 피처 + anchor_stage1 params, 10 seeds
│   ├── extratrees_wlight_prob.csv        # ET + GPS + wLight (Public 0.6053, 효과 없음)
│   ├── extratrees_gps_slim85_prob.csv   # ET GPS Slim 85% (Public 0.6055 — 현재 최고)
│   ├── extratrees_gps_slim80_prob.csv   # ET GPS Slim 80% (Public 0.6044)
│   ├── extratrees_gps_slim75_prob.csv   # ET GPS Slim 75% (OOF 0.6398)
│   ├── extratrees_gps_pertarget_slim_prob.csv # 타깃별 개별 slim
│   ├── gru_ensemble_prob.csv            # GRU 앙상블 (OOF ~0.655, 부적합)
│   └── catboost_gps_slim85_prob.csv     # CatBoost GPS Slim 85% (실행 중)
│   ├── extratrees_extreme_prob.csv       # ET 극한 최적화 (OOF 0.6466, 미제출)
│   ├── extratrees_semisup_prob.csv       # ET 준지도 (OOF 0.6459, 미제출)
│   ├── lgbm_catboost_ensemble_v2_prob.csv# LGBM+CatBoost v2 (Public 0.6127)
│   ├── optuna_params.json                # 모델별 Optuna best_params 캐시
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
| 스마트폰 | mGps | GPS 좌표/속도/고도 | ✅ 완료 (gps_features.py: 속도/장소/홈/회전반경 19개 피처 → OOF 0.6465, v2 대비 -0.0003 동률) |
| 스마트폰 | mACStatus | 충전 여부 (0/1) | ✅ 완료 (parquet_features_v4.py: sleep/presleep/morning/daily 비율 4개 피처) |
| 스마트폰 | mBle | 주변 BLE 기기 RSSI | ⏳ 미시도 (사회적 접촉·기기 연결 패턴) |
| 스마트폰 | mWifi | 주변 WiFi AP RSSI | ⏳ 미시도 (위치 규칙성·실내외 패턴) |
| 스마트워치 | wLight | 워치 조도 | ⏳ 미시도 (수면 중 광량 변화, mLight와 측정 위치 상이) |

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
| lgbm_optuna_prob | 0.6178 | Optuna 30 trials |
| lgbm_catboost_ensemble_prob | 0.6170 | LGBM + CatBoost 앙상블 v1 |
| catboost_optuna_prob | 0.6183 | CatBoost 단독 |
| lgbm_catboost_ensemble_v2_prob | 0.6127 | v2 피처 (roll21/28+rollstd) 추가 |
| lgbm_catboost_ensemble_v3_prob | 역효과 | 트렌드 피처 추가 — 성능 저하 |
| lgbm_catboost_weighted_prob | 역효과 | OOF 가중 앙상블 — NaN 문제로 OOF 0.79 |
| lgbm_catboost_et_ensemble_prob | 역효과 | 3모델 앙상블 — LGBM NaN 문제로 OOF 0.71 |
| extratrees_ensemble_prob | 0.6061 | ExtraTrees 단독 앙상블 (ET v2 기준선) |
| hgb_et_ensemble_prob | 0.6103 | HGB+ET 앙상블 |
| hgb_et_xt_ensemble_prob | 0.6078 | HGB+ET+교차 타깃 피처 (역효과) |
| mlp_hgb_et_ensemble_prob | 0.6062 | MLP+HGB+ET 3모델 (OOF 기준 최고) |
| extratrees_v3_ensemble_prob | 0.6094 | ET v3 피처 단독 (z-score 없음) |
| extratrees_clip005_prob | 0.6068 | ET clip [0.05, 0.95] + 30 seeds (역효과) |
| mlp_hgb_et_weighted_prob | 0.6132 | MLP+HGB+ET per-target 가중치 (OOF 과적합 → 역효과) |
| **extratrees_optuna_v4_prob** | **0.6051** | ET v4 max_feat 0.1~0.3 집중 재탐색, 200 trials, 10 seeds (ET v2 0.6061 대비 개선) |
| extratrees_gps_prob | 0.6044 | ET + GPS 19개 피처 (OOF 0.6465) |
| hgb_v2_ensemble_prob | 미제출 | HGB v2 단독 (OOF 0.6466, ET와 동률) |
| mlp_hgb_et_v4_ensemble_prob | 미제출 | MLP+HGB+ET v4 (OOF 0.6395) |
| extratrees_wlight_prob | 0.6053 | ET + GPS + wLight (OOF 0.6461, 개선 없음) |
| extratrees_gps_slim85_prob | **0.6055** | **현재 최고 공개점수** — ET GPS Slim 85% (OOF 0.6406, GPS 대비 +0.0021) |
| extratrees_gps_slim80_prob | 0.6044 | ET GPS Slim 80% (OOF 0.6403) |
| extratrees_gps_slim75_prob | 미제출 | ET GPS Slim 75% (OOF 0.6398, 제출 시 성능 악화 예상) |
| catboost_gps_slim85_prob | 실행 중 | CatBoost GPS Slim 85% |

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

### 11단계: CatBoost 앙상블

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

**Public Score**: **0.6170** — CatBoost의 낮은 과적합 덕분에 처음으로 앙상블 효과 실현.

### 12단계: v2 피처 확장 (roll21/28 + rollstd7/14)

lag/roll 피처에 3~4주 이동평균(roll21/28)과 단·중기 변동성(rollstd7/14)을 추가해 features_all_v2.csv 생성.

| 피처 | 윈도우 | 포착 패턴 |
|------|--------|-----------|
| roll21_{t} | 90일 이내 직전 21개 평균 | 3주 기준선 |
| roll28_{t} | 90일 이내 직전 28개 평균 | 4주 기준선 |
| rollstd7_{t} | 30일 이내 직전 7개 표준편차 | 단기 변동성 |
| rollstd14_{t} | 60일 이내 직전 14개 표준편차 | 중기 변동성 |

피처 수: 136개 → **150개** (lag/roll 42 → 70개). `optuna_params_io.py`로 Optuna best_params JSON 중앙 관리 체계 도입.

**Public Score**: **0.6127** (v1 대비 개선)

### 14단계: HistGradientBoosting 단독 앙상블

sklearn의 `HistGradientBoostingClassifier`는 NaN을 자체 처리(별도 빈 할당)해 LOSO NaN 함정에서 자유롭고, gradient boosting 계열이므로 ET와 다른 귀납적 편향을 가짐.

```python
# HGB Optuna 탐색 공간 (50/100 trials)
learning_rate: 0.01~0.3 (log), max_iter: 100~800
max_leaf_nodes: 15~80, min_samples_leaf: 10~60
l2_regularization: 0~10, max_features: 0.1~1.0
```

| 타깃 | HGB v2 OOF LL | ET OOF LL | 승자 |
|------|:-------------:|:---------:|:----:|
| Q1 | **0.6988** | 0.6998 | HGB |
| Q2 | 0.6577 | **0.6478** | ET |
| Q3 | **0.6336** | 0.6443 | HGB (+0.011) |
| S1 | **0.6141** | 0.6161 | HGB |
| S2 | 0.6193 | **0.6161** | ET |
| S3 | 0.6230 | **0.6124** | ET |
| S4 | **0.6795** | 0.6870 | HGB |
| **평균** | **0.6466** | **0.6462** | ET (≈동률) |

**핵심**: HGB와 ET는 OOF LL 기준 사실상 동률(차이 0.0004). HGB는 Q3·S1·S4에서 ET 대비 우위, ET는 Q2·S2·S3에서 우위. 모델 간 보완 관계 존재.

### 15단계: HGB+ET 앙상블 — OOF 기준 최고

서로 약점·강점이 다른 HGB와 ET를 동일 가중치로 앙상블.

| 타깃 | 앙상블 OOF LL | HGB LL | ET LL |
|------|:------------:|:------:|:-----:|
| Q1 | **0.6974** | 0.6988 | 0.6998 |
| Q2 | 0.6509 | 0.6577 | **0.6478** |
| Q3 | **0.6330** | 0.6336 | 0.6443 |
| S1 | **0.6109** | 0.6141 | 0.6161 |
| S2 | **0.6146** | 0.6193 | 0.6161 |
| S3 | 0.6149 | 0.6230 | **0.6124** |
| S4 | 0.6823 | **0.6795** | 0.6870 |
| **평균** | **0.6434** | 0.6466 | 0.6462 |

Q1·Q3·S1·S2 4개 타깃에서 두 모델 모두를 동시에 이김. **Jensen 부등식 효과** — log-loss는 확률에 대해 볼록(convex)함수이므로, 두 예측의 평균이 각 예측의 log-loss 평균보다 항상 낮거나 같음.

**OOF LL 0.6434** (ET 0.6462 대비 0.003 개선) — 드라마틱하지 않으나 통계적으로 의미 있음. 공개점수 미검증.

### 13단계: ExtraTrees 단독 앙상블 — 신규 최고

LGBM/CatBoost의 LOSO 성능 저하 원인 분석 결과, val subject의 `subj_mean` 피처 NaN이 gradient boosting 계열에 치명적임을 확인. ExtraTrees는 fold별 train median imputation으로 NaN을 처리하며 LOSO에서 강한 일반화 성능을 보임.

```python
# ExtraTrees Optuna best params (주요 타깃)
# S1: n_est=516, depth=12, max_features=0.11  → OOF LL 0.6152
# S3: n_est=588, depth=15, max_features=0.11  → OOF LL 0.6136
# max_features=0.11 (11%) — 극히 적은 피처 샘플링으로 subject 간 과적합 방지
```

| 타깃 | ET F1 | ET OOF LL | v2앙상블 OOF LL |
|------|:-----:|:---------:|:--------------:|
| Q1 | 0.548 | 0.6986 | ~0.709 |
| Q2 | 0.711 | 0.6519 | ~0.665 |
| Q3 | 0.753 | 0.6417 | ~0.656 |
| S1 | 0.807 | 0.6152 | ~0.633 |
| S2 | 0.768 | 0.6196 | ~0.639 |
| S3 | 0.797 | 0.6136 | ~0.634 |
| S4 | 0.690 | 0.6875 | ~0.697 |
| **평균** | **0.725** | **0.6469** | **0.6618** |

**Public Score**: **0.6061 (현재 최고)** — OOF LL 기준으로도 v2 앙상블(0.6618)보다 개선.

**핵심 발견**: LOSO 환경에서 gradient boosting(LGBM/CatBoost)보다 ExtraTrees의 완전 무작위 분할이 unseen subject 일반화에 유리. max_features=0.11이라는 극단적 피처 샘플링이 오히려 subject 간 노이즈를 효과적으로 차단.

### 16단계: 교차 타깃 lag 피처 — 역효과

타깃 간 시계열 관계를 포착하기 위해 86개 새 피처를 `data/cross_target_features.csv`로 사전 생성.

| 피처 유형 | 수 | 내용 |
|-----------|:--:|------|
| `xcorr_{t1}_{t2}_14` | 21 | 타깃 쌍 rolling Pearson correlation (14개 이내, 60일) |
| `xcorr_{t1}_{t2}_28` | 21 | 타깃 쌍 rolling Pearson correlation (28개 이내, 90일) |
| `lag1_diff_{t1}_{t2}` | 21 | 전날 타깃 쌍 차이 (∈ {-1, 0, 1}) |
| `momentum_{t1}_{t2}` | 21 | 두 타깃 변화 방향 일치 여부 ((Δt1)×(Δt2)) |
| `n_pos_lag1` | 1 | 전날 양성 타깃 수 (0~7) |
| `n_pos_roll7` | 1 | 최근 7일 평균 양성 타깃 수 |

| 타깃 | HGB+ET OOF LL | HGB+ET+XT OOF LL | 변화 |
|------|:------------:|:----------------:|:----:|
| Q1 | 0.6994 | 0.6983 | -0.001 |
| Q2 | 0.6478 | 0.6454 | -0.002 |
| Q3 | 0.6443 | **0.6312** | **-0.013** |
| S1 | 0.6161 | **0.6083** | **-0.008** |
| S2 | 0.6161 | 0.6255 | +0.009 |
| S3 | 0.6124 | 0.6215 | +0.009 |
| S4 | 0.6870 | 0.6866 | -0.000 |
| **평균** | **0.6434** | **0.6453** | **+0.002 (악화)** |

**역효과 원인 분석**:
1. **정보 중복** — `lag1_diff_{t1}_{t2}`·`n_pos_lag1`은 기존 `lag1_Q1 ... lag1_S4`의 선형 결합 → 새 정보 없음
2. **소표본 상관계수 불안정** — 이진 데이터 × 14개 관측으로 Pearson correlation이 노이즈에 민감
3. **momentum NaN 42%** — lag1·lag2 모두 유효해야 계산 가능, 데이터 부족으로 절반 가까이 결측

**결론**: 교차 타깃 lag 피처는 효과 없음. HGB+ET 기준(OOF 0.6434) 유지.

### 17단계: Multi-task MLP 단독 앙상블

트리 계열과 완전히 다른 귀납적 편향 확보를 위해 PyTorch 기반 Multi-task MLP 도입.

**아키텍처**
```
Input (144피처: subj_mean 7개 + mUsageStats 13개 제거)
  → [Linear(256) → BN → ReLU → Drop(0.37)]
  → [Linear(128) → BN → ReLU → Drop(0.37)]
  → Q1 / Q2 / Q3 / S1 / S2 / S3 / S4  (각각 Linear(128→1) + Sigmoid)
BCEWithLogitsLoss 7타깃 평균 + Adam + CosineAnnealingLR
```

- **subj_mean 전체 제거**: 트리 모델과의 오차 독립성 확보 (다양성 의도적 설계)
- **Optuna 30 trials**: Best OOF avg_ll = 0.6454 (hidden=[256,128], drop=0.367, lr=2e-4, ep=50)
- **10 seeds 멀티 시드**: 최종 OOF **0.6398**

| 타깃 | MLP LL | HGB+ET LL | 비교 |
|------|:------:|:---------:|:----:|
| Q1 | 0.7145 | 0.6994 | ❌ 악화 (+0.015) |
| Q2 | 0.6587 | 0.6478 | ❌ 악화 (+0.011) |
| Q3 | 0.6514 | 0.6443 | ❌ 악화 (+0.007) |
| **S1** | **0.6055** | 0.6161 | ✅ 개선 (-0.011) |
| **S2** | **0.5841** | 0.6161 | ✅ **개선 (-0.032)** |
| **S3** | **0.5847** | 0.6124 | ✅ **개선 (-0.028)** |
| **S4** | **0.6798** | 0.6870 | ✅ 개선 (-0.007) |
| **평균** | **0.6398** | 0.6434 | ✅ **-0.0036** |

**핵심 발견**: Q1~Q3(주관적 설문 타깃)은 subj_mean 없이 약하지만, S1~S4(객관적 센서 타깃)에서 MLP가 HGB+ET를 크게 앞섬. 특히 S2(-0.032), S3(-0.028)에서 두드러진 우위.

### 18단계: MLP + HGB + ET 3모델 앙상블 — OOF 기준 최고

MLP(센서 강점) + HGB+ET(설문 강점)의 상호 보완 관계를 1/3 동일 가중치로 결합.

| 타깃 | 3모델 앙상블 | HGB+ET | MLP | 개선 |
|------|:-----------:|:------:|:---:|:----:|
| Q1 | 0.6994 | 0.6994 | 0.7145 | ±0.000 |
| Q2 | 0.6500 | 0.6478 | 0.6587 | +0.002 ❌ |
| Q3 | **0.6342** | 0.6443 | 0.6514 | -0.010 ✅ |
| S1 | **0.6066** | 0.6161 | 0.6055 | -0.010 ✅ |
| S2 | **0.6008** | 0.6161 | 0.5841 | -0.015 ✅ |
| S3 | **0.5990** | 0.6124 | 0.5847 | -0.013 ✅ |
| S4 | **0.6782** | 0.6870 | 0.6798 | -0.009 ✅ |
| **평균** | **0.6383** | 0.6434 | 0.6398 | **-0.0051** ✅ |

Q2만 소폭 악화(MLP가 Q2에서 약하기 때문), 나머지 6개 타깃 개선. **OOF 0.6383** — 지금까지 최고.

**OOF 누적 성능**
```
ET 단독:      0.6469  (Public 0.6061)
HGB+ET:       0.6434
MLP 단독:     0.6398
MLP+HGB+ET:   0.6383  ← 현재 최고
```

### 19단계: parquet 피처 v3 심층 재설계

기존 parquet v2가 시간대별 평균/표준편차 수준에 그쳤던 한계를 극복. 수면 특화 심층 피처 34개 추가 → `parquet_features_v3.py`

| 센서 | 신규 피처 | 포착 신호 |
|------|-----------|-----------|
| **wHr** | `hr_sleep_min_abs` — 수면 구간 실제 최솟값 | 수면 중 최저 심박수 (nocturnal dip) |
| **wHr** | `hr_sleep_intra_var` — 시간 내 std 평균 | 수면 심박 변동성 (REM/NREM 주기 근사) |
| **wHr** | `hr_presleep_to_sleep_drop` — 취침 전→수면 심박 강하폭 | 수면 진입 심박 반응 |
| **wHr** | `hr_nocturnal_dip` — 저녁 대비 수면 심박 강하율 | 자율신경계 야간 이완 |
| **mScreenStatus** | `screen_presleep_last_minute` — 22시 기준 마지막 켜짐 경과 분 | 취침 전 스크린 노출 종료 시각 |
| **mScreenStatus** | `screen_presleep_max_run` — 취침 전 최대 연속 켜짐 구간 | 취침 전 연속 화면 사용 시간 |
| **mScreenStatus** | `screen_sleep_any` — 수면 중 화면 켜짐 여부 | 야간 스마트폰 사용 유무 |
| **mAmbience** | `amb_sleep_quiet_ratio`, `amb_sleep_noisy_ratio` | 수면 구간 소음 노출 강도 |
| **mAmbience** | `amb_presleep_*` (취침 전 구간 분리) | 취침 전 소음/quiet 환경 비율 |
| **wPedo** | `pedo_active_hours` — step>100 시간 수 | 하루 실제 활동 시간 |
| **wPedo** | `pedo_walk_ratio` — 활동 시간 / 전체 시간 | 생활 활동도 |
| **wPedo** | `pedo_evening_step_ratio` — 저녁 걸음 비율 | 저녁 활동 집중도 |

**결과**: 700행, 83컬럼(v2) → 117컬럼(v3, +34개)

### 20단계: HGB+ET v4 — v3 피처 검증 (v2 파라미터 유지)

v3 피처의 효과를 기존 파라미터(hgb_v2, extratrees_v2)로 검증 (`hgb_et_v4_ensemble.py`).

| 타깃 | v4 LL | v2 LL | 차이 |
|------|:-----:|:-----:|:----:|
| Q1 | 0.6966 | 0.6974 | **-0.0008** ✅ |
| Q2 | 0.6521 | 0.6509 | +0.0012 ❌ |
| Q3 | 0.6332 | 0.6330 | +0.0002 ≈ |
| S1 | 0.6135 | 0.6109 | +0.0026 ❌ |
| S2 | 0.6148 | 0.6146 | +0.0002 ≈ |
| S3 | 0.6142 | 0.6149 | **-0.0007** ✅ |
| S4 | 0.6820 | 0.6823 | -0.0003 ✅ |
| **평균** | **0.6438** | **0.6434** | **+0.0004 (사실상 동률)** |

피처 수: 149개(v2) → 184개(v3, +35). 전체 OOF는 동률이나 ET 단독으로 보면 7/7 타깃 개선, HGB는 5/7 악화 → **v3 피처는 ET 친화적, HGB 비친화적**.

### 21단계: Optuna v3 재튜닝 (hgb_v3, extratrees_v3)

v3 피처 기준으로 HGB(50 trials) + ET(50 trials, max_feat 0.05~0.5 세밀화) 재탐색 (`hgb_et_v4_optuna.py`).

**Phase 1 — HGB v3 (5/7 타깃 악화)**

| 타깃 | hgb_v3 LL | hgb_v2 LL | 변화 |
|------|:---------:|:---------:|:----:|
| Q1 | 0.7042 | 0.6988 | +0.0054 ❌ |
| Q2 | 0.6623 | 0.6577 | +0.0046 ❌ |
| Q3 | 0.6307 | 0.6336 | **-0.0029** ✅ |
| S1 | 0.6085 | 0.6141 | **-0.0056** ✅ |
| S2 | 0.6209 | 0.6193 | +0.0016 ❌ |
| S3 | 0.6311 | 0.6230 | +0.0081 ❌ |
| S4 | 0.6817 | 0.6795 | +0.0022 ❌ |

**Phase 2 — ET v3 (7/7 타깃 개선)**

| 타깃 | et_v3 LL | et_v2 LL | 변화 |
|------|:--------:|:--------:|:----:|
| Q1 | 0.6919 | 0.6998 | **-0.0079** ✅ |
| Q2 | 0.6414 | 0.6478 | **-0.0064** ✅ |
| Q3 | 0.6344 | 0.6443 | **-0.0099** ✅ |
| S1 | 0.6074 | 0.6161 | **-0.0087** ✅ |
| S2 | 0.6082 | 0.6161 | **-0.0079** ✅ |
| S3 | 0.6039 | 0.6124 | **-0.0085** ✅ |
| S4 | 0.6761 | 0.6870 | **-0.0109** ✅ |

**Phase 3 — HGB_v3+ET_v3 앙상블**: OOF 0.6441 (v2 앙상블 0.6434 대비 +0.0007 악화)

ET v3의 7/7 타깃 개선이 HGB v3의 악화에 상쇄됨. **핵심 발견**: v3 피처는 HGB gradient boosting에 노이즈로 작용, ET 무작위 분할은 새 피처 중 유용한 것만 선택적 활용.

### 22단계: MLP+HGB+ET v4 앙상블

parquet v3 피처 + hgb_v3 + extratrees_v3 + mlp_v1 파라미터로 3모델 앙상블 (`mlp_hgb_et_v4_ensemble.py`). MLP 입력: ~178개(v2 144개 대비 +34).

| 타깃 | v4 앙상블 LL | v2 앙상블 LL | 차이 |
|------|:-----------:|:-----------:|:----:|
| Q1 | **0.6951** | 0.6994 | **-0.0043** ✅ |
| Q2 | 0.6489 | 0.6478 | +0.0011 ❌ |
| Q3 | **0.6393** | 0.6443 | **-0.0050** ✅ |
| S1 | **0.6092** | 0.6161 | **-0.0069** ✅ |
| S2 | **0.6052** | 0.6161 | **-0.0109** ✅ |
| S3 | **0.6013** | 0.6124 | **-0.0111** ✅ |
| S4 | **0.6774** | 0.6870 | **-0.0096** ✅ |
| **평균** | **0.6395** | **0.6434** | **-0.0039 (ET v2 대비)** |

6/7 타깃에서 ET v2 LL 대비 개선. 하지만 v2 MLP+HGB+ET 앙상블(0.6383) 대비 +0.0012 악화. **HGB v3 파라미터가 발목** — v3 피처 기반 HGB 재튜닝이 오히려 역효과를 일으켜 전체 앙상블을 끌어내림.

**OOF 누적 성능 (v4 실험 포함)**
```
ET 단독:            0.6469  (Public 0.6061)
HGB+ET v2:          0.6434  (Public 0.6103)
MLP+HGB+ET v2:      0.6383  (Public 0.6062) ← OOF 현재 최고
HGB+ET v4:          0.6438  (v3 피처, v2 파라미터 — 사실상 동률)
HGB+ET Optuna v4:   0.6441  (v3 피처, v3 파라미터 — 소폭 악화)
MLP+HGB+ET v4:      0.6395  (v3 피처, v3 파라미터 — 소폭 악화)
ET v3 단독:         0.6438  (Public 0.6094 — ET v2 대비 Public 악화)
ET 극한 최적화:     0.6466  (OOF ET v2 대비 +0.0004 악화)
ET 준지도:          0.6459  (≈ ET v2 동률 — train/test 동일 10명으로 효과 없음)
ET clip005 30s:     0.6464  (Public 0.6068 — clip 완화 역효과)
ET 가중치 앙상블:   0.6329  (Public 0.6132 — OOF 과적합, Public 역전)
```

### 23단계: clip 범위 완화 + seeds 30개 — OOF 무변화

ET v2 파라미터 기반으로 clip 범위를 [0.1, 0.9] → [0.05, 0.95]로 완화하고 seeds를 10개 → 30개로 확장 (`extratrees_clip005_ensemble.py`).

| 타깃 | clip005 LL | v2 LL | 차이 |
|------|:----------:|:-----:|:----:|
| Q1 | 0.6998 | 0.6998 | ±0.0000 |
| Q2 | 0.6488 | 0.6478 | +0.0010 ❌ |
| Q3 | 0.6435 | 0.6443 | -0.0008 ✅ |
| S1 | 0.6160 | 0.6161 | -0.0001 ✅ |
| S2 | 0.6167 | 0.6161 | +0.0006 ❌ |
| S3 | 0.6124 | 0.6124 | ±0.0000 |
| S4 | 0.6876 | 0.6870 | +0.0006 ❌ |
| **평균** | **0.6464** | **0.6462** | **+0.0002 (사실상 동률)** |

예측 분포: S1 max=0.950, S2 max=0.918, S3 max=0.939 — clip 완화로 고신뢰 예측이 보존됨. OOF는 clip 범위에 무관(OOF는 원시 확률로 계산)하므로 Public 제출 시에만 효과 확인 가능.

**Public 결과: 0.6068 (ET v2 0.6061 대비 악화)** — clip 완화가 오히려 극단 예측의 페널티를 증가시킴.

### 24단계: MLP+HGB+ET per-target 가중치 앙상블 — OOF 역전

각 모델의 OOF를 분리 수집 후 타깃별 Nelder-Mead 최적화로 최적 가중치 탐색 (`mlp_hgb_et_weighted_ensemble.py`).

**최적 가중치 결과**

| 타깃 | w_MLP | w_HGB | w_ET | 최적 LL | 균등 LL | 개선 |
|------|:-----:|:-----:|:----:|:-------:|:-------:|:----:|
| Q1 | 0.014 | 0.513 | 0.473 | 0.6979 | 0.6998 | +0.0018 |
| Q2 | 0.199 | 0.134 | 0.667 | 0.6494 | 0.6504 | +0.0010 |
| Q3 | 0.030 | 0.713 | 0.257 | 0.6317 | 0.6347 | +0.0030 |
| S1 | 0.728 | 0.214 | 0.058 | 0.6050 | 0.6068 | +0.0018 |
| **S2** | **1.000** | 0.000 | 0.000 | **0.5846** | 0.6007 | **+0.0160** |
| **S3** | **1.000** | 0.000 | 0.000 | **0.5861** | 0.5998 | **+0.0136** |
| S4 | 0.463 | 0.537 | 0.000 | 0.6758 | 0.6784 | +0.0027 |
| **평균** | — | — | — | **0.6329** | **0.6383** | **-0.0054** |

OOF 기준 전 타깃 개선, 평균 -0.0054로 역대 최고 OOF. 그러나 **Public 0.6132 — ET v2 0.6061 대비 크게 악화**.

**원인 분석**: S2·S3에서 MLP 100% 같은 극단 가중치가 450행 OOF 노이즈에 과적합. OOF가 낮아질수록 Public이 나빠지는 역전 현상 발생. 가중치 최적화 자체가 LOSO OOF라는 소규모·노이즈 많은 지표를 과도하게 최적화한 결과.

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
| lgbm_optuna | 0.671 | 0.6178 |
| lgbm_catboost_ensemble | 0.6623 | 0.6170 |
| lgbm_catboost_ensemble_v2 | 0.6616 | 0.6127 (✅ v2 피처 개선) |
| lgbm_catboost_ensemble_v3 | 0.6618 | 역효과 (❌ 트렌드 피처 추가 실패) |
| lgbm_catboost_weighted | 0.7501 | 역효과 (❌ NaN 문제 — OOF 신뢰 불가) |
| lgbm_catboost_et_ensemble | 0.7074 | 역효과 (❌ LGBM NaN 문제 OOF 1.009) |
| **extratrees_ensemble** | **0.6469** | **0.6061 (✅ 현재 최고 공개점수)** |
| hgb_ensemble (v2, 100 trials) | 0.6466 | 미제출 (ET와 동률) |
| hgb_et_ensemble | 0.6434 | 0.6103 |
| hgb_et_xt_ensemble | 0.6453 | 0.6078 (❌ 교차 타깃 피처 추가 — 역효과) |
| multitask_mlp_ensemble | 0.6398 | 미제출 (✅ S1~S4 센서 타깃 강점) |
| **mlp_hgb_et_ensemble** | **0.6383** | **0.6062 (✅ OOF 기준 최고)** |
| hgb_et_v4_ensemble | 0.6438 | 미제출 (parquet v3 피처, v2 파라미터 — ET 7/7 개선, HGB 5/7 악화) |
| hgb_et_v4_optuna | 0.6441 | 미제출 (❌ v3 파라미터 재튜닝 — ET 7/7 개선이 HGB 악화에 상쇄) |
| mlp_hgb_et_v4_ensemble | 0.6395 | 미제출 (❌ v3 피처+파라미터 — HGB v3 발목, v2 앙상블 대비 +0.0012 악화) |
| extratrees_v3_ensemble | 0.6438 | 0.6094 (❌ ET v2 대비 Public 악화) |
| extratrees_extreme | 0.6466 | 미제출 (❌ OOF ET v2 대비 +0.0004 악화) |
| extratrees_semisup | 0.6459 | 미제출 (≈ ET v2와 동률 — train/test 동일 10명으로 효과 없음) |
| extratrees_clip005 (clip 0.05~0.95, 30 seeds) | 0.6464 | 0.6068 (❌ clip 완화 역효과) |
| mlp_hgb_et_weighted (per-target 가중치) | 0.6329 | 0.6132 (❌ OOF 최적화 과적합 — OOF-Public 역전) |
| extratrees_optuna_v4 (max_feat 0.1~0.3) | ~0.6462 | **0.6051** (✅ ET v2 대비 개선) |
| extratrees_gps (GPS 19개 피처) | 0.6465 | Public 0.6044 |
| anchor_et Stage2 (v4 피처+OOF 앵커) | 0.6490 | 미제출 (❌ Stage2 앵커 역효과 — 220피처 과적합) |
| extratrees_v4_ensemble (v4 피처, 10 seeds) | 0.6479 | 미확인 (❌ OOF v2 0.6462 대비 +0.0017 악화. Q1/Q2/S1 개선, Q3/S2/S3 악화) |
| ET + GPS + wLight | 0.6461 | 효과 없음 (GPS 동률) |
| ET + GPS + mBle | 0.6461 | 효과 없음 (GPS 동률) |
| **ET GPS Slim 90%** | **0.6422** | GPS 피처 + 상위 90% 커버 피처만 선택 (GPS 대비 +0.0043 개선) |
| **ET GPS Slim 85%** | **0.6406** | **Public 0.6055 (현재 최고)** — slim90 대비 추가 개선 |
| **ET GPS Slim 80%** | **0.6403** | Public 0.6044 — slim85 대비 Public 악화 |
| ET GPS Slim 75% | 0.6398 | OOF 지속 개선. Public 악화 예상 (피처 과도 제거) |
| v3GPS Slim 90% | 0.6425 | parquet v3+GPS Slim 90% — v2 Slim 대비 역효과 |
| GRU ensemble (10 seeds) | ~0.655 | 훈련 윈도우 310개로 데이터 부족 — 부적합 |
| CatBoost GPS Slim 85% | 실행 중 | — |

### 25단계: 추가 센서 피처 — wLight, mBle (효과 없음)

GPS 성공에 이어 미사용 센서 소스 추가 실험.

**wLight (손목 조도, 16개 피처)**

| 피처 | 내용 |
|------|------|
| `wlight_{zone}_mean/std` | 시간대별 조도 평균/표준편차 |
| `wlight_sleep_dark_ratio` | 수면 구간 어두운 비율 |
| `wlight_presleep_to_sleep_drop` | 취침 전 → 수면 조도 강하폭 |

OOF 0.6461 — GPS(0.6465)와 동률. GPS가 이미 이동 패턴을 포착해 wLight 정보 중복.

**mBle (BLE 스캔, 25개 피처)**

| 피처 | 내용 |
|------|------|
| `ble_devices_per_scan_mean/std` | 스캔당 기기 수 통계 |
| `ble_n_unique_daily` | 하루 고유 기기 수 |
| `ble_close/medium_ratio` | RSSI 기반 근접/중거리 비율 |

OOF 0.6461 — GPS와 동률. BLE 기기 패턴이 LOSO 환경(10명)에서 cross-subject 일반화 실패.

**핵심 교훈**: 피처 추가보다 **피처 제거(slim)**가 LOSO 환경에서 더 효과적.

### 26단계: Feature Importance Slim — 현재 최고 공개점수

ExtraTrees Feature Importance 기반 피처 선택으로 노이즈 피처 제거.

**방법론**
```
Phase 0: 기존 extratrees_gps params로 10 LOSO fold × 7 타깃 ET 학습 → 피처 중요도 계산
         cumulative importance >= N% 커버 피처만 유지 (나머지 제거)
Phase 1: slim 피처셋으로 Optuna 재튜닝 (100 trials/target)
Phase 2: 10 seeds 앙상블
```

**slim 비율별 결과**

| 모델 | 피처 수 | OOF LL | Public | 비고 |
|------|:-------:|:------:|:------:|------|
| GPS (기준) | 184개 | 0.6465 | 0.6044 | — |
| Slim 90% | 132개 | 0.6422 | — | +0.0043 |
| **Slim 85%** | **~115개** | **0.6406** | **0.6055** | **현재 최고 Public** |
| Slim 80% | ~100개 | 0.6403 | 0.6044 | slim85 대비 Public 악화 |
| Slim 75% | 87개 | 0.6398 | — | Public 악화 예상 |

**핵심 발견**: OOF는 slim 비율을 낮출수록 단조 개선(87개에서 0.6398). 그러나 Public은 slim85(0.6055)가 최적 — 그 이하는 과도한 피처 제거로 일반화 손실. 피처 제거의 sweet spot 존재.

**slim 피처 선택 원리**: 전체 184개 피처 중 하위 중요도 피처들은 cross-subject 일반화에 노이즈로 작용. 상위 85% 커버 피처만 사용함으로써 subject별 특이 패턴에 과적합하는 피처를 차단.

### 27단계: GRU 시계열 모델 — 데이터 부족으로 부적합

시계열 의존성 명시 포착을 위해 GRU 도입.

**아키텍처**
```
Input: 과거 14일 [센서 피처 + 레이블] → predict 오늘의 7 타깃
SleepGRU: GRU(hidden=32, n_layers=1, dropout=0.4) + Linear(7)
LOSO 학습, 10 seeds, BCEWithLogitsLoss
```

| seed | OOF LL |
|------|:------:|
| 42 | 0.6629 |
| 123 | 0.6573 |
| ... | ... |
| **평균** | **~0.655** |

**실패 원인**: 310개 훈련 윈도우로 GRU 파라미터 대비 데이터 부족. lag/roll 피처가 이미 시계열 의존성을 충분히 포착. 트리 계열(ET Slim 85% OOF 0.6406)에 크게 뒤짐.

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
| **LGBM+CatBoost v2 피처** | ✅ 0.6170→**0.6127** | roll21/28+rollstd7/14 피처 확장 효과 |
| ~~Subject 트렌드 피처 (trend7/14, dev_roll7/14)~~ | ❌ 역효과 | 7/14일 기울기·편차 피처가 노이즈 증가. OOF 개선 없고 Public 악화 |
| ~~OOF 기반 가중 앙상블~~ | ❌ OOF 0.7501 (비정상) | LOSO val subject의 subj_mean 6개가 NaN → LGBM/CatBoost 예측 붕괴 |
| ~~LGBM+CatBoost+ET 3모델 앙상블~~ | ❌ OOF 0.7074 | 동일 NaN 문제로 LGBM OOF 1.009. ET(0.647)만 정상 — LGBM이 앙상블 전체를 끌어내림 |
| **ExtraTrees 단독** | ✅ OOF 0.6469→**0.6088** | median imputation으로 NaN 해결. max_feat=0.11의 극단적 피처 샘플링이 LOSO 일반화에 최적 |
| **HGB 단독** | ✅ OOF 0.6466 (미제출) | NaN 자체 처리. ET와 OOF 동률이나 타깃별 상이한 강·약점 보유 |
| **HGB+ET 앙상블** | ✅ OOF **0.6434** (미제출) | Jensen 부등식 효과로 두 모델 동시 개선. 공개점수 미확인 |
| ~~HGB+ET + 교차 타깃 피처(86개)~~ | ❌ OOF 0.6453 (악화) | xcorr 소표본 불안정 + lag1_diff는 기존 피처의 선형 결합으로 정보 중복 |
| **Multi-task MLP 단독** | ✅ OOF **0.6398** (미제출) | subj_mean 제거로 트리와 오차 독립성 확보. S2(-0.032), S3(-0.028) 대폭 개선. Q1~Q3 취약 |
| **MLP+HGB+ET 3모델 앙상블** | ✅ OOF **0.6383** (미제출, 현재 최고) | MLP 센서 강점 + HGB+ET 설문 강점 결합. 6/7 타깃 개선 |
| **parquet v3 피처 재설계** | ✅ ET 7/7 개선 (HGB 5/7 악화) | wHr 수면 심박 변동성, mScreenStatus 취침 전 행동, mAmbience 구간 분리, wPedo 활동 패턴 — ET에만 효과적 |
| **HGB+ET v4 (v3 피처+v2 파라미터)** | OOF 0.6438 (v2 0.6434 대비 사실상 동률) | ET 개선이 HGB 악화에 상쇄. v3 피처의 HGB 비친화성 확인 |
| **Optuna v3 재튜닝 (hgb_v3, et_v3)** | ❌ OOF 0.6441 (소폭 악화) | HGB v3 5/7 타깃 악화로 ET v3 7/7 개선 효과 상쇄. ET v3 파라미터는 유효 |
| **MLP+HGB+ET v4 앙상블** | ❌ OOF 0.6395 (v2 0.6383 대비 +0.0012 악화) | HGB v3 파라미터 발목. ET/MLP 기여로 ET v2 대비 6/7 개선이나 전체 앙상블은 v2 대비 열세 |
| **ET v3 단독 앙상블 (z-score 없음)** | ❌ OOF 0.6438, Public 0.6094 | OOF 개선(ET v2 0.6462 대비 -0.0024)에도 Public 악화. LOSO 과적합 의심 |
| **ET 극한 최적화 (300 trials, max_feat 0.05~0.15)** | ❌ OOF 0.6466 (+0.0004 악화) | Q3 탐색 상한(0.15)에서 최적값 발견 → 탐색 공간 제약이 발목. ET v2 기본 범위(0.1~1.0)가 더 안정적 |
| **준지도 학습 (센서 유사도 기반 subj_mean 보정)** | ≈ OOF 0.6459 (ET v2 동률) | train/test 동일 10명 구조에서 cosine similarity가 자기 자신을 top-1으로 찾음 → 효과 없음 |
| **ET clip [0.05, 0.95] + 30 seeds** | OOF 0.6464 ≈ 동률, Public 0.6068 (❌ 역효과) | clip 완화로 극단 예측 보존 → 오히려 틀린 고신뢰 예측의 페널티 증가 |
| **MLP+HGB+ET per-target 가중치 앙상블** | OOF 0.6329 (최고), Public 0.6132 (❌ 크게 악화) | S2·S3 MLP 100% 극단 가중치가 450행 OOF 노이즈에 과적합. OOF 개선이 Public 악화로 역전되는 패턴 확인 |
| **ET Optuna v4 재탐색 (max_feat 0.1~0.3)** | OOF ~0.6462, **Public 0.6051** | max_feat 0.1~0.3 집중 200 trials, 10 seeds. ET v2(0.6061) 대비 개선 |
| **ET + GPS 피처 (mGps 19개)** | OOF 0.6465 (v2 동률), **Public 0.6044** | Q1(+0.0026), S1(+0.0017), S3(+0.0003) 개선. Q3(-0.0036), Q2(-0.0008) 악화. OOF 동률이나 Public 개선 — GPS가 test 분포에서 유효한 추가 정보 제공 |
| **앵커 ET (v4 피처 + OOF 앵커, 윈도우 페어)** | Stage1 OOF=0.6457 / Stage2 OOF=0.6490 (역효과) | parquet_v4 + trend 피처 Stage1(1 seed)은 v2 0.6462 대비 개선. Stage2 앵커(220피처) 450행 과적합. Q1만 개선(-0.0039) |
| **ET v4 앙상블 (v4 피처, anchor_stage1 params, 10 seeds)** | OOF 0.6479 (v2 대비 +0.0017 악화) | Q1/Q2/S1 개선, Q3/S2/S3 악화. v4 피처가 OOF에서 일관적 개선 미흡. Public 제출 필요 |
| **ET + GPS + wLight (16개 피처)** | OOF 0.6461 (GPS 동률) | 손목 조도 피처 16개 추가. 효과 없음 — GPS가 이미 이동 패턴을 포착해 wLight 중복 |
| **ET + GPS + mBle (25개 피처)** | OOF 0.6461 (GPS 동률) | BLE 스캔 피처 25개 추가. 효과 없음 — 소규모 LOSO에서 BLE 기기 패턴이 일반화 안 됨 |
| **ET GPS Feature Importance Slim (90/85/80/75%)** | OOF 0.6422/0.6406/0.6403/0.6398 | 피처 중요도 상위 N% 커버 피처만 선택 → 하위 피처 노이즈 제거. slim85 Public 0.6055 (현재 최고). slim75는 피처 과도 제거로 Public 악화 예상 |
| **GRU ensemble (window=14, 10 seeds)** | OOF ~0.655 | 14일 윈도우 GRU 시계열 모델. 훈련 윈도우 310개 부족으로 부적합. lag/roll 피처가 이미 시계열 의존성 포착 |

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
| **LOSO NaN 함정** | LOSO val fold에서 val subject가 train_fold에 없으므로 subj_mean 6개 피처가 NaN. LGBM/CatBoost는 이 NaN을 기본 branch로 처리해 예측이 붕괴(OOF 1.0 이상). ET는 median imputation으로 해결 |
| **ExtraTrees LOSO 적합성** | gradient boosting 계열보다 LOSO에서 강한 이유: ① 완전 무작위 분할로 subject 패턴 과적합 방지 ② max_features=0.11 극단적 샘플링으로 특정 subject 의존 피처 차단 ③ iteration 없어 fold당 45행에서도 안정 |
| **Optuna params 캐시 관리** | `{model}_{feature_version}` 키 체계로 optuna_params.json 중앙 관리. 동일 피처셋·모델이면 재탐색 없이 재사용 |
| **HGB vs ET 특성 비교** | HGB: Q3·S1·S4 강, ET: Q2·S2·S3 강. 앙상블 시 Q1·Q3·S1·S2 4개 타깃에서 Jensen 부등식으로 두 모델 동시 개선 |
| **트리 계열 앙상블 한계** | 동일 피처셋 트리 모델 간 오차 상관이 높아 OOF 개선 폭이 작음(0.6462→0.6434). 진짜 다양성은 신경망 등 완전히 다른 귀납적 편향 필요 |
| **교차 타깃 피처 한계** | lag1_diff·n_pos_lag1은 기존 lag1 피처의 선형 결합 — 새 정보 없음. 이진 데이터 14개 샘플로 Pearson correlation 추정이 불안정(NaN 11~42%). 기존 label_features.py가 이미 모든 타깃의 lag/roll을 제공하므로 교차 타깃 상관은 이미 내재됨 |
| **MLP 타깃별 강·약점** | Q1~Q3(주관적 설문): subj_mean 없이 MLP 약함. S1~S4(객관적 센서): MLP가 트리보다 크게 앞섬(S2 -0.032, S3 -0.028). smooth decision boundary가 센서 데이터의 연속 패턴 포착에 유리 |
| **MLP 다양성 설계** | subj_mean 전체 제거가 핵심. 트리와 동일 피처를 쓰면 오차 상관이 높아져 앙상블 효과 감소. 의도적 피처 차별화로 오차 독립성 확보 → 3모델 앙상블에서 Jensen 부등식 효과 극대화 |
| **Multi-task 학습 효과** | 7개 타깃 동시 학습으로 360행을 7배 활용하는 효과. 공유 백본이 "모든 타깃을 동시에 설명하는 표현" 강제 → 자연스러운 정규화. n_epochs=50 최적(Optuna)은 과적합 방지 효과 |
| **v3 피처의 모델별 친화도 차이** | parquet v3 심층 피처는 ET에서 7/7 개선(최대 -0.011), HGB에서 5/7 악화. ET의 무작위 피처 샘플링이 새 피처 중 유용한 것만 선택적 활용. HGB는 gradient boosting 특성상 새 피처를 노이즈로 취급 |
| **파라미터 재튜닝 vs 피처 추가 상호작용** | ET v3 파라미터(max_feat 0.05~0.5 세밀화)는 7/7 개선을 달성하나, HGB v3 재튜닝은 오히려 역효과. 피처 구성이 변해도 HGB의 최적 파라미터 방향이 안정적이지 않음. 모델별로 피처 추가 영향을 독립 검증해야 함 |
| **앙상블 시 약한 고리 효과** | v4 앙상블에서 HGB v3(악화) + ET v3(개선) + MLP = 전체 악화. 한 모델의 성능 저하가 앙상블 전체를 끌어내림 → 구성 모델 각각의 성능 개선 확인 후 앙상블해야 함 |
| **피처 천장 진단** | OOF-Public 갭이 ET 0.041 / HGB+ET 0.033 / MLP+HGB+ET 0.032로 모델을 바꿔도 일정. 모델 개선의 한계에 도달 — 미사용 데이터 소스(GPS, wLight, BLE, WiFi)를 통한 피처 추가가 다음 성능 돌파의 핵심 |
| **OOF 가중치 최적화의 함정** | per-target Nelder-Mead 가중치 최적화: OOF 0.6329(역대 최고)이나 Public 0.6132(역대 최저). S2·S3에서 MLP 100% 극단 가중치가 소규모 OOF 노이즈에 과적합. OOF 최적화가 Public으로 역전되는 현상 확인 — 향후 OOF 기반 메타 학습 시 강한 정규화 필수 |
| **앵커 구조 vs 2단계 스태킹 차이** | 2단계 스태킹(실패)은 Stage2가 Stage1 OOF만을 입력으로 사용 → 원본 피처 정보 손실 + 과적합. 앵커 구조는 OOF를 원본 피처에 **추가**해 Stage2 입력을 풍부하게 함 → OOF가 cross-target 상관과 subject-residual을 담은 7차원 앵커로 작동 |
| **앵커 구조 소표본 한계** | v4 피처 자체는 Stage1 OOF 0.6457 (ET v2 0.6462 대비 개선). 그러나 Stage2에서 앵커 7개 추가 시 220피처 → 450행에서 과적합 → Stage2 OOF 0.6490 (역효과). DACON 0.5917 우승 코드는 371피처지만 대상자 분포가 더 균질해 유효. 소표본 LOSO에서 앵커 피처는 리스크. |
| **피처 추가 vs 피처 제거 역전** | LOSO 10명 환경에서 센서 피처 추가(wLight, mBle)는 OOF 개선 없음. 반면 Feature Importance 기반 slim(하위 피처 제거)은 OOF 0.6465 → 0.6406(+0.006) 개선. 추가보다 제거가 cross-subject 일반화에 효과적. |
| **Feature Importance Slim sweet spot** | slim 비율 낮출수록 OOF 단조 개선(75%=0.6398). 하지만 Public에서 slim85(0.6055) > slim80(0.6044) — 피처 과도 제거 시 일반화 손실. 85%가 OOF-Public 균형 최적점. |
| **GRU/LSTM 부적합 판단** | 310개 훈련 윈도우는 GRU 파라미터(hidden×n_layers×7 타깃) 대비 데이터 부족. lag/roll 피처가 이미 시계열 의존성 포착. 짧은 시퀀스(33~57일)와 LOSO 콜드스타트 문제로 시계열 모델은 이 문제에 맞지 않음. |
| **윈도우 페어 트렌드 피처** | roll3-roll7(단기 모멘텀), lag1-roll7(오늘 vs 주간 평균), roll7-roll28(주간 vs 월간 추세)을 개인별 시계열 방향 포착에 활용. NaN 안전 처리: 두 roll 중 하나라도 NaN이면 NaN으로 처리 |
| **mACStatus 수면 프록시** | 스마트폰 충전 여부(0/1)가 수면 스케줄 규칙성의 간접 지표. 수면 중(00-06h) 충전 비율 44%, 취침 전(22-24h) 27%, 기상 후(06-09h) 저충전 → 규칙적 충전 패턴은 규칙적 수면 스케줄과 상관 |
| **수면 확장 구간 wHr (00-09h)** | 현재 수면 구간(00-06h)을 확장해 기상 직전까지 포함. 코드셰어 0.6003 접근법이 사용한 수면 HR 스파이크(HR > mean+1std) 비율 피처 도입. 수면 중 일시적 심박 상승은 각성 이벤트 신호 |

---

## 향후 개선 방향

> 현재 최고 공개점수: **0.6055** (ET GPS Slim 85%) / OOF 최고: **0.6329** (MLP+HGB+ET 가중치, Public 역전) / OOF-Public 선형 최고: **0.6398** (ET GPS Slim 75%) / 리더보드 1위: 0.56119 / gap: ~0.044

### 성능 병목 분석

#### OOF-Public 갭 고착 (~0.038 일정)

| 모델 | OOF LL | Public Score | 갭 |
|------|:------:|:------------:|:--:|
| ET 단독 | 0.6469 | 0.6061 | **0.041** |
| HGB+ET | 0.6434 | 0.6103 | 0.033 |
| MLP+HGB+ET | 0.6383 | 0.6062 | 0.032 |
| ET GPS Slim 85% | 0.6406 | 0.6055 | 0.035 |

모델을 바꿔도 OOF-Public 갭이 일정 → **모델 천장이 아닌 피처 천장(feature ceiling)** 에 도달한 상태. 더 좋은 모델보다 더 좋은 정보(피처)가 성능을 결정함.

#### Q1 고착 문제 (모든 모델 공통)

| 모델 | Q1 LL | 비교 |
|------|:-----:|------|
| ET 단독 | 0.6986 | — |
| HGB+ET | 0.6974 | 거의 동일 |
| MLP 단독 | 0.7145 | 오히려 악화 |
| MLP+HGB+ET | 0.6994 | 거의 동일 |

Q1("수면의 질 — 기상 직후")은 모든 모델에서 LL ≈ 0.699 고착 → **현재 피처셋으로 Q1을 설명할 수 없음**. 수면 중 생리 신호(야간 심박수 변화, 뒤척임 등)가 없는 한 Q1 개선은 어려움.

### 완료된 시도

| 순위 | 방법 | 상태 |
|------|------|------|
| 1 | ~~CatBoost 앙상블~~ | ✅ 0.6170 |
| 2 | ~~v2 피처 확장 (roll21/28+rollstd)~~ | ✅ 0.6127 |
| 3 | ~~ExtraTrees 단독~~ | ✅ **0.6088 (현재 최고 공개점수)** |
| 4 | ~~ET Optuna 100 trials~~ | ✅ 완료 |
| 5 | ~~subj_mean NaN 수정 → 3모델 앙상블~~ | ✅ 완료 — LGBM ET 대비 열세 |
| 6 | ~~HistGradientBoosting 단독/앙상블~~ | ✅ 완료 — OOF 0.6466 (ET 동률), HGB+ET 0.6434 |
| 7 | ~~교차 타깃 lag 피처~~ | ✅ 완료 — 역효과 (OOF 0.6453) |
| 8 | ~~Multi-task MLP~~ | ✅ 완료 — OOF 0.6383 (3모델 앙상블) |
| 9 | ~~parquet 피처 심층 재설계 (v3)~~ | ✅ 완료 — ET 7/7 개선, 전체 앙상블 개선 미흡 (HGB 비친화) |
| 10 | ~~Optuna v3 재튜닝~~ | ✅ 완료 — ET v3 파라미터 유효, HGB v3 역효과 확인 |
| 11 | ~~MLP+HGB+ET v4 앙상블~~ | ✅ 완료 — OOF 0.6395 (v2 0.6383 대비 소폭 악화) |
| 12 | ~~ET v3 단독 앙상블~~ | ✅ 완료 — OOF 0.6438, Public 0.6094 (ET v2 Public 대비 악화) |
| 13 | ~~ET 극한 최적화 (300 trials)~~ | ✅ 완료 — OOF 0.6466 (악화, Q3 탐색 상한 제약) |
| 14 | ~~준지도 학습 (센서 유사도)~~ | ✅ 완료 — 효과 없음 (train/test 동일 10명) |
| 15 | ~~clip [0.05, 0.95] + 30 seeds~~ | ✅ 완료 — Public 0.6068 (ET v2 대비 악화) |
| 16 | ~~per-target 최적 가중치 앙상블~~ | ✅ 완료 — OOF 0.6329(최고)이나 Public 0.6132 (OOF 과적합) |
| 17 | ~~ET Optuna v4 재탐색 (max_feat 0.1~0.3, 200 trials)~~ | ✅ 완료 — OOF ~0.6462, **Public 0.6051** (ET v2 0.6061 대비 개선) |
| 18 | ~~GPS 피처 추가 (mGps 19개)~~ | ✅ 완료 — OOF 0.6465, **Public 0.6044 (현재 최고)** (OOF 동률이나 Public 개선) |
| 19 | ~~윈도우 페어 트렌드 피처 (trend_short/l1r7/long)~~ | ✅ 완료 — label_features.py에 추가 (21개 피처: 7타깃 × 3방향) |
| 20 | ~~수면 확장 wHr (00-09h) + mACStatus~~ | ✅ 완료 — parquet_features_v4.py (123컬럼: v3 117 + 신규 8개) |
| 21 | ~~앵커 ET 앙상블 (Stage1 OOF → Stage2 피처)~~ | ✅ 완료 — Stage1 OOF 0.6457 (개선), Stage2 OOF 0.6490 (역효과 — 220피처 과적합) |
| 22 | ~~ET v4 앙상블 (v4 피처 + anchor_stage1 params, 10 seeds)~~ | ✅ 완료 — OOF 0.6479 (v2 대비 +0.0017 악화. Q1/Q2/S1 개선, Q3/S2/S3 악화) |
| 23 | ~~wLight 피처 추가 (16개)~~ | ✅ 완료 — OOF 0.6461 (GPS 동률, 효과 없음) |
| 24 | ~~mBle 피처 추가 (25개)~~ | ✅ 완료 — OOF 0.6461 (GPS 동률, 효과 없음) |
| 25 | ~~ET GPS Feature Importance Slim (90/85/80/75%)~~ | ✅ 완료 — slim85 OOF 0.6406, **Public 0.6055 (현재 최고)**. slim75 OOF 0.6398이나 Public 악화 예상 |
| 26 | ~~GRU 시계열 모델 (window=14, 10 seeds)~~ | ✅ 완료 — OOF ~0.655 (훈련 윈도우 310개 부족, 부적합) |
| 27 | CatBoost GPS Slim 85% | 실행 중 |

### 남은 개선 방향 (우선순위 순)

> 현재 피처 천장(feature ceiling) 진단: OOF-Public 갭이 모델을 바꿔도 ~0.038 고착 → 더 좋은 모델보다 **더 좋은 정보(피처)** 가 성능을 결정.

| 순위 | 방법 | 기대 근거 | 리스크 |
|------|------|-----------|--------|
| ~~1순위~~ | ~~GPS 피처 추가 (mGps.parquet)~~ | ~~완료 — OOF 0.6465 (v2 동률). Q1/S1 개선, Q3/Q2 악화~~ | — |
| ~~2순위~~ | ~~앵커 구조 (Stage1 OOF를 피처로)~~ | ~~실행 중 — anchor_et_ensemble.py (parquet v4 + trend + OOF 앵커)~~ | — |
| ~~3순위~~ | ~~윈도우 페어 트렌드 + mACStatus + 수면확장 wHr~~ | ~~완료 — label_features.py trend 21개, parquet_features_v4.py 8개 신규~~ | — |
| **4순위** | **wLight + mBle/mWifi 피처 추가** | wLight: 손목 조도로 수면 중 광 노출(mLight와 측정 위치 다름). mBle/mWifi: 장소 다양성 entropy, 하루 접속 기기 수 | 낮음 |
| **5순위** | **요일 효과 피처** | 현재 lag/roll만 있고 dayofweek 기반 피처 없음. 주말/주중 패턴 차이가 수면·심리에 큰 영향. 개인별 "평소 요일 대비 편차" | 낮음 |
| **6순위** | **Pseudo-labeling** | 예측 확률 > 0.85 또는 < 0.15인 테스트 샘플을 학습 데이터에 추가. 유효 범위: train/test가 동일 10명이므로 subject 정보 활용 가능 | 중간 |
| **7순위** | **LSTM 시계열 모델** | 현재 모든 모델이 날짜 순서 무시. LSTM은 주간 리듬·피로 누적 등 시계열 의존성 명시 포착 | 높음 (LOSO 콜드스타트, 시퀀스 33~57일로 짧음) |

#### 1순위: GPS 피처 추가

`mGps.parquet`는 GPS 좌표·속도·고도를 포함. 현재 `wPedo`가 걸음수·칼로리를 집계하나, GPS는 **공간적 이동 패턴**이라는 별개 정보를 제공.

| 예정 피처 | 포착 신호 |
|-----------|-----------|
| `gps_dist_total` — 하루 총 이동 거리 | 신체 활동 강도 (wPedo 보완) |
| `gps_n_places` — 방문 장소 수 (반경 100m 클러스터) | 생활 다양성·사회적 활동 |
| `gps_home_duration` — 추정 자택 체류 시간 | 은둔 성향·피로 지표 |
| `gps_radius_gyration` — 이동 범위 (회전 반경) | 생활권 규모 |
| `gps_presleep_dist` — 취침 전 이동 거리 | 취침 전 활동 강도 |
| `gps_place_entropy` — 장소 방문 분포 entropy | 생활 규칙성 |

#### 2순위: 스태킹 앙상블

현재 단순 1/3 동일 가중치 앙상블 → OOF 예측을 메타 피처로 Level-2 학습.

```
Level 1: ET, HGB, MLP 각각 OOF 예측 수집 (LOSO)
Level 2: 타깃별 LightGBM 메타 학습기 (입력: 3모델 OOF × 7타깃 = 21열)
         → 메타 모델이 "어느 모델이 어느 타깃에 더 신뢰할 만한가"를 학습
```

주의: 소표본(450행) 메타 과적합 방지를 위해 Ridge(alpha=1.0, positive=True) 우선 시도.

#### 3순위: wLight + mBle/mWifi 피처

| 센서 | 예정 피처 | 근거 |
|------|-----------|------|
| `wLight` | `wlight_sleep_mean`, `wlight_presleep_trend` | mLight(폰)와 달리 손목 위 조도 — 수면 자세·이불 차단 반영 가능 |
| `mBle` | `ble_n_devices_day`, `ble_n_devices_evening` | 하루 연결 기기 수로 사회적 활동 규모 근사 |
| `mWifi` | `wifi_n_aps_day`, `wifi_place_entropy` | 접속 AP 수·분포로 위치 다양성 보완 |

#### 4순위: 요일 효과 피처

```python
# dayofweek 기반 개인화 피처 예시
feat["dayofweek"] = target_sleep_date.dayofweek  # 0=월 ~ 6=일
feat["is_weekend"] = int(target_sleep_date.dayofweek >= 5)

# subject별 요일 평균 대비 편차 (개인화)
subj_dow_mean = train[train.subject_id == sid].groupby("dayofweek")[t].mean()
feat[f"dow_dev_{t}"] = current_val - subj_dow_mean.get(dayofweek, np.nan)
```

#### Q1 고착 문제

Q1("수면의 질 — 기상 직후")은 모든 모델에서 LL ≈ 0.699 고착. **현재 피처셋으로 Q1 설명 한계.**

| 원인 | 대응 방향 |
|------|-----------|
| 수면의 질은 수면 중 생리신호 의존 (뒤척임·REM 비율) | GPS 집 체류 시간, wLight 수면 중 조도가 간접 신호 |
| 개인 편차가 커서 unseen subject 일반화 어려움 | 스태킹 메타 모델이 subject별 예측 편향 보정 가능 |
| subj_mean_Q1 항상 NaN (leakage 방지) | 대체 개인화 피처: 센서 z-score 기준선, 요일 효과 |

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
| ~~Subject 트렌드 피처 (v3)~~ | ❌ 역효과 | 7/14일 기울기·편차가 노이즈 — v2 피처 유지 |
| ~~OOF 가중 앙상블~~ | ❌ OOF 비정상(0.75) | val subject subj_mean NaN으로 LGBM/CatBoost 붕괴 |
| ~~LGBM+CatBoost+ET 3모델~~ | ❌ OOF 0.7074 | NaN 미처리 LGBM OOF 1.009로 앙상블 전체 저하 |

---

## 확률 clip 전략

모든 제출 파일에 `clip(0.1, 0.9)` 적용 중.

### clip을 사용하는 이유

Log-loss는 `log(0) = -∞`이므로 극단값(0 또는 1)이 실제 정답과 다를 경우 loss가 폭발함.

| 예측값 p | 실제 y=1일 때 loss | 비고 |
|:--------:|:-----------------:|------|
| 0.9 | 0.105 | 자신 있게 맞춤 |
| 0.1 | 2.303 | 자신 있게 틀림 |
| 0.01 | 4.605 | 매우 자신 있게 틀림 |

**비대칭성**: 극단적으로 틀렸을 때의 페널티(4.6)가 극단적으로 맞았을 때의 보상(0.01)보다 훨씬 큼. clip은 이 위험에 대한 보험으로, 기댓값 기준 손실을 줄임.

```
p=0.02 → clip → 0.1 : loss 3.91 → 2.30  (1.61 절약)
p=0.98 → clip → 0.9 : loss 0.02 → 0.105 (0.085 손해)
```

### clip 범위 비교

| 범위 | 특성 |
|------|------|
| [0.05, 0.95] | 극단값만 보호, 자신감 허용 |
| **[0.1, 0.9]** | **현재 사용 — 균형적 보호** |
| [0.2, 0.8] | 강한 보호, 정보 손실 큼 |

LOSO처럼 완전히 새로운 subject를 예측하는 상황에서 모델이 과도한 자신감을 가질 위험이 높아 [0.1, 0.9]가 합리적. 다만 최적 범위는 실험적으로 결정 가능 (향후 개선 방향 7번).

---

## 환경 설정

```bash
uv pip install pandas pyarrow lightgbm scikit-learn optuna shap xgboost catboost
```

### 실행 순서

```bash
# 현재 최고 공개점수 (ET GPS Slim 85%, Public 0.6055)
uv run scripts/extratrees_gps_slim85_ensemble.py

# GPS 피처 기반
uv run scripts/extratrees_gps_ensemble.py        # ET + GPS (Public 0.6044)
uv run scripts/extratrees_gps_slim80_ensemble.py  # ET GPS Slim 80% (OOF 최고)

# CatBoost (실행 중)
uv run scripts/catboost_gps_slim85_ensemble.py
```
