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
│   ├── features_all.csv                  # 통합 피처 데이터셋 (700행 × 107컬럼)
│   └── ch2025_data_items/                # 라이프로그 parquet (12종)
├── scripts/
│   ├── baseline_subject_mean.py          # 베이스라인: subject 평균 분류
│   ├── lgbm_csv_only.py                  # LightGBM (CSV 피처만, StratifiedKFold)
│   ├── logistic_csv_only.py              # Logistic Regression (피처 중요도 분석)
│   ├── lgbm_with_parquet.py              # LightGBM + parquet 피처 (StratifiedKFold)
│   ├── lgbm_csv_group.py                 # LightGBM (CSV만, GroupKFold LOSO)
│   ├── lgbm_parquet_group.py             # LightGBM + parquet v1 (GroupKFold LOSO)
│   ├── parquet_features.py               # parquet 집계 v1 (일별 + 야간 2구간)
│   ├── parquet_features_v2.py            # parquet 집계 v2 (시간대 5구간 + mAmbience)
│   ├── label_features.py                 # lag1/2 + roll3/7 피처 빌더
│   ├── build_feature_csv.py              # 전체 피처 통합 CSV 생성
│   ├── lgbm_final.py                     # LightGBM (모든 피처 + GroupKFold LOSO)
│   ├── lgbm_zscore.py                    # lgbm_final + subject z-score 정규화
│   └── lgbm_multiseed.py                 # lgbm_zscore + 멀티 시드(10개) 평균
├── submission/
│   ├── baseline_subject_mean.csv
│   ├── lgbm_csv_only.csv
│   ├── lgbm_parquet_group_prob.csv
│   ├── lgbm_zscore_prob.csv
│   ├── lgbm_multiseed_prob.csv           # 현재 최고 제출 파일
│   └── submission_result.md
├── data_summary.md
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
| 스마트폰 | mGps | GPS 좌표/속도/고도 | ❌ wPedo와 중복 |
| 스마트폰 | mBle | 주변 BLE 기기 RSSI | ❌ 밀도 낮음 |
| 스마트폰 | mUsageStats | 앱별 사용 시간 | ❌ 구조 복잡, 보류 |
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
| lgbm_roll14lag7_prob | 0.6890 | |
| lgbm_lag_full_prob | 0.6822 | 동일 타깃 lag 허용 |
| **lgbm_optuna_prob** | **0.6178** | **현재 최고** |

> lgbm_csv_only(12.07) → lgbm_zscore_prob(0.695): hard→확률 전환으로 **94% 개선**
> lgbm_zscore_prob(0.695) → lgbm_multiseed_prob(0.692): 멀티 시드로 **0.003 추가 개선**

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

| 피처 | 내용 | 주요 효과 |
|------|------|-----------|
| parquet v1 | wPedo/mActivity/mScreen/wHr/mLight 일별 집계 | S3 +0.122 |
| parquet v2 | 시간대 5구간 분리 + mAmbience 추가 | Q3/S1 소폭 개선 |
| lag1/2 | 전날·2일 전 레이블 (날짜 간격 ≤2일 조건) | S3 개선 |
| roll3/7 | 최근 3/7개 레이블 이동 평균 (30일 이내) | 안정적 개인 기준선 |

**피처 수**: base 12 + parquet 68 + lag/roll 28 = 총 108개

### 4단계: Subject Z-score 정규화

각 센서 피처를 subject별 평균/표준편차로 정규화.

```
기존: hr_mean = 80.0 (id03과 id08 동일 취급)
개선: hr_mean_zscore = (80 - subject_mean) / subject_std
     → id03: +2.0 (비정상적으로 높은 날)
     → id08: -1.0 (평소보다 낮은 날)
```

| 구현 방식 | 내용 |
|-----------|------|
| train fold | fold 내 각 subject 통계로 정규화 |
| val fold | held-out subject 자신의 데이터로 정규화 |
| test | 전체 train 통계로 정규화 |

**효과**: S2 F1 0.579 → 0.611 (+0.032), 평균 LogLoss 1.181 → 1.151

### 5단계: 멀티 시드 평균

10개 시드(+ subsample=0.8, colsample_bytree=0.8) 학습 후 확률 평균.

```python
SEEDS = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]
```

**효과**: 모든 타깃 F1 개선, 평균 LogLoss 1.151 → 1.086 (5.6% 개선)

---

## 모델별 GroupKFold CV 결과

### F1 비교

| 타깃 | csv_group | parquet_group | lgbm_multiseed | lgbm_lag_full | **lgbm_optuna** |
|------|:---:|:---:|:---:|:---:|:---:|
| Q1 | 0.629 | 0.647 | 0.649 | 0.653 | **0.642** |
| Q2 | 0.689 | 0.688 | 0.666 | 0.701 | **0.713** |
| Q3 | 0.727 | 0.740 | **0.726** | 0.712 | 0.727 |
| S1 | 0.796 | 0.811 | 0.805 | 0.806 | **0.811** |
| S2 | 0.745 | 0.688 | 0.636 | 0.701 | **0.783** |
| S3 | 0.372 | 0.494 | 0.543 | 0.603 | **0.733** |
| S4 | 0.675 | 0.675 | 0.624 | 0.637 | **0.693** |

### 평균 LogLoss 비교

| 모델 | 평균 LogLoss | Public Score |
|------|:---:|:---:|
| lgbm_parquet_group | 1.592 | 미제출 |
| lgbm_final | 1.181 | 미제출 |
| lgbm_zscore | 1.151 | 0.6950 |
| lgbm_multiseed | 1.086 | 0.6924 |
| lgbm_roll14lag7 | 1.064 | 0.6890 |
| lgbm_lag_full | 1.042 | 0.6822 |
| **lgbm_optuna** | **0.671** | **0.6178** |
| lgbm_stacking | 0.915 | 0.7163 (❌ 역효과) |

---

## 피처 엔지니어링 상세

### 통합 피처 데이터셋: `data/features_all.csv`
700행 × 107컬럼 (train 450 + test 250)

#### 시간대 구분 (parquet v2)

| 시간대 | 범위 | 의미 |
|--------|------|------|
| morning | 06~12시 | 오전 활동 |
| afternoon | 12~18시 | 오후 활동 |
| evening | 18~22시 | 저녁 활동 |
| presleep | 22~24시 | 취침 전 |
| sleep | 00~06시 | 수면 중 |

#### lag/roll 피처 조건

| 피처 | 조건 |
|------|------|
| lag1_{t} | 날짜 간격 ≤ 2일, 아니면 NaN |
| lag2_{t} | lag1 유효 + 연속 날짜, 아니면 NaN |
| roll3_{t} | 30일 이내 직전 3개 평균 |
| roll7_{t} | 30일 이내 직전 7개 평균 |

---

## 주요 발견 및 교훈

| 발견 | 내용 |
|------|------|
| **확률 제출 효과 압도적** | hard 0/1(12.07) → 확률(0.695): 94% 개선. log-loss metric에선 필수 |
| **LOSO lag 처리** | val fold에서 lag를 train_fold로 계산 시 val subject 전부 NaN → 자신의 데이터로 계산해야 함 |
| **S3 취약 → parquet+lag로 개선** | CSV only F1=0.372 → 멀티시드 0.543 |
| **S2 역설** | 피처 추가 시 F1 하락했다가 z-score 적용으로 회복 |
| **멀티 시드 효과** | subsample 추가로 시드 간 다양성 확보 → 모든 타깃 안정적 개선 |
| **subj_mean leakage** | 예측 타깃 t의 subj_mean_t 사용 시 F1=1.0 → 반드시 제거 |
| **스태킹 역효과** | OOF log-loss 0.915로 크게 개선됐으나 Public Score 0.6924→0.7163으로 악화. OOF 과적합으로 추정: 10명 소규모 데이터에서 Stage 2가 Stage 1 OOF 패턴을 외워버림 |

---

## 향후 개선 방향

> 현재 최고 점수: 0.6178 / 리더보드 1위: 0.56119 / 상위권 목표: ~0.4

### 핵심 개선 방향 (우선순위순)

| 순위 | 방법 | 예상 점수 개선 | 구현 난이도 |
|------|------|:---:|:---:|
| ~~1~~ | ~~2단계 스태킹~~ | ~~0.05~0.15~~ | ~~중간~~ |
| 1 | **Subject 가중치** | 0.03~0.08 | 낮음 |
| 2 | **Calibration (Platt Scaling)** | 0.02~0.05 | 낮음 |
| 3 | **roll14 + lag7** | 0.01~0.03 | 낮음 |

> ~~2단계 스태킹~~: OOF log-loss는 개선됐으나 실제 Public Score 악화(0.6924→0.7163).
> 10명 소규모 데이터에서 Stage 2가 Stage 1 OOF를 과적합하는 것으로 판단. **제외**.

#### 방법별 설명

**1. Subject 가중치**
예측 대상 subject의 학습 데이터에 더 높은 가중치를 부여해 개인 패턴 강화.
```
예: id03 test 예측 시 → id03 train 행 weight=3.0, 나머지 weight=1.0
```

**2. Calibration (Platt Scaling)**
현재 고정된 clip(0.1~0.9) 대신 OOF 예측으로 타깃별 최적 보정 파라미터 학습.
log-loss는 확률 보정 품질에 민감하므로 효과 기대.

**3. roll14 + lag7**
현재 roll7까지만 사용. 2주 이동 평균(roll14)과 7일 전 레이블(lag7)로 주간 패턴 포착.

### 추가 검토 항목

| 방법 | 기대 효과 |
|------|-----------|
| XGBoost / CatBoost 앙상블 | LightGBM과 다양성 확보, 분산 추가 감소 |
| 하이퍼파라미터 튜닝 (Optuna) | 타깃별 최적 파라미터 탐색 |
| mUsageStats 피처 추가 | 취침 전 앱 사용 패턴 → S3 개선 가능 |

---

## 환경 설정

```bash
uv venv
uv pip install pandas pyarrow lightgbm scikit-learn
```

### 실행 순서

```bash
# 피처 데이터셋 생성
uv run python scripts/build_feature_csv.py

# 모델 학습 (권장 순서)
uv run python scripts/lgbm_multiseed.py     # 현재 최고 성능 (권장)
uv run python scripts/lgbm_zscore.py        # 멀티시드 없는 버전
uv run python scripts/lgbm_final.py         # z-score 없는 버전
```
