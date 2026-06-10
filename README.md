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
│   ├── catboost_optuna.py                # CatBoost 단독 Optuna 튜닝
│   ├── extratrees_ensemble.py            # ExtraTrees 단독 앙상블
│   ├── extratrees_v3_ensemble.py         # ExtraTrees v3 피처 단독 앙상블 (z-score 없음)
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
│   ├── extratrees_gps_slim75_ensemble.py # ET GPS Slim 75% (OOF 0.6398, Public 0.6056 slim80 대비 악화)
│   ├── extratrees_v3gps_slim_ensemble.py # v3 피처+GPS Slim 90% (역효과)
│   ├── extratrees_gps_pertarget_slim_ensemble.py # 타깃별 개별 slim 85%
│   ├── gru_ensemble.py                   # GRU 시계열 앙상블 (10 seeds, window=14) — 데이터 부족으로 부적합
│   ├── catboost_gps_slim85_ensemble.py   # CatBoost GPS Slim 85% (OOF 0.6446, ET 대비 열세)
│   ├── mwifi_features.py                 # mWifi.parquet → 21개 WiFi 피처 (wifi_entropy/home_ratio 등)
│   ├── parquet_features_v5.py            # parquet 집계 v5 — v2 + wLight + mBle + mWifi (parquet_features_v2 확장)
│   ├── dow_deviation_features.py         # 요일 효과 편차 피처 (개인별 요일 평균 대비 편차, 23개 센서 기준)
│   ├── extratrees_v5_gps_slim85_ensemble.py # ET v5 피처+GPS+DOW편차 Slim 85% (OOF 0.6410, Public 0.6046)
│   ├── mlp_gps_slim85_ensemble.py        # Multi-task MLP GPS Slim 85% (OOF avg_ll 0.6396, 10 seeds)
│   ├── et_cb_hgb_mlp_ensemble.py         # 4모델 균등 앙상블 (ET+CatBoost+HGB+MLP, Public 0.6070 역효과)
│   ├── et_gps_slim80_calibrated.py       # ET GPS Slim 80% + LOSO OOF 기반 피험자별 logit bias 보정 (REG=0.5)
│   ├── et_gps_slim80_ws_calibrated.py    # ET GPS Slim 80% + Within-subject 시간순 hold-out 기반 logit bias 보정
│   ├── et_gps_slim80_reg_tuning.py       # LOSO logit bias 보정 REG 튜닝 (0.1/0.25/1.0 비교, Phase 0~2 1회 실행)
│   ├── et_gps_slim80_transductive.py     # Transductive Z-score ET GPS Slim 80% (train+test 센서 통계, Public 0.6020)
│   ├── sensor_lag_features.py            # 센서 lag1/roll7 피처 빌더 (전날 센서값, 7일 이동평균)
│   ├── et_gps_slim80_trans_sensorlag.py  # Transductive + 센서 lag 피처 (OOF 0.6483, 미제출)
│   ├── et_ws_cv_transductive.py          # ET Within-Subject CV + Transductive (Public 0.6268, WS val 신뢰성 확인)
│   ├── mlp_loso_transductive.py          # Multi-task MLP LOSO + Transductive (OOF 1.1551, MLP×LOSO 부적합 확인)
│   ├── et_gps_slim80_trans_varreg.py     # Transductive Z-score + 피험자별 가변 REG 보정 (reg = BASE_REG * BASE_N / n_dates)
│   ├── et_gps_slim80_alldata_ws.py       # 전체 데이터 학습 + WS OOF 편향 보정 (100% warm-start, 구조 불일치 해소)
│   ├── et_gps_slim80_personal_blend.py  # 전체 모델 + 피험자별 개인 모델 블렌딩 (Global+Personal ET, alpha WS OOF 최적화)
│   ├── et_gps_slim80_pers_pertarget.py # 개인 모델 블렌딩 + 타깃별 alpha 독립 최적화 (per-target: Q2=0.0, S1=0.5054, S2=0.4677)
│   ├── et_gps_slim80_pers_grid.py     # 개인 모델 파라미터 grid search (depth×max_feat×min_leaf 18조합, Global WS OOF 1회 계산)
│   ├── lgb_gps_slim80_personal_blend.py # LGB GPS Slim 80% + Personal Blend (WS OOF Optuna 50 trials, Public 0.6079)
│   ├── cb_gps_slim80_personal_blend.py  # CB GPS Slim 80% + Personal Blend (depth 4-6, iterations 150-600, WS OOF 0.6485)
│   ├── xgb_gps_slim80_personal_blend.py # XGB GPS Slim 80% + Personal Blend (WS OOF 0.6120, 미제출)
│   ├── ensemble_et_lgb_cb.py           # ET+LGB+CB 3-way 앙상블 (1:1:1 및 2:1:1)
│   ├── ensemble_et_xgb.py             # ET+XGB 앙상블 (1:1, 1:2, 3-way, 4-way)
│   ├── whr_variability_features.py    # wHr 수면 심박 변동성 피처 (IQR/p10/p90/within_std/night_dip 등 7개)
│   ├── et_gps_whrvar_slim80_personal_blend.py # ET GPS + wHr변동성 Slim 80% Personal Blend (WS OOF 0.6486, wHr 0/7 선택)
│   ├── et_gps_whrvar_slim80_hgbpers_blend.py  # ET Global + HGB 개인 모델 블렌딩 (WS OOF 0.6508, alpha=0.0977)
│   ├── rolling_features.py            # 피험자별 일별 피처 이동평균/delta 빌더 (ma3/ma7/delta3, 19개 기준 -> 57개)
│   ├── et_gps_rolling_slim80_personal_blend.py # ET GPS + Rolling Slim 80% Personal Blend (WS OOF 0.6443, 22/57 선택)
│   ├── hgb_gps_rolling_slim80_personal_blend.py # HGB GPS + Rolling Slim 80% Personal Blend (WS OOF 0.6204)
│   ├── et_fwdlabel_slim80_personal_blend.py   # ET GPS + 역방향 시간 피처(fwd_lag1/roll7/roll14) + Personal Blend (WS OOF 0.6357, 21/21 fwd 선택)
│   ├── et_gps_slim80_seasonal_blend.py       # ET GPS Slim 80% + Seasonal Re-weight (Jun-Jul x2.0) + Personal Blend (WS OOF 0.6504)
│   ├── lgb_gps_slim80_seasonal_blend.py      # LGB GPS Slim 80% + Seasonal Re-weight + Personal Blend
│   ├── xgb_gps_slim80_seasonal_blend.py      # XGB GPS Slim 80% + Seasonal Re-weight + Personal Blend
│   ├── cb_gps_slim80_seasonal_blend.py       # CB GPS Slim 80% + Seasonal Re-weight + Personal Blend (WS OOF 0.6485)
│   ├── hgb_gps_slim80_seasonal_blend.py      # HGB GPS Slim 80% + Seasonal Re-weight + Personal Blend (WS OOF 0.6249)
│   ├── ensemble_seasonal_5way.py             # ET+LGB+XGB+CB+HGB 5모델 Seasonal 앙상블 (29가지 조합)
│   ├── et_gps_slim80_summer_holdout_blend.py # ET GPS Slim 80% + Summer Holdout Val (Jun-Jul=val, Oct-Nov=train, id04/id05 WS fallback)
│   ├── lgb_gps_slim80_summer_holdout_blend.py # LGB GPS Slim 80% + Summer Holdout Val
│   ├── xgb_gps_slim80_summer_holdout_blend.py # XGB GPS Slim 80% + Summer Holdout Val
│   ├── ensemble_summer_holdout.py            # Seasonal 3-way + Summer Holdout 앙상블 (63가지 조합)
│   ├── et_gps_slim80_rolling_whr_seasonal_blend.py  # ET GPS + Rolling(ma3/ma7/delta3) + WHR변동성 + Seasonal (WS OOF 0.6427, 136피처)
│   ├── et_gps_slim80_pertarget_seasonal_blend.py    # ET GPS Slim 80% 타깃별 독립 피처 선택 + Seasonal (WS OOF 0.6398)
│   └── et_gps_slim80_density_blend.py        # ET GPS Slim 80% + Density Ratio Weighting (P(test|x)/P(train|x), WS OOF 0.6470)
├── submission/
│   ├── extratrees_ensemble_prob.csv      # ET 단독 앙상블 (Public 0.6061)
│   ├── mlp_hgb_et_ensemble_prob.csv      # MLP+HGB+ET 3모델 (OOF 0.6383)
│   ├── extratrees_v3_ensemble_prob.csv   # ET v3 단독 (Public 0.6094)
│   ├── hgb_et_xt_ensemble_prob.csv       # HGB+ET+교차타깃 (OOF 0.6453, 역효과)
│   ├── hgb_v2_ensemble_prob.csv          # HGB v2 단독 (OOF 0.6466, 미제출)
│   ├── hgb_et_ensemble_prob.csv          # HGB+ET 앙상블 (OOF 0.6434, Public 0.6103)
│   ├── multitask_mlp_ensemble_prob.csv   # MLP 단독 (OOF 0.6398, 미제출)
│   ├── hgb_et_v4_ensemble_prob.csv       # HGB+ET v4 (OOF 0.6438, 미제출)
│   ├── mlp_hgb_et_v4_ensemble_prob.csv   # MLP+HGB+ET v4 (OOF 0.6395, 미제출)
│   ├── mlp_hgb_et_weighted_prob.csv      # MLP+HGB+ET 가중치 (OOF 0.6329, Public 0.6132, 악화)
│   ├── extratrees_optuna_v4_prob.csv     # ET v4 max_feat 0.1~0.3 집중 (Public 0.6051)
│   ├── extratrees_gps_prob.csv          # ET + GPS 피처 앙상블 (Public 0.6044)
│   ├── anchor_et_prob.csv               # 앵커 ET Stage2 (OOF 0.6490 — 역효과)
│   ├── extratrees_v4_ensemble_prob.csv  # ET v4 피처 + anchor_stage1 params, 10 seeds
│   ├── extratrees_wlight_prob.csv        # ET + GPS + wLight (Public 0.6053, 효과 없음)
│   ├── extratrees_gps_slim85_prob.csv   # ET GPS Slim 85% (Public 0.6055)
│   ├── extratrees_gps_slim80_prob.csv   # ET GPS Slim 80% (Public 0.6044)
│   ├── extratrees_gps_slim75_prob.csv   # ET GPS Slim 75% (OOF 0.6398)
│   ├── extratrees_gps_pertarget_slim_prob.csv # 타깃별 개별 slim
│   ├── gru_ensemble_prob.csv            # GRU 앙상블 (OOF ~0.655, 부적합)
│   ├── catboost_gps_slim85_prob.csv     # CatBoost GPS Slim 85% (OOF 0.6446, 미제출)
│   ├── mlp_gps_slim85_prob.csv          # MLP GPS Slim 85% (OOF avg_ll 0.6396, 미제출)
│   ├── et_cb_hgb_mlp_ensemble_prob.csv  # 4모델 균등 앙상블 (Public 0.6070, 역효과)
│   ├── extratrees_v5_gps_slim85_prob.csv# ET v5 피처 GPS Slim 85% (Public 0.6046)
│   ├── et_gps_slim80_calibrated_prob.csv# LOSO OOF 기반 logit bias 보정 (Public 0.6027)
│   ├── et_gps_slim80_ws_calibrated_prob.csv # WS hold-out 기반 logit bias 보정 (Public 0.6031)
│   ├── et_gps_slim80_reg025_prob.csv    # REG=0.25 보정 (Public 0.6029, REG=0.5 대비 소폭 악화)
│   ├── et_gps_slim80_reg01_prob.csv     # REG=0.10 보정 (미제출 — 과보정 우려)
│   ├── et_gps_slim80_reg10_prob.csv     # REG=1.00 보정 (미제출 — 과소보정)
│   ├── et_gps_slim80_transductive_prob.csv # Transductive Z-score (Public 0.6020)
│   ├── et_gps_slim80_trans_sensorlag_prob.csv # Transductive + 센서 lag (미제출 — OOF 0.6483 악화)
│   ├── et_ws_cv_transductive_prob.csv   # ET WS CV Transductive (Public 0.6268, WS val 신뢰성 확인)
│   ├── et_gps_slim80_trans_hard85_prob.csv # Hard threshold p>0.85→1 (Public 0.9682, 실험용)
│   ├── mlp_loso_transductive_prob.csv   # MLP LOSO Transductive (미제출 — OOF 1.1551, 랜덤보다 나쁨)
│   ├── et_gps_slim80_trans_varreg_prob.csv  # Transductive + 피험자별 가변 REG (미제출 — OOF 0.6235)
│   ├── et_gps_slim80_alldata_ws_prob.csv    # 전체 데이터 학습 + WS OOF 보정 (Public 0.6009)
│   ├── et_gps_slim80_personal_blend_prob.csv # 전체+개인 모델 블렌딩 (Public 0.5992)
│   ├── et_gps_slim80_pers_pertarget_prob.csv # per-target alpha 블렌딩 (Public 0.6027 — WS OOF 개선 but Public 역전)
│   ├── et_gps_slim80_pers_grid_best_prob.csv # 개인 모델 params grid best (depth=2,max_feat=0.5,min_leaf=3)
│   ├── lgb_gps_slim80_personal_blend_prob.csv  # LGB GPS Slim 80% + Personal Blend (Public 0.6079)
│   ├── cb_gps_slim80_personal_blend_prob.csv   # CB GPS Slim 80% + Personal Blend (WS OOF 0.6485)
│   ├── xgb_gps_slim80_personal_blend_prob.csv  # XGB GPS Slim 80% + Personal Blend (WS OOF 0.6120, 미제출)
│   ├── et_lgb_ensemble_prob.csv          # ET + LGB 앙상블 (Public 0.5982)
│   ├── et_lgb_cb_ensemble_prob.csv       # ET + LGB + CB 3-way 앙상블 (Public 0.5963)
│   ├── et_lgb_cb_w2_ensemble_prob.csv    # ET + LGB + CB 2:1:1 앙상블 (Public 0.5962, 현재 최고)
│   ├── et_xgb_ensemble_prob.csv          # ET + XGB 1:1 앙상블 (Public 0.5968)
│   ├── et_xgb_w2_ensemble_prob.csv       # ET + XGB 1:2 앙상블 (미제출)
│   ├── et_xgb_lgb_ensemble_prob.csv      # ET + XGB + LGB 3-way 앙상블 (미제출)
│   ├── et_lgb_cb_xgb_ensemble_prob.csv   # ET + LGB + CB + XGB 4-way 앙상블 (Public 0.5955, 현재 최고)
│   ├── et_gps_whrvar_slim80_personal_blend_prob.csv  # ET GPS wHr변동성 Slim 80% Personal Blend (미제출, WS OOF 0.6486)
│   ├── et_gps_rolling_slim80_personal_blend_prob.csv # ET GPS Rolling Slim 80% Personal Blend (미제출, WS OOF 0.6443)
│   ├── hgb_gps_rolling_slim80_personal_blend_prob.csv # HGB GPS Rolling Slim 80% Personal Blend (미제출, WS OOF 0.6204)
│   ├── et_gps_slim80_seasonal_blend_prob.csv  # ET GPS Slim 80% Seasonal (WS OOF 0.6504, 3-way 앙상블 일부)
│   ├── lgb_gps_slim80_seasonal_blend_prob.csv # LGB GPS Slim 80% Seasonal (3-way 앙상블 일부)
│   ├── xgb_gps_slim80_seasonal_blend_prob.csv # XGB GPS Slim 80% Seasonal (3-way 앙상블 일부)
│   ├── et_gps_slim80_summer_holdout_blend_prob.csv  # ET Summer Holdout (OOF 0.6224, Public 0.6048)
│   ├── et_gps_slim80_rolling_whr_seasonal_blend_prob.csv # Rolling+WHR Seasonal (WS OOF 0.6427, Public 0.6001)
│   ├── et_gps_slim80_pertarget_seasonal_blend_prob.csv   # Per-Target 피처 선택 Seasonal (WS OOF 0.6398, Public 0.6031)
│   ├── et_gps_slim80_density_blend_prob.csv   # Density Ratio Weighting (WS OOF 0.6470, 미제출)
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
| 스마트폰 | mBle | 주변 BLE 기기 RSSI | ✅ 완료 (mble_features.py: 25개 피처, OOF GPS 동률 → parquet_features_v5에 포함) |
| 스마트폰 | mWifi | 주변 WiFi AP RSSI | ✅ 완료 (mwifi_features.py: 21개 피처, wifi_entropy/home_ratio 등 → parquet_features_v5에 포함) |
| 스마트워치 | wLight | 워치 조도 | ✅ 완료 (wlight_features.py: 16개 피처, OOF GPS 동률 → parquet_features_v5에 포함) |

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
| extratrees_gps_slim85_prob | 0.6055 | ET GPS Slim 85% (OOF 0.6406) — slim80 대비 Public 열세 |
| **extratrees_gps_slim80_prob** | **0.6044** | **현재 최고 공개점수** — ET GPS Slim 80% (OOF 0.6403, slim85 대비 Public 개선) |
| extratrees_gps_slim75_prob | 0.6056 | ET GPS Slim 75% (OOF 0.6398, 피처 과도 제거로 slim80 대비 악화) |
| catboost_gps_slim85_prob | 미제출 | CatBoost GPS Slim 85% (OOF 0.6446, ET 0.6406 대비 열세) |
| et_cb_hgb_mlp_ensemble_prob | 0.6070 | ET+CatBoost+HGB+MLP 4모델 균등 앙상블 (역효과 — 약한 모델 3개가 ET 희석) |
| extratrees_v5_gps_slim85_prob | 0.6046 | ET v5 피처 (wLight+mBle+mWifi+DOW편차) GPS Slim 85% (OOF 0.6410, v4 대비 +0.0012) |
| et_gps_slim80_calibrated_prob | 0.6027 | LOSO OOF 기반 피험자별 logit bias 보정 (REG=0.5) |
| et_gps_slim80_ws_calibrated_prob | 0.6031 | Within-subject 시간순 hold-out 기반 보정 (LOSO 보정 대비 소폭 악화) |
| et_gps_slim80_reg025_prob | 0.6029 | REG=0.25 보정 (REG=0.5 대비 소폭 악화 — REG=0.5가 최적 확인) |
| et_gps_slim80_transductive_prob | 0.6020 | Transductive Z-score 정규화 (train+test 전체 센서 통계 활용) |
| et_gps_slim80_trans_sensorlag_prob | 미제출 | Transductive + 센서 lag/roll7 피처 (OOF 0.6483 악화 — LOSO 환경 과적합) |
| et_ws_cv_transductive_prob | 0.6268 | ET Within-Subject CV + Transductive (WS val 0.6266 ≈ Public 0.6267 — WS val 신뢰 가능 확인, 모델 자체 약함) |
| et_gps_slim80_trans_hard85_prob | 0.9682 | Hard threshold p>0.85→1, p<0.15→0 (S1~S3 고신뢰 예측 오답 확인 실험) |
| mlp_loso_transductive_prob | 미제출 | MLP LOSO + Transductive (OOF 1.1551, 랜덤 0.693보다 나쁨 — MLP×LOSO 부적합) |
| et_gps_slim80_trans_varreg_prob | 0.6019 | Transductive + 피험자별 가변 REG (reg=BASE_REG*BASE_N/n_dates) |
| et_gps_slim80_alldata_ws_prob | 0.6009 | 전체 데이터 학습 + WS OOF 편향 보정 (100% warm-start) |
| et_gps_slim80_personal_blend_prob | 0.5992 | 전체 모델(80%) + 피험자별 개인 모델(20%) 블렌딩 (alpha=0.2056, WS OOF 0.6493) |
| et_gps_slim80_pers_pertarget_prob | 0.6027 | per-target alpha (Q2=0.0, S1=0.5054, S2=0.4677) — WS OOF 0.6440 개선 but Public 역전 (단일 alpha 대비 악화) |
| **et_gps_slim80_pers_grid_best_prob** | **0.5989** | 개인 모델 params grid search 최적 — depth=2, max_feat=0.5, min_leaf=3, alpha=0.3595, WS OOF 0.6408 |
| et_gps_slim80_global_ws_optuna_prob | 0.5999 | Global ET WS OOF Optuna 50 trials (WS 맥락 재최적화) |
| lgb_gps_slim80_personal_blend_prob | 0.6079 | LGB GPS Slim 80% + Personal Blend (WS OOF Optuna 50 trials) |
| **et_lgb_ensemble_prob** | **0.5982** | ET(pers_grid_best) + LGB 앙상블 — ET+LGB 이종 모델 결합 |
| et_gps_slim80_cross_target_prob | 미제출 | Cross-Target Stacking (WS OOF 0.6427 악화 — 미제출) |
| cb_gps_slim80_personal_blend_prob | 미제출 | CB GPS Slim 80% + Personal Blend (WS OOF 0.6485, 앙상블 다양성용) |
| et_lgb_cb_ensemble_prob | 0.5963 | ET+LGB+CB 3-way 앙상블 (1:1:1) — et_lgb(0.5982) 대비 +0.0019 개선 |
| **et_lgb_cb_w2_ensemble_prob** | **0.5962** | **ET+LGB+CB 2:1:1 앙상블 (ET 비중 강화) — 현재 최고** |
| xgb_gps_slim80_personal_blend_prob | 0.6037 | XGB GPS Slim 80% + Personal Blend — WS OOF 0.6120(최고)이나 Public 심한 과적합 |
| et_xgb_ensemble_prob | 0.5968 | ET + XGB 1:1 앙상블 — et_lgb_cb(0.5963) 대비 소폭 악화 |
| et_xgb_w2_ensemble_prob | 미제출 | ET + XGB 1:2 앙상블 (XGB 비중 강화) — XGB 과적합으로 제출 불필요 |
| et_xgb_lgb_ensemble_prob | 미제출 | ET + XGB + LGB 3-way — XGB 과적합으로 제출 불필요 |
| **et_lgb_cb_xgb_ensemble_prob** | **0.5955** | **ET + LGB + CB + XGB 4-way 앙상블 — 현재 최고. XGB 단독 나빠도 다양성 기여** |
| hgb_gps_slim80_personal_blend_prob | 미제출 | HGB GPS Slim 80% + Personal Blend (WS OOF 0.6249, alpha=0.3385, S3=0.5584 강점) |
| et_lgb_cb_hgb_ensemble_prob | 미제출 | ET+LGB+CB+HGB 4-way (XGB 제외) |
| et_lgb_cb_xgb_hgb_ensemble_prob | 미제출 | ET+LGB+CB+XGB+HGB 5-way 1:1:1:1:1 |
| et_lgb_cb_xgb_hgb_w2_ensemble_prob | 미제출 | ET+LGB+CB+XGB+HGB 5-way ET 비중 2배 |
| et_gps_whrvar_slim80_personal_blend_prob | 미제출 | ET GPS + wHr변동성 Slim 80% Personal Blend (WS OOF 0.6486 — wHr 0/7 선택, 방향 폐기) |
| et_gps_rolling_slim80_personal_blend_prob | 미제출 | ET GPS + Rolling Slim 80% Personal Blend (WS OOF 0.6443 — 22/57 rolling 선택, 방향 폐기) |
| hgb_gps_rolling_slim80_personal_blend_prob | 미제출 | HGB GPS + Rolling Slim 80% Personal Blend (WS OOF 0.6204 신기록 — Public 역전 확인, 방향 폐기) |
| et_fwdlabel_slim80_personal_blend_prob | 미제출 | ET GPS + 역방향 피처(fwd_lag1/roll7/roll14 21개) + Personal Blend (WS OOF 0.6357, alpha=0.3637 — 21/21 fwd 피처 선택, S1=0.5324 대폭 개선) |
| et_gps_slim80_seasonal_blend_prob | - | ET GPS Slim 80% + Seasonal Re-weight (Jun-Jul x2.0) + Personal Blend (WS OOF 0.6504, alpha=0.2056) |
| lgb_gps_slim80_seasonal_blend_prob | - | LGB GPS Slim 80% + Seasonal Re-weight + Personal Blend |
| xgb_gps_slim80_seasonal_blend_prob | - | XGB GPS Slim 80% + Seasonal Re-weight + Personal Blend |
| **ensemble_sh_et_sea_lgb_sea_xgb_sea_prob** | **0.5987** | **ET+LGB+XGB Seasonal 3-way 균등 앙상블 — 현재 최고** |
| ensemble_seasonal_5way (et+lgb+xgb+cb+hgb) | 0.6006 | 5-way Seasonal 앙상블 — CB/HGB 추가로 역효과 |
| et_gps_slim80_summer_holdout_blend_prob | 0.6048 | Summer Holdout Val (Jun-Jul=val, Oct-Nov=train, id04/05 WS fallback) — 계절 가중치보다 효과 없음 |
| ensemble_sh_et_sea_lgb_sea_xgb_sea_xgb_sh_prob | 0.5984 | Seasonal 3-way + XGB Summer Holdout 앙상블 — 개선 미미 |
| et_gps_slim80_rolling_whr_seasonal_blend_prob | 0.6001 | Rolling(ma3/ma7/delta3 57개)+WHR변동성(7개) 추가, WS OOF 0.6427 개선→Public 역전 |
| et_gps_slim80_pertarget_seasonal_blend_prob | 0.6031 | 타깃별 독립 피처 선택 (Q3:69, S2:64, S4:118개), WS OOF 0.6398 개선→Public 역전 |
| et_gps_slim80_density_blend_prob | 미제출 | Density Ratio Weighting — Jun-Jul(0.187)이 Oct(2.431)보다 낮은 DR, WS OOF 0.6470 |

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
| **ET GPS Slim 85%** | **0.6406** | Public 0.6055 — slim80 대비 Public 열세 |
| **ET GPS Slim 80%** | **0.6403** | **Public 0.6044 (현재 최고 Public)** — OOF-Public 균형 최적점 |
| ET GPS Slim 75% | 0.6398 | Public 0.6056 — slim80 대비 악화 (피처 과도 제거) |
| v3GPS Slim 90% | 0.6425 | parquet v3+GPS Slim 90% — v2 Slim 대비 역효과 |
| ET v5 GPS Slim 85% (wLight+mBle+mWifi+DOW편차) | 0.6410 | Public 0.6046 — v4 대비 +0.0012 OOF 개선이나 Public은 slim80 대비 열세 |
| 4모델 균등 앙상블 (ET+CatBoost+HGB+MLP) | — | Public 0.6070 — 약한 모델 3개가 ET(Public 0.6044)를 희석. 균등 가중치 앙상블 역효과 확인 |
| GRU ensemble (10 seeds) | ~0.655 | 훈련 윈도우 310개로 데이터 부족 — 부적합 |
| CatBoost GPS Slim 85% | 0.6446 | ET 0.6406 대비 열세. Q3만 개선(+0.0083), Q2/S2 크게 악화. 앙상블 효과 없음 |
| **ET GPS Slim 80% + LOSO logit bias 보정** | OOF 0.6401→0.6225 | **Public 0.6027 (현재 최고)** — 피험자×타깃 logit bias LOSO OOF 추정 후 test 적용. REG=0.5. OOF 수치는 자기 참조로 과낙관적 |
| ET GPS Slim 80% + WS hold-out logit bias 보정 | OOF(WS) 0.6305 | Public 0.6031 (LOSO 보정 대비 소폭 열세) — WS 30% hold-out 기반 보정. 이론적으로 test 구조에 더 가깝나 경험적으로 LOSO 보정 우위 |
| LOSO logit bias REG 튜닝 (REG=0.25) | OOF 0.6129 | Public 0.6029 (REG=0.5 0.6027 대비 소폭 악화) — REG=0.5가 현재 최적 확인. REG=0.1(OOF 0.5993)·1.0(OOF 0.6304)은 미제출 |

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
| GPS (기준) | 184개 | 0.6465 | 0.6044502 | — |
| Slim 90% | 132개 | 0.6422 | 미제출 | +0.0043 OOF |
| Slim 85% | ~115개 | 0.6406 | 0.6054882 | slim80 대비 Public 열세 |
| **Slim 80%** | **~100개** | **0.6403** | **0.6044461** | **현재 최고 Public** |
| Slim 75% | 87개 | 0.6398 | 0.6056109 | slim80 대비 Public 악화 |

**핵심 발견**: OOF는 slim 비율을 낮출수록 단조 개선(87개에서 0.6398). Public은 slim80(0.6044)가 최적 — slim85(0.6055)보다 낮은 비율(더 적은 피처)이 오히려 Public에서 유리. 단 slim75(0.6056)는 다시 악화 — 피처 제거의 sweet spot은 slim80.

**slim 피처 선택 원리**: 전체 184개 피처 중 하위 중요도 피처들은 cross-subject 일반화에 노이즈로 작용. 상위 85% 커버 피처만 사용함으로써 subject별 특이 패턴에 과적합하는 피처를 차단.

### 28단계: CatBoost GPS Slim 85% — ET 대비 열세

ET GPS Slim 85%와 동일한 피처셋·구조로 CatBoost 도입 (모델 다양성 확보 목적).

**Optuna 탐색 공간**
```python
iterations: 200~2000, learning_rate: 0.01~0.3 (log)
depth: 3~10, l2_leaf_reg: 1.0~10.0
subsample: 0.5~1.0, colsample_bylevel: 0.3~1.0
```

**타깃별 OOF LL 비교**

| 타깃 | CatBoost | ET Slim 85 | 차이 |
|------|:--------:|:----------:|:----:|
| Q1 | 0.7016 | 0.6947 | -0.0069 ❌ |
| Q2 | 0.6532 | 0.6446 | -0.0086 ❌ |
| Q3 | **0.6358** | 0.6441 | **+0.0083** ✅ |
| S1 | 0.6119 | 0.6065 | -0.0054 ❌ |
| S2 | 0.6209 | 0.6074 | -0.0135 ❌ |
| S3 | 0.6031 | 0.6015 | -0.0016 ❌ |
| S4 | 0.6857 | 0.6856 | ≈동률 |
| **평균** | **0.6446** | **0.6406** | **-0.0040** ❌ |

Optuna 소요 시간: Q1만 4시간 11분, 7 타깃 전체 약 16시간.

**결론**: Q3 하나만 CatBoost 우세, 나머지 6개 타깃 ET 우세. 앙상블 시 Q3 소폭 개선되나 Q2·S2 악화가 더 커 전체 OOF 악화 예상. **ET의 완전 무작위 분할이 CatBoost 대칭 트리보다 LOSO cross-subject 일반화에 적합**.

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

### 29단계: parquet_features_v5 + DOW편차 피처 — 미미한 효과

wLight(16개), mBle(25개), mWifi(21개) 피처를 parquet_features_v5.py로 통합. 개인별 요일 편차 피처(23개 기준 센서 × 2: dow_dev + dow_mean)를 dow_deviation_features.py로 구현.

ET GPS Slim 85%와 동일 구조에 v5 피처 + DOW편차 적용 (`extratrees_v5_gps_slim85_ensemble.py`).

| 구성 | 특이사항 |
|------|---------|
| parquet_features_v5 | v2 + wLight + mBle + mWifi — 센서 소스 3개 추가 |
| DOW mean/dev 피처 | 참조 데이터(train dates)로 subject×weekday 평균 계산 후 편차 |
| GPS Slim 85% | 기존과 동일 — slim 비율 유지 |

| 타깃 | v5 OOF LL | v4 참조 LL | 차이 |
|------|:---------:|:---------:|:----:|
| Q1 | 0.6927 | 0.6947 | -0.0020 |
| Q2 | 0.6467 | 0.6446 | +0.0021 |
| Q3 | 0.6416 | 0.6441 | -0.0025 |
| S1 | 0.6076 | 0.6065 | +0.0011 |
| S2 | 0.6101 | 0.6074 | +0.0027 |
| S3 | 0.6086 | 0.6015 | +0.0071 (악화) |
| S4 | 0.6789 | 0.6856 | -0.0067 |
| **평균** | **0.6410** | **0.6406** | **+0.0012 개선** |

**결과**: OOF +0.0012 개선(0.6422 ref 대비), **Public 0.6046** — slim80(0.6044) 대비 열세. S3 악화가 두드러짐.
**핵심**: DOW mean 피처가 DOW dev보다 중요도 높음. 그러나 v5 피처 전체 효과는 미미 — 피처 확장보다 **학습 구조 변화**가 필요한 단계 진입.

### 30단계: 4모델 균등 앙상블 — 역효과

ET GPS Slim 85% + CatBoost GPS Slim 85% + HGB GPS Slim 85% + MLP GPS Slim 85%를 동일 가중치(1/4)로 결합 (`et_cb_hgb_mlp_ensemble.py`).

**Public 0.6070** — ET 단독 Public(0.6044) 대비 크게 악화.

**실패 원인**: CatBoost(OOF 0.6446), HGB, MLP의 Public 성능이 ET(0.6406) 대비 열세임을 확인하지 않고 앙상블. 각 모델이 OOF에서 다른 패턴을 보여도, Public에서 ET보다 못한 모델들이 ET 신호를 희석시킴.

**핵심 교훈**: 앙상블 구성 전 각 모델의 **Public 점수 독립 검증 필수**. OOF 다양성만으로 앙상블 효과를 기대하는 것은 위험.

### 31단계: 데이터 분할 구조 분석 — 향후 전략 재검토

train/test 날짜 분포를 분석한 결과, 대회의 train/test 분리는 **피험자 내부 시점 기준의 블록 분할**임을 발견.

```
피험자 내 타임라인 예시:
[train_block1: 6월~7월] → [test_block: 7월~9월] → [train_block2: 9월~11월]
```

**LOSO CV vs 실제 test의 근본 불일치**:
- LOSO CV 평가: "완전히 새로운(본 적 없는) 피험자를 예측할 수 있는가?"
- 실제 test 구조: "이미 학습한 피험자의 중간 날짜를 예측하는가?"

이 불일치가 OOF(~0.640) - Public(~0.604) 갭의 핵심 원인. LOSO OOF가 실제 test보다 어려운 문제를 평가하고 있어 낙관적으로 보임.

**다음 전략 방향** (우선순위):
1. **within-subject CV**: subject×time 블록으로 실제 대회 구조 근사
2. **개인별 fine-tuning**: 베이스 모델 → 각 피험자 train으로 개인화
3. **피험자 임베딩**: subject_id를 명시 피처로 주입

### 32단계: LOSO OOF 기반 피험자별 logit bias 보정 — 현재 최고 공개점수

31단계에서 발견한 LOSO-test 구조 불일치를 활용해, LOSO OOF 예측에서 피험자·타깃별 **로짓 편향(logit bias)**을 추정하고 test 예측에 적용 (`et_gps_slim80_calibrated.py`).

**보정 방법**

```python
# 피험자별 타깃별 bias 최적화
def fit_logit_bias(pred_oof, y_true, reg=0.5):
    logit_p = logit(clip(pred_oof))
    def obj(b):
        return log_loss(y_true, expit(logit_p + b)) + 0.5 * b**2
    return minimize_scalar(obj, bounds=(-2.0, 2.0), method="bounded").x

# test 예측에 적용
p_cal = expit(logit(clip(p_raw)) + bias[subject_id][target])
```

**단계별 구성**
- Phase 0: Feature Importance (slim 80% — ~100개 피처 유지)
- Phase 1: 캐시된 extratrees_gps_slim80 Optuna 파라미터 로드
- Phase 2: 10 seeds LOSO 앙상블 → OOF 수집
- Phase 3: 피험자×타깃 logit bias 피팅 (CALIB_REG=0.5)
- Phase 4: bias 적용 후 test 예측 저장

**주목할 편향 (절댓값 기준)**

| 피험자 | 타깃 | LOSO bias | 의미 |
|--------|------|:--------:|------|
| id03 | Q1 | +0.281 | 베이스 모델이 id03 Q1을 과소예측 |
| id06 | Q1 | -0.350 | 베이스 모델이 id06 Q1을 과대예측 |
| id05 | S3 | -0.386 | 베이스 모델이 id05 S3를 과대예측 |

**결과**

| 지표 | 값 |
|------|:--:|
| OOF LL (보정 전) | 0.6401 |
| OOF LL (보정 후) | 0.6225 |
| Public Score | **0.6027 (현재 최고 공개점수)** |

OOF LL 개선(-0.018)이 Public 개선(-0.017)으로 연결됨. 단, OOF LL은 보정 데이터와 평가 데이터가 같아 과낙관적 — Public과의 실질적 개선을 Public 점수로 확인.

### 33단계: Within-subject 시간순 hold-out 기반 logit bias 보정 — LOSO 보정 대비 소폭 악화

LOSO 보정의 이론적 약점(실제 test에서 모델이 피험자를 알고 있으나, 보정 추정 시 피험자 정보 없음)을 개선하기 위해 within-subject 시간순 hold-out으로 보정 데이터를 구성 (`et_gps_slim80_ws_calibrated.py`).

**핵심 차이**: 각 피험자의 학습 데이터 앞 70%로 학습 후, 뒤 30%를 hold-out 보정 데이터로 사용.

```python
# 피험자 i의 날짜를 시간순 정렬 후 70:30 분할
dates_sorted = sort(subject_i_dates)
train_dates = dates_sorted[:int(n * 0.70)]   # 다른 9명 + 자신의 앞 70%로 학습
holdout_dates = dates_sorted[int(n * 0.70):]  # 자신의 뒤 30%로 보정 추정
```

**이론적 근거**: 실제 test에서 모델은 해당 피험자의 train 데이터 전체로 학습됨 → WS hold-out은 "피험자 알고 있음" 시나리오를 근사, LOSO보다 test 구조에 더 가까움.

**실험 결과 (OOF LL 비교)**

| 타깃 | 보정 전 | LOSO 보정 | WS 보정 | 비고 |
|------|:------:|:---------:|:-------:|------|
| Q1 | 0.6977 | 0.6744 | 0.6784 | WS > LOSO |
| Q2 | 0.6395 | 0.6264 | 0.6301 | WS > LOSO |
| S3 | 0.5985 | 0.5741 | 0.5917 | WS > LOSO |
| **평균** | **0.6401** | **0.6225** | **0.6305** | WS > LOSO |

**Public Score: 0.6031** — LOSO 보정(0.6027) 대비 소폭 악화.

**분석**: 이론적으로 WS 보정이 더 적합하나, 실제 공개점수는 LOSO 보정이 우위. 이유:
1. WS 보정 데이터(각 피험자 뒤 30% 날짜)가 실제 test 날짜 분포와 다를 수 있음
2. id04의 WS 평균 |bias| 0.197 (LOSO 0.069 대비 3배) — 18날짜로 보정 추정 시 분산 과다
3. LOSO 편향이 적어 더 보수적으로 작동 → 과도한 보정 위험 감소

### 34단계: LOSO logit bias 보정 REG 튜닝 — REG=0.5가 현재 최적

REG=0.5(Public 0.6027)를 기준으로 REG=0.1·0.25·1.0을 비교 (`et_gps_slim80_reg_tuning.py`).

Phase 0~2(피처 계산 + 10 seeds LOSO 앙상블)를 1회만 실행하고, Phase 3에서 REG별 bias 재추정으로 효율적 비교.

**REG별 결과**

| REG | OOF LL (보정 후) | 평균 \|bias\| 대표값 | Public | 비고 |
|:---:|:---:|---|:---:|---|
| 0.10 | 0.5993 | id03 Q1=+0.813, id06 Q1=-0.984, id05 S3=-1.088 | 미제출 | 과보정 우려 — 극단 편향 |
| 0.25 | 0.6129 | id03 Q1=+0.475, id06 Q1=-0.582 | **0.6029** | REG=0.5 대비 소폭 악화 |
| **0.50** | **0.6225** | id03 Q1=+0.281, id06 Q1=-0.350 | **0.6027** | **현재 최고 — 최적 REG** |
| 1.00 | 0.6304 | id03 Q1=+0.157, id06 Q1=-0.193 | 미제출 | 과소보정 예상 |

**결론**: REG가 낮아질수록 OOF LL은 개선되지만(자기참조 과적합), Public에서는 REG=0.5가 최적. REG=0.25도 소폭 악화(0.6029)에 그쳐 REG=0.5의 L2 페널티 강도가 이 데이터에 잘 맞음을 확인. REG 방향의 추가 튜닝은 한계 도달.

### 35단계: Transductive Z-score 정규화 — 현재 최고 공개점수

기존 z-score 정규화는 train 데이터만으로 피험자 통계를 계산. 그러나 test 센서 데이터(레이블 없음)를 추가하면 특히 학습 날짜가 적은 피험자(id03·id10: 33일)의 개인 기준선 추정이 안정화됨.

**핵심 변경**

```python
# 기존: train 데이터만으로 피험자 통계 계산
def compute_subj_stats(df, sensor_cols):
    return df.groupby("subject_id")[sensor_cols].agg(["mean", "std"])

# 개선: parquet_feat(train+test 날짜 전체)로 통계 계산 — 레이블 없는 X만 사용
def compute_transductive_stats(parquet_feat, sensor_cols):
    avail = [c for c in sensor_cols if c in parquet_feat.columns]
    return parquet_feat.groupby("subject_id")[avail].agg(["mean", "std"])
```

**구성**: slim 80% 피처 선택(184개 → 99개 유지 / 85개 제거) + subj_mean 추가(106개) → LOSO 10 seeds 앙상블 + logit bias 보정(REG=0.5)

**결과**

| 지표 | 값 |
|------|:--:|
| 피처 수 | 106개 (slim 80% 99개 + subj_mean 7개) |
| Public Score | **0.6020 (현재 최고 공개점수)** |

calibrated 방식(0.6027) 대비 0.0007 추가 개선. 레이블 없는 센서 데이터만 활용하므로 information leakage 없음.

### 36단계: 센서 Lag 피처 추가 — OOF 악화, 미제출

전날 센서값(lag1)과 7일 이동평균(roll7)을 피처로 추가. 날짜 shift 트릭으로 구현: parquet date D의 센서값 → 레이블 날짜 D+1에 연결.

```python
# 날짜 shift 트릭 (lag1)
lag1_df["date"] = (lag1_df["date_dt"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
# 날짜 shift 트릭 (roll7)
roll7_df["date"] = (roll7_df["date_dt"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
```

**결과**: 총 피처 384개 → slim 80% 후 210개(기존 slim 80% ~100개 대비 2배 이상). OOF 0.6483(transductive 0.6405 대비 악화).

**원인**: LOSO 환경에서 센서 lag 피처가 개별 피험자의 특이 시계열 패턴에 과적합. 피처 수 과다(210개)로 노이즈 증가. 미제출.

### 37단계: ET Within-Subject CV + Transductive — WS val 신뢰성 확인

LOSO CV 대신 피험자 내부 시간순 분할(앞 80% train, 뒤 20% val)로 CV 전략 전환. 실제 test 구조("기존 피험자의 날짜 예측")와 더 근접한 평가 방식.

**WS val OOF 결과**

| 타깃 | WS val OOF LL |
|------|:------------:|
| Q1 | 0.7122 |
| Q2 | 0.6456 |
| Q3 | 0.6387 |
| S1 | 0.5501 |
| S2 | 0.6134 |
| S3 | 0.6084 |
| S4 | 0.6179 |
| **평균** | **0.6266** |

**WS val(0.6266) ≈ Public(0.6267) — WS val이 실제 test 분포를 정확히 반영하는 신뢰 가능한 estimator임 확인.**

그러나 Public 0.6268 — transductive LOSO(0.6020) 대비 크게 악화.

**원인**: Optuna가 WS val 20행(소규모)에서 depth=3~5의 매우 얕은 트리를 최적으로 찾음. LOSO params(depth=5~22)보다 훨씬 단순한 모델 → WS val 평가는 신뢰 가능하나 WS로 학습한 모델 자체가 LOSO 모델보다 약함.

### 38단계: Hard threshold 실험 — Pseudo-labeling 전략 폐기

et_gps_slim80_transductive 예측에서 p > 0.85 → 1, p < 0.15 → 0으로 hard 변환 후 제출.

**목적**: 고신뢰 예측의 정확도 확인 및 pseudo-labeling 가능성 평가.

**결과**: Public **0.9682** — 모든 제출 중 최악(0보다 나쁨).

**원인 분석**: S1·S2·S3에서 모델이 0.9 이상을 예측하는 샘플 다수가 실제 오답. ET 모델이 해당 타깃들에 대해 체계적 과신 편향을 가짐. hard threshold로 오답을 확신으로 변환하면 log-loss 극대화.

**결론**: pseudo-labeling 전략 완전 폐기. 고신뢰 예측조차 신뢰할 수 없음을 확인.

### 39단계: Multi-task MLP LOSO + Transductive — MLP×LOSO 부적합 확인

Transductive Z-score 정규화 + MLP(106→128→64→7) + LOSO GroupKFold(10) + 10 seeds 앙상블.

**아키텍처**
```
Input (106피처: slim 80% 99개 + subj_mean 7개)
  → [Linear(128) → BN → ReLU → Drop(0.4)]
  → [Linear(64) → BN → ReLU → Drop(0.4)]
  → Q1 / Q2 / Q3 / S1 / S2 / S3 / S4  (각각 Linear(64→1) + Sigmoid)
```

**LOSO OOF 결과 (10 seeds 누적)**

| 타깃 | OOF LL |
|------|:------:|
| Q1 | 1.6213 |
| Q2 | 1.0939 |
| Q3 | 1.2091 |
| S1 | 1.0885 |
| S2 | 1.0297 |
| S3 | 0.9390 |
| S4 | 1.1038 |
| **평균** | **1.1551** |

**랜덤 예측(0.693)보다 나쁨 — 미제출.**

**근본 원인**: LOSO 구조에서 MLP는 9명의 학습 데이터 패턴을 암기. held-out subject에서 "9명과 다른 패턴"이 감지되면 역예측 발생 → OOF > 1.0. 이는 코드 버그가 아닌 MLP와 LOSO의 구조적 불일치. ET의 완전 무작위 분할은 암기를 방지하지만 MLP는 gradient descent로 암기를 강화함.

MLP가 의미있는 성능을 내려면 WS CV 구조(학습 fold에 해당 피험자 포함)가 필요.

### 40단계: 피험자별 가변 REG 보정 — Transductive 기반 미세 조정

Transductive Z-score(35단계)에서 LOSO logit bias 보정 REG=0.5를 피험자별로 가변 적용.

**핵심 아이디어**: 훈련 날짜가 많은 피험자는 bias 추정이 안정적 → REG 낮춰도 됨. 날짜가 적은 피험자는 추정 불안정 → REG 높여서 보수적 보정.

```python
BASE_REG = 0.5
BASE_N   = 45   # 전체 피험자 평균 훈련 날짜 수 (기준)
reg = BASE_REG * (BASE_N / n_dates)
# id04 (57일) → REG=0.395  (더 많이 보정 허용)
# id03 (33일) → REG=0.682  (보수적 보정)
# id10 (33일) → REG=0.682
```

**피험자별 REG 결과**

| 피험자 | 훈련일수 | REG |
|--------|:-------:|:---:|
| id03 | 33일 | 0.682 |
| id10 | 33일 | 0.682 |
| id01 | 41일 | 0.549 |
| id09 | 41일 | 0.549 |
| id05 | 44일 | 0.511 |
| id02 | 48일 | 0.469 |
| id06 | 48일 | 0.469 |
| id07 | 49일 | 0.459 |
| id08 | 56일 | 0.402 |
| id04 | 57일 | 0.395 |

**결과**: OOF 0.6235 (LOSO 기준 자기참조, 과낙관적). 미제출 — Public 제출로 실제 효과 확인 필요.

### 41단계: 전체 데이터 학습 + WS OOF 편향 보정 — 구조 불일치 해소 시도

**핵심 문제 정의**: LOSO는 test 예측에서 90% warm(피험자 X: 10 fold 중 9개에서 학습됨)이지만, 실제 이상적인 test 구조는 100% warm(항상 피험자 X를 학습에 포함). 이 10% 구조 불일치가 성능 손실의 일부.

**해소 전략**:
- 전체 450행으로 단일 모델 학습 (100% warm test 예측)
- WS OOF(피험자별 마지막 20% val, 나머지 전체로 학습)로 편향 보정 데이터 수집 (보정 구조도 warm-start)
- best_slim(LOSO 최적화 params) 그대로 사용 — WS Optuna 재튜닝 시 depth=3~5 얕은 트리 문제 회피

**WS OOF 결과 (타깃별 LL)**

| 타깃 | WS val LL |
|------|:---------:|
| Q1 | 0.7141 |
| Q2 | 0.6503 |
| Q3 | 0.6716 |
| S1 | 0.5892 |
| S2 | 0.6669 |
| S3 | 0.6287 |
| S4 | 0.6650 |
| **평균** | **0.6551** |

**결과**: **Public 0.6009 (현재 최고)** — varreg(0.6019), transductive(0.6020) 대비 큰 폭 개선. 100% warm-start 전략이 실제로 유효함을 확인.

### 42단계: 피험자별 개인 모델 블렌딩 — WS OOF 기준 최고

전체 모델(Global)과 피험자별 개인 모델(Personal)을 블렌딩해 각 피험자의 개인 특성을 추가로 포착.

**구조**
```
Global ET:  전체 450행 학습 (best_slim params, 10 seeds)
Personal ET: 피험자 자신의 데이터만 학습 (depth=3, max_feat=0.3, n_est=50, 10 seeds)
블렌드: alpha * personal + (1-alpha) * global
alpha: WS OOF val log-loss 최소화로 전역 최적화
편향 보정: 블렌드된 WS OOF 예측으로 피험자별 로짓 편향 보정 (REG=0.5)
```

**WS OOF 타깃별 LL (alpha=0.2056)**

| 타깃 | blend LL | global-only LL | 차이 |
|------|:--------:|:--------------:|:----:|
| Q1 | 0.7011 | 0.7141 (alldata_ws) | -0.0130 |
| Q2 | 0.6665 | 0.6503 (alldata_ws) | +0.0162 (악화) |
| Q3 | 0.6658 | 0.6716 (alldata_ws) | -0.0058 |
| S1 | 0.5718 | 0.5892 (alldata_ws) | -0.0174 |
| S2 | 0.6519 | 0.6669 (alldata_ws) | -0.0150 |
| S3 | 0.6244 | 0.6287 (alldata_ws) | -0.0043 |
| S4 | 0.6635 | 0.6650 (alldata_ws) | -0.0015 |
| **평균** | **0.6493** | **0.6547** (phase내 global-only) | **-0.0054** |

**최적 alpha = 0.2056**: 개인 모델을 약 20% 반영. Q2를 제외한 6/7 타깃에서 개선.

**결과**: **Public 0.5992 (현재 최고)** — alldata_ws(0.6009) 대비 +0.0017 추가 개선. 전체 데이터 학습(100% warm-start) + 개인 모델 블렌딩이 시너지 효과 발휘.

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
| **CatBoost GPS Slim 85%** | OOF 0.6446 (❌ ET 0.6406 대비 열세) | Q3(+0.0083)만 ET 대비 우세. Q2(-0.0086), S2(-0.0135) 크게 열세. ET의 완전 무작위 분할이 CatBoost 대칭 트리보다 LOSO cross-subject 일반화에 유리 |
| **parquet_features_v5 (wLight+mBle+mWifi)** | OOF GPS 동률 (개별 추가 효과 없음) | 센서 소스 3개 추가해도 LOSO OOF 변화 없음. GPS가 이미 이동 패턴 포착, BLE/WiFi는 10명 환경에서 cross-subject 일반화 실패 |
| **DOW deviation 피처 (요일별 편차)** | ET v5: OOF 0.6410 (+0.0012 v4 대비) | DOW mean(요일 평균) 피처가 DOW dev(편차)보다 중요도 높음. 단독 기여보다 상호 보완 효과. S3 소폭 악화 |
| **4모델 균등 앙상블 (ET+CatBoost+HGB+MLP GPS Slim 85%)** | Public 0.6070 (❌ ET 0.6044 대비 크게 악화) | 약한 모델 3개(CatBoost 0.6446, HGB 미확인, MLP 미확인)가 ET(0.6406) 신호를 희석. 균등 가중치 전략 실패 — 각 구성 모델의 Public 성능이 확인된 후 앙상블해야 함 |
| **LOSO OOF 기반 피험자별 logit bias 보정 (REG=0.5)** | OOF 0.6401→0.6225, Public 0.6027 | 10 seeds LOSO 앙상블 OOF에서 피험자×타깃 logit bias 추정 후 test 적용. id03/id06/id05의 큰 편향이 보정됨. OOF-based 보정이므로 OOF LL 수치 자체는 과낙관적 |
| **WS 시간순 hold-out 기반 logit bias 보정** | OOF LOSO_cal 0.6225 → WS_cal 0.6305, Public 0.6031 (❌ LOSO 보정 0.6027 대비 소폭 악화) | 각 피험자 앞 70% 날짜로 학습, 뒤 30%로 보정 추정. 이론적으로 test 구조에 더 가까우나 id04의 WS |bias|=0.197(LOSO 0.069 대비 3배)으로 분산 과다. LOSO 보정이 경험적으로 우위 |
| **LOSO logit bias 보정 REG 튜닝** | REG=0.25: OOF 0.6129, Public 0.6029 (❌ REG=0.5 대비 소폭 악화) | REG 낮출수록 OOF 개선(자기참조 과적합). Public에서는 REG=0.5(0.6027)가 최적 — REG 방향 추가 개선 여지 없음 확인 |
| **Transductive Z-score 정규화** | **Public 0.6020 (현재 최고)** | train+test 전체 센서 통계로 피험자 기준선 추정 안정화. calibrated(0.6027) 대비 추가 개선. 레이블 없는 X만 사용 — leakage 없음 |
| **센서 Lag 피처 (sensorlag1 + sensorroll7)** | OOF 0.6483 (❌ 악화), 미제출 | 전날 센서값/7일 이동평균을 날짜 shift 트릭으로 구현. 피처 수 210개로 증가 → LOSO 환경에서 subject 특이 시계열 패턴 과적합 |
| **ET Within-Subject CV + Transductive** | WS val OOF 0.6266 ≈ Public 0.6268 (❌ LOSO 대비 크게 악화) | WS val(0.6266)이 Public(0.6267)과 일치 — WS val이 신뢰 가능한 estimator 확인. 단 Optuna가 depth=3~5 얕은 트리 선택 → 모델 자체 약함 |
| **Hard threshold 실험 (p>0.85→1, p<0.15→0)** | Public 0.9682 (❌ 실험 목적) | S1~S3 고신뢰 예측 다수가 오답임 확인. ET 모델의 체계적 과신 편향 발견. pseudo-labeling 전략 완전 폐기 |
| **MLP LOSO + Transductive** | OOF 1.1551 (❌ 랜덤 0.693보다 나쁨), 미제출 | MLP가 9명 학습 패턴 암기 후 held-out subject에 역예측. LOSO cold-start × 신경망 암기의 구조적 불일치 확인 |

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
| **Feature Importance Slim sweet spot** | slim 비율 낮출수록 OOF 단조 개선(75%=0.6398). Public에서는 slim80(0.6044)이 최적 — slim85(0.6055)보다 낮고, slim75(0.6056)보다도 낮음. slim80이 OOF-Public 균형 최적점. |
| **GRU/LSTM 부적합 판단** | 310개 훈련 윈도우는 GRU 파라미터(hidden×n_layers×7 타깃) 대비 데이터 부족. lag/roll 피처가 이미 시계열 의존성 포착. 짧은 시퀀스(33~57일)와 LOSO 콜드스타트 문제로 시계열 모델은 이 문제에 맞지 않음. |
| **CatBoost LOSO 열세** | CatBoost GPS Slim 85% OOF 0.6446 vs ET 0.6406. Q3만 개선(+0.0083), Q2(-0.0086)·S2(-0.0135) 크게 열세. ET 무작위 분할이 CatBoost 대칭 트리보다 unseen subject 일반화에 유리. Optuna도 trial당 1~4분(ET 대비 10배 이상 느림)으로 비효율. |
| **윈도우 페어 트렌드 피처** | roll3-roll7(단기 모멘텀), lag1-roll7(오늘 vs 주간 평균), roll7-roll28(주간 vs 월간 추세)을 개인별 시계열 방향 포착에 활용. NaN 안전 처리: 두 roll 중 하나라도 NaN이면 NaN으로 처리 |
| **mACStatus 수면 프록시** | 스마트폰 충전 여부(0/1)가 수면 스케줄 규칙성의 간접 지표. 수면 중(00-06h) 충전 비율 44%, 취침 전(22-24h) 27%, 기상 후(06-09h) 저충전 → 규칙적 충전 패턴은 규칙적 수면 스케줄과 상관 |
| **수면 확장 구간 wHr (00-09h)** | 현재 수면 구간(00-06h)을 확장해 기상 직전까지 포함. 코드셰어 0.6003 접근법이 사용한 수면 HR 스파이크(HR > mean+1std) 비율 피처 도입. 수면 중 일시적 심박 상승은 각성 이벤트 신호 |
| **train/test 블록 구조 발견** | 대회의 train/test 분리는 각 피험자 내부의 시점으로 이루어짐. 한 피험자의 타임라인이 [train_block1 → test_block → train_block2] 형태로 교차. 즉 일부 test 날짜는 training 날짜보다 앞서고, 일부 training 날짜는 test 날짜보다 뒤. |
| **LOSO CV vs 실제 test 구조 불일치** | LOSO CV는 "완전히 새로운 피험자에게 일반화 가능한가?"를 평가하나, 실제 test는 "이미 학습에 사용된 피험자의 특정 날짜를 예측"하는 문제. 이 불일치가 OOF-Public 갭(~0.035)의 근본 원인. 동일 피험자이므로 실제로는 within-subject 예측에 가까움. |
| **OOF 낙관론의 역설** | LOSO OOF(~0.640)가 Public(~0.604)보다 나빠 보이는 이유: OOF는 완전 미지 피험자 예측이라 어렵고, Public은 기존 피험자 예측이라 쉬움. 즉 Public이 더 유리한 구조. OOF를 개선해도 Public이 비례해서 따라오지 않는 이유 중 하나. |
| **4모델 균등 앙상블 함정** | 구성 모델의 Public 성능을 확인하지 않고 OOF 기준으로 앙상블하면 역효과. CatBoost/HGB/MLP 각각의 Public 성능이 ET보다 열세인 상태에서 균등 가중치 앙상블은 ET 신호 희석. **앙상블 전 각 모델의 Public 검증 필수**. |
| **피처 버전 v5 효과 한계** | wLight+mBle+mWifi+DOW편차를 추가해도 OOF +0.0012 개선에 그침. 이는 더 많은 피처보다 **학습 구조 자체의 변화**(within-subject CV, 피험자 임베딩 등)가 필요한 단계임을 시사. |
| **피험자별 logit bias 보정 효과** | LOSO OOF에서 추정한 피험자×타깃 logit bias를 test에 적용해 Public 0.6044→0.6027 개선. 보정은 모델이 각 피험자를 과대·과소 예측하는 체계적 오류를 로짓 공간에서 보정. REG=0.5의 L2 페널티로 소표본 bias 추정의 과적합 방지. |
| **LOSO vs WS 보정의 역설** | WS 보정(0.6031)이 LOSO 보정(0.6027)보다 이론적으로 test 구조에 더 가까우나 경험적으로 열세. 원인: ① WS hold-out(뒤 30%)이 실제 test 날짜 분포와 불일치 ② id04의 WS |bias| 0.197(LOSO 0.069 대비 3배)으로 편향 추정 불안정 ③ LOSO 보정이 더 보수적(편향 작음)으로 작동해 과보정 위험 감소. |
| **OOF 보정의 자기 참조 주의** | LOSO 보정 후 OOF LL 0.6225는 과낙관적 — 보정 편향이 OOF 데이터에서 추정되고 동일 OOF에서 평가되므로 완전히 신뢰할 수 없음. Public 점수(0.6027)가 실질적 성능 지표. |
| **보정 REG의 최적점 존재** | REG=0.1(OOF 0.5993) → 0.25(0.6129) → 0.5(0.6225) → 1.0(0.6304): OOF는 REG 감소에 단조 개선. Public은 REG=0.5가 최적(0.6027), REG=0.25는 소폭 악화(0.6029). 과보정(REG 너무 낮음)이 Public에서 역효과. REG 방향의 추가 개선은 한계 도달. |
| **Transductive Z-score 효과** | train+test 전체 센서 통계로 피험자 기준선 추정이 더 안정적. 특히 33일 데이터(id03·id10)에서 통계 추정 개선. calibrated(0.6027) 대비 0.0007 추가 개선으로 현재 최고(0.6020) 달성. 레이블 없는 X만 사용하므로 leakage 없음. |
| **센서 lag 피처의 LOSO 함정** | sensorlag1(전날 센서)/sensorroll7(7일 평균)은 날짜 shift 트릭으로 구현 가능하나, LOSO 환경에서 subject 특이 시계열 패턴에 과적합. 피처 수 384개(slim 80% → 210개)가 기존 slim 80%(~100개)보다 많아 노이즈 증가. OOF 0.6483(0.6405 대비 악화). |
| **WS val이 신뢰 가능한 estimator** | et_ws_cv_transductive의 WS val OOF(0.6266)가 Public(0.6267)과 매우 근접. LOSO OOF(~0.640)보다 더 실제 test 분포를 반영. 단 WS 모델이 Optuna에서 depth=3~5 얕은 트리를 찾아 모델 자체가 약함 → WS val이 좋은 측정 도구이나 모델 품질은 LOSO가 우위. |
| **MLP × LOSO 근본 부적합** | MLP는 9명 학습 데이터의 공통 패턴을 암기. held-out subject에서 역예측 발생. OOF 1.1551 > 랜덤 0.693 → 아무것도 모르는 것보다 나쁜 모델. 코드 버그가 아닌 구조적 문제(LOSO cold-start × 신경망 암기). MLP는 WS CV 구조에서만 시도 가능. |
| **Hard threshold → Pseudo-labeling 폐기** | p>0.85 고신뢰 예측을 hard 1로 바꾸면 Public 0.9682(대폭 악화). S1~S3 다수 예측이 0.9 부근에서 오답. ET 모델이 해당 타깃들에 대해 체계적 과신 편향을 가짐. pseudo-labeling은 오답 레이블을 학습에 추가하는 역효과 확정. |
| **WS OOF 신기록의 역설** | hgb_gps_rolling_slim80 WS OOF 0.6204(역대 최고)이나 Public 악화 확인. Rolling 피처(ma3/ma7/delta3) + Personal Blend 조합이 WS holdout(피험자별 마지막 20%)에만 과적합. WS split의 시간순 구조가 rolling 피처를 유독 유리하게 평가 — WS OOF 낮다고 항상 Public 개선되지 않음. NaN 네이티브 HGB가 rolling 43/57 선택 vs ET 22/57 차이도 주목. |
| **wHr 변동성 피처의 NaN 함정** | hr_sleep_iqr/p10/p90 등 wHr 변동성 7개 피처가 ET slim 80%에서 0/7 선택. 원인: wHr 데이터 225행(50%)이라 NaN 50% 피처 → ET 중요도 자동 낮아짐. HGB(NaN 네이티브)에서는 선택될 수 있으나 개인 모델 분산 과다 문제로 상쇄. 50% 이상 NaN 피처는 fillna 기반 모델에서 원천적으로 불리. |

---

## 향후 개선 방향

> 현재 최고 공개점수: **0.5955** (et_lgb_cb_xgb — ET+LGB+CB+XGB 4-way 앙상블) / 이전 최고: **0.5962** (et_lgb_cb_w2 — ET+LGB+CB 2:1:1) / WS OOF 최고: **0.6204** (hgb_gps_rolling, Public 역전 확인) / 리더보드 1위: 0.56119

### 성능 병목 분석

#### OOF-Public 갭 고착 (~0.038 일정)

| 모델 | OOF LL | Public Score | 갭 |
|------|:------:|:------------:|:--:|
| ET 단독 | 0.6469 | 0.6061 | **0.041** |
| HGB+ET | 0.6434 | 0.6103 | 0.033 |
| MLP+HGB+ET | 0.6383 | 0.6062 | 0.032 |
| ET GPS Slim 85% | 0.6406 | 0.6055 | 0.035 |
| ET GPS Slim 80% | 0.6403 | 0.6044 | 0.036 |
| ET v5 GPS Slim 85% | 0.6410 | 0.6046 | 0.036 |
| ET Slim 80% + LOSO 보정 | 0.6401 (보정 전 기준) | 0.6027 | 0.037 |
| **ET Slim 80% + Transductive Z-score** | **0.640x (LOSO OOF)** | **0.6020** | **~0.038** |

모델을 바꿔도 OOF-Public 갭이 ~0.035 고착. **핵심 원인**: LOSO CV는 "미지 피험자 예측"을 평가하나 실제 test는 "기존 피험자의 날짜 예측" — 구조적 불일치. 피처 추가·모델 변경보다 **CV 전략과 학습 구조 변화**가 다음 돌파구. Transductive Z-score(35단계)와 logit bias 보정(32단계)으로 갭 내에서 추가 개선 확인.

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
| 25 | ~~ET GPS Feature Importance Slim (90/85/80/75%)~~ | ✅ 완료 — slim80 OOF 0.6403, **Public 0.6044 (현재 최고)**. slim85(0.6055)보다 slim80(0.6044)가 Public 우세 |
| 26 | ~~GRU 시계열 모델 (window=14, 10 seeds)~~ | ✅ 완료 — OOF ~0.655 (훈련 윈도우 310개 부족, 부적합) |
| 27 | ~~CatBoost GPS Slim 85%~~ | ✅ 완료 — OOF 0.6446 (ET 0.6406 대비 +0.0040 열세. Q3만 우세, Q2/S2 크게 악화) |
| 28 | ~~wLight + mBle + mWifi 피처 추가 (parquet_features_v5)~~ | ✅ 완료 — 개별 OOF GPS 동률. parquet_features_v5.py로 통합. DOW편차 피처와 결합 시 OOF 0.6410 (+0.0012) |
| 29 | ~~요일 효과 편차 피처 (DOW deviation)~~ | ✅ 완료 — dow_deviation_features.py 구현. DOW mean 피처가 DOW dev보다 중요도 높음. 단독 기여 미미 |
| 30 | ~~4모델 균등 앙상블 (ET+CatBoost+HGB+MLP)~~ | ✅ 완료 — Public 0.6070 (역효과). 약한 모델 3개가 ET 희석. 균등 가중치 앙상블 전략 한계 재확인 |
| 31 | ~~데이터 분할 구조 분석 (train/test 블록 구조 발견)~~ | ✅ 완료 — LOSO CV vs 실제 test 구조 불일치 확인. OOF-Public 갭 근본 원인 파악. 다음 전략 방향 도출 |
| 32 | ~~LOSO OOF 기반 피험자별 logit bias 보정 (REG=0.5)~~ | ✅ 완료 — Public 0.6027. OOF LL 0.6401→0.6225 개선 |
| 33 | ~~Within-subject 시간순 hold-out 기반 logit bias 보정~~ | ✅ 완료 — Public 0.6031 (LOSO 보정 대비 소폭 악화). WS OOF LL 0.6305 (LOSO 0.6225보다 보수적) |
| 34 | ~~LOSO logit bias 보정 REG 튜닝 (0.1/0.25/1.0)~~ | ✅ 완료 — REG=0.25 Public 0.6029 (REG=0.5 0.6027 대비 소폭 악화). REG=0.5가 최적. REG 방향 추가 탐색 한계 |
| 35 | ~~Transductive Z-score 정규화~~ | ✅ **완료 — Public 0.6020 (현재 최고 공개점수)**. train+test 전체 센서 통계로 피험자 기준선 추정 안정화 |
| 36 | ~~센서 Lag 피처 (sensorlag1 + sensorroll7)~~ | ✅ 완료 — 미제출 (OOF 0.6483 악화). LOSO 환경에서 센서 lag 과적합. 피처 수 과다(210개) |
| 37 | ~~ET Within-Subject CV + Transductive~~ | ✅ 완료 — Public 0.6268 (역효과). WS val OOF(0.6266)≈Public(0.6267) — WS val이 신뢰 가능한 estimator 확인. 단 모델 자체 약함(depth=3~5) |
| 38 | ~~Hard threshold 실험 (p>0.85→1, p<0.15→0)~~ | ✅ 완료 — Public 0.9682 (실험 목적). S1~S3 고신뢰 예측 오답 확인. pseudo-labeling 전략 폐기 |
| 39 | ~~MLP LOSO + Transductive~~ | ✅ 완료 — 미제출 (OOF 1.1551, 랜덤 0.693보다 나쁨). MLP×LOSO 근본 부적합 확인 |
| 40 | ~~피험자별 가변 REG 보정 (varreg)~~ | ✅ 완료 — **Public 0.6019 (현재 최고)**. transductive(0.6020) 대비 +0.0001 개선. reg = BASE_REG * BASE_N / n_dates |
| 41 | ~~전체 데이터 학습 + WS OOF 편향 보정 (alldata_ws)~~ | ✅ 완료 — **Public 0.6009 (현재 최고)**. 100% warm-start 구조 불일치 해소 효과 확인 |
| 42 | ~~피험자별 개인 모델 블렌딩 (personal_blend)~~ | ✅ 완료 — **Public 0.5992 (현재 최고)**. alpha=0.2056(Global 80% + Personal 20%). alldata_ws(0.6009) 대비 추가 개선 |
| 43 | ~~per-target alpha 최적화 (pers_pertarget)~~ | ✅ 완료 — Public 0.6027 (WS OOF 0.6440 개선 but Public 역전). per-target 자유도 7개가 WS OOF 과적합 유발 |
| 44 | ~~개인 모델 파라미터 grid search (depth×max_feat×min_leaf)~~ | ✅ 완료 — depth=2, max_feat=0.5, min_leaf=3, alpha=0.3595, **Public 0.5989** |
| 45 | ~~Global ET WS OOF Optuna 재탐색~~ | ✅ 완료 — **Public 0.5999**. WS 맥락 재최적화로 pers_grid_best 대비 미개선. WS OOF 0.6329 |
| 46 | ~~LGB GPS Slim 80% Personal Blend~~ | ✅ 완료 — **Public 0.6079**. LGB WS OOF Optuna 50 trials. ET(0.5989) 대비 열세 — 소규모 데이터에서 ET 강함 재확인 |
| 47 | ~~ET + LGB 앙상블~~ | ✅ 완료 — **Public 0.5982 (당시 최고)**. 이종 모델 결합으로 분산 감소. pers_grid_best(0.5989) 대비 +0.0007 개선 |
| 48 | ~~Cross-Target Stacking (LOSO OOF ct 피처)~~ | ✅ 완료 — 미제출 (WS OOF 0.6427 악화). LOSO OOF ct 피처가 WS split에서 노이즈로 작용. personal_blend(0.6329) 대비 열세 |
| 49 | ~~CatBoost GPS Slim 80% Personal Blend~~ | ✅ 완료 — WS OOF 0.6485 (단독 최약). subject_enc categorical 처리로 앙상블 다양성 기여 의도 |
| 50 | ~~ET+LGB+CB 3-way 앙상블~~ | ✅ 완료 — **Public 0.5963**. ET+LGB(0.5982) 대비 +0.0019 개선. CB 이종 오류 패턴 기여 |
| 51 | ~~ET+LGB+CB 2:1:1 앙상블 (ET 비중 강화)~~ | ✅ **완료 — Public 0.5962 (현재 최고)**. 3-way 1:1:1(0.5963) 대비 +0.0001 추가 개선. ET 신뢰성 우월 재확인 |
| 52 | ~~XGB GPS Slim 80% Personal Blend~~ | ✅ 완료 — WS OOF blend LL **0.6120** (모든 단일 모델 최고, alpha=0.2382). 미제출 |
| 53 | ~~ET + XGB 1:1 앙상블~~ | ✅ 완료 — **Public 0.5968**. et_lgb_cb_w2(0.5962) 대비 악화. XGB WS OOF 신뢰성 낮음 시사 |
| 54 | ~~XGB GPS Slim 80% Personal Blend 단독~~ | ✅ 완료 — **Public 0.6037** (WS OOF 0.6120 대비 큰 괴리). XGB 소규모 데이터 과적합 심함 |
| 55 | ~~ET + LGB + CB + XGB 4-way 앙상블~~ | ✅ **완료 — Public 0.5955 (현재 최고)**. 단독 성능 나빠도 앙상블 다양성 기여. 이종 모델 전략 재확인 |
| 56 | ~~HGB GPS Slim 80% Personal Blend~~ | ✅ 완료 — WS OOF blend LL **0.6249** (alpha=0.3385). S1=0.6017, S3=0.5584 강점. ET보다 우수한 WS OOF |
| 57 | ~~wHr 변동성 피처 + ET Personal Blend~~ | ❌ WS OOF 0.6486 (personal_blend 0.6329 대비 악화). ET slim 80%에서 wHr 변동성 피처 **0/7 선택** — 50% NaN 피처는 ET 중요도 낮음. 방향 폐기 |
| 58 | ~~HGB 개인 모델 블렌딩 (Global ET + Personal HGB)~~ | ❌ WS OOF 0.6508 (ET 개인 모델 대비 악화). alpha=0.0977 극소 — 30~50행에 ~190개 피처 HGB 분산 과다. 방향 폐기 |
| 59 | ~~Rolling 피처 추가 ET (ma3/ma7/delta3, 19기준→57개)~~ | ❌ WS OOF 0.6443. 22/57 rolling 피처 선택. LOSO 환경 센서 lag 과적합 패턴 반복 |
| 60 | ~~HGB GPS + Rolling Slim 80% Personal Blend~~ | ❌ WS OOF **0.6204** (신기록)이나 Public 악화 확인. 43/57 rolling 피처 선택(HGB NaN 네이티브 덕분). WS OOF 최적화 방향 자체가 Public 역전 — Personal Blend + Rolling 전체 방향 폐기 |

### 남은 개선 방향 (우선순위 순)

> **현재 상황 (60단계 완료)**: ET+LGB+CB+XGB 4-way **0.5955** (현재 최고). wHr 변동성, HGB 개인 모델, Rolling 피처 (ET/HGB) 실험 완료 — 모두 Public 역전 확인. WS OOF 신기록(0.6204) 달성이나 Public 악화 — WS OOF 기반 최적화 방향 자체 폐기. 앙상블 가중치 Optuna, Stacking 구조적 한계 확인. **근본적 전략 전환 필요: LOSO 패러다임 탈피 → WS CV + 피험자 직접 모델링.**

**핵심 교훈**:
- 모델 다양성(ET+LGB+CB) > 피처 복잡도 증가
- WS OOF 개선 ≈ Public 개선 (단, per-target alpha 과적합 예외, XGB WS OOF도 주의)
- 소규모 데이터에서 ET가 LGB/CB/XGB보다 강함 (ET 0.5989 < LGB 0.6079 < CB 0.6485 < XGB WS 0.6120)
- ET 비중 강화(2:1:1)가 균등(1:1:1)보다 유리 — ET 신뢰성 재확인
- XGB WS OOF 0.6120은 앙상블 노이즈 가능성 → 4-way 시 낮은 가중치 권장

#### 완료된 방향

| 방법 | 결과 |
|------|------|
| ~~GPS 피처 추가~~ | ~~완료 — Public 0.6044~~ |
| ~~앵커 구조~~ | ~~완료 — Stage2 역효과~~ |
| ~~윈도우 페어 트렌드 + mACStatus + 수면확장 wHr~~ | ~~완료~~ |
| ~~wLight + mBle/mWifi 피처 추가~~ | ~~완료 — OOF GPS 동률~~ |
| ~~요일 효과 피처~~ | ~~완료 — 단독 효과 미미~~ |
| ~~개인별 fine-tuning (logit bias 보정)~~ | ~~완료 — Public 0.6027. REG 튜닝 포화~~ |
| ~~Transductive Z-score 정규화~~ | ~~완료 — Public 0.6020~~ |
| ~~Within-subject CV + Transductive~~ | ~~완료 — Public 0.6268 (역효과). WS val 신뢰성 확인~~ |
| ~~Pseudo-labeling~~ | ~~폐기 — hard threshold(0.9682) 실험으로 고신뢰 예측 오답 확인~~ |
| ~~MLP LOSO~~ | ~~폐기 — OOF 1.1551, MLP×LOSO 구조적 부적합~~ |
| ~~per-target alpha 최적화~~ | ~~완료 — Public 0.6027 (역전). WS OOF 자유도 7개 과적합~~ |
| ~~개인 모델 파라미터 grid search~~ | ~~완료 — 신기록. depth=2, max_feat=0.5, min_leaf=3 최적~~ |
| ~~wHr 변동성 피처 추가~~ | ~~완료 — WS OOF 0.6486 (악화). ET slim 80%에서 0/7 선택, 50% NaN으로 ET 중요도 낮음~~ |
| ~~HGB 개인 모델 교체 (ET depth=2 → HGB)~~ | ~~완료 — WS OOF 0.6508 (악화). alpha=0.097 극소, 30~50행에 190개 피처 분산 과다~~ |
| ~~Rolling 피처 (ma3/ma7/delta3)~~ | ~~완료 — ET WS OOF 0.6443, HGB WS OOF 0.6204. 신기록 달성이나 Public 역전 — WS OOF 기반 최적화 방향 자체 폐기~~ |

#### 완료 후 제출 대기 방향

| 방법 | 상태 | 비고 |
|------|------|------|
| ~~피험자별 가변 REG 보정~~ | ✅ **제출 완료 — Public 0.6019** | transductive(0.6020) 대비 +0.0001 개선 |
| ~~전체 데이터 학습 + WS OOF (100% warm-start)~~ | ✅ **제출 완료 — Public 0.6009** | varreg, transductive 대비 개선. 100% warm-start 효과 확인 |
| ~~피험자별 개인 모델 블렌딩 (Global+Personal ET)~~ | ✅ **제출 완료 — Public 0.5992** | alpha=0.2056. alldata_ws(0.6009) 대비 +0.0017 추가 개선 |
| ~~per-target alpha 최적화~~ | ✅ **제출 완료 — Public 0.6027 (역전)** | WS OOF 개선 but 자유도 7개 과적합. 단일 alpha가 더 robust |
| ~~개인 모델 파라미터 grid search~~ | ✅ **제출 완료 — 신기록** | depth=2, max_feat=0.5, min_leaf=3, alpha=0.3595, WS OOF 0.6408 |

#### 미완료 방향 (우선순위 순)

> **구조적 진단 (업데이트)**: WS val(마지막 20%) 구조가 실제 테스트를 잘못 시뮬레이션. 실제 test = [Train앞]→[TEST중간]→[Train뒤] 구조인데, WS val = [Train]→[VAL끝]로 val 이후 training이 없음 → fwd features가 val에서 NaN, test에서는 실제 값 → 분포 불일치로 WS OOF 신뢰 불가. **해결: Middle-Block CV** (val=중간 블록, train=앞+뒷블록).

| 순위 | 방법 | 핵심 근거 | 난이도 | 리스크 |
|------|------|-----------|--------|--------|
| ~~폐기~~ | ~~앙상블 가중치 Optuna~~ | ~~WS OOF 기반 최적 비율 탐색 → et_gps_slim80_global_ws_optuna(0.5999)로 효과 미미 확인~~ | ~~—~~ | ~~완료~~ |
| ~~폐기~~ | ~~Stacking (meta-learner)~~ | ~~Q1/Q3/S1에서 상수 예측 — 90행 WS OOF로는 5모델 차별 가중치 학습 불가. 구조적 한계~~ | ~~—~~ | ~~완료~~ |
| ~~폐기~~ | ~~Multi-window WS CV + subject_id 직접 피처화~~ | ~~WS OOF 0.6324 — Q3(0.6782)/S3(0.6345) 심각 악화. 방향 폐기~~ | ~~—~~ | ~~완료~~ |
| ~~완료~~ | ~~역방향 시간 피처 (WS val 기반 평가)~~ | ~~WS OOF 0.6357, 21/21 선택. WS val fwd=NaN → val/test 분포 불일치. Middle-Block CV로 대체 필요~~ | ~~—~~ | ~~완료~~ |
| ~~완료~~ | ~~Kernel Smoothing blend~~ | ~~bandwidth=60일 WS OOF 0.6489. a2 blend(KS 20%+ET 80%) 소폭 개선. S1/S4 강점, Q2/Q3 약점~~ | ~~—~~ | ~~완료~~ |
| **1순위** | **Middle-Block CV + fwd 피처** | val=중간 블록(40~60%), train=앞블록+뒷블록. 실제 테스트 구조 완전 일치. fwd features가 val에서도 실제 값 → WS OOF 신뢰성 회복 | **중** | **낮음** — 기존 피처/모델 유지, CV 구조만 변경 |
| **2순위** | **피험자별 독립 모델 (Per-subject LogReg)** | LOSO 패러다임 원천 차단. 각 피험자 S 타깃 T: LogReg(C=grid) on S's 33-57 rows. Q1 고착(0.699) 해결 가능성 | **낮음** | **중** — 소표본 분산 위험 |
| **3순위** | **GP 시계열 보간 (Gaussian Process interpolation)** | 각 (피험자, 타깃)을 시간축 GP로 모델링. train labels=관측, test dates=보간점. 양방향 블록 구조를 자연스럽게 포착 | **높음** | **중** — 이진 GP 커널 선택 복잡 |
| **4순위** | **MLP + WS CV + Subject Embedding** | WS val ≈ Public 확인(0.6266 ≈ 0.6267). Subject embedding으로 Q1 고착(0.699) 해결 가능성. MLP×LOSO는 실패이나 MLP×WS CV는 미시도 | **높음** | **중** — 구현 복잡, 시간 소요 |
| **5순위** | **Ridge personal model (ET depth=2 교체)** | 30~50행 소표본에서 ET(depth=2)보다 L2 logistic regression이 이론적으로 더 안정적 | **낮음** | **낮음** — 단독 효과 제한적 |

#### 1순위 상세: 역방향 시간 피처 (Forward-label features)

**데이터 구조 활용**
```
train/test 날짜는 피험자별 블록 교차:
  [Train: Jun~Jul] -> [TEST: Aug~Sep] -> [Train: Oct~Nov]

현재 사용: Aug~Sep test rows -> Jun~Jul labels만 참조 (lag/roll)
미사용: Oct~Nov training labels (test보다 미래지만 train split에 존재)
```

**피처 정의**
```
query_sleep_date = test row의 sleep_date

fwd_lag1_{t}: query_sleep_date 이후 가장 가까운 train label (nearest future)
fwd_roll7_{t}: query_sleep_date + 1 ~ +7일 이내 train labels 평균
fwd_roll14_{t}: query_sleep_date + 1 ~ +14일 이내 train labels 평균

총 21개 피처 (7 targets x 3 window)
```

**구현 핵심**
```
Train rows fwd 계산: same subject의 나중 training dates에서 참조
  Jun~Jul train rows: Oct~Nov training labels -> fwd 피처 값 존재
  Oct~Nov train rows: fwd features = NaN (더 나중 training 없음)

LOSO fold val rows fwd 계산: fold의 train_fold로만 참조
  val subject는 fold의 training에 없음 -> fwd = NaN (leakage 없음)
  test 시점에는 Oct~Nov labels 실제 존재 -> fwd 값 채워짐

ET NaN 처리: fold train median imputation
  학습 시: NaN을 중앙값으로 처리 (neutral)
  test 시: 실제 forward label 값 입력 -> 예측 개선 기대
```

**기대 효과**
```
동일 피험자의 Oct~Nov 수면 질 = Aug~Sep 수면 질의 강한 예측 신호
  -> 특히 Q1(수면의 질) 고착(0.699) 해결 가능성
  -> 개인별 sleep pattern의 시간 안정성 활용
```

**구현 파일**: `scripts/et_fwdlabel_slim80_personal_blend.py`

#### 2순위 상세: GP 시계열 보간 (Gaussian Process interpolation)

```
각 (피험자, 타깃) 쌍을 시간축 위의 GP로 모델링:
  관측: train labels (이진 0/1 -> probit/logit 변환 후 연속값 근사)
  보간: test dates (train 블록 사이에 위치)

커널 후보:
  RBF (Radial Basis Function): 시간적 smooth 패턴 포착
  Matern 3/2 또는 5/2: 거칠기 조절, 이진 레이블에 더 현실적
  Periodic + RBF: 주간 패턴 + 장기 트렌드

장점: 양방향 블록 구조 (Jun~Jul / Oct~Nov) 자연스럽게 포착
단점: 이진 레이블 GP (Bernoulli likelihood) 구현 복잡
     피험자당 30~50 관측으로 커널 학습 불안정 가능성
```

#### 이전 1순위(폐기) 상세: Multi-window WS CV + subject_id 직접 피처화

**현재 WS CV의 약점**
```
단일 분할(80/20): 피험자당 val 20행 → Optuna 불안정 → depth=3~5 약한 트리
해결: 피험자당 3개 분할점의 OOF 평균을 목적함수로 사용

windows = [0.60, 0.70, 0.80]
  window_1: train [0~60%], val [60~80%]
  window_2: train [0~70%], val [70~90%]
  window_3: train [0~80%], val [80~100%]
  OOF = 3 window 평균 → Optuna 신호 3배 → depth=8~15 수준 모델 기대
```

**subject_id 직접 피처화**
```
LOSO에서 불가: held-out subject의 ID는 학습에서 본 적 없음
WS CV에서 가능: 모든 피험자가 train/val에 동시 등장

효과: 트리가 "id03이면 Q1 기준선 높음" 같은 피험자별 규칙을 직접 학습
     → 현재 subj_mean보다 훨씬 풍부한 피험자 개인화
     → Q1 고착(0.699) 해결 가능성 — subj_mean_Q1은 항상 NaN이나 subject_id는 사용 가능
```

#### 2순위 상세: MLP + WS CV + Subject Embedding

```
WS val 검증 결과 (et_ws_cv_transductive에서 확인):
  WS val OOF: 0.6266 ≈ Public 0.6267  — WS val이 신뢰 가능한 estimator

MLP + LOSO (실패):  LOSO cold-start → MLP 역예측 → OOF 1.1551
MLP + WS CV:        학습 fold에 해당 피험자 포함 → MLP가 피험자 패턴 학습 가능

아키텍처 (subject embedding 추가):
  Input: sensor features + lag features
  Subject Embedding: nn.Embedding(10, 8) → 피험자별 학습 가능한 8차원 벡터
  Forward: [concat(X, subj_emb)] → Linear(256) → BN → ReLU → Drop
                                 → Linear(128) → BN → ReLU → Drop
                                 → 7 targets (Q1~S4)

기대:
  Q1 고착 해결 — subject embedding이 피험자별 수면의 질 기준선 직접 학습
  S1~S4 강점 유지 — MLP가 센서 타깃에서 ET 대비 우위 확인된 패턴 재현
  기존 LOSO 앙상블(ET+LGB+CB+XGB)과 다른 귀납 편향 → 앙상블 다양성 기여
```

#### 3순위 상세: Ridge personal model

```
현재 개인 모델: ET(depth=2, max_feat=0.5, min_leaf=3) — 30~50행에서 여전히 과적합 가능성
대안: L2 Logistic Regression (Ridge)
  - C 파라미터로 정규화 강도 조절 (C=0.01~1.0 grid search)
  - 30~50행 소표본에서 이론적으로 최적
  - 선형 결정 경계 → 피험자별 개인 편차만 학습

구현:
  Global 모델 (LOSO ET): 기존 유지
  Personal 모델: LogisticRegression(C=C_opt, max_iter=1000) per subject per target
  Blend: alpha × personal + (1-alpha) × global  (WS OOF 최적화)
```

#### Q1 고착 문제

Q1("수면의 질 — 기상 직후")은 모든 모델에서 LL ≈ 0.699 고착. **현재 피처셋으로 Q1 설명 한계.**

| 원인 | 대응 방향 |
|------|-----------|
| 수면의 질은 수면 중 생리신호 의존 (뒤척임·REM 비율) | GPS 집 체류 시간, wLight 수면 중 조도가 간접 신호 |
| 개인 편차가 커서 unseen subject 일반화 어려움 | subject_id 피처 + WS CV로 피험자별 오프셋 직접 학습 |
| subj_mean_Q1 항상 NaN (leakage 방지) | 대체 개인화 피처: Transductive z-score, 요일 효과 |

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

### 35단계: Seasonal Re-weighting + 이종 모델 앙상블 — 현재 최고 공개점수

Train(Jun-Jul + Oct-Nov) → Test(Aug-Sep) 분포 이동을 직접 겨냥해 여름 샘플에 높은 가중치 부여.

**Seasonal Re-weighting 원리**
```python
SUMMER_WEIGHT = 2.0
sample_weight = 2.0 if month in [6, 7] else 1.0
```
Jun-Jul(여름) 200행 × 2.0 = 효과적 400행, Oct-Nov 250행 × 1.0. 여름 패턴이 test(Aug-Sep)에 더 가깝다는 가정.

**ET+LGB+XGB Seasonal 3-way 결과**

| 모델 | WS OOF LL | Public |
|------|:---------:|:------:|
| ET Seasonal | 0.6504 | (3-way 구성) |
| LGB Seasonal | - | (3-way 구성) |
| XGB Seasonal | - | (3-way 구성) |
| **3-way 균등 앙상블** | — | **0.5987 (현재 최고)** |

CB Seasonal(WS OOF 0.6485)·HGB Seasonal(WS OOF 0.6249)을 추가한 5-way 앙상블은 Public 0.6006으로 역효과.

**핵심 발견**: CB/HGB가 ET/LGB/XGB보다 Public에서 열세 — 이종 모델 추가가 항상 유리하지 않음.

### 36단계: 도메인 이동 해소 실험 — 반복 실패

**Summer Holdout Validation** (Jun-Jul=val, Oct-Nov=train per subject)
- id04/id05는 Jun-Jul 데이터 없어 WS fallback 적용
- ET Summer Holdout OOF 0.6224, Public **0.6048** — seasonal보다 효과 없음
- 여름 데이터가 별도 val로 쓰이면 정보 손실이 더 크다는 결론

**Rolling + WHR 변동성 피처** (57개 rolling + 7개 WHR)
- WS OOF 0.6427 (seasonal 0.6504보다 개선)
- Public **0.6001** — OOF 개선이 또다시 Public 역전

**타깃별 독립 피처 선택 (Per-Target Slim 80%)**
- Q3: 69개, S2: 64개, S4: 118개 — 타겟마다 최적 피처셋 상이
- WS OOF 0.6398 (역대 최고)
- Public **0.6031** — OOF-Public 역전 패턴 재확인

**핵심 교훈**: WS OOF 개선이 Public 개선으로 이어지지 않는 패턴이 반복. 검증 지표 신뢰도 자체가 한계.

### 37단계: Density Ratio Weighting — Jun-Jul이 test와 덜 유사

고정 계절 가중치(2.0)를 학습 가능한 density ratio로 교체.

**방법론**
```python
# train+test 피처로 binary classifier (train=0, test=1) 학습
clf = LogisticRegression(C=0.1)
clf.fit(X_combined, y_combined)
p_test = clf.predict_proba(X_train)[:, 1]
weight = p_test / (1 - p_test) * (n_train / n_test)  # density ratio
```

**월별 density ratio 분포**

| 월 | mean_dr | n | 예상 대비 |
|----|:-------:|:-:|---------|
| 06 | 0.187 | 54 | 예상 밖 낮음 |
| 07 | 0.401 | 146 | 예상 밖 낮음 |
| 08 | 1.127 | 131 | — |
| 09 | 1.826 | 84 | — |
| 10 | **2.431** | 28 | 예상 밖 높음 |
| 11 | 1.758 | 7 | — |

**핵심 발견**: "Jun-Jul이 test(Aug-Sep)에 가깝다"는 계절 가중치의 근본 가정이 피처 공간에서 틀림. Oct-Nov이 오히려 test와 더 유사. 계절 패턴보다 **개별 피험자의 월별 특성**이 더 복잡하게 작용.

**결과**: WS OOF 0.6470 (미제출) — seasonal OOF 0.6504 소폭 개선이나 Public 검증 불필요 판단.

---

## 환경 설정

```bash
uv pip install pandas pyarrow lightgbm scikit-learn optuna shap xgboost catboost
```

### 실행 순서

```bash
# 현재 최고 공개점수 (ET+LGB+XGB Seasonal 3-way 앙상블, Public 0.5987)
# -> ensemble_seasonal_5way.py 실행 후 et_sea+lgb_sea+xgb_sea 3-way 선택

# ET GPS Slim 80% Seasonal (3-way 앙상블 구성 요소)
uv run scripts/et_gps_slim80_seasonal_blend.py
uv run scripts/lgb_gps_slim80_seasonal_blend.py
uv run scripts/xgb_gps_slim80_seasonal_blend.py

# Density Ratio Weighting (WS OOF 0.6470, 미제출)
uv run scripts/et_gps_slim80_density_blend.py

# Transductive Z-score (Public 0.6020)
uv run scripts/et_gps_slim80_transductive.py

# LOSO logit bias 보정 REG=0.5 (Public 0.6027)
uv run scripts/et_gps_slim80_calibrated.py

# 피험자별 가변 REG 보정 (미제출, WS OOF 기준 비교 대상)
uv run scripts/et_gps_slim80_trans_varreg.py

# 전체 데이터 학습 + WS OOF 편향 보정 (미제출, 100% warm-start)
uv run scripts/et_gps_slim80_alldata_ws.py

# 전체+개인 모델 블렌딩 (Public 0.5992)
uv run scripts/et_gps_slim80_personal_blend.py

# per-target alpha 블렌딩 (Public 0.6027 — 역전)
uv run scripts/et_gps_slim80_pers_pertarget.py

# 개인 모델 파라미터 grid search — 신기록 (depth=2, max_feat=0.5, min_leaf=3)
uv run scripts/et_gps_slim80_pers_grid.py

# WS CV + Transductive (Public 0.6268 — WS val 신뢰성 확인용)
uv run scripts/et_ws_cv_transductive.py

# REG 튜닝 (REG=0.1/0.25/1.0 한 번에 비교)
uv run scripts/et_gps_slim80_reg_tuning.py

# Within-subject 보정 (Public 0.6031)
uv run scripts/et_gps_slim80_ws_calibrated.py

# 베이스 모델 (ET GPS Slim 80%, Public 0.6044)
uv run scripts/extratrees_gps_slim80_ensemble.py

# 현재 최고 공개점수 (ET+LGB+CB+XGB 4-way 앙상블, Public 0.5955)
uv run scripts/ensemble_et_xgb.py

# wHr 변동성 피처 + ET Personal Blend (미제출, WS OOF 0.6486 — 방향 폐기)
uv run scripts/et_gps_whrvar_slim80_personal_blend.py

# ET Global + HGB 개인 모델 블렌딩 (미제출, WS OOF 0.6508 — 방향 폐기)
uv run scripts/et_gps_whrvar_slim80_hgbpers_blend.py

# ET GPS + Rolling Slim 80% Personal Blend (미제출, WS OOF 0.6443 — 방향 폐기)
uv run scripts/et_gps_rolling_slim80_personal_blend.py

# HGB GPS + Rolling Slim 80% Personal Blend (미제출, WS OOF 0.6204 신기록 — Public 역전, 방향 폐기)
uv run scripts/hgb_gps_rolling_slim80_personal_blend.py
```
