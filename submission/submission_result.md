# 제출 파일에 대한 대회 자체 Metric 점수

**현재 최고: gp_noq2_blend_prob = 0.592709746** (2026-06-10)

현재 최고 파이프라인:
- GP(날짜 RBF, ls=90, MP-bias보정, 10%, Q2=0) + LR_v2ac(v2+mACStatus, 20%) + ET앙상블(ET+LGB+CB+XGB, 70%)

## SSM+biascorr 미제출 후보 (2026-06-11 생성)

GP와 동일한 per-subject-target logit bias correction 추가.

| 파일 | 설명 | MP OOF (avg) |
| --- | --- | --- |
| ssm_biascorr_noq2_a10_blend_prob.csv | SSM+bias(10%, Q2=0) + base | 보정 후 0.6098 |
| ssm_biascorr_pertarget_blend_prob.csv | 타겟별 alpha (Q1=0.119, Q3=0.088, S1-S4=0.15) | 동일 |
| ssm_biascorr_noq2_a05_blend_prob.csv | SSM+bias(5%, Q2=0) + base | 보수적 |
| ssm_biascorr_noq2_a15_blend_prob.csv | SSM+bias(15%, Q2=0) + base | 공격적 |

bias correction 전후 타겟별 MP OOF:

| 타겟 | 보정 전 | 보정 후 | 개선 | GP |
| --- | --- | --- | --- | --- |
| Q1 | 0.6691 | 0.6338 | +0.0353 | 0.6803 |
| Q2 | 0.7972 | 0.7549 | +0.0424 | 0.8530 |
| Q3 | 0.6645 | 0.6289 | +0.0357 | 0.6849 |
| S1 | 0.5667 | 0.5406 | +0.0262 | 0.5734 |
| S2 | 0.5434 | 0.5340 | +0.0094 | 0.5419 |
| S3 | 0.5804 | 0.5651 | +0.0153 | 0.5805 |
| S4 | 0.6222 | 0.6113 | +0.0109 | 0.6312 |
| avg | 0.6348 | **0.6098** | **+0.0250** | 0.6493 |

## SSM 미제출 후보 (2026-06-11 생성)

Bernoulli SSM (RTS Smoother, sigma_q=0.30) — GP 대비 MP OOF +0.0145 개선

| 파일 | 설명 | 핵심 차이 |
| --- | --- | --- |
| ssm_noq2_a10_blend_prob.csv | SSM(10%, Q2=0) + base | Q3도 포함 (SSM이 Q3 도움) |
| ssm_pertarget_blend_prob.csv | 타겟별 alpha (Q1=0.048, Q3=0.017, S1=0.117, S2=0.15, S3=0.118, S4=0.127) | 가장 정교한 튜닝 |
| ssm_noq2_a05_blend_prob.csv | SSM(5%, Q2=0) + base | 보수적 |
| ssm_noq2_a15_blend_prob.csv | SSM(15%, Q2=0) + base | 공격적 |

SSM vs GP 차이:
- Q3: GP는 해로웠지만 SSM은 도움 (SSM Q3 OOF 0.6645 < naive 0.6730)
- S4: SSM=0.6222 vs GP=0.6312 (+0.0090)
- Q1: SSM=0.6691 vs GP=0.6803 (+0.0112)

## 기타 미제출 후보 (생성 완료)

| 파일 | 설명 | 예상 |
| --- | --- | --- |
| gp_noq2q3_blend_prob.csv | Q2=Q3=0, 나머지=0.10 | gp_noq2보다 ~0.0002 개선 예상 (미미) |
| gp_helpful_only_blend_prob.csv | = gp_noq2q3와 동일 | 제출 불필요 |

## 탐색 완료 (추가 신호 없음, 2026-06-11)

| 방향 | 결과 | 비고 |
| --- | --- | --- |
| BLE 피처 | within-subj r<=0.20 | 신호 없음 |
| wLight 피처 | within-subj r<=0.17 | 신호 없음 |
| mAmbience 피처 | within-subj r<=0.08 | 신호 없음 |
| mActivity 피처 | within-subj r<=0.11 | 신호 없음 |
| WiFi home 피처 | within-subj r<=0.13 | 신호 없음 |
| GPS 귀가 시간 | within-subj r<=0.16, coverage 35% | 신호 없음 |
| GPS 루틴 Jaccard | within-subj r<=0.09 | 신호 없음 |
| 양방향 라벨 보간 | MP OOF 0.9264 (naive 0.5966) | 이진 라벨 노이즈로 실패 |
| EWMA label 피처 | LR MP OOF 0.6435 (기존 0.6370) | 기존 roll 피처 대비 나쁨 |
| CB personal blend | ET4 corr=0.92-0.99 | 앙상블 다양성 없음 |

## 전체 제출 이력

| 제출 파일 | 점수 |
| --- | --- |
| submission_sample | 21.1269 |
| lgbm_optuna_prob | 0.6178 |
| lgbm_catboost_ensemble_prob | 0.6170 |
| catboost_optuna_prob | 0.6183 |
| lgbm_catboost_ensemble_v2_prob | 0.6127 |
| lgbm_catboost_ensemble_v3_prob | 0.6142 |
| lgbm_catboost_weighted_prob | 0.6195 |
| lgbm_catboost_et_ensemble_prob | 0.6146 |
| extratrees_ensemble_prob | 0.6061 |
| hgb_et_ensemble_prob | 0.6103 |
| hgb_et_xt_ensemble_prob | 0.6078 |
| mlp_hgb_et_ensemble_prob | 0.6062 |
| extratrees_v3_ensemble_prob | 0.6094 |
| extratrees_clip005_prob | 0.6068 |
| mlp_hgb_et_weighted_prob | 0.6132 |
| extratrees_optuna_v4_prob | 0.6051 |
| extratrees_gps_prob | 0.6044502749 |
| extratrees_v4_ensemble_prob | 0.6162153422 |
| extratrees_wlight_prob | 0.6052869869 |
| extratrees_gps_slim85_prob | 0.6054882763 |
| extratrees_gps_slim80_prob | 0.6044461004 |
| extratrees_gps_slim75_prob | 0.6056109238 |
| et_cb_hgb_mlp_ensemble_prob | 0.6070169324 |
| et_cb_hgb_mlp_ensemble_submission | 11.4383427432 |
| extratrees_v5_gps_slim85_prob | 0.6046318856 |
| et_gps_slim80_calibrated_prob | 0.6027395452 |
| et_gps_slim80_ws_calibrated_prob | 0.6030646769 |
| et_gps_slim80_reg025_prob | 0.6029453794 |
| et_gps_slim80_reg01_prob | 미제출 |
| et_gps_slim80_reg10_prob | 미제출 |
| et_gps_slim80_transductive_prob | 0.6020864059 |
| et_gps_slim80_trans_sensorlag_prob | 미제출 (OOF 0.6483, 악화) |
| et_ws_cv_transductive_prob | 0.6267950764 |
| et_gps_slim80_trans_hard85_prob | 0.9681811619 |
| mlp_loso_transductive_prob | 미제출 (OOF 1.1551, 랜덤보다 나쁨) |
| et_gps_slim80_trans_hard85_prob | 0.9681811619 |
| et_gps_slim80_trans_varreg_prob | 0.6019247537 |
| et_gps_slim80_alldata_ws_prob | 0.6009226938 |
| et_gps_slim80_personal_blend_prob | 0.5992275974 |
| et_gps_slim80_pers_pertarget_prob | 0.6027042926 |
| et_gps_slim80_pers_grid_best_prob | 0.598867734 |
| et_gps_slim80_global_ws_optuna_prob | 0.599952318 |
| lgb_gps_slim80_personal_blend_prob | 0.607939536 |
| et_lgb_ensemble_prob | 0.5982497695 |
| et_lgb_cb_ensemble_prob | 0.5963033028 |
| et_lgb_cb_w2_ensemble_prob | 0.5961904978 |
| et_xgb_ensemble_prob | 0.5968477757 |
| xgb_gps_slim80_personal_blend_prob | 0.6037207666 |
| et_lgb_cb_xgb_ensemble_prob | 0.5955437932 |
| et_lgb_cb_xgb_hgb_ensemble_prob | 0.5973291749 |
| hgb_gps_rolling_slim80_personal_blend_prob | 0.60587054 |
| et_lgb_cb_hgb_ensemble_prob | 0.5985029091 |
| et_multiws_subjectid_prob | 0.6039898137 |
| et_fwdlabel_slim80_personal_blend_prob | 0.6019446797 |
| kernel_et_a2_blend_prob | 0.5954770692 |
| et_midblock_fwdlabel_slim80_personal_blend_prob | 0.6035068105 |
| stacking_optweights_prob | 0.6005423699 |
| ensemble_cross_3way_prob | 0.6009239099 |
| et_gps_deep_slim80_personal_blend_prob | 0.5992521759 |
| ensemble_deep_lgbxgb_prob | 0.5991891949 |
| ensemble_seasonal_3way_prob | 0.5987098486 |
| ensemble_summer_3way_prob | 0.6003930759 |
| et_gps_slim80_chain_blend_prob | 0.5982151223 |
| mlp_gps_slim80_personal_blend_prob | 0.6155500831|
| mlp_gps_slim80_seasonal_blend_prob | 0.6173123431 |
| ensemble_seasonal_et_lgb_xgb_cb_hgb_prob | 0.6006310275 |
| ensemble_seasonal_xgb_hgb_prob | 0.6072243451 |
| et_gps_slim80_summer_holdout_blend_prob | 0.60481657 |
| ensemble_sh_et_sea_lgb_sea_xgb_sea_xgb_sh_prob | 0.598413797 |
| et_gps_slim80_rolling_whr_seasonal_blend_prob | 0.60014174 |
| et_gps_slim80_pertarget_seasonal_blend_prob | 0.6031296505 |
| et_gps_slim80_stabfeat_seasonal_blend_prob | 0.6026260731 |
| lgr_gps_slim80_seasonal_blend_prob | 0.6016037577 |
| et_gps_slim80_ks_blend_prob | 0.6009543044 |
| et_gps_slim80_midpoint_blend_prob | 0.6021865065 |
| ks_gauss30_a2_blend_prob | 0.5948686261 |
| logreg_et_a1_blend_prob | 0.594407112 |
| gauss_logreg_blend_prob | 0.5944438944 |
| logreg_et_a2_blend_prob | 0.5940370938 |
| logreg_et_a4_blend_prob | 0.5954975174 |
| twlr_bw30_a3_blend_prob | 0.5939857645 |
| calib_logit_lam5_blend_prob | 0.6006419636 |
| logreg_v2ac_et_a2_blend_prob | 0.5938618198 |
| gp_logreg_a1_blend_prob | 0.5928672104 |
| gp_mp_logreg_a10_blend_prob | 0.5927858004 |
| gp_mp_logreg_a05_blend_prob | 0.5932539188 |
| gp_noq2_blend_prob | 0.592709746 |
| ssm_noq2_a10_blend_prob | 0.5931270518 |
| ssm_pertarget_blend_prob | 미제출 |
| ssm_biascorr_noq2_a10_blend_prob | 미제출 |
| ssm_biascorr_pertarget_blend_prob | 미제출 |
