"""
MLP (Multi-task) + LOSO + Transductive Z-score + 피험자별 로짓 편향 보정

ExtraTrees 대신 PyTorch Multi-task MLP 사용:
  - 7개 타깃을 공유 백본으로 동시 학습 (타깃 간 상관관계 활용)
  - subj_mean_{t} 피처 포함: 피험자 기저 예측 + 잔차 학습
  - 폴드 내 전체 피처 표준화 (transductive z-score 이후 추가 정규화)
  - BatchNorm + Dropout + weight decay 정규화
  - LOSO GroupKFold(10) + 10 seeds 앙상블

파이프라인:
  - Phase 0: Feature Importance (ET GPS 파라미터로 keep_feats 계산)
  - Phase 1: LOSO 10 seeds MLP 앙상블 (OOF 수집)
  - Phase 2: 피험자별 로짓 편향 보정 (REG=0.5)
  - Phase 3: 보정된 test 예측 저장

출력: submission/mlp_loso_transductive_prob.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize_scalar
from scipy.special import logit as logit_fn, expit as expit_fn
from pathlib import Path
from functools import reduce

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from gps_features import build_gps
from optuna_params_io import load_params

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

ET_KEY_GPS  = "extratrees_gps"
ET_KEY_SLIM = "extratrees_gps_slim80"
IMP_COVERAGE = 0.80

TARGETS  = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS    = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

CALIB_REG  = 0.5
BIAS_BOUND = 2.0

HIDDEN     = [128, 64]
DROPOUT    = 0.4
WEIGHT_DECAY = 1e-3
LR         = 1e-3
EPOCHS     = 500
BATCH_SIZE = 64

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Multi-task MLP
# ──────────────────────────────────────────────

class MultiTaskMLP(nn.Module):
    def __init__(self, n_feats, hidden=None, dropout=DROPOUT, n_targets=7):
        super().__init__()
        if hidden is None:
            hidden = HIDDEN
        layers = []
        prev = n_feats
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, 1) for _ in range(n_targets)])

    def forward(self, x):
        h = self.backbone(x)
        return torch.sigmoid(
            torch.cat([head(h) for head in self.heads], dim=1)
        )


def train_mlp(X_tr, y_tr, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n, d = X_tr.shape
    model = MultiTaskMLP(d).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCELoss()

    X_t = torch.FloatTensor(X_tr).to(DEVICE)
    y_t = torch.FloatTensor(y_tr).to(DEVICE)
    dataset = TensorDataset(X_t, y_t)
    loader  = DataLoader(dataset, batch_size=min(BATCH_SIZE, n), shuffle=True,
                         generator=torch.Generator().manual_seed(seed))

    model.train()
    for _ in range(EPOCHS):
        for Xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


def predict_mlp(model, X):
    model.eval()
    with torch.no_grad():
        X_t = torch.FloatTensor(X).to(DEVICE)
        return model(X_t).cpu().numpy()


# ──────────────────────────────────────────────
# 피처 빌드 유틸
# ──────────────────────────────────────────────

def get_sensor_cols(parquet_feat):
    return [c for c in parquet_feat.columns if c not in ("subject_id", "date")]


def compute_transductive_stats(parquet_feat, sensor_cols):
    avail = [c for c in sensor_cols if c in parquet_feat.columns]
    return parquet_feat.groupby("subject_id")[avail].agg(["mean", "std"])


def apply_zscore(subject_ids, X, stats, sensor_cols):
    X = X.copy()
    for col in sensor_cols:
        if col not in X.columns or (col, "mean") not in stats.columns:
            continue
        means = subject_ids.map(stats[(col, "mean")])
        stds  = subject_ids.map(stats[(col, "std")])
        valid = stds.notna() & (stds > 0)
        X[col] = X[col].astype(float)
        X.loc[valid, col] = (X.loc[valid, col] - means[valid]) / stds[valid]
    return X


def add_date_features(df):
    dt = pd.to_datetime(df["sleep_date"])
    df = df.copy()
    df["day_of_week"]  = dt.dt.dayofweek
    df["month"]        = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def add_subject_mean_features(df, ref, is_train):
    subject_sum   = ref.groupby("subject_id")[TARGETS].sum()
    subject_count = ref.groupby("subject_id")[TARGETS].count()
    if is_train:
        for t in TARGETS:
            s_sum = df["subject_id"].map(subject_sum[t])
            s_cnt = df["subject_id"].map(subject_count[t])
            df[f"subj_mean_{t}"] = (s_sum - df[t]) / (s_cnt - 1).clip(lower=1)
    else:
        subject_mean = subject_sum / subject_count
        for t in TARGETS:
            df[f"subj_mean_{t}"] = df["subject_id"].map(subject_mean[t])
    return df


def build_features(df, ref, parquet_feat, label_feat, is_train, le):
    df = add_date_features(df)
    df = add_subject_mean_features(df, ref, is_train)
    df["subject_enc"] = df["subject_id"].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    df = df.merge(parquet_feat, left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")
    df = df.merge(label_feat, left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")
    base_cols    = (["subject_enc", "day_of_week", "month", "day_of_month",
                     "is_weekend", "week_of_year"]
                    + [f"subj_mean_{t}" for t in TARGETS])
    parquet_cols = [c for c in parquet_feat.columns if c not in ("subject_id", "date")]
    label_cols   = [c for c in label_feat.columns   if c not in ("subject_id", "date")]
    all_cols = ["subject_id"] + base_cols + parquet_cols + label_cols
    return df[[c for c in all_cols if c in df.columns]].reset_index(drop=True)


# ──────────────────────────────────────────────
# Phase 0: Feature Importance (ET GPS 파라미터)
# ──────────────────────────────────────────────

def compute_importance(train, parquet_feat, le, sensor_cols, fold_label_feats,
                       best_gps, transductive_stats):
    from sklearn.ensemble import ExtraTreesClassifier
    importance_dict = {t: [] for t in TARGETS}
    feature_names_ref = {}

    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr   = build_features(train_fold, train_fold, parquet_feat, lf_tr, True, le)
        sid_tr = X_tr["subject_id"].reset_index(drop=True)
        X_tr_z = apply_zscore(sid_tr, X_tr.drop(columns=["subject_id"]),
                              transductive_stats, sensor_cols)
        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            X_tr_t = X_tr_z.drop(columns=drop_cols, errors="ignore")
            if t not in feature_names_ref:
                feature_names_ref[t] = X_tr_t.columns.tolist()
            X_tr_filled = X_tr_t.fillna(X_tr_t.median())
            y = train[t].values
            model = ExtraTreesClassifier(**{**best_gps[t], "random_state": 42})
            model.fit(X_tr_filled, y[tr_idx])
            importance_dict[t].append(model.feature_importances_)

    all_imp = {t: pd.Series(np.mean(importance_dict[t], axis=0),
                            index=feature_names_ref[t])
               for t in TARGETS}
    common = list(reduce(lambda a, b: a & b, [set(v.index) for v in all_imp.values()]))
    combined = pd.DataFrame({t: all_imp[t][common] for t in TARGETS})
    combined["mean_imp"] = combined.mean(axis=1)
    combined = combined.sort_values("mean_imp", ascending=False)
    total = combined["mean_imp"].sum()
    combined["cumsum"] = combined["mean_imp"].cumsum() / total
    keep = combined[combined["cumsum"] <= IMP_COVERAGE].index.tolist()
    print(f"  전체 공통 피처: {len(combined)}개")
    print(f"  상위 {IMP_COVERAGE*100:.0f}% 커버: {len(keep)}개 유지 / {len(combined)-len(keep)}개 제거")
    return set(keep)


# ──────────────────────────────────────────────
# 피험자별 로짓 편향 보정
# ──────────────────────────────────────────────

def fit_logit_bias(pred_oof, y_true, reg=CALIB_REG, bound=BIAS_BOUND):
    eps = 1e-6
    p  = np.clip(pred_oof, eps, 1 - eps)
    lp = logit_fn(p)

    def obj(b):
        return log_loss(y_true, expit_fn(lp + b)) + reg * (b ** 2)

    res = minimize_scalar(obj, bounds=(-bound, bound), method="bounded")
    return float(res.x)


def apply_logit_bias(pred_raw, bias, eps=1e-6):
    p = np.clip(pred_raw, eps, 1 - eps)
    return expit_fn(logit_fn(p) + bias)


# ──────────────────────────────────────────────
# 메인 학습/예측 파이프라인
# ──────────────────────────────────────────────

def train_and_predict(train, test, parquet_feat):
    le          = LabelEncoder().fit(train["subject_id"])
    groups      = train["subject_id"].values
    cv          = GroupKFold(n_splits=10)
    sensor_cols = get_sensor_cols(parquet_feat)
    n_seeds     = len(SEEDS)

    fold_indices = list(cv.split(np.zeros(len(train)), train[TARGETS[0]].values, groups))

    transductive_stats = compute_transductive_stats(parquet_feat, sensor_cols)
    print(f"  transductive 통계: {len(transductive_stats)}명, {len(sensor_cols)}개 센서")
    print(f"  DEVICE: {DEVICE}")

    print("label 피처 사전 계산 중...")
    label_feat_test = build_label_features(train, test)
    fold_label_feats = []
    for tr_idx, val_idx in fold_indices:
        train_fold = train.iloc[tr_idx].copy()
        val_fold   = train.iloc[val_idx].copy()
        lf_tr  = build_label_features(train_fold, train_fold)
        lf_val = build_label_features(val_fold,   val_fold)
        fold_label_feats.append((tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val))
    print("  완료")

    gps_params = load_params(ET_KEY_GPS)
    if not gps_params:
        print("ERROR: extratrees_gps params 캐시 없음.")
        return None
    best_gps = {t: gps_params[t] for t in TARGETS}

    print("\n=== Phase 0: Feature Importance (slim 80%, ET GPS 파라미터) ===")
    keep_feats = compute_importance(train, parquet_feat, le, sensor_cols,
                                    fold_label_feats, best_gps, transductive_stats)
    # MLP: subj_mean_{t} 전체 포함 (잔차 학습)
    for t in TARGETS:
        keep_feats.add(f"subj_mean_{t}")
    print(f"  subj_mean 추가 후: {len(keep_feats)}개\n")

    # 폴드별 피처 행렬 사전 계산
    print("폴드 피처 사전 계산 중 (MLP용, transductive z-score + 표준화)...")
    fold_data = []
    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr_raw  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
        X_val_raw = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)

        sid_tr  = X_tr_raw["subject_id"].reset_index(drop=True)
        sid_val = X_val_raw["subject_id"].reset_index(drop=True)
        X_tr_z  = apply_zscore(sid_tr,  X_tr_raw.drop(columns=["subject_id"]),
                               transductive_stats, sensor_cols)
        X_val_z = apply_zscore(sid_val, X_val_raw.drop(columns=["subject_id"]),
                               transductive_stats, sensor_cols)

        # slim 선택 + DROP_USAGE 제거
        slim_cols = [c for c in X_tr_z.columns if c in keep_feats
                     and c not in DROP_USAGE]
        X_tr_slim  = X_tr_z[[c for c in slim_cols if c in X_tr_z.columns]]
        X_val_slim = X_val_z[[c for c in slim_cols if c in X_val_z.columns]]

        # 폴드 내 전체 피처 표준화 (신경망용)
        mu  = X_tr_slim.mean()
        sig = X_tr_slim.std().replace(0, 1)
        X_tr_norm  = ((X_tr_slim  - mu) / sig).fillna(0).values.astype(np.float32)
        X_val_norm = ((X_val_slim - mu) / sig).fillna(0).values.astype(np.float32)

        y_tr  = train.loc[train_fold.index, TARGETS].values.astype(np.float32)
        y_val = train.loc[val_fold.index,   TARGETS].values.astype(np.float32)

        fold_data.append((tr_idx, val_idx, X_tr_norm, X_val_norm, y_tr, y_val, mu, sig, slim_cols))

    # 테스트 피처 (전체 train 기준)
    lf_full   = build_label_features(train, train)
    X_full    = build_features(train, train, parquet_feat, lf_full,      True,  le)
    X_te_raw  = build_features(test.copy(), train, parquet_feat, label_feat_test, False, le)
    sid_full  = X_full["subject_id"].reset_index(drop=True)
    sid_te    = X_te_raw["subject_id"].reset_index(drop=True)
    X_full_z  = apply_zscore(sid_full, X_full.drop(columns=["subject_id"]),
                             transductive_stats, sensor_cols)
    X_te_z    = apply_zscore(sid_te,   X_te_raw.drop(columns=["subject_id"]),
                             transductive_stats, sensor_cols)

    # 전체 데이터 표준화 파라미터 (full train 기준)
    slim_cols_ref = fold_data[0][8]
    X_full_slim   = X_full_z[[c for c in slim_cols_ref if c in X_full_z.columns]]
    X_te_slim     = X_te_z[[c  for c in slim_cols_ref if c in X_te_z.columns]]
    mu_full  = X_full_slim.mean()
    sig_full = X_full_slim.std().replace(0, 1)
    X_full_norm = ((X_full_slim - mu_full) / sig_full).fillna(0).values.astype(np.float32)
    X_te_norm   = ((X_te_slim  - mu_full) / sig_full).fillna(0).values.astype(np.float32)

    n_feats = X_full_norm.shape[1]
    print(f"  완료 (피처 수: {n_feats}개)\n")

    # Phase 1: LOSO 10 seeds MLP 앙상블
    print(f"=== Phase 1: MLP LOSO 앙상블 ({n_seeds} seeds) ===")
    print(f"  구조: {n_feats} -> {' -> '.join(map(str, HIDDEN))} -> {len(TARGETS)}")
    print(f"  epochs={EPOCHS}, lr={LR}, dropout={DROPOUT}, wd={WEIGHT_DECAY}")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 8))

    mlp_oof  = {t: np.zeros(len(train)) for t in TARGETS}
    mlp_test = {t: np.zeros(len(test))  for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        seed_oof  = {t: np.zeros(len(train)) for t in TARGETS}
        seed_test = {t: np.zeros(len(test))  for t in TARGETS}

        for fold_i, (tr_idx, val_idx, X_tr_n, X_val_n, y_tr, y_val, mu, sig, scols) in enumerate(fold_data):
            # 테스트 피처를 이 폴드의 표준화 파라미터로 변환
            X_te_f = ((X_te_slim - mu) / sig).fillna(0).values.astype(np.float32)

            model = train_mlp(X_tr_n, y_tr, seed + fold_i * 100)
            val_pred = predict_mlp(model, X_val_n)   # [n_val, 7]
            te_pred  = predict_mlp(model, X_te_f)    # [n_te, 7]

            for t_i, t in enumerate(TARGETS):
                seed_oof[t][val_idx]  = val_pred[:, t_i]
                seed_test[t]         += te_pred[:, t_i] / cv.n_splits

        for t in TARGETS:
            mlp_oof[t]  += seed_oof[t]  / n_seeds
            mlp_test[t] += seed_test[t] / n_seeds

        lls = [log_loss(train[t].values, mlp_oof[t] * n_seeds / (seed_i + 1))
               for t in TARGETS]
        ll_str = "  ".join(f"{ll:>6.4f}" for ll in lls)
        print(f"{seed:>6}  {ll_str}  {np.mean(lls):>6.4f}")

    print()
    print("=== Phase 1 최종 OOF (보정 전) ===")
    print(f"{'타깃':<5}  {'OOF LL':>8}")
    print("-" * 20)
    lls_before = {}
    for t in TARGETS:
        ll = log_loss(train[t].values, mlp_oof[t])
        lls_before[t] = ll
        print(f"{t:<5}  {ll:>8.4f}")
    print(f"{'평균':<5}  {np.mean(list(lls_before.values())):>8.4f}")

    # Phase 2: 피험자별 로짓 편향 보정
    print(f"\n=== Phase 2: 피험자별 로짓 편향 보정 (REG={CALIB_REG}) ===")
    subjects = sorted(train["subject_id"].unique())
    biases = {sid: {} for sid in subjects}

    for sid in subjects:
        mask = (train["subject_id"] == sid).values
        for t in TARGETS:
            pred = mlp_oof[t][mask]
            y    = train[t].values[mask]
            if len(np.unique(y)) < 2:
                biases[sid][t] = 0.0
                continue
            biases[sid][t] = fit_logit_bias(pred, y)

    print(f"\n{'피험자':<8}  " + "  ".join(f"{t:>7}" for t in TARGETS) + "  평균|편향|")
    print("-" * (10 + 10 * len(TARGETS)))
    for sid in subjects:
        vals = [biases[sid][t] for t in TARGETS]
        val_str = "  ".join(f"{v:>+7.4f}" for v in vals)
        print(f"{sid:<8}  {val_str}  {np.mean(np.abs(vals)):>8.4f}")

    mlp_oof_cal = {t: np.zeros(len(train)) for t in TARGETS}
    for sid in subjects:
        mask = (train["subject_id"] == sid).values
        for t in TARGETS:
            mlp_oof_cal[t][mask] = apply_logit_bias(mlp_oof[t][mask], biases[sid][t])

    print("\n=== OOF LL 보정 전후 비교 ===")
    print(f"{'타깃':<5}  {'보정전':>8}  {'보정후':>8}  {'개선':>8}")
    print("-" * 38)
    lls_after = {}
    for t in TARGETS:
        ll_b = lls_before[t]
        ll_a = log_loss(train[t].values, mlp_oof_cal[t])
        lls_after[t] = ll_a
        sign = "[+]" if ll_b > ll_a else "[-]"
        print(f"{t:<5}  {ll_b:>8.4f}  {ll_a:>8.4f}  {ll_b-ll_a:>+7.4f} {sign}")
    avg_b = np.mean(list(lls_before.values()))
    avg_a = np.mean(list(lls_after.values()))
    print(f"{'평균':<5}  {avg_b:>8.4f}  {avg_a:>8.4f}  {avg_b-avg_a:>+7.4f}")
    print()
    print("주의: OOF 보정후 LL은 과낙관적 해석 금지. Public score로 확인 필요.")

    # test 예측에 보정 적용
    mlp_test_cal = {t: mlp_test[t].copy() for t in TARGETS}
    for sid in subjects:
        mask_te = (test["subject_id"] == sid).values
        if mask_te.sum() == 0:
            continue
        for t in TARGETS:
            mlp_test_cal[t][mask_te] = apply_logit_bias(mlp_test[t][mask_te], biases[sid][t])

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = mlp_test_cal[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS 피처 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    print("=== MLP Multi-task + LOSO + Transductive Z-score ===\n")
    result = train_and_predict(train, sample, parquet_feat)
    if result is None:
        return

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 최종 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "mlp_loso_transductive_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
