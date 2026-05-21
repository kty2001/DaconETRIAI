"""
MLP + HGB + ET — per-target 최적 가중치 앙상블
- 3모델 OOF를 분리 수집 후 타깃별로 scipy Nelder-Mead로 가중치 최적화
- 균등(1/3) 대비 개선 여부 확인
- clip [0.05, 0.95]
출력: submission/mlp_hgb_et_weighted_prob.csv
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.optimize import minimize
from pathlib import Path
import random

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from optuna_params_io import load_params

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

MLP_KEY = "mlp_v1"
HGB_KEY = "hgb_v2"
ET_KEY  = "extratrees_v2"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]
DROP_MLP = DROP_USAGE + [f"subj_mean_{t}" for t in TARGETS]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIMS_MAP = {
    "256-128-64":     [256, 128, 64],
    "512-256-128":    [512, 256, 128],
    "256-128":        [256, 128],
    "128-64-32":      [128, 64, 32],
    "512-256-128-64": [512, 256, 128, 64],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ── MLP ────────────────────────────────────────────────────────────────────────

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, 1) for _ in TARGETS])

    def forward(self, x):
        z = self.backbone(x)
        return torch.cat([h(z) for h in self.heads], dim=1)


def train_mlp(model, X, Y, params, seed):
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["weight_decay"])
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=params["n_epochs"])
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32).to(DEVICE),
                      torch.tensor(Y, dtype=torch.float32).to(DEVICE)),
        batch_size=params["batch_size"], shuffle=True, drop_last=True,
    )
    model.train()
    for _ in range(params["n_epochs"]):
        for xb, yb in loader:
            opt.zero_grad()
            nn.BCEWithLogitsLoss()(model(xb), yb).backward()
            opt.step()
        sch.step()
    return model


def mlp_predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        ).cpu().numpy()


# ── 피처 빌더 ──────────────────────────────────────────────────────────────────

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


def precompute_fold_features(train, test, parquet_feat, le, fold_label_feats, label_feat_test):
    fold_raw = []
    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
        X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)
        fold_raw.append((tr_idx, val_idx,
                         X_tr.drop(columns=["subject_id"]),
                         X_val.drop(columns=["subject_id"])))

    X_te_raw = build_features(test.copy(), train, parquet_feat, label_feat_test, False, le)
    X_te_raw = X_te_raw.drop(columns=["subject_id"])

    # 트리 피처 (타깃별 subj_mean 제거는 외부에서)
    fold_by_target = {}
    te_by_target   = {}
    for t in TARGETS:
        dc = [f"subj_mean_{t}"] + DROP_USAGE
        fold_by_target[t] = [
            (tr_idx, val_idx,
             X_tr.drop(columns=dc, errors="ignore"),
             X_val.drop(columns=dc, errors="ignore"))
            for tr_idx, val_idx, X_tr, X_val in fold_raw
        ]
        te_by_target[t] = X_te_raw.drop(columns=dc, errors="ignore")

    # MLP 피처 (subj_mean 전체 제거 + imputation + scaling)
    fold_mlp = []
    for tr_idx, val_idx, X_tr, X_val in fold_raw:
        Xtr = X_tr.drop(columns=DROP_MLP, errors="ignore")
        Xvl = X_val.drop(columns=DROP_MLP, errors="ignore")
        Xte = X_te_raw.drop(columns=DROP_MLP, errors="ignore")
        med    = Xtr.median()
        scaler = StandardScaler()
        fold_mlp.append((
            tr_idx, val_idx,
            scaler.fit_transform(Xtr.fillna(med).to_numpy()),
            scaler.transform(Xvl.fillna(med).to_numpy()),
            scaler.transform(Xte.fillna(med).to_numpy()),
        ))

    mlp_input_dim = fold_mlp[0][2].shape[1]
    return fold_by_target, te_by_target, fold_mlp, mlp_input_dim


def train_and_predict(train, test, parquet_feat):
    le     = LabelEncoder().fit(train["subject_id"])
    groups = train["subject_id"].values
    cv     = GroupKFold(n_splits=10)
    n_seeds = len(SEEDS)
    n_folds = cv.n_splits

    fold_indices = list(cv.split(np.zeros(len(train)), train[TARGETS[0]].values, groups))

    print("label 피처 사전 계산 중...")
    label_feat_test  = build_label_features(train, test)
    fold_label_feats = []
    for tr_idx, val_idx in fold_indices:
        train_fold = train.iloc[tr_idx].copy()
        val_fold   = train.iloc[val_idx].copy()
        lf_tr  = build_label_features(train_fold, train_fold)
        lf_val = build_label_features(val_fold,   val_fold)
        fold_label_feats.append((tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val))
    print("  완료")

    print("폴드 피처 사전 계산 중...")
    fold_by_target, te_by_target, fold_mlp, mlp_input_dim = precompute_fold_features(
        train, test, parquet_feat, le, fold_label_feats, label_feat_test,
    )
    print(f"  완료 (트리: {fold_by_target[TARGETS[0]][0][2].shape[1]}개, MLP: {mlp_input_dim}개)\n")

    # ── params 로드 ───────────────────────────────────────────────────────────
    cached_hgb = load_params(HGB_KEY)
    cached_et  = load_params(ET_KEY)
    cached_mlp = load_params(MLP_KEY)

    if not cached_hgb:
        raise RuntimeError(f"캐시에 {HGB_KEY} params 없음. hgb_ensemble.py 먼저 실행하세요.")
    if not cached_et:
        raise RuntimeError(f"캐시에 {ET_KEY} params 없음. extratrees_ensemble.py 먼저 실행하세요.")
    if not cached_mlp or "shared" not in cached_mlp:
        raise RuntimeError(f"캐시에 {MLP_KEY} params 없음. multitask_mlp_ensemble.py 먼저 실행하세요.")

    mlp_params = dict(cached_mlp["shared"])
    if "hidden_dims" not in mlp_params:
        mlp_params["hidden_dims"] = HIDDEN_DIMS_MAP[mlp_params["hidden_dims_key"]]

    print("=== Phase 1: 캐시 params 로드 ===")
    print(f"  HGB: Q1 lr={cached_hgb['Q1']['learning_rate']:.4f}")
    print(f"  ET:  Q1 n_est={cached_et['Q1']['n_estimators']}, depth={cached_et['Q1']['max_depth']}")
    print(f"  MLP: hidden={mlp_params['hidden_dims']}, drop={mlp_params['dropout']:.3f}\n")

    # ── Phase 2: 모델별 OOF 수집 ─────────────────────────────────────────────
    Y_all = train[TARGETS].values

    # 누적 합산용
    mlp_oof_acc  = np.zeros((len(train), len(TARGETS)))
    mlp_test_acc = np.zeros((len(test),  len(TARGETS)))
    hgb_oof_acc  = {t: np.zeros(len(train)) for t in TARGETS}
    hgb_test_acc = {t: np.zeros(len(test))  for t in TARGETS}
    et_oof_acc   = {t: np.zeros(len(train)) for t in TARGETS}
    et_test_acc  = {t: np.zeros(len(test))  for t in TARGETS}

    print(f"=== Phase 2: 3모델 OOF 수집 ({n_seeds} seeds) ===")
    for seed_i, seed in enumerate(SEEDS):
        # MLP
        mlp_oof_seed  = np.zeros((len(train), len(TARGETS)))
        mlp_test_seed = np.zeros((len(test),  len(TARGETS)))
        for tr_idx, val_idx, X_tr_sc, X_val_sc, X_te_sc in fold_mlp:
            model = MultiTaskMLP(mlp_input_dim,
                                 mlp_params["hidden_dims"],
                                 mlp_params["dropout"]).to(DEVICE)
            train_mlp(model, X_tr_sc, Y_all[tr_idx], mlp_params, seed=seed)
            mlp_oof_seed[val_idx]  = mlp_predict(model, X_val_sc)
            mlp_test_seed         += mlp_predict(model, X_te_sc) / n_folds

        mlp_oof_acc  += mlp_oof_seed  / n_seeds
        mlp_test_acc += mlp_test_seed / n_seeds

        # HGB + ET (타깃별)
        for ti, t in enumerate(TARGETS):
            y = train[t].values
            hgb_params = {**cached_hgb[t], "random_state": seed}
            et_params  = {**cached_et[t],  "random_state": seed}

            for tr_idx, val_idx, X_tr, X_val in fold_by_target[t]:
                X_te   = te_by_target[t]
                tr_med = X_tr.median()

                hgb = HistGradientBoostingClassifier(**hgb_params)
                hgb.fit(X_tr.to_numpy(), y[tr_idx])
                hgb_oof_acc[t][val_idx] += hgb.predict_proba(X_val.to_numpy())[:, 1] / n_seeds
                hgb_test_acc[t]         += hgb.predict_proba(X_te.to_numpy())[:, 1] / n_seeds / n_folds

                et = ExtraTreesClassifier(**et_params)
                et.fit(X_tr.fillna(tr_med).to_numpy(), y[tr_idx])
                et_oof_acc[t][val_idx] += et.predict_proba(X_val.fillna(tr_med).to_numpy())[:, 1] / n_seeds
                et_test_acc[t]         += et.predict_proba(X_te.fillna(tr_med).to_numpy())[:, 1] / n_seeds / n_folds

        print(f"  seed {seed:>6} ({seed_i+1}/{n_seeds}) 완료")

    # ── Phase 3: 타깃별 최적 가중치 탐색 ─────────────────────────────────────
    print("\n=== Phase 3: 타깃별 가중치 최적화 (Nelder-Mead) ===")
    print(f"{'타깃':<5}  {'w_MLP':>6}  {'w_HGB':>6}  {'w_ET':>6}  {'최적 LL':>8}  {'균등 LL':>8}  {'개선':>7}")
    print("-" * 60)

    optimal_weights = {}
    oof_final  = {t: np.zeros(len(train)) for t in TARGETS}
    test_final = {t: np.zeros(len(test))  for t in TARGETS}

    for ti, t in enumerate(TARGETS):
        y      = train[t].values
        mlp_p  = mlp_oof_acc[:, ti]
        hgb_p  = hgb_oof_acc[t]
        et_p   = et_oof_acc[t]

        def objective(logits):
            w = softmax(np.array(logits))
            p = (w[0] * mlp_p + w[1] * hgb_p + w[2] * et_p).clip(1e-7, 1 - 1e-7)
            return log_loss(y, p)

        # 균등(1/3) 기준 LL
        p_eq   = ((mlp_p + hgb_p + et_p) / 3).clip(1e-7, 1 - 1e-7)
        ll_eq  = log_loss(y, p_eq)

        # 여러 초기값으로 다중 시작 최적화
        best_ll, best_w = ll_eq, np.array([1/3, 1/3, 1/3])
        for init in [[0, 0, 0], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
                     [2, -1, -1], [-1, 2, -1], [-1, -1, 2]]:
            res = minimize(objective, init, method="Nelder-Mead",
                           options={"xatol": 1e-7, "fatol": 1e-7, "maxiter": 2000})
            w = softmax(res.x)
            ll = res.fun
            if ll < best_ll:
                best_ll, best_w = ll, w

        optimal_weights[t] = best_w
        improvement = ll_eq - best_ll

        print(f"{t:<5}  {best_w[0]:>6.3f}  {best_w[1]:>6.3f}  {best_w[2]:>6.3f}"
              f"  {best_ll:>8.4f}  {ll_eq:>8.4f}  {improvement:>+7.4f}")

        # 최적 가중치로 OOF/test 생성
        oof_final[t]  = best_w[0]*mlp_p + best_w[1]*hgb_p + best_w[2]*et_p
        test_final[t] = (best_w[0] * mlp_test_acc[:, ti]
                         + best_w[1] * hgb_test_acc[t]
                         + best_w[2] * et_test_acc[t])

    # ── 최종 OOF 요약 ─────────────────────────────────────────────────────────
    print()
    ref_eq_ll = {"Q1": 0.6994, "Q2": 0.6478, "Q3": 0.6443,
                 "S1": 0.6161, "S2": 0.6161, "S3": 0.6124, "S4": 0.6870}

    print(f"{'타깃':<5}  {'F1':>6}  {'최적 LL':>8}  {'균등v2':>8}  {'차이':>7}")
    print("-" * 44)
    for t in TARGETS:
        f1 = f1_score(train[t].values, (oof_final[t] > 0.5).astype(int))
        ll = log_loss(train[t].values, oof_final[t])
        diff = ll - ref_eq_ll[t]
        sign = "+" if diff >= 0 else ""
        print(f"{t:<5}  {f1:>6.3f}  {ll:>8.4f}  {ref_eq_ll[t]:>8.4f}  {sign}{diff:.4f}")

    avg_ll = np.mean([log_loss(train[t].values, oof_final[t]) for t in TARGETS])
    avg_f1 = np.mean([f1_score(train[t].values, (oof_final[t] > 0.5).astype(int)) for t in TARGETS])
    diff   = avg_ll - 0.6383
    sign   = "+" if diff >= 0 else ""
    print(f"{'평균':<5}  {avg_f1:>6.3f}  {avg_ll:>8.4f}  {'0.6383':>8}  {sign}{diff:.4f}")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = test_final[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== MLP + HGB + ET per-target 가중치 앙상블 ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.05, 0.95)

    print("\n=== 예측 분포 (clip 0.05~0.95) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "mlp_hgb_et_weighted_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
