"""
Multi-task MLP + HGB + ExtraTrees 3모델 앙상블
- MLP: S1/S2/S3/S4 센서 타깃 강점
- HGB+ET: Q1/Q2/Q3 설문 타깃 강점
- 동일 가중치 평균 (1/3 씩)
- 캐시된 mlp_v1 / hgb_v2 / extratrees_v2 params 로드
출력: submission/mlp_hgb_et_ensemble_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

BASELINE_F1 = {
    "Q1": 0.649, "Q2": 0.662, "Q3": 0.712,
    "S1": 0.788, "S2": 0.611, "S3": 0.526, "S4": 0.611,
}

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]
# MLP용 추가 제거 (subj_mean 전체 — 트리와 다양성 확보)
DROP_MLP = DROP_USAGE + [f"subj_mean_{t}" for t in TARGETS]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIMS_MAP = {
    "256-128-64":     [256, 128, 64],
    "512-256-128":    [512, 256, 128],
    "256-128":        [256, 128],
    "128-64-32":      [128, 64, 32],
    "512-256-128-64": [512, 256, 128, 64],
}


# ── 유틸 ───────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── MLP 모델 ───────────────────────────────────────────────────────────────────

class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
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
    """
    두 가지 피처 행렬을 한 번에 준비
    - tree_*: DROP_USAGE만 제거 (subj_mean 유지, 타깃별 제거는 외부에서)
    - mlp_*:  DROP_MLP 제거 + fold-median imputation + StandardScaler
    """
    fold_raw = []
    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
        X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)
        fold_raw.append((tr_idx, val_idx,
                         X_tr.drop(columns=["subject_id"]),
                         X_val.drop(columns=["subject_id"])))

    X_te_raw = build_features(test.copy(), train, parquet_feat, label_feat_test, False, le)
    X_te_raw = X_te_raw.drop(columns=["subject_id"])

    # ── 트리 피처 (subj_mean_t는 각 타깃별 외부 제거) ─────────────────────────
    fold_tree = []
    for tr_idx, val_idx, X_tr, X_val in fold_raw:
        Xtr = X_tr.drop(columns=DROP_USAGE, errors="ignore")
        Xvl = X_val.drop(columns=DROP_USAGE, errors="ignore")
        fold_tree.append((tr_idx, val_idx, Xtr, Xvl))
    X_te_tree = X_te_raw.drop(columns=DROP_USAGE, errors="ignore")

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

    # ── MLP 피처 (subj_mean 전체 제거 + imputation + scaling) ─────────────────
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
    tree_feat_n = fold_by_target[TARGETS[0]][0][2].shape[1]
    print(f"  완료 (트리 피처: {tree_feat_n}개, MLP 입력: {mlp_input_dim}개, device: {DEVICE})\n")

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

    mlp_params = cached_mlp["shared"]
    if "hidden_dims" not in mlp_params:
        mlp_params["hidden_dims"] = HIDDEN_DIMS_MAP[mlp_params["hidden_dims_key"]]

    print("=== Phase 1: 캐시 params 로드 ===")
    print(f"  HGB ({HGB_KEY}):  Q1 lr={cached_hgb['Q1']['learning_rate']:.4f}, iter={cached_hgb['Q1']['max_iter']}")
    print(f"  ET  ({ET_KEY}):   Q1 n_est={cached_et['Q1']['n_estimators']}, depth={cached_et['Q1']['max_depth']}")
    print(f"  MLP ({MLP_KEY}):  hidden={mlp_params['hidden_dims']}, drop={mlp_params['dropout']:.3f}, "
          f"ep={mlp_params['n_epochs']}\n")

    # ── Phase 2: 멀티 시드 앙상블 ────────────────────────────────────────────
    Y_all   = train[TARGETS].values
    n_seeds = len(SEEDS)
    n_folds = len(fold_mlp)

    print(f"=== Phase 2: MLP+HGB+ET 3모델 앙상블 ({n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균F1   평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 16))

    ens_oof  = {t: np.zeros(len(train)) for t in TARGETS}
    ens_test = {t: np.zeros(len(test))  for t in TARGETS}

    for seed_i, seed in enumerate(SEEDS):
        seed_oof  = {t: np.zeros(len(train)) for t in TARGETS}
        seed_test = {t: np.zeros(len(test))  for t in TARGETS}

        # ── MLP (전체 타깃 동시) ──────────────────────────────────────────────
        mlp_oof  = np.zeros((len(train), 7))
        mlp_test = np.zeros((len(test),  7))
        for tr_idx, val_idx, X_tr_sc, X_val_sc, X_te_sc in fold_mlp:
            model = MultiTaskMLP(mlp_input_dim,
                                 mlp_params["hidden_dims"],
                                 mlp_params["dropout"]).to(DEVICE)
            train_mlp(model, X_tr_sc, Y_all[tr_idx], mlp_params, seed=seed)
            mlp_oof[val_idx]  = mlp_predict(model, X_val_sc)
            mlp_test         += mlp_predict(model, X_te_sc) / n_folds

        # ── HGB + ET (타깃별) ─────────────────────────────────────────────────
        for ti, t in enumerate(TARGETS):
            y = train[t].values

            hgb_params = {**cached_hgb[t], "random_state": seed}
            et_params  = {**cached_et[t],  "random_state": seed}

            for tr_idx, val_idx, X_tr, X_val in fold_by_target[t]:
                X_te = te_by_target[t]

                hgb = HistGradientBoostingClassifier(**hgb_params)
                hgb.fit(X_tr.to_numpy(), y[tr_idx])
                hgb_val  = hgb.predict_proba(X_val.to_numpy())[:, 1]
                hgb_test = hgb.predict_proba(X_te.to_numpy())[:, 1]

                tr_med = X_tr.median()
                et  = ExtraTreesClassifier(**et_params)
                et.fit(X_tr.fillna(tr_med).to_numpy(), y[tr_idx])
                et_val  = et.predict_proba(X_val.fillna(tr_med).to_numpy())[:, 1]
                et_test = et.predict_proba(X_te.fillna(tr_med).to_numpy())[:, 1]

                # 3모델 동일 가중치 (1/3 씩)
                seed_oof[t][val_idx] = (mlp_oof[val_idx, ti] + hgb_val + et_val) / 3
                seed_test[t]        += (mlp_test[:, ti] + hgb_test + et_test) / 3 / cv.n_splits

        for t in TARGETS:
            ens_oof[t]  += seed_oof[t]  / n_seeds
            ens_test[t] += seed_test[t] / n_seeds

        f1s = [f1_score(train[t].values, (ens_oof[t] * n_seeds / (seed_i + 1) > 0.5).astype(int))
               for t in TARGETS]
        lls = [log_loss(train[t].values, ens_oof[t] * n_seeds / (seed_i + 1))
               for t in TARGETS]
        f1_str = "  ".join(f"{f:>6.3f}" for f in f1s)
        print(f"{seed:>6}  {f1_str}  {np.mean(f1s):>6.3f}  {np.mean(lls):>6.4f}")

    # ── 최종 결과 출력 ────────────────────────────────────────────────────────
    print()
    ref_ll = {"Q1": 0.6994, "Q2": 0.6478, "Q3": 0.6443,
              "S1": 0.6161, "S2": 0.6161, "S3": 0.6124, "S4": 0.6870}
    mlp_ll = {"Q1": 0.7145, "Q2": 0.6587, "Q3": 0.6514,
              "S1": 0.6055, "S2": 0.5841, "S3": 0.5847, "S4": 0.6798}

    print(f"{'타깃':<5}  {'F1':>6}  {'앙상블 LL':>9}  {'HGB+ET':>7}  {'MLP':>7}  {'개선':>7}")
    print("-" * 50)
    for t in TARGETS:
        f1 = f1_score(train[t].values, (ens_oof[t] > 0.5).astype(int))
        ll = log_loss(train[t].values, ens_oof[t])
        diff = ll - ref_ll[t]
        print(f"{t:<5}  {f1:>6.3f}  {ll:>9.4f}  {ref_ll[t]:>7.4f}  {mlp_ll[t]:>7.4f}  {diff:>+7.4f}")

    avg_ll = np.mean([log_loss(train[t].values, ens_oof[t]) for t in TARGETS])
    avg_f1 = np.mean([f1_score(train[t].values, (ens_oof[t] > 0.5).astype(int)) for t in TARGETS])
    print(f"{'평균':<5}  {avg_f1:>6.3f}  {avg_ll:>9.4f}  {'0.6434':>7}  {'0.6398':>7}  {avg_ll - 0.6434:>+7.4f}")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = ens_test[t]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet 피처 v2 집계 중 ===")
    parquet_feat = build_parquet_features()
    print()

    print("=== MLP + HGB + ET 3모델 앙상블 ===\n")
    result = train_and_predict(train, sample, parquet_feat)

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== MLP+HGB+ET 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "mlp_hgb_et_ensemble_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
