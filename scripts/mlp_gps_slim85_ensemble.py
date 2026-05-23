"""
MLP GPS Slim 85% ensemble
- parquet v2 + GPS features, feature importance top 85% coverage only
- Multi-task MLP: shared backbone + 7 independent output heads
- subj_mean 전체 제거 (트리 모델과의 다양성 확보)
- Phase 0: importance (ExtraTrees with extratrees_gps params)
- Phase 1: Optuna 30 trials (mlp_gps_slim85)
- Phase 2: 10 seeds ensemble
output: submission/mlp_gps_slim85_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import optuna
import random
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from functools import reduce

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from gps_features import build_gps
from optuna_params_io import load_params, save_params

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

ET_KEY_GPS   = "extratrees_gps"
MLP_KEY      = "mlp_gps_slim85"
N_TRIALS     = 30
IMP_COVERAGE = 0.85

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS   = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]
DROP_GLOBAL = DROP_USAGE + [f"subj_mean_{t}" for t in TARGETS]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIMS_MAP = {
    "256-128-64":     [256, 128, 64],
    "512-256-128":    [512, 256, 128],
    "256-128":        [256, 128],
    "128-64-32":      [128, 64, 32],
    "512-256-128-64": [512, 256, 128, 64],
}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(prev, 1) for _ in TARGETS])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.backbone(x)
        return torch.cat([h(z) for h in self.heads], dim=1)


def train_model(model: nn.Module, X: np.ndarray, Y: np.ndarray,
                params: dict, seed: int) -> nn.Module:
    set_seed(seed)
    criterion  = nn.BCEWithLogitsLoss()
    optimizer  = torch.optim.Adam(model.parameters(),
                                  lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params["n_epochs"])
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32).to(DEVICE),
                      torch.tensor(Y, dtype=torch.float32).to(DEVICE)),
        batch_size=params["batch_size"], shuffle=True, drop_last=True,
    )
    model.train()
    for _ in range(params["n_epochs"]):
        for xb, yb in loader:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()
    return model


def predict_proba(model: nn.Module, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32).to(DEVICE))
        return torch.sigmoid(logits).cpu().numpy()


def get_sensor_cols(parquet_feat):
    return [c for c in parquet_feat.columns if c not in ("subject_id", "date")]


def compute_subj_stats(subject_ids, X, sensor_cols):
    avail = [c for c in sensor_cols if c in X.columns]
    tmp = X[avail].copy()
    tmp["subject_id"] = subject_ids.values
    return tmp.groupby("subject_id")[avail].agg(["mean", "std"])


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


def compute_importance(train, parquet_feat, le, sensor_cols, fold_label_feats, best_gps):
    importance_dict   = {t: [] for t in TARGETS}
    feature_names_ref = {}

    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr   = build_features(train_fold, train_fold, parquet_feat, lf_tr, True, le)
        sid_tr = X_tr["subject_id"].reset_index(drop=True)
        tr_stats = compute_subj_stats(sid_tr, X_tr, sensor_cols)
        X_tr_z   = apply_zscore(sid_tr, X_tr.drop(columns=["subject_id"]), tr_stats, sensor_cols)

        for t in TARGETS:
            drop_cols = [f"subj_mean_{t}"] + DROP_USAGE
            X_tr_t = X_tr_z.drop(columns=drop_cols, errors="ignore")
            if t not in feature_names_ref:
                feature_names_ref[t] = X_tr_t.columns.tolist()
            tr_median   = X_tr_t.median()
            X_tr_filled = X_tr_t.fillna(tr_median)
            y = train[t].values
            model = ExtraTreesClassifier(**{**best_gps[t], "random_state": 42})
            model.fit(X_tr_filled, y[tr_idx])
            importance_dict[t].append(model.feature_importances_)

    all_imp = {t: pd.Series(np.mean(importance_dict[t], axis=0),
                            index=feature_names_ref[t])
               for t in TARGETS}
    common_feats = list(reduce(lambda a, b: a & b,
                               [set(all_imp[t].index) for t in TARGETS]))
    combined = pd.DataFrame({t: all_imp[t][common_feats] for t in TARGETS})
    combined["mean_imp"] = combined.mean(axis=1)
    combined = combined.sort_values("mean_imp", ascending=False)
    total = combined["mean_imp"].sum()
    combined["cumsum"] = combined["mean_imp"].cumsum() / total
    keep = combined[combined["cumsum"] <= IMP_COVERAGE].index.tolist()

    print(f"  전체 공통 피처: {len(combined)}개")
    print(f"  상위 {IMP_COVERAGE*100:.0f}% 커버: {len(keep)}개 유지 / {len(combined)-len(keep)}개 제거")
    return set(keep)


def precompute_fold_features(train, test, parquet_feat, le, sensor_cols,
                             fold_label_feats, label_feat_test, full_stats, keep_feats):
    fold_base = []
    for tr_idx, val_idx, train_fold, val_fold, lf_tr, lf_val in fold_label_feats:
        X_tr  = build_features(train_fold, train_fold, parquet_feat, lf_tr,  True,  le)
        X_val = build_features(val_fold,   train_fold, parquet_feat, lf_val, False, le)
        sid_tr  = X_tr["subject_id"].reset_index(drop=True)
        sid_val = X_val["subject_id"].reset_index(drop=True)
        tr_stats  = compute_subj_stats(sid_tr,  X_tr,  sensor_cols)
        val_stats = compute_subj_stats(sid_val, X_val, sensor_cols)
        X_tr_z  = apply_zscore(sid_tr,  X_tr.drop(columns=["subject_id"]),  tr_stats,  sensor_cols)
        X_val_z = apply_zscore(sid_val, X_val.drop(columns=["subject_id"]), val_stats, sensor_cols)
        fold_base.append((tr_idx, val_idx, X_tr_z, X_val_z))

    X_te_raw = build_features(test.copy(), train, parquet_feat, label_feat_test, False, le)
    sid_te   = X_te_raw["subject_id"].reset_index(drop=True)
    X_te_z   = apply_zscore(sid_te, X_te_raw.drop(columns=["subject_id"]), full_stats, sensor_cols)

    first_tr_cols = fold_base[0][2].drop(columns=DROP_GLOBAL, errors="ignore").columns
    slim_cols = [c for c in first_tr_cols if c in keep_feats]

    fold_prepped = []
    for tr_idx, val_idx, X_tr_z, X_val_z in fold_base:
        X_tr_t = X_tr_z.drop(columns=DROP_GLOBAL, errors="ignore")
        X_val_t = X_val_z.drop(columns=DROP_GLOBAL, errors="ignore")
        X_te_t  = X_te_z.drop(columns=DROP_GLOBAL, errors="ignore")

        X_tr_slim  = X_tr_t[[c for c in slim_cols if c in X_tr_t.columns]]
        X_val_slim = X_val_t[[c for c in slim_cols if c in X_val_t.columns]]
        X_te_slim  = X_te_t[[c for c in slim_cols if c in X_te_t.columns]]

        tr_med   = X_tr_slim.median()
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr_slim.fillna(tr_med).to_numpy())
        X_val_sc = scaler.transform(X_val_slim.fillna(tr_med).to_numpy())
        X_te_sc  = scaler.transform(X_te_slim.fillna(tr_med).to_numpy())

        fold_prepped.append((tr_idx, val_idx, X_tr_sc, X_val_sc, X_te_sc))

    input_dim = fold_prepped[0][2].shape[1]
    return fold_prepped, input_dim


def make_objective(fold_prepped, Y_all, input_dim):
    def objective(trial):
        hd_key = trial.suggest_categorical("hidden_dims_key", list(HIDDEN_DIMS_MAP.keys()))
        params = {
            "hidden_dims":  HIDDEN_DIMS_MAP[hd_key],
            "hidden_key":   hd_key,
            "dropout":      trial.suggest_float("dropout",      0.1,  0.5),
            "lr":           trial.suggest_float("lr",           1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
            "n_epochs":     trial.suggest_int("n_epochs",       50,   300,  step=50),
            "batch_size":   trial.suggest_categorical("batch_size", [32, 64, 128]),
        }
        set_seed(42)
        oof = np.zeros((len(Y_all), 7))
        for tr_idx, val_idx, X_tr_sc, X_val_sc, _ in fold_prepped:
            model = MultiTaskMLP(input_dim, params["hidden_dims"], params["dropout"]).to(DEVICE)
            train_model(model, X_tr_sc, Y_all[tr_idx], params, seed=42)
            oof[val_idx] = predict_proba(model, X_val_sc)
        return float(np.mean([log_loss(Y_all[:, i], oof[:, i]) for i in range(7)]))
    return objective


def train_and_predict(train, test, parquet_feat):
    le          = LabelEncoder().fit(train["subject_id"])
    groups      = train["subject_id"].values
    cv          = GroupKFold(n_splits=10)
    sensor_cols = get_sensor_cols(parquet_feat)
    n_seeds     = len(SEEDS)

    fold_indices = list(cv.split(np.zeros(len(train)), train[TARGETS[0]].values, groups))

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

    lf_full    = build_label_features(train, train)
    X_full     = build_features(train, train, parquet_feat, lf_full, True, le)
    full_stats = compute_subj_stats(X_full["subject_id"], X_full, sensor_cols)

    gps_params = load_params(ET_KEY_GPS)
    if not gps_params:
        print("ERROR: extratrees_gps params 캐시 없음")
        return None
    best_gps = {t: gps_params[t] for t in TARGETS}

    print("=== Phase 0: Feature Importance 계산 ===")
    keep_feats = compute_importance(train, parquet_feat, le, sensor_cols,
                                    fold_label_feats, best_gps)
    print()

    print(f"폴드 피처 사전 계산 중 (slim {IMP_COVERAGE*100:.0f}%, MLP)...")
    fold_prepped, input_dim = precompute_fold_features(
        train, test, parquet_feat, le, sensor_cols,
        fold_label_feats, label_feat_test, full_stats, keep_feats,
    )
    print(f"  완료 (입력 차원: {input_dim}개, device: {DEVICE})\n")

    Y_all = train[TARGETS].values

    cached = load_params(MLP_KEY)
    if cached and "shared" in cached:
        print(f"=== Phase 1: 저장된 params 로드 ({MLP_KEY}) ===")
        best_params = cached["shared"]
        if "hidden_dims" not in best_params and "hidden_key" in best_params:
            best_params["hidden_dims"] = HIDDEN_DIMS_MAP[best_params["hidden_key"]]
        print(f"  hidden_dims={best_params['hidden_dims']}, dropout={best_params['dropout']:.3f}")
        print(f"  lr={best_params['lr']:.5f}, wd={best_params['weight_decay']:.5f}")
        print(f"  n_epochs={best_params['n_epochs']}, batch_size={best_params['batch_size']}\n")
    else:
        print(f"=== Phase 1: MLP Optuna GPS Slim85 ({N_TRIALS} trials) ===")
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(make_objective(fold_prepped, Y_all, input_dim),
                       n_trials=N_TRIALS, show_progress_bar=True)
        best_params = dict(study.best_params)
        best_params["hidden_dims"] = HIDDEN_DIMS_MAP[best_params["hidden_dims_key"]]
        print(f"\n  Best OOF avg_ll: {study.best_value:.4f}")
        print(f"  hidden_dims={best_params['hidden_dims']}, dropout={best_params['dropout']:.3f}")
        print(f"  lr={best_params['lr']:.5f}, wd={best_params['weight_decay']:.5f}")
        print(f"  n_epochs={best_params['n_epochs']}, batch_size={best_params['batch_size']}\n")
        save_params(MLP_KEY, {"shared": best_params})

    print(f"=== Phase 2: MLP GPS Slim85 앙상블 ({n_seeds} seeds) ===")
    print(f"{'시드':>6}  " + "  ".join(f"{t:>6}" for t in TARGETS) + "  평균F1   평균LL")
    print("-" * (8 + 9 * len(TARGETS) + 16))

    ens_oof  = np.zeros((len(train), 7))
    ens_test = np.zeros((len(test),  7))
    n_folds  = len(fold_prepped)

    for seed_i, seed in enumerate(SEEDS):
        seed_oof  = np.zeros((len(train), 7))
        seed_test = np.zeros((len(test),  7))

        for tr_idx, val_idx, X_tr_sc, X_val_sc, X_te_sc in fold_prepped:
            model = MultiTaskMLP(input_dim, best_params["hidden_dims"],
                                 best_params["dropout"]).to(DEVICE)
            train_model(model, X_tr_sc, Y_all[tr_idx], best_params, seed=seed)
            seed_oof[val_idx]  = predict_proba(model, X_val_sc)
            seed_test         += predict_proba(model, X_te_sc) / n_folds

        ens_oof  += seed_oof  / n_seeds
        ens_test += seed_test / n_seeds

        f1s = [f1_score(Y_all[:, i], (ens_oof[:, i] * n_seeds / (seed_i + 1) > 0.5).astype(int))
               for i in range(7)]
        lls = [log_loss(Y_all[:, i], ens_oof[:, i] * n_seeds / (seed_i + 1))
               for i in range(7)]
        f1_str = "  ".join(f"{f:>6.3f}" for f in f1s)
        print(f"{seed:>6}  {f1_str}  {np.mean(f1s):>6.3f}  {np.mean(lls):>6.4f}")

    print()
    print(f"{'타깃':<5}  {'F1':>6}  {'OOF LL':>8}")
    print("-" * 28)
    lls_final = []
    for i, t in enumerate(TARGETS):
        f1 = f1_score(Y_all[:, i], (ens_oof[:, i] > 0.5).astype(int))
        ll = log_loss(Y_all[:, i], ens_oof[:, i])
        lls_final.append(ll)
        print(f"{t:<5}  {f1:>6.3f}  {ll:>8.4f}")
    avg_ll = np.mean(lls_final)
    print(f"{'평균':<5}  {'':>6}  {avg_ll:>8.4f}")

    result = test[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for i, t in enumerate(TARGETS):
        result[t] = ens_test[:, i]
    return result


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    print(f"=== MLP GPS Slim {IMP_COVERAGE*100:.0f}% ===\n")
    result = train_and_predict(train, sample, parquet_feat)
    if result is None:
        return

    result_prob = result[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result_prob[t] = result[t].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "mlp_gps_slim85_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
