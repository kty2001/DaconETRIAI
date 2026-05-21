"""
GRU 기반 수면 지표 예측 앙상블
- 과거 W일 (sensor features + labels) 시퀀스 -> 오늘 7개 타깃 예측
- LOSO: 9개 피험자 시퀀스로 학습, 1개로 검증
- 멀티태스크: 7개 타깃 동시 예측 (BCEWithLogitsLoss)
- 멀티 시드 앙상블
출력: submission/gru_ensemble_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features
from gps_features import build_gps

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"
SUBMISSION_DIR.mkdir(exist_ok=True)

TARGETS     = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
SEEDS       = [42, 123, 456, 789, 1024, 2024, 3141, 5678, 9999, 31415]
WINDOW      = 14    # 과거 14일 시퀀스
HIDDEN_SIZE = 32
N_LAYERS    = 1
DROPOUT     = 0.4
LR          = 0.001
WEIGHT_DECAY = 0.01
EPOCHS      = 300
PATIENCE    = 30
BATCH_SIZE  = 64

ET_LL_GPS_SLIM = {
    "Q1": 0.6976, "Q2": 0.6410, "Q3": 0.6430,
    "S1": 0.6075, "S2": 0.6079, "S3": 0.6015, "S4": 0.6856,
}

DROP_USAGE = [
    "usage_ms_morning", "usage_ms_afternoon", "usage_ms_evening",
    "usage_ms_presleep", "usage_ms_sleep", "usage_ms_total",
    "usage_apps_morning", "usage_apps_afternoon", "usage_apps_evening",
    "usage_apps_presleep", "usage_apps_sleep",
    "usage_presleep_ratio", "usage_sleep_ratio",
]


# ── 모델 정의 ─────────────────────────────────────────────────────────────────
class SleepGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout, n_targets):
        super().__init__()
        self.gru = nn.GRU(
            input_size, hidden_size, n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, n_targets)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)   # logits (BCEWithLogitsLoss 사용)


# ── 데이터셋 ──────────────────────────────────────────────────────────────────
class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.X = torch.tensor(sequences, dtype=torch.float32)
        self.y = torch.tensor(labels,    dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── 피처 준비 ─────────────────────────────────────────────────────────────────
def get_feature_cols(df, parquet_feat, label_feat):
    """센서 피처 컬럼 목록 반환 (DROP_USAGE, 타깃 제외)"""
    parquet_cols = [c for c in parquet_feat.columns if c not in ("subject_id", "date")]
    label_cols   = [c for c in label_feat.columns   if c not in ("subject_id", "date")]
    sensor_cols  = [c for c in parquet_cols + label_cols if c not in DROP_USAGE]
    return sensor_cols


def build_subject_sequence(subject_df, feature_cols, col_medians, col_stds):
    """단일 피험자 시퀀스 행렬 반환 (NaN 채우기 + z-score)"""
    mat = subject_df[feature_cols].values.astype(np.float32)
    # NaN → 전체 median
    for j, col in enumerate(feature_cols):
        nan_mask = np.isnan(mat[:, j])
        if nan_mask.any():
            mat[nan_mask, j] = col_medians[j]
    # 피험자 z-score
    subj_mean = np.nanmean(mat, axis=0)
    subj_std  = np.nanstd(mat, axis=0)
    valid = subj_std > 0
    mat[:, valid] = (mat[:, valid] - subj_mean[valid]) / subj_std[valid]
    return mat


def build_sequences(subjects_data, feature_cols, target_cols, col_medians, col_stds,
                    window=14):
    """슬라이딩 윈도우 시퀀스 생성
    입력: 과거 window일의 [features + targets]
    출력: 오늘의 targets
    """
    seqs, labs = [], []
    n_feat = len(feature_cols)
    n_tgt  = len(target_cols)

    for sid, sdf in subjects_data.items():
        sdf = sdf.sort_values("lifelog_date").reset_index(drop=True)
        feat_mat = build_subject_sequence(sdf, feature_cols, col_medians, col_stds)
        tgt_mat  = sdf[target_cols].values.astype(np.float32)

        for i in range(window, len(sdf)):
            # [features, past_targets] for days i-window to i-1, then features for day i
            seq = []
            for j in range(window):
                day_feat = feat_mat[i - window + j]          # (n_feat,)
                day_tgt  = tgt_mat[i - window + j]           # (n_tgt,)
                seq.append(np.concatenate([day_feat, day_tgt]))
            seqs.append(np.array(seq, dtype=np.float32))    # (window, n_feat+n_tgt)
            labs.append(tgt_mat[i])                          # (n_tgt,)

    return np.array(seqs), np.array(labs)


def build_test_sequences(subject_df, feature_cols, target_cols, col_medians, col_stds,
                         window=14):
    """테스트 피험자: 각 날짜에 대해 과거 window일 시퀀스 생성"""
    sdf = subject_df.sort_values("lifelog_date").reset_index(drop=True)
    feat_mat = build_subject_sequence(sdf, feature_cols, col_medians, col_stds)
    tgt_mat  = sdf[target_cols].values.astype(np.float32)

    seqs = []
    for i in range(window, len(sdf)):
        seq = []
        for j in range(window):
            day_feat = feat_mat[i - window + j]
            day_tgt  = tgt_mat[i - window + j]
            seq.append(np.concatenate([day_feat, day_tgt]))
        seqs.append(np.array(seq, dtype=np.float32))
    return np.array(seqs)


# ── 학습/평가 ─────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(loader.dataset)


def eval_model(model, X_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        probs  = torch.sigmoid(logits).numpy()
    return probs


# ── 메인 학습 흐름 ────────────────────────────────────────────────────────────
def run_loso(train_df, test_df, parquet_feat, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 전체 피처 준비
    label_feat_all = build_label_features(train_df, train_df)
    feature_cols   = get_feature_cols(train_df, parquet_feat, label_feat_all)

    # 전체 데이터 merge
    full_df = train_df.merge(
        parquet_feat, left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"], how="left"
    ).drop(columns=["date"], errors="ignore")
    full_df = full_df.merge(
        label_feat_all, left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"], how="left"
    ).drop(columns=["date"], errors="ignore")

    # 전체 column median (NaN 대체용)
    avail_cols  = [c for c in feature_cols if c in full_df.columns]
    col_medians = np.nanmedian(full_df[avail_cols].values.astype(float), axis=0)
    col_stds    = np.nanstd(full_df[avail_cols].values.astype(float), axis=0)
    feature_cols = avail_cols

    subjects = sorted(train_df["subject_id"].unique())
    cv = GroupKFold(n_splits=len(subjects))
    groups = train_df["subject_id"].values

    oof_probs = np.full((len(train_df), len(TARGETS)), np.nan)

    for fold_idx, (tr_idx, val_idx) in enumerate(
            cv.split(np.zeros(len(train_df)), train_df[TARGETS[0]].values, groups)):
        val_sid = train_df.iloc[val_idx]["subject_id"].iloc[0]

        # 훈련/검증 피험자 데이터
        tr_subjects = {}
        for sid in subjects:
            if sid != val_sid:
                sdf = full_df[full_df["subject_id"] == sid]
                if len(sdf) > WINDOW:
                    tr_subjects[sid] = sdf

        val_df_fold = full_df[full_df["subject_id"] == val_sid]

        # 시퀀스 생성
        X_tr, y_tr = build_sequences(
            tr_subjects, feature_cols, TARGETS, col_medians, col_stds, WINDOW
        )
        if len(X_tr) == 0:
            continue

        X_val_seq = build_test_sequences(
            val_df_fold, feature_cols, TARGETS, col_medians, col_stds, WINDOW
        )
        y_val_true = val_df_fold.sort_values("lifelog_date")[TARGETS].values[WINDOW:]

        input_size = X_tr.shape[2]
        model = SleepGRU(input_size, HIDDEN_SIZE, N_LAYERS, DROPOUT, len(TARGETS))
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, min_lr=1e-5
        )
        criterion = nn.BCEWithLogitsLoss()

        loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        X_val_t = torch.tensor(X_val_seq, dtype=torch.float32)

        best_val_loss = float("inf")
        patience_cnt  = 0

        for epoch in range(EPOCHS):
            train_loss = train_epoch(model, loader, optimizer, criterion)
            val_probs  = eval_model(model, X_val_t)
            val_loss   = np.mean([
                log_loss(y_val_true[:, i], val_probs[:, i].clip(1e-6, 1-1e-6))
                for i in range(len(TARGETS))
            ])
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-5:
                best_val_loss = val_loss
                best_probs    = val_probs.copy()
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE:
                    break

        # OOF 저장 (window 이후 날짜만)
        val_dates = val_df_fold.sort_values("lifelog_date")["lifelog_date"].values
        val_dates_window = val_dates[WINDOW:]
        for day_i, date in enumerate(val_dates_window):
            row_mask = (train_df["subject_id"] == val_sid) & (train_df["lifelog_date"] == date)
            row_indices = np.where(row_mask.values)[0]
            if len(row_indices) > 0:
                oof_probs[row_indices[0]] = best_probs[day_i]

    return oof_probs


def predict_test(train_df, test_df, parquet_feat, seed):
    """전체 훈련 데이터로 모델 학습 후 테스트 예측"""
    torch.manual_seed(seed)
    np.random.seed(seed)

    label_feat_all  = build_label_features(train_df, train_df)
    label_feat_test = build_label_features(train_df, test_df)
    feature_cols    = get_feature_cols(train_df, parquet_feat, label_feat_all)

    full_df = train_df.merge(
        parquet_feat, left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"], how="left"
    ).drop(columns=["date"], errors="ignore")
    full_df = full_df.merge(
        label_feat_all, left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"], how="left"
    ).drop(columns=["date"], errors="ignore")

    avail_cols  = [c for c in feature_cols if c in full_df.columns]
    col_medians = np.nanmedian(full_df[avail_cols].values.astype(float), axis=0)
    col_stds    = np.nanstd(full_df[avail_cols].values.astype(float), axis=0)
    feature_cols = avail_cols

    # 테스트 데이터 merge
    test_full = test_df.merge(
        parquet_feat, left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"], how="left"
    ).drop(columns=["date"], errors="ignore")
    test_full = test_full.merge(
        label_feat_test, left_on=["subject_id", "lifelog_date"],
        right_on=["subject_id", "date"], how="left"
    ).drop(columns=["date"], errors="ignore")

    subjects = sorted(train_df["subject_id"].unique())
    tr_subjects = {}
    for sid in subjects:
        sdf = full_df[full_df["subject_id"] == sid]
        if len(sdf) > WINDOW:
            tr_subjects[sid] = sdf

    X_tr, y_tr = build_sequences(
        tr_subjects, feature_cols, TARGETS, col_medians, col_stds, WINDOW
    )

    input_size = X_tr.shape[2]
    model = SleepGRU(input_size, HIDDEN_SIZE, N_LAYERS, DROPOUT, len(TARGETS))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    # 검증 없이 고정 epoch 학습
    for epoch in range(150):
        train_epoch(model, loader, optimizer, criterion)

    # 피험자별 테스트 예측
    test_preds = []
    for _, row in test_df.iterrows():
        sid  = row["subject_id"]
        date = row["lifelog_date"]

        # 해당 피험자의 과거 데이터
        hist = full_df[full_df["subject_id"] == sid].sort_values("lifelog_date")
        hist = hist[hist["lifelog_date"] < date]

        if len(hist) < WINDOW:
            # 히스토리 부족 시 0.5 반환
            test_preds.append([0.5] * len(TARGETS))
            continue

        seq = []
        hist_tail = hist.tail(WINDOW)
        feat_mat = build_subject_sequence(hist_tail, feature_cols, col_medians, col_stds)
        tgt_mat  = hist_tail[TARGETS].values.astype(np.float32)
        for j in range(WINDOW):
            seq.append(np.concatenate([feat_mat[j], tgt_mat[j]]))
        seq_tensor = torch.tensor([seq], dtype=torch.float32)

        model.eval()
        with torch.no_grad():
            logits = model(seq_tensor)
            probs  = torch.sigmoid(logits).numpy()[0]
        test_preds.append(probs.tolist())

    return np.array(test_preds)


def main():
    train  = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    sample = pd.read_csv(DATA / "ch2026_submission_sample.csv")

    print("=== parquet v2 + GPS 피처 집계 중 ===")
    parquet_feat = build_parquet_features()
    gps_feat     = build_gps()
    parquet_feat = parquet_feat.merge(gps_feat, on=["subject_id", "date"], how="left")
    print()

    n_seeds = len(SEEDS)
    print(f"=== GRU 앙상블 (window={WINDOW}, hidden={HIDDEN_SIZE}, {n_seeds} seeds) ===")
    print(f"    dropout={DROPOUT}, lr={LR}, weight_decay={WEIGHT_DECAY}")
    print()

    oof_sum  = np.zeros((len(train),  len(TARGETS)))
    test_sum = np.zeros((len(sample), len(TARGETS)))
    oof_cnt  = np.zeros(len(train))

    for seed_i, seed in enumerate(SEEDS):
        print(f"  seed {seed} ({seed_i+1}/{n_seeds}) LOSO ...")
        oof_probs  = run_loso(train, sample, parquet_feat, seed)
        test_probs = predict_test(train, sample, parquet_feat, seed)

        valid_mask = ~np.isnan(oof_probs[:, 0])
        oof_sum[valid_mask] += oof_probs[valid_mask]
        oof_cnt[valid_mask] += 1
        test_sum += test_probs

        # 중간 OOF 출력
        avg_oof = np.where(oof_cnt[:, None] > 0,
                           oof_sum / oof_cnt[:, None], np.nan)
        valid = ~np.isnan(avg_oof[:, 0])
        lls = [log_loss(train[t].values[valid], avg_oof[valid, i].clip(1e-6, 1-1e-6))
               for i, t in enumerate(TARGETS)]
        print(f"    OOF LL = {np.mean(lls):.4f}  "
              + "  ".join(f"{t}:{ll:.4f}" for t, ll in zip(TARGETS, lls)))

    # 최종 OOF
    oof_final = oof_sum / np.maximum(oof_cnt[:, None], 1)
    valid = ~np.isnan(oof_final[:, 0])

    print()
    print(f"{'타깃':<5}  {'F1':>6}  {'OOF LL':>8}  {'slim85':>8}  {'개선':>7}")
    print("-" * 44)
    lls_final = []
    for i, t in enumerate(TARGETS):
        mask = valid & train[t].notna()
        f1 = f1_score(train[t].values[mask], (oof_final[mask, i] > 0.5).astype(int))
        ll = log_loss(train[t].values[mask], oof_final[mask, i].clip(1e-6, 1-1e-6))
        lls_final.append(ll)
        ref  = ET_LL_GPS_SLIM[t]
        diff = ref - ll
        sign = "[+]" if diff > 0 else "[-]"
        print(f"{t:<5}  {f1:>6.3f}  {ll:>8.4f}  {ref:>8.4f}  {diff:>+6.4f} {sign}")
    avg_ll  = np.mean(lls_final)
    avg_ref = np.mean(list(ET_LL_GPS_SLIM.values()))
    print(f"{'평균':<5}  {'':>6}  {avg_ll:>8.4f}  {avg_ref:>8.4f}  {avg_ref-avg_ll:>+6.4f}")

    # 테스트 예측 저장
    test_avg = test_sum / n_seeds
    result_prob = sample[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for i, t in enumerate(TARGETS):
        result_prob[t] = test_avg[:, i].clip(0.1, 0.9)

    print("\n=== 예측 분포 (clip 0.1~0.9) ===")
    for t in TARGETS:
        print(f"  {t}: min={result_prob[t].min():.3f}, "
              f"mean={result_prob[t].mean():.3f}, "
              f"max={result_prob[t].max():.3f}")

    out_path = SUBMISSION_DIR / "gru_ensemble_prob.csv"
    result_prob.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")


if __name__ == "__main__":
    main()
