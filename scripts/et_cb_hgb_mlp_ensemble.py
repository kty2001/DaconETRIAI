"""
ET + CatBoost + HGB + MLP 4-model ensemble (GPS Slim 85%)
- 4개 모델 OOF 확률 기반 타깃별 최적 가중치 (Nelder-Mead)
- 입력: extratrees_gps_slim85_prob.csv, catboost_gps_slim85_prob.csv,
        hgb_gps_slim85_prob.csv, mlp_gps_slim85_prob.csv
- 출력: submission/et_cb_hgb_mlp_ensemble_prob.csv (clip 0.1~0.9)
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA = ROOT / "data"
SUBMISSION_DIR = ROOT / "submission"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

MODEL_FILES = [
    ("ET",       "extratrees_gps_slim85_prob.csv"),
    ("CatBoost", "catboost_gps_slim85_prob.csv"),
    ("HGB",      "hgb_gps_slim85_prob.csv"),
    ("MLP",      "mlp_gps_slim85_prob.csv"),
]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def load_probs():
    dfs = {}
    for name, fname in MODEL_FILES:
        path = SUBMISSION_DIR / fname
        if not path.exists():
            print(f"ERROR: {path} 없음")
            return None
        dfs[name] = pd.read_csv(path)
        print(f"  {name}: {fname} ({len(dfs[name])}행)")
    return dfs


def equal_weight_ensemble(dfs):
    n = len(MODEL_FILES)
    result = dfs["ET"][["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = sum(dfs[name][t] for name, _ in MODEL_FILES) / n
    return result


def main():
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")

    print("=== 확률 파일 로드 ===")
    dfs = load_probs()
    if dfs is None:
        return

    print("\n=== 각 모델 예측 분포 ===")
    for name, _ in MODEL_FILES:
        means = [f"{t}={dfs[name][t].mean():.3f}" for t in TARGETS]
        print(f"  {name}: " + " ".join(means))

    print("\n=== 균등 가중치 앙상블 ===")
    result = equal_weight_ensemble(dfs)

    print("\n=== 앙상블 예측 분포 (clip 전) ===")
    for t in TARGETS:
        print(f"  {t}: min={result[t].min():.3f}, "
              f"mean={result[t].mean():.3f}, "
              f"max={result[t].max():.3f}")

    result_clip = result.copy()
    for t in TARGETS:
        result_clip[t] = result[t].clip(0.1, 0.9)

    out_path = SUBMISSION_DIR / "et_cb_hgb_mlp_ensemble_prob.csv"
    result_clip.to_csv(out_path, index=False)
    print(f"\n저장 완료: {out_path}")

    print("\n=== 제출 파일 (binary threshold 0.5) ===")
    submission = result_clip[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        submission[t] = (result_clip[t] >= 0.5).astype(int)

    sub_path = SUBMISSION_DIR / "et_cb_hgb_mlp_ensemble_submission.csv"
    submission.to_csv(sub_path, index=False)
    print(f"바이너리 제출 파일: {sub_path}")

    print("\n=== 예측값 분포 (타깃별 양성 비율) ===")
    train_ratio = {t: train[t].mean() for t in TARGETS}
    for t in TARGETS:
        pred_ratio = submission[t].mean()
        print(f"  {t}: train={train_ratio[t]:.3f}  pred={pred_ratio:.3f}")


if __name__ == "__main__":
    main()
