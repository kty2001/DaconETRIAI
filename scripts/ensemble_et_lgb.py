"""
ET + LGB 앙상블
pers_grid_best (ET) + lgb_gps_slim80_personal_blend (LGB) 확률 평균

출력:
  submission/et_lgb_ensemble_prob.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUBMISSION_DIR = ROOT / "submission"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def main():
    et_path  = SUBMISSION_DIR / "et_gps_slim80_pers_grid_best_prob.csv"
    lgb_path = SUBMISSION_DIR / "lgb_gps_slim80_personal_blend_prob.csv"

    if not et_path.exists():
        print(f"ERROR: {et_path} not found")
        return
    if not lgb_path.exists():
        print(f"ERROR: {lgb_path} not found")
        return

    et  = pd.read_csv(et_path)
    lgb = pd.read_csv(lgb_path)

    if not (et[["subject_id", "sleep_date"]].equals(lgb[["subject_id", "sleep_date"]])):
        print("ERROR: row order mismatch between files")
        return

    result = et[["subject_id", "sleep_date", "lifelog_date"]].copy()
    for t in TARGETS:
        result[t] = ((et[t] + lgb[t]) / 2).clip(0.1, 0.9)

    print("=== ET + LGB ensemble ===")
    for t in TARGETS:
        print(f"  {t}: ET={et[t].mean():.3f}, LGB={lgb[t].mean():.3f}, "
              f"ensemble={result[t].mean():.3f}")

    out_path = SUBMISSION_DIR / "et_lgb_ensemble_prob.csv"
    result.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}")
    print(f"rows: {len(result)}, NaN: {result[TARGETS].isnull().any().any()}")


if __name__ == "__main__":
    main()
