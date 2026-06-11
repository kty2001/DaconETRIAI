"""
ET + LGB + CB 3-way 앙상블

입력:
  et_gps_slim80_pers_grid_best_prob.csv  (ET, public 0.5989)
  lgb_gps_slim80_personal_blend_prob.csv (LGB, public 0.6079)
  cb_gps_slim80_personal_blend_prob.csv  (CB)

출력:
  submission/et_lgb_cb_ensemble_prob.csv      (equal weight 1:1:1)
  submission/et_lgb_cb_w2_ensemble_prob.csv   (ET-heavy 2:1:1)
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
SUBMISSION_DIR = ROOT / "submission"

TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]


def load(name):
    path = SUBMISSION_DIR / f"{name}.csv"
    if not path.exists():
        print(f"ERROR: {path} not found")
        return None
    return pd.read_csv(path)


def main():
    et  = load("et_gps_slim80_pers_grid_best_prob")
    lgb = load("lgb_gps_slim80_personal_blend_prob")
    cb  = load("cb_gps_slim80_personal_blend_prob")

    if et is None or lgb is None or cb is None:
        return

    key_cols = ["subject_id", "sleep_date"]
    if not (et[key_cols].equals(lgb[key_cols]) and et[key_cols].equals(cb[key_cols])):
        print("ERROR: row order mismatch")
        return

    base = et[["subject_id", "sleep_date", "lifelog_date"]].copy()

    # 1:1:1 equal weight
    result_eq = base.copy()
    for t in TARGETS:
        result_eq[t] = ((et[t] + lgb[t] + cb[t]) / 3).clip(0.1, 0.9)

    # 2:1:1 ET-heavy
    result_w2 = base.copy()
    for t in TARGETS:
        result_w2[t] = ((2 * et[t] + lgb[t] + cb[t]) / 4).clip(0.1, 0.9)

    print("=== ET + LGB + CB ensemble ===")
    for t in TARGETS:
        print(f"  {t}: ET={et[t].mean():.3f}, LGB={lgb[t].mean():.3f}, "
              f"CB={cb[t].mean():.3f}, eq={result_eq[t].mean():.3f}, w2={result_w2[t].mean():.3f}")

    out1 = SUBMISSION_DIR / "et_lgb_cb_ensemble_prob.csv"
    out2 = SUBMISSION_DIR / "et_lgb_cb_w2_ensemble_prob.csv"
    result_eq.to_csv(out1, index=False)
    result_w2.to_csv(out2, index=False)
    print(f"\nsaved: {out1}")
    print(f"saved: {out2}")
    print(f"rows: {len(base)}")


if __name__ == "__main__":
    main()
