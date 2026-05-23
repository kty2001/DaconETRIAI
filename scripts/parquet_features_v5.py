"""
parquet 피처 v5: v2 + wLight + mBle + mWifi
- v2  : wPedo/mActivity/mScreenStatus/wHr/mLight/mAmbience/mUsageStats
- wLight: 손목 조도 (수면 중 광 노출, presleep 블루라이트)
- mBle  : 블루투스 스캔 (장소 다양성, 기기 수)
- mWifi : WiFi 스캔 (장소 다양성 entropy, AP 수, 홈 비율)
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from parquet_features_v2 import build_all as build_v2
from wlight_features import build_wlight
from mble_features import build_mble
from mwifi_features import build_mwifi


def build_all() -> pd.DataFrame:
    builders = [
        ("wPedo/mActivity/mScreenStatus/wHr/mLight/mAmbience/mUsageStats (v2)", build_v2),
        ("wLight", build_wlight),
        ("mBle",   build_mble),
        ("mWifi",  build_mwifi),
    ]
    feat = None
    for name, fn in builders:
        print(f"  {name} 집계 중...")
        df = fn()
        if feat is None:
            feat = df
        else:
            feat = feat.merge(df, on=["subject_id", "date"], how="outer")

    drop_cols = [c for c in feat.columns if c.startswith("level_")]
    feat = feat.drop(columns=drop_cols, errors="ignore")

    print(f"  완료: {len(feat)}행, {len(feat.columns)}컬럼")
    return feat


if __name__ == "__main__":
    print("=== parquet 피처 v5 집계 ===")
    feat = build_all()
    parquet_cols = [c for c in feat.columns if c not in ("subject_id", "date")]
    print(f"\n총 {len(parquet_cols)}개 피처")
    print(parquet_cols)
