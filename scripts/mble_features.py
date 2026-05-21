"""
mBle 피처 빌더 (ch2025_mBle.parquet)
m_ble: 블루투스 스캔 결과 배열 [{address, device_class, rssi}, ...]

피처 구성:
  일별 요약  : ble_devices_per_scan_mean/std, ble_n_unique_daily,
               ble_close_ratio (RSSI > -60), ble_medium_ratio (RSSI > -70)
  시간대별   : ble_unique_{zone}, ble_rssi_mean_{zone}, ble_scans_{zone}
               (morning, afternoon, evening, presleep, sleep)
  디바이스클래스: ble_class0_ratio (미분류), ble_class_known_ratio (분류됨)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"

ZONES = ["morning", "afternoon", "evening", "presleep", "sleep"]
CLOSE_RSSI  = -60   # 매우 근접
MEDIUM_RSSI = -70   # 중간 거리


def _zone(hour: pd.Series) -> pd.Series:
    conditions = [
        (hour >= 6)  & (hour < 12),
        (hour >= 12) & (hour < 18),
        (hour >= 18) & (hour < 22),
        (hour >= 22),
        (hour < 6),
    ]
    return np.select(conditions, ZONES, default="sleep")


def _parse_scan(arr):
    """한 스캔 배열 -> (addresses, rssis) 리스트"""
    if arr is None or len(arr) == 0:
        return [], [], []
    addrs, rssis, classes = [], [], []
    for item in arr:
        addrs.append(str(item.get("address", "")))
        rssis.append(item.get("rssi", np.nan))
        classes.append(str(item.get("device_class", "0")))
    return addrs, rssis, classes


def build_mble() -> pd.DataFrame:
    """
    반환: (subject_id, date) 기준 mBle 피처 DataFrame
          date = lifelog_date (parquet_features_v2와 동일 키)
    """
    df = pd.read_parquet(PARQUET_DIR / "ch2025_mBle.parquet")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour
    df["zone"] = _zone(df["hour"])

    # 스캔별 파싱
    records = []
    for _, row in df.iterrows():
        addrs, rssis, classes = _parse_scan(row["m_ble"])
        if len(addrs) == 0:
            records.append({
                "subject_id": row["subject_id"],
                "date": row["date"],
                "zone": row["zone"],
                "n_devices": 0,
                "addrs": [],
                "rssis": [],
                "classes": [],
            })
        else:
            records.append({
                "subject_id": row["subject_id"],
                "date": row["date"],
                "zone": row["zone"],
                "n_devices": len(addrs),
                "addrs": addrs,
                "rssis": rssis,
                "classes": classes,
            })

    rows = []
    grouped = pd.DataFrame({"subject_id": df["subject_id"],
                             "date": df["date"],
                             "zone": df["zone"]}).copy()
    grouped["_idx"] = range(len(grouped))

    for (sid, date), grp_idx in grouped.groupby(["subject_id", "date"])["_idx"]:
        recs = [records[i] for i in grp_idx]
        feat = {"subject_id": sid, "date": date}

        # 전체 일별
        all_addrs  = [a for r in recs for a in r["addrs"]]
        all_rssis  = [v for r in recs for v in r["rssis"] if not np.isnan(v)]
        all_classes = [c for r in recs for c in r["classes"]]
        n_scans = len(recs)

        n_per_scan = [r["n_devices"] for r in recs]
        feat["ble_devices_per_scan_mean"] = float(np.mean(n_per_scan))  if n_per_scan else np.nan
        feat["ble_devices_per_scan_std"]  = float(np.std(n_per_scan))   if len(n_per_scan) > 1 else np.nan
        feat["ble_n_unique_daily"]        = len(set(all_addrs))
        feat["ble_scans_daily"]           = n_scans

        if all_rssis:
            rssi_arr = np.array(all_rssis)
            feat["ble_close_ratio"]  = float((rssi_arr > CLOSE_RSSI).mean())
            feat["ble_medium_ratio"] = float((rssi_arr > MEDIUM_RSSI).mean())
            feat["ble_rssi_mean"]    = float(rssi_arr.mean())
            feat["ble_rssi_std"]     = float(rssi_arr.std())
        else:
            feat["ble_close_ratio"]  = np.nan
            feat["ble_medium_ratio"] = np.nan
            feat["ble_rssi_mean"]    = np.nan
            feat["ble_rssi_std"]     = np.nan

        if all_classes:
            feat["ble_class0_ratio"]     = all_classes.count("0") / len(all_classes)
            feat["ble_class_known_ratio"] = 1.0 - feat["ble_class0_ratio"]
        else:
            feat["ble_class0_ratio"]     = np.nan
            feat["ble_class_known_ratio"] = np.nan

        # 시간대별
        for zone in ZONES:
            z_recs = [r for r in recs if r["zone"] == zone]
            z_addrs = [a for r in z_recs for a in r["addrs"]]
            z_rssis = [v for r in z_recs for v in r["rssis"] if not np.isnan(v)]

            feat[f"ble_unique_{zone}"] = len(set(z_addrs)) if z_recs else np.nan
            feat[f"ble_scans_{zone}"]  = len(z_recs)
            if z_rssis:
                feat[f"ble_rssi_mean_{zone}"] = float(np.mean(z_rssis))
            else:
                feat[f"ble_rssi_mean_{zone}"] = np.nan

        rows.append(feat)

    result = pd.DataFrame(rows)
    print(f"  mBle 피처 완료: {result.shape[1]-2}개 피처 x {len(result)}행")
    return result


if __name__ == "__main__":
    feat = build_mble()
    print(feat.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    nan_rates = feat.drop(columns=["subject_id", "date"]).isna().mean()
    print("\nNaN 비율:")
    print(nan_rates.round(3).to_string())
