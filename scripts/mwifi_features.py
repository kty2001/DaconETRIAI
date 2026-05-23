"""
mWifi 피처 빌더 (ch2025_mWifi.parquet)
m_wifi: [{bssid, rssi}, ...] 배열

피처 구성:
  일별 요약  : wifi_n_unique_daily (고유 AP 수), wifi_entropy (다양성 엔트로피),
               wifi_scans_daily, wifi_rssi_mean/std
  시간대별   : wifi_unique_{zone}, wifi_scans_{zone}, wifi_rssi_mean_{zone}
               (morning, afternoon, evening, presleep, sleep)
  홈 네트워크: wifi_home_ratio (수면 시간대에 주로 등장하는 AP 접속 비율)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"

ZONES = ["morning", "afternoon", "evening", "presleep", "sleep"]


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
    if arr is None or len(arr) == 0:
        return [], []
    bssids, rssis = [], []
    for item in arr:
        bssids.append(str(item.get("bssid", "")))
        rssis.append(item.get("rssi", np.nan))
    return bssids, rssis


def build_mwifi() -> pd.DataFrame:
    """
    반환: (subject_id, date) 기준 mWifi 피처 DataFrame
          date = lifelog_date (parquet_features_v2와 동일 키)
    """
    df = pd.read_parquet(PARQUET_DIR / "ch2025_mWifi.parquet")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour
    df["zone"] = _zone(df["hour"])

    # subject별 수면 구간(00~06h)에서 주로 보이는 BSSID = "홈 네트워크"
    sleep_mask = df["zone"] == "sleep"
    home_bssids_by_subject = {}
    for sid, grp in df[sleep_mask].groupby("subject_id"):
        bssid_list = []
        for arr in grp["m_wifi"]:
            b, _ = _parse_scan(arr)
            bssid_list.extend(b)
        if bssid_list:
            counts = pd.Series(bssid_list).value_counts()
            top_n = set(counts[counts >= counts.quantile(0.7)].index.tolist())
            home_bssids_by_subject[sid] = top_n
        else:
            home_bssids_by_subject[sid] = set()

    rows = []
    for (sid, date), grp in df.groupby(["subject_id", "date"]):
        feat = {"subject_id": sid, "date": date}
        home_nets = home_bssids_by_subject.get(sid, set())

        all_bssids, all_rssis = [], []
        zone_bssids = {z: [] for z in ZONES}
        zone_rssis  = {z: [] for z in ZONES}
        zone_scans  = {z: 0  for z in ZONES}

        for _, row in grp.iterrows():
            b, r = _parse_scan(row["m_wifi"])
            z = row["zone"]
            all_bssids.extend(b)
            all_rssis.extend([v for v in r if not np.isnan(v)])
            zone_bssids[z].extend(b)
            zone_rssis[z].extend([v for v in r if not np.isnan(v)])
            zone_scans[z] += 1

        # 일별 요약
        feat["wifi_n_unique_daily"] = len(set(all_bssids))
        feat["wifi_scans_daily"]    = len(grp)

        if all_bssids:
            counts = pd.Series(all_bssids).value_counts()
            probs = counts / counts.sum()
            feat["wifi_entropy"] = float(-(probs * np.log2(probs + 1e-10)).sum())
        else:
            feat["wifi_entropy"] = np.nan

        feat["wifi_rssi_mean"] = float(np.mean(all_rssis)) if all_rssis else np.nan
        feat["wifi_rssi_std"]  = float(np.std(all_rssis))  if len(all_rssis) > 1 else np.nan

        # 홈 네트워크 비율
        if all_bssids and home_nets:
            feat["wifi_home_ratio"] = sum(1 for b in all_bssids if b in home_nets) / len(all_bssids)
        else:
            feat["wifi_home_ratio"] = np.nan

        # 시간대별
        for z in ZONES:
            feat[f"wifi_unique_{z}"] = len(set(zone_bssids[z])) if zone_bssids[z] else np.nan
            feat[f"wifi_scans_{z}"]  = zone_scans[z]
            feat[f"wifi_rssi_mean_{z}"] = float(np.mean(zone_rssis[z])) if zone_rssis[z] else np.nan

        rows.append(feat)

    result = pd.DataFrame(rows)
    print(f"  mWifi 피처 완료: {result.shape[1]-2}개 피처 x {len(result)}행")
    return result


if __name__ == "__main__":
    feat = build_mwifi()
    print(feat.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    nan_rates = feat.drop(columns=["subject_id", "date"]).isna().mean()
    print("\nNaN 비율:")
    print(nan_rates.round(3).to_string())
