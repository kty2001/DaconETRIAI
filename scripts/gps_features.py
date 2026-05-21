"""
GPS 피처 빌더 (mGps.parquet)
좌표는 익명화(스케일링)된 단위, 속도는 실제 m/s.

피처 구성:
  속도 기반  : gps_speed_mean/max, gps_moving_ratio, gps_total_dist_m
              gps_speed_{zone}, gps_moving_{zone} (morning~presleep, sleep 제외)
  공간 기반  : gps_n_places (격자 셀 수), gps_place_entropy, gps_radius_gyration
  홈 기반    : gps_home_ratio, gps_dist_from_home_mean/max, gps_night_home_ratio
              (홈 = subject별 수면 구간(00~06h) centroid)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"

GRID_SIZE       = 0.001   # 장소 판별 격자 크기 (익명화 좌표 단위)
HOME_RADIUS     = 0.002   # 홈 판정 반경
MOVING_THRESH   = 0.5     # 이동 판정 속도 (m/s)


def _zone(hour: pd.Series) -> pd.Series:
    conditions = [
        (hour >= 6)  & (hour < 12),
        (hour >= 12) & (hour < 18),
        (hour >= 18) & (hour < 22),
        (hour >= 22),
        (hour < 6),
    ]
    choices = ["morning", "afternoon", "evening", "presleep", "sleep"]
    return np.select(conditions, choices, default="sleep")


def _parse_batch(m_gps_series: pd.Series):
    """m_gps 컬럼 → (lat, lon, speed) 배열 (분 단위 평균)"""
    lats, lons, speeds = [], [], []
    for arr in m_gps_series:
        if arr is None or len(arr) == 0:
            lats.append(np.nan)
            lons.append(np.nan)
            speeds.append(np.nan)
        else:
            lats.append(float(np.mean([x["latitude"] for x in arr])))
            lons.append(float(np.mean([x["longitude"] for x in arr])))
            speeds.append(float(np.mean([x["speed"] for x in arr])))
    return np.array(lats), np.array(lons), np.array(speeds)


def build_gps() -> pd.DataFrame:
    """
    반환: (subject_id, date) 기준 GPS 피처 DataFrame
          date = lifelog_date (parquet_features_v2와 동일 키)
    """
    df = pd.read_parquet(PARQUET_DIR / "ch2025_mGps.parquet")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour
    df["zone"] = _zone(df["hour"])

    print("  GPS 파싱 중 (800k rows)...")
    lats, lons, speeds = _parse_batch(df["m_gps"])
    df["lat"]   = lats
    df["lon"]   = lons
    df["speed"] = speeds
    df = df.dropna(subset=["lat", "lon", "speed"])

    # ── subject별 홈 위치 (수면 구간 centroid) ────────────────────────────────
    sleep_df = df[df["zone"] == "sleep"]
    home = (
        sleep_df.groupby("subject_id")[["lat", "lon"]]
        .mean()
        .rename(columns={"lat": "home_lat", "lon": "home_lon"})
    )
    df = df.merge(home, on="subject_id", how="left")
    df["dist_home"] = np.sqrt(
        (df["lat"] - df["home_lat"]) ** 2 +
        (df["lon"] - df["home_lon"]) ** 2
    )
    df["at_home"] = (df["dist_home"] < HOME_RADIUS).astype(float)
    df["moving"]  = (df["speed"] > MOVING_THRESH).astype(float)

    # ── 격자 셀 (장소 판별) ───────────────────────────────────────────────────
    df["lat_cell"]  = (df["lat"] / GRID_SIZE).round().astype(int)
    df["lon_cell"]  = (df["lon"] / GRID_SIZE).round().astype(int)
    df["place_id"]  = df["lat_cell"].astype(str) + "_" + df["lon_cell"].astype(str)

    # ── 일별 집계 ─────────────────────────────────────────────────────────────
    ZONES = ["morning", "afternoon", "evening", "presleep"]

    rows = []
    for (sid, date), grp in df.groupby(["subject_id", "date"]):
        feat = {"subject_id": sid, "date": date}

        # 전일 속도 기반
        feat["gps_speed_mean"]    = grp["speed"].mean()
        feat["gps_speed_max"]     = grp["speed"].max()
        feat["gps_moving_ratio"]  = grp["moving"].mean()
        feat["gps_total_dist_m"]  = grp["speed"].sum() * 60  # ≈ 총 이동 거리(m)

        # 시간대별 속도
        for zone in ZONES:
            z = grp[grp["zone"] == zone]
            feat[f"gps_speed_{zone}"]  = z["speed"].mean()  if len(z) > 0 else np.nan
            feat[f"gps_moving_{zone}"] = z["moving"].mean() if len(z) > 0 else np.nan

        # 장소 다양성
        place_cnt = grp["place_id"].value_counts()
        feat["gps_n_places"] = len(place_cnt)
        probs = place_cnt / place_cnt.sum()
        feat["gps_place_entropy"] = float(-(probs * np.log2(probs + 1e-10)).sum())

        # 회전 반경 (공간 분산)
        lats_a = grp["lat"].values
        lons_a = grp["lon"].values
        feat["gps_radius_gyration"] = float(
            np.sqrt(((lats_a - lats_a.mean()) ** 2 + (lons_a - lons_a.mean()) ** 2).mean())
        )

        # 홈 체류
        feat["gps_home_ratio"]          = grp["at_home"].mean()
        feat["gps_dist_from_home_mean"] = grp["dist_home"].mean()
        feat["gps_dist_from_home_max"]  = grp["dist_home"].max()

        # 야간(취침 전+수면) 홈 체류
        night = grp[grp["zone"].isin(["presleep", "sleep"])]
        feat["gps_night_home_ratio"] = night["at_home"].mean() if len(night) > 0 else np.nan

        rows.append(feat)

    result = pd.DataFrame(rows)
    print(f"  GPS 피처 완료: {result.shape[1]-2}개 피처 × {len(result)}행")
    return result


if __name__ == "__main__":
    feat = build_gps()
    print(feat.describe().T[["mean", "std", "min", "max"]].round(4).to_string())
