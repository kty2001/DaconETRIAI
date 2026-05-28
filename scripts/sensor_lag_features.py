"""
센서 lag 피처: 이전 날짜의 센서 데이터 (레이블 미사용, 정보 유출 없음)
- sensorlag1_{col}: 전날 센서값
- sensorroll7_{col}: 직전 7 레코딩의 센서 평균

date shift 방식 구현:
  parquet_feat의 원래 날짜 +1일 = lifelog_date 기준 merge 가능
  -> query row with lifelog_date=D gets sensor data from date=D-1 (lag1) ✓
  -> 날짜 간격이 있는 경우 NaN 처리됨 (left merge)
"""

import pandas as pd

ROLL_WINDOW = 7


def build_sensor_lags(parquet_feat: pd.DataFrame) -> pd.DataFrame:
    """
    parquet_feat: (subject_id, date, sensor_cols...) - train + test 날짜 모두 포함
    반환: (subject_id, date, sensorlag1_*, sensorroll7_*) DataFrame
          date 컬럼 = lifelog_date와 조인할 날짜 (원래 센서 날짜 +1일)
    """
    sensor_cols = [c for c in parquet_feat.columns if c not in ("subject_id", "date")]

    pf = parquet_feat[["subject_id", "date"] + sensor_cols].copy()
    pf["date_dt"] = pd.to_datetime(pf["date"])
    pf = pf.sort_values(["subject_id", "date_dt"]).reset_index(drop=True)

    # lag1: 이전 날 센서 (date +1일 shift)
    # parquet date D -> join date D+1 -> query lifelog_date D+1 gets sensor from D
    lag1_df = pf[["subject_id", "date_dt"] + sensor_cols].copy()
    lag1_df = lag1_df.rename(columns={c: f"sensorlag1_{c}" for c in sensor_cols})
    lag1_df["date"] = (lag1_df["date_dt"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    lag1_df = lag1_df.drop(columns=["date_dt"])

    # roll7: 직전 7 레코딩의 센서 평균 (row 기반 rolling, date +1일 shift)
    # parquet date D의 roll7 = 피험자 내 직전 7개 행의 평균 (D 포함)
    # date +1일 shift 후: join date D+1 -> query gets "7-day rolling ending at D"
    roll7_vals = (
        pf.groupby("subject_id")[sensor_cols]
        .transform(lambda x: x.rolling(ROLL_WINDOW, min_periods=1).mean())
    )
    roll7_df = pf[["subject_id", "date_dt"]].copy()
    for col in sensor_cols:
        roll7_df[f"sensorroll7_{col}"] = roll7_vals[col].values
    roll7_df["date"] = (roll7_df["date_dt"] + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
    roll7_df = roll7_df.drop(columns=["date_dt"])

    result = lag1_df.merge(roll7_df, on=["subject_id", "date"], how="outer")
    return result.reset_index(drop=True)
