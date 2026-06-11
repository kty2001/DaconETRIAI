"""
심층 센서 피처: parquet_features_v2 에서 추출하지 않은 새로운 신호

1. wHr 심층 분석
   - hr_sleep_min_clean   : 수면(00-06h) 최저 HR (이상값 제거) = resting HR
   - hr_morning_resting   : 기상 직후(06-08h) 최저 HR
   - hr_presleep_delta    : 취침 전(22-24h) 평균 HR - 수면 평균 HR (스트레스 지표)
   - hr_sleep_first_h     : 수면 첫 1시간(00-01h) 평균 HR
   - hr_sleep_last_h      : 수면 마지막 1시간(05-06h) 평균 HR
   - hr_sleep_depth       : sleep_first_h - sleep_last_h (수면 깊이 대리 지표)
   - hr_daily_resting_p10 : 하루 전체 HR 10th percentile (resting HR 근사)

2. mScreenStatus 심층 분석
   - screen_pickups        : 일별 화면 픽업 횟수 (0->1 전환)
   - screen_pickups_eve    : 저녁(18-22h) 픽업 횟수
   - screen_pickups_night  : 야간(22-24h + 00-06h) 픽업 횟수

3. mUsageStats 앱 다양성
   - n_unique_apps         : 하루 고유 앱 수
   - n_apps_presleep       : 취침 전(22-24h) 고유 앱 수
   - app_top1_share        : 최다 사용 앱의 점유율
   - app_gini              : 앱 사용 시간 지니계수 (집중도)

4. mActivity 활동 단편화
   - sedentary_bouts       : 정적 상태 연속 구간 수
   - sedentary_longest     : 가장 긴 정적 구간 길이(분)
   - activity_transitions  : active <-> sedentary 전환 횟수
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"

HR_LOW  = 30
HR_HIGH = 180


def _load(filename):
    df = pd.read_parquet(PARQUET_DIR / filename)
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour
    return df


def build_deep_hr():
    df = _load("ch2025_wHr.parquet")
    df["hr_val"] = df["heart_rate"].apply(
        lambda x: float(np.mean(x)) if len(x) > 0 else np.nan
    )
    df["hr_min_row"] = df["heart_rate"].apply(
        lambda x: float(np.min(x)) if len(x) > 0 else np.nan
    )
    # 이상값 제거
    df.loc[(df["hr_val"] < HR_LOW) | (df["hr_val"] > HR_HIGH), "hr_val"] = np.nan
    df.loc[(df["hr_min_row"] < HR_LOW) | (df["hr_min_row"] > HR_HIGH), "hr_min_row"] = np.nan

    # 수면 최저 HR (resting HR)
    sleep = df[df["hour"] < 6].copy()
    sleep_agg = sleep.groupby(["subject_id", "date"]).agg(
        hr_sleep_min_clean=("hr_min_row", "min"),
        hr_sleep_mean_clean=("hr_val",    "mean"),
    ).reset_index()

    # 수면 첫 1시간 / 마지막 1시간
    first_h = df[df["hour"] == 0].groupby(["subject_id", "date"]).agg(
        hr_sleep_first_h=("hr_val", "mean")
    ).reset_index()
    last_h  = df[df["hour"] == 5].groupby(["subject_id", "date"]).agg(
        hr_sleep_last_h=("hr_val", "mean")
    ).reset_index()

    # 기상 직후 (06-08h) 최저 HR
    morning = df[(df["hour"] >= 6) & (df["hour"] < 8)].groupby(["subject_id", "date"]).agg(
        hr_morning_resting=("hr_min_row", "min")
    ).reset_index()

    # 취침 전 (22-24h) 평균 HR
    presleep = df[df["hour"] >= 22].groupby(["subject_id", "date"]).agg(
        hr_presleep_mean_deep=("hr_val", "mean")
    ).reset_index()

    # 일별 10th percentile HR (전체 resting HR 근사)
    p10 = df.groupby(["subject_id", "date"]).agg(
        hr_daily_resting_p10=("hr_val", lambda x: float(np.nanpercentile(x.dropna(), 10)) if x.notna().sum() > 0 else np.nan)
    ).reset_index()

    result = sleep_agg
    for df2 in [first_h, last_h, morning, presleep, p10]:
        result = result.merge(df2, on=["subject_id", "date"], how="left")

    # 합성 피처
    result["hr_sleep_depth"]    = result["hr_sleep_first_h"] - result["hr_sleep_last_h"]
    result["hr_presleep_delta"] = result["hr_presleep_mean_deep"] - result["hr_sleep_mean_clean"]

    return result


def build_deep_screen():
    df = _load("ch2025_mScreenStatus.parquet")

    def count_pickups(grp):
        s = grp.sort_values("timestamp")["m_screen_use"]
        return int(((s.shift(1) == 0) & (s == 1)).sum())

    # 전체 일별 픽업 수
    daily = df.groupby(["subject_id", "date"]).apply(count_pickups).reset_index()
    daily.columns = ["subject_id", "date", "screen_pickups"]

    # 저녁(18-22h) 픽업
    eve = df[df["hour"].between(18, 21)].groupby(["subject_id", "date"]).apply(
        count_pickups
    ).reset_index()
    eve.columns = ["subject_id", "date", "screen_pickups_eve"]

    # 야간(22-06h) 픽업
    night_df = df[(df["hour"] >= 22) | (df["hour"] < 6)].copy()
    night = night_df.groupby(["subject_id", "date"]).apply(
        count_pickups
    ).reset_index()
    night.columns = ["subject_id", "date", "screen_pickups_night"]

    result = daily.merge(eve,   on=["subject_id", "date"], how="left")
    result = result.merge(night, on=["subject_id", "date"], how="left")
    return result


def _gini(arr):
    """1차원 양수 배열의 지니계수"""
    arr = arr[arr > 0]
    if len(arr) < 2:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def build_deep_apps():
    df = pd.read_parquet(PARQUET_DIR / "ch2025_mUsageStats.parquet")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour
    df["zone"] = df["hour"].apply(
        lambda h: "presleep" if h >= 22 else ("sleep" if h < 6 else "other")
    )

    records = []
    for _, row in df.iterrows():
        for entry in row["m_usage_stats"]:
            records.append({
                "subject_id": row["subject_id"],
                "date": row["date"],
                "zone": row["zone"],
                "app": entry["app_name"],
                "ms": entry["total_time"],
            })
    flat = pd.DataFrame(records)

    # 일별 고유 앱 수
    n_apps = flat.groupby(["subject_id", "date"])["app"].nunique().rename("n_unique_apps").reset_index()

    # 취침 전 고유 앱 수
    n_apps_pre = (
        flat[flat["zone"] == "presleep"]
        .groupby(["subject_id", "date"])["app"].nunique()
        .rename("n_apps_presleep").reset_index()
    )

    # 최다 사용 앱 점유율
    def top1_share(grp):
        total = grp["ms"].sum()
        if total == 0:
            return np.nan
        return float(grp.groupby("app")["ms"].sum().max() / total)

    top1 = flat.groupby(["subject_id", "date"]).apply(top1_share).rename("app_top1_share").reset_index()

    # 지니계수
    def gini_apps(grp):
        app_times = grp.groupby("app")["ms"].sum().values.astype(float)
        return _gini(app_times)

    gini = flat.groupby(["subject_id", "date"]).apply(gini_apps).rename("app_gini").reset_index()

    result = n_apps.merge(n_apps_pre, on=["subject_id", "date"], how="left")
    result = result.merge(top1,       on=["subject_id", "date"], how="left")
    result = result.merge(gini,       on=["subject_id", "date"], how="left")
    return result


def build_deep_activity():
    df = _load("ch2025_mActivity.parquet")
    # active=4,7 / sedentary=3 / others
    df["is_active"]    = df["m_activity"].isin([4, 7]).astype(int)
    df["is_sedentary"] = (df["m_activity"] == 3).astype(int)

    rows = []
    for (sid, date), grp in df.groupby(["subject_id", "date"]):
        grp = grp.sort_values("timestamp")
        sed = grp["is_sedentary"].values
        act = grp["is_active"].values

        # 정적 구간 수 및 최장 구간 (연속 정적 run)
        def runs(arr):
            if len(arr) == 0:
                return 0, 0
            changes = np.diff(np.concatenate([[0], arr, [0]]))
            starts = np.where(changes == 1)[0]
            ends   = np.where(changes == -1)[0]
            lengths = ends - starts
            return len(lengths), int(max(lengths)) if len(lengths) > 0 else 0

        n_sed_bouts, longest_sed = runs(sed)

        # active <-> sedentary 전환 횟수
        state = np.where(act == 1, 1, np.where(sed == 1, -1, 0))
        transitions = int((np.diff(np.sign(state)) != 0).sum())

        rows.append({
            "subject_id":       sid,
            "date":             date,
            "sedentary_bouts":  n_sed_bouts,
            "sedentary_longest": longest_sed,
            "activity_transitions": transitions,
        })

    return pd.DataFrame(rows)


def build_all():
    print("  deep HR features...", flush=True)
    hr   = build_deep_hr()
    print("  deep screen features...", flush=True)
    sc   = build_deep_screen()
    print("  deep app features...", flush=True)
    apps = build_deep_apps()
    print("  deep activity features...", flush=True)
    act  = build_deep_activity()

    result = hr
    for df2 in [sc, apps, act]:
        result = result.merge(df2, on=["subject_id", "date"], how="outer")

    cols = [c for c in result.columns if c not in ("subject_id", "date")]
    print(f"  deep features: {len(cols)} cols -> {cols}", flush=True)
    return result


if __name__ == "__main__":
    df = build_all()
    print(df.describe().to_string())
