"""
parquet 피처 v2: 시간대별 집계 + 수면 특화 피처 + mAmbience
시간대 구분:
  morning   : 06~12h
  afternoon : 12~18h
  evening   : 18~22h
  presleep  : 22~24h
  sleep     : 00~06h (수면 중)
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"


def _load(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_DIR / filename)
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour
    return df


def _zone(hour: pd.Series) -> pd.Series:
    """hour → 시간대 문자열"""
    conditions = [
        (hour >= 6) & (hour < 12),
        (hour >= 12) & (hour < 18),
        (hour >= 18) & (hour < 22),
        (hour >= 22),
        (hour < 6),
    ]
    choices = ["morning", "afternoon", "evening", "presleep", "sleep"]
    return np.select(conditions, choices, default="sleep")


# ── wPedo ────────────────────────────────────────────────────────────────────
def build_wpedo() -> pd.DataFrame:
    df = _load("ch2025_wPedo.parquet")
    df["zone"] = _zone(df["hour"])

    daily = df.groupby(["subject_id", "date"]).agg(
        pedo_step_sum=("step", "sum"),
        pedo_calories_sum=("burned_calories", "sum"),
        pedo_distance_sum=("distance", "sum"),
        pedo_speed_mean=("speed", "mean"),
    ).reset_index()

    # 시간대별 걸음수
    zone_step = (
        df.groupby(["subject_id", "date", "zone"])["step"]
        .sum()
        .unstack(fill_value=0)
        .add_prefix("pedo_step_")
        .reset_index()
    )
    # 없는 zone 컬럼 보완
    for z in ["morning", "afternoon", "evening", "presleep", "sleep"]:
        col = f"pedo_step_{z}"
        if col not in zone_step.columns:
            zone_step[col] = 0

    return daily.merge(zone_step, on=["subject_id", "date"], how="left")


# ── mActivity ────────────────────────────────────────────────────────────────
def build_mactivity() -> pd.DataFrame:
    df = _load("ch2025_mActivity.parquet")
    df["zone"] = _zone(df["hour"])
    total = df.groupby(["subject_id", "date"]).size().rename("act_total")

    active = (
        df[df["m_activity"].isin([4, 7])]
        .groupby(["subject_id", "date"]).size().rename("act_active_cnt")
    )
    still = (
        df[df["m_activity"] == 3]
        .groupby(["subject_id", "date"]).size().rename("act_still_cnt")
    )
    agg = pd.concat([total, active, still], axis=1).fillna(0).reset_index()
    agg["act_active_ratio"] = agg["act_active_cnt"] / agg["act_total"].clip(lower=1)
    agg["act_still_ratio"] = agg["act_still_cnt"] / agg["act_total"].clip(lower=1)

    # 시간대별 활동 비율 (active = 4/7)
    zone_active = (
        df[df["m_activity"].isin([4, 7])]
        .groupby(["subject_id", "date", "zone"]).size()
        .unstack(fill_value=0)
        .add_prefix("act_active_")
        .reset_index()
    )
    zone_total = (
        df.groupby(["subject_id", "date", "zone"]).size()
        .unstack(fill_value=0)
        .add_prefix("act_total_")
        .reset_index()
    )
    for z in ["morning", "afternoon", "evening", "presleep", "sleep"]:
        for prefix, df2 in [("act_active_", zone_active), ("act_total_", zone_total)]:
            col = f"{prefix}{z}"
            if col not in df2.columns:
                df2[col] = 0

    zone_merged = zone_active.merge(zone_total, on=["subject_id", "date"], how="outer").fillna(0)
    zone_ratio = zone_merged[["subject_id", "date"]].copy()
    for z in ["morning", "afternoon", "evening", "presleep", "sleep"]:
        act_col = f"act_active_{z}"
        tot_col = f"act_total_{z}"
        if act_col not in zone_merged.columns:
            zone_merged[act_col] = 0
        if tot_col not in zone_merged.columns:
            zone_merged[tot_col] = 0
        zone_ratio[f"act_ratio_{z}"] = (
            zone_merged[act_col] / zone_merged[tot_col].clip(lower=1)
        ).values

    result = agg[["subject_id", "date", "act_active_ratio", "act_still_ratio", "act_active_cnt"]]
    return result.merge(zone_ratio, on=["subject_id", "date"], how="left")


# ── mScreenStatus ─────────────────────────────────────────────────────────────
def build_mscreen() -> pd.DataFrame:
    df = _load("ch2025_mScreenStatus.parquet")
    df["zone"] = _zone(df["hour"])

    daily = df.groupby(["subject_id", "date"]).agg(
        screen_on_ratio=("m_screen_use", "mean"),
        screen_on_count=("m_screen_use", "sum"),
    ).reset_index()

    zone_screen = (
        df.groupby(["subject_id", "date", "zone"])["m_screen_use"]
        .mean()
        .unstack()
        .add_prefix("screen_ratio_")
        .reset_index()
    )
    for z in ["morning", "afternoon", "evening", "presleep", "sleep"]:
        col = f"screen_ratio_{z}"
        if col not in zone_screen.columns:
            zone_screen[col] = np.nan

    return daily.merge(zone_screen, on=["subject_id", "date"], how="left")


# ── wHr ───────────────────────────────────────────────────────────────────────
def build_whr() -> pd.DataFrame:
    df = _load("ch2025_wHr.parquet")
    df["hr_min"] = df["heart_rate"].apply(
        lambda x: float(np.mean(x)) if len(x) > 0 else np.nan
    )
    df["zone"] = _zone(df["hour"])

    daily = df.groupby(["subject_id", "date"]).agg(
        hr_mean=("hr_min", "mean"),
        hr_std=("hr_min", "std"),
        hr_min_val=("hr_min", "min"),
        hr_max_val=("hr_min", "max"),
    ).reset_index()

    # 시간대별 심박 평균
    zone_hr = (
        df.groupby(["subject_id", "date", "zone"])["hr_min"]
        .agg(["mean", "std"])
        .unstack()
        .reset_index()
    )
    zone_hr.columns = [
        "_".join(filter(None, map(str, c))) if isinstance(c, tuple) else c
        for c in zone_hr.columns
    ]
    zone_hr.columns = [
        c.replace("mean_", "hr_mean_").replace("std_", "hr_std_")
        if c not in ("subject_id", "date") else c
        for c in zone_hr.columns
    ]

    # 수면 특화: sleep zone RMSSD 근사 (연속 차분 절대값 평균)
    sleep_df = df[df["zone"] == "sleep"].copy()
    rmssd_rows = []
    for (sid, date), grp in sleep_df.groupby(["subject_id", "date"]):
        vals = grp.sort_values("hour")["hr_min"].dropna().values
        if len(vals) >= 2:
            rmssd = float(np.mean(np.abs(np.diff(vals))))
        else:
            rmssd = np.nan
        rmssd_rows.append({"subject_id": sid, "date": date, "hr_sleep_rmssd": rmssd})
    rmssd_df = pd.DataFrame(rmssd_rows)

    result = daily.merge(zone_hr, on=["subject_id", "date"], how="left")
    result = result.merge(rmssd_df, on=["subject_id", "date"], how="left")
    return result


# ── mLight ────────────────────────────────────────────────────────────────────
def build_mlight() -> pd.DataFrame:
    df = _load("ch2025_mLight.parquet")
    df["zone"] = _zone(df["hour"])

    daily = df.groupby(["subject_id", "date"]).agg(
        light_mean=("m_light", "mean"),
        light_max=("m_light", "max"),
    ).reset_index()

    zone_light = (
        df.groupby(["subject_id", "date", "zone"])["m_light"]
        .mean()
        .unstack()
        .add_prefix("light_mean_")
        .reset_index()
    )
    for z in ["morning", "afternoon", "evening", "presleep", "sleep"]:
        col = f"light_mean_{z}"
        if col not in zone_light.columns:
            zone_light[col] = np.nan

    return daily.merge(zone_light, on=["subject_id", "date"], how="left")


# presleep/sleep 시간대에서 유효한 sound 카테고리 (상위 빈도 기준)
KEEP_LABELS = {"Silence", "Speech", "Music", "Inside, small room",
               "Inside, large room or hall", "Breathing", "Snoring",
               "White noise", "Television", "Computer keyboard"}


# ── mAmbience ─────────────────────────────────────────────────────────────────
def build_mambience() -> pd.DataFrame:
    df = _load("ch2025_mAmbience.parquet")
    df["zone"] = _zone(df["hour"])

    # m_ambience 컬럼 구조 확인 후 파싱
    sample_val = df["m_ambience"].dropna().iloc[0] if not df["m_ambience"].dropna().empty else None

    def extract_label(x):
        """numpy array of arrays [[label, prob], ...] 또는 list of dicts 처리"""
        if x is None:
            return None
        try:
            if isinstance(x, (list, np.ndarray)) and len(x) > 0:
                first = x[0]
                if isinstance(first, (list, np.ndarray)) and len(first) >= 2:
                    # [[label, prob_str], ...] 형태
                    best_label = None
                    best_prob = -1.0
                    for item in x:
                        try:
                            prob = float(item[1])
                            if prob > best_prob:
                                best_prob = prob
                                best_label = str(item[0])
                        except Exception:
                            continue
                    return best_label
                elif isinstance(first, dict):
                    return max(x, key=lambda d: d.get("probability", 0)).get("label", None)
            elif isinstance(x, str):
                return x
        except Exception:
            pass
        return None

    df["ambience_label"] = df["m_ambience"].apply(extract_label)
    if df["ambience_label"].isna().all():
        return pd.DataFrame(columns=["subject_id", "date"])

    # KEEP_LABELS에 없는 레이블은 "Other"로 병합
    df["ambience_label"] = df["ambience_label"].apply(
        lambda x: x if x in KEEP_LABELS else "Other"
    )

    def compute_ratios(sub_df, prefix):
        total = sub_df.groupby(["subject_id", "date"]).size().rename("_total")
        cnt = (
            sub_df.groupby(["subject_id", "date", "ambience_label"]).size()
            .unstack(fill_value=0)
        )
        # 모든 KEEP_LABELS + Other 컬럼 보장
        for lbl in list(KEEP_LABELS) + ["Other"]:
            if lbl not in cnt.columns:
                cnt[lbl] = 0
        cnt = cnt.join(total)
        for lbl in list(KEEP_LABELS) + ["Other"]:
            col = lbl.replace(" ", "_").replace(",", "")
            cnt[f"{prefix}_{col}"] = cnt[lbl] / cnt["_total"].clip(lower=1)
        keep = [c for c in cnt.columns if c.startswith(f"{prefix}_")]
        return cnt.reset_index()[["subject_id", "date"] + keep]

    daily_amb = compute_ratios(df, "amb")
    night_df = df[df["zone"].isin(["presleep", "sleep"])].copy()
    if len(night_df) > 0:
        night_amb = compute_ratios(night_df, "amb_night")
        result = daily_amb.merge(night_amb, on=["subject_id", "date"], how="left")
    else:
        result = daily_amb

    return result


# ── mUsageStats ───────────────────────────────────────────────────────────────
def build_musagestats() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_DIR / "ch2025_mUsageStats.parquet")
    df["date"] = df["timestamp"].dt.date.astype(str)
    df["hour"] = df["timestamp"].dt.hour
    df["zone"] = _zone(df["hour"])

    # 배열 → 행 단위로 펼치기
    records = []
    for _, row in df.iterrows():
        for entry in row["m_usage_stats"]:
            records.append({
                "subject_id": row["subject_id"],
                "date":       row["date"],
                "zone":       row["zone"],
                "total_time": entry["total_time"],  # ms
            })
    flat = pd.DataFrame(records)

    ZONES = ["morning", "afternoon", "evening", "presleep", "sleep"]

    # 시간대별 집계
    zone_agg = flat.groupby(["subject_id", "date", "zone"]).agg(
        _total=("total_time", "sum"),
        _apps=("total_time", "count"),
    ).reset_index()

    zone_total = (
        zone_agg.pivot_table(index=["subject_id", "date"], columns="zone", values="_total", fill_value=0)
        .add_prefix("usage_ms_")
        .reset_index()
    )
    zone_apps = (
        zone_agg.pivot_table(index=["subject_id", "date"], columns="zone", values="_apps", fill_value=0)
        .add_prefix("usage_apps_")
        .reset_index()
    )

    # 없는 zone 컬럼 보완
    for z in ZONES:
        for df_z, pfx in [(zone_total, "usage_ms_"), (zone_apps, "usage_apps_")]:
            col = f"{pfx}{z}"
            if col not in df_z.columns:
                df_z[col] = 0

    result = zone_total.merge(zone_apps, on=["subject_id", "date"], how="outer")

    # 일별 합산 및 비율 피처
    ms_cols = [f"usage_ms_{z}" for z in ZONES]
    result["usage_ms_total"]          = result[ms_cols].sum(axis=1)
    result["usage_presleep_ratio"]    = (
        result["usage_ms_presleep"] / result["usage_ms_total"].replace(0, np.nan)
    )
    result["usage_sleep_ratio"]       = (
        result["usage_ms_sleep"] / result["usage_ms_total"].replace(0, np.nan)
    )

    # subject별 z-score 정규화: 절대값(ms, 앱 수) → 개인 내 상대적 수치로 변환
    # ratio 피처(presleep_ratio, sleep_ratio)는 이미 상대값이므로 제외
    apps_cols = [f"usage_apps_{z}" for z in ZONES]
    abs_cols  = ms_cols + apps_cols + ["usage_ms_total"]
    for col in abs_cols:
        subj_mean = result.groupby("subject_id")[col].transform("mean")
        subj_std  = result.groupby("subject_id")[col].transform("std").replace(0, np.nan)
        result[col] = (result[col] - subj_mean) / subj_std

    return result


# ── 통합 ──────────────────────────────────────────────────────────────────────
def build_all() -> pd.DataFrame:
    builders = [
        ("wPedo",         build_wpedo),
        ("mActivity",     build_mactivity),
        ("mScreenStatus", build_mscreen),
        ("wHr",           build_whr),
        ("mLight",        build_mlight),
        ("mAmbience",     build_mambience),
        ("mUsageStats",   build_musagestats),
    ]
    feat = None
    for name, fn in builders:
        print(f"  {name} 집계 중...")
        df = fn()
        if feat is None:
            feat = df
        else:
            feat = feat.merge(df, on=["subject_id", "date"], how="outer")

    # 컬럼 중복 정리 (level_* 등 불필요 컬럼 제거)
    drop_cols = [c for c in feat.columns if c.startswith("level_")]
    feat = feat.drop(columns=drop_cols, errors="ignore")

    print(f"  완료: {len(feat)}행, {len(feat.columns)}컬럼")
    return feat


if __name__ == "__main__":
    print("=== parquet 피처 v2 집계 ===")
    feat = build_all()
    print(feat.head(3).to_string())
    print("\n컬럼 목록:")
    parquet_cols = [c for c in feat.columns if c not in ("subject_id", "date")]
    print(parquet_cols)
    print(f"\n총 {len(parquet_cols)}개 피처")
