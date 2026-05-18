"""
parquet 피처 v3: v2 기반 심층 센서 피처 추가

신규 피처:
  wHr:          hr_sleep_min_abs, hr_sleep_intra_var,
                hr_presleep_to_sleep_drop, hr_nocturnal_dip
  mScreenStatus: screen_presleep_last_minute, screen_presleep_max_run,
                screen_sleep_any
  mAmbience:    amb_presleep_* / amb_sleep_* (구간 분리 집계),
                amb_sleep_quiet_ratio, amb_sleep_noisy_ratio
  wPedo:        pedo_active_hours, pedo_walk_ratio, pedo_evening_step_ratio
"""

import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))
from parquet_features_v2 import build_all as _build_v2, _load, _zone

ROOT = Path(__file__).parent.parent
PARQUET_DIR = ROOT / "data" / "ch2025_data_items"

KEEP_LABELS = {
    "Silence", "Speech", "Music",
    "Inside, small room", "Inside, large room or hall",
    "Breathing", "Snoring", "White noise",
    "Television", "Computer keyboard",
}


# ── 공통 유틸 ─────────────────────────────────────────────────────────────────
def _extract_ambience_label(x):
    if x is None:
        return None
    try:
        if isinstance(x, (list, np.ndarray)) and len(x) > 0:
            first = x[0]
            if isinstance(first, (list, np.ndarray)) and len(first) >= 2:
                best_label, best_prob = None, -1.0
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


def _ambience_ratios(sub_df, prefix):
    """주어진 구간 DataFrame에 대해 레이블별 비율 집계"""
    if len(sub_df) == 0:
        return pd.DataFrame(columns=["subject_id", "date"])
    total = sub_df.groupby(["subject_id", "date"]).size().rename("_total")
    cnt = (
        sub_df.groupby(["subject_id", "date", "ambience_label"])
        .size()
        .unstack(fill_value=0)
    )
    for lbl in list(KEEP_LABELS) + ["Other"]:
        if lbl not in cnt.columns:
            cnt[lbl] = 0
    cnt = cnt.join(total)
    for lbl in list(KEEP_LABELS) + ["Other"]:
        col = lbl.replace(" ", "_").replace(",", "")
        cnt[f"{prefix}_{col}"] = cnt[lbl] / cnt["_total"].clip(lower=1)
    keep_cols = [c for c in cnt.columns if c.startswith(f"{prefix}_")]
    return cnt.reset_index()[["subject_id", "date"] + keep_cols]


# ── wHr 심층 ─────────────────────────────────────────────────────────────────
def build_whr_v3() -> pd.DataFrame:
    df = _load("ch2025_wHr.parquet")
    # 각 행 배열 → 통계값 추출
    df["hr_mean_row"] = df["heart_rate"].apply(
        lambda x: float(np.mean(x)) if len(x) > 0 else np.nan
    )
    df["hr_std_row"] = df["heart_rate"].apply(
        lambda x: float(np.std(x)) if len(x) > 1 else np.nan
    )
    df["hr_min_row"] = df["heart_rate"].apply(
        lambda x: float(np.min(x)) if len(x) > 0 else np.nan
    )
    df["zone"] = _zone(df["hour"])

    sleep_df    = df[df["zone"] == "sleep"]
    presleep_df = df[df["zone"] == "presleep"]
    evening_df  = df[df["zone"] == "evening"]

    # 수면 중 절대 최저 심박수 (hourly 평균이 아닌 실측 최솟값)
    hr_sleep_min_abs = (
        sleep_df.groupby(["subject_id", "date"])["hr_min_row"]
        .min().rename("hr_sleep_min_abs")
    )
    # 수면 중 intra-hour 변동성 (시간당 std의 평균 — HRV 근사)
    hr_sleep_intra_var = (
        sleep_df.groupby(["subject_id", "date"])["hr_std_row"]
        .mean().rename("hr_sleep_intra_var")
    )
    # 구간별 평균 심박수 (내부용)
    hr_presleep_mean = (
        presleep_df.groupby(["subject_id", "date"])["hr_mean_row"]
        .mean().rename("_hr_presleep")
    )
    hr_sleep_mean = (
        sleep_df.groupby(["subject_id", "date"])["hr_mean_row"]
        .mean().rename("_hr_sleep")
    )
    hr_evening_mean = (
        evening_df.groupby(["subject_id", "date"])["hr_mean_row"]
        .mean().rename("_hr_evening")
    )

    result = pd.concat(
        [hr_sleep_min_abs, hr_sleep_intra_var,
         hr_presleep_mean, hr_sleep_mean, hr_evening_mean],
        axis=1,
    ).reset_index()

    # 취침 전 → 수면 중 심박 감소량 (심박 이완 효율)
    result["hr_presleep_to_sleep_drop"] = (
        result["_hr_presleep"] - result["_hr_sleep"]
    )
    # 야간 심박 감소율 (저녁 대비 수면 중 % 감소)
    result["hr_nocturnal_dip"] = (
        (result["_hr_evening"] - result["_hr_sleep"])
        / result["_hr_evening"].clip(lower=1)
    )

    return result[[
        "subject_id", "date",
        "hr_sleep_min_abs", "hr_sleep_intra_var",
        "hr_presleep_to_sleep_drop", "hr_nocturnal_dip",
    ]]


# ── mScreenStatus 심층 ────────────────────────────────────────────────────────
def build_mscreen_v3() -> pd.DataFrame:
    df = _load("ch2025_mScreenStatus.parquet")
    df["zone"] = _zone(df["hour"])
    df["minute_of_day"] = df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute

    presleep = df[df["zone"] == "presleep"].copy()
    sleep    = df[df["zone"] == "sleep"].copy()

    # 취침 전(22~24h) 마지막 화면 켜짐 시각 (22:00 기준 분, 0~119)
    presleep_on = presleep[presleep["m_screen_use"] == 1]
    last_min = (
        presleep_on
        .groupby(["subject_id", "date"])["minute_of_day"]
        .max()
        .sub(22 * 60)
        .rename("screen_presleep_last_minute")
        .reset_index()
    )

    # 취침 전 최장 연속 화면 켜짐 구간 (readings 수 기준)
    def max_run(series):
        mx = cur = 0
        for v in series.values:
            cur = cur + 1 if v == 1 else 0
            if cur > mx:
                mx = cur
        return mx

    max_cont = (
        presleep.sort_values("minute_of_day")
        .groupby(["subject_id", "date"])["m_screen_use"]
        .apply(max_run)
        .rename("screen_presleep_max_run")
        .reset_index()
    )

    # 수면 중(00~06h) 화면 사용 여부
    sleep_any = (
        sleep.groupby(["subject_id", "date"])["m_screen_use"]
        .max()
        .rename("screen_sleep_any")
        .reset_index()
    )

    result = last_min.merge(max_cont, on=["subject_id", "date"], how="outer")
    result = result.merge(sleep_any, on=["subject_id", "date"], how="outer")
    return result


# ── mAmbience 구간 분리 집계 ─────────────────────────────────────────────────
def build_mambience_v3() -> pd.DataFrame:
    df = _load("ch2025_mAmbience.parquet")
    df["zone"] = _zone(df["hour"])
    df["ambience_label"] = df["m_ambience"].apply(_extract_ambience_label)
    df["ambience_label"] = df["ambience_label"].apply(
        lambda x: x if x in KEEP_LABELS else "Other"
    )

    presleep_df = df[df["zone"] == "presleep"]
    sleep_df    = df[df["zone"] == "sleep"]

    presleep_amb = _ambience_ratios(presleep_df, "amb_presleep")
    sleep_amb    = _ambience_ratios(sleep_df,    "amb_sleep")

    # 수면 구간 종합 피처
    if len(sleep_amb) > 0:
        quiet_cols = [c for c in sleep_amb.columns
                      if any(kw in c for kw in ["_Silence", "_Breathing", "_White_noise"])]
        if quiet_cols:
            sleep_amb["amb_sleep_quiet_ratio"] = sleep_amb[quiet_cols].sum(axis=1)
        if "amb_sleep_Other" in sleep_amb.columns:
            sleep_amb["amb_sleep_noisy_ratio"] = sleep_amb["amb_sleep_Other"]

    if len(presleep_amb) > 0 and len(sleep_amb) > 0:
        result = presleep_amb.merge(sleep_amb, on=["subject_id", "date"], how="outer")
    elif len(presleep_amb) > 0:
        result = presleep_amb
    else:
        result = sleep_amb

    return result


# ── wPedo 심층 ────────────────────────────────────────────────────────────────
def build_wpedo_v3() -> pd.DataFrame:
    df = _load("ch2025_wPedo.parquet")
    df["zone"] = _zone(df["hour"])

    # 활동 시간 (걸음수 > 100인 시간 수)
    active_hours = (
        df[df["step"] > 100]
        .groupby(["subject_id", "date"])
        .size()
        .rename("pedo_active_hours")
        .reset_index()
    )

    # 걷기 비율 (달리기 vs 걷기)
    walk_run = df.groupby(["subject_id", "date"]).agg(
        _walk=("walking_step", "sum"),
        _run=("running_step", "sum"),
    ).reset_index()
    walk_run["pedo_walk_ratio"] = (
        walk_run["_walk"] / (walk_run["_walk"] + walk_run["_run"] + 1)
    )
    walk_run = walk_run[["subject_id", "date", "pedo_walk_ratio"]]

    # 저녁 걸음수 비율
    total_step = (
        df.groupby(["subject_id", "date"])["step"]
        .sum().rename("_total").reset_index()
    )
    evening_step = (
        df[df["zone"] == "evening"]
        .groupby(["subject_id", "date"])["step"]
        .sum().rename("_evening").reset_index()
    )
    step_ratio = total_step.merge(evening_step, on=["subject_id", "date"], how="left")
    step_ratio["_evening"] = step_ratio["_evening"].fillna(0)
    step_ratio["pedo_evening_step_ratio"] = (
        step_ratio["_evening"] / step_ratio["_total"].clip(lower=1)
    )
    step_ratio = step_ratio[["subject_id", "date", "pedo_evening_step_ratio"]]

    result = active_hours.merge(walk_run, on=["subject_id", "date"], how="outer")
    result = result.merge(step_ratio, on=["subject_id", "date"], how="outer")
    return result


# ── 통합 ──────────────────────────────────────────────────────────────────────
def build_all() -> pd.DataFrame:
    """v2 피처 전체 + v3 심층 피처 추가"""
    print("  v2 기반 피처 집계 중...")
    feat = _build_v2()

    extras = [
        ("wHr 심층",        build_whr_v3),
        ("mScreenStatus 심층", build_mscreen_v3),
        ("mAmbience 구간분리", build_mambience_v3),
        ("wPedo 심층",      build_wpedo_v3),
    ]
    for name, fn in extras:
        print(f"  {name} 집계 중...")
        extra_feat = fn()
        feat = feat.merge(extra_feat, on=["subject_id", "date"], how="left")

    drop_cols = [c for c in feat.columns if c.startswith("level_")]
    feat = feat.drop(columns=drop_cols, errors="ignore")

    print(f"  완료: {len(feat)}행, {len(feat.columns)}컬럼")
    return feat


if __name__ == "__main__":
    print("=== parquet 피처 v3 집계 ===")
    feat = build_all()
    v3_keywords = [
        "hr_sleep_min_abs", "hr_sleep_intra_var",
        "hr_presleep_to_sleep_drop", "hr_nocturnal_dip",
        "screen_presleep_last", "screen_presleep_max", "screen_sleep_any",
        "amb_presleep_", "amb_sleep_",
        "pedo_active_hours", "pedo_walk_ratio", "pedo_evening_step",
    ]
    new_cols = [c for c in feat.columns
                if any(kw in c for kw in v3_keywords)]
    print(f"\n신규 피처 ({len(new_cols)}개):")
    for c in new_cols:
        print(f"  {c}")
    print(f"\n전체 parquet 피처 수: {len(feat.columns) - 2}개")
