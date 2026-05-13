"""
feature_result.md 생성 스크립트
LightGBM gain importance 전체 결과를 마크다운으로 저장
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features

DATA = Path(__file__).parent.parent / "data"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]

PARAMS = {
    "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
    "random_state": 42, "num_leaves": 31, "learning_rate": 0.05,
    "n_estimators": 300, "min_child_samples": 20,
    "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
}

def get_sensor_cols(pf):
    return [c for c in pf.columns if c not in ("subject_id", "date")]

def compute_subj_stats(sids, X, sc):
    avail = [c for c in sc if c in X.columns]
    tmp = X[avail].copy(); tmp["subject_id"] = sids.values
    return tmp.groupby("subject_id")[avail].agg(["mean", "std"])

def apply_zscore(sids, X, stats, sc):
    X = X.copy()
    for col in sc:
        if col not in X.columns or (col, "mean") not in stats.columns:
            continue
        m = sids.map(stats[(col, "mean")]); s = sids.map(stats[(col, "std")])
        v = s.notna() & (s > 0); X[col] = X[col].astype(float)
        X.loc[v, col] = (X.loc[v, col] - m[v]) / s[v]
    return X

def add_date_features(df):
    dt = pd.to_datetime(df["sleep_date"]); df = df.copy()
    df["day_of_week"]  = dt.dt.dayofweek
    df["month"]        = dt.dt.month
    df["day_of_month"] = dt.dt.day
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df

def add_subj_mean(df, ref, is_train):
    ss = ref.groupby("subject_id")[TARGETS].sum()
    sc = ref.groupby("subject_id")[TARGETS].count()
    if is_train:
        for t in TARGETS:
            df[f"subj_mean_{t}"] = (df["subject_id"].map(ss[t]) - df[t]) / \
                                    (df["subject_id"].map(sc[t]) - 1).clip(lower=1)
    else:
        sm = ss / sc
        for t in TARGETS:
            df[f"subj_mean_{t}"] = df["subject_id"].map(sm[t])
    return df

def build_features(df, ref, pf, lf, is_train, le):
    df = add_date_features(df); df = add_subj_mean(df, ref, is_train)
    df["subject_enc"] = df["subject_id"].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1)
    df = df.merge(pf, left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")
    df = df.merge(lf, left_on=["subject_id", "lifelog_date"],
                  right_on=["subject_id", "date"], how="left").drop(columns=["date"], errors="ignore")
    base = (["subject_enc", "day_of_week", "month", "day_of_month", "is_weekend", "week_of_year"]
            + [f"subj_mean_{t}" for t in TARGETS])
    pc = [c for c in pf.columns if c not in ("subject_id", "date")]
    lc = [c for c in lf.columns if c not in ("subject_id", "date")]
    all_cols = ["subject_id"] + base + pc + lc
    return df[[c for c in all_cols if c in df.columns]].reset_index(drop=True)

print("=== 피처 중요도 계산 중 ===")
train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
pf    = build_parquet_features()
le    = LabelEncoder().fit(train["subject_id"])
sc    = get_sensor_cols(pf)
cv    = GroupKFold(n_splits=10)
fold_indices = list(cv.split(np.zeros(len(train)), train[TARGETS[0]].values, train["subject_id"].values))

fold_label_feats = []
for tr_idx, val_idx in fold_indices:
    tf = train.iloc[tr_idx].copy(); vf = train.iloc[val_idx].copy()
    fold_label_feats.append((tr_idx, val_idx, tf, vf,
                             build_label_features(tf, tf), build_label_features(vf, vf)))

importance_by_target = {}
for t in TARGETS:
    print(f"  {t} 계산 중...")
    fold_imps = []; feat_names = None
    for tr_idx, val_idx, tf, vf, lf_tr, lf_val in fold_label_feats:
        Xtr = build_features(tf, tf, pf, lf_tr, True, le)
        sid_tr = Xtr["subject_id"].reset_index(drop=True)
        tr_stats = compute_subj_stats(sid_tr, Xtr, sc)
        Xtr_z = apply_zscore(sid_tr, Xtr.drop(columns=["subject_id"]), tr_stats, sc)
        Xtr_z = Xtr_z.drop(columns=[f"subj_mean_{t}"], errors="ignore")
        if feat_names is None:
            feat_names = Xtr_z.columns.tolist()
        m = lgb.LGBMClassifier(**PARAMS)
        m.fit(Xtr_z, train[t].values[tr_idx])
        fold_imps.append(m.feature_importances_)
    mean_imp = np.array(fold_imps).mean(axis=0)
    importance_by_target[t] = pd.Series(mean_imp, index=feat_names).sort_values(ascending=False)

comb = pd.DataFrame({t: importance_by_target[t] for t in TARGETS}).fillna(0)
comb["mean"] = comb.mean(axis=1)
comb = comb.sort_values("mean", ascending=False)
by_t = importance_by_target
print("  완료\n")

lines = []
lines.append("# 피처 중요도 분석 결과 (LightGBM gain 기준)")
lines.append("")
lines.append("> - 분석 일자: 2026-05-12")
lines.append("> - CV 전략: LOSO GroupKFold(n=10), 폴드 평균 gain importance")
lines.append("> - 파이프라인: z-score 정규화, 타깃별 subj_mean_{t} 제거")
lines.append("> - 총 피처 수: 135개 (타깃별)")
lines.append("> - 비교 목적: 향후 Permutation Importance / SHAP 결과와 대조")
lines.append("")
lines.append("---")
lines.append("")

# ── 1. 전체 평균 (모든 피처) ──────────────────────────────────────────────────
lines.append("## 1. 전체 타깃 평균 importance (135개 전체)")
lines.append("")
header = "| 순위 | 피처명 | " + " | ".join(TARGETS) + " | 평균 |"
sep    = "|------|--------|" + "|".join(["------"] * len(TARGETS)) + "|------|"
lines.append(header)
lines.append(sep)
for rank, (feat, row) in enumerate(comb.iterrows(), 1):
    vals = " | ".join(f"{row[t]:>6.1f}" for t in TARGETS)
    lines.append(f"| {rank} | `{feat}` | {vals} | **{row['mean']:.1f}** |")

lines.append("")
lines.append("---")
lines.append("")

# ── 2. 타깃별 전체 랭킹 ──────────────────────────────────────────────────────
for idx, t in enumerate(TARGETS):
    imp   = by_t[t]
    total = imp.sum()
    lines.append(f"## 2-{idx+1}. [{t}] 타깃 importance 전체 순위 ({len(imp)}개)")
    lines.append("")
    lines.append(f"> 전체 importance 합: {total:.1f}")
    lines.append("")
    lines.append("| 순위 | 피처명 | importance | 비율(%) |")
    lines.append("|------|--------|:----------:|:-------:|")
    for rank, (feat, val) in enumerate(imp.items(), 1):
        ratio = val / total * 100
        lines.append(f"| {rank} | `{feat}` | {val:.1f} | {ratio:.2f}% |")
    lines.append("")

lines.append("---")
lines.append("")

# ── 3. 그룹별 합계 ────────────────────────────────────────────────────────────
def get_group(feat):
    if feat.startswith("lag") or feat.startswith("roll"):
        return "lag/roll (시계열 lag)"
    if feat.startswith("subj_mean"):
        return "subj_mean (타 타깃 평균)"
    if feat.startswith("hr_") or feat == "hr_mean":
        return "wHr (심박수)"
    if feat.startswith("light_"):
        return "mLight (조도)"
    if feat.startswith("screen_"):
        return "mScreenStatus (스크린)"
    if feat.startswith("pedo_"):
        return "wPedo (만보계)"
    if feat.startswith("act_"):
        return "mActivity (활동)"
    if feat.startswith("amb_"):
        return "mAmbience (주변음)"
    if feat.startswith("usage_"):
        return "mUsageStats (앱 사용)"
    if feat in ("day_of_week", "month", "day_of_month", "is_weekend",
                "week_of_year", "subject_enc"):
        return "base (날짜/subject)"
    return "기타"

group_data = {}
for feat, row in comb.iterrows():
    g = get_group(feat)
    if g not in group_data:
        group_data[g] = {"count": 0, "mean_sum": 0.0}
    group_data[g]["count"]    += 1
    group_data[g]["mean_sum"] += row["mean"]

group_df    = pd.DataFrame(group_data).T.sort_values("mean_sum", ascending=False)
total_mean  = comb["mean"].sum()

lines.append("## 3. 피처 그룹별 importance 합계")
lines.append("")
lines.append("| 그룹 | 피처 수 | importance 합계 | 전체 비율(%) |")
lines.append("|------|:-------:|:---------------:|:------------:|")
for g, row in group_df.iterrows():
    ratio = row["mean_sum"] / total_mean * 100
    lines.append(
        f"| {g} | {int(row['count'])} | {row['mean_sum']:.1f} | {ratio:.1f}% |"
    )

lines.append("")
lines.append("---")
lines.append("")

# ── 4. 주요 발견 요약 ─────────────────────────────────────────────────────────
lines.append("## 4. 주요 발견 요약")
lines.append("")
lines.append("| 구분 | 내용 |")
lines.append("|------|------|")
lines.append("| **공통 top 피처** | `light_mean_presleep`(1위), `hr_min_val`(2위), `light_mean_morning`(3위), `screen_ratio_presleep`(4위) — 조도·심박수·스크린 시간이 핵심 신호 |")
lines.append("| **조도(mLight) 그룹** | 피처 수 대비 importance 비율 높음 — 취침 전/아침/저녁 빛 노출이 수면 품질의 강한 예측 변수 |")
lines.append("| **mUsageStats 개별 유효성** | 전체 추가 시 과적합이었으나 개별 피처(usage_apps_afternoon, usage_apps_morning 등)는 상위권 — 선별 추가 가능성 있음 |")
lines.append("| **lag/roll 피처 낮은 기여** | 42개 피처 중 top-20 진입은 Q3의 roll3_Q1(3위), S2의 roll14_S3(9위), S3의 roll14_S4(14위) 등 극소수 |")
lines.append("| **Q1 특이사항** | `hr_mean`이 Q1에서만 1위 (전체 평균 30위) — Q1(수면의 질) 예측에 전체 심박수 평균이 특히 중요 |")
lines.append("| **S1 특이사항** | `subj_mean_Q1`이 1위 — 총 수면 시간(S1)과 수면의 질 설문(Q1)의 강한 교차 상관 |")
lines.append("| **S2 특이사항** | mUsageStats 피처(usage_apps_afternoon, usage_ms_evening 등)가 상위 집중 — 수면 효율과 앱 사용 패턴 상관 |")

out_path = ROOT / "submission" / "feature_result.md"
out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"저장 완료: {out_path}")
print(f"총 {len(lines)}줄 작성")
