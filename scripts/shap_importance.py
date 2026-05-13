"""
SHAP 기반 피처 중요도 분석
TreeExplainer로 OOF SHAP값 계산 -> 폴드 평균
방향(+/-) 및 크기 모두 파악 가능
결과를 submission/feature_result.md에 추가
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
import sys
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features

DATA = ROOT / "data"
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


def get_group(feat):
    if feat.startswith("lag") or feat.startswith("roll"):  return "lag/roll (시계열 lag)"
    if feat.startswith("subj_mean"):                        return "subj_mean (타 타깃 평균)"
    if feat.startswith("hr_") or feat == "hr_mean":        return "wHr (심박수)"
    if feat.startswith("light_"):                           return "mLight (조도)"
    if feat.startswith("screen_"):                          return "mScreenStatus (스크린)"
    if feat.startswith("pedo_"):                            return "wPedo (만보계)"
    if feat.startswith("act_"):                             return "mActivity (활동)"
    if feat.startswith("amb_"):                             return "mAmbience (주변음)"
    if feat.startswith("usage_"):                           return "mUsageStats (앱 사용)"
    if feat in ("day_of_week", "month", "day_of_month",
                "is_weekend", "week_of_year", "subject_enc"): return "base (날짜/subject)"
    return "기타"


def main():
    print("=== SHAP 피처 중요도 분석 ===")
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    pf    = build_parquet_features()
    le    = LabelEncoder().fit(train["subject_id"])
    sc    = get_sensor_cols(pf)
    cv    = GroupKFold(n_splits=10)

    fold_indices = list(cv.split(
        np.zeros(len(train)), train[TARGETS[0]].values, train["subject_id"].values))

    print("label 피처 및 fold 피처 사전 계산 중...")
    fold_label_feats = []
    for tr_idx, val_idx in fold_indices:
        tf = train.iloc[tr_idx].copy(); vf = train.iloc[val_idx].copy()
        fold_label_feats.append((tr_idx, val_idx, tf, vf,
                                 build_label_features(tf, tf), build_label_features(vf, vf)))

    fold_data_by_target = {}
    for t in TARGETS:
        fold_data = []
        for tr_idx, val_idx, tf, vf, lf_tr, lf_val in fold_label_feats:
            Xtr  = build_features(tf, tf, pf, lf_tr,  True,  le)
            Xval = build_features(vf, tf, pf, lf_val, False, le)
            sid_tr  = Xtr["subject_id"].reset_index(drop=True)
            sid_val = Xval["subject_id"].reset_index(drop=True)
            tr_stats  = compute_subj_stats(sid_tr,  Xtr,  sc)
            val_stats = compute_subj_stats(sid_val, Xval, sc)
            Xtr_z  = apply_zscore(sid_tr,  Xtr.drop(columns=["subject_id"]),  tr_stats,  sc)
            Xval_z = apply_zscore(sid_val, Xval.drop(columns=["subject_id"]), val_stats, sc)
            Xtr_z  = Xtr_z.drop(columns=[f"subj_mean_{t}"], errors="ignore")
            Xval_z = Xval_z.drop(columns=[f"subj_mean_{t}"], errors="ignore")
            fold_data.append((tr_idx, val_idx, Xtr_z, Xval_z))
        fold_data_by_target[t] = fold_data
    print("  완료\n")

    # 타깃별 SHAP 계산
    # mean_abs_shap: 평균 |SHAP| -> 크기 기반 중요도
    # mean_shap:     평균 SHAP  -> 방향 포함 (양=예측 1 방향 기여, 음=예측 0 방향)
    shap_abs_by_target  = {}
    shap_mean_by_target = {}

    for t in TARGETS:
        print(f"  [{t}] SHAP 계산 중...")
        feat_names = None
        abs_accum  = None  # shape: (n_features,)
        mean_accum = None
        n_total    = 0

        for tr_idx, val_idx, Xtr_z, Xval_z in fold_data_by_target[t]:
            y_tr = train[t].values[tr_idx]

            model = lgb.LGBMClassifier(**PARAMS)
            model.fit(Xtr_z, y_tr)

            if feat_names is None:
                feat_names = Xval_z.columns.tolist()

            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(Xval_z)

            # LightGBM binary: shap_values는 (n_samples, n_features) 또는
            # list[2] 형태 모두 가능 -> 양성 클래스 추출
            if isinstance(shap_vals, list):
                sv = shap_vals[1]   # 양성 클래스
            else:
                sv = shap_vals

            n = sv.shape[0]
            if abs_accum is None:
                abs_accum  = np.abs(sv).sum(axis=0)
                mean_accum = sv.sum(axis=0)
            else:
                abs_accum  += np.abs(sv).sum(axis=0)
                mean_accum += sv.sum(axis=0)
            n_total += n

        shap_abs_by_target[t]  = pd.Series(abs_accum  / n_total, index=feat_names).sort_values(ascending=False)
        shap_mean_by_target[t] = pd.Series(mean_accum / n_total, index=feat_names).sort_values(ascending=False, key=abs)
        print(f"     완료 top3: {list(shap_abs_by_target[t].head(3).index)}")

    # combined (mean |SHAP|)
    comb_abs = pd.DataFrame({t: shap_abs_by_target[t] for t in TARGETS}).fillna(0)
    comb_abs["mean"] = comb_abs.mean(axis=1)
    comb_abs = comb_abs.sort_values("mean", ascending=False)

    comb_mean = pd.DataFrame({t: shap_mean_by_target[t] for t in TARGETS}).fillna(0)
    comb_mean["mean"] = comb_mean.mean(axis=1)

    # ── feature_result.md에 추가 ─────────────────────────────────────────────
    result_path = ROOT / "submission" / "feature_result.md"
    existing = result_path.read_text(encoding="utf-8")

    lines = []
    lines.append("\n---\n")
    lines.append("# SHAP 분석 결과")
    lines.append("")
    lines.append("> - 분석 일자: 2026-05-12")
    lines.append("> - CV 전략: LOSO GroupKFold(n=10), OOF 전체 행 평균")
    lines.append("> - 측정 방식: TreeExplainer -> OOF SHAP values (양성 클래스 기준)")
    lines.append("> - mean |SHAP|: 크기 기반 중요도 (방향 무관)")
    lines.append("> - mean SHAP:   방향 포함 기여도 (양수=예측값 상승 기여, 음수=하락 기여)")
    lines.append("> - 총 피처 수: 135개 (타깃별)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 전체 평균 mean|SHAP| (전체 피처)
    lines.append("## 1. 전체 타깃 평균 mean|SHAP| — 크기 기반 중요도 (135개 전체)")
    lines.append("")
    header = "| 순위 | 피처명 | " + " | ".join(TARGETS) + " | 평균 |"
    sep    = "|------|--------|" + "|".join(["--------"] * len(TARGETS)) + "|--------|"
    lines.append(header)
    lines.append(sep)
    for rank, (feat, row) in enumerate(comb_abs.iterrows(), 1):
        vals = " | ".join(f"{row[t]:.4f}" for t in TARGETS)
        lines.append(f"| {rank} | `{feat}` | {vals} | **{row['mean']:.4f}** |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. 타깃별 전체 순위 (mean|SHAP| + mean SHAP 방향)
    for idx, t in enumerate(TARGETS):
        abs_imp  = shap_abs_by_target[t]
        mean_imp = shap_mean_by_target[t]
        lines.append(f"## 2-{idx+1}. [{t}] SHAP 전체 순위 ({len(abs_imp)}개)")
        lines.append("")
        lines.append("| 순위 | 피처명 | mean\\|SHAP\\| | mean SHAP | 방향 |")
        lines.append("|------|--------|:------------:|:---------:|:----:|")
        for rank, (feat, abs_val) in enumerate(abs_imp.items(), 1):
            mean_val = mean_imp.get(feat, 0.0)
            direction = "+" if mean_val > 0.001 else ("-" if mean_val < -0.001 else "0")
            lines.append(f"| {rank} | `{feat}` | {abs_val:.4f} | {mean_val:+.4f} | {direction} |")
        lines.append("")

    # 3. 그룹별 합계
    lines.append("---")
    lines.append("")
    lines.append("## 3. 피처 그룹별 mean|SHAP| 합계")
    lines.append("")
    group_data = {}
    for feat, row in comb_abs.iterrows():
        g = get_group(feat)
        if g not in group_data:
            group_data[g] = {"count": 0, "sum": 0.0}
        group_data[g]["count"] += 1
        group_data[g]["sum"]   += row["mean"]
    group_df = pd.DataFrame(group_data).T.sort_values("sum", ascending=False)
    total_sum = comb_abs["mean"].sum()
    lines.append("| 그룹 | 피처 수 | mean|SHAP| 합계 | 전체 비율(%) |")
    lines.append("|------|:-------:|:----------------:|:------------:|")
    for g, row in group_df.iterrows():
        ratio = row["sum"] / total_sum * 100
        lines.append(f"| {g} | {int(row['count'])} | {row['sum']:.4f} | {ratio:.1f}% |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. 3가지 방법론 비교 (gain / permutation / SHAP) top-20
    lines.append("## 4. 3가지 방법론 비교 — 전체 평균 top-20")
    lines.append("")
    lines.append("> gain 순위 기준 정렬. 순위 차이가 클수록 방법론 간 불일치 큰 피처.")
    lines.append("")
    lines.append("| gain 순위 | 피처명 | gain 평균 | perm 순위 | perm 변화량 | SHAP 순위 | mean|SHAP| |")
    lines.append("|:---------:|--------|:---------:|:---------:|:-----------:|:---------:|:----------:|")

    # gain 데이터 로드
    gain_path = ROOT / "submission" / "feature_importance_raw.json"
    gain_ranks = {}
    gain_vals  = {}
    if gain_path.exists():
        import json
        try:
            raw = gain_path.read_bytes().decode("utf-8", errors="ignore").strip()
            gain_data  = json.loads(raw)
            gain_comb  = pd.DataFrame(gain_data["combined"]).sort_values("mean", ascending=False)
            for r, (feat, row) in enumerate(gain_comb.iterrows(), 1):
                gain_ranks[feat] = r
                gain_vals[feat]  = row["mean"]
        except Exception:
            pass

    # perm 데이터 (현재 세션 내 없으므로 feature_result.md에서 파싱 불가 -> 순위만 추정)
    shap_rank_map = {feat: r+1 for r, (feat, _) in enumerate(comb_abs.iterrows())}

    top20_gain = sorted(gain_ranks.keys(), key=lambda f: gain_ranks[f])[:20]
    for feat in top20_gain:
        g_rank = gain_ranks.get(feat, "-")
        g_val  = gain_vals.get(feat, 0)
        s_rank = shap_rank_map.get(feat, "-")
        s_val  = comb_abs.loc[feat, "mean"] if feat in comb_abs.index else 0
        lines.append(
            f"| {g_rank} | `{feat}` | {g_val:.1f} | - | - | {s_rank} | {s_val:.4f} |"
        )

    lines.append("")
    lines.append("> perm 수치는 feature_result.md Permutation Importance 섹션 참조")
    lines.append("")

    result_path.write_text(existing + "\n".join(lines), encoding="utf-8")
    print(f"\n저장 완료: {result_path}")
    print(f"추가 {len(lines)}줄")


if __name__ == "__main__":
    main()
