"""
Permutation Importance 분석
각 피처를 랜덤 셔플 후 OOF LogLoss 변화량 측정
재학습 없이 예측만 재수행 → 실제 예측 기여도 직접 측정
결과를 submission/feature_result.md에 추가
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from label_features import build_label_features
from parquet_features_v2 import build_all as build_parquet_features

DATA = ROOT / "data"
TARGETS = ["Q1", "Q2", "Q3", "S1", "S2", "S3", "S4"]
N_REPEAT = 5  # 셔플 반복 횟수 (분산 감소)
RANDOM_SEED = 42

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


def permutation_importance_target(t, fold_data, y_all, n_repeat, rng):
    """타깃 t의 Permutation Importance 계산. LogLoss 증가량 반환."""
    feat_names = None
    feat_delta_folds = []

    for tr_idx, val_idx, Xtr_z, Xval_z in fold_data:
        y_tr  = y_all[tr_idx]
        y_val = y_all[val_idx]

        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(Xtr_z, y_tr)

        base_prob = model.predict_proba(Xval_z)[:, 1]
        base_ll   = log_loss(y_val, base_prob)

        if feat_names is None:
            feat_names = Xval_z.columns.tolist()

        feat_delta = {}
        for feat in feat_names:
            deltas = []
            for _ in range(n_repeat):
                Xval_shuf = Xval_z.copy()
                Xval_shuf[feat] = rng.permutation(Xval_shuf[feat].values)
                shuf_prob = model.predict_proba(Xval_shuf)[:, 1]
                shuf_ll   = log_loss(y_val, shuf_prob)
                deltas.append(shuf_ll - base_ll)
            feat_delta[feat] = np.mean(deltas)

        feat_delta_folds.append(feat_delta)

    # 폴드 평균
    result = {}
    for feat in feat_names:
        result[feat] = np.mean([fd[feat] for fd in feat_delta_folds])

    return pd.Series(result).sort_values(ascending=False)


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
    print("=== Permutation Importance 분석 ===")
    train = pd.read_csv(DATA / "ch2026_metrics_train.csv")
    pf    = build_parquet_features()
    le    = LabelEncoder().fit(train["subject_id"])
    sc    = get_sensor_cols(pf)
    cv    = GroupKFold(n_splits=10)
    rng   = np.random.default_rng(RANDOM_SEED)

    fold_indices = list(cv.split(
        np.zeros(len(train)), train[TARGETS[0]].values, train["subject_id"].values))

    print("label 피처 및 fold 피처 사전 계산 중...")
    fold_label_feats = []
    for tr_idx, val_idx in fold_indices:
        tf = train.iloc[tr_idx].copy(); vf = train.iloc[val_idx].copy()
        fold_label_feats.append((tr_idx, val_idx, tf, vf,
                                 build_label_features(tf, tf), build_label_features(vf, vf)))

    # 타깃별 fold 피처 사전 계산 (z-score 포함)
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

            Xtr_z  = Xtr_z.drop(columns=[f"subj_mean_{t}"],  errors="ignore")
            Xval_z = Xval_z.drop(columns=[f"subj_mean_{t}"], errors="ignore")

            fold_data.append((tr_idx, val_idx, Xtr_z, Xval_z))
        fold_data_by_target[t] = fold_data
    print("  완료\n")

    # 타깃별 Permutation Importance
    perm_by_target = {}
    for t in TARGETS:
        print(f"  [{t}] Permutation Importance 계산 중 (repeat={N_REPEAT}, folds=10)...")
        y_all = train[t].values
        perm_by_target[t] = permutation_importance_target(
            t, fold_data_by_target[t], y_all, N_REPEAT, rng)
        print(f"       완료 top3: {list(perm_by_target[t].head(3).index)}")

    # combined
    comb = pd.DataFrame({t: perm_by_target[t] for t in TARGETS}).fillna(0)
    comb["mean"] = comb.mean(axis=1)
    comb = comb.sort_values("mean", ascending=False)

    # ── feature_result.md에 추가 ─────────────────────────────────────────────
    result_path = ROOT / "submission" / "feature_result.md"
    existing = result_path.read_text(encoding="utf-8")

    lines = []
    lines.append("\n---\n")
    lines.append("# Permutation Importance 분석 결과")
    lines.append("")
    lines.append("> - 분석 일자: 2026-05-12")
    lines.append("> - CV 전략: LOSO GroupKFold(n=10), 폴드 평균")
    lines.append("> - 측정 방식: 각 피처를 랜덤 셔플 후 OOF LogLoss 변화량 (셔플 반복 5회 평균)")
    lines.append("> - 양수(+) = LogLoss 증가 = 중요한 피처 / 음수(-) = 셔플 후 오히려 개선 = 노이즈 가능성")
    lines.append("> - 총 피처 수: 135개 (타깃별)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 전체 평균
    lines.append("## 1. 전체 타깃 평균 Permutation Importance (135개 전체)")
    lines.append("")
    header = "| 순위 | 피처명 | " + " | ".join(TARGETS) + " | 평균 |"
    sep    = "|------|--------|" + "|".join(["--------"] * len(TARGETS)) + "|--------|"
    lines.append(header)
    lines.append(sep)
    for rank, (feat, row) in enumerate(comb.iterrows(), 1):
        vals = " | ".join(f"{row[t]:+.4f}" for t in TARGETS)
        lines.append(f"| {rank} | `{feat}` | {vals} | **{row['mean']:+.4f}** |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. 타깃별
    for idx, t in enumerate(TARGETS):
        imp   = perm_by_target[t]
        lines.append(f"## 2-{idx+1}. [{t}] Permutation Importance 전체 순위 ({len(imp)}개)")
        lines.append("")
        lines.append("| 순위 | 피처명 | LogLoss 변화량 | 해석 |")
        lines.append("|------|--------|:--------------:|------|")
        for rank, (feat, val) in enumerate(imp.items(), 1):
            interp = "중요" if val > 0.005 else ("무관" if val < -0.002 else "미미")
            lines.append(f"| {rank} | `{feat}` | {val:+.4f} | {interp} |")
        lines.append("")

    # 3. 그룹별 합계
    lines.append("---")
    lines.append("")
    lines.append("## 3. 피처 그룹별 Permutation Importance 합계")
    lines.append("")

    group_data = {}
    for feat, row in comb.iterrows():
        g = get_group(feat)
        if g not in group_data:
            group_data[g] = {"count": 0, "mean_sum": 0.0}
        group_data[g]["count"]    += 1
        group_data[g]["mean_sum"] += row["mean"]

    group_df = pd.DataFrame(group_data).T.sort_values("mean_sum", ascending=False)
    lines.append("| 그룹 | 피처 수 | LogLoss 변화량 합계 | 비고 |")
    lines.append("|------|:-------:|:-------------------:|------|")
    for g, row in group_df.iterrows():
        note = "핵심 신호" if row["mean_sum"] > 0.1 else ("유효" if row["mean_sum"] > 0 else "노이즈 가능성")
        lines.append(f"| {g} | {int(row['count'])} | {row['mean_sum']:+.4f} | {note} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. gain vs permutation 비교 (top-20)
    lines.append("## 4. LightGBM gain vs Permutation Importance 비교 (전체 평균 기준 top-20)")
    lines.append("")
    lines.append("| gain 순위 | 피처명 (gain) | gain 평균 | perm 순위 | 피처명 (perm) | perm 변화량 |")
    lines.append("|:---------:|---------------|:---------:|:---------:|---------------|:-----------:|")

    gain_path = ROOT / "submission" / "feature_importance_raw.json"
    if gain_path.exists():
        import json
        with open(gain_path, encoding="utf-8", errors="ignore") as f:
            raw = f.read().strip()
        try:
            gain_data = json.loads(raw)
            gain_comb = pd.DataFrame(gain_data["combined"]).sort_values("mean", ascending=False)
            perm_rank = {feat: r+1 for r, (feat, _) in enumerate(comb.iterrows())}
            for g_rank, (g_feat, g_row) in enumerate(gain_comb.head(20).iterrows(), 1):
                p_feat = comb.index[g_rank - 1]
                p_val  = comb.loc[p_feat, "mean"]
                p_rank = perm_rank.get(g_feat, "-")
                lines.append(
                    f"| {g_rank} | `{g_feat}` | {g_row['mean']:.1f} | "
                    f"{p_rank} | `{p_feat}` | {p_val:+.4f} |"
                )
        except Exception:
            lines.append("| (gain raw JSON 파싱 실패) | | | | | |")
    else:
        lines.append("| (feature_importance_raw.json 없음) | | | | | |")

    lines.append("")

    result_path.write_text(existing + "\n".join(lines), encoding="utf-8")
    print(f"\n저장 완료: {result_path}")
    print(f"추가 {len(lines)}줄")


if __name__ == "__main__":
    main()
