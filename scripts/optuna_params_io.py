"""
Optuna best_params 중앙 저장소 헬퍼

키 형식: {model}_{feature_version}  예) lgbm_v2, catboost_v2, extratrees_v2
동일 피처 버전이면 어떤 앙상블 스크립트에서도 params 공유·재사용 가능

사용 예:
    from optuna_params_io import load_params, save_params

    # 로드 (없으면 None → Optuna 실행)
    cached = load_params("lgbm_v2")

    # 저장 (다른 키는 그대로 유지)
    save_params("lgbm_v2", best_lgbm)
"""

import json
from pathlib import Path

PARAMS_PATH = Path(__file__).parent.parent / "submission" / "optuna_params.json"


def load_params(key: str) -> dict | None:
    if not PARAMS_PATH.exists():
        return None
    with open(PARAMS_PATH, "r") as f:
        data = json.load(f)
    return data.get(key)


def save_params(key: str, params: dict) -> None:
    data = {}
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH, "r") as f:
            data = json.load(f)
    data[key] = params
    with open(PARAMS_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  params 저장 완료: '{key}' → {PARAMS_PATH.name}")
