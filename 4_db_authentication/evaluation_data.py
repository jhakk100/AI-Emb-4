"""FAR, FRR, EER 계산을 위해선 정답을 알려주고 하는 테스트가 필요"""

import json
from datetime import datetime
from pathlib import Path

from models import ROOT


DATA_FILE = ROOT / "data" / "evaluation_data.json"


def load_data():
    """저장된 평가 데이터를 불러옵니다."""
    if not DATA_FILE.exists():
        return {"tests": []}

    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError("평가 데이터 형식이 올바르지 않습니다.")

        if "tests" not in data:
            data["tests"] = []

        return data

    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"평가 데이터를 읽을 수 없습니다: {DATA_FILE}\n{exc}"
        ) from exc


def save_data(data):
    """평가 데이터를 저장합니다."""
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def add_test_result(
    actual_user,
    recognized_user,
    similarity,
    threshold,
    test_type,
):
    """
    평가 결과를 하나 저장합니다.

    test_type:
        genuine  -> 본인 테스트
        impostor -> 타인 테스트
    """

    if test_type not in ("genuine", "impostor"):
        raise ValueError(
            "test_type은 genuine 또는 impostor여야 합니다."
        )

    data = load_data()

    result = {
        "time": datetime.now().isoformat(),
        "actual_user": actual_user,
        "recognized_user": recognized_user,
        "similarity": float(similarity),
        "threshold": float(threshold),
        "type": test_type,
    }

    data["tests"].append(result)
    save_data(data)

    return result


def get_scores():
    """Genuine과 Impostor similarity 데이터를 분리해서 반환합니다."""
    data = load_data()

    genuine_scores = []
    impostor_scores = []

    for test in data["tests"]:
        if test.get("type") == "genuine":
            genuine_scores.append(float(test["similarity"]))

        elif test.get("type") == "impostor":
            impostor_scores.append(float(test["similarity"]))

    return genuine_scores, impostor_scores


def get_test_count():
    """전체 평가 데이터 개수를 반환합니다."""
    data = load_data()

    genuine = 0
    impostor = 0

    for test in data["tests"]:
        if test.get("type") == "genuine":
            genuine += 1
        elif test.get("type") == "impostor":
            impostor += 1

    return genuine, impostor
