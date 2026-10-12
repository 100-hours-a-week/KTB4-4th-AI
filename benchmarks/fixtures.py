import hashlib
import json

VERSION = "taste-smoke-v1"
SYSTEM = """한국어 발화의 명시적인 취향을 JSON으로 추출하라.
likes: 좋아하는 물건·음식·스타일의 명사 표현.
dislikes: 싫어하거나 피하는 물건·음식·스타일의 명사 표현.
activities: 현재 즐긴다고 명시한 활동의 명사 표현.
추측하지 말고, 취향이 없거나 철회한 항목은 넣지 않는다.
정정이 있다면 최신 진술만 반영한다. 중복 없이 아래 세 배열만 출력한다.
입력의 '사용자:' 줄만 취향 근거로 삼는다. 빈 항목은 빈 배열로 반환한다."""
SCHEMA = {
    "type": "object",
    "properties": {
        field: {"type": "array", "items": {"type": "string"}}
        for field in ("likes", "dislikes", "activities")
    },
    "required": ["likes", "dislikes", "activities"],
    "additionalProperties": False,
}
CASES = [
    {
        "id": "explicit-like",
        "label": "명시적 선호",
        "input": "사용자: 나는 커피를 좋아해.",
        "expected": {"likes": ["커피"], "dislikes": [], "activities": []},
    },
    {
        "id": "negative",
        "label": "비선호 분리",
        "input": "사용자: 향수는 싫어해.",
        "expected": {"likes": [], "dislikes": ["향수"], "activities": []},
    },
    {
        "id": "mixed",
        "label": "선호와 비선호",
        "input": "사용자: 커피는 좋아하고 향수는 싫어해.",
        "expected": {"likes": ["커피"], "dislikes": ["향수"], "activities": []},
    },
    {
        "id": "activity",
        "label": "활동 추출",
        "input": "사용자: 주말에 캠핑을 즐겨.",
        "expected": {"likes": [], "dislikes": [], "activities": ["캠핑"]},
    },
    {
        "id": "correction",
        "label": "취향 정정",
        "input": "사용자: 커피를 좋아해.\n사용자: 정정할게. 커피는 싫어하고 차를 좋아해.",
        "expected": {"likes": ["차"], "dislikes": ["커피"], "activities": []},
    },
    {
        "id": "unknown",
        "label": "미확정 취향",
        "input": "사용자: 향수를 좋아하는지 아직 모르겠어.",
        "expected": {"likes": [], "dislikes": [], "activities": []},
    },
    {
        "id": "other-person",
        "label": "타인의 취향 제외",
        "input": "사용자: 친구는 커피를 좋아해. 내 취향은 아직 말 안 했어.",
        "expected": {"likes": [], "dislikes": [], "activities": []},
    },
    {
        "id": "assistant-evidence",
        "label": "AI 제안 제외",
        "input": "AI: 캠핑과 러닝을 좋아하시죠?\n사용자: 아니, 나는 수영을 즐겨.",
        "expected": {"likes": [], "dislikes": [], "activities": ["수영"]},
    },
]


def fixture_hash():
    data = {"cases": CASES, "system": SYSTEM, "schema": SCHEMA}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def evaluate(text, expected):
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return {"schema_valid": False, "exact_match": False, "reason": "JSON 파싱 실패"}
    valid = (
        isinstance(data, dict)
        and set(data) == set(expected)
        and all(
            isinstance(value, list) and all(isinstance(item, str) for item in value)
            for value in data.values()
        )
    )
    exact = valid and all(
        len(data[key]) == len(set(data[key]))
        and {v.strip() for v in data[key]} == set(expected[key])
        for key in expected
    )
    return {
        "schema_valid": valid,
        "exact_match": exact,
        "reason": "정답 일치" if exact else "내용 불일치" if valid else "스키마 위반",
        "parsed": data,
    }
