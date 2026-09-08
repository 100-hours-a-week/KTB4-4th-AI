"""대화가 끝나고 나오는 것.

  ① profile  — 계약(TasteProfile)으로 확정한 취향 프로필
  ② summary  — 사람이 읽는 요약문 (친구 공개 항목만)
  ③ queries  — 임베딩에 던질 검색 문장
  ④ filters  — SQL WHERE 에 들어갈 조건
  ⑤ weights  — 순위 조정에만 쓰는 항목

여기까지가 이 하네스의 범위다. 임베딩 호출과 벡터 검색은 하지 않는다.
`queries` 가 임베딩 단계에 그대로 넘어갈 입력이다.

모델이 하는 건 문장 쓰기뿐이다. mode(self/gift) 판정은 서버가 칸 이름으로 정한다.
"""

from __future__ import annotations

import llm
from schema import State

# 검색 문장이 어느 목록에 쓰이는지 — 칸에서 결정적으로 파생된다
QUERY_MODE = {
    "interests": "both",
    "hobbies": "both",
    "wants": "both",
    "unaffordable": "gift",   # 못 산 것은 선물의 핵심 재료
    "consumables": "self",    # 소모품은 선물에서 뺀다
}

# 보류 이유별 선물 신호 강도. 순위 단계에서 쓰지만 쿼리에 실어 넘긴다.
GIFT_WEIGHT = {
    "justification": 1.5,   # "나한테 사주긴 좀 그래" — 돈이 있어도 스스로 해결되지 않는다
    "price": 1.4,           # "비싸서 못 샀어"
    "timing": 1.15,         # "언젠가 사고 싶어"
}

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "rewritten": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "text": {"type": "string"}},
                "required": ["source", "text"],
                "additionalProperties": False,
            },
        },
        "derived": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source": {"type": "string"}, "text": {"type": "string"}},
                "required": ["source", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["rewritten", "derived"],
    "additionalProperties": False,
}

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
    "additionalProperties": False,
}


REWRITE_SYSTEM = """너는 사용자의 취향 항목을 **상품 검색용 문장**으로 바꾸는 역할이다.

## 왜 바꾸나

사용자는 "짐 많아지는 게 싫어요" 라고 말하고, 상품 설명에는 "초경량 220g 접이식" 이라고 적혀 있다.
겹치는 말이 없어서 그대로는 못 찾는다. 그래서 **상품 설명과 비슷한 결의 문장**으로 바꾼다.

## rewritten — 취향 항목 다듬기

받은 항목 하나당 문장 하나를 만든다. 형태는 이렇다.

  "무엇을 하는 데 쓰는 물건인지 + 어떤 상황에서 쓰는지"

예시
  "캠핑장에서 핸드드립"  →  "야외나 캠핑장에서 커피를 직접 내려 마실 때 쓰는 도구"
  "스웨터 뜨기"          →  "손으로 뜨개질을 할 때 쓰는 실과 도구"
  "러닝 워치"            →  "달리기 기록을 재는 손목 착용 기기"

- 상품 이름을 지어내지 않는다. 브랜드나 모델명을 넣지 않는다.
- 한 문장, 40자 안쪽.
- `source` 는 받은 항목을 **글자 그대로** 옮긴다.

## derived — 생활 맥락에서 필요한 것 뽑기

생활 맥락 항목은 그대로 검색하면 엉뚱한 게 걸린다.
("1인 가구" 로 검색하면 "1인용 밥솥" 이 나온다)

그래서 **그 생활에서 반복되는 상황**을 뽑고, 그 상황에 필요한 물건을 문장으로 쓴다.

예시
  "버스로 왕복 2시간 통학"  →  "이동 중에 목을 받쳐주는 물건"
                              "이동 중에 마실 것을 담는 용기"
  "하루 4시간 반 수면"      →  "짧게 자도 잘 자게 도와주는 물건"
  "귀가 후 야간 작업"       →  "밤에 오래 앉아 작업할 때 쓰는 물건"

- 항목 하나에서 두세 개까지 나와도 된다. 억지로 늘리지 않는다.
- 뽑을 게 없는 항목은 건너뛴다.
- 사용자가 힘들다고 한 것을 지적하는 문장은 쓰지 않는다. 필요한 물건만 적는다.
- `source` 는 근거가 된 생활 맥락 항목을 그대로 옮긴다."""


SUMMARY_SYSTEM = """너는 사용자의 취향을 소개하는 짧은 글을 쓰는 역할이다.
이 글은 **선물을 고르려는 친구가 읽는다.**

## 무엇을 쓰나

- 이 사람이 무엇에 관심이 있고 무엇을 즐기는지
- 물건을 고를 때 어떤 걸 선호하고 무엇을 피하는지

3~4문장. 한 문단으로 자연스럽게 이어 쓴다. 존댓말.

## 어떻게 쓰나

**항목을 나열하지 않는다.** 항목들을 엮어서 "어떤 사람인지"가 드러나게 쓴다.

나쁜 예 — 그냥 이어붙임
  캠핑을 좋아하시고, 핸드드립을 좋아하시고, 가벼운 것을 선호하시고,
  향이 강한 것을 싫어하세요.

좋은 예
  주말이면 밖으로 나가는 걸 좋아하세요. 캠핑장에서 직접 커피를 내려 드실 만큼
  손으로 하는 일을 즐기시고요. 장비는 가볍고 부피 작은 쪽을 고르시는 편이라
  짐이 늘어나는 물건은 잘 안 쓰세요.

## 절대 넣지 않는 것

- **선물·추천 이야기.** "선물 고르실 때" 같은 문장을 쓰지 않는다
- **전언 표현.** "들었어요", "알고 있습니다", "~라고 하시더라고요" 를 쓰지 않는다
- **받은 항목에 없는 말.** "복잡한 도구를 싫어한다" 처럼 그럴듯하게 덧붙이지 않는다
- 건강·수면·피로·체력 이야기
- 돈 사정, 직업이나 학업 상태, 취업 준비 여부
- 구체적인 지명·회사·학교 이름, 시간표
- 받은 항목에 없는 내용

받은 게 적으면 짧게 쓴다. 억지로 늘리지 않는다.
한 항목만 있으면 한 문장이어도 된다."""


def _rewrite(s: State) -> tuple[list[dict], list[dict]]:
    q = s.query_items
    life = s.by_field("lifestyle")
    if not q and not life:
        return [], []

    lines = []
    if q:
        lines.append("[취향 항목]")
        lines += [f"- {i.value}" for i in q]
    if life:
        lines.append("\n[생활 맥락]")
        lines += [f"- {i.value}" for i in life]

    out = llm.json_chat(
        [{"role": "system", "content": REWRITE_SYSTEM},
         {"role": "user", "content": "\n".join(lines)}],
        REWRITE_SCHEMA,
    )
    return out.get("rewritten", []), out.get("derived", [])


def _summarize(s: State) -> str:
    # 요약은 친구가 읽는 글이다. 취향·관심사만 넣는다.
    # lifestyle 은 "만성 피로", "취업 준비 중" 같은 게 섞이므로 통째로 뺀다.
    # owned·consumables·constraints·unaffordable 도 요약에 넣을 성격이 아니다.
    src = s.by_field("interests", "hobbies", "preferences", "dislikes")
    src = [i for i in src if i.visibility == "friends"]
    if not src:
        return ""

    lines = []
    for label, fields in [("좋아하는 것", ("interests", "hobbies")),
                          ("고를 때 선호", ("preferences",)),
                          ("피하는 것", ("dislikes",))]:
        got = [i.value for i in src if i.field in fields]
        if got:
            lines.append(f"{label}: " + " · ".join(got))
    if s.axes:
        lines.append("판단 기준: " + " · ".join(s.axes))

    # 요약은 스키마를 채우는 일이 아니라 글을 쓰는 일이다.
    # 추출용 작은 모델 말고 응답 모델(더 큰 쪽)에 맡긴다. 대기 경로가 아니라 느려도 된다.
    out = "".join(llm.stream_chat(
        [{"role": "system", "content": SUMMARY_SYSTEM},
         {"role": "user", "content": "\n".join(lines)}],
        model=llm.RESPONSE_MODEL, temperature=0.5,
    ))
    return llm.strip_meta(out).strip().strip('"')


def build(s: State) -> dict:
    """대화 상태 → 임베딩·검색 단계에 넘길 것.

    순서는 1단계 최종본의 내부 처리 순서를 그대로 따른다.
      세션 상태를 프로필로 확정
      → query 역할 항목을 상품 검색 문장으로 재작성
      → lifestyle 항목에서 파생 쿼리 생성 (self 전용)
      → 요약문 생성 (친구 공개 항목만)
      → [여기까지. 다음 단계인 임베딩·벡터 검색은 이 하네스 밖이다]
    """
    rewritten, derived = _rewrite(s)

    by_value = {i.value: i for i in s.items}
    queries: list[dict] = []

    # ① 취향 항목에서 나온 쿼리
    for r in rewritten:
        item = by_value.get(r.get("source", ""))
        if not item or item.link_role != "query":
            continue
        queries.append({
            "text": r.get("text", "").strip(),
            "source": item.value,
            "field": item.field,
            "kind": "direct",
            "mode": QUERY_MODE.get(item.field, "both"),
            "confidence": round(item.confidence, 2),
            "giftWeight": GIFT_WEIGHT.get(item.deferral_reason or "", 1.0),
        })

    # ② 생활 맥락에서 파생된 쿼리 — self 목록에만 쓴다.
    #    "너 잠 못 자니까" 를 친구에게 노출하지 않기 위해서다.
    for d in derived:
        item = by_value.get(d.get("source", ""))
        queries.append({
            "text": d.get("text", "").strip(),
            "source": d.get("source", ""),
            "field": "lifestyle",
            "kind": "derived",
            "mode": "self",
            # 한 다리 건너 나온 쿼리다. 직접 말한 것과 같은 무게로 두지 않는다.
            "confidence": round((item.confidence * 0.7) if item else 0.5, 2),
            "giftWeight": 1.0,
        })

    queries = [q for q in queries if q["text"]]

    # 임베딩은 문장마다 한 번씩 부르지 않는다. 중복을 걷어내고 한 번에 넘긴다.
    seen: set[str] = set()
    batch: list[str] = []
    for q in queries:
        if q["text"] not in seen:
            seen.add(q["text"])
            batch.append(q["text"])

    summary = _summarize(s)

    return {
        "profile": s.to_profile(summary=summary),
        "summary": summary,
        "queries": queries,
        "embeddingBatch": batch,
        "filters": {
            "dislikes": [i.value for i in s.by_field("dislikes")],
            "constraints": [i.value for i in s.by_field("constraints")],
            "owned": [i.value for i in s.by_field("owned")],
        },
        "weights": [i.value for i in s.by_field("preferences", "lifestyle")],
        "axes": list(s.axes),
    }
