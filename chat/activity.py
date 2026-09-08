"""취향 프로필 → 메인 화면 행동 카드.

**사용자 본인 화면에만 나온다.** 선물·상품 이야기를 하지 않는다.

  ① 서버가 '무엇에 대해 만들지' 재료(seed)를 고른다 — 결정적, LLM 안 씀
  ② 모델은 그 재료로 문장을 쓴다. 안 엮이는 재료는 버린다
  ③ 서버가 걸러내고 갈래별로 잘라 3~5개를 낸다

`requires` 는 **화면에 내보내지 않는다.** 그 행동에 필요한 물건이고,
선물 리스트로 넘어가는 재료다. 사용자에게 보이면 "물건 팔려는 것"이 된다.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import llm
from schema import Item, State

# 갈래 — 각각 다른 재료에서 나온다
PAIR = "pair"              # 교집합: 관심사 둘이 겹치는 지점
PRECEDE = "precede"        # 선행 체험: 사기 전에 해보는 것
SUBSTITUTE = "substitute"  # 대체: 반복 구매를 다른 방식으로
FILL = "fill"              # 생활 보정: 반복되는 상황의 빈 자리

KIND_LABEL = {
    PAIR: "겹치는 관심사",
    PRECEDE: "사고 싶어하는 것",
    SUBSTITUTE: "반복해서 사는 것",
    FILL: "생활 맥락",
}

MAX_CARDS = 5
PER_KIND_SEED = 3   # 재료 단계 — 모델이 버릴 것을 감안해 넉넉히 넘긴다
PER_KIND_OUT = 2    # 출력 단계 — 한 갈래가 화면을 다 먹지 않게
SEED_BUDGET = 10
MIN_CONF = 0.5      # 이보다 낮은 항목은 재료로 쓰지 않는다


@dataclass
class Seed:
    id: str
    kind: str
    sources: list[str]
    score: float


@dataclass
class Card:
    title: str
    reason: str
    requires: list[str]   # 화면에 안 나간다
    effort: str
    kind: str
    sources: list[str]


# ────────────────────────────────────────────────────── ① 재료 고르기

def _pairs(s: State) -> list[Seed]:
    """관심사 둘을 붙인다. 어울리는지는 모델이 판단한다.

    `query_items` 를 쓰면 안 된다. wants·unaffordable·consumables 까지 딸려와서
    "캠핑 × 커피 캡슐" 같은 짝이 생기고, 그것들은 이미 자기 갈래를 갖고 있다.
    여기 재료는 **실제로 하거나 관심 있는 것**뿐이다.
    """
    q = sorted((i for i in s.by_field("interests", "hobbies")
                if i.confidence >= MIN_CONF),
               key=lambda i: -i.confidence)[:5]
    out = []
    for a, b in itertools.combinations(q, 2):
        # 같은 물건의 다른 표현이 짝이 되는 것을 막는다 ("커피" × "핸드드립 커피")
        if a.value in b.value or b.value in a.value:
            continue
        out.append(Seed("", PAIR, [a.value, b.value], a.confidence * b.confidence))
    return out


def _precede(s: State) -> list[Seed]:
    """갖고 싶다고 한 것 — 사기 전에 해볼 수 있는 행동으로 바꾼다."""
    out = []
    for i in s.by_field("wants", "unaffordable"):
        if i.confidence < MIN_CONF:
            continue
        # 미뤄둔 것일수록 '사기 전에 해보기'가 잘 맞는다
        out.append(Seed("", PRECEDE, [i.value],
                        i.confidence + (0.2 if i.deferral_signal else 0.0)))
    return out


def _substitute(s: State) -> list[Seed]:
    return [Seed("", SUBSTITUTE, [i.value], i.confidence)
            for i in s.by_field("consumables") if i.confidence >= MIN_CONF]


def _fill(s: State) -> list[Seed]:
    # lifestyle 은 확신이 높아도 쓸 만한 카드가 잘 안 나온다. 점수를 낮춰 뒤로 민다.
    return [Seed("", FILL, [i.value], i.confidence * 0.8)
            for i in s.by_field("lifestyle") if i.confidence >= MIN_CONF]


def seeds(s: State, memory: dict | None = None) -> list[Seed]:
    memory = memory or {}
    dead = set(memory.get("rejectedSources", []))

    cand = _pairs(s) + _precede(s) + _substitute(s) + _fill(s)
    cand = [c for c in cand if not (dead & set(c.sources))]
    cand.sort(key=lambda c: -c.score)

    taken: dict[str, int] = {}
    out: list[Seed] = []
    for c in cand:
        if taken.get(c.kind, 0) >= PER_KIND_SEED:
            continue
        taken[c.kind] = taken.get(c.kind, 0) + 1
        c.id = f"s{len(out) + 1}"
        out.append(c)
        if len(out) >= SEED_BUDGET:
            break
    return out


# ────────────────────────────────────────────────────── ② 문장 쓰기

CARD_SCHEMA = {
    "type": "object",
    "properties": {
        "cards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "seedId": {"type": "string"},
                    "title": {"type": "string"},
                    "reason": {"type": "string"},
                    "requires": {"type": "array", "items": {"type": "string"}},
                    "effort": {"type": "string", "enum": ["low", "mid", "high"]},
                },
                "required": ["seedId", "title", "reason", "requires", "effort"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["cards"],
    "additionalProperties": False,
}


CARD_SYSTEM = """너는 사용자의 취향을 보고 **해볼 만한 행동**을 제안하는 역할이다.
상품을 추천하지 않는다. 선물 이야기를 하지 않는다.

## 무엇을 내나

받은 재료 하나당 카드 하나. **사용자가 미처 생각하지 못했을 행동**을 낸다.
이미 하고 있다고 말한 것을 그대로 다시 쓰면 카드가 아니다.

  나쁜 예 — 이미 하는 것
    재료: 캠핑 · 핸드드립     →  "캠핑 가서 커피 마시기"
  좋은 예 — 둘을 붙여서 한 걸음 나간 것
    재료: 캠핑 · 핸드드립     →  "산 정상에서 원두 갈아 내리기"

## 갈래별로 쓰는 법

| 갈래 | 어떻게 |
|---|---|
| 겹치는 관심사 | 둘이 만나는 지점에서 새 행동을 만든다 |
| 사고 싶어하는 것 | **사기 전에 먼저 해볼 수 있는 것**을 낸다 (턴테이블 → LP 바 가보기) |
| 반복해서 사는 것 | 그 반복을 다른 방식으로 바꿔본다 (캡슐 → 원두 갈아보기) |
| 생활 맥락 | 그 상황에 반복되는 빈 시간을 채운다 (통근 1시간 → 오디오북) |

## 각 칸

- `title` — **행동**이다. 동사로 끝낸다. 15자 안쪽. 물건 이름이 제목이 되면 안 된다.
- `reason` — "~라고 하셔서" 한 줄. 25자 안쪽. **재료에 있는 말만 쓴다.**
- `requires` — 그 행동에 실제로 필요한 물건 1~3개. 일반명사로. 브랜드·모델명 금지.
  이미 갖고 있다고 한 것은 넣지 않는다. 필요한 게 없으면 빈 배열.
- `effort` — 오늘 바로 되면 `low`, 준비가 필요하면 `mid`, 날 잡아야 하면 `high`.

## 버려야 할 재료

- 두 재료가 서로 안 엮이면 **만들지 말고 건너뛴다.** 억지로 잇지 않는다.
- 피해야 할 것 목록에 걸리면 만들지 않는다.
- 뻔한 말밖에 안 나오면 건너뛴다. 개수를 채우는 것보다 안 내는 게 낫다.

`seedId` 는 받은 재료의 id 를 **그대로** 옮긴다."""


def _ask(s: State, sd: list[Seed]) -> list[dict]:
    avoid = [i.value for i in s.by_field("dislikes", "constraints")]
    owned = [i.value for i in s.by_field("owned")]
    doing = [i.value for i in s.by_field("hobbies")]

    lines = ["[재료]"]
    for c in sd:
        lines.append(f"- {c.id} ({KIND_LABEL[c.kind]}): " + " × ".join(c.sources))
    lines.append("\n[피해야 할 것]\n" + (" · ".join(avoid) or "(없음)"))
    lines.append("[이미 갖고 있는 것]\n" + (" · ".join(owned) or "(없음)"))
    lines.append("[이미 하고 있는 것]\n" + (" · ".join(doing) or "(없음)"))

    # 카드 문구는 스키마를 채우는 일이 아니라 쓰는 일이다. 작은 추출 모델로는
    # 뻔한 말밖에 안 나온다. finalize._summarize 와 같은 이유로 응답 모델을 쓴다.
    # 대기 경로가 아니라 느려도 된다.
    out = llm.json_chat(
        [{"role": "system", "content": CARD_SYSTEM},
         {"role": "user", "content": "\n".join(lines)}],
        CARD_SCHEMA, model=llm.RESPONSE_MODEL, temperature=0.6,
    )
    return out.get("cards", [])


# ────────────────────────────────────────────────────── ③ 걸러내기

def _hits(text: str, values: list[str]) -> bool:
    return any(v and v in text for v in values)


def build(s: State, memory: dict | None = None) -> list[Card]:
    memory = memory or {}
    sd = seeds(s, memory)
    if not sd:
        return []

    by_id = {c.id: c for c in sd}
    avoid = [i.value for i in s.by_field("dislikes", "constraints")]
    owned = [i.value for i in s.by_field("owned")]
    doing = [i.value for i in s.by_field("hobbies")]
    shown = set(memory.get("shownCards", []))

    cards: list[Card] = []
    for raw in _ask(s, sd):
        seed = by_id.get(raw.get("seedId", ""))
        title = (raw.get("title") or "").strip()
        if not seed or not title or title in shown:
            continue

        # 후검증 — 피해야 할 것이 제목이나 준비물에 들어오면 버린다.
        # 카드 한 장이 잘못 나가면 목록 전체의 신뢰가 깎인다.
        blob = title + " " + " ".join(raw.get("requires", []))
        if _hits(blob, avoid):
            continue
        # 이미 하고 있는 것을 그대로 되돌려주는 카드는 값이 없다
        if _hits(title, doing):
            continue

        cards.append(Card(
            title=title,
            reason=(raw.get("reason") or "").strip(),
            # 이미 가진 물건은 니즈가 아니다. 선물 리스트로 넘기지 않는다.
            requires=[r for r in raw.get("requires", []) if not _hits(r, owned)],
            effort=raw.get("effort") or "mid",
            kind=seed.kind,
            sources=seed.sources,
        ))

    # 갈래별 상한 — 한 갈래가 화면을 다 먹으면 목록이 단조로워진다
    taken: dict[str, int] = {}
    out: list[Card] = []
    for c in cards:
        if taken.get(c.kind, 0) >= PER_KIND_OUT:
            continue
        taken[c.kind] = taken.get(c.kind, 0) + 1
        out.append(c)

    # 바로 할 수 있는 게 하나는 있어야 탭이 일어난다. 없으면 low 를 끌어올린다.
    if len(out) > MAX_CARDS and not any(c.effort == "low" for c in out[:MAX_CARDS]):
        low = next((c for c in out[MAX_CARDS:] if c.effort == "low"), None)
        if low:
            out.remove(low)
            out.insert(0, low)

    # 3개를 못 채워도 억지로 늘리지 않는다. 빈약한 목록보다 짧은 목록이 낫다.
    return out[:MAX_CARDS]


# ────────────────────────────────────────────────────── 반응 → 프로필

# 카드 탭은 3초다. 대화보다 싸게 프로필을 갱신하는 유일한 경로.
#
# 주의 — 카드 제목은 **활동**이고 wants·owned 는 **물건** 칸이다.
# 제목을 그대로 wants 에 넣으면 "산 정상에서 원두 갈아 내리기" 가 선물 검색어가 된다.
# 니즈가 되는 것은 제목이 아니라 requires 다.


def tap(card: Card, action: str) -> tuple[dict, dict]:
    """카드 반응 → (프로필 델타, 메모리 패치).

    델타는 state.apply() 가 그대로 먹는다.
    """
    delta = {"items": [], "axes": [], "drop": []}
    patch: dict = {"shownCards": [card.title]}

    def add(value, field, conf, intent="none"):
        delta["items"].append({
            "value": value, "field": field, "confidence": conf,
            "evidence": f"카드 '{card.title}' · {action}",
            "intent": intent, "defer": "none", "urgency": "none",
        })

    if action == "want":
        # 활동은 관심사로, 그 활동에 필요한 물건은 want 로.
        # 선물 프레임 없이 눌렀으므로 오염되지 않은 want 다.
        add(card.title, "interests", 0.7)
        for r in card.requires:
            add(r, "wants", 0.75, intent="want")

    elif action == "done":
        # 해봤다는 것이 그 물건을 갖고 있다는 뜻은 아니다. owned 로 옮기지 않는다.
        add(card.title, "hobbies", 0.7)

    elif action == "no":
        # 활동 제목을 dislikes 에 넣지 않는다. dislikes 는 상품 필터라
        # 활동 문장이 들어가면 아무것도 못 거른다. 카드 억제로만 쓴다.
        patch["rejectedSources"] = list(card.sources)

    return delta, patch


if __name__ == "__main__":
    # 재료 선정은 LLM 없이 돌아간다. 모델 없이도 여기까지 검증할 수 있다.
    def it(v, f, c):
        return Item(v, f, c, "", None, None, None, 0, 0)

    demo = State(turn=6, items=[
        it("캠핑", "interests", 0.9),
        it("핸드드립 커피", "interests", 0.85),
        it("팟캐스트 듣기", "interests", 0.7),
        it("가벼운 것", "preferences", 0.6),
        it("강한 향", "dislikes", 0.7),
        it("자취", "lifestyle", 0.8),
        it("지하철로 한 시간 반 통근", "lifestyle", 0.9),
        it("턴테이블", "wants", 0.9),
        it("티타늄 머그컵", "unaffordable", 0.9),
        it("커피 캡슐", "consumables", 0.9),
        it("텐트", "owned", 0.8),
    ])
    for c in seeds(demo):
        print(f"{c.id:>3}  {c.score:.2f}  {KIND_LABEL[c.kind]:<12} {' × '.join(c.sources)}")
