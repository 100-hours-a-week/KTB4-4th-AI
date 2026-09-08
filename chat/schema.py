"""스키마 정의와 서버가 책임지는 상태 관리.

모델은 '이번 턴에 새로 알아낸 것'만 낸다. 나머지는 전부 여기서 채운다.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from datetime import datetime, timezone

MAX_ITEMS = 12          # 세션 중 들고 있는 항목 상한
MAX_AXES = 3

FIELDS = [
    "interests", "hobbies", "preferences", "dislikes", "lifestyle",
    "wants", "unaffordable", "owned", "consumables", "constraints",
]

# 이 항목이 상품에 닿는 경로 — field 에서 결정적으로 파생된다
LINK_ROLE = {
    "interests": "query", "hobbies": "query", "wants": "query",
    "unaffordable": "query", "consumables": "query",
    "dislikes": "filter", "constraints": "filter", "owned": "filter",
    "preferences": "weight", "lifestyle": "weight",
}

# 친구에게 보일지 — 계약 enum 은 private · friends · public
# constraints(알레르기·사이즈)와 lifestyle(수면·사정)은 친구 화면에 나가지 않는다.
PRIVATE_FIELDS = {"constraints", "lifestyle"}
VISIBILITY = {f: ("private" if f in PRIVATE_FIELDS else "friends") for f in FIELDS}

# 의도 기본값
INTENT_DEFAULT = {
    "consumables": ("need", None, "soon"),
    "wants": ("want", None, None),
    "unaffordable": ("want", "price", None),
    "interests": ("both", None, None),
    "hobbies": ("both", None, None),
    "preferences": ("both", None, None),
    "lifestyle": ("both", None, None),
    "dislikes": (None, None, None),
    "owned": (None, None, None),
    "constraints": (None, None, None),
}

# 방출에서 제외되는 칸 — 개수가 적고 하나 빠지면 오답이 나온다
NEVER_EJECT = {"dislikes", "constraints"}

# 계약(ProfileItem)의 길이 상한
MAX_VALUE = 20
MAX_EVIDENCE = 40

SCHEMA_VERSION = "3.0"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# 추출 모델이 따를 JSON 스키마 (구조화 출력)
DELTA_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "field": {"type": "string", "enum": FIELDS},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                    "intent": {"type": "string", "enum": ["need", "want", "both", "none"]},
                    "defer": {"type": "string", "enum": ["price", "justification", "timing", "none"]},
                    "urgency": {"type": "string", "enum": ["soon", "later", "none"]},
                },
                "required": ["value", "field", "confidence", "evidence",
                             "intent", "defer", "urgency"],
                "additionalProperties": False,
            },
        },
        "axes": {"type": "array", "items": {"type": "string"}},
        "drop": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["items", "axes", "drop"],
    "additionalProperties": False,
}


@dataclass
class Item:
    value: str
    field: str
    confidence: float
    evidence: str
    intent_type: str | None
    deferral_reason: str | None
    urgency: str | None
    first_turn: int
    updated_turn: int
    first_seen_at: str = ""
    updated_at: str = ""
    # 택소노미 사전이 아직 없다. 매핑 실패는 오류가 아니라 null 이다.
    taxonomy_path: list[str] | None = None

    @property
    def link_role(self) -> str:
        return LINK_ROLE[self.field]

    @property
    def visibility(self) -> str:
        return VISIBILITY[self.field]

    @property
    def deferral_signal(self) -> bool:
        return self.deferral_reason is not None

    def to_contract(self) -> dict:
        """계약의 ProfileItem 객체로 직렬화한다."""
        out = {
            "value": self.value,
            "confidence": round(self.confidence, 2),
            "linkRole": self.link_role,
            "visibility": self.visibility,
            # 배제 전용 칸(dislikes·owned·constraints)에는 need/want 구분이 없다
            "intentType": self.intent_type,
            "deferralSignal": self.deferral_signal,
            "evidence": self.evidence,
            "taxonomyPath": self.taxonomy_path,
            "firstSeenAt": self.first_seen_at,
            "updatedAt": self.updated_at,
        }
        # deferralReason 은 조건부 필수 — 보류 신호일 때만 넣는다
        if self.deferral_signal:
            out["deferralReason"] = self.deferral_reason
        return out


def _norm(v: str | None) -> str | None:
    if v in (None, "", "none", "null"):
        return None
    return v


@dataclass
class State:
    turn: int = 0
    items: list[Item] = dc_field(default_factory=list)
    axes: list[str] = dc_field(default_factory=list)
    history: list[dict] = dc_field(default_factory=list)      # 대화 이력 (누적)
    user_lengths: list[int] = dc_field(default_factory=list)  # 유저 발화 길이
    goal_log: list[str] = dc_field(default_factory=list)      # 턴마다 어떤 목표였나
    goal_stats: dict = dc_field(default_factory=dict)         # 목표별 시도/성과
    last_reply: str = ""

    def note_try(self, goal: str) -> None:
        st = self.goal_stats.setdefault(goal, {"tries": 0, "wins": 0})
        st["tries"] += 1

    def note_result(self, goal: str, gained: int) -> None:
        st = self.goal_stats.setdefault(goal, {"tries": 0, "wins": 0})
        if gained > 0:
            st["wins"] += 1

    def tries(self, goal: str) -> int:
        return self.goal_stats.get(goal, {}).get("tries", 0)

    def wins(self, goal: str) -> int:
        return self.goal_stats.get(goal, {}).get("wins", 0)

    # ---------------------------------------------------------------- 조회

    def by_role(self, role: str) -> list[Item]:
        return [i for i in self.items if i.link_role == role]

    def by_field(self, *fields: str) -> list[Item]:
        return [i for i in self.items if i.field in fields]

    @property
    def query_items(self) -> list[Item]:
        return self.by_role("query")

    @property
    def strong_query_items(self) -> list[Item]:
        return [i for i in self.query_items if i.confidence >= 0.6]

    @property
    def filter_items(self) -> list[Item]:
        return self.by_field("dislikes", "constraints")

    @property
    def want_items(self) -> list[Item]:
        return [i for i in self.items if i.intent_type == "want"]

    @property
    def can_close(self) -> bool:
        return (len(self.query_items) >= 3
                and len(self.strong_query_items) >= 2
                and len(self.filter_items) >= 1)

    def goal_attempts(self, goal: str) -> int:
        return self.goal_log.count(goal)

    # ---------------------------------------------------------------- 갱신

    def apply(self, delta: dict) -> tuple[list[Item], list[str]]:
        """모델 델타를 병합한다. 반환값은 (새로 들어온 항목, 지워진 값)."""
        added: list[Item] = []

        for raw in delta.get("items", []):
            f = raw.get("field")
            if f not in LINK_ROLE:
                continue
            value = (raw.get("value") or "").strip()[:MAX_VALUE]
            if not value:
                continue

            d_intent, d_defer, d_urgency = INTENT_DEFAULT[f]
            intent = _norm(raw.get("intent")) or d_intent
            defer = _norm(raw.get("defer")) or d_defer
            urgency = _norm(raw.get("urgency")) or d_urgency
            conf = float(raw.get("confidence") or 0.5)

            existing = next((i for i in self.items if i.value == value), None)
            if existing:
                existing.confidence = max(existing.confidence, conf)
                existing.field = f
                existing.intent_type = intent
                existing.deferral_reason = defer
                existing.urgency = urgency
                existing.updated_turn = self.turn
                existing.updated_at = _now()
            else:
                now = _now()
                item = Item(
                    value=value, field=f, confidence=conf,
                    evidence=(raw.get("evidence") or "")[:MAX_EVIDENCE],
                    intent_type=intent, deferral_reason=defer, urgency=urgency,
                    first_turn=self.turn, updated_turn=self.turn,
                    first_seen_at=now, updated_at=now,
                )
                self.items.append(item)
                added.append(item)

        dropped = [v for v in delta.get("drop", []) if any(i.value == v for i in self.items)]
        if dropped:
            self.items = [i for i in self.items if i.value not in dropped]

        for a in delta.get("axes", []):
            a = a.strip()
            if a and a not in self.axes:
                self.axes.append(a)
        self.axes = self.axes[:MAX_AXES]

        self._eject()
        # 방금 넣었어도 방출됐으면 새로 파악한 것이 아니다.
        survived = {id(i) for i in self.items}
        return [i for i in added if id(i) in survived], dropped

    def _eject(self) -> None:
        """12개를 넘으면 점수 낮은 것부터 버린다."""
        if len(self.items) <= MAX_ITEMS:
            return

        def keep(i: Item) -> bool:
            return i.field in NEVER_EJECT or i.deferral_signal

        protected = [i for i in self.items if keep(i)]
        rest = [i for i in self.items if not keep(i)]

        def score(i: Item) -> float:
            먼저언급 = 1.0 / (1 + i.first_turn)
            경과 = self.turn - i.updated_turn
            return i.confidence + 먼저언급 - 0.1 * 경과

        # 보호 항목만으로 상한을 넘으면 보호 안에서도 낮은 것부터 버린다.
        # 상한은 프롬프트 비용을 막는 장치라 예외를 두면 의미가 없다.
        if len(protected) > MAX_ITEMS:
            protected.sort(key=score, reverse=True)
            self.items = protected[:MAX_ITEMS]
            return

        rest.sort(key=score, reverse=True)
        room = MAX_ITEMS - len(protected)
        self.items = protected + rest[:room]

    # ---------------------------------------------------------------- 출력

    def to_profile(self, user_id: str = "u_local", summary: str = "") -> dict:
        """세션 상태를 계약의 TasteProfile 객체로 확정한다.

        칸(field)이 곧 배열 이름이다. linkRole·visibility·intentType 은
        모델이 내지 않고 여기서 칸으로부터 파생된다.
        """
        profile: dict = {
            "schemaVersion": SCHEMA_VERSION,
            "userId": user_id,
            "summary": summary,
        }
        for f in FIELDS:
            profile[f] = [i.to_contract() for i in self.items if i.field == f]
        profile["axes"] = list(self.axes)
        return profile

    def state_block(self) -> str:
        """추출 모델에게 넘길 현재 상태 요약."""
        if not self.items:
            return "(아직 없음)"
        lines = []
        for f in FIELDS:
            got = [i for i in self.items if i.field == f]
            if got:
                lines.append(f"{f}: " + " · ".join(f"{i.value}({i.confidence:.1f})" for i in got))
        if self.axes:
            lines.append("axes: " + " · ".join(self.axes))
        return "\n".join(lines)

    def summary_counts(self) -> dict:
        return {
            "턴": self.turn,
            "검색용": len(self.query_items),
            "확신0.6+": len(self.strong_query_items),
            "필터": len(self.filter_items),
            "want": len(self.want_items),
            "전체": len(self.items),
            "canClose": self.can_close,
        }
