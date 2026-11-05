from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ConversationDirection(StrEnum):
    DEVELOPING = "developing"
    SETTLED = "settled"
    BROADEN = "broaden"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ConversationContext:
    """대화용 관찰이다. 확인된 프로필이나 현재 사용자 의도를 대신하지 않는다."""

    focus: str
    direction: ConversationDirection
    evidence: str
    source_turn: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "focus": self.focus,
            "direction": self.direction.value,
            "evidence": self.evidence,
            "source_turn": self.source_turn,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ConversationContext":
        return cls(
            focus=value["focus"],
            direction=ConversationDirection(value["direction"]),
            evidence=value["evidence"],
            source_turn=int(value.get("source_turn", 0)),
        )
