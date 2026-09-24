from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.domain.profile.models import ProfileState, utc_now


class ConversationGoal(StrEnum):
    OPENING = "OPENING"
    INTEREST = "INTEREST"
    INTEREST_VIA_ROUTINE = "INTEREST_VIA_ROUTINE"
    DISLIKE = "DISLIKE"
    GEAR = "GEAR"
    DEEPEN = "DEEPEN"
    CORRECT = "CORRECT"
    WRAP = "WRAP"


class GoalArea(StrEnum):
    INTEREST = "interest"
    GEAR = "gear"
    EXCLUSION = "exclusion"
    DEEPEN = "deepen"


class CoverageStatus(StrEnum):
    PENDING = "pending"
    FOUND = "found"
    CONFIRMED_NONE = "confirmed_none"
    UNRESOLVED = "unresolved"
    EXHAUSTED = "exhausted"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    INPUT_LOCKED = "input_locked"
    REVIEW = "review"
    CLOSED = "closed"


class CompletionReason(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    USER_EXIT = "USER_EXIT"
    MAX_CYCLES = "MAX_CYCLES"


@dataclass(slots=True, frozen=True)
class ConversationTurn:
    role: str
    content: str
    created_at: datetime

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        fallback_created_at: datetime,
    ) -> ConversationTurn:
        created_at = payload.get("created_at")
        return cls(
            role=payload["role"],
            content=payload["content"],
            created_at=(datetime.fromisoformat(created_at) if created_at else fallback_created_at),
        )


def _default_coverage() -> dict[GoalArea, CoverageStatus]:
    return {area: CoverageStatus.PENDING for area in GoalArea}


@dataclass(slots=True)
class ConversationState:
    user_id: int
    conversation_room_id: int
    status: SessionStatus = SessionStatus.ACTIVE
    turn_count: int = 0
    history: list[ConversationTurn] = field(default_factory=list)
    profile: ProfileState = field(default_factory=ProfileState)
    goal_coverage: dict[GoalArea, CoverageStatus] = field(default_factory=_default_coverage)
    goal_attempts: dict[str, int] = field(default_factory=dict)
    last_goal: ConversationGoal | None = None
    completion_reason: CompletionReason | None = None
    created_at: datetime = field(default_factory=utc_now)
    last_active_at: datetime = field(default_factory=utc_now)
    expiration_at: datetime | None = None
    last_turn_extraction_failed: bool = False
    finalized: bool = False
    analysis_turn_count: int | None = None
    analysis_summary: str | None = None
    analysis_taste_keywords: list[str] = field(default_factory=list)
    analysis_interest_keywords: list[str] = field(default_factory=list)

    @property
    def session_id(self) -> int:
        return self.conversation_room_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "conversation_room_id": self.conversation_room_id,
            "status": self.status.value,
            "turn_count": self.turn_count,
            "history": [turn.to_dict() for turn in self.history],
            "profile": self.profile.to_dict(),
            "goal_coverage": {
                area.value: status.value for area, status in self.goal_coverage.items()
            },
            "goal_attempts": dict(self.goal_attempts),
            "last_goal": self.last_goal.value if self.last_goal else None,
            "completion_reason": (self.completion_reason.value if self.completion_reason else None),
            "created_at": self.created_at.isoformat(),
            "last_active_at": self.last_active_at.isoformat(),
            "expiration_at": self.expiration_at.isoformat() if self.expiration_at else None,
            "last_turn_extraction_failed": self.last_turn_extraction_failed,
            "finalized": self.finalized,
            "analysis_turn_count": self.analysis_turn_count,
            "analysis_summary": self.analysis_summary,
            "analysis_taste_keywords": list(self.analysis_taste_keywords),
            "analysis_interest_keywords": list(self.analysis_interest_keywords),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConversationState:
        created_at = datetime.fromisoformat(payload["created_at"])
        coverage = _default_coverage()
        for area, status in payload.get("goal_coverage", {}).items():
            coverage[GoalArea(area)] = CoverageStatus(status)
        last_goal = payload.get("last_goal")
        completion_reason = payload.get("completion_reason")
        return cls(
            user_id=int(payload["user_id"]),
            conversation_room_id=int(payload["conversation_room_id"]),
            status=SessionStatus(payload.get("status", SessionStatus.ACTIVE)),
            turn_count=int(payload.get("turn_count", 0)),
            history=[
                ConversationTurn.from_dict(turn, fallback_created_at=created_at)
                for turn in payload.get("history", [])
            ],
            profile=ProfileState.from_dict(payload.get("profile")),
            goal_coverage=coverage,
            goal_attempts={
                key: int(value) for key, value in payload.get("goal_attempts", {}).items()
            },
            last_goal=ConversationGoal(last_goal) if last_goal else None,
            completion_reason=(CompletionReason(completion_reason) if completion_reason else None),
            created_at=created_at,
            last_active_at=datetime.fromisoformat(payload["last_active_at"]),
            expiration_at=(
                datetime.fromisoformat(payload["expiration_at"])
                if payload.get("expiration_at")
                else None
            ),
            last_turn_extraction_failed=bool(payload.get("last_turn_extraction_failed", False)),
            finalized=bool(payload.get("finalized", False)),
            analysis_turn_count=(
                int(payload["analysis_turn_count"])
                if payload.get("analysis_turn_count") is not None
                else None
            ),
            analysis_summary=payload.get("analysis_summary"),
            analysis_taste_keywords=list(payload.get("analysis_taste_keywords", [])),
            analysis_interest_keywords=list(payload.get("analysis_interest_keywords", [])),
        )


@dataclass(slots=True, frozen=True)
class ReadinessResult:
    sufficient: bool
    missing_signals: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class GoalDecision:
    goal: ConversationGoal
    completion_reason: CompletionReason | None = None
