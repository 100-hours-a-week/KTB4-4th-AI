from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from app.domain.profile.models import ProfileState, utc_now
from app.domain.profile.onboarding import OnboardingProfile


class ConversationGoal(StrEnum):
    OPENING = "OPENING"
    # 대화를 이어가는 차례. 구체적인 반응 방식은 move가 정한다.
    CHAT = "CHAT"
    WRAP = "WRAP"


class ConversationMove(StrEnum):
    """니쥬가 이번 턴에 어떻게 반응할지. 무엇을 알아낼지가 아니다."""

    # 방금 이야기를 이어 묻는다. 기본 move.
    FOLLOW = "follow"
    # 질문 없이 니쥬의 의견, 감상, 아는 이야기를 보탠다.
    ADD = "add"
    # 열린 질문 대신 결이 다른 선택지 두 개를 주고 고르게 한다. 질문-답만 이어지는 흐름을 바꾸고
    # 답하기 어려운 것도 쉽게 답하게 한다. 한 턴으로 끝나는 대화 방식이다.
    CHOICE = "choice"
    # 앞에서 신나게 이야기한 화제로 돌아간다.
    RETURN = "return"
    # 한 화제에서 꼬리 질문을 충분히 했으니 다른 생활 이야기로 넘어간다.
    SHIFT = "shift"

    @classmethod
    def _missing_(cls, value: object) -> ConversationMove | None:
        # 이전 버전 세션에는 선택지 move가 "play"로 저장돼 있다.
        return cls.CHOICE if value == "play" else None


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


@dataclass(slots=True)
class ConversationState:
    user_id: int
    conversation_room_id: int
    status: SessionStatus = SessionStatus.ACTIVE
    turn_count: int = 0
    history: list[ConversationTurn] = field(default_factory=list)
    profile: ProfileState = field(default_factory=ProfileState)
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
    analysis_keyword_scores: dict[str, float] = field(default_factory=dict)
    analysis_patch_used: bool = False
    # 대화 흐름 상태. 최근 move, 사용자 에너지(0~1)와 그 최고치,
    # 에너지가 낮은 턴이 이어진 횟수, 사용자 평균 발화 길이.
    move_history: list[str] = field(default_factory=list)
    energy: float | None = None
    peak_energy: float = 0.0
    low_energy_streak: int = 0
    user_length_avg: float | None = None
    # 사용자가 가장 신나게 이야기한 관심사와 그때 에너지. 되돌아가기에 쓴다.
    best_topic: str | None = None
    best_topic_energy: float = 0.0
    returned_topics: list[str] = field(default_factory=list)
    used_scenes: list[str] = field(default_factory=list)
    # 한 번 오른 진행률은 내리지 않는다.
    progress_floor: int = 0
    # 턴을 기록한 뒤 종료로 판정되면 다음 턴을 마무리 턴으로 예약한다.
    # 그 턴의 질문은 이미 사용자에게 흘려보냈으므로 바꾸지 않고, 답을 받은 뒤 마무리한다.
    pending_wrap: CompletionReason | None = None
    # 마지막으로 확정한 턴의 BE 사용자 메시지 ID. 같은 메시지 재요청이면 다시 생성하지 않는다.
    last_user_message_id: int | None = None
    # 대화방을 만들 때 BE가 보낸 온보딩 값. 이전 버전에서 만든 세션에는 없다.
    onboarding: OnboardingProfile | None = None

    @property
    def last_move(self) -> ConversationMove | None:
        return ConversationMove(self.move_history[-1]) if self.move_history else None

    @property
    def session_id(self) -> int:
        return self.conversation_room_id

    def user_turns(self) -> list[tuple[int, str]]:
        """history의 사용자 발화를 턴 번호(1부터)와 함께 돌려준다."""
        turns = [turn.content for turn in self.history if turn.role == "user"]
        return list(enumerate(turns, start=1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "conversation_room_id": self.conversation_room_id,
            "status": self.status.value,
            "turn_count": self.turn_count,
            "history": [turn.to_dict() for turn in self.history],
            "profile": self.profile.to_dict(),
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
            "analysis_keyword_scores": dict(self.analysis_keyword_scores),
            "analysis_patch_used": self.analysis_patch_used,
            "move_history": list(self.move_history),
            "energy": self.energy,
            "peak_energy": self.peak_energy,
            "low_energy_streak": self.low_energy_streak,
            "user_length_avg": self.user_length_avg,
            "best_topic": self.best_topic,
            "best_topic_energy": self.best_topic_energy,
            "returned_topics": list(self.returned_topics),
            "used_scenes": list(self.used_scenes),
            "progress_floor": self.progress_floor,
            "pending_wrap": self.pending_wrap.value if self.pending_wrap else None,
            "last_user_message_id": self.last_user_message_id,
            "onboarding": self.onboarding.to_dict() if self.onboarding else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConversationState:
        created_at = datetime.fromisoformat(payload["created_at"])
        completion_reason = payload.get("completion_reason")
        pending_wrap = payload.get("pending_wrap")
        onboarding = payload.get("onboarding")
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
            analysis_keyword_scores={
                key: float(value)
                for key, value in payload.get("analysis_keyword_scores", {}).items()
            },
            analysis_patch_used=bool(payload.get("analysis_patch_used", False)),
            move_history=list(payload.get("move_history", [])),
            energy=float(payload["energy"]) if payload.get("energy") is not None else None,
            peak_energy=float(payload.get("peak_energy", 0.0)),
            low_energy_streak=int(payload.get("low_energy_streak", 0)),
            user_length_avg=(
                float(payload["user_length_avg"])
                if payload.get("user_length_avg") is not None
                else None
            ),
            best_topic=payload.get("best_topic"),
            best_topic_energy=float(payload.get("best_topic_energy", 0.0)),
            returned_topics=list(payload.get("returned_topics", [])),
            used_scenes=list(payload.get("used_scenes", [])),
            progress_floor=int(payload.get("progress_floor", 0)),
            pending_wrap=CompletionReason(pending_wrap) if pending_wrap else None,
            last_user_message_id=(
                int(payload["last_user_message_id"])
                if payload.get("last_user_message_id") is not None
                else None
            ),
            onboarding=OnboardingProfile.from_dict(onboarding) if onboarding else None,
        )


@dataclass(slots=True, frozen=True)
class ReadinessResult:
    sufficient: bool
    missing_signals: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class GoalDecision:
    goal: ConversationGoal
    completion_reason: CompletionReason | None = None
    # 이번 턴의 반응 방식과 그때 참고할 화제.
    move: ConversationMove | None = None
    scene: str | None = None
