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
    TASTE = "TASTE"
    DISLIKE = "DISLIKE"
    GEAR = "GEAR"
    DEEPEN = "DEEPEN"
    # 곰곰이 생각하게 하는 대화(reflective)에서만 쓴다.
    BRIDGE = "BRIDGE"
    REFLECT = "REFLECT"
    # 니쥬 수다(companion)에서 대화를 이어가는 차례. 구체적인 반응 방식은 move가 정한다.
    CHAT = "CHAT"
    WRAP = "WRAP"


class ConversationStyle(StrEnum):
    # 부족한 정보를 차례로 묻는 기존 방식
    EXPLORE = "explore"
    # 한 이야기를 무엇 → 어떻게 → 왜 → 반대편으로 따라가며 곰곰이 생각하게 하는 방식
    REFLECTIVE = "reflective"
    # 니쥬가 반응하고 자기 생각을 보태며 수다를 떨고,
    # 사용자 에너지를 보고 반응 방식(move)을 고르는 방식
    COMPANION = "companion"


class ConversationMove(StrEnum):
    """companion 대화에서 니쥬가 이번 턴에 어떻게 반응할지. 무엇을 알아낼지가 아니다."""

    # 방금 이야기를 이어 묻는다. 기본 move.
    FOLLOW = "follow"
    # 질문 없이 니쥬의 의견, 감상, 아는 이야기를 보탠다.
    ADD = "add"
    # 들은 이야기의 공통된 결을 짐작으로 되비춘다.
    REFLECT_BACK = "reflect_back"
    # 방금 이야기에서 다리를 놓아 다른 생활 화제로 건너간다.
    BRIDGE = "bridge"
    # 밸런스 게임이나 둘 중 고르기로 답하는 부담을 낮춘다.
    PLAY = "play"
    # 앞에서 신나게 이야기한 화제로 돌아간다.
    RETURN = "return"


class ThreadStage(StrEnum):
    """reflective 대화에서 지금 이야기 줄기가 어디까지 왔는지."""

    WHAT = "what"
    HOW = "how"
    WHY = "why"
    CONTRAST = "contrast"
    BRIDGE = "bridge"


class GoalArea(StrEnum):
    INTEREST = "interest"
    TASTE = "taste"
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
    turns_since_new_interest: int = 0
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
    conversation_style: ConversationStyle = ConversationStyle.EXPLORE
    # reflective 대화의 이야기 줄기. 줄기 관심사와 단계, 줄기 안에서 주고받은 턴 수.
    thread_topic: str | None = None
    thread_stage: ThreadStage = ThreadStage.WHAT
    thread_turns: int = 0
    # "왜" 질문을 한 횟수. 대화가 무거워지지 않게 제한한다.
    why_count: int = 0
    # 직전 사용자 답의 깊이(0 짧게 넘김 ~ 2 이유나 느낌을 담음)와 짧은 답이 이어진 횟수.
    last_answer_depth: float | None = None
    shallow_streak: int = 0
    # 되비추기를 했는지, 되비추기에 쓴 항목 값. 다음 답에서 정정 대상 후보가 된다.
    reflected: bool = False
    reflection_values: list[str] = field(default_factory=list)
    # companion 대화의 흐름 상태.
    # 최근 move, 질문으로 끝난 니쥬 응답이 이어진 횟수, 사용자 에너지(0~1)와 그 최고치,
    # 에너지가 낮은 턴이 이어진 횟수, 사용자 평균 발화 길이.
    move_history: list[str] = field(default_factory=list)
    question_streak: int = 0
    energy: float | None = None
    peak_energy: float = 0.0
    low_energy_streak: int = 0
    user_length_avg: float | None = None
    # 사용자가 가장 신나게 이야기한 관심사와 그때 에너지. 되돌아가기에 쓴다.
    best_topic: str | None = None
    best_topic_energy: float = 0.0
    returned_topics: list[str] = field(default_factory=list)
    used_scenes: list[str] = field(default_factory=list)
    reflect_back_count: int = 0
    # 한 번 오른 진행률은 내리지 않는다.
    progress_floor: int = 0

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
            "goal_coverage": {
                area.value: status.value for area, status in self.goal_coverage.items()
            },
            "goal_attempts": dict(self.goal_attempts),
            "last_goal": self.last_goal.value if self.last_goal else None,
            "turns_since_new_interest": self.turns_since_new_interest,
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
            "conversation_style": self.conversation_style.value,
            "thread_topic": self.thread_topic,
            "thread_stage": self.thread_stage.value,
            "thread_turns": self.thread_turns,
            "why_count": self.why_count,
            "last_answer_depth": self.last_answer_depth,
            "shallow_streak": self.shallow_streak,
            "reflected": self.reflected,
            "reflection_values": list(self.reflection_values),
            "move_history": list(self.move_history),
            "question_streak": self.question_streak,
            "energy": self.energy,
            "peak_energy": self.peak_energy,
            "low_energy_streak": self.low_energy_streak,
            "user_length_avg": self.user_length_avg,
            "best_topic": self.best_topic,
            "best_topic_energy": self.best_topic_energy,
            "returned_topics": list(self.returned_topics),
            "used_scenes": list(self.used_scenes),
            "reflect_back_count": self.reflect_back_count,
            "progress_floor": self.progress_floor,
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
            turns_since_new_interest=int(payload.get("turns_since_new_interest", 0)),
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
            conversation_style=ConversationStyle(
                payload.get("conversation_style", ConversationStyle.EXPLORE)
            ),
            thread_topic=payload.get("thread_topic"),
            thread_stage=ThreadStage(payload.get("thread_stage", ThreadStage.WHAT)),
            thread_turns=int(payload.get("thread_turns", 0)),
            why_count=int(payload.get("why_count", 0)),
            last_answer_depth=(
                float(payload["last_answer_depth"])
                if payload.get("last_answer_depth") is not None
                else None
            ),
            shallow_streak=int(payload.get("shallow_streak", 0)),
            reflected=bool(payload.get("reflected", False)),
            reflection_values=list(payload.get("reflection_values", [])),
            move_history=list(payload.get("move_history", [])),
            question_streak=int(payload.get("question_streak", 0)),
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
            reflect_back_count=int(payload.get("reflect_back_count", 0)),
            progress_floor=int(payload.get("progress_floor", 0)),
        )


@dataclass(slots=True, frozen=True)
class ReadinessResult:
    sufficient: bool
    missing_signals: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class GoalDecision:
    goal: ConversationGoal
    completion_reason: CompletionReason | None = None
    # companion 대화에서만 쓴다. 이번 턴의 반응 방식과 건너갈 때 쓸 후보 화제.
    move: ConversationMove | None = None
    scene: str | None = None
