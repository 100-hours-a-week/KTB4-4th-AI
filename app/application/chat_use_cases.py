from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass

from app.application.conversation_prompts import build_friend_summary_messages
from app.application.conversation_service import CompletedTurn, ConversationService
from app.application.conversation_state_store import ConversationStateStore
from app.application.ports.model_gateway import ModelGateway
from app.domain.conversation.models import (
    CompletionReason,
    ConversationState,
    ReadinessResult,
    SessionStatus,
)
from app.domain.conversation.policy import MAX_TURNS, recommendation_readiness
from app.domain.profile.merger import (
    friend_summary_values,
    ranked_friend_signals,
    taste_rank_score,
)
from app.domain.profile.models import ProfileState, TasteField

ANALYSIS_KEYWORD_LIMIT = 3
_TASTE_KEYWORD_FIELDS = (TasteField.PREFERENCES,)
_INTEREST_KEYWORD_FIELDS = (TasteField.INTERESTS, TasteField.HOBBIES)


class ChatUseCaseError(RuntimeError):
    pass


class SessionNotFoundError(ChatUseCaseError):
    pass


class SessionExistsError(ChatUseCaseError):
    pass


class SessionClosedError(ChatUseCaseError):
    pass


class SessionInputLockedError(ChatUseCaseError):
    pass


class TurnInProgressError(ChatUseCaseError):
    pass


class ResponseModelUnavailableError(ChatUseCaseError):
    pass


class ProfileTooSparseError(ChatUseCaseError):
    pass


class AnalysisRequiredError(ChatUseCaseError):
    pass


class AnalysisOutdatedError(ChatUseCaseError):
    pass


class AnalysisAlreadyCorrectedError(ChatUseCaseError):
    pass


class AnalysisKeywordsCanOnlyBeDeletedError(ChatUseCaseError):
    pass


@dataclass(slots=True, frozen=True)
class StartedSession:
    state: ConversationState
    greeting: str


@dataclass(slots=True, frozen=True)
class ProfileAnalysis:
    state: ConversationState
    profile: ProfileState
    summary: str | None
    taste_keywords: tuple[str, ...]
    interest_keywords: tuple[str, ...]
    keyword_scores: Mapping[str, float]
    readiness: ReadinessResult


class ChatUseCases:
    def __init__(
        self,
        *,
        conversations: ConversationService,
        states: ConversationStateStore,
        model_gateway: ModelGateway,
        response_model: str,
        response_timeout_seconds: float = 60.0,
        summary_timeout_seconds: float = 10.0,
    ) -> None:
        self._conversations = conversations
        self._states = states
        self._model_gateway = model_gateway
        self._response_model = response_model
        self._response_timeout_seconds = response_timeout_seconds
        self._summary_timeout_seconds = summary_timeout_seconds
        self._locks: dict[int, asyncio.Lock] = {}

    def _lock(self, session_id: int) -> asyncio.Lock:
        return self._locks.setdefault(session_id, asyncio.Lock())

    async def _complete(self, messages, *, timeout_seconds: float) -> str:
        try:
            async with asyncio.timeout(timeout_seconds):
                return await self._model_gateway.complete(
                    messages,
                    model=self._response_model,
                )
        except Exception as error:
            raise ResponseModelUnavailableError from error

    async def create_session(self, *, user_id: int, conversation_room_id: int) -> StartedSession:
        lock = self._lock(conversation_room_id)
        async with lock:
            if await self._states.load(conversation_room_id) is not None:
                raise SessionExistsError(conversation_room_id)
            prepared = self._conversations.prepare_session(
                user_id=user_id,
                conversation_room_id=conversation_room_id,
            )
            greeting = await self._complete(
                prepared.messages,
                timeout_seconds=self._response_timeout_seconds,
            )
            greeting = self._conversations.record_opening(prepared.state, greeting)
            await self._states.save(prepared.state)
            return StartedSession(state=prepared.state, greeting=greeting)

    async def send_message(
        self,
        session_id: int,
        message: str,
        *,
        user_id: int,
    ) -> CompletedTurn:
        lock = self._lock(session_id)
        if lock.locked():
            raise TurnInProgressError(session_id)
        async with lock:
            state = await self._states.load(session_id)
            if state is None or state.user_id != user_id:
                raise SessionNotFoundError(session_id)
            if state.status == SessionStatus.CLOSED or state.finalized:
                raise SessionClosedError(session_id)
            if state.status in {SessionStatus.INPUT_LOCKED, SessionStatus.REVIEW}:
                raise SessionInputLockedError(session_id)

            prepared = self._conversations.prepare_turn(
                state,
                message,
            )
            # 추출은 응답 결과를 쓰지 않으므로 응답 생성과 동시에 시작한다.
            # 한 턴의 시간은 둘의 합이 아니라 더 오래 걸리는 쪽이 된다.
            extraction = asyncio.create_task(self._conversations.extract(state, message))
            try:
                raw_reply = await self._complete(
                    prepared.messages,
                    timeout_seconds=self._response_timeout_seconds,
                )
            except BaseException:
                extraction.cancel()
                raise
            completed = await self._conversations.complete_turn(
                state,
                utterance=message,
                raw_reply=raw_reply,
                goal=prepared.decision.goal,
                completion_reason=prepared.decision.completion_reason,
                extraction=await extraction,
            )
            await self._states.save(completed.state)
            return completed

    def _stored_analysis(self, state: ConversationState) -> ProfileAnalysis:
        return ProfileAnalysis(
            state=state,
            profile=state.profile,
            summary=state.analysis_summary,
            taste_keywords=tuple(state.analysis_taste_keywords),
            interest_keywords=tuple(state.analysis_interest_keywords),
            keyword_scores=dict(state.analysis_keyword_scores),
            readiness=recommendation_readiness(state),
        )

    @staticmethod
    def _validate_closable(state: ConversationState) -> None:
        if state.status == SessionStatus.CLOSED or state.finalized:
            raise SessionClosedError(state.session_id)
        if state.analysis_turn_count is None:
            raise AnalysisRequiredError(state.session_id)
        if state.analysis_turn_count != state.turn_count:
            raise AnalysisOutdatedError(state.session_id)

    @staticmethod
    def _profile_too_sparse(state: ConversationState) -> bool:
        if state.turn_count < MAX_TURNS:
            return not state.profile.active_signals()
        # 최대 턴까지 대화했는데 분석 키워드로 보여줄 취향·관심사가 하나도 없으면 분석하지 않는다.
        return not ranked_friend_signals(
            state.profile, (*_TASTE_KEYWORD_FIELDS, *_INTEREST_KEYWORD_FIELDS)
        )

    async def get_close_analysis(self, session_id: int, *, user_id: int) -> ProfileAnalysis:
        async with self._lock(session_id):
            state = await self._states.load(session_id)
            if state is None or state.user_id != user_id:
                raise SessionNotFoundError(session_id)
            self._validate_closable(state)
            return self._stored_analysis(state)

    async def analyze_session(self, session_id: int) -> ProfileAnalysis:
        async with self._lock(session_id):
            state = await self._states.load(session_id)
            if state is None:
                raise SessionNotFoundError(session_id)
            if state.status == SessionStatus.CLOSED or state.finalized:
                raise SessionClosedError(session_id)

            if self._profile_too_sparse(state):
                # 새 대화로 다시 시작하도록 세션을 만료시켜 같은 방 번호로 재생성할 수 있게 한다.
                await self._states.delete(session_id)
                raise ProfileTooSparseError(session_id)

            if state.analysis_turn_count == state.turn_count:
                state.status = SessionStatus.REVIEW
                await self._states.save(state)
                return self._stored_analysis(state)

            safe_values = friend_summary_values(state.profile)
            summary = None
            if any(safe_values.values()):
                try:
                    summary = await self._complete(
                        build_friend_summary_messages(state.profile),
                        timeout_seconds=self._summary_timeout_seconds,
                    )
                except ResponseModelUnavailableError:
                    summary = None

            state.analysis_turn_count = state.turn_count
            state.analysis_summary = summary
            taste_signals = ranked_friend_signals(state.profile, _TASTE_KEYWORD_FIELDS)[
                :ANALYSIS_KEYWORD_LIMIT
            ]
            interest_signals = ranked_friend_signals(state.profile, _INTEREST_KEYWORD_FIELDS)[
                :ANALYSIS_KEYWORD_LIMIT
            ]
            state.analysis_taste_keywords = [signal.value for signal in taste_signals]
            state.analysis_interest_keywords = [signal.value for signal in interest_signals]
            state.analysis_keyword_scores = {
                signal.value: taste_rank_score(signal)
                for signal in (*taste_signals, *interest_signals)
            }
            state.analysis_patch_used = False
            state.status = SessionStatus.REVIEW
            await self._states.save(state)
            return self._stored_analysis(state)

    async def update_analysis(
        self,
        session_id: int,
        *,
        user_id: int,
        summary: str | None,
        taste_keywords: list[str],
        interest_keywords: list[str],
    ) -> ProfileAnalysis:
        async with self._lock(session_id):
            state = await self._states.load(session_id)
            if state is None or state.user_id != user_id:
                raise SessionNotFoundError(session_id)
            self._validate_closable(state)
            if state.analysis_patch_used:
                raise AnalysisAlreadyCorrectedError(session_id)
            if not set(taste_keywords).issubset(state.analysis_taste_keywords) or not set(
                interest_keywords
            ).issubset(state.analysis_interest_keywords):
                raise AnalysisKeywordsCanOnlyBeDeletedError(session_id)

            state.analysis_summary = summary
            # 키워드는 지우기만 가능하므로 남은 키워드는 분석 때의 점수 순서를 따른다.
            state.analysis_taste_keywords = [
                keyword for keyword in state.analysis_taste_keywords if keyword in taste_keywords
            ]
            state.analysis_interest_keywords = [
                keyword
                for keyword in state.analysis_interest_keywords
                if keyword in interest_keywords
            ]
            state.analysis_patch_used = True
            state.status = SessionStatus.REVIEW
            await self._states.save(state)
            return self._stored_analysis(state)

    async def close_session(self, session_id: int, *, user_id: int) -> ProfileAnalysis:
        async with self._lock(session_id):
            state = await self._states.load(session_id)
            if state is None or state.user_id != user_id:
                raise SessionNotFoundError(session_id)
            self._validate_closable(state)

            readiness = recommendation_readiness(state)
            state.status = SessionStatus.CLOSED
            state.completion_reason = state.completion_reason or (
                CompletionReason.SUFFICIENT if readiness.sufficient else CompletionReason.USER_EXIT
            )
            state.finalized = True
            await self._states.save(state)
            return self._stored_analysis(state)
