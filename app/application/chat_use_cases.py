from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass

from app.application.conversation_prompts import build_friend_summary_messages
from app.application.conversation_service import CompletedTurn, ConversationService
from app.application.conversation_state_store import ConversationStateStore
from app.application.ports.model_gateway import ModelGateway
from app.domain.conversation.guards import fix_summary_intro, stable_reply_prefix
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
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
from app.domain.profile.onboarding import OnboardingProfile

logger = logging.getLogger(__name__)

ANALYSIS_KEYWORD_LIMIT = 3
_TASTE_KEYWORD_FIELDS = (TasteField.PREFERENCES,)
_INTEREST_KEYWORD_FIELDS = (TasteField.INTERESTS, TasteField.HOBBIES)


def _rounded(value: float | None) -> int | None:
    return round(value) if value is not None else None


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
class ReplyDelta:
    """스트리밍 중인 응답 조각. 모든 조각을 이어 붙이면 확정된 응답과 같다."""

    text: str


@dataclass(slots=True, frozen=True)
class ConfirmedTurn:
    """같은 사용자 메시지로 이미 확정한 턴. 다시 생성하지 않고 저장된 상태로 응답한다."""

    state: ConversationState


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

    async def _complete(
        self,
        messages,
        *,
        timeout_seconds: float,
        purpose: str,
        session_id: int,
    ) -> str:
        """응답 모델을 부른다.

        실패하면 원인을 로그 한 줄로 남기고 ResponseModelUnavailableError로 바꾼다.
        로그의 reason으로 실패를 나눈다. timeout은 이 서버가 기다리다 끊은 것이고,
        나머지는 게이트웨이가 붙인 이름(http_status, upstream_error, network 등)이다.
        """
        started = time.monotonic()
        try:
            async with asyncio.timeout(timeout_seconds):
                return await self._model_gateway.complete(
                    messages,
                    model=self._response_model,
                )
        except TimeoutError as error:
            logger.warning(
                "response model failed: purpose=%s session=%s model=%s elapsed_ms=%d "
                "reason=timeout limit_s=%s",
                purpose,
                session_id,
                self._response_model,
                (time.monotonic() - started) * 1000,
                timeout_seconds,
            )
            raise ResponseModelUnavailableError("timeout") from error
        except Exception as error:
            describe = getattr(error, "describe", None)
            detail = describe() if callable(describe) else f"reason=unexpected {error!r}"
            logger.warning(
                "response model failed: purpose=%s session=%s model=%s elapsed_ms=%d %s",
                purpose,
                session_id,
                self._response_model,
                (time.monotonic() - started) * 1000,
                detail,
                exc_info=True,
            )
            raise ResponseModelUnavailableError(getattr(error, "reason", "unexpected")) from error

    async def create_session(
        self,
        *,
        user_id: int,
        conversation_room_id: int,
        onboarding: OnboardingProfile | None = None,
    ) -> StartedSession:
        lock = self._lock(conversation_room_id)
        async with lock:
            if await self._states.load(conversation_room_id) is not None:
                raise SessionExistsError(conversation_room_id)
            prepared = self._conversations.prepare_session(
                user_id=user_id,
                conversation_room_id=conversation_room_id,
                onboarding=onboarding,
            )
            greeting = await self._complete(
                prepared.messages,
                timeout_seconds=self._response_timeout_seconds,
                purpose="opening",
                session_id=conversation_room_id,
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
            self._validate_sendable(state)

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
                    purpose="reply",
                    session_id=session_id,
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
                move=prepared.decision.move,
                scene=prepared.decision.scene,
            )
            await self._states.save(completed.state)
            return completed

    async def find_confirmed_turn(
        self,
        session_id: int,
        *,
        user_id: int,
        user_message_id: int,
    ) -> ConfirmedTurn | None:
        """스트림을 열기 전에 확인한다. 같은 사용자 메시지로 이미 확정한 턴이면 그 턴을 돌려준다.

        여기서 막히는 요청은 HTTP 오류로 응답한다. 확정된 턴은 입력이 잠긴 뒤에도 다시 받을 수
        있어야 하므로 상태 검사보다 먼저 본다.
        """
        if self._lock(session_id).locked():
            raise TurnInProgressError(session_id)
        state = await self._states.load(session_id)
        if state is None or state.user_id != user_id:
            raise SessionNotFoundError(session_id)
        if state.last_user_message_id == user_message_id:
            return ConfirmedTurn(state)
        self._validate_sendable(state)
        return None

    async def stream_message(
        self,
        session_id: int,
        message: str,
        *,
        user_id: int,
        user_message_id: int,
    ) -> AsyncIterator[ReplyDelta | CompletedTurn | ConfirmedTurn]:
        """응답 조각을 생성되는 대로 내보내고, 마지막에 확정한 턴을 내보낸다.

        턴은 끝까지 생성하고 정리한 뒤에만 저장한다. 도중에 실패하거나 호출한 쪽이 끊으면(취소)
        저장하지 않으므로, 같은 메시지를 다시 보내 처음부터 생성할 수 있다.
        """
        lock = self._lock(session_id)
        if lock.locked():
            raise TurnInProgressError(session_id)
        async with lock:
            state = await self._states.load(session_id)
            if state is None or state.user_id != user_id:
                raise SessionNotFoundError(session_id)
            # 잠금을 기다리는 사이 같은 메시지가 먼저 확정됐을 수 있다.
            if state.last_user_message_id == user_message_id:
                yield ConfirmedTurn(state)
                return
            self._validate_sendable(state)

            prepared = self._conversations.prepare_turn(state, message)
            # 마무리 턴만 끝 질문을 종료 멘트로 바꿀 수 있다. 생성 전에 정해지므로
            # 일반 턴은 질문까지 바로 내보낸다.
            hold_question = prepared.decision.goal == ConversationGoal.WRAP
            started = time.monotonic()
            first_token_ms: float | None = None
            first_delta_ms: float | None = None
            reply_ms: float | None = None
            delta_count = 0
            outcome = "cancelled"
            extraction = asyncio.create_task(self._conversations.extract(state, message))
            emitted = ""
            try:
                raw_reply = ""
                async for token in self._stream_reply(prepared.messages, session_id=session_id):
                    if first_token_ms is None:
                        first_token_ms = (time.monotonic() - started) * 1000
                    raw_reply += token
                    stable = stable_reply_prefix(raw_reply, hold_question=hold_question)
                    if len(stable) > len(emitted) and stable.startswith(emitted):
                        if first_delta_ms is None:
                            first_delta_ms = (time.monotonic() - started) * 1000
                        delta_count += 1
                        yield ReplyDelta(stable[len(emitted) :])
                        emitted = stable
                reply_ms = (time.monotonic() - started) * 1000
                completed = await self._conversations.complete_turn(
                    state,
                    utterance=message,
                    raw_reply=raw_reply,
                    goal=prepared.decision.goal,
                    completion_reason=prepared.decision.completion_reason,
                    extraction=await extraction,
                    move=prepared.decision.move,
                    scene=prepared.decision.scene,
                )
                completed.state.last_user_message_id = user_message_id
                await self._states.save(completed.state)

                if completed.reply.startswith(emitted):
                    if len(completed.reply) > len(emitted):
                        if first_delta_ms is None:
                            first_delta_ms = (time.monotonic() - started) * 1000
                        delta_count += 1
                        yield ReplyDelta(completed.reply[len(emitted) :])
                else:
                    # stable_reply_prefix가 정리 규칙과 어긋난 경우다. 확정 응답은 completed로 간다.
                    logger.warning(
                        "streamed reply diverged from final reply: session=%s emitted_chars=%d",
                        session_id,
                        len(emitted),
                    )
                outcome = "completed"
                yield completed
            except BaseException as error:
                extraction.cancel()
                if not isinstance(error, (asyncio.CancelledError, GeneratorExit)):
                    outcome = "failed"
                raise
            finally:
                # 대화 원문은 남기지 않고 시간과 조각 수만 남긴다.
                logger.info(
                    "reply stream: session=%s user_message=%s model=%s goal=%s outcome=%s "
                    "first_token_ms=%s first_delta_ms=%s reply_ms=%s total_ms=%d deltas=%d",
                    session_id,
                    user_message_id,
                    self._response_model,
                    prepared.decision.goal.value,
                    outcome,
                    _rounded(first_token_ms),
                    _rounded(first_delta_ms),
                    _rounded(reply_ms),
                    (time.monotonic() - started) * 1000,
                    delta_count,
                )

    async def _stream_reply(self, messages, *, session_id: int) -> AsyncIterator[str]:
        """응답 모델 스트림. 실패는 _complete와 같은 기준으로 로그를 남기고 바꾼다."""
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._response_timeout_seconds):
                async for token in self._model_gateway.stream(
                    messages,
                    model=self._response_model,
                ):
                    yield token
        except TimeoutError as error:
            logger.warning(
                "response model failed: purpose=reply_stream session=%s model=%s elapsed_ms=%d "
                "reason=timeout limit_s=%s",
                session_id,
                self._response_model,
                (time.monotonic() - started) * 1000,
                self._response_timeout_seconds,
            )
            raise ResponseModelUnavailableError("timeout") from error
        except Exception as error:
            describe = getattr(error, "describe", None)
            detail = describe() if callable(describe) else f"reason=unexpected {error!r}"
            logger.warning(
                "response model failed: purpose=reply_stream session=%s model=%s elapsed_ms=%d %s",
                session_id,
                self._response_model,
                (time.monotonic() - started) * 1000,
                detail,
                exc_info=True,
            )
            raise ResponseModelUnavailableError(getattr(error, "reason", "unexpected")) from error

    @staticmethod
    def _validate_sendable(state: ConversationState) -> None:
        if state.status == SessionStatus.CLOSED or state.finalized:
            raise SessionClosedError(state.session_id)
        if state.status in {SessionStatus.INPUT_LOCKED, SessionStatus.REVIEW}:
            raise SessionInputLockedError(state.session_id)

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
            name = state.onboarding.display_name if state.onboarding else None
            address = state.onboarding.address_name if state.onboarding else None
            if any(safe_values.values()):
                try:
                    summary = fix_summary_intro(
                        await self._complete(
                            build_friend_summary_messages(state.profile, name=name),
                            timeout_seconds=self._summary_timeout_seconds,
                            purpose="summary",
                            session_id=session_id,
                        ),
                        name=name,
                        address=address,
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
