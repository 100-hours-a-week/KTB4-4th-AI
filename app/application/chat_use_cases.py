from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass

from app.application.conversation_service import CompletedTurn, ConversationService
from app.application.conversation_state_store import ConversationStateStore
from app.application.llm_trace import bind_llm_session, llm_call
from app.application.ports.fault_injection import (
    ReplyDelayStore,
    TestAccountConfig,
    TestAccountStore,
)
from app.application.ports.model_gateway import ModelGateway
from app.application.ports.session_lock import SessionLease, SessionLock
from app.application.profile_keyword_prompts import KeywordSetPayload, build_keyword_messages
from app.application.profile_summary_prompts import build_friend_summary_messages
from app.application.search_intent_service import (
    SearchIntentService,
    intent_targets,
    onboarding_intent_targets,
)
from app.core.metrics import record_mock_reply, record_reply_delay
from app.domain.conversation.guards import fix_summary_intro, reply_question, stable_reply_prefix
from app.domain.conversation.models import (
    CompletionReason,
    ConversationGoal,
    ConversationState,
    GoalDecision,
    ReadinessResult,
    SessionStatus,
)
from app.domain.conversation.policy import MAX_TURNS, recommendation_readiness
from app.domain.profile.keywords import (
    KEYWORD_LIMIT,
    Keyword,
    composed_keywords,
    fallback_keywords,
)
from app.domain.profile.memory import LongTermMemory, combined_profile, merge_memory
from app.domain.profile.merger import friend_summary_values, normalize_text, ranked_friend_signals
from app.domain.profile.models import ExtractionDelta, ProfileState, TasteField, utc_now
from app.domain.profile.onboarding import OnboardingProfile

logger = logging.getLogger(__name__)

ANALYSIS_KEYWORD_LIMIT = KEYWORD_LIMIT
# 키워드를 묶을 때 모델에 넘기는 항목 수. 같은 결의 항목을 합칠 수 있게 노출 한도보다 넉넉히 준다.
KEYWORD_INPUT_LIMIT = 12
_TASTE_KEYWORD_FIELDS = (TasteField.PREFERENCES,)
_INTEREST_KEYWORD_FIELDS = (TasteField.INTERESTS, TasteField.HOBBIES)


# 질문 보완에서 니쥬가 대화를 마쳤다고 판단했을 때 모델이 질문 대신 쓰는 표시.
_END_MARK = "종료"


def _is_end_mark(text: str) -> bool:
    return text.strip().strip("'\"`.!~ ") == _END_MARK


# 부하 테스트 세션은 대화가 끝나지 않으므로 매 턴 읽고 쓰는 상태가 커지지 않게 최근 기록만 둔다.
# 프롬프트는 최근 기록만 쓰므로(conversation_prompts.MAX_HISTORY_MESSAGES) 응답 경로는 같다.
TEST_SESSION_HISTORY_LIMIT = 40

# 부하 테스트 Mock 응답. 모델 응답처럼 여러 조각으로 흘려보내고 매 턴 다른 질문으로 끝낸다.
MOCK_GREETING = "안녕하세요, 부하 테스트용 니쥬예요. 요즘 어떻게 지내세요?"


def _mock_reply_chunks(turn: int) -> tuple[str, ...]:
    return (
        "테스트 응답이에요. ",
        f"{turn}번째 이야기 고마워요. ",
        f"{turn + 1}번째 이야기도 들려줄래요?",
    )


async def _no_extraction() -> tuple[ExtractionDelta, bool]:
    return ExtractionDelta(), False


def _trim_test_history(state: ConversationState) -> None:
    if state.turn_limits_waived and len(state.history) > TEST_SESSION_HISTORY_LIMIT:
        state.history = state.history[-TEST_SESSION_HISTORY_LIMIT:]


def _turn_goal(
    decision: GoalDecision, ended: bool
) -> tuple[ConversationGoal, CompletionReason | None]:
    """니쥬가 응답에서 대화를 마쳤으면 이번 턴을 사용자 종료로 마무리한다."""
    if ended:
        return ConversationGoal.WRAP, CompletionReason.USER_EXIT
    return decision.goal, decision.completion_reason


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
    # 대화를 닫을 때만 채운다. Backend가 저장했다가 다음 대화방을 만들 때 다시 보낸다.
    memory: LongTermMemory | None = None


class ChatUseCases:
    def __init__(
        self,
        *,
        conversations: ConversationService,
        states: ConversationStateStore,
        model_gateway: ModelGateway,
        response_model: str,
        session_lock: SessionLock,
        lock_wait_seconds: float = 20.0,
        response_timeout_seconds: float = 60.0,
        summary_timeout_seconds: float = 10.0,
        search_intents: SearchIntentService | None = None,
        keyword_model: str | None = None,
        reply_delay: ReplyDelayStore | None = None,
        reply_delay_max_seconds: float = 10.0,
        test_accounts: TestAccountStore | None = None,
    ) -> None:
        """search_intents 가 있으면 새로 들어온 관심사의 검색 의도를 백그라운드로 만들어 둔다.

        keyword_model 이 있으면 분석 키워드를 그 모델로 묶고, 없으면 저장 값에서 만든다.

        메시지 전송은 같은 세션이 처리 중이면 바로 거절하고, 나머지는 lock_wait_seconds 까지
        기다린다. Backend 호출 타임아웃보다 짧게 둬서 끊기기 전에 처리 중 오류로 답한다.
        """
        self._conversations = conversations
        self._keyword_model = keyword_model
        self._search_intents = search_intents
        self._states = states
        self._model_gateway = model_gateway
        self._response_model = response_model
        self._response_timeout_seconds = response_timeout_seconds
        self._summary_timeout_seconds = summary_timeout_seconds
        self._session_lock = session_lock
        self._lock_wait_seconds = lock_wait_seconds
        self._reply_delay = reply_delay
        self._reply_delay_max_seconds = reply_delay_max_seconds
        self._test_accounts = test_accounts

    async def _injected_reply_delay(self, *, session_id: int, user_id: int) -> float:
        """부하·장애 테스트용 지연 시간. 설정이 없거나 읽지 못하면 0이라 평소 요청은 그대로 간다."""
        if self._reply_delay is None:
            return 0.0
        try:
            config = await self._reply_delay.get()
        except Exception:
            logger.warning("reply delay config read failed: session=%s", session_id, exc_info=True)
            return 0.0
        if config is None:
            return 0.0
        return min(
            config.delay_for(session_id=session_id, user_id=user_id),
            self._reply_delay_max_seconds,
        )

    async def _delay_reply_for_test(self, *, session_id: int, user_id: int) -> None:
        delay = await self._injected_reply_delay(session_id=session_id, user_id=user_id)
        if delay > 0:
            logger.info("fault injection: reply delay session=%s delay_s=%s", session_id, delay)
            record_reply_delay(delay)
            await asyncio.sleep(delay)

    async def _test_account(self, *, session_id: int, user_id: int) -> TestAccountConfig | None:
        """user_id가 승인된 부하 테스트 계정이면 그 설정. 목록을 읽지 못하면 일반 사용자로 본다."""
        if self._test_accounts is None:
            return None
        try:
            config = await self._test_accounts.get()
        except Exception:
            logger.warning("test account list read failed: session=%s", session_id, exc_info=True)
            return None
        if config is None or user_id not in config.user_ids:
            return None
        return config

    @staticmethod
    def _mark_test_session(state: ConversationState, config: TestAccountConfig) -> None:
        state.turn_limits_waived = True
        state.mock_model_calls = config.mock_model
        logger.info(
            "load test account: session=%s user=%s mock_model=%s",
            state.session_id,
            state.user_id,
            config.mock_model,
        )

    async def _apply_test_account_policy(self, state: ConversationState, user_id: int) -> None:
        """승인된 부하 테스트 계정이면 이 세션의 턴 제한과 자동 마무리를 끄고 Mock 여부를 정한다.

        한 번 정하면 세션 상태에 남아서 계정 목록이 만료돼도 그 세션은 끝까지 같은 정책을 쓴다.
        """
        if state.turn_limits_waived:
            return
        config = await self._test_account(session_id=state.session_id, user_id=user_id)
        if config is not None:
            self._mark_test_session(state, config)

    async def _mock_reply(self, state: ConversationState, *, user_id: int) -> AsyncIterator[str]:
        """모델 대신 고정 응답을 흘려보낸다. 응답 지연 설정은 실제 모델 호출과 똑같이 받는다."""
        await self._delay_reply_for_test(session_id=state.session_id, user_id=user_id)
        record_mock_reply("reply")
        for chunk in _mock_reply_chunks(state.turn_count + 1):
            yield chunk

    def _prepare_intents(self, completed: CompletedTurn) -> None:
        """이번 턴에 저장된 관심사의 검색 의도를 만든다. 세션 종료 때 추천이 캐시를 읽는다."""
        if self._search_intents is not None:
            self._search_intents.schedule(intent_targets(completed.merge_result.accepted))

    @asynccontextmanager
    async def _hold(self, session_id: int, *, wait: bool = True) -> AsyncIterator[SessionLease]:
        lease = await self._session_lock.acquire(
            session_id,
            wait_seconds=self._lock_wait_seconds if wait else 0.0,
        )
        if lease is None:
            raise TurnInProgressError(session_id)
        try:
            yield lease
        finally:
            # 끊긴 스트림의 취소 중에도 락을 풀어야 재전송이 바로 들어온다.
            await asyncio.shield(lease.release())

    @staticmethod
    async def _ensure_held(lease: SessionLease, session_id: int) -> None:
        """저장 직전에 확인한다. 락을 잃었으면 다른 요청이 같은 턴을 만들 수 있다."""
        if not await lease.renew():
            raise TurnInProgressError(session_id)

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
                with llm_call(purpose):
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

    async def _ensure_reply_question(
        self,
        reply: str,
        messages,
        *,
        goal: ConversationGoal,
        session_id: int,
        lenient: bool = False,
    ) -> tuple[str, bool]:
        """질문이 빠졌거나 반복된 응답에 질문을 보완한다. 두 번째 값은 니쥬가 대화를 마쳤는지다.

        니쥬가 방금 응답에서 대화를 마치겠다고 했으면 질문을 붙이지 않고 마무리 턴으로 바꾼다.
        사용자 말의 표현이 아니라 니쥬 응답을 보고 판단하므로, 사용자가 공부나 관계를
        그만두고 싶다는 이야기는 니쥬가 질문을 이어 가는 한 종료로 보지 않는다.

        lenient면(부하 테스트 계정) 보완해도 질문이 빠지거나 반복될 때 실패하지 않고 원래
        응답을 쓴다. 같은 문장을 반복해서 보내는 테스트가 응답 품질 검사로 끊기지 않게 한다.
        """
        cleaned = self._conversations.guard_reply(reply, goal)
        if goal == ConversationGoal.WRAP:
            return cleaned, False
        question = reply_question(cleaned)
        asked_questions = {
            normalize_text(reply_question(message["content"]))
            for message in messages
            if message["role"] == "assistant"
        }
        if question and normalize_text(question) not in asked_questions:
            return cleaned, False
        # 질문은 검증 전까지 스트리밍을 보류하므로 반응은 유지한 채 질문만 보완할 수 있다.
        reaction = cleaned[: len(cleaned) - len(question)].rstrip() if question else cleaned
        repair = await self._complete(
            [
                *messages,
                {"role": "assistant", "content": reaction},
                {
                    "role": "system",
                    "content": "방금 반응 뒤에 이어질 질문 하나만 작성한다. "
                    "원래 대화의 직전 질문과 최신 사용자 답을 함께 읽는다. "
                    "공통 지침과, 이번 턴 메모가 있으면 그 메모에 따라 질문의 화제를 고른다. "
                    "한두 마디로 답할 수 있게 묻는다. "
                    "이미 답했거나 모르겠다고 한 내용과 그 이유를 다시 묻지 않는다. "
                    "추출 항목을 채우려고 화제를 바꾸지 않는다. "
                    "인사·설명·맞장구 없이 질문 한 문장만 쓴다. "
                    + (
                        ""
                        if goal == ConversationGoal.OPENING
                        else "단, 방금 반응에서 니쥬가 대화를 여기서 마치겠다거나 더 묻지 않겠다고 "
                        "사용자에게 말했으면 질문 대신 "
                        f"'{_END_MARK}'만 쓴다. 사용자가 공부, 일, 관계 같은 다른 일을 그만두고 "
                        "싶다고 한 이야기는 대화를 마치는 것이 아니다."
                    ),
                },
            ],
            timeout_seconds=self._response_timeout_seconds,
            purpose="question_repair",
            session_id=session_id,
        )
        if goal != ConversationGoal.OPENING and _is_end_mark(repair):
            if not reaction:
                if lenient:
                    return cleaned, False
                raise ResponseModelUnavailableError("question_missing_or_repeated")
            return reaction, True
        question = reply_question(self._conversations.guard_reply(repair, goal))
        if not question or normalize_text(question) in asked_questions:
            if lenient:
                return cleaned, False
            raise ResponseModelUnavailableError("question_missing_or_repeated")
        if reaction and reaction[-1] not in ".!~ㅎㅠ":
            reaction += "."
        result = f"{reaction} {question}".strip()
        if not reply_question(result):
            if lenient:
                return cleaned, False
            raise ResponseModelUnavailableError("question_not_renderable")
        return result, False

    async def create_session(
        self,
        *,
        user_id: int,
        conversation_room_id: int,
        onboarding: OnboardingProfile | None = None,
        memory: LongTermMemory | None = None,
    ) -> StartedSession:
        bind_llm_session(conversation_room_id, user_id=user_id)
        async with self._hold(conversation_room_id) as lease:
            if await self._states.load(conversation_room_id) is not None:
                raise SessionExistsError(conversation_room_id)
            prepared = self._conversations.prepare_session(
                user_id=user_id,
                conversation_room_id=conversation_room_id,
                onboarding=onboarding,
                memory=memory,
            )
            test_account = await self._test_account(
                session_id=conversation_room_id, user_id=user_id
            )
            if test_account is not None:
                self._mark_test_session(prepared.state, test_account)
            if prepared.state.mock_model_calls:
                greeting = MOCK_GREETING
                record_mock_reply("opening")
            else:
                greeting = await self._complete(
                    prepared.messages,
                    timeout_seconds=self._response_timeout_seconds,
                    purpose="opening",
                    session_id=conversation_room_id,
                )
                greeting, _ = await self._ensure_reply_question(
                    greeting,
                    prepared.messages,
                    goal=ConversationGoal.OPENING,
                    session_id=conversation_room_id,
                )
            greeting = self._conversations.record_opening(prepared.state, greeting)
            await self._ensure_held(lease, conversation_room_id)
            await self._states.save(prepared.state)
            if (
                self._search_intents is not None
                and onboarding is not None
                and not prepared.state.mock_model_calls
            ):
                self._search_intents.schedule(onboarding_intent_targets(onboarding.interests))
            return StartedSession(state=prepared.state, greeting=greeting)

    async def send_message(
        self,
        session_id: int,
        message: str,
        *,
        user_id: int,
    ) -> CompletedTurn:
        bind_llm_session(session_id, user_id=user_id)
        async with self._hold(session_id, wait=False) as lease:
            state = await self._states.load(session_id)
            if state is None or state.user_id != user_id:
                raise SessionNotFoundError(session_id)
            self._validate_sendable(state)
            await self._apply_test_account_policy(state, user_id)

            prepared = self._conversations.prepare_turn(
                state,
                message,
            )
            mock = state.mock_model_calls
            # 추출은 응답 결과를 쓰지 않으므로 응답 생성과 동시에 시작한다.
            # 한 턴의 시간은 둘의 합이 아니라 더 오래 걸리는 쪽이 된다.
            extraction = asyncio.create_task(
                _no_extraction() if mock else self._conversations.extract(state, message)
            )
            try:
                if mock:
                    chunks = [chunk async for chunk in self._mock_reply(state, user_id=user_id)]
                    raw_reply = self._conversations.guard_reply(
                        "".join(chunks), prepared.decision.goal
                    )
                    ended = False
                else:
                    await self._delay_reply_for_test(session_id=session_id, user_id=user_id)
                    raw_reply = await self._complete(
                        prepared.messages,
                        timeout_seconds=self._response_timeout_seconds,
                        purpose="reply",
                        session_id=session_id,
                    )
                    raw_reply, ended = await self._ensure_reply_question(
                        raw_reply,
                        prepared.messages,
                        goal=prepared.decision.goal,
                        session_id=session_id,
                        lenient=state.turn_limits_waived,
                    )
            except BaseException:
                extraction.cancel()
                raise
            goal, completion_reason = _turn_goal(
                prepared.decision, ended and not state.turn_limits_waived
            )
            completed = await self._conversations.complete_turn(
                state,
                utterance=message,
                raw_reply=raw_reply,
                goal=goal,
                completion_reason=completion_reason,
                extraction=await extraction,
                move=prepared.decision.move,
                scene=prepared.decision.scene,
            )
            _trim_test_history(completed.state)
            await self._ensure_held(lease, session_id)
            await self._states.save(completed.state)
            if not mock:
                self._prepare_intents(completed)
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
        if await self._session_lock.is_locked(session_id):
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
        bind_llm_session(session_id, user_id=user_id)
        async with self._hold(session_id, wait=False) as lease:
            state = await self._states.load(session_id)
            if state is None or state.user_id != user_id:
                raise SessionNotFoundError(session_id)
            # 잠금을 기다리는 사이 같은 메시지가 먼저 확정됐을 수 있다.
            if state.last_user_message_id == user_message_id:
                yield ConfirmedTurn(state)
                return
            self._validate_sendable(state)
            await self._apply_test_account_policy(state, user_id)

            prepared = self._conversations.prepare_turn(state, message)
            # 질문 누락·반복을 고쳐도 이미 보낸 반응은 바뀌지 않도록 질문 문장을 보류한다.
            hold_question = True
            started = time.monotonic()
            first_token_ms: float | None = None
            first_delta_ms: float | None = None
            reply_ms: float | None = None
            delta_count = 0
            outcome = "cancelled"
            goal = prepared.decision.goal
            mock = state.mock_model_calls
            extraction = asyncio.create_task(
                _no_extraction() if mock else self._conversations.extract(state, message)
            )
            reply_tokens = (
                self._mock_reply(state, user_id=user_id)
                if mock
                else self._stream_reply(prepared.messages, session_id=session_id, user_id=user_id)
            )
            emitted = ""
            try:
                raw_reply = ""
                async for token in reply_tokens:
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
                if mock:
                    raw_reply = self._conversations.guard_reply(raw_reply, prepared.decision.goal)
                    ended = False
                else:
                    raw_reply, ended = await self._ensure_reply_question(
                        raw_reply,
                        prepared.messages,
                        goal=prepared.decision.goal,
                        session_id=session_id,
                        lenient=state.turn_limits_waived,
                    )
                goal, completion_reason = _turn_goal(
                    prepared.decision, ended and not state.turn_limits_waived
                )
                if goal != ConversationGoal.WRAP and raw_reply.startswith(emitted):
                    remainder = raw_reply[len(emitted) :]
                    if remainder:
                        if first_delta_ms is None:
                            first_delta_ms = (time.monotonic() - started) * 1000
                        delta_count += 1
                        yield ReplyDelta(remainder)
                        emitted = raw_reply
                completed = await self._conversations.complete_turn(
                    state,
                    utterance=message,
                    raw_reply=raw_reply,
                    goal=goal,
                    completion_reason=completion_reason,
                    extraction=await extraction,
                    move=prepared.decision.move,
                    scene=prepared.decision.scene,
                )
                completed.state.last_user_message_id = user_message_id
                _trim_test_history(completed.state)
                await self._ensure_held(lease, session_id)
                await self._states.save(completed.state)
                if not mock:
                    self._prepare_intents(completed)

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
                    "mock" if mock else self._response_model,
                    goal.value,
                    outcome,
                    _rounded(first_token_ms),
                    _rounded(first_delta_ms),
                    _rounded(reply_ms),
                    (time.monotonic() - started) * 1000,
                    delta_count,
                )

    async def _stream_reply(self, messages, *, session_id: int, user_id: int) -> AsyncIterator[str]:
        """응답 모델 스트림. 실패는 _complete와 같은 기준으로 로그를 남기고 바꾼다.

        테스트용 지연은 응답 제한 시간과 모델 호출 추적 바깥에서 건다. 그래야 지연이 응답
        제한 시간을 잡아먹지 않고, Langfuse의 모델 지연 시간에도 섞이지 않는다.
        """
        await self._delay_reply_for_test(session_id=session_id, user_id=user_id)
        started = time.monotonic()
        try:
            async with asyncio.timeout(self._response_timeout_seconds):
                with llm_call("reply"):
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

    @staticmethod
    def _analysis_profile(state: ConversationState) -> ProfileState:
        """추천에 쓰는 프로필. 지난 대화의 기억 위에 이번 대화 결과를 얹는다.

        요약과 키워드는 그날 대화만 보여 주므로 state.profile 을 쓴다.
        """
        return combined_profile(
            state.memory,
            state.profile,
            now=state.last_active_at,
            dropped=state.memory_dropped,
        )

    def _stored_analysis(
        self,
        state: ConversationState,
        *,
        memory: LongTermMemory | None = None,
    ) -> ProfileAnalysis:
        return ProfileAnalysis(
            state=state,
            profile=self._analysis_profile(state),
            summary=state.analysis_summary,
            taste_keywords=tuple(state.analysis_taste_keywords),
            interest_keywords=tuple(state.analysis_interest_keywords),
            keyword_scores=dict(state.analysis_keyword_scores),
            readiness=recommendation_readiness(state),
            memory=memory,
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
        async with self._hold(session_id):
            state = await self._states.load(session_id)
            if state is None or state.user_id != user_id:
                raise SessionNotFoundError(session_id)
            self._validate_closable(state)
            return self._stored_analysis(state)

    async def analyze_session(self, session_id: int) -> ProfileAnalysis:
        async with self._hold(session_id) as lease:
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

            bind_llm_session(session_id, user_id=state.user_id)
            # 요약과 키워드는 서로의 결과를 쓰지 않으므로 함께 만든다.
            summary, (taste_keywords, interest_keywords) = await asyncio.gather(
                self._friend_summary(state, state.profile, session_id),
                self._keywords(state.profile, session_id),
            )

            state.analysis_turn_count = state.turn_count
            state.analysis_summary = summary
            state.analysis_taste_keywords = [keyword.label for keyword in taste_keywords]
            state.analysis_interest_keywords = [keyword.label for keyword in interest_keywords]
            state.analysis_keyword_scores = {
                keyword.label: keyword.score for keyword in (*taste_keywords, *interest_keywords)
            }
            state.analysis_patch_used = False
            state.status = SessionStatus.REVIEW
            await self._ensure_held(lease, session_id)
            await self._states.save(state)
            return self._stored_analysis(state)

    async def _friend_summary(
        self, state: ConversationState, profile: ProfileState, session_id: int
    ) -> str | None:
        if not any(friend_summary_values(profile).values()):
            return None
        name = state.onboarding.display_name if state.onboarding else None
        address = state.onboarding.address_name if state.onboarding else None
        try:
            return fix_summary_intro(
                await self._complete(
                    build_friend_summary_messages(profile, name=name),
                    timeout_seconds=self._summary_timeout_seconds,
                    purpose="summary",
                    session_id=session_id,
                ),
                name=name,
                address=address,
            )
        except ResponseModelUnavailableError:
            return None

    async def _keywords(
        self, profile: ProfileState, session_id: int
    ) -> tuple[list[Keyword], list[Keyword]]:
        """취향·관심사 키워드. 모델이 저장 값을 묶어 이름 붙이고, 못 쓰면 저장 값에서 만든다.

        취향과 관심사는 따로 검증하고 따로 대체한다. 한쪽만 틀려도 다른 쪽 결과는 살린다.
        """
        # 요약처럼 활성 한도 밖으로 밀린 항목도 넘긴다. 같은 결의 항목을 묶는 재료가 된다.
        tastes = ranked_friend_signals(profile, _TASTE_KEYWORD_FIELDS, include_inactive=True)[
            :KEYWORD_INPUT_LIMIT
        ]
        interests = ranked_friend_signals(profile, _INTEREST_KEYWORD_FIELDS, include_inactive=True)[
            :KEYWORD_INPUT_LIMIT
        ]
        raw: Mapping[str, object] = {}
        if self._keyword_model is not None and (tastes or interests):
            started = time.monotonic()
            try:
                async with asyncio.timeout(self._summary_timeout_seconds):
                    with llm_call("keywords"):
                        raw = await self._model_gateway.structured(
                            build_keyword_messages(tastes, interests),
                            model=self._keyword_model,
                            json_schema=KeywordSetPayload.model_json_schema(),
                        )
            except Exception as error:
                describe = getattr(error, "describe", None)
                logger.warning(
                    "keyword model failed: session=%s model=%s elapsed_ms=%d %s",
                    session_id,
                    self._keyword_model,
                    (time.monotonic() - started) * 1000,
                    describe() if callable(describe) else f"reason={type(error).__name__}",
                )
        if not isinstance(raw, Mapping):
            raw = {}
        composed = (
            composed_keywords(raw.get("taste"), tastes, taste=True),
            composed_keywords(raw.get("interest"), interests, taste=False),
        )
        return (
            composed[0] or fallback_keywords(tastes, taste=True),
            composed[1] or fallback_keywords(interests, taste=False),
        )

    async def update_analysis(
        self,
        session_id: int,
        *,
        user_id: int,
        summary: str | None,
        taste_keywords: list[str],
        interest_keywords: list[str],
    ) -> ProfileAnalysis:
        async with self._hold(session_id):
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
        async with self._hold(session_id):
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
            memory = merge_memory(
                state.memory,
                state.profile,
                user_id=state.user_id,
                now=utc_now(),
                dropped=state.memory_dropped,
                recalled=state.memory_recalled,
            )
            return self._stored_analysis(state, memory=memory)
