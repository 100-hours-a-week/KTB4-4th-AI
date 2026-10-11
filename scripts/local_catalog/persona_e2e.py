"""페르소나 하나로 대화부터 추천까지 로컬에서 끝까지 돌려 본다.

대화 모델과 추출 모델은 운영과 같은 ChatUseCases 를 쓰고, 사용자 답변은 페르소나 설정을 받은
모델이 대신 쓴다. 대화가 끝나면 운영 종료 처리와 같은 경로로 추천을 만들고 로컬 카탈로그에서 찾는다.

실행:
    CATALOG_DATABASE_URL=postgresql://$(whoami)@127.0.0.1:5432/needu_catalog_local \\
        uv run python scripts/local_catalog/persona_e2e.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import asyncpg
import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.api.routes.v1_chat import recommendation_profile  # noqa: E402
from app.application.chat_use_cases import ChatUseCases  # noqa: E402
from app.application.conversation_service import ConversationService  # noqa: E402
from app.application.conversation_state_store import ConversationStateStore  # noqa: E402
from app.application.recommendation_engine import RecommendationEngine  # noqa: E402
from app.application.recommendation_service import (  # noqa: E402
    V1RecommendationService,
    recommendation_inputs,
)
from app.core.config import get_settings  # noqa: E402
from app.domain.conversation.models import ConversationStyle, SessionStatus  # noqa: E402
from app.domain.conversation.policy import recommendation_readiness  # noqa: E402
from app.domain.recommendation import build_search_queries  # noqa: E402
from app.infrastructure.embedding import OpenAICompatibleEmbedder  # noqa: E402
from app.infrastructure.model_gateway import (  # noqa: E402
    OpenAICompatibleModelGateway,
    SystemOneJudgmentGateway,
)
from app.infrastructure.persistence import InMemorySessionStore  # noqa: E402
from app.infrastructure.persistence.postgres_catalog_repository import (  # noqa: E402
    PostgresCatalogRepository,
)

PERSONA = """너는 선물 추천 서비스와 대화하는 28살 직장인 사용자다.
아래 설정대로, 메신저에서 친구에게 말하듯 짧고 자연스럽게 한두 문장으로 답한다.
설정에 없는 내용은 지어내지 말고 잘 모르겠다고 하거나 넘어간다.
한 번에 모든 취향을 늘어놓지 말고 질문받은 것 위주로 답한다.

- 아침마다 핸드드립으로 커피를 내려 마시는 게 낙이다. 산미 있는 원두를 좋아한다.
- 주말에 친구들과 오토캠핑을 자주 간다. 캠핑장에서 커피 내려 마시는 시간을 제일 좋아한다.
- 요즘 퇴근 후에 러닝을 시작했다. 아직 5km 정도 뛴다.
- 물건은 무채색이나 베이지처럼 차분한 색, 심플한 디자인을 좋아한다.
- 향이 강한 제품(향수, 디퓨저, 방향제)은 머리가 아파서 싫어한다.
- 화장품이나 액세서리에는 관심이 거의 없다."""

MAX_TURNS = 14


async def persona_reply(gateway, model: str, history: list[dict[str, str]]) -> str:
    # 서비스 입장의 assistant/user 를 뒤집어 페르소나 모델에게 넘긴다.
    flipped = [
        {"role": "user" if turn["role"] == "assistant" else "assistant", "content": turn["content"]}
        for turn in history
    ]
    reply = await gateway.complete([{"role": "system", "content": PERSONA}, *flipped], model=model)
    return reply.strip()


async def main() -> None:
    settings = get_settings()
    secret = settings.model_api_key or settings.openrouter_api_key
    api_key = secret.get_secret_value() if secret is not None else None
    client = httpx.AsyncClient(timeout=None)
    pool = await asyncpg.create_pool(dsn=settings.catalog_database_url, min_size=1, max_size=4)
    try:
        base_urls = {}
        if settings.response_model_base_url is not None:
            base_urls[settings.response_model_name] = str(settings.response_model_base_url)
        if settings.extraction_model_base_url is not None:
            base_urls[settings.extraction_model_name] = str(settings.extraction_model_base_url)
        gateway = OpenAICompatibleModelGateway(
            client=client,
            model_base_urls=base_urls,
            model_max_tokens={
                settings.response_model_name: settings.response_model_max_tokens,
                settings.extraction_model_name: settings.extraction_model_max_tokens,
            },
            api_key=api_key,
        )
        judgment = (
            SystemOneJudgmentGateway(
                client=client, base_url=str(settings.judgment_model_base_url), api_key=api_key
            )
            if settings.judgment_model_base_url is not None
            else None
        )
        chat = ChatUseCases(
            conversations=ConversationService(
                model_gateway=gateway,
                extraction_model=settings.extraction_model_name,
                extraction_timeout_seconds=settings.extraction_model_timeout_seconds,
                judgment_gateway=judgment,
                judgment_model=settings.judgment_model_name,
                judgment_timeout_seconds=settings.judgment_model_timeout_seconds,
                conversation_style=ConversationStyle(settings.conversation_style),
            ),
            states=ConversationStateStore(InMemorySessionStore()),
            model_gateway=gateway,
            response_model=settings.response_model_name,
            response_timeout_seconds=settings.response_model_timeout_seconds,
            summary_timeout_seconds=settings.summary_model_timeout_seconds,
        )
        embedder = OpenAICompatibleEmbedder(
            client=client,
            base_url=str(settings.embedding_base_url),
            space_id=settings.embedding_space_id,
            passage_model=settings.embedding_passage_model_name,
            query_model=settings.embedding_query_model_name,
            dimensions=settings.embedding_dimensions,
            api_key=(
                settings.upstage_api_key.get_secret_value()
                if settings.upstage_api_key is not None
                else None
            ),
            batch_size=settings.embedding_batch_size,
            timeout_seconds=settings.embedding_timeout_seconds,
        )
        recommender = V1RecommendationService(
            RecommendationEngine(embedder=embedder, catalog=PostgresCatalogRepository(pool=pool))
        )

        started = await chat.create_session(user_id=1, conversation_room_id=1)
        session_id = started.state.session_id
        history = [{"role": "assistant", "content": started.greeting}]
        print(f"\n[니쥬] {started.greeting}")
        state = started.state
        for _ in range(MAX_TURNS):
            if state.status != SessionStatus.ACTIVE or recommendation_readiness(state).sufficient:
                break
            message = await persona_reply(gateway, settings.response_model_name, history)
            print(f"[나] {message}")
            completed = await chat.send_message(session_id, message, user_id=1)
            state = completed.state
            history += [
                {"role": "user", "content": message},
                {"role": "assistant", "content": completed.reply},
            ]
            print(f"[니쥬] {completed.reply}")

        analysis = await chat.analyze_session(session_id)
        print("\n== 취향 신호")
        for signal in analysis.profile.active_signals():
            print(f"  {signal.field.value:12s} {signal.value} ({signal.confidence:.2f})")

        closable = await chat.get_close_analysis(session_id, user_id=1)
        payload = {"userId": 1, "profile": recommendation_profile(closable)}
        signals, guide = recommendation_inputs(payload)
        print("\n== 검색 쿼리")
        for query in build_search_queries(signals):
            print(f"  [{query.space.value:7s}] {query.text}")
        print(f"  제외: {list(guide.exclusions)}  가산: {list(guide.preferences)}")

        result = await recommender.recommend_lists(payload)
        names = {
            row["external_id"]: (row["name"], row["category"])
            for row in await pool.fetch("SELECT external_id, name, category FROM catalog_products")
        }
        for mode in ("self", "gift"):
            print(f"\n== 추천 ({mode})")
            for rank, item in enumerate(result[mode]["items"], start=1):
                name, category = names.get(item["externalId"], ("?", "?"))
                print(
                    f"  {rank:2d}. [{category}] {name[:45]}"
                    f"  score={item['score']}  {item['reason']}"
                )
    finally:
        await pool.close()
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
