from typing import cast

from fastapi import Request

from app.application.chat_use_cases import ChatUseCases
from app.application.conversation_state_store import ConversationStateStore


def get_chat_use_cases(request: Request) -> ChatUseCases:
    return cast(ChatUseCases, request.app.state.chat_use_cases)


def get_conversation_state_store(request: Request) -> ConversationStateStore:
    return cast(ConversationStateStore, request.app.state.conversation_state_store)
