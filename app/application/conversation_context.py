from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.domain.conversation.context import ConversationContext, ConversationDirection
from app.domain.profile.merger import normalize_text


class ConversationContextPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus: str = Field(min_length=1, max_length=60)
    direction: ConversationDirection
    evidence: str = Field(min_length=1, max_length=160)


def parse_conversation_context(
    raw: object, utterance: str, source_turn: int
) -> ConversationContext | None:
    try:
        payload = ConversationContextPayload.model_validate(raw)
    except ValidationError:
        return None
    if (
        not payload.focus.strip()
        or not payload.evidence.strip()
        or normalize_text(payload.evidence) not in normalize_text(utterance)
    ):
        return None
    return ConversationContext(
        focus=payload.focus.strip(),
        direction=payload.direction,
        evidence=payload.evidence.strip(),
        source_turn=source_turn,
    )
