import re

from app.domain.conversation.models import ConversationGoal

# 이 표시가 나오면 그 뒤는 지시문이나 메모가 새어 나온 것이라 잘라 낸다.
# "사용자:"는 프롬프트 예시를 흉내 내 다음 차례까지 이어 쓴 경우다.
_LEAK_MARKERS = (
    "<<<STATE>>>",
    "[STATE]",
    "[시스템 지시]",
    "시스템 프롬프트:",
    "현재 목표:",
    "[대화하는 방식]",
    "[말투]",
    "[안전]",
    "[예시]",
    "[소재]",
    "[지금]",
    "지금까지 들은 것:",
    "사용자:",
)
# 응답 앞에 붙은 화자 이름표. 예시 형식을 따라 "니쥬: ..."로 쓰는 경우가 있다.
_SPEAKER_LABEL = re.compile(r"^\s*(니쥬|AI|assistant)\s*[:：]\s*", re.IGNORECASE)
_WRAPPING_QUOTES = ('"', "“", "”", "'")

_FALLBACK_QUESTIONS = {
    ConversationGoal.OPENING: "요즘 어떻게 지내세요?",
    ConversationGoal.CHAT: "그 얘기 조금 더 들려줄래요?",
    ConversationGoal.WRAP: "이야기해 주신 내용으로 취향을 정리해 둘게요.",
}


# 질문 뒤에 남겨 두는 말의 최대 길이. 니쥬가 고른 답 한 마디 정도다.
COMPANION_TAIL_CHARS = 30


def sanitize_response(text: str, goal: ConversationGoal) -> str:
    cleaned = _SPEAKER_LABEL.sub("", text.strip())
    cut_positions = [cleaned.find(marker) for marker in _LEAK_MARKERS]
    cut_positions = [position for position in cut_positions if position >= 0]
    if cut_positions:
        cleaned = cleaned[: min(cut_positions)].rstrip()
    # 응답은 한 문단으로 보낸다. 모델이 넣은 줄바꿈은 띄어쓰기로 바꾼다.
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned).strip()
    if len(cleaned) > 1 and cleaned[0] in _WRAPPING_QUOTES and cleaned[-1] in _WRAPPING_QUOTES:
        cleaned = cleaned[1:-1].strip()

    # 질문 없이 끝나는 응답도 있고, "뭐 골라요? 저는 빔프요."처럼 질문 뒤에
    # 니쥬의 답이 짧게 붙기도 한다. 질문 뒤에 길게 늘어놓은 말만 뗀다.
    question_end = cleaned.rfind("?")
    if question_end >= 0 and len(cleaned[question_end + 1 :].strip()) > COMPANION_TAIL_CHARS:
        cleaned = cleaned[: question_end + 1].strip()
    return cleaned or _FALLBACK_QUESTIONS[goal]


CLOSING_MESSAGE = "들려주신 이야기로 취향과 관심사를 정리해 볼게요."


def closing_reply(reply: str) -> str:
    """응답 끝의 질문 문장을 빼고 종료 멘트를 붙인다. 앞의 짧은 반응은 남긴다."""
    text = reply.strip()
    if text.endswith("?"):
        cut = max(text.rfind(mark, 0, len(text) - 1) for mark in (".", "!", "~"))
        text = text[: cut + 1].strip() if cut >= 0 else ""
    return f"{text} {CLOSING_MESSAGE}".strip()
