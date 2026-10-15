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
    # 마무리 턴은 뒤에 종료 멘트가 붙으므로 정리 안내를 겹쳐 쓰지 않는다.
    ConversationGoal.WRAP: "이야기 들려주셔서 고마워요.",
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


_SENTENCE_END_MARKS = (".", "!", "~")
_LABEL_NAMES = ("니쥬", "ai", "assistant")


def _may_become_label(text: str) -> bool:
    """응답 앞부분이 아직 화자 이름표("니쥬:")로 이어질 수 있는지 본다."""
    lowered = text.lower()
    name = lowered.rstrip()
    if name in _LABEL_NAMES:
        return True
    return lowered == name and any(label.startswith(name) for label in _LABEL_NAMES)


def _partial_marker_length(text: str) -> int:
    """끝에 걸린 유출 표시의 앞부분 길이. 다음 글자가 와야 표시인지 알 수 있다."""
    longest = max(len(marker) for marker in _LEAK_MARKERS) - 1
    for size in range(min(longest, len(text)), 0, -1):
        tail = text[-size:]
        if any(marker.startswith(tail) for marker in _LEAK_MARKERS):
            return size
    return 0


def stable_reply_prefix(partial: str, *, hold_question: bool) -> str:
    """생성 중인 응답에서 정리 규칙이 나중에 바꾸지 않는 앞부분을 돌려준다.

    스트리밍으로 보낸 글자는 되돌릴 수 없으므로 sanitize_response와 closing_reply가
    확정 응답에서도 그대로 둘 부분만 내보낸다. 결과는 항상 확정 응답의 앞부분이어야 하므로
    두 규칙을 바꾸면 이 함수도 함께 확인한다.

    일반 턴은 생성된 글자를 바로 내보내고 아래만 잠깐 보류한다.
    - 아직 이름표("니쥬:")가 될 수 있는 응답 맨 앞
    - 끝에 걸린 유출 표시의 앞부분("사용자:"의 "사용")
    - 끝의 공백과 줄바꿈. 지워지거나 띄어쓰기 하나로 합쳐질 수 있다.
    - 마지막 `?` 뒤의 말. 길면 잘린다.
    - 따옴표로 시작한 응답 전체. 끝 따옴표를 봐야 앞 따옴표를 지울지 안다.

    hold_question은 마무리 턴처럼 끝 질문 문장을 종료 멘트로 바꿀 수 있는 턴이다. 그때는
    마지막 `?` 앞에서 끝난 문장까지만 낸다.
    """
    text = partial.lstrip()
    if not hold_question and _may_become_label(text):
        return ""
    cleaned = _SPEAKER_LABEL.sub("", text)
    cut_positions = [cleaned.find(marker) for marker in _LEAK_MARKERS]
    cut_positions = [position for position in cut_positions if position >= 0]
    if cut_positions:
        cleaned = cleaned[: min(cut_positions)]
    elif not hold_question:
        cleaned = cleaned[: len(cleaned) - _partial_marker_length(cleaned)]
    cleaned = re.sub(r"\s*\n+\s*", " ", cleaned).lstrip()
    if cleaned[:1] in _WRAPPING_QUOTES:
        return ""
    question_end = cleaned.rfind("?")
    if not hold_question:
        return cleaned[: question_end + 1] if question_end >= 0 else cleaned.rstrip()
    search_end = question_end if question_end >= 0 else len(cleaned)
    bound = max(cleaned.rfind(mark, 0, search_end) for mark in _SENTENCE_END_MARKS) + 1
    return cleaned[:bound]


CLOSING_MESSAGE = "들려주신 이야기로 취향과 관심사를 정리해 볼게요."


def closing_reply(reply: str) -> str:
    """응답 끝의 질문 문장을 빼고 종료 멘트를 붙인다. 앞의 짧은 반응은 남긴다."""
    text = reply.strip()
    if text.endswith("?"):
        cut = max(text.rfind(mark, 0, len(text) - 1) for mark in (".", "!", "~"))
        text = text[: cut + 1].strip() if cut >= 0 else ""
    return f"{text} {CLOSING_MESSAGE}".strip()
