from app.domain.conversation.models import ConversationGoal

_LEAK_MARKERS = (
    "<<<STATE>>>",
    "[STATE]",
    "[시스템 지시]",
    "시스템 프롬프트:",
    "현재 목표:",
)

_FALLBACK_QUESTIONS = {
    ConversationGoal.OPENING: "요즘 어떻게 지내세요?",
    ConversationGoal.INTEREST: "최근에는 뭐 하면서 시간을 보내셨어요?",
    ConversationGoal.INTEREST_VIA_ROUTINE: "어제 저녁에는 뭐 하셨어요?",
    ConversationGoal.DISLIKE: "요즘 하면서 좀 별로였던 일도 있었어요?",
    ConversationGoal.GEAR: "그거 할 때 늘 챙기는 게 있으세요?",
    ConversationGoal.DEEPEN: "그중에 제일 기억에 남는 순간은 언제였어요?",
    ConversationGoal.CORRECT: "말씀해 주신 내용으로 취향 분석을 바로잡아 둘게요.",
    ConversationGoal.WRAP: "이야기해 주신 내용으로 취향을 정리해 둘게요.",
}


def sanitize_response(text: str, goal: ConversationGoal) -> str:
    cleaned = text.strip()
    cut_positions = [cleaned.find(marker) for marker in _LEAK_MARKERS]
    cut_positions = [position for position in cut_positions if position >= 0]
    if cut_positions:
        cleaned = cleaned[: min(cut_positions)].rstrip()

    first_question = cleaned.find("?")
    if first_question >= 0:
        cleaned = cleaned[: first_question + 1].strip()
    return cleaned or _FALLBACK_QUESTIONS[goal]


CLOSING_MESSAGE = "들려주신 이야기로 당신의 취향과 관심사를 찾아볼게요."


def closing_reply(reply: str) -> str:
    """응답 끝의 질문 문장을 빼고 종료 멘트를 붙인다. 앞의 짧은 반응은 남긴다."""
    text = reply.strip()
    if text.endswith("?"):
        cut = max(text.rfind(mark, 0, len(text) - 1) for mark in (".", "!", "~"))
        text = text[: cut + 1].strip() if cut >= 0 else ""
    return f"{text} {CLOSING_MESSAGE}".strip()
