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
    ConversationGoal.TASTE: "그건 주로 혼자 하세요, 누구랑 같이 하세요?",
    ConversationGoal.DISLIKE: "요즘 하면서 좀 별로였던 일도 있었어요?",
    ConversationGoal.GEAR: "그거 할 때 늘 챙기는 게 있으세요?",
    ConversationGoal.DEEPEN: "그중에 제일 기억에 남는 순간은 언제였어요?",
    ConversationGoal.BRIDGE: "그런 시간은 평소에 또 어디서 만들어요?",
    ConversationGoal.REFLECT: "얘기 들어 보니 좋아하는 게 조금 보이는데, 제가 이해한 게 맞아요?",
    ConversationGoal.CHAT: "그 얘기 조금 더 들려줄래요?",
    ConversationGoal.WRAP: "이야기해 주신 내용으로 취향을 정리해 둘게요.",
}


# reflective 대화의 응답 길이 상한. 질문 40자에 짧은 반응 하나가 들어가는 정도다.
# 넘으면 앞의 반응 문장을 떼고 질문만 남긴다.
CONCISE_REPLY_CHARS = 60
# companion 응답에서 질문 뒤에 남겨 두는 말의 최대 길이. 니쥬가 고른 답 한 마디 정도다.
COMPANION_TAIL_CHARS = 30


def sanitize_response(
    text: str,
    goal: ConversationGoal,
    *,
    concise: bool = False,
    companion: bool = False,
) -> str:
    cleaned = text.strip()
    cut_positions = [cleaned.find(marker) for marker in _LEAK_MARKERS]
    cut_positions = [position for position in cut_positions if position >= 0]
    if cut_positions:
        cleaned = cleaned[: min(cut_positions)].rstrip()

    if companion:
        # companion은 질문 없이 끝나는 응답도 있고, "뭐 골라요? 저는 빔프요."처럼 질문 뒤에
        # 니쥬의 답이 짧게 붙기도 한다. 질문 뒤에 길게 늘어놓은 말만 뗀다.
        question_end = cleaned.rfind("?")
        if question_end >= 0 and len(cleaned[question_end + 1 :].strip()) > COMPANION_TAIL_CHARS:
            cleaned = cleaned[: question_end + 1].strip()
    else:
        # 질문 뒤에 붙은 말은 뗀다.
        question_end = cleaned.find("?")
        if question_end >= 0:
            cleaned = cleaned[: question_end + 1].strip()
    if concise and len(cleaned) > CONCISE_REPLY_CHARS and cleaned.endswith("?"):
        cut = max(cleaned.rfind(mark, 0, len(cleaned) - 1) for mark in (".", "!", "~"))
        if cut >= 0 and cleaned[cut + 1 :].strip():
            cleaned = cleaned[cut + 1 :].strip()
    return cleaned or _FALLBACK_QUESTIONS[goal]


CLOSING_MESSAGE = "들려주신 이야기로 당신의 취향과 관심사를 찾아볼게요."


def closing_reply(reply: str) -> str:
    """응답 끝의 질문 문장을 빼고 종료 멘트를 붙인다. 앞의 짧은 반응은 남긴다."""
    text = reply.strip()
    if text.endswith("?"):
        cut = max(text.rfind(mark, 0, len(text) - 1) for mark in (".", "!", "~"))
        text = text[: cut + 1].strip() if cut >= 0 else ""
    return f"{text} {CLOSING_MESSAGE}".strip()
