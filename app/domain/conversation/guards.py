from app.domain.conversation.models import ConversationGoal

_LEAK_MARKERS = (
    "<<<STATE>>>",
    "[STATE]",
    "[시스템 지시]",
    "시스템 프롬프트:",
    "현재 목표:",
)

_FALLBACK_QUESTIONS = {
    ConversationGoal.OPENING: "요즘 즐겨 하거나 관심이 가는 일이 있으세요?",
    ConversationGoal.INTEREST: "요즘 시간 가는 줄 모르고 즐기는 일이 있으세요?",
    ConversationGoal.INTEREST_VIA_ROUTINE: "평소 쉬는 날에는 주로 어떻게 보내세요?",
    ConversationGoal.DISLIKE: "물건을 고를 때 피하고 싶은 종류나 조건이 있으세요?",
    ConversationGoal.GEAR: "자주 쓰는 물건에서 특히 중요하게 보는 점이 있으세요?",
    ConversationGoal.DEEPEN: "그중에서 특히 마음에 드는 이유는 무엇인가요?",
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
