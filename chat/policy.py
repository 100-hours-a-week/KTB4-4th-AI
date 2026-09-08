"""이번 턴에 무엇을 물을지 서버가 정한다.

모델에게 '알아서 판단해라'를 시키지 않는다. 여기서 정하고 지시만 내려보낸다.
상태(JSON)를 보고 개수를 세는 일이라 결정적이고, 테스트할 수 있다.

같은 목표가 계속 나오면 같은 질문이 반복된다. 그래서 두 가지를 둔다.
  ① 지시문을 시도 횟수마다 바꾼다 (같은 목표라도 각도를 바꿔 묻는다)
  ② 세 번 시도했는데 아무것도 못 건지면 그 목표를 접고 다음으로 넘어간다

선물 관련 질문("사고 싶은데 못 산 거 있어요?")은 목표에 두지 않는다.
사용자가 선물 받으려는 대화로 느끼는 순간 답이 꾸며지기 때문이다.
wants·unaffordable 은 대화 중에 저절로 나오면 줍기만 한다.
"""

from __future__ import annotations

from dataclasses import dataclass

from schema import State

MAX_TURNS = 20
EARLY_CLOSE_TURN = 12  # 조건을 다 채워도 이 턴 전에는 끝내지 않는다
GIVE_UP_AFTER = 3      # 무성과 시도가 이만큼이면 그 목표를 접는다
DROPOFF_LEN = 12       # 최근 3턴 발화가 계속 줄고 마지막이 이 아래면 그만 묻는다

_WRAP = ("새 질문을 하지 말고 대화를 자연스럽게 마무리해라. "
         "짧게 인사하고 끝낸다.")

# 목표마다 지시문을 여러 개 둔다. 같은 목표를 반복할 때 순서대로 쓴다.
DIRECTIVES: dict[str, list[str]] = {
    "OPENING": [
        "첫 인사를 건네고, 요즘 하루를 어떻게 보내는지 가볍게 물어라. "
        "취향이나 취미를 직접 묻지 않는다.",
    ],
    "INTEREST": [
        "무엇에 시간을 쓰는지 물어라. '취미가 뭐예요' 처럼 대놓고 묻지 말고 "
        "방금 나온 이야기에 이어서 물어라.",
        "최근에 재미있었거나 기억에 남는 일이 있었는지 물어라. 취미라는 말을 쓰지 않는다.",
        "쉬는 날에는 주로 뭘 하면서 보내는지 물어라.",
    ],
    "INTEREST_VIA_ROUTINE": [
        "사용자가 앞서 말한 일과나 상황 중 하나를 골라 그 안으로 파고들어라. "
        "그 시간에 무엇을 하는지 구체적으로 물어라. 취향을 직접 묻지 않는다.",
        "앞서 말한 일과 중 하나를 골라, 그때 **무엇을 쓰는지**(물건·기기·도구) 물어라.",
        "하루 중 그나마 마음 편한 시간이 언제인지, 그때 뭘 하는지 물어라.",
    ],
    "DISLIKE": [
        "피하는 것이나 잘 안 맞는 것이 있는지 물어라.",
        "몸에 안 받거나 못 쓰는 것이 있는지 물어라. 알레르기 같은 것도 포함된다.",
        "받으면 곤란한 물건이 있는지 물어라.",
    ],
    # 선물 얘기를 대놓고 묻지 않는다. 물어서 얻은 답은 이미 꾸며져 있다.
    # 대신 "지금 쓰는 물건"을 물으면 같은 재료(보유·선호·아쉬운 점)가 자연스럽게 나온다.
    "GEAR": [
        "앞서 나온 활동에 대해, 지금 어떤 물건을 쓰고 계신지 물어라.",
        "지금 쓰는 것에 아쉬운 점이나 불편한 점이 있는지 물어라.",
        "그걸 할 때 뭐가 하나 더 있으면 좋겠다 싶은 게 있는지 물어라.",
    ],
    "DEEPEN": [
        "이미 나온 이야기 중 하나를 골라 한 단계 더 구체적으로 물어라. "
        "얼마나 자주 하는지, 어떤 걸 쓰는지 같은 것.",
        "이미 나온 것 중 하나를 골라, 지금 쓰는 물건에 아쉬운 점이 있는지 물어라.",
        "이미 나온 것 중 하나를 골라, 처음 시작하게 된 계기를 물어라.",
    ],
    "WRAP": [_WRAP],
}


@dataclass
class Goal:
    name: str
    directive: str
    reason: str


def _make(s: State, name: str, reason: str) -> Goal:
    d = DIRECTIVES[name]
    n = s.tries(name)
    extra = ""
    if n and name not in ("WRAP", "OPENING"):
        extra = " 앞에서 이미 비슷한 질문을 했으니 같은 말을 반복하지 마라."
    return Goal(name, d[min(n, len(d) - 1)] + extra, reason)


def _dropping_off(s: State) -> bool:
    L = s.user_lengths
    if len(L) < 3:
        return False
    a, b, c = L[-3:]
    return a > b > c and c < DROPOFF_LEN


def _dead(s: State, name: str) -> bool:
    """세 번 물었는데 하나도 못 건졌으면 접는다."""
    return s.tries(name) >= GIVE_UP_AFTER and s.wins(name) == 0


def decide(s: State) -> Goal:
    q = len(s.query_items)
    strong = len(s.strong_query_items)
    filt = len(s.filter_items)

    if s.turn == 0:
        return _make(s, "OPENING", "첫 턴")

    # ── 끝낼 때인가
    if s.turn >= MAX_TURNS:
        return _make(s, "WRAP", f"턴 상한 {MAX_TURNS} 도달")
    if _dropping_off(s):
        return _make(s, "WRAP", f"이탈 신호 (발화 길이 {s.user_lengths[-3:]})")
    if s.can_close and s.turn >= EARLY_CLOSE_TURN:
        return _make(s, "WRAP", f"조건 충족 (검색용 {q}, 확신 {strong}, 필터 {filt})")

    # ── 채워야 할 것을 급한 순서로 세운다
    todo: list[tuple[str, str]] = []

    if q < 3:
        # 직접 묻기가 두 번 실패했으면 일과에서 캔다. 한 번 넘어가면 안 돌아온다.
        if s.tries("INTEREST_VIA_ROUTINE") > 0 or (s.tries("INTEREST") >= 2 and q == 0):
            todo.append(("INTEREST_VIA_ROUTINE",
                         f"검색용 {q}/3 · 직접 질문 {s.tries('INTEREST')}회 무성과 → 일과에서 캔다"))
        else:
            todo.append(("INTEREST", f"검색용 {q}/3 (직접 질문 {s.tries('INTEREST')}회)"))

    if filt == 0 and s.turn >= 3:
        todo.append(("DISLIKE", "싫어하는 것·제약 0개"))
    if q >= 1 and len(s.by_field("owned", "preferences")) < 2:
        todo.append(("GEAR", "지금 쓰는 물건 미확인 (보유·선호 재료)"))
    if strong < 2:
        todo.append(("DEEPEN", f"확신 0.6 이상 {strong}/2"))
    todo.append(("DEEPEN", "깊이 보강"))

    # ── 죽은 목표는 건너뛴다. 이게 없으면 같은 질문에 갇힌다.
    skipped = []
    for name, reason in todo:
        if _dead(s, name):
            skipped.append(name)
            continue
        if skipped:
            reason += f" · {'·'.join(skipped)} 접음"
        return _make(s, name, reason)

    return _make(s, "WRAP", f"물어볼 게 남지 않음 ({'·'.join(skipped)} 전부 무성과)")
