"""카카오톡 이름으로 만든 username 을 프롬프트에 쓸 이름으로 바꾼다.

요약과 선물 추천 이유에는 원문을 쓰고(display_name), 대화에서는 성을 뗀 이름으로
부른다(address_name).
"""

from __future__ import annotations

import re
import unicodedata

# 카카오톡 이름 길이 상한. 이보다 길면 이름이 아닌 문장으로 보고 쓰지 않는다.
MAX_NAME_LENGTH = 20
_COMPOUND_SURNAMES = frozenset(
    {"남궁", "제갈", "선우", "황보", "독고", "사공", "서문", "동방", "어금", "망절"}
)
_SURNAMES = frozenset(
    "김이박최정강조윤장임한오서신권황안송류유전홍고문양손배백허남심노하곽성차주우구민진나지엄"
    "채원천방공현함변염여추도소석선설마길연위표명기반왕금옥육인맹제모탁국은편용예경봉사부황"
    "가복태목형피두감음빈동온호범좌갈"
)
_BRACKETED = re.compile(r"\([^)]*\)|\[[^\]]*\]|\{[^}]*\}|<[^>]*>")
_HANGUL_NAME = re.compile(r"[가-힣]{2,4}")


def display_name(nickname: str | None) -> str | None:
    """요약과 추천 이유에 쓰는 이름. 원문에서 이모지, 기호, 제어 문자만 뺀다.

    "."처럼 글자가 없거나 지나치게 길면 None 이다. 프롬프트에 그대로 들어가므로 따옴표도 뺀다.
    """
    if not nickname:
        return None
    # 줄바꿈 같은 공백은 단어가 붙지 않게 띄어쓰기로 바꾼 뒤 뺀다.
    kept = "".join(
        " " if char.isspace() else char
        for char in unicodedata.normalize("NFC", nickname)
        if char.isspace()
        or (unicodedata.category(char)[0] not in {"C", "S"} and char not in {'"', "'", "`"})
    )
    name = " ".join(kept.split())
    if len(name) > MAX_NAME_LENGTH or not any(
        unicodedata.category(char).startswith("L") for char in name
    ):
        return None
    return name


def address_name(nickname: str | None) -> str | None:
    """대화에서 "동휘님"처럼 부를 이름. 뒤에 "님"을 붙여 쓴다.

    괄호 안 내용, 이모지, 기호를 뺀 뒤 한글 2~4글자면 이름으로 보고 성을 뗀다.
    - 2글자: 그대로 쓴다. "서진"이 이름인지 성+이름인지 구분할 수 없고 어느 쪽이든 어색하지 않다.
    - 3글자: 첫 글자가 성씨면 뗀다. "서동휘" -> "동휘"
    - 4글자: 앞 두 글자가 두 글자 성이면 뗀다. "남궁민수" -> "민수"
    그 밖의 이름(별명, 영문)은 display_name 원문을 그대로 쓴다.
    """
    name = display_name(nickname)
    if name is None:
        return None
    compact = "".join(
        char for char in _BRACKETED.sub("", name) if unicodedata.category(char)[0] in {"L", "N"}
    )
    if not _HANGUL_NAME.fullmatch(compact):
        return name
    if len(compact) == 3 and compact[0] in _SURNAMES:
        return compact[1:]
    if len(compact) == 4 and compact[:2] in _COMPOUND_SURNAMES:
        return compact[2:]
    return compact
