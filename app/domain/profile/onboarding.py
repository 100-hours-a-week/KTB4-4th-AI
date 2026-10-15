"""대화방을 만들 때 BE가 함께 보내는 온보딩 값.

코드 값은 BE enum 이름과 같다. 관심 카테고리는 추천 쿼리의 재료가 되고, 알레르기와 선물 제외
조건은 추천 후보를 거르는 데 쓴다. 대화에서 나온 취향과 달리 사용자가 직접 고른 값이라
대화 내용으로 덮어쓰지 않는다. nickname 은 카카오톡 이름으로 만든 username 이고, 프롬프트에
쓸 이름으로 바꾸는 규칙은 app.domain.profile.naming 에 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any

from app.domain.profile import naming


class InterestCategory(StrEnum):
    FASHION = "FASHION"
    BEAUTY = "BEAUTY"
    HOME_INTERIOR = "HOME_INTERIOR"
    COFFEE_TEA = "COFFEE_TEA"
    COOKING = "COOKING"
    GOURMET = "GOURMET"
    TRAVEL = "TRAVEL"
    FITNESS = "FITNESS"
    OUTDOOR = "OUTDOOR"
    GAME = "GAME"
    MUSIC = "MUSIC"
    MOVIE_DRAMA = "MOVIE_DRAMA"
    READING = "READING"
    TECH = "TECH"
    PET = "PET"


class Allergy(StrEnum):
    PEANUT = "PEANUT"
    NUTS = "NUTS"
    MILK = "MILK"
    EGG = "EGG"
    WHEAT = "WHEAT"
    SOY = "SOY"
    SHELLFISH = "SHELLFISH"
    FISH = "FISH"


class GiftExclusion(StrEnum):
    PERFUME = "PERFUME"
    ALCOHOL = "ALCOHOL"
    CLOTHING = "CLOTHING"
    FOOD = "FOOD"
    COSMETICS = "COSMETICS"
    VOUCHER = "VOUCHER"


class Gender(StrEnum):
    MALE = "MALE"
    FEMALE = "FEMALE"


# 화면 칩 이름(BE 라벨)이 아니라 검색 쿼리에 넣을 말이다. "커피·차 관련 제품"보다
# "커피와 차 관련 제품"이 상품 문서 문장과 가깝다.
INTEREST_QUERY_TERMS: dict[InterestCategory, str] = {
    InterestCategory.FASHION: "패션",
    InterestCategory.BEAUTY: "뷰티",
    InterestCategory.HOME_INTERIOR: "홈 인테리어",
    InterestCategory.COFFEE_TEA: "커피와 차",
    InterestCategory.COOKING: "요리와 베이킹",
    InterestCategory.GOURMET: "맛집 탐방",
    InterestCategory.TRAVEL: "여행",
    InterestCategory.FITNESS: "운동과 헬스",
    InterestCategory.OUTDOOR: "캠핑과 아웃도어",
    InterestCategory.GAME: "게임",
    InterestCategory.MUSIC: "음악",
    InterestCategory.MOVIE_DRAMA: "영화와 드라마",
    InterestCategory.READING: "독서",
    InterestCategory.TECH: "IT 전자기기",
    InterestCategory.PET: "반려동물",
}


# 대화 관심사의 분야(app.domain.profile.taxonomy.INTEREST_CATEGORIES 키). 추천 쿼리를 고를 때
# 같은 분야의 대화 관심사와 한 묶음이 되어 한 분야가 쿼리를 몰아 차지하지 않는다.
INTEREST_TAXONOMY_KEYS: dict[InterestCategory, str] = {
    InterestCategory.FASHION: "fashion",
    InterestCategory.BEAUTY: "beauty",
    InterestCategory.HOME_INTERIOR: "living",
    InterestCategory.COFFEE_TEA: "food",
    InterestCategory.COOKING: "food",
    InterestCategory.GOURMET: "food",
    InterestCategory.TRAVEL: "travel",
    InterestCategory.FITNESS: "fitness",
    InterestCategory.OUTDOOR: "outdoor",
    InterestCategory.GAME: "games",
    InterestCategory.MUSIC: "music",
    InterestCategory.MOVIE_DRAMA: "video",
    InterestCategory.READING: "reading",
    InterestCategory.TECH: "tech",
    InterestCategory.PET: "pets",
}


@dataclass(slots=True, frozen=True)
class OnboardingProfile:
    nickname: str | None = None
    birth_date: date | None = None
    gender: Gender | None = None
    interests: tuple[InterestCategory, ...] = ()
    allergies: tuple[Allergy, ...] = ()
    gift_exclusions: tuple[GiftExclusion, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "nickname": self.nickname,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "gender": self.gender.value if self.gender else None,
            "interests": [code.value for code in self.interests],
            "allergies": [code.value for code in self.allergies],
            "gift_exclusions": [code.value for code in self.gift_exclusions],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OnboardingProfile:
        birth_date = payload.get("birth_date")
        gender = payload.get("gender")
        return cls(
            nickname=payload.get("nickname"),
            birth_date=date.fromisoformat(birth_date) if birth_date else None,
            gender=Gender(gender) if gender else None,
            interests=tuple(InterestCategory(code) for code in payload.get("interests", [])),
            allergies=tuple(Allergy(code) for code in payload.get("allergies", [])),
            gift_exclusions=tuple(
                GiftExclusion(code) for code in payload.get("gift_exclusions", [])
            ),
        )

    @property
    def display_name(self) -> str | None:
        return naming.display_name(self.nickname)

    @property
    def address_name(self) -> str | None:
        return naming.address_name(self.nickname)

    def recommendation_payload(self) -> dict[str, Any]:
        """추천 서비스에 넘기는 부분. 생년월일과 성별은 아직 추천에 쓰지 않는다."""
        return {
            "displayName": self.display_name,
            "interestCategoryCodes": [code.value for code in self.interests],
            "allergyCodes": [code.value for code in self.allergies],
            "giftExclusionCodes": [code.value for code in self.gift_exclusions],
        }
