"""취향과 관심사를 어떤 종류의 내용으로 볼지 정한 분류 기준.

프로필 값은 사용자 표현을 살린 구체적인 구("무채색 옷", "혼자 가는 캠핑")로 두고,
이 분류는 그 값이 어느 축의 취향인지, 어느 분야의 관심사인지를 판단하는 기준으로만 쓴다.
분류 결과는 프로필 항목의 taxonomy_path에 남긴다.

- 취향: 대상 안에서, 또는 대상을 가로질러 무엇을 더 좋아하는지 보여 주는 축 위의 선호
- 관심사: 사용자가 시간과 마음을 쓰는 구체적인 분야, 대상, 활동
"""

from __future__ import annotations

from dataclasses import dataclass

from app.domain.profile.models import PreferenceAspect

TASTE_ROOT = "취향"
INTEREST_ROOT = "관심사"


@dataclass(slots=True, frozen=True)
class TasteAxis:
    key: str
    group: str
    name: str
    examples: str
    # 추천에서 이 취향을 어떻게 쓸지. 상품 속성 가산, 감각, 사용 상황 쿼리, 고르는 기준.
    aspect: PreferenceAspect
    # 판단 모델(Jev)에 주는 설명. 영어가 주 학습 언어라 영어로 쓴다.
    description: str
    # 대화 모델이 이 축을 자연스럽게 물을 때 쓰는 방향. 비어 있으면 대화로 묻지 않는다.
    ask: str = ""

    @property
    def path(self) -> tuple[str, ...]:
        return (TASTE_ROOT, self.group, self.name)


@dataclass(slots=True, frozen=True)
class InterestCategory:
    key: str
    name: str
    examples: str
    description: str

    @property
    def path(self) -> tuple[str, ...]:
        return (INTEREST_ROOT, self.name)


_A = PreferenceAspect.ATTRIBUTE
_S = PreferenceAspect.SENSORY
_T = PreferenceAspect.SITUATION
_C = PreferenceAspect.CRITERION

TASTE_AXES: tuple[TasteAxis, ...] = (
    TasteAxis(
        "visual.color",
        "시각",
        "색상",
        "무채색, 파스텔, 원색, 어두운 색, 따뜻한 색",
        _A,
        "Preferred colors, such as achromatic, pastel, vivid, dark, or warm colors",
        "즐겨 쓰는 물건이나 옷의 색",
    ),
    TasteAxis(
        "visual.design",
        "시각",
        "디자인",
        "미니멀, 화려함, 심플, 장식적",
        _A,
        "Preferred design, such as minimal, flashy, simple, or ornate",
        "물건이 어떤 느낌의 디자인인지",
    ),
    TasteAxis(
        "visual.shape",
        "시각",
        "형태",
        "둥근, 각진, 슬림한, 볼륨감 있는",
        _A,
        "Preferred shapes, such as round, angular, slim, or voluminous",
    ),
    TasteAxis(
        "style.mood",
        "스타일",
        "분위기",
        "모던, 클래식, 빈티지, 레트로, 미래적",
        _A,
        "Preferred stylistic mood, such as modern, classic, vintage, retro, or futuristic",
        "좋아하는 물건이나 공간의 스타일",
    ),
    TasteAxis(
        "style.fashion",
        "스타일",
        "패션/디자인",
        "캐주얼, 스트리트, 포멀, 내추럴, 키치",
        _A,
        "Preferred fashion style, such as casual, street, formal, natural, or kitsch",
        "평소 즐겨 입는 옷 느낌",
    ),
    TasteAxis(
        "sense.taste",
        "미각",
        "맛",
        "단맛, 짠맛, 매운맛, 산미, 쓴맛, 고소함",
        _S,
        "Preferred flavors, such as sweet, salty, spicy, sour, bitter, or nutty",
        "좋아하는 맛",
    ),
    TasteAxis(
        "sense.scent",
        "후각",
        "향",
        "우디, 머스크, 플로럴, 시트러스, 비누향",
        _S,
        "Preferred scents, such as woody, musk, floral, citrus, or clean soap",
        "좋아하는 향",
    ),
    TasteAxis(
        "sense.sound",
        "청각",
        "음악적 감각",
        "잔잔함, 강한 비트, 어쿠스틱, 웅장함",
        _S,
        "Preferred sound or music feel, such as calm, strong beat, acoustic, or grand",
        "좋아하는 음악 분위기",
    ),
    TasteAxis(
        "sense.texture",
        "촉각",
        "질감",
        "부드러운, 거친, 폭신한, 매끄러운",
        _A,
        "Preferred textures, such as soft, rough, fluffy, or smooth",
    ),
    TasteAxis(
        "mood.space",
        "분위기",
        "공간",
        "조용한, 활기찬, 아늑한, 개방적인",
        _T,
        "Preferred atmosphere of a space, such as quiet, lively, cozy, or open",
        "좋아하는 공간 분위기",
    ),
    TasteAxis(
        "activity.energy",
        "활동",
        "활동성",
        "활동적인 ↔ 정적인",
        _T,
        "Preference for active versus calm, still ways of spending time",
        "쉴 때 몸을 움직이는 쪽인지 가만히 쉬는 쪽인지",
    ),
    TasteAxis(
        "experience.novelty",
        "경험",
        "새로운 경험",
        "새로운 것 선호 ↔ 익숙한 것 선호",
        _T,
        "Preference for trying new things versus sticking to familiar ones",
        "새로운 곳을 찾아가는지 가던 곳을 가는지",
    ),
    TasteAxis(
        "experience.planning",
        "경험",
        "계획성",
        "계획적인 ↔ 즉흥적인",
        _T,
        "Preference for planning ahead versus acting spontaneously",
        "미리 계획을 세우는지 즉흥적으로 움직이는지",
    ),
    TasteAxis(
        "experience.pace",
        "경험",
        "속도",
        "여유로운 ↔ 빠른",
        _T,
        "Preference for a relaxed, slow pace versus a fast, packed pace",
        "여유롭게 즐기는지 알차게 몰아서 하는지",
    ),
    TasteAxis(
        "social.size",
        "사회",
        "인원",
        "혼자 ↔ 소수 ↔ 다수",
        _T,
        "Preference for doing things alone, with a few people, or in a large group",
        "혼자 하는지 여럿이 하는지",
    ),
    TasteAxis(
        "social.relation",
        "사회",
        "관계",
        "친한 사람 중심 ↔ 새로운 사람",
        _T,
        "Preference for spending time with close people versus meeting new people",
        "주로 누구와 함께하는지",
    ),
    TasteAxis(
        "place.setting",
        "장소",
        "공간",
        "실내 ↔ 야외",
        _T,
        "Preference for indoor versus outdoor places",
        "실내에서 즐기는지 밖에서 즐기는지",
    ),
    TasteAxis(
        "place.crowd",
        "장소",
        "밀집도",
        "한적함 ↔ 붐빔",
        _T,
        "Preference for quiet, uncrowded places versus busy, crowded places",
        "한적한 곳을 찾는지 북적이는 곳이 좋은지",
    ),
    TasteAxis(
        "spending.price",
        "소비",
        "가격",
        "가성비 ↔ 프리미엄",
        _C,
        "Preference for good value for money versus premium products",
        "하나를 사도 좋은 걸 사는지 가볍게 여러 개 사는지",
    ),
    TasteAxis(
        "spending.function",
        "소비",
        "기능",
        "실용성 ↔ 심미성",
        _C,
        "Preference for practicality versus looks when choosing things",
        "물건을 쓸모로 고르는지 예뻐서 고르는지",
    ),
    TasteAxis(
        "spending.brand",
        "소비",
        "브랜드",
        "브랜드 비중 낮음 ↔ 브랜드 중시",
        _C,
        "How much the user cares about brands, from not at all to strongly",
    ),
    TasteAxis(
        "spending.trend",
        "소비",
        "유행",
        "유행 추종 ↔ 독특함",
        _C,
        "Preference for following trends versus choosing unique, uncommon things",
    ),
    TasteAxis(
        "spending.proven",
        "소비",
        "제품 성향",
        "검증된 제품 ↔ 신제품",
        _C,
        "Preference for proven, well-reviewed products versus new releases",
    ),
    TasteAxis(
        "spending.ownership",
        "소비",
        "소유",
        "오래 사용 ↔ 자주 교체",
        _C,
        "Preference for using things for a long time versus replacing them often",
        "물건을 오래 쓰는 편인지 자주 바꾸는 편인지",
    ),
    TasteAxis(
        "value.simplicity",
        "가치",
        "단순성",
        "단순함 ↔ 복잡함",
        _C,
        "Preference for simple things versus complex, feature-rich things",
    ),
    TasteAxis(
        "value.rarity",
        "가치",
        "희소성",
        "대중적 ↔ 희소함",
        _C,
        "Preference for popular, common things versus rare, limited things",
    ),
    TasteAxis(
        "value.tradition",
        "가치",
        "전통성",
        "전통 ↔ 혁신",
        _C,
        "Preference for traditional things versus innovative, new approaches",
    ),
    TasteAxis(
        "value.sustainability",
        "가치",
        "지속가능성",
        "일반 소비 ↔ 친환경 중시",
        _C,
        "How much the user values eco-friendly, sustainable choices",
    ),
    TasteAxis(
        "value.quality",
        "가치",
        "품질",
        "가격 중심 ↔ 품질 중심",
        _C,
        "Preference for choosing by price versus by quality",
    ),
    TasteAxis(
        "content.depth",
        "콘텐츠",
        "난이도",
        "가볍고 쉬움 ↔ 깊고 복잡함",
        _A,
        "Preference for light, easy content versus deep, complex content",
        "가볍게 보는 걸 좋아하는지 깊이 파고드는 걸 좋아하는지",
    ),
    TasteAxis(
        "content.emotion",
        "콘텐츠",
        "감정",
        "편안함 ↔ 자극적",
        _A,
        "Preference for comforting, relaxing content versus thrilling, intense content",
        "편하게 보는 걸 좋아하는지 자극적인 걸 좋아하는지",
    ),
    TasteAxis(
        "content.narrative",
        "콘텐츠",
        "서사",
        "현실적 ↔ 판타지적",
        _A,
        "Preference for realistic stories versus fantasy stories",
    ),
)

# 동기: 그 일에서 무엇을 얻는지, 왜 중요한지. 주신 분류표를 넓힌 부분이다.
# "왜" 질문의 답에서 뽑히고, 상품 문서와는 맞지 않아 추천 쿼리 대신 요약문에 쓴다.
_M = PreferenceAspect.MOTIVE
TASTE_AXES = (
    *TASTE_AXES,
    TasteAxis(
        "motive.recharge",
        "동기",
        "휴식·재충전",
        "머리 비우기, 쉼, 재충전",
        _M,
        "Resting and recharging, such as clearing one's head or unwinding",
    ),
    TasteAxis(
        "motive.immersion",
        "동기",
        "몰입",
        "집중, 푹 빠져듦, 시간 가는 줄 모름",
        _M,
        "Deep focus and immersion, losing track of time",
    ),
    TasteAxis(
        "motive.achievement",
        "동기",
        "성취·성장",
        "해냈다는 느낌, 실력이 느는 것",
        _M,
        "A sense of achievement or getting better at something",
    ),
    TasteAxis(
        "motive.connection",
        "동기",
        "관계·교감",
        "함께 나누는 시간, 사람과 가까워지는 것",
        _M,
        "Sharing time with others and feeling close to people",
    ),
    TasteAxis(
        "motive.expression",
        "동기",
        "자기표현",
        "나다움, 취향 드러내기, 만드는 즐거움",
        _M,
        "Expressing oneself, showing one's taste, or the joy of making things",
    ),
    TasteAxis(
        "motive.freedom",
        "동기",
        "자유",
        "얽매이지 않음, 내 마음대로 하는 시간",
        _M,
        "Freedom and autonomy, doing things on one's own terms",
    ),
)

INTEREST_CATEGORIES: tuple[InterestCategory, ...] = (
    InterestCategory(
        "food",
        "음식·미식",
        "커피, 차, 디저트, 베이킹, 요리, 맛집, 와인, 전통음식",
        "Food and drink, such as coffee, tea, desserts, baking, cooking, restaurants, wine",
    ),
    InterestCategory(
        "fashion",
        "패션",
        "의류, 신발, 가방, 액세서리, 스트리트웨어, 빈티지, 명품",
        "Fashion, such as clothes, shoes, bags, accessories, streetwear, vintage, luxury",
    ),
    InterestCategory(
        "beauty",
        "뷰티",
        "향수, 스킨케어, 메이크업, 헤어, 네일",
        "Beauty, such as perfume, skincare, makeup, hair, nails",
    ),
    InterestCategory(
        "fitness",
        "운동·피트니스",
        "러닝, 헬스, 요가, 필라테스, 수영, 자전거, 클라이밍",
        "Exercise and fitness, such as running, gym, yoga, pilates, swimming, cycling, climbing",
    ),
    InterestCategory(
        "outdoor",
        "아웃도어",
        "캠핑, 등산, 낚시, 서핑, 스키, 스노보드",
        "Outdoor activities, such as camping, hiking, fishing, surfing, skiing, snowboarding",
    ),
    InterestCategory(
        "travel",
        "여행",
        "국내여행, 해외여행, 호캉스, 배낭여행, 드라이브",
        "Travel, such as domestic or overseas trips, hotel stays, backpacking, road trips",
    ),
    InterestCategory(
        "music",
        "음악",
        "K-pop, 힙합, R&B, 락, 재즈, 클래식, EDM, 악기",
        "Music, such as K-pop, hip-hop, R&B, rock, jazz, classical, EDM, instruments",
    ),
    InterestCategory(
        "video",
        "영화·영상",
        "영화, 드라마, 애니메이션, 다큐멘터리, 유튜브",
        "Movies and video, such as films, dramas, animation, documentaries, YouTube",
    ),
    InterestCategory(
        "games",
        "게임",
        "PC게임, 콘솔게임, 모바일게임, 보드게임, e스포츠",
        "Games, such as PC, console, mobile, board games, esports",
    ),
    InterestCategory(
        "reading",
        "독서·글쓰기",
        "소설, 에세이, 인문학, 자기계발, 웹소설, 글쓰기",
        "Reading and writing, such as novels, essays, humanities, self-help, web novels",
    ),
    InterestCategory(
        "art",
        "미술·예술",
        "전시, 그림, 일러스트, 디자인, 공예, 사진",
        "Art, such as exhibitions, painting, illustration, design, crafts, photography",
    ),
    InterestCategory(
        "performance",
        "공연",
        "뮤지컬, 연극, 콘서트, 페스티벌",
        "Live performances, such as musicals, plays, concerts, festivals",
    ),
    InterestCategory(
        "tech",
        "기술",
        "AI, 프로그래밍, 로봇, 스마트폰, PC, 전자기기",
        "Technology, such as AI, programming, robots, smartphones, PCs, gadgets",
    ),
    InterestCategory(
        "vehicles",
        "자동차",
        "자동차, 바이크, 튜닝, 드라이브, 모터스포츠",
        "Cars and bikes, such as cars, motorcycles, tuning, driving, motorsports",
    ),
    InterestCategory(
        "learning",
        "학습",
        "외국어, 역사, 과학, 경제, 자격증, 온라인 강의",
        "Learning, such as languages, history, science, certificates, online courses",
    ),
    InterestCategory(
        "finance",
        "금융",
        "주식, 코인, 부동산, 재테크, 경제",
        "Finance, such as stocks, crypto, real estate, personal investing",
    ),
    InterestCategory(
        "living",
        "라이프스타일",
        "인테리어, 가구, 식물, 홈카페, 정리, 생활용품",
        "Home and living, such as interior design, furniture, plants, home cafe, organizing",
    ),
    InterestCategory(
        "pets",
        "반려동물",
        "강아지, 고양이, 물고기, 파충류, 반려동물 용품",
        "Pets, such as dogs, cats, fish, reptiles, pet supplies",
    ),
    InterestCategory(
        "creation",
        "창작",
        "사진촬영, 영상제작, 음악제작, 그림, 글쓰기, DIY",
        "Making things, such as photography, video making, music production, drawing, DIY",
    ),
    InterestCategory(
        "collecting",
        "수집",
        "LP, 피규어, 신발, 카드, 굿즈, 시계",
        "Collecting, such as vinyl records, figures, sneakers, cards, merchandise, watches",
    ),
    InterestCategory(
        "social",
        "사회활동",
        "동호회, 모임, 네트워킹, 봉사활동",
        "Social activities, such as clubs, meetups, networking, volunteering",
    ),
    InterestCategory(
        "selfcare",
        "자기관리",
        "명상, 습관관리, 생산성, 다이어리, 자기계발",
        "Self-management, such as meditation, habit tracking, productivity, journaling",
    ),
)

TASTE_AXES_BY_KEY = {axis.key: axis for axis in TASTE_AXES}
# 대화로 물을 때의 순서. 어떤 활동 이야기에서든 이어 묻기 쉬운 축(누구와, 어디서, 어떤 분위기)을
# 앞에 두고, 물건이나 소비 이야기가 나와야 자연스러운 축은 뒤에 둔다.
ASK_PRIORITY = (
    "social.size",
    "place.crowd",
    "mood.space",
    "place.setting",
    "experience.planning",
    "experience.novelty",
    "experience.pace",
    "activity.energy",
    "social.relation",
    "content.emotion",
    "content.depth",
    "sense.taste",
    "sense.sound",
    "sense.scent",
    "style.mood",
    "visual.design",
    "visual.color",
    "style.fashion",
    "spending.ownership",
    "spending.price",
    "spending.function",
)
INTEREST_CATEGORIES_BY_KEY = {category.key: category for category in INTEREST_CATEGORIES}
_AXES_BY_PATH = {axis.path: axis for axis in TASTE_AXES}


def taste_axis_for_path(path: tuple[str, ...] | None) -> TasteAxis | None:
    return _AXES_BY_PATH.get(path) if path else None


def taste_axes_guide() -> str:
    """프롬프트에 넣는 취향 축 표. 값의 예시이지 라벨 목록이 아니다."""
    return "\n".join(f"- {axis.group}/{axis.name}: {axis.examples}" for axis in TASTE_AXES)


def interest_categories_guide() -> str:
    return "\n".join(f"- {category.name}: {category.examples}" for category in INTEREST_CATEGORIES)
