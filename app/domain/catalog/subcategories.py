"""상품 세부 카테고리. AI DB에만 있고 원본 DB와 화면에는 없다.

대분류(11개) 안에서 상품이 무엇인지를 한 단계 더 나눈다. 판단 모델(Jev)이 대분류별 선택지 중
하나를 고르고, 문서 생성은 이 값을 상품 종류의 기준으로 쓴다. 추천에서는 같은 종류가 몰리지 않게
섞고, 싫어하는 종류를 빼고, 관심사 태그를 붙이는 기준이 된다.

패션과 명품·패션은 같은 종류를 다루므로 선택지를 같이 쓴다. 둘의 차이는 대분류가 이미 나눠 준다.
"""

from __future__ import annotations

from types import MappingProxyType

from app.domain.catalog.labels import ProductCategory

# 판단 모델의 확신이 이보다 낮은 세부 카테고리는 저장만 하고 문서 생성 입력으로 넘기지 않는다.
# 틀린 종류가 문서에 박히는 것보다 문서 생성 모델이 상품명만 보고 쓰는 편이 낫다.
SUBCATEGORY_MIN_CONFIDENCE = 0.6

_FASHION = {
    "jewelry": "주얼리",
    "bag": "가방",
    "wallet": "지갑",
    "hat": "모자",
    "accessories": "패션 잡화",
    "apparel": "의류·신발",
    "loungewear": "속옷·홈웨어",
}

SUBCATEGORIES: MappingProxyType[ProductCategory, MappingProxyType[str, str]] = MappingProxyType(
    {
        category: MappingProxyType(labels)
        for category, labels in {
            ProductCategory.VOUCHER: {
                "cafe_drink": "카페 음료",
                "dessert": "디저트",
                "meal": "식사",
                "movie": "영화 관람",
                "convenience": "편의점 상품",
                "digital_content": "디지털 이용권",
                "money_voucher": "금액권",
            },
            ProductCategory.BEAUTY: {
                "skincare": "스킨케어",
                "makeup": "메이크업",
                "fragrance": "향수",
                "body_care": "바디·핸드케어",
                "hair": "헤어케어",
            },
            ProductCategory.FOOD: {
                "meat_seafood": "정육·수산",
                "fruit": "과일",
                "bakery": "베이커리·떡",
                "sweets": "초콜릿·과자",
                "beverage": "차·커피·음료",
                "pantry": "간편식·가공식품",
            },
            ProductCategory.HEALTH: {
                "supplements": "영양제",
                "traditional_health": "홍삼·전통 건강식품",
                "sports_nutrition": "운동 보충식품",
                "diet": "다이어트·식단",
                "gut_inner_beauty": "유산균·이너뷰티",
                "recovery": "수면·피로회복",
            },
            ProductCategory.LIVING: {
                "kitchen": "주방·식기",
                "bedding_bath": "침구·욕실",
                "home_fragrance": "디퓨저·방향제",
                "decor": "조명·인테리어 소품",
                "flowers_plants": "꽃·식물",
                "character_goods": "인형·캐릭터 굿즈",
                "hobby": "보드게임·문구",
                "household": "생활용품",
            },
            ProductCategory.FASHION: _FASHION,
            ProductCategory.LUXURY: _FASHION,
            ProductCategory.DIGITAL: {
                "mobile_accessory": "모바일 액세서리",
                "massage_device": "마사지·온열 기기",
                "home_appliance": "생활 가전",
                "beauty_device": "뷰티 기기",
                "camera": "카메라",
                "audio": "음향 기기",
                "gaming": "게임",
                "pc_office": "PC·사무 기기",
            },
            ProductCategory.SPORTS: {
                "golf": "골프 용품",
                "sportswear": "스포츠웨어",
                "workout": "홈트·요가 용품",
                "running": "러닝 용품",
                "car": "차량용품",
                "camping": "캠핑 용품",
                "other_sports": "스포츠·아웃도어 용품",
            },
            ProductCategory.BOOKS_TICKETS: {
                "book": "도서",
                "comics_kids": "만화·어린이 책",
                "album": "음반",
                "fan_goods": "팬 굿즈",
                "ticket": "공연·전시 티켓",
            },
            ProductCategory.PET: {
                "pet_food": "간식·사료",
                "pet_toy": "장난감",
                "pet_wear": "의류·산책용품",
                "pet_care": "위생용품",
                "pet_living": "생활용품",
            },
        }.items()
    }
)


def subcategory_label(category: ProductCategory, subcategory: str) -> str:
    """대분류에 속한 세부 카테고리의 한국어 이름. 대분류에 없는 값이면 ValueError."""
    labels = SUBCATEGORIES[category]
    if subcategory not in labels:
        raise ValueError(f"{subcategory!r} is not a subcategory of {category.value}")
    return labels[subcategory]
