-- 상품 카테고리(11개)와 선물 테마(6개)를 담는다.
-- 카테고리는 메인 추천(self)과 v2 카테고리 탭에, 테마는 선물 목록(gift)에 나올 수 있는지
-- (gift_available) 판정에 쓴다.
--
-- 기존 행은 다음 동기화에서 원본 해시가 달라져 라벨이 채워지고 pending 으로 돌아간다.
-- 여기서 미리 pending 으로 돌리면 라벨 없이 문서를 한 번 더 만들게 되므로 그러지 않는다.

ALTER TABLE catalog_products
    -- 원본 product_category 그대로. 카테고리·테마 규칙이 바뀌면 이 값으로 다시 계산한다.
    ADD COLUMN source_labels text[] NOT NULL DEFAULT '{}',
    ADD COLUMN category text
        CHECK (category IN (
            'voucher', 'beauty', 'food', 'health', 'living', 'fashion',
            'digital', 'sports', 'luxury', 'books_tickets', 'pet'
        )),
    -- source: 원본 product_category 에 있던 값, inferred: 원본에 없어 문서 생성 모델이 고름
    ADD COLUMN category_source text
        CHECK (category_source IN ('source', 'inferred')),
    -- inferred 일 때만 채운다.
    ADD COLUMN category_confidence real
        CHECK (category_confidence BETWEEN 0 AND 1),
    ADD COLUMN themes text[] NOT NULL DEFAULT '{}'
        CHECK (themes <@ ARRAY[
            'birthday', 'housewarming', 'thanks', 'cheer',
            'friend', 'fun'
        ]::text[]),
    -- 선물 목록(마이 페이지·친구 상세 페이지)에 나올 수 있는 상품. 선물테마가 붙었거나
    -- 교환권·명품·패션이면 true 다. gift 문서는 이 상품에만 만든다.
    ADD COLUMN gift_available boolean NOT NULL DEFAULT false,
    ADD CONSTRAINT ck_catalog_products_category_source
        CHECK ((category IS NULL) = (category_source IS NULL));

-- v2 카테고리 탭과 개인추천의 카테고리 필터.
CREATE INDEX ix_catalog_products_category
    ON catalog_products (category)
    WHERE is_active AND processing_status = 'ready';

-- 선물 목록 검색. 상품 수가 적어 지금은 순차 검색과 차이가 없지만 풀이 커질 때를 대비한다.
CREATE INDEX ix_catalog_products_gift_available
    ON catalog_products (platform, external_id)
    WHERE gift_available AND is_active AND processing_status = 'ready';
