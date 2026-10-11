-- 세부 카테고리와 중복 묶음. 원본 DB에는 없고 AI DB에만 있는 판단 결과다.
--
-- 세부 카테고리는 판단 모델이 대분류별 선택지(app/domain/catalog/subcategories.py)에서 고른다.
-- 선택지는 추천을 다듬으면서 바뀔 수 있어 CHECK 로 묶지 않고 코드에서 검증한다.
-- 동기화는 이 컬럼들을 쓰지 않으므로, 원본이 바뀌어 다시 pending 이 되어도 값이 남는다.

ALTER TABLE catalog_products
    ADD COLUMN subcategory text,
    -- 판단 모델이 고른 선택지의 확률. 낮으면 문서 생성 입력과 추천의 종류별 처리에서 뺀다.
    ADD COLUMN subcategory_confidence real
        CHECK (subcategory_confidence BETWEEN 0 AND 1),
    -- 두 번째 후보와 확률("massage_device:0.42"). 추천에는 쓰지 않고 선택지 기준을 고칠 때 본다.
    ADD COLUMN subcategory_second text,
    -- 정규화한 상품명과 카테고리의 해시. 매장·옵션만 다른 같은 이름 상품을 묶는다.
    ADD COLUMN dedup_group text,
    -- 매장 이름이 없는 금액권("1만원권"). 어디서 쓰는지 알 수 없어 취향 추천에서 뺀다.
    ADD COLUMN is_generic_voucher boolean NOT NULL DEFAULT false,
    ADD CONSTRAINT ck_catalog_products_subcategory_confidence
        CHECK ((subcategory IS NULL) = (subcategory_confidence IS NULL));
