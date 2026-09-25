-- LLM이 작성한 content/usage/gift 검색 문서와 그 임베딩을 담는다.
-- 벡터 차원은 임베딩 모델이 확정된 뒤에 vector(N)으로 좁히고 인덱스를 건다.
-- 지금은 차원을 고정하지 않는다.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE catalog_embedding_models (
    model_version text PRIMARY KEY,
    provider text NOT NULL,
    model_name text NOT NULL,
    dimensions integer NOT NULL CHECK (dimensions > 0),
    is_active boolean NOT NULL DEFAULT false,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- 활성 모델은 항상 하나뿐이다. 새 모델로 전체를 다시 만드는 동안
-- 기존 모델 벡터로 추천을 계속 제공하고, 적재가 끝난 뒤에만 전환한다.
CREATE UNIQUE INDEX uq_catalog_one_active_embedding_model
    ON catalog_embedding_models (is_active)
    WHERE is_active;

CREATE TABLE catalog_product_documents (
    platform varchar(50) NOT NULL,
    external_id varchar(255) NOT NULL,
    -- content: 이 상품은 무엇인가
    -- usage:   어떤 활동에서 쓰는가
    -- gift:    누구에게 왜 선물하기 좋은가
    space text NOT NULL CHECK (space IN ('content', 'usage', 'gift')),
    document_text text NOT NULL CHECK (length(btrim(document_text)) > 0),
    document_hash char(64) NOT NULL,
    -- 문서 생성 프롬프트와 검증 규칙의 버전. 규칙이 바뀌면 재생성 대상이 된다.
    document_version text NOT NULL,
    embedding_model_version text NOT NULL
        REFERENCES catalog_embedding_models(model_version),
    embedding vector NOT NULL,
    generated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (platform, external_id, space, embedding_model_version),
    FOREIGN KEY (platform, external_id)
        REFERENCES catalog_products(platform, external_id) ON DELETE CASCADE
);

-- 추천 검색은 항상 공간 하나와 활성 모델 버전 하나로 좁힌 뒤 벡터를 정렬한다.
CREATE INDEX ix_catalog_documents_search
    ON catalog_product_documents (space, embedding_model_version);

-- 문서 배치는 embedding_model_version 으로 이 표를 참조한다.
-- 행이 없으면 저장이 외래키 위반으로 실패하므로 먼저 넣어둔다.
-- model_version 값은 EMBEDDING_SPACE_ID 설정과 같아야 한다.
INSERT INTO catalog_embedding_models (model_version, provider, model_name, dimensions, is_active)
VALUES ('solar-embedding-2', 'upstage', 'solar-embedding-2-passage', 1024, true)
ON CONFLICT (model_version) DO NOTHING;
