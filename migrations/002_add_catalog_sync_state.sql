-- MySQL 원본 동기화에 필요한 실행 기록과 상태 컬럼을 추가한다.
-- 001은 원본 4개 필드만 담고 있어서 변경 감지와 재처리 대상 선별을 할 수 없다.

CREATE TABLE catalog_sync_runs (
    run_id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    mode text NOT NULL CHECK (mode IN ('full', 'incremental')),
    status text NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'succeeded', 'failed')),
    received_count integer NOT NULL DEFAULT 0 CHECK (received_count >= 0),
    changed_count integer NOT NULL DEFAULT 0 CHECK (changed_count >= 0),
    deactivated_count integer NOT NULL DEFAULT 0 CHECK (deactivated_count >= 0),
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz
);

ALTER TABLE catalog_products
    -- 원본 4개 값 전체의 해시. 달라진 행만 실제로 쓴다.
    ADD COLUMN source_hash char(64) NOT NULL,
    -- 문서 생성 입력(상품명 + 가격 구간)만의 해시.
    -- 가격이 같은 구간 안에서 움직이면 이 값은 그대로라 LLM을 다시 부르지 않는다.
    ADD COLUMN doc_input_hash char(64) NOT NULL,
    ADD COLUMN is_active boolean NOT NULL DEFAULT true,
    ADD COLUMN processing_status text NOT NULL DEFAULT 'pending'
        CHECK (processing_status IN ('pending', 'processing', 'ready', 'failed')),
    ADD COLUMN processing_error text,
    ADD COLUMN last_seen_run_id bigint REFERENCES catalog_sync_runs(run_id),
    ADD COLUMN source_synced_at timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN created_at timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN updated_at timestamptz NOT NULL DEFAULT now();

-- 문서 생성 배치가 처리 대상을 고르는 경로.
CREATE INDEX ix_catalog_products_pending
    ON catalog_products (processing_status, updated_at)
    WHERE is_active;

-- 전체 동기화 후 이번 실행에서 안 보인 상품을 내릴 때 쓰는 경로.
CREATE INDEX ix_catalog_products_last_seen
    ON catalog_products (last_seen_run_id)
    WHERE is_active;
