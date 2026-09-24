-- 적재 결과 확인. psql -d needu_catalog_local -f scripts/local_catalog/verify.sql

\echo '== 동기화 실행 기록'
SELECT run_id, mode, status, received_count, changed_count, deactivated_count
  FROM catalog_sync_runs ORDER BY run_id;

\echo '== 상품 처리 상태'
SELECT processing_status, count(*) FROM catalog_products GROUP BY 1 ORDER BY 1;

\echo '== 실패 사유'
SELECT platform, external_id, name, processing_error
  FROM catalog_products WHERE processing_status = 'failed';

\echo '== 공간별 문서 수와 벡터 차원'
SELECT space, embedding_model_version, count(*) AS docs,
       min(vector_dims(embedding)) AS min_dims, max(vector_dims(embedding)) AS max_dims
  FROM catalog_product_documents GROUP BY 1, 2 ORDER BY 1;

\echo '== 문서 3개가 다 없는 ready 상품 (0행이어야 정상)'
SELECT p.platform, p.external_id, count(d.space) AS docs
  FROM catalog_products p
  LEFT JOIN catalog_product_documents d USING (platform, external_id)
 WHERE p.processing_status = 'ready'
 GROUP BY 1, 2 HAVING count(d.space) <> 3;

\echo '== 문서 샘플'
SELECT p.name, d.space, d.document_text
  FROM catalog_product_documents d JOIN catalog_products p USING (platform, external_id)
 ORDER BY p.external_id, d.space LIMIT 9;

\echo '== 벡터 검색 스모크: 조말론 코롱의 gift 벡터와 가까운 상품'
SELECT p.name, round((d.embedding <=> q.embedding)::numeric, 4) AS cosine_distance
  FROM catalog_product_documents d
  JOIN catalog_products p USING (platform, external_id)
  CROSS JOIN (SELECT embedding FROM catalog_product_documents
               WHERE external_id = 'K-1003' AND space = 'gift') q
 WHERE d.space = 'gift'
 ORDER BY d.embedding <=> q.embedding LIMIT 5;
