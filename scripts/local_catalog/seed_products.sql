-- 백엔드 MySQL products 테이블을 흉내 낸 로컬 원본. 동기화가 읽는 4개 컬럼만 둔다.
-- 가격 구간(low < 20000 <= mid < 80000 <= high)과 가격 없음이 골고루 섞이도록 30개를 넣는다.

DROP TABLE IF EXISTS products;

CREATE TABLE products (
    platform_type varchar(50) NOT NULL,
    external_id varchar(255) NOT NULL,
    name varchar(300) NOT NULL,
    price decimal(12, 2) NULL,
    PRIMARY KEY (platform_type, external_id)
) DEFAULT CHARSET = utf8mb4;

INSERT INTO products (platform_type, external_id, name, price) VALUES
('kakao', 'K-1001', '스타벅스 e카드 3만원 교환권', 30000),
('kakao', 'K-1002', '이솝 레저렉션 아로마틱 핸드밤 75ml', 39000),
('kakao', 'K-1003', '조말론 런던 우드 세이지 앤 씨 솔트 코롱 30ml', 115000),
('kakao', 'K-1004', '설빙 인절미 설빙 기프티콘', 10900),
('kakao', 'K-1005', '록시땅 시어 버터 핸드크림 30ml 3종 세트', 42000),
('kakao', 'K-1006', '딥티크 필로시코스 캔들 190g', 108000),
('kakao', 'K-1007', '교촌 허니콤보 + 콜라 1.25L', 26000),
('kakao', 'K-1008', '배스킨라빈스 파인트 아이스크림', 9800),
('kakao', 'K-1009', '정관장 홍삼정 에브리타임 10ml x 30포', 105000),
('kakao', 'K-1010', '르라보 상탈 33 오 드 퍼퓸 50ml', 298000),
('naver', 'N-2001', '로지텍 MX Master 3S 무선 마우스', 139000),
('naver', 'N-2002', '앤커 20000mAh 고속충전 보조배터리', 45900),
('naver', 'N-2003', '무인양품 초음파 아로마 디퓨저', 59000),
('naver', 'N-2004', '레고 크리에이터 3in1 해적선 31109', 129000),
('naver', 'N-2005', '몰스킨 클래식 하드커버 노트 라지 룰드', 32000),
('naver', 'N-2006', '라미 사파리 만년필 EF 블랙', 38000),
('naver', 'N-2007', '나이키 에브리데이 쿠션 크루 양말 3팩', 19000),
('naver', 'N-2008', '스탠리 퀜처 H2.0 텀블러 887ml', 59000),
('naver', 'N-2009', '브라운 시리즈 7 전기면도기', 249000),
('naver', 'N-2010', '핸드드립 입문 세트 드리퍼 서버 필터', 27900),
('coupang', 'C-3001', '필립스 에어프라이어 4.1L', 159000),
('coupang', 'C-3002', '요가매트 TPE 8mm 논슬립', 24900),
('coupang', 'C-3003', '닌텐도 스위치 마리오카트 8 디럭스', 64800),
('coupang', 'C-3004', '아이 원목 블록 놀이 세트 100피스', 18900),
('coupang', 'C-3005', '캠핑용 LED 랜턴 충전식', 29900),
('coupang', 'C-3006', '고디바 골드 컬렉션 초콜릿 15구', 52000),
('coupang', 'C-3007', '반려견 노즈워크 담요', 15900),
('coupang', 'C-3008', '다이슨 슈퍼소닉 헤어드라이어', 599000),
('coupang', 'C-3009', '손글씨 캘리그라피 입문 키트', NULL),
('coupang', 'C-3010', '제주 한라봉 선물세트 3kg', NULL);
