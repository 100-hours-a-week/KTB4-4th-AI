#!/usr/bin/env bash
# 로컬 카탈로그 적재 테스트 환경을 만든다.
#   1. 테스트 전용 MySQL(3307)을 띄우고 상품 30개를 넣는다.
#   2. PostgreSQL 에 needu_catalog_local 을 새로 만들고 migrations/*.sql 을 적용한다.
# 다시 실행하면 두 DB 모두 처음 상태로 돌아간다.
# 최초 실행 전: cp scripts/local_catalog/env.example.sh scripts/local_catalog/env.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export LOCAL_CATALOG_ROOT="$ROOT"
# shellcheck source=env.sh
source "$ROOT/scripts/local_catalog/env.sh"

MYSQL_BASE="${MYSQL_BASE:-/usr/local/mysql}"
MYSQL_DIR="$ROOT/.local/mysql"
MYSQL_SOCK="$MYSQL_DIR/mysql.sock"
PG_DB=needu_catalog_local

mysql_root() {
    "$MYSQL_BASE/bin/mysql" --socket="$MYSQL_SOCK" -uroot "$@"
}

# ---- MySQL 원본 ----
mkdir -p "$MYSQL_DIR"
if [ ! -d "$MYSQL_DIR/data" ]; then
    echo "[mysql] initializing data directory"
    "$MYSQL_BASE/bin/mysqld" --initialize-insecure \
        --basedir="$MYSQL_BASE" --datadir="$MYSQL_DIR/data" \
        --log-error="$MYSQL_DIR/init.err"
fi

if ! mysql_root -e "SELECT 1" >/dev/null 2>&1; then
    echo "[mysql] starting on 127.0.0.1:$SOURCE_MYSQL_PORT"
    nohup "$MYSQL_BASE/bin/mysqld" \
        --basedir="$MYSQL_BASE" --datadir="$MYSQL_DIR/data" \
        --port="$SOURCE_MYSQL_PORT" --bind-address=127.0.0.1 \
        --socket="$MYSQL_SOCK" --mysqlx=OFF \
        --pid-file="$MYSQL_DIR/mysqld.pid" --log-error="$MYSQL_DIR/mysqld.err" \
        >/dev/null 2>&1 &
    for _ in $(seq 1 30); do
        mysql_root -e "SELECT 1" >/dev/null 2>&1 && break
        sleep 1
    done
    mysql_root -e "SELECT 1" >/dev/null || { echo "mysql did not start; see $MYSQL_DIR/mysqld.err"; exit 1; }
fi

echo "[mysql] seeding $SOURCE_MYSQL_DATABASE.$SOURCE_MYSQL_TABLE"
mysql_root <<SQL
CREATE DATABASE IF NOT EXISTS $SOURCE_MYSQL_DATABASE DEFAULT CHARSET utf8mb4;
# aiomysql 은 cryptography 없이 caching_sha2_password 로 붙지 못한다.
CREATE USER IF NOT EXISTS '$SOURCE_MYSQL_USER'@'%' IDENTIFIED WITH mysql_native_password BY '$SOURCE_MYSQL_PASSWORD';
ALTER USER '$SOURCE_MYSQL_USER'@'%' IDENTIFIED WITH mysql_native_password BY '$SOURCE_MYSQL_PASSWORD';
GRANT SELECT ON $SOURCE_MYSQL_DATABASE.* TO '$SOURCE_MYSQL_USER'@'%';
SQL
mysql_root --default-character-set=utf8mb4 "$SOURCE_MYSQL_DATABASE" < "$ROOT/scripts/local_catalog/seed_products.sql"
mysql_root -N -e "SELECT CONCAT('[mysql] products = ', COUNT(*)) FROM $SOURCE_MYSQL_DATABASE.$SOURCE_MYSQL_TABLE"

# ---- PostgreSQL 카탈로그 ----
echo "[postgres] recreating $PG_DB"
dropdb --if-exists "$PG_DB"
createdb "$PG_DB"
for migration in "$ROOT"/migrations/*.sql; do
    echo "[postgres] applying $(basename "$migration")"
    psql -q -v ON_ERROR_STOP=1 -d "$PG_DB" -f "$migration"
done
psql -Atq -d "$PG_DB" -c "SELECT '[postgres] pgvector ' || extversion FROM pg_extension WHERE extname = 'vector'"

echo "done. next: source scripts/local_catalog/env.sh && uv run python -m app.jobs.catalog_sync"
