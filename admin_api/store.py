import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def now() -> str:
    return datetime.now(UTC).isoformat()


class Store:
    def __init__(self, path: Path | None = None):
        self.path = path or Path(os.environ.get("AI_LAB_DB", ROOT / ".local/ai-lab/lab.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS connections (
                    id TEXT PRIMARY KEY, data TEXT NOT NULL,
                    probe TEXT, checked_at TEXT
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, spec TEXT NOT NULL,
                    connection TEXT NOT NULL, manifest TEXT NOT NULL,
                    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    summary TEXT, error TEXT
                );
                CREATE TABLE IF NOT EXISTS samples (
                    run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    ordinal INTEGER NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY (run_id, ordinal)
                );
                CREATE TABLE IF NOT EXISTS heartbeat (
                    id INTEGER PRIMARY KEY CHECK(id=1), updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    id TEXT PRIMARY KEY, data TEXT NOT NULL
                );
            """)
            for connection_id, name, provider, port in [
                ("lmstudio", "LM Studio", "lmstudio", 1234),
                ("ollama", "Ollama", "ollama", 11434),
            ]:
                data = dict(
                    name=name,
                    provider=provider,
                    environment="local",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    model="",
                    revision="",
                    precision="",
                    hardware="",
                    api_key_env="",
                )
                db.execute(
                    "INSERT OR IGNORE INTO connections(id,data) VALUES (?,?)",
                    (connection_id, json.dumps(data, ensure_ascii=False)),
                )
            db.execute(
                "INSERT OR IGNORE INTO connections(id,data) VALUES (?,?)",
                (
                    "huggingface",
                    json.dumps(
                        dict(
                            name="Hugging Face · Qwen3.5 9B",
                            provider="huggingface",
                            environment="other",
                            base_url="https://router.huggingface.co/v1",
                            model="Qwen/Qwen3.5-9B:deepinfra",
                            api_key_env="HF_TOKEN",
                            revision="",
                            precision="",
                            hardware="호스팅 제공업체 관리 · 정밀도와 GPU 미확인",
                        ),
                        ensure_ascii=False,
                    ),
                ),
            )

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def connection_row(row):
        return {
            "id": row["id"],
            **json.loads(row["data"]),
            "probe": json.loads(row["probe"]) if row["probe"] else None,
            "checked_at": row["checked_at"],
        }

    def connections(self):
        with self.db() as db:
            return [self.connection_row(r) for r in db.execute("SELECT * FROM connections")]

    def connection(self, connection_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM connections WHERE id=?", (connection_id,)).fetchone()
            return self.connection_row(row) if row else None

    def save_connection(self, connection_id, data):
        with self.db() as db:
            db.execute(
                "INSERT INTO connections(id,data) VALUES (?,?) ON CONFLICT(id) "
                "DO UPDATE SET data=excluded.data, probe=NULL, checked_at=NULL",
                (connection_id, json.dumps(data, ensure_ascii=False)),
            )
        return self.connection(connection_id)

    def save_probe(self, connection_id, probe):
        with self.db() as db:
            db.execute(
                "UPDATE connections SET probe=?, checked_at=? WHERE id=?",
                (json.dumps(probe, ensure_ascii=False), now(), connection_id),
            )

    def create_run(self, spec, connection, manifest):
        run_id = uuid4().hex
        with self.db() as db:
            db.execute(
                "INSERT INTO runs(id,status,spec,connection,manifest,created_at) "
                "VALUES (?, 'queued', ?, ?, ?, ?)",
                (
                    run_id,
                    json.dumps(spec, ensure_ascii=False),
                    json.dumps(connection, ensure_ascii=False),
                    json.dumps(manifest, ensure_ascii=False),
                    now(),
                ),
            )
        return self.run(run_id)

    @staticmethod
    def run_row(row):
        data = dict(row)
        for key in ("spec", "connection", "manifest", "summary"):
            data[key] = json.loads(data[key]) if data[key] else None
        data["cancel_requested"] = bool(data["cancel_requested"])
        return data

    def runs(self):
        with self.db() as db:
            return [
                {**self.run_row(r), "completed": r["completed"]}
                for r in db.execute(
                    "SELECT runs.*, (SELECT count(*) FROM samples WHERE run_id=runs.id) "
                    "AS completed FROM runs ORDER BY created_at DESC LIMIT 100"
                )
            ]

    def run(self, run_id):
        with self.db() as db:
            row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            if not row:
                return None
            data = self.run_row(row)
            data["samples"] = [
                {"ordinal": r["ordinal"], **json.loads(r["data"])}
                for r in db.execute(
                    "SELECT * FROM samples WHERE run_id=? ORDER BY ordinal", (run_id,)
                )
            ]
            return data

    def claim(self):
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id FROM runs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if not row:
                return None
            db.execute(
                "UPDATE runs SET status='running', started_at=? WHERE id=?", (now(), row["id"])
            )
        return self.run(row["id"])

    def sample(self, run_id, ordinal, data):
        with self.db() as db:
            db.execute(
                "INSERT INTO samples(run_id,ordinal,data) VALUES (?,?,?)",
                (run_id, ordinal, json.dumps(data, ensure_ascii=False)),
            )

    def cancel(self, run_id):
        with self.db() as db:
            db.execute(
                "UPDATE runs SET cancel_requested=1, "
                "status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END, "
                "finished_at=CASE WHEN status='queued' THEN ? ELSE finished_at END "
                "WHERE id=? AND status IN ('queued','running')",
                (now(), run_id),
            )

    def cancelled(self, run_id):
        with self.db() as db:
            row = db.execute("SELECT cancel_requested FROM runs WHERE id=?", (run_id,)).fetchone()
            return not row or bool(row[0])

    def finish(self, run_id, status, summary, error=None):
        with self.db() as db:
            db.execute(
                "UPDATE runs SET status=?, summary=?, error=?, finished_at=? WHERE id=?",
                (status, json.dumps(summary), error, now(), run_id),
            )

    def delete(self, run_id):
        with self.db() as db:
            cursor = db.execute("DELETE FROM runs WHERE id=? AND status!='running'", (run_id,))
            return cursor.rowcount > 0

    def recover(self):
        # Called only while holding the exclusive worker lock; never retry a billed request.
        with self.db() as db:
            db.execute(
                "UPDATE runs SET status='interrupted', finished_at=?, "
                "error='워커가 종료되어 실행이 중단됐습니다. 자동 재실행하지 않습니다.' "
                "WHERE status='running'",
                (now(),),
            )

    def beat(self):
        with self.db() as db:
            db.execute(
                "INSERT INTO heartbeat VALUES (1,?) ON CONFLICT(id) "
                "DO UPDATE SET updated_at=excluded.updated_at",
                (now(),),
            )

    def worker_status(self):
        with self.db() as db:
            row = db.execute("SELECT updated_at FROM heartbeat WHERE id=1").fetchone()
        seen = row[0] if row else None
        return {
            "online": bool(
                seen and (datetime.now(UTC) - datetime.fromisoformat(seen)).total_seconds() < 8
            ),
            "last_seen": seen,
        }

    def save_chat(self, session):
        with self.db() as db:
            db.execute(
                "INSERT INTO chat_sessions VALUES (?,?) ON CONFLICT(id) "
                "DO UPDATE SET data=excluded.data",
                (session["id"], json.dumps(session, ensure_ascii=False)),
            )

    def chat(self, session_id):
        with self.db() as db:
            row = db.execute("SELECT data FROM chat_sessions WHERE id=?", (session_id,)).fetchone()
            return json.loads(row[0]) if row else None

    def chats(self):
        with self.db() as db:
            return [
                json.loads(row[0])
                for row in db.execute("SELECT data FROM chat_sessions ORDER BY rowid DESC LIMIT 50")
            ]

    def delete_chat(self, session_id):
        with self.db() as db:
            return db.execute("DELETE FROM chat_sessions WHERE id=?", (session_id,)).rowcount > 0
