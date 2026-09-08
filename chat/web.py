#!/usr/bin/env python3
"""브라우저에서 대화해보는 UI. 의존성 없음(stdlib).

  python3 chat/web.py        →  http://127.0.0.1:7860

왼쪽은 사용자가 보는 채팅, 오른쪽은 서버만 보는 것(이번 턴 목표·추출된 스키마).
"""

from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import finalize
import llm
import policy
import prompts
from schema import DELTA_SCHEMA, FIELDS, State

PORT = 7860

_lock = threading.Lock()
_s = State()
_closed = False


def state_json(s: State) -> dict:
    buckets: dict[str, list] = {f: [] for f in FIELDS}
    for i in s.items:
        buckets[i.field].append({
            "value": i.value,
            "confidence": round(i.confidence, 2),
            "role": i.link_role,
            "visibility": i.visibility,
            "intent": i.intent_type,
            "defer": i.deferral_reason,
            "evidence": i.evidence,
        })
    return {
        "buckets": {k: v for k, v in buckets.items() if v},
        "axes": list(s.axes),
        "counts": s.summary_counts(),
        "goals": list(s.goal_log),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 콘솔 조용히
        pass

    # ------------------------------------------------------------ helpers
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse_open(self):
        # keep-alive 로 두면 스트림이 끝나도 브라우저가 계속 기다린다.
        # 매 요청을 닫아야 fetch reader 가 done 을 받는다.
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, event, data):
        chunk = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        try:
            self.wfile.write(chunk.encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 브라우저가 먼저 끊음

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    # ------------------------------------------------------------ routes
    def do_GET(self):
        if self.path.split("?")[0] not in ("/", "/index.html"):
            self.send_error(404)
            return
        html = (HERE / "ui.html").read_text(encoding="utf-8").encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)

    def do_POST(self):
        route = self.path.split("?")[0]
        if route == "/api/reset":
            return self._reset()
        if route == "/api/open":
            return self._open()
        if route == "/api/turn":
            return self._turn()
        if route == "/api/finish":
            return self._finish()
        self.send_error(404)

    # ------------------------------------------------------------ handlers
    def _reset(self):
        global _s, _closed
        with _lock:
            _s = State()
            _closed = False
        self._json({"ok": True, "models": {
            "response": llm.RESPONSE_MODEL, "extract": llm.EXTRACT_MODEL,
            "base": llm.BASE_URL,
        }})

    def _open(self):
        """첫 턴 — 유저 발화 없이 AI 가 먼저 연다."""
        s = _s
        self._sse_open()
        g = policy.decide(s)
        s.goal_log.append(g.name)
        s.note_try(g.name)
        self._sse("goal", {"name": g.name, "reason": g.reason, "directive": g.directive})

        reply = ""
        t0 = time.time()
        ttft = None
        try:
            for piece in llm.stream_chat(prompts.opening_messages(g.directive)):
                if ttft is None:
                    ttft = time.time() - t0
                reply += piece
                self._sse("token", {"t": piece})
        except Exception as e:
            self._sse("error", {"message": str(e)})
            return
        cleaned = llm.one_question(llm.strip_meta(reply))
        if cleaned != reply.strip():
            self._sse("replace", {"text": cleaned})
            reply = cleaned
        s.history.append({"role": "user", "content": "(대화 시작)"})
        s.history.append({"role": "assistant", "content": reply})
        s.last_reply = reply
        self._sse("done", {"ttft": round(ttft or 0, 2),
                           "total": round(time.time() - t0, 2),
                           "state": state_json(s), "closed": False})

    def _turn(self):
        global _closed
        user = (self._body().get("message") or "").strip()
        s = _s
        self._sse_open()
        if not user:
            self._sse("error", {"message": "빈 발화"})
            return
        if _closed:
            self._sse("error", {"message": "대화가 종료됨. 새로 시작하세요."})
            return

        with _lock:
            s.turn += 1
            s.user_lengths.append(len(user))
            g = policy.decide(s)
            s.goal_log.append(g.name)
            s.note_try(g.name)
        s.note_try(g.name)

        self._sse("goal", {"name": g.name, "reason": g.reason, "directive": g.directive})

        # 사용자가 듣고 답한 문장. ② 가 s.last_reply 를 덮어쓰기 전에 붙잡는다.
        prev_reply = s.last_reply

        # ② 응답 모델 — 사용자가 기다리는 유일한 호출
        reply, t0, ttft = "", time.time(), None
        try:
            for piece in llm.stream_chat(prompts.response_messages(s, user, g.directive)):
                if ttft is None:
                    ttft = time.time() - t0
                reply += piece
                self._sse("token", {"t": piece})
        except Exception as e:
            self._sse("error", {"message": f"응답 실패: {e}"})
            return
        reply_ms = time.time() - t0

        cleaned = llm.one_question(llm.strip_meta(reply))
        if cleaned != reply.strip():
            self._sse("replace", {"text": cleaned})
            reply = cleaned

        s.history.append({"role": "user", "content": user})
        s.history.append({"role": "assistant", "content": reply})
        s.last_reply = reply

        # ③ 추출 모델 — 사용자는 안 기다림
        t1 = time.time()
        try:
            raw = llm.json_chat(prompts.extract_messages(s, user, prev_reply), DELTA_SCHEMA)
        except Exception as e:
            self._sse("extract_error", {"message": str(e)})
            raw = {"items": [], "axes": [], "drop": []}

        with _lock:
            added, dropped = s.apply(raw)
            s.note_result(g.name, len(added))

        self._sse("extract", {
            "added": [{"value": i.value, "field": i.field,
                       "confidence": round(i.confidence, 2),
                       "role": i.link_role, "intent": i.intent_type,
                       "defer": i.deferral_reason} for i in added],
            "dropped": dropped,
            "raw": raw,
            "ms": round((time.time() - t1) * 1000),
        })

        # 대화를 마칠 차례면 여기서 자동 종료한다
        closing = g.name == "WRAP"
        if closing:
            _closed = True

        self._sse("done", {
            "ttft": round(ttft or 0, 2), "total": round(reply_ms, 2),
            "state": state_json(s), "closed": closing,
        })

    def _finish(self):
        s = _s
        self._sse_open()
        self._sse("stage", {"text": "요약문과 검색 쿼리를 만드는 중"})
        t0 = time.time()
        try:
            r = finalize.build(s)
        except Exception as e:
            self._sse("error", {"message": f"마무리 실패: {e}"})
            return
        r["ms"] = round((time.time() - t0) * 1000)
        self._sse("result", r)


def main():
    print(f"LM Studio  {llm.BASE_URL}")
    print(f"응답 모델   {llm.RESPONSE_MODEL}")
    print(f"추출 모델   {llm.EXTRACT_MODEL}")
    print(f"\n  http://127.0.0.1:{PORT}\n")
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Timer(0.8, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n종료")


if __name__ == "__main__":
    main()
