#!/usr/bin/env python3
"""터미널에서 직접 대화해보는 하네스.

  python3 chat/run.py

한 턴의 흐름
  ① 서버가 이번 턴 목표를 정한다      (policy.decide — 순수 로직)
  ② 응답 모델이 그 지시대로 답한다     (스트리밍)
  ③ 추출 모델이 델타 JSON 을 낸다      (구조화 출력, 사용자는 안 기다림)
  ④ 서버가 병합·방출한다              (schema.State.apply)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import finalize
import llm
import policy
import prompts
from schema import DELTA_SCHEMA, State

DIM = "\033[2m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YEL = "\033[33m"
GRN = "\033[32m"
RED = "\033[31m"
END = "\033[0m"


def show_goal(g, verbose: bool) -> None:
    print(f"{DIM}┌ 목표 {BOLD}{g.name}{END}{DIM}  ({g.reason}){END}")
    if verbose:
        print(f"{DIM}└ 지시: {g.directive}{END}")


def show_delta(added, dropped, raw, verbose: bool) -> None:
    if not added and not dropped:
        print(f"{DIM}  · 새로 알아낸 것 없음{END}")
    for i in added:
        bits = [i.field, f"{i.confidence:.1f}"]
        if i.intent_type:
            bits.append(i.intent_type)
        if i.deferral_reason:
            bits.append(f"defer:{i.deferral_reason}")
        print(f"{GRN}  + {i.value}{END} {DIM}({' · '.join(bits)}){END}")
    for v in dropped:
        print(f"{RED}  - {v}{END}")
    if verbose:
        print(f"{DIM}  raw: {json.dumps(raw, ensure_ascii=False)}{END}")


def show_state(s: State) -> None:
    c = s.summary_counts()
    line = " · ".join(f"{k} {v}" for k, v in c.items())
    print(f"{DIM}  [{line}]{END}")


def dump_state(s: State) -> None:
    print(f"\n{BOLD}── 현재 상태 ──{END}")
    print(s.state_block() or "(비어 있음)")
    print()
    for k, v in s.summary_counts().items():
        print(f"  {k:10} {v}")
    print(f"  {'목표이력':10} {' → '.join(s.goal_log)}")
    print()


def show_result(s: State, out_path: str = "") -> None:
    print(f"\n{BOLD}── 대화 종료. 임베딩 단계로 넘길 것 ──{END}\n")
    t0 = time.time()
    try:
        r = finalize.build(s)
    except Exception as e:
        print(f"{RED}마무리 실패: {e}{END}")
        return

    print(f"{BOLD}요약문{END}  {DIM}(친구도 보는 글){END}")
    print(f"  {r['summary'] or '(만들 재료가 없음)'}\n")

    print(f"{BOLD}임베딩에 던질 쿼리{END} {DIM}({len(r['queries'])}개 · 배치 {len(r['embeddingBatch'])}건 1회 호출){END}")
    if not r["queries"]:
        print(f"  {RED}없음 — 검색을 시작할 수 없다{END}")
    for q in r["queries"]:
        tag = "직접" if q["kind"] == "direct" else "생활파생"
        color = GRN if q["kind"] == "direct" else CYAN
        gw = f" · 선물가중 {q['giftWeight']}" if q["giftWeight"] != 1.0 else ""
        print(f"  {color}[{tag}]{END} {q['text']}")
        print(f"       {DIM}← {q['source']} · {q['field']} · {q['mode']} · {q['confidence']:.2f}{gw}{END}")

    print(f"\n{BOLD}하드필터{END} {DIM}(SQL WHERE 에 들어감){END}")
    for k, v in r["filters"].items():
        print(f"  {k:12} {' · '.join(v) if v else DIM + '없음' + END}")

    print(f"\n{BOLD}순위 조정용{END}")
    print(f"  {' · '.join(r['weights']) if r['weights'] else DIM + '없음' + END}")
    if r["axes"]:
        print(f"\n{BOLD}판단 기준{END}\n  {' · '.join(r['axes'])}")

    print(f"\n{DIM}({time.time() - t0:.1f}s){END}")
    print(f"\n{DIM}mode: both=양쪽 · self=메인만 · gift=선물만{END}")
    print(f"{DIM}생활파생은 self 전용 — 친구에게 사정을 노출하지 않기 위해서다{END}")
    print(f"{DIM}임베딩 호출과 벡터 검색은 이 하네스 밖이다. 여기까지가 그 입력이다.{END}\n")

    if out_path:
        Path(out_path).write_text(
            json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{DIM}→ {out_path} 에 저장{END}\n")


def show_profile(s: State) -> None:
    """계약(TasteProfile)으로 확정한 프로필. 요약문은 LLM 을 부르지 않고 비워둔다."""
    print(f"\n{BOLD}── TasteProfile (계약 형태) ──{END}")
    print(json.dumps(s.to_profile(), ensure_ascii=False, indent=2))
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true", help="지시문과 델타 원본까지 출력")
    ap.add_argument("--no-extract", action="store_true", help="추출 모델 끄고 대화만")
    ap.add_argument("--json", dest="json_out", default="", metavar="PATH",
                    help="마무리 결과(프로필·쿼리)를 JSON 파일로 저장")
    args = ap.parse_args()

    print(f"{BOLD}니쥬 대화 하네스{END}")
    print(f"{DIM}응답 모델 {llm.RESPONSE_MODEL} · 추출 모델 {llm.EXTRACT_MODEL}{END}")
    print(f"{DIM}{llm.BASE_URL}{END}")
    print(f"{DIM}/state 상태  /profile 프로필  /finish 결과  /quit 종료{END}\n")

    s = State()

    # ── 첫 턴: 유저 발화 없이 AI 가 먼저 연다
    g = policy.decide(s)
    s.goal_log.append(g.name)
    show_goal(g, args.verbose)
    print(f"{CYAN}AI ▸ {END}", end="", flush=True)
    reply = ""
    try:
        for piece in llm.stream_chat(prompts.opening_messages(g.directive)):
            print(piece, end="", flush=True)
            reply += piece
    except Exception as e:
        print(f"\n{RED}응답 실패: {e}{END}")
        return
    print("\n")
    s.history.append({"role": "user", "content": "(대화 시작)"})
    s.history.append({"role": "assistant", "content": reply})
    s.last_reply = reply

    # ── 본 루프
    while True:
        try:
            user = input(f"{YEL}나 ▸ {END}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user in ("/quit", "/q"):
            break
        if user == "/state":
            dump_state(s)
            continue
        if user in ("/profile", "/p"):
            show_profile(s)
            continue
        if user in ("/finish", "/f"):
            show_result(s, args.json_out)
            continue

        s.turn += 1
        s.user_lengths.append(len(user))

        # ① 서버가 목표를 정한다
        g = policy.decide(s)
        s.goal_log.append(g.name)
        s.note_try(g.name)
        print()
        show_goal(g, args.verbose)

        # 사용자가 듣고 답한 문장. ② 가 s.last_reply 를 덮어쓰기 전에 붙잡는다.
        prev_reply = s.last_reply

        # ② 응답 모델 (사용자가 기다리는 유일한 호출)
        print(f"{CYAN}AI ▸ {END}", end="", flush=True)
        t0 = time.time()
        ttft = None
        reply = ""
        try:
            for piece in llm.stream_chat(prompts.response_messages(s, user, g.directive)):
                if ttft is None:
                    ttft = time.time() - t0
                print(piece, end="", flush=True)
                reply += piece
        except Exception as e:
            print(f"\n{RED}응답 실패: {e}{END}")
            continue
        total = time.time() - t0
        print(f"\n{DIM}  ({ttft or 0:.2f}s 첫글자 / {total:.2f}s 완료){END}")

        reply = llm.one_question(llm.strip_meta(reply))
        s.history.append({"role": "user", "content": user})
        s.history.append({"role": "assistant", "content": reply})
        s.last_reply = reply

        # ③ 추출 모델 (사용자는 안 기다림)
        if not args.no_extract:
            t1 = time.time()
            try:
                raw = llm.json_chat(prompts.extract_messages(s, user, prev_reply), DELTA_SCHEMA)
            except Exception as e:
                print(f"{RED}  추출 실패: {e}{END}")
                raw = {"items": [], "axes": [], "drop": []}
            # ④ 서버가 병합
            added, dropped = s.apply(raw)
            s.note_result(g.name, len(added))
            show_delta(added, dropped, raw, args.verbose)
            print(f"{DIM}  (추출 {time.time() - t1:.2f}s){END}")

        show_state(s)
        print()

        if g.name == "WRAP":
            print(f"{BOLD}대화가 마무리 단계입니다. /state 로 결과를 보거나 계속 이어가세요.{END}\n")

    dump_state(s)
    if s.items:
        show_result(s, args.json_out)


if __name__ == "__main__":
    main()
