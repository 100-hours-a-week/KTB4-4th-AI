import asyncio
import fcntl
import signal

from admin_api.provenance import source_hash
from admin_api.store import Store
from benchmarks.needu import execute_needu
from benchmarks.runner import execute


async def work(store, stop):
    store.recover()
    worker_source = source_hash()

    async def heartbeat():
        while not stop.is_set():
            store.beat()
            await asyncio.sleep(2)

    beat = asyncio.create_task(heartbeat())
    try:
        while not stop.is_set():
            run = store.claim()
            if run:
                if run["manifest"].get("source_hash") != worker_source:
                    store.finish(
                        run["id"],
                        "failed",
                        {},
                        "관리 API와 워커 코드 버전이 다릅니다. 두 프로세스를 다시 시작하세요.",
                    )
                    continue
                print(f"Running {run['id']} ({run['spec']['mode']})", flush=True)
                try:
                    if run["spec"]["mode"] == "needu":
                        await execute_needu(store, run, stop)
                    else:
                        await execute(store, run, stop)
                except Exception as error:
                    # Avoid logging prompts, headers, provider messages or keys.
                    store.finish(run["id"], "failed", {}, f"워커 처리 오류: {type(error).__name__}")
            else:
                await asyncio.sleep(0.4)
    finally:
        beat.cancel()
        await asyncio.gather(beat, return_exceptions=True)


async def main():
    store = Store()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    with store.path.with_suffix(".worker.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("AI Lab worker가 이미 실행 중입니다.") from None
        print("AI Lab worker ready", flush=True)
        await work(store, stop)


if __name__ == "__main__":
    asyncio.run(main())
