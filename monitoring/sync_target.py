"""RunPod에서 이름 접두어가 맞는 최신 RUNNING Pod를 찾아 Prometheus 대상 파일을 고친다.

AI 서버의 RunPodPodLocator와 같은 기준(접두어 + RUNNING + 가장 최근 createdAt)으로 고른다.
대상 파일의 run 라벨은 그대로 둔다. Prometheus가 파일을 15초마다 다시 읽으므로 재시작은 없다.

    python3 monitoring/sync_target.py           # 한 번 고치고 끝
    python3 monitoring/sync_target.py --watch   # 30초마다 확인
"""

import argparse
import json
import os
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = Path(__file__).resolve().parent / "prometheus" / "targets" / "vllm.json"
EXAMPLE = TARGETS.with_name("vllm.example.json")
PODS_URL = "https://api.runpod.io/v2/pods"


def read_env() -> dict[str, str]:
    values: dict[str, str] = {}
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({k: v for k, v in os.environ.items() if k.startswith("RUNPOD_")})
    return values


def latest_running_pod(api_key: str, prefix: str) -> dict | None:
    request = urllib.request.Request(
        PODS_URL,
        # RunPod 쪽 Cloudflare가 Python-urllib UA를 403으로 막는다.
        headers={"Authorization": f"Bearer {api_key}", "User-Agent": "curl/8.7.1"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        pods = json.load(response).get("pods", [])
    running = [
        pod
        for pod in pods
        if str(pod.get("name", "")).startswith(prefix) and pod.get("status") == "RUNNING"
    ]
    return max(running, key=lambda pod: str(pod.get("createdAt", "")), default=None)


def sync(api_key: str, prefix: str, port: int) -> None:
    # 대상 파일은 실제 Pod 주소가 담겨 git에서 빠진다. 처음에는 예시 파일에서 시작한다.
    current = json.loads((TARGETS if TARGETS.exists() else EXAMPLE).read_text())
    labels = current[0].get("labels", {}) if current else {}
    pod = latest_running_pod(api_key, prefix)
    if pod is None:
        print(f"RUNNING Pod 없음 (prefix={prefix}). 대상 파일은 그대로 둔다.")
        return
    if labels.get("pod") == pod["id"]:
        return
    labels = {**labels, "pod": pod["id"]}
    labels.setdefault("run", "baseline")
    target = [{"targets": [f"{pod['id']}-{port}.proxy.runpod.net"], "labels": labels}]
    temp = TARGETS.with_suffix(".tmp")
    temp.write_text(json.dumps(target, ensure_ascii=False, indent=2) + "\n")
    temp.replace(TARGETS)
    print(f"대상 변경: {pod['id']} ({pod.get('name')}), run={labels['run']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", action="store_true", help="30초마다 다시 확인한다")
    args = parser.parse_args()

    env = read_env()
    api_key = env.get("RUNPOD_API_KEY")
    if not api_key:
        raise SystemExit("RUNPOD_API_KEY가 .env에 없다")
    prefix = env.get("RUNPOD_POD_NAME_PREFIX", "needu-kanana-vllm")
    port = int(env.get("RUNPOD_POD_PORT", "8000"))

    while True:
        try:
            sync(api_key, prefix, port)
        except OSError as error:
            print(f"RunPod 조회 실패: {error}")
        if not args.watch:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
