# 모니터링 (Prometheus + Grafana)

RunPod Pod에서 도는 vLLM의 `/metrics`를 로컬 Prometheus가 5초마다 긁고, Grafana 대시보드로 본다. 부하 테스트와 vLLM 설정 비교 실험에 쓴다.

```text
RunPod vLLM /metrics  ←(5초 scrape, HTTPS)─  Prometheus :9090  ←  Grafana :3000
```

## 처음 실행

```bash
# vLLM API 키 파일. 저장소 루트 .env의 MODEL_API_KEY 값을 쓴다. git에는 올라가지 않는다.
mkdir -p monitoring/prometheus/secrets
grep "^MODEL_API_KEY=" .env | cut -d= -f2- | tr -d '"' | tr -d '\n' > monitoring/prometheus/secrets/vllm_api_key

# 수집 대상 파일. 실제 Pod 주소가 들어가므로 git에서 빠져 있고, 예시에서 만든다.
python3 monitoring/sync_target.py

docker compose -f monitoring/docker-compose.yml up -d
```

`sync_target.py`는 RUNNING Pod가 없으면 파일을 만들지 않는다. Pod가 꺼져 있을 때는 `cp monitoring/prometheus/targets/vllm.example.json monitoring/prometheus/targets/vllm.json`으로 만들어 두고, Pod를 켠 뒤 다시 실행한다.

- Grafana: http://localhost:3000 (admin / admin, 비밀번호는 `GRAFANA_ADMIN_PASSWORD` 환경 변수로 바꿀 수 있다)
- 대시보드: Need U 폴더의 "Need U · vLLM 서빙"
- Prometheus 수집 상태: http://localhost:9090/targets

두 포트 모두 127.0.0.1에만 열려 있다.

## Pod 주소가 바뀌었을 때

`python3 monitoring/sync_target.py`를 실행하면 AI 서버와 같은 기준(이름 접두어, RUNNING, 가장 최근 생성)으로 Pod를 찾아 대상 파일을 고친다. `--watch`를 붙이면 30초마다 확인한다. 직접 고칠 때는 `prometheus/targets/vllm.json`의 `targets`와 `pod`만 바꾸면 된다. 이 파일은 git에서 빠져 있고, 형식은 [vllm.example.json](prometheus/targets/vllm.example.json)과 같다. Prometheus가 15초 안에 다시 읽으므로 재시작은 필요 없다. 주소 형식은 `<pod-id>-8000.proxy.runpod.net`이고 `https://`와 `/v1`은 붙이지 않는다.

Pod가 꺼져 있으면 RunPod 프록시가 404를 돌려줘서 대시보드의 "수집 상태"가 0으로 나온다.

## 실험 구분

같은 파일의 `run` 라벨에 실험 이름을 넣는다. 부하 테스트를 시작하기 전에 바꿔 두면 그 뒤로 쌓이는 지표에 이 이름이 붙고, 대시보드 상단의 "실험" 선택으로 실험끼리 겹쳐 볼 수 있다.

```json
"labels": { "pod": "<pod-id>", "run": "fp8-kv" }
```

이름은 바꾼 설정을 알 수 있게 짓는다(`baseline`, `maxlen-8k`, `fp8-kv`, `seqs-32`). 어떤 실행 옵션으로 띄웠는지는 실험마다 따로 기록해 둔다.

## Pod를 처음 켰을 때 확인할 것

vLLM 버전마다 지표 이름이 조금씩 다르다. 대시보드는 v1 엔진 기준 이름(`vllm:kv_cache_usage_perc`, `vllm:inter_token_latency_seconds` 등)으로 만들었다. Pod를 켠 뒤 아래 명령으로 실제 이름을 보고, 비어 있는 패널이 있으면 이름을 맞춘다.

```bash
curl -s -H "Authorization: Bearer $(cat monitoring/prometheus/secrets/vllm_api_key)" \
  https://<pod-id>-8000.proxy.runpod.net/metrics | grep -E "^# HELP vllm:" | cut -d' ' -f3
```

RunPod 프록시는 일부 User-Agent를 403으로 막는다. 수집 상태에 403이 나오면 Prometheus의 UA가 막힌 것이다.

## 대시보드에서 보는 것

| 패널 | 볼 점 |
| --- | --- |
| 대기 요청 | 0보다 커지기 시작하는 동시 사용자 수가 처리 한계 |
| KV 캐시 사용률, 선점 | 사용률이 1에 붙고 선점이 생기면 KV 용량이 병목 |
| TTFT, 대기열 시간 | 대기열 시간이 TTFT 대부분을 차지하면 배치가 꽉 찬 것 |
| 토큰 사이 간격 | 동시 요청이 늘수록 스트리밍 속도가 얼마나 떨어지는지 |
| 토큰 처리량 | 설정을 바꿨을 때 GPU가 실제로 더 많이 처리하는지 |
| prefix 캐시 적중률 | system 프롬프트 재사용이 되는지 |

GPU 사용률과 메모리는 vLLM 지표에 없다. 필요하면 Pod 안에서 `nvidia_gpu_exporter`를 띄우고 포트를 노출한 뒤 scrape 대상을 하나 더 추가한다.

## 데이터

지표는 Docker 볼륨 `monitoring_prometheus-data`에 180일 보관된다. `docker compose down`으로는 지워지지 않고 `down -v`를 하면 지워진다. 대시보드를 화면에서 고쳤으면 JSON으로 내보내 [grafana/dashboards/vllm.json](grafana/dashboards/vllm.json)에 덮어써야 저장소에 남는다.
