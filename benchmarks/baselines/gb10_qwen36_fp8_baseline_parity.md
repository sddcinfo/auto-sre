# baseline: gb10_qwen36_fp8_baseline_parity (20260418T203821)

## Environment

- **hostname**: `promaxgb10-f426`
- **platform**: `Linux-6.17.0-1014-nvidia-aarch64-with-glibc2.39`
- **python**: `3.14.4`
- **model_id**: `Qwen/Qwen3.6-35B-A3B-FP8`
- **vllm_image**: `sha256:1c135f7bef18f74535c9d24a4ae6612bd1c2d563529d52d4247e41e6e252ee5e`
- **vllm_args**: `serve Qwen/Qwen3.6-35B-A3B-FP8 --tensor-parallel-size=1 --max-model-len=262144 --gpu-memory-utilization=0.7 --kv-cache-dtype=auto --host=0.0.0.0 --max-num-seqs=8 --max-num-batched-tokens=4096 --quantization=fp8 --chat-template=/home/bradlay/.local/share/autosre/qwen35-chat-template.jinja --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 --enable-prefix-caching --enable-chunked-prefill --load-format=fastsafetensors --scheduling-policy=priority --port=8010`
- **autosre_sha**: `bed4e95`
- **gpu**: `NVIDIA GB10`
- **recipe_hashes**: `{}`

## Config

```json
  "duration_seconds": 60,
  "warmup_seconds": 15,
  "translation_concurrency": 1,
  "translation_rps": 2.0,
  "coding_concurrency": 2,
  "saturate_slots": False,
  "saturate_concurrency": 16,
  "run_proxy_check": False,
  "vllm_url": 'http://localhost:8010',
  "proxy_url": 'http://localhost:8011',
  "model_id": 'Qwen/Qwen3.6-35B-A3B-FP8',
```

## Per-workload results

| Workload | Phase | Samples | TTFT p50 (ms) | TTFT p95 (ms) | TTFT p99 (ms) | TPS p50 | TPS agg | Errors | Wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| translation | isolated | 185 | 114 | 119 | 120 | 51.73 | 33.32 | 0 | 60.0 |
| coding | isolated | 24 | 272 | 597 | 751 | 41.02 | 74.97 | 0 | 77.6 |
| translation | contention | 119 | 158 | 163 | 189 | 31.85 | 20.00 | 0 | 65.0 |
| coding | contention | 17 | 351 | 400 | 401 | 27.83 | 50.89 | 0 | 65.0 |

## Scheduler counters (during contention)

- `preemptions_delta`: **0**
- `requests_running_peak`: 3.0
- `requests_waiting_peak`: 0.0
- `kv_cache_pct_peak`: 0.007
- `queue_time_avg_ms`: 0.0
- `prefix_cache_hit_pct`: 72.2

