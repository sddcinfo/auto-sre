# baseline: gb10_qwen36_fp8_flashinfer_8192 (20260418T215120)

## Environment

- **hostname**: `promaxgb10-f426`
- **platform**: `Linux-6.17.0-1014-nvidia-aarch64-with-glibc2.39`
- **python**: `3.14.4`
- **model_id**: `Qwen/Qwen3.6-35B-A3B-FP8`
- **vllm_image**: `sha256:1c135f7bef18f74535c9d24a4ae6612bd1c2d563529d52d4247e41e6e252ee5e`
- **vllm_args**: `serve Qwen/Qwen3.6-35B-A3B-FP8 --tensor-parallel-size=1 --max-model-len=262144 --gpu-memory-utilization=0.7 --kv-cache-dtype=auto --host=0.0.0.0 --max-num-seqs=8 --max-num-batched-tokens=8192 --quantization=fp8 --chat-template=/home/bradlay/.local/share/autosre/qwen35-chat-template.jinja --enable-auto-tool-choice --tool-call-parser=qwen3_coder --reasoning-parser=qwen3 --enable-prefix-caching --enable-chunked-prefill --load-format=fastsafetensors --scheduling-policy=priority --attention-backend=flashinfer --port=8010`
- **autosre_sha**: `a687ba4`
- **gpu**: `NVIDIA GB10`
- **recipe_hashes**: `{'autosre/backends/recipes/qwen3-coder-30b-a3b.yaml': '6186f40bbfd82a652127dcee5e1a2bfe9ea0e26ec4c0491d7e51f7be939a8b39', 'autosre/backends/recipes/qwen3.5-35b-a3b-int4.yaml': '32699e61c8cf7436475321fa9ffad7772bbf94e9d6b8ff994fe9a6fdfa0147b1', 'autosre/backends/recipes/qwen3.5-35b-a3b.yaml': '177ef0b2cbd7e7aab2746d91b11d756f9de94e3f6acf39da41a1617974226dcf', 'autosre/backends/recipes/qwen3.6-35b-a3b-fp8.yaml': '044ef46a71ca1e802f1890efbcdf337d1dcbb5a6f72e204d3bea2a89f33f210c', 'autosre/backends/recipes/qwen3.6-35b-a3b-int4.yaml': '34aa004f6517a42c58fc78295d303c26c9a75c4f8ac4f8ff5664a7bc4f4d57b6', 'autosre/backends/recipes/qwen3.6-fp8-nightly.yaml': '98a64ea5e2004de4e64d79c2d49583caaaa9e55a774beee0daabfa9b8021e2bc'}`

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
| translation | isolated | 184 | 115 | 119 | 122 | 51.66 | 33.10 | 0 | 60.2 |
| coding | isolated | 24 | 269 | 286 | 307 | 41.41 | 71.44 | 0 | 73.8 |
| translation | contention | 119 | 157 | 164 | 191 | 31.73 | 16.94 | 0 | 76.2 |
| coding | contention | 16 | 371 | 401 | 404 | 27.95 | 49.79 | 0 | 76.2 |

## Scheduler counters (during contention)

- `preemptions_delta`: **0**
- `requests_running_peak`: 3.0
- `requests_waiting_peak`: 0.0
- `kv_cache_pct_peak`: 0.008
- `queue_time_avg_ms`: 0.0
- `prefix_cache_hit_pct`: 69.4

