# auto-sre performance baselines

Committed regression baselines for the `autosre perf` harness. These
files are the source of truth for "is vLLM serving both Claude Code and
meeting-scribe translation within the performance envelope we shipped?"

## Layout

```
benchmarks/
├── baselines/
│   ├── <name>.json   # machine-readable PhaseResult + tolerances
│   └── <name>.md     # human-readable summary (reviewed in PRs)
└── README.md         # this file
```

## Workflow

### Establish a new baseline (one-time per config)

```bash
# 1. Make sure the vLLM + proxy you want to freeze are running.
autosre start

# 2. Run the harness and eyeball the output.
autosre perf run --duration 60 --no-compare

# 3. If the numbers look right, promote the run to a committed baseline.
autosre perf save-baseline gb10_qwen36_fp8

# 4. Commit both the .json and .md together.
git add repos/auto-sre/benchmarks/baselines/gb10_qwen36_fp8.*
git commit -m "perf: add gb10 qwen3.6 fp8 baseline"
```

### Validate a change (every time)

```bash
# After editing vllm flags, proxy code, or the priority-preempt hook
autosre stop && autosre start       # pick up changes
autosre perf run                    # exit 0 → ship; exit 2 → regression
```

Exit codes:

| code | meaning                                                        |
|------|----------------------------------------------------------------|
| 0    | clean — all metrics within tolerance of the committed baseline |
| 1    | warn-only — inspect the violations, may still be OK            |
| 2    | one or more **fail** violations — do not merge                 |

### Intentional baseline update

If a change is *supposed* to move the baseline (model swap, hardware
bump, deliberate quality-for-throughput tradeoff):

```bash
autosre perf run --duration 60 --no-compare
autosre perf save-baseline gb10_qwen36_fp8          # overwrites in place
# OR, to keep the old baseline for comparison:
autosre perf save-baseline gb10_qwen36_fp8_v2
```

Bundle the new baseline in the same PR as the change so reviewers can
see both. The `.md` file is the thing humans review.

## Pre-benchmark checklist

The harness refuses to run when meeting-scribe containers or non-autosre
GPU processes are detected. Follow this every time:

```bash
# 1. Stop the systemd supervisor (it resurrects compose-down'd containers)
systemctl --user stop meeting-scribe.service

# 2. Stop the compose stack + autoheal
meeting-scribe gb10 down

# 3. Verify clean GPU (expect 1 process: VLLM::EngineCore)
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# 4. Run the harness
autosre perf run

# 5. Restore after benchmarking
systemctl --user start meeting-scribe.service
```

If you need to run with scribe up (e.g. to measure coexistence), pass
`--allow-contention`. Numbers will not be comparable to the committed
baseline.

## Current baselines

| Baseline | Config | Translation TPS p50 | Coding TPS p50 | Coding TTFT p50 |
|----------|--------|--------------------:|---------------:|----------------:|
| `gb10_qwen36_fp8` | enforce-eager, seqs=12, gpu-mem=0.50 | 44 tok/s | 37 tok/s | 270 ms |
| `gb10_qwen36_fp8_tuned` | CUDA graphs, seqs=8, gpu-mem=0.75 | **79 tok/s** | **54 tok/s** | **181 ms** |

Default comparison target: `gb10_qwen36_fp8_tuned` (set via `--baseline`).

## Naming convention

`<hardware>_<model>_<quant>[_variant].json`

Examples:

- `gb10_qwen36_fp8.json` — pre-tune reference (fork-point)
- `gb10_qwen36_fp8_tuned.json` — current production
- `gb10_qwen35_fp8.json` — if we swap quantization
- `gb10_qwen3_coder_30b_fp8.json` — if we swap models

Pick names that survive contact with future model families.

## Tolerances

The tolerance block inside each `.json` is editable — ship a looser or
tighter tolerance alongside the change that justifies it, in the same
commit. Defaults live in `autosre/perf/baseline.py::_DEFAULT_TOLERANCES`
and cover both ratio checks (vs baseline) and absolute SLO checks (e.g.
translation TTFT p95 under contention must not be >2x its isolated
measurement, independent of the baseline numbers).

Phases with fewer than 20 successful samples emit a "warn" instead of
ratio-based checks — percentile estimates are too noisy at low sample
counts. Increase `--duration` to 120s for baseline captures if coding
consistently produces fewer than 20 samples.

## Harness details

- **Warmup phase** (15s default): runs both workloads concurrently to
  absorb CUDA graph capture and KV cache cold-start. Not measured.
- **Isolated phase**: each workload runs alone (translation at
  concurrency=1, coding at concurrency=2). No contention.
- **Contention phase**: both workloads run simultaneously. Translation
  is paced at `translation_rps` (default 2.0 req/s). Coding is unpaced.
- **Saturation phase** (optional, `--saturate-slots`): high coding
  concurrency (default 16) that fills `max_num_seqs` and forces
  priority preemption. Used to validate the `vllm_priority_preempt.pth`
  monkey-patch.
- **Proxy sanity**: hits `:8011/v1/messages` end-to-end to validate
  the Anthropic proxy streaming path.

## Additional harness commands

### `autosre perf boot` — cold-boot benchmark

Measures wall-clock time from service stop to all backends healthy.
Isolates the run from the compose watchdog timer by saving/restoring
its exact `is-enabled` + `is-active` state. Polls completion using
the systemd `InvocationID` and the same health checks as production
preflight.

```bash
autosre perf boot --no-compare                    # measure only
autosre perf boot --save-baseline gb10_current    # save as baseline
autosre perf boot --baseline gb10_current         # compare against
```

Boot baselines are stored as `boot_<name>.json` with `"kind": "boot"`
to prevent cross-type comparison with vLLM baselines.

### `autosre perf smoke` — end-to-end backend validation

Sends one request to each backend and checks protocol-level response:

| Backend | Port | Endpoint | Pass criteria |
|---------|------|----------|---------------|
| ASR | 8003 | `/v1/chat/completions` | HTTP 200, valid choices structure |
| TTS | 8002 | `/v1/audio/speech` | HTTP 200, audio bytes > 1000 |
| Diarization | 8001 | `/v1/diarize` | HTTP 200, JSON with segments + num_speakers |
| Translation | 8010 | `/v1/chat/completions` | HTTP 200, non-empty content |

```bash
autosre perf smoke   # exit 0 = all pass, exit 2 = any failure
```

## Full stack configuration reference

### vLLM (autosre-vllm-local, port 8010)

Recipe: `autosre/backends/recipes/qwen3.6-35b-a3b-fp8.yaml` (production-locked 2026-04-25; replaced the Qwen3.5-INT4-AutoRound + custom albond `vllm-qwen35-v2` fork stack).

| Setting | Value | Why |
|---------|-------|-----|
| `model` | `Qwen/Qwen3.6-35B-A3B-FP8` | 35B MoE, 3B active params. Native Qwen-published FP8 (~35 GB) — no calibration. |
| `docker_image` | `vllm/vllm-openai:latest` | Stock upstream vLLM. |
| `--tensor-parallel-size=1` | 1 GPU | Single B200 on GB10. No TP overhead. |
| `--max-model-len=262144` | 256K context | Qwen3.6 context window. |
| `--gpu-memory-utilization=0.70` | ~90 GB | 128 GB unified. Leaves ~17 GB for CPU/OS when scribe coexists. |
| `--max-num-seqs=8` | 8 slots | Reverted from 12: splitting KV blocks across 12 slots caused 2.5x TTFT regression with chunked-prefill. |
| `--max-num-batched-tokens=4096` | Chunked prefill cap | Drops throughput ~5%, halves tail latency for concurrent decoders. |
| `--kv-cache-dtype=auto` | FP16 | FP8 KV unstable on SM121 (vllm#26646). |
| `--quantization=fp8` | native FP8 | Qwen-published FP8 weights. |
| `--attention-backend=flashinfer` | on | Required pairing with FP8 on SM121 per NVIDIA forum thread 366822. |
| `--enable-prefix-caching` | on | ~84% hit rate on coding system prompt. Removing it → 8.6x TTFT regression. |
| `--enable-chunked-prefill` | on | Required for `max_num_batched_tokens`. Interleaves prefill with decode. |
| `--load-format=fastsafetensors` | on | 19x faster model loading on GB10. |
| `--scheduling-policy=priority` | on | Translation (priority=-10) preempts coding (priority=10). Requires monkey-patch. |
| `--enable-auto-tool-choice` | on | Auto tool-call detection for Claude Code. |
| `--tool-call-parser=qwen3_coder` | Qwen3 XML format | Parses `<tool_call>` tags. |
| `--reasoning-parser=qwen3` | on | Extracts `<think>` blocks into `reasoning_content`. |
| No `--enforce-eager` | CUDA graphs on | +80% translation TPS, +46% coding TPS vs eager. |

**Environment variables:**

| Variable | Value | Why |
|----------|-------|-----|
| `HF_HUB_OFFLINE=1` | on | Prevents DNS-dependency crashloop at boot. |
| `TRANSFORMERS_OFFLINE=1` | on | Belt-and-suspenders offline. |
| `NVIDIA_DISABLE_REQUIRE=1` | on | Bypasses CUDA version gate for custom image. |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | on | Allows 131072 despite model config mismatch. |
| `VLLM_MARLIN_USE_ATOMIC_ADD=1` | on | Required for SM121 Blackwell Marlin kernel correctness. |

### Compose services (docker-compose.gb10.yml)

| Service | Container | Port | GPU alloc | Key settings |
|---------|-----------|------|-----------|-------------|
| `pyannote-diarize` | `scribe-diarization` | 8001 | ~4 GB | `asyncio.Lock()` serialized (Sortformer CUDA safety). `DIARIZE_MAX_SPEAKERS=4`. |
| `qwen3-tts` | `scribe-tts` | 8002 | ~1.2 GB | Single-sequence autoregressive. `autoheal=false` (GIL blocks /health during synthesis). |
| `qwen3-tts-2` | `scribe-tts-2` | 8012 | ~1.2 GB | Second TTS replica for round-robin. Same config as above. |
| `vllm-asr` | `scribe-asr` | 8003 | ~13 GB (10% util) | `--max-model-len=4096`, `--max-num-seqs=4`, `--enforce-eager`, `--load-format=fastsafetensors`. |
| `autoheal` | `scribe-autoheal` | — | — | `AUTOHEAL_CONTAINER_LABEL=all`, 15s interval, 300s start period. |

**All GPU services share:** `shm_size: 16g`, `ulimits.memlock: -1`, `network_mode: host`, `restart: unless-stopped`.

**All services set:** `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (offline boot safety).

**GPU services (except ASR) set:** `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,expandable_segments:False` (fragmentation prevention on SM121).

### systemd (meeting-scribe.service)

| Directive | Value | Why |
|-----------|-------|-----|
| `Type=notify` | — | Server sends `sd_notify(READY=1)` when all backends initialized. |
| `ExecCondition` | `preflight --mode=precondition` | Filesystem-only gates. Exit 64/65 = non-retriable. |
| `ExecStartPre/1` | `gb10 up --offline` | Compose stack up with `--pull never`. |
| `ExecStartPre/2` | `preflight --mode=boot --wait 720` | 720s budget for cold-boot vLLM (~7 min worst case). |
| `ExecStart` | `start --foreground` | `os.execvpe()` so server IS MainPID for sd_notify. |
| `TimeoutStartSec=1200` | 20 min | Outer safety net; preflight's 720s is the real failure detector. |
| `Restart=on-failure` | — | Auto-restart on crash, not on clean stop. |
| `RestartPreventExitStatus=64 65` | — | Don't retry structural failures (missing venv, corrupt config). |
| `StartLimitBurst=3 / IntervalSec=600` | — | Max 3 starts per 10 min. |

### Boot sequence timing (typical)

| Phase | Duration | Notes |
|-------|----------|-------|
| Compose up | ~0.4s | Idempotent with cached images. |
| Preflight Phase 1 (Docker checks) | ~1s | docker info + compose config + image inspect. |
| Preflight Phase 2 (backend health) | ~50s | Concurrent polling; translation (vLLM cold-load) is the long pole. |
| Server startup + sd_notify | ~2s | FastAPI lifespan + asyncio.gather of backend clients. |
| **Total** | **~54s** | Baseline: `boot_gb10_current` |
