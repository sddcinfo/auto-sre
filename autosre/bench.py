"""Model benchmarking for autosre.

Benchmarks vLLM models on the local GPU:
  - Single-request throughput (TTFT + decode TPS)
  - Concurrent throughput (aggregate TPS at N parallel)
  - Tool calling validation
  - Memory usage

Results are saved to ~/.local/share/autosre/benchmarks/.

KNOWN GAP (see ../../../UPGRADE-NOTES-2026-04.md): `autosre bench` assumes
exclusive vLLM access. If the shared vLLM on port 8010 is already serving
a different model (e.g. the coding agent), `bench -m <other_model>`
reports "Ready in 0s" without actually swapping the model and then
records 0 tok/s + "Tools: fail". Either detect the already-loaded model
and skip the swap, or fall back to benchmarking the live model.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import httpx

DOCKER_IMAGE = "ghcr.io/bjk110/vllm-spark:turboquant"
HF_CACHE = "/data/huggingface"
CONTAINER_NAME = "autosre-bench"


@dataclass
class ModelSpec:
    """A model to benchmark."""

    name: str
    model_id: str
    vllm_args: list[str]
    weight_size_gb: float = 0.0


@dataclass
class BenchResult:
    """Benchmark results for a single model."""

    name: str
    model_id: str
    startup_seconds: float = 0.0
    gpu_memory_mb: int = 0
    swap_used: str = ""
    single_tps: float = 0.0
    single_ttft_ms: float = 0.0
    single_tokens: int = 0
    concurrent_agg_tps: float = 0.0
    concurrent_avg_tps: float = 0.0
    concurrent_n: int = 3
    tool_calling: str = "untested"
    error: str = ""


# Common vLLM args shared across all models
_COMMON_ARGS = [
    "--enforce-eager",
    "--enable-auto-tool-choice",
    "--tool-call-parser=qwen3_coder",
    "--enable-prefix-caching",
    "--max-num-seqs=8",
]

_NVFP4_ARGS = [*_COMMON_ARGS, "--load-format=fastsafetensors", "--kv-cache-dtype=turboquant"]
_FP8_ARGS = [*_COMMON_ARGS, "--load-format=safetensors", "--kv-cache-dtype=turboquant"]


def _model(
    name: str,
    model_id: str,
    weight_gb: float,
    args: list[str],
    *,
    gpu_util: float = 0.85,
    max_model_len: int = 131072,
    extra_args: list[str] | None = None,
) -> ModelSpec:
    return ModelSpec(
        name=name,
        model_id=model_id,
        weight_size_gb=weight_gb,
        vllm_args=[
            *args,
            f"--gpu-memory-utilization={gpu_util}",
            f"--max-model-len={max_model_len}",
            *(extra_args or []),
        ],
    )


# Models to benchmark, ordered smallest to largest weights.
#
# Generation context (memory bandwidth, NOT compute, is the GB10 ceiling):
#   - LPDDR5x bandwidth = 273 GB/s → ceiling ~38 tok/s for 35-B FP8
#   - NVFP4 on SM121 was slow under the FP4 CUTLASS kernels we tried
#     (Phase 6.B 2026-04-25 close-out). Marlin fallback works but loses
#     ~8 BLEU. Flagged here as "expected slower" — re-bench when
#     Blackwell-native FP4 kernels stabilize.
#
# Qwen3.5 / Intel AutoRound entries were retired 2026-04-29 alongside
# the rest of the 3.5 stack. Add a 3.6 equivalent (with a proper
# baseline run) when one becomes load-bearing again.
MODELS: list[ModelSpec] = [
    # --- Small models (fit alongside scribe services) ---
    _model(
        "Nemotron-Nano-30B NVFP4",
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
        19.3,
        _NVFP4_ARGS,
        extra_args=["--reasoning-parser=nano_v3", "--moe-backend=cutlass"],
    ),
    _model(
        "Qwen3.6-35B-A3B NVFP4 (RedHat)",
        "RedHatAI/Qwen3.6-35B-A3B-NVFP4",
        25.0,
        _NVFP4_ARGS,
    ),
    # --- Production translation + coding agent ---
    _model(
        "Qwen3.6-35B-A3B FP8 (production)",
        "Qwen/Qwen3.6-35B-A3B-FP8",
        35.0,
        _FP8_ARGS,
    ),
]


def _stop_container() -> None:
    subprocess.run(
        ["docker", "rm", "-f", CONTAINER_NAME],
        capture_output=True,
        check=False,
    )
    time.sleep(2)


def _start_container(spec: ModelSpec, port: int = 8010) -> bool:
    """Start a vLLM container for the given model. Returns True if started."""
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "--gpus",
        "all",
        "--network",
        "host",
        "--shm-size",
        "16g",
        "--ulimit",
        "memlock=-1",
        "-v",
        f"{HF_CACHE}:/root/.cache/huggingface",
        "-e",
        "NVIDIA_DISABLE_REQUIRE=1",
        "-e",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
    ]

    # Pass HF_TOKEN for authenticated model downloads
    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token:
        cmd.extend(["-e", f"HF_TOKEN={hf_token}"])

    cmd.extend(
        [
            DOCKER_IMAGE,
            "vllm",
            "serve",
            spec.model_id,
            "--host=0.0.0.0",
            f"--port={port}",
            "--tensor-parallel-size=1",
            *spec.vllm_args,
        ]
    )
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    return result.returncode == 0


def _wait_for_health(port: int = 8010, timeout: int = 600) -> bool:
    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = httpx.get(url, timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def _get_gpu_memory_mb() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=False,
        )
        return sum(int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip())
    except Exception:
        return 0


def _get_swap_used() -> str:
    try:
        result = subprocess.run(
            ["swapon", "--show=USED", "--noheadings"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() or "0"
    except Exception:
        return "?"


def _clear_swap() -> None:
    subprocess.run(["sudo", "swapoff", "-a"], capture_output=True, check=False)
    subprocess.run(["sudo", "swapon", "-a"], capture_output=True, check=False)


def _stream_request(
    url: str,
    model_id: str,
    prompt: str,
    max_tokens: int = 150,
) -> tuple[float, int, float, float]:
    """Returns (ttft_ms, token_count, gen_time_s, tps)."""
    payload = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": 0.6,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        # Benchmarks are background work — live translation (-10) preempts.
        "priority": 10,
    }
    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0

    with httpx.Client(timeout=120) as client, client.stream("POST", url, json=payload) as resp:
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1

    t1 = time.perf_counter()
    ttft = (first_token_time - t0) * 1000 if first_token_time else 0
    gen_time = t1 - first_token_time if first_token_time else t1 - t0
    tps = token_count / gen_time if gen_time > 0 else 0
    return ttft, token_count, gen_time, tps


def _test_tool_calling(url: str, model_id: str) -> str:
    """Test if the model generates proper tool calls. Returns 'pass' or 'fail'."""
    payload: dict[str, Any] = {
        "model": model_id,
        "max_tokens": 200,
        "messages": [{"role": "user", "content": "Read the file /etc/hostname"}],
        # Bench runs are background work — live meeting-scribe translation
        # (priority -10) must always preempt.
        "priority": 10,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "Read",
                    "description": "Read a file from disk",
                    "parameters": {
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"],
                    },
                },
            }
        ],
    }
    try:
        resp = httpx.post(url, json=payload, timeout=60)
        d = resp.json()
        tool_calls = d.get("choices", [{}])[0].get("message", {}).get("tool_calls", [])
        has_read = any(tc.get("function", {}).get("name") == "Read" for tc in tool_calls)
        return "pass" if has_read else "fail"
    except Exception as e:
        return f"error: {e}"


def run_single_benchmark(
    spec: ModelSpec,
    port: int = 8010,
    concurrent_n: int = 3,
) -> BenchResult:
    """Benchmark a single model end-to-end."""
    result = BenchResult(name=spec.name, model_id=spec.model_id, concurrent_n=concurrent_n)
    url = f"http://localhost:{port}/v1/chat/completions"

    click.echo(f"\n  Starting {spec.name}...")

    _stop_container()
    _clear_swap()

    t_start = time.time()
    if not _start_container(spec, port):
        result.error = "Docker launch failed"
        return result

    if not _wait_for_health(port, timeout=600):
        result.error = "Health check timeout (10 min)"
        # Grab last few log lines for debugging
        logs = subprocess.run(
            ["docker", "logs", "--tail=5", CONTAINER_NAME],
            capture_output=True,
            text=True,
            check=False,
        )
        result.error += f"\n{logs.stderr[-200:]}" if logs.stderr else ""
        _stop_container()
        return result

    result.startup_seconds = time.time() - t_start
    result.gpu_memory_mb = _get_gpu_memory_mb()
    result.swap_used = _get_swap_used()

    click.echo(
        f"  Ready in {result.startup_seconds:.0f}s (GPU: {result.gpu_memory_mb // 1024}GB, swap: {result.swap_used})"
    )
    click.echo("  Benchmarking...")

    # Warmup
    import contextlib

    with contextlib.suppress(Exception):
        _stream_request(url, spec.model_id, "Say hello", max_tokens=10)

    # Single request
    try:
        ttft, tokens, _, tps = _stream_request(
            url,
            spec.model_id,
            "Write a Python quicksort function. Be concise.",
            max_tokens=200,
        )
        result.single_tps = tps
        result.single_ttft_ms = ttft
        result.single_tokens = tokens
    except Exception as e:
        result.error = f"Single request failed: {e}"

    # Concurrent requests
    import concurrent.futures

    prompts = [
        "Write a Python quicksort function.",
        "Write a Python binary search function.",
        "Write a Python merge sort function.",
        "Write a Python linked list class.",
    ][:concurrent_n]

    try:
        t_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_n) as pool:
            futs = [pool.submit(_stream_request, url, spec.model_id, p, 150) for p in prompts]
            concurrent_results = [f.result() for f in futs]
        t_total = time.perf_counter() - t_start

        total_tokens = sum(r[1] for r in concurrent_results)
        result.concurrent_avg_tps = sum(r[3] for r in concurrent_results) / len(concurrent_results)
        result.concurrent_agg_tps = total_tokens / t_total if t_total > 0 else 0
    except Exception as e:
        result.error = f"Concurrent test failed: {e}"

    # Tool calling
    try:
        result.tool_calling = _test_tool_calling(url, spec.model_id)
    except Exception:
        result.tool_calling = "error"

    _stop_container()
    return result


def print_result(r: BenchResult) -> None:
    """Print a single benchmark result."""
    if r.error:
        click.secho(f"  {r.name}: FAILED — {r.error}", fg="red")
        return

    click.echo(f"  {r.name}")
    click.echo(
        f"    Single:     {r.single_tps:.1f} tok/s  (TTFT {r.single_ttft_ms:.0f}ms, {r.single_tokens} tokens)"
    )
    click.echo(
        f"    {r.concurrent_n}x Conc:    {r.concurrent_agg_tps:.1f} tok/s aggregate, {r.concurrent_avg_tps:.1f} tok/s per-req"
    )
    click.echo(f"    Tools:      {r.tool_calling}")
    click.echo(f"    Memory:     {r.gpu_memory_mb // 1024}GB GPU, swap {r.swap_used}")
    click.echo(f"    Startup:    {r.startup_seconds:.0f}s")


def print_summary_table(results: list[BenchResult]) -> None:
    """Print a comparison table of all results."""
    click.echo()
    click.echo("  Model                         │ Single │ 3x Agg │ Tools │  GPU  │ Startup")
    click.echo("  ──────────────────────────────┼────────┼────────┼───────┼───────┼────────")
    for r in results:
        if r.error:
            click.echo(f"  {r.name:30s} │  FAIL  │  FAIL  │  ---  │  ---  │ ---")
        else:
            tools = "  ✓  " if r.tool_calling == "pass" else "  ✗  "
            click.echo(
                f"  {r.name:30s} │ {r.single_tps:5.1f}  │ {r.concurrent_agg_tps:5.1f}  │{tools}│ {r.gpu_memory_mb // 1024:3d}GB │ {r.startup_seconds:4.0f}s"
            )


def save_results(results: list[BenchResult]) -> Path:
    """Save results to JSON in the autosre data directory."""
    import os

    data_dir = (
        Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        / "autosre"
        / "benchmarks"
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = data_dir / f"bench-{timestamp}.json"

    data = {
        "timestamp": timestamp,
        "results": [
            {
                "name": r.name,
                "model_id": r.model_id,
                "single_tps": r.single_tps,
                "single_ttft_ms": r.single_ttft_ms,
                "concurrent_agg_tps": r.concurrent_agg_tps,
                "concurrent_avg_tps": r.concurrent_avg_tps,
                "concurrent_n": r.concurrent_n,
                "tool_calling": r.tool_calling,
                "gpu_memory_mb": r.gpu_memory_mb,
                "swap_used": r.swap_used,
                "startup_seconds": r.startup_seconds,
                "error": r.error,
            }
            for r in results
        ],
    }

    path.write_text(json.dumps(data, indent=2))
    return path
