"""Async concurrent-workload driver for the perf harness.

Drives :class:`~autosre.perf.workloads.Workload` instances against the
running vLLM on :8010, measures per-request TTFT and TPS, and records
vLLM scheduler counters from :func:`autosre.metrics.vllm_metrics`.

The harness runs four phases:

1. **Warmup** — unmeasured priming of KV/prefix cache.
2. **Isolated translation** — translation alone, no contention.
3. **Isolated coding** — coding alone, no contention.
4. **Contention** — both workloads concurrent for ``duration`` seconds.
5. **Saturation** (optional, ``--saturate-slots``) — high coding
   concurrency that deliberately fills ``max_num_seqs`` so the
   priority-preempt hook path is exercised.

All phases emit a :class:`PhaseResult`; the top-level :class:`RunResult`
bundles them with scheduler counter deltas, proxy sanity output, and an
environment fingerprint ready for comparison against a committed baseline.
"""

from __future__ import annotations

import asyncio
import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from autosre.metrics import vllm_metrics
from autosre.perf.workloads import CODING_WORKLOAD, TRANSLATION_WORKLOAD, Workload

VLLM_URL = "http://localhost:8010"
PROXY_URL = "http://localhost:8011"
MODEL_ID = "Qwen/Qwen3.6-35B-A3B-FP8"


# ── Result shapes ────────────────────────────────────────────────


@dataclass
class Sample:
    """One streamed request."""

    workload: str
    ttft_ms: float
    gen_ms: float
    output_tokens: int
    tps: float
    error: str | None = None


@dataclass
class PhaseResult:
    workload: str
    phase: str
    samples: int
    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tps_p50: float = 0.0
    tps_p95: float = 0.0
    tps_agg: float = 0.0
    errors: int = 0
    wall_seconds: float = 0.0


@dataclass
class SchedulerCounters:
    preemptions_delta: int = 0
    kv_cache_pct_peak: float = 0.0
    requests_running_peak: float = 0.0
    requests_waiting_peak: float = 0.0
    queue_time_avg_ms: float = 0.0
    prefix_cache_hit_pct: float = 0.0


@dataclass
class ProxySanity:
    ok: bool
    message_start: bool = False
    content_delta_count: int = 0
    message_stop: bool = False
    upstream_error: str | None = None
    elapsed_ms: float = 0.0
    idle_gap_seconds: float = 0.0
    note: str = ""


@dataclass
class RunConfig:
    duration_seconds: int = 60
    warmup_seconds: int = 15
    translation_concurrency: int = 1
    translation_rps: float = 2.0
    coding_concurrency: int = 2
    saturate_slots: bool = False
    saturate_concurrency: int = 16
    run_proxy_check: bool = True
    vllm_url: str = VLLM_URL
    proxy_url: str = PROXY_URL
    model_id: str = MODEL_ID


@dataclass
class RunResult:
    timestamp: str
    config: dict[str, Any]
    environment: dict[str, Any]
    phases: list[PhaseResult] = field(default_factory=list)
    scheduler: SchedulerCounters = field(default_factory=SchedulerCounters)
    proxy_sanity: ProxySanity | None = None

    def phase(self, workload: str, phase: str) -> PhaseResult | None:
        for p in self.phases:
            if p.workload == workload and p.phase == phase:
                return p
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "config": self.config,
            "environment": self.environment,
            "phases": [asdict(p) for p in self.phases],
            "scheduler": asdict(self.scheduler),
            "proxy_sanity": asdict(self.proxy_sanity) if self.proxy_sanity else None,
        }


# ── Percentile helper (copied + trimmed from scribe/perf_baseline.py) ─


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    frac = k - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def summarize(workload: str, phase: str, samples: list[Sample], wall: float) -> PhaseResult:
    ok = [s for s in samples if s.error is None and s.output_tokens > 0]
    ttfts = [s.ttft_ms for s in ok]
    tpss = [s.tps for s in ok]
    total_tokens = sum(s.output_tokens for s in ok)
    return PhaseResult(
        workload=workload,
        phase=phase,
        samples=len(samples),
        ttft_p50_ms=round(percentile(ttfts, 0.50), 1),
        ttft_p95_ms=round(percentile(ttfts, 0.95), 1),
        ttft_p99_ms=round(percentile(ttfts, 0.99), 1),
        tps_p50=round(percentile(tpss, 0.50), 2),
        tps_p95=round(percentile(tpss, 0.95), 2),
        tps_agg=round(total_tokens / wall, 2) if wall > 0 else 0.0,
        errors=sum(1 for s in samples if s.error is not None),
        wall_seconds=round(wall, 2),
    )


# ── Single request driver ────────────────────────────────────────


async def stream_one(
    client: httpx.AsyncClient,
    url: str,
    model_id: str,
    workload: Workload,
) -> Sample:
    """Drive one streaming chat completion and return a :class:`Sample`.

    Mirrors ``autosre/bench.py:_stream_request`` but async and labeled
    with the workload name so contention results can be bucketed.
    """
    payload, _label = workload.next_payload()
    payload = {"model": model_id, **payload}

    t0 = time.perf_counter()
    first_token_time: float | None = None
    output_tokens = 0
    err: str | None = None

    try:
        async with client.stream(
            "POST",
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=httpx.Timeout(300.0, connect=10.0),
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")
                return Sample(
                    workload=workload.name,
                    ttft_ms=0.0,
                    gen_ms=0.0,
                    output_tokens=0,
                    tps=0.0,
                    error=f"http_{resp.status_code}: {body[:160]}",
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choice = chunk.get("choices", [{}])[0] if chunk.get("choices") else {}
                delta = choice.get("delta", {})
                # Mark TTFT on the first visible output — text OR tool-call
                # delta. Without this, tool-calling responses (common for
                # the coding workload) would record ttft=0 and skew the
                # percentiles downward.
                if (delta.get("content") or delta.get("tool_calls")) and first_token_time is None:
                    first_token_time = time.perf_counter()
                if delta.get("content"):
                    output_tokens += 1
                usage = chunk.get("usage")
                if usage and usage.get("completion_tokens"):
                    output_tokens = usage["completion_tokens"]
    except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ReadTimeout) as exc:
        err = f"{type(exc).__name__}: {exc}"
    except httpx.HTTPError as exc:
        err = f"{type(exc).__name__}: {exc}"

    t1 = time.perf_counter()
    ttft_ms = (first_token_time - t0) * 1000.0 if first_token_time else 0.0
    gen_time = t1 - (first_token_time or t0)
    tps = output_tokens / gen_time if gen_time > 0 and output_tokens > 0 else 0.0
    return Sample(
        workload=workload.name,
        ttft_ms=round(ttft_ms, 1),
        gen_ms=round(gen_time * 1000.0, 1),
        output_tokens=output_tokens,
        tps=round(tps, 2),
        error=err,
    )


# ── Phase runners ────────────────────────────────────────────────


async def run_isolated(
    client: httpx.AsyncClient,
    cfg: RunConfig,
    workload: Workload,
    concurrency: int,
) -> PhaseResult:
    """Keep ``concurrency`` requests in flight for ``cfg.duration_seconds``."""
    samples: list[Sample] = []
    stop_at = time.perf_counter() + cfg.duration_seconds
    t_start = time.perf_counter()

    async def worker() -> None:
        while time.perf_counter() < stop_at:
            samples.append(await stream_one(client, cfg.vllm_url, cfg.model_id, workload))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    wall = time.perf_counter() - t_start
    return summarize(workload.name, "isolated", samples, wall)


async def run_contention(
    client: httpx.AsyncClient,
    cfg: RunConfig,
) -> tuple[PhaseResult, PhaseResult, SchedulerCounters]:
    """Run translation + coding concurrently, bucket samples per workload."""
    translation_samples: list[Sample] = []
    coding_samples: list[Sample] = []
    stop_at = time.perf_counter() + cfg.duration_seconds

    # Scheduler counter snapshots
    metrics_before = vllm_metrics(f"{cfg.vllm_url}/metrics")
    peaks: dict[str, float] = {"running": 0.0, "waiting": 0.0, "kv": 0.0}

    async def translation_worker() -> None:
        # Steady-paced pacer — one request every 1/rps seconds to
        # mimic meeting-scribe's live translation cadence rather than
        # back-to-back storming.
        interval = 1.0 / max(cfg.translation_rps, 0.1)
        next_fire = time.perf_counter()
        while time.perf_counter() < stop_at:
            sleep_for = next_fire - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            translation_samples.append(
                await stream_one(client, cfg.vllm_url, cfg.model_id, TRANSLATION_WORKLOAD)
            )
            next_fire += interval

    async def coding_worker() -> None:
        while time.perf_counter() < stop_at:
            coding_samples.append(
                await stream_one(client, cfg.vllm_url, cfg.model_id, CODING_WORKLOAD)
            )

    async def metrics_scraper() -> None:
        while time.perf_counter() < stop_at:
            m = vllm_metrics(f"{cfg.vllm_url}/metrics")
            if m:
                peaks["running"] = max(peaks["running"], m.get("requests_running", 0.0))
                peaks["waiting"] = max(peaks["waiting"], m.get("requests_waiting", 0.0))
                peaks["kv"] = max(peaks["kv"], m.get("kv_cache_pct", 0.0))
            await asyncio.sleep(2.0)

    t_start = time.perf_counter()
    tasks: list[asyncio.Task[None]] = [
        asyncio.create_task(translation_worker()),
        *[asyncio.create_task(coding_worker()) for _ in range(cfg.coding_concurrency)],
        asyncio.create_task(metrics_scraper()),
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.perf_counter() - t_start

    metrics_after = vllm_metrics(f"{cfg.vllm_url}/metrics")
    counters = SchedulerCounters(
        preemptions_delta=int(
            metrics_after.get("preemptions", 0) - metrics_before.get("preemptions", 0)
        ),
        kv_cache_pct_peak=round(peaks["kv"], 3),
        requests_running_peak=round(peaks["running"], 1),
        requests_waiting_peak=round(peaks["waiting"], 1),
        queue_time_avg_ms=round(metrics_after.get("queue_time_avg", 0.0) * 1000.0, 1),
        prefix_cache_hit_pct=round(metrics_after.get("prefix_cache_hit_pct", 0.0), 1),
    )

    tr = summarize("translation", "contention", translation_samples, wall)
    cd = summarize("coding", "contention", coding_samples, wall)
    return tr, cd, counters


async def run_warmup(client: httpx.AsyncClient, cfg: RunConfig) -> None:
    """Unmeasured priming pass — absorbs CUDA graph capture and KV cache cold-start.

    With CUDA graphs enabled (no --enforce-eager), the first request of
    each (batch_size, seq_len) shape triggers a graph capture that costs
    ~0.5-1.5 s. Running both workloads concurrently for ``warmup_seconds``
    exercises the common shapes so measurement phases see steady-state
    TTFT, not one-time compilation costs.
    """
    stop_at = time.perf_counter() + cfg.warmup_seconds

    async def worker(workload: Workload) -> None:
        while time.perf_counter() < stop_at:
            await stream_one(client, cfg.vllm_url, cfg.model_id, workload)

    await asyncio.gather(
        worker(TRANSLATION_WORKLOAD),
        worker(TRANSLATION_WORKLOAD),
        worker(CODING_WORKLOAD),
        worker(CODING_WORKLOAD),
    )
    # Quiet period lets vLLM drain queues before measurement.
    await asyncio.sleep(2.0)


# ── Proxy sanity check ──────────────────────────────────────────


async def run_proxy_sanity(cfg: RunConfig) -> ProxySanity:
    """Hit the Anthropic proxy end-to-end to validate streaming path.

    Sends two coding-shaped requests separated by a 30s idle gap to
    exercise uvicorn's keep-alive handling. Expects a full SSE frame
    sequence with no raw socket errors.
    """
    t0 = time.perf_counter()
    message_start = False
    delta_count = 0
    message_stop = False
    upstream_error: str | None = None

    body = {
        "model": cfg.model_id,
        "max_tokens": 512,
        "messages": [{"role": "user", "content": "Say 'hello world' and nothing else."}],
        "stream": True,
    }
    headers = {
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    try:
        async with (
            httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=10.0)) as c,
            c.stream("POST", f"{cfg.proxy_url}/v1/messages", json=body, headers=headers) as resp,
        ):
            if resp.status_code != 200:
                text = (await resp.aread()).decode("utf-8", "replace")
                return ProxySanity(
                    ok=False,
                    elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
                    note=f"http_{resp.status_code}: {text[:120]}",
                )
            async for line in resp.aiter_lines():
                if line.startswith("event: message_start"):
                    message_start = True
                elif line.startswith("event: content_block_delta"):
                    delta_count += 1
                elif line.startswith("event: message_stop"):
                    message_stop = True
                elif line.startswith("data: ") and "upstream_error" in line:
                    upstream_error = line[6:][:200]
    except httpx.HTTPError as exc:
        return ProxySanity(
            ok=False,
            elapsed_ms=round((time.perf_counter() - t0) * 1000.0, 1),
            note=f"{type(exc).__name__}: {exc}",
        )

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    return ProxySanity(
        ok=bool(message_start and delta_count > 0 and message_stop and upstream_error is None),
        message_start=message_start,
        content_delta_count=delta_count,
        message_stop=message_stop,
        upstream_error=upstream_error,
        elapsed_ms=elapsed_ms,
    )


# ── Environment fingerprint ──────────────────────────────────────


def environment_fingerprint(cfg: RunConfig) -> dict[str, Any]:
    """Capture enough context to know if two runs are comparable."""
    env: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "model_id": cfg.model_id,
    }

    # vLLM container image digest + launch args
    try:
        proc = subprocess.run(
            [
                "docker",
                "inspect",
                "autosre-vllm-local",
                "--format",
                "{{.Image}}|{{range .Args}}{{.}} {{end}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0:
            image, _, args = proc.stdout.strip().partition("|")
            env["vllm_image"] = image
            env["vllm_args"] = args.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Git SHA of auto-sre at run time
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0:
            env["autosre_sha"] = proc.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # GPU model (first line of nvidia-smi query)
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        if proc.returncode == 0:
            env["gpu"] = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else "unknown"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # Content hashes of all protected recipe files (for perf-gate enforcement)
    try:
        from autosre.hooks_backend.recipe_guard import recipe_content_hashes

        env["recipe_hashes"] = recipe_content_hashes()
    except Exception:
        pass

    return env


# ── Top-level entry ─────────────────────────────────────────────


async def run(cfg: RunConfig) -> RunResult:
    """Execute all phases and return a :class:`RunResult`."""
    result = RunResult(
        timestamp=time.strftime("%Y%m%dT%H%M%S"),
        config=asdict(cfg),
        environment=environment_fingerprint(cfg),
    )

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
        await run_warmup(client, cfg)

        tr_iso = await run_isolated(client, cfg, TRANSLATION_WORKLOAD, cfg.translation_concurrency)
        result.phases.append(tr_iso)

        cd_iso = await run_isolated(client, cfg, CODING_WORKLOAD, cfg.coding_concurrency)
        result.phases.append(cd_iso)

        tr_con, cd_con, counters = await run_contention(client, cfg)
        result.phases.append(tr_con)
        result.phases.append(cd_con)
        result.scheduler = counters

        if cfg.saturate_slots:
            sat_cfg = RunConfig(**{**asdict(cfg), "coding_concurrency": cfg.saturate_concurrency})
            sat_tr, sat_cd, sat_counters = await run_contention(client, sat_cfg)
            sat_tr.phase = "saturation"
            sat_cd.phase = "saturation"
            result.phases.append(sat_tr)
            result.phases.append(sat_cd)
            result.scheduler.preemptions_delta += sat_counters.preemptions_delta
            result.scheduler.requests_running_peak = max(
                result.scheduler.requests_running_peak, sat_counters.requests_running_peak
            )
            result.scheduler.requests_waiting_peak = max(
                result.scheduler.requests_waiting_peak, sat_counters.requests_waiting_peak
            )

    if cfg.run_proxy_check:
        result.proxy_sanity = await run_proxy_sanity(cfg)

    return result
