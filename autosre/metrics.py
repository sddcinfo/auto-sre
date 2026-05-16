"""vLLM metrics scraper and proxy request analytics.

Scrapes vLLM's Prometheus /metrics endpoint directly (no prometheus_client
library needed). Reads the proxy request JSONL log for historical analytics.

All data is computed on-the-fly from HTTP scrapes and log file reads —
no external time-series database required.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

import httpx

# ── Request logs ─────────────────────────────────────────────────
# Two sources:
#   1. Anthropic proxy (Claude Code requests) — proxy-requests.jsonl
#   2. Meeting-scribe translation backend (direct vLLM) — scribe-translations.jsonl
# Both have the same schema and are merged by timestamp.

PROXY_LOG_PATH = Path.home() / ".local" / "share" / "autosre" / "proxy-requests.jsonl"
SCRIBE_TRANSLATION_LOG_PATH = (
    Path.home() / ".local" / "share" / "autosre" / "scribe-translations.jsonl"
)


def _read_jsonl_tail(path: Path, n: int) -> list[dict[str, Any]]:
    """Read last n valid JSON entries from a JSONL file."""
    if not path.exists():
        return []
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            file_size = f.tell()
            if file_size == 0:
                return []
            # Read last chunk — generous 10KB per entry
            chunk_size = min(file_size, n * 10240)
            f.seek(file_size - chunk_size)
            data = f.read().decode("utf-8", errors="replace")
        lines = [line for line in data.strip().splitlines() if line.strip()]
        entries = []
        for line in lines[-n:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
    except (OSError, ValueError):
        return []


def read_recent_requests(n: int = 20) -> list[dict[str, Any]]:
    """Read last n entries from BOTH request logs, merged by timestamp.

    Combines proxy-requests.jsonl (Claude Code) and scribe-translations.jsonl
    (meeting-scribe direct-to-vLLM translations) into a single chronological feed.
    """
    # Read 2n from each so after merge + trim we still have n
    proxy_entries = _read_jsonl_tail(PROXY_LOG_PATH, n * 2)
    scribe_entries = _read_jsonl_tail(SCRIBE_TRANSLATION_LOG_PATH, n * 2)

    merged = proxy_entries + scribe_entries
    merged.sort(key=lambda r: r.get("ts", 0))
    return merged[-n:] if len(merged) > n else merged


def request_analytics(requests: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Compute analytics from proxy request log.

    Returns rolling averages by source, token breakdown, req/min rate.
    """
    if requests is None:
        requests = read_recent_requests(100)
    if not requests:
        return {}

    translation = [r for r in requests if r.get("source") == "translation"]
    coding = [r for r in requests if r.get("source") == "coding"]

    def _avg(items: list[dict[str, Any]], key: str) -> float:
        vals = [r[key] for r in items if key in r]
        return sum(vals) / len(vals) if vals else 0

    # Time range for req/min
    timestamps = [r.get("ts", 0) for r in requests if r.get("ts")]
    duration_min = (max(timestamps) - min(timestamps)) / 60 if len(timestamps) > 1 else 1

    return {
        "translation_avg_ms": _avg(translation, "elapsed_ms"),
        "translation_count": len(translation),
        "translation_tokens": sum(r.get("output_tokens", 0) for r in translation),
        "coding_avg_ms": _avg(coding, "elapsed_ms"),
        "coding_count": len(coding),
        "coding_tokens": sum(r.get("output_tokens", 0) for r in coding),
        "total_count": len(requests),
        "requests_per_min": len(requests) / max(duration_min, 0.01),
    }


# ── vLLM Prometheus metrics scraper ─────────────────────────────

# Module-level state for rate computation (two-sample delta)
_prev: dict[str, float] = {}
_prev_ts: float = 0.0


def vllm_metrics(url: str = "http://localhost:8010/metrics") -> dict[str, Any]:
    """Scrape vLLM /metrics and return parsed metrics dict.

    Returns keys: ttft_p50, ttft_p99, tpot_p50, tpot_p99,
    requests_running, requests_waiting, kv_cache_pct,
    prefix_cache_hit_pct, prompt_tps, gen_tps, total_requests,
    queue_time_avg, preemptions.
    """
    global _prev, _prev_ts

    try:
        resp = httpx.get(url, timeout=2)
        lines = resp.text.splitlines()
    except Exception:
        return {}

    now = time.monotonic()
    result: dict[str, float] = {}

    # Simple gauge/counter metrics
    gauges = {
        "vllm:num_requests_running": "requests_running",
        "vllm:num_requests_waiting": "requests_waiting",
        "vllm:kv_cache_usage_perc": "kv_cache_pct",
    }
    counters = {
        "vllm:prompt_tokens_total": "prompt_tokens_total",
        "vllm:generation_tokens_total": "gen_tokens_total",
        "vllm:request_success_total": "request_success",
        "vllm:prefix_cache_queries_total": "prefix_cache_queries",
        "vllm:prefix_cache_hits_total": "prefix_cache_hits",
        "vllm:num_preemptions_total": "preemptions",
    }

    # Histogram buckets for TTFT and TPOT
    ttft_buckets: list[tuple[float, float]] = []
    tpot_buckets: list[tuple[float, float]] = []
    queue_sum = 0.0
    queue_count = 0.0

    for line in lines:
        if line.startswith("#"):
            continue

        # Gauges
        for prefix, key in gauges.items():
            if line.startswith(prefix):
                with contextlib.suppress(ValueError, IndexError):
                    result[key] = float(line.rsplit(None, 1)[-1])
                break

        # Counters (may have labels — take last match)
        for prefix, key in counters.items():
            if line.startswith(prefix):
                try:
                    val = float(line.rsplit(None, 1)[-1])
                    result[key] = result.get(key, 0) + val
                except (ValueError, IndexError):
                    pass
                break

        # TTFT histogram buckets
        if line.startswith("vllm:time_to_first_token_seconds_bucket{"):
            bucket = _parse_bucket(line)
            if bucket:
                ttft_buckets.append(bucket)

        # TPOT histogram buckets
        if line.startswith("vllm:inter_token_latency_seconds_bucket{"):
            bucket = _parse_bucket(line)
            if bucket:
                tpot_buckets.append(bucket)

        # Queue time (sum/count for average)
        if line.startswith("vllm:request_queue_time_seconds_sum"):
            with contextlib.suppress(ValueError, IndexError):
                queue_sum = float(line.rsplit(None, 1)[-1])
        if line.startswith("vllm:request_queue_time_seconds_count"):
            with contextlib.suppress(ValueError, IndexError):
                queue_count = float(line.rsplit(None, 1)[-1])

    # Compute TTFT/TPOT percentiles from histogram buckets
    if ttft_buckets:
        result["ttft_p50"] = _percentile(ttft_buckets, 0.5)
        result["ttft_p99"] = _percentile(ttft_buckets, 0.99)

    if tpot_buckets:
        result["tpot_p50"] = _percentile(tpot_buckets, 0.5)
        result["tpot_p99"] = _percentile(tpot_buckets, 0.99)

    # Average queue time
    if queue_count > 0:
        result["queue_time_avg"] = queue_sum / queue_count

    # Prefix cache hit rate
    queries = result.get("prefix_cache_queries", 0)
    hits = result.get("prefix_cache_hits", 0)
    result["prefix_cache_hit_pct"] = (hits / queries * 100) if queries > 0 else 0

    # Token rates (delta from previous scrape)
    if _prev_ts > 0:
        dt = now - _prev_ts
        if dt > 0.5:
            for counter_key, rate_key in [
                ("prompt_tokens_total", "prompt_tps"),
                ("gen_tokens_total", "gen_tps"),
            ]:
                cur = result.get(counter_key, 0)
                prev = _prev.get(counter_key, cur)
                result[rate_key] = max(0, (cur - prev) / dt)

    # Save for next delta
    _prev = {k: v for k, v in result.items() if k.endswith("_total")}
    _prev_ts = now

    # Total successful requests
    result["total_requests"] = int(result.get("request_success", 0))

    return result


def _parse_bucket(line: str) -> tuple[float, float] | None:
    """Parse a Prometheus histogram bucket line → (le, count)."""
    try:
        # Extract le="..." value
        le_start = line.index('le="') + 4
        le_end = line.index('"', le_start)
        le_str = line[le_start:le_end]
        le = float("inf") if le_str == "+Inf" else float(le_str)
        count = float(line.rsplit(None, 1)[-1])
        return (le, count)
    except (ValueError, IndexError):
        return None


def _percentile(buckets: list[tuple[float, float]], p: float) -> float:
    """Estimate percentile from histogram buckets via linear interpolation."""
    if not buckets:
        return 0.0

    # Sort by le boundary
    buckets = sorted(buckets, key=lambda x: x[0])

    # Total count is the +Inf bucket (or last bucket)
    total = buckets[-1][1]
    if total == 0:
        return 0.0

    target = total * p
    prev_le = 0.0
    prev_count = 0.0

    for le, count in buckets:
        if count >= target:
            # Interpolate within this bucket
            if count == prev_count:
                return le
            fraction = (target - prev_count) / (count - prev_count)
            return prev_le + fraction * (le - prev_le)
        prev_le = le
        prev_count = count

    # Fallback: return the last finite bucket boundary
    for le, _ in reversed(buckets):
        if le != float("inf"):
            return le
    return 0.0
