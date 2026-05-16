"""Tests for autosre.metrics — vLLM metrics scraper and proxy request analytics.

Unit tests use synthetic Prometheus text. Integration tests hit real endpoints.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from autosre.metrics import (
    _parse_bucket,
    _percentile,
    read_recent_requests,
    request_analytics,
    vllm_metrics,
)

# ── Unit Tests (no external services) ────────────────────────


class TestParseBucket:
    def test_normal_bucket(self):
        line = (
            'vllm:time_to_first_token_seconds_bucket{engine="0",le="0.25",model_name="test"} 48.0'
        )
        result = _parse_bucket(line)
        assert result == (0.25, 48.0)

    def test_inf_bucket(self):
        line = (
            'vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="test"} 230.0'
        )
        result = _parse_bucket(line)
        assert result[0] == float("inf")
        assert result[1] == 230.0

    def test_invalid_line(self):
        assert _parse_bucket("not a bucket line") is None

    def test_zero_count(self):
        line = 'vllm:ttft_bucket{le="0.001"} 0.0'
        result = _parse_bucket(line)
        assert result == (0.001, 0.0)


class TestPercentile:
    def test_p50_simple(self):
        buckets = [(0.1, 5), (0.5, 10), (1.0, 15), (float("inf"), 15)]
        p50 = _percentile(buckets, 0.5)
        assert 0.1 < p50 < 1.0  # 50th percentile is between 0.1 and 1.0

    def test_p99(self):
        buckets = [(0.1, 90), (0.5, 99), (1.0, 100), (float("inf"), 100)]
        p99 = _percentile(buckets, 0.99)
        assert 0.1 < p99 <= 1.0

    def test_empty_buckets(self):
        assert _percentile([], 0.5) == 0.0

    def test_all_zero(self):
        buckets = [(0.1, 0), (0.5, 0), (float("inf"), 0)]
        assert _percentile(buckets, 0.5) == 0.0

    def test_all_in_first_bucket(self):
        buckets = [(0.05, 100), (0.1, 100), (float("inf"), 100)]
        p50 = _percentile(buckets, 0.5)
        assert p50 <= 0.05


class TestReadRecentRequests:
    """read_recent_requests reads from BOTH proxy + scribe logs and merges."""

    def _patch_both(self, proxy_path, scribe_path):
        """Patch both log paths to point into tmp_path."""
        return (
            patch("autosre.metrics.PROXY_LOG_PATH", proxy_path),
            patch("autosre.metrics.SCRIBE_TRANSLATION_LOG_PATH", scribe_path),
        )

    def test_empty_log(self, tmp_path):
        proxy = tmp_path / "proxy.jsonl"
        scribe = tmp_path / "scribe.jsonl"
        proxy.touch()
        p1, p2 = self._patch_both(proxy, scribe)
        with p1, p2:
            assert read_recent_requests() == []

    def test_missing_log(self, tmp_path):
        proxy = tmp_path / "nonexistent1.jsonl"
        scribe = tmp_path / "nonexistent2.jsonl"
        p1, p2 = self._patch_both(proxy, scribe)
        with p1, p2:
            assert read_recent_requests() == []

    def test_reads_last_n(self, tmp_path):
        proxy = tmp_path / "proxy.jsonl"
        scribe = tmp_path / "scribe.jsonl"
        entries = [
            json.dumps({"ts": time.time() - i, "source": "coding", "elapsed_ms": 100 * i})
            for i in range(10)
        ]
        proxy.write_text("\n".join(entries) + "\n")
        p1, p2 = self._patch_both(proxy, scribe)
        with p1, p2:
            result = read_recent_requests(3)
            assert len(result) == 3

    def test_handles_corrupt_lines(self, tmp_path):
        proxy = tmp_path / "proxy.jsonl"
        scribe = tmp_path / "scribe.jsonl"
        proxy.write_text('{"ts": 1}\nnot json\n{"ts": 2}\n')
        p1, p2 = self._patch_both(proxy, scribe)
        with p1, p2:
            result = read_recent_requests(10)
            assert len(result) == 2

    def test_merges_both_logs_by_timestamp(self, tmp_path):
        """Entries from proxy and scribe are merged chronologically."""
        proxy = tmp_path / "proxy.jsonl"
        scribe = tmp_path / "scribe.jsonl"
        now = time.time()
        proxy.write_text(
            json.dumps({"ts": now - 30, "source": "coding", "elapsed_ms": 5000})
            + "\n"
            + json.dumps({"ts": now - 10, "source": "coding", "elapsed_ms": 4000})
            + "\n"
        )
        scribe.write_text(
            json.dumps({"ts": now - 20, "source": "translation", "elapsed_ms": 400})
            + "\n"
            + json.dumps({"ts": now - 5, "source": "translation", "elapsed_ms": 300})
            + "\n"
        )
        p1, p2 = self._patch_both(proxy, scribe)
        with p1, p2:
            result = read_recent_requests(10)
            assert len(result) == 4
            # Sorted by timestamp ascending
            assert result[0]["source"] == "coding"  # -30
            assert result[1]["source"] == "translation"  # -20
            assert result[2]["source"] == "coding"  # -10
            assert result[3]["source"] == "translation"  # -5

    def test_scribe_only(self, tmp_path):
        """Works when only scribe log exists."""
        proxy = tmp_path / "nonexistent.jsonl"
        scribe = tmp_path / "scribe.jsonl"
        scribe.write_text(
            json.dumps({"ts": time.time(), "source": "translation", "elapsed_ms": 200}) + "\n"
        )
        p1, p2 = self._patch_both(proxy, scribe)
        with p1, p2:
            result = read_recent_requests(5)
            assert len(result) == 1
            assert result[0]["source"] == "translation"


class TestRequestAnalytics:
    def test_empty(self):
        assert request_analytics([]) == {}

    def test_translation_and_coding(self):
        requests = [
            {
                "source": "translation",
                "elapsed_ms": 200,
                "output_tokens": 10,
                "ts": time.time() - 60,
            },
            {
                "source": "translation",
                "elapsed_ms": 300,
                "output_tokens": 15,
                "ts": time.time() - 30,
            },
            {"source": "coding", "elapsed_ms": 5000, "output_tokens": 200, "ts": time.time()},
        ]
        a = request_analytics(requests)
        assert a["translation_count"] == 2
        assert a["coding_count"] == 1
        assert a["translation_avg_ms"] == 250
        assert a["coding_avg_ms"] == 5000
        assert a["translation_tokens"] == 25
        assert a["coding_tokens"] == 200


class TestVllmMetricsParsing:
    """Test vllm_metrics with synthetic Prometheus text."""

    SAMPLE_METRICS = """
# HELP vllm:num_requests_running Number of requests currently running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="test"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="test"} 1.0
# HELP vllm:kv_cache_usage_perc KV cache usage
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="test"} 0.42
# HELP vllm:generation_tokens_total Total generation tokens
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="test"} 5000.0
# HELP vllm:prompt_tokens_total Total prompt tokens
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="test"} 12000.0
# HELP vllm:prefix_cache_queries_total Total prefix cache queries
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="test"} 100.0
# HELP vllm:prefix_cache_hits_total Total prefix cache hits
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="test"} 75.0
# HELP vllm:time_to_first_token_seconds Time to first token
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",model_name="test"} 10.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.25",model_name="test"} 50.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.5",model_name="test"} 90.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="1.0",model_name="test"} 95.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",model_name="test"} 100.0
""".strip()

    def test_parses_gauges(self):
        mock_resp = MagicMock()
        mock_resp.text = self.SAMPLE_METRICS

        with patch("autosre.metrics.httpx.get", return_value=mock_resp):
            m = vllm_metrics()

        assert m["requests_running"] == 3.0
        assert m["requests_waiting"] == 1.0
        assert m["kv_cache_pct"] == 0.42

    def test_parses_counters(self):
        mock_resp = MagicMock()
        mock_resp.text = self.SAMPLE_METRICS

        with patch("autosre.metrics.httpx.get", return_value=mock_resp):
            m = vllm_metrics()

        assert m["gen_tokens_total"] == 5000.0
        assert m["prompt_tokens_total"] == 12000.0

    def test_computes_prefix_cache_hit_pct(self):
        mock_resp = MagicMock()
        mock_resp.text = self.SAMPLE_METRICS

        with patch("autosre.metrics.httpx.get", return_value=mock_resp):
            m = vllm_metrics()

        assert m["prefix_cache_hit_pct"] == 75.0

    def test_parses_ttft_percentiles(self):
        mock_resp = MagicMock()
        mock_resp.text = self.SAMPLE_METRICS

        with patch("autosre.metrics.httpx.get", return_value=mock_resp):
            m = vllm_metrics()

        assert "ttft_p50" in m
        assert 0.1 < m["ttft_p50"] < 0.5  # 50th percentile
        assert "ttft_p99" in m
        assert m["ttft_p99"] > 0.5  # 99th percentile near the tail

    def test_handles_connection_error(self):
        with patch("autosre.metrics.httpx.get", side_effect=Exception("connection refused")):
            m = vllm_metrics()
        assert m == {}


# ── Integration Tests (require live vLLM on :8010) ───────────

pytestmark_integration = (
    pytest.mark.skipif(not _is_vllm_running(), reason="vLLM not running on :8010")
    if (
        _is_vllm_running := lambda: (
            __import__("autosre.metrics", fromlist=["vllm_metrics"]).vllm_metrics() != {}
        )
    )
    and False
    else pytest.mark.skipif(True, reason="placeholder")
)


class TestVllmMetricsLive:
    """Integration tests against live vLLM — skipped if not running."""

    @pytest.fixture(autouse=True)
    def _check_vllm(self):
        m = vllm_metrics()
        if not m:
            pytest.skip("vLLM not running on :8010")

    def test_live_metrics_have_keys(self):
        m = vllm_metrics()
        assert "requests_running" in m
        assert "kv_cache_pct" in m
        assert "total_requests" in m

    def test_kv_cache_in_range(self):
        m = vllm_metrics()
        kv = m.get("kv_cache_pct", 0)
        assert 0 <= kv <= 1.0, f"KV cache {kv} out of range"

    def test_ttft_non_negative(self):
        m = vllm_metrics()
        if "ttft_p50" in m:
            assert m["ttft_p50"] >= 0
        if "ttft_p99" in m:
            assert m["ttft_p99"] >= 0
