"""Tests for autosre.eval.capture — stream-json + proxy parity."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from autosre.eval.capture import (
    TurnRecord,
    normalize_run,
    parse_proxy_slice,
    parse_stream_json,
    write_turns,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


class TestStreamJsonParser:
    def test_single_turn_with_tool_call(self, tmp_path: Path) -> None:
        transcript = tmp_path / "transcript.jsonl"
        _write_jsonl(
            transcript,
            [
                {
                    "type": "message_start",
                    "timestamp": 100.0,
                    "message": {
                        "model": "claude-opus-4-6",
                        "usage": {
                            "input_tokens": 2500,
                            "cache_creation_input_tokens": 100,
                            "cache_read_input_tokens": 400,
                            "output_tokens": 0,
                        },
                    },
                },
                {
                    "type": "content_block_start",
                    "content_block": {
                        "type": "text",
                        "text": "Let me check the file.",
                    },
                },
                {
                    "type": "content_block_start",
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "Read",
                        "input": {"file_path": "/x/src/app.py"},
                    },
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 140},
                },
                {"type": "message_stop", "timestamp": 102.5},
            ],
        )
        turns = parse_stream_json(transcript)
        assert len(turns) == 1
        t = turns[0]
        assert t.provider == "anthropic"
        assert t.model == "claude-opus-4-6"
        assert t.input_tokens == 2500
        assert t.cache_creation_input_tokens == 100
        assert t.cache_read_input_tokens == 400
        assert t.output_tokens == 140
        assert t.elapsed_ms == 2500.0
        assert t.stop_reason == "tool_use"
        assert len(t.tool_calls) == 1
        tc = t.tool_calls[0]
        assert tc.name == "Read"
        assert tc.id == "toolu_01"
        assert "file_path=/x/src/app.py" in tc.input_summary
        assert "check the file" in t.response_prefix

    def test_multi_turn_sequence(self, tmp_path: Path) -> None:
        transcript = tmp_path / "transcript.jsonl"
        _write_jsonl(
            transcript,
            [
                {
                    "type": "message_start",
                    "timestamp": 1.0,
                    "message": {"model": "m", "usage": {"input_tokens": 1, "output_tokens": 0}},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 5},
                },
                {"type": "message_stop", "timestamp": 2.0},
                {
                    "type": "message_start",
                    "timestamp": 3.0,
                    "message": {"model": "m", "usage": {"input_tokens": 10, "output_tokens": 0}},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 7},
                },
                {"type": "message_stop", "timestamp": 4.0},
            ],
        )
        turns = parse_stream_json(transcript)
        assert len(turns) == 2
        assert turns[0].output_tokens == 5
        assert turns[1].input_tokens == 10
        assert turns[1].output_tokens == 7

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        transcript = tmp_path / "empty.jsonl"
        transcript.write_text("")
        assert parse_stream_json(transcript) == []

    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        assert parse_stream_json(tmp_path / "nope.jsonl") == []

    def test_malformed_line_skipped(self, tmp_path: Path) -> None:
        transcript = tmp_path / "t.jsonl"
        transcript.write_text(
            "not json\n"
            + json.dumps(
                {
                    "type": "message_start",
                    "timestamp": 0,
                    "message": {"model": "m", "usage": {"input_tokens": 1, "output_tokens": 2}},
                }
            )
            + "\n"
            + json.dumps({"type": "message_stop", "timestamp": 1})
            + "\n"
        )
        turns = parse_stream_json(transcript)
        assert len(turns) == 1

    def test_unterminated_turn_still_flushed(self, tmp_path: Path) -> None:
        """A message_start without a matching message_stop is still captured."""
        transcript = tmp_path / "t.jsonl"
        _write_jsonl(
            transcript,
            [
                {
                    "type": "message_start",
                    "timestamp": 5.0,
                    "message": {"model": "m", "usage": {"input_tokens": 1, "output_tokens": 0}},
                },
                {"type": "message_delta", "usage": {"output_tokens": 9}},
            ],
        )
        turns = parse_stream_json(transcript)
        assert len(turns) == 1
        assert turns[0].output_tokens == 9


class TestProxySliceParser:
    def test_filters_by_run_id(self, tmp_path: Path) -> None:
        log = tmp_path / "proxy.jsonl"
        _write_jsonl(
            log,
            [
                {
                    "ts": 1.0,
                    "model": "qwen",
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "elapsed_ms": 1200,
                    "run_id": "r1::security",
                },
                {
                    "ts": 2.0,
                    "model": "qwen",
                    "input_tokens": 120,
                    "output_tokens": 20,
                    "elapsed_ms": 300,
                    "run_id": "",
                },
                {
                    "ts": 3.0,
                    "model": "qwen",
                    "input_tokens": 80,
                    "output_tokens": 10,
                    "elapsed_ms": 400,
                    "run_id": "r1::security",
                },
                {
                    "ts": 4.0,
                    "model": "qwen",
                    "input_tokens": 5,
                    "output_tokens": 5,
                    "elapsed_ms": 50,
                    "run_id": "r1::leakage",
                },
            ],
        )
        turns = parse_proxy_slice(log, run_id="r1::security")
        assert len(turns) == 2
        assert turns[0].input_tokens == 100
        assert turns[1].input_tokens == 80
        assert all(t.provider == "local" for t in turns)

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        assert parse_proxy_slice(tmp_path / "nope.jsonl", run_id="x") == []


class TestCaptureParity:
    """Both sources must produce the exact same TurnRecord field set.

    Any future field added to TurnRecord must be populatable from BOTH
    sources or this test fails. That is the invariant the whole eval
    pipeline relies on for symmetric comparison.
    """

    def test_field_set_identical(self, tmp_path: Path) -> None:
        # Build a minimal turn from each source and compare keys.
        stream_path = tmp_path / "stream.jsonl"
        _write_jsonl(
            stream_path,
            [
                {
                    "type": "message_start",
                    "timestamp": 0.0,
                    "message": {
                        "model": "claude",
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
                {"type": "message_delta", "usage": {"output_tokens": 2}},
                {"type": "message_stop", "timestamp": 0.5},
            ],
        )
        proxy_path = tmp_path / "proxy.jsonl"
        _write_jsonl(
            proxy_path,
            [
                {
                    "ts": 10.0,
                    "model": "qwen",
                    "input_tokens": 1,
                    "output_tokens": 2,
                    "elapsed_ms": 500,
                    "run_id": "p",
                },
            ],
        )

        a_turns = parse_stream_json(stream_path)
        l_turns = parse_proxy_slice(proxy_path, run_id="p")

        assert len(a_turns) == 1
        assert len(l_turns) == 1

        a_keys = set(a_turns[0].model_dump().keys())
        l_keys = set(l_turns[0].model_dump().keys())
        assert a_keys == l_keys, (
            f"symmetric field sets required; diff: "
            f"anthropic_only={a_keys - l_keys}, local_only={l_keys - a_keys}"
        )

    def test_totals_match_for_equivalent_inputs(self, tmp_path: Path) -> None:
        """Input/output token totals for matched-synthetic inputs are equal."""
        stream_path = tmp_path / "stream.jsonl"
        _write_jsonl(
            stream_path,
            [
                {
                    "type": "message_start",
                    "timestamp": 0.0,
                    "message": {
                        "model": "claude",
                        "usage": {"input_tokens": 123, "output_tokens": 0},
                    },
                },
                {"type": "message_delta", "usage": {"output_tokens": 45}},
                {"type": "message_stop", "timestamp": 1.0},
            ],
        )
        proxy_path = tmp_path / "proxy.jsonl"
        _write_jsonl(
            proxy_path,
            [
                {
                    "ts": 0,
                    "model": "qwen",
                    "input_tokens": 123,
                    "output_tokens": 45,
                    "elapsed_ms": 1000,
                    "run_id": "z",
                },
            ],
        )
        a = parse_stream_json(stream_path)[0]
        local = parse_proxy_slice(proxy_path, run_id="z")[0]
        assert a.input_tokens == local.input_tokens == 123
        assert a.output_tokens == local.output_tokens == 45


class TestWriteTurns:
    def test_roundtrip(self, tmp_path: Path) -> None:
        turns = [
            TurnRecord(ts=1.0, provider="local", model="m", output_tokens=5),
            TurnRecord(ts=2.0, provider="local", model="m", output_tokens=8),
        ]
        out = tmp_path / "turns.jsonl"
        write_turns(turns, out)
        lines = out.read_text().strip().splitlines()
        assert len(lines) == 2
        recovered = [TurnRecord.model_validate_json(line) for line in lines]
        assert recovered[0].output_tokens == 5
        assert recovered[1].output_tokens == 8


class TestNormalizeRun:
    def test_anthropic_path(self, tmp_path: Path) -> None:
        capture_dir = tmp_path / "capture"
        capture_dir.mkdir()
        _write_jsonl(
            capture_dir / "transcript.jsonl",
            [
                {
                    "type": "message_start",
                    "timestamp": 0,
                    "message": {
                        "model": "claude",
                        "usage": {"input_tokens": 1, "output_tokens": 0},
                    },
                },
                {"type": "message_delta", "usage": {"output_tokens": 2}},
                {"type": "message_stop", "timestamp": 1},
            ],
        )
        out = tmp_path / "turns.jsonl"
        turns = normalize_run(
            provider="anthropic",
            run_id="x",
            capture_dir=capture_dir,
            proxy_log_path=tmp_path / "never.jsonl",
            out_path=out,
        )
        assert len(turns) == 1
        assert out.exists()

    def test_local_path(self, tmp_path: Path) -> None:
        proxy = tmp_path / "proxy.jsonl"
        _write_jsonl(
            proxy,
            [
                {
                    "ts": 1,
                    "model": "q",
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "elapsed_ms": 100,
                    "run_id": "myrun",
                },
            ],
        )
        out = tmp_path / "turns.jsonl"
        turns = normalize_run(
            provider="local",
            run_id="myrun",
            capture_dir=tmp_path / "capture",
            proxy_log_path=proxy,
            out_path=out,
        )
        assert len(turns) == 1
        assert turns[0].provider == "local"
        assert out.exists()
