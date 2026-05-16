"""Capture layer — normalize transcript telemetry into ``TurnRecord`` rows.

Two sources, one output schema:

- **Anthropic provider** streams a ``stream-json`` transcript via
  ``claude --output-format=stream-json --verbose``. Each line is a JSON
  event: ``message_start`` / ``content_block_start`` /
  ``content_block_delta`` / ``message_delta`` / ``message_stop``. We walk
  that sequence with a small state machine and emit one
  :class:`TurnRecord` per assistant message.
- **Local provider** has the same telemetry available from
  ``anthropic_proxy.py`` which already writes one JSONL row per request
  to ``~/.local/share/autosre/proxy-requests.jsonl``. Rows produced
  during an eval run are tagged with ``AUTOSRE_RUN_ID`` so we can slice
  them out.

The downstream eval pipeline only consumes ``TurnRecord`` and
``ToolCall`` — never the raw sources — so that symmetry holds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from pathlib import Path

Role = Literal["assistant", "tool_result", "user", "system"]
CaptureProvider = Literal["local", "anthropic"]


class ToolCall(BaseModel):
    """One tool invocation made by the assistant during a turn."""

    id: str = ""
    name: str
    # input_summary is a short string so the dataset is greppable.
    # We never store the full argument blob here; that lives in the
    # raw transcript file if anyone needs it.
    input_summary: str = ""


class TurnRecord(BaseModel):
    """Provider-agnostic normalized view of one assistant turn."""

    ts: float
    provider: CaptureProvider
    model: str
    agent: str | None = None
    role: Role = "assistant"
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: float = 0.0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    prompt_prefix: str = ""
    response_prefix: str = ""
    stop_reason: str | None = None


# ── Anthropic stream-json parser ────────────────────────────────────


def parse_stream_json(path: Path, *, model_hint: str = "") -> list[TurnRecord]:
    """Parse a ``stream-json`` transcript into ``TurnRecord`` rows.

    Claude Code's ``--output-format=stream-json`` emits one JSON object
    per line. We recognize the events that carry the fields we need:

    - ``message_start``: pulls ``message.model`` and initial ``usage``.
    - ``content_block_start`` with a ``tool_use`` block: appends to the
      current turn's ``tool_calls``.
    - ``message_delta``: updates final ``usage`` + ``stop_reason``.
    - ``message_stop``: finalizes the current turn.

    Claude Code also wraps some events in an outer envelope with fields
    like ``{"type": "assistant", "message": {...}}``; we unwrap those
    too. Unknown events are ignored — the goal is to extract telemetry,
    not to be a full spec-conformant parser.
    """
    out: list[TurnRecord] = []
    if not path.exists():
        return out

    current: _PartialTurn | None = None

    for raw_line in path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue

        inner = _inner_message(event)

        ev_type = event.get("type") or (inner.get("type") if inner else None)

        if ev_type == "message_start":
            msg_obj = event.get("message") or inner or {}
            msg = msg_obj if isinstance(msg_obj, dict) else {}
            current = _PartialTurn(
                model=str(msg.get("model") or model_hint or ""),
                ts=_float(event.get("timestamp") or event.get("ts")),
            )
            usage_obj = msg.get("usage") or {}
            usage = usage_obj if isinstance(usage_obj, dict) else {}
            current.input_tokens = _int(usage.get("input_tokens"))
            current.cache_creation_input_tokens = _int(usage.get("cache_creation_input_tokens"))
            current.cache_read_input_tokens = _int(usage.get("cache_read_input_tokens"))
            current.output_tokens = _int(usage.get("output_tokens"))

        elif ev_type == "content_block_start":
            if current is None:
                current = _PartialTurn(model=model_hint)
            block_obj = event.get("content_block") or {}
            block = block_obj if isinstance(block_obj, dict) else {}
            if block.get("type") == "tool_use":
                name = str(block.get("name") or "")
                tool_id = str(block.get("id") or "")
                summary = _summarize_tool_input(block.get("input") or {})
                current.tool_calls.append(ToolCall(id=tool_id, name=name, input_summary=summary))
            elif block.get("type") == "text":
                text = str(block.get("text") or "")
                if text and not current.response_prefix:
                    current.response_prefix = text[:200]

        elif ev_type == "content_block_delta":
            delta_obj = event.get("delta") or {}
            delta = delta_obj if isinstance(delta_obj, dict) else {}
            if delta.get("type") == "text_delta" and current is not None:
                chunk = str(delta.get("text") or "")
                if chunk and not current.response_prefix:
                    current.response_prefix = chunk[:200]

        elif ev_type == "message_delta":
            if current is None:
                continue
            delta_obj = event.get("delta") or {}
            delta = delta_obj if isinstance(delta_obj, dict) else {}
            usage_obj = event.get("usage") or {}
            usage = usage_obj if isinstance(usage_obj, dict) else {}
            updated = _int(usage.get("output_tokens"))
            if updated:
                current.output_tokens = updated
            stop = delta.get("stop_reason")
            if stop:
                current.stop_reason = str(stop)

        elif ev_type == "message_stop":
            if current is None:
                continue
            end_ts = _float(event.get("timestamp") or event.get("ts"))
            if end_ts and current.ts:
                current.elapsed_ms = max(0.0, (end_ts - current.ts) * 1000.0)
            out.append(current.to_turn_record())
            current = None

    if current is not None:
        out.append(current.to_turn_record())

    return out


def _inner_message(event: dict[str, object]) -> dict[str, object] | None:
    """Claude Code wraps some stream-json events in an envelope with
    ``{"type": "assistant", "message": {...}}``. Return the inner
    message dict if present, else ``None``.
    """
    msg = event.get("message")
    if isinstance(msg, dict):
        return msg
    return None


def _int(val: object) -> int:
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    if isinstance(val, str):
        try:
            return int(val)
        except ValueError:
            return 0
    return 0


def _float(val: object) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val)
        except ValueError:
            return 0.0
    return 0.0


def _summarize_tool_input(inp: object) -> str:
    """Short, greppable summary of a tool_use input block."""
    if not isinstance(inp, dict):
        return ""
    parts: list[str] = []
    for key in ("file_path", "pattern", "command", "path", "url", "prompt"):
        if key in inp:
            val = str(inp[key])
            parts.append(f"{key}={val[:80]}")
    if not parts and inp:
        first = next(iter(inp.items()))
        parts.append(f"{first[0]}={str(first[1])[:80]}")
    return " ".join(parts)[:200]


@dataclass
class _PartialTurn:
    """Mutable accumulator used while walking the stream-json events."""

    model: str = ""
    ts: float = 0.0
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    elapsed_ms: float = 0.0
    stop_reason: str | None = None
    response_prefix: str = ""
    prompt_prefix: str = ""

    def __post_init__(self) -> None:
        self.tool_calls: list[ToolCall] = []

    def to_turn_record(self) -> TurnRecord:
        return TurnRecord(
            ts=self.ts,
            provider="anthropic",
            model=self.model,
            role="assistant",
            input_tokens=self.input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens,
            output_tokens=self.output_tokens,
            elapsed_ms=self.elapsed_ms,
            tool_calls=list(self.tool_calls),
            prompt_prefix=self.prompt_prefix,
            response_prefix=self.response_prefix,
            stop_reason=self.stop_reason,
        )


# ── Local proxy jsonl parser ────────────────────────────────────────


def parse_proxy_slice(
    log_path: Path,
    *,
    run_id: str,
) -> list[TurnRecord]:
    """Read proxy-requests.jsonl, filter to ``run_id``, return ``TurnRecord`` rows.

    The proxy schema lives in
    ``autosre/backends/anthropic_proxy.py:_enqueue_log``. We map each
    row directly to a ``TurnRecord`` since the proxy already stores
    everything we need. The ``agent`` field is left null because the
    proxy does not know which sub-agent issued a given request.
    """
    if not log_path.exists():
        return []
    out: list[TurnRecord] = []
    for raw_line in log_path.read_text().splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if row.get("run_id") != run_id:
            continue
        out.append(
            TurnRecord(
                ts=float(row.get("ts") or 0.0),
                provider="local",
                model=str(row.get("model") or ""),
                role="assistant",
                input_tokens=int(row.get("input_tokens") or 0),
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                output_tokens=int(row.get("output_tokens") or 0),
                elapsed_ms=float(row.get("elapsed_ms") or 0.0),
                tool_calls=[],  # proxy does not capture per-tool detail
                prompt_prefix=str(row.get("prompt_prefix") or "")[:200],
                response_prefix=str(row.get("response_prefix") or "")[:200],
                stop_reason=None,
            )
        )
    return out


# ── Write normalized output ─────────────────────────────────────────


def write_turns(turns: list[TurnRecord], out_path: Path) -> None:
    """Write one TurnRecord per line to ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for turn in turns:
            f.write(turn.model_dump_json() + "\n")


def normalize_run(
    *,
    provider: CaptureProvider,
    run_id: str,
    capture_dir: Path,
    proxy_log_path: Path,
    out_path: Path,
    model_hint: str = "",
) -> list[TurnRecord]:
    """Produce ``turns.jsonl`` for one (provider x suite) run.

    Routes to the right source by provider and writes the result.
    """
    if provider == "anthropic":
        turns = parse_stream_json(
            capture_dir / "transcript.jsonl",
            model_hint=model_hint,
        )
    else:
        turns = parse_proxy_slice(proxy_log_path, run_id=run_id)
    write_turns(turns, out_path)
    return turns
