"""Anthropic-to-OpenAI API proxy for vLLM.

Translates Anthropic Messages API (/v1/messages) to OpenAI Chat Completions
API (/v1/chat/completions) so Claude Code can talk to vLLM backends.

Handles: messages, system prompts, tool definitions, tool_use/tool_result
blocks, and streaming.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
import uuid
from typing import Any, cast

import click
import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request  # noqa: TC002 — used at runtime
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route


def _repair_tool_json(raw: str) -> str:
    """Attempt to repair common JSON issues from local models.

    Local models sometimes produce:
    - Trailing commas before } or ]
    - Single-quoted strings
    - Unquoted keys
    - Truncated JSON (missing closing braces)
    """
    import re

    s = raw.strip()
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Balance braces — append missing closing braces/brackets
    opens = s.count("{") - s.count("}")
    s += "}" * max(0, opens)
    opens = s.count("[") - s.count("]")
    s += "]" * max(0, opens)
    return s


def _convert_tools_to_openai(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert Anthropic tool definitions to OpenAI function format."""
    openai_tools = []
    for tool in tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
        )
    return openai_tools


def _convert_content_to_openai(content: str | list[dict[str, Any]]) -> str | list[dict[str, Any]]:
    """Convert Anthropic content blocks to OpenAI format."""
    if isinstance(content, str):
        return content

    # For simple text-only content, return as string
    if len(content) == 1 and content[0].get("type") == "text":
        return str(content[0]["text"])

    # Multi-part content (text + images)
    parts: list[dict[str, Any]] = []
    for block in content:
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{source.get('media_type', 'image/png')};base64,{source['data']}",
                        },
                    }
                )
    return parts


def _convert_messages_to_openai(
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert Anthropic messages to OpenAI chat format."""
    openai_messages: list[dict[str, Any]] = []

    # System message
    if system:
        if isinstance(system, str):
            openai_messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text_parts = [b["text"] for b in system if b.get("type") == "text"]
            openai_messages.append({"role": "system", "content": "\n\n".join(text_parts)})

    for msg in messages:
        role = msg["role"]
        content = msg.get("content")

        if role == "user":
            if isinstance(content, str):
                openai_messages.append({"role": "user", "content": content})
            elif isinstance(content, list):
                # Check for tool_result blocks (these become separate messages in OpenAI)
                tool_results = [b for b in content if b.get("type") == "tool_result"]
                other_blocks = [b for b in content if b.get("type") != "tool_result"]

                if other_blocks:
                    converted = _convert_content_to_openai(other_blocks)
                    openai_messages.append({"role": "user", "content": converted})

                for tr in tool_results:
                    # Extract text content from tool_result
                    tr_content = tr.get("content", "")
                    if isinstance(tr_content, list):
                        tr_content = "\n".join(
                            b.get("text", "") for b in tr_content if b.get("type") == "text"
                        )
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tr.get("tool_use_id", ""),
                            "content": str(tr_content),
                        }
                    )

        elif role == "assistant":
            if isinstance(content, str):
                openai_messages.append({"role": "assistant", "content": content})
            elif isinstance(content, list):
                # Separate text and tool_use blocks
                text_parts = []
                tool_calls = []
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block["text"])
                    elif block.get("type") == "tool_use":
                        tool_calls.append(
                            {
                                "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                                "type": "function",
                                "function": {
                                    "name": block["name"],
                                    "arguments": json.dumps(block.get("input", {})),
                                },
                            }
                        )

                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if text_parts:
                    assistant_msg["content"] = "\n".join(text_parts)
                else:
                    assistant_msg["content"] = None
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls
                openai_messages.append(assistant_msg)

    return openai_messages


def _convert_openai_response_to_anthropic(
    openai_resp: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    """Convert OpenAI chat completion response to Anthropic Messages format."""
    choice = openai_resp["choices"][0]
    message = choice["message"]

    content: list[dict[str, Any]] = []

    # Add text content
    if message.get("content"):
        # Strip <think>...</think> blocks from reasoning models
        text = message["content"]
        import re

        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        if text.strip():
            content.append({"type": "text", "text": text})

    # Add tool_use blocks
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            func = tc["function"]
            try:
                input_data = json.loads(func["arguments"])
            except (json.JSONDecodeError, TypeError):
                # Try repairing common JSON issues from local models
                try:
                    input_data = json.loads(_repair_tool_json(func.get("arguments", "{}")))
                except (json.JSONDecodeError, TypeError):
                    input_data = {"raw": func.get("arguments", "")}

            content.append(
                {
                    "type": "tool_use",
                    "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                    "name": func["name"],
                    "input": input_data,
                }
            )

    # Determine stop reason
    finish = choice.get("finish_reason", "stop")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "tool_use",
    }
    stop_reason = stop_reason_map.get(finish, "end_turn")

    usage = openai_resp.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content or [{"type": "text", "text": ""}],
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _build_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Build a Server-Sent Event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


async def _handle_streaming(
    openai_resp: httpx.Response,
    model: str,
    anthropic_msg_id: str,
) -> Any:
    """Convert OpenAI streaming response to Anthropic SSE streaming format."""
    # Emit message_start
    yield _build_sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": anthropic_msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
    )

    content_block_idx = 0
    current_tool_call_id: str | None = None
    current_tool_name: str | None = None
    in_text_block = False
    in_tool_block = False
    in_think_block = False
    accumulated_text = ""
    stream_output_tokens = 0

    async for line in openai_resp.aiter_lines():
        if not line.startswith("data: "):
            continue
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            break

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # vLLM emits a final usage-only chunk (`choices: []`) when
        # stream_options.include_usage=True — handle its usage payload
        # but skip delta/finish_reason processing.
        choices = chunk.get("choices") or []
        if not choices:
            chunk_usage = chunk.get("usage")
            if chunk_usage and chunk_usage.get("completion_tokens"):
                stream_output_tokens = chunk_usage["completion_tokens"]
            continue
        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Handle text content
        text_content = delta.get("content", "")
        if text_content:
            # Filter <think> blocks
            accumulated_text += text_content
            if "<think>" in accumulated_text and not in_think_block:
                in_think_block = True
                # Emit any text before <think>
                before = accumulated_text.split("<think>")[0]
                if before and in_text_block:
                    yield _build_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": content_block_idx,
                            "delta": {"type": "text_delta", "text": before},
                        },
                    )
                accumulated_text = accumulated_text[accumulated_text.index("<think>") :]
                continue

            if in_think_block:
                if "</think>" in accumulated_text:
                    in_think_block = False
                    after = accumulated_text.split("</think>", 1)[1]
                    accumulated_text = after
                    if not after:
                        continue
                    # Fall through to emit the text after </think>
                else:
                    continue

            if not in_text_block:
                in_text_block = True
                yield _build_sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": content_block_idx,
                        "content_block": {"type": "text", "text": ""},
                    },
                )

            yield _build_sse_event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": content_block_idx,
                    "delta": {"type": "text_delta", "text": accumulated_text},
                },
            )
            accumulated_text = ""

        # Handle tool calls
        if delta.get("tool_calls"):
            for tc_delta in delta["tool_calls"]:
                func = tc_delta.get("function", {})

                if func.get("name"):
                    # New tool call starting
                    if in_text_block:
                        yield _build_sse_event(
                            "content_block_stop",
                            {
                                "type": "content_block_stop",
                                "index": content_block_idx,
                            },
                        )
                        content_block_idx += 1
                        in_text_block = False
                    elif in_tool_block:
                        yield _build_sse_event(
                            "content_block_stop",
                            {
                                "type": "content_block_stop",
                                "index": content_block_idx,
                            },
                        )
                        content_block_idx += 1

                    current_tool_call_id = tc_delta.get("id", f"toolu_{uuid.uuid4().hex[:12]}")
                    current_tool_name = func["name"]
                    in_tool_block = True

                    yield _build_sse_event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": content_block_idx,
                            "content_block": {
                                "type": "tool_use",
                                "id": current_tool_call_id,
                                "name": current_tool_name,
                                "input": {},
                            },
                        },
                    )

                if func.get("arguments"):
                    yield _build_sse_event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": content_block_idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": func["arguments"],
                            },
                        },
                    )

        # Capture usage from stream_options.include_usage (final chunk)
        chunk_usage = chunk.get("usage")
        if chunk_usage:
            chunk_usage.get("prompt_tokens", 0)
            stream_output_tokens = chunk_usage.get("completion_tokens", 0)

        # Handle finish
        if finish_reason:
            if in_text_block or in_tool_block:
                yield _build_sse_event(
                    "content_block_stop",
                    {
                        "type": "content_block_stop",
                        "index": content_block_idx,
                    },
                )

            stop_reason_map = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
            }

            yield _build_sse_event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": stop_reason_map.get(finish_reason, "end_turn"),
                        "stop_sequence": None,
                    },
                    "usage": {"output_tokens": stream_output_tokens},
                },
            )

    yield _build_sse_event("message_stop", {"type": "message_stop"})


def create_proxy_app(vllm_url: str) -> Starlette:
    """Create the proxy Starlette application."""

    # ── Request logger (async, non-blocking) ──────────────────
    import asyncio
    from pathlib import Path

    _log_path = Path.home() / ".local" / "share" / "autosre" / "proxy-requests.jsonl"
    _log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=2000)

    async def _log_writer() -> None:
        """Background task: drain queue to JSONL file."""
        _cleanup_counter = 0
        while True:
            line = await _log_queue.get()
            try:
                # Rotate if >100MB or date changed
                if _log_path.exists():
                    stat = _log_path.stat()
                    import time as _t

                    file_date = _t.strftime("%Y-%m-%d", _t.localtime(stat.st_mtime))
                    today = _t.strftime("%Y-%m-%d")
                    if stat.st_size > 100_000_000 or file_date != today:
                        rotated = _log_path.with_suffix(f".{file_date}.jsonl")
                        _log_path.rename(rotated)
                with _log_path.open("a") as f:
                    f.write(line + "\n")

                # Cleanup rotated files older than 7 days (check every ~100 writes)
                _cleanup_counter += 1
                if _cleanup_counter >= 100:
                    _cleanup_counter = 0
                    import time as _t

                    cutoff = _t.time() - 7 * 86400
                    for old in _log_path.parent.glob("proxy-requests.*.jsonl"):
                        try:
                            if old.stat().st_mtime < cutoff:
                                old.unlink()
                        except Exception:
                            pass
            except Exception:
                pass

    def _classify_source(messages: list[Any], max_tokens: int) -> str:  # noqa: ARG001
        """Classify request source.

        All requests through this proxy are from Claude Code (coding agent).
        Meeting-scribe talks directly to vLLM, so its requests never pass here.
        """
        return "coding"

    def _prompt_prefix(messages: list[Any], limit: int = 100) -> str:
        """Extract first N chars of the user prompt."""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    texts = [
                        b.get("text", "")
                        for b in content
                        if isinstance(b, dict) and b.get("type") == "text"
                    ]
                    content = " ".join(texts)
                return str(content)[:limit]
        return ""

    def _enqueue_log(
        model: str,
        messages: list[Any],
        max_tokens: int,
        input_tokens: int,
        output_tokens: int,
        elapsed_ms: float,
        response_text: str = "",
        stream: bool = False,
        tools_count: int = 0,
        has_thinking: bool = False,
    ) -> None:
        """Non-blocking enqueue of a log entry."""
        import os as _os
        import time as _t

        tps = output_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 and output_tokens > 0 else 0
        source = _classify_source(messages, max_tokens)
        msg_count = len(messages)

        # AUTOSRE_RUN_ID is set by the eval runner so that every proxy row
        # produced during a given (run, suite) can be sliced out later.
        # When unset (normal interactive use) the field is empty string,
        # which keeps downstream jq/grep queries trivial.
        run_id = _os.environ.get("AUTOSRE_RUN_ID", "")

        entry = json.dumps(
            {
                "ts": _t.time(),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "elapsed_ms": round(elapsed_ms, 1),
                "tps": round(tps, 1),
                "max_tokens": max_tokens,
                "source": source,
                "stream": stream,
                "tools": tools_count,
                "thinking": has_thinking,
                "messages": msg_count,
                "prompt_prefix": _prompt_prefix(messages),
                "response_prefix": response_text[:200],
                "run_id": run_id,
            }
        )
        with contextlib.suppress(asyncio.QueueFull):
            _log_queue.put_nowait(entry)  # Drop rather than block

    # Start log writer as background task
    _log_writer_task: asyncio.Task[None] | None = None

    async def _ensure_log_writer() -> None:
        nonlocal _log_writer_task
        if _log_writer_task is None or _log_writer_task.done():
            _log_writer_task = asyncio.create_task(_log_writer())

    async def handle_messages(request: Request) -> JSONResponse | StreamingResponse:
        """Handle POST /v1/messages — the Anthropic Messages API endpoint."""
        body = await request.json()

        model = body.get("model", "")
        messages = body.get("messages", [])
        system = body.get("system")
        max_tokens = body.get("max_tokens", 4096)
        temperature = body.get("temperature", 0.0)
        tools = body.get("tools", [])
        stream = body.get("stream", False)
        top_p = body.get("top_p")
        stop_sequences = body.get("stop_sequences")

        # Cap max_tokens to fit within model context.
        # Query model's actual max context, cache it. We stash the value on
        # the function object itself as a simple singleton; mypy doesn't
        # model arbitrary attribute attachment to Callables, so cast around
        # it explicitly.
        model_max_ctx = cast("int", getattr(handle_messages, "_max_ctx", 0))
        if not model_max_ctx:
            try:
                async with httpx.AsyncClient(timeout=5) as ctx_client:
                    r = await ctx_client.get(f"{vllm_url}/v1/models")
                    model_max_ctx = r.json()["data"][0].get("max_model_len", 131072)
            except Exception:
                model_max_ctx = 131072
            setattr(handle_messages, "_max_ctx", model_max_ctx)  # noqa: B010

        # Estimate input tokens from full message payload size.
        # JSON-serialized messages include keys, brackets, tool schemas etc.
        # Using ~3 chars/token on the full JSON is more accurate than content-only.
        payload_chars = len(json.dumps(messages)) + len(json.dumps(system or ""))
        input_estimate = int(payload_chars / 3)
        # Reserve 10% buffer for tokenizer variance
        available = max(1024, int((model_max_ctx - input_estimate) * 0.9))
        max_tokens = min(max_tokens, available)

        # Build OpenAI request.
        #
        # Priority ordering on the shared vLLM coder (lower value = higher
        # priority in vLLM `--scheduling-policy=priority`):
        #
        #   -10  meeting-scribe live translation  (translate_vllm.py:155)
        #    10  Claude Code coding agent         (this file)
        #    20  autosre plan-review runner       (_local_provider_runner.py)
        #
        # Keep coding strictly above -10 so every live-translation request
        # preempts Claude Code when the GPU is busy. Do NOT use 0 — it is
        # the vLLM default for requests that omit the field, which makes
        # it ambiguous with third-party clients that never set priority.
        openai_req: dict[str, Any] = {
            "model": model,
            "messages": _convert_messages_to_openai(messages, system),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "priority": 10,
        }

        # Get accurate token counts in streaming responses
        if stream:
            openai_req["stream_options"] = {"include_usage": True}

        if tools:
            openai_req["tools"] = _convert_tools_to_openai(tools)

        if top_p is not None:
            openai_req["top_p"] = top_p

        if stop_sequences:
            openai_req["stop"] = stop_sequences

        anthropic_msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        _request_t0 = time.monotonic()

        await _ensure_log_writer()

        if stream:
            client = httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0))
            try:
                openai_resp = await client.send(
                    client.build_request(
                        "POST",
                        f"{vllm_url}/v1/chat/completions",
                        json=openai_req,
                    ),
                    stream=True,
                )
            except httpx.ConnectError:
                await client.aclose()
                return JSONResponse(
                    {"error": {"type": "server_error", "message": "vLLM backend unreachable"}},
                    status_code=502,
                )

            async def stream_and_close() -> Any:
                _stream_tokens = 0
                _stream_input = 0
                _has_thinking = False
                _upstream_failed = False
                try:
                    try:
                        async for chunk in _handle_streaming(openai_resp, model, anthropic_msg_id):
                            if '"text_delta"' in chunk or '"input_json_delta"' in chunk:
                                _stream_tokens += 1
                            if '"thinking"' in chunk:
                                _has_thinking = True
                            # Extract usage from message_delta (populated by stream_options)
                            if '"output_tokens"' in chunk:
                                try:
                                    evt_data = json.loads(chunk.split("data: ", 1)[1])
                                    u = evt_data.get("usage", {})
                                    if u.get("output_tokens"):
                                        _stream_tokens = u["output_tokens"]
                                except Exception:
                                    pass
                            yield chunk
                    except (
                        httpx.RemoteProtocolError,
                        httpx.ReadError,
                        httpx.ReadTimeout,
                        httpx.StreamError,
                    ) as exc:
                        # vLLM dropped the connection mid-stream — common when
                        # priority preemption evicts a running request, or when
                        # concurrent load stalls streaming (vllm#36826).
                        # Terminate the SSE stream cleanly so Claude Code sees a
                        # legitimate end-of-stream instead of a raw socket close,
                        # which undici surfaces as "socket connection was closed
                        # unexpectedly".
                        _upstream_failed = True
                        yield _build_sse_event(
                            "message_delta",
                            {
                                "type": "message_delta",
                                "delta": {
                                    "stop_reason": "end_turn",
                                    "stop_sequence": None,
                                },
                                "usage": {"output_tokens": _stream_tokens},
                            },
                        )
                        yield _build_sse_event(
                            "message_stop",
                            {
                                "type": "message_stop",
                                "upstream_error": f"{type(exc).__name__}: {exc}",
                            },
                        )
                finally:
                    _enqueue_log(
                        model=model,
                        messages=messages,
                        max_tokens=max_tokens,
                        input_tokens=_stream_input,
                        output_tokens=_stream_tokens,
                        elapsed_ms=(time.monotonic() - _request_t0) * 1000,
                        stream=True,
                        tools_count=len(tools),
                        has_thinking=_has_thinking,
                    )
                    with contextlib.suppress(Exception):
                        await openai_resp.aclose()
                    with contextlib.suppress(Exception):
                        await client.aclose()

            return StreamingResponse(
                stream_and_close(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # Non-streaming
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=10.0)) as client:
            try:
                openai_resp = await client.post(
                    f"{vllm_url}/v1/chat/completions",
                    json=openai_req,
                )
            except httpx.ConnectError:
                return JSONResponse(
                    {"error": {"type": "server_error", "message": "vLLM backend unreachable"}},
                    status_code=502,
                )

        if openai_resp.status_code != 200:
            return JSONResponse(
                {"error": {"type": "api_error", "message": openai_resp.text}},
                status_code=openai_resp.status_code,
            )

        openai_data = openai_resp.json()
        elapsed_ms = (time.monotonic() - _request_t0) * 1000
        usage = openai_data.get("usage", {})
        response_text = ""
        has_thinking = False
        choices = openai_data.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            response_text = msg.get("content", "") or ""
            has_thinking = bool(msg.get("reasoning"))
        _enqueue_log(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            elapsed_ms=elapsed_ms,
            response_text=response_text,
            stream=False,
            tools_count=len(tools),
            has_thinking=has_thinking,
        )

        return JSONResponse(
            _convert_openai_response_to_anthropic(openai_data, model),
        )

    async def handle_health(request: Request) -> JSONResponse:  # noqa: ARG001
        """Health check — proxy itself is always healthy if it can respond.

        Backend status is reported separately so the TUI can distinguish
        'proxy up, backend loading' from 'proxy down'.
        """
        backend_ok = False
        async with httpx.AsyncClient(timeout=3) as client:
            try:
                resp = await client.get(f"{vllm_url}/health")
                backend_ok = resp.status_code == 200
            except Exception:
                pass
        return JSONResponse(
            {
                "status": "ok",
                "backend": "ok" if backend_ok else "loading",
                "vllm_url": vllm_url,
            }
        )

    return Starlette(
        routes=[
            Route("/v1/messages", handle_messages, methods=["POST"]),
            Route("/health", handle_health, methods=["GET"]),
        ],
    )


def run_proxy(
    vllm_url: str = "http://localhost:8010",
    proxy_port: int = 8011,
) -> None:
    """Run the Anthropic proxy server."""
    click.echo(f"Anthropic proxy: :{proxy_port} -> {vllm_url}")
    click.echo(f"Claude Code URL: http://localhost:{proxy_port}")

    app = create_proxy_app(vllm_url)
    # timeout_keep_alive: undici (Claude Code's HTTP client) holds idle
    # keep-alive sockets in its pool and reuses them for the next request.
    # uvicorn's default 5s idle close races undici's reuse attempt and surfaces
    # as "socket connection was closed unexpectedly" on the client. Bump well
    # above any reasonable inter-request gap so the server never closes first.
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=proxy_port,
        log_level="warning",
        timeout_keep_alive=600,
    )


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8011
    vllm = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8010"
    run_proxy(vllm, port)
