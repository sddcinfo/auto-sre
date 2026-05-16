"""Local-model plan-review runner.

Invoked as a subprocess by ``autosre.review.chain.run_chain()`` when the
chain contains ``"local"``. The chain passes the prompt as the last argv
element (matching the contract of the CLI providers — ``codex exec <prompt>``,
``gemini --model X -p <prompt>``, ``claude --print <prompt>``).

This runner:

1. Discovers the currently active autosre backend via ``load_active_state()``.
2. Resolves the served model id. Priority:
   a. ``AUTOSRE_REVIEW_MODEL`` env var (set by ``autosre claude`` after it
      already resolved the id via ``backend.get_claude_model_arg()``).
   b. ``GET /v1/models`` on the backend's base URL and use ``data[0].id``
      (matches the pattern at ``autosre/backends/anthropic_proxy.py:587``
      and ``autosre/backends/vllm.py:719``).
   c. Backend's ``get_claude_model_arg(active_state["model"])`` mapping.
3. POSTs to ``{base_url}/v1/chat/completions`` with the prompt as the user
   message, under a short system prompt reiterating the JSON output contract.
4. Strips any ``<think>…</think>`` blocks (qwen3 reasoning parser wraps
   thinking output there) before printing.
5. Prints the model response to stdout. Exits ``0`` on HTTP-2xx, non-zero on
   any error so the chain executor can fall through to the next provider.

The chain's ``_parse_response_json()`` extracts the first ``{…}`` substring
from stdout, so there's no need for us to strictly JSON-validate the model's
output — we just need to get it through without corrupting it.

Backend-agnostic: works against any autosre backend that exposes an
OpenAI-compatible ``/v1/chat/completions`` endpoint (vLLM, Ollama, llamacpp).
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import httpx

from autosre.backends.base import BackendType, load_active_state

_DEFAULT_TIMEOUT = 600
_DEFAULT_TEMPERATURE = 0.1
_DEFAULT_MAX_TOKENS = 4096
_DEFAULT_CONNECT_TIMEOUT = 10


_SYSTEM_PROMPT = """\
You are a plan reviewer. The user will show you an implementation plan.

Return your analysis as a single JSON object with this exact schema:
{"findings": [{"severity": "P0|P1|P2", "title": "...", "description": "...", "recommendation": "..."}], "questions": ["..."]}

Severity levels:
- P0 (Critical): data loss, security vulnerability, system outage
- P1 (High): incorrect behavior, significant technical debt
- P2 (Medium): design improvements, missing edge cases

Only flag genuine issues with concrete recommendations. If the plan looks good,
return {"findings": [], "questions": []}.

Output ONLY the JSON object. No markdown fences, no narrative, no prose.
"""


def _strip_think_blocks(text: str) -> str:
    """Remove ``<think>…</think>`` blocks emitted by qwen3 reasoning parser.

    Non-greedy match across newlines so multiple think blocks are all stripped.
    Leaves the rest of the text unchanged.
    """
    return re.sub(r"<think>[\s\S]*?</think>", "", text)


def _get_base_url() -> str:
    """Determine the OpenAI-compatible base URL for the active backend.

    Raises ``RuntimeError`` if no backend is running.
    """
    active = load_active_state()
    if not active:
        raise RuntimeError(
            "No active autosre backend. Start one with 'autosre start' first.",
        )

    api_host = active.get("api_host", "localhost")
    api_port = active.get("api_port")
    if not api_port:
        raise RuntimeError(f"active backend has no api_port: {active!r}")

    return f"http://{api_host}:{api_port}"


def _resolve_model_id(base_url: str, active: dict[str, Any]) -> str:
    """Resolve the served model id.

    Priority:
      1. ``AUTOSRE_REVIEW_MODEL`` env var.
      2. ``GET {base_url}/v1/models`` first entry.
      3. Backend's ``get_claude_model_arg(short_key)`` lookup.
    """
    env_model = os.environ.get("AUTOSRE_REVIEW_MODEL")
    if env_model:
        return env_model

    try:
        with httpx.Client(timeout=5) as client:
            resp = client.get(f"{base_url}/v1/models")
            if resp.status_code == 200:
                data = resp.json()
                models = data.get("data", [])
                if models:
                    model_id: str = models[0]["id"]
                    return model_id
    except (httpx.HTTPError, KeyError, ValueError):
        pass

    # Last resort: instantiate the backend and ask it to resolve the recipe key.
    try:
        from autosre.backends import get_backend

        backend_name = str(active.get("backend", ""))
        backend_type = BackendType(backend_name)
        backend = get_backend(backend_type, active_state=active)
        short_key = str(active.get("model", backend.default_model))
        return backend.get_claude_model_arg(short_key)
    except (ValueError, RuntimeError, KeyError) as e:
        raise RuntimeError(f"could not resolve model id: {e}") from e


def _call_model(base_url: str, model_id: str, prompt: str) -> str:
    """POST to ``/v1/chat/completions`` and return the assistant content."""
    timeout = int(os.environ.get("AUTOSRE_REVIEW_TIMEOUT", _DEFAULT_TIMEOUT))
    temperature = float(os.environ.get("AUTOSRE_REVIEW_TEMPERATURE", _DEFAULT_TEMPERATURE))
    max_tokens = int(os.environ.get("AUTOSRE_REVIEW_MAX_TOKENS", _DEFAULT_MAX_TOKENS))

    payload: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # vLLM honors this when --scheduling-policy=priority is set (lower
        # value = higher priority). Ordering on the shared coder:
        #   -10  meeting-scribe live translation
        #    10  Claude Code coding agent
        #    20  plan-review runner (this)
        # Plan reviews are batch work and preempted by both live translation
        # and interactive Claude Code.
        "priority": 20,
    }

    http_timeout = httpx.Timeout(float(timeout), connect=_DEFAULT_CONNECT_TIMEOUT)
    with httpx.Client(timeout=http_timeout) as client:
        resp = client.post(f"{base_url}/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        content: str = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"unexpected response shape: {data!r}") from e

    return content


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m autosre.review._local_provider_runner``.

    Reads the prompt from argv (last argument) to match the contract of the
    CLI providers (``codex exec <prompt>``, ``gemini -p <prompt>``,
    ``claude --print <prompt>``). Falls back to reading from stdin if no
    argv prompt is given.
    """
    if argv is None:
        argv = sys.argv[1:]

    prompt = argv[-1] if argv else sys.stdin.read()

    if not prompt.strip():
        print("error: empty prompt", file=sys.stderr)
        return 2

    try:
        active = load_active_state()
        if not active:
            print(
                "local provider error: No active autosre backend. "
                "Start one with 'autosre start' first.",
                file=sys.stderr,
            )
            return 1

        base_url = _get_base_url()
        model_id = _resolve_model_id(base_url, active)
        content = _call_model(base_url, model_id, prompt)
    except httpx.HTTPError as e:
        print(f"local provider HTTP error: {e}", file=sys.stderr)
        return 1
    except RuntimeError as e:
        print(f"local provider error: {e}", file=sys.stderr)
        return 1

    cleaned = _strip_think_blocks(content)
    sys.stdout.write(cleaned)
    if not cleaned.endswith("\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
