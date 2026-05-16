# ruff: noqa: RUF001, TC003
"""Workload definitions for the concurrent perf harness.

Two workloads share a single vLLM instance:

- **Translation** — short EN↔JA sentences at ``priority=-10``, mirroring
  meeting-scribe's live translation path.
- **Coding** — prefill-heavy requests (~800-token system + 20 tool schemas
  + rotating user prompt) at ``priority=10``, mirroring Claude Code via
  the Anthropic proxy.

Everything here is pure data — no I/O. The harness consumes
:class:`Workload` instances to build OpenAI chat payloads.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Workload:
    """A driver specification for one of the two concurrent workloads."""

    name: str
    priority: int
    max_tokens: int
    # Callable returns an (openai_payload, label) tuple each invocation.
    # Round-robins through an internal prompt pool so successive requests
    # don't all hit the same prefix cache entry.
    next_payload: Callable[[], tuple[dict[str, Any], str]]


# ── Translation corpus (copied verbatim from meeting-scribe) ─────
#
# Source: repos/meeting-scribe/benchmarks/translation_benchmark.py:28-52.
# Kept in-tree so this harness has zero runtime dependency on the
# meeting-scribe package. If the corpus drifts upstream, we intentionally
# do NOT auto-sync — the whole point of a regression harness is a frozen
# input.

_TRANSLATION_CORPUS: list[dict[str, str]] = [
    {
        "ja": "今日の会議の議題について確認しましょう。",
        "en": "Let's confirm the agenda for today's meeting.",
    },
    {
        "ja": "この提案について何かご質問はありますか？",
        "en": "Do you have any questions about this proposal?",
    },
    {"ja": "次のステップを決めましょう。", "en": "Let's decide on the next steps."},
    {"ja": "四半期の売上目標を達成しました。", "en": "We achieved our quarterly sales target."},
    {
        "ja": "新しいプロジェクトの予算を承認する必要があります。",
        "en": "We need to approve the budget for the new project.",
    },
    {
        "ja": "来週のデモに向けて準備を進めています。",
        "en": "We are preparing for next week's demo.",
    },
    {"ja": "APIのレスポンスタイムが改善されました。", "en": "The API response time has improved."},
    {
        "ja": "データベースのマイグレーションは明日実行します。",
        "en": "We will run the database migration tomorrow.",
    },
    {
        "ja": "本番環境にデプロイする前にテストを完了してください。",
        "en": "Please complete testing before deploying to production.",
    },
    {
        "ja": "すみません、もう一度言っていただけますか？",
        "en": "Excuse me, could you say that again?",
    },
    {"ja": "その点については同意します。", "en": "I agree with that point."},
    {"ja": "詳しく説明していただけますか？", "en": "Could you explain in more detail?"},
    {
        "ja": "この問題を解決するためには、チーム全体で協力する必要があると考えています。",
        "en": "I believe we need the whole team to cooperate to solve this problem.",
    },
    {
        "ja": "スケジュールの遅延を最小限に抑えるため、優先順位を再検討しましょう。",
        "en": "Let's re-examine priorities to minimize schedule delays.",
    },
    {
        "ja": "お客様からのフィードバックに基づいて、UIを改善する計画です。",
        "en": "We plan to improve the UI based on customer feedback.",
    },
]


def _translation_prompt(source_name: str, target_name: str) -> str:
    # Mirrors ``meeting_scribe.languages.get_translation_prompt`` so the
    # harness sends byte-identical payloads to what meeting-scribe sends
    # in production. Copied to avoid a cross-repo import.
    return (
        f"You are a professional {source_name}-to-{target_name} translator. "
        f"Translate the following {source_name} text into natural, fluent {target_name}. "
        f"Preserve the meaning, tone, and context. "
        f"Return only the translation, no explanation or commentary."
    )


_JA_EN_SYSTEM = _translation_prompt("Japanese", "English")
_EN_JA_SYSTEM = _translation_prompt("English", "Japanese")


def _translation_cycle() -> Iterator[tuple[dict[str, Any], str]]:
    """Yield (payload, label) forever, alternating JA→EN and EN→JA."""
    for item in itertools.cycle(_TRANSLATION_CORPUS):
        yield (
            {
                "messages": [
                    {"role": "system", "content": _JA_EN_SYSTEM},
                    {"role": "user", "content": item["ja"]},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
                "stream": True,
                "stream_options": {"include_usage": True},
                "priority": -10,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            f"ja_en:{item['ja'][:20]}",
        )
        yield (
            {
                "messages": [
                    {"role": "system", "content": _EN_JA_SYSTEM},
                    {"role": "user", "content": item["en"]},
                ],
                "temperature": 0.0,
                "max_tokens": 256,
                "stream": True,
                "stream_options": {"include_usage": True},
                "priority": -10,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            f"en_ja:{item['en'][:20]}",
        )


# ── Coding workload ─────────────────────────────────────────────

# ~800-token system prompt modeled on Claude Code's actual runtime
# prompt. Deliberately verbose to exercise prefill like the real thing.
_CODING_SYSTEM = """You are an expert software engineer working in an existing Python codebase.
You have access to tools for reading files, editing files, searching the codebase,
and running shell commands. When the user asks you to make a change:

1. Start by reading the relevant files to understand the current state.
2. Search for related code patterns, callers, and tests that might be affected.
3. Plan the minimum change that addresses the user's request — do not introduce
   speculative abstractions, premature optimizations, or unrelated cleanup.
4. Apply edits using the Edit tool for targeted changes or Write for new files.
5. Run the project's lint, type-check, and test commands to validate your change.
6. Summarize what you did in 1–2 sentences; do not narrate internal deliberation.

Codebase conventions:
- Python 3.11+, strict mypy, 100-char lines, double-quoted strings.
- Prefer dedicated tools (Read/Edit/Grep/Glob) over Bash when possible.
- Never use git commands destructively without explicit user authorization.
- Reuse existing utilities before writing new ones; search first.
- Test with pytest; lint with ruff; format with ruff format.
- Follow XDG base-directory conventions for data/config/state paths.
- Do not commit unless explicitly asked. Do not push unless explicitly asked.

When writing code:
- Default to no comments. Add one only when the WHY is non-obvious.
- Never write multi-paragraph docstrings or multi-line comment blocks.
- Do not explain WHAT the code does — well-named identifiers already do that.
- Do not reference the current task or PR — that context belongs in the PR description.
- Trust internal code and framework guarantees; only validate at system boundaries.

When debugging:
- Find the root cause, not a symptom. Do not paper over errors with try/except.
- Do not blame the browser cache, CDN, or other opaque external state until
  you have ruled out actual code paths.
- If you hit an obstacle, investigate rather than bypassing safety checks."""


def _coding_tools() -> list[dict[str, Any]]:
    """Twenty realistic tool schemas mirroring Claude Code's built-ins.

    The exact count matters for prefill token budget — keep this list
    stable across baseline runs.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read a file from the local filesystem.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute file path."},
                        "offset": {"type": "integer", "description": "Line to start reading from."},
                        "limit": {"type": "integer", "description": "Number of lines to read."},
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file, creating or overwriting it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Edit",
                "description": "Replace an exact string in a file with a new string.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_string": {"type": "string"},
                        "new_string": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command and return stdout and stderr.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "description": {"type": "string"},
                        "timeout": {"type": "number"},
                    },
                    "required": ["command", "description"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Glob",
                "description": "Find files matching a glob pattern.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Grep",
                "description": "Search file contents with a regular expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                        "glob": {"type": "string"},
                        "output_mode": {"type": "string"},
                        "context": {"type": "integer"},
                    },
                    "required": ["pattern"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "WebFetch",
                "description": "Fetch a URL and return the rendered markdown.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "prompt": {"type": "string"},
                    },
                    "required": ["url", "prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "WebSearch",
                "description": "Search the web and return results.",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "TodoWrite",
                "description": "Create or update a structured task list.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "todos": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "content": {"type": "string"},
                                    "status": {"type": "string"},
                                    "activeForm": {"type": "string"},
                                },
                                "required": ["content", "status", "activeForm"],
                            },
                        }
                    },
                    "required": ["todos"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "NotebookEdit",
                "description": "Edit a specific cell in a Jupyter notebook.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "notebook_path": {"type": "string"},
                        "cell_id": {"type": "string"},
                        "new_source": {"type": "string"},
                        "edit_mode": {"type": "string"},
                    },
                    "required": ["notebook_path", "new_source"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_status",
                "description": "Run git status --porcelain in the working tree.",
                "parameters": {"type": "object", "properties": {}},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_diff",
                "description": "Show unstaged or staged diff.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "staged": {"type": "boolean"},
                        "path": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "git_log",
                "description": "Show recent commits.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "n": {"type": "integer"},
                        "path": {"type": "string"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_tests",
                "description": "Run the project test suite (pytest).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "verbose": {"type": "boolean"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_lint",
                "description": "Run ruff check on a path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_format",
                "description": "Run ruff format on a path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_typecheck",
                "description": "Run mypy on a path.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List the contents of a directory.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "describe_symbol",
                "description": "Return the definition and references of a named symbol.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "request_review",
                "description": "Submit the current change set for external review.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "provider": {"type": "string"},
                    },
                    "required": ["summary"],
                },
            },
        },
    ]


_CODING_PROMPTS: list[str] = [
    "Refactor this quicksort implementation to use an iterative approach with an explicit stack. Keep the function signature identical and add one-line doctests for the empty, single-element, and already-sorted cases.",
    "A user reports that our date-parsing helper drops the timezone when the input string ends in 'Z'. Find the helper, diagnose the root cause, and write a failing test that demonstrates the bug before fixing it.",
    "Write a Python function that takes a list of dicts and groups them by a given key, returning a dict-of-lists. Handle the case where the key is missing on some dicts by grouping them under a None bucket.",
    "Explain how Python's asyncio.Semaphore differs from threading.Semaphore in terms of fairness and release ordering. Give a short code example that demonstrates the difference.",
    "Given a directed graph represented as adjacency lists, write a function that returns one topological order if the graph is a DAG, or raises a clear exception naming any node that participates in a cycle.",
    "I have a pytest fixture that creates a temporary sqlite database. Rewrite it as an async fixture using aiosqlite, keeping the same teardown guarantees. Show the before and after.",
]


def _coding_cycle() -> Iterator[tuple[dict[str, Any], str]]:
    tools = _coding_tools()
    for prompt in itertools.cycle(_CODING_PROMPTS):
        yield (
            {
                "messages": [
                    {"role": "system", "content": _CODING_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 1200,
                "stream": True,
                "stream_options": {"include_usage": True},
                "priority": 10,
                "tools": tools,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            f"coding:{prompt[:32]}",
        )


def _make_workload(
    name: str,
    priority: int,
    max_tokens: int,
    factory: Callable[[], Iterator[tuple[dict[str, Any], str]]],
) -> Workload:
    it = factory()

    def _next() -> tuple[dict[str, Any], str]:
        return next(it)

    return Workload(name=name, priority=priority, max_tokens=max_tokens, next_payload=_next)


TRANSLATION_WORKLOAD = _make_workload("translation", -10, 256, _translation_cycle)
CODING_WORKLOAD = _make_workload("coding", 10, 1200, _coding_cycle)
