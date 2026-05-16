"""Provider chain executor — plan-review fallback chain.

Writes per-cycle log files with start/end timestamps under
``autosre.paths.review_log_dir()`` for post-mortem inspection.

The ``"local"`` provider maps to an in-process Python runner that
targets whichever local backend is active (vLLM / Ollama / llamacpp) via
its OpenAI-compatible ``/v1/chat/completions`` endpoint. See
``autosre.review._local_provider_runner``.

Chain overrides via ``AUTOSRE_REVIEW_CHAIN`` env var (comma-separated
provider names). When set by ``autosre claude``, the review loop runs
against local models without any cloud fallback.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from autosre import paths

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Upstream provider model constants (used by CLI subprocess commands)
# ---------------------------------------------------------------------------

GEMINI_MODEL = "auto-gemini-3"
CLAUDE_SONNET_MODEL = "claude-sonnet-4-6"
CODEX_MODEL = "gpt-5.4"


# ---------------------------------------------------------------------------
# Logging — append log + per-cycle detail files
# ---------------------------------------------------------------------------


def _review_log_file() -> Path:
    return paths.data_dir() / "review-chain.log"


def _review_log_dir() -> Path:
    return paths.review_log_dir()


# Generous caps — large enough for any real review, prevents runaway logs
MAX_PROMPT_LOG = 512_000  # 512KB
MAX_STDOUT_LOG = 2_097_152  # 2MB
MAX_STDERR_LOG = 512_000  # 512KB


def _file_log(msg: str) -> None:
    """Append to persistent log file (in addition to stdlib logging)."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        path = _review_log_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(f"[{ts}] {msg}\n")
    except OSError:
        pass


def _save_cycle_log(
    provider: str,
    prompt: str,
    stdout: str,
    stderr: str,
    returncode: int | None,
    elapsed: float,
    findings: list[dict[str, Any]] | None,
    *,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    """Save a detailed per-cycle JSON log file for debugging.

    Args:
        metadata: Optional dict with plan_path, plan_filename, plan_title, iteration, project.

    Returns the log file path, or None if saving failed.
    """
    meta: dict[str, Any] = dict(metadata) if metadata else {}
    # Stable filename fragment: hash of the abs plan path + stem, mirroring
    # cli_plan.py's state keying so cross-repo collisions don't happen here either.
    plan_path_raw = meta.get("plan_path") or ""
    plan_filename = meta.get("plan_filename", "") or ""
    plan_stem = plan_filename.removesuffix(".md") or "unknown"
    if plan_path_raw:
        import hashlib

        path_hash = hashlib.sha256(str(plan_path_raw).encode()).hexdigest()[:12]
        plan_stem = f"{path_hash}_{plan_stem}"
    iteration = meta.get("iteration", 0)
    ts = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{plan_stem}_{iteration}_{ts}_{provider}.json"

    try:
        log_dir = _review_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / filename
        log_data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "provider": provider,
            "elapsed_seconds": round(elapsed, 2),
            "returncode": returncode,
            "error": error,
            # Plan metadata
            "plan_path": meta.get("plan_path"),
            "plan_filename": meta.get("plan_filename"),
            "plan_title": meta.get("plan_title"),
            "iteration": iteration,
            "project": meta.get("project"),
            # Full content with generous caps
            "prompt_size": len(prompt),
            "prompt": prompt[:MAX_PROMPT_LOG],
            "stdout_size": len(stdout),
            "stdout": stdout[:MAX_STDOUT_LOG],
            "stderr_size": len(stderr),
            "stderr": stderr[:MAX_STDERR_LOG],
            "findings_count": len(findings) if findings else 0,
            "findings": findings,
        }
        log_path.write_text(json.dumps(log_data, indent=2, ensure_ascii=False) + "\n")
        _file_log(f"cycle log saved: {log_path}")
        return log_path
    except OSError as e:
        _file_log(f"failed to save cycle log: {e}")
        return None


# ---------------------------------------------------------------------------
# Provider definitions
# ---------------------------------------------------------------------------

PROVIDER_CMDS: dict[str, list[str]] = {
    # Codex — primary reviewer, gpt-5.4 at xhigh reasoning. We have codex credit
    # and the user explicitly requested "claude plans, codex reviews". The
    # TOML-style -c overrides make the review pipeline self-contained so it
    # doesn't depend on the user's ~/.codex/config.toml defaults changing.
    "codex": [
        "codex",
        "exec",
        "-c",
        'model="gpt-5.4"',
        "-c",
        'model_reasoning_effort="xhigh"',
        "--skip-git-repo-check",
    ],
    "gemini": ["gemini", "--model", GEMINI_MODEL, "-p"],
    "claude": ["claude", "--print", "--model", CLAUDE_SONNET_MODEL],
    # Local provider — runs our in-repo Python runner as a subprocess so
    # it obeys the same "prompt as last arg, stdout is the response, non-zero
    # exit = failure" contract as the CLI providers. Used as fallback when
    # codex is unavailable or when the user explicitly opts into local-only.
    "local": [sys.executable, "-m", "autosre.review._local_provider_runner"],
}

# Default chains per review mode
DEFAULT_CHAINS: dict[str, list[str]] = {
    # Plan review: codex gpt-5.4/xhigh is primary, local model is offline fallback,
    # gemini/claude CLIs are further fallback for redundancy.
    "plan": ["codex", "local", "gemini", "claude"],
    # Commit review (not currently wired into hooks, preserved from upstream
    # ordering in case we add git-commit review later).
    "commit": ["codex", "gemini", "claude", "local"],
}

# Per-provider timeouts based on observed behavior
PROVIDER_TIMEOUTS: dict[str, int] = {
    "codex": 900,  # codex exec gpt-5.4/xhigh can take 5-15 minutes on complex plans
    "gemini": 180,  # Normally <60s
    "claude": 180,  # Normally <90s
    "local": 600,  # Local reasoning models can be slow at long context
}
DEFAULT_PROVIDER_TIMEOUT = 300

# Chain-level budget — ensures deterministic termination. Bumped from upstream's
# 720s to 1200s to accommodate a full codex gpt-5.4/xhigh attempt plus fallbacks.
MAX_CHAIN_SECONDS = 1200


# ---------------------------------------------------------------------------
# Chain-override env var
# ---------------------------------------------------------------------------


def _chain_from_env() -> list[str] | None:
    """Return chain override from ``AUTOSRE_REVIEW_CHAIN`` if set.

    Comma-separated provider names. Unknown providers are dropped with a
    warning but valid entries are still used.
    """
    raw = os.environ.get("AUTOSRE_REVIEW_CHAIN")
    if not raw:
        return None
    names = [n.strip() for n in raw.split(",") if n.strip()]
    valid = [n for n in names if n in PROVIDER_CMDS]
    invalid = [n for n in names if n not in PROVIDER_CMDS]
    if invalid:
        logger.warning(
            "chain_env_unknown_providers: invalid=%r allowed=%r",
            invalid,
            list(PROVIDER_CMDS),
        )
    return valid or None


# ---------------------------------------------------------------------------
# Chain result
# ---------------------------------------------------------------------------


@dataclass
class ProviderAttempt:
    """Record of a single provider attempt."""

    provider: str
    success: bool
    elapsed_seconds: float
    returncode: int | None = None
    stdout_size: int = 0
    stderr_size: int = 0
    error: str | None = None
    findings_count: int | None = None
    raw_stdout: str = ""


@dataclass
class ChainResult:
    """Result of a provider chain execution."""

    provider: str
    findings: list[dict[str, Any]] | None
    questions: list[str] | None
    raw_output: str
    elapsed_seconds: float
    attempts: list[ProviderAttempt] = field(default_factory=list)

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    def format_findings(self) -> str:
        """Format findings as readable text for hook output."""
        if not self.findings:
            return ""

        lines = [f"Plan review findings (via {self.provider}):\n"]
        counts: dict[str, int] = {}

        for f in self.findings:
            sev = f.get("severity", "P2")
            title = f.get("title", "Untitled")
            desc = f.get("description", "")
            rec = f.get("recommendation", "")
            counts[sev] = counts.get(sev, 0) + 1

            lines.append(f"  {sev}: {title}")
            if desc:
                lines.append(f"    {desc}")
            if rec:
                lines.append(f"    Recommendation: {rec}")
            lines.append("")

        summary_parts = [f"{counts.get(s, 0)} {s}" for s in ("P0", "P1", "P2") if counts.get(s)]
        lines.insert(1, f"  ({', '.join(summary_parts)})\n")

        lines.append(
            "BLOCKING: Address ALL findings above (P0, P1, and P2) in the plan, "
            "then call ExitPlanMode again for re-review.",
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core chain executor
# ---------------------------------------------------------------------------


def run_chain(
    prompt: str,
    *,
    chain: list[str] | None = None,
    mode: str = "plan",
    metadata: dict[str, Any] | None = None,
) -> ChainResult:
    """Execute a provider chain, trying each provider in order until one succeeds.

    Args:
        prompt: The full prompt to send to providers.
        chain: Ordered list of provider names. Defaults to ``AUTOSRE_REVIEW_CHAIN``
            env var if set, otherwise ``DEFAULT_CHAINS[mode]``.
        mode: Review mode ("plan" or "commit") — determines default chain.
        metadata: Plan metadata dict (plan_path, plan_filename, plan_title, iteration).

    Returns:
        ChainResult with findings (or ``None`` if all providers failed).
    """
    if chain is None:
        chain = _chain_from_env() or DEFAULT_CHAINS.get(mode, DEFAULT_CHAINS["plan"])

    meta: dict[str, Any] = dict(metadata) if metadata else {}
    plan_name = meta.get("plan_filename", "").replace(".md", "") or "unknown"

    total_start = time.monotonic()
    attempts: list[ProviderAttempt] = []

    _file_log("=" * 60)
    _file_log(
        f"chain start: mode={mode}, chain={chain}, prompt_size={len(prompt)}, "
        f"plan={plan_name}, budget={MAX_CHAIN_SECONDS}s",
    )
    logger.info(
        "chain_start: mode=%r chain=%r prompt_size=%d plan=%r",
        mode,
        chain,
        len(prompt),
        plan_name,
    )

    for provider_name in chain:
        # Check chain-level time budget
        chain_elapsed = time.monotonic() - total_start
        if chain_elapsed >= MAX_CHAIN_SECONDS:
            _file_log(
                f"chain budget exhausted ({chain_elapsed:.0f}s >= {MAX_CHAIN_SECONDS}s), "
                f"skipping {provider_name}",
            )
            logger.warning(
                "chain_budget_exhausted: elapsed=%.1f skipped=%r",
                chain_elapsed,
                provider_name,
            )
            break

        if provider_name not in PROVIDER_CMDS:
            _file_log(f"skipping unknown provider: {provider_name}")
            logger.warning("unknown_provider: %r", provider_name)
            attempts.append(
                ProviderAttempt(
                    provider=provider_name,
                    success=False,
                    elapsed_seconds=0,
                    error=f"unknown provider (allowed: {', '.join(PROVIDER_CMDS)})",
                ),
            )
            continue

        cmd_prefix = PROVIDER_CMDS[provider_name]
        cli_binary = cmd_prefix[0]

        # sys.executable is always present; only check external binaries.
        if cli_binary != sys.executable and not shutil.which(cli_binary):
            _file_log(f"skipping {provider_name} ({cli_binary} not on PATH)")
            logger.info("provider_skip: provider=%r reason=not_on_path", provider_name)
            attempts.append(
                ProviderAttempt(
                    provider=provider_name,
                    success=False,
                    elapsed_seconds=0,
                    error="not on PATH",
                ),
            )
            continue

        provider_timeout = PROVIDER_TIMEOUTS.get(provider_name, DEFAULT_PROVIDER_TIMEOUT)

        # Cap provider timeout by remaining chain budget
        remaining = MAX_CHAIN_SECONDS - (time.monotonic() - total_start)
        effective_timeout = min(provider_timeout, int(remaining))

        attempt = _try_provider(provider_name, prompt, timeout=effective_timeout, metadata=meta)
        attempts.append(attempt)

        if attempt.success and attempt.findings_count is not None:
            elapsed = time.monotonic() - total_start
            _file_log(
                f"chain complete: provider={provider_name}, "
                f"findings={attempt.findings_count}, elapsed={elapsed:.1f}s",
            )
            logger.info(
                "chain_complete: provider=%r findings=%d elapsed=%.1f",
                provider_name,
                attempt.findings_count,
                elapsed,
            )

            raw = attempt.raw_stdout
            findings, questions = _parse_response_json(raw)

            return ChainResult(
                provider=provider_name,
                findings=findings,
                questions=questions,
                raw_output=raw,
                elapsed_seconds=elapsed,
                attempts=attempts,
            )

    # All providers failed or returned no findings
    elapsed = time.monotonic() - total_start
    _file_log(f"chain exhausted: all providers failed or clean, elapsed={elapsed:.1f}s")
    logger.warning("chain_exhausted: elapsed=%.1f attempts=%d", elapsed, len(attempts))

    return ChainResult(
        provider="",
        findings=None,
        questions=None,
        raw_output="",
        elapsed_seconds=elapsed,
        attempts=attempts,
    )


# ---------------------------------------------------------------------------
# Single provider attempt
# ---------------------------------------------------------------------------


def _try_provider(
    name: str,
    prompt: str,
    *,
    timeout: int,
    metadata: dict[str, Any] | None = None,
) -> ProviderAttempt:
    """Run a single provider and return the attempt record."""
    if name not in PROVIDER_CMDS:
        raise ValueError(f"Unknown provider: {name}. Allowed: {', '.join(PROVIDER_CMDS)}")
    cmd_prefix = PROVIDER_CMDS[name]
    cmd = [*cmd_prefix, prompt]

    _file_log(f"trying {name}: timeout={timeout}s")
    logger.info("provider_start: provider=%r timeout=%d", name, timeout)

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start

        _file_log(
            f"{name} completed: rc={result.returncode}, "
            f"stdout={len(result.stdout)} chars, stderr={len(result.stderr)} chars, "
            f"elapsed={elapsed:.1f}s",
        )
        logger.info(
            "provider_complete: provider=%r rc=%d stdout=%d stderr=%d elapsed=%.1f",
            name,
            result.returncode,
            len(result.stdout),
            len(result.stderr),
            elapsed,
        )

        if result.returncode != 0:
            stderr_preview = result.stderr[:500]
            _file_log(f"{name} FAILED stderr: {stderr_preview}")
            logger.error(
                "provider_failed: provider=%r rc=%d stderr=%r",
                name,
                result.returncode,
                stderr_preview,
            )
            _save_cycle_log(
                name,
                prompt,
                result.stdout,
                result.stderr,
                result.returncode,
                elapsed,
                None,
                error=stderr_preview[:200],
                metadata=metadata,
            )

            return ProviderAttempt(
                provider=name,
                success=False,
                elapsed_seconds=elapsed,
                returncode=result.returncode,
                stdout_size=len(result.stdout),
                stderr_size=len(result.stderr),
                error=f"rc={result.returncode}: {stderr_preview[:200]}",
            )

        # Parse findings
        findings = _parse_findings_json(result.stdout)
        findings_count = len(findings) if findings else 0

        if findings_count > 0:
            _file_log(f"{name} returned {findings_count} findings")
            logger.info("provider_findings: provider=%r count=%d", name, findings_count)
        else:
            _file_log(f"{name} returned no findings (clean or unparseable)")
            if result.stdout:
                _file_log(f"{name} stdout preview: {result.stdout[:300]}")
            logger.info(
                "provider_no_findings: provider=%r stdout_preview=%r",
                name,
                result.stdout[:200],
            )

        # Save detailed cycle log (full prompt + response + metadata)
        _save_cycle_log(
            name,
            prompt,
            result.stdout,
            result.stderr,
            0,
            elapsed,
            findings,
            metadata=metadata,
        )

        return ProviderAttempt(
            provider=name,
            success=True,
            elapsed_seconds=elapsed,
            returncode=0,
            stdout_size=len(result.stdout),
            stderr_size=len(result.stderr),
            findings_count=findings_count,
            raw_stdout=result.stdout,
        )

    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        _file_log(f"{name} TIMED OUT after {elapsed:.1f}s (limit={timeout}s)")
        logger.error(
            "provider_timeout: provider=%r elapsed=%.1f timeout=%d",
            name,
            elapsed,
            timeout,
        )
        _save_cycle_log(
            name,
            prompt,
            "",
            "",
            None,
            elapsed,
            None,
            error=f"timed out after {elapsed:.1f}s",
            metadata=metadata,
        )
        return ProviderAttempt(
            provider=name,
            success=False,
            elapsed_seconds=elapsed,
            error=f"timed out after {elapsed:.1f}s",
        )

    except (FileNotFoundError, OSError) as e:
        elapsed = time.monotonic() - start
        _file_log(f"{name} OS ERROR: {e}")
        logger.error("provider_os_error: provider=%r error=%r", name, str(e))
        _save_cycle_log(name, prompt, "", "", None, elapsed, None, error=str(e), metadata=metadata)
        return ProviderAttempt(
            provider=name,
            success=False,
            elapsed_seconds=elapsed,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def _parse_response_json(
    raw: str,
) -> tuple[list[dict[str, Any]] | None, list[str] | None]:
    """Extract findings and questions from provider output.

    Providers may wrap the JSON in markdown fences or narrative text. This
    helper finds the largest brace-matched substring and parses that.
    Returns ``(findings, questions)`` — either may be ``None``.
    """
    if not raw or not raw.strip():
        return None, None

    json_match = re.search(r"\{[\s\S]*\}", raw)
    if not json_match:
        return None, None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None, None

    findings = data.get("findings", []) or None
    questions = data.get("questions", []) or None
    return findings, questions


def _parse_findings_json(raw: str) -> list[dict[str, Any]] | None:
    """Extract findings array from provider output. Convenience wrapper."""
    findings, _ = _parse_response_json(raw)
    return findings
