"""Recipe file guard — blocks unvalidated changes to performance-sensitive parameters.

Protected files:
  - ``*/backends/recipes/*.yaml``  (auto-sre vLLM recipes)
  - ``*/meeting_scribe/recipes/*.yaml``  (meeting-scribe model recipes)
  - ``*/meeting_scribe/stage_configs/*.yaml``  (meeting-scribe stage configs)

Protected parameters (root-level or nested under ``stage_args[*].engine_args``):
  gpu_memory_utilization, max_num_seqs, max_num_batched_tokens,
  kv_cache_dtype, quantization, max_model_len, tensor_parallel

Protected extra_args flags:
  --enforce-eager, --enable-prefix-caching, --enable-chunked-prefill,
  --scheduling-policy, --load-format, --gpu-memory-utilization, etc.

The guard is used by:
  1. PreToolUse hook (guard.py) — blocks Edit/Write on recipe files
  2. Pre-commit hook (recipe_precommit.py) — blocks commits without baselines
  3. CI check (recipe_ci_check.py) — blocks PRs without baselines
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from autosre import paths

logger = logging.getLogger(__name__)

# ── Protected parameter definitions ──────────────────────────────

_PERF_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "gpu_memory_utilization",
        "max_num_seqs",
        "max_num_batched_tokens",
        "kv_cache_dtype",
        "quantization",
        "max_model_len",
        "tensor_parallel",
    }
)

_PERF_SENSITIVE_EXTRA_ARG_PREFIXES: frozenset[str] = frozenset(
    {
        "--enforce-eager",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--scheduling-policy",
        "--load-format",
        "--gpu-memory-utilization",
        "--max-num-seqs",
        "--max-num-batched-tokens",
        "--kv-cache-dtype",
        "--quantization",
        "--max-model-len",
        # --attention-backend flipped coding TTFT p99 by 3x during the
        # Qwen3.6-FP8 tuning pass (flashinfer vs default).  Treat as
        # perf-sensitive so future swaps go through the baseline-capture
        # approve-edit flow.  See reports/phase2-tuneA/.
        "--attention-backend",
    }
)

# Approval tokens expire after this many seconds (2 hours).
_APPROVAL_TTL_SECONDS = 7200


# ── Path detection ───────────────────────────────────────────────


def is_protected_recipe(file_path: str) -> bool:
    """Return True if *file_path* is a protected recipe or stage_config YAML."""
    if not file_path.endswith(".yaml"):
        return False
    parts = PurePosixPath(file_path).parts
    for i in range(len(parts)):
        remaining = parts[i:]
        if remaining[:2] == ("backends", "recipes"):
            return True
        if remaining[:2] == ("meeting_scribe", "recipes"):
            return True
        if remaining[:2] == ("meeting_scribe", "stage_configs"):
            return True
    return False


# ── Extra-args parsing ───────────────────────────────────────────


def _parse_extra_arg(arg: str) -> tuple[str, str | bool]:
    """Parse ``--flag=value`` → (``--flag``, ``value``) or ``--flag`` → (``--flag``, True)."""
    if "=" in arg:
        key, _, value = arg.partition("=")
        return key, value
    return arg, True


def _filter_perf_extra_args(args: list[Any]) -> dict[str, str | bool]:
    """Extract only perf-sensitive flags from an extra_args list."""
    result: dict[str, str | bool] = {}
    for arg in args:
        if not isinstance(arg, str):
            continue
        key, value = _parse_extra_arg(arg)
        if key in _PERF_SENSITIVE_EXTRA_ARG_PREFIXES:
            result[key] = value
    return result


# ── Value extraction ─────────────────────────────────────────────


def extract_perf_values(doc: dict[str, Any] | None) -> dict[str, Any]:
    """Extract performance-sensitive values from a parsed YAML document.

    Handles both root-level recipe params and nested
    ``stage_args[*].engine_args`` params from stage_configs.
    """
    if not isinstance(doc, dict):
        return {}

    values: dict[str, Any] = {}

    # Root-level keys
    for key in _PERF_SENSITIVE_KEYS:
        if key in doc:
            values[key] = doc[key]

    # extra_args at root
    if "extra_args" in doc and isinstance(doc["extra_args"], list):
        perf_args = _filter_perf_extra_args(doc["extra_args"])
        if perf_args:
            values["extra_args"] = tuple(sorted(perf_args.items()))

    # Nested stage_args[*].engine_args (meeting-scribe stage_configs)
    stage_args = doc.get("stage_args")
    if isinstance(stage_args, list):
        for idx, stage in enumerate(stage_args):
            if not isinstance(stage, dict):
                continue
            engine_args = stage.get("engine_args")
            if not isinstance(engine_args, dict):
                continue
            for key in _PERF_SENSITIVE_KEYS:
                if key in engine_args:
                    values[f"stage[{idx}].{key}"] = engine_args[key]

    return values


# ── YAML diff ────────────────────────────────────────────────────


def diff_perf_values(before_yaml: str, after_yaml: str) -> list[str]:
    """Return names of perf-sensitive params that changed between two YAML docs.

    Returns an empty list if no perf-sensitive params changed.
    Returns a non-empty list with a parse-error sentinel on YAML parse failure
    (fail-closed behavior).
    """
    try:
        before_doc = yaml.safe_load(before_yaml)
    except yaml.YAMLError:
        return ["<yaml parse error in before content — fail closed>"]

    try:
        after_doc = yaml.safe_load(after_yaml)
    except yaml.YAMLError:
        return ["<yaml parse error in after content — fail closed>"]

    if before_doc is None:
        before_doc = {}
    if after_doc is None:
        after_doc = {}

    if not isinstance(before_doc, dict):
        return ["<before content is not a YAML mapping — fail closed>"]
    if not isinstance(after_doc, dict):
        return ["<after content is not a YAML mapping — fail closed>"]

    before_vals = extract_perf_values(before_doc)
    after_vals = extract_perf_values(after_doc)

    all_keys = sorted(set(before_vals) | set(after_vals))
    changed: list[str] = []
    for key in all_keys:
        if before_vals.get(key) != after_vals.get(key):
            changed.append(key)

    return changed


# ── Content hashing ──────────────────────────────────────────────


def content_hash(content: str) -> str:
    """SHA-256 hex digest of *content*."""
    return hashlib.sha256(content.encode()).hexdigest()


def recipe_content_hashes() -> dict[str, str]:
    """Return ``{relative_path: content_hash}`` for all protected recipes in the repo.

    Uses ``git ls-files`` to discover tracked files.  Falls back to an empty
    dict if not in a git repo.
    """
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--full-name"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return {}
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}

    # Find the repo root so we can resolve relative paths to absolute
    try:
        root_proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        repo_root = Path(root_proc.stdout.strip()) if root_proc.returncode == 0 else Path.cwd()
    except (subprocess.SubprocessError, FileNotFoundError):
        repo_root = Path.cwd()

    hashes: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        rel_path = line.strip()
        if not rel_path:
            continue
        if is_protected_recipe(rel_path):
            full_path = repo_root / rel_path
            with contextlib.suppress(OSError):
                hashes[rel_path] = content_hash(full_path.read_text())

    return hashes


# ── Approval tokens ──────────────────────────────────────────────


def _approvals_dir() -> Path:
    """Return (and create) the recipe-approvals directory."""
    d = paths.state_dir() / "recipe-approvals"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _approval_path(file_path: str) -> Path:
    """Token path keyed on the canonical file path."""
    canonical = str(Path(file_path).resolve())
    path_hash = hashlib.sha256(canonical.encode()).hexdigest()
    return _approvals_dir() / f"{path_hash}.json"


def has_perf_approval(file_path: str, after_content: str) -> bool:
    """Return True if a fresh approval token matches *file_path* and *after_content*."""
    token_path = _approval_path(file_path)
    if not token_path.exists():
        return False
    try:
        token = json.loads(token_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    # Check content hash matches
    expected_hash = content_hash(after_content)
    if token.get("content_hash") != expected_hash:
        return False

    # Check path matches
    canonical = str(Path(file_path).resolve())
    if token.get("canonical_path") != canonical:
        return False

    # Check freshness
    token_ts = token.get("timestamp", 0)
    return not time.time() - token_ts > _APPROVAL_TTL_SECONDS


def write_perf_approval(
    file_path: str,
    validated_hash: str,
    *,
    source: str = "save-baseline",
    ttl_seconds: int = _APPROVAL_TTL_SECONDS,
) -> Path:
    """Write an approval token for *file_path* with the given content hash.

    The *validated_hash* is the SHA-256 of the file's validated content.
    Historically this came exclusively from the perf run's recorded
    ``environment.recipe_hashes``; the explicit-approval CLI
    (``autosre perf approve-edit``) also writes tokens here, in which
    case *source* is ``"approve-edit"`` and reflects the user's direct
    consent rather than a save-baseline chain.

    Returns the token path so callers can show it to the operator.
    """
    canonical = str(Path(file_path).resolve())
    token = {
        "canonical_path": canonical,
        "content_hash": validated_hash,
        "timestamp": time.time(),
        "source": source,
        "ttl_seconds": ttl_seconds,
    }
    token_path = _approval_path(file_path)
    token_path.write_text(json.dumps(token, indent=2) + "\n")
    logger.info("Recipe approval token written: %s → %s (source=%s)", file_path, token_path, source)
    return token_path


def preview_edit(
    file_path: str,
    old_string: str,
    new_string: str,
) -> tuple[str, str, list[str]]:
    """Compute the Edit-tool-equivalent ``after`` content for a proposed edit.

    Returns ``(after_content, unified_diff, perf_sensitive_changes)``.

    Matches the guard's own ``after = before.replace(old, new, 1)`` rule
    byte-for-byte so an approval minted against the returned hash will
    match the Edit tool's hash when the same old/new strings are applied.
    Raises :class:`FileNotFoundError` if the recipe is missing and
    :class:`ValueError` if ``old_string`` does not occur in the file or
    the computed edit is a no-op.
    """
    import difflib

    path = Path(file_path).resolve()
    before = path.read_text()

    if old_string not in before:
        msg = f"old_string not found in {path}"
        raise ValueError(msg)

    after = before.replace(old_string, new_string, 1)
    if after == before:
        msg = "Edit is a no-op (old_string == new_string)"
        raise ValueError(msg)

    diff = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{path.name}",
            tofile=f"b/{path.name}",
            n=3,
        )
    )
    changed = diff_perf_values(before, after)
    return after, diff, changed
