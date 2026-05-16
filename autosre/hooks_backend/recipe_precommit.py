"""Pre-commit hook: block commits that change perf-sensitive recipe params
without a corresponding baseline update.

Registered via ``.pre-commit-config.yaml`` with::

    files: '(recipes|stage_configs)/.*\\.yaml$'

Only runs when recipe/stage_config YAML files are staged.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
from typing import Any

from autosre.hooks_backend.recipe_guard import (
    content_hash,
    diff_perf_values,
    is_protected_recipe,
)


def _git_show(ref_path: str) -> str:
    """Return file content at a git ref (e.g. ``HEAD:path`` or ``:path``)."""
    proc = subprocess.run(
        ["git", "show", ref_path],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def main(argv: list[str] | None = None) -> int:
    """Entry point — receives staged filenames from pre-commit."""
    files = argv if argv is not None else sys.argv[1:]

    # Filter to protected recipe files
    protected = [f for f in files if is_protected_recipe(f)]
    if not protected:
        return 0

    # Check which protected files have perf-sensitive changes.
    # New recipes (HEAD content empty) are allowed without a baseline at
    # commit time — Phase 2 bench runs mint the first baseline for the new
    # model; blocking creation here is circular.  Edits to existing recipes
    # still require a staged baseline.
    changed_recipes: list[tuple[str, list[str]]] = []
    for path in protected:
        head_content = _git_show(f"HEAD:{path}")
        staged_content = _git_show(f":{path}")
        if not head_content:
            continue
        changed = diff_perf_values(head_content, staged_content)
        if changed:
            changed_recipes.append((path, changed))

    if not changed_recipes:
        return 0

    # Check that at least one baseline is also staged and its recipe_hashes match
    staged_baselines = _find_staged_baselines()
    errors: list[str] = []

    for path, changed_params in changed_recipes:
        staged_content = _git_show(f":{path}")
        staged_hash = content_hash(staged_content)

        validated = False
        for baseline in staged_baselines:
            recipe_hashes = baseline.get("environment", {}).get("recipe_hashes", {})
            if recipe_hashes.get(path) == staged_hash:
                validated = True
                break

        if not validated:
            errors.append(
                f"  {path}: changes {', '.join(changed_params)}\n"
                f"    No staged baseline validates this content (hash: {staged_hash[:12]}…)\n"
                f"    Run: autosre perf run && autosre perf save-baseline <name>"
            )

    if errors:
        print("Recipe perf gate FAILED — perf-sensitive params changed without baseline:\n")
        print("\n".join(errors))
        print(
            "\nCommit the baseline alongside the recipe change:\n"
            "  git add benchmarks/baselines/<name>.{json,md}"
        )
        return 1

    return 0


def _find_staged_baselines() -> list[dict[str, Any]]:
    """Load all staged baseline JSON files."""
    proc = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if proc.returncode != 0:
        return []

    baselines: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        path = line.strip()
        if not path.startswith("benchmarks/baselines/") or not path.endswith(".json"):
            continue
        content = _git_show(f":{path}")
        if content:
            with contextlib.suppress(json.JSONDecodeError):
                baselines.append(json.loads(content))

    return baselines


if __name__ == "__main__":
    sys.exit(main())
