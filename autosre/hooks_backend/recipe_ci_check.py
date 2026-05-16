"""CI check: block PRs that change perf-sensitive recipe params without a
corresponding baseline update.

Usage::

    python -m autosre.hooks_backend.recipe_ci_check --base-ref <sha>

Exit codes:
    0 — clean (no perf-sensitive recipe changes, or baseline validates them)
    1 — fail (perf-sensitive changes without matching baseline)
"""

from __future__ import annotations

import argparse
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
    """Return file content at ``ref:path``."""
    proc = subprocess.run(
        ["git", "show", ref_path],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Recipe perf gate for CI")
    parser.add_argument("--base-ref", required=True, help="Base branch SHA or ref")
    args = parser.parse_args()

    base_ref = args.base_ref

    # Find files changed between base and HEAD
    proc = subprocess.run(
        ["git", "diff", "--name-only", f"{base_ref}...HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if proc.returncode != 0:
        print(f"WARNING: git diff failed (exit {proc.returncode}), skipping recipe gate")
        return 0

    changed_files = [f.strip() for f in proc.stdout.splitlines() if f.strip()]
    protected = [f for f in changed_files if is_protected_recipe(f)]

    if not protected:
        return 0

    # Check which protected files have perf-sensitive changes
    changed_recipes: list[tuple[str, list[str]]] = []
    for path in protected:
        base_content = _git_show(f"{base_ref}:{path}")
        head_content = _git_show(f"HEAD:{path}")
        changed = diff_perf_values(base_content, head_content)
        if changed:
            changed_recipes.append((path, changed))

    if not changed_recipes:
        return 0

    # Find changed baseline files and load them
    changed_baselines: list[dict[str, Any]] = []
    for f in changed_files:
        if f.startswith("benchmarks/baselines/") and f.endswith(".json"):
            content = _git_show(f"HEAD:{f}")
            if content:
                with contextlib.suppress(json.JSONDecodeError):
                    changed_baselines.append(json.loads(content))

    # Validate each changed recipe against baselines
    errors: list[str] = []
    for path, changed_params in changed_recipes:
        head_content = _git_show(f"HEAD:{path}")
        head_hash = content_hash(head_content)

        validated = False
        for baseline in changed_baselines:
            recipe_hashes = baseline.get("environment", {}).get("recipe_hashes", {})
            if recipe_hashes.get(path) == head_hash:
                validated = True
                break

        if not validated:
            errors.append(
                f"  {path}: changes {', '.join(changed_params)}\n"
                f"    No baseline in this PR validates content hash {head_hash[:12]}…\n"
                f"    Run `autosre perf run` and `autosre perf save-baseline` locally,\n"
                f"    then commit the baseline alongside the recipe change."
            )

    if errors:
        print("::error::Recipe perf gate FAILED")
        print()
        print("Perf-sensitive recipe params changed without a validating baseline:\n")
        print("\n".join(errors))
        return 1

    print(f"Recipe perf gate passed: {len(changed_recipes)} recipe(s) validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
