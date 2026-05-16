"""Immutable filesystem snapshots of a target repository.

Two modes:

- ``git-worktree`` (default, clean tree required): ``git worktree add`` at
  ``HEAD``. Fast, deterministic, git knows about it, easy to clean up.
- ``file-copy`` (``--allow-dirty``): ``shutil.copytree`` with a fixed
  ignore set so VCS metadata, caches, and virtualenvs are excluded. This
  faithfully captures untracked files too, which several suites (leakage,
  tech-debt, dead-code) legitimately need to inspect.

Both modes lock every regular file in the snapshot to ``0o444`` so any
``Write``/``Edit`` attempt by an eval agent fails at the OS layer. The
findings-file directory lives *outside* the snapshot, under the run
directory, so it stays writable. This is the enforcement layer the rest
of the eval pipeline relies on — the Claude Code permission list is only
a hint, not a boundary.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

SnapshotMode = Literal["git-worktree", "file-copy"]


class DirtyTreeError(RuntimeError):
    """Raised when the caller requested a git snapshot of a dirty tree."""


class NotARepoError(RuntimeError):
    """Raised when the target is not a git repo and a git snapshot was requested."""


# Top-level directory / file patterns excluded from file-copy mode. These
# are noise for every eval suite and would blow up the snapshot size.
# Untracked files that don't match these patterns ARE included — that's
# the whole point of the file-copy mode.
_IGNORE_PATTERNS: tuple[str, ...] = (
    ".git",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".nox",
    "build",
    "dist",
    "*.egg-info",
    ".DS_Store",
)


@dataclass(frozen=True)
class Snapshot:
    """A materialized snapshot of a target repository at a pinned state."""

    path: Path
    mode: SnapshotMode
    source_repo: Path
    source_sha: str | None
    snapshot_digest: str
    file_count: int
    included_untracked: bool
    cleanup_hint: str = field(default="")

    def as_manifest_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation for ``manifest.json``."""
        return {
            "path": str(self.path),
            "mode": self.mode,
            "source_repo": str(self.source_repo),
            "source_sha": self.source_sha,
            "snapshot_digest": self.snapshot_digest,
            "file_count": self.file_count,
            "included_untracked": self.included_untracked,
        }


# ── Public API ─────────────────────────────────────────────────────


def materialize(
    target: Path,
    dest: Path,
    *,
    allow_dirty: bool = False,
) -> Snapshot:
    """Materialize a read-only snapshot of ``target`` at ``dest``.

    Default path: ``materialize_git``. When ``allow_dirty`` is True, fall
    back to ``materialize_copy`` — either because the tree is dirty or
    because ``target`` is not a git repo at all.
    """
    target = target.resolve()
    dest = dest.resolve()

    if _is_git_repo(target):
        if _is_dirty(target):
            if not allow_dirty:
                raise DirtyTreeError(
                    f"target {target} has uncommitted changes; pass "
                    f"allow_dirty=True to snapshot with a file-copy"
                )
            return materialize_copy(target, dest)
        return materialize_git(target, dest)

    if not allow_dirty:
        raise NotARepoError(
            f"target {target} is not a git repo; pass allow_dirty=True to snapshot with a file-copy"
        )
    return materialize_copy(target, dest)


def materialize_git(target: Path, dest: Path) -> Snapshot:
    """Create a ``git worktree`` at ``HEAD`` and lock it read-only."""
    target = target.resolve()
    dest = dest.resolve()
    if dest.exists():
        raise FileExistsError(f"snapshot destination already exists: {dest}")

    sha = _rev_parse_head(target)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(target),
            "worktree",
            "add",
            "--detach",
            str(dest),
            sha,
        ],
        check=True,
        capture_output=True,
    )

    file_count = _count_files(dest)
    digest = _hash_tree(dest)
    _chmod_tree_readonly(dest)

    return Snapshot(
        path=dest,
        mode="git-worktree",
        source_repo=target,
        source_sha=sha,
        snapshot_digest=digest,
        file_count=file_count,
        included_untracked=False,
        cleanup_hint="git worktree remove",
    )


def materialize_copy(target: Path, dest: Path) -> Snapshot:
    """Copy ``target`` → ``dest`` with ``_IGNORE_PATTERNS`` and lock it."""
    target = target.resolve()
    dest = dest.resolve()
    if dest.exists():
        raise FileExistsError(f"snapshot destination already exists: {dest}")

    ignore = shutil.ignore_patterns(*_IGNORE_PATTERNS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(target, dest, ignore=ignore, symlinks=True)

    head_sha = _rev_parse_head(target) if _is_git_repo(target) else None
    file_count = _count_files(dest)
    digest = _hash_tree(dest)

    # Write a sidecar so downstream readers can prove exactly which files
    # the agents saw without re-reading the snapshot. Lives OUTSIDE the
    # snapshot so it is not itself included in the digest and stays
    # writable after _chmod_tree_readonly.
    sidecar = dest.parent / f"{dest.name}.digest.json"
    sidecar.write_text(
        json.dumps(
            {
                "path": str(dest),
                "mode": "file-copy",
                "source_repo": str(target),
                "source_sha": head_sha,
                "snapshot_digest": digest,
                "file_count": file_count,
            },
            indent=2,
        )
    )

    _chmod_tree_readonly(dest)

    return Snapshot(
        path=dest,
        mode="file-copy",
        source_repo=target,
        source_sha=head_sha,
        snapshot_digest=digest,
        file_count=file_count,
        included_untracked=True,
        cleanup_hint="shutil.rmtree",
    )


def cleanup(snapshot: Snapshot) -> None:
    """Remove the snapshot. Restores write perms first so rm can proceed."""
    _chmod_tree_writable(snapshot.path)

    if snapshot.mode == "git-worktree":
        # Use the source repo's worktree machinery so git stays consistent.
        try:
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(snapshot.source_repo),
                    "worktree",
                    "remove",
                    "--force",
                    str(snapshot.path),
                ],
                check=True,
                capture_output=True,
            )
            return
        except subprocess.CalledProcessError:
            # Fall through to rmtree if git disagrees.
            pass

    if snapshot.path.exists():
        shutil.rmtree(snapshot.path)


# ── Internals ──────────────────────────────────────────────────────


def _is_git_repo(target: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
        return r.returncode == 0 and r.stdout.strip() == "true"
    except FileNotFoundError:
        return False


def _is_dirty(target: Path) -> bool:
    r = subprocess.run(
        ["git", "-C", str(target), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return bool(r.stdout.strip())


def _rev_parse_head(target: Path) -> str:
    r = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def _count_files(root: Path) -> int:
    n = 0
    for _ in _walk_regular_files(root):
        n += 1
    return n


def _walk_regular_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            continue
        if p.is_file():
            out.append(p)
    return out


def _hash_tree(root: Path) -> str:
    """sha256 over (relpath, sha256(content)) for every regular file.

    Stable across runs: paths are sorted, line endings are preserved
    byte-for-byte, and symlinks are excluded (they would introduce
    nondeterminism from mtime and target resolution).
    """
    h = hashlib.sha256()
    for p in _walk_regular_files(root):
        rel = p.relative_to(root).as_posix().encode()
        content_hash = hashlib.sha256(p.read_bytes()).digest()
        h.update(len(rel).to_bytes(4, "big"))
        h.update(rel)
        h.update(content_hash)
    return h.hexdigest()


def _chmod_tree_readonly(root: Path) -> None:
    """Strip write bits from every regular file under ``root``.

    Directories keep their x+r bits so traversal still works. We only
    touch regular files — symlinks are left alone (chmod on a symlink
    changes the link target on Linux, which is not what we want), and
    special files are ignored.
    """
    for p in root.rglob("*"):
        if p.is_symlink():
            continue
        if p.is_file():
            try:
                mode = p.stat().st_mode
                p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            except (OSError, PermissionError):
                pass


def _chmod_tree_writable(root: Path) -> None:
    """Restore user-writable bits so rmtree / git worktree remove can proceed."""
    if not root.exists():
        return
    for p in root.rglob("*"):
        if p.is_symlink():
            continue
        try:
            mode = p.stat().st_mode
            p.chmod(mode | stat.S_IWUSR)
        except (OSError, PermissionError):
            pass
    # Also handle the root itself.
    with contextlib.suppress(OSError, PermissionError):
        root.chmod(root.stat().st_mode | stat.S_IWUSR)
