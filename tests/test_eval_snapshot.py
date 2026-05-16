"""Tests for autosre.eval.snapshot."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest

from autosre.eval.snapshot import (
    DirtyTreeError,
    NotARepoError,
    _hash_tree,
    cleanup,
    materialize,
    materialize_copy,
    materialize_git,
)

if TYPE_CHECKING:
    from pathlib import Path


def _git(cwd: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return r.stdout.strip()


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")


@pytest.fixture()
def clean_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "src"
    repo.mkdir()
    _init_repo(repo)
    (repo / "README.md").write_text("hello\n")
    (repo / "code.py").write_text("def f():\n    return 1\n")
    (repo / "sub").mkdir()
    (repo / "sub" / "more.py").write_text("x = 2\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


@pytest.fixture()
def dirty_repo(clean_repo: Path) -> Path:
    # Modify a tracked file and add an untracked one.
    (clean_repo / "code.py").write_text("def f():\n    return 42\n")
    (clean_repo / "untracked.txt").write_text("brand new\n")
    return clean_repo


class TestMaterializeGit:
    def test_clean_tree_produces_worktree(self, clean_repo: Path, tmp_path: Path) -> None:
        dest = tmp_path / "snapshot"
        snap = materialize_git(clean_repo, dest)
        assert snap.mode == "git-worktree"
        assert snap.path == dest.resolve()
        assert (snap.path / "code.py").read_text() == "def f():\n    return 1\n"
        assert snap.source_sha is not None
        assert len(snap.source_sha) == 40
        cleanup(snap)

    def test_read_only_after_materialize(self, clean_repo: Path, tmp_path: Path) -> None:
        snap = materialize_git(clean_repo, tmp_path / "snap")
        with pytest.raises(PermissionError):
            (snap.path / "code.py").write_text("mutation attempt\n")
        cleanup(snap)

    def test_refuses_dirty_tree(self, dirty_repo: Path, tmp_path: Path) -> None:
        with pytest.raises(DirtyTreeError):
            materialize(dirty_repo, tmp_path / "snap")

    def test_allow_dirty_routes_to_copy(self, dirty_repo: Path, tmp_path: Path) -> None:
        snap = materialize(dirty_repo, tmp_path / "snap", allow_dirty=True)
        assert snap.mode == "file-copy"
        # Dirty tracked change must be present.
        assert (snap.path / "code.py").read_text() == ("def f():\n    return 42\n")
        # Untracked file must be present too.
        assert (snap.path / "untracked.txt").read_text() == "brand new\n"
        cleanup(snap)

    def test_cleanup_removes_worktree(self, clean_repo: Path, tmp_path: Path) -> None:
        snap = materialize_git(clean_repo, tmp_path / "snap")
        assert snap.path.exists()
        cleanup(snap)
        assert not snap.path.exists()


class TestMaterializeCopy:
    def test_ignores_noise_dirs(self, clean_repo: Path, tmp_path: Path) -> None:
        (clean_repo / "__pycache__").mkdir()
        (clean_repo / "__pycache__" / "x.pyc").write_text("noise")
        (clean_repo / "node_modules").mkdir()
        (clean_repo / "node_modules" / "junk.js").write_text("noise")

        snap = materialize_copy(clean_repo, tmp_path / "snap")
        assert not (snap.path / "__pycache__").exists()
        assert not (snap.path / "node_modules").exists()
        assert (snap.path / "code.py").exists()
        cleanup(snap)

    def test_digest_sidecar_written(self, clean_repo: Path, tmp_path: Path) -> None:
        snap = materialize_copy(clean_repo, tmp_path / "snap")
        sidecar = snap.path.parent / f"{snap.path.name}.digest.json"
        assert sidecar.exists()
        import json as _json

        payload = _json.loads(sidecar.read_text())
        assert payload["snapshot_digest"] == snap.snapshot_digest
        assert payload["mode"] == "file-copy"
        cleanup(snap)

    def test_digest_stable_across_identical_trees(self, clean_repo: Path, tmp_path: Path) -> None:
        a = materialize_copy(clean_repo, tmp_path / "a")
        b = materialize_copy(clean_repo, tmp_path / "b")
        assert a.snapshot_digest == b.snapshot_digest
        cleanup(a)
        cleanup(b)

    def test_digest_changes_with_content(self, clean_repo: Path, tmp_path: Path) -> None:
        a = materialize_copy(clean_repo, tmp_path / "a")
        (clean_repo / "code.py").write_text("def f():\n    return 999\n")
        b = materialize_copy(clean_repo, tmp_path / "b")
        assert a.snapshot_digest != b.snapshot_digest
        cleanup(a)
        cleanup(b)


class TestMaterializeRouting:
    def test_non_git_requires_allow_dirty(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        (plain / "a.txt").write_text("x")
        with pytest.raises(NotARepoError):
            materialize(plain, tmp_path / "snap")

    def test_non_git_allow_dirty_succeeds(self, tmp_path: Path) -> None:
        plain = tmp_path / "plain"
        plain.mkdir()
        (plain / "a.txt").write_text("x")
        snap = materialize(plain, tmp_path / "snap", allow_dirty=True)
        assert snap.mode == "file-copy"
        assert snap.source_sha is None
        cleanup(snap)

    def test_clean_repo_uses_git_mode(self, clean_repo: Path, tmp_path: Path) -> None:
        snap = materialize(clean_repo, tmp_path / "snap")
        assert snap.mode == "git-worktree"
        cleanup(snap)


class TestHashTree:
    def test_order_independent(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        a.mkdir()
        (a / "z.txt").write_text("z-content")
        (a / "a.txt").write_text("a-content")
        h1 = _hash_tree(a)

        b = tmp_path / "b"
        b.mkdir()
        (b / "a.txt").write_text("a-content")
        (b / "z.txt").write_text("z-content")
        h2 = _hash_tree(b)

        assert h1 == h2

    def test_path_matters(self, tmp_path: Path) -> None:
        a = tmp_path / "a"
        a.mkdir()
        (a / "one.txt").write_text("same")
        b = tmp_path / "b"
        b.mkdir()
        (b / "two.txt").write_text("same")
        assert _hash_tree(a) != _hash_tree(b)


class TestSnapshotManifest:
    def test_as_manifest_dict_is_json_serializable(self, clean_repo: Path, tmp_path: Path) -> None:
        import json as _json

        snap = materialize_git(clean_repo, tmp_path / "snap")
        payload = snap.as_manifest_dict()
        _json.dumps(payload)  # must not raise
        assert payload["mode"] == "git-worktree"
        assert payload["source_sha"] == snap.source_sha
        cleanup(snap)

    def test_snapshot_is_frozen_dataclass(self, clean_repo: Path, tmp_path: Path) -> None:
        snap = materialize_git(clean_repo, tmp_path / "snap")
        with pytest.raises(AttributeError):
            snap.mode = "file-copy"  # type: ignore[misc]
        cleanup(snap)
