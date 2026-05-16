"""Tests for autosre.review.cli_plan — state management and prompt selection.

Key regression test: ``_state_key`` must include a hash of the absolute
plan path, otherwise two repos each containing a ``plan.md`` would share
iteration state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from autosre.review import cli_plan

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


class TestStateKey:
    def test_includes_stem(self, tmp_path: Path) -> None:
        plan = tmp_path / "my-plan.md"
        plan.write_text("# plan")
        key = cli_plan._state_key(plan)
        assert key.endswith("_my-plan")

    def test_includes_path_hash_prefix(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text("# plan")
        key = cli_plan._state_key(plan)
        # 12-char hex hash + underscore + stem
        prefix = key.split("_", 1)[0]
        assert len(prefix) == 12
        assert all(c in "0123456789abcdef" for c in prefix)

    def test_different_repos_same_stem_different_keys(
        self,
        tmp_path: Path,
    ) -> None:
        """Regression: upstream keyed state by stem only, causing collisions
        across repos. Our port hashes the absolute path so each plan gets
        its own state namespace."""
        repo_a = tmp_path / "repo_a"
        repo_b = tmp_path / "repo_b"
        repo_a.mkdir()
        repo_b.mkdir()
        plan_a = repo_a / "plan.md"
        plan_b = repo_b / "plan.md"
        plan_a.write_text("# A")
        plan_b.write_text("# B")

        key_a = cli_plan._state_key(plan_a)
        key_b = cli_plan._state_key(plan_b)

        assert key_a != key_b, (
            "different-repo plans with identical stems must get different "
            "state keys to avoid cross-repo iteration contamination"
        )
        # Both still end with the stem for human debugging
        assert key_a.endswith("_plan")
        assert key_b.endswith("_plan")

    def test_same_path_stable_key(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text("# A")
        key1 = cli_plan._state_key(plan)
        key2 = cli_plan._state_key(plan)
        assert key1 == key2


class TestStatePath:
    def test_state_path_under_review_state_dir(self, tmp_path: Path) -> None:
        plan = tmp_path / "my-plan.md"
        plan.write_text("# plan")
        path = cli_plan._state_path(plan)
        assert path.parent == tmp_path / "data" / "autosre" / "review-state"
        assert path.name.startswith("_state_")
        assert path.name.endswith(".json")

    def test_cross_repo_state_files_distinct(self, tmp_path: Path) -> None:
        repo_a = tmp_path / "a"
        repo_b = tmp_path / "b"
        repo_a.mkdir()
        repo_b.mkdir()
        plan_a = repo_a / "plan.md"
        plan_b = repo_b / "plan.md"
        plan_a.write_text("x")
        plan_b.write_text("x")

        assert cli_plan._state_path(plan_a) != cli_plan._state_path(plan_b)


class TestLoadSaveState:
    def test_load_missing_returns_default(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text("# plan")
        state = cli_plan._load_state(plan)
        assert state == {
            "iteration": 0,
            "previous_findings": [],
            "plan_mtime": 0,
            "plan_hash": "",
            "last_reviewed_at": "",
            "last_review_status": "not_reviewed",
            "last_review_findings_count": 0,
        }

    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text("# plan")
        state = {
            "iteration": 3,
            "previous_findings": [{"severity": "P1", "title": "x"}],
            "plan_mtime": 123.45,
            "plan_hash": "abc",
            "last_reviewed_at": "2026-04-11T10:00:00+0000",
            "last_review_status": "blocking",
            "last_review_findings_count": 1,
        }
        cli_plan._save_state(plan, state)
        loaded = cli_plan._load_state(plan)
        assert loaded == state

    def test_load_corrupted_json_returns_default(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text("# plan")
        cli_plan._state_path(plan).parent.mkdir(parents=True, exist_ok=True)
        cli_plan._state_path(plan).write_text("{not valid json")
        state = cli_plan._load_state(plan)
        assert state["iteration"] == 0

    def test_two_plans_different_states(self, tmp_path: Path) -> None:
        """Regression: writing state for plan A must not affect plan B even
        if their stems collide."""
        repo_a = tmp_path / "a"
        repo_b = tmp_path / "b"
        repo_a.mkdir()
        repo_b.mkdir()
        plan_a = repo_a / "plan.md"
        plan_b = repo_b / "plan.md"
        plan_a.write_text("x")
        plan_b.write_text("y")

        cli_plan._save_state(plan_a, {"iteration": 5, "previous_findings": []})
        cli_plan._save_state(plan_b, {"iteration": 2, "previous_findings": []})

        assert cli_plan._load_state(plan_a)["iteration"] == 5
        assert cli_plan._load_state(plan_b)["iteration"] == 2


class TestPlanMetadataHelpers:
    def test_extract_title_with_h1(self) -> None:
        assert cli_plan._extract_title("# My Plan\n\nContent") == "My Plan"

    def test_extract_title_no_heading(self) -> None:
        assert cli_plan._extract_title("Just text, no heading") == "Untitled"

    def test_extract_title_only_h2(self) -> None:
        assert cli_plan._extract_title("## sub\ntext") == "Untitled"

    def test_compute_hash_stable(self) -> None:
        h1 = cli_plan._compute_hash("content")
        h2 = cli_plan._compute_hash("content")
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex

    def test_compute_hash_different_content(self) -> None:
        assert cli_plan._compute_hash("a") != cli_plan._compute_hash("b")

    def test_detect_project_with_marker(self) -> None:
        content = "# Plan\n\n**Project**: `my-repo`\n\nContent"
        assert cli_plan._detect_project(content) == "my-repo"

    def test_detect_project_without_marker(self) -> None:
        assert cli_plan._detect_project("# Plan\n\nContent") is None


class TestPromptTemplates:
    def test_initial_prompt_has_placeholders(self) -> None:
        assert "{plan_content}" in cli_plan.INITIAL_REVIEW_PROMPT
        assert "P0" in cli_plan.INITIAL_REVIEW_PROMPT
        assert "P1" in cli_plan.INITIAL_REVIEW_PROMPT
        assert "P2" in cli_plan.INITIAL_REVIEW_PROMPT

    def test_rereview_prompt_has_placeholders(self) -> None:
        t = cli_plan.RE_REVIEW_PROMPT
        assert "{iteration}" in t
        assert "{prev_count}" in t
        assert "{previous_findings_text}" in t
        assert "{plan_content}" in t
        assert "convergence" in t.lower()

    def test_initial_prompt_format(self) -> None:
        rendered = cli_plan.INITIAL_REVIEW_PROMPT.format(plan_content="## Step 1\ndo X")
        assert "## Step 1" in rendered
        assert "{plan_content}" not in rendered

    def test_rereview_prompt_format(self) -> None:
        rendered = cli_plan.RE_REVIEW_PROMPT.format(
            iteration=3,
            prev_count=2,
            previous_findings_text="  1. [P0] foo",
            plan_content="updated plan",
        )
        assert "iteration 3" in rendered
        assert "2 issues" in rendered
        assert "updated plan" in rendered
