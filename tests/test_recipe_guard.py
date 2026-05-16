"""Tests for autosre.hooks_backend.recipe_guard — recipe file protection."""

from __future__ import annotations

import json
import textwrap
import time
from typing import TYPE_CHECKING

import pytest

from autosre.hooks_backend import recipe_guard

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))


# ── is_protected_recipe ──────────────────────────────────────────


class TestIsProtectedRecipe:
    def test_autosre_recipe(self) -> None:
        assert recipe_guard.is_protected_recipe(
            "/home/user/repos/auto-sre/autosre/backends/recipes/qwen3.6-35b-a3b-fp8.yaml"
        )

    def test_meeting_scribe_recipe(self) -> None:
        assert recipe_guard.is_protected_recipe(
            "/home/user/repos/meeting-scribe/src/meeting_scribe/recipes/qwen3-asr.yaml"
        )

    def test_meeting_scribe_stage_config(self) -> None:
        assert recipe_guard.is_protected_recipe(
            "/home/user/repos/meeting-scribe/src/meeting_scribe/stage_configs/qwen3_tts.yaml"
        )

    def test_relative_autosre_recipe(self) -> None:
        assert recipe_guard.is_protected_recipe("autosre/backends/recipes/qwen3.6-35b-a3b-fp8.yaml")

    def test_unrelated_yaml(self) -> None:
        assert not recipe_guard.is_protected_recipe("/home/user/config.yaml")

    def test_non_yaml(self) -> None:
        assert not recipe_guard.is_protected_recipe("autosre/backends/recipes/README.md")

    def test_demo_run_recipe(self) -> None:
        # Demo run copies should also be protected
        assert recipe_guard.is_protected_recipe(
            "demo/runs/20260409/src/meeting_scribe/recipes/qwen3-asr.yaml"
        )


# ── extract_perf_values ──────────────────────────────────────────


class TestExtractPerfValues:
    def test_root_level_params(self) -> None:
        doc = {
            "model_key": "test",
            "gpu_memory_utilization": 0.75,
            "max_num_seqs": 8,
            "description": "ignored",
        }
        values = recipe_guard.extract_perf_values(doc)
        assert values == {
            "gpu_memory_utilization": 0.75,
            "max_num_seqs": 8,
        }

    def test_extra_args(self) -> None:
        doc = {
            "extra_args": [
                "--enable-prefix-caching",
                "--scheduling-policy=priority",
                "--tool-call-parser=qwen3_coder",  # not perf-sensitive
            ],
        }
        values = recipe_guard.extract_perf_values(doc)
        assert "extra_args" in values
        extra = dict(values["extra_args"])
        assert extra["--enable-prefix-caching"] is True
        assert extra["--scheduling-policy"] == "priority"
        assert "--tool-call-parser" not in extra

    def test_attention_backend_flagged(self) -> None:
        doc = {
            "extra_args": [
                "--attention-backend=flashinfer",
                "--reasoning-parser=qwen3",  # not perf-sensitive
            ],
        }
        values = recipe_guard.extract_perf_values(doc)
        assert "extra_args" in values
        extra = dict(values["extra_args"])
        assert extra["--attention-backend"] == "flashinfer"
        assert "--reasoning-parser" not in extra

    def test_nested_stage_args(self) -> None:
        doc = {
            "stage_args": [
                {
                    "stage_id": 0,
                    "engine_args": {
                        "gpu_memory_utilization": 0.1,
                        "max_num_seqs": 4,
                        "hf_config_name": "ignored",
                    },
                },
                {
                    "stage_id": 1,
                    "engine_args": {
                        "gpu_memory_utilization": 0.6,
                    },
                },
            ],
        }
        values = recipe_guard.extract_perf_values(doc)
        assert values["stage[0].gpu_memory_utilization"] == 0.1
        assert values["stage[0].max_num_seqs"] == 4
        assert values["stage[1].gpu_memory_utilization"] == 0.6

    def test_none_doc(self) -> None:
        assert recipe_guard.extract_perf_values(None) == {}

    def test_non_mapping(self) -> None:
        assert recipe_guard.extract_perf_values("just a string") == {}  # type: ignore[arg-type]


# ── diff_perf_values ─────────────────────────────────────────────


class TestDiffPerfValues:
    def test_no_change(self) -> None:
        yaml_str = "gpu_memory_utilization: 0.75\nmax_num_seqs: 8\n"
        assert recipe_guard.diff_perf_values(yaml_str, yaml_str) == []

    def test_gpu_mem_changed(self) -> None:
        before = "gpu_memory_utilization: 0.75\nmax_num_seqs: 8\n"
        after = "gpu_memory_utilization: 0.70\nmax_num_seqs: 8\n"
        changed = recipe_guard.diff_perf_values(before, after)
        assert changed == ["gpu_memory_utilization"]

    def test_comment_only(self) -> None:
        before = "# old comment\ngpu_memory_utilization: 0.75\n"
        after = "# new comment\ngpu_memory_utilization: 0.75\n"
        assert recipe_guard.diff_perf_values(before, after) == []

    def test_extra_args_value_changed(self) -> None:
        before = textwrap.dedent("""\
            extra_args:
              - "--scheduling-policy=priority"
              - "--enable-prefix-caching"
        """)
        after = textwrap.dedent("""\
            extra_args:
              - "--scheduling-policy=fcfs"
              - "--enable-prefix-caching"
        """)
        changed = recipe_guard.diff_perf_values(before, after)
        assert "extra_args" in changed

    def test_extra_args_toggle_removed(self) -> None:
        before = textwrap.dedent("""\
            extra_args:
              - "--enable-prefix-caching"
              - "--enable-chunked-prefill"
        """)
        after = textwrap.dedent("""\
            extra_args:
              - "--enable-chunked-prefill"
        """)
        changed = recipe_guard.diff_perf_values(before, after)
        assert "extra_args" in changed

    def test_parse_error_before(self) -> None:
        changed = recipe_guard.diff_perf_values("{{invalid", "key: value\n")
        assert len(changed) > 0
        assert "parse error" in changed[0]

    def test_parse_error_after(self) -> None:
        changed = recipe_guard.diff_perf_values("key: value\n", "{{invalid")
        assert len(changed) > 0
        assert "parse error" in changed[0]

    def test_non_mapping_before(self) -> None:
        changed = recipe_guard.diff_perf_values("just a string", "key: value\n")
        assert len(changed) > 0
        assert "mapping" in changed[0]

    def test_empty_before_new_file(self) -> None:
        after = "gpu_memory_utilization: 0.75\nmax_num_seqs: 8\n"
        changed = recipe_guard.diff_perf_values("", after)
        assert "gpu_memory_utilization" in changed
        assert "max_num_seqs" in changed

    def test_non_sensitive_param_change(self) -> None:
        before = "description: old\ngpu_memory_utilization: 0.75\n"
        after = "description: new\ngpu_memory_utilization: 0.75\n"
        assert recipe_guard.diff_perf_values(before, after) == []


# ── Approval tokens ──────────────────────────────────────────────


class TestApprovalTokens:
    def test_fresh_matching_token(self, tmp_path: Path) -> None:
        recipe = tmp_path / "test.yaml"
        recipe.write_text("gpu_memory_utilization: 0.75\n")
        content = recipe.read_text()

        recipe_guard.write_perf_approval(str(recipe), recipe_guard.content_hash(content))
        assert recipe_guard.has_perf_approval(str(recipe), content)

    def test_wrong_content_hash(self, tmp_path: Path) -> None:
        recipe = tmp_path / "test.yaml"
        recipe.write_text("gpu_memory_utilization: 0.75\n")

        recipe_guard.write_perf_approval(
            str(recipe), recipe_guard.content_hash("gpu_memory_utilization: 0.75\n")
        )
        # Check with different content
        assert not recipe_guard.has_perf_approval(str(recipe), "gpu_memory_utilization: 0.70\n")

    def test_stale_token(self, tmp_path: Path) -> None:
        recipe = tmp_path / "test.yaml"
        content = "gpu_memory_utilization: 0.75\n"
        recipe.write_text(content)

        recipe_guard.write_perf_approval(str(recipe), recipe_guard.content_hash(content))

        # Manually backdate the token
        token_path = recipe_guard._approval_path(str(recipe))
        token = json.loads(token_path.read_text())
        token["timestamp"] = time.time() - recipe_guard._APPROVAL_TTL_SECONDS - 1
        token_path.write_text(json.dumps(token))

        assert not recipe_guard.has_perf_approval(str(recipe), content)

    def test_missing_token(self, tmp_path: Path) -> None:
        assert not recipe_guard.has_perf_approval(str(tmp_path / "nonexistent.yaml"), "content")
