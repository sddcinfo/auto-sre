"""Tests for autosre.hooks_backend.guard — rule evaluation and CLI."""

from __future__ import annotations

import json
import textwrap
from typing import TYPE_CHECKING

import pytest
import yaml
from click.testing import CliRunner

from autosre.hooks_backend import guard

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    # Drop any prior config cache so each test loads fresh rules.
    guard._CONFIG_CACHE = None
    guard._CONFIG_CACHE_PATH = None
    guard._CONFIG_CACHE_MTIME = 0.0


@pytest.fixture
def rules_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write a minimal rules file and point AUTOSRE_GUARD_RULES at it."""
    path = tmp_path / "rules.yaml"
    path.write_text(
        textwrap.dedent(
            """\
            version: "1.0.0"
            settings:
              approval_expiry_seconds: 60
            blocked_env_vars: [AUTOSRE_BYPASS_GUARD]
            rules:
              - id: block-stash
                pattern: '^git\\s+stash'
                decision: deny
                reason: "no stash"
              - id: ask-reset
                pattern: '^git\\s+reset'
                decision: ask
                reason: "reset is risky"
              - id: allow-echo
                pattern: '^echo\\s+'
                decision: allow
            """,
        ),
    )
    monkeypatch.setenv("AUTOSRE_GUARD_RULES", str(path))
    return path


class TestLoadConfig:
    def test_loads_yaml(self, rules_file: Path) -> None:
        cfg = guard.load_config()
        assert "rules" in cfg
        assert len(cfg["rules"]) == 3
        assert cfg["blocked_env_vars"] == ["AUTOSRE_BYPASS_GUARD"]

    def test_hot_reload_on_mtime_change(
        self,
        rules_file: Path,
    ) -> None:
        import time

        cfg1 = guard.load_config()
        assert len(cfg1["rules"]) == 3

        # Wait a tick then rewrite with fewer rules
        time.sleep(0.01)
        rules_file.write_text(
            textwrap.dedent(
                """\
                version: "1.0.0"
                rules:
                  - id: only
                    pattern: '^foo'
                    decision: deny
                    reason: "nope"
                """,
            ),
        )
        # Force mtime to differ even if fs resolution is coarse
        import os

        new_mtime = rules_file.stat().st_mtime + 10
        os.utime(rules_file, (new_mtime, new_mtime))

        cfg2 = guard.load_config()
        assert len(cfg2["rules"]) == 1
        assert cfg2["rules"][0]["id"] == "only"

    def test_missing_file_raises_click_exception(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("AUTOSRE_GUARD_RULES", str(tmp_path / "nonexistent.yaml"))
        guard._CONFIG_CACHE = None
        import click

        with pytest.raises(click.ClickException, match="Guard rules not found"):
            guard.load_config()


class TestSplitChainedCommands:
    def test_plain(self) -> None:
        assert guard.split_chained_commands("git status") == ["git status"]

    def test_logical_and(self) -> None:
        assert guard.split_chained_commands("cd foo && git status") == [
            "cd foo",
            "git status",
        ]

    def test_logical_or(self) -> None:
        assert guard.split_chained_commands("foo || bar") == ["foo", "bar"]

    def test_semicolon(self) -> None:
        assert guard.split_chained_commands("a; b; c") == ["a", "b", "c"]

    def test_pipe(self) -> None:
        assert guard.split_chained_commands("a | b") == ["a", "b"]

    def test_background(self) -> None:
        assert guard.split_chained_commands("long-running &") == ["long-running"]

    def test_quoted_separator_preserved(self) -> None:
        # Separator inside quotes should not split
        assert guard.split_chained_commands('echo "a && b"') == ['echo "a && b"']

    def test_newline(self) -> None:
        assert guard.split_chained_commands("a\nb") == ["a", "b"]


class TestNormalizeForPatterns:
    def test_strips_git_c_flag(self) -> None:
        assert guard.normalize_for_patterns("git -C /path push origin dev") == "git push origin dev"

    def test_preserves_non_git(self) -> None:
        assert guard.normalize_for_patterns("ls -la") == "ls -la"

    def test_collapses_whitespace(self) -> None:
        assert guard.normalize_for_patterns("git  -C  /path   status") == "git status"


class TestEvaluateRules:
    def test_deny_rule(self, rules_file: Path) -> None:
        cfg = guard.load_config()
        _mapped, reason, decision = guard.evaluate_rules("git stash push", cfg)
        assert decision == "deny"
        assert reason == "no stash"

    def test_ask_rule(self, rules_file: Path) -> None:
        cfg = guard.load_config()
        _mapped, reason, decision = guard.evaluate_rules("git reset --hard", cfg)
        assert decision == "ask"
        assert reason == "reset is risky"

    def test_allow_rule(self, rules_file: Path) -> None:
        cfg = guard.load_config()
        _mapped, _reason, decision = guard.evaluate_rules("echo hello", cfg)
        assert decision == "allow"

    def test_no_match_allows(self, rules_file: Path) -> None:
        cfg = guard.load_config()
        _mapped, _reason, decision = guard.evaluate_rules("ls -la", cfg)
        assert decision == "allow"

    def test_python_inline_blocked(self, rules_file: Path) -> None:
        cfg = guard.load_config()
        # python -c isn't in the test rules, but the code-logic python allowlist
        # still applies. Unknown `python` invocations fall through — not denied
        # by the Python allowlist alone.
        _mapped, _reason, decision = guard.evaluate_rules("python scripts/foo.py", cfg)
        assert decision == "allow"  # scripts/*.py is allowlisted by code logic

    def test_python_m_pytest_allowed(self, rules_file: Path) -> None:
        cfg = guard.load_config()
        _mapped, _reason, decision = guard.evaluate_rules("python -m pytest tests/", cfg)
        assert decision == "allow"


class TestBlockedEnvVars:
    def test_bypass_env_blocked(
        self,
        rules_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AUTOSRE_BYPASS_GUARD", "1")
        runner = CliRunner()
        stdin = json.dumps(
            {"tool_name": "Bash", "tool_input": {"command": "ls"}},
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        decision = output["hookSpecificOutput"]["permissionDecision"]
        assert decision == "deny"
        assert "SECURITY VIOLATION" in output["hookSpecificOutput"]["permissionDecisionReason"]


class TestGuardCmd:
    @pytest.fixture(autouse=True)
    def _cleanup(self) -> None:
        yield
        # Reset the config cache between tests — otherwise a prior test's
        # rules leak into the next.
        guard._CONFIG_CACHE = None

    def test_bash_deny_flow(self, rules_file: Path) -> None:
        runner = CliRunner()
        stdin = json.dumps(
            {"tool_name": "Bash", "tool_input": {"command": "git stash pop"}},
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "no stash" in output["hookSpecificOutput"]["permissionDecisionReason"]

    def test_bash_allow_flow(self, rules_file: Path) -> None:
        runner = CliRunner()
        stdin = json.dumps(
            {"tool_name": "Bash", "tool_input": {"command": "ls -la"}},
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_non_bash_tool_allows(self, rules_file: Path) -> None:
        runner = CliRunner()
        stdin = json.dumps(
            {"tool_name": "Read", "tool_input": {"file_path": "/some/file"}},
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        assert json.loads(result.stdout)["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_gitignore_secrets_guard(self, rules_file: Path) -> None:
        runner = CliRunner()
        stdin = json.dumps(
            {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": "/repo/.gitignore",
                    "new_string": "node_modules/\nsecrets/\n",
                },
            },
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert (
            "gitignore the secrets/ directory"
            in (output["hookSpecificOutput"]["permissionDecisionReason"])
        )

    def test_chained_command_first_deny_wins(self, rules_file: Path) -> None:
        runner = CliRunner()
        stdin = json.dumps(
            {
                "tool_name": "Bash",
                "tool_input": {"command": "echo hi && git stash"},
            },
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        output = json.loads(result.stdout)
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"


class TestPackagedRulesSmoke:
    """Smoke tests against the real packaged rules file to catch YAML errors."""

    @pytest.fixture(autouse=True)
    def _unset_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AUTOSRE_GUARD_RULES", raising=False)
        guard._CONFIG_CACHE = None

    def test_packaged_rules_parse(self) -> None:
        from pathlib import Path as _Path

        src = _Path(guard.__file__).resolve().parent / "config" / "guard-rules.yaml"
        assert src.exists()
        with src.open() as f:
            data = yaml.safe_load(f)
        assert "rules" in data
        # Default ruleset: git safety, python/venv backdoors, heredoc,
        # code-quality routing through the project CLI, deps hygiene.
        # Expect ~40 rules. Guard against accidental deletion (drops
        # below 30) and against accidental re-inflation (regressions
        # that pull upstream rules back in).
        assert 30 <= len(data["rules"]) <= 80, (
            f"unexpected rule count {len(data['rules'])} — check guard-rules.yaml"
        )

    def test_git_stash_blocked_by_packaged_rules(self) -> None:
        cfg = guard.load_config()
        _, reason, decision = guard.evaluate_rules("git stash", cfg)
        assert decision == "deny"
        assert "stash" in reason.lower()

    def test_git_reset_hard_blocked_by_packaged_rules(self) -> None:
        cfg = guard.load_config()
        _, reason, decision = guard.evaluate_rules("git reset --hard HEAD~1", cfg)
        assert decision == "deny"
        assert "reset --hard" in reason

    def test_ls_allowed_by_packaged_rules(self) -> None:
        cfg = guard.load_config()
        _, _, decision = guard.evaluate_rules("ls -la", cfg)
        assert decision == "allow"

    def test_no_verify_commit_blocked(self) -> None:
        cfg = guard.load_config()
        _, _, decision = guard.evaluate_rules("git commit -m msg --no-verify", cfg)
        assert decision == "deny"

    def test_sed_on_recipe_blocked(self) -> None:
        cfg = guard.load_config()
        _, _, decision = guard.evaluate_rules(
            "sed -i 's/0.75/0.70/' autosre/backends/recipes/foo.yaml", cfg
        )
        assert decision == "deny"

    def test_cp_over_recipe_blocked(self) -> None:
        cfg = guard.load_config()
        _, _, decision = guard.evaluate_rules(
            "cp /tmp/new.yaml autosre/backends/recipes/foo.yaml", cfg
        )
        assert decision == "deny"

    def test_python_on_recipe_blocked(self) -> None:
        cfg = guard.load_config()
        _, _, decision = guard.evaluate_rules(
            "python3 edit_recipe.py autosre/backends/recipes/foo.yaml", cfg
        )
        assert decision == "deny"


class TestRecipeGuardIntegration:
    """Integration tests for Edit/Write recipe file protection."""

    @pytest.fixture
    def recipe_file(self, tmp_path: Path) -> Path:
        """Create a recipe YAML file in a path that looks like a recipe."""
        recipes_dir = tmp_path / "autosre" / "backends" / "recipes"
        recipes_dir.mkdir(parents=True)
        recipe = recipes_dir / "test-model.yaml"
        recipe.write_text(
            textwrap.dedent("""\
                model_key: test
                gpu_memory_utilization: 0.75
                max_num_seqs: 8
                max_num_batched_tokens: 4096
                extra_args:
                  - "--enable-prefix-caching"
                  - "--scheduling-policy=priority"
            """),
        )
        return recipe

    def test_edit_recipe_perf_param_denied(self, rules_file: Path, recipe_file: Path) -> None:
        runner = CliRunner()
        stdin = json.dumps(
            {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": str(recipe_file),
                    "old_string": "gpu_memory_utilization: 0.75",
                    "new_string": "gpu_memory_utilization: 0.70",
                },
            }
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert "gpu_memory_utilization" in output["hookSpecificOutput"]["permissionDecisionReason"]

    def test_edit_recipe_comment_only_allowed(self, rules_file: Path, recipe_file: Path) -> None:
        # Prepend a comment to the file content
        content = recipe_file.read_text()
        recipe_file.write_text("# old comment\n" + content)

        runner = CliRunner()
        stdin = json.dumps(
            {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": str(recipe_file),
                    "old_string": "# old comment",
                    "new_string": "# new comment",
                },
            }
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_write_recipe_perf_param_denied(self, rules_file: Path, recipe_file: Path) -> None:
        runner = CliRunner()
        stdin = json.dumps(
            {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(recipe_file),
                    "content": textwrap.dedent("""\
                    model_key: test
                    gpu_memory_utilization: 0.50
                    max_num_seqs: 12
                """),
                },
            }
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_edit_recipe_with_approval_allowed(self, rules_file: Path, recipe_file: Path) -> None:
        from autosre.hooks_backend import recipe_guard

        # Build the after content
        before = recipe_file.read_text()
        after = before.replace("gpu_memory_utilization: 0.75", "gpu_memory_utilization: 0.70")

        # Mint approval token for the after content
        recipe_guard.write_perf_approval(str(recipe_file), recipe_guard.content_hash(after))

        runner = CliRunner()
        stdin = json.dumps(
            {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": str(recipe_file),
                    "old_string": "gpu_memory_utilization: 0.75",
                    "new_string": "gpu_memory_utilization: 0.70",
                },
            }
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_write_new_recipe_allowed(self, rules_file: Path, tmp_path: Path) -> None:
        # Creating a new recipe is a create, not an edit — there's no prior
        # baseline to validate against, and Phase 2 bench runs mint the
        # first baseline for a new model.  Gating creation on an approval
        # token is circular; allow the write.
        recipes_dir = tmp_path / "autosre" / "backends" / "recipes"
        recipes_dir.mkdir(parents=True)
        new_recipe = recipes_dir / "brand-new-model.yaml"
        assert not new_recipe.exists()

        runner = CliRunner()
        stdin = json.dumps(
            {
                "tool_name": "Write",
                "tool_input": {
                    "file_path": str(new_recipe),
                    "content": textwrap.dedent("""\
                        model_key: brand-new
                        gpu_memory_utilization: 0.75
                        max_num_seqs: 8
                        max_num_batched_tokens: 4096
                        extra_args:
                          - "--enable-prefix-caching"
                          - "--scheduling-policy=priority"
                    """),
                },
            }
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"

    def test_edit_non_recipe_yaml_allowed(self, rules_file: Path, tmp_path: Path) -> None:
        normal_yaml = tmp_path / "config.yaml"
        normal_yaml.write_text("gpu_memory_utilization: 0.75\n")

        runner = CliRunner()
        stdin = json.dumps(
            {
                "tool_name": "Edit",
                "tool_input": {
                    "file_path": str(normal_yaml),
                    "old_string": "gpu_memory_utilization: 0.75",
                    "new_string": "gpu_memory_utilization: 0.50",
                },
            }
        )
        result = runner.invoke(guard.guard_cmd, input=stdin)
        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["hookSpecificOutput"]["permissionDecision"] == "allow"
