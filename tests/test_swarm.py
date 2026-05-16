"""Tests for autosre.swarm module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from autosre.swarm.launcher import (
    DEFAULT_ANTHROPIC_MODEL,
    EvalLaunchSpec,
    SwarmLauncher,
)
from autosre.swarm.templates import TASK_TEMPLATES


class TestTaskTemplates:
    def test_all_templates_defined(self) -> None:
        expected = {
            "code-review",
            "architecture-analysis",
            "incident-response",
            "content-generation",
            "data-analysis",
        }
        assert set(TASK_TEMPLATES.keys()) == expected

    def test_code_review_template(self) -> None:
        t = TASK_TEMPLATES["code-review"]
        assert t.num_agents == 4
        assert len(t.agent_roles) == 4
        assert "Security" in t.agent_roles[0]

    def test_incident_response_template(self) -> None:
        t = TASK_TEMPLATES["incident-response"]
        assert t.num_agents == 5
        assert "Incident Commander" in t.agent_roles[0]

    def test_all_templates_have_roles(self) -> None:
        for name, tmpl in TASK_TEMPLATES.items():
            assert len(tmpl.agent_roles) == tmpl.num_agents, f"{name}: role count mismatch"

    def test_format_prompt(self) -> None:
        t = TASK_TEMPLATES["code-review"]
        prompt = t.format_prompt()
        assert "Agent 1:" in prompt
        assert "Security" in prompt
        assert "comprehensive code review" in prompt


class TestSwarmLauncher:
    def test_build_env_includes_agent_teams(self) -> None:
        mock_backend = MagicMock()
        mock_backend.get_claude_env.return_value = {
            "ANTHROPIC_BASE_URL": "http://192.168.1.101:8000",
            "ANTHROPIC_AUTH_TOKEN": "vllm",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        }

        launcher = SwarmLauncher(mock_backend)
        env = launcher.build_env()

        assert env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"
        assert env["ANTHROPIC_BASE_URL"] == "http://192.168.1.101:8000"

    def test_build_env_purges_cloud_credentials(self) -> None:
        mock_backend = MagicMock()
        mock_backend.get_claude_env.return_value = {
            "ANTHROPIC_BASE_URL": "http://localhost:8000",
            "ANTHROPIC_AUTH_TOKEN": "vllm",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        }

        launcher = SwarmLauncher(mock_backend)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-cloud-key"}):
            env = launcher.build_env()
            # Cloud key is replaced with local-vllm, not removed
            assert env["ANTHROPIC_API_KEY"] == "local-vllm"

    def test_build_launch_cmd_basic(self) -> None:
        mock_backend = MagicMock()
        launcher = SwarmLauncher(mock_backend)

        cmd = launcher.build_launch_cmd("qwen3.6:35b-a3b")
        assert "--bare" in cmd
        assert "--model=qwen3.6:35b-a3b" in cmd
        assert any("--mcp-config" in c for c in cmd)
        assert any("--system-prompt" in c for c in cmd)

    def test_build_launch_cmd_with_template(self) -> None:
        mock_backend = MagicMock()
        template = TASK_TEMPLATES["code-review"]
        launcher = SwarmLauncher(mock_backend, template=template)

        cmd = launcher.build_launch_cmd("qwen3.6-fp8")
        assert any("code review" in arg.lower() for arg in cmd)

    def test_launch_no_claude(self) -> None:
        mock_backend = MagicMock()
        launcher = SwarmLauncher(mock_backend)

        with patch("shutil.which", return_value=None):
            try:
                launcher.launch()
            except RuntimeError as e:
                assert "claude" in str(e).lower()


class TestAnthropicProvider:
    """Provider=anthropic must never inject Anthropic env overrides."""

    def _backend(self) -> MagicMock:
        mock_backend = MagicMock()
        mock_backend.get_claude_env.return_value = {
            "ANTHROPIC_BASE_URL": "http://localhost:8011",
            "ANTHROPIC_AUTH_TOKEN": "vllm",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        }
        mock_backend.name = "stub"
        mock_backend.default_model = "stub-model"
        mock_backend.get_claude_model_arg.return_value = "stub-model"
        mock_backend.get_api_url.return_value = "http://localhost:8000"
        return mock_backend

    def test_build_env_purges_all_overrides(self) -> None:
        launcher = SwarmLauncher(self._backend(), provider="anthropic")

        dirty = {
            "ANTHROPIC_API_KEY": "sk-should-be-gone",
            "ANTHROPIC_AUTH_TOKEN": "should-be-gone",
            "ANTHROPIC_BASE_URL": "http://wrong-proxy:1234",
            "ANTHROPIC_MODEL": "wrong-model",
            "ANTHROPIC_SMALL_FAST_MODEL": "wrong-small",
            "ANTHROPIC_LARGE_MODEL": "wrong-large",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "99",
        }
        with patch.dict("os.environ", dirty, clear=False):
            env = launcher.build_env()

        for key in dirty:
            assert key not in env, (
                f"anthropic provider must not propagate {key} (got {env.get(key)!r})"
            )
        # Agent teams must still be enabled.
        assert env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] == "1"

    def test_build_env_does_not_call_backend(self) -> None:
        backend = self._backend()
        launcher = SwarmLauncher(backend, provider="anthropic")
        launcher.build_env()
        backend.get_claude_env.assert_not_called()

    def test_build_launch_cmd_uses_anthropic_model(self) -> None:
        launcher = SwarmLauncher(
            self._backend(),
            provider="anthropic",
            anthropic_model="claude-opus-4-6",
        )
        cmd = launcher.build_launch_cmd("claude-opus-4-6")
        assert "--model=claude-opus-4-6" in cmd
        # Interactive mode: no stream-json
        assert "--output-format=stream-json" not in cmd

    def test_default_anthropic_model_is_opus_4_6(self) -> None:
        assert DEFAULT_ANTHROPIC_MODEL.startswith("claude-opus-4-6")


class TestLocalProviderUnchanged:
    """Existing local-mode behavior must not regress."""

    def test_local_still_sets_local_vllm_key(self) -> None:
        mock_backend = MagicMock()
        mock_backend.get_claude_env.return_value = {
            "ANTHROPIC_BASE_URL": "http://localhost:8011",
        }
        launcher = SwarmLauncher(mock_backend, provider="local")
        env = launcher.build_env()
        assert env["ANTHROPIC_API_KEY"] == "local-vllm"
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:8011"


class TestEvalLaunchSpec:
    def test_eval_mode_build_launch_cmd_adds_stream_json(self, tmp_path: Path) -> None:
        mock_backend = MagicMock()
        mock_backend.get_claude_model_arg.return_value = "stub"
        launcher = SwarmLauncher(mock_backend, provider="local")

        spec = EvalLaunchSpec(
            capture_dir=tmp_path / "capture",
            worktree_path=tmp_path / "worktree",
            findings_file=tmp_path / "agent-outputs" / "findings.json",
            suite_name="security",
            run_id="test-run",
        )
        # worktree must exist for settings write
        spec.capture_dir.mkdir(parents=True)
        spec.worktree_path.mkdir(parents=True)

        cmd = launcher.build_launch_cmd("stub", eval_mode=spec)
        assert "--output-format=stream-json" in cmd
        assert "--verbose" in cmd
        # Eval settings file must live inside the capture dir, not tempdir.
        settings_flags = [c for c in cmd if c.startswith("--settings=")]
        assert len(settings_flags) == 1
        settings_path = Path(settings_flags[0].split("=", 1)[1])
        assert settings_path.parent == spec.capture_dir

    def test_eval_settings_denies_bash(self, tmp_path: Path) -> None:
        import json as _json

        mock_backend = MagicMock()
        launcher = SwarmLauncher(mock_backend, provider="local")
        spec = EvalLaunchSpec(
            capture_dir=tmp_path / "capture",
            worktree_path=tmp_path / "worktree",
            findings_file=tmp_path / "f.json",
            suite_name="security",
            run_id="r",
        )
        settings_path = launcher._eval_settings_path(spec)
        settings = _json.loads(settings_path.read_text())
        allow = settings["permissions"]["allow"]
        deny = settings["permissions"]["deny"]
        assert "Bash(*)" not in allow
        assert "Bash" not in allow
        assert "Bash" in deny
