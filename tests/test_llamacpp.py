"""Tests for llama.cpp backend."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autosre.backends.llamacpp import LlamaCppBackend


class TestLlamaCppBackend:
    """Tests for llama.cpp backend."""

    def test_name_and_description(self) -> None:
        """Test backend name and description."""
        backend = LlamaCppBackend()
        assert backend.name == "llamacpp"
        assert "llama" in backend.description.lower()

    def test_models_defined(self) -> None:
        """Test that models are defined."""
        backend = LlamaCppBackend()
        assert backend.models == {"qwen3.6": "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"}
        assert backend.default_model == "qwen3.6"

    def test_models_are_unsloth_ggufs(self) -> None:
        """Test model IDs are unsloth GGUFs."""
        backend = LlamaCppBackend()
        for model_id in backend.models.values():
            assert "unsloth/" in model_id
            assert "GGUF" in model_id

    def test_api_port(self) -> None:
        """Test default API port."""
        backend = LlamaCppBackend()
        assert backend.api_port == 8080


class TestLlamaCppRequirements:
    """Tests for llama.cpp requirements check."""

    def test_check_requirements_installed(self) -> None:
        """Test requirements check when llama-server is installed."""
        backend = LlamaCppBackend()

        with patch("shutil.which", return_value="/opt/homebrew/bin/llama-server"):
            ok, missing = backend.check_requirements()
            assert ok is True
            assert missing == []

    def test_check_requirements_not_installed(self) -> None:
        """Test requirements check when llama-server is not installed."""
        backend = LlamaCppBackend()

        with patch("shutil.which", return_value=None):
            ok, missing = backend.check_requirements()
            assert ok is False
            assert len(missing) == 1
            assert "llama-server" in missing[0]


class TestLlamaCppClaudeEnv:
    """Tests for Claude Code environment configuration."""

    def test_get_claude_env_keys(self) -> None:
        """Test get_claude_env returns correct keys."""
        backend = LlamaCppBackend()
        env = backend.get_claude_env()

        assert "ANTHROPIC_BASE_URL" in env
        assert "ANTHROPIC_AUTH_TOKEN" in env
        assert "CLAUDE_CODE_ATTRIBUTION_HEADER" in env

    def test_get_claude_env_no_model(self) -> None:
        """Test get_claude_env does NOT include ANTHROPIC_MODEL."""
        backend = LlamaCppBackend()
        env = backend.get_claude_env()

        assert "ANTHROPIC_MODEL" not in env

    def test_get_claude_env_base_url(self) -> None:
        """Test base URL points to llama-server port."""
        backend = LlamaCppBackend()
        env = backend.get_claude_env()

        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:8080"

    def test_get_claude_env_auth_token(self) -> None:
        """Test auth token is set."""
        backend = LlamaCppBackend()
        env = backend.get_claude_env()

        assert env["ANTHROPIC_AUTH_TOKEN"] == "llamacpp"


class TestLlamaCppClaudeModelArg:
    """Tests for Claude --model argument."""

    def test_get_claude_model_arg_default(self) -> None:
        """Test model arg for default model."""
        backend = LlamaCppBackend()
        assert backend.get_claude_model_arg("qwen3.6") == "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"

    def test_get_claude_model_arg_custom(self) -> None:
        """Test model arg for custom model ID."""
        backend = LlamaCppBackend()
        assert backend.get_claude_model_arg("my/custom:Q8") == "my/custom:Q8"


class TestLlamaCppStart:
    """Tests for llama-server start behavior."""

    def test_start_constructs_correct_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that start constructs the correct llama-server command."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        state_file = tmp_path / "autosre" / "active.json"

        def _save(s: dict[str, object]) -> None:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(s))

        monkeypatch.setattr("autosre.backends.llamacpp.save_active_state", _save)

        backend = LlamaCppBackend()

        mock_proc = MagicMock()
        mock_proc.pid = 12345

        with (
            patch.object(backend, "_stop_tracked_processes"),
            patch("subprocess.Popen", return_value=mock_proc) as mock_popen,
            patch.object(backend, "_wait_for_server", return_value=True),
        ):
            result = backend.start(model="qwen3.6")

        # Verify command
        cmd = mock_popen.call_args[0][0]
        assert cmd[0] == "llama-server"
        assert "-hf" in cmd
        assert "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M" in cmd
        assert "--port" in cmd
        assert "8080" in cmd
        assert "-ngl" in cmd
        assert "99" in cmd

        assert result["api_port"] == 8080
        assert result["model"] == "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M"

    def test_start_fails_on_timeout(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that start raises on server timeout."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        backend = LlamaCppBackend()

        mock_proc = MagicMock()
        mock_proc.pid = 12345

        with (  # noqa: SIM117
            patch.object(backend, "_stop_tracked_processes"),
            patch("subprocess.Popen", return_value=mock_proc),
            patch.object(backend, "_wait_for_server", return_value=False),
        ):
            with pytest.raises(RuntimeError, match="failed to start"):
                backend.start(model="qwen3.6")


class TestLlamaCppStop:
    """Tests for llama-server stop behavior."""

    def test_stop_clears_active_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that stop clears active.json."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

        cleared: list[bool] = []
        monkeypatch.setattr(
            "autosre.backends.llamacpp.clear_active_state", lambda: cleared.append(True)
        )

        backend = LlamaCppBackend()
        with patch.object(backend, "_stop_tracked_processes", return_value=True):
            backend.stop()

        assert len(cleared) == 1
