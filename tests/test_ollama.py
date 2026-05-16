"""Tests for Ollama backend."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autosre.backends.ollama import OLLAMA_MIN_ANTHROPIC_VERSION, OllamaBackend


class TestOllamaBackend:
    """Tests for Ollama backend."""

    def test_name_and_description(self) -> None:
        """Test backend name and description."""
        backend = OllamaBackend()
        assert backend.name == "ollama"
        assert "Ollama" in backend.description

    def test_models_defined(self) -> None:
        """Test that models are defined."""
        backend = OllamaBackend()
        assert backend.models == {"qwen3.6": "qwen3.6:35b-a3b"}
        assert backend.default_model == "qwen3.6"

    def test_qwen36_model(self) -> None:
        """Test Qwen3.6 model tag."""
        backend = OllamaBackend()
        assert backend.models["qwen3.6"] == "qwen3.6:35b-a3b"

    def test_check_requirements_ollama_installed(self) -> None:
        """Test requirements check when Ollama is installed."""
        backend = OllamaBackend()

        with patch("shutil.which", return_value="/usr/local/bin/ollama"):
            ok, missing = backend.check_requirements()
            assert ok is True
            assert missing == []

    def test_check_requirements_ollama_not_installed(self) -> None:
        """Test requirements check when Ollama is not installed."""
        backend = OllamaBackend()

        with patch("shutil.which", return_value=None):
            ok, missing = backend.check_requirements()
            assert ok is False
            assert len(missing) == 1
            assert "Ollama" in missing[0]


class TestOllamaVersionDetection:
    """Tests for Ollama version detection."""

    def test_get_version_valid(self) -> None:
        """Test getting Ollama version from valid output."""
        backend = OllamaBackend()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ollama version 0.14.2"

        with patch("subprocess.run", return_value=mock_result):
            version = backend._get_ollama_version()
            assert version == (0, 14, 2)

    def test_get_version_not_installed(self) -> None:
        """Test getting version when Ollama not installed."""
        backend = OllamaBackend()

        with patch("subprocess.run", side_effect=FileNotFoundError):
            version = backend._get_ollama_version()
            assert version is None

    def test_get_version_invalid_output(self) -> None:
        """Test getting version with invalid output."""
        backend = OllamaBackend()

        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "no version here"

        with patch("subprocess.run", return_value=mock_result):
            version = backend._get_ollama_version()
            assert version is None


class TestOllamaAnthropicSupport:
    """Tests for Ollama Anthropic API support detection."""

    def test_supports_anthropic_api_new_version(self) -> None:
        """Test that Ollama >= 0.14.0 supports Anthropic API."""
        backend = OllamaBackend()

        with patch.object(backend, "_get_ollama_version", return_value=(0, 14, 2)):
            assert backend._supports_anthropic_api() is True

    def test_supports_anthropic_api_min_version(self) -> None:
        """Test that minimum supported version works."""
        backend = OllamaBackend()

        with patch.object(
            backend, "_get_ollama_version", return_value=OLLAMA_MIN_ANTHROPIC_VERSION
        ):
            assert backend._supports_anthropic_api() is True

    def test_supports_anthropic_api_old_version(self) -> None:
        """Test that old Ollama versions don't support Anthropic API."""
        backend = OllamaBackend()

        with patch.object(backend, "_get_ollama_version", return_value=(0, 5, 0)):
            assert backend._supports_anthropic_api() is False

    def test_supports_anthropic_api_no_version(self) -> None:
        """Test when version can't be determined."""
        backend = OllamaBackend()

        with patch.object(backend, "_get_ollama_version", return_value=None):
            assert backend._supports_anthropic_api() is False

    def test_min_anthropic_version_constant(self) -> None:
        """Test the minimum version constant."""
        assert OLLAMA_MIN_ANTHROPIC_VERSION == (0, 14, 0)


class TestOllamaClaudeEnv:
    """Tests for Claude Code environment configuration."""

    def test_get_claude_env_keys(self) -> None:
        """Test get_claude_env returns correct keys."""
        backend = OllamaBackend()
        env = backend.get_claude_env()

        assert "ANTHROPIC_BASE_URL" in env
        assert "ANTHROPIC_AUTH_TOKEN" in env
        assert "CLAUDE_CODE_ATTRIBUTION_HEADER" in env

    def test_get_claude_env_no_model(self) -> None:
        """Test get_claude_env does NOT include ANTHROPIC_MODEL."""
        backend = OllamaBackend()
        env = backend.get_claude_env()

        assert "ANTHROPIC_MODEL" not in env

    def test_get_claude_env_base_url(self) -> None:
        """Test base URL points to Ollama port."""
        backend = OllamaBackend()
        env = backend.get_claude_env()

        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:11434"

    def test_get_claude_env_attribution_header(self) -> None:
        """Test attribution header is disabled."""
        backend = OllamaBackend()
        env = backend.get_claude_env()

        assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"


class TestOllamaClaudeModelArg:
    """Tests for Claude --model argument."""

    def test_get_claude_model_arg_default(self) -> None:
        """Test model arg for default model."""
        backend = OllamaBackend()
        assert backend.get_claude_model_arg("qwen3.6") == "qwen3.6:35b-a3b"

    def test_get_claude_model_arg_custom(self) -> None:
        """Test model arg for custom model ID."""
        backend = OllamaBackend()
        assert backend.get_claude_model_arg("custom:latest") == "custom:latest"


class TestOllamaStart:
    """Tests for Ollama start behavior."""

    def test_start_requires_anthropic_api(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that start fails if Ollama doesn't support Anthropic API."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        backend = OllamaBackend()

        with (  # noqa: SIM117
            patch.object(backend, "_supports_anthropic_api", return_value=False),
            patch.object(backend, "_get_ollama_version", return_value=(0, 5, 0)),
        ):
            with pytest.raises(RuntimeError, match="Anthropic API"):
                backend.start(model="qwen3.6")

    def test_start_writes_active_state(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that start writes active.json."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        state_file = tmp_path / "autosre" / "active.json"

        def _save(s: dict[str, object]) -> None:
            state_file.parent.mkdir(parents=True, exist_ok=True)
            state_file.write_text(json.dumps(s))

        monkeypatch.setattr("autosre.backends.ollama.save_active_state", _save)

        backend = OllamaBackend()

        with (
            patch.object(backend, "_supports_anthropic_api", return_value=True),
            patch.object(backend, "_is_ollama_running", return_value=True),
            patch.object(backend, "_has_model", return_value=True),
            patch.object(backend, "_stop_tracked_processes"),
        ):
            result = backend.start(model="qwen3.6")

        assert result["api_port"] == 11434
        assert result["model"] == "qwen3.6:35b-a3b"
