"""Tests for base backend functionality."""

from pathlib import Path
from unittest.mock import patch

import pytest

from autosre.backends.base import (
    Backend,
    BackendType,
    clear_active_state,
    detect_platform,
    load_active_state,
    save_active_state,
)


class TestDetectPlatform:
    """Tests for platform detection."""

    def test_always_returns_ollama(self) -> None:
        """detect_platform always returns OLLAMA (universal default)."""
        assert detect_platform() == BackendType.OLLAMA

    def test_returns_ollama_on_linux(self) -> None:
        """detect_platform returns OLLAMA even on Linux."""
        with patch("platform.system", return_value="Linux"):
            assert detect_platform() == BackendType.OLLAMA

    def test_returns_ollama_on_mac(self) -> None:
        """detect_platform returns OLLAMA on Mac."""
        with patch("platform.system", return_value="Darwin"):
            assert detect_platform() == BackendType.OLLAMA


class TestBackendType:
    """Tests for BackendType enum."""

    def test_ollama_value(self) -> None:
        assert BackendType.OLLAMA.value == "ollama"

    def test_llamacpp_value(self) -> None:
        assert BackendType.LLAMACPP.value == "llamacpp"

    def test_backend_count(self) -> None:
        assert len(BackendType) == 4  # ollama, llamacpp, vllm, mlx-dflash


class ConcreteBackend(Backend):
    """Concrete implementation for testing abstract Backend."""

    name = "test"
    description = "Test backend"

    def check_requirements(self) -> tuple[bool, list[str]]:
        return True, []

    def setup(self, **kwargs: object) -> bool:
        return True

    def start(self, model: str | None = None, **kwargs: object) -> dict[str, object]:
        return {"status": "started"}

    def stop(self) -> bool:
        return True

    def status(self) -> dict[str, object]:
        return {"running": False}

    def get_claude_env(self) -> dict[str, str]:
        return {
            "ANTHROPIC_BASE_URL": "http://localhost:8080",
            "ANTHROPIC_AUTH_TOKEN": "test",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        }

    def get_claude_model_arg(self, model: str) -> str:
        return self.get_model_id(model)

    def is_healthy(self) -> bool:
        return False


class TestBackendBase:
    """Tests for Backend base class."""

    def test_data_dir_creation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that data directory is created correctly."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        backend = ConcreteBackend()

        data_dir = backend.data_dir
        assert data_dir.exists()
        assert data_dir == tmp_path / "autosre"

    def test_pids_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test PIDs file path."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        backend = ConcreteBackend()

        assert backend.pids_file == tmp_path / "autosre" / "test.pids"

    def test_save_and_load_pids(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test saving and loading PIDs."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        backend = ConcreteBackend()

        pids = {"process1": 1234, "process2": 5678}
        backend._save_pids(pids)

        loaded = backend._load_pids()
        assert loaded == pids

    def test_clear_pids(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test clearing PIDs file."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        backend = ConcreteBackend()

        backend._save_pids({"test": 123})
        assert backend.pids_file.exists()

        backend._clear_pids()
        assert not backend.pids_file.exists()

    def test_load_pids_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading PIDs when file doesn't exist."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        backend = ConcreteBackend()

        assert backend._load_pids() == {}

    def test_get_model_id_known(self) -> None:
        """Test getting model ID for known model."""
        backend = ConcreteBackend()
        backend.models = {"short": "full/model/path"}

        assert backend.get_model_id("short") == "full/model/path"

    def test_get_model_id_unknown(self) -> None:
        """Test getting model ID for unknown model returns input."""
        backend = ConcreteBackend()
        backend.models = {}

        assert backend.get_model_id("custom/model") == "custom/model"

    def test_get_claude_env(self) -> None:
        """Test get_claude_env returns expected keys."""
        backend = ConcreteBackend()
        env = backend.get_claude_env()
        assert "ANTHROPIC_BASE_URL" in env
        assert "ANTHROPIC_AUTH_TOKEN" in env
        assert "CLAUDE_CODE_ATTRIBUTION_HEADER" in env

    def test_get_claude_model_arg(self) -> None:
        """Test get_claude_model_arg resolves model key."""
        backend = ConcreteBackend()
        backend.models = {"short": "full/model/path"}
        assert backend.get_claude_model_arg("short") == "full/model/path"


class TestActiveState:
    """Tests for active backend state file helpers."""

    def test_save_and_load(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test saving and loading active state."""
        state_file = tmp_path / "autosre" / "active.json"
        monkeypatch.setattr("autosre.backends.base.ACTIVE_STATE_FILE", state_file)

        state = {"backend": "ollama", "model": "qwen3.6", "api_port": 11434}
        save_active_state(state)

        loaded = load_active_state()
        assert loaded == state

    def test_load_nonexistent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading when file doesn't exist."""
        state_file = tmp_path / "autosre" / "active.json"
        monkeypatch.setattr("autosre.backends.base.ACTIVE_STATE_FILE", state_file)

        assert load_active_state() is None

    def test_clear(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test clearing active state."""
        state_file = tmp_path / "autosre" / "active.json"
        monkeypatch.setattr("autosre.backends.base.ACTIVE_STATE_FILE", state_file)

        save_active_state({"backend": "ollama"})
        assert state_file.exists()

        clear_active_state()
        assert not state_file.exists()

    def test_load_corrupt_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading corrupt JSON returns None."""
        state_file = tmp_path / "autosre" / "active.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text("not json{{{")
        monkeypatch.setattr("autosre.backends.base.ACTIVE_STATE_FILE", state_file)

        assert load_active_state() is None
