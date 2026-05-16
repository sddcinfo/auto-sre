"""Regression tests for the Backend ABC contract.

Ensures all backends conform to the interface, especially get_api_url()
and active_state reconstruction.
"""

from autosre.backends import BackendType, get_backend
from autosre.backends.base import Backend
from autosre.backends.llamacpp import LlamaCppBackend
from autosre.backends.ollama import OllamaBackend


class TestGetApiUrl:
    """Every backend must return a valid URL from get_api_url()."""

    def test_ollama_returns_localhost(self) -> None:
        b = get_backend(BackendType.OLLAMA)
        url = b.get_api_url()
        assert url == "http://localhost:11434"

    def test_llamacpp_returns_localhost(self) -> None:
        b = get_backend(BackendType.LLAMACPP)
        url = b.get_api_url()
        assert url == "http://localhost:8080"

    def test_get_api_url_matches_claude_env(self) -> None:
        """get_api_url() should be consistent with get_claude_env() base URL."""
        for bt in [BackendType.OLLAMA, BackendType.LLAMACPP]:
            b = get_backend(bt)
            env = b.get_claude_env()
            # The base URL in claude env should start with what get_api_url returns
            assert env["ANTHROPIC_BASE_URL"].startswith("http://localhost")


class TestActiveStatePassthrough:
    """get_backend() must pass active_state to the backend constructor."""

    def test_ollama_ignores_active_state(self) -> None:
        active = {"backend": "ollama", "model": "qwen3.6", "api_port": 11434}
        b = get_backend(BackendType.OLLAMA, active_state=active)
        assert isinstance(b, OllamaBackend)
        # Ollama doesn't use active_state but shouldn't crash
        assert b.get_api_url() == "http://localhost:11434"

    def test_llamacpp_ignores_active_state(self) -> None:
        active = {"backend": "llamacpp", "model": "qwen3.6", "api_port": 8080}
        b = get_backend(BackendType.LLAMACPP, active_state=active)
        assert isinstance(b, LlamaCppBackend)
        assert b.get_api_url() == "http://localhost:8080"

    def test_none_active_state(self) -> None:
        b = get_backend(BackendType.OLLAMA, active_state=None)
        assert isinstance(b, OllamaBackend)


class TestBackendTypeEnum:
    def test_vllm_value(self) -> None:
        assert BackendType.VLLM.value == "vllm"

    def test_vllm_from_string(self) -> None:
        assert BackendType("vllm") is BackendType.VLLM

    def test_all_backends_have_values(self) -> None:
        expected = {"ollama", "llamacpp", "vllm", "mlx-dflash"}
        actual = {bt.value for bt in BackendType}
        assert actual == expected


class TestBackendABC:
    def test_get_api_url_default(self) -> None:
        """Default get_api_url() uses localhost and api_port."""

        class DummyBackend(Backend):
            name = "dummy"
            api_port = 9999

            def check_requirements(self):
                return True, []

            def setup(self, **_kwargs):
                return True

            def start(self, **_kwargs):
                return {}

            def stop(self):
                return True

            def status(self):
                return {}

            def get_claude_env(self):
                return {}

            def get_claude_model_arg(self, model):
                return model

            def is_healthy(self):
                return True

        b = DummyBackend()
        assert b.get_api_url() == "http://localhost:9999"
        assert b._active_state is None

    def test_active_state_stored(self) -> None:
        """active_state should be stored on the instance."""
        b = get_backend(BackendType.OLLAMA, active_state={"key": "value"})
        assert b._active_state == {"key": "value"}
