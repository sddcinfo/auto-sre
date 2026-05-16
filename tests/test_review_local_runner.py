"""Tests for autosre.review._local_provider_runner.

Mocks httpx at the module level (pattern from tests/test_proxy_logger.py)
and monkey-patches ``load_active_state`` to avoid hitting the real active
state file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from autosre.review import _local_provider_runner as runner

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolate_xdg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.delenv("AUTOSRE_REVIEW_MODEL", raising=False)
    monkeypatch.delenv("AUTOSRE_REVIEW_TIMEOUT", raising=False)


def _active_state_vllm() -> dict[str, Any]:
    return {
        "backend": "vllm",
        "model": "qwen3.6-fp8",
        "api_host": "localhost",
        "api_port": 8010,
    }


class TestStripThinkBlocks:
    def test_no_think_blocks(self) -> None:
        assert runner._strip_think_blocks("plain text") == "plain text"

    def test_single_think_block(self) -> None:
        text = '<think>reasoning here</think>\n{"findings": []}'
        assert runner._strip_think_blocks(text).strip() == '{"findings": []}'

    def test_multiline_think_block(self) -> None:
        text = '<think>\nlong\nreasoning\nchain\n</think>\n{"findings": [{"severity": "P0"}]}'
        out = runner._strip_think_blocks(text)
        assert "reasoning" not in out
        assert '"severity": "P0"' in out

    def test_multiple_think_blocks(self) -> None:
        text = "<think>a</think>middle<think>b</think>end"
        assert runner._strip_think_blocks(text) == "middleend"

    def test_non_greedy(self) -> None:
        # If there are two blocks, we should strip both, not merge across them.
        text = "<think>first</think> real text <think>second</think> more"
        out = runner._strip_think_blocks(text)
        assert "real text" in out
        assert "more" in out
        assert "first" not in out
        assert "second" not in out


class TestGetBaseUrl:
    def test_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "load_active_state", _active_state_vllm)
        assert runner._get_base_url() == "http://localhost:8010"

    def test_no_active_state_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "load_active_state", lambda: None)
        with pytest.raises(RuntimeError, match="No active"):
            runner._get_base_url()

    def test_missing_port_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(runner, "load_active_state", lambda: {"backend": "vllm"})
        with pytest.raises(RuntimeError, match="no api_port"):
            runner._get_base_url()

    def test_defaults_host_to_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            runner,
            "load_active_state",
            lambda: {"backend": "vllm", "api_port": 8010},
        )
        assert runner._get_base_url() == "http://localhost:8010"


class TestResolveModelId:
    def test_env_var_takes_priority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AUTOSRE_REVIEW_MODEL", "Qwen/Qwen3.6-35B-A3B-FP8")
        resolved = runner._resolve_model_id("http://localhost:8010", _active_state_vllm())
        assert resolved == "Qwen/Qwen3.6-35B-A3B-FP8"

    @patch("autosre.review._local_provider_runner.httpx.Client")
    def test_fetches_from_v1_models_when_env_unset(
        self,
        mock_client_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("AUTOSRE_REVIEW_MODEL", raising=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": [{"id": "Qwen/Qwen3.6-35B-A3B-FP8"}]}
        mock_client = MagicMock()
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        resolved = runner._resolve_model_id("http://localhost:8010", _active_state_vllm())
        assert resolved == "Qwen/Qwen3.6-35B-A3B-FP8"
        mock_client.get.assert_called_once_with("http://localhost:8010/v1/models")

    @patch("autosre.review._local_provider_runner.httpx.Client")
    def test_falls_back_to_backend_when_v1_models_fails(
        self,
        mock_client_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("AUTOSRE_REVIEW_MODEL", raising=False)
        # httpx raises inside the context manager
        mock_client_cls.return_value.__enter__.side_effect = httpx.ConnectError("boom")

        # Stub get_backend to return a mock backend with get_claude_model_arg
        fake_backend = MagicMock()
        fake_backend.default_model = "qwen3.6-fp8"
        fake_backend.get_claude_model_arg.return_value = "Intel/resolved-from-backend"
        fake_get_backend = MagicMock(return_value=fake_backend)

        import autosre.backends as backends_pkg

        monkeypatch.setattr(backends_pkg, "get_backend", fake_get_backend, raising=False)

        resolved = runner._resolve_model_id("http://localhost:8010", _active_state_vllm())
        assert resolved == "Intel/resolved-from-backend"
        fake_backend.get_claude_model_arg.assert_called_once_with("qwen3.6-fp8")


class TestCallModel:
    @patch("autosre.review._local_provider_runner.httpx.Client")
    def test_posts_to_chat_completions(self, mock_client_cls: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '<think>thinking</think>\n{"findings": [], "questions": []}',
                    },
                },
            ],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        out = runner._call_model(
            "http://localhost:8010",
            "Qwen/Qwen3.6-35B-A3B-FP8",
            "review this plan",
        )
        assert "<think>" in out  # stripping happens in main(), not here

        mock_client.post.assert_called_once()
        _args, kwargs = mock_client.post.call_args
        assert mock_client.post.call_args[0][0] == "http://localhost:8010/v1/chat/completions"
        payload = kwargs["json"]
        assert payload["model"] == "Qwen/Qwen3.6-35B-A3B-FP8"
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][1]["role"] == "user"
        assert payload["messages"][1]["content"] == "review this plan"
        assert payload["priority"] == 20

    @patch("autosre.review._local_provider_runner.httpx.Client")
    def test_raises_on_http_error(self, mock_client_cls: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500",
            request=MagicMock(),
            response=MagicMock(),
        )
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        with pytest.raises(httpx.HTTPStatusError):
            runner._call_model("http://localhost:8010", "model-id", "prompt")

    @patch("autosre.review._local_provider_runner.httpx.Client")
    def test_raises_on_unexpected_shape(self, mock_client_cls: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"unexpected": "shape"}
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value.__enter__.return_value = mock_client

        with pytest.raises(RuntimeError, match="unexpected response shape"):
            runner._call_model("http://localhost:8010", "model-id", "prompt")


class TestMain:
    def test_empty_prompt_returns_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        rc = runner.main([""])
        assert rc == 2
        captured = capsys.readouterr()
        assert "empty prompt" in captured.err

    def test_no_active_backend_returns_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(runner, "load_active_state", lambda: None)
        rc = runner.main(["review this plan"])
        assert rc == 1
        captured = capsys.readouterr()
        assert "No active" in captured.err

    def test_happy_path_strips_think_blocks_in_output(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(runner, "load_active_state", _active_state_vllm)
        monkeypatch.setattr(
            runner,
            "_resolve_model_id",
            lambda _base, _active: "Qwen/Qwen3.6-35B-A3B-FP8",
        )
        monkeypatch.setattr(
            runner,
            "_call_model",
            lambda _base, _model, _prompt: (
                '<think>some reasoning</think>\n{"findings": [], "questions": []}'
            ),
        )

        rc = runner.main(["review this plan"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "<think>" not in out
        assert '"findings"' in out

    def test_http_error_returns_1(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(runner, "load_active_state", _active_state_vllm)
        monkeypatch.setattr(
            runner,
            "_resolve_model_id",
            lambda _base, _active: "model-id",
        )

        def _raise(*_args: object, **_kwargs: object) -> None:
            raise httpx.ConnectError("no route")

        monkeypatch.setattr(runner, "_call_model", _raise)
        rc = runner.main(["prompt"])
        assert rc == 1
        assert "HTTP error" in capsys.readouterr().err
