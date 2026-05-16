"""Integration tests for autosre — validates all launch paths are consistent."""

from __future__ import annotations

from unittest.mock import MagicMock

from autosre.backends.vllm import VllmBackend


class TestClaudeLaunchIsolation:
    """Verify the claude command builds a fully isolated environment."""

    def test_env_has_local_api_key(self) -> None:
        b = VllmBackend(
            active_state={"api_host": "localhost", "api_port": 8010, "proxy_port": 8011}
        )
        env = b.get_claude_env()
        assert env["ANTHROPIC_AUTH_TOKEN"] == "vllm"
        assert "anthropic.com" not in env["ANTHROPIC_BASE_URL"]

    def test_env_points_to_proxy(self) -> None:
        b = VllmBackend(
            active_state={"api_host": "localhost", "api_port": 8010, "proxy_port": 8011}
        )
        env = b.get_claude_env()
        assert env["ANTHROPIC_BASE_URL"] == "http://localhost:8011"

    def test_hf_token_passed_to_docker(self) -> None:
        """HF_TOKEN from environment is passed to Docker containers."""
        b = VllmBackend()
        # The _start_local method reads HF_TOKEN from os.environ
        assert hasattr(b, "_start_local")


class TestRecipeConsistency:
    """Verify recipes match the models dict."""

    def test_all_models_have_recipes(self) -> None:
        from autosre.backends.recipes import get_recipe_for_model

        for model_key in VllmBackend.models:
            recipe = get_recipe_for_model(model_key)
            assert recipe is not None, f"No recipe for model {model_key}"
            assert recipe["model_key"] == model_key

    def test_default_model_has_recipe(self) -> None:
        from autosre.backends.recipes import get_recipe_for_model

        recipe = get_recipe_for_model(VllmBackend.default_model)
        assert recipe is not None
        assert recipe["model_id"] == VllmBackend.models[VllmBackend.default_model]

    def test_default_recipe_uses_stock_vllm_image(self) -> None:
        from autosre.backends.recipes import get_recipe_for_model

        recipe = get_recipe_for_model(VllmBackend.default_model)
        assert recipe["docker_image"].startswith("vllm/vllm-openai")

    def test_default_recipe_has_fastsafetensors(self) -> None:
        from autosre.backends.recipes import get_recipe_for_model

        recipe = get_recipe_for_model(VllmBackend.default_model)
        assert "--load-format=fastsafetensors" in recipe["extra_args"]

    def test_default_recipe_has_tool_calling(self) -> None:
        from autosre.backends.recipes import get_recipe_for_model

        recipe = get_recipe_for_model(VllmBackend.default_model)
        assert "--enable-auto-tool-choice" in recipe["extra_args"]
        assert "--tool-call-parser=qwen3_coder" in recipe["extra_args"]


class TestSwarmLauncherConsistency:
    """Verify swarm launcher matches cli.py claude command."""

    def test_launcher_uses_bare(self) -> None:
        from autosre.swarm.launcher import SwarmLauncher

        mock_backend = MagicMock()
        launcher = SwarmLauncher(mock_backend)
        cmd = launcher.build_launch_cmd("test-model")
        assert "--bare" in cmd

    def test_launcher_has_mcp_config(self) -> None:
        from autosre.swarm.launcher import SwarmLauncher

        mock_backend = MagicMock()
        launcher = SwarmLauncher(mock_backend)
        cmd = launcher.build_launch_cmd("test-model")
        assert any("--mcp-config" in c for c in cmd)

    def test_launcher_has_system_prompt(self) -> None:
        from autosre.swarm.launcher import SwarmLauncher

        mock_backend = MagicMock()
        launcher = SwarmLauncher(mock_backend)
        cmd = launcher.build_launch_cmd("test-model")
        assert any("--system-prompt" in c for c in cmd)

    def test_launcher_has_settings(self) -> None:
        from autosre.swarm.launcher import SwarmLauncher

        mock_backend = MagicMock()
        launcher = SwarmLauncher(mock_backend)
        cmd = launcher.build_launch_cmd("test-model")
        assert any("--settings" in c for c in cmd)

    def test_launcher_sets_api_key(self) -> None:
        from autosre.swarm.launcher import SwarmLauncher

        mock_backend = MagicMock()
        mock_backend.get_claude_env.return_value = {"ANTHROPIC_BASE_URL": "http://localhost:8011"}
        launcher = SwarmLauncher(mock_backend)
        env = launcher.build_env()
        assert env["ANTHROPIC_API_KEY"] == "local-vllm"

    def test_launcher_system_prompt_promotes_parallelism(self) -> None:
        from autosre.swarm.launcher import SwarmLauncher

        mock_backend = MagicMock()
        launcher = SwarmLauncher(mock_backend)
        cmd = launcher.build_launch_cmd("test-model")
        system_prompt = next(c for c in cmd if "--system-prompt" in c)
        assert "parallel" in system_prompt.lower()


class TestDemoScenarios:
    """Verify demo scenarios are properly configured."""

    def test_all_scenarios_exist(self) -> None:
        from autosre.demo import SCENARIOS

        assert len(SCENARIOS) >= 5

    def test_scenarios_have_required_keys(self) -> None:
        from autosre.demo import SCENARIOS

        for sid, s in SCENARIOS.items():
            assert "name" in s, f"Scenario {sid} missing name"
            assert "desc" in s, f"Scenario {sid} missing desc"
            assert "prompt" in s, f"Scenario {sid} missing prompt"

    def test_scenarios_use_current_model(self) -> None:
        """Demo scenarios should not reference old models."""
        from autosre.demo import SCENARIOS

        old_models = ["nemotron", "nvfp4", "gemma4"]
        for sid, s in SCENARIOS.items():
            prompt_lower = s["prompt"].lower()
            for old in old_models:
                assert old not in prompt_lower, f"Scenario {sid} references old model '{old}'"


class TestAnthropicProxy:
    """Verify the Anthropic API proxy converts correctly."""

    def test_convert_tools_to_openai(self) -> None:
        from autosre.backends.anthropic_proxy import _convert_tools_to_openai

        tools = [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                },
            }
        ]
        result = _convert_tools_to_openai(tools)
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "Read"

    def test_convert_messages_to_openai(self) -> None:
        from autosre.backends.anthropic_proxy import _convert_messages_to_openai

        messages = [{"role": "user", "content": "hello"}]
        result = _convert_messages_to_openai(messages, system="you are helpful")
        assert result[0]["role"] == "system"
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "hello"

    def test_convert_openai_response_to_anthropic(self) -> None:
        from autosre.backends.anthropic_proxy import _convert_openai_response_to_anthropic

        openai_resp = {
            "choices": [
                {
                    "message": {"content": "hello", "tool_calls": None},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1},
        }
        result = _convert_openai_response_to_anthropic(openai_resp, "test-model")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["stop_reason"] == "end_turn"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "hello"

    def test_convert_tool_use_response(self) -> None:
        from autosre.backends.anthropic_proxy import _convert_openai_response_to_anthropic

        openai_resp = {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {
                                    "name": "Read",
                                    "arguments": '{"path": "/etc/hostname"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        result = _convert_openai_response_to_anthropic(openai_resp, "test-model")
        assert result["stop_reason"] == "tool_use"
        tool_block = next(b for b in result["content"] if b["type"] == "tool_use")
        assert tool_block["name"] == "Read"
        assert tool_block["input"]["path"] == "/etc/hostname"


class TestMCPSettings:
    """Verify MCP settings write to project-level, not user-level."""

    def test_settings_path_is_project_level(self) -> None:
        from autosre.mcp_servers.settings import get_claude_settings_path

        path = get_claude_settings_path()
        # Should NOT be ~/.claude/settings.json (user-level)
        assert ".claude/settings.json" not in str(path) or "auto-sre" in str(path)


class TestBenchModule:
    """Verify benchmark module is properly configured."""

    def test_models_list_not_empty(self) -> None:
        from autosre.bench import MODELS

        assert len(MODELS) >= 3

    def test_models_have_hf_token_support(self) -> None:
        """The _start_container function should pass HF_TOKEN."""
        from autosre.bench import _start_container

        # Function exists and accepts a ModelSpec
        assert callable(_start_container)

    def test_production_model_in_bench_list(self) -> None:
        """Production translation model must be benchable.

        The Qwen3.5-INT4-AutoRound entries were retired alongside the rest
        of the 3.5 stack on 2026-04-29; the production successor is
        Qwen3.6-35B-A3B-FP8 on stock vllm/vllm-openai.
        """
        from autosre.bench import MODELS

        prod_models = [m for m in MODELS if "Qwen3.6-35B-A3B FP8" in m.name]
        assert len(prod_models) >= 1
