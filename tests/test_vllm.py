"""Tests for autosre.backends.vllm module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autosre.backends.vllm import CONTAINER_PREFIX, VllmBackend
from autosre.backends.vllm_config import VllmConfig
from autosre.infra.types import GB10Node, NodeRole


@pytest.fixture
def two_node_config(tmp_path: Path) -> VllmConfig:
    config = VllmConfig(
        nodes=[
            GB10Node(hostname="gb10-1", ip="192.168.1.101", role=NodeRole.HEAD),
            GB10Node(hostname="gb10-2", ip="192.168.1.102", role=NodeRole.WORKER),
        ],
    )
    config.save(tmp_path / "vllm.yaml")
    return config


@pytest.fixture
def single_node_config(tmp_path: Path) -> VllmConfig:
    config = VllmConfig(
        nodes=[
            GB10Node(hostname="gb10-1", ip="192.168.1.101", role=NodeRole.HEAD),
        ],
    )
    config.save(tmp_path / "vllm.yaml")
    return config


@pytest.fixture
def backend_with_config(tmp_path: Path, two_node_config: VllmConfig) -> VllmBackend:
    _ = two_node_config  # used for side effect: saves config to tmp_path
    active = {
        "backend": "vllm",
        "config_path": str(tmp_path / "vllm.yaml"),
        "api_host": "192.168.1.101",
    }
    return VllmBackend(active_state=active)


@pytest.fixture
def solo_backend(tmp_path: Path, single_node_config: VllmConfig) -> VllmBackend:
    _ = single_node_config  # used for side effect: saves config to tmp_path
    active = {
        "backend": "vllm",
        "config_path": str(tmp_path / "vllm.yaml"),
        "api_host": "192.168.1.101",
    }
    return VllmBackend(active_state=active)


class TestVllmBackendBasics:
    def test_name(self) -> None:
        assert VllmBackend.name == "vllm"

    def test_description(self) -> None:
        assert "TurboQuant" in VllmBackend.description

    def test_api_port(self) -> None:
        assert VllmBackend.api_port == 8010

    def test_default_model(self) -> None:
        # Single canonical recipe since the customer-install simplification
        # 2026-04-30; the nightly + qwen3-coder siblings were retired.
        # ``qwen3.6-fp8`` pins ``vllm/vllm-openai:latest``, which is the
        # image meeting-scribe's bootstrap actually pulls — defaulting
        # to the nightly recipe broke ``autosre start`` on a fresh GB10.
        assert VllmBackend.default_model == "qwen3.6-fp8"

    def test_models_defined(self) -> None:
        # Single-recipe registry: ``qwen3.6-fp8`` is the only key.
        assert VllmBackend.models == {
            "qwen3.6-fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
        }

    def test_model_ids_are_hf_paths(self) -> None:
        for model_id in VllmBackend.models.values():
            assert "/" in model_id, f"Model ID should be HF path: {model_id}"


class TestVllmApiUrl:
    def test_api_url_from_active_state(self, backend_with_config: VllmBackend) -> None:
        assert backend_with_config.get_api_url() == "http://192.168.1.101:8010"

    def test_api_url_from_config(self, tmp_path: Path, two_node_config: VllmConfig) -> None:
        _ = two_node_config  # side effect: saves config
        with patch.object(VllmConfig, "default_path", return_value=tmp_path / "vllm.yaml"):
            b = VllmBackend()
            assert b.get_api_url() == "http://192.168.1.101:8010"

    def test_api_url_fallback_no_config(self) -> None:
        b = VllmBackend()
        # No config file exists, should fall back to localhost
        assert b.get_api_url() == "http://localhost:8010"


class TestVllmClaudeEnv:
    def test_env_keys(self, backend_with_config: VllmBackend) -> None:
        env = backend_with_config.get_claude_env()
        assert "ANTHROPIC_BASE_URL" in env
        assert "ANTHROPIC_AUTH_TOKEN" in env
        assert "CLAUDE_CODE_ATTRIBUTION_HEADER" in env

    def test_base_url_uses_proxy(self, backend_with_config: VllmBackend) -> None:
        env = backend_with_config.get_claude_env()
        # Claude env points at the proxy, not vLLM directly
        assert "localhost" in env["ANTHROPIC_BASE_URL"]
        assert str(VllmBackend.proxy_port) in env["ANTHROPIC_BASE_URL"]

    def test_auth_token(self, backend_with_config: VllmBackend) -> None:
        env = backend_with_config.get_claude_env()
        assert env["ANTHROPIC_AUTH_TOKEN"] == "vllm"


class TestVllmRecipeLoading:
    def test_build_vllm_serve_cmd_solo(self, solo_backend: VllmBackend) -> None:
        recipe = {
            "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
            "tensor_parallel": 1,
            "max_model_len": 262144,
            "gpu_memory_utilization": 0.90,
            "kv_cache_dtype": "turboquant",
            "max_num_seqs": 10,
            "moe_backend": "cutlass",
            "attention_backend": "TRITON_ATTN",
            "extra_args": ["--enable-prefix-caching"],
        }
        cmd = solo_backend._build_vllm_serve_cmd(recipe, no_turboquant=False)

        assert cmd[0] == "vllm"
        assert cmd[1] == "serve"
        assert "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4" in cmd
        assert "--tensor-parallel-size=1" in cmd
        assert "--max-model-len=262144" in cmd
        assert "--kv-cache-dtype=turboquant" in cmd
        assert "--moe-backend=cutlass" in cmd
        assert "--enable-prefix-caching" in cmd

    def test_build_vllm_serve_cmd_no_turboquant(self, solo_backend: VllmBackend) -> None:
        recipe = {
            "model_id": "test/model",
            "kv_cache_dtype": "turboquant",
        }
        cmd = solo_backend._build_vllm_serve_cmd(recipe, no_turboquant=True)
        assert "--kv-cache-dtype=fp8" in cmd

    def test_build_vllm_serve_cmd_cluster(self, backend_with_config: VllmBackend) -> None:
        recipe = {
            "model_id": "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
            "tensor_parallel": 2,
            "max_model_len": 262144,
            "gpu_memory_utilization": 0.70,
            "kv_cache_dtype": "turboquant",
            "extra_args": ["--distributed-executor-backend=ray"],
        }
        cmd = backend_with_config._build_vllm_serve_cmd(recipe, no_turboquant=False)

        assert "--tensor-parallel-size=2" in cmd
        assert "--distributed-executor-backend=ray" in cmd


class TestVllmRequirements:
    def test_check_requirements_no_config(self) -> None:
        b = VllmBackend()
        ok, missing = b.check_requirements()
        assert ok is False
        assert any("config not found" in m for m in missing)

    @patch("autosre.backends.vllm.SSHRunner")
    def test_check_requirements_nodes_unreachable(
        self,
        mock_ssh_cls: MagicMock,
        backend_with_config: VllmBackend,
    ) -> None:
        mock_runner = MagicMock()
        mock_runner.is_reachable.return_value = False
        mock_ssh_cls.return_value = mock_runner

        ok, missing = backend_with_config.check_requirements()
        assert ok is False
        assert any("unreachable" in m.lower() for m in missing)

    @patch("autosre.backends.vllm.SSHRunner")
    def test_check_requirements_all_ok(
        self,
        mock_ssh_cls: MagicMock,
        backend_with_config: VllmBackend,
    ) -> None:
        mock_runner = MagicMock()
        mock_runner.is_reachable.return_value = True
        mock_result = MagicMock()
        mock_result.stdout = "28.5.1"
        mock_runner.run.return_value = mock_result
        mock_ssh_cls.return_value = mock_runner

        ok, missing = backend_with_config.check_requirements()
        assert ok is True
        assert missing == []


class TestVllmStartSolo:
    @patch("autosre.backends.vllm.SSHRunner")
    @patch("autosre.backends.vllm.save_active_state")
    def test_start_solo_model(
        self,
        mock_save: MagicMock,
        mock_ssh_cls: MagicMock,
        solo_backend: VllmBackend,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

        mock_runner = MagicMock()
        mock_runner.docker_run.return_value = "abc123container"
        mock_ssh_cls.return_value = mock_runner

        # Mock _wait_for_vllm to return True
        monkeypatch.setattr(solo_backend, "_wait_for_vllm", lambda *a, **kw: True)

        result = solo_backend.start(model="qwen3.6-fp8")

        assert result["backend"] == "vllm"
        assert result["model"] == "qwen3.6-fp8"
        assert result["api_host"] == "192.168.1.101"
        assert "containers" in result
        mock_runner.docker_run.assert_called_once()

    @patch("autosre.backends.vllm.SSHRunner")
    @patch("autosre.backends.vllm.save_active_state")
    def test_start_solo_passes_memory_cap(
        self,
        mock_save: MagicMock,
        mock_ssh_cls: MagicMock,
        solo_backend: VllmBackend,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Customer GB10 path must enforce the same hard memory cap
        added 2026-05-01 after the local OOM incident — without it,
        the customer host would still be exposed to vLLM-driven global
        OOMs of meeting-scribe."""
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
        monkeypatch.setenv("AUTOSRE_VLLM_MEM_LIMIT", "72g")

        mock_runner = MagicMock()
        mock_runner.docker_run.return_value = "abc123container"
        mock_ssh_cls.return_value = mock_runner
        monkeypatch.setattr(solo_backend, "_wait_for_vllm", lambda *a, **kw: True)

        solo_backend.start(model="qwen3.6-fp8")

        _, kwargs = mock_runner.docker_run.call_args
        extra = kwargs.get("extra_args") or []
        assert "--memory" in extra
        assert "--memory-swap" in extra
        assert extra[extra.index("--memory") + 1] == "72g"
        assert extra[extra.index("--memory-swap") + 1] == "72g"

    @patch("autosre.backends.vllm.SSHRunner")
    def test_start_nonexistent_model_fails(
        self,
        mock_ssh_cls: MagicMock,
        solo_backend: VllmBackend,
    ) -> None:
        with pytest.raises(RuntimeError, match="No recipe"):
            solo_backend.start(model="nonexistent-cluster-model")


class TestVllmStartCluster:
    """Cluster mode tests — currently no cluster recipes active on GB10."""

    def test_cluster_model_not_available(self, backend_with_config: VllmBackend) -> None:
        """No cluster recipes exist in the current config."""
        with pytest.raises(RuntimeError, match="No recipe"):
            backend_with_config.start(model="nonexistent-cluster-model")


class TestVllmStop:
    @patch("subprocess.run")
    @patch("autosre.backends.vllm.SSHRunner")
    @patch("autosre.backends.vllm.clear_active_state")
    def test_stop_with_saved_containers(
        self,
        mock_clear: MagicMock,
        mock_ssh_cls: MagicMock,
        mock_subprocess: MagicMock,
        backend_with_config: VllmBackend,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

        # Save some container state
        backend_with_config._active_state["containers"] = {"head": "abc123"}

        mock_runner = MagicMock()
        mock_runner.docker_stop.return_value = True
        mock_ssh_cls.return_value = mock_runner
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="", stderr=b"")

        # Default stop keeps model running (no container removal, no state clear)
        result = backend_with_config.stop()
        assert result is True
        mock_clear.assert_not_called()
        # Verify NO docker rm -f was called (critical: tests must never kill live containers)
        for call in mock_subprocess.call_args_list:
            cmd = call[0][0] if call[0] else []
            assert not (
                isinstance(cmd, list) and "docker" in cmd and ("rm" in cmd or "stop" in cmd)
            ), f"stop() without unload_model called docker: {cmd}"

        mock_subprocess.reset_mock()

        # With unload_model=True, containers are stopped and state is cleared
        result = backend_with_config.stop(unload_model=True)
        assert result is True
        mock_clear.assert_called_once()


class TestVllmHealth:
    @patch("autosre.backends.vllm.httpx.Client")
    def test_is_healthy_true(
        self,
        mock_client_cls: MagicMock,
        backend_with_config: VllmBackend,
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        assert backend_with_config.is_healthy() is True

    @patch("autosre.backends.vllm.httpx.Client")
    def test_is_healthy_false(
        self,
        mock_client_cls: MagicMock,
        backend_with_config: VllmBackend,
    ) -> None:
        mock_client_cls.side_effect = Exception("connection refused")
        assert backend_with_config.is_healthy() is False


class TestContainerPrefix:
    def test_container_prefix(self) -> None:
        assert CONTAINER_PREFIX == "autosre-vllm"


class TestRecipeParityCheck:
    """Test the boot-time recipe-parity helper (the prevention layer
    that surfaces drift between the recipe yaml and the live `vllm
    serve ...` cmdline). Each test crafts a fake `ps` output via
    monkeypatch to stay hermetic — no real vllm process required."""

    @staticmethod
    def _patch_ps_output(monkeypatch, output: str) -> None:
        import subprocess as _subprocess
        from unittest.mock import MagicMock

        def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
            res = MagicMock()
            res.stdout = output
            res.returncode = 0
            return res

        # The parity helper does a lazy `import subprocess` inside the
        # function body, so patch the module-level subprocess.run.
        monkeypatch.setattr(_subprocess, "run", fake_run)

    def test_no_mismatch_when_aligned(self, monkeypatch) -> None:
        recipe = {
            "model_id": "Qwen/Qwen3.6-35B-A3B-FP8",
            "max_model_len": 262144,
            "gpu_memory_utilization": 0.70,
            "max_num_seqs": 8,
            "max_num_batched_tokens": 4096,
            "quantization": "fp8",
            "attention_backend": "flashinfer",
            "extra_args": ["--enable-prefix-caching", "--enable-chunked-prefill"],
        }
        ps_output = (
            "/usr/bin/python3 /usr/local/bin/vllm serve Qwen/Qwen3.6-35B-A3B-FP8 "
            "--max-model-len=262144 --gpu-memory-utilization=0.70 --max-num-seqs=8 "
            "--max-num-batched-tokens=4096 --quantization=fp8 "
            "--attention-backend=flashinfer --port=8010 "
            "--enable-prefix-caching --enable-chunked-prefill\n"
        )
        self._patch_ps_output(monkeypatch, ps_output)

        mismatches = VllmBackend.assert_running_vllm_matches_recipe(recipe, api_port=8010)
        assert mismatches == [], f"unexpected mismatches: {mismatches}"

    def test_detects_missing_attention_backend(self, monkeypatch) -> None:
        """The 2026-04-30 acceptance test: the cmdline missing
        `--attention-backend=flashinfer` (the customer-GB10 case) MUST
        be caught."""
        recipe = {
            "model_id": "Qwen/Qwen3.6-35B-A3B-FP8",
            "attention_backend": "flashinfer",
            "extra_args": [],
        }
        ps_output = (
            "/usr/bin/python3 /usr/local/bin/vllm serve Qwen/Qwen3.6-35B-A3B-FP8 "
            "--port=8010\n"
        )
        self._patch_ps_output(monkeypatch, ps_output)

        mismatches = VllmBackend.assert_running_vllm_matches_recipe(recipe, api_port=8010)
        assert any(
            "attention-backend" in m and "flashinfer" in m for m in mismatches
        ), mismatches

    def test_detects_value_drift(self, monkeypatch) -> None:
        recipe = {
            "model_id": "Qwen/Qwen3.6-35B-A3B-FP8",
            "max_model_len": 262144,
            "extra_args": [],
        }
        ps_output = (
            "/usr/bin/python3 /usr/local/bin/vllm serve Qwen/Qwen3.6-35B-A3B-FP8 "
            "--max-model-len=131072 --port=8010\n"
        )
        self._patch_ps_output(monkeypatch, ps_output)

        mismatches = VllmBackend.assert_running_vllm_matches_recipe(recipe, api_port=8010)
        assert any(
            "max-model-len" in m and "131072" in m and "262144" in m
            for m in mismatches
        ), mismatches

    def test_detects_missing_extra_arg(self, monkeypatch) -> None:
        recipe = {
            "model_id": "Qwen/Qwen3.6-35B-A3B-FP8",
            "extra_args": ["--enable-prefix-caching", "--enable-chunked-prefill"],
        }
        ps_output = (
            "/usr/bin/python3 /usr/local/bin/vllm serve Qwen/Qwen3.6-35B-A3B-FP8 "
            "--port=8010 --enable-prefix-caching\n"
        )
        self._patch_ps_output(monkeypatch, ps_output)

        mismatches = VllmBackend.assert_running_vllm_matches_recipe(recipe, api_port=8010)
        assert any(
            "missing live extra_arg" in m and "enable-chunked-prefill" in m
            for m in mismatches
        ), mismatches

    def test_returns_empty_when_no_live_process(self, monkeypatch) -> None:
        """If no `vllm serve ...` process is running, return [] —
        not an error. Caller (boot-time hook) decides what to do."""
        recipe = {"model_id": "Qwen/Qwen3.6-35B-A3B-FP8", "extra_args": []}
        self._patch_ps_output(monkeypatch, "/usr/bin/some-other-process\n")

        mismatches = VllmBackend.assert_running_vllm_matches_recipe(recipe, api_port=8010)
        assert mismatches == []

    def test_warn_helper_emits_warning_per_mismatch(self, monkeypatch, caplog) -> None:
        import logging as _logging

        recipe = {
            "model_id": "Qwen/Qwen3.6-35B-A3B-FP8",
            "attention_backend": "flashinfer",
            "extra_args": [],
        }
        ps_output = (
            "/usr/bin/python3 /usr/local/bin/vllm serve Qwen/Qwen3.6-35B-A3B-FP8 "
            "--port=8010\n"
        )
        self._patch_ps_output(monkeypatch, ps_output)

        import autosre.backends.vllm as vllm_module
        with caplog.at_level(_logging.WARNING, logger=vllm_module.logger.name):
            VllmBackend.warn_on_recipe_drift(recipe, api_port=8010)

        warns = [r for r in caplog.records if "autosre recipe drift" in r.message]
        assert len(warns) >= 1, [r.message for r in caplog.records]


class TestBuildRuntimeEnv:
    """`_build_runtime_env` is the single source of truth for the env
    vars passed to the vLLM container. Both the local dev path
    (`_start_local`, container `autosre-vllm-local`) and the customer
    SSH path (`_start_solo`, container `autosre-vllm-head`) funnel
    through it, so any drift gap closes here.

    These tests pin the contract that the 2026-04-30 audit found
    broken — the customer-facing path was previously missing
    HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE/HF_HUB_CACHE, which would have
    re-exposed the same bug after a wipe-and-reinstall."""

    def test_default_env_includes_offline_and_cache(self) -> None:
        env = VllmBackend._build_runtime_env({})
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["HF_HUB_CACHE"] == "/data/huggingface/hub"
        # NVIDIA_DISABLE_REQUIRE must propagate to BOTH start paths.
        # Pre-2026-04-30 it was only set on the local dev path; the
        # customer-facing _start_solo path silently dropped it,
        # leaving customer hosts vulnerable to driver-envelope refusals.
        assert env["NVIDIA_DISABLE_REQUIRE"] == "1"

    def test_recipe_env_overrides_defaults(self) -> None:
        env = VllmBackend._build_runtime_env(
            {"env": {"HF_HUB_OFFLINE": "0", "VLLM_FOO": "bar"}}
        )
        assert env["HF_HUB_OFFLINE"] == "0"  # overridden by recipe
        assert env["VLLM_FOO"] == "bar"  # additive
        # Other defaults still present.
        assert env["HF_HUB_CACHE"] == "/data/huggingface/hub"

    def test_hf_token_picked_up_from_environ(self, monkeypatch) -> None:
        monkeypatch.setenv("HF_TOKEN", "hf_test_token_12345")
        env = VllmBackend._build_runtime_env({})
        assert env["HF_TOKEN"] == "hf_test_token_12345"

    def test_hf_token_absent_when_not_in_environ(self, monkeypatch) -> None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
        env = VllmBackend._build_runtime_env({})
        assert "HF_TOKEN" not in env

    def test_recipe_cublas_hardening_passes_through(self) -> None:
        """The 2026-04-30 acceptance test: customer needs the cuBLAS
        hardening env vars (production-required per CLAUDE.md). After
        the recipe is updated, the env-builder must propagate them."""
        recipe_env = {
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "VLLM_MARLIN_USE_ATOMIC_ADD": "1",
        }
        env = VllmBackend._build_runtime_env({"env": recipe_env})
        for key, val in recipe_env.items():
            assert env[key] == val, f"missing/wrong {key}: {env.get(key)!r}"
