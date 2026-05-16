"""vLLM backend for GB10 (NVIDIA Grace Blackwell) nodes.

Manages vLLM Docker containers on remote GB10 nodes via SSH.
Supports single-node (solo) and multi-node (cluster, TP=2) modes.
TurboQuant KV cache compression enabled by default.

Primary Docker image: bjk110/spark-vllm (feat/turboquant branch)
  - vLLM 0.19.1 with SM121 support
  - CUDA WPH decode AOT-compiled for SM121
  - TurboQuant KV cache compression

Fallback: eugr/spark-vllm-docker (no TurboQuant)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, ClassVar, cast

import click
import httpx

from autosre.infra.ssh import SSHRunner

from .base import Backend, clear_active_state, save_active_state
from .recipes import get_recipe_for_model
from .vllm_config import VllmConfig

logger = logging.getLogger(__name__)

# Container name prefix for autosre-managed containers
CONTAINER_PREFIX = "autosre-vllm"


class VllmBackend(Backend):
    """vLLM on GB10 via Docker over SSH.

    Manages Docker containers on remote GB10 nodes. Solo models run on the
    head node only. Cluster models use Ray for tensor parallelism across
    head + worker nodes via NCCL over ConnectX-7.
    """

    name: ClassVar[str] = "vllm"
    description: ClassVar[str] = "vLLM on GB10 (Docker, NVFP4, TurboQuant)"
    api_port: ClassVar[int] = 8010
    proxy_port: ClassVar[int] = 8011  # Anthropic API proxy port

    # Single canonical recipe. Customers get the stable image
    # (``vllm/vllm-openai:latest``) which is what meeting-scribe's
    # bootstrap pulls; power users who want bleeding-edge can override
    # with ``--docker-image vllm/vllm-openai:nightly`` at start time.
    models: ClassVar[dict[str, str]] = {
        "qwen3.6-fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
    }
    default_model: ClassVar[str] = "qwen3.6-fp8"

    # Docker images: recipe can override via "docker_image" key. Stock
    # vllm/vllm-openai is the only image a customer GB10 should expect
    # to have — the prior custom albond fork (vllm-qwen35-v2) was
    # retired alongside the Qwen3.5 stack.
    default_docker_image: ClassVar[str] = "vllm/vllm-openai:latest"
    fallback_docker_image: ClassVar[str] = "vllm/vllm-openai:nightly"

    def __init__(self, active_state: dict[str, object] | None = None) -> None:
        super().__init__(active_state)
        self._config: VllmConfig | None = None
        # Reconstruct config from stored path if available
        if active_state and active_state.get("config_path"):
            config_path = Path(str(active_state["config_path"])).expanduser()
            if config_path.exists():
                self._config = VllmConfig.load(config_path)

    @property
    def config(self) -> VllmConfig:
        """Lazily load config from default path."""
        if self._config is None:
            self._config = VllmConfig.load()
        return self._config

    def get_api_url(self) -> str:
        """Return URL of the vLLM API on the head node."""
        port = (
            int(cast("int", self._active_state.get("api_port", self.api_port)))
            if self._active_state
            else self.api_port
        )
        if self._active_state and self._active_state.get("api_host"):
            return f"http://{self._active_state['api_host']}:{port}"
        try:
            return f"http://{self.config.head_node.ip}:{port}"
        except (FileNotFoundError, ValueError):
            return f"http://localhost:{port}"

    def check_requirements(self) -> tuple[bool, list[str]]:
        """Check SSH connectivity to configured GB10 nodes."""
        missing: list[str] = []

        try:
            config = self.config
        except FileNotFoundError:
            return False, ["vLLM config not found. Run 'autosre configure vllm --node <ip>' first."]

        for node in config.nodes:
            runner = SSHRunner(node)
            if not runner.is_reachable():
                missing.append(f"SSH unreachable: {node.ssh_target}")

        if not missing:
            # Check Docker and nvidia-smi on head node
            head_runner = SSHRunner(config.head_node)
            try:
                result = head_runner.run(["docker", "info", "--format", "{{.ServerVersion}}"])
                if not result.stdout.strip():
                    missing.append(f"Docker not responding on {config.head_node.ip}")
            except Exception:
                missing.append(f"Docker check failed on {config.head_node.ip}")

            try:
                result = head_runner.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
                )
                if not result.stdout.strip():
                    missing.append(f"nvidia-smi failed on {config.head_node.ip}")
            except Exception:
                missing.append(f"GPU check failed on {config.head_node.ip}")

        return len(missing) == 0, missing

    def setup(self, force: bool = False, **_kwargs: object) -> bool:  # noqa: ARG002
        """Validate config and check Docker image availability."""
        try:
            config = self.config
        except FileNotFoundError as e:
            click.secho(f"ERROR: {e}", fg="red")
            return False

        click.echo(f"Nodes: {len(config.nodes)} ({config.head_node.ip} = head)")
        click.echo(f"Docker image: {config.docker_image}")
        click.echo(f"HF cache: {config.hf_cache_dir}")

        # Check Docker image exists on head node
        head_runner = SSHRunner(config.head_node)
        result = head_runner.run(
            ["docker", "images", "-q", config.docker_image],
            check=False,
        )
        if result.stdout.strip():
            click.secho("Docker image found on head node.", fg="green")
        else:
            click.secho(
                f"Docker image '{config.docker_image}' not found on head node.",
                fg="yellow",
            )
            click.echo("Run 'autosre provision setup' to build it.")

        return True

    def start(
        self,
        model: str | None = None,
        foreground: bool = False,  # noqa: ARG002
        **kwargs: object,
    ) -> dict[str, object]:
        """Launch vLLM container on the local machine or remote GB10 node(s).

        Local mode (default): runs Docker directly on this machine.
        Remote mode: requires vllm.yaml config with SSH nodes.

        Solo models: single container on head node.
        Cluster models: Ray head + worker, then vLLM serve with TP=2.
        TurboQuant enabled by default via --kv-cache-dtype turboquant.

        Args:
            model: Model key (e.g., "qwen3.6-fp8"). None = default.
            foreground: Not supported for remote Docker (ignored).

        Returns:
            Dict with model, api_port, api_host, containers.

        Raises:
            RuntimeError: If launch fails.
        """
        model_key = model or self.default_model
        no_turboquant = bool(kwargs.get("no_turboquant", False))

        # Load recipe
        try:
            recipe = get_recipe_for_model(model_key)
        except FileNotFoundError as e:
            msg = f"No recipe for model '{model_key}': {e}"
            raise RuntimeError(msg) from e

        # Try local mode first (no SSH config needed)
        try:
            self.config  # noqa: B018 — check if remote config exists
        except (FileNotFoundError, ValueError):
            return self._start_local(model_key, recipe, no_turboquant)

        config = self.config
        model_id = self.get_model_id(model_key)
        is_cluster = recipe.get("mode") == "cluster"

        # Validate cluster models have enough nodes
        if is_cluster and not config.is_cluster:
            msg = (
                f"Model '{model_key}' requires cluster mode (TP={recipe.get('tensor_parallel', 2)}) "
                f"but only {len(config.nodes)} node(s) configured. "
                "Add more nodes with 'autosre configure vllm --node <ip> --node <ip>'."
            )
            raise RuntimeError(msg)

        # Stop any existing containers
        self._stop_containers()

        click.echo(f"Launching {recipe.get('short_name', model_id)}...")
        click.echo(f"  Mode: {'cluster (TP=2)' if is_cluster else 'solo'}")
        click.echo(f"  TurboQuant: {'disabled' if no_turboquant else 'enabled'}")

        containers: dict[str, str] = {}

        if is_cluster:
            containers = self._start_cluster(recipe, config, no_turboquant)
        else:
            containers = self._start_solo(recipe, config, no_turboquant)

        # Wait for vLLM to be ready
        api_url = f"http://{config.head_node.ip}:{self.api_port}"
        click.echo(f"Waiting for vLLM at {api_url}/health ...")

        if not self._wait_for_vllm(config.head_node.ip, self.api_port, timeout=600):
            # Cleanup on failure
            self._stop_containers(containers)
            msg = "vLLM failed to start within 5 minutes."
            raise RuntimeError(msg)

        click.secho("vLLM is ready!", fg="green")

        # Boot-time recipe-parity sentinel — emit a WARNING per drift
        # point so a host running stale flags (older recipe state,
        # manual override) surfaces in logs immediately. The
        # 2026-04-30 customer-GB10 audit found multiple drifts that
        # would have been silent without this check; the helper is
        # deliberately non-fatal so a recipe edit that lags the
        # running process doesn't refuse the start.
        VllmBackend.warn_on_recipe_drift(recipe, api_port=self.api_port)

        # Save state
        state: dict[str, object] = {
            "backend": self.name,
            "model": model_key,
            "api_port": self.api_port,
            "api_host": config.head_node.ip,
            "proxy_port": self.proxy_port,
            "config_path": str(config.default_path()),
            "containers": containers,
        }
        save_active_state(state)
        self._save_containers(containers)

        return state

    def _start_local(
        self,
        model_key: str,
        recipe: dict[str, Any],
        no_turboquant: bool,
    ) -> dict[str, object]:
        """Start vLLM container locally via Docker (no SSH)."""
        import shutil
        import subprocess

        if not shutil.which("docker"):
            msg = "Docker not found. Install Docker to run vLLM locally."
            raise RuntimeError(msg)

        model_id = recipe["model_id"]
        port = int(recipe.get("api_port", 8010))
        docker_image = str(recipe.get("docker_image", self.default_docker_image))

        # Verify Docker image exists
        check = subprocess.run(
            ["docker", "image", "inspect", docker_image],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode != 0:
            msg = (
                f"Docker image '{docker_image}' not found.\n"
                f"Build it first — see repos/albond-spark-int4/README.md"
            )
            raise RuntimeError(msg)

        # Verify model weights are cached
        hf_cache = "/data/huggingface"
        user_hf_cache = str(Path.home() / ".cache" / "huggingface")
        model_slug = model_id.replace("/", "--")
        model_cached = (Path(hf_cache) / "hub" / f"models--{model_slug}").exists() or (
            Path(user_hf_cache) / "hub" / f"models--{model_slug}"
        ).exists()
        if not model_cached:
            msg = (
                f"Model '{model_id}' not found in HuggingFace cache.\n"
                f"Download first: hf download {model_id}"
            )
            raise RuntimeError(msg)

        container_name = f"{CONTAINER_PREFIX}-local"

        # Only remove if container exists but is dead — never kill a healthy one
        check = subprocess.run(
            ["docker", "inspect", "--format={{.State.Running}}", container_name],
            capture_output=True,
            text=True,
            check=False,
        )
        if check.returncode == 0 and "true" in check.stdout:
            # Container is running — don't touch it
            click.secho(f"  Container {container_name} already running, skipping", fg="yellow")
            # Return existing state
            from .base import load_active_state as _load_active

            active = _load_active()
            if active:
                return dict(active)
            return {
                "backend": self.name,
                "model": model_key,
                "api_port": port,
                "api_host": "localhost",
            }

        # Capture logs from dead container before removing
        if check.returncode == 0:
            logs = subprocess.run(
                ["docker", "logs", "--tail", "50", container_name],
                capture_output=True,
                text=True,
                check=False,
            )
            exit_info = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format={{.State.ExitCode}} {{.State.FinishedAt}} {{.State.Error}}",
                    container_name,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if exit_info.stdout.strip():
                click.secho(f"  Previous container died: {exit_info.stdout.strip()}", fg="yellow")
            if logs.stderr:
                crash_log = Path.home() / ".local" / "share" / "autosre" / "vllm-crash.log"
                crash_log.write_text(f"Exit: {exit_info.stdout}\n{logs.stderr[-2000:]}\n")
                click.secho(f"  Crash log saved: {crash_log}", fg="yellow")

        # Remove dead/stopped container
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            check=False,
        )

        vllm_cmd = self._build_vllm_serve_cmd(recipe, no_turboquant)
        # Override port in the command
        vllm_cmd = [arg for arg in vllm_cmd if not arg.startswith("--port=")]
        vllm_cmd.append(f"--port={port}")

        # Upstream vllm/vllm-openai has ENTRYPOINT=["vllm","serve"]; strip
        # both leading tokens so the model id lands as the first positional
        # and the --flags follow. Any image tag in the vllm/vllm-openai
        # family ships with this entrypoint shape.
        if docker_image.startswith(("vllm/vllm-openai", "vllm-openai")):
            while vllm_cmd and vllm_cmd[0] in ("vllm", "serve"):
                vllm_cmd = vllm_cmd[1:]

        # Hard memory cap. Added 2026-05-01 after the 2026-04-30 OOM
        # incident: vLLM's mmap'd weights + KV cache live in host RAM
        # on GB10's unified-memory architecture, and an unconstrained
        # container can push the host into a global OOM that kills the
        # operator's foreground workload (e.g. meeting-scribe) instead
        # of failing locally. Setting --memory == --memory-swap means
        # the container is denied swap entirely, so a runaway
        # allocation hits a per-cgroup OOM (vLLM raises
        # OutOfMemoryError) instead of dragging the kernel into a
        # global swap-thrash spiral. Default 80g matches gpu_memory_utilization=0.70
        # (~77 GB working set) plus a thin runtime margin. Override
        # via AUTOSRE_VLLM_MEM_LIMIT for non-GB10 hosts or larger models.
        mem_limit = os.environ.get("AUTOSRE_VLLM_MEM_LIMIT", "80g")
        docker_cmd = [
            "docker",
            "run",
            "-d",
            "--restart",
            "unless-stopped",
            "--name",
            container_name,
            "--gpus",
            "all",
            "--network",
            "host",
            "--shm-size",
            "16g",
            "--memory",
            mem_limit,
            "--memory-swap",
            mem_limit,
            "--ulimit",
            "memlock=-1",
            "-v",
            f"{hf_cache}:/data/huggingface",
            "-v",
            f"{user_hf_cache}:/root/.cache/huggingface",
        ]

        # Single source of truth for env vars — see
        # `_build_runtime_env` for the rationale. The customer-facing
        # `_start_solo` path uses the same helper, so the two paths
        # produce bit-for-bit identical container env (including
        # NVIDIA_DISABLE_REQUIRE, HF offline mode, HF_HUB_CACHE).
        for key, val in VllmBackend._build_runtime_env(recipe).items():
            docker_cmd.extend(["-e", f"{key}={val}"])

        # Mount chat template if specified
        chat_template = recipe.get("chat_template")
        if chat_template:
            expanded = str(Path(chat_template).expanduser())
            docker_cmd.extend(["-v", f"{expanded}:{expanded}:ro"])

        # vLLM V1 priority-preemption pre-hook.
        #
        # The patch itself lives in ``vllm_priority_preempt.py`` and
        # installs a monkey-patch on ``Scheduler.schedule`` that actually
        # preempts lower-priority running requests for higher-priority
        # waiting ones. Upstream V1 only reorders the waiting queue —
        # see the module docstring for details.
        #
        # Activation mechanism: a ``.pth`` file dropped into the
        # container's site-packages directory. Python's site.py processes
        # every ``*.pth`` in each site-dir at interpreter startup and
        # executes lines beginning with ``import`` — this runs for
        # *every* Python invocation inside the container, including
        # ``vllm serve`` itself (which is a console-script entry point,
        # not interactive). ``PYTHONSTARTUP`` does NOT work here because
        # it only fires in interactive mode.
        preempt_py = Path(__file__).resolve().parent / "vllm_priority_preempt.py"
        preempt_pth = Path(__file__).resolve().parent / "vllm_priority_preempt.pth"
        if preempt_py.exists() and preempt_pth.exists():
            preempt_py_in_container = "/opt/autosre/vllm_priority_preempt.py"
            # Debian/Ubuntu layout in the vllm image. Any ``.pth`` file
            # dropped into this directory gets processed by ``site.py``.
            preempt_pth_in_container = (
                "/usr/local/lib/python3.12/dist-packages/vllm_priority_preempt.pth"
            )
            docker_cmd.extend(
                [
                    "-v",
                    f"{preempt_py}:{preempt_py_in_container}:ro",
                    "-v",
                    f"{preempt_pth}:{preempt_pth_in_container}:ro",
                ]
            )

        docker_cmd.extend([docker_image, *vllm_cmd])

        kv_dtype = recipe.get("kv_cache_dtype", "auto")
        quant = recipe.get("quantization", "none")
        click.echo(f"Launching {recipe.get('short_name', model_id)} locally...")
        click.echo(f"  Port: {port}  |  KV: {kv_dtype}  |  Quant: {quant}")

        result = subprocess.run(docker_cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            msg = f"Docker launch failed: {result.stderr.strip()}"
            raise RuntimeError(msg)

        container_id = result.stdout.strip()[:12]
        click.echo(f"  Container: {container_id}")

        # SM121 fix: disable Marlin MoE kernels (crash with cudaErrorUnknown on Blackwell)
        self._apply_sm121_moe_fix(container_name)
        # SM121 fix: disable CUTLASS block-FP8 linear kernel (Internal error on Blackwell)
        self._apply_sm121_fp8_fix(container_name)

        # Wait for vLLM to be ready
        click.echo(f"Waiting for vLLM at http://localhost:{port}/health ...")
        if not self._wait_for_vllm("localhost", port, timeout=600):
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                check=False,
            )
            msg = "vLLM failed to start within 5 minutes."
            raise RuntimeError(msg)

        click.secho("vLLM is ready!", fg="green")

        # Boot-time recipe-parity sentinel — see the matching call in
        # the SSH-cluster path above for the rationale.
        VllmBackend.warn_on_recipe_drift(recipe, api_port=port)

        state: dict[str, object] = {
            "backend": self.name,
            "model": model_key,
            "api_port": port,
            "api_host": "localhost",
            "proxy_port": self.proxy_port,
            "containers": {"local": result.stdout.strip()},
        }
        save_active_state(state)
        self._save_containers({"local": result.stdout.strip()})

        return state

    @staticmethod
    def _build_runtime_env(recipe: dict[str, Any]) -> dict[str, str]:
        """Single source of truth for environment variables passed to
        the vLLM container. Both `_start_local` and `_start_solo`
        funnel through here so the customer-facing path (`_start_solo`,
        container `autosre-vllm-head`) can NEVER drift from the local
        dev path (`_start_local`, container `autosre-vllm-local`).

        Pre-2026-04-30, only `_start_local` set the offline + cache
        env vars; `_start_solo` only forwarded `recipe.get("env", {})`.
        That meant the customer GB10 was running without
        `HF_HUB_OFFLINE=1` (containers could hang on cold-boot DNS
        races), without `HF_HUB_CACHE` (models had to be at
        `~/.cache/huggingface/hub`, not `/data/huggingface/hub`),
        AND without `NVIDIA_DISABLE_REQUIRE=1` (driver enforcement
        could refuse to start the container on a host whose driver
        version landed outside vLLM's strict NVIDIA_REQUIRE_CUDA
        envelope). Codifying all of this here closes the gaps and
        keeps the two paths bit-for-bit identical."""
        env: dict[str, str] = {
            # Run fully offline so vLLM never races systemd-resolved
            # at cold boot (otherwise: "Temporary failure in name
            # resolution" until DNS comes up, blowing the boot budget).
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            # Direct the HF library at /data/huggingface — that's where
            # `pull-models` places weights on customer + dev hosts.
            # Without this the library silently defaults to
            # /root/.cache/huggingface/hub and a fresh container
            # crash-loops with `LocalEntryNotFoundError` if the user
            # cache doesn't already have the model.
            "HF_HUB_CACHE": "/data/huggingface/hub",
            # NVIDIA driver-version requirement check — the upstream
            # vllm/vllm-openai image bakes a strict NVIDIA_REQUIRE_CUDA
            # envelope into the container that some legitimate driver
            # versions on GB10 fall outside of. Disabling the check
            # avoids "Container exec failed: NVIDIA driver doesn't
            # match" on hosts that are functionally compatible.
            "NVIDIA_DISABLE_REQUIRE": "1",
        }
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env["HF_TOKEN"] = hf_token
        # Recipe env wins over the defaults so an operator can override
        # any of these per-model if needed.
        env.update(recipe.get("env", {}))
        return env

    def _start_solo(
        self,
        recipe: dict[str, Any],
        config: VllmConfig,
        no_turboquant: bool,
    ) -> dict[str, str]:
        """Start a single vLLM container on the head node."""
        head = SSHRunner(config.head_node)
        vllm_cmd = self._build_vllm_serve_cmd(recipe, no_turboquant)

        # Same entrypoint-strip the local path does — upstream
        # vllm/vllm-openai images have ENTRYPOINT=["vllm","serve"], so
        # passing our cmd verbatim doubles ``vllm serve`` and vLLM
        # rejects the model id as "unrecognized arguments".
        docker_image = config.docker_image
        if docker_image.startswith(("vllm/vllm-openai", "vllm-openai")):
            while vllm_cmd and vllm_cmd[0] in ("vllm", "serve"):
                vllm_cmd = vllm_cmd[1:]

        # Canonical mount: ALWAYS bind /data/huggingface to itself so
        # HF_HUB_CACHE=/data/huggingface/hub (set by `_build_runtime_env`)
        # resolves inside the container. The legacy
        # config.hf_cache_dir → /root/.cache/huggingface mount stays
        # for compat with older operators who set hf_cache_dir to a
        # custom location, but we no longer rely on it for model
        # discovery. Pre-2026-04-30 the customer-facing path bound
        # only the legacy mount, so a recipe pointing HF_HUB_CACHE at
        # /data/huggingface/hub would have failed inside the container.
        volumes = ["/data/huggingface:/data/huggingface"]
        if config.hf_cache_dir != "/data/huggingface":
            volumes.append(f"{config.hf_cache_dir}:/root/.cache/huggingface")

        # Same OOM-cap as `_start_local` — see the docstring there for
        # the full rationale. Forwarded via extra_args so the cap
        # follows whatever flags SSHRunner.docker_run already sets.
        mem_limit = os.environ.get("AUTOSRE_VLLM_MEM_LIMIT", "80g")
        mem_args = ["--memory", mem_limit, "--memory-swap", mem_limit]

        container_id = head.docker_run(
            docker_image,
            cmd=vllm_cmd,
            name=f"{CONTAINER_PREFIX}-head",
            volumes=volumes,
            env=VllmBackend._build_runtime_env(recipe),
            extra_args=mem_args,
        )

        click.echo(f"  Head container: {container_id[:12]}")
        return {"head": container_id}

    def _start_cluster(
        self,
        recipe: dict[str, Any],
        config: VllmConfig,
        no_turboquant: bool,
    ) -> dict[str, str]:
        """Start Ray cluster + vLLM across head and worker nodes."""
        containers: dict[str, str] = {}
        head = SSHRunner(config.head_node)
        head_ip = config.head_node.ip
        ray_port = 6379

        # Merge recipe env with NCCL config
        env = {
            **recipe.get("env", {}),
            "NCCL_SOCKET_IFNAME": config.nccl_socket_ifname,
            "NCCL_IB_DISABLE": "0",
            "VLLM_HOST_IP": head_ip,
        }

        # Start Ray head
        ray_head_cmd = [
            "ray",
            "start",
            "--block",
            "--head",
            f"--port={ray_port}",
        ]
        head_container = head.docker_run(
            config.docker_image,
            cmd=ray_head_cmd,
            name=f"{CONTAINER_PREFIX}-ray-head",
            volumes=[f"{config.hf_cache_dir}:/root/.cache/huggingface"],
            env={**env, "VLLM_HOST_IP": head_ip},
        )
        containers["ray_head"] = head_container
        click.echo(f"  Ray head: {head_container[:12]} on {head_ip}")

        # Start Ray workers
        for i, worker_node in enumerate(config.worker_nodes):
            worker = SSHRunner(worker_node)
            ray_worker_cmd = [
                "ray",
                "start",
                "--block",
                f"--address={head_ip}:{ray_port}",
            ]
            worker_container = worker.docker_run(
                config.docker_image,
                cmd=ray_worker_cmd,
                name=f"{CONTAINER_PREFIX}-ray-worker-{i}",
                volumes=[f"{config.hf_cache_dir}:/root/.cache/huggingface"],
                env={**env, "VLLM_HOST_IP": worker_node.ip},
            )
            containers[f"ray_worker_{i}"] = worker_container
            click.echo(f"  Ray worker {i}: {worker_container[:12]} on {worker_node.ip}")

        # Start vLLM serve on head (connects to Ray cluster)
        vllm_cmd = self._build_vllm_serve_cmd(recipe, no_turboquant)
        # Strip the upstream ENTRYPOINT=["vllm","serve"] tokens (see
        # ``_start_solo`` for the rationale).
        docker_image = config.docker_image
        if docker_image.startswith(("vllm/vllm-openai", "vllm-openai")):
            while vllm_cmd and vllm_cmd[0] in ("vllm", "serve"):
                vllm_cmd = vllm_cmd[1:]
        vllm_container = head.docker_run(
            docker_image,
            cmd=vllm_cmd,
            name=f"{CONTAINER_PREFIX}-serve",
            volumes=[f"{config.hf_cache_dir}:/root/.cache/huggingface"],
            env=env,
        )
        containers["vllm_serve"] = vllm_container
        click.echo(f"  vLLM serve: {vllm_container[:12]}")

        return containers

    def _build_vllm_serve_cmd(self, recipe: dict[str, Any], no_turboquant: bool) -> list[str]:
        """Build the vLLM serve command from a recipe."""
        model_id = recipe["model_id"]
        tp = recipe.get("tensor_parallel", 1)
        max_model_len = recipe.get("max_model_len", 131072)
        gpu_mem = recipe.get("gpu_memory_utilization", 0.90)

        kv_dtype = "fp8" if no_turboquant else recipe.get("kv_cache_dtype", "turboquant")

        cmd = [
            "vllm",
            "serve",
            model_id,
            f"--tensor-parallel-size={tp}",
            f"--max-model-len={max_model_len}",
            f"--gpu-memory-utilization={gpu_mem}",
            f"--kv-cache-dtype={kv_dtype}",
            "--host=0.0.0.0",
            f"--port={self.api_port}",
        ]

        max_seqs = recipe.get("max_num_seqs")
        if max_seqs:
            cmd.append(f"--max-num-seqs={max_seqs}")

        max_batched = recipe.get("max_num_batched_tokens")
        if max_batched:
            cmd.append(f"--max-num-batched-tokens={max_batched}")

        quantization = recipe.get("quantization")
        if quantization:
            cmd.append(f"--quantization={quantization}")

        moe_backend = recipe.get("moe_backend")
        if moe_backend:
            cmd.append(f"--moe-backend={moe_backend}")

        attention = recipe.get("attention_backend")
        if attention:
            cmd.append(f"--attention-backend={attention}")

        chat_template = recipe.get("chat_template")
        if chat_template:
            expanded = str(Path(chat_template).expanduser())
            cmd.append(f"--chat-template={expanded}")

        # Extra args from recipe
        cmd.extend(recipe.get("extra_args", []))

        return cmd

    def stop(self, **kwargs: object) -> bool:
        """Stop proxy and optionally the vLLM model container.

        By default, the vLLM container is KEPT RUNNING — it takes minutes
        to load the model and must survive proxy/UI restarts.

        Keyword args:
            unload_model: If True, also stop the vLLM container. Only set
                          this when explicitly requested by the user.
        """
        unload_model = bool(kwargs.get("unload_model", False))
        # Stop proxy (fast, cheap to restart)
        self.stop_proxy()

        if not unload_model:
            click.echo("  vLLM model kept running (use --unload-model to stop)")
            self._clear_pids()
            # Don't clear active state — model is still running
            return True

        containers = self._load_containers()

        # Stop local containers
        local_stopped = self._stop_local_containers(containers)

        # Stop remote containers
        remote_stopped = False
        if not local_stopped:
            remote_stopped = (
                self._stop_containers(containers) if containers else self._stop_containers()
            )

        self._clear_pids()
        clear_active_state()
        return local_stopped or remote_stopped

    def _stop_local_containers(self, containers: dict[str, str] | None = None) -> bool:
        """Stop locally-managed Docker containers."""
        import subprocess

        stopped = False
        # Stop by saved container IDs
        if containers:
            for role, cid in containers.items():
                if role.startswith("local") or role == "local":
                    result = subprocess.run(
                        ["docker", "rm", "-f", cid],
                        capture_output=True,
                        check=False,
                    )
                    if result.returncode == 0:
                        click.echo(f"Stopped local container: {cid[:12]}")
                        stopped = True

        # Also stop by name prefix
        result = subprocess.run(
            ["docker", "rm", "-f", f"{CONTAINER_PREFIX}-local"],
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 and result.stderr and b"No such container" not in result.stderr:
            stopped = True

        return stopped

    def _stop_containers(self, containers: dict[str, str] | None = None) -> bool:
        """Stop containers by ID, or by name prefix if no IDs given."""
        stopped = False
        try:
            config = self.config
        except (FileNotFoundError, ValueError):
            return False

        if containers:
            # Stop specific containers by ID
            for role, cid in containers.items():
                # Determine which node this container is on
                if "worker" in role:
                    # Find the right worker node
                    idx = 0
                    parts = role.split("_")
                    if len(parts) > 2:
                        try:
                            idx = int(parts[-1])
                        except ValueError:
                            idx = 0
                    nodes = config.worker_nodes
                    node = nodes[idx] if idx < len(nodes) else config.head_node
                else:
                    node = config.head_node

                runner = SSHRunner(node)
                if runner.docker_stop(cid):
                    stopped = True
        else:
            # Stop by name prefix on all nodes
            for node in config.nodes:
                runner = SSHRunner(node)
                ps_output = runner.docker_ps(name_filter=CONTAINER_PREFIX)
                for line in ps_output.splitlines():
                    if not line.strip():
                        continue
                    cid = line.split("\t")[0]
                    if runner.docker_stop(cid):
                        stopped = True

        return stopped

    def status(self) -> dict[str, object]:
        """Get vLLM backend status."""
        import subprocess

        api_port = (
            int(cast("int", self._active_state.get("api_port", self.api_port)))
            if self._active_state
            else self.api_port
        )
        api_host = (
            self._active_state.get("api_host", "localhost") if self._active_state else "localhost"
        )
        result: dict[str, object] = {
            "backend": self.name,
            "api_url": f"http://{api_host}:{api_port}",
            "api_port": api_port,
        }

        # Check for running container
        try:
            ps = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    f"name={CONTAINER_PREFIX}",
                    "--format",
                    "{{.Names}}\t{{.Status}}",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            containers = ps.stdout.strip()
            result["containers"] = containers or "none"
        except Exception:
            containers = ""
            result["containers"] = "unknown"

        # Check if vLLM API responds
        try:
            resp = httpx.get(f"http://{api_host}:{api_port}/v1/models", timeout=3)
            result["server_running"] = resp.status_code == 200
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                result["model"] = models[0]["id"] if models else "unknown"
        except Exception:
            result["server_running"] = bool(containers)

        # Node info from config (optional — may not have remote config)
        try:
            config = self.config
            result["nodes"] = len(config.nodes)
            result["head_ip"] = config.head_node.ip
        except (FileNotFoundError, ValueError):
            result["nodes"] = 1
            result["head_ip"] = api_host

        return result

    def get_claude_env(self) -> dict[str, str]:
        """Get environment variables for Claude Code.

        Points at the Anthropic API proxy (not vLLM directly), since Claude Code
        speaks Anthropic Messages API but vLLM speaks OpenAI Chat Completions.
        """
        proxy = (
            int(cast("int", self._active_state.get("proxy_port", self.proxy_port)))
            if self._active_state
            else self.proxy_port
        )
        return {
            "ANTHROPIC_BASE_URL": f"http://localhost:{proxy}",
            "ANTHROPIC_AUTH_TOKEN": "vllm",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        }

    def get_claude_model_arg(self, model: str) -> str:
        """Get the model ID for claude --model flag."""
        return self.get_model_id(model)

    def is_healthy(self) -> bool:
        """Check if vLLM is responding (via proxy if running, else direct)."""
        # Check proxy first
        proxy_port = (
            int(cast("int", self._active_state.get("proxy_port", self.proxy_port)))
            if self._active_state
            else self.proxy_port
        )
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"http://localhost:{proxy_port}/health")
                if resp.status_code == 200:
                    return True
        except Exception:
            pass
        # Fall back to direct vLLM check
        try:
            with httpx.Client(timeout=5) as client:
                resp = client.get(f"{self.get_api_url()}/health")
                return resp.status_code == 200
        except Exception:
            return False

    def start_proxy(self) -> int:
        """Start the Anthropic API proxy as a background process.

        Returns the proxy PID.
        """
        import subprocess

        vllm_url = self.get_api_url()
        proxy_port = (
            int(cast("int", self._active_state.get("proxy_port", self.proxy_port)))
            if self._active_state
            else self.proxy_port
        )

        # Check if proxy already running
        try:
            with httpx.Client(timeout=2) as client:
                resp = client.get(f"http://localhost:{proxy_port}/health")
                if resp.status_code == 200:
                    click.echo(f"Proxy already running on :{proxy_port}")
                    return 0
        except Exception:
            pass

        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "autosre.backends.anthropic_proxy",
                str(proxy_port),
                vllm_url,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # Wait briefly for startup
        import time

        time.sleep(1)

        if proc.poll() is not None:
            msg = "Anthropic proxy failed to start"
            raise RuntimeError(msg)

        # Save proxy PID
        pids = self._load_pids()
        pids["proxy"] = proc.pid
        self._save_pids(pids)

        click.echo(f"Anthropic proxy started on :{proxy_port} (PID {proc.pid})")
        return proc.pid

    def stop_proxy(self) -> bool:
        """Stop the Anthropic API proxy."""
        pids = self._load_pids()
        proxy_pid = pids.get("proxy")
        if proxy_pid and isinstance(proxy_pid, int):
            stopped = self._stop_process(proxy_pid, "autosre.backends.anthropic_proxy")
            if not stopped:
                # Try harder — the cmdline match might differ
                try:
                    import os

                    os.kill(proxy_pid, 0)
                    import signal

                    os.kill(proxy_pid, signal.SIGTERM)
                    stopped = True
                    click.echo(f"Stopped proxy (PID {proxy_pid})")
                except (ProcessLookupError, PermissionError):
                    pass
            pids.pop("proxy", None)
            self._save_pids(pids)
            return stopped
        return False

    @staticmethod
    def _apply_sm121_moe_fix(container_name: str) -> None:
        """Disable Marlin MoE on SM121 (Blackwell) to prevent CUDA crashes.

        The fused_marlin_moe kernel hits 'cudaErrorUnknown' on SM121.
        This patches check_moe_marlin_supports_layer to return False on
        SM12+, forcing the stable MoeWNA16 Triton-based fallback for MoE
        layers. Linear layers still use Marlin for performance.
        """
        import subprocess

        patch_script = r'''
import torch
cap = torch.cuda.get_device_capability()
if cap[0] < 12:
    exit(0)  # Not Blackwell, no patch needed

from vllm.model_executor.layers.quantization.utils import marlin_utils
import inspect
src_file = inspect.getfile(marlin_utils)
with open(src_file) as f:
    content = f.read()
if "SM121_MOE_FIX" in content:
    print("[SM121] MoE fix already applied")
    exit(0)
old = "def check_moe_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:"
new = """def check_moe_marlin_supports_layer(layer: LinearBase, group_size: int) -> bool:
    # SM121_MOE_FIX: Disable Marlin MoE on Blackwell (crashes with cudaErrorUnknown)
    import torch
    _cap = torch.cuda.get_device_capability()
    if _cap[0] >= 12:
        return False"""
if old not in content:
    print("[SM121] WARNING: Could not find function to patch")
    exit(0)
with open(src_file, "w") as f:
    f.write(content.replace(old, new))
print("[SM121] MoE fix applied: Marlin MoE disabled, Triton MoE enabled")
'''
        result = subprocess.run(
            ["docker", "exec", container_name, "python3", "-c", patch_script],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.stdout.strip():
            click.echo(f"  {result.stdout.strip()}")

    @staticmethod
    def _apply_sm121_fp8_fix(container_name: str) -> None:
        """Disable CUTLASS block-FP8 linear kernel on SM121 (Blackwell).

        vLLM's ``CUTLASS_BLOCK_FP8_SUPPORTED`` capability probe reports True
        on SM121 but the kernel crashes with ``cutlass_gemm_caller Error
        Internal`` during the first forward pass when serving
        Qwen/Qwen3.6-35B-A3B-FP8 (block-128 weights).  Force the flag to
        False so vLLM falls through to the DeepGEMM / FlashInfer / Triton
        FP8 path.  DeepGEMM is separately disabled via the
        ``VLLM_USE_DEEP_GEMM=0`` env var set in the recipe, leaving Triton
        as the working fallback on Blackwell today.

        Safe no-op on non-Blackwell hardware (compute capability < 12).
        """
        import subprocess

        patch_script = r"""
import torch
cap = torch.cuda.get_device_capability()
if cap[0] < 12:
    exit(0)  # Not Blackwell, no patch needed

src_file = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/quantization/utils/w8a8_utils.py"
with open(src_file) as f:
    content = f.read()
if "SM121_FP8_FIX" in content:
    print("[SM121] FP8 fix already applied")
    exit(0)
old = "CUTLASS_BLOCK_FP8_SUPPORTED = cutlass_block_fp8_supported()"
new = "CUTLASS_BLOCK_FP8_SUPPORTED = False  # SM121_FP8_FIX: cutlass block-FP8 crashes on Blackwell"
if old not in content:
    print("[SM121] WARNING: Could not find CUTLASS_BLOCK_FP8_SUPPORTED assignment to patch")
    exit(0)
with open(src_file, "w") as f:
    f.write(content.replace(old, new))
print("[SM121] FP8 fix applied: CUTLASS block-FP8 disabled, Triton fallback enabled")
"""
        result = subprocess.run(
            ["docker", "exec", container_name, "python3", "-c", patch_script],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if result.stdout.strip():
            click.echo(f"  {result.stdout.strip()}")

    def _wait_for_vllm(self, host: str, port: int, timeout: int = 300) -> bool:
        """Poll vLLM /health endpoint until ready."""
        import time

        url = f"http://{host}:{port}/health"
        start = time.time()
        while time.time() - start < timeout:
            try:
                with httpx.Client(timeout=5) as client:
                    resp = client.get(url)
                    if resp.status_code == 200:
                        return True
            except Exception:
                pass
            time.sleep(1)
        return False

    @staticmethod
    def assert_running_vllm_matches_recipe(
        recipe: dict[str, Any],
        api_port: int = 8010,
    ) -> list[str]:
        """Compare the live `vllm serve ...` cmdline to what the recipe
        would have generated. Returns a list of human-readable mismatch
        descriptions (empty if aligned).

        Called from `autosre start` after the vLLM is up so that a
        host running stale flags (older recipe state, manual override)
        emits a WARNING per drift point. Same-shape protection as
        meeting-scribe's `infra.compose.warn_on_recipe_source_drift`.

        Concrete past failure (2026-04-30): customer GB10 was missing
        `--attention-backend=flashinfer` + the cuBLAS hardening env
        vars because they were never in the recipe yaml — only present
        on the local autosre process via uncodified manual overrides.
        Recipe was updated in commit 0bbe204; this check ensures any
        subsequent drift between recipe + live process surfaces in
        logs immediately."""
        import shlex
        import subprocess

        mismatches: list[str] = []

        # Find the live vllm serve process for this api_port.
        try:
            ps = subprocess.run(
                ["ps", "-eo", "cmd"],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return [f"could not run `ps` to inspect live vllm: {exc!s}"]

        live_cmd: list[str] | None = None
        for line in ps.stdout.splitlines():
            if (
                "vllm" in line
                and "serve" in line
                and recipe.get("model_id", "") in line
                and (f"--port={api_port}" in line or f"--port {api_port}" in line)
            ):
                live_cmd = shlex.split(line)
                break
        if live_cmd is None:
            # Not an error — vllm may not be up yet on this codepath.
            return []

        # Extract live flag dict.
        live: dict[str, str] = {}
        i = 0
        while i < len(live_cmd):
            tok = live_cmd[i]
            if tok.startswith("--"):
                name = tok.lstrip("-")
                if "=" in name:
                    key, _, val = name.partition("=")
                    live[key] = val
                    i += 1
                    continue
                if i + 1 < len(live_cmd) and not live_cmd[i + 1].startswith("--"):
                    live[name] = live_cmd[i + 1]
                    i += 2
                    continue
                live[name] = ""
            i += 1

        # Check the perf-sensitive recipe-driven flags.
        check_pairs: list[tuple[str, str, object]] = [
            ("max-model-len", str(recipe.get("max_model_len", "")), recipe.get("max_model_len")),
            ("gpu-memory-utilization", str(recipe.get("gpu_memory_utilization", "")), recipe.get("gpu_memory_utilization")),
            ("max-num-seqs", str(recipe.get("max_num_seqs", "")), recipe.get("max_num_seqs")),
            ("max-num-batched-tokens", str(recipe.get("max_num_batched_tokens", "")), recipe.get("max_num_batched_tokens")),
            ("quantization", str(recipe.get("quantization", "")), recipe.get("quantization")),
            ("attention-backend", str(recipe.get("attention_backend", "")), recipe.get("attention_backend")),
        ]
        for live_key, expected_str, expected_val in check_pairs:
            if expected_val is None:
                continue
            actual = live.get(live_key)
            if actual is None:
                mismatches.append(
                    f"missing live flag --{live_key}={expected_str} (recipe specifies it)"
                )
                continue
            try:
                if float(actual) != float(expected_str):
                    mismatches.append(
                        f"--{live_key}: live={actual} recipe={expected_str}"
                    )
            except (TypeError, ValueError):
                if actual != expected_str:
                    mismatches.append(
                        f"--{live_key}: live={actual} recipe={expected_str}"
                    )

        # Extra args — every flag in extra_args must appear on the cmdline.
        for arg in recipe.get("extra_args", []) or []:
            arg_name = arg.split("=", 1)[0].lstrip("-")
            if arg_name not in live:
                mismatches.append(f"missing live extra_arg {arg}")

        return mismatches

    @staticmethod
    def warn_on_recipe_drift(
        recipe: dict[str, Any], api_port: int = 8010
    ) -> None:
        """Boot-time hook: emit a WARNING per recipe ↔ live-cmdline
        mismatch. Never raises. Safe to call from `autosre start`
        after the vLLM is healthy."""
        try:
            mismatches = VllmBackend.assert_running_vllm_matches_recipe(
                recipe, api_port=api_port
            )
        except Exception as exc:
            logger.warning(
                "autosre recipe-parity check raised %s: %s",
                type(exc).__name__,
                exc,
            )
            return
        for m in mismatches:
            logger.warning("autosre recipe drift: %s", m)

    def _save_containers(self, containers: dict[str, str]) -> None:
        """Save container IDs to state file."""
        state_file = self.data_dir / "vllm_containers.json"
        state_file.write_text(json.dumps(containers))

    def _load_containers(self) -> dict[str, str]:
        """Load saved container IDs."""
        state_file = self.data_dir / "vllm_containers.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text())
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError):
                pass
        # Also check active_state
        if self._active_state and isinstance(self._active_state.get("containers"), dict):
            containers = cast("dict[str, Any]", self._active_state["containers"])
            return {str(k): str(v) for k, v in containers.items()}
        return {}

    @classmethod
    def reconcile_state(cls) -> dict[str, object] | None:
        """Reconstruct active.json from a running vLLM container, if any.

        Called by ``load_active_state()`` when active.json is missing but
        ``vllm_containers.json`` still points at a container. We verify the
        container is running, probe vLLM's ``/v1/models`` for the served model
        id, reverse-lookup the short model key, and write a fresh active.json.
        Returns the reconstructed state, or ``None`` if nothing to attach to.
        """
        import subprocess

        xdg_data = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        containers_file = xdg_data / "autosre" / "vllm_containers.json"
        try:
            raw = json.loads(containers_file.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(raw, dict) or not raw:
            return None

        live: dict[str, str] = {}
        for key, cid in raw.items():
            if not isinstance(cid, str):
                continue
            result = subprocess.run(
                ["docker", "inspect", "--format={{.State.Running}}", cid],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and "true" in result.stdout:
                live[str(key)] = cid
        if not live:
            # Stored IDs may be stale (container recreated by restart policy).
            # Fall back to well-known container name.
            name = f"{CONTAINER_PREFIX}-local"
            result = subprocess.run(
                ["docker", "inspect", "--format={{.Id}} {{.State.Running}}", name],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and "true" in result.stdout:
                new_id = result.stdout.strip().split()[0]
                live["local"] = new_id
            else:
                return None

        served_model: str | None = None
        try:
            with httpx.Client(timeout=3) as client:
                resp = client.get(f"http://localhost:{cls.api_port}/v1/models")
                if resp.status_code == 200:
                    data = resp.json().get("data") or []
                    if data:
                        served_model = data[0].get("id")
        except Exception:
            return None
        if not served_model:
            return None

        model_key = cls.default_model
        for key, full_id in cls.models.items():
            if full_id == served_model:
                model_key = key
                break

        state: dict[str, object] = {
            "backend": cls.name,
            "model": model_key,
            "api_port": cls.api_port,
            "api_host": "localhost",
            "proxy_port": cls.proxy_port,
            "containers": live,
        }
        save_active_state(state)
        containers_file.write_text(json.dumps(live))
        return state
