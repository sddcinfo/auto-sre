"""llama.cpp backend using llama-server with native Anthropic Messages API."""

import shutil
import subprocess

import click
import httpx

from .base import Backend, clear_active_state, save_active_state


class LlamaCppBackend(Backend):
    """llama.cpp backend using llama-server.

    Uses the native Anthropic Messages API at /v1/messages
    (merged in llama.cpp PR #17570, Nov 2025, enabled by default).
    """

    name = "llamacpp"
    description = "llama.cpp (llama-server, native Anthropic API)"

    api_port = 8080

    models = {
        "qwen3.6": "unsloth/Qwen3.6-35B-A3B-GGUF:Q4_K_M",
    }
    default_model = "qwen3.6"

    def check_requirements(self) -> tuple[bool, list[str]]:
        """Check if llama-server is available."""
        missing = []

        if not shutil.which("llama-server"):
            missing.append("llama-server (install: brew install llama.cpp)")

        return len(missing) == 0, missing

    def setup(
        self,
        force: bool = False,  # noqa: ARG002
        **kwargs: object,  # noqa: ARG002
    ) -> bool:
        """Set up llama.cpp environment."""
        click.secho("=== llama.cpp Backend Setup ===", fg="cyan", bold=True)

        if not shutil.which("llama-server"):
            click.secho("ERROR: llama-server not found", fg="red")
            click.echo("Install: brew install llama.cpp")
            return False

        click.secho("llama-server found!", fg="green")
        click.echo("Models are downloaded automatically on first use via -hf flag.")
        click.echo("For offline use, pre-download with: huggingface-cli download <model>")

        click.secho("llama.cpp setup complete!", fg="green")
        return True

    def get_claude_env(self) -> dict[str, str]:
        """Get environment variables for Claude Code."""
        return {
            "ANTHROPIC_BASE_URL": f"http://localhost:{self.api_port}",
            "ANTHROPIC_AUTH_TOKEN": "llamacpp",
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
        }

    def get_claude_model_arg(self, model: str) -> str:
        """Get the model ID to pass via claude --model."""
        return self.get_model_id(model)

    def is_healthy(self) -> bool:
        """Check if llama-server is healthy."""
        try:
            with httpx.Client(timeout=2) as client:
                resp = client.get(f"http://localhost:{self.api_port}/health")
                return resp.status_code == 200
        except Exception:
            return False

    def start(
        self,
        model: str | None = None,
        foreground: bool = False,
        **kwargs: object,  # noqa: ARG002
    ) -> dict[str, object]:
        """Start llama-server with a model.

        Args:
            model: Model to use (HuggingFace model ID or local path)
            foreground: Run in foreground with logs
        """
        model = model or self.default_model
        model_id = self.get_model_id(model)

        click.echo(f"Using model: {model_id}")

        # Stop any tracked processes
        self._stop_tracked_processes()

        # Build command
        cmd = [
            "llama-server",
            "-hf",
            model_id,
            "--port",
            str(self.api_port),
            "-ngl",
            "99",
        ]

        click.echo()
        click.secho("=== Starting llama-server ===", fg="cyan", bold=True)
        click.echo(f"Command: {' '.join(cmd)}")

        proc = subprocess.Popen(
            cmd,
            stdout=None if foreground else subprocess.DEVNULL,
            stderr=None if foreground else subprocess.DEVNULL,
        )

        pids: dict[str, object] = {"llama-server": proc.pid}

        # Wait for server to be ready
        click.echo("Waiting for llama-server...")
        if not self._wait_for_server(f"http://localhost:{self.api_port}/health", timeout=300):
            proc.kill()
            raise RuntimeError("llama-server failed to start (model download may take a while)")

        click.secho("llama-server is ready!", fg="green")
        click.echo(f"Anthropic Messages API: http://localhost:{self.api_port}/v1/messages")

        # Save PIDs
        self._save_pids(pids)

        # Write active state
        save_active_state(
            {
                "backend": "llamacpp",
                "model": model,
                "api_port": self.api_port,
            }
        )

        return {
            "pids": pids,
            "model": model_id,
            "api_port": self.api_port,
        }

    def stop(self, **kwargs: object) -> bool:  # noqa: ARG002
        """Stop llama-server."""
        stopped = self._stop_tracked_processes()
        clear_active_state()
        return stopped

    def status(self) -> dict[str, object]:
        """Get llama.cpp backend status."""
        pids = self._load_pids()
        healthy = self.is_healthy()

        return {
            "backend": self.name,
            "server_running": healthy,
            "pids": pids,
            "api_port": self.api_port,
        }
