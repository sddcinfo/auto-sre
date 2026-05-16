"""Swarm launcher — launches Claude Code agent teams with task templates.

Supports two providers:

- ``local`` (default): routes Claude Code at the local vLLM/llama.cpp/Ollama
  backend through the existing Anthropic-compatible proxy. This is the
  behavior auto-sre has always had.
- ``anthropic``: launches Claude Code against the online Anthropic API using
  Claude Code's own native auth. No environment variables are injected; all
  ``ANTHROPIC_*`` overrides the user may have in their shell are purged so
  that the online session does not accidentally point at a local proxy.

Supports two execution modes:

- Interactive (default): replaces the current process with
  ``os.execvpe``. This is what ``autosre swarm launch`` has always done.
- Eval (``eval_mode`` set): runs Claude Code as a subprocess with stdout
  piped to a transcript file. Used by ``autosre eval run`` so every turn
  can be captured and later normalized into ``turns.jsonl``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import click

if TYPE_CHECKING:
    from autosre.backends.base import Backend
    from autosre.swarm.templates import TaskTemplate


Provider = Literal["local", "anthropic"]


# Env vars that select which Anthropic endpoint Claude Code talks to. We
# purge all of them in both provider modes and then (only for local) put
# the correct ones back. In anthropic mode they stay purged so Claude Code
# falls through to its native configured auth.
_ANTHROPIC_OVERRIDE_KEYS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_LARGE_MODEL",
    "CLAUDE_CODE_ATTRIBUTION_HEADER",
)


DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-6[1m]"


@dataclass
class EvalLaunchSpec:
    """Per-run inputs that turn ``SwarmLauncher`` into an eval capture driver.

    When this is passed the launcher:

    - runs Claude Code as a subprocess (not ``os.execvpe``);
    - sets cwd to ``worktree_path`` (a read-only snapshot of the target);
    - appends ``--output-format=stream-json --verbose`` so the transcript
      can be parsed into ``TurnRecord`` rows;
    - writes a restricted settings file that drops ``Bash`` from the
      allow-list (the filesystem layer is what actually prevents writes to
      the snapshot — see ``autosre/eval/snapshot.py``).
    """

    capture_dir: Path
    worktree_path: Path
    findings_file: Path
    suite_name: str
    run_id: str
    read_only: bool = True
    timeout_seconds: int = 1800
    extra_env: dict[str, str] = field(default_factory=dict)


@dataclass
class CaptureResult:
    """Return value of :meth:`SwarmLauncher.launch` in eval mode."""

    exit_code: int
    transcript_path: Path
    stderr_path: Path
    duration_ms: float


class SwarmLauncher:
    """Launches Claude Code agent teams against local or online providers.

    The launcher is stateful for exactly one launch. Reuse is fine, but each
    ``launch()`` call builds a fresh env + cmd.
    """

    def __init__(
        self,
        backend: Backend,
        template: TaskTemplate | None = None,
        *,
        provider: Provider = "local",
        anthropic_model: str = DEFAULT_ANTHROPIC_MODEL,
    ) -> None:
        self.backend = backend
        self.template = template
        self.provider: Provider = provider
        self.anthropic_model = anthropic_model

    # ── Environment ────────────────────────────────────────────────

    def build_env(self) -> dict[str, str]:
        """Build the complete environment for the swarm.

        Both providers purge every ``ANTHROPIC_*`` override key. Then:

        - ``local`` re-applies the backend's ``get_claude_env()`` overrides
          and sets ``ANTHROPIC_API_KEY=local-vllm`` (any value works; the
          local proxy does not validate it).
        - ``anthropic`` leaves all of them unset, so Claude Code's native
          config (``~/.claude`` credentials) is the only source of truth.
        """
        env = os.environ.copy()

        for key in _ANTHROPIC_OVERRIDE_KEYS:
            env.pop(key, None)

        if self.provider == "local":
            env.update(self.backend.get_claude_env())
            env["ANTHROPIC_API_KEY"] = "local-vllm"

        # Agent teams flag is required for both providers. Claude Code
        # respects it regardless of whether the backend is local or cloud.
        env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"

        return env

    # ── Command building ───────────────────────────────────────────

    def _model_arg(self) -> str:
        """Return the ``--model=`` argument value for the current provider."""
        if self.provider == "anthropic":
            return self.anthropic_model
        return self.backend.get_claude_model_arg(self.backend.default_model)

    def _mcp_config_path(self) -> Path:
        mcp_config = {
            "mcpServers": {
                "autosre-fetch": {"command": "autosre-mcp-fetch"},
                "autosre-search": {"command": "autosre-mcp-search"},
            },
        }
        mcp_file = Path(tempfile.gettempdir()) / "autosre-mcp.json"
        mcp_file.write_text(json.dumps(mcp_config))
        return mcp_file

    def _interactive_settings_path(self) -> Path:
        settings = {
            "permissions": {
                "allow": [
                    "Bash(*)",
                    "Read",
                    "Write",
                    "Edit",
                    "Glob",
                    "Grep",
                    "Agent",
                    "mcp__autosre-search__web_search",
                    "mcp__autosre-fetch__web_fetch",
                ],
            },
        }
        settings_file = Path(tempfile.gettempdir()) / "autosre-settings.json"
        settings_file.write_text(json.dumps(settings))
        return settings_file

    def _eval_settings_path(self, spec: EvalLaunchSpec) -> Path:
        """Read-only settings profile for eval runs.

        Drops ``Bash`` entirely and keeps only file-inspection tools plus
        a narrow Write/Edit hint. The real enforcement that writes can
        only land in the agent-outputs directory comes from the filesystem
        layer in ``autosre/eval/snapshot.py`` (worktree files are chmod
        0o444), not from Claude Code permissions — this repo's permission
        model is coarse tool-level allow/deny, not path-scoped.
        """
        settings = {
            "permissions": {
                "allow": [
                    "Read",
                    "Glob",
                    "Grep",
                    "Agent",
                    "Write",
                    "Edit",
                    "mcp__autosre-search__web_search",
                    "mcp__autosre-fetch__web_fetch",
                ],
                "deny": ["Bash"],
            },
        }
        settings_file = spec.capture_dir / f"claude-settings-{spec.suite_name}.json"
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text(json.dumps(settings, indent=2))
        return settings_file

    def build_launch_cmd(
        self,
        model_arg: str,
        *,
        eval_mode: EvalLaunchSpec | None = None,
    ) -> list[str]:
        """Build the ``claude`` launch command.

        When ``eval_mode`` is ``None`` the command matches the interactive
        behavior auto-sre has always produced. When it is set the command
        instead points at a read-only snapshot, streams structured JSON
        output, and uses the restricted eval settings profile.
        """
        mcp_file = self._mcp_config_path()

        if eval_mode is None:
            settings_file = self._interactive_settings_path()
            system_prompt = (
                "You are running on a local vLLM backend with continuous "
                "batching. ALWAYS invoke multiple independent tool calls in "
                "a SINGLE response. Never do sequentially what can be done "
                "in parallel. Be precise with file paths."
            )
        else:
            settings_file = self._eval_settings_path(eval_mode)
            system_prompt = (
                "You are running inside an eval harness. The working "
                "directory is a read-only snapshot of the target repository. "
                "Write your findings to the exact absolute path the user "
                "gives you and nothing else. Do not print findings in chat. "
                "Do not attempt to modify the snapshot."
            )

        cmd = [
            "claude",
            "--bare",
            f"--settings={settings_file}",
            f"--model={model_arg}",
            f"--mcp-config={mcp_file}",
            f"--system-prompt={system_prompt}",
        ]

        if eval_mode is not None:
            cmd.append("--output-format=stream-json")
            cmd.append("--verbose")

        if self.template is not None:
            cmd.append(self.template.format_prompt())

        return cmd

    # ── Launch ─────────────────────────────────────────────────────

    def launch(
        self,
        model_key: str | None = None,
        *,
        eval_mode: EvalLaunchSpec | None = None,
    ) -> CaptureResult | dict[str, object]:
        """Launch the agent swarm.

        Interactive mode (``eval_mode is None``) replaces the current
        process via ``os.execvpe``. Eval mode spawns a subprocess, tees
        stdout into ``capture_dir/transcript.jsonl``, stderr into
        ``capture_dir/stderr.log``, waits for exit, and returns a
        :class:`CaptureResult`.
        """
        if not shutil.which("claude"):
            msg = "'claude' command not found. Install: npm install -g @anthropic-ai/claude-code"
            raise RuntimeError(msg)

        # Model argument resolution. model_key is only honored for local;
        # for anthropic the caller must set anthropic_model on the launcher.
        if self.provider == "anthropic":
            model_arg = self.anthropic_model
        else:
            chosen = model_key or self.backend.default_model
            model_arg = self.backend.get_claude_model_arg(chosen)

        env = self.build_env()
        cmd = self.build_launch_cmd(model_arg, eval_mode=eval_mode)

        if eval_mode is None:
            self._print_launch_banner(model_arg)
            os.execvpe("claude", cmd, env)
            # Unreachable in practice; keeps the return type honest.
            return {"launched": True}  # pragma: no cover

        return self._launch_capture(cmd, env, eval_mode)

    # ── Internal: capture subprocess ───────────────────────────────

    def _launch_capture(
        self,
        cmd: list[str],
        env: dict[str, str],
        spec: EvalLaunchSpec,
    ) -> CaptureResult:
        spec.capture_dir.mkdir(parents=True, exist_ok=True)
        transcript_path = spec.capture_dir / "transcript.jsonl"
        stderr_path = spec.capture_dir / "stderr.log"

        env = {**env, **spec.extra_env, "AUTOSRE_RUN_ID": spec.run_id}

        t0 = time.monotonic()
        with (
            transcript_path.open("wb") as t_out,
            stderr_path.open("wb") as e_out,
        ):
            proc = subprocess.Popen(
                cmd,
                cwd=str(spec.worktree_path),
                env=env,
                stdout=t_out,
                stderr=e_out,
            )
            try:
                exit_code = proc.wait(timeout=spec.timeout_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
                exit_code = -1
        duration_ms = (time.monotonic() - t0) * 1000

        return CaptureResult(
            exit_code=exit_code,
            transcript_path=transcript_path,
            stderr_path=stderr_path,
            duration_ms=duration_ms,
        )

    # ── Internal: banner ───────────────────────────────────────────

    def _print_launch_banner(self, model_arg: str) -> None:
        click.echo()
        click.secho("Launching agent swarm...", fg="cyan", bold=True)
        click.echo(f"  Provider: {self.provider}")
        click.echo(f"  Backend:  {self.backend.name}")
        click.echo(f"  Model:    {model_arg}")
        if self.provider == "local":
            click.echo(f"  URL:      {self.backend.get_api_url()}")
        else:
            click.echo("  URL:      (native Claude Code auth)")
        click.echo("  Swarm:    enabled (agent teams)")
        if self.template is not None:
            click.echo(f"  Template: {self.template.name} ({self.template.num_agents} agents)")
            for i, role in enumerate(self.template.agent_roles):
                click.echo(f"    Agent {i + 1}: {role}")
        click.echo()
