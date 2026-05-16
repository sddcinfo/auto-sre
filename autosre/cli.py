"""CLI entry point for auto-sre."""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import click
import httpx

from . import __version__
from .backends import BackendType, detect_platform, get_backend
from .backends.base import load_active_state


def _resolve_backend(backend: str) -> BackendType:
    """Resolve backend string to BackendType."""
    return detect_platform() if backend == "auto" else BackendType(backend)


@click.group(invoke_without_command=True)
@click.version_option(version=__version__)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Auto-SRE: Local LLM server management for Claude Code.

    Run with no arguments to open the interactive dashboard.
    """
    if ctx.invoked_subcommand is None:
        from .tui import main

        main()


@cli.command()
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["ollama", "llamacpp", "vllm", "mlx-dflash", "auto"]),
    default="auto",
    help="Backend to use (auto-detects by default)",
)
@click.option("--force", "-f", is_flag=True, help="Force reinstall")
def setup(backend: str, force: bool) -> None:
    """Set up the environment for running local LLMs.

    \b
    Backends:
    - ollama: Universal default (requires Ollama >= 0.14.0)
    - llamacpp: llama-server (brew install llama.cpp)
    """
    if backend == "auto":
        backend_type = detect_platform()
        click.echo(f"Auto-detected backend: {backend_type.value}")
    else:
        backend_type = BackendType(backend)

    b = get_backend(backend_type)

    ok, missing = b.check_requirements()
    if not ok:
        click.secho("Missing requirements:", fg="red")
        for m in missing:
            click.echo(f"  - {m}")
        sys.exit(1)

    if not b.setup(force=force):
        sys.exit(1)


@cli.command()
@click.option("--no-scribe", is_flag=True, help="Skip starting meeting-scribe services and UI")
@click.option(
    "--model",
    "model_key",
    default=None,
    help=(
        "Override the vLLM model to load (model_key or recipe basename, e.g. "
        "'qwen3.6-fp8').  Default: the backend's default_model."
    ),
)
def start(no_scribe: bool, model_key: str | None) -> None:
    """Start the full stack: meeting-scribe + coding agent + proxy.

    \b
    Starts:
      1. nv-monitor (hardware metrics)
      2. Anthropic API proxy
      3. Meeting-scribe services + UI
      4. Coding agent (Qwen3.6-35B-A3B-FP8 on vLLM by default)
      5. Warmup request

    \b
    Examples:
      autosre start                           # Start everything (default model)
      autosre start --no-scribe               # Coding agent only
    """
    import concurrent.futures

    scribe_compose = (
        Path(__file__).resolve().parent.parent.parent / "meeting-scribe" / "docker-compose.gb10.yml"
    )

    t_start = time.time()

    click.echo("Starting full stack...")

    # 0. Captive portal (ports 80 + 443 for hotspot WiFi guests)
    scribe_scripts = Path(__file__).resolve().parent.parent.parent / "meeting-scribe" / "scripts"
    for script_name, port, pid_file in [
        ("captive-portal-80.py", 80, "/tmp/meeting-captive-80.pid"),
        ("captive-portal-443.py", 443, "/tmp/meeting-captive-443.pid"),
    ]:
        script = scribe_scripts / script_name
        if script.exists() and not no_scribe:
            # Check if already running
            try:
                import socket

                s = socket.socket()
                s.settimeout(0.5)
                s.connect(("127.0.0.1", port))
                s.close()
                click.secho(f"  captive portal :{port}: already running", fg="green")
            except Exception:
                proc = subprocess.Popen(
                    ["sudo", sys.executable, str(script)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                Path(pid_file).write_text(str(proc.pid))
                click.secho(f"  captive portal :{port}: started", fg="green")

    # 1. nv-monitor (instant, needed for TUI)
    nv_monitor = Path.home() / "tools" / "nv-monitor" / "nv-monitor"
    if nv_monitor.exists():
        try:
            httpx.get("http://localhost:9100/metrics", timeout=1)
            click.secho("  nv-monitor: already running", fg="green")
        except Exception:
            subprocess.Popen(
                [str(nv_monitor), "-n", "-p", "9100"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            click.secho("  nv-monitor: started (:9100)", fg="green")

    # 2. Proxy (instant, ready before model loads)
    from .backends.vllm import VllmBackend

    b_proxy = get_backend(BackendType.VLLM)
    if isinstance(b_proxy, VllmBackend):
        try:
            httpx.get("http://localhost:8011/health", timeout=2)
            click.secho("  proxy: already running (:8011)", fg="green")
        except Exception:
            try:
                b_proxy.start_proxy()
                click.secho("  proxy: started (:8011)", fg="green")
            except RuntimeError as e:
                click.secho(f"  proxy: failed ({e})", fg="yellow")

    # 3. Scribe + LLM in parallel (LLM is the slow step)
    def _start_scribe() -> str:
        if no_scribe or not scribe_compose.exists():
            return "skipped"
        r = subprocess.run(
            ["docker", "compose", "-f", str(scribe_compose), "up", "-d"],
            capture_output=True,
            text=True,
            check=False,
        )
        return "started" if r.returncode == 0 else f"failed: {r.stderr[:80]}"

    def _start_agent() -> dict[str, object]:
        # Skip if already running UNLESS caller explicitly asked for a
        # different model — in that case let the new start() fail fast
        # with the existing "model already loaded" guard rather than
        # silently ignoring the --model override.
        b = get_backend(BackendType.VLLM)
        active = load_active_state()
        if active and not model_key:
            port = int(cast("int", active.get("api_port", 8010)))
            try:
                resp = httpx.get(f"http://localhost:{port}/health", timeout=3)
                if resp.status_code == 200:
                    return dict(active)  # already running
            except Exception:
                pass
        return b.start(model=model_key or b.default_model)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        scribe_fut = pool.submit(_start_scribe)
        agent_fut = pool.submit(_start_agent)

        startup_futs: list[concurrent.futures.Future[Any]] = [scribe_fut, agent_fut]
        for fut in concurrent.futures.as_completed(startup_futs):
            if fut is scribe_fut:
                status = fut.result()
                click.secho(f"  scribe: {status}", fg="green" if "started" in status else "yellow")
            elif fut is agent_fut:
                try:
                    result = fut.result()
                    click.secho(
                        f"  LLM: {result.get('model', '?')} on :{result.get('api_port', 8010)}",
                        fg="green",
                    )
                except RuntimeError as e:
                    click.secho(f"  LLM: FAILED ({e})", fg="red")
                    sys.exit(1)

    # 4. Warmup ALL components (eliminates cold-start latency)
    active = load_active_state()
    if active:
        api_port = int(cast("int", active.get("api_port", 8010)))
        click.echo("  Warming up all components...")

        def _warmup_vllm() -> str | None:
            """Warmup vLLM with a translation request (fills KV cache, compiles kernels)."""
            try:
                model_resp = httpx.get(f"http://localhost:{api_port}/v1/models", timeout=5)
                model_id = model_resp.json()["data"][0]["id"]
                # Translation warmup (most common request type)
                httpx.post(
                    f"http://localhost:{api_port}/v1/chat/completions",
                    json={
                        "model": model_id,
                        "max_tokens": 50,
                        "messages": [
                            {"role": "system", "content": "Translate Japanese to English."},
                            {"role": "user", "content": "テスト"},
                        ],
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    timeout=60,
                )
                # Code warmup (triggers reasoning path)
                httpx.post(
                    f"http://localhost:{api_port}/v1/chat/completions",
                    json={
                        "model": model_id,
                        "max_tokens": 10,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                    timeout=60,
                )
                return "vllm"
            except Exception:
                return None

        # IMPORTANT: these three must trigger a full inference round, not
        # just hit /health. Health only confirms the container started —
        # it does not compile the CUDA kernels. Without a real warmup
        # the first user meeting has a ~1 min cold start while vLLM's
        # ASR attention mask, pyannote's powerset head, and TTS's CUDA
        # graph all JIT on the first request. We generate a real PCM
        # blob, send it through each backend's real API path, and only
        # return success once each one produces a valid output.
        #
        # Both TTS and pyannote occasionally come up in a broken CUDA
        # state on Blackwell SM_121 (see /UPGRADE-NOTES-2026-04.md):
        #   - pyannote: powerset.to_multilabel → one_hot → CUDA error
        #   - TTS: "GET was unable to find an engine to execute this
        #     computation" (torch kernel dispatch miss)
        # Mitigation in both cases is `docker compose ... --force-recreate`
        # of the affected container. So each warmup helper tries once,
        # and if that fails, force-recreates its container, waits for
        # health, and retries. Only the second failure is fatal.
        import base64 as _base64
        import io as _io
        import wave as _wave

        SCRIBE_COMPOSE = (
            Path(__file__).resolve().parent.parent.parent
            / "meeting-scribe"
            / "docker-compose.gb10.yml"
        )

        def _wait_port(port: int, timeout: int) -> bool:
            """Wait for /health to return 200. Used before both the
            initial inference warmup and the retry-after-recreate.
            """
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    r = httpx.get(f"http://localhost:{port}/health", timeout=2)
                    if r.status_code == 200:
                        return True
                except Exception:
                    pass
                time.sleep(1.5)
            return False

        def _wait_vllm_models(port: int, timeout: int) -> bool:
            """Wait for vLLM /v1/models to return a valid model list.
            vLLM reports /health 200 immediately but the inference path
            isn't ready until /v1/models returns the served model id.
            """
            deadline = time.time() + timeout
            while time.time() < deadline:
                try:
                    r = httpx.get(f"http://localhost:{port}/v1/models", timeout=5)
                    if r.status_code == 200 and r.json().get("data"):
                        return True
                except Exception:
                    pass
                time.sleep(2.0)
            return False

        def _force_recreate(service: str, ready_port: int, timeout: int = 90) -> bool:
            """Force-recreate a scribe compose service and wait for
            its /health to return 200. Returns True on success.
            """
            try:
                r = subprocess.run(
                    [
                        "docker",
                        "compose",
                        "-f",
                        str(SCRIBE_COMPOSE),
                        "up",
                        "-d",
                        "--force-recreate",
                        service,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if r.returncode != 0:
                    return False
            except Exception:
                return False
            return _wait_port(ready_port, timeout)

        def _make_pcm_wav(seconds: float = 3.0, sr: int = 16000) -> tuple[bytes, bytes]:
            """Build (raw_s16le_pcm, wav_bytes) of non-silent audio.

            A sine sweep gives ASR/VAD enough signal to leave the
            silence-discard path and run the model. Pure silence
            would be rejected before it ever reaches the CUDA kernel,
            defeating the warmup.
            """
            import math

            n = int(seconds * sr)
            amp = 16000  # ~-6dBFS for s16le
            pcm_ints = bytearray()
            for i in range(n):
                t = i / sr
                # Sweep 200-800 Hz so the spectrogram has content
                freq = 200.0 + 600.0 * (i / n)
                v = int(amp * math.sin(2 * math.pi * freq * t))
                pcm_ints += int(v).to_bytes(2, "little", signed=True)
            pcm = bytes(pcm_ints)

            buf = _io.BytesIO()
            with _wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(pcm)
            return pcm, buf.getvalue()

        def _try_asr() -> bool:
            """One real vLLM ASR inference call. True on 2xx."""
            try:
                _, wav_bytes = _make_pcm_wav(seconds=3.0)
                audio_b64 = _base64.b64encode(wav_bytes).decode()
                models_resp = httpx.get("http://localhost:8003/v1/models", timeout=10)
                models_resp.raise_for_status()
                model_id = models_resp.json()["data"][0]["id"]
                r = httpx.post(
                    "http://localhost:8003/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "input_audio",
                                        "input_audio": {"data": audio_b64, "format": "wav"},
                                    },
                                    {"type": "text", "text": "<|startoftranscript|>"},
                                ],
                            }
                        ],
                        "max_tokens": 64,
                        "temperature": 0.0,
                    },
                    timeout=90,
                )
                return r.status_code < 500
            except Exception:
                return False

        def _try_tts() -> bool:
            """One real TTS voice-clone synthesis call. True on a valid WAV."""
            try:
                _, wav_bytes = _make_pcm_wav(seconds=2.0)
                voice_b64 = _base64.b64encode(wav_bytes).decode()
                r = httpx.post(
                    "http://localhost:8002/v1/audio/speech",
                    json={
                        "model": "qwen3-tts",
                        "input": "warm up",
                        "voice": voice_b64,
                        "language": "English",
                        "response_format": "wav",
                        "speed": 1.0,
                    },
                    timeout=120,
                )
                if r.status_code != 200:
                    return False
                # Confirm we got a RIFF/WAVE payload, not a JSON error
                return len(r.content) >= 100 and r.content.startswith(b"RIFF")
            except Exception:
                return False

        def _try_diarize() -> bool:
            """One real pyannote /v1/diarize call. True on 200 JSON."""
            try:
                pcm, _ = _make_pcm_wav(seconds=20.0)
                r = httpx.post(
                    "http://localhost:8001/v1/diarize",
                    content=pcm,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-Sample-Rate": "16000",
                        "X-Min-Speakers": "0",
                        "X-Max-Speakers": "2",
                    },
                    timeout=120,
                )
                return r.status_code == 200
            except Exception:
                return False

        def _warmup_asr() -> str | None:
            # Wait for vLLM ASR to actually serve /v1/models — docker
            # compose returns before the Qwen3-ASR weights finish
            # loading (start_period=60s). Without this wait the first
            # inference call races the container startup and fails
            # with Connection refused.
            if not _wait_vllm_models(8003, timeout=150):
                return None
            if _try_asr():
                return "asr"
            time.sleep(2)
            return "asr" if _try_asr() else None

        def _warmup_tts() -> str | None:
            if not _wait_port(8002, timeout=120):
                return None
            if _try_tts():
                return "tts"
            # TTS came up in a broken CUDA dispatch state — recreate
            click.secho("  tts: first inference failed, force-recreating…", fg="yellow")
            if not _force_recreate("qwen3-tts", 8002, timeout=120):
                return None
            return "tts" if _try_tts() else None

        def _warmup_diarize() -> str | None:
            if not _wait_port(8001, timeout=120):
                return None
            if _try_diarize():
                return "diarize"
            # pyannote CUDA kernel dispatch error on Blackwell SM_121 —
            # clear by force-recreating the container (see
            # containers/pyannote/server.py:_apply_blackwell_patches
            # and /UPGRADE-NOTES-2026-04.md).
            click.secho("  diarize: first inference failed, force-recreating…", fg="yellow")
            if not _force_recreate("pyannote-diarize", 8001, timeout=120):
                return None
            return "diarize" if _try_diarize() else None

        import concurrent.futures

        # Track success/failure so we can report + block the "stack is
        # running" banner until everything is truly warm.
        warmed: list[str] = []
        warmup_failures: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futs = {
                pool.submit(_warmup_vllm): "vLLM",
                pool.submit(_warmup_asr): "ASR",
                pool.submit(_warmup_tts): "TTS",
                pool.submit(_warmup_diarize): "Diarize",
            }
            for fut in concurrent.futures.as_completed(futs):
                result_name = fut.result()
                name = futs[fut]
                if result_name:
                    warmed.append(name)
                else:
                    warmup_failures.append(name)

        if warmup_failures:
            # Loud failure — do NOT silently mark the stack "ready" if a
            # backend refuses the first inference, because the user will
            # hit a broken meeting. Report and exit non-zero so systemd
            # and interactive shells see it.
            click.secho(
                f"  warmup: {', '.join(warmed) or 'none'} ready, "
                f"FAILED: {', '.join(warmup_failures)}",
                fg="red",
                bold=True,
            )
            click.secho(
                "  Stack is NOT ready — a real inference call to "
                "each failed backend errored out. Check:",
                fg="red",
            )
            click.echo("    docker logs scribe-asr")
            click.echo("    docker logs scribe-tts")
            click.echo("    docker logs scribe-diarization")
            click.echo(
                "  Diarize CUDA errors can be cleared with:"
                " docker compose -f repos/meeting-scribe/docker-compose.gb10.yml"
                " up -d --force-recreate pyannote-diarize"
            )
            sys.exit(1)

        click.secho(f"  warmup: {', '.join(warmed)} ready", fg="green")

    elapsed = time.time() - t_start

    # 5. Scribe UI server (needs meeting-scribe venv)
    scribe_ui_started = False
    if not no_scribe:
        scribe_dir = Path(__file__).resolve().parent.parent.parent / "meeting-scribe"
        scribe_venv = scribe_dir / ".venv" / "bin" / "meeting-scribe"
        if scribe_venv.exists():
            # Check if already running. Scribe admin listener is HTTPS with a
            # self-signed cert and binds 127.0.0.1 + the management IP.
            try:
                httpx.get(
                    "https://127.0.0.1:8080/api/status",
                    timeout=2,
                    verify=False,  # noqa: S501 — self-signed cert on localhost
                )
                scribe_ui_started = True
                click.secho("  scribe UI: already running (:8080)", fg="green")
            except Exception:
                click.echo("  Starting meeting-scribe UI server...")
                r = subprocess.run(
                    [str(scribe_venv), "start"],
                    capture_output=True,
                    cwd=str(scribe_dir),
                    check=False,
                )
                scribe_ui_started = r.returncode == 0
                if scribe_ui_started:
                    click.secho("  scribe UI: started (:8080)", fg="green")
                else:
                    stderr_text = (
                        r.stderr.decode("utf-8", "replace")[:100]
                        if isinstance(r.stderr, bytes)
                        else r.stderr[:100]
                        if r.stderr
                        else "unknown"
                    )
                    click.secho(
                        f"  scribe UI: failed ({stderr_text})",
                        fg="yellow",
                    )

    click.echo()
    click.secho("=" * 50, fg="green")
    click.secho(f"  Auto-SRE stack is running! ({elapsed:.0f}s)", fg="green", bold=True)
    click.secho("=" * 50, fg="green")
    click.echo()
    click.echo(f"  Coding agent:  {result.get('model', 'qwen3.6-fp8')}")
    click.echo(f"  vLLM:          http://localhost:{result.get('api_port', 8010)}")
    click.echo(f"  Proxy:         http://localhost:{result.get('proxy_port', 8011)}")
    if not no_scribe:
        click.echo("  Scribe:        translation, ASR, diarization, TTS")
    if scribe_ui_started:
        click.echo("  Scribe UI:     https://127.0.0.1:8080 (or management IP)")
    click.echo()
    click.echo("  autosre claude    — launch Claude Code (local)")
    click.echo("  autosre stop      — stop everything")


@cli.command("install-service")
@click.option(
    "--no-enable",
    is_flag=True,
    help="Write the unit file but do not enable/start it.",
)
@click.option(
    "--no-start",
    is_flag=True,
    help="Enable for boot but skip starting now (avoids the 3-7 min vLLM cold-load).",
)
@click.option("-q", "--quiet", is_flag=True, help="Reduce output to errors only.")
def install_service_cmd(no_enable: bool, no_start: bool, quiet: bool) -> None:
    """Install the user systemd unit so autosre's vLLM starts on boot.

    Writes ``~/.config/systemd/user/autosre.service`` (or
    ``$XDG_CONFIG_HOME/systemd/user/...``), runs ``daemon-reload``,
    enables the unit for boot, and optionally starts it now. Also
    runs ``loginctl enable-linger`` so user services run before
    login — required for a customer GB10 that boots without a
    console session.

    The unit's ``ExecStart`` runs ``autosre start --no-scribe``;
    meeting-scribe owns its own user unit (``meeting-scribe.service``)
    via ``meeting-scribe install-service``.
    """
    from autosre.service_installer import install as _install_service

    rc = _install_service(no_enable=no_enable, no_start=no_start, quiet=quiet)
    if rc != 0:
        sys.exit(rc)


@cli.command("uninstall-service")
@click.option("-q", "--quiet", is_flag=True, help="Reduce output to errors only.")
def uninstall_service_cmd(quiet: bool) -> None:
    """Stop, disable, and remove the user systemd unit.

    Does not undo ``loginctl enable-linger`` — that flag is generic
    and may be wanted by other user services.
    """
    from autosre.service_installer import uninstall as _uninstall_service

    rc = _uninstall_service(quiet=quiet)
    if rc != 0:
        sys.exit(rc)


@cli.command()
@click.option("--no-scribe", is_flag=True, help="Skip stopping meeting-scribe services")
@click.option(
    "--unload-model",
    is_flag=True,
    help="Also stop the vLLM model container (takes minutes to reload!)",
)
def stop(no_scribe: bool, unload_model: bool) -> None:
    """Stop proxy + services. The vLLM model is kept running by default.

    \b
    The 35B model takes 3+ minutes to load, so it is preserved across
    stop/start cycles. Use --unload-model to explicitly remove it.

    Firewall rules are intentionally left in place (harmless when no hotspot active).
    """
    # Stop proxy (and optionally vLLM container)
    active = load_active_state()
    if active:
        backend_type = BackendType(str(active["backend"]))
        b = get_backend(backend_type, active_state=active)
        b.stop(unload_model=unload_model)
        if unload_model:
            click.secho("  coding agent + proxy: stopped (model unloaded)", fg="green")
        else:
            click.secho("  proxy: stopped (model still running)", fg="green")
    else:
        click.echo("  coding agent: not running")

    # Stop scribe UI server
    if not no_scribe:
        scribe_dir = Path(__file__).resolve().parent.parent.parent / "meeting-scribe"
        scribe_venv = scribe_dir / ".venv" / "bin" / "meeting-scribe"
        if scribe_venv.exists():
            r = subprocess.run(
                [str(scribe_venv), "stop"],
                capture_output=True,
                text=True,
                cwd=str(scribe_dir),
                check=False,
            )
            if "not running" not in (r.stdout + r.stderr):
                click.secho("  scribe UI: stopped", fg="green")

    # Stop scribe containers
    if not no_scribe:
        scribe_compose = (
            Path(__file__).resolve().parent.parent.parent
            / "meeting-scribe"
            / "docker-compose.gb10.yml"
        )
        if scribe_compose.exists():
            click.echo("Stopping meeting-scribe services...")
            subprocess.run(
                ["docker", "compose", "-f", str(scribe_compose), "down"],
                capture_output=True,
                check=False,
            )
            click.secho("  meeting-scribe containers: stopped", fg="green")
        else:
            click.echo("  meeting-scribe: compose file not found")

    # Stop captive portal processes
    if not no_scribe:
        for pid_file in ("/tmp/meeting-captive-80.pid", "/tmp/meeting-captive-443.pid"):
            pid_path = Path(pid_file)
            if pid_path.exists():
                try:
                    pid = int(pid_path.read_text().strip())
                    subprocess.run(
                        ["sudo", "kill", str(pid)],
                        capture_output=True,
                        check=False,
                        timeout=5,
                    )
                    pid_path.unlink(missing_ok=True)
                except (ValueError, OSError):
                    pass
        # Also kill any orphaned captive portal processes
        subprocess.run(
            ["sudo", "pkill", "-f", "captive-portal-[48]"],
            capture_output=True,
            check=False,
            timeout=5,
        )
        click.secho("  captive portals: stopped", fg="green")

    # Stop nv-monitor
    nv_result = subprocess.run(
        ["pkill", "-f", "nv-monitor"],
        capture_output=True,
        check=False,
        timeout=5,
    )
    if nv_result.returncode == 0:
        click.secho("  nv-monitor: stopped", fg="green")

    # Clean state files (but NOT firewall rules — they're persistent)
    Path("/tmp/meeting-hotspot.json").unlink(missing_ok=True)
    Path("/tmp/meeting-scribe-active.json").unlink(missing_ok=True)


@cli.command()
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["ollama", "llamacpp", "vllm", "mlx-dflash", "auto"]),
    default="auto",
    help="Backend to check",
)
def status(backend: str) -> None:
    """Check server status."""
    # Try active state first
    active = load_active_state()
    if active and backend == "auto":
        backend_type = BackendType(str(active["backend"]))
    else:
        backend_type = _resolve_backend(backend)

    b = get_backend(backend_type, active_state=active)
    s = b.status()

    click.echo(f"Backend: {s['backend']}")
    click.echo()

    if s["backend"] == "ollama":
        ollama_status = (
            click.style("RUNNING", fg="green")
            if s.get("ollama_running")
            else click.style("STOPPED", fg="red")
        )
        click.echo(f"  Ollama Server:  {ollama_status}")
        click.echo(f"  Version:        {s.get('ollama_version', 'unknown')}")
        click.echo(f"  Anthropic API:  {'yes' if s.get('supports_anthropic') else 'no'}")
    elif s["backend"] == "llamacpp":
        server_status = (
            click.style("RUNNING", fg="green")
            if s.get("server_running")
            else click.style("STOPPED", fg="red")
        )
        click.echo(f"  llama-server:   {server_status}")
    elif s["backend"] == "vllm":
        server_status = (
            click.style("RUNNING", fg="green")
            if s.get("server_running")
            else click.style("STOPPED", fg="red")
        )
        click.echo(f"  vLLM Server:    {server_status}")
        click.echo(f"  Head Node:      {s.get('head_ip', 'unknown')}")
        click.echo(f"  Nodes:          {s.get('nodes', '?')}")
        click.echo(f"  API URL:        {s.get('api_url', 'unknown')}")
        containers = s.get("containers")
        if containers and containers != "none":
            click.echo(f"  Containers:     {containers}")
        if s.get("error"):
            click.secho(f"  Error:          {s['error']}", fg="red")

    click.echo(f"  API Port:       {s.get('api_port', b.api_port)}")

    pids = s.get("pids")
    if isinstance(pids, dict) and pids:
        click.echo()
        click.echo("PIDs:")
        for name, pid in pids.items():
            click.echo(f"  {name}: {pid}")

    models_list = s.get("models")
    if isinstance(models_list, list) and models_list:
        click.echo()
        click.echo("Available models:")
        for m in models_list:
            click.echo(f"  - {m}")


@cli.command("precommit")
@click.option("--verbose", "-v", is_flag=True, default=False)
@click.option(
    "--warn-only",
    is_flag=True,
    default=False,
    help="Treat all hits as warnings (exit 0 even on block-level findings).",
)
@click.option(
    "--include-all-tracked",
    is_flag=True,
    default=False,
    help="Scan every tracked file (deep scan) instead of only the working tree.",
)
@click.pass_context
def precommit(
    ctx: click.Context,
    verbose: bool,
    warn_only: bool,
    include_all_tracked: bool,
) -> None:
    """Scan auto-sre working tree for sensitive data before commit.

    Uses the vendored ``precommit_scanner`` module — no external tool
    required. Flags credentials, private keys, MAC/IP leaks, and other
    block-level patterns.
    """
    from autosre import precommit_scanner

    repo_root = Path(__file__).resolve().parents[1]
    ctx.invoke(
        precommit_scanner.precommit,
        repo=repo_root,
        include_staged=True,
        include_all_tracked=include_all_tracked,
        verbose=verbose,
        warn_only=warn_only,
    )


@cli.command()
@click.option(
    "--prompt", "-p", default="Say 'hello world' and nothing else.", help="Test prompt to send"
)
def test(prompt: str) -> None:
    """Send a test prompt to verify the server is working."""
    click.echo("Testing server connection...")
    click.echo()

    active = load_active_state()
    if not active:
        click.secho("ERROR: No active server. Run 'autosre start' first.", fg="red")
        sys.exit(1)

    backend_type = BackendType(str(active["backend"]))
    b = get_backend(backend_type, active_state=active)

    if not b.is_healthy():
        click.secho("ERROR: Server not responding", fg="red")
        sys.exit(1)

    model_key = str(active.get("model", b.default_model))
    model_id = b.get_model_id(model_key)

    click.echo(f"Backend: {backend_type.value}")
    click.echo(f"Model: {model_id}")
    click.echo(f"Prompt: {prompt}")
    click.echo()

    # Send test request via Anthropic Messages API
    try:
        with httpx.Client(timeout=60) as client:
            start_time = time.time()
            resp = client.post(
                f"{b.get_api_url()}/v1/messages",
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": b.get_claude_env().get("ANTHROPIC_AUTH_TOKEN", "test"),
                },
                json={
                    "model": model_id,
                    "max_tokens": 100,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            elapsed = time.time() - start_time

            if resp.status_code != 200:
                click.secho(f"ERROR: Server returned {resp.status_code}", fg="red")
                click.echo(resp.text)
                sys.exit(1)

            data = resp.json()
            content_blocks = data.get("content", [])

            # Extract text and thinking blocks
            text_parts = []
            for block in content_blocks:
                if block.get("type") == "thinking":
                    text_parts.append(f"[thinking] {block.get('thinking', '')}")
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            text = "\n".join(text_parts)
            usage = data.get("usage", {})

            click.secho("Response:", fg="cyan", bold=True)
            click.echo(text or "(empty — model may need different prompt format)")
            click.echo()
            click.echo(
                f"Tokens: {usage.get('input_tokens', '?')} in / {usage.get('output_tokens', '?')} out"
            )
            click.echo(f"Time: {elapsed:.2f}s")
            click.echo()
            click.secho("Server is working correctly!", fg="green")

    except httpx.TimeoutException:
        click.secho("ERROR: Request timed out", fg="red")
        sys.exit(1)
    except Exception as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)


@cli.command(context_settings={"ignore_unknown_options": True})
@click.option("--auto-start", is_flag=True, help="Auto-start server if not running")
@click.argument("prompt", nargs=-1, type=click.UNPROCESSED)
def claude(auto_start: bool, prompt: tuple[str, ...]) -> None:
    """Launch Claude Code with local server configuration.

    Reads the active backend from active.json, purges cloud credentials,
    sets local-only credentials, and launches claude with --model.

    Any extra arguments are passed through as the initial prompt.
    Agent teams are always enabled.
    """
    active = load_active_state()
    if not active:
        if auto_start:
            click.echo("No active server, starting...")
            backend_type = detect_platform()
            b = get_backend(backend_type)
            try:
                b.start()
            except RuntimeError as e:
                click.secho(f"ERROR: {e}", fg="red")
                sys.exit(1)
            active = load_active_state()
            if not active:
                click.secho("ERROR: Failed to start server", fg="red")
                sys.exit(1)
        else:
            click.secho("ERROR: No server running. Run 'autosre start' first.", fg="red")
            click.echo("Or use: autosre claude --auto-start")
            sys.exit(1)

    backend_type = BackendType(str(active["backend"]))
    b = get_backend(backend_type, active_state=active)
    model_key = str(active.get("model", b.default_model))

    # Auto-start Anthropic API proxy for vLLM backends
    if backend_type == BackendType.VLLM:
        from autosre.backends.vllm import VllmBackend

        if isinstance(b, VllmBackend):
            try:
                b.start_proxy()
            except RuntimeError as e:
                click.secho(f"ERROR: {e}", fg="red")
                sys.exit(1)

    # Check health
    if not b.is_healthy():
        click.secho("ERROR: Server not responding", fg="red")
        sys.exit(1)

    # PURGE all cloud credentials — enforce 100% local execution
    env = os.environ.copy()
    for key in [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_MODEL",
        "ANTHROPIC_SMALL_FAST_MODEL",
        "ANTHROPIC_LARGE_MODEL",
        "CLAUDE_CODE_ATTRIBUTION_HEADER",
        # Block OAuth/keychain fallback
        "CLAUDE_CODE_USE_OAUTH",
        "CLAUDE_CODE_API_KEY",
    ]:
        env.pop(key, None)

    # SET local-only credentials — ALL traffic goes to localhost proxy
    env.update(b.get_claude_env())
    env["ANTHROPIC_API_KEY"] = "local-vllm"

    # Pre-trust the CWD so Claude Code doesn't show the workspace trust dialog
    import json as _json_trust
    import uuid

    cwd = Path.cwd().resolve()
    slug = str(cwd).replace("/", "-").removeprefix("-")
    project_dir = Path.home() / ".claude" / "projects" / f"-{slug}"
    project_dir.mkdir(parents=True, exist_ok=True)
    # Write a minimal session marker — Claude Code trusts dirs that have session data
    marker = project_dir / "autosre-trust.jsonl"
    if not marker.exists():
        marker.write_text(
            _json_trust.dumps(
                {
                    "type": "summary",
                    "sessionId": str(uuid.uuid4()),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
            + "\n"
        )

    # Always enable agent teams
    env["CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS"] = "1"

    # Get model arg
    model_arg = b.get_claude_model_arg(model_key)

    # Wire up the plan-review loop in LOCAL-ONLY mode. The symmetric design:
    #   - autosre claude (this path) → chain = ["local"] — local model self-reviews
    #     with a fresh HTTP context. No codex/gemini/claude CLI fallback, because
    #     the whole point of autosre claude is offline / local-only operation.
    #   - bare claude (no autosre wrapper) → hooks installed via `autosre hooks
    #     install` use the default chain (codex gpt-5.4 xhigh → local → gemini →
    #     claude), so Claude plans and Codex reviews via the external API chain.
    # AUTOSRE_REVIEW_MODEL is passed through so _local_provider_runner can skip
    # a /v1/models roundtrip.
    env["AUTOSRE_REVIEW_CHAIN"] = "local"
    env["AUTOSRE_REVIEW_MODEL"] = model_arg
    env["AUTOSRE_RUN_ID"] = env.get("AUTOSRE_RUN_ID") or str(uuid.uuid4())

    # Force ALL model usage to local — no cloud fallback for subagents/teams
    env["ANTHROPIC_MODEL"] = model_arg
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model_arg
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model_arg
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model_arg
    env["CLAUDE_CODE_SUBAGENT_MODEL"] = model_arg
    # Deprecated but still checked by some code paths
    env["ANTHROPIC_SMALL_FAST_MODEL"] = model_arg

    # Find claude executable
    if not shutil.which("claude"):
        click.secho("ERROR: 'claude' command not found", fg="red")
        click.echo("Install Claude Code: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    click.echo()
    click.secho(
        f"─── Local LLM: {model_arg.split('/')[-1]} @ :{b.get_claude_env()['ANTHROPIC_BASE_URL'].split(':')[-1]} ───",
        fg="cyan",
        dim=True,
    )
    click.echo()

    # Wire up local MCP servers for web search/fetch.
    # Unique filenames per launch (keyed on AUTOSRE_RUN_ID) so concurrent
    # `autosre claude` sessions don't clobber each other's config files.
    import json as _json
    import tempfile

    launch_id = env["AUTOSRE_RUN_ID"]
    tmp_root = Path(tempfile.gettempdir())

    mcp_config = {
        "mcpServers": {
            "autosre-fetch": {"command": "autosre-mcp-fetch"},
            "autosre-search": {"command": "autosre-mcp-search"},
            "autosre-capabilities": {"command": "autosre-mcp-capabilities"},
        },
    }
    mcp_file = tmp_root / f"autosre-mcp-{launch_id}.json"
    mcp_file.write_text(_json.dumps(mcp_config))

    # Build permissive settings for local sandbox (no permission prompts)
    # env block is inherited by teammates spawned via --settings
    local_settings: dict[str, object] = {
        "permissions": {
            "allow": [
                "Bash(*)",
                "Read",
                "Write",
                "Edit",
                "Glob",
                "Grep",
                "Agent",
                "TeamCreate",
                "TeamDelete",
                "SendMessage",
                "TaskCreate",
                "TaskUpdate",
                "TaskGet",
                "TaskList",
                "mcp__autosre-search__web_search",
                "mcp__autosre-fetch__web_fetch",
                "mcp__autosre-capabilities__list_modules",
                "mcp__autosre-capabilities__search_commands",
                "mcp__autosre-capabilities__get_command",
            ],
        },
        "model": model_arg,
        "modelOverrides": {
            "claude-opus-4-6": model_arg,
            "claude-sonnet-4-6": model_arg,
            "claude-haiku-4-5-20251001": model_arg,
            "claude-sonnet-4-5-20241022": model_arg,
        },
        "availableModels": [model_arg],
        "env": {
            "ANTHROPIC_BASE_URL": b.get_claude_env()["ANTHROPIC_BASE_URL"],
            "ANTHROPIC_API_KEY": "local-vllm",
            "ANTHROPIC_MODEL": model_arg,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model_arg,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": model_arg,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": model_arg,
            "CLAUDE_CODE_SUBAGENT_MODEL": model_arg,
            "CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS": "1",
        },
    }

    # Wire up the autosre claude_hooks scripts. Event keys live under a
    # top-level ``hooks`` wrapper per the Claude Code settings.json schema —
    # entries written at the root are silently ignored. Command strings
    # invoke ``autosre hooks run <module>`` so no absolute Python path leaks
    # into the settings file and the PATH-resolved ``autosre`` entrypoint
    # dispatches into ``autosre.claude_hooks.<module>.main()``.
    def _hook_cmd(module: str) -> str:
        return f"autosre hooks run {module}"

    local_hooks: dict[str, Any] = {}
    local_settings["hooks"] = local_hooks

    local_hooks["PreToolUse"] = [
        {
            "matcher": "Bash",
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_cmd("pretooluse_bash_guard"),
                    "timeout": 10,
                },
            ],
        },
        {
            "matcher": "Edit",
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_cmd("pretooluse_bash_guard"),
                    "timeout": 10,
                },
            ],
        },
        {
            "matcher": "Write",
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_cmd("pretooluse_bash_guard"),
                    "timeout": 10,
                },
            ],
        },
        {
            "matcher": "ExitPlanMode",
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_cmd("pretooluse_plan_review"),
                    "timeout": 1260,
                },
            ],
        },
    ]
    local_hooks["PostToolUse"] = [
        {
            "matcher": "Bash",
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_cmd("posttooluse_audit"),
                    "timeout": 5,
                },
                {
                    "type": "command",
                    "command": _hook_cmd("telemetry_async"),
                    "timeout": 5,
                },
                {
                    "type": "command",
                    "command": _hook_cmd("post_commit_scan_update"),
                    "timeout": 5,
                },
            ],
        },
    ]
    local_hooks["Stop"] = [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_cmd("stop_session_check"),
                    "timeout": 10,
                },
                {
                    "type": "command",
                    "command": _hook_cmd("telemetry_async"),
                    "timeout": 5,
                },
            ],
        },
    ]
    local_hooks["UserPromptSubmit"] = [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_cmd("user_prompt_submit_branch_check"),
                    "timeout": 5,
                },
            ],
        },
    ]
    local_hooks["PreCompact"] = [
        {
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_cmd("precompact_context"),
                    "timeout": 10,
                },
            ],
        },
    ]
    local_hooks["SubagentStart"] = [
        {
            "matcher": "Plan",
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_cmd("subagent_plan_context"),
                    "timeout": 10,
                },
            ],
        },
        {
            "matcher": "Explore",
            "hooks": [
                {
                    "type": "command",
                    "command": _hook_cmd("subagent_plan_context"),
                    "timeout": 10,
                },
            ],
        },
    ]

    settings_file = tmp_root / f"autosre-settings-{launch_id}.json"
    settings_file.write_text(_json.dumps(local_settings))

    # System prompt for local LLM optimization
    system_prompt = (
        "You are running on a local vLLM server with continuous batching. "
        "Make parallel tool calls whenever possible for performance.\n\n"
        "CRITICAL: The codebase is in the CURRENT WORKING DIRECTORY. "
        "NEVER cd to any other directory. Use relative paths only.\n\n"
        "RULES:\n"
        "1. Make multiple tool calls in a SINGLE response when independent.\n"
        "2. Read multiple files at once, not one at a time.\n"
        "3. Run multiple Bash commands simultaneously when independent.\n"
        "4. When spawning agents, launch ALL in ONE message.\n"
        "5. Use RELATIVE paths only. NEVER use cd or absolute paths.\n"
        "6. NEVER use sleep. Messages from teammates arrive automatically.\n\n"
        "=== AGENT TEAM TOOL SCHEMAS ===\n\n"
        "TeamCreate — creates infrastructure only, does NOT spawn agents:\n"
        "  PARAMETERS: {team_name: string (required), description: string (optional)}\n"
        '  EXAMPLE: TeamCreate({"team_name": "analysis", "description": "Code analysis"})\n\n'
        "Agent — spawn a teammate (call AFTER TeamCreate):\n"
        "  PARAMETERS: {description: string, prompt: string, subagent_type: string,\n"
        "    name: string (teammate display name), team_name: string (must match TeamCreate),\n"
        "    run_in_background: boolean (set true for parallel spawning)}\n"
        '  EXAMPLE: Agent({"description": "Review code", "prompt": "Review src/ for bugs",\n'
        '    "subagent_type": "general-purpose", "name": "reviewer",\n'
        '    "team_name": "analysis", "run_in_background": true})\n\n'
        "SendMessage — communicate with teammates:\n"
        "  PARAMETERS: {to: string (REQUIRED), summary: string (REQUIRED), message: string (REQUIRED)}\n"
        "  ALL THREE fields are MANDATORY. Omitting any field causes an error.\n"
        '  The "to" field MUST be a bare name like "reviewer".\n'
        '  NEVER use "@reviewer" or "analysis/reviewer" or any prefix.\n'
        '  EXAMPLE: SendMessage({"to": "reviewer", "summary": "check progress",\n'
        '    "message": "What have you found so far?"})\n\n'
        "=== COMMON MISTAKES TO AVOID ===\n"
        '- TeamCreate does NOT accept a "teammates" array\n'
        '- SendMessage REQUIRES "summary" field — omitting it causes an error\n'
        '- SendMessage "to" must be bare name — no @ prefix, no team/ prefix\n'
        "- Do NOT use sleep to wait for teammates — messages arrive automatically\n"
        '- Do NOT broadcast to "*" before teammates are spawned\n'
    )

    # Build claude command
    cmd = [
        "claude",
        f"--settings={settings_file}",
        f"--model={model_arg}",
        f"--mcp-config={mcp_file}",
        f"--append-system-prompt={system_prompt}",
        "--dangerously-skip-permissions",
    ]
    if prompt:
        cmd.append(" ".join(prompt))

    # Execute claude (interactive — user sees agent working)
    os.execvpe("claude", cmd, env)


@cli.command("ui")
def tui() -> None:
    """Interactive TUI dashboard — status, benchmarks, demos."""
    from .tui import main

    main()


@cli.command()
@click.option("--requests", "-n", default=20, help="Number of recent requests to show")
@click.option("--follow", "-f", is_flag=True, help="Live-updating view (2s refresh)")
def metrics(requests: int, follow: bool) -> None:
    """Show vLLM inference metrics and proxy request analytics."""
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    from .metrics import read_recent_requests, request_analytics, vllm_metrics

    console = Console()

    def _fmt_lat(ms: float) -> str:
        if ms < 1000:
            return f"{ms:.0f}ms"
        if ms < 60000:
            return f"{ms / 1000:.1f}s"
        return f"{int(ms // 60000)}m{int((ms % 60000) // 1000):02d}s"

    def _render() -> None:
        console.clear()

        # vLLM metrics
        vm = vllm_metrics()
        if vm:
            inf = Table(title="[bold]Inference Metrics[/]", show_header=False, padding=(0, 2))
            inf.add_column("Metric", style="bold")
            inf.add_column("Value", justify="right")

            inf.add_row("TTFT p50", _fmt_lat(vm.get("ttft_p50", 0) * 1000))
            inf.add_row("TTFT p99", _fmt_lat(vm.get("ttft_p99", 0) * 1000))
            inf.add_row("TPOT p50", _fmt_lat(vm.get("tpot_p50", 0) * 1000))
            inf.add_row("TPOT p99", _fmt_lat(vm.get("tpot_p99", 0) * 1000))
            inf.add_row("", "")

            running = int(vm.get("requests_running", 0))
            waiting = int(vm.get("requests_waiting", 0))
            q_style = "green" if waiting == 0 else "yellow" if waiting <= 2 else "red"
            inf.add_row("Active/Waiting", Text(f"{running}/{waiting}", style=q_style))

            kv_pct = vm.get("kv_cache_pct", 0) * 100
            kv_style = "green" if kv_pct < 70 else "yellow" if kv_pct < 90 else "red"
            inf.add_row("KV Cache", Text(f"{kv_pct:.1f}%", style=kv_style))
            inf.add_row("Prefix Cache", f"{vm.get('prefix_cache_hit_pct', 0):.0f}% hit")
            inf.add_row("Avg Queue", _fmt_lat(vm.get("queue_time_avg", 0) * 1000))
            inf.add_row("", "")
            inf.add_row("Prompt tok/s", f"{vm.get('prompt_tps', 0):.0f}")
            inf.add_row("Gen tok/s", f"{vm.get('gen_tps', 0):.0f}")
            inf.add_row("Total Requests", f"{vm.get('total_requests', 0)}")

            console.print(inf)
        else:
            console.print("[dim]vLLM metrics unavailable[/]")

        console.print()

        # Request analytics
        recent = read_recent_requests(requests)
        if recent:
            analytics = request_analytics(recent)
            console.print(
                f"[bold]Request Analytics[/]  "
                f"[yellow]Translation[/]: {analytics.get('translation_count', 0)} reqs, "
                f"avg {_fmt_lat(analytics.get('translation_avg_ms', 0))}  |  "
                f"[cyan]Coding[/]: {analytics.get('coding_count', 0)} reqs, "
                f"avg {_fmt_lat(analytics.get('coding_avg_ms', 0))}  |  "
                f"{analytics.get('requests_per_min', 0):.1f} req/min"
            )
            console.print()

            req_tbl = Table(title="[bold]Recent Requests[/]", show_header=True, header_style="bold")
            req_tbl.add_column("Time", style="dim")
            req_tbl.add_column("Source")
            req_tbl.add_column("Latency", justify="right")
            req_tbl.add_column("In/Out", justify="right")
            req_tbl.add_column("Prompt", max_width=50)

            for r in recent[-requests:]:
                ts = time.strftime("%H:%M:%S", time.localtime(r.get("ts", 0)))
                src = r.get("source", "?")
                src_style = "yellow" if src == "translation" else "cyan"
                lat = r.get("elapsed_ms", 0)
                lat_style = "green" if lat < 2000 else "yellow" if lat < 5000 else "red"
                in_tok = r.get("input_tokens", 0)
                out_tok = r.get("output_tokens", 0)
                prompt = r.get("prompt_prefix", "")[:50]
                req_tbl.add_row(
                    ts,
                    Text(src, style=src_style),
                    Text(_fmt_lat(lat), style=lat_style),
                    f"{in_tok}/{out_tok}",
                    Text(prompt, style="dim"),
                )
            console.print(req_tbl)
        else:
            console.print("[dim]No proxy requests logged yet[/]")

    if follow:
        console.print("[dim]Live metrics (Ctrl+C to exit)...[/]")
        try:
            while True:
                _render()
                time.sleep(2)
        except KeyboardInterrupt:
            pass
    else:
        _render()


@cli.command()
def backends() -> None:
    """List available backends."""
    detected = detect_platform()

    click.echo("Available backends:")
    click.echo()

    for bt in BackendType:
        try:
            b = get_backend(bt)
        except KeyError:
            # Backend registered in enum but not yet implemented
            click.echo(f"  {bt.value:10} - {click.style('(not installed)', fg='yellow')}")
            continue
        ok, missing = b.check_requirements()

        if bt == detected:
            bt_status = click.style("(auto-detected)", fg="cyan")
        elif ok:
            bt_status = click.style("(available)", fg="green")
        else:
            bt_status = click.style("(missing requirements)", fg="yellow")

        click.echo(f"  {bt.value:10} - {b.description} {bt_status}")

        if not ok and missing:
            for m in missing:
                click.echo(f"               - {m}")

    click.echo()
    click.echo(f"Default backend: {detected.value}")


@cli.command()
@click.option("--model", "-m", multiple=True, help="Model name or index to bench (default: all)")
@click.option("--list", "list_models", is_flag=True, help="List available models for benchmarking")
@click.option("--concurrent", "-c", default=3, help="Number of concurrent requests to test")
@click.option("--history", is_flag=True, help="Show past benchmark results")
def bench(model: tuple[str, ...], list_models: bool, concurrent: int, history: bool) -> None:
    """Benchmark vLLM models on the local GPU.

    \b
    Tests each model for:
      - Single-request throughput (TTFT + decode TPS)
      - Concurrent throughput (aggregate TPS)
      - Tool calling validation
      - GPU memory usage

    \b
    Examples:
      autosre bench                    # benchmark all models
      autosre bench --list             # show available models
      autosre bench -m 0 -m 2         # bench models #0 and #2
      autosre bench -m "Nemotron"     # bench by name substring
      autosre bench --history         # show past results
    """
    from .bench import MODELS, print_result, print_summary_table, run_single_benchmark, save_results

    if list_models:
        click.echo("Available models for benchmarking:")
        click.echo()
        for i, spec in enumerate(MODELS):
            click.echo(f"  {i}) {spec.name}")
            click.echo(f"     {spec.model_id} (~{spec.weight_size_gb:.0f}GB weights)")
        return

    if history:
        import os

        data_dir = (
            Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
            / "autosre"
            / "benchmarks"
        )
        if not data_dir.exists():
            click.echo("No benchmark history.")
            return
        for f in sorted(data_dir.glob("bench-*.json"), reverse=True):
            import json as _json

            data = _json.loads(f.read_text())
            click.echo(f"\n  {data['timestamp']}")
            for r in data["results"]:
                if r.get("error"):
                    click.echo(f"    {r['name']:30s}  FAILED")
                else:
                    click.echo(
                        f"    {r['name']:30s}  {r['single_tps']:5.1f} tok/s  │  {r['concurrent_agg_tps']:5.1f} agg  │  {r['gpu_memory_mb'] // 1024}GB"
                    )
        return

    # Select models
    specs = []
    if model:
        for m in model:
            if m.isdigit():
                idx = int(m)
                if 0 <= idx < len(MODELS):
                    specs.append(MODELS[idx])
                else:
                    click.secho(f"Invalid index: {m} (0-{len(MODELS) - 1})", fg="red")
                    sys.exit(1)
            else:
                matched = [
                    s
                    for s in MODELS
                    if m.lower() in s.name.lower() or m.lower() in s.model_id.lower()
                ]
                if matched:
                    specs.extend(matched)
                else:
                    click.secho(f"No model matching: {m}", fg="red")
                    sys.exit(1)
    else:
        specs = list(MODELS)

    click.secho("Auto-SRE Model Benchmark", bold=True)
    click.echo(f"Models: {len(specs)}  |  Concurrent: {concurrent}x")
    click.echo()

    results = []
    for spec in specs:
        click.secho(f"━━━ {spec.name} ({spec.weight_size_gb:.0f}GB weights) ━━━", bold=True)
        result = run_single_benchmark(spec, concurrent_n=concurrent)
        print_result(result)
        results.append(result)

    click.echo()
    click.secho("Summary", bold=True)
    print_summary_table(results)

    path = save_results(results)
    click.echo(f"\nResults saved to: {path}")


@cli.group("perf")
def perf() -> None:
    """Concurrent-workload regression harness for the shared vLLM.

    \b
    Drives translation (priority=-10) and coding (priority=10) workloads
    simultaneously against the running vLLM on :8010, records per-workload
    TTFT / TPS percentiles plus vLLM scheduler counters, and compares
    against a committed named baseline under benchmarks/baselines/.

    \b
    Exit codes:
      0 — clean (within tolerance, or no baseline to compare)
      1 — warn-only violations
      2 — one or more fail-severity violations
    """


def _check_gpu_contention(allow: bool) -> None:
    """Preflight for `autosre perf run`: refuse to benchmark when anything
    other than the target vLLM is holding GPU context.

    Detects (a) running meeting-scribe containers by name and (b) any
    non-autosre-vllm-local process in ``nvidia-smi --query-compute-apps``.
    Both produce wildly inflated TTFT / depressed TPS numbers that are
    not comparable to a committed baseline. Exits non-zero unless
    ``--allow-contention`` is passed.
    """
    import subprocess as _sp

    if allow:
        click.secho(
            "  [warn] --allow-contention set — numbers will not be comparable to the baseline",
            fg="yellow",
        )
        return

    offenders: list[str] = []

    try:
        r = _sp.run(
            [
                "docker",
                "ps",
                "--filter",
                "name=scribe",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        scribe_names = [n for n in r.stdout.splitlines() if n.strip()]
        if scribe_names:
            offenders.append(f"meeting-scribe containers: {', '.join(scribe_names)}")
    except (FileNotFoundError, _sp.TimeoutExpired):
        pass

    try:
        r = _sp.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        gpu_procs = [line for line in r.stdout.splitlines() if line.strip()]
        # Expect exactly the autosre vLLM EngineCore(s). Flag anything else.
        extra = [line for line in gpu_procs if "VLLM::EngineCore" not in line or len(gpu_procs) > 2]
        # If there's more than one distinct PID and one of them is not part of
        # the main container, it's contention.
        if len(gpu_procs) > 2:
            offenders.append(
                f"{len(gpu_procs)} GPU processes (expected ≤2 for a single vLLM engine):\n    "
                + "\n    ".join(gpu_procs)
            )
        elif extra and not any("VLLM::EngineCore" in line for line in gpu_procs):
            offenders.append("no vLLM EngineCore found on the GPU — is autosre-vllm-local up?")
    except (FileNotFoundError, _sp.TimeoutExpired):
        pass

    if offenders:
        click.secho(
            "refusing to run perf benchmark — GPU contention detected:",
            fg="red",
            bold=True,
        )
        for line in offenders:
            click.echo(f"  • {line}")
        click.echo()
        click.echo("Quiesce the environment and retry:")
        click.echo(
            "  systemctl --user stop meeting-scribe.service  # stop systemd supervisor first"
        )
        click.echo(
            "  meeting-scribe gb10 down                      # then the compose stack + autoheal"
        )
        click.echo("  autosre perf run                              # re-run")
        click.echo()
        click.echo("Restore after benchmarking:")
        click.echo("  systemctl --user start meeting-scribe.service")
        click.echo()
        click.echo("Or bypass (numbers will not match the baseline):")
        click.echo("  autosre perf run --allow-contention")
        raise click.exceptions.Exit(code=3)


@perf.command("run")
@click.option(
    "--baseline",
    "-b",
    default="gb10_qwen36_fp8_flashinfer",
    help="Named baseline to compare against.",
)
@click.option("--duration", "-d", default=60, help="Seconds per measured phase.")
@click.option("--coding-concurrency", default=2, help="Parallel coding requests during contention.")
@click.option(
    "--translation-rps", default=2.0, help="Translation requests per second during contention."
)
@click.option(
    "--saturate-slots", is_flag=True, help="Add a saturation phase that forces priority preemption."
)
@click.option("--no-proxy-check", is_flag=True, help="Skip the :8011 proxy sanity check.")
@click.option("--no-save", is_flag=True, help="Don't write the run JSON/markdown to disk.")
@click.option(
    "--no-compare", is_flag=True, help="Skip baseline comparison even if the baseline exists."
)
@click.option(
    "--allow-contention",
    is_flag=True,
    help="Bypass the GPU-contention preflight (meeting-scribe ASR/TTS/diarize containers sharing the GPU). "
    "Numbers will not be comparable to the baseline.",
)
@click.option(
    "--model",
    "model_id",
    default=None,
    help="Override the model id sent in chat-completion payloads. Defaults to the hardcoded "
    "3.5-INT4 id; set this when benching a different loaded model (e.g. 3.6-FP8) against "
    "the running vLLM endpoint.  When unset but an active backend is detected, the id is "
    "auto-resolved from `/v1/models` so a cold-swapped server doesn't silently produce "
    "4xx errors against the old hardcoded id.",
)
def perf_run(
    baseline: str,
    duration: int,
    coding_concurrency: int,
    translation_rps: float,
    saturate_slots: bool,
    no_proxy_check: bool,
    no_save: bool,
    no_compare: bool,
    allow_contention: bool,
    model_id: str | None,
) -> None:
    """Run the concurrent-workload benchmark and compare against a baseline."""
    import asyncio
    import json as _json
    import os as _os
    import sys as _sys
    from pathlib import Path as _Path

    import httpx as _httpx

    from .perf.baseline import compare, load_baseline
    from .perf.harness import RunConfig
    from .perf.harness import run as run_harness
    from .perf.report import render_markdown, render_stdout

    _check_gpu_contention(allow_contention)

    # Resolve the served model id from /v1/models when the caller didn't
    # pin one explicitly.  Without this, `perf run` silently defaults to
    # the hardcoded 3.5-INT4 id and fires thousands of 4xx-erroring
    # requests against any other loaded model.
    if model_id is None:
        vllm_url = "http://localhost:8010"
        try:
            resp = _httpx.get(f"{vllm_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                served = data.get("data", [])
                if served:
                    resolved = served[0].get("id")
                    if isinstance(resolved, str) and resolved:
                        model_id = resolved
                        click.echo(f"  model: auto-detected served id = {model_id}")
        except Exception:
            pass

    if model_id:
        cfg = RunConfig(
            duration_seconds=duration,
            coding_concurrency=coding_concurrency,
            translation_rps=translation_rps,
            saturate_slots=saturate_slots,
            run_proxy_check=not no_proxy_check,
            model_id=model_id,
        )
    else:
        cfg = RunConfig(
            duration_seconds=duration,
            coding_concurrency=coding_concurrency,
            translation_rps=translation_rps,
            saturate_slots=saturate_slots,
            run_proxy_check=not no_proxy_check,
        )

    click.secho(
        f"perf run — duration={duration}s  coding_conc={coding_concurrency}  translation_rps={translation_rps}",
        bold=True,
    )
    if saturate_slots:
        click.echo("  (saturation phase enabled)")
    click.echo()

    result = asyncio.run(run_harness(cfg))

    base = None
    violations = []
    if not no_compare:
        try:
            base = load_baseline(baseline)
            violations = compare(result, base)
        except FileNotFoundError:
            click.secho(f"Baseline '{baseline}' not found — skipping comparison.", fg="yellow")

    render_stdout(result, base, violations)

    if not no_save:
        data_dir = (
            _Path(_os.environ.get("XDG_DATA_HOME", _Path.home() / ".local" / "share"))
            / "autosre"
            / "perf"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        json_path = data_dir / f"run-{result.timestamp}.json"
        md_path = data_dir / f"run-{result.timestamp}.md"
        json_path.write_text(_json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n")
        md_path.write_text(render_markdown(result, base, violations, title="autosre perf run"))
        click.echo(f"run saved: {json_path}")
        click.echo(f"markdown:  {md_path}")

    fails = [v for v in violations if v.severity == "fail"]
    warns = [v for v in violations if v.severity == "warn"]
    if fails:
        _sys.exit(2)
    if warns:
        _sys.exit(1)
    _sys.exit(0)


@perf.command("save-baseline")
@click.argument("name")
@click.option(
    "--from",
    "from_run",
    type=click.Path(exists=True, path_type=Path),
    help="Promote a specific run JSON (defaults to most recent).",
)
def perf_save_baseline(name: str, from_run: Path | None) -> None:
    """Promote a run to a committed baseline under benchmarks/baselines/."""
    import json as _json
    import os as _os
    from pathlib import Path as _Path

    from .perf.baseline import baselines_dir, save_baseline
    from .perf.harness import (
        PhaseResult,
        ProxySanity,
        RunResult,
        SchedulerCounters,
    )
    from .perf.report import render_markdown

    if from_run is None:
        data_dir = (
            _Path(_os.environ.get("XDG_DATA_HOME", _Path.home() / ".local" / "share"))
            / "autosre"
            / "perf"
        )
        candidates = sorted(data_dir.glob("run-*.json"))
        if not candidates:
            click.secho(
                "No runs found in $XDG_DATA_HOME/autosre/perf/. Run `autosre perf run` first.",
                fg="red",
            )
            sys.exit(1)
        from_run = candidates[-1]
        click.echo(f"Promoting most recent run: {from_run}")

    raw = _json.loads(from_run.read_text())
    run = RunResult(
        timestamp=raw["timestamp"],
        config=raw["config"],
        environment=raw["environment"],
        phases=[PhaseResult(**p) for p in raw["phases"]],
        scheduler=SchedulerCounters(**raw["scheduler"]),
        proxy_sanity=ProxySanity(**raw["proxy_sanity"]) if raw.get("proxy_sanity") else None,
    )

    json_path = save_baseline(name, run)
    md_path = baselines_dir() / f"{name}.md"
    md_path.write_text(render_markdown(run, None, [], title=f"baseline: {name}"))

    # Verify recipe files haven't changed since perf run, then mint approval tokens
    from autosre.hooks_backend import recipe_guard as _rg

    run_recipe_hashes: dict[str, str] = raw.get("environment", {}).get("recipe_hashes", {})
    for rel_path, validated_hash in run_recipe_hashes.items():
        full = Path.cwd() / rel_path
        if not full.exists():
            continue
        current_hash = _rg.content_hash(full.read_text())
        if current_hash != validated_hash:
            click.secho(
                f"ERROR: {rel_path} has changed since perf run "
                f"(run hash: {validated_hash[:12]}…, current: {current_hash[:12]}…). "
                f"Re-run `autosre perf run` with the current recipe.",
                fg="red",
            )
            sys.exit(1)

    approved = 0
    for rel_path, validated_hash in run_recipe_hashes.items():
        if _rg.is_protected_recipe(rel_path):
            _rg.write_perf_approval(str(Path.cwd() / rel_path), validated_hash)
            approved += 1
    if approved:
        click.echo(f"Minted {approved} recipe approval token(s)")

    click.secho(f"Wrote {json_path}", fg="green")
    click.secho(f"Wrote {md_path}", fg="green")
    click.echo()
    click.echo("To commit:")
    click.echo(f"  git add {json_path.relative_to(Path.cwd())} {md_path.relative_to(Path.cwd())}")
    click.echo(f"  git commit -m 'perf: add {name} baseline'")


@perf.command("compare")
@click.argument("run_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--baseline",
    "-b",
    default="gb10_qwen36_fp8_flashinfer",
    help="Baseline name to compare against.",
)
def perf_compare(run_path: Path, baseline: str) -> None:
    """Re-run baseline comparison on a stored run JSON (offline)."""
    import json as _json

    from .perf.baseline import compare, load_baseline
    from .perf.harness import (
        PhaseResult,
        ProxySanity,
        RunResult,
        SchedulerCounters,
    )
    from .perf.report import render_stdout

    raw = _json.loads(run_path.read_text())
    run = RunResult(
        timestamp=raw["timestamp"],
        config=raw["config"],
        environment=raw["environment"],
        phases=[PhaseResult(**p) for p in raw["phases"]],
        scheduler=SchedulerCounters(**raw["scheduler"]),
        proxy_sanity=ProxySanity(**raw["proxy_sanity"]) if raw.get("proxy_sanity") else None,
    )
    base = load_baseline(baseline)
    violations = compare(run, base)
    render_stdout(run, base, violations)

    fails = [v for v in violations if v.severity == "fail"]
    warns = [v for v in violations if v.severity == "warn"]
    if fails:
        sys.exit(2)
    if warns:
        sys.exit(1)


@perf.command("list")
def perf_list() -> None:
    """List committed baselines and recent runs."""
    import os as _os
    from pathlib import Path as _Path

    from .perf.baseline import baselines_dir

    click.secho("Committed baselines", bold=True)
    bdir = baselines_dir()
    if bdir.exists():
        for f in sorted(bdir.glob("*.json")):
            click.echo(f"  {f.stem}  ({f})")
    else:
        click.echo("  (none)")

    click.echo()
    click.secho("Recent runs", bold=True)
    data_dir = (
        _Path(_os.environ.get("XDG_DATA_HOME", _Path.home() / ".local" / "share"))
        / "autosre"
        / "perf"
    )
    if data_dir.exists():
        runs = sorted(data_dir.glob("run-*.json"), reverse=True)[:10]
        for f in runs:
            click.echo(f"  {f.stem}  ({f})")
    else:
        click.echo("  (none)")


@perf.command("show")
@click.argument("name")
def perf_show(name: str) -> None:
    """Render a baseline's markdown to stdout."""
    from .perf.baseline import baselines_dir

    md = baselines_dir() / f"{name}.md"
    if not md.exists():
        click.secho(f"Baseline markdown not found: {md}", fg="red")
        sys.exit(1)
    click.echo(md.read_text())


@perf.command("approve-edit")
@click.option("--recipe", "recipe_path", required=True, help="Path to the protected recipe YAML")
@click.option(
    "--old-string",
    "old_string",
    required=True,
    help="Exact substring to replace (same semantics as the Edit tool)",
)
@click.option(
    "--new-string",
    "new_string",
    required=True,
    help="Replacement substring (same semantics as the Edit tool)",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Skip the interactive confirmation prompt (for scripted use)",
)
@click.option(
    "--ttl-seconds",
    type=int,
    default=None,
    help="Approval lifetime in seconds (default: 7200 = 2h)",
)
def perf_approve_edit(
    recipe_path: str,
    old_string: str,
    new_string: str,
    yes: bool,
    ttl_seconds: int | None,
) -> None:
    """Mint an approval token for a specific recipe edit without re-benching.

    Companion to the PreToolUse recipe guard: an agent can suggest a
    recipe change but cannot apply it until a fresh approval token
    exists for the exact post-edit content.  ``save-baseline`` mints
    such tokens automatically after a validated perf run; this
    command is the explicit human-consent path for changes that are
    known safe or are about to be perf-validated in a separate
    tuning loop.

    The flow:

    \b
    1. Agent proposes a change (--old-string / --new-string).
    2. Operator runs this command, reviews the diff + list of
       perf-sensitive params that will change, and either confirms
       (interactive [y/N] prompt) or passes --yes.
    3. Token is written keyed on hash(post-edit-content); any
       subsequent Edit-tool call applying the exact same old/new
       strings passes the PreToolUse recipe guard.

    Example::

        autosre perf approve-edit \\
          --recipe autosre/backends/recipes/qwen3.6-35b-a3b-fp8.yaml \\
          --old-string "max_num_batched_tokens: 4096" \\
          --new-string "max_num_batched_tokens: 8192"
    """
    from autosre.hooks_backend import recipe_guard

    path = Path(recipe_path)
    if not path.exists():
        click.secho(f"Recipe not found: {path}", fg="red")
        sys.exit(1)

    if not recipe_guard.is_protected_recipe(str(path)):
        click.secho(
            f"Not a protected recipe: {path}\n"
            "  The Edit tool guard only gates files under backends/recipes/, "
            "meeting_scribe/recipes/, or meeting_scribe/stage_configs/.  "
            "No approval needed here.",
            fg="yellow",
        )
        sys.exit(1)

    try:
        after, diff, perf_changed = recipe_guard.preview_edit(str(path), old_string, new_string)
    except ValueError as exc:
        click.secho(f"ERROR: {exc}", fg="red")
        sys.exit(1)

    click.secho("Proposed recipe edit:", bold=True)
    click.echo(f"  recipe:   {path.resolve()}")
    if perf_changed:
        click.echo("  changes:  " + ", ".join(perf_changed))
    else:
        click.secho(
            "  changes:  (no perf-sensitive params — Edit would not need approval anyway)",
            fg="yellow",
        )

    click.echo()
    click.secho("Diff:", bold=True)
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            click.secho(line, fg="green")
        elif line.startswith("-") and not line.startswith("---"):
            click.secho(line, fg="red")
        else:
            click.echo(line)

    click.echo()

    if not yes:
        confirmed = click.confirm(
            "Approve this edit? (This grants the recipe-guard approval "
            "for the exact hash of the post-edit content.)",
            default=False,
        )
        if not confirmed:
            click.secho("Aborted — no token written.", fg="yellow")
            sys.exit(1)

    write_kwargs: dict[str, Any] = {"source": "approve-edit"}
    if ttl_seconds is not None:
        write_kwargs["ttl_seconds"] = ttl_seconds
    token_path = recipe_guard.write_perf_approval(
        str(path),
        recipe_guard.content_hash(after),
        **write_kwargs,
    )
    click.secho("Approval token written.", fg="green")
    click.echo(f"  path:      {token_path}")
    click.echo(f"  ttl:       {ttl_seconds if ttl_seconds is not None else 'default (7200 s)'}")
    click.echo(
        "The agent can now apply the proposed Edit via the standard tool call; "
        "the guard will match the hash and pass."
    )


@perf.command("boot")
@click.option(
    "--baseline",
    "-b",
    default=None,
    help="Boot baseline name to compare against (without boot_ prefix).",
)
@click.option(
    "--save-baseline",
    "save_name",
    default=None,
    help="Save result as a boot baseline with this name.",
)
@click.option("--no-compare", is_flag=True, help="Skip baseline comparison.")
@click.option("--no-save", is_flag=True, help="Don't write run result to disk.")
def perf_boot(
    baseline: str | None,
    save_name: str | None,
    no_compare: bool,
    no_save: bool,
) -> None:
    """Benchmark meeting-scribe cold-boot time.

    Stops the service and compose stack, starts the service, and measures
    wall-clock time until all backends are healthy. Isolates the run from
    the compose watchdog timer.
    """
    import json as _json
    import os as _os
    from pathlib import Path as _Path

    from .perf.boot import (
        compare_boot,
        load_boot_baseline,
        render_boot_markdown,
        render_boot_stdout,
        run_boot_benchmark,
        save_boot_baseline,
    )

    click.secho("perf boot — cold-start benchmark", bold=True)
    click.echo()

    result = run_boot_benchmark()

    base = None
    violations = []
    if not no_compare and baseline:
        try:
            base = load_boot_baseline(baseline)
            violations = compare_boot(result, base)
        except FileNotFoundError:
            click.secho(f"Boot baseline '{baseline}' not found — skipping comparison.", fg="yellow")
        except ValueError as e:
            click.secho(f"Boot baseline error: {e}", fg="red")

    render_boot_stdout(result, base, violations)

    if not no_save:
        data_dir = (
            _Path(_os.environ.get("XDG_DATA_HOME", _Path.home() / ".local" / "share"))
            / "autosre"
            / "perf"
        )
        data_dir.mkdir(parents=True, exist_ok=True)
        json_path = data_dir / f"boot-{result.timestamp}.json"
        md_path = data_dir / f"boot-{result.timestamp}.md"
        json_path.write_text(_json.dumps(result.to_json(), indent=2, ensure_ascii=False) + "\n")
        md_path.write_text(render_boot_markdown(result, base, violations))
        click.echo(f"run saved: {json_path}")

    if save_name:
        from .perf.boot import boot_baselines_dir

        json_path = save_boot_baseline(save_name, result)
        md_path = boot_baselines_dir() / f"boot_{save_name}.md"
        md_path.write_text(
            render_boot_markdown(result, None, [], title=f"boot baseline: {save_name}")
        )
        click.secho(f"Baseline saved: {json_path}", fg="green")
        click.secho(f"Markdown:       {md_path}", fg="green")

    fails = [v for v in violations if v.severity == "fail"]
    warns = [v for v in violations if v.severity == "warn"]
    if fails:
        sys.exit(2)
    if warns:
        sys.exit(1)


@perf.command("smoke")
def perf_smoke() -> None:
    """Run end-to-end smoke tests against all meeting-scribe backends.

    \b
    Tests each backend with a single request:
      ASR (8003)         — audio transcription via /v1/chat/completions
      TTS (8002)         — text-to-speech via /v1/audio/speech
      Diarization (8001) — speaker diarization via /v1/diarize
      Translation (8010) — JA→EN translation via /v1/chat/completions

    Exit codes: 0 = all pass, 2 = any failure.
    """
    import asyncio

    from .perf.smoke import render_smoke_stdout
    from .perf.smoke import run_smoke as _run_smoke

    click.secho("perf smoke — backend health validation", bold=True)
    click.echo()

    results = asyncio.run(_run_smoke())
    render_smoke_stdout(results)

    if any(not r.passed for r in results):
        sys.exit(2)


@cli.group("swarm-demo")
def swarm_demo() -> None:
    """Agent swarm demo targeting meeting-scribe.

    Creates sandbox copies of meeting-scribe, launches Claude Code
    with agent teams on the local LLM. Each run is timestamped.
    """
    pass


@swarm_demo.command("check")
def demo_check() -> None:
    """Check prerequisites and auto-start servers."""
    from .demo import _active_config, run_check

    cfg = _active_config()
    run_check(cfg)


@swarm_demo.command("run")
@click.argument("scenario", type=int, required=False)
def swarm_run(scenario: int | None) -> None:
    """Run a demo scenario (1-6). Shows menu if no scenario given."""
    from .demo import SCENARIOS, _active_config, _ensure_server, launch_scenario, run_check

    cfg = _active_config()

    if scenario is None:
        click.secho("Demo Scenarios", bold=True)
        click.echo()
        for sid, s in SCENARIOS.items():
            click.echo(f"  {sid}) {s['name']}")
            click.echo(f"     {s['desc']}")
            click.echo()
        scenario = click.prompt("Select scenario", type=int)

    if scenario not in SCENARIOS:
        click.secho(f"Invalid scenario: {scenario} (1-{len(SCENARIOS)})", fg="red")
        sys.exit(1)

    if not run_check(cfg):
        sys.exit(1)

    _ensure_server(cfg)
    launch_scenario(cfg, scenario)


@swarm_demo.command("smoke")
def demo_smoke() -> None:
    """Quick smoke test — verify completion, tool calling, streaming."""
    from .demo import _active_config, _ensure_server, _get_model_id, _proxy_url, _vllm_url

    cfg = _active_config()
    _ensure_server(cfg)

    vllm = _vllm_url(cfg)
    proxy = _proxy_url(cfg)
    model_id = _get_model_id(vllm)

    click.secho("Smoke Test", bold=True)
    click.echo()

    # Basic completion
    click.echo("  Basic completion: ", nl=False)
    try:
        resp = httpx.post(
            f"{proxy}/v1/messages",
            json={
                "model": model_id,
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Say OK"}],
            },
            timeout=30,
        )
        assert resp.json()["type"] == "message"
        click.secho("OK", fg="green")
    except Exception as e:
        click.secho(f"FAIL ({e})", fg="red")

    # Tool calling
    click.echo("  Tool calling: ", nl=False)
    try:
        resp = httpx.post(
            f"{proxy}/v1/messages",
            json={
                "model": model_id,
                "max_tokens": 200,
                "tools": [
                    {
                        "name": "Read",
                        "description": "Read a file",
                        "input_schema": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}},
                            "required": ["path"],
                        },
                    }
                ],
                "messages": [{"role": "user", "content": "Read /etc/hostname"}],
            },
            timeout=30,
        )
        assert any(b["type"] == "tool_use" for b in resp.json()["content"])
        click.secho("OK", fg="green")
    except Exception as e:
        click.secho(f"FAIL ({e})", fg="red")

    # Streaming
    click.echo("  Streaming: ", nl=False)
    try:
        with (
            httpx.Client(timeout=15) as client,
            client.stream(
                "POST",
                f"{proxy}/v1/messages",
                json={
                    "model": model_id,
                    "max_tokens": 20,
                    "stream": True,
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            ) as resp,
        ):
            first_line = next(resp.iter_lines())
            assert "message_start" in first_line
        click.secho("OK", fg="green")
    except Exception as e:
        click.secho(f"FAIL ({e})", fg="red")

    click.echo()
    click.secho("Done!", fg="green", bold=True)


@swarm_demo.command("runs")
def demo_runs() -> None:
    """List past demo runs."""
    from .demo import list_runs

    click.secho("Demo Run History", bold=True)
    click.echo()
    list_runs()


@swarm_demo.command("clean")
def demo_clean() -> None:
    """Remove all demo run sandboxes."""
    from .demo import _runs_dir

    runs = _runs_dir()
    if runs.exists():
        import shutil as _shutil

        _shutil.rmtree(runs)
        click.secho("All runs cleaned.", fg="green")
    else:
        click.echo("No runs to clean.")


@cli.group()
def models() -> None:
    """Manage models for the current backend."""
    pass


@models.command("list")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["ollama", "llamacpp", "vllm", "mlx-dflash", "auto"]),
    default="auto",
    help="Backend to check",
)
def models_list(backend: str) -> None:
    """List available models for the backend."""
    backend_type = _resolve_backend(backend)
    b = get_backend(backend_type)

    click.echo(f"Backend: {backend_type.value}")
    click.echo()
    click.echo("Configured models:")
    for key, model_id in b.models.items():
        default = " (default)" if key == b.default_model else ""
        click.echo(f"  {key:12} -> {model_id}{default}")

    # Show loaded models if server is running
    s = b.status()
    loaded_models = s.get("models")
    if isinstance(loaded_models, list) and loaded_models:
        click.echo()
        click.echo("Currently loaded:")
        for m in loaded_models:
            click.echo(f"  - {m}")


@models.command("pull")
@click.argument("model")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["ollama", "llamacpp", "vllm", "mlx-dflash", "auto"]),
    default="auto",
    help="Backend to use",
)
def models_pull(model: str, backend: str) -> None:
    """Pull/download a model.

    \b
    For Ollama: runs 'ollama pull'
    For llama.cpp: downloads via 'huggingface-cli download' or auto-downloads on first use
    """
    backend_type = _resolve_backend(backend)
    b = get_backend(backend_type)
    model_id = b.get_model_id(model)

    click.echo(f"Pulling model: {model_id}")
    click.echo(f"Backend: {backend_type.value}")
    click.echo()

    if backend_type == BackendType.OLLAMA:
        subprocess.run(["ollama", "pull", model_id], check=True)
    elif backend_type == BackendType.LLAMACPP:
        if shutil.which("huggingface-cli"):
            # Parse HF repo from model ID (format: repo:quant)
            hf_repo = model_id.split(":")[0] if ":" in model_id else model_id
            subprocess.run(["huggingface-cli", "download", hf_repo], check=True)
        else:
            click.echo("Model will be downloaded automatically on first use via -hf flag.")
            click.echo("To pre-download: pip install huggingface-hub && huggingface-cli download")

    click.secho("Done!", fg="green")


@cli.group()
def mcp() -> None:
    """Manage MCP servers for web research."""
    pass


def _add_mcp_server(name: str, command: list[str], env: dict[str, str] | None = None) -> bool:
    """Add an MCP server via claude mcp add. Returns True on success."""
    cmd = ["claude", "mcp", "add", "-s", "user", name]
    if env:
        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])
    cmd.append("--")
    cmd.extend(command)

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def _remove_mcp_server(name: str) -> bool:
    """Remove an MCP server via claude mcp remove. Returns True on success."""
    result = subprocess.run(["claude", "mcp", "remove", name], capture_output=True, text=True)
    return result.returncode == 0


@mcp.command("setup")
@click.option("--brave-api-key", envvar="BRAVE_API_KEY", default=None, help="Also add Brave Search")
def mcp_setup(brave_api_key: str | None) -> None:
    """Set up local web research MCP servers for Claude Code.

    Installs local-only web fetch and search servers (no API keys needed).
    Uses curl_cffi for browser TLS fingerprint impersonation and DuckDuckGo
    for search. Optionally adds Brave Search if --brave-api-key is provided.
    """
    if not shutil.which("claude"):
        click.secho("ERROR: 'claude' CLI not found", fg="red")
        click.echo("Install: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)

    # Verify local MCP server entry points are installed
    for entry_point in ["autosre-mcp-fetch", "autosre-mcp-search"]:
        if not shutil.which(entry_point):
            click.secho(f"ERROR: '{entry_point}' not found on PATH", fg="red")
            click.echo("Reinstall: pip install -e '.[dev]' or pip install autosre")
            sys.exit(1)

    errors = []

    # Add local web fetch server
    click.echo("Adding local web fetch MCP server...")
    if _add_mcp_server("autosre-fetch", ["autosre-mcp-fetch"]):
        click.secho("  autosre-fetch: added", fg="green")
    else:
        click.secho("  autosre-fetch: failed", fg="red")
        errors.append("autosre-fetch")

    # Add local web search server
    click.echo("Adding local web search MCP server...")
    if _add_mcp_server("autosre-search", ["autosre-mcp-search"]):
        click.secho("  autosre-search: added", fg="green")
    else:
        click.secho("  autosre-search: failed", fg="red")
        errors.append("autosre-search")

    # Optional: add Brave Search alongside local servers
    if brave_api_key:
        if not shutil.which("npx"):
            click.secho("  brave-search: skipped (npx not found)", fg="yellow")
        else:
            click.echo("Adding Brave Search MCP server...")
            if _add_mcp_server(
                "brave-search",
                ["npx", "-y", "@anthropic-ai/mcp-server-brave-search"],
                env={"BRAVE_API_KEY": brave_api_key},
            ):
                click.secho("  brave-search: added", fg="green")
            else:
                click.secho("  brave-search: failed", fg="red")
                errors.append("brave-search")

    # Deny built-in web tools that don't work locally
    from autosre.mcp_servers.settings import (
        deny_builtin_web_tools,
        load_claude_settings,
        save_claude_settings,
    )

    settings = load_claude_settings()
    settings = deny_builtin_web_tools(settings)
    save_claude_settings(settings)
    click.echo("Denied built-in WebFetch/WebSearch (they require Anthropic cloud)")

    if errors:
        click.secho(f"\nFailed to add: {', '.join(errors)}", fg="red")
        sys.exit(1)

    click.echo()
    click.secho("MCP setup complete! Local web research ready.", fg="green")


@mcp.command("status")
def mcp_status() -> None:
    """Check MCP server status."""
    if not shutil.which("claude"):
        click.secho("ERROR: 'claude' CLI not found", fg="red")
        sys.exit(1)

    result = subprocess.run(["claude", "mcp", "list"], capture_output=True, text=True)
    if result.returncode != 0:
        click.echo("Could not list MCP servers")
        return

    output = result.stdout

    # Check each server
    servers = {
        "autosre-fetch": "Local web fetch (curl_cffi + markdownify)",
        "autosre-search": "Local web search (DuckDuckGo)",
        "brave-search": "Brave Search (API key required)",
    }
    for name, desc in servers.items():
        if name in output:
            click.secho(f"  {name}: configured — {desc}", fg="green")
        else:
            click.echo(f"  {name}: not configured")

    # Check if built-in tools are denied
    from autosre.mcp_servers.settings import load_claude_settings

    settings = load_claude_settings()
    permissions = settings.get("permissions", {})
    deny = permissions.get("deny", []) if isinstance(permissions, dict) else []
    if isinstance(deny, list) and "WebSearch" in deny and "WebFetch" in deny:
        click.secho("  Built-in WebFetch/WebSearch: denied (correct for local)", fg="green")
    else:
        click.secho(
            "  Built-in WebFetch/WebSearch: not denied (run 'autosre mcp setup')", fg="yellow"
        )

    if output.strip():
        click.echo()
        click.echo(output)


@mcp.command("remove")
def mcp_remove() -> None:
    """Remove local MCP servers and restore built-in web tools."""
    if not shutil.which("claude"):
        click.secho("ERROR: 'claude' CLI not found", fg="red")
        sys.exit(1)

    for name in ["autosre-fetch", "autosre-search"]:
        if _remove_mcp_server(name):
            click.secho(f"  {name}: removed", fg="green")
        else:
            click.echo(f"  {name}: not found or already removed")

    # Restore built-in web tools
    from autosre.mcp_servers.settings import (
        allow_builtin_web_tools,
        load_claude_settings,
        save_claude_settings,
    )

    settings = load_claude_settings()
    settings = allow_builtin_web_tools(settings)
    save_claude_settings(settings)
    click.echo("Restored built-in WebFetch/WebSearch in Claude Code settings")
    click.secho("\nMCP servers removed.", fg="green")


@cli.group()
def cluster() -> None:
    """Manage k3s cluster on GB10 nodes (optional overlay)."""
    pass


@cluster.command("bootstrap")
def cluster_bootstrap() -> None:
    """Bootstrap k3s cluster with GPU and Network operators."""
    from autosre.backends.vllm_config import VllmConfig
    from autosre.cluster.manager import ClusterManager

    try:
        config = VllmConfig.load()
    except FileNotFoundError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    mgr = ClusterManager(config)
    if not mgr.bootstrap():
        sys.exit(1)


@cluster.command("teardown")
@click.confirmation_option(prompt="Tear down the k3s cluster?")
def cluster_teardown() -> None:
    """Tear down the k3s cluster from all nodes."""
    from autosre.backends.vllm_config import VllmConfig
    from autosre.cluster.manager import ClusterManager

    try:
        config = VllmConfig.load()
    except FileNotFoundError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    mgr = ClusterManager(config)
    mgr.teardown()


@cluster.command("status")
def cluster_status_cmd() -> None:
    """Show k3s cluster status."""
    from autosre.backends.vllm_config import VllmConfig
    from autosre.cluster.manager import ClusterManager

    try:
        config = VllmConfig.load()
    except FileNotFoundError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    mgr = ClusterManager(config)
    s = mgr.status()

    status_str = (
        click.style("READY", fg="green") if s.cluster_ready else click.style("NOT READY", fg="red")
    )
    click.echo(f"Cluster:         {status_str}")
    click.echo(f"k3s server:      {'running' if s.k3s_server_running else 'stopped'}")
    click.echo(f"GPU Operator:    {'ready' if s.gpu_operator_ready else 'not deployed'}")
    click.echo(f"Network Operator: {'ready' if s.network_operator_ready else 'not deployed'}")

    if s.nodes:
        click.echo(f"\nNodes ({len(s.nodes)}):")
        for n in s.nodes:
            ready = (
                click.style("Ready", fg="green") if n.ready else click.style("NotReady", fg="red")
            )
            click.echo(f"  {n.hostname:15} {n.ip:15} {ready} {','.join(n.roles)}")

    if s.error:
        click.secho(f"\nError: {s.error}", fg="red")


@cluster.command("validate")
def cluster_validate() -> None:
    """Run GPU, NCCL, and RDMA validation tests."""
    from autosre.backends.vllm_config import VllmConfig
    from autosre.cluster.manager import ClusterManager

    try:
        config = VllmConfig.load()
    except FileNotFoundError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    mgr = ClusterManager(config)

    click.secho("GPU Validation:", bold=True)
    gpus = mgr.validate_gpus()
    for node_name, count in gpus.items():
        click.echo(f"  {node_name}: {count} GPU(s)")

    click.echo()
    click.secho("NCCL Validation:", bold=True)
    mgr.validate_nccl()

    click.echo()
    click.secho("RDMA Validation:", bold=True)
    mgr.validate_rdma()


@cli.group()
def demo() -> None:
    """Manage enterprise demo scenarios."""
    pass


@demo.command("list")
def demo_list() -> None:
    """List available demo scenarios."""
    from autosre.demos.scenario import DemoScenario, list_scenarios

    scenarios = list_scenarios()
    if not scenarios:
        click.echo("No scenarios found.")
        return

    click.echo("Available demo scenarios:")
    click.echo()
    for name in scenarios:
        try:
            s = DemoScenario.load(name)
            cluster = " [cluster]" if s.cluster_required else ""
            click.echo(f"  {name:25} {s.total_minutes:3}min  {s.model:20} {s.audience}{cluster}")
        except Exception:
            click.echo(f"  {name:25} (error loading)")


@demo.command("run")
@click.argument("scenario")
@click.option(
    "--audience",
    "-a",
    type=click.Choice(["cxo", "engineering", "finance", "hr", "marketing", "product"]),
    help="Override audience profile",
)
@click.option("--skip-preflight", is_flag=True, help="Skip preflight checks")
def demo_run(scenario: str, audience: str | None, skip_preflight: bool) -> None:
    """Run a demo scenario.

    \b
    Examples:
      autosre demo run enterprise-overview
      autosre demo run deep-tech --audience engineering
      autosre demo run quick-impact --skip-preflight
    """
    from autosre.demos.audience import AUDIENCE_PROFILES
    from autosre.demos.runner import DemoRunner
    from autosre.demos.scenario import DemoScenario

    try:
        s = DemoScenario.load(scenario)
    except FileNotFoundError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    profile = None
    if audience:
        profile = AUDIENCE_PROFILES.get(audience)
    elif s.audience in AUDIENCE_PROFILES:
        profile = AUDIENCE_PROFILES[s.audience]

    runner = DemoRunner(s, audience=profile)
    if not runner.run(skip_preflight=skip_preflight):
        sys.exit(1)


@demo.command("preflight")
@click.argument("scenario")
def demo_preflight(scenario: str) -> None:
    """Run preflight checks for a demo scenario."""
    from autosre.demos.runner import DemoRunner
    from autosre.demos.scenario import DemoScenario

    try:
        s = DemoScenario.load(scenario)
    except FileNotFoundError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    runner = DemoRunner(s)
    ok, issues = runner.preflight()

    if ok:
        click.secho("All preflight checks passed!", fg="green")
    else:
        click.secho("Preflight FAILED:", fg="red")
        for issue in issues:
            click.echo(f"  - {issue}")
        sys.exit(1)


@cli.group()
def swarm() -> None:
    """Manage agent swarm (Claude Code agent teams)."""
    pass


@swarm.command("launch")
@click.option("--agents", "-n", type=int, default=3, help="Number of agents (from template)")
@click.option(
    "--template",
    "-t",
    type=click.Choice(
        [
            "code-review",
            "architecture-analysis",
            "incident-response",
            "content-generation",
            "data-analysis",
        ]
    ),
    help="Task template for the swarm",
)
@click.option("--model", "-m", help="Override model (default from active server)")
@click.option(
    "--provider",
    type=click.Choice(["local", "anthropic"]),
    default="local",
    help="Which provider to route through: local vLLM (default) or online Anthropic API",
)
@click.option(
    "--anthropic-model",
    default="claude-opus-4-6[1m]",
    help="Model id for --provider=anthropic (default claude-opus-4-6[1m])",
)
def swarm_launch(
    agents: int,  # noqa: ARG001
    template: str | None,
    model: str | None,
    provider: str,
    anthropic_model: str,
) -> None:
    """Launch an agent swarm with optional task template.

    \b
    Examples:
      autosre swarm launch                                   # Local basic swarm
      autosre swarm launch -t code-review                    # Local, code review template
      autosre swarm launch -t incident-response -m nemotron-super
      autosre swarm launch --provider=anthropic -t code-review
      autosre swarm launch --provider=anthropic --anthropic-model=claude-opus-4-6
    """
    from autosre.backends.base import load_active_state
    from autosre.swarm.launcher import SwarmLauncher
    from autosre.swarm.templates import TASK_TEMPLATES

    task_template = TASK_TEMPLATES.get(template) if template else None

    if provider == "anthropic":
        # Online mode uses Claude Code's native auth. We still need a
        # backend instance for the launcher API, but its env/model args
        # are ignored once the provider is anthropic. Use whichever
        # backend is active; fall back to ollama as a pure stub if none.
        active = load_active_state()
        backend_type = BackendType(str(active["backend"])) if active else BackendType("ollama")
        b = get_backend(backend_type, active_state=active)
        launcher = SwarmLauncher(
            b,
            template=task_template,
            provider="anthropic",
            anthropic_model=anthropic_model,
        )
        try:
            launcher.launch()
        except RuntimeError as e:
            click.secho(f"ERROR: {e}", fg="red")
            sys.exit(1)
        return

    active = load_active_state()
    if not active:
        click.secho("ERROR: No server running. Run 'autosre start' first.", fg="red")
        sys.exit(1)

    backend_type = BackendType(str(active["backend"]))
    b = get_backend(backend_type, active_state=active)

    launcher = SwarmLauncher(b, template=task_template, provider="local")
    try:
        launcher.launch(model_key=model)
    except RuntimeError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)


@swarm.command("templates")
def swarm_templates() -> None:
    """List available task templates for agent swarms."""
    from autosre.swarm.templates import TASK_TEMPLATES

    click.echo("Available swarm templates:")
    click.echo()
    for name, tmpl in TASK_TEMPLATES.items():
        click.echo(f"  {name:25} {tmpl.num_agents} agents  {tmpl.description}")
        for i, role in enumerate(tmpl.agent_roles):
            click.echo(f"    Agent {i + 1}: {role}")
        click.echo()


# ── eval command group ─────────────────────────────────────────────


@cli.group("eval")
def eval_group() -> None:
    """Run eval suites, list past runs, compare runs bidirectionally."""
    pass


@eval_group.command("suites")
@click.option("--show", default=None, help="Show one suite in detail")
def eval_suites(show: str | None) -> None:
    """List available eval suites, or show one in detail."""
    from autosre.eval.suite import load_all_suites

    suites = load_all_suites()
    if show:
        suite = suites.get(show)
        if suite is None:
            click.secho(f"ERROR: unknown suite: {show}", fg="red")
            sys.exit(1)
        click.echo(f"{suite.name}: {suite.description}")
        click.echo(f"  category: {suite.category}")
        click.echo(f"  agents:   {suite.num_agents}")
        for i, role in enumerate(suite.agent_roles):
            click.echo(f"    Agent {i + 1}: {role}")
        return
    click.echo("Available eval suites:")
    click.echo()
    for name, suite in suites.items():
        click.echo(f"  {name:16} {suite.num_agents} agents  {suite.description}")


@eval_group.command("run")
@click.option(
    "--provider",
    type=click.Choice(["local", "anthropic"]),
    required=True,
    help="Which provider to route through for this run",
)
@click.option(
    "--suite",
    "suites",
    multiple=True,
    required=True,
    help="Suite name(s); repeat --suite or use a comma-separated list",
)
@click.option(
    "--target",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=Path(),
    help="Target repository path (default: cwd)",
)
@click.option("--run-id", default=None, help="Short tag appended to the run directory name")
@click.option(
    "--anthropic-model",
    default="claude-opus-4-6[1m]",
    help="Model id when provider=anthropic",
)
@click.option(
    "--allow-dirty",
    is_flag=True,
    help="Allow snapshotting an unclean tree via a full file-copy",
)
@click.option(
    "--keep-worktrees",
    is_flag=True,
    help="Do not remove per-suite worktrees after the run (debugging)",
)
def eval_run(
    provider: str,
    suites: tuple[str, ...],
    target: Path,
    run_id: str | None,
    anthropic_model: str,
    allow_dirty: bool,
    keep_worktrees: bool,
) -> None:
    """Capture a single-provider eval run.

    \b
    Examples:
      autosre eval run --provider=local --suite=security --target=.
      autosre eval run --provider=anthropic --suite=security,a11y,i18n --target=.
    """
    from autosre.eval.report import write_report_md
    from autosre.eval.runner import EvalRunner
    from autosre.swarm.launcher import SwarmLauncher

    suite_list = _split_suites(suites)

    def factory(prov, _suite):  # type: ignore[no-untyped-def]
        if prov == "anthropic":
            from autosre.backends import BackendType, get_backend

            b = get_backend(BackendType("ollama"))
            return SwarmLauncher(
                b,
                provider="anthropic",
                anthropic_model=anthropic_model,
            )
        from autosre.backends import BackendType, get_backend
        from autosre.backends.base import load_active_state

        active = load_active_state()
        if not active:
            click.secho(
                "ERROR: provider=local requires 'autosre start' first.",
                fg="red",
            )
            sys.exit(1)
        b = get_backend(
            BackendType(str(active["backend"])),
            active_state=active,
        )
        return SwarmLauncher(b, provider="local")

    runner = EvalRunner(launcher_factory=factory)
    try:
        result = runner.run(
            provider=provider,  # type: ignore[arg-type]
            suites=suite_list,
            target=target.resolve(),
            run_id=run_id,
            anthropic_model=anthropic_model,
            allow_dirty=allow_dirty,
            keep_worktrees=keep_worktrees,
        )
    except ValueError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    report_path = write_report_md(result)
    click.secho(f"Run complete: {result.run_dir}", fg="green", bold=True)
    click.echo(f"  Findings: {sum(len(s.findings) for s in result.suites)}")
    click.echo(f"  Report:   {report_path}")


@eval_group.command("list")
def eval_list() -> None:
    """List past eval runs under ~/.local/share/autosre/eval-runs/."""
    from autosre.eval.runner import EVAL_RUNS_ROOT

    root = EVAL_RUNS_ROOT
    if not root.exists():
        click.echo("(no runs yet)")
        return
    runs = sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").exists()],
        reverse=True,
    )
    if not runs:
        click.echo("(no runs yet)")
        return
    for run in runs:
        import json as _json

        data = _json.loads((run / "manifest.json").read_text())
        click.echo(
            f"  {run.name:60} provider={data.get('provider'):9} "
            f"suites={len(data.get('suites') or [])}"
        )


@eval_group.command("show")
@click.argument("run")
def eval_show(run: str) -> None:
    """Print the report.md for a run (by id or directory path)."""
    from autosre.eval.report import load_report_md
    from autosre.eval.runner import EVAL_RUNS_ROOT

    run_dir = _resolve_run_dir(run, EVAL_RUNS_ROOT)
    click.echo(load_report_md(run_dir))


@eval_group.command("compare")
@click.argument("run_a")
@click.argument("run_b")
@click.option("--suite", "suite_filter", multiple=True, help="Limit to these suites")
@click.option("--allow-sha-mismatch", is_flag=True)
@click.option("--allow-suite-mismatch", is_flag=True)
@click.option("--allow-same-provider", is_flag=True)
@click.option("--include-failed-suites", is_flag=True)
@click.option("--no-judge", is_flag=True, help="Skip the Opus judge tier")
def eval_compare(
    run_a: str,
    run_b: str,
    suite_filter: tuple[str, ...],
    allow_sha_mismatch: bool,
    allow_suite_mismatch: bool,
    allow_same_provider: bool,
    include_failed_suites: bool,
    no_judge: bool,
) -> None:
    """Compare two independent eval runs bidirectionally."""
    import time

    from autosre.eval.differ import (
        CompareOverrides,
        CompareRefusedError,
        compare,
        write_compare_dir,
    )
    from autosre.eval.judge import Judge
    from autosre.eval.report import write_compare_md
    from autosre.eval.runner import EVAL_RUNS_ROOT

    a_dir = _resolve_run_dir(run_a, EVAL_RUNS_ROOT)
    b_dir = _resolve_run_dir(run_b, EVAL_RUNS_ROOT)

    overrides = CompareOverrides(
        allow_sha_mismatch=allow_sha_mismatch,
        allow_suite_mismatch=allow_suite_mismatch,
        allow_same_provider=allow_same_provider,
        include_failed_suites=include_failed_suites,
    )

    judge = None if no_judge else Judge()

    try:
        result = compare(
            a_dir,
            b_dir,
            overrides=overrides,
            suite_filter=list(suite_filter) if suite_filter else None,
            judge=judge,
        )
    except CompareRefusedError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(2)

    ts = time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime())
    out_dir = EVAL_RUNS_ROOT / "compares" / f"{ts}-{a_dir.name}-vs-{b_dir.name}"
    index_path = EVAL_RUNS_ROOT / "compares.jsonl"
    write_compare_dir(result, out_dir, index_path=index_path)
    write_compare_md(result, out_dir)
    click.secho(f"Compare written: {out_dir}", fg="green", bold=True)
    click.echo(f"  compare.md:   {out_dir / 'compare.md'}")
    click.echo(f"  compare.json: {out_dir / 'compare.json'}")


def _split_suites(suites: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for entry in suites:
        for part in entry.split(","):
            stripped = part.strip()
            if stripped:
                out.append(stripped)
    return out


def _resolve_run_dir(run: str, root: Path) -> Path:
    """Accept either a full path or a bare run id under ``root``."""
    p = Path(run)
    if p.is_absolute() and p.exists():
        return p
    candidate = root / run
    if candidate.exists():
        return candidate
    # Fuzzy: any dir ending with run
    if root.exists():
        for child in root.iterdir():
            if child.name.endswith(run) and (child / "manifest.json").exists():
                return child
    click.secho(f"ERROR: run not found: {run}", fg="red")
    sys.exit(1)


@cli.group()
def provision() -> None:
    """Provision and manage GB10 node lifecycle."""
    pass


@provision.command("setup")
@click.argument("node_ip")
@click.option("--ssh-user", default="root", help="SSH username")
@click.option("--hf-token", envvar="HF_TOKEN", help="HuggingFace API token for gated models")
def provision_setup(node_ip: str, ssh_user: str, hf_token: str | None) -> None:
    """Provision a GB10 node from vanilla DGX OS to production-ready.

    Idempotent — safe to re-run on an already provisioned node.
    """
    from autosre.infra.types import GB10Node
    from autosre.provision.provisioner import Provisioner

    node = GB10Node(hostname="gb10", ip=node_ip, ssh_user=ssh_user)
    p = Provisioner(node, hf_token=hf_token)
    if not p.provision():
        sys.exit(1)


@provision.command("validate")
@click.argument("node_ip")
@click.option("--ssh-user", default="root", help="SSH username")
def provision_validate(node_ip: str, ssh_user: str) -> None:
    """Validate a GB10 node is properly provisioned."""
    from autosre.infra.types import GB10Node
    from autosre.provision.provisioner import Provisioner

    node = GB10Node(hostname="gb10", ip=node_ip, ssh_user=ssh_user)
    p = Provisioner(node)

    ok, issues = p.validate()
    if ok:
        click.secho("All checks passed!", fg="green")
    else:
        click.secho("Validation FAILED:", fg="red")
        for issue in issues:
            click.echo(f"  - {issue}")
        sys.exit(1)


@provision.command("backup")
@click.argument("node_ip")
@click.option("--ssh-user", default="root", help="SSH username")
def provision_backup(node_ip: str, ssh_user: str) -> None:
    """Backup critical state on a GB10 node before wipe."""
    from autosre.infra.types import GB10Node
    from autosre.provision.provisioner import Provisioner

    node = GB10Node(hostname="gb10", ip=node_ip, ssh_user=ssh_user)
    p = Provisioner(node)
    p.pre_wipe_backup()
    p.save_docker_images()
    click.secho("Backup complete.", fg="green")


@provision.command("restore")
@click.argument("node_ip")
@click.option("--ssh-user", default="root", help="SSH username")
def provision_restore(node_ip: str, ssh_user: str) -> None:
    """Restore state and re-provision a GB10 node after wipe."""
    from autosre.infra.types import GB10Node
    from autosre.provision.provisioner import Provisioner

    node = GB10Node(hostname="gb10", ip=node_ip, ssh_user=ssh_user)
    p = Provisioner(node)
    if not p.post_wipe_restore():
        click.secho("Restore failed.", fg="red")
        sys.exit(1)
    p.load_docker_images()
    click.secho("Restore and re-provision complete.", fg="green")


@provision.command("sync-models")
@click.argument("source_ip")
@click.argument("dest_ip")
@click.option("--ssh-user", default="root", help="SSH username")
def provision_sync_models(source_ip: str, dest_ip: str, ssh_user: str) -> None:
    """Copy HuggingFace models between GB10 nodes."""
    from autosre.infra.types import GB10Node
    from autosre.provision.lifecycle import NodeLifecycle

    source = GB10Node(hostname="source", ip=source_ip, ssh_user=ssh_user)
    dest = GB10Node(hostname="dest", ip=dest_ip, ssh_user=ssh_user)
    lifecycle = NodeLifecycle([source, dest])
    lifecycle.sync_models(source, dest)


@provision.command("sync-images")
@click.argument("source_ip")
@click.argument("dest_ip")
@click.option("--ssh-user", default="root", help="SSH username")
def provision_sync_images(source_ip: str, dest_ip: str, ssh_user: str) -> None:
    """Copy saved Docker images between GB10 nodes."""
    from autosre.infra.types import GB10Node
    from autosre.provision.lifecycle import NodeLifecycle

    source = GB10Node(hostname="source", ip=source_ip, ssh_user=ssh_user)
    dest = GB10Node(hostname="dest", ip=dest_ip, ssh_user=ssh_user)
    lifecycle = NodeLifecycle([source, dest])
    lifecycle.sync_docker_images(source, dest)


@provision.command("rolling-rebuild")
def provision_rolling_rebuild() -> None:
    """Guided rolling rebuild of all configured GB10 nodes.

    Rebuilds one node at a time. Cluster models degrade to a solo
    model during each rebuild window.
    """
    from autosre.backends.vllm_config import VllmConfig
    from autosre.provision.lifecycle import NodeLifecycle

    try:
        config = VllmConfig.load()
    except FileNotFoundError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    lifecycle = NodeLifecycle(config.nodes)
    if not lifecycle.rolling_rebuild():
        sys.exit(1)


@provision.command("wipe-all")
def provision_wipe_all() -> None:
    """Full wipe and rebuild of all GB10 nodes (destructive)."""
    from autosre.backends.vllm_config import VllmConfig
    from autosre.provision.lifecycle import NodeLifecycle

    try:
        config = VllmConfig.load()
    except FileNotFoundError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    lifecycle = NodeLifecycle(config.nodes)
    if not lifecycle.full_wipe_both():
        sys.exit(1)


@cli.group("hooks")
def hooks() -> None:
    """Install / uninstall / inspect autosre Claude Code hooks.

    Manages the autosre hook block in ``~/.claude/settings.json`` so
    that bare ``claude`` (not just ``autosre claude``) fires the plan
    review loop, bash guard, and session checks. Used for "online" mode:
    Claude plans, codex gpt-5.4 reviews, Claude addresses findings.
    """
    pass


@hooks.command("install", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--settings",
    "settings_path",
    type=click.Path(),
    default=None,
    help="Override the settings.json path (default: ~/.claude/settings.json).",
)
def hooks_install(settings_path: str | None) -> None:
    """Install autosre hook entries into ~/.claude/settings.json.

    Idempotent: running twice is a no-op. Existing user hook entries are
    preserved; we only append to the event lists under ``hooks``. Hook
    commands invoke ``autosre hooks run <module>`` and rely on
    ``autosre`` being on ``$PATH``.
    """
    from autosre.hooks_installer import install

    path = Path(settings_path) if settings_path else None
    result = install(settings_path=path)

    click.secho(f"Installed autosre hooks into: {result['settings_path']}", fg="green")
    click.echo(f"Sidecar: {result['sidecar_path']}")
    added = len(result["added"])
    skipped = len(result["skipped"])
    click.echo(f"Added {added} entries, skipped {skipped} (already present).")
    for entry in result["added"]:
        matcher = entry.get("matcher") or "*"
        click.echo(f"  [+] {entry['event']}[{matcher}] -> {entry['command']}")


@hooks.command("uninstall", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--settings",
    "settings_path",
    type=click.Path(),
    default=None,
    help="Override the settings.json path (default: ~/.claude/settings.json).",
)
def hooks_uninstall(settings_path: str | None) -> None:
    """Remove autosre hook entries from ~/.claude/settings.json.

    Uses the sidecar state file to remove exactly the entries autosre
    installed. Entries the user added by hand are preserved.
    """
    from autosre.hooks_installer import uninstall

    path = Path(settings_path) if settings_path else None
    result = uninstall(settings_path=path)

    if "note" in result:
        click.secho(result["note"], fg="yellow")
        return

    click.secho(f"Uninstalled from: {result['settings_path']}", fg="green")
    removed = len(result["removed"])
    click.echo(f"Removed {removed} entries.")
    for entry in result["removed"]:
        matcher = entry.get("matcher") or "*"
        click.echo(f"  [-] {entry['event']}[{matcher}] -> {entry['command']}")


@hooks.command("status", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--settings",
    "settings_path",
    type=click.Path(),
    default=None,
    help="Override the settings.json path.",
)
def hooks_status(settings_path: str | None) -> None:
    """Show autosre hook installation state.

    Reports whether hooks are installed, the python interpreter used,
    and any drift (entries the sidecar claims we installed but that no
    longer appear in settings.json -- meaning the user edited it by hand).
    """
    from autosre.hooks_installer import status

    path = Path(settings_path) if settings_path else None
    result = status(settings_path=path)

    click.echo(f"Settings:  {result['settings_path']}")
    click.echo(f"Sidecar:   {result['sidecar_path']}")
    if not result["installed"]:
        click.secho("Status:    not installed", fg="yellow")
        click.echo("Run: autosre hooks install")
        return

    click.secho("Status:    installed", fg="green")
    click.echo(f"Entries:   {len(result['entries'])}")
    for entry in result["entries"]:
        matcher = entry.get("matcher") or "*"
        click.echo(f"  {entry['event']}[{matcher}] -> {entry['command']}")

    if result["drift"]:
        click.secho(
            f"\nDrift: {len(result['drift'])} entries missing from settings "
            "(user edited by hand?):",
            fg="yellow",
        )
        for entry in result["drift"]:
            matcher = entry.get("matcher") or "*"
            click.echo(f"  [drift] {entry['event']}[{matcher}] -> {entry['command']}")


@hooks.command(
    "run",
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.argument("module")
def hooks_run(module: str) -> None:
    """Invoke a Claude Code hook script by module name.

    Dispatches to ``autosre.claude_hooks.<module>.main()``. Used as the
    command body of installed hook entries so we never embed an absolute
    Python path in settings.json — ``autosre`` resolves from ``$PATH``.
    """
    import importlib

    # Guard against arbitrary module names — only the claude_hooks submodules
    # are legitimate hook scripts.
    allowed = {
        "post_commit_scan_update",
        "posttooluse_audit",
        "precompact_context",
        "pretooluse_bash_guard",
        "pretooluse_plan_review",
        "stop_session_check",
        "subagent_plan_context",
        "telemetry_async",
        "user_prompt_submit_branch_check",
    }
    if module not in allowed:
        click.echo(f"unknown hook module: {module}", err=True)
        raise SystemExit(2)

    mod = importlib.import_module(f"autosre.claude_hooks.{module}")
    rc = mod.main()
    raise SystemExit(int(rc) if rc is not None else 0)


@cli.group("hooks-backend")
def hooks_backend() -> None:
    """Backend for Claude Code hooks (guard, stop-check, init)."""
    pass


@hooks_backend.command("guard", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--event",
    default="",
    help="Hook event name (unused, kept for compatibility).",
)
@click.pass_context
def hooks_backend_guard(ctx: click.Context, event: str) -> None:
    """Evaluate a Bash command against guard rules.

    Reads Claude Code hook JSON from stdin and writes a decision JSON to
    stdout. Exits 0 always; the decision is encoded in the JSON's
    ``hookSpecificOutput.permissionDecision`` field.
    """
    from autosre.hooks_backend.guard import guard_cmd

    ctx.invoke(guard_cmd, event=event)


@hooks_backend.command("stop-check", context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_context
def hooks_backend_stop_check(ctx: click.Context) -> None:
    """Evaluate session-end completion checklist.

    Detects the current repo, its type, and what steps remain
    (commit, push, deploy, test). Output is JSON for the Claude Code
    Stop hook.
    """
    from autosre.hooks_backend.stop_check import stop_check_cmd

    ctx.invoke(stop_check_cmd)


@hooks_backend.command("init", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--force", is_flag=True, help="Overwrite an existing user rules file.")
def hooks_backend_init(force: bool) -> None:
    """Copy the packaged guard-rules.yaml into $XDG_CONFIG_HOME/autosre/.

    Idempotent: leaves an existing user file alone unless ``--force``.
    """
    import shutil as _shutil

    from autosre import paths

    src = Path(__file__).resolve().parent / "hooks_backend" / "config" / "guard-rules.yaml"
    dest = paths.guard_rules_file()

    if dest.exists() and not force:
        click.secho(f"Already installed: {dest}", fg="yellow")
        click.echo("Use --force to overwrite.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    _shutil.copy2(src, dest)
    click.secho(f"Installed: {dest}", fg="green")


@cli.group()
def review() -> None:
    """Run AI code/plan reviews against local or online providers."""
    pass


@review.command("plan", context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("plan_path", type=click.Path(exists=True))
@click.option(
    "--chain",
    "chain_str",
    default=None,
    help="Comma-separated provider chain. Default: AUTOSRE_REVIEW_CHAIN env or codex,gemini,claude.",
)
@click.option(
    "--json-output",
    "json_output",
    is_flag=True,
    help="Output raw JSON (for hook consumption). Exit code 1 if blocking findings.",
)
@click.option(
    "--reset",
    is_flag=True,
    help="Reset iteration counter (start fresh review).",
)
@click.pass_context
def review_plan(
    ctx: click.Context,
    plan_path: str,
    chain_str: str | None,
    json_output: bool,
    reset: bool,
) -> None:
    """Review an implementation plan using an AI provider chain.

    Tracks iterations per plan. After iteration 2, switches to a focused
    re-review prompt that verifies previous findings were addressed.
    P0/P1 findings block; P2-only is advisory.
    """
    from autosre.review.cli_plan import plan_review_cmd

    # Delegate to the command implementation, bypassing the inner click
    # decorator wiring — we just call its underlying callback.
    ctx.invoke(
        plan_review_cmd,
        plan_path=plan_path,
        chain_str=chain_str,
        json_output=json_output,
        reset=reset,
    )


@cli.group()
def keys() -> None:
    """Manage SSH key pairs for operational remote access."""
    pass


@keys.command("generate")
@click.option("--name", "-n", default=None, help="Key name (default: autosre_ed25519)")
@click.option(
    "--type",
    "key_type",
    type=click.Choice(["ed25519", "rsa", "ecdsa"]),
    default="ed25519",
    help="Key algorithm",
)
@click.option("--comment", "-c", default=None, help="Comment embedded in the public key")
@click.option("--force", "-f", is_flag=True, help="Overwrite an existing key with this name")
def keys_generate(name: str | None, key_type: str, comment: str | None, force: bool) -> None:
    """Generate a passphrase-less SSH key pair for autosre operations."""
    from autosre.infra.keys import DEFAULT_KEY_NAME, SSHKeyManager

    mgr = SSHKeyManager()
    resolved_name = name or DEFAULT_KEY_NAME

    existed = mgr.path_for(resolved_name).exists and not force
    try:
        pair = mgr.generate(resolved_name, key_type=key_type, comment=comment, force=force)
    except (RuntimeError, ValueError) as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    if existed:
        click.secho(f"Key already exists: {pair.private_key}", fg="yellow")
    else:
        click.secho(f"Generated key: {pair.private_key}", fg="green")
    click.echo(f"Public key:  {pair.public_key}")
    click.echo()
    click.echo(pair.read_public_key())


@keys.command("list")
def keys_list() -> None:
    """List SSH key pairs in ~/.ssh."""
    from autosre.infra.keys import SSHKeyManager

    mgr = SSHKeyManager()
    pairs = mgr.list_keys()
    if not pairs:
        click.echo(f"No SSH key pairs found in {mgr.key_dir}")
        return
    for pair in pairs:
        click.echo(f"  {pair.name:<24} {pair.private_key}")


@keys.command("show")
@click.argument("name", required=False)
def keys_show(name: str | None) -> None:
    """Print the public key for NAME (default: autosre_ed25519)."""
    from autosre.infra.keys import DEFAULT_KEY_NAME, SSHKeyManager

    mgr = SSHKeyManager()
    pair = mgr.path_for(name or DEFAULT_KEY_NAME)
    if not pair.exists:
        click.secho(f"Key not found: {pair.private_key}", fg="red")
        click.echo("Run: autosre keys generate")
        sys.exit(1)
    click.echo(pair.read_public_key())


@keys.command("copy-command")
@click.argument("target")
@click.option("--name", "-n", default=None, help="Key name (default: autosre_ed25519)")
def keys_copy_command(target: str, name: str | None) -> None:
    """Print the ssh-copy-id command to install a key on TARGET (user@host).

    Autosre does not run ssh-copy-id itself — installing a key requires
    the remote password, which you should enter interactively.
    """
    from autosre.infra.keys import DEFAULT_KEY_NAME, SSHKeyManager

    mgr = SSHKeyManager()
    pair = mgr.path_for(name or DEFAULT_KEY_NAME)
    if not pair.exists:
        click.secho(f"Key not found: {pair.private_key}", fg="red")
        click.echo("Run: autosre keys generate")
        sys.exit(1)
    try:
        cmd = mgr.copy_id_command(target, pair)
    except ValueError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)
    click.echo(" ".join(cmd))


@keys.command("agent-setup")
@click.option("--name", "-n", default=None, help="Key name (default: autosre_ed25519)")
@click.option("--socket", default=None, help="Override ssh-agent socket path")
def keys_agent_setup(name: str | None, socket: str | None) -> None:
    """Load the key into ssh-agent and configure ~/.ssh/config to use it.

    After running this once, ``ssh user@host`` will use the loaded key
    without ``-i`` or per-host config entries.
    """
    from pathlib import Path

    from autosre.infra.keys import (
        DEFAULT_KEY_NAME,
        SSHKeyManager,
        add_to_agent,
        agent_keys,
        detect_agent_socket,
        ensure_config_uses_agent,
    )

    mgr = SSHKeyManager()
    pair = mgr.path_for(name or DEFAULT_KEY_NAME)
    if not pair.exists:
        click.secho(f"Key not found: {pair.private_key}", fg="red")
        click.echo("Run: autosre keys generate")
        sys.exit(1)

    sock_path: Path | None
    if socket:
        sock_path = Path(socket)
        if not sock_path.exists():
            click.secho(f"ERROR: socket does not exist: {sock_path}", fg="red")
            sys.exit(1)
    else:
        sock_path = detect_agent_socket()
        if sock_path is None:
            click.secho(
                "ERROR: no ssh-agent socket found "
                "(checked $XDG_RUNTIME_DIR/gcr/ssh and $SSH_AUTH_SOCK)",
                fg="red",
            )
            sys.exit(1)

    click.echo(f"Using ssh-agent socket: {sock_path}")

    try:
        modified = ensure_config_uses_agent(sock_path)
    except OSError as e:
        click.secho(f"ERROR: failed to update ssh config: {e}", fg="red")
        sys.exit(1)
    if modified:
        click.secho("Updated ~/.ssh/config managed block", fg="green")
    else:
        click.secho("~/.ssh/config managed block already up-to-date", fg="yellow")

    try:
        added = add_to_agent(pair, sock_path)
    except RuntimeError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)
    if added:
        click.secho(f"Added {pair.private_key} to ssh-agent", fg="green")
    else:
        click.secho(f"{pair.private_key} already loaded in ssh-agent", fg="yellow")

    try:
        loaded = agent_keys(sock_path)
    except RuntimeError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    click.echo()
    click.echo("Loaded keys:")
    if not loaded:
        click.echo("  (none)")
    else:
        for line in loaded:
            click.echo(f"  {line}")

    click.echo()
    click.secho("Try: ssh <user>@<host>", fg="green")


@keys.command("agent-status")
def keys_agent_status() -> None:
    """Show ssh-agent socket and loaded keys."""
    from autosre.infra.keys import agent_keys, detect_agent_socket

    sock = detect_agent_socket()
    if sock is None:
        click.secho(
            "No ssh-agent socket found (checked $XDG_RUNTIME_DIR/gcr/ssh and $SSH_AUTH_SOCK)",
            fg="red",
        )
        sys.exit(1)

    click.echo(f"Socket: {sock}")
    try:
        loaded = agent_keys(sock)
    except RuntimeError as e:
        click.secho(f"ERROR: {e}", fg="red")
        sys.exit(1)

    click.echo("Loaded keys:")
    if not loaded:
        click.echo("  (none)")
    else:
        for line in loaded:
            click.echo(f"  {line}")


@cli.group()
def configure() -> None:
    """Configure backend connections."""
    pass


@configure.command("vllm")
@click.option(
    "--node",
    "-n",
    "node_ips",
    multiple=True,
    required=True,
    help="GB10 node IP (first = head, rest = workers). Repeat for multiple nodes.",
)
@click.option("--ssh-user", default="root", help="SSH username for all nodes")
@click.option("--ssh-key", default=None, help="Path to SSH private key")
@click.option("--docker-image", default=None, help="Override Docker image")
@click.option("--hf-cache", default="/data/huggingface", help="HuggingFace cache dir on nodes")
def configure_vllm(
    node_ips: tuple[str, ...],
    ssh_user: str,
    ssh_key: str | None,
    docker_image: str | None,
    hf_cache: str,
) -> None:
    """Configure vLLM backend with GB10 node addresses.

    \b
    First --node is the head, additional --node values are workers.

    \b
    Examples:
      autosre configure vllm --node 192.168.1.101                    # Single node
      autosre configure vllm --node 192.168.1.101 --node 192.168.1.102  # 2-node cluster
    """
    from autosre.backends.vllm_config import VllmConfig
    from autosre.infra.ssh import SSHRunner
    from autosre.infra.types import GB10Node, NodeRole

    nodes = []
    for i, ip in enumerate(node_ips):
        role = NodeRole.HEAD if i == 0 else NodeRole.WORKER
        node = GB10Node(
            hostname=f"gb10-{i + 1}",
            ip=ip,
            ssh_user=ssh_user,
            ssh_key=ssh_key,
            role=role,
        )
        nodes.append(node)

    # Validate SSH connectivity
    click.echo("Validating SSH connectivity...")
    for node in nodes:
        runner = SSHRunner(node)
        role_label = "head" if node.role is NodeRole.HEAD else "worker"
        if runner.is_reachable():
            click.secho(f"  {node.ip} ({role_label}): OK", fg="green")
        else:
            click.secho(f"  {node.ip} ({role_label}): UNREACHABLE", fg="red")
            click.echo("Fix SSH connectivity before proceeding.")
            sys.exit(1)

    # Build config
    config = VllmConfig(nodes=nodes, hf_cache_dir=hf_cache)
    if docker_image:
        config.docker_image = docker_image

    # Save
    config.save()
    click.echo()
    click.secho(f"vLLM config saved to {config.default_path()}", fg="green")
    click.echo(f"  Head:    {config.head_node.ip}")
    for w in config.worker_nodes:
        click.echo(f"  Worker:  {w.ip}")
    click.echo(f"  Image:   {config.docker_image}")
    click.echo(f"  HF cache: {config.hf_cache_dir}")
    click.echo()
    click.echo("Start serving with: autosre start -b vllm")


@cli.group("ssh")
def ssh_group() -> None:
    """Blessed remote-execution wrappers for GB10 / workstation nodes.

    Use these instead of bare ``ssh`` in scripts and Claude Code sessions.
    They route through :class:`autosre.infra.ssh.SSHRunner`, log to the
    XDG state dir, and are exempt from the bare-SSH guard rule.
    """


@ssh_group.command("exec", context_settings={"ignore_unknown_options": True})
@click.argument("target")
@click.argument("remote_cmd", nargs=-1, required=True)
@click.option("--user", default=None, help="SSH user (defaults to TARGET's user@ or current).")
@click.option("--key", default=None, help="Path to an SSH private key.")
@click.option("--timeout", type=int, default=60, show_default=True)
def ssh_exec(
    target: str,
    remote_cmd: tuple[str, ...],
    user: str | None,
    key: str | None,
    timeout: int,
) -> None:
    """Run REMOTE_CMD on TARGET via the autosre SSH wrapper.

    Examples:

        autosre ssh exec <user>@<host> -- docker info
        autosre ssh exec <host> --user <user> -- nvidia-smi
    """
    import sys as _sys

    from autosre.infra.ssh import SSHRunner
    from autosre.infra.types import GB10Node

    if "@" in target:
        ssh_user, host = target.split("@", 1)
    else:
        host = target
        ssh_user = user or ""

    node = GB10Node(
        hostname=host,
        ip=host,
        ssh_user=ssh_user or user or "root",
        ssh_key=key,
    )
    runner = SSHRunner(node)
    result = runner.run(list(remote_cmd), timeout=timeout, check=False)
    if result.stdout:
        _sys.stdout.write(result.stdout)
    if result.stderr:
        _sys.stderr.write(result.stderr)
    raise SystemExit(result.returncode)


# ---------------------------------------------------------------------------
# Dropbox command group
# ---------------------------------------------------------------------------


@cli.group("dropbox")
def dropbox() -> None:
    """Self-hosted file dropbox with a stealth password gate.

    Runs a ``filebrowser`` backend on ``127.0.0.1:<upstream_port>`` behind
    an autosre TLS+HTTP proxy on ``<listen_port>`` (default ``8443``).
    The proxy peek-dispatches TLS vs plain HTTP on the same port and
    guards access with an HMAC-signed cookie; there's a minimal
    password-only login page (no branding, no CSS, no username field).

    Three-step setup::

      autosre dropbox install --config-file ~/.config/autosre/dropbox.toml
      autosre dropbox init    --config-file ~/.config/autosre/dropbox.toml --password-stdin
      autosre dropbox start

    Install is non-destructive (unit files + binary only); init is
    destructive (certs + DB + password + secret) and refuses while the
    service is running.
    """
    pass


def _dropbox_load_config(config_file: str | None) -> "DropboxConfigType":
    from autosre.dropbox.config import DropboxConfig

    path = Path(config_file).expanduser() if config_file else None
    return DropboxConfig.load(config_file=path)


# Type alias for type checker (lazy import pattern — mypy tolerates the string)
from typing import TYPE_CHECKING  # noqa: E402

if TYPE_CHECKING:
    from autosre.dropbox.config import DropboxConfig as DropboxConfigType


@dropbox.command("install", context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--config-file", type=click.Path(), default=None, help="Path to a dropbox.toml config file."
)
@click.option(
    "--data-dir", type=click.Path(), default=None, help="Override dropbox data directory."
)
@click.option("--listen-port", type=int, default=None, help="Override public listen port.")
@click.option("--upstream-port", type=int, default=None, help="Override filebrowser upstream port.")
@click.option(
    "--system",
    "system_mode",
    is_flag=True,
    help="Install into /etc/systemd/system/ (requires sudo).",
)
@click.option(
    "--service-user", default=None, help="(system-mode only) account the service runs as."
)
@click.option(
    "--filebrowser-bin",
    type=click.Path(),
    default=None,
    help="Path to filebrowser binary (default: auto-download).",
)
@click.option(
    "--force",
    is_flag=True,
    help="Overwrite existing unit files that differ from the planned content.",
)
def dropbox_install(
    config_file: str | None,
    data_dir: str | None,
    listen_port: int | None,
    upstream_port: int | None,
    system_mode: bool,
    service_user: str | None,
    filebrowser_bin: str | None,
    force: bool,
) -> None:
    """Write unit files + ensure the filebrowser binary is present.

    Non-destructive: never touches certs, DB, or password files. Use
    ``autosre dropbox init`` for the destructive state bootstrap.
    """
    import os as _os

    from autosre.dropbox import filebrowser as fb_helper
    from autosre.dropbox.installer import (
        InstallMode,
        build_install_context,
        detect_init_system,
        install,
        probe_systemd_system_operable,
        probe_systemd_user_operable,
    )

    # CLI overrides flow through the env vars so DropboxConfig.load picks them up.
    if data_dir:
        _os.environ["AUTOSRE_DROPBOX_DATA_DIR"] = data_dir
    if listen_port is not None:
        _os.environ["AUTOSRE_DROPBOX_LISTEN_PORT"] = str(listen_port)
    if upstream_port is not None:
        _os.environ["AUTOSRE_DROPBOX_UPSTREAM_PORT"] = str(upstream_port)

    config = _dropbox_load_config(config_file)

    init_system = detect_init_system()
    if init_system.value != "systemd":
        click.secho(
            f"unsupported init system ({init_system.value}); run the proxy manually:",
            fg="red",
        )
        click.echo(
            f"  AUTOSRE_DROPBOX_DATA_DIR={config.data_dir} "
            f"AUTOSRE_DROPBOX_LISTEN_PORT={config.listen_port} "
            f"python -m autosre.dropbox.proxy"
        )
        raise SystemExit(1)

    mode = InstallMode.SYSTEM if system_mode else InstallMode.USER
    if mode is InstallMode.USER:
        ok, detail = probe_systemd_user_operable()
        if not ok:
            click.secho(f"systemd --user is not operable: {detail}", fg="red")
            click.echo("try --system (runs units from /etc/systemd/system, requires sudo)")
            raise SystemExit(1)
    else:
        ok, detail = probe_systemd_system_operable()
        if not ok:
            click.secho(f"systemd system manager unreachable: {detail}", fg="red")
            raise SystemExit(1)

    fb_path: Path | None = None
    if filebrowser_bin:
        fb_path = Path(filebrowser_bin).expanduser()
    else:
        fb_path = fb_helper.find_or_download()

    ctx = build_install_context(
        config=config,
        mode=mode,
        service_user=service_user,
        python_bin=None,
        filebrowser_bin=fb_path,
        repo_dir=None,
        config_file=Path(config_file).expanduser() if config_file else None,
    )

    try:
        result = install(ctx, force=force)
    except (RuntimeError, ValueError) as exc:
        click.secho(f"install failed: {exc}", fg="red")
        raise SystemExit(1) from exc

    click.secho(f"Installed dropbox units ({result['mode']} mode).", fg="green")
    click.echo(f"Service user: {result['service_user']}")
    click.echo(f"Target dir:   {result['target_dir']}")
    for path in result["written"]:
        click.echo(f"  [+] wrote    {path}")
    for path in result["skipped"]:
        click.echo(f"  [·] skipped  {path} (unchanged)")
    click.echo(f"Sidecar: {result['sidecar']}")
    click.echo()
    click.echo("Next step:")
    click.echo(
        f"  autosre dropbox init{(' --config-file ' + config_file) if config_file else ''} --password-stdin"
    )


@dropbox.command("init", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--config-file", type=click.Path(), default=None)
@click.option("--password-stdin", "from_stdin", is_flag=True, help="Read password from stdin.")
@click.option(
    "--password-file",
    "from_file",
    type=click.Path(),
    default=None,
    help="Read password from a 0600 file.",
)
@click.option("--cert-source", type=click.Choice(["self-signed", "import"]), default="self-signed")
@click.option(
    "--cert", type=click.Path(), default=None, help="(import only) path to existing cert PEM."
)
@click.option(
    "--key", type=click.Path(), default=None, help="(import only) path to existing key PEM."
)
def dropbox_init(
    config_file: str | None,
    from_stdin: bool,
    from_file: str | None,
    cert_source: str,
    cert: str | None,
    key: str | None,
) -> None:
    """Generate cert, init filebrowser DB, write password + HMAC secret.

    Destructive: writes to the configured ``tls_dir`` and ``config_dir``.
    Refuses while the dropbox service is active — stop it first. Password
    must be supplied via interactive prompt, ``--password-stdin``, or
    ``--password-file`` (no literal flag).
    """
    from autosre.dropbox import filebrowser as fb_helper
    from autosre.dropbox.credentials import PasswordError, resolve_password
    from autosre.dropbox.installer import is_any_service_active
    from autosre.dropbox.state_init import StateInitError, perform_init

    if is_any_service_active():
        click.secho(
            "dropbox service is running; stop it before init (`autosre dropbox stop`)",
            fg="red",
        )
        raise SystemExit(1)

    config = _dropbox_load_config(config_file)

    try:
        password = resolve_password(
            from_stdin=from_stdin,
            from_file=Path(from_file).expanduser() if from_file else None,
        )
    except PasswordError as exc:
        click.secho(f"password input error: {exc}", fg="red")
        raise SystemExit(1) from exc

    fb_path = fb_helper.find_or_download()

    try:
        result = perform_init(
            config,
            admin_password=password,
            filebrowser_bin=fb_path,
            cert_source=cert_source,
            cert=Path(cert).expanduser() if cert else None,
            key=Path(key).expanduser() if key else None,
        )
    except StateInitError as exc:
        click.secho(f"init failed: {exc}", fg="red")
        raise SystemExit(1) from exc

    click.secho("Dropbox state initialized.", fg="green")
    for label, value in result.items():
        click.echo(f"  {label}: {value}")
    click.echo()
    click.echo("Next step: `autosre dropbox start`")


@dropbox.command("start")
def dropbox_start() -> None:
    """Enable + start both dropbox units."""
    from autosre.dropbox.installer import enable_and_start

    try:
        result = enable_and_start()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        click.secho(f"start failed: {exc}", fg="red")
        raise SystemExit(1) from exc
    click.secho(f"Started: {', '.join(result['started'])} ({result['mode']})", fg="green")


@dropbox.command("stop")
def dropbox_stop() -> None:
    """Stop both dropbox units."""
    from autosre.dropbox.installer import systemctl_action

    result = systemctl_action("stop")
    for r in result["results"]:
        colour = "green" if r["returncode"] == 0 else "red"
        click.secho(f"  [{r['returncode']}] {r['unit']}", fg=colour)


@dropbox.command("restart")
def dropbox_restart() -> None:
    """Restart both dropbox units."""
    from autosre.dropbox.installer import systemctl_action

    result = systemctl_action("restart")
    for r in result["results"]:
        colour = "green" if r["returncode"] == 0 else "red"
        click.secho(f"  [{r['returncode']}] {r['unit']}", fg=colour)


@dropbox.command("status")
def dropbox_status() -> None:
    """Show dropbox installation + unit state."""
    from autosre.dropbox.credentials import verify_password_file_mode
    from autosre.dropbox.installer import status

    result = status()
    if not result["installed"]:
        click.secho("not installed", fg="yellow")
        return

    click.echo(f"mode:        {result['mode']}")
    click.echo(f"service user: {result['service_user']}")
    click.echo(f"config file: {result['config_file']}")
    click.echo(f"data dir:    {result['config_data_dir']}")
    click.echo(f"listen port: {result['config_listen_port']}")
    click.echo("units:")
    for unit in result["units"]:
        colour = "green" if unit["status"] == "active" else "yellow"
        click.secho(f"  [{unit['status']}] {unit['unit']}", fg=colour)
    if result["drift"]:
        click.secho("drift:", fg="red")
        for unit in result["drift"]:
            click.echo(f"  {unit['status']}: {unit['path']}")

    # Password file mode sanity check
    config = _dropbox_load_config(None)
    ok, detail = verify_password_file_mode(config.password_file)
    colour = "green" if ok else "red"
    click.secho(f"password file: {detail}", fg=colour)


@dropbox.command("logs", context_settings={"ignore_unknown_options": True})
@click.option("-f", "--follow", is_flag=True, help="Follow log output (journalctl -f).")
@click.option("-n", "--lines", type=int, default=100, help="Show the last N lines.")
def dropbox_logs(follow: bool, lines: int) -> None:
    """Tail journalctl for both dropbox units."""
    from autosre.dropbox.installer import _mode_from_sidecar, _unit_names_from_sidecar

    try:
        mode = _mode_from_sidecar()
        units = _unit_names_from_sidecar()
    except RuntimeError as exc:
        click.secho(str(exc), fg="red")
        raise SystemExit(1) from exc

    cmd = ["journalctl"]
    if mode.value == "user":
        cmd.append("--user")
    else:
        cmd.insert(0, "sudo")
    for unit in units:
        cmd.extend(["-u", unit])
    cmd.extend(["-n", str(lines), "--no-pager"])
    if follow:
        cmd.append("-f")
    subprocess.run(cmd, check=False)


@dropbox.command("passwd")
@click.option("--password-stdin", "from_stdin", is_flag=True)
@click.option("--password-file", "from_file", type=click.Path(), default=None)
def dropbox_passwd(from_stdin: bool, from_file: str | None) -> None:
    """Change the dropbox admin password.

    Refuses while the service is running. Uses the same credential input
    rules as ``init``: interactive prompt, ``--password-stdin``, or
    ``--password-file``.
    """
    from autosre.dropbox import filebrowser as fb_helper
    from autosre.dropbox.credentials import (
        PasswordError,
        resolve_password,
        write_password_file,
    )
    from autosre.dropbox.installer import is_any_service_active

    if is_any_service_active():
        click.secho(
            "dropbox service is running; stop it before changing the password",
            fg="red",
        )
        raise SystemExit(1)

    config = _dropbox_load_config(None)

    try:
        password = resolve_password(
            from_stdin=from_stdin,
            from_file=Path(from_file).expanduser() if from_file else None,
        )
    except PasswordError as exc:
        click.secho(f"password input error: {exc}", fg="red")
        raise SystemExit(1) from exc

    write_password_file(config.password_file, password)

    # Update filebrowser's own user record so its UI auth (if ever used) stays in sync.
    fb_path = fb_helper.find_or_download()
    result = subprocess.run(
        [
            str(fb_path),
            "users",
            "update",
            "admin",
            "--password",
            password,
            "-d",
            str(config.filebrowser_db),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.secho(
            f"filebrowser user update failed (password file rewritten, service stopped): {result.stderr.strip()}",
            fg="yellow",
        )
    else:
        click.secho("password updated.", fg="green")
    click.echo("Start the service again with `autosre dropbox start`.")


@dropbox.command("uninstall")
@click.option("--purge", is_flag=True, help="Also remove the data directory (NOT files_dir).")
def dropbox_uninstall(purge: bool) -> None:
    """Stop + disable + remove unit files. ``--purge`` also removes the data dir."""
    from autosre.dropbox.installer import uninstall

    result = uninstall(purge=purge)
    if "note" in result:
        click.secho(result["note"], fg="yellow")
        return
    for path in result.get("removed", []):
        click.echo(f"  [-] {path}")
    for path in result.get("purged", []):
        click.secho(f"  purged {path}", fg="yellow")
    click.secho("Uninstalled.", fg="green")


@dropbox.group("config")
def dropbox_config() -> None:
    """Inspect dropbox configuration."""
    pass


@dropbox_config.command("show")
@click.option("--config-file", type=click.Path(), default=None)
def dropbox_config_show(config_file: str | None) -> None:
    """Print the resolved dropbox config (never includes password material)."""
    from dataclasses import asdict

    config = _dropbox_load_config(config_file)
    data = asdict(config)
    for key, value in sorted(data.items()):
        click.echo(f"  {key}: {value}")
    click.echo(f"  cert_file (derived): {config.cert_file}")
    click.echo(f"  key_file  (derived): {config.key_file}")
    click.echo(f"  filebrowser_db (derived): {config.filebrowser_db}")


if __name__ == "__main__":
    cli()
