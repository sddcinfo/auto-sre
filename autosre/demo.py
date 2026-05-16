"""Agent swarm demo — Click-based CLI for showcasing local LLM agent teams.

Targets the meeting-scribe codebase (sandbox copy, safe to modify).
All ports, models, and config read dynamically from autosre active state.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import cast

import click
import httpx

from .backends.base import load_active_state


def _data_dir() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "autosre"


def _active_config() -> dict[str, object]:
    """Load active autosre config, with defaults."""
    state = load_active_state() or {}
    return {
        "api_port": int(cast("int", state.get("api_port", 8010))),
        "proxy_port": int(cast("int", state.get("proxy_port", 8011))),
        "model": str(state.get("model", "qwen3.6-fp8")),
        "api_host": str(state.get("api_host", "localhost")),
        "backend": str(state.get("backend", "vllm")),
    }


def _vllm_url(cfg: dict[str, object]) -> str:
    return f"http://{cfg['api_host']}:{cfg['api_port']}"


def _proxy_url(cfg: dict[str, object]) -> str:
    return f"http://localhost:{cfg['proxy_port']}"


def _is_healthy(url: str) -> bool:
    try:
        resp = httpx.get(f"{url}/health", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def _get_model_id(url: str) -> str:
    try:
        resp = httpx.get(f"{url}/v1/models", timeout=5)
        return str(resp.json()["data"][0]["id"])
    except Exception:
        return "unknown"


def _get_model_ctx(url: str) -> str:
    try:
        resp = httpx.get(f"{url}/v1/models", timeout=5)
        ctx = resp.json()["data"][0]["max_model_len"]
        return f"{ctx // 1024}K"
    except Exception:
        return "?"


def _scribe_src() -> Path:
    """Find meeting-scribe source repo.

    Search order:
      1. ``$MEETING_SCRIBE_REPO`` env override.
      2. Sibling ``meeting-scribe`` directory next to auto-sre's checkout.
      3. ``~/meeting-scribe`` — the layout produced by meeting-scribe's
         own bootstrap.sh.
    """
    import os

    auto_sre = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    env = os.environ.get("MEETING_SCRIBE_REPO", "").strip()
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            auto_sre.parent / "meeting-scribe",
            Path.home() / "meeting-scribe",
        ]
    )
    for p in candidates:
        if (p / "src" / "meeting_scribe").is_dir():
            return p
    msg = "meeting-scribe repo not found (tried $MEETING_SCRIBE_REPO, sibling dir, ~/meeting-scribe)"
    raise FileNotFoundError(msg)


def _runs_dir() -> Path:
    auto_sre = Path(__file__).resolve().parent.parent
    return auto_sre / "demo" / "runs"


def _ensure_server(cfg: dict[str, object]) -> bool:
    """Ensure vLLM + proxy are running, auto-starting if needed."""
    vllm = _vllm_url(cfg)
    proxy = _proxy_url(cfg)

    # Check vLLM
    if not _is_healthy(vllm):
        click.echo(f"  Starting full stack ({cfg['model']})...")
        result = subprocess.run(
            ["autosre", "start"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            click.secho(f"  Failed to start vLLM: {result.stderr[:200]}", fg="red")
            return False
        # Wait for vLLM to be fully healthy (model loading can take minutes)
        click.echo("  Waiting for vLLM to be ready...")
        for i in range(300):
            if _is_healthy(vllm):
                break
            if i % 10 == 0 and i > 0:
                click.echo(f"  Still loading... ({i}s)")
            time.sleep(1)
        if not _is_healthy(vllm):
            click.secho("  vLLM failed to become healthy after 5 minutes", fg="red")
            return False
        click.secho("  vLLM started", fg="green")

    # Check proxy
    if not _is_healthy(proxy):
        click.echo("  Starting Anthropic proxy...")
        subprocess.Popen(
            ["python3", "-m", "autosre.backends.anthropic_proxy", str(cfg["proxy_port"]), vllm],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)
        if _is_healthy(proxy):
            click.secho("  Proxy started", fg="green")
        else:
            click.secho("  Proxy will auto-start with 'autosre claude'", fg="yellow")

    return True


def _warmup(cfg: dict[str, object]) -> None:
    """Fire a tiny request to warm up the model."""
    import contextlib

    vllm = _vllm_url(cfg)
    model_id = _get_model_id(vllm)
    with contextlib.suppress(Exception):
        httpx.post(
            f"{vllm}/v1/chat/completions",
            json={
                "model": model_id,
                "max_tokens": 1,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=30,
        )


# --- Sandbox management ---


def create_run() -> Path:
    """Create a fresh timestamped sandbox from meeting-scribe."""
    runs = _runs_dir()
    run_id = time.strftime("%Y%m%d-%H%M%S")
    sandbox = runs / run_id
    sandbox.mkdir(parents=True, exist_ok=True)

    src = _scribe_src()
    git_proc = subprocess.run(
        ["git", "-C", str(src), "archive", "HEAD"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["tar", "-x", "-C", str(sandbox)],
        input=git_proc.stdout,
        check=True,
        capture_output=True,
    )
    # Trust mise config in sandbox (otherwise mise errors on untrusted .mise.toml)
    mise_toml = sandbox / ".mise.toml"
    if mise_toml.exists():
        subprocess.run(["mise", "trust", str(sandbox)], capture_output=True, check=False)
    subprocess.run(["git", "init", "-q"], cwd=sandbox, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "demo"], cwd=sandbox, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "demo@local"], cwd=sandbox, check=True, capture_output=True
    )
    subprocess.run(["git", "add", "-A"], cwd=sandbox, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", f"meeting-scribe sandbox — run {run_id}"],
        cwd=sandbox,
        check=True,
        capture_output=True,
    )

    py_count = len(list(sandbox.rglob("*.py")))
    click.secho(f"Created run {run_id} ({py_count} py files)", fg="green")
    return sandbox


def list_runs() -> None:
    """List past demo runs."""
    runs = _runs_dir()
    if not runs.exists():
        click.echo("  No runs yet.")
        return

    for run_dir in sorted(runs.iterdir(), reverse=True):
        if not (run_dir / ".git").is_dir():
            continue
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        commits = len(result.stdout.strip().splitlines()) if result.stdout.strip() else 0
        click.echo(f"  {run_dir.name}  —  {commits} commits")


# --- Check ---


def run_check(cfg: dict[str, object]) -> bool:
    """Check all prerequisites, auto-starting servers as needed."""
    ok = True
    vllm = _vllm_url(cfg)
    proxy = _proxy_url(cfg)

    click.secho("Infrastructure", bold=True)
    click.echo()

    # vLLM + Proxy — single _ensure_server call handles both
    if not _is_healthy(vllm):
        click.echo(f"  \033[1;33m~\033[0m vLLM not running — starting ({cfg['model']})...")
        if not _ensure_server(cfg):
            ok = False
        else:
            click.secho(f"  ✓ vLLM started on :{cfg['api_port']}", fg="green")

    if _is_healthy(vllm):
        model_id = _get_model_id(vllm)
        ctx = _get_model_ctx(vllm)
        click.echo(f"  \033[0;32m✓\033[0m vLLM: {model_id} ({ctx} context) on :{cfg['api_port']}")

    if _is_healthy(proxy):
        click.echo(
            f"  \033[0;32m✓\033[0m Anthropic proxy: :{cfg['proxy_port']} → :{cfg['api_port']}"
        )
    else:
        click.echo("  \033[1;33m~\033[0m Proxy: will start with 'autosre claude'")

    # Claude CLI
    if shutil.which("claude"):
        click.echo("  \033[0;32m✓\033[0m Claude Code")
    else:
        click.secho("  ✗ Claude Code not installed", fg="red")
        ok = False

    # MCP
    for name, cmd in [("web search", "autosre-mcp-search"), ("web fetch", "autosre-mcp-fetch")]:
        if shutil.which(cmd):
            click.echo(f"  \033[0;32m✓\033[0m MCP {name}")
        else:
            click.secho(f"  ✗ MCP {name} not found", fg="red")
            ok = False

    # Agents
    agents_dir = Path(__file__).resolve().parent.parent / ".claude" / "agents"
    agent_files = list(agents_dir.glob("*.md")) if agents_dir.exists() else []
    if agent_files:
        click.echo(f"  \033[0;32m✓\033[0m Agent definitions: {len(agent_files)}")
        for f in agent_files:
            click.echo(f"      \033[2m{f.stem}\033[0m")
    else:
        click.secho("  ✗ No agent definitions", fg="red")
        ok = False

    # Credential isolation
    click.echo()
    click.secho("Credential Isolation", bold=True)
    click.echo()
    click.echo(f"  \033[0;32m✓\033[0m ANTHROPIC_BASE_URL → localhost:{cfg['proxy_port']}")
    click.echo("  \033[0;32m✓\033[0m ANTHROPIC_API_KEY → purged (local-vllm only)")
    click.echo("  \033[2m  autosre claude --bare strips all cloud credentials\033[0m")

    # Target codebase
    click.echo()
    click.secho("Target Codebase", bold=True)
    click.echo()
    try:
        src = _scribe_src()
        py_count = len(list((src / "src").rglob("*.py")))
        test_count = len(list((src / "tests").rglob("test_*.py")))
        click.echo(f"  \033[0;32m✓\033[0m meeting-scribe: {py_count} source, {test_count} tests")
    except FileNotFoundError:
        click.secho("  ✗ meeting-scribe not found", fg="red")
        ok = False

    # GPU
    click.echo()
    click.secho("GPU Memory", bold=True)
    click.echo()
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=name,used_memory", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.strip().splitlines():
        if line.strip():
            click.echo(f"    {line.strip()}")

    click.echo()
    if ok:
        click.secho("All prerequisites met!", fg="green", bold=True)
    else:
        click.secho("Some prerequisites missing.", fg="red", bold=True)
    return ok


# --- Scenarios ---

SCENARIOS = {
    1: {
        "name": "Parallel Codebase Analysis",
        "desc": "3 agents analyze meeting-scribe simultaneously",
        "prompt": (
            "You MUST use agent teams for this task. Do NOT do the work yourself.\n\n"
            'Step 1: Call TeamCreate with {"team_name": "codebase-analysis", '
            '"description": "Parallel codebase analysis"}\n\n'
            "Step 2: In ONE message, spawn 3 agents with the Agent tool. Each agent call needs:\n"
            '  Agent({"description": "...", "prompt": "...", '
            '"subagent_type": "general-purpose", "name": "...", '
            '"team_name": "codebase-analysis", "run_in_background": true})\n\n'
            "The 3 teammates:\n"
            '1. name="researcher" — map the full architecture of src/meeting_scribe/\n'
            '2. name="reviewer" — audit for security issues and code quality\n'
            "3. name=\"test-runner\" — run 'python -m pytest tests/ -v' and report results\n\n"
            "Step 3: Wait for messages from teammates (they arrive automatically). "
            "Do NOT use sleep. When all finish, synthesize their findings.\n\n"
            'To message a teammate: SendMessage({"to": "researcher", '
            '"summary": "check status", "message": "How is your analysis going?"})'
        ),
    },
    2: {
        "name": "Feature Pipeline: Research → Implement → Test",
        "desc": "Add a new language to meeting-scribe's translation system",
        "prompt": (
            "You MUST use agent teams for this task. Do NOT do the work yourself.\n\n"
            'Step 1: Call TeamCreate with {"team_name": "feature-pipeline", '
            '"description": "Feature pipeline: research, implement, test"}\n\n'
            "Step 2: Spawn 3 agents with the Agent tool (all in ONE message):\n"
            '  Each needs: description, prompt, subagent_type="general-purpose", '
            'name, team_name="feature-pipeline", run_in_background=true\n\n'
            "The 3 teammates:\n"
            '1. name="researcher" — study src/meeting_scribe/backends/translate_vllm.py '
            "and src/meeting_scribe/languages.py. Report how languages are registered.\n"
            '2. name="implementer" — add Korean (ko) as a new language.\n'
            "3. name=\"tester\" — run 'python -m pytest tests/test_languages.py -v'\n\n"
            "Pipeline: researcher first, then tell implementer to start, then tester.\n"
            'Use SendMessage({"to": "implementer", "summary": "start implementing", '
            '"message": "Research is done. Add Korean support now."}) to coordinate.'
        ),
    },
    3: {
        "name": "Multi-Module Refactor (worktree isolation)",
        "desc": "2 agents refactor different modules in parallel worktrees",
        "prompt": (
            "You MUST use agent teams for this task. Do NOT do the work yourself.\n\n"
            'Step 1: Call TeamCreate with {"team_name": "parallel-refactor", '
            '"description": "Parallel module refactoring"}\n\n'
            "Step 2: Spawn 2 agents with the Agent tool (both in ONE message):\n"
            '  Each needs: description, prompt, subagent_type="general-purpose", '
            'name, team_name="parallel-refactor", run_in_background=true, '
            'isolation="worktree"\n\n'
            "The 2 teammates:\n"
            '1. name="asr-refactor" — refactor src/meeting_scribe/backends/asr_vllm.py '
            "to add structured error handling and retry logic for vLLM failures.\n"
            '2. name="translate-refactor" — refactor '
            "src/meeting_scribe/backends/translate_vllm.py to add timeout configuration.\n\n"
            "Both work simultaneously. Wait for their messages (arrive automatically)."
        ),
    },
    4: {
        "name": "Web Research → Plan → Implement",
        "desc": "Research latest vLLM optimizations online, then apply them",
        "prompt": (
            "You MUST use agent teams for this task. Do NOT do the work yourself.\n\n"
            'Step 1: Call TeamCreate with {"team_name": "vllm-upgrade", '
            '"description": "vLLM optimization research and implementation"}\n\n'
            "Step 2: Spawn 3 agents with the Agent tool (all in ONE message):\n"
            '  Each needs: description, prompt, subagent_type="general-purpose", '
            'name, team_name="vllm-upgrade", run_in_background=true\n\n'
            "The 3 teammates:\n"
            "1. name=\"researcher\" — search the web for 'vLLM performance tuning 2025' "
            "and 'NVIDIA GB10 Grace Blackwell vLLM optimization'. Fetch pages and extract "
            "actionable recommendations.\n"
            '2. name="planner" — synthesize research into concrete changes.\n'
            '3. name="implementer" — apply changes to docker-compose.gb10.yml.\n\n'
            "Pipeline: researcher first, then tell planner, then implementer.\n"
            'Use SendMessage({"to": "planner", "summary": "research done", '
            '"message": "Here are the findings..."}) to coordinate.'
        ),
    },
    5: {
        "name": "Full System Audit",
        "desc": "4 agents: architecture, dependencies, security, performance",
        "prompt": (
            "You MUST use agent teams for this task. Do NOT do the work yourself.\n\n"
            'Step 1: Call TeamCreate with {"team_name": "system-audit", '
            '"description": "Full system audit"}\n\n'
            "Step 2: Spawn 4 agents with the Agent tool (all in ONE message):\n"
            '  Each needs: description, prompt, subagent_type="general-purpose", '
            'name, team_name="system-audit", run_in_background=true\n\n'
            "The 4 teammates:\n"
            '1. name="architect" — document the full system design and data flow\n'
            '2. name="dep-auditor" — check pyproject.toml deps for known issues\n'
            '3. name="security" — review endpoints, input validation, docker security\n'
            '4. name="perf" — analyze GPU memory budget in docker-compose.gb10.yml\n\n'
            "All 4 work in parallel. Wait for their messages (arrive automatically). "
            "Synthesize findings when done."
        ),
    },
    6: {
        "name": "Hardcoded Values Scan",
        "desc": "Find and fix hardcoded ports, URLs, paths, and magic numbers",
        "prompt": (
            "You MUST use agent teams for this task. Do NOT do the work yourself.\n\n"
            'Step 1: Call TeamCreate with {"team_name": "hardcode-scan", '
            '"description": "Scan and fix hardcoded values"}\n\n'
            "Step 2: Spawn 3 agents with the Agent tool (all in ONE message):\n"
            '  Each needs: description, prompt, subagent_type="general-purpose", '
            'name, team_name="hardcode-scan", run_in_background=true\n\n'
            "The 3 teammates:\n"
            '1. name="scanner" — scan all Python and config files for hardcoded ports, '
            "IPs, URLs, paths, magic numbers, model names. Report file:line.\n"
            '2. name="fixer" — extract worst offenders into config.py or env vars.\n'
            "3. name=\"tester\" — run 'python -m pytest tests/ -v' to verify nothing broke.\n\n"
            "Pipeline: scanner first, then tell fixer, then tester.\n"
            'Use SendMessage({"to": "fixer", "summary": "scan complete", '
            '"message": "Start fixing these..."}) to coordinate.'
        ),
    },
}


def launch_scenario(cfg: dict[str, object], scenario_id: int) -> None:
    """Launch a demo scenario."""
    scenario = SCENARIOS[scenario_id]

    click.echo()
    click.secho(f"Scenario {scenario_id}: {scenario['name']}", bold=True)
    click.echo(f"  {scenario['desc']}")
    click.echo()

    # Create sandbox
    sandbox = create_run()

    # Warmup
    click.echo("  Warming up model...")
    _warmup(cfg)
    click.secho("  Ready", fg="green")

    click.echo()
    click.echo(f"  \033[2mRun: {sandbox.name}  |  Model: local  |  No cloud credentials\033[0m")
    click.echo()

    # Launch from the sandbox itself — no path confusion
    # The sandbox is a git repo so Claude Code treats it as a project
    os.chdir(sandbox)
    os.execvp("autosre", ["autosre", "claude", scenario["prompt"]])
