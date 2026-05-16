"""Interactive TUI for autosre — Rich-based terminal interface."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from .backends.base import load_active_state

if TYPE_CHECKING:
    from collections.abc import Callable

console = Console()


def _health(url: str) -> bool:
    try:
        return httpx.get(url, timeout=1, verify=False).status_code == 200  # noqa: S501
    except Exception:
        return False


def _parse_container_logs(container: str, tail: int = 50) -> str:
    """Get recent docker logs for a container. Returns empty string on failure."""
    try:
        r = subprocess.run(
            ["docker", "logs", f"--tail={tail}", container],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        return (r.stderr + r.stdout) if r.returncode == 0 else ""
    except Exception:
        return ""


def _get_loading_progress(container: str = "autosre-vllm-local") -> str | None:
    """Parse docker logs to get vLLM loading progress with stage tracking.

    Startup stages (typical ~3 min for INT4 35B):
      1. Patching       (~2s)   — INT8 LM Head runtime patch
      2. Initializing    (~5s)   — vLLM engine config
      3. Loading weights (~15s)  — fastsafetensors shards
      4. Profiling       (~60s)  — KV cache + MoE kernel profiling
      5. Starting API    (~5s)   — uvicorn startup
    """
    import re

    logs = _parse_container_logs(container)
    if not logs.strip():
        return None

    # Check stages in reverse order (latest stage wins)
    stages: list[tuple[str, str]] = [
        ("Application startup complete", "[5/5] API starting..."),
        ("Available KV cache memory", "[4/5] KV cache allocation..."),
        ("Model loading took", "[4/5] Profiling..."),
        ("Initializing", "[2/5] Initializing vLLM engine..."),
        ("non-default args", "[2/5] Initializing vLLM engine..."),
        ("INT8 LM Head", "[1/5] Applying patches..."),
        ("Starting vLLM", "[1/5] Starting vLLM..."),
    ]

    # Weight loading progress (most detailed)
    matches = re.findall(
        r"Loading.*safetensors.*?(\d+)%\s+Completed\s+\|\s+(\d+)/(\d+)",
        logs,
    )
    if matches:
        pct, done, total = matches[-1]
        return f"[3/5] Loading weights: {pct}% ({done}/{total} shards)"

    for marker, msg in stages:
        if marker in logs:
            return msg

    return "[1/5] Starting container..."


def _get_scribe_loading(container: str) -> str | None:
    """Parse scribe container loading status."""
    import re

    logs = _parse_container_logs(container, tail=20)
    if not logs.strip():
        return None

    if "Application startup complete" in logs:
        return None  # healthy

    matches = re.findall(r"Loading.*?(\d+)%\s+Completed\s+\|\s+(\d+)/(\d+)", logs)
    if matches:
        pct, done, total = matches[-1]
        return f"Loading: {pct}% ({done}/{total})"

    if "Model loading took" in logs:
        return "Profiling..."

    if "Initializing" in logs or "Starting" in logs:
        return "Initializing..."

    return None


def _gpu_processes() -> list[dict[str, Any]]:
    """Get GPU processes with PID, friendly name, and VRAM usage.

    Resolves PIDs to docker containers or known process names so the TUI
    can show "scribe-tts: 4.4 GB" instead of "python: 4.4 GB".
    """
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return []

    procs = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            mb = int(parts[2])
        except ValueError:
            continue
        pname = parts[1]
        label = _resolve_gpu_pid(pid, pname)
        procs.append({"pid": pid, "name": label, "raw_name": pname, "mb": mb})

    procs.sort(key=lambda p: -cast("int", p["mb"]))
    return procs


# Cache PID → friendly name (rebuilt on each TUI tick is fine; PIDs are stable)
_pid_label_cache: dict[int, str] = {}


def _resolve_gpu_pid(pid: int, fallback_name: str) -> str:
    """Map a GPU process PID to its container name or model identity."""
    cached = _pid_label_cache.get(pid)
    if cached:
        return cached

    # Walk up the process tree to find a containerd-shim parent → docker container
    try:
        with Path(f"/proc/{pid}/status").open() as f:
            content = f.read()
        ppid = None
        for line in content.splitlines():
            if line.startswith("PPid:"):
                ppid = int(line.split()[1])
                break

        # Check if direct parent is containerd-shim → docker container
        # The cgroup file gives us the container ID directly
        with Path(f"/proc/{pid}/cgroup").open() as f:
            cgroup = f.read()
        for line in cgroup.splitlines():
            if "docker" in line or "containers" in line:
                # Extract container ID hex (12+ chars)
                import re

                m = re.search(r"([0-9a-f]{12,})", line)
                if m:
                    cid = m.group(1)[:12]
                    # Resolve to container name
                    try:
                        nr = subprocess.run(
                            ["docker", "inspect", "-f", "{{.Name}}", cid],
                            capture_output=True,
                            text=True,
                            check=False,
                            timeout=1,
                        )
                        if nr.returncode == 0 and nr.stdout.strip():
                            name = nr.stdout.strip().lstrip("/")
                            _pid_label_cache[pid] = name
                            return name
                    except Exception:
                        pass
                    _pid_label_cache[pid] = f"docker:{cid[:8]}"
                    return _pid_label_cache[pid]

        # Not in a container — check command line for vllm/uvicorn hints
        try:
            with Path(f"/proc/{pid}/cmdline").open() as f:
                cmdline = f.read().replace("\0", " ")
        except Exception:
            cmdline = ""

        # Check parent if this is an EngineCore subprocess
        if "EngineCore" in fallback_name and ppid:
            try:
                with Path(f"/proc/{ppid}/cmdline").open() as f:
                    parent_cmdline = f.read().replace("\0", " ")
                cmdline = parent_cmdline or cmdline
            except Exception:
                pass

        # Extract model name from vllm cmdline: "vllm serve <model>"
        if "vllm" in cmdline.lower():
            import re

            m = re.search(r"vllm\s+serve\s+(\S+)", cmdline)
            if m:
                model = m.group(1).split("/")[-1][:24]
                _pid_label_cache[pid] = f"vllm:{model}"
                return _pid_label_cache[pid]
            _pid_label_cache[pid] = "vllm"
            return "vllm"
    except Exception:
        pass

    # Fallback: just the process name
    _pid_label_cache[pid] = fallback_name[:20]
    return _pid_label_cache[pid]


def _gpu_pid_utilization() -> dict[int, int]:
    """Return {PID: SM%} for GPU processes.

    Uses `nvidia-smi pmon` for per-process compute utilization.
    Empty dict means GPU is fully idle.
    """
    try:
        r = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1", "-s", "u"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except Exception:
        return {}

    util: dict[int, int] = {}
    for line in r.stdout.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        # Format: gpu pid type sm mem enc dec ... command
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[1])
            sm_str = parts[3]
            if sm_str != "-":
                util[pid] = int(sm_str)
        except (ValueError, IndexError):
            continue
    return util


def _mem_info() -> dict[str, int]:
    try:
        r = subprocess.run(["free", "-m"], capture_output=True, text=True, check=False)
        lines = r.stdout.strip().splitlines()
        mem = lines[1].split()
        swap_line = [ln for ln in lines if ln.startswith("Swap")]
        swap = swap_line[0].split() if swap_line else ["Swap:", "0", "0", "0"]
        return {
            "total": int(mem[1]),
            "used": int(mem[2]),
            "available": int(mem[6]),
            "swap_used": int(swap[2]),
        }
    except Exception:
        return {"total": 0, "used": 0, "available": 0, "swap_used": 0}


def _nv_metrics() -> dict[str, str | float]:
    """Fetch key metrics from nv-monitor prometheus endpoint."""
    result: dict[str, str | float] = {}
    try:
        resp = httpx.get("http://localhost:9100/metrics", timeout=1)
        for line in resp.text.splitlines():
            if line.startswith("#"):
                continue
            if "nv_cpu_usage_percent" in line and 'cpu="overall"' in line:
                result["cpu_pct"] = float(line.split()[-1])
            elif "nv_cpu_temperature" in line:
                result["cpu_temp"] = float(line.split()[-1])
            elif "nv_gpu_utilization_percent" in line:
                result["gpu_pct"] = float(line.split()[-1])
            elif "nv_gpu_temperature" in line:
                result["gpu_temp"] = float(line.split()[-1])
            elif "nv_gpu_power_watts" in line:
                result["gpu_power"] = float(line.split()[-1])
            elif "nv_memory_used_bytes" in line and "bufcache" not in line:
                result["mem_used_gb"] = float(line.split()[-1]) / 1e9
            elif "nv_memory_total_bytes" in line:
                result["mem_total_gb"] = float(line.split()[-1]) / 1e9
            elif "nv_swap_used_bytes" in line:
                result["swap_mb"] = float(line.split()[-1]) / 1e6
            elif "nv_load_average" in line and "1m" in line:
                result["load_1m"] = float(line.split()[-1])
    except Exception:
        pass
    return result


def _model_info(port: int) -> dict[str, str]:
    try:
        r = httpx.get(f"http://localhost:{port}/v1/models", timeout=1)
        d = r.json()["data"][0]
        return {"id": d["id"], "ctx": f"{d['max_model_len'] // 1024}K"}
    except Exception:
        return {"id": "unknown", "ctx": "?"}


def _container_info() -> list[dict[str, Any]]:
    """Get container names, status, uptime, restart count."""
    try:
        r = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--format",
                "{{.Names}}\t{{.Status}}\t{{.RunningFor}}",
                "--filter",
                "name=scribe",
                "--filter",
                "name=autosre",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        containers = []
        for line in r.stdout.strip().splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            name = parts[0] if parts else ""
            status = parts[1] if len(parts) > 1 else ""
            uptime = parts[2] if len(parts) > 2 else ""
            # Extract restart count from status like "Up 2 hours (Restarting)"
            restarts = 0
            try:
                ri = subprocess.run(
                    ["docker", "inspect", "-f", "{{.RestartCount}}", name],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=1,
                )
                restarts = int(ri.stdout.strip()) if ri.returncode == 0 else 0
            except Exception:
                pass
            containers.append(
                {"name": name, "status": status, "uptime": uptime, "restarts": restarts}
            )
        return containers
    except Exception:
        return []


def _disk_usage() -> dict[str, str]:
    """Get sizes of key directories.

    The meeting-scribe path is taken from ``$MEETING_SCRIBE_REPO``
    when set, falling back to ``~/meeting-scribe`` (the layout
    produced by meeting-scribe's bootstrap.sh)."""
    import os

    scribe_root = (
        os.environ.get("MEETING_SCRIBE_REPO", "").strip()
        or str(Path.home() / "meeting-scribe")
    )
    result = {}
    dirs = {
        "HF Cache": "/data/huggingface",
        "Meetings": str(Path(scribe_root).expanduser() / "meetings"),
        "Proxy Logs": str(Path.home() / ".local/share/autosre"),
    }
    for label, path in dirs.items():
        try:
            r = subprocess.run(
                ["du", "-sh", path],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            if r.returncode == 0:
                result[label] = r.stdout.split()[0]
        except Exception:
            pass
    return result


def _hotspot_info() -> dict[str, Any] | None:
    """Get hotspot status from nmcli."""
    try:
        r = subprocess.run(
            ["nmcli", "-t", "-f", "NAME,TYPE,DEVICE", "con", "show", "--active"],
            capture_output=True,
            text=True,
            check=False,
            timeout=1,
        )
        for line in r.stdout.strip().splitlines():
            if "wireless" in line and ("AP" in line or "Demo" in line or "Hotspot" in line):
                name = line.split(":")[0]
                # Get SSID
                sr = subprocess.run(
                    ["nmcli", "-t", "-f", "802-11-wireless.ssid", "con", "show", name],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=1,
                )
                ssid = sr.stdout.strip().split(":")[-1] if sr.returncode == 0 else name
                # Count connected clients
                cr = subprocess.run(
                    ["bash", "-c", "arp -an | grep -c '10.42.0.'"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=1,
                )
                clients = int(cr.stdout.strip()) if cr.returncode == 0 else 0
                return {"ssid": ssid, "clients": clients}
    except Exception:
        pass
    return None


def _bar(pct: float, width: int = 15) -> str:
    """Bar chart: ████████░░░░░░░ 50%"""
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def render_status() -> Panel:
    """Render the full system status panel — 3 equal columns, no dead space."""
    active = load_active_state() or {}
    api_port = int(cast("int", active.get("api_port", 8010)))
    proxy_port = int(cast("int", active.get("proxy_port", 8011)))

    import socket
    import time as _time

    from .metrics import read_recent_requests, vllm_metrics

    def _dot(up: bool, loading: bool = False) -> str:
        if up:
            return "[green]●[/]"
        if loading:
            return "[yellow]◐[/]"
        return "[red]○[/]"

    def _fmt_latency(ms: float) -> str:
        if ms < 1000:
            return f"{ms:.0f}ms"
        if ms < 60000:
            return f"{ms / 1000:.1f}s"
        return f"{int(ms // 60000)}m{int((ms % 60000) // 1000):02d}s"

    # ── Gather all data ──────────────────────────────────────
    scribe_up = False
    meeting_rec = False
    meeting_clients = 0
    scribe_data: dict[str, Any] = {}
    try:
        r = httpx.get("https://localhost:8080/api/status", verify=False, timeout=1)  # noqa: S501
        if r.status_code == 200:
            scribe_up = True
            scribe_data = r.json()
            m = scribe_data.get("meeting", {})
            meeting_clients = scribe_data.get("connections", 0)
            meeting_rec = m.get("state") == "recording"
    except Exception:
        pass

    portal_up = False
    try:
        s = socket.socket()
        s.settimeout(0.2)
        s.connect(("127.0.0.1", 80))
        s.close()
        portal_up = True
    except Exception:
        pass

    scribe_containers = {"ASR": "scribe-asr", "Diarize": "scribe-diarization", "TTS": "scribe-tts"}
    scribe_states = {}
    for name, port in [("ASR", 8003), ("Diarize", 8001), ("TTS", 8002)]:
        up = _health(f"http://localhost:{port}/health")
        loading = None if up else _get_scribe_loading(scribe_containers.get(name, ""))
        scribe_states[name] = (up, loading)

    vllm_up = _health(f"http://localhost:{api_port}/v1/models")
    vllm_info = _model_info(api_port) if vllm_up else None
    vllm_loading = None if vllm_up else _get_loading_progress()

    proxy_ok = False
    proxy_backend = False
    try:
        pr = httpx.get(f"http://localhost:{proxy_port}/health", timeout=1)
        proxy_ok = pr.status_code == 200
        proxy_backend = pr.json().get("backend") == "ok" if proxy_ok else False
    except Exception:
        pass

    mcp_ok = bool(shutil.which("autosre-mcp-search") and shutil.which("autosre-mcp-fetch"))
    nv = _nv_metrics()
    gpu_proc_list = _gpu_processes()  # Rich PID-resolved list
    gpu_pid_util = _gpu_pid_utilization()  # {PID: SM%}
    vm = vllm_metrics()
    containers = _container_info()
    disk = _disk_usage()

    # ── Header ───────────────────────────────────────────────
    header_parts = []
    if vllm_info:
        header_parts.append(f"{vllm_info['id'].split('/')[-1]}  ({vllm_info['ctx']})")
    header_parts.append(f"vLLM :{api_port}  Proxy :{proxy_port}")
    if meeting_rec:
        header_parts.append(
            f"[bold green]● REC[/] {meeting_clients} client{'s' if meeting_clients != 1 else ''}"
        )
    header_sub = "  │  ".join(header_parts)

    # ══════════════════════════════════════════════════════════
    # COLUMN 1: Services + Containers
    # ══════════════════════════════════════════════════════════
    col1_lines = []
    # Services
    col1_lines.append("[dim]── Services ──[/]")
    ui_detail = ":8080" if scribe_up else ""
    col1_lines.append(f" {_dot(scribe_up)} Scribe   [dim]{ui_detail}[/]")
    for name in ("ASR", "Diarize", "TTS"):
        up, loading = scribe_states[name]
        port_map = {"ASR": 8003, "Diarize": 8001, "TTS": 8002}
        detail = f":{port_map[name]}" if up else (loading or "")
        col1_lines.append(f" {_dot(up, bool(loading))} {name:8s} [dim]{detail}[/]")
    vllm_detail = f":{api_port}" if vllm_up else (vllm_loading or "")
    col1_lines.append(f" {_dot(vllm_up, bool(vllm_loading))} vLLM     [dim]{vllm_detail}[/]")
    col1_lines.append(
        f" {_dot(proxy_ok, proxy_ok and not proxy_backend)} Proxy    [dim]{f':{proxy_port}' if proxy_ok else ''}[/]"
    )
    col1_lines.append(f" {_dot(mcp_ok)} MCP      {_dot(portal_up)} Portal")

    # Containers
    col1_lines.append("")
    col1_lines.append("[dim]── Containers ──[/]")
    for c in containers:
        cname = c["name"].replace("scribe-", "").replace("autosre-", "")
        if "Up" in c["status"]:
            st = "[green]Up[/]"
        elif "Restarting" in c["status"]:
            st = "[yellow]Restart[/]"
        elif "Exited" in c["status"]:
            st = "[red]Exit[/]"
        else:
            st = f"[dim]{c['status'][:6]}[/]"
        uptime = c["uptime"].replace("About ", "~").replace(" ago", "")
        restart_str = f" [red bold]↻{c['restarts']}[/]" if c["restarts"] > 0 else ""
        col1_lines.append(f" {cname[:12]:12s} {st} [dim]{uptime[:8]}[/]{restart_str}")

    col1 = Text.from_markup("\n".join(col1_lines))

    # ══════════════════════════════════════════════════════════
    # COLUMN 2: Hardware + VRAM + Disk
    # ══════════════════════════════════════════════════════════
    col2 = Table(show_header=False, box=None, padding=(0, 1))
    col2.add_column("", style="bold", no_wrap=True)
    col2.add_column("", justify="right", no_wrap=True)

    if nv:
        cpu_pct = float(nv.get("cpu_pct", 0) or 0)
        gpu_pct = float(nv.get("gpu_pct", 0) or 0)
        mem_used = float(nv.get("mem_used_gb", 0) or 0)
        mem_total = float(nv.get("mem_total_gb", 128) or 128)
        mem_pct = mem_used / mem_total * 100 if mem_total > 0 else 0.0
        swap_mb = float(nv.get("swap_mb", 0) or 0)

        col2.add_row("CPU", f"{_bar(cpu_pct)} {cpu_pct:.0f}%")
        # GPU row — show top consumer when GPU is busy
        gpu_value = f"{_bar(gpu_pct)} {gpu_pct:.0f}%"
        if gpu_pct > 5 and gpu_pid_util and gpu_proc_list:
            top = max(gpu_proc_list, key=lambda p: gpu_pid_util.get(p["pid"], 0))
            top_sm = gpu_pid_util.get(top["pid"], 0)
            if top_sm > 0:
                gpu_value += f"  [dim]{top['name'][:14]}[/]"
        col2.add_row("GPU", gpu_value)
        col2.add_row("Memory", f"{_bar(mem_pct)} {mem_used:.0f}/{mem_total:.0f}GB")
        col2.add_row("Temp", f"CPU {nv.get('cpu_temp', 0):.0f}°  GPU {nv.get('gpu_temp', 0):.0f}°")
        col2.add_row("Power / Load", f"{nv.get('gpu_power', 0):.0f}W / {nv.get('load_1m', 0):.1f}")
        swap_style = "green" if swap_mb < 100 else "yellow" if swap_mb < 1000 else "red bold"
        col2.add_row("Swap", Text(f"{swap_mb:.0f}MB", style=swap_style))
    else:
        mem = _mem_info()
        col2.add_row("RAM", f"{mem['available'] // 1024}GB avail")

    # VRAM breakdown — show actual process names with VRAM and SM%
    if gpu_proc_list:
        col2.add_row(Text("── GPU processes ──", style="dim"), Text("VRAM   SM%", style="dim"))
        total_gb = 0.0
        for p in gpu_proc_list:
            gb = p["mb"] / 1024
            total_gb += gb
            sm = gpu_pid_util.get(p["pid"], 0)
            # Color: bold green if hot, yellow if warm, dim if idle
            if sm >= 30:
                style = "bold green"
                marker = "●"
            elif sm >= 5:
                style = "yellow"
                marker = "◐"
            else:
                style = "dim"
                marker = " "
            label_text = Text(f"{marker} ", style=style)
            label_text.append(p["name"][:18], style=style)
            value_text = Text(f"{gb:5.1f}GB ", style=style)
            value_text.append(f"{sm:3d}%", style="bold green" if sm >= 30 else style)
            col2.add_row(label_text, value_text)
        col2.add_row(Text("  Total", style="bold"), Text(f"{total_gb:.0f} / 128 GB", style="bold"))

    # Disk
    if disk:
        col2.add_row(Text("── Disk ──", style="dim"), Text(""))
        for label, size in disk.items():
            col2.add_row(f"  {label}", size)

    # ══════════════════════════════════════════════════════════
    # COLUMN 3: Inference + Requests
    # ══════════════════════════════════════════════════════════
    col3 = Table(show_header=False, box=None, padding=(0, 1))
    col3.add_column("", style="bold", no_wrap=True)
    col3.add_column("", justify="right", no_wrap=True)

    TTFT_P50_TARGET = 250
    TTFT_P99_TARGET = 1000
    TPOT_P50_TARGET = 30
    TPOT_P99_TARGET = 60

    def _metric_vs(val: float, target: float) -> Text:
        ratio = val / target if target > 0 else 0
        label = _fmt_latency(val)
        if ratio <= 1.0:
            return Text(f"{label} ✓", style="green")
        if ratio <= 2.0:
            return Text(f"{label} !", style="yellow")
        return Text(f"{label} ✗", style="red")

    if vm:
        ttft_p50 = vm.get("ttft_p50", 0) * 1000
        ttft_p99 = vm.get("ttft_p99", 0) * 1000
        tpot_p50 = vm.get("tpot_p50", 0) * 1000
        tpot_p99 = vm.get("tpot_p99", 0) * 1000
        running = int(vm.get("requests_running", 0))
        waiting = int(vm.get("requests_waiting", 0))
        kv_pct = vm.get("kv_cache_pct", 0) * 100
        gen_tps = vm.get("gen_tps", 0)
        total_reqs = vm.get("total_requests", 0)
        is_idle = running == 0 and waiting == 0

        if is_idle:
            col3.add_row("Status", Text("Idle", style="dim"))
        else:
            q_style = "green" if waiting == 0 else "yellow" if waiting <= 2 else "red"
            col3.add_row("Active/Queue", Text(f"{running} / {waiting}", style=q_style))

        if total_reqs > 0:
            if is_idle:
                col3.add_row(
                    "TTFT p50/99",
                    Text(f"{_fmt_latency(ttft_p50)} / {_fmt_latency(ttft_p99)}", style="dim"),
                )
                col3.add_row(
                    "TPOT p50/99",
                    Text(f"{_fmt_latency(tpot_p50)} / {_fmt_latency(tpot_p99)}", style="dim"),
                )
            else:
                col3.add_row("TTFT p50", _metric_vs(ttft_p50, TTFT_P50_TARGET))
                col3.add_row("TTFT p99", _metric_vs(ttft_p99, TTFT_P99_TARGET))
                col3.add_row("TPOT p50", _metric_vs(tpot_p50, TPOT_P50_TARGET))
                col3.add_row("TPOT p99", _metric_vs(tpot_p99, TPOT_P99_TARGET))

        kv_style = "green" if kv_pct < 70 else "yellow" if kv_pct < 90 else "red"
        col3.add_row("KV Cache", Text(f"{_bar(kv_pct)} {kv_pct:.0f}%", style=kv_style))
        col3.add_row("Prefix Hit", f"{vm.get('prefix_cache_hit_pct', 0):.0f}%")

        if is_idle:
            col3.add_row("Throughput", Text("— idle —", style="dim"))
        else:
            gen_style = "green" if gen_tps >= 30 else "yellow" if gen_tps >= 15 else "dim"
            col3.add_row("Gen tok/s", Text(f"{gen_tps:.0f}", style=gen_style))

        col3.add_row("Requests", f"{total_reqs:,}")
        preemptions = int(vm.get("preemptions", 0))
        if preemptions > 0:
            col3.add_row("Preemptions", Text(f"{preemptions}", style="red bold"))
    else:
        col3.add_row(Text("vLLM offline", style="dim"), Text(""))

    # Recent requests (inline in col3)
    now_ts = _time.time()
    recent = read_recent_requests(8)

    # Determine freshness: LIVE (< 10s), RECENT (< 5 min), STALE (older)
    freshness = "no data"
    last_age_s = None
    if recent:
        last_age_s = now_ts - recent[-1].get("ts", 0)
        if last_age_s < 10:
            freshness = "live"
        elif last_age_s < 300:
            freshness = "recent"
        else:
            freshness = "stale"

    # Header row with freshness badge
    if freshness == "live":
        badge = Text("● LIVE", style="green bold")
    elif freshness == "recent":
        badge = Text("◐ RECENT", style="yellow")
    elif freshness == "stale":
        badge = Text("○ STALE", style="dim red")
    else:
        badge = Text("", style="dim")
    col3.add_row(Text("── Requests ──", style="dim"), badge)

    # Show actively-running vLLM requests first (from metrics, most important)
    if vm and vm.get("requests_running", 0) > 0:
        running = int(vm["requests_running"])
        waiting = int(vm.get("requests_waiting", 0))
        msg = f"{running} active"
        if waiting > 0:
            msg += f", {waiting} queued"
        col3.add_row(Text("In Flight", style="bold"), Text(msg, style="green"))

    # Recent log entries
    active_recent = [req for req in recent if (now_ts - req.get("ts", 0)) < 300]
    if active_recent:
        for req in active_recent[-6:]:
            ts = _time.strftime("%H:%M:%S", _time.localtime(req.get("ts", 0)))
            lat = int(req.get("elapsed_ms", 0))
            lat_style = "green" if lat < 2000 else "yellow" if lat < 5000 else "red"
            out_tok = req.get("output_tokens", 0)
            src = req.get("source", "?")
            src_prefix = "T " if src == "translation" else "C "
            col3.add_row(
                Text(f"{src_prefix}{ts}", style="dim"),
                Text(f"{_fmt_latency(lat)}  {out_tok}tok", style=lat_style),
            )
    elif freshness == "stale" and last_age_s is not None:
        # Dim message for stale data
        if last_age_s > 3600:
            age_str = f"{int(last_age_s // 3600)}h ago"
        else:
            age_str = f"{int(last_age_s // 60)}m ago"
        col3.add_row(Text(f"Last: {age_str}", style="dim"), Text("(stale)", style="dim red"))
    elif not recent:
        col3.add_row(Text("No requests yet", style="dim"), Text(""))

    # ══════════════════════════════════════════════════════════
    # LAYOUT: 3 equal columns
    # ══════════════════════════════════════════════════════════
    col1_panel = Panel(
        col1, title="[bold]Services & Containers[/]", border_style="cyan", padding=(0, 1)
    )
    col2_panel = Panel(
        col2, title="[bold]Hardware & Storage[/]", border_style="cyan", padding=(0, 1)
    )
    col3_panel = Panel(
        col3, title="[bold]Inference & Requests[/]", border_style="cyan", padding=(0, 1)
    )

    layout = Table(show_header=False, box=None, padding=(0, 0), expand=True)
    layout.add_column("c1", ratio=1)
    layout.add_column("c2", ratio=1)
    layout.add_column("c3", ratio=1)
    layout.add_row(col1_panel, col2_panel, col3_panel)

    content = layout

    return Panel(
        content,
        title=Text.from_markup(f"[bold white on blue] Auto-SRE [/]  {header_sub}"),
        subtitle="[dim]autosre claude | autosre metrics -f | autosre stop[/]",
        border_style="blue",
        padding=(0, 0),
    )


def render_status_vertical() -> Panel:
    """Narrow stacked status panel for a left-sidebar tmux pane.

    Target width ~58 cols (fits in a 60-col pane after borders). Sections
    are stacked top-to-bottom: Services, Containers, Hardware, GPU
    processes, Inference, Recent requests. Uses the same data sources as
    :func:`render_status` so values stay consistent when the user resizes
    the window and the layout flips between full/vertical/compact.
    """
    from .metrics import read_recent_requests, vllm_metrics

    active = load_active_state() or {}
    api_port = int(cast("int", active.get("api_port", 8010)))
    proxy_port = int(cast("int", active.get("proxy_port", 8011)))

    def _dot(up: bool, loading: bool = False) -> str:
        if up:
            return "[green]●[/]"
        if loading:
            return "[yellow]◐[/]"
        return "[red]○[/]"

    # ── Service health ───────────────────────────────────────
    scribe_up = _health("https://localhost:8080/api/status")
    scribe_states: dict[str, tuple[bool, str | None]] = {}
    scribe_containers = {"ASR": "scribe-asr", "Diarize": "scribe-diarization", "TTS": "scribe-tts"}
    for name, port in [("ASR", 8003), ("Diarize", 8001), ("TTS", 8002)]:
        up = _health(f"http://localhost:{port}/health")
        loading = None if up else _get_scribe_loading(scribe_containers.get(name, ""))
        scribe_states[name] = (up, loading)

    vllm_up = _health(f"http://localhost:{api_port}/v1/models")
    vllm_info = _model_info(api_port) if vllm_up else None
    vllm_loading = None if vllm_up else _get_loading_progress()

    proxy_ok = False
    proxy_backend = False
    try:
        pr = httpx.get(f"http://localhost:{proxy_port}/health", timeout=1)
        proxy_ok = pr.status_code == 200
        proxy_backend = pr.json().get("backend") == "ok" if proxy_ok else False
    except Exception:
        pass

    nv = _nv_metrics()
    vm = vllm_metrics()
    gpu_proc_list = _gpu_processes()
    gpu_pid_util = _gpu_pid_utilization()
    containers = _container_info()

    # ── Build a flat list of (label, value) rows ────────────
    lines: list[str] = []

    lines.append("[bold cyan]── Services ──[/]")
    lines.append(f" {_dot(scribe_up)} Scribe    [dim]:8080[/]")
    for name in ("ASR", "Diarize", "TTS"):
        up, loading = scribe_states[name]
        port_map = {"ASR": 8003, "Diarize": 8001, "TTS": 8002}
        detail = f":{port_map[name]}" if up else (loading or "")
        lines.append(f" {_dot(up, bool(loading))} {name:<9s} [dim]{detail}[/]")
    vllm_detail = f":{api_port}" if vllm_up else (vllm_loading or "")
    if vllm_info:
        vllm_detail = f":{api_port}  [dim]{vllm_info['id'].split('/')[-1][:18]}[/]"
    lines.append(f" {_dot(vllm_up, bool(vllm_loading))} vLLM      [dim]{vllm_detail}[/]")
    lines.append(
        f" {_dot(proxy_ok, proxy_ok and not proxy_backend)} Proxy     [dim]:{proxy_port}[/]"
    )

    if containers:
        lines.append("")
        lines.append("[bold cyan]── Containers ──[/]")
        for c in containers[:6]:
            cname = c["name"].replace("scribe-", "").replace("autosre-", "")
            if "Up" in c["status"]:
                st = "[green]Up[/]"
            elif "Restarting" in c["status"]:
                st = "[yellow]Restart[/]"
            elif "Exited" in c["status"]:
                st = "[red]Exit[/]"
            else:
                st = f"[dim]{c['status'][:6]}[/]"
            uptime = c["uptime"].replace("About ", "~").replace(" ago", "")
            restart_str = f" [red bold]↻{c['restarts']}[/]" if c["restarts"] > 0 else ""
            lines.append(f" {cname[:14]:<14s} {st} [dim]{uptime[:8]}[/]{restart_str}")

    lines.append("")
    lines.append("[bold cyan]── Hardware ──[/]")
    if nv:
        cpu_pct = float(nv.get("cpu_pct", 0) or 0)
        gpu_pct = float(nv.get("gpu_pct", 0) or 0)
        mem_used = float(nv.get("mem_used_gb", 0) or 0)
        mem_total = float(nv.get("mem_total_gb", 128) or 128)
        mem_pct = mem_used / mem_total * 100 if mem_total > 0 else 0.0
        lines.append(f" CPU   {_bar(cpu_pct, 12)} {cpu_pct:>3.0f}%")
        lines.append(f" GPU   {_bar(gpu_pct, 12)} {gpu_pct:>3.0f}%")
        lines.append(f" VRAM  {_bar(mem_pct, 12)} {mem_used:.0f}/{mem_total:.0f}GB")
        lines.append(
            f" Temp  CPU {nv.get('cpu_temp', 0):.0f}°  GPU {nv.get('gpu_temp', 0):.0f}°  "
            f"[dim]{nv.get('gpu_power', 0):.0f}W[/]"
        )
    else:
        lines.append(" [dim]metrics unavailable[/]")

    if gpu_proc_list:
        lines.append("")
        lines.append("[bold cyan]── GPU processes ──[/]")
        total_gb = 0.0
        for p in gpu_proc_list[:5]:
            gb = p["mb"] / 1024
            total_gb += gb
            sm = gpu_pid_util.get(p["pid"], 0)
            if sm >= 30:
                style = "bold green"
                marker = "●"
            elif sm >= 5:
                style = "yellow"
                marker = "◐"
            else:
                style = "dim"
                marker = " "
            name = p["name"][:18]
            lines.append(f" [{style}]{marker} {name:<18s}[/] [dim]{gb:>5.1f}GB {sm:>3d}%[/]")

    lines.append("")
    lines.append("[bold cyan]── Inference ──[/]")
    if vm:
        running = int(vm.get("requests_running", 0))
        waiting = int(vm.get("requests_waiting", 0))
        kv_pct = vm.get("kv_cache_pct", 0) * 100
        gen_tps = vm.get("gen_tps", 0)
        total_reqs = vm.get("total_requests", 0)
        ttft_p50 = vm.get("ttft_p50", 0) * 1000
        tpot_p50 = vm.get("tpot_p50", 0) * 1000
        preemptions = int(vm.get("preemptions", 0))

        q_style = "green" if waiting == 0 else "yellow" if waiting <= 2 else "red"
        lines.append(f" Active/Queue  [{q_style}]{running} / {waiting}[/]")
        lines.append(f" Gen tok/s     {gen_tps:.0f}")
        lines.append(f" TTFT p50      {ttft_p50:.0f}ms")
        lines.append(f" TPOT p50      {tpot_p50:.0f}ms")
        kv_style = "green" if kv_pct < 70 else "yellow" if kv_pct < 90 else "red"
        lines.append(f" KV Cache      [{kv_style}]{_bar(kv_pct, 10)} {kv_pct:.0f}%[/]")
        lines.append(f" Requests      {int(total_reqs):,}")
        if preemptions > 0:
            lines.append(f" [red bold]Preemptions   {preemptions}[/]")
    else:
        lines.append(" [dim]vLLM offline[/]")

    # Recent requests (compact, last 4)
    import time as _time

    recent = read_recent_requests(4)
    if recent:
        lines.append("")
        lines.append("[bold cyan]── Recent ──[/]")
        for req in recent[-4:]:
            ts = _time.strftime("%H:%M:%S", _time.localtime(req.get("ts", 0)))
            lat = int(req.get("elapsed_ms", 0))
            lat_style = "green" if lat < 2000 else "yellow" if lat < 5000 else "red"
            out_tok = int(req.get("output_tokens", 0))
            src = req.get("source", "?")
            src_prefix = "T" if src == "translation" else "C"
            lines.append(f" {src_prefix} {ts}  [{lat_style}]{lat:>5}ms[/]  [dim]{out_tok}tok[/]")

    header_bits = []
    if vllm_info:
        header_bits.append(f"{vllm_info['id'].split('/')[-1][:20]}")
    header_bits.append(f":{api_port}/{proxy_port}")
    header_sub = " │ ".join(header_bits)

    body = Text.from_markup("\n".join(lines))
    return Panel(
        body,
        title=Text.from_markup("[bold white on blue] Auto-SRE [/]"),
        subtitle=f"[dim]{header_sub}[/]",
        border_style="blue",
        padding=(0, 1),
    )


def render_status_compact() -> Panel:
    """Compact single-row status panel for the top tmux pane (~6 rows).

    Deliberately minimal: one row per (services, containers, hardware,
    requests) so the whole thing fits in a pane that has been resized
    down to ~8 rows. Degrades silently if data sources are unreachable.
    """
    active = load_active_state() or {}
    api_port = int(cast("int", active.get("api_port", 8010)))
    proxy_port = int(cast("int", active.get("proxy_port", 8011)))

    from .metrics import request_analytics, vllm_metrics

    def _dot(up: bool) -> str:
        return "[green]●[/]" if up else "[red]○[/]"

    vllm_up = _health(f"http://localhost:{api_port}/v1/models")
    proxy_ok = _health(f"http://localhost:{proxy_port}/health")
    scribe_up = _health("https://localhost:8080/api/status")

    nv = _nv_metrics() or {}
    vm = vllm_metrics() or {}
    analytics = request_analytics() or {}

    services_line = (
        f"{_dot(vllm_up)} vLLM :{api_port}  "
        f"{_dot(proxy_ok)} Proxy :{proxy_port}  "
        f"{_dot(scribe_up)} Scribe :8080"
    )
    hw_line = (
        f"GPU {nv.get('gpu_pct', 0):>3}%  "
        f"VRAM {nv.get('mem_used_gb', 0):.0f}/{nv.get('mem_total_gb', 0):.0f}GB  "
        f"CPU {nv.get('cpu_pct', 0):>3}%  "
        f"Temp {nv.get('gpu_temp_c', 0):>3}°C"
    )
    running = int(vm.get("running", 0))
    waiting = int(vm.get("waiting", 0))
    gen_tps = float(vm.get("gen_tps", 0))
    recent_reqs = int(analytics.get("req_per_min", 0))
    req_line = (
        f"Running {running:>2}  Waiting {waiting:>2}  "
        f"Gen {gen_tps:>4.0f} tok/s  "
        f"Recent {recent_reqs:>3}/min"
    )

    body = Text.from_markup(f"{services_line}\n[dim]{hw_line}[/]\n[dim]{req_line}[/]")
    return Panel(
        body,
        title="[bold white on blue] Auto-SRE (compact) [/]",
        subtitle="[dim]q quit  s refresh  1 Claude  2 Swarm[/]",
        border_style="blue",
        padding=(0, 1),
    )


def render_menu() -> None:
    """Render the action menu (standalone, used by tests)."""
    actions = Table(show_header=False, box=None, padding=(0, 1), expand=True)
    for _ in range(6):
        actions.add_column("", justify="center")
    actions.add_row(
        "[bold cyan]1[/] Launch Claude",
        "[bold cyan]2[/] Swarm Demo",
        "[bold cyan]3[/] Benchmark",
        "[bold cyan]4[/] Smoke Test",
        "[bold cyan]5[/] Tech Audit",
        "[bold cyan]q[/] Quit",
    )
    console.print(Panel(actions, border_style="dim", padding=(0, 1)))


def run_benchmark() -> None:
    """Quick inline benchmark."""
    active = load_active_state() or {}
    api_port = int(cast("int", active.get("api_port", 8010)))

    if not _health(f"http://localhost:{api_port}"):
        console.print("[red]Coding agent not running. Run 'autosre start' first.[/]")
        return

    import concurrent.futures
    import json

    url = f"http://localhost:{api_port}/v1/chat/completions"
    model = _model_info(api_port)["id"]

    def stream_req(prompt: str, max_tokens: int = 150) -> tuple[float, int, float]:
        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0.6,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            # Same priority as Claude Code coding agent so live translation
            # (priority -10 in meeting-scribe) preempts TUI benchmarks.
            "priority": 10,
        }
        t0 = time.perf_counter()
        ft = None
        tc = 0
        with httpx.Client(timeout=120) as c, c.stream("POST", url, json=payload) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                d = line[6:]
                if d == "[DONE]":
                    break
                ch = json.loads(d)["choices"][0]["delta"]
                if ch.get("content"):
                    if ft is None:
                        ft = time.perf_counter()
                    tc += 1
        t1 = time.perf_counter()
        tps = tc / (t1 - ft) if ft else 0
        return tps, tc, (ft - t0) * 1000 if ft else 0

    prompts = [
        "Write a Python quicksort.",
        "Write a Python binary search.",
        "Write a merge sort.",
        "Write a linked list.",
        "Write a hash map.",
        "Write a BFS.",
        "Write a DFS.",
        "Write a trie.",
    ]

    with console.status("[cyan]Warming up...[/]"):
        stream_req("hi", 5)

    table = Table(title="Benchmark Results", show_header=True, header_style="bold")
    table.add_column("Concurrency", justify="center")
    table.add_column("Per-request", justify="right")
    table.add_column("Aggregate", justify="right")
    table.add_column("Speedup", justify="right")

    with console.status("[cyan]Single request...[/]"):
        tps, _, ttft = stream_req(prompts[0])
        base = tps

    table.add_row("1", f"{tps:.1f} tok/s", f"{tps:.1f} tok/s", "1.0x")

    for n in [2, 4, 8]:
        with console.status(f"[cyan]{n}x concurrent...[/]"):
            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
                futs = [pool.submit(stream_req, p) for p in prompts[:n]]
                results = [f.result() for f in futs]
            t_total = time.perf_counter() - t0
            total_tokens = sum(r[1] for r in results)
            avg = sum(r[0] for r in results) / len(results)
            agg = total_tokens / t_total
            speedup = agg / base if base > 0 else 0

        table.add_row(str(n), f"{avg:.1f} tok/s", f"{agg:.1f} tok/s", f"{speedup:.1f}x")

    console.print()
    console.print(table)
    console.print(f"\n[dim]TTFT: {ttft:.0f}ms  |  Model: {model}[/]")


def run_smoke() -> None:
    """Quick smoke test."""
    active = load_active_state() or {}
    proxy_port = int(cast("int", active.get("proxy_port", 8011)))
    api_port = int(cast("int", active.get("api_port", 8010)))
    model = _model_info(api_port)["id"]
    proxy = f"http://localhost:{proxy_port}"

    tests: list[tuple[str, Callable[[], object]]] = [
        ("Proxy health", lambda: _health(f"http://localhost:{proxy_port}")),
        (
            "Completion",
            lambda: (
                httpx.post(
                    f"{proxy}/v1/messages",
                    json={
                        "model": model,
                        "max_tokens": 100,
                        "messages": [{"role": "user", "content": "Say OK"}],
                    },
                    timeout=30,
                )
                .json()
                .get("type")
                == "message"
            ),
        ),
        (
            "Tool calling",
            lambda: any(
                b["type"] == "tool_use"
                for b in httpx.post(
                    f"{proxy}/v1/messages",
                    json={
                        "model": model,
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
                .json()
                .get("content", [])
            ),
        ),
    ]

    for name, test in tests:
        with console.status(f"[cyan]Testing {name}...[/]"):
            try:
                ok = test()
                if ok:
                    console.print(f"  [green]✓[/] {name}")
                else:
                    console.print(f"  [red]✗[/] {name}")
            except Exception as e:
                console.print(f"  [red]✗[/] {name}: {e}")


def _read_key_with_timeout(timeout: float = 5.0) -> str | None:
    """Read a single keypress with timeout. Returns None on timeout."""
    import select
    import sys

    if not sys.stdin.isatty():
        return Prompt.ask(
            "\n[bold cyan]>[/]", choices=["1", "2", "3", "4", "5", "s", "q"], default="s"
        )

    import termios
    import tty

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ready, _, _ = select.select([sys.stdin], [], [], timeout)
        if ready:
            return sys.stdin.read(1)
        return None
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def main() -> None:
    """Interactive TUI main loop with auto-refresh.

    The TUI runs full-screen in the current pane. When the user picks
    an action (Claude Code, swarm demo, audit) the TUI process is
    replaced in-place via ``os.execvp`` so that pane becomes the main
    Claude / team-leader window — no tmux splits, no cramped sidebar.
    """
    from io import StringIO
    from pathlib import Path

    from . import tmux as tmux_mod

    refresh_interval = 2.0  # seconds — enough to avoid flicker, short enough to feel live
    _last_output = ""  # cache last render to avoid blank screen on error

    def _dispatch_cmd(cmd: list[str], label: str) -> None:  # noqa: ARG001 — label kept for API symmetry
        """Replace the TUI with ``cmd`` in the current pane.

        The dispatched process inherits the TUI's cwd, which is
        whatever the operator invoked ``autosre`` from. Override via
        ``$AUTOSRE_DISPATCH_CWD`` if you want commands to land in a
        specific working tree (e.g. the auto-sre source checkout)."""
        override = os.environ.get("AUTOSRE_DISPATCH_CWD", "").strip()
        if override:
            target = Path(override).expanduser()
            if target.is_dir():
                os.chdir(target)
        os.execvp(cmd[0], cmd)

    while True:
        # Render to buffer first, then clear+print atomically — no blank flashes
        try:
            buf = StringIO()
            buf_console = Console(
                file=buf, width=console.width, height=console.height, force_terminal=True
            )
            ph = tmux_mod.pane_height() if tmux_mod.in_tmux() else None
            pw = tmux_mod.pane_width() if tmux_mod.in_tmux() else None
            # Render-variant selection is a pure (width, height) function
            # so manual resizes (tmux prefix-</>/resize-pane) flip the
            # layout naturally without any signal handling:
            #
            #   wide + tall  → full 3-column render_status
            #   narrow+tall  → vertical stacked sidebar
            #   short        → 3-line compact glance
            if pw is not None and ph is not None:
                if pw >= 100 and ph >= 26:
                    variant = "full"
                elif pw >= 50 and ph >= 28:
                    variant = "vertical"
                else:
                    variant = "compact"
            else:
                variant = "full"

            if variant == "full":
                panel = render_status()
            elif variant == "vertical":
                panel = render_status_vertical()
            else:
                panel = render_status_compact()
            buf_console.print(panel)

            show_menu_inline = variant == "full"
            if show_menu_inline:
                actions = Table(show_header=False, box=None, padding=(0, 1), expand=True)
                for _ in range(6):
                    actions.add_column("", justify="center")
                actions.add_row(
                    "[bold cyan]1[/] Launch Claude",
                    "[bold cyan]2[/] Swarm Demo",
                    "[bold cyan]3[/] Benchmark",
                    "[bold cyan]4[/] Smoke Test",
                    "[bold cyan]5[/] Tech Audit",
                    "[bold cyan]q[/] Quit",
                )
                buf_console.print(Panel(actions, border_style="dim", padding=(0, 1)))
                buf_console.print("\n[dim]Live refresh. Press a key to act.[/]")
            _last_output = buf.getvalue()
        except Exception as e:
            # On render error, show last known good output + error
            if not _last_output:
                _last_output = f"[red]Render error: {e}[/]\n"

        console.clear()
        console.file.write(_last_output)
        console.file.flush()

        choice = _read_key_with_timeout(refresh_interval)

        if choice is None:
            continue  # auto-refresh
        if choice in ("q", "Q", "\x03"):  # q or Ctrl+C
            break
        if choice in ("s", "S"):
            continue  # manual refresh
        if choice == "1":
            _dispatch_cmd(["autosre", "claude"], "Launching Claude Code")
        elif choice == "2":
            console.print()
            from .demo import SCENARIOS

            for sid, s in SCENARIOS.items():
                console.print(f"  [bold cyan]{sid}[/]) {s['name']}")
                console.print(f"     [dim]{s['desc']}[/]")
            scenario_choice = Prompt.ask("\n[bold]Scenario[/]", choices=[str(i) for i in SCENARIOS])
            _dispatch_cmd(
                ["autosre", "swarm-demo", "run", scenario_choice],
                "Launching swarm demo",
            )
        elif choice == "3":
            run_benchmark()
        elif choice == "4":
            run_smoke()
        elif choice == "5":
            _dispatch_cmd(
                [
                    "autosre",
                    "claude",
                    "Use agent teams to audit this codebase for tech debt. Spawn 4 agents in parallel: "
                    "(1) scan all Python files for hardcoded ports, IPs, URLs, and model names; "
                    "(2) find stale references to old models like nemotron, NVFP4, or FP8; "
                    "(3) check consistency between recipes, vllm.py models dict, and docker configs; "
                    "(4) validate that README and CLAUDE.md match actual behavior. "
                    "Each agent reports independently.",
                ],
                "Launching tech debt audit",
            )
