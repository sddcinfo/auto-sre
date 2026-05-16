"""Integration test: end-to-end plan-review loop with a stub backend.

Stands up a fake OpenAI-compatible HTTP server in a background thread,
writes a fake active.json pointing at it, then runs ``autosre review plan``
twice — the first call simulates a blocking review (deny), the second
simulates a clean review (allow). Verifies that iteration state advances
between calls and that per-iteration logs are written to disk.

Marked ``slow`` because it involves real subprocesses and a live HTTP
server. Run with ``pytest -m slow tests/integration/``.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from contextlib import closing
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.slow


def _free_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _FakeVllmServer:
    """Minimal OpenAI-compatible server for ``/v1/models`` and ``/v1/chat/completions``.

    The completion response is provided per-test by mutating ``next_response``.
    """

    def __init__(self) -> None:
        self.next_response: dict[str, Any] = {
            "choices": [
                {"message": {"content": '{"findings": [], "questions": []}'}},
            ],
        }
        self.port = _free_port()
        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        server = self  # capture for the handler

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any, **_kwargs: Any) -> None:
                pass  # silence test noise

            def do_GET(self) -> None:
                if self.path == "/v1/models":
                    self._send_json(
                        {
                            "object": "list",
                            "data": [{"id": "fake-test-model"}],
                        },
                    )
                else:
                    self.send_response(404)
                    self.end_headers()

            def do_POST(self) -> None:
                if self.path == "/v1/chat/completions":
                    length = int(self.headers.get("Content-Length", "0"))
                    _body = self.rfile.read(length)  # consume
                    self._send_json(server.next_response)
                else:
                    self.send_response(404)
                    self.end_headers()

            def _send_json(self, data: dict[str, Any]) -> None:
                payload = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._httpd = HTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        # Wait briefly for the server to be ready
        deadline = time.time() + 3
        while time.time() < deadline:
            try:
                with closing(
                    socket.create_connection(("127.0.0.1", self.port), timeout=0.2),
                ):
                    return
            except OSError:
                time.sleep(0.05)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()


@pytest.fixture
def fake_vllm() -> _FakeVllmServer:
    server = _FakeVllmServer()
    server.start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def isolated_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_vllm: _FakeVllmServer,
) -> dict[str, str]:
    """Point autosre's XDG dirs at ``tmp_path`` and write a fake active.json."""
    xdg_data = tmp_path / "xdg-data"
    xdg_config = tmp_path / "xdg-config"
    xdg_state = tmp_path / "xdg-state"
    xdg_cache = tmp_path / "xdg-cache"
    for d in (xdg_data, xdg_config, xdg_state, xdg_cache):
        d.mkdir(parents=True)

    env = {
        "XDG_DATA_HOME": str(xdg_data),
        "XDG_CONFIG_HOME": str(xdg_config),
        "XDG_STATE_HOME": str(xdg_state),
        "XDG_CACHE_HOME": str(xdg_cache),
        "AUTOSRE_REVIEW_CHAIN": "local",  # force local provider only
        "AUTOSRE_REVIEW_MODEL": "fake-test-model",
        "PATH": subprocess.os.environ.get("PATH", ""),
    }

    # Write fake active.json so _local_provider_runner can resolve the backend.
    active = {
        "backend": "vllm",
        "model": "fake-test-model",
        "api_host": "127.0.0.1",
        "api_port": fake_vllm.port,
        "proxy_port": fake_vllm.port + 1,
    }
    autosre_data = xdg_data / "autosre"
    autosre_data.mkdir()
    (autosre_data / "active.json").write_text(json.dumps(active))

    return env


def _run_review_plan(env: dict[str, str], plan_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "autosre.cli",
            "review",
            "plan",
            str(plan_path),
            "--chain",
            "local",
            "--json-output",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


class TestReviewLoop:
    def test_blocking_then_clean_advances_iteration(
        self,
        tmp_path: Path,
        isolated_env: dict[str, str],
        fake_vllm: _FakeVllmServer,
    ) -> None:
        plan = tmp_path / "plan.md"
        plan.write_text(
            "# Plan: Add rate limiting\n\n"
            "1. Add middleware that counts requests per IP in memory.\n"
            "2. Use a simple dict keyed by IP with lazy cleanup.\n"
            "3. No persistence, no cluster coordination.\n",
        )

        # --- Iteration 1: blocking response ---
        fake_vllm.next_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "findings": [
                                    {
                                        "severity": "P0",
                                        "title": "In-memory rate limit",
                                        "description": "Won't scale across processes.",
                                        "recommendation": "Use Redis or a shared store.",
                                    },
                                ],
                                "questions": [],
                            },
                        ),
                    },
                },
            ],
        }

        r1 = _run_review_plan(isolated_env, plan)
        assert r1.returncode == 1, (
            f"expected blocking, got rc={r1.returncode}\n{r1.stdout}\n{r1.stderr}"
        )
        data1 = json.loads(r1.stdout)
        assert data1["iteration"] == 1
        assert data1["blocking"] is True
        assert len(data1["findings"]) == 1
        assert data1["findings"][0]["severity"] == "P0"

        # --- Iteration 2: clean response (Claude has "updated" the plan) ---
        # Bump the mtime to simulate a file update so the cached-clean path
        # doesn't short-circuit (upstream port doesn't implement that
        # short-circuit, so it's a no-op — just a safety belt).
        plan.write_text(plan.read_text() + "\n4. Use Redis-backed sliding window.\n")

        fake_vllm.next_response = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({"findings": [], "questions": []}),
                    },
                },
            ],
        }

        r2 = _run_review_plan(isolated_env, plan)
        assert r2.returncode == 0, (
            f"expected clean, got rc={r2.returncode}\n{r2.stdout}\n{r2.stderr}"
        )
        data2 = json.loads(r2.stdout)
        assert data2["iteration"] == 2
        assert data2["blocking"] is False
        assert data2["findings"] == []

        # --- State file verification ---
        from pathlib import Path as _Path

        state_dir = _Path(isolated_env["XDG_DATA_HOME"]) / "autosre" / "review-state"
        state_files = list(state_dir.glob("_state_*.json"))
        assert len(state_files) == 1, f"expected one state file, got {state_files}"
        state = json.loads(state_files[0].read_text())
        assert state["iteration"] == 2
        assert state["last_review_status"] == "clean"
        assert state["last_review_findings_count"] == 0
        # Previous findings should be from iteration 2 (empty after clean review)
        assert state["previous_findings"] == []

        # --- Per-iteration logs exist ---
        log_dir = _Path(isolated_env["XDG_DATA_HOME"]) / "autosre" / "review-log"
        assert log_dir.exists()
        log_files = list(log_dir.glob("*.json"))
        assert len(log_files) >= 2, f"expected >= 2 per-iteration logs, got {log_files}"
