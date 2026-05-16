# Auto-SRE

Local LLM server management for Claude Code. Runs Ollama, llama.cpp, or
vLLM behind a native Anthropic Messages API proxy — no LiteLLM needed, no
cloud API keys required.

Ships with a plan-review hook that turns Claude Code's plan mode into a
closed feedback loop (plan → review → update → re-review), a Bash-guard
hook, local MCP servers for web search/fetch, and an optional stealth
file-sharing dropbox for pushing artefacts to the box over HTTPS.

## Requirements

- **Python 3.11+**
- A backend — at least one of:
  - **Ollama ≥ 0.14.0** — broadest model catalogue, macOS + Linux
  - **llama.cpp** (`brew install llama.cpp`) — single-binary, GGUF models
  - **vLLM** — multi-GPU, cluster-aware; requires NVIDIA CUDA + Docker

## Quick start

```bash
git clone https://github.com/sddcinfo/auto-sre.git
cd auto-sre
pip install -e '.[dev]'
autosre setup
autosre start
autosre test
autosre claude         # launches Claude Code against the local server
```

`autosre claude` injects the environment and settings that make Claude
Code talk to the local backend instead of the Anthropic API — see
[How it works](#how-it-works) for details.

## Backends

| Backend | When to pick it | Install |
|---------|-----------------|---------|
| Ollama (default) | General use, laptops, fastest setup | [ollama.com](https://ollama.com) |
| llama.cpp | Single static binary, GGUF-based models | `brew install llama.cpp` |
| vLLM | Multi-GPU, cluster-aware, recipe-driven | Docker + NVIDIA toolkit |

Backend selection is automatic unless overridden:

```bash
autosre setup --backend auto       # default — detects what's installed
autosre setup --backend ollama
autosre setup --backend llamacpp
autosre setup --backend vllm
```

Models are pulled per backend:

```bash
autosre models list                # what's available for the current backend
autosre models pull qwen3.6:35b-a3b-fp8
```

vLLM is recipe-driven; recipes live in `autosre/backends/recipes/*.yaml`
and pin the model, quantization, context length, and concurrency knobs.

## Commands

Run `autosre --help` for the full list. The most-used ones:

| Command | Purpose |
|---------|---------|
| `autosre setup` | Environment validation + backend setup |
| `autosre start` | Start the backend (and, on GB10, meeting-scribe + warmup) |
| `autosre stop` | Stop the proxy (vLLM container is kept running unless `--unload-model`) |
| `autosre status` | Backend / model health |
| `autosre test` | Send a probe message via the proxy |
| `autosre claude` | Launch Claude Code with local proxy + local review chain |
| `autosre backends` | List available backends + requirements |
| `autosre models list` / `models pull <id>` | Manage backend model weights |
| `autosre bench` | Latency + throughput benchmarks |
| `autosre metrics` | vLLM Prometheus metrics + proxy request analytics |
| `autosre ui` | Terminal dashboard (Rich TUI) |
| `autosre hooks install` / `uninstall` / `status` | Wire the review hook into bare `claude` |
| `autosre mcp setup` / `status` / `remove` | Local web search + fetch MCP servers |
| `autosre dropbox install` / `init` / `start` / `status` | Self-hosted HTTPS file dropbox |
| `autosre review plan <path>` | Run a plan review manually (debugging / scripting) |
| `autosre ssh exec <host> -- <cmd>` | Routed SSH wrapper (required by the Bash guard) |
| `autosre precommit` | Scan working tree for sensitive data (credentials, keys, LAN identity) before commit |

## Plan-review loop (plan → review → update → re-review)

autosre ships a Claude Code hook that turns plan mode into a closed feedback
loop. When Claude exits plan mode, the hook runs the plan through an AI
reviewer that returns JSON findings tagged `P0`/`P1`/`P2`. If any `P0`/`P1`
issues are found, the hook denies the `ExitPlanMode` call with the findings
embedded in `additionalContext`; Claude re-enters plan mode, updates the
plan, exits again, and the loop closes on the next iteration. State is
tracked per-plan by absolute-path hash so two repos with a `plan.md` file
don't share iteration counters.

Two launch modes, symmetric design:

| Launcher | Claude Code points at | Reviewer | How it's wired |
|---|---|---|---|
| **`autosre claude`** | Local vLLM/Ollama/llamacpp via the Anthropic proxy | Local model self-review (fresh context per iteration) | Temp `--settings=<file>` written at launch; `AUTOSRE_REVIEW_CHAIN=local` forced; `~/.claude/settings.json` untouched |
| **`claude`** (bare, after `autosre hooks install`) | Anthropic API (normal) | `codex exec` with `gpt-5.4` at `xhigh` reasoning — falls back to `local` → `gemini` → `claude` CLIs | Managed block in `~/.claude/settings.json`; uninstallable via `autosre hooks uninstall` |

Setup for bare-claude mode:

```bash
autosre hooks install          # writes the managed block into ~/.claude/settings.json
autosre hooks status           # verify + show drift if any
# ...use bare `claude` normally; ExitPlanMode fires the review loop...
autosre hooks uninstall        # surgical removal; user's own hook entries preserved
```

The installer preserves any existing user hook entries (it appends to
`PreToolUse`/`PostToolUse`/etc. event lists rather than replacing), tracks
exactly what it added via a sidecar file at
`~/.claude/.autosre-hooks-installed.json`, and the uninstall command removes
only the entries it installed. Rerunning `install` is idempotent: no files
are rewritten unless the planned entry set or the embedded Python
interpreter path has changed.

Direct invocation (for debugging or scripting) bypasses the hook:

```bash
autosre review plan /tmp/my-plan.md --chain local --json-output
autosre review plan /tmp/my-plan.md --chain codex --json-output
```

State and logs (XDG-compliant):

| Path | Purpose |
|---|---|
| `$XDG_DATA_HOME/autosre/review-state/_state_<hash>_<stem>.json` | Per-plan iteration state — iteration count, previous findings, plan mtime |
| `$XDG_DATA_HOME/autosre/review-log/<hash>_<stem>_<iter>_<provider>.json` | Per-iteration full prompt + response for post-mortems |
| `$XDG_CONFIG_HOME/autosre/guard-rules.yaml` | User-editable guard rules; `autosre hooks-backend init` seeds the default |
| `$XDG_STATE_HOME/autosre/approvals/` | One-shot approval cache for the `ask` guard decision |
| `$XDG_STATE_HOME/autosre/hook-{audit,blocked,errors}.log` | Hook audit trails |

## MCP web research (local)

Zero-API-key web search and fetch. Uses `curl_cffi` for browser TLS
fingerprint impersonation (bypasses Cloudflare/bot detection) and
DuckDuckGo for search:

```bash
autosre mcp setup                         # install local MCP servers (no API keys)
autosre mcp status                        # verify
autosre mcp remove                        # uninstall, restore defaults
autosre mcp setup --brave-api-key KEY     # opt into Brave Search alongside local tools
```

The `autosre claude` flow wires these into Claude Code's MCP config
automatically; for bare `claude`, set them up once via `autosre mcp setup`.

## Dropbox — self-hosted HTTPS file dropbox

Optional. Runs a `filebrowser` backend behind a stealth TLS+HTTP proxy on
a single port. The proxy peeks the first byte of every connection: TLS
gets terminated and gated with an HMAC-signed cookie, plain HTTP gets 301'd
to HTTPS on the same port. The login page is a single password input —
no branding, no CSS, no username field.

Three-step setup:

```bash
cp deploy/example/dropbox.toml ~/.config/autosre/dropbox.toml
$EDITOR ~/.config/autosre/dropbox.toml          # set data_dir + ports

autosre dropbox install --config-file ~/.config/autosre/dropbox.toml
autosre dropbox init    --config-file ~/.config/autosre/dropbox.toml --password-stdin
autosre dropbox start
```

`install` is non-destructive (unit files + `filebrowser` binary only).
`init` is destructive (certs, sqlite DB, password file, HMAC secret) and
refuses to run while the service is active. Add `--system --service-user <name>`
to install into `/etc/systemd/system/` instead of `~/.config/systemd/user/`.

Passwords are **never** passed on the command line — only via interactive
prompt, `--password-stdin`, or a 0600 `--password-file`. See
`autosre dropbox --help` for the full subcommand reference.

## Swarm / agent teams

Launch with `--swarm` to enable multiple Claude Code agents working in
parallel on a shared task list:

```bash
autosre claude --swarm
# Inside Claude Code: "Create an agent team to explore this from 3 angles."
```

Requires Claude Code ≥ 2.1.32. Use `Shift+Down` to cycle between teammates
in your terminal.

## How it works

```
┌────────────┐       ┌────────────────────────┐       ┌─────────────┐
│ Claude     │──────▶│ autosre Anthropic      │──────▶│ Backend     │
│ Code (bare │       │ proxy (:8011)          │       │ Ollama /    │
│ or autosre │       │ translates Messages →  │       │ llamacpp /  │
│ claude)    │       │ OpenAI Chat Completions│       │ vLLM        │
└────────────┘       └────────────────────────┘       └─────────────┘
```

1. `autosre start` launches the backend (and, on a GB10 box with
   meeting-scribe present, the full stack including warmup). Active state
   is written to `~/.local/share/autosre/active.json`.
2. `autosre claude` reads that state, purges cloud Anthropic credentials,
   sets local-only env vars, writes a temp `~/.claude/settings.json`-style
   file and an MCP config, then execs `claude --settings=<temp>`.
3. The review-chain env var is pinned to `local` so the plan-review hook
   runs against the local model, not external services.

Credential isolation — `autosre claude` ensures Claude Code only talks to
localhost:

- `ANTHROPIC_API_KEY` is removed from the environment
- `ANTHROPIC_AUTH_TOKEN` is set to a local dummy value
- `ANTHROPIC_BASE_URL` points at the local proxy
- `CLAUDE_CODE_ATTRIBUTION_HEADER=0` disables billing headers

## Data directories

All paths respect the XDG base-dir spec:

| Path | Purpose |
|------|---------|
| `$XDG_DATA_HOME/autosre/active.json` | Active backend state |
| `$XDG_DATA_HOME/autosre/review-state/` | Per-plan review iteration state |
| `$XDG_DATA_HOME/autosre/review-log/` | Per-iteration review logs |
| `$XDG_DATA_HOME/autosre/dropbox/bin/` | Cached `filebrowser` binary |
| `$XDG_CONFIG_HOME/autosre/guard-rules.yaml` | User-editable Bash-guard rules |
| `$XDG_CONFIG_HOME/autosre/dropbox.toml` | Dropbox config (optional) |
| `$XDG_STATE_HOME/autosre/.dropbox-installed.json` | Dropbox installer sidecar |
| `$XDG_STATE_HOME/autosre/hook-{audit,blocked,errors}.log` | Hook audit trails |

## Deployment

See `deploy/example/` for templated systemd units and an example
`dropbox.toml`. The `autosre.service.template` uses `envsubst` placeholders
so you can render it for any account:

```bash
# From a checkout:
cd deploy/example/systemd
SERVICE_USER=$(id -un) SERVICE_HOME=$HOME REPO_DIR=$(pwd)/../.. \
PYTHON_BIN_DIR=$(dirname "$(command -v python3)") \
NODE_BIN_DIR=$(dirname "$(command -v node 2>/dev/null || echo /usr/bin/true)") \
envsubst < autosre.service.template | sudo tee /etc/systemd/system/autosre.service
sudo systemctl daemon-reload && sudo systemctl enable --now autosre
```

## Development (for forks)

External contributions aren't accepted — see the **License & contributions**
section below. These commands are for your own fork's CI gate:

```bash
pip install -e '.[dev]'

ruff check autosre/ tests/
ruff format --check autosre/ tests/
mypy autosre/
pytest tests/ --ignore=tests/integration -q
autosre precommit                 # sensitive-data scan before commit
```

Mypy runs in `--strict` mode with zero overrides of autosre modules.

## License & contributions

MIT. See [LICENSE](LICENSE). Anyone is free to use, fork, and modify this
software.

This repository is published for consumption, not co-development. Pull
requests, feature requests, and issues from external contributors are **not
accepted**. Fork freely — you own your fork.
