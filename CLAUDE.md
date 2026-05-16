# CLAUDE.md — auto-sre

## Project Overview
Local LLM server management for Claude Code. Python CLI (Click), httpx for HTTP, PyYAML for config.
No Anthropic API key required — all backends use native Anthropic Messages API locally.

## Running Locally
This project runs Claude Code with fully local LLM backends (Ollama, llama.cpp, vLLM on GB10).
Built-in WebFetch/WebSearch are replaced by local MCP servers (`autosre-mcp-fetch`, `autosre-mcp-search`).

## Development
- Python 3.11+, Hatchling build system
- Install: `pip install -e '.[dev]'`
- Lint: `ruff check autosre/ tests/`
- Format: `ruff format autosre/ tests/`
- Type check: `mypy autosre/`
- Test: `pytest`

## Code Conventions
- 100-char line length, double-quoted strings
- Strict mypy, comprehensive ruff rule set
- Lazy imports in CLI commands (from X import Y inside function body)
- Click for CLI, httpx for HTTP, PyYAML for config
- curl_cffi for web fetching (browser TLS fingerprint impersonation)
- XDG-compliant data storage: ~/.local/share/autosre/
- Type annotations on all functions

## Key Architecture
- `autosre/backends/` — Backend ABC + Ollama, llama.cpp, vLLM implementations
- `autosre/backends/recipes/` — YAML model configs (add new models here, no code changes)
- `autosre/backends/anthropic_proxy.py` — Anthropic Messages API → OpenAI Chat Completions translator (the proxy that makes local backends look like Anthropic's API to Claude Code)
- `autosre/backends/vllm_priority_preempt.py` — PYTHONSTARTUP-installed sitecustomize hook that adds priority preemption to vLLM's V1 scheduler (evicts low-priority running requests to admit high-priority waiting ones)
- `autosre/mcp_servers/` — Local MCP servers:
  - `fetch` / `search` — web fetch (curl_cffi) + web search (DuckDuckGo)
  - `capabilities` — command-discovery server that introspects the live autosre click group so Claude can search autosre commands with `list_modules`, `search_commands`, `get_command`
- `autosre/cli.py` — All Click commands (command groups: `setup`, `start`, `stop`, `claude`, `hooks`, `mcp`, `swarm`, `dropbox`, `cluster`, `provision`, `models`, `review`, `keys`, `ssh`, `configure`, `demo`, `eval`, `hooks-backend`, `ui`, `metrics`, `bench`, `backends`)
- `autosre/dropbox/` — Self-hosted HTTPS file dropbox subsystem:
  - `config.py` — `DropboxConfig` dataclass + single `.load()` resolver (defaults → TOML → env, in that order of precedence)
  - `proxy.py` — TLS/HTTP sniffing proxy: peeks the first byte, terminates TLS + HMAC-cookie gate for TLS traffic, 301-redirects plain HTTP on the same port
  - `installer.py` — OS-aware systemd installer with bus-operability probing (explicitly avoids `is-system-running` which fails on degraded systems) and `pwd`-based user resolution (never reads `$USER` / `$HOME`)
  - `state_init.py` — destructive state bootstrap: self-signed cert generation, filebrowser DB init (noauth mode), HMAC secret seeding
  - `credentials.py` — password handling: interactive prompt / `--password-stdin` / `--password-file` only; no `--password` literal flag ever exists
  - `filebrowser.py` — pinned download + SHA256 verification for the upstream `filebrowser` binary
- `autosre/infra/` — SSH, node types, config for remote GB10 nodes
- `autosre/swarm/` — Agent team orchestration
- `autosre/paths.py` — XDG-compliant path helpers (`data_dir`, `config_dir`, `state_dir`, `cache_dir` + subsystem-specific subdirs)
- `autosre/review/` — Plan-review loop: `chain.py` (provider-chain executor), `cli_plan.py` (iteration state + prompt), `_local_provider_runner.py` (OpenAI-compat HTTP client for local backends)
- `autosre/hooks_backend/` — Guard rules (`guard.py`) + session-end checklist (`stop_check.py`) + packaged rules YAML
- `autosre/hooks_installer.py` — Claude Code settings-file merger for bare `claude` mode (used by `autosre hooks install` to drop the managed block into `~/.claude/settings.json`)
- `autosre/claude_hooks/` — Claude Code hook scripts invoked via `sys.executable -m autosre.claude_hooks.<module>`. Shared I/O plumbing lives in `autosre/claude_hooks/_io.py`
- `deploy/example/` — Templated systemd units + `dropbox.toml` sample; consume via `envsubst`
- `.github/workflows/ci.yml` — CI gate: ruff check, ruff format --check, mypy (strict, no autosre-module overrides), pytest
- `autosre/perf/` — Concurrent-workload regression harness:
  - `harness.py` — `RunConfig`, `stream_one`, phase runners (`run_isolated`, `run_contention`, `run_warmup`), `run_proxy_sanity`, top-level `run()`
  - `workloads.py` — `TRANSLATION_WORKLOAD` (priority=-10, JA<->EN corpus, `enable_thinking=false`) and `CODING_WORKLOAD` (priority=10, ~800-token system prompt + 20 tool schemas)
  - `baseline.py` — `Baseline`, `compare()` with ratio-based tolerances + absolute SLOs, `save_baseline()`, `load_baseline()`
  - `report.py` — stdout table renderer + markdown generator
- `benchmarks/baselines/` — Committed baselines (`<name>.json` + `<name>.md`). `gb10_qwen36_fp8_baseline_parity` = parity reference, `gb10_qwen36_fp8_flashinfer` = current production (with `_8192` and `_16384` `max_num_batched_tokens` variants)
- Tests in `tests/` — pytest with monkeypatch/MagicMock patterns

## Performance benchmarking

The perf harness validates that vLLM serves both Claude Code (coding workload) and meeting-scribe (translation workload) within the committed performance envelope. It is the gate for any change to vLLM flags, recipe tuning, model upgrades, or image rebuilds.

### Quick reference

```bash
# Before benchmarking: quiesce the GPU
systemctl --user stop meeting-scribe.service
meeting-scribe gb10 down

# Run against committed baseline (exit 0=clean, 1=warn, 2=fail)
autosre perf run

# Run with saturation to exercise priority preemption
autosre perf run --saturate-slots

# Save a new baseline after intentional config changes
autosre perf run --no-compare       # inspect numbers first
autosre perf save-baseline <name>   # commit when satisfied
git add benchmarks/baselines/<name>.{json,md}

# Restore after benchmarking
systemctl --user start meeting-scribe.service
```

### Current vLLM configuration and rationale

Recipe: `autosre/backends/recipes/qwen3.6-35b-a3b-fp8.yaml` — the single canonical vLLM recipe (production-locked 2026-04-25 alongside the Qwen3.5 → 3.6 migration; the `vllm-qwen35-v2` custom albond fork was retired with 3.5; the `qwen3.6-fp8-nightly` and `qwen3-coder` siblings were retired 2026-04-30 alongside the customer-install simplification).

| Flag | Value | Why |
|------|-------|-----|
| `gpu_memory_utilization` | `0.70` | 128 GB unified memory. 0.70 gives ~77 GB for vLLM (weights + KV cache). Leaves ~17 GB for CPU/OS when scribe is running alongside. |
| `max_num_seqs` | `8` | Slots for concurrent requests. Was briefly 12 (2026-04 audit), reverted after the perf harness showed 2.5x coding TTFT regression from splitting KV blocks across too many slots with chunked-prefill. 8 is sufficient for the default workload (coding_concurrency=2, translation_rps=2). |
| `max_num_batched_tokens` | `4096` | Chunked-prefill budget. Caps per-step token batch so a single 38K-token eval/review prefill can't stall live decode streams. Drops aggregate throughput ~5% but halves tail latency for concurrent translation. |
| `--enable-prefix-caching` | set | Reuses the coding workload's ~800-token system prompt + 20 tool schemas across requests. Hit rate ~84%. Removing it caused 8.6x coding TTFT regression in harness validation. |
| `--enable-chunked-prefill` | set | Required for `max_num_batched_tokens` to take effect. Interleaves prefill chunks with decode steps. |
| `--scheduling-policy=priority` | set | Priority-based scheduling. Translation (priority=-10) preempts coding (priority=10) when max_num_seqs saturates. Requires the `vllm_priority_preempt.pth` monkey-patch because upstream chunked-prefill disables native priority-preemption (vllm#10101). |
| `--load-format=fastsafetensors` | set | 19x faster model loading on GB10. |
| `--quantization=fp8` | set | Native Qwen-published FP8 checkpoint — no calibration needed. The prior 3.5-INT4 stack used `--quantization=inc` (Intel Neural Compressor) for AutoRound weights; that path is gone with 3.5. |
| `--attention-backend=flashinfer` | set | Required pairing with FP8 on SM121 per NVIDIA forum thread 366822. Codified as `attention_backend: flashinfer` in the recipe yaml (was missing pre-2026-04-30 — local autosre had it via stale process state, customer GB10 didn't, ~20% concurrent-translate latency penalty + 8 GB extra VRAM). |
| `--kv-cache-dtype=auto` | set | FP8 KV cache is unstable on SM121 (vllm#26646). |
| No `--enforce-eager` | (absent) | CUDA graphs enabled. Stock vllm/vllm-openai supports SM121 graph capture on the 2026-04+ images. Gives +80% translation TPS, +46% coding TPS vs eager mode. First-request TTFT pays ~1s graph-capture cost (absorbed by the harness warmup phase). |

### Container env vars (codified in `_build_runtime_env`)

`VllmBackend._build_runtime_env(recipe)` is the single source of truth for the env vars passed to the vLLM container.  Both `_start_local` (dev box, container `autosre-vllm-local`) and `_start_solo` (customer GB10 via SSH, container `autosre-vllm-head`) funnel through it so the two paths produce bit-for-bit identical container env. Pre-2026-04-30 they diverged — `_start_solo` only forwarded `recipe.get("env", {})` and silently dropped HF offline mode, HF_HUB_CACHE, and NVIDIA_DISABLE_REQUIRE.

| Env var | Value | Why |
|---|---|---|
| `HF_HUB_OFFLINE` | `1` | Block any network I/O at boot — otherwise the container races systemd-resolved at cold boot and crash-loops on `Temporary failure in name resolution`. |
| `TRANSFORMERS_OFFLINE` | `1` | Companion to `HF_HUB_OFFLINE` for the transformers library. |
| `HF_HUB_CACHE` | `/data/huggingface/hub` | Direct the HF library at the canonical bind-mount where `meeting-scribe gb10 pull-models` places weights. Without this the library defaults to `/root/.cache/huggingface/hub` and a fresh container crash-loops with `LocalEntryNotFoundError` if the user-cache mount doesn't have the model yet. |
| `NVIDIA_DISABLE_REQUIRE` | `1` | Disable the upstream image's strict `NVIDIA_REQUIRE_CUDA` envelope; some legitimate driver versions on GB10 fall outside the baked-in range. |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` | Pre-sizes 8 × 4 MiB cuBLAS workspaces — required to prevent the `CUBLAS_STATUS_INTERNAL_ERROR` cascade observed 2026-04-18 23:45 under burst concurrent prefill. (Set via the recipe's `env:` block.) |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `1` | NVIDIA DGX Spark thread 366822 recommendation; companion to the cuBLAS workspace config. (Set via the recipe's `env:` block.) |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | Allow `max_model_len=262144` despite vLLM's default safety cap. (Set via the recipe's `env:` block.) |
| `HF_TOKEN` | `$HF_TOKEN` from operator environ | Forwarded only when present in the operator's environment; not strictly required at runtime since the container is fully offline, but kept available for the `pull-models` path that runs at install time. |

### Boot-time recipe-parity sentinel

`VllmBackend.warn_on_recipe_drift(recipe, api_port)` runs after `_wait_for_vllm` returns OK on both `_start_local` and `_start_solo`. It diffs the live `vllm serve ...` cmdline (read from `ps -eo cmd`) against what `_build_vllm_serve_cmd(recipe)` would have produced and emits a WARNING per mismatch.  Catches three drift classes the recipe-edit guard alone can't see: host hand-edited cmdline, stale process from older recipe state, and missing `extra_args`.  Deliberately non-fatal; the CI test in `tests/test_vllm.py::TestRecipeParityCheck` is the strict gate.

### Deferred tuning items

- **MTP speculative decoding** (`--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":1}'`) — requires vLLM >= 0.16 with qwen3_next_mtp. May degrade prefix cache hit rate (vllm#38182). Stay at num_speculative_tokens=1 (bugs at 2: vllm#36498, #36872).
- **`--quantization=auto_round`** — dispatches to gptq_marlin on Blackwell. High compat risk with custom fork. Research-grade, not a flag flip.

## Plan-review loop — two invocation paths

The same `autosre/claude_hooks/*.py` scripts fire from two different launch paths. They share a single source of truth for the planned hook list in `autosre/hooks_installer.py:_planned_entries()`.

### Bare `claude` (online mode)

Run once per machine:
```
autosre hooks install
```

Idempotently merges the 11 autosre hook entries into `~/.claude/settings.json` alongside any user-owned settings (`permissions`, `enabledPlugins`, etc.) and records the installed set in `~/.claude/.autosre-hooks-installed.json` so `autosre hooks uninstall` can remove them surgically. The hook commands embed the absolute path of the python interpreter autosre is installed under, so re-running `autosre hooks install` after a mise Python bump refreshes the paths.

Bare `claude` then reads `~/.claude/settings.json` on every launch and gets the full hook chain with **zero additional flags**. The review provider chain defaults to `["codex", "local", "gemini", "claude"]` — Codex (gpt-5.4 xhigh) is the primary reviewer, so plans produced by bare `claude` get reviewed by an external model.

### `autosre claude` (offline / local mode)

When invoked, `autosre claude`:

1. Starts the local vLLM (or Ollama / llamacpp) backend and its Anthropic proxy.
2. Writes two **per-launch** temp files (keyed on a fresh `AUTOSRE_RUN_ID` UUID so concurrent launches don't collide):
   - `/tmp/autosre-settings-<uuid>.json` — Claude Code settings with top-level `PreToolUse`, `PostToolUse`, `Stop`, `UserPromptSubmit`, `PreCompact`, `SubagentStart` keys pointing at our hook scripts under `autosre/claude_hooks/`, plus `permissions`/`model`/`env` overrides that route all model calls to the local Anthropic proxy.
   - `/tmp/autosre-mcp-<uuid>.json` — MCP config with `autosre-fetch`, `autosre-search`, `autosre-capabilities` servers.
3. Passes them via `--settings=<file>` and `--mcp-config=<file>`, then `os.execvpe("claude", ...)`.
4. Sets `AUTOSRE_REVIEW_CHAIN=local` in the environment so the review chain runs the local model against itself (no external Codex/Gemini/Claude CLI calls). Intended for offline / airgapped work.

Both paths share the same hook scripts, so every feature below (plan review, bash guard, session checks, audit/telemetry) works identically in both modes. The only difference is where the review chain runs: `local` for `autosre claude`, `codex → local → gemini → claude` for bare `claude`.

**The plan→review→update→re-review loop** fires on `PreToolUse(ExitPlanMode)`:
- `autosre/claude_hooks/pretooluse_plan_review.py` receives the plan file path from Claude Code.
- Shells out to `autosre review plan <path> --json-output`.
- `autosre review plan` loads per-plan iteration state (keyed on `sha256(abs_path)[:12]_<stem>` — fixes upstream bug of cross-repo collisions via `plan.md` stem), builds a prompt (`INITIAL_REVIEW_PROMPT` for iteration 0, `RE_REVIEW_PROMPT` with previous findings for iteration ≥1), and runs the provider chain.
- Chain default: `["codex", "local", "gemini", "claude"]`. `codex` uses `codex exec -c model="gpt-5.4" -c model_reasoning_effort="xhigh"` as the **primary reviewer**; `local` is the offline fallback via the vLLM runner; gemini/claude CLIs are further fallback.
- The hook translates the chain result into Claude Code hook JSON: P0/P1 findings → `permissionDecision=deny` with formatted findings as `additionalContext` so Claude re-enters plan mode with the feedback, updates the plan, and ExitPlanMode fires the loop again. Clean / P2-only / chain-failure all allow.

**State/log paths** (XDG-respecting):
- `$XDG_DATA_HOME/autosre/review-state/_state_<hash>_<stem>.json` — iteration count, previous findings, plan mtime.
- `$XDG_DATA_HOME/autosre/review-log/<hash>_<stem>_<iter>_<provider>.json` — per-iteration full prompt + response for post-mortems.
- `$XDG_CONFIG_HOME/autosre/guard-rules.yaml` — user-editable guard rules (`autosre hooks-backend init` installs the packaged default).
- `$XDG_STATE_HOME/autosre/approvals/` — one-shot approval cache for the `ask` decision type.
- `$XDG_STATE_HOME/autosre/hook-{audit,blocked,errors}.log` — hook audit trails.
