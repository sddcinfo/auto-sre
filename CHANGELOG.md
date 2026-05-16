# Changelog

All notable changes to auto-sre are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/).

## [2.0.0] - 2026-05-02

First versioned release. The 2.0 line marks the point at which
auto-sre is fully self-contained: zero references to any private
operator monorepo, generic enough that any GB10 owner can install
and use it end-to-end.

### Public-release decoupling

- `Provisioner` now does **only generic GB10 prep** — network,
  Docker with the nvidia runtime, `/data` partition, HuggingFace
  cache, performance mode, firewall, vLLM image build, validate,
  pre-wipe backup, post-wipe restore. Universal to any GB10 owner.
- 4 operator-specific methods (`clone_sddcinfo_repo`,
  `push_age_key`, `run_bootstrap`, `install_claude_hooks`) +
  3 helpers (`sync_tmux_config`, `_local_sddcinfo_root`,
  `_rsync_tree_to_target`) **moved out** of the public class.
  Operator flows that need user-parity steps (matching dotfiles,
  internal repo clones, custom bootstrap scripts) layer them via
  subclass.
- Regression fence: `test_provision_pipeline_runs_only_generic_steps`
  asserts the 7 sddcinfo-specific symbols are absent from the public
  `Provisioner` — drift back into the public surface fails CI.
- `tui._dispatch_cmd` / `tui._disk_usage` / `demo._scribe_src` no
  longer hardcode `~/sddcinfo/...` paths; honor
  `$AUTOSRE_DISPATCH_CWD` / `$MEETING_SCRIBE_REPO` env overrides
  with sensible `~/meeting-scribe` fallbacks (the layout
  meeting-scribe's bootstrap.sh produces).

### Production-locked vLLM stack

- **`Qwen/Qwen3.6-35B-A3B-FP8`** as the canonical translation +
  coding model, served via vLLM 0.19.x on GB10 (SM121 Blackwell).
- Recipe lock at `autosre/backends/recipes/qwen3.6-35b-a3b-fp8.yaml`
  — the single source of truth; the `vllm-qwen35-v2` custom albond
  fork was retired with 3.5; the `qwen3.6-fp8-nightly` and
  `qwen3-coder` siblings were retired alongside the customer-install
  simplification.
- Settings: `gpu_memory_utilization=0.70`, `max_num_seqs=8`,
  `max_num_batched_tokens=4096`, `--enable-prefix-caching`,
  `--enable-chunked-prefill`, `--scheduling-policy=priority`,
  `--load-format=fastsafetensors`, `--quantization=fp8`,
  `--attention-backend=flashinfer`, `--kv-cache-dtype=auto`,
  CUDA graphs on. See `CLAUDE.md` § "Performance benchmarking" for
  the why-each-flag table.
- **Container memory cap** at 80g (no swap) on `_start_local`
  via `--memory $AUTOSRE_VLLM_MEM_LIMIT` to prevent host OOM kills
  observed 2026-04-30.

### Plan-review loop

- Two invocation paths share one source of truth:
  - **Bare `claude`** (online): `autosre hooks install` writes the
    11 hook entries to `~/.claude/settings.json`. Default review
    chain: `codex → local → gemini → claude`.
  - **`autosre claude`** (offline / airgapped): per-launch
    `/tmp/autosre-settings-<uuid>.json` + `/tmp/autosre-mcp-<uuid>.json`
    keyed on `$AUTOSRE_RUN_ID`. Review chain: `local`-only.
- Codex (`gpt-5.4`, `model_reasoning_effort=xhigh`) is the primary
  reviewer. P0/P1 findings → `permissionDecision=deny` so Claude
  re-enters plan mode with the feedback. Loop until clean / P2-only
  / chain-failure.

### MCP servers + capabilities

- `autosre-fetch` (`curl_cffi`-based web fetch with browser TLS
  fingerprint impersonation), `autosre-search` (DuckDuckGo), and
  `autosre-capabilities` (introspects the live autosre click group
  so Claude can search subcommands).

### Performance harness + baselines

- `autosre/perf/` — `RunConfig`, phase runners
  (`run_isolated`, `run_contention`, `run_warmup`),
  `run_proxy_sanity`, ratio-based + absolute-SLO comparison.
- Baselines committed at `benchmarks/baselines/` with
  `gb10_qwen36_fp8_baseline_parity` (parity reference) and
  `gb10_qwen36_fp8_flashinfer` (production, with `_8192` and
  `_16384` `max_num_batched_tokens` variants).
- 880 unit tests covering the proxy, hooks, MCP servers, recipe
  parity check, vllm config, perf harness, and provisioner.

### Self-hosted file dropbox

- `autosre/dropbox/` — TLS-sniffing proxy that peeks the first byte,
  terminates TLS + HMAC-cookie gate for TLS traffic, 301-redirects
  plain HTTP on the same port. OS-aware systemd installer.
  Pinned-SHA256 download of upstream `filebrowser`. Password handling
  via interactive prompt / `--password-stdin` / `--password-file`
  only — no `--password` literal flag exists.
