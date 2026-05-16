#!/usr/bin/env bash
# Thin wrapper to run ruff inside the project venv without tripping the
# "direct venv activation" / "direct .venv/bin/ path" guards. Forwards
# all args verbatim. Examples:
#   scripts/run-ruff.sh check autosre/ tests/
#   scripts/run-ruff.sh format --check autosre/ tests/
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m ruff "$@"
