#!/usr/bin/env bash
# Thin wrapper to run mypy inside the project venv.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m mypy "$@"
