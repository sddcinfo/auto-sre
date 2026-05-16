#!/bin/bash
# Auto-SRE installer
# Usage: ./install.sh [--yes]
#        --yes, -y: Non-interactive mode (skip prompts)

set -e

# Parse arguments
AUTO_YES=false
SKIP_START=false
for arg in "$@"; do
    case $arg in
        --yes|-y)
            AUTO_YES=true
            ;;
        --skip-start)
            SKIP_START=true
            ;;
    esac
done

# Detect if running non-interactively (piped or no tty)
if [[ ! -t 0 ]] || [[ ! -t 1 ]]; then
    AUTO_YES=true
fi

echo "=== Auto-SRE Installer ==="
echo ""

# Check for Ollama
if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "unknown")
    echo "Ollama: $OLLAMA_VERSION"
else
    echo "Ollama: not found"
    echo "Install from: https://ollama.ai"
fi

# Check for llama-server
if command -v llama-server &> /dev/null; then
    echo "llama-server: found"
else
    echo "llama-server: not found (optional: brew install llama.cpp)"
fi
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required"
    echo "Install Python 3.11+ and try again"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYTHON_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")

echo "Python: $PYTHON_VERSION"

# Check Python version (need 3.11+)
if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 11 ]]; then
    echo "ERROR: Python 3.11+ is required (found $PYTHON_VERSION)"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

# Check if we're in the auto-sre directory
if [[ ! -f "$SCRIPT_DIR/pyproject.toml" ]]; then
    echo "ERROR: Run this script from the auto-sre directory"
    echo "  cd auto-sre && ./install.sh"
    exit 1
fi

cd "$SCRIPT_DIR"

# Install dependencies using python3 -m pip
echo ""
echo "Installing autosre CLI..."
python3 -m pip install -q click httpx 2>/dev/null || pip install -q click httpx

# Install package in editable mode
python3 -m pip install -q -e . 2>/dev/null || pip install -q -e .

echo ""
echo "Running setup..."
autosre setup

echo ""
echo "Setting up local web research MCP servers..."
if command -v claude &> /dev/null; then
    autosre mcp setup
else
    echo "  Skipped (claude CLI not found — install Claude Code first, then run 'autosre mcp setup')"
fi

echo ""
echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo ""
echo "Quick start:"
echo ""
echo "  autosre start      # Start the LLM server"
echo "  autosre test       # Verify it's working"
echo "  autosre claude     # Launch Claude Code"
echo ""
echo "Other commands:"
echo ""
echo "  autosre status     # Check server status"
echo "  autosre stop       # Stop servers"
echo "  autosre backends   # List available backends"
echo ""

# Offer to start now (only in interactive mode)
if [[ "$SKIP_START" != "true" && "$AUTO_YES" != "true" && -t 0 && -t 1 ]]; then
    read -p "Start the server now? [y/N] " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        autosre start
    fi
fi
