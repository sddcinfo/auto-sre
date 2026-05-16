"""Configuration I/O helpers for auto-sre."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# XDG-compliant data directory
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "autosre"


def data_path(filename: str) -> Path:
    """Get a path within the autosre data directory.

    Creates the directory if it doesn't exist.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / filename


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and return its contents as a dict.

    Returns empty dict if file doesn't exist or is empty.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    if not text.strip():
        return {}
    result = yaml.safe_load(text)
    if not isinstance(result, dict):
        return {}
    return result


def save_yaml(path: Path, data: dict[str, Any]) -> None:
    """Save a dict to a YAML file.

    Creates parent directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
