"""Embedded vLLM recipes for GB10 NVFP4 models.

Recipes are YAML files derived from spark-vllm-docker,
curated for Dell Pro Max GB10 (SM121a) hardware.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from autosre.infra.config import load_yaml

RECIPES_DIR = Path(__file__).parent


def list_recipes() -> list[str]:
    """List available recipe short names."""
    return sorted(p.stem for p in RECIPES_DIR.glob("*.yaml"))


def load_recipe(name: str) -> dict[str, Any]:
    """Load a recipe by short name or model key.

    Args:
        name: Recipe short name (e.g., "nemotron-3-nano-nvfp4") or
              model key (e.g., "nemotron-nano").

    Returns:
        Recipe dict with all fields.

    Raises:
        FileNotFoundError: If recipe doesn't exist.
    """
    # Try direct filename match first
    path = RECIPES_DIR / f"{name}.yaml"
    if path.exists():
        return load_yaml(path)

    # Try matching by model_key field in all recipes
    for recipe_path in RECIPES_DIR.glob("*.yaml"):
        recipe = load_yaml(recipe_path)
        if recipe.get("model_key") == name:
            return recipe

    available = ", ".join(list_recipes())
    msg = f"Recipe '{name}' not found. Available: {available}"
    raise FileNotFoundError(msg)


def get_recipe_for_model(model_key: str) -> dict[str, Any]:
    """Get the recipe for a model key.

    Model keys are the short names used in VllmBackend.models
    (e.g., "nemotron-nano", "nemotron-super").

    Raises:
        FileNotFoundError: If no recipe matches the model key.
    """
    return load_recipe(model_key)
