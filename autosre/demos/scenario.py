"""Demo scenario definitions loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autosre.infra.config import load_yaml

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


@dataclass
class DemoPhase:
    """A phase within a demo timeline."""

    name: str
    duration_minutes: int
    description: str
    commands: list[str] = field(default_factory=list)
    talking_points: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DemoPhase:
        return cls(
            name=str(data["name"]),
            duration_minutes=int(data.get("duration_minutes", 5)),
            description=str(data.get("description", "")),
            commands=list(data.get("commands", [])),
            talking_points=list(data.get("talking_points", [])),
            success_criteria=list(data.get("success_criteria", [])),
        )


@dataclass
class DemoScenario:
    """Complete demo scenario definition."""

    name: str
    description: str
    audience: str
    total_minutes: int
    model: str
    backend: str
    cluster_required: bool
    phases: list[DemoPhase] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DemoScenario:
        phases = [DemoPhase.from_dict(p) for p in data.get("phases", [])]
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            audience=str(data.get("audience", "any")),
            total_minutes=int(data.get("total_minutes", 45)),
            model=str(data.get("model", "nemotron-nano")),
            backend=str(data.get("backend", "vllm")),
            cluster_required=bool(data.get("cluster_required", False)),
            phases=phases,
        )

    @classmethod
    def load(cls, name: str) -> DemoScenario:
        """Load a scenario by name from the scenarios directory.

        Raises:
            FileNotFoundError: If scenario doesn't exist.
        """
        path = SCENARIOS_DIR / f"{name}.yaml"
        if not path.exists():
            available = list_scenarios()
            msg = f"Scenario '{name}' not found. Available: {', '.join(available)}"
            raise FileNotFoundError(msg)
        data = load_yaml(path)
        return cls.from_dict(data)

    @property
    def computed_duration(self) -> int:
        """Total duration from phases (may differ from total_minutes)."""
        return sum(p.duration_minutes for p in self.phases)


def list_scenarios() -> list[str]:
    """List available scenario names."""
    return sorted(p.stem for p in SCENARIOS_DIR.glob("*.yaml"))
