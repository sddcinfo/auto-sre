"""Demo runner — orchestrates demo execution with real-time status."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from autosre.demos.audience import AudienceProfile
    from autosre.demos.scenario import DemoPhase, DemoScenario


class DemoRunner:
    """Orchestrates demo scenario execution with timing and audience adaptation."""

    def __init__(
        self,
        scenario: DemoScenario,
        audience: AudienceProfile | None = None,
    ) -> None:
        self.scenario = scenario
        self.audience = audience
        self._current_phase: int = -1
        self._start_time: float | None = None

    def preflight(self) -> tuple[bool, list[str]]:
        """Check all prerequisites for the demo.

        Validates: backend available, model loadable, cluster healthy (if needed).
        """
        issues: list[str] = []

        # Check scenario has phases
        if not self.scenario.phases:
            issues.append("Scenario has no phases defined")

        # Check model is in the vLLM model list
        from autosre.backends.vllm import VllmBackend

        if self.scenario.model not in VllmBackend.models:
            issues.append(f"Model '{self.scenario.model}' not in VllmBackend.models")

        # Check cluster requirement
        if self.scenario.cluster_required:
            try:
                from autosre.backends.vllm_config import VllmConfig

                config = VllmConfig.load()
                if not config.is_cluster:
                    issues.append(
                        f"Scenario requires cluster but only {len(config.nodes)} node(s) configured"
                    )
            except FileNotFoundError:
                issues.append("vLLM config not found (cluster required)")

        return len(issues) == 0, issues

    def run(self, skip_preflight: bool = False) -> bool:
        """Execute the demo scenario phase by phase.

        Returns True if all phases completed.
        """
        if not skip_preflight:
            ok, issues = self.preflight()
            if not ok:
                click.secho("Preflight FAILED:", fg="red")
                for issue in issues:
                    click.echo(f"  - {issue}")
                return False

        self._start_time = time.time()

        click.echo()
        click.secho(f"{'═' * 60}", fg="cyan")
        click.secho(f"  Demo: {self.scenario.name}", fg="cyan", bold=True)
        click.secho(f"  {self.scenario.description}", fg="cyan")
        if self.audience:
            click.secho(
                f"  Audience: {self.audience.name} ({self.audience.description})", fg="cyan"
            )
        click.secho(
            f"  Duration: {self.scenario.total_minutes} min | Model: {self.scenario.model}",
            fg="cyan",
        )
        click.secho(f"{'═' * 60}", fg="cyan")

        for i, phase in enumerate(self.scenario.phases):
            self._current_phase = i
            if not self._run_phase(phase, i, len(self.scenario.phases)):
                return False

        elapsed = time.time() - self._start_time
        click.echo()
        click.secho(f"{'═' * 60}", fg="green")
        click.secho(f"  Demo complete! ({elapsed / 60:.1f} min elapsed)", fg="green", bold=True)
        click.secho(f"{'═' * 60}", fg="green")
        return True

    def _run_phase(self, phase: DemoPhase, index: int, total: int) -> bool:
        """Execute a single demo phase."""

        click.echo()
        click.secho(f"{'─' * 60}", fg="blue")
        click.secho(
            f"  Phase {index + 1}/{total}: {phase.name} ({phase.duration_minutes} min)",
            fg="blue",
            bold=True,
        )
        click.echo(f"  {phase.description}")
        click.secho(f"{'─' * 60}", fg="blue")

        # Show talking points (audience-adapted if available)
        self._print_talking_points(phase)

        # Show commands to run
        if phase.commands:
            click.echo()
            click.secho("  Commands:", fg="yellow")
            for cmd in phase.commands:
                click.echo(f"    $ {cmd}")

        # Show success criteria
        if phase.success_criteria:
            click.echo()
            click.secho("  Success criteria:", fg="yellow")
            for criterion in phase.success_criteria:
                click.echo(f"    [ ] {criterion}")

        # Wait for phase duration (or let presenter advance)
        click.echo()
        click.echo(
            f"  Press Enter to advance to next phase (or wait {phase.duration_minutes} min)..."
        )

        return True

    def _print_talking_points(self, phase: DemoPhase) -> None:
        """Print audience-adapted talking points for a phase."""

        points = list(phase.talking_points)

        # Overlay audience-specific talking points
        if self.audience and phase.name in self.audience.talking_points:
            audience_points = self.audience.talking_points[phase.name]
            points = [*audience_points, *points]

        if points:
            click.echo()
            click.secho("  Talking points:", fg="green")
            for point in points:
                click.echo(f"    - {point}")

    def adapt_audience(self, audience: AudienceProfile) -> None:
        """Switch audience profile mid-demo for real-time adaptation."""
        self.audience = audience
        click.secho(f"  Audience switched to: {audience.name}", fg="cyan")

    def status(self) -> dict[str, object]:
        """Current demo status."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        phase_name = (
            self.scenario.phases[self._current_phase].name
            if 0 <= self._current_phase < len(self.scenario.phases)
            else "not started"
        )
        return {
            "scenario": self.scenario.name,
            "current_phase": phase_name,
            "phase_index": self._current_phase,
            "total_phases": len(self.scenario.phases),
            "elapsed_minutes": elapsed / 60,
            "total_minutes": self.scenario.total_minutes,
            "audience": self.audience.name if self.audience else None,
        }
