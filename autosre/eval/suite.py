"""EvalSuite schema + YAML loader.

A suite is a declarative spec for one kind of eval run. It tells the
runner how many agents to spawn, what role each has, what categories
of finding to emit, and which files to read. Suites live as YAML under
``autosre/eval/suites/`` so adding a new one is a pure-data change.

The runner formats the absolute path of the per-run findings file into
the suite's ``initial_prompt`` template via ``{findings_file}``. The
template is otherwise rendered as-is — no f-string interpolation,
no Jinja, no surprises.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

# Imported at runtime (not TYPE_CHECKING) because pydantic needs to
# resolve the type of the ``category`` field during model validation.
from autosre.eval.findings import Category  # noqa: TC001

SUITES_DIR = Path(__file__).resolve().parent / "suites"


SuiteTarget = Literal["repo", "globs"]


class EvalSuite(BaseModel):
    """A single eval suite loaded from YAML."""

    name: str
    description: str
    category: Category
    num_agents: int = Field(..., ge=1, le=10)
    agent_roles: list[str]
    system_prompt: str
    initial_prompt: str
    target_globs: list[str] | None = None
    max_tokens_per_agent: int = 16000
    timeout_seconds: int = 1800

    @field_validator("agent_roles", mode="after")
    @classmethod
    def _roles_match_count(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("agent_roles must not be empty")
        return v

    def model_post_init(self, context: object, /) -> None:  # noqa: ARG002
        if len(self.agent_roles) != self.num_agents:
            raise ValueError(
                f"suite {self.name}: agent_roles has "
                f"{len(self.agent_roles)} entries but num_agents={self.num_agents}"
            )
        if "{findings_file}" not in self.initial_prompt:
            raise ValueError(
                f"suite {self.name}: initial_prompt must contain "
                f"{{findings_file}} placeholder so the runner can inject "
                f"the absolute findings-file path"
            )

    def render_prompt(self, *, findings_file: Path) -> str:
        """Return the initial prompt with placeholders substituted."""
        roles_block = "\n".join(
            f"- Agent {i + 1}: {role}" for i, role in enumerate(self.agent_roles)
        )
        return self.initial_prompt.format(
            findings_file=str(findings_file),
            agent_roles=roles_block,
            num_agents=self.num_agents,
            suite_name=self.name,
            category=self.category,
        )


def load_suite(path: Path) -> EvalSuite:
    """Load one suite from a YAML file."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise TypeError(f"{path}: suite YAML must be a mapping")
    return EvalSuite.model_validate(data)


def load_all_suites(directory: Path | None = None) -> dict[str, EvalSuite]:
    """Load every ``*.yaml`` under ``directory`` (default: builtin suites dir)."""
    directory = directory or SUITES_DIR
    out: dict[str, EvalSuite] = {}
    if not directory.exists():
        return out
    for path in sorted(directory.glob("*.yaml")):
        suite = load_suite(path)
        out[suite.name] = suite
    return out
