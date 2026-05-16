"""Auto-SRE eval harness.

This package provides a provider-agnostic pipeline for running Claude Code
agent swarms against a target repository, capturing every turn in a
normalized schema, extracting structured findings, and (as a separate step)
diffing two independent runs bidirectionally.

Subpackages and modules:

- :mod:`autosre.eval.snapshot`  — immutable, filesystem-enforced snapshots
- :mod:`autosre.eval.capture`   — stream-json + proxy log → ``TurnRecord``
- :mod:`autosre.eval.findings`  — ``Finding`` schema + id + normalization
- :mod:`autosre.eval.suite`     — YAML suite loader + ``EvalSuite`` schema
- :mod:`autosre.eval.runner`    — orchestrates a single-provider run
- :mod:`autosre.eval.extract`   — findings extraction pipeline with fallbacks
- :mod:`autosre.eval.differ`    — bipartite matching + strict refusal rules
- :mod:`autosre.eval.judge`     — Opus LLM-as-judge subprocess wrapper
- :mod:`autosre.eval.report`    — markdown / json report rendering
- :mod:`autosre.eval.lenient_json` — tiny JSON cleaner (no new deps)
"""

from __future__ import annotations
