"""Semantic boundaries for Planner decision components.

The Planner owns scientific claims and dependencies. Execution mode and tool
choice belong to the Researcher's local graph, so this module deliberately
does not classify global nodes as Search, Aggregate, or Answer.
"""

from __future__ import annotations

from typing import Iterable


def validate_component_requirements(*, evidence_needed: Iterable[str], component_id: str) -> None:
    """Validate requirement shape without closing it over current tools."""

    values = list(evidence_needed)
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{component_id} evidence requirements must be non-empty strings")
