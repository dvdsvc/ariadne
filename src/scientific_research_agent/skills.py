"""Scientific reasoning skills: reusable investigation patterns, not tool recipes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScientificSkill:
    skill_id: str
    description: str
    reasoning_pattern: tuple[str, ...]


class SkillRegistry:
    def __init__(self) -> None:
        self._skills: dict[str, ScientificSkill] = {}

    def register(self, skill: ScientificSkill) -> None:
        self._skills[skill.skill_id] = skill

    def all(self) -> list[ScientificSkill]:
        """Return the full strategy catalogue without deciding applicability."""
        return list(self._skills.values())

    def get(self, skill_id: str) -> ScientificSkill:
        return self._skills[skill_id]


def default_skill_registry() -> SkillRegistry:
    """Return no built-in scientific strategy.

    A strategy may be registered only after it has an auditable provenance,
    such as a curator-reviewed research trajectory distilled from a paper.
    Hand-written examples must not silently constrain an open research question.
    """
    return SkillRegistry()
