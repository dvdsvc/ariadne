"""Interpret the user's question into a small, stable Research Brief."""

from __future__ import annotations

import re
from typing import Any

from .contracts import QuestionTask, ResearchBrief


class QuestionInterpreter:
    """The Interpretation-layer adapter.

    The interpreter defines the problem boundary. It does not choose a source,
    method, tool, component, dependency, or execution sequence.
    """

    def __init__(self, advisor: Any | None = None) -> None:
        self.advisor = advisor

    def interpret(self, question: str) -> ResearchBrief:
        if self.advisor is not None:
            raw = self.advisor.interpret(question)
            if isinstance(raw, ResearchBrief):
                self._validate(raw)
                return raw
            if not isinstance(raw, dict):
                raise ValueError("Interpretation advisor must return a ResearchBrief or object")
            return self._from_raw(raw)
        # Without a model, preserve the open question. This fallback is not a
        # classifier and never manufactures scope, method, or scientific facts.
        return ResearchBrief(
            research_object="unspecified",
            spatial_scope={},
            temporal_scope={},
            core_phenomenon_or_relation=question,
            must_answer=[question],
            key_assumptions_and_unresolved_ambiguities=["研究对象、时空范围和可识别的关系仍需 Interpretation 明确。"],
            conclusion_level="不超过当前观测所能支持的现象或关系层级。",
        )

    @staticmethod
    def _from_raw(raw: dict[str, Any]) -> ResearchBrief:
        required = {
            "research_object",
            "spatial_scope",
            "temporal_scope",
            "core_phenomenon_or_relation",
            "must_answer",
            "key_assumptions_and_unresolved_ambiguities",
            "conclusion_level",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ValueError(f"Interpretation Brief is missing fields: {missing}")
        brief = ResearchBrief(
            research_object=str(raw["research_object"]),
            spatial_scope=dict(raw["spatial_scope"]),
            temporal_scope=dict(raw["temporal_scope"]),
            core_phenomenon_or_relation=str(raw["core_phenomenon_or_relation"]),
            must_answer=[str(item) for item in raw["must_answer"]],
            key_assumptions_and_unresolved_ambiguities=[str(item) for item in raw["key_assumptions_and_unresolved_ambiguities"]],
            conclusion_level=str(raw["conclusion_level"]),
        )
        QuestionInterpreter._validate(brief)
        return brief

    @staticmethod
    def _validate(brief: ResearchBrief) -> None:
        if not brief.research_object.strip() or not brief.core_phenomenon_or_relation.strip() or not brief.must_answer:
            raise ValueError("Interpretation Brief requires an object, a core relation, and at least one knowledge gap")
        if any(not isinstance(item, str) or not item.strip() for item in brief.must_answer):
            raise ValueError("Research Brief must_answer entries must be non-empty epistemic questions")
        procedural_zh = re.compile(r"^(?:下载|清洗|清理|预处理|做统计|统计|计算|聚合|画图|绘图|输出|导出|检索|爬取|调用|运行)")
        procedural_en = re.compile(r"^(?:download|clean|preprocess|compute|calculate|aggregate|plot|draw|export|fetch|query|search|run)\b", re.IGNORECASE)
        invalid = [item for item in brief.must_answer if procedural_zh.search(item.strip()) or procedural_en.search(item.strip())]
        if invalid:
            raise ValueError("Research Brief must_answer contains execution tasks, not epistemic closure gaps: " + "; ".join(invalid))


def task_from_brief(question: str, brief: ResearchBrief) -> QuestionTask:
    """Build the narrow legacy execution projection from an Interpretation Brief."""

    return QuestionTask(
        question=question,
        object_of_inference=brief.research_object,
        spatial_scope=dict(brief.spatial_scope),
        temporal_scope=dict(brief.temporal_scope),
        # Knowledge gaps remain in the Brief and are not turned into Planner
        # instructions through the legacy execution projection.
        scientific_goals=[],
        inference_level=[],
        allowed_evidence=[],
        requested_scale={},
    )


def brief_from_task(question: str, task: QuestionTask) -> ResearchBrief:
    """Adapt older test/integration interpreters at the graph boundary."""

    return ResearchBrief(
        research_object=task.object_of_inference,
        spatial_scope=dict(task.spatial_scope),
        temporal_scope=dict(task.temporal_scope),
        core_phenomenon_or_relation=question,
        must_answer=list(task.scientific_goals) or [question],
        key_assumptions_and_unresolved_ambiguities=[],
        conclusion_level=", ".join(task.inference_level) or "不超过当前观测所能支持的层级。",
    )


def normalize_interpretation(question: str, value: ResearchBrief | QuestionTask) -> tuple[ResearchBrief, QuestionTask]:
    """Normalize the I-node result without moving interpretation into Planner."""

    if isinstance(value, ResearchBrief):
        return value, task_from_brief(question, value)
    if isinstance(value, QuestionTask):
        return brief_from_task(question, value), value
    raise TypeError(f"Unsupported Interpretation output: {type(value).__name__}")
