"""Nested scientific research loops: outer world-sensing and inner evidence completion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import tempfile

from .contracts import ComponentConclusion, EvidenceGap, ResearchReport, WriterSection, to_dict
from .aiq_grounding import ReportClaim, ResearchSource, ScientificResearchNotes
from .persistence import RunStore
from .planner import ScientificPlanner
from .question import QuestionInterpreter
from .registry import DataRegistry, ExternalDataRegistry, ToolRegistry
from .resilience import RetryPolicy
from .research import ResearchPolicy
from .tools_builtin import default_tool_registry
from .runtime import (
    ResearchRuntime,
    _decision,
    _entry_from_dict,
    _final_answer,
    _goal,
    _research_brief,
    _plan,
    _scout,
    _snapshot_from_dict,
    _task,
    _synthesis,
)
from .validation import ContractViolation


@dataclass
class WorkflowResult:
    report: ResearchReport
    outer_rounds: int
    decisions: list[dict[str, Any]]
    run_id: str = "ephemeral"


class ScientificResearchAgent:
    def __init__(self, sources: DataRegistry | None = None, external: ExternalDataRegistry | None = None, tools: ToolRegistry | None = None, *, interpreter: QuestionInterpreter | None = None, planner: ScientificPlanner | None = None, research_policy: ResearchPolicy | None = None, run_store: RunStore | None = None, scope_advisor: Any | None = None, action_advisor: Any | None = None, claim_advisor: Any | None = None, evidence_reviewer: Any | None = None, node_writer_advisor: Any | None = None, writer_advisor: Any | None = None, goal_verifier_advisor: Any | None = None, final_answer_advisor: Any | None = None, execution_policy: RetryPolicy | None = None) -> None:
        self.sources = sources or DataRegistry()
        self.external = external or ExternalDataRegistry()
        self.tools = tools or default_tool_registry()
        self.interpreter = interpreter or QuestionInterpreter()
        self.planner = planner or ScientificPlanner()
        self.research_policy = research_policy
        self.run_store, self.scope_advisor, self.action_advisor, self.claim_advisor, self.evidence_reviewer = run_store, scope_advisor, action_advisor, claim_advisor, evidence_reviewer
        self.node_writer_advisor, self.writer_advisor, self.goal_verifier_advisor, self.final_answer_advisor = node_writer_advisor, writer_advisor, goal_verifier_advisor, final_answer_advisor
        self.execution_policy = execution_policy
        self._ephemeral_store: RunStore | None = None
        self._langgraph_runner: Any | None = None

    def _graph_runner(self):
        """Return the sole public orchestration runtime for this agent."""
        if self.run_store is None:
            # The old agent supported non-durable runs.  Keep that convenience
            # while still sending them through the same durable state machine;
            # the temporary SQLite store is removed by the OS, not a second
            # execution path.
            if self._ephemeral_store is None:
                directory = Path(tempfile.mkdtemp(prefix="scientific-agent-"))
                self._ephemeral_store = RunStore(directory / "run.sqlite")
            store = self._ephemeral_store
            # The service runtime used by LangGraph must use the same durable
            # store as the graph runner.  Keeping this only in a local
            # variable would silently drop every node checkpoint.
            self.run_store = store
        else:
            store = self.run_store
        if self._langgraph_runner is None:
            from scientific_research_agent_langgraph import LangGraphScientificRunner
            self._langgraph_runner = LangGraphScientificRunner(lambda: self, store)
        return self._langgraph_runner

    @staticmethod
    def _workflow_result(state: dict[str, Any]) -> WorkflowResult:
        raw = state.get("result")
        if not raw:
            errors = list(state.get("contract_errors", []))
            message = "; ".join(str(item) for item in errors) or f"LangGraph ended without a workflow result: {state.get('error')}"
            if any("ContractViolation" in item or "illegal control" in item for item in errors):
                raise ContractViolation(message)
            raise RuntimeError(message)
        report_raw = raw["report"]
        decisions = [_decision(item) for item in raw.get("decisions", [])]
        gaps_by_id = {
            item.gap_id: item
            for item in [EvidenceGap(**item) for item in report_raw.get("evidence_gaps", [])]
        }
        for decision in decisions:
            for gap in decision.evidence_gaps:
                gaps_by_id.setdefault(gap.gap_id, gap)
        judgments = [
            judgment
            for decision in decisions
            for judgment in decision.component_judgments
        ]
        report_judgments = [
            _decision({"component_judgments": [item], "diagnoses": [], "evidence_gaps": [], "reason": "", "action": "continue_research"}).component_judgments[0]
            for item in report_raw.get("judgments", [])
        ]
        if report_judgments:
            judgments = report_judgments
        report = ResearchReport(
            question=report_raw["question"],
            markdown=report_raw["markdown"],
            research_brief=_research_brief(report_raw["research_brief"]) if report_raw.get("research_brief") else None,
            task=_task(report_raw["task"]),
            plan=_plan(report_raw["plan"]),
            scout_reports=[_scout(item) for item in report_raw.get("scout_reports", [])],
            evidence=[_entry_from_dict(item) for item in report_raw.get("evidence", [])],
            judgments=judgments,
            evidence_gaps=list(gaps_by_id.values()),
            next_observations=list(report_raw.get("next_observations", [])),
            research_notes=[ScientificResearchNotes.model_validate(item) for item in report_raw.get("research_notes", [])],
            claim_manifest=[ReportClaim.model_validate(item) for item in report_raw.get("claim_manifest", [])],
            verified_sources=[ResearchSource.model_validate(item) for item in report_raw.get("verified_sources", [])],
            component_conclusions=[ComponentConclusion(**item) for item in report_raw.get("component_conclusions", [])],
            dag_synthesis=_synthesis(report_raw["dag_synthesis"]) if report_raw.get("dag_synthesis") else None,
            goal_assessment=_goal(report_raw["goal_assessment"]) if report_raw.get("goal_assessment") else None,
            final_answer=_final_answer(report_raw["final_answer"]) if report_raw.get("final_answer") else None,
            writer_sections=[WriterSection(**item) for item in report_raw.get("writer_sections", [])],
        )
        return WorkflowResult(report, int(raw.get("outer_rounds", 0)), [to_dict(item) for item in decisions], raw.get("run_id", state.get("run_id", "")))

    def run(self, question: str, *, max_outer_rounds: int = 8, max_inner_steps: int = 12) -> WorkflowResult:
        state = self._graph_runner().run(question, thread_id=f"run:{id(self)}:{question}", max_outer_rounds=max_outer_rounds, max_inner_steps=max_inner_steps)
        return self._workflow_result(state)

    def resume(self, run_id: str, *, max_outer_rounds: int = 8, max_inner_steps: int = 12) -> WorkflowResult:
        state = self._graph_runner().resume(run_id, thread_id=f"resume:{run_id}", max_outer_rounds=max_outer_rounds, max_inner_steps=max_inner_steps)
        return self._workflow_result(state)

    def _runtime(self) -> ResearchRuntime:
        return ResearchRuntime(sources=self.sources, external=self.external, tools=self.tools, interpreter=self.interpreter, planner=self.planner, research_policy=self.research_policy, store=self.run_store, scope_advisor=self.scope_advisor, action_advisor=self.action_advisor, claim_advisor=self.claim_advisor, evidence_reviewer=self.evidence_reviewer, node_writer_advisor=self.node_writer_advisor, writer_advisor=self.writer_advisor, goal_verifier_advisor=self.goal_verifier_advisor, final_answer_advisor=self.final_answer_advisor, execution_policy=self.execution_policy)
