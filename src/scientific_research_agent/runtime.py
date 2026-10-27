"""Durable state machine for the nested Deep Research loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .aiq_grounding import EvidenceSourceRegistry, ScientificResearchNotes
from .adjudication import EvidenceAdjudicator
from .conclusion import DAGSynthesizer, FinalAnswerWriter, GoalPlanVerifier, IsolatedNodeWriter
from .contracts import (
    ComponentConclusion, ComponentJudgment, CriterionEvaluation, DAGSynthesis, EvidenceAssessment, EvidenceEntry, EvidenceGap, FailureRecord, GoalAssessment, Hypothesis, JudgeDecision, LocalExecutionGraph, ResearchAction,
    MissingDiagnosis, PlannerReview, QuestionTask, RequiredComponent, ResearchReport, RunSnapshot,
    ResearchBrief, ScoutReport, ScientificAnswerStrategy, ScientificPlan, PlanEdge, PlanGraph, PlanNode, PlanPatch, PlanPatchOperation,
    ScientificTaskAnalysis, ScopeResolution, to_dict,
)
from .judge import SufficiencyJudge
from .ledger import EvidenceLedger
from .evolver import PlanEvolver
from .persistence import RunStore
from .planner import ScientificPlanner
from .question import QuestionInterpreter
from .registry import DataRegistry, ExternalDataRegistry, ToolRegistry
from .resilience import RetryPolicy, classify_error, retry_metadata
from .reasoning import StrictSchemaViolation
from .research import FlexibleResearcher, ResearchPolicy
from .scout import Scout
from .scope import ScopeCompiler
from .skills import default_skill_registry
from .writer import DAGBoundWriter
from .validation import ContractViolation, affected_components, validate_plan


@dataclass
class DurableWorkflowResult:
    run_id: str
    report: ResearchReport
    outer_rounds: int
    decisions: list[dict[str, Any]]


def _durable_result_from_graph_state(state: dict[str, Any]) -> DurableWorkflowResult:
    """Convert the graph's serialized terminal result for legacy callers."""
    from .workflow import ScientificResearchAgent

    result = ScientificResearchAgent._workflow_result(state)
    return DurableWorkflowResult(result.run_id, result.report, result.outer_rounds, result.decisions)


class ResearchRuntime:
    """No stage is only held in LLM context: each transition is traced and snapshotted."""

    def __init__(self, *, sources: DataRegistry, external: ExternalDataRegistry, tools: ToolRegistry, interpreter: QuestionInterpreter | None = None, planner: ScientificPlanner | None = None, research_policy: ResearchPolicy | None = None, store: RunStore | None = None, scope_advisor: Any | None = None, action_advisor: Any | None = None, claim_advisor: Any | None = None, evidence_reviewer: Any | None = None, node_writer_advisor: Any | None = None, writer_advisor: Any | None = None, goal_verifier_advisor: Any | None = None, final_answer_advisor: Any | None = None, execution_policy: RetryPolicy | None = None) -> None:
        self.sources, self.external, self.tools = sources, external, tools
        self.interpreter, self.planner = interpreter or QuestionInterpreter(), planner or ScientificPlanner()
        # Bind the Planner to the live registry before any model call.  The
        # registry supplies execution capabilities; only conclusion nodes are
        # compiled into the semantic Planner -> Ledger -> Judge interface.
        self.research_policy, self.store, self.action_advisor = research_policy, store, action_advisor
        self.node_writer_advisor = node_writer_advisor
        self.writer_advisor = writer_advisor
        self.goal_verifier_advisor = goal_verifier_advisor
        self.final_answer_advisor = final_answer_advisor
        self.claim_advisor = claim_advisor
        self.evidence_reviewer = evidence_reviewer
        self.execution_policy = execution_policy or RetryPolicy()
        self.scout = Scout(sources, external)
        self.scope_compiler = ScopeCompiler(scope_advisor)
        self.judge = SufficiencyJudge(EvidenceAdjudicator(evidence_reviewer))
        self.evolver, self.writer = PlanEvolver(), DAGBoundWriter(writer_advisor)
        self.node_writer = IsolatedNodeWriter(node_writer_advisor, claim_advisor)
        self.synthesizer, self.goal_verifier = DAGSynthesizer(), GoalPlanVerifier(goal_verifier_advisor)
        self.final_answer_writer = FinalAnswerWriter(final_answer_advisor)
        self.skills = getattr(self.planner, "skills", default_skill_registry())

    def run(self, question: str, *, max_outer_rounds: int = 8, max_inner_steps: int = 12) -> DurableWorkflowResult:
        return self._run_through_langgraph(question, max_outer_rounds=max_outer_rounds, max_inner_steps=max_inner_steps)

    def resume(self, run_id: str, *, max_outer_rounds: int = 8, max_inner_steps: int = 12) -> DurableWorkflowResult:
        if self.store is None:
            raise RuntimeError("resume requires RunStore")
        return self._run_through_langgraph_resume(run_id, max_outer_rounds=max_outer_rounds, max_inner_steps=max_inner_steps)

    def _service_agent(self):
        """Build a non-orchestrating service facade for LangGraph nodes."""
        from .workflow import ScientificResearchAgent
        return ScientificResearchAgent(
            self.sources,
            self.external,
            tools=self.tools,
            interpreter=self.interpreter,
            planner=self.planner,
            research_policy=self.research_policy,
            run_store=self.store,
            scope_advisor=self.scope_compiler.advisor,
            action_advisor=self.action_advisor,
            claim_advisor=self.claim_advisor,
            evidence_reviewer=self.judge.adjudicator.reviewer,
            node_writer_advisor=self.node_writer_advisor,
            writer_advisor=self.writer_advisor,
            goal_verifier_advisor=self.goal_verifier_advisor,
            final_answer_advisor=self.final_answer_advisor,
            execution_policy=self.execution_policy,
        )

    def _run_through_langgraph(self, question: str, *, max_outer_rounds: int, max_inner_steps: int) -> DurableWorkflowResult:
        from scientific_research_agent_langgraph import LangGraphScientificRunner
        service_agent = self._service_agent()
        store = self.store or service_agent._graph_runner().store
        runner = LangGraphScientificRunner(lambda: service_agent, store)
        state = runner.run(question, thread_id=f"runtime:{id(self)}:{question}", max_outer_rounds=max_outer_rounds, max_inner_steps=max_inner_steps)
        return _durable_result_from_graph_state(state)

    def _run_through_langgraph_resume(self, run_id: str, *, max_outer_rounds: int, max_inner_steps: int) -> DurableWorkflowResult:
        from scientific_research_agent_langgraph import LangGraphScientificRunner
        service_agent = self._service_agent()
        runner = LangGraphScientificRunner(lambda: service_agent, self.store)
        state = runner.resume(run_id, thread_id=f"runtime-resume:{run_id}", max_outer_rounds=max_outer_rounds, max_inner_steps=max_inner_steps)
        return _durable_result_from_graph_state(state)

    def _prepare_retryable_toolchain_gaps(self, snapshot: RunSnapshot, entries: list[EvidenceEntry]) -> None:
        """Reopen completed gaps caused by an unavailable data-handle bridge.

        A completed report with ``needs_plan_patch`` is still a durable
        checkpoint, not an immutable dead end.  Only gaps whose local
        capability record explicitly says that a materialized dataset input
        was unavailable are reopened; genuine data/resolution gaps remain
        terminal and continue to report as gaps on resume.
        """

        # A process can disappear after a durable checkpoint has been written
        # but before the run reaches ``completed``/``failed``.  Resume must
        # repair retryable toolchain gaps from any non-terminal checkpoint,
        # not only from the two terminal labels.
        if snapshot.plan is None or snapshot.stage == "completed":
            return
        relation_retried: list[str] = []
        evidence_by_id = {entry.evidence_id: entry for entry in entries}
        for component_id, graph in snapshot.local_execution_graphs.items():
            for action in graph.actions:
                request = action.data_request or {}
                if request.get("operation") not in {"relate", "materialize"}:
                    continue
                evidence_ids = graph.action_evidence_ids.get(action.action_id, [])
                if not any(self._relation_limit_overflow(evidence_by_id.get(evidence_id)) for evidence_id in evidence_ids):
                    continue
                request["row_limit"] = max(int(request.get("row_limit") or 0), 3_000_000)
                action.data_request = request
                graph.action_status[action.action_id] = "pending"
                graph.action_evidence_ids.pop(action.action_id, None)
                # The old attempt fingerprint includes the failed request's
                # declared limit.  Remove it so recovery cannot mistake the
                # old rejected evidence for a successful execution of the
                # repaired request.
                graph.action_input_fingerprints.pop(action.action_id, None)
                graph.execution_status = "open"
                graph.scientific_status = "not_tested"
                graph.missing_evidence = ["data_relate"]
                graph.diagnostic_epoch += 1
                graph.revision += 1
                relation_retried.append(component_id)
                if snapshot.plan.graph is not None:
                    snapshot.plan.graph.component_node(component_id).execution_status = "active"
                break
        if relation_retried:
            snapshot.goal_assessment = None
            snapshot.dag_synthesis = None
            snapshot.final_answer = None
            snapshot.stage = "planned"
            snapshot.inner_step = 0
            self._save(snapshot, "resume_reopened_relation_overflow", {
                "component_ids": list(dict.fromkeys(relation_retried)),
                "row_limit": 3_000_000,
                "reason": "retry relation materialization after declared row-limit overflow",
            })
        retryable: list[str] = []
        for component_id, graph in snapshot.local_execution_graphs.items():
            try:
                component = snapshot.plan.component(component_id)
            except (ValueError, StopIteration):
                continue
            if component.scientific_status != "insufficient":
                continue
            for coverage in graph.capability_coverage.values():
                if coverage.get("status") != "input_not_ready":
                    continue
                missing = {
                    str(requirement_id)
                    for provider in coverage.get("provider_readiness", [])
                    for requirement_id in provider.get("missing_requirement_ids", [])
                }
                if "materialized_dataset" in missing:
                    retryable.append(component_id)
                    break
        retryable = list(dict.fromkeys(retryable))
        if not retryable or not snapshot.scout_reports:
            return
        operations = [
            PlanPatchOperation("reopen_component", component_id, {"reason": "Resume after repairing the materialized-data handle bridge."})
            for component_id in retryable
        ]
        review = PlannerReview(
            "patch_graph", retryable[0],
            "Reopen only components blocked by the repaired typed materialized-data handle interface.",
            affected_component_ids=retryable, operations=operations,
        )
        # This is a durable infrastructure retry, not a new scientific plan
        # proposal. Reopening the same components does not redefine the
        # Interpretation Brief.
        patched = self.evolver.apply_global_review(snapshot.plan, snapshot.scout_reports[-1], review)
        snapshot.plan = validate_plan(patched)
        for component_id in retryable:
            snapshot.local_execution_graphs.pop(component_id, None)
            snapshot.component_conclusions = [item for item in snapshot.component_conclusions if item.component_id != component_id]
        snapshot.goal_assessment = None
        snapshot.dag_synthesis = None
        snapshot.final_answer = None
        snapshot.stage = "planned"
        snapshot.inner_step = 0
        self._save(snapshot, "resume_reopened_toolchain_gaps", {
            "component_ids": retryable,
            "plan_revision": snapshot.plan.plan_revision,
            "reason": "typed materialized-data handle bridge repaired",
        })

    @staticmethod
    def _relation_limit_overflow(entry: EvidenceEntry | None) -> bool:
        if entry is None:
            return False
        findings = entry.findings if isinstance(entry.findings, dict) else {}
        summary = str(entry.result_summary or "").lower()
        operation = findings.get("data_operation") if isinstance(findings.get("data_operation"), dict) else {}
        return (
            "row limit" in summary
            or bool(findings.get("estimated_exceeds_declared_limit"))
            or (operation.get("status") == "invalid_request" and "limit" in summary)
        )

    def _save(self, snapshot: RunSnapshot, event_type: str, payload: dict[str, Any]) -> None:
        """Persist one domain snapshot; routing is owned by LangGraph."""
        snapshot.revision += 1
        if self.store:
            self.store.checkpoint(snapshot, event_type, payload)

    @staticmethod
    def _operational_graph_complete(graph: LocalExecutionGraph) -> bool:
        """Treat a completed terminal action as closure of a data graph.

        Historical failed attempts remain diagnostic history.  They must not
        reopen a graph after a later idempotent ``finish_component`` action
        has committed its terminal state.
        """
        return any(
            action.kind == "finish_component"
            and graph.action_status.get(action.action_id) == "completed"
            for action in graph.actions
        )


    def _record_node_conclusions(self, snapshot: RunSnapshot, ledger: EvidenceLedger, decision: JudgeDecision) -> None:
        if snapshot.plan is None:
            return
        assessments = {item.component_id: item for item in decision.assessments}
        by_component = {item.component_id: item for item in snapshot.component_conclusions}
        for judgment in decision.component_judgments:
            component = snapshot.plan.component(judgment.component_id)
            assessment = assessments.get(judgment.component_id)
            if assessment is None:
                continue
            by_component[judgment.component_id] = self.node_writer.write(
                component,
                assessment,
                self.judge.adjudicator.entries_for_component(snapshot.plan, component, ledger),
            )
        snapshot.component_conclusions = [by_component[component.component_id] for component in snapshot.plan.required_components if component.component_id in by_component]

    @staticmethod
    def _active_frontier(plan: ScientificPlan) -> list[str]:
        """Return executable leaves of the current graph, never every component."""

        active = []
        for component in plan.required_components:
            if plan.component_is_terminal(component.component_id):
                continue
            if all(
                plan.component(dependency).scientific_status == "supported"
                for dependency in component.depends_on
            ):
                active.append(component.component_id)
        return active

    @staticmethod
    def _stamp_planner_review(snapshot: RunSnapshot, graph: LocalExecutionGraph, review: PlannerReview) -> None:
        review.review_id = review.review_id or f"PR{len(snapshot.planner_reviews) + 1:04d}"
        review.plan_revision = snapshot.plan.plan_revision if snapshot.plan is not None else 0
        review.local_graph_revision = graph.revision

    def _trace(self, run_id: str, kind: str, payload: dict[str, Any]) -> str | None:
        return self.store.append_trace(run_id, kind, payload) if self.store else None

    @staticmethod
    def _rebind_active_plan_scope(plan: ScientificPlan, scout: ScoutReport, affected_component_ids: set[str]) -> None:
        """Use a newly compiled scope only for future work after local repair."""

        for component in plan.required_components:
            if component.component_id in affected_component_ids:
                component.scope = scout.task_scope


def _task(raw: dict[str, Any]) -> QuestionTask:
    return QuestionTask(**raw)


def _research_brief(raw: dict[str, Any]) -> ResearchBrief:
    return ResearchBrief(
        research_object=str(raw["research_object"]),
        spatial_scope=dict(raw["spatial_scope"]),
        temporal_scope=dict(raw["temporal_scope"]),
        core_phenomenon_or_relation=str(raw["core_phenomenon_or_relation"]),
        must_answer=[str(item) for item in raw.get("must_answer", [])],
        key_assumptions_and_unresolved_ambiguities=[str(item) for item in raw.get("key_assumptions_and_unresolved_ambiguities", [])],
        conclusion_level=str(raw.get("conclusion_level", "")),
    )


def _scout(raw: dict[str, Any]) -> ScoutReport:
    from .contracts import CoverageProfile
    return ScoutReport(raw["generated_at"], raw["task_scope"], [CoverageProfile(**item) for item in raw["sources"]], raw["external_datasets"], raw["sensor_limits"], raw["world_fingerprint"], raw.get("warnings", []), raw.get("probes", {}), [ScopeResolution(**item) for item in raw.get("scope_resolutions", [])])


def _plan(raw: dict[str, Any]) -> ScientificPlan:
    task = _task(raw["task"])
    components = [_component(item) for item in raw["required_components"]]
    hypotheses = [Hypothesis(**item) for item in raw["candidate_hypotheses"]]
    raw_analysis = raw.get("task_analysis", {})
    # A persisted run from before the Planner-Agent contract remains readable,
    # but newly created plans always contain the complete structured analysis.
    analysis = ScientificTaskAnalysis(**raw_analysis) if isinstance(raw_analysis, dict) else ScientificTaskAnalysis(
        user_intent=str(raw_analysis), explicit_requirements=list(task.scientific_goals),
        implicit_requirements=[], out_of_scope=[], language="unknown",
    )
    raw_strategy = raw.get("answer_strategy")
    strategy = ScientificAnswerStrategy(**raw_strategy) if raw_strategy else ScientificAnswerStrategy(
        "scientific_report", "Evidence-bound scientific inference",
        [component.component_id for component in components],
        ["Separate observation, interpretation, alternatives, and gaps."],
    )
    graph = _graph(raw["graph"]) if raw.get("graph") else None
    # Old snapshots may contain the retired Search/Aggregate/Answer graph.
    # Discard only that compiled projection; the Planner components and their
    # scientific dependencies remain the durable source of truth.
    if graph is not None and any(node.kind != "decision" for node in graph.nodes):
        graph = None
    return ScientificPlan(
        task=task, task_analysis=analysis, scientific_goal=raw["scientific_goal"],
        required_components=components, candidate_hypotheses=hypotheses,
        current_sensor_limits=raw["current_sensor_limits"], known_evidence_gaps=raw["known_evidence_gaps"],
        scout_fingerprint=raw["scout_fingerprint"], answer_strategy=strategy,
        constraints=list(raw.get("constraints", [])),
        plan_revision=raw.get("plan_revision", 1), graph=graph,
        research_brief=_research_brief(raw["research_brief"]) if raw.get("research_brief") else None,
    )


def _graph(raw: dict[str, Any]) -> PlanGraph:
    assessments = [
        EvidenceAssessment(
            assessment_id=item["assessment_id"], component_id=item["component_id"], outcome=item["outcome"],
            criterion_evaluations=[CriterionEvaluation(**criterion) for criterion in item.get("criterion_evaluations", [])],
            evidence_ids=item.get("evidence_ids", []), missing_evidence=item.get("missing_evidence", []), gap_kind=item.get("gap_kind"),
            allowed_claim_level=item.get("allowed_claim_level", "none"), rationale=item.get("rationale", ""),
            packet_fingerprint=item.get("packet_fingerprint", ""), adjudicator=item.get("adjudicator", "r_layer_scientific_judge"), reviewer_verdict=item.get("reviewer_verdict", "not_requested"), confidence=item.get("confidence"),
        )
        for item in raw.get("assessments", [])
    ]
    patches = [
        PlanPatch(
            patch_id=item["patch_id"], base_revision=item["base_revision"], next_revision=item["next_revision"], kind=item["kind"],
            trigger_assessment_ids=item.get("trigger_assessment_ids", []), affected_component_ids=item.get("affected_component_ids", []),
            reason=item.get("reason", ""), operations=[PlanPatchOperation(**operation) for operation in item.get("operations", [])], created_at=item["created_at"],
        )
        for item in raw.get("patches", [])
    ]
    return PlanGraph(
        nodes=[_node(item) for item in raw.get("nodes", [])],
        edges=[PlanEdge(**item) for item in raw.get("edges", [])],
        assessments=assessments, patches=patches,
    )


def _diagnosis(raw: dict[str, Any]) -> MissingDiagnosis:
    return MissingDiagnosis(**raw)


def _failure(raw: dict[str, Any]) -> FailureRecord:
    return FailureRecord(**raw)


def _planner_review(raw: dict[str, Any]) -> PlannerReview:
    return PlannerReview(
        route=raw["route"], component_id=raw["component_id"], rationale=raw.get("rationale", ""),
        diagnoses=[_diagnosis(item) for item in raw.get("diagnoses", [])],
        affected_component_ids=list(raw.get("affected_component_ids", [])),
        operations=[PlanPatchOperation(**item) for item in raw.get("operations", [])],
        review_id=raw.get("review_id", ""), plan_revision=raw.get("plan_revision", 0),
        local_graph_revision=raw.get("local_graph_revision", 0),
    )


def _decision(raw: dict[str, Any]) -> JudgeDecision:
    judgments = [ComponentJudgment(
        component_id=item["component_id"], claim=item["claim"], status=item["status"], confidence=item["confidence"],
        evidence_ids=item["evidence_ids"], rationale=item["rationale"], missing_evidence=item.get("missing_evidence", []),
        diagnoses=[_diagnosis(d) for d in item.get("diagnoses", [])], assessment_ids=item.get("assessment_ids", []),
        allowed_claim_level=item.get("allowed_claim_level", "none"),
    ) for item in raw["component_judgments"]]
    gaps = [EvidenceGap(**item) for item in raw["evidence_gaps"]]
    assessments = [
        EvidenceAssessment(
            assessment_id=item["assessment_id"], component_id=item["component_id"], outcome=item["outcome"],
            criterion_evaluations=[CriterionEvaluation(**criterion) for criterion in item.get("criterion_evaluations", [])],
            evidence_ids=item.get("evidence_ids", []), missing_evidence=item.get("missing_evidence", []), gap_kind=item.get("gap_kind"),
            allowed_claim_level=item.get("allowed_claim_level", "none"), rationale=item.get("rationale", ""),
            packet_fingerprint=item.get("packet_fingerprint", ""), adjudicator=item.get("adjudicator", "r_layer_scientific_judge"), reviewer_verdict=item.get("reviewer_verdict", "not_requested"), confidence=item.get("confidence"),
        )
        for item in raw.get("assessments", [])
    ]
    return JudgeDecision(raw["action"], judgments, [_diagnosis(item) for item in raw["diagnoses"]], gaps, raw["reason"], raw.get("affected_component_ids", []), assessments)


def _snapshot_from_dict(raw: dict[str, Any]) -> RunSnapshot:
    conclusions = [ComponentConclusion(**item) for item in raw.get("component_conclusions", [])]
    synthesis = _synthesis(raw["dag_synthesis"]) if raw.get("dag_synthesis") else None
    goal = _goal(raw["goal_assessment"]) if raw.get("goal_assessment") else None
    return RunSnapshot(
        run_id=raw["run_id"], stage=raw["stage"], question=raw["question"],
        research_brief=_research_brief(raw["research_brief"]) if raw.get("research_brief") else None,
        task=_task(raw["task"]) if raw.get("task") else None,
        scout_reports=[_scout(item) for item in raw.get("scout_reports", [])],
        plan=_plan(raw["plan"]) if raw.get("plan") else None,
        research_notes=[ScientificResearchNotes.model_validate(item) for item in raw.get("research_notes", [])],
        decisions=[_decision(item) for item in raw.get("decisions", [])],
        component_conclusions=conclusions, dag_synthesis=synthesis, goal_assessment=goal,
        active_component_ids=raw.get("active_component_ids", []), outer_round=raw.get("outer_round", 0),
        inner_step=raw.get("inner_step", 0), revision=raw.get("revision", 0),
        local_execution_graphs={component_id: _local_execution_graph(item) for component_id, item in raw.get("local_execution_graphs", {}).items()},
        final_answer=_final_answer(raw["final_answer"]) if raw.get("final_answer") else None,
        planner_reviews=[_planner_review(item) for item in raw.get("planner_reviews", [])],
        pending_planner_review=_planner_review(raw["pending_planner_review"]) if raw.get("pending_planner_review") else None,
        failure_history=[_failure(item) for item in raw.get("failure_history", [])],
        schema_version=max(2, int(raw.get("schema_version", 1))),
    )


def _local_execution_graph(raw: dict[str, Any]) -> LocalExecutionGraph:
    legacy_execution_failures = [
        FailureRecord(
            failure_id=f"F:legacy:{raw.get('component_id', '')}:{index}", failure_kind="tool_execution",
            phase="legacy_action_execution", error_type="LegacyActionError", category="legacy_unclassified",
            retryable=False, message=str(message), component_id=raw.get("component_id", ""),
            action_id=action_id, graph_revision=raw.get("revision", 1), diagnostic_epoch=raw.get("diagnostic_epoch", 1),
        )
        for index, (action_id, message) in enumerate(raw.get("action_errors", {}).items(), 1)
    ]
    legacy_proposal_failures, legacy_contract_failures = [], []
    for index, item in enumerate(raw.get("action_proposal_errors", []), 1):
        contract = item.get("error_type") in {"ContractViolation", "StrictSchemaViolation", "ValidationError"}
        record = FailureRecord(
            failure_id=f"F:legacy-proposal:{raw.get('component_id', '')}:{index}",
            failure_kind="contract_validation" if contract else "model_action_proposal",
            phase=item.get("phase", "legacy_researcher_action_proposal"),
            error_type=item.get("error_type", "LegacyProposalError"),
            category="schema_contract" if contract else item.get("category", "legacy_unclassified"),
            retryable=False if contract else bool(item.get("retryable", False)),
            message=item.get("message", ""), component_id=raw.get("component_id", ""),
            graph_revision=raw.get("revision", 1), diagnostic_epoch=raw.get("diagnostic_epoch", 1),
        )
        (legacy_contract_failures if contract else legacy_proposal_failures).append(record)
    return LocalExecutionGraph(
        graph_id=raw["graph_id"], component_id=raw["component_id"],
        actions=[ResearchAction(**item) for item in raw.get("actions", [])],
        action_status=dict(raw.get("action_status", {})),
        action_evidence_ids={key: list(value) for key, value in raw.get("action_evidence_ids", {}).items()},
        tool_execution_failures=[_failure(item) for item in raw.get("tool_execution_failures", [])] + legacy_execution_failures,
        model_action_proposal_failures=[_failure(item) for item in raw.get("model_action_proposal_failures", [])] + legacy_proposal_failures,
        contract_failures=[_failure(item) for item in raw.get("contract_failures", [])] + legacy_contract_failures,
        action_input_fingerprints=dict(raw.get("action_input_fingerprints", {})),
        rejected_action_fingerprints=list(raw.get("rejected_action_fingerprints", [])),
        execution_status=raw.get("execution_status", "open"),
        scientific_status=raw.get("scientific_status", "not_tested"), missing_evidence=list(raw.get("missing_evidence", [])),
        capability_coverage=dict(raw.get("capability_coverage", {})), repair_history=list(raw.get("repair_history", [])),
        escalation_note=raw.get("escalation_note", ""), planner_directive=raw.get("planner_directive", ""),
        revision=raw.get("revision", 1), diagnostic_epoch=raw.get("diagnostic_epoch", 1),
    )


def _synthesis(raw: dict[str, Any]) -> DAGSynthesis:
    return DAGSynthesis(
        plan_revision=raw["plan_revision"], root_goal=raw["root_goal"],
        component_conclusions=[ComponentConclusion(**item) for item in raw.get("component_conclusions", [])],
        required_component_ids=list(raw.get("required_component_ids", [])), blocked_component_ids=list(raw.get("blocked_component_ids", [])),
        terminal_component_ids=list(raw.get("terminal_component_ids", [])), dependency_summary=list(raw.get("dependency_summary", [])),
    )


def _goal(raw: dict[str, Any]) -> GoalAssessment:
    return GoalAssessment(
        status=raw["status"], reason=raw["reason"], blocking_component_ids=list(raw.get("blocking_component_ids", [])),
        missing_requirements=list(raw.get("missing_requirements", [])), diagnoses=[_diagnosis(item) for item in raw.get("diagnoses", [])],
        synthesis_fingerprint=raw.get("synthesis_fingerprint", ""), verifier_verdict=raw.get("verifier_verdict", "deterministic"),
    )


def _final_answer(raw: dict[str, Any]):
    from .contracts import FinalAnswer
    return FinalAnswer(**raw)


def _entry_from_dict(raw: dict[str, Any]) -> EvidenceEntry:
    value = dict(raw)
    value.setdefault("calculation_status", "computed")
    return EvidenceEntry(**value)


def _component(raw: dict[str, Any]) -> RequiredComponent:
    allowed = {
        "component_id", "claim", "depends_on", "evidence_needed", "constraints",
        "failure_meaning", "scope", "execution_status", "scientific_status",
        "candidate_hypothesis_ids", "skill_ids",
    }
    return RequiredComponent(**{key: value for key, value in raw.items() if key in allowed})


def _node(raw: dict[str, Any]) -> PlanNode:
    return PlanNode(**dict(raw))
