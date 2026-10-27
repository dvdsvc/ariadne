"""Serializable scientific contracts shared by every layer of the agent."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

EvidenceStatus = Literal["supported", "contradicted", "insufficient", "not_tested"]
ToolStatus = Literal["ok", "data_unavailable", "error", "not_applicable"]
CalculationStatus = Literal["computed", "insufficient_data", "not_computable", "not_applicable"]
ExecutionStatus = Literal["pending", "active", "completed", "blocked", "failed", "exhausted", "escalated", "superseded"]
PlanNodeKind = Literal["decision"]
PlanEdgeKind = Literal["depends_on"]
PatchKind = Literal["assessment", "continue_research", "local_rescout", "redirect", "evidence_gap", "supersede", "global_patch"]
SensorKind = Literal[
    "event_clock_sensor", "spatial_constraint", "material_state_sensor",
    "external_boundary_condition", "forward_physical_model", "opportunistic_sensor",
]
ResolutionKind = Literal["site", "locality", "county", "regional", "unresolved"]
GapKind = Literal[
    "calculation_missing", "wrong_scope_or_data_assumption", "missing_external_sensor",
    "missing_tool_capability", "tool_input_unavailable", "weak_optional_source",
    "fundamentally_unidentifiable", "resolution_limit",
]
NextAction = Literal["continue_research", "local_rescout", "patch_plan", "write"]
PlannerRoute = Literal["repair_local", "patch_graph", "formalize_gap"]
RunStage = Literal["created", "interpreted", "scouted", "planned", "researching", "judged", "writing", "completed", "failed"]
FailureKind = Literal["tool_execution", "model_action_proposal", "contract_validation", "runtime"]


@dataclass
class SpatialObservation:
    latitude: float | None = None
    longitude: float | None = None
    uncertainty_km: float | None = None
    resolution: ResolutionKind = "unresolved"
    provenance: str = "unknown"

    @property
    def sigma_xy_km(self) -> float | None:
        """Explicit alias used by spatial tools; never silently substitute zero."""
        return self.uncertainty_km


@dataclass
class SourceDescriptor:
    source_id: str
    name: str
    sensor_kind: SensorKind
    version: str
    description: str
    variables: list[str]
    capabilities: list[str]
    schema: dict[str, Any]
    provenance: dict[str, Any] = field(default_factory=dict)
    authoritative: bool = True


@dataclass
class CoverageProfile:
    source_id: str
    matched_records: int
    total_records: int
    dimensions: dict[str, Any]
    missingness: dict[str, float]
    resolution: dict[str, Any]
    uncertainty: dict[str, Any]
    estimated_query_cost: dict[str, Any] = field(default_factory=dict)
    sample_records: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    data_version: str = "unknown"


@dataclass
class QuestionTask:
    question: str
    object_of_inference: str
    spatial_scope: dict[str, Any]
    temporal_scope: dict[str, Any]
    scientific_goals: list[str]
    inference_level: list[Literal["phenomenon", "mechanism", "observation_design"]]
    allowed_evidence: list[str]
    requested_scale: dict[str, Any]

    @property
    def scope(self) -> dict[str, Any]:
        return {**self.spatial_scope, **self.temporal_scope}


@dataclass
class ResearchBrief:
    """The Interpretation layer's bounded definition of the question.

    This is an epistemic brief, not an execution plan.  ``must_answer`` items
    are knowledge gaps required to close the question; they must never become
    download, cleaning, computation, plotting, or export instructions.
    Planner, Researcher, Judge, and Writer may consume this brief, but only the
    Interpretation node may create or revise it for a run.
    """

    research_object: str
    spatial_scope: dict[str, Any]
    temporal_scope: dict[str, Any]
    core_phenomenon_or_relation: str
    must_answer: list[str]
    key_assumptions_and_unresolved_ambiguities: list[str]
    conclusion_level: str

    @property
    def scope(self) -> dict[str, Any]:
        return {**self.spatial_scope, **self.temporal_scope}


@dataclass
class ScopeResolution:
    """A Planner-bound compilation of a data request into query predicates.

    ``canonical_scope`` is the only scope accepted by scientific execution.
    The Planner receives this readout but does not author physical field names
    or values itself.
    """

    source_id: str
    status: Literal["resolved", "partial", "scope_unresolvable", "unconstrained"]
    canonical_scope: dict[str, Any]
    query_contract: dict[str, Any]
    unresolved_terms: list[str] = field(default_factory=list)
    non_filter_terms: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoutReport:
    generated_at: str
    task_scope: dict[str, Any]
    sources: list[CoverageProfile]
    external_datasets: list[dict[str, Any]]
    sensor_limits: list[str]
    world_fingerprint: str
    warnings: list[str] = field(default_factory=list)
    probes: dict[str, dict[str, Any]] = field(default_factory=dict)
    scope_resolutions: list[ScopeResolution] = field(default_factory=list)


@dataclass
class Hypothesis:
    hypothesis_id: str
    label: str
    description: str
    predictions: list[str]
    differentiating_observables: list[str]
    status: EvidenceStatus = "not_tested"


@dataclass
class ScientificTaskAnalysis:
    """Planner-owned reading of the request, kept distinct from evidence claims."""

    user_intent: str
    explicit_requirements: list[str]
    implicit_requirements: list[str]
    out_of_scope: list[str]
    language: str
    ambiguities: list[str] = field(default_factory=list)


@dataclass
class ScientificAnswerStrategy:
    """What a final scientific report must establish, never a tool recipe."""

    answer_type: Literal["scientific_report", "comparative_inference", "hypothesis_test", "observation_design", "custom"]
    title: str
    required_component_ids: list[str]
    synthesis_requirements: list[str]


@dataclass
class RequiredComponent:
    component_id: str
    claim: str
    depends_on: list[str]
    evidence_needed: list[str]
    constraints: dict[str, Any]
    failure_meaning: str
    scope: dict[str, Any]
    # Every Planner component is a scientific decision node. Data acquisition
    # and computation are local Researcher actions, not global pipeline stages.
    # Component lifecycle belongs to the global Plan DAG. A Judge still
    # emits the narrower EvidenceStatus; the Planner may additionally bound
    # an obsolete route as ``superseded`` or an unavailable route as an
    # ``evidence_gap`` without pretending either is an evidence verdict.
    execution_status: ExecutionStatus = "pending"
    scientific_status: EvidenceStatus = "not_tested"
    candidate_hypothesis_ids: list[str] = field(default_factory=list)
    skill_ids: list[str] = field(default_factory=list)


@dataclass
class PlanNode:
    """One Planner decision node in the unconstrained scientific DAG."""

    node_id: str
    kind: PlanNodeKind
    label: str
    parent_id: str | None
    execution_status: ExecutionStatus = "pending"
    scientific_status: EvidenceStatus = "not_tested"
    component_id: str | None = None
    created_in_revision: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanEdge:
    source_id: str
    target_id: str
    relation: PlanEdgeKind = "depends_on"
    output_name: str = ""
    input_name: str = ""
    rationale: str = ""


@dataclass
class CriterionEvaluation:
    criterion_id: str
    status: Literal["passed", "failed", "missing", "not_applicable"]
    evidence_ids: list[str]
    detail: str


@dataclass
class EvidenceAssessment:
    """A context-sliced scientific assessment of one component."""

    assessment_id: str
    component_id: str
    outcome: EvidenceStatus
    criterion_evaluations: list[CriterionEvaluation]
    evidence_ids: list[str]
    missing_evidence: list[str]
    gap_kind: GapKind | None
    allowed_claim_level: Literal["none", "pattern_only", "component_only", "mechanism_association"]
    rationale: str
    packet_fingerprint: str
    adjudicator: str = "r_layer_scientific_judge"
    # ``entailed``/``not_entailed`` are retained for old reviewer records.
    # New R-layer semantic adjudication records the actual three-way result.
    reviewer_verdict: Literal["not_requested", "entailed", "not_entailed", "supported", "contradicted", "insufficient"] = "not_requested"
    # Confidence is optional because it belongs to the R-Judge output, not to
    # a deterministic status-to-number mapping.
    confidence: float | None = None


@dataclass
class PlanPatchOperation:
    operation: Literal[
        "set_status", "add_gap", "redirect", "supersede",
        "add_component", "reconnect_dependencies", "reopen_component", "rescout",
    ]
    target_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlanPatch:
    """Append-only mutation record for an evolving plan graph."""

    patch_id: str
    base_revision: int
    next_revision: int
    kind: PatchKind
    trigger_assessment_ids: list[str]
    affected_component_ids: list[str]
    reason: str
    operations: list[PlanPatchOperation]
    created_at: str


@dataclass
class PlanGraph:
    """The Planner's arbitrary acyclic scientific decision graph."""

    nodes: list[PlanNode]
    edges: list[PlanEdge]
    assessments: list[EvidenceAssessment] = field(default_factory=list)
    patches: list[PlanPatch] = field(default_factory=list)

    def node(self, node_id: str) -> PlanNode:
        return next(node for node in self.nodes if node.node_id == node_id)

    def component_node(self, component_id: str) -> PlanNode:
        return next(node for node in self.nodes if node.component_id == component_id)

    def terminal_component(self, component_id: str) -> bool:
        node = self.component_node(component_id)
        return node.execution_status in {"completed", "exhausted", "escalated", "superseded"} or node.scientific_status in {"supported", "contradicted"}


@dataclass
class ScientificPlan:
    task: QuestionTask
    task_analysis: ScientificTaskAnalysis
    scientific_goal: str
    required_components: list[RequiredComponent]
    candidate_hypotheses: list[Hypothesis]
    current_sensor_limits: list[str]
    known_evidence_gaps: list[str]
    scout_fingerprint: str
    answer_strategy: ScientificAnswerStrategy | None = None
    constraints: list[str] = field(default_factory=list)
    plan_revision: int = 1
    graph: PlanGraph | None = None
    research_brief: ResearchBrief | None = None

    def component(self, component_id: str) -> RequiredComponent:
        return next(component for component in self.required_components if component.component_id == component_id)

    def component_is_terminal(self, component_id: str) -> bool:
        component = self.component(component_id)
        if self.graph is not None:
            return self.graph.terminal_component(component_id)
        return component.execution_status in {"completed", "exhausted", "escalated", "superseded"} or component.scientific_status in {"supported", "contradicted"}


@dataclass
class ToolResult:
    result_id: str
    tool_id: str
    status: ToolStatus
    calculation_status: CalculationStatus
    summary: str
    findings: dict[str, Any]
    evidence_tags: list[str]
    uncertainty: dict[str, Any]
    data_versions: dict[str, str]
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class EvidenceEntry:
    evidence_id: str
    kind: Literal["source_query", "tool_call", "external_data", "forward_model"]
    subject_id: str
    component_ids: list[str]
    input_scope: dict[str, Any]
    result_summary: str
    findings: dict[str, Any]
    evidence_tags: list[str]
    uncertainty: dict[str, Any]
    data_versions: dict[str, str]
    calculation_status: CalculationStatus
    status: ToolStatus | Literal["ok"]
    tool_parameters: dict[str, Any] = field(default_factory=dict)
    trace_id: str | None = None
    recorded_at: str | None = None
    limitations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Provenance links are append-only facts, not model-written citations.
    upstream_evidence_ids: list[str] = field(default_factory=list)
    input_fingerprint: str | None = None
    result_fingerprint: str | None = None
    # Stable hash of the producing component's scientific inputs.  It changes
    # when a Planner patch reopens/reconnects/rescopes that component, allowing
    # the current Judge packet to exclude durable evidence from an older route.
    execution_epoch: str = ""


@dataclass
class FailureRecord:
    """One durable failure in exactly one runtime channel.

    Tool/provider execution, model action proposal, and contract validation are
    deliberately disjoint.  ``diagnostic_epoch`` lets a repaired local graph
    retain history without exposing old failures as the current diagnosis.
    """

    failure_id: str
    failure_kind: FailureKind
    phase: str
    error_type: str
    category: str
    retryable: bool
    message: str
    component_id: str = ""
    action_id: str = ""
    graph_revision: int = 0
    diagnostic_epoch: int = 0
    retry_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MissingDiagnosis:
    component_id: str
    kind: GapKind
    reason: str
    missing_observables: list[str]
    suggested_sensor_kinds: list[SensorKind]
    expected_information_gain: Literal["low", "medium", "high"]
    patch_scope: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceGap:
    gap_id: str
    target_component: str
    unresolved_hypotheses: list[str]
    reason: str
    missing_observable: str
    candidate_sensor: str | None
    expected_information_gain: Literal["low", "medium", "high"]
    conclusion_limit: str


@dataclass
class ComponentJudgment:
    component_id: str
    claim: str
    status: EvidenceStatus
    confidence: float | None
    evidence_ids: list[str]
    rationale: str
    missing_evidence: list[str] = field(default_factory=list)
    diagnoses: list[MissingDiagnosis] = field(default_factory=list)
    assessment_ids: list[str] = field(default_factory=list)
    allowed_claim_level: Literal["none", "pattern_only", "component_only", "mechanism_association"] = "none"


@dataclass
class ComponentConclusion:
    """One node's isolated, execution-bound conclusion.

    It is written from one R-Judge assessment and its local Ledger packet.  It
    intentionally has no user-question, Scout, Plan, or Researcher rationale
    field.
    """

    component_id: str
    status: EvidenceStatus
    conclusion: str
    observed_results: list[str]
    limitations: list[str]
    evidence_ids: list[str]
    allowed_claim_level: Literal["none", "pattern_only", "component_only", "mechanism_association"]
    packet_fingerprint: str
    writer_verdict: Literal["deterministic", "entailed", "rejected"] = "deterministic"


@dataclass
class DAGSynthesis:
    """Graph-only combination of isolated node conclusions; never raw evidence."""

    plan_revision: int
    root_goal: str
    component_conclusions: list[ComponentConclusion]
    required_component_ids: list[str]
    blocked_component_ids: list[str]
    terminal_component_ids: list[str]
    dependency_summary: list[str]


@dataclass(frozen=True)
class WriterSubgraphPacket:
    """A local, conclusion-only subgraph handed to the Writer.

    This is a compilation input, not a judging packet.  It contains no raw
    evidence, tool output, Scout state, or Researcher history.  The global
    Plan DAG determines which upstream nodes and edges are visible here.
    """

    root_component_id: str
    component_ids: list[str]
    node_ids: list[str]
    edge_ids: list[str]
    conclusions: list[ComponentConclusion]


@dataclass(frozen=True)
class WriterSection:
    """Natural-language compilation of one Writer local subgraph."""

    section_id: str
    root_component_id: str
    component_ids: list[str]
    node_ids: list[str]
    edge_ids: list[str]
    text: str
    status: EvidenceStatus
    allowed_claim_level: Literal["none", "pattern_only", "component_only", "mechanism_association"]


@dataclass
class GoalAssessment:
    """Question-level check over the DAG synthesis, without raw tool context."""

    status: Literal["proven", "needs_plan_patch", "evidence_gap"]
    reason: str
    blocking_component_ids: list[str]
    missing_requirements: list[str]
    diagnoses: list[MissingDiagnosis] = field(default_factory=list)
    synthesis_fingerprint: str = ""
    verifier_verdict: Literal["deterministic", "proven", "needs_plan_patch", "evidence_gap"] = "deterministic"


@dataclass
class FinalAnswer:
    """The durable presentation output after the scientific DAG is verified.

    The Planner owns the answer strategy; this object is written only after
    arbitrary decision-DAG synthesis and goal verification. It is deliberately separated from
    component conclusions so an incomplete run still publishes a precise
    statement of its current conclusion boundary instead of an empty report.
    """

    answer_node_id: str
    status: Literal["proven", "needs_plan_patch", "evidence_gap"]
    conclusion: str
    observed_pattern: list[str]
    supported_interpretation: list[str]
    alternative_explanations: list[str]
    evidence_gaps: list[str]
    next_actions: list[str]
    component_ids: list[str]
    synthesis_fingerprint: str
    writer_verdict: Literal["deterministic", "synthesized", "rejected"] = "deterministic"


@dataclass
class JudgeDecision:
    action: NextAction
    component_judgments: list[ComponentJudgment]
    diagnoses: list[MissingDiagnosis]
    evidence_gaps: list[EvidenceGap]
    reason: str
    affected_component_ids: list[str] = field(default_factory=list)
    assessments: list[EvidenceAssessment] = field(default_factory=list)


@dataclass
class PlannerReview:
    """Top-level planning decision after a bounded local execution attempt.

    This is deliberately distinct from a Judge verdict.  The Judge reports
    a frozen test outcome; the Planner preserves, revises, or bounds the
    global scientific map without directing local tool execution.
    """

    route: PlannerRoute
    component_id: str
    rationale: str
    diagnoses: list[MissingDiagnosis] = field(default_factory=list)
    affected_component_ids: list[str] = field(default_factory=list)
    operations: list[PlanPatchOperation] = field(default_factory=list)
    review_id: str = ""
    plan_revision: int = 0
    local_graph_revision: int = 0


@dataclass
class ResearchAction:
    """One local execution-graph node proposed by the Researcher.

    ``depends_on_action_ids`` are the directed edges.  They are deliberately
    separate from the scientific-plan edges: these nodes describe how the
    current scientific claim is tested, never what the global claim means.
    """

    action_id: str
    component_id: str
    kind: Literal["tool", "data_tool", "source_query", "local_rescout", "finish_component"]
    tool_id: str | None = None
    source_id: str | None = None
    input_scope: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, Any] = field(default_factory=dict)
    # ``data_request`` is intentionally separate from generic tool parameters.
    # It is a strict, source-bound relational operation proposed by the
    # Researcher, not an opaque JSON/SQL string or a permanent source adapter.
    data_request: dict[str, Any] = field(default_factory=dict)
    # A durable reference to the materialize evidence node whose rows feed a
    # numerical provider.  It is emitted through the strict action schema,
    # never guessed from an action label or a source name.
    data_handle_evidence_id: str = ""
    # Cross-local dependencies are converted to durable evidence references;
    # action ids from another LocalExecutionGraph never cross the boundary.
    input_evidence_ids: list[str] = field(default_factory=list)
    rationale: str = ""
    expected_evidence_tags: list[str] = field(default_factory=list)
    depends_on_action_ids: list[str] = field(default_factory=list)
    local_node_label: str = ""


LocalActionStatus = Literal["pending", "running", "completed", "failed", "skipped", "blocked"]
LocalGraphExecutionStatus = Literal["open", "running", "completed", "exhausted", "escalated"]


@dataclass
class LocalExecutionGraph:
    """Durable, component-scoped execution DAG owned by the Researcher.

    The graph is intentionally an execution artifact.  Its nodes can be
    added by the Researcher as evidence arrives, but it cannot alter a global
    claim or dependency.  The runtime only schedules
    nodes whose declared inputs have completed; it does not infer a route.
    """

    graph_id: str
    component_id: str
    actions: list[ResearchAction] = field(default_factory=list)
    action_status: dict[str, LocalActionStatus] = field(default_factory=dict)
    action_evidence_ids: dict[str, list[str]] = field(default_factory=dict)
    tool_execution_failures: list[FailureRecord] = field(default_factory=list)
    model_action_proposal_failures: list[FailureRecord] = field(default_factory=list)
    contract_failures: list[FailureRecord] = field(default_factory=list)
    action_input_fingerprints: dict[str, str] = field(default_factory=dict)
    rejected_action_fingerprints: list[str] = field(default_factory=list)
    execution_status: LocalGraphExecutionStatus = "open"
    scientific_status: EvidenceStatus = "not_tested"
    missing_evidence: list[str] = field(default_factory=list)
    capability_coverage: dict[str, Any] = field(default_factory=dict)
    repair_history: list[dict[str, Any]] = field(default_factory=list)
    escalation_note: str = ""
    planner_directive: str = ""
    revision: int = 1
    diagnostic_epoch: int = 1

    def action(self, action_id: str) -> ResearchAction:
        for action in self.actions:
            if action.action_id == action_id:
                return action
        raise KeyError(action_id)

    def ready_actions(self) -> list[ResearchAction]:
        """Return actions whose explicit local inputs have completed."""

        completed = {
            action_id for action_id, status in self.action_status.items()
            if status == "completed"
        }
        return [
            action for action in self.actions
            if self.action_status.get(action.action_id, "pending") == "pending"
            and set(action.depends_on_action_ids).issubset(completed)
        ]

    def add_actions(self, actions: list[ResearchAction]) -> None:
        known = {action.action_id for action in self.actions}
        for action in actions:
            if action.action_id in known:
                continue
            self.actions.append(action)
            self.action_status[action.action_id] = "pending"
            known.add(action.action_id)
        if actions:
            self.execution_status = "open"
            self.revision += 1

    def open_local_repair(self, *, missing_evidence: list[str], directive: str = "") -> None:
        """Open a bounded repair attempt without transferring method choice upstream."""

        self.execution_status = "open"
        self.missing_evidence = list(missing_evidence)
        self.planner_directive = directive
        self.repair_history.append({
            "revision": self.revision,
            "missing_evidence": list(missing_evidence),
            "previous_action_ids": [action.action_id for action in self.actions],
            "directive": directive,
        })
        self.diagnostic_epoch += 1
        self.scientific_status = "not_tested"
        self.escalation_note = ""
        self.revision += 1

    def to_local_view(self, scientific_neighbourhood: dict[str, Any]) -> dict[str, Any]:
        """The bounded shared workspace passed to the Researcher model."""

        execution_failures = [item for item in self.tool_execution_failures if item.diagnostic_epoch == self.diagnostic_epoch]
        proposal_failures = [item for item in self.model_action_proposal_failures if item.diagnostic_epoch == self.diagnostic_epoch]
        contract_failures = [item for item in self.contract_failures if item.diagnostic_epoch == self.diagnostic_epoch]
        by_action: dict[str, list[FailureRecord]] = {}
        for item in [*execution_failures, *contract_failures]:
            if item.action_id:
                by_action.setdefault(item.action_id, []).append(item)

        return {
            "scientific_neighbourhood": scientific_neighbourhood,
            "execution_graph": {
                "graph_id": self.graph_id,
                "component_id": self.component_id,
                "execution_status": self.execution_status,
                "scientific_status": self.scientific_status,
                "revision": self.revision,
                "nodes": [
                    {
                        "action_id": action.action_id,
                        "kind": action.kind,
                        "label": action.local_node_label,
                        "tool_id": action.tool_id,
                        "source_id": action.source_id,
                        "input_scope": action.input_scope,
                        "parameters": action.parameters,
                        "expected_evidence_tags": action.expected_evidence_tags,
                        "status": self.action_status.get(action.action_id, "pending"),
                        "evidence_ids": self.action_evidence_ids.get(action.action_id, []),
                        "input_evidence_ids": list(action.input_evidence_ids),
                        "current_failures": [to_dict(item) for item in by_action.get(action.action_id, [])],
                        "input_fingerprint": self.action_input_fingerprints.get(action.action_id, ""),
                    }
                    for action in self.actions
                ],
                "edges": [
                    {"from": upstream, "to": action.action_id, "meaning": "upstream output is required input"}
                    for action in self.actions for upstream in action.depends_on_action_ids
                ],
                "missing_evidence": self.missing_evidence,
                "capability_coverage": self.capability_coverage,
                "failure_channels": {
                    "tool_execution": [to_dict(item) for item in execution_failures],
                    "model_action_proposal": [to_dict(item) for item in proposal_failures],
                    "contract_validation": [to_dict(item) for item in contract_failures],
                },
                "historical_failure_counts": {
                    "tool_execution": len(self.tool_execution_failures) - len(execution_failures),
                    "model_action_proposal": len(self.model_action_proposal_failures) - len(proposal_failures),
                    "contract_validation": len(self.contract_failures) - len(contract_failures),
                },
                "rejected_action_fingerprints": self.rejected_action_fingerprints,
                "repair_history": self.repair_history,
                "planner_directive": self.planner_directive,
            },
        }


@dataclass
class RunSnapshot:
    run_id: str
    stage: RunStage
    question: str
    research_brief: ResearchBrief | None = None
    task: QuestionTask | None = None
    scout_reports: list[ScoutReport] = field(default_factory=list)
    plan: ScientificPlan | None = None
    research_notes: list[Any] = field(default_factory=list)
    decisions: list[JudgeDecision] = field(default_factory=list)
    component_conclusions: list[ComponentConclusion] = field(default_factory=list)
    dag_synthesis: DAGSynthesis | None = None
    goal_assessment: GoalAssessment | None = None
    active_component_ids: list[str] = field(default_factory=list)
    outer_round: int = 0
    inner_step: int = 0
    revision: int = 0
    local_execution_graphs: dict[str, LocalExecutionGraph] = field(default_factory=dict)
    final_answer: FinalAnswer | None = None
    planner_reviews: list[PlannerReview] = field(default_factory=list)
    pending_planner_review: PlannerReview | None = None
    failure_history: list[FailureRecord] = field(default_factory=list)
    schema_version: int = 2


@dataclass
class ResearchReport:
    question: str
    markdown: str
    task: QuestionTask
    plan: ScientificPlan
    scout_reports: list[ScoutReport]
    evidence: list[EvidenceEntry]
    judgments: list[ComponentJudgment]
    evidence_gaps: list[EvidenceGap]
    next_observations: list[str]
    research_notes: list[Any] = field(default_factory=list)
    claim_manifest: list[Any] = field(default_factory=list)
    verified_sources: list[Any] = field(default_factory=list)
    component_conclusions: list[ComponentConclusion] = field(default_factory=list)
    dag_synthesis: DAGSynthesis | None = None
    goal_assessment: GoalAssessment | None = None
    final_answer: FinalAnswer | None = None
    writer_sections: list[WriterSection] = field(default_factory=list)
    research_brief: ResearchBrief | None = None


def to_dict(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return to_dict(value.model_dump(mode="json"))
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, list):
        return [to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: to_dict(item) for key, item in value.items()}
    return value
