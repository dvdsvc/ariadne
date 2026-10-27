"""Context-isolated node conclusions and question-level DAG verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .aiq_grounding import ReportClaim
from .contracts import (
        ComponentConclusion, ComponentJudgment, DAGSynthesis, EvidenceAssessment,
    EvidenceEntry, EvidenceGap, FinalAnswer, GoalAssessment, MissingDiagnosis,
    ScientificAnswerStrategy, ScientificPlan, RequiredComponent,
    WriterSection,
)


@dataclass(frozen=True)
class NodeConclusionPacket:
    """The complete and only input context of a node writer."""

    component: RequiredComponent
    assessment: EvidenceAssessment
    entries: list[EvidenceEntry]

    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class NodeConclusionAdvisor(Protocol):
    def conclude(self, packet: NodeConclusionPacket) -> dict[str, Any]: ...


class GoalVerificationAdvisor(Protocol):
    def verify(self, *, question: str, plan: ScientificPlan, synthesis: DAGSynthesis) -> dict[str, Any]: ...


@dataclass(frozen=True)
class FinalAnswerPacket:
    """The complete context of the final presentation writer.

    It deliberately contains the original question and Planner-owned answer
    contract, but never raw Ledger entries, Scout records, tools, or Researcher
    trajectory.  Component conclusions are already evidence-bound outputs.
    """

    question: str
    research_brief: Any | None
    answer_strategy: ScientificAnswerStrategy | None
    synthesis: DAGSynthesis
    goal_assessment: GoalAssessment
    evidence_gaps: list[EvidenceGap]
    local_sections: list[WriterSection]


class FinalAnswerAdvisor(Protocol):
    def write_final_answer(self, packet: FinalAnswerPacket) -> dict[str, Any]: ...


class IsolatedNodeWriter:
    """Write one bounded conclusion without access to research context.

    A model may improve wording, but it cannot alter the R-Judge assessment or
    cite evidence outside the supplied packet.  Invalid/unavailable model output
    fails closed to an ``insufficient`` local conclusion.
    """

    def __init__(self, advisor: NodeConclusionAdvisor | None = None, claim_advisor: Any | None = None) -> None:
        self.advisor, self.claim_advisor = advisor, claim_advisor

    def write(self, component: RequiredComponent, assessment: EvidenceAssessment, entries: list[EvidenceEntry]) -> ComponentConclusion:
        packet = NodeConclusionPacket(component, assessment, list(entries))
        allowed_ids = set(assessment.evidence_ids)
        direct = [entry.result_summary for entry in entries if entry.evidence_id in allowed_ids]
        limitations = [item for entry in entries for item in [*entry.limitations, *entry.warnings]]
        status = assessment.outcome
        conclusion = self._default_conclusion(component, assessment)
        verdict = "deterministic"
        if self.advisor is not None:
            try:
                proposal = self.advisor.conclude(packet)
                proposed_status = proposal.get("status")
                proposed_ids = [item for item in proposal.get("evidence_ids", []) if isinstance(item, str)]
                proposed_text = proposal.get("conclusion")
                if proposed_status == status and isinstance(proposed_text, str) and proposed_text.strip() and set(proposed_ids).issubset(allowed_ids):
                    conclusion = proposed_text.strip()
                    direct = [item for item in proposal.get("observed_results", []) if isinstance(item, str)] or direct
                    limitations = [item for item in proposal.get("limitations", []) if isinstance(item, str)] or limitations
                    allowed_ids = set(proposed_ids) or allowed_ids
                    verdict = "entailed"
                else:
                    status, conclusion, verdict = "insufficient", "The isolated node writer could not bind its wording to the supplied contract and evidence packet.", "rejected"
            except Exception:
                status, conclusion, verdict = "insufficient", "The isolated node writer was unavailable; this node cannot be promoted beyond an evidence gap.", "rejected"
        if self.claim_advisor is not None and status in {"supported", "contradicted"}:
            try:
                claim = ReportClaim(
                    claim_id=f"claim_{component.component_id}", component_id=component.component_id,
                    category="interpretation" if assessment.allowed_claim_level == "mechanism_association" else "observed_pattern",
                    text=conclusion, evidence_ids=sorted(allowed_ids),
                )
                evidence = [entry for entry in entries if entry.evidence_id in allowed_ids]
                if self.claim_advisor.verify_claim(claim=claim, evidence=evidence).get("verdict") != "entailed":
                    status, conclusion, verdict = "insufficient", "The isolated claim-entailment check rejected this node conclusion.", "rejected"
            except Exception:
                status, conclusion, verdict = "insufficient", "The isolated claim-entailment check was unavailable; this node conclusion is withheld.", "rejected"
        return ComponentConclusion(
            component_id=component.component_id, status=status,
            conclusion=conclusion, observed_results=direct, limitations=list(dict.fromkeys(limitations)),
            evidence_ids=sorted(allowed_ids), allowed_claim_level=assessment.allowed_claim_level if status == assessment.outcome else "none",
            packet_fingerprint=packet.fingerprint(), writer_verdict=verdict,
        )

    @staticmethod
    def _default_conclusion(component: RequiredComponent, assessment: EvidenceAssessment) -> str:
        if assessment.outcome == "supported":
            return f"Within the declared component scope, the supplied calculation results support: {component.claim}"
        if assessment.outcome == "contradicted":
            return f"Within the declared component scope, the supplied calculation results contradict: {component.claim}"
        return f"Within the declared component scope, no conclusion is released for: {component.claim}. Missing evidence: {', '.join(assessment.missing_evidence) or 'unspecified'}"


class DAGSynthesizer:
    """Combines node conclusions through the Plan DAG without viewing Ledger data."""

    def synthesize(self, plan: ScientificPlan, conclusions: list[ComponentConclusion]) -> DAGSynthesis:
        by_component = {item.component_id: item for item in conclusions}
        # The Planner's scientific DAG is the answer boundary. There is no
        # second answer-strategy list that can silently omit or force nodes.
        required = [item.component_id for item in plan.required_components]
        blocked, terminal, summary = [], [], []
        for component in plan.required_components:
            if component.execution_status == "superseded":
                summary.append(f"{component.component_id}: superseded by a later Planner map revision; prior evidence remains in the Ledger but is not an active answer requirement.")
                continue
            conclusion = by_component.get(component.component_id)
            if conclusion is None:
                blocked.append(component.component_id)
                summary.append(f"{component.component_id}: no isolated conclusion was produced.")
                continue
            dependency_statuses = [by_component[dependency].status if dependency in by_component else "not_tested" for dependency in component.depends_on]
            if "contradicted" in dependency_statuses:
                terminal.append(component.component_id)
                summary.append(f"{component.component_id}: pruned because an upstream prerequisite was contradicted.")
            elif any(status != "supported" for status in dependency_statuses):
                blocked.append(component.component_id)
                summary.append(f"{component.component_id}: dependency conclusion is not supported.")
            elif conclusion.status in {"supported", "contradicted"}:
                terminal.append(component.component_id)
                summary.append(f"{component.component_id}: {conclusion.status} from the R-Judge result packet.")
            else:
                blocked.append(component.component_id)
                summary.append(f"{component.component_id}: {conclusion.status}.")
        return DAGSynthesis(
            plan_revision=plan.plan_revision, root_goal=plan.scientific_goal,
            component_conclusions=[by_component[item.component_id] for item in plan.required_components if item.component_id in by_component and item.execution_status != "superseded"],
            required_component_ids=required, blocked_component_ids=list(dict.fromkeys(blocked)),
            terminal_component_ids=list(dict.fromkeys(terminal)), dependency_summary=summary,
        )


class GoalPlanVerifier:
    """Checks whether graph-combined node conclusions answer the original task.

    It receives the question and Plan only at this final control boundary, plus
    the conclusion-only DAG synthesis.  Raw Ledger entries and the Researcher
    trajectory are deliberately unavailable.
    """

    def __init__(self, advisor: GoalVerificationAdvisor | None = None) -> None:
        self.advisor = advisor

    def verify(self, question: str, plan: ScientificPlan, synthesis: DAGSynthesis, *, exhausted: bool = False) -> GoalAssessment:
        fingerprint = hashlib.sha256(json.dumps(asdict(synthesis), ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        deterministic = self._deterministic(plan, synthesis, exhausted, fingerprint)
        if self.advisor is None:
            return deterministic
        try:
            proposal = self.advisor.verify(question=question, plan=plan, synthesis=synthesis)
            status = proposal.get("status")
            blockers = [item for item in proposal.get("blocking_component_ids", []) if item in {component.component_id for component in plan.required_components}]
            missing = [item for item in proposal.get("missing_requirements", []) if isinstance(item, str)]
            reason = proposal.get("reason")
            if status not in {"proven", "needs_plan_patch", "evidence_gap"} or not isinstance(reason, str):
                return GoalAssessment("needs_plan_patch", "Goal verifier returned an invalid bounded decision.", deterministic.blocking_component_ids, deterministic.missing_requirements, deterministic.diagnoses, fingerprint, "needs_plan_patch")
            # A model cannot release an answer while the deterministic graph has
            # unresolved required nodes.
            if status == "proven" and deterministic.status != "proven":
                return GoalAssessment("needs_plan_patch", "Goal verifier attempted to promote an unresolved DAG.", deterministic.blocking_component_ids, deterministic.missing_requirements, deterministic.diagnoses, fingerprint, "needs_plan_patch")
            resolved_blockers = blockers or deterministic.blocking_component_ids
            if status == "needs_plan_patch" and not resolved_blockers:
                return deterministic
            diagnoses = deterministic.diagnoses or [
                MissingDiagnosis(component_id=item, kind="calculation_missing", reason="The bounded goal verifier requested a focused repair for this existing Plan node.", missing_observables=[item], suggested_sensor_kinds=[], expected_information_gain="medium")
                for item in resolved_blockers
            ]
            return GoalAssessment(status, reason, resolved_blockers, missing or deterministic.missing_requirements, diagnoses, fingerprint, status)
        except Exception:
            return deterministic

    @staticmethod
    def _deterministic(plan: ScientificPlan, synthesis: DAGSynthesis, exhausted: bool, fingerprint: str) -> GoalAssessment:
        required = set(synthesis.required_component_ids)
        terminal = set(synthesis.terminal_component_ids)
        blockers = [item for item in synthesis.blocked_component_ids if item in required] or sorted(required - terminal)
        blockers = list(dict.fromkeys(blockers))
        if not blockers:
            return GoalAssessment("proven", "Every active scientific decision node has a terminal isolated conclusion and all DAG dependencies are satisfied.", [], [], [], fingerprint)
        diagnoses = []
        for item in blockers:
            sensor_kinds: list[str] = []
            if "spatial" in item and "spatial_constraint" not in sensor_kinds:
                sensor_kinds.insert(0, "spatial_constraint")
            elif not sensor_kinds and ("temporal" in item or "event" in item):
                sensor_kinds = ["event_clock_sensor"]
            diagnoses.append(MissingDiagnosis(
                component_id=item,
                kind="calculation_missing",
                reason="The graph-level verifier cannot establish the root question because this required node lacks a terminal isolated conclusion.",
                missing_observables=[item],
                suggested_sensor_kinds=list(dict.fromkeys(sensor_kinds)),
                expected_information_gain="medium",
            ))
        if exhausted:
            missing = list(plan.research_brief.must_answer) if plan.research_brief else list(blockers)
            return GoalAssessment("evidence_gap", "The available observation world does not close the Research Brief's knowledge gaps.", blockers, missing, diagnoses, fingerprint)
        missing = list(plan.research_brief.must_answer) if plan.research_brief else list(blockers)
        return GoalAssessment("needs_plan_patch", "The current DAG synthesis still has unresolved scientific decision nodes; repair the blocking nodes without redefining the Research Brief.", blockers, missing, diagnoses, fingerprint)


class FinalAnswerWriter:
    """Write the final presentation from conclusion-only graph state.

    This is not the Planner: the Planner supplies ``ScientificAnswerStrategy``
    and owns any plan change.  This writer can only synthesize terminal node
    conclusions or state the verified boundary of the current run.
    """

    def __init__(self, advisor: FinalAnswerAdvisor | None = None) -> None:
        self.advisor = advisor

    def write(
        self,
        question: str,
        plan: ScientificPlan,
        synthesis: DAGSynthesis,
        goal: GoalAssessment,
        evidence_gaps: list[EvidenceGap],
        local_sections: list[WriterSection] | None = None,
    ) -> FinalAnswer:
        packet = FinalAnswerPacket(question, plan.research_brief, plan.answer_strategy, synthesis, goal, list(evidence_gaps), list(local_sections or []))
        deterministic = self._deterministic(plan, packet)
        if self.advisor is None:
            return deterministic
        try:
            proposal = self.advisor.write_final_answer(packet)
            valid_ids = {item.component_id for item in synthesis.component_conclusions}
            component_ids = [item for item in proposal.get("component_ids", []) if isinstance(item, str) and item in valid_ids]
            status = proposal.get("status")
            conclusion = proposal.get("conclusion")
            if status != goal.status or not isinstance(conclusion, str) or not conclusion.strip():
                return self._rejected(deterministic)
            return FinalAnswer(
                answer_node_id=deterministic.answer_node_id,
                status=status,
                conclusion=conclusion.strip(),
                observed_pattern=self._strings(proposal.get("observed_pattern")),
                supported_interpretation=self._strings(proposal.get("supported_interpretation")),
                alternative_explanations=self._strings(proposal.get("alternative_explanations")),
                evidence_gaps=self._strings(proposal.get("evidence_gaps")),
                next_actions=self._strings(proposal.get("next_actions")),
                component_ids=component_ids or deterministic.component_ids,
                synthesis_fingerprint=goal.synthesis_fingerprint,
                writer_verdict="synthesized",
            )
        except Exception:
            return self._rejected(deterministic)

    @staticmethod
    def _strings(value: Any) -> list[str]:
        return [item.strip() for item in value if isinstance(item, str) and item.strip()] if isinstance(value, list) else []

    @staticmethod
    def _rejected(fallback: FinalAnswer) -> FinalAnswer:
        return FinalAnswer(
            **{**asdict(fallback), "writer_verdict": "rejected"},
        )

    @staticmethod
    def _deterministic(plan: ScientificPlan, packet: FinalAnswerPacket) -> FinalAnswer:
        conclusions = {item.component_id: item for item in packet.synthesis.component_conclusions}
        observed, interpretations, alternatives = [], [], []
        for component_id in packet.synthesis.terminal_component_ids:
            item = conclusions.get(component_id)
            if item is None:
                continue
            text = item.conclusion
            if item.status == "contradicted":
                alternatives.append(text)
            elif item.allowed_claim_level == "mechanism_association":
                interpretations.append(text)
            else:
                observed.append(text)
        formal_gaps = [gap.conclusion_limit or gap.reason for gap in packet.evidence_gaps]
        formal_gaps.extend(packet.goal_assessment.missing_requirements)
        formal_gaps = list(dict.fromkeys(item for item in formal_gaps if item))
        if packet.goal_assessment.status == "proven":
            conclusion = "The global Plan requirements are satisfied by the supplied isolated node conclusions. The statements below are limited to those verified component conclusions."
            next_actions: list[str] = []
        elif packet.goal_assessment.status == "evidence_gap":
            conclusion = "The available observation world does not establish a scientific answer to the question. No claim beyond the supplied component conclusions is released; the items below are formal evidence gaps, not negative geological findings."
            next_actions = ["Treat the listed missing requirements as the next observation or data-integration targets."]
        else:
            conclusion = "The current run has not established a scientific answer. No geological conclusion is released because the global Plan still has blocking requirements that require repair or further execution."
            next_actions = ["Return the listed blocking requirements to the Planner for a focused Plan revision."]
        answer_node_id = "answer:final"
        return FinalAnswer(
            answer_node_id=answer_node_id,
            status=packet.goal_assessment.status,
            conclusion=conclusion,
            observed_pattern=observed,
            supported_interpretation=interpretations,
            alternative_explanations=alternatives,
            evidence_gaps=formal_gaps,
            next_actions=next_actions,
            component_ids=[item.component_id for item in packet.synthesis.component_conclusions],
            synthesis_fingerprint=packet.goal_assessment.synthesis_fingerprint,
        )
