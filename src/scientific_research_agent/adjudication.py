"""Context-isolated assessment of Planner decision claims against Ledger evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .contracts import CriterionEvaluation, EvidenceAssessment, EvidenceEntry, EvidenceStatus, RequiredComponent, ScientificPlan
from .ledger import EvidenceLedger
from .plan_graph import component_execution_epoch, ensure_plan_graph, evidence_input_component_ids


@dataclass(frozen=True)
class EvidencePacket:
    """The only context the isolated R-layer Judge is allowed to receive."""

    component_id: str
    claim: str
    scope: dict[str, Any]
    entries: list[EvidenceEntry]

    def fingerprint(self) -> str:
        value = {
            "component_id": self.component_id,
            "claim": self.claim,
            "scope": self.scope,
            "entries": [asdict(entry) for entry in self.entries],
        }
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class EvidenceReviewer(Protocol):
    """Judge one isolated subproblem from its claim and calculation results."""

    def review(self, packet: EvidencePacket) -> dict[str, Any]: ...


class EvidenceAdjudicator:
    """Evaluates one component without receiving question or researcher context."""

    def __init__(self, reviewer: EvidenceReviewer | None = None) -> None:
        self.reviewer = reviewer

    def assess_component(self, plan: ScientificPlan, component: RequiredComponent, ledger: EvidenceLedger) -> list[EvidenceAssessment]:
        ensure_plan_graph(plan)
        entries = self.entries_for_component(plan, component, ledger)
        return [self._assess_result_packet(component, entries)]

    @staticmethod
    def entries_for_component(plan: ScientificPlan, component: RequiredComponent, ledger: EvidenceLedger) -> list[EvidenceEntry]:
        """Expose only evidence that reaches this component through DAG edges."""

        visible_component_ids = evidence_input_component_ids(plan, component.component_id)
        epochs = {component_id: component_execution_epoch(plan, component_id) for component_id in visible_component_ids}
        current: list[EvidenceEntry] = []
        for entry in ledger.for_components(visible_component_ids):
            producers = visible_component_ids.intersection(entry.component_ids)
            if entry.execution_epoch:
                if any(entry.execution_epoch == epochs[producer] for producer in producers):
                    current.append(entry)
                continue
            # Legacy snapshots without an epoch remain readable only until a
            # route-changing Planner patch invalidates their producer.
            if any(_legacy_entry_survives_plan_patches(plan, producer, entry) for producer in producers):
                current.append(entry)
        return current

    def _assess_result_packet(self, component: RequiredComponent, entries: list[EvidenceEntry]) -> EvidenceAssessment:
        packet = EvidencePacket(component.component_id, component.claim, dict(component.scope), list(entries))
        valid_entries = [entry for entry in entries if entry.status == "ok"]
        all_evidence_ids = [entry.evidence_id for entry in valid_entries]
        criteria: list[CriterionEvaluation] = []
        outcome = "insufficient"
        missing = ["semantic_judgment"]
        reviewer_verdict = "not_requested"
        confidence: float | None = None
        if self.reviewer is not None and valid_entries:
            try:
                review = self.reviewer.review(packet)
                # The isolated reviewer is the only producer of the scientific
                # component verdict. Calculation status, values, uncertainty,
                # and provenance are inputs to that review, never pass/fail
                # rules owned by a provider or this deterministic layer.
                semantic_status = review.get("status")
                if semantic_status in {"supported", "contradicted", "insufficient"}:
                    allowed_ids = {entry.evidence_id for entry in valid_entries}
                    selected_ids = [item for item in review.get("evidence_ids", []) if item in allowed_ids]
                    reported_missing = [item for item in review.get("missing_evidence", []) if isinstance(item, str)]
                    if not selected_ids:
                        semantic_status = "insufficient"
                        reported_missing = list(dict.fromkeys([*reported_missing, "independent_evidence_binding"]))
                    outcome = semantic_status
                    reviewer_verdict = semantic_status
                    reported_confidence = review.get("confidence")
                    if isinstance(reported_confidence, (int, float)) and not isinstance(reported_confidence, bool):
                        confidence = float(reported_confidence)
                    missing = reported_missing if semantic_status == "insufficient" else []
                    all_evidence_ids = selected_ids
                    criteria.append(CriterionEvaluation("subproblem_entailment", "passed" if semantic_status == "supported" else "failed" if semantic_status == "contradicted" else "missing", selected_ids, str(review.get("rationale", ""))))
                    rationale = str(review.get("rationale", "")) or _rationale(outcome, missing)
                else:
                    reviewer_verdict = review.get("verdict", "not_entailed")
                    outcome = "supported" if reviewer_verdict == "entailed" else "insufficient"
                    missing = [] if outcome == "supported" else ["independent_entailment_review"]
                    criteria.append(CriterionEvaluation("independent_entailment_review", "passed" if outcome == "supported" else "missing", [], "The isolated reviewer supplied the semantic verdict."))
                    rationale = _rationale(outcome, missing)
            except Exception:
                reviewer_verdict = "not_entailed"
                outcome = "insufficient"
                missing = ["independent_entailment_review"]
                criteria.append(CriterionEvaluation("independent_entailment_review", "missing", [], "The isolated reviewer was unavailable; fail closed."))
                rationale = _rationale(outcome, missing)
        else:
            rationale = _rationale(outcome, missing)
        gap_kind = _gap_kind(entries, missing, component)
        return EvidenceAssessment(
            assessment_id=f"A:{component.component_id}:{_short(packet.fingerprint())}",
            component_id=component.component_id, outcome=outcome,
            criterion_evaluations=criteria, evidence_ids=sorted(set(all_evidence_ids)),
            missing_evidence=missing, gap_kind=gap_kind if outcome == "insufficient" else None,
            allowed_claim_level=_allowed_claim_level(component, outcome),
            rationale=rationale, packet_fingerprint=packet.fingerprint(),
            reviewer_verdict=reviewer_verdict,
            confidence=confidence,
        )

def _legacy_entry_survives_plan_patches(plan: ScientificPlan, component_id: str, entry: EvidenceEntry) -> bool:
    """Fail closed when an old unversioned entry predates a route-changing patch."""

    graph = ensure_plan_graph(plan)
    invalidating = [
        patch for patch in graph.patches
        if component_id in patch.affected_component_ids
        and patch.kind in {"global_patch", "local_rescout", "redirect", "supersede"}
    ]
    if not invalidating:
        return True
    if not entry.recorded_at:
        return False
    return all(entry.recorded_at >= patch.created_at for patch in invalidating)


def _gap_kind(entries: list[EvidenceEntry], missing: list[str], component: RequiredComponent) -> str:
    if any(entry.status == "data_unavailable" for entry in entries):
        external = {"plate_rotation", "reconstruct_paleolocation", "external_boundary_condition", "distance_to_trench", "plate_velocity"}
        if external.intersection(missing) or external.intersection(component.evidence_needed):
            return "missing_external_sensor"
        return "wrong_scope_or_data_assumption"
    if {"spatial_location", "spatial_uncertainty", "resolution_assessment"}.intersection(missing):
        return "resolution_limit"
    if component.component_id == "mechanism_discrimination" and "material_state" in missing:
        return "fundamentally_unidentifiable"
    return "calculation_missing"


def _allowed_claim_level(component: RequiredComponent, outcome: str) -> str:
    if outcome != "supported":
        return "none"
    mechanism_words = ("机制", "外部边界", "候选机制", "mechanism", "boundary")
    return "mechanism_association" if any(word in component.claim for word in mechanism_words) else "component_only"


def _rationale(outcome: str, missing: list[str]) -> str:
    if outcome == "supported":
        return "The R-Judge found that the supplied calculation results support the component claim."
    if outcome == "contradicted":
        return "The R-Judge found that the supplied calculation results contradict the component claim."
    return f"The supplied calculation results do not resolve the component claim: {', '.join(missing) or 'unspecified evidence gap'}."


def _short(value: str) -> str:
    return value[:12]
