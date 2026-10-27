"""Append-only evolution of a scientific plan after independently assessed evidence."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Iterable

from .contracts import (
    EvidenceAssessment, JudgeDecision, MissingDiagnosis, PlannerReview,
    PlanNode, PlanPatch, PlanPatchOperation, RequiredComponent,
    ScientificPlan, ScoutReport,
)
from .plan_graph import ensure_plan_graph, expected_plan_edges, validate_plan_graph


class PlanEvolver:
    """Changes the future frontier while preserving result assessments."""

    def apply_decision(self, plan: ScientificPlan, scout: ScoutReport, decision: JudgeDecision) -> ScientificPlan:
        """Append a revision for every evidence verdict, even when no re-scout occurs."""

        patched = deepcopy(plan)
        graph = ensure_plan_graph(patched)
        base_revision = patched.plan_revision
        next_revision = base_revision + 1
        operations: list[PlanPatchOperation] = []
        affected: set[str] = set()
        existing_assessment_ids = {assessment.assessment_id for assessment in graph.assessments}
        for assessment in decision.assessments:
            if not any(existing.assessment_id == assessment.assessment_id for existing in graph.assessments):
                graph.assessments.append(assessment)
        for judgment in decision.component_judgments:
            if judgment.assessment_ids and all(assessment_id in existing_assessment_ids for assessment_id in judgment.assessment_ids):
                # Resume may replay the same deterministic Judge packet after a
                # durable Planner-review checkpoint.  The assessment is an
                # identity-bearing fact, so its reducer transition is idempotent.
                continue
            affected.add(judgment.component_id)
            component = patched.component(judgment.component_id)
            component.scientific_status = judgment.status
            decision_node = graph.component_node(judgment.component_id)
            decision_node.scientific_status = judgment.status
            operations.append(PlanPatchOperation("set_status", decision_node.node_id, {"scientific_status": judgment.status, "assessment_ids": judgment.assessment_ids}))
        # The Judge records why the current component packet is insufficient, but never
        # changes the scientific map.  Only the top-level Planner may turn
        # that diagnosis into a new component, dependency, re-opened route,
        # or evidence gap.
        if not operations:
            # A decision with no newly assessed component is not a legitimate
            # plan mutation.  The runtime simply leaves the revision untouched.
            return patched
        patch_kind = self._patch_kind(decision)
        patch = PlanPatch(
            patch_id=f"P{next_revision:04d}", base_revision=base_revision, next_revision=next_revision,
            kind=patch_kind,
            trigger_assessment_ids=[assessment.assessment_id for assessment in decision.assessments],
            affected_component_ids=sorted(affected), reason=decision.reason,
            operations=operations, created_at=datetime.now(timezone.utc).isoformat(),
        )
        graph.patches.append(patch)
        patched.plan_revision = next_revision
        patched.scout_fingerprint = scout.world_fingerprint
        patched.current_sensor_limits = list(scout.sensor_limits)
        patched.known_evidence_gaps = list(patched.known_evidence_gaps)
        validate_plan_graph(patched, graph)
        return patched

    def apply_global_review(self, plan: ScientificPlan, scout: ScoutReport, review: PlannerReview) -> ScientificPlan:
        """Apply a Planner-owned scientific-map patch without touching Ledger history.

        This reducer deliberately has no knowledge of tools or local action
        graphs.  It accepts only typed map operations already proposed by the
        Planner, rebuilds affected scientific dependency edges, and leaves old
        evidence and prior component assessments attached to the old component.
        """

        if review.route != "patch_graph" or not review.operations:
            raise ValueError("apply_global_review requires a non-empty patch_graph review")
        patched = deepcopy(plan)
        graph = ensure_plan_graph(patched)
        base_revision = patched.plan_revision
        next_revision = base_revision + 1
        known_ids = {item.component_id for item in patched.required_components}
        operations: list[PlanPatchOperation] = []
        affected: set[str] = set(review.affected_component_ids)

        for proposed in review.operations:
            operation = proposed.operation
            if operation == "add_component":
                component = self._component_from_payload(proposed.payload, scout)
                if component.component_id in known_ids:
                    raise ValueError(f"Planner patch duplicates component {component.component_id}")
                if not set(component.depends_on).issubset(known_ids):
                    raise ValueError("A newly added component may depend only on components already present in this revision")
                patched.required_components.append(component)
                known_ids.add(component.component_id)
                self._add_component_graph_nodes(patched, graph, component, next_revision)
                if patched.answer_strategy and component.component_id not in patched.answer_strategy.required_component_ids:
                    patched.answer_strategy.required_component_ids.append(component.component_id)
                operations.append(PlanPatchOperation("add_component", component.component_id, {"component": proposed.payload, "reason": proposed.payload.get("reason", "")}))
                affected.add(component.component_id)
            elif operation == "supersede":
                target = proposed.target_id
                self._require_component(target, known_ids)
                patched.component(target).execution_status = "superseded"
                patched.component(target).scientific_status = "not_tested"
                node = graph.component_node(target)
                node.execution_status = "superseded"
                node.metadata.setdefault("superseded", []).append({"revision": next_revision, "reason": proposed.payload.get("reason", "")})
                operations.append(PlanPatchOperation("supersede", node.node_id, {"reason": proposed.payload.get("reason", "")}))
                affected.add(target)
            elif operation == "reconnect_dependencies":
                target = proposed.target_id
                self._require_component(target, known_ids)
                dependencies = list(proposed.payload.get("depends_on", []))
                if target in dependencies or not set(dependencies).issubset(known_ids):
                    raise ValueError(f"Planner patch has invalid dependencies for {target}")
                patched.component(target).depends_on = dependencies
                graph.edges = expected_plan_edges(patched)
                operations.append(PlanPatchOperation("redirect", graph.component_node(target).node_id, {"depends_on": dependencies, "reason": proposed.payload.get("reason", "")}))
                affected.add(target)
            elif operation == "reopen_component":
                target = proposed.target_id
                self._require_component(target, known_ids)
                patched.component(target).execution_status = "active"
                patched.component(target).scientific_status = "not_tested"
                node = graph.component_node(target)
                node.execution_status = "active"
                node.scientific_status = "not_tested"
                operations.append(PlanPatchOperation("reopen_component", node.node_id, {"reason": proposed.payload.get("reason", ""), "reopened": True}))
                affected.add(target)
            elif operation == "rescout":
                target = proposed.target_id
                self._require_component(target, known_ids)
                node = graph.component_node(target)
                node.metadata.setdefault("planner_rescout", []).append({"revision": next_revision, "scope": proposed.payload.get("patch_scope", {}), "reason": proposed.payload.get("reason", "")})
                operations.append(PlanPatchOperation("redirect", node.node_id, {"rescout": True, "patch_scope": proposed.payload.get("patch_scope", {}), "reason": proposed.payload.get("reason", "")}))
                affected.add(target)
            else:
                raise ValueError(f"Unsupported Planner map operation: {operation}")

        patch = PlanPatch(
            patch_id=f"P{next_revision:04d}", base_revision=base_revision, next_revision=next_revision,
            kind="global_patch", trigger_assessment_ids=[], affected_component_ids=sorted(affected),
            reason=review.rationale, operations=operations, created_at=datetime.now(timezone.utc).isoformat(),
        )
        graph.patches.append(patch)
        patched.plan_revision = next_revision
        patched.scout_fingerprint = scout.world_fingerprint
        patched.current_sensor_limits = list(scout.sensor_limits)
        patched.known_evidence_gaps = sorted(set([*patched.known_evidence_gaps, *[item.reason for item in review.diagnoses]]))
        validate_plan_graph(patched, graph)
        return patched

    @staticmethod
    def _require_component(component_id: str, ids: set[str]) -> None:
        if component_id not in ids:
            raise ValueError(f"Planner patch targets unknown component {component_id}")

    @staticmethod
    def _component_from_payload(payload: dict[str, Any], scout: ScoutReport) -> RequiredComponent:
        raw = payload.get("component")
        if not isinstance(raw, dict):
            raise ValueError("Planner add_component patch lacks a typed component")
        component = RequiredComponent(
            component_id=str(raw["component_id"]), claim=str(raw["claim"]), depends_on=list(raw["depends_on"]),
            evidence_needed=list(raw["evidence_needed"]), constraints=dict(raw["constraints"]),
            failure_meaning=str(raw["failure_meaning"]), scope=dict(scout.task_scope),
            candidate_hypothesis_ids=list(raw.get("candidate_hypothesis_ids", [])), skill_ids=list(raw.get("skill_ids", [])),
        )
        return component

    @staticmethod
    def _add_component_graph_nodes(plan: ScientificPlan, graph: object, component: RequiredComponent, revision: int) -> None:
        graph.nodes.append(PlanNode(
            f"component:{component.component_id}", "decision", component.claim, None,
            component.execution_status, component.scientific_status, component.component_id,
            revision, {"evidence_needed": list(component.evidence_needed), "scope": dict(component.scope)},
        ))
        graph.edges = expected_plan_edges(plan)

    def apply_diagnoses(self, plan: ScientificPlan, scout: ScoutReport, diagnoses: Iterable[MissingDiagnosis]) -> ScientificPlan:
        """Compatibility reducer for older callers and deterministic tests.

        The production runtime no longer invokes this method: a Judge
        diagnosis must first be interpreted by the Planner.  Keeping the
        narrow legacy method preserves persisted-fixture migration without
        reintroducing Judge-owned routing into the normal loop.
        """

        diagnoses = list(diagnoses)
        patched = deepcopy(plan)
        graph = ensure_plan_graph(patched)
        base_revision = patched.plan_revision
        operations: list[PlanPatchOperation] = []
        for diagnosis in diagnoses:
            self._apply_diagnosis(patched, graph, diagnosis, base_revision + 1, operations)
        if not operations:
            return patched
        graph.patches.append(PlanPatch(
            patch_id=f"P{base_revision + 1:04d}", base_revision=base_revision, next_revision=base_revision + 1,
            kind="local_rescout", trigger_assessment_ids=[], affected_component_ids=sorted({item.component_id for item in diagnoses}),
            reason="Compatibility diagnosis patch.", operations=operations, created_at=datetime.now(timezone.utc).isoformat(),
        ))
        patched.plan_revision += 1
        patched.scout_fingerprint = scout.world_fingerprint
        patched.current_sensor_limits = list(scout.sensor_limits)
        patched.known_evidence_gaps = sorted(set([*patched.known_evidence_gaps, *[item.reason for item in diagnoses]]))
        validate_plan_graph(patched, graph)
        return patched

    def formalize_evidence_gap(self, plan: ScientificPlan, scout: ScoutReport, component_id: str, reason: str, diagnoses: Iterable[MissingDiagnosis] = ()) -> ScientificPlan:
        """Reducer for a Planner-owned terminal gap decision.

        The Planner supplies the decision and reason; this method only records
        the append-only graph mutation and never infers why research stopped.
        """

        patched = deepcopy(plan)
        graph = ensure_plan_graph(patched)
        base_revision = patched.plan_revision
        node = graph.component_node(component_id)
        node.metadata.setdefault("evidence_gaps", []).append({"revision": base_revision + 1, "reason": reason})
        patched.component(component_id).execution_status = "exhausted"
        patch = PlanPatch(
            patch_id=f"P{base_revision + 1:04d}", base_revision=base_revision, next_revision=base_revision + 1,
            kind="evidence_gap", trigger_assessment_ids=[], affected_component_ids=[component_id], reason=reason,
            operations=[PlanPatchOperation("add_gap", node.node_id, {"reason": reason})],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        graph.patches.append(patch)
        patched.plan_revision += 1
        patched.scout_fingerprint = scout.world_fingerprint
        patched.current_sensor_limits = list(scout.sensor_limits)
        patched.known_evidence_gaps = sorted(set([*patched.known_evidence_gaps, reason, *[item.reason for item in diagnoses]]))
        validate_plan_graph(patched, graph)
        return patched

    @staticmethod
    def _patch_kind(decision: JudgeDecision) -> str:
        if any(diagnosis.kind == "fundamentally_unidentifiable" for diagnosis in decision.diagnoses):
            return "evidence_gap"
        if decision.action == "local_rescout":
            return "local_rescout"
        if decision.action == "continue_research":
            return "continue_research"
        if any(judgment.status == "contradicted" for judgment in decision.component_judgments):
            return "redirect"
        return "assessment"

    @staticmethod
    def _apply_diagnosis(plan: ScientificPlan, graph: object, diagnosis: MissingDiagnosis, revision: int, operations: list[PlanPatchOperation]) -> None:
        node = graph.component_node(diagnosis.component_id)
        if diagnosis.kind in {"fundamentally_unidentifiable", "missing_tool_capability", "tool_input_unavailable"}:
            node.metadata.setdefault("evidence_gaps", []).append({
                "revision": revision, "reason": diagnosis.reason,
                "missing_observables": diagnosis.missing_observables,
                "suggested_sensor_kinds": diagnosis.suggested_sensor_kinds,
            })
            operations.append(PlanPatchOperation("add_gap", node.node_id, {"reason": diagnosis.reason, "missing_observables": diagnosis.missing_observables}))
            return
        if diagnosis.kind == "calculation_missing":
            operations.append(PlanPatchOperation("redirect", node.node_id, {"missing_observables": list(diagnosis.missing_observables), "reason": diagnosis.reason}))
            return
        if diagnosis.kind in {"wrong_scope_or_data_assumption", "missing_external_sensor", "resolution_limit", "weak_optional_source"}:
            node.metadata.setdefault("local_rescout", []).append({
                "revision": revision, "diagnosis": diagnosis.kind,
                "patch_scope": diagnosis.patch_scope, "reason": diagnosis.reason,
            })
            operations.append(PlanPatchOperation("redirect", node.node_id, {"diagnosis": diagnosis.kind, "patch_scope": diagnosis.patch_scope}))
