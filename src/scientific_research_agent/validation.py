"""Hard boundaries between generative proposals and executable scientific state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from .contracts import LocalExecutionGraph, RequiredComponent, ResearchAction, ScientificPlan
from .node_contracts import validate_component_requirements
from .plan_graph import ensure_plan_graph
from .registry import ToolRegistry


class ContractViolation(ValueError):
    pass


REQUIRED_COMPONENT_FIELDS = {"component_id", "claim", "depends_on", "evidence_needed", "constraints", "failure_meaning", "scope"}


def validate_plan(plan: ScientificPlan) -> ScientificPlan:
    ids = [component.component_id for component in plan.required_components]
    if not ids or len(ids) != len(set(ids)):
        raise ContractViolation("ResearchPlan requires unique, non-empty component ids.")
    valid_ids = set(ids)
    for component in plan.required_components:
        raw = asdict(component)
        # Root claims legitimately have an empty dependency list.
        missing = [field for field in REQUIRED_COMPONENT_FIELDS if raw.get(field) in (None, "") or (field not in {"depends_on"} and raw.get(field) == [])]
        if missing:
            raise ContractViolation(f"{component.component_id} lacks required scientific fields: {missing}")
        unknown = set(component.depends_on) - valid_ids
        if unknown:
            raise ContractViolation(f"{component.component_id} depends on unknown components: {sorted(unknown)}")
        if component.component_id in component.depends_on:
            raise ContractViolation(f"{component.component_id} cannot depend on itself")
        try:
            validate_component_requirements(
                evidence_needed=component.evidence_needed,
                component_id=component.component_id,
            )
        except ValueError as error:
            raise ContractViolation(str(error)) from error
        # A plan may name evidence properties but must never require a fixed tool sequence.
        forbidden = {"tool_id", "tool_ids", "run_tool", "call_tool"}
        if forbidden.intersection(component.constraints):
            raise ContractViolation(f"{component.component_id} prescribes tools; planner must specify evidence, not tool chain")
    _assert_acyclic(plan.required_components)
    try:
        ensure_plan_graph(plan)
    except ValueError as error:
        raise ContractViolation(str(error)) from error
    return plan


def _assert_acyclic(components: list[RequiredComponent]) -> None:
    graph = {component.component_id: component.depends_on for component in components}
    visiting, complete = set(), set()

    def visit(node: str) -> None:
        if node in complete:
            return
        if node in visiting:
            raise ContractViolation(f"ResearchPlan dependency cycle detected at {node}")
        visiting.add(node)
        for parent in graph[node]:
            visit(parent)
        visiting.remove(node)
        complete.add(node)

    for node in graph:
        visit(node)


def validate_actions(actions: list[ResearchAction], plan: ScientificPlan, registry: ToolRegistry) -> list[ResearchAction]:
    components = {component.component_id for component in plan.required_components}
    seen: set[str] = set()
    validated: list[ResearchAction] = []
    for action in actions:
        if not action.action_id or action.action_id in seen:
            raise ContractViolation("Research action ids must be unique and non-empty")
        seen.add(action.action_id)
        if action.component_id not in components:
            raise ContractViolation(f"Action {action.action_id} targets an unknown component")
        if action.kind == "tool":
            if not action.tool_id or action.tool_id not in {tool.tool_id for tool in registry.all()}:
                raise ContractViolation(f"Action {action.action_id} names an unregistered tool")
        elif action.kind == "data_tool":
            if action.tool_id != "data.tool":
                raise ContractViolation(f"Data tool action {action.action_id} must name data.tool")
            if not action.source_id:
                raise ContractViolation(f"Data tool action {action.action_id} needs a source id")
            if not isinstance(action.data_request, dict) or not action.data_request:
                raise ContractViolation(f"Data tool action {action.action_id} needs a structured data request")
        elif action.tool_id is not None:
            raise ContractViolation(f"Non-tool action {action.action_id} cannot carry a tool id")
        if action.kind == "source_query" and not action.source_id:
            raise ContractViolation(f"Source query {action.action_id} needs a source id")
        if action.kind == "finish_component" and (action.tool_id is not None or action.source_id is not None):
            raise ContractViolation(f"Finish action {action.action_id} cannot name a tool or source")
        validated.append(action)
    action_ids = {action.action_id for action in validated}
    for action in validated:
        unknown = set(action.depends_on_action_ids) - action_ids
        if unknown or action.action_id in action.depends_on_action_ids:
            raise ContractViolation(f"Action {action.action_id} has invalid local dependencies: {sorted(unknown)}")
    visiting, complete = set(), set()
    by_id = {action.action_id: action for action in validated}
    def visit(action_id: str) -> None:
        if action_id in complete:
            return
        if action_id in visiting:
            raise ContractViolation("Research actions contain a local dependency cycle")
        visiting.add(action_id)
        for upstream in by_id[action_id].depends_on_action_ids:
            visit(upstream)
        visiting.remove(action_id); complete.add(action_id)
    for action in by_id:
        visit(action)
    return validated


def validate_local_execution_graph(graph: LocalExecutionGraph, plan: ScientificPlan, registry: ToolRegistry) -> LocalExecutionGraph:
    """Validate a persisted local DAG over every accumulated action wave."""

    if graph.component_id not in {component.component_id for component in plan.required_components}:
        raise ContractViolation(f"Local graph {graph.graph_id} targets an unknown component")
    validate_actions(graph.actions, plan, registry)
    action_ids = {action.action_id for action in graph.actions}
    missing_statuses = action_ids - set(graph.action_status)
    if missing_statuses:
        raise ContractViolation(f"Local graph omits states for actions: {sorted(missing_statuses)}")
    unknown_statuses = set(graph.action_status) - action_ids
    if unknown_statuses:
        raise ContractViolation(f"Local graph contains states for unknown actions: {sorted(unknown_statuses)}")
    invalid_statuses = {
        action_id: status for action_id, status in graph.action_status.items()
        if status not in {"pending", "running", "completed", "failed", "skipped", "blocked"}
    }
    if invalid_statuses:
        raise ContractViolation(f"Local graph contains invalid action states: {invalid_statuses}")
    for name, mapping in {
        "evidence": graph.action_evidence_ids,
        "input fingerprints": graph.action_input_fingerprints,
    }.items():
        unknown = set(mapping) - action_ids
        if unknown:
            raise ContractViolation(f"Local graph contains {name} for unknown actions: {sorted(unknown)}")
    for channel_name, failures in {
        "tool execution": graph.tool_execution_failures,
        "model action proposal": graph.model_action_proposal_failures,
        "contract validation": graph.contract_failures,
    }.items():
        for failure in failures:
            if failure.component_id and failure.component_id != graph.component_id:
                raise ContractViolation(f"Local graph contains {channel_name} failure for another component")
            if failure.action_id and failure.action_id not in action_ids:
                raise ContractViolation(f"Local graph contains {channel_name} failure for unknown action {failure.action_id}")
            if failure.failure_kind not in {"tool_execution", "model_action_proposal", "contract_validation"}:
                raise ContractViolation(f"Local graph contains an invalid failure kind: {failure.failure_kind}")
    if any(action.component_id != graph.component_id for action in graph.actions):
        raise ContractViolation("A local execution graph may contain actions for only one scientific component")
    return graph


def affected_components(plan: ScientificPlan, roots: set[str]) -> set[str]:
    """Return a changed component and only its dependency descendants."""
    affected = set(roots)
    changed = True
    while changed:
        changed = False
        for component in plan.required_components:
            if component.component_id not in affected and set(component.depends_on).intersection(affected):
                affected.add(component.component_id)
                changed = True
    return affected
