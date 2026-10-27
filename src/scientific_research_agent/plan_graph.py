"""Compilation and invariant checks for the Planner's scientific decision DAG."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Iterable

from .contracts import PlanEdge, PlanGraph, PlanNode, RequiredComponent, ScientificPlan


def component_plan_edges(plan: ScientificPlan, component: RequiredComponent) -> list[PlanEdge]:
    """Derive only the scientific dependency edges declared by the Planner."""

    known = {item.component_id for item in plan.required_components}
    target = f"component:{component.component_id}"
    edges: list[PlanEdge] = []
    for dependency in component.depends_on:
        if dependency not in known:
            raise ValueError(f"{component.component_id} depends on unknown component {dependency}")
        edges.append(PlanEdge(f"component:{dependency}", target, "depends_on", rationale="Planner-declared scientific prerequisite."))
    return edges


def expected_plan_edges(plan: ScientificPlan) -> list[PlanEdge]:
    """Return the exact dependency edges declared by the current plan."""

    return [edge for component in plan.required_components for edge in component_plan_edges(plan, component)]


def compile_plan_graph(plan: ScientificPlan) -> PlanGraph:
    """Compile Planner components directly into an unconstrained decision DAG.

    There are no global Search, Aggregate, or Answer stages. Data acquisition,
    computation, and local materialization remain Researcher actions attached
    to whichever scientific decision node needs them.
    """

    nodes = [
        PlanNode(
            f"component:{component.component_id}", "decision", component.claim, None,
            component.execution_status, component.scientific_status, component.component_id,
            plan.plan_revision, {"evidence_needed": list(component.evidence_needed), "scope": dict(component.scope)},
        )
        for component in plan.required_components
    ]
    graph = PlanGraph(nodes, expected_plan_edges(plan))
    validate_plan_graph(plan, graph)
    return graph


def ensure_plan_graph(plan: ScientificPlan) -> PlanGraph:
    if plan.graph is None:
        plan.graph = compile_plan_graph(plan)
    validate_plan_graph(plan, plan.graph)
    return plan.graph


def evidence_input_component_ids(plan: ScientificPlan, component_id: str) -> set[str]:
    """Return the component and all Planner-declared scientific ancestors."""

    graph = ensure_plan_graph(plan)
    target = f"component:{component_id}"
    nodes = {node.node_id: node for node in graph.nodes}
    if target not in nodes:
        raise ValueError(f"Unknown decision node for component {component_id}")
    reverse: dict[str, list[str]] = {}
    for edge in graph.edges:
        reverse.setdefault(edge.target_id, []).append(edge.source_id)
    visited, component_ids, stack = set(), set(), [target]
    while stack:
        node_id = stack.pop()
        if node_id in visited:
            continue
        visited.add(node_id)
        node = nodes[node_id]
        if node.component_id:
            component_ids.add(node.component_id)
        stack.extend(reverse.get(node_id, []))
    return component_ids


def component_execution_epoch(plan: ScientificPlan, component_id: str) -> str:
    """Hash the scientific inputs that require fresh evidence for a node."""

    memo: dict[str, str] = {}

    def build(current_id: str) -> str:
        if current_id in memo:
            return memo[current_id]
        component = plan.component(current_id)
        payload = {
            "component_id": component.component_id,
            "claim": component.claim,
            "depends_on": list(component.depends_on),
            "evidence_needed": list(component.evidence_needed),
            "constraints": component.constraints,
            "scope": component.scope,
            "scout_fingerprint": plan.scout_fingerprint,
            "route_patch_ids": [
                patch.patch_id for patch in ensure_plan_graph(plan).patches
                if current_id in patch.affected_component_ids
                and patch.kind in {"global_patch", "local_rescout", "redirect", "supersede"}
            ],
            "upstream_epochs": {dependency: build(dependency) for dependency in component.depends_on},
        }
        value = sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()
        memo[current_id] = value
        return value

    return build(component_id)


def validate_plan_graph(plan: ScientificPlan, graph: PlanGraph) -> None:
    node_ids = [node.node_id for node in graph.nodes]
    if len(node_ids) != len(set(node_ids)):
        raise ValueError("PlanGraph decision node ids must be unique.")
    component_ids = {component.component_id for component in plan.required_components}
    expected_node_ids = {f"component:{component_id}" for component_id in component_ids}
    if {node.node_id for node in graph.nodes} != expected_node_ids:
        raise ValueError("PlanGraph must contain exactly one decision node for every Planner component.")
    if any(node.kind != "decision" or node.component_id not in component_ids for node in graph.nodes):
        raise ValueError("PlanGraph nodes must be Planner decision nodes bound to known components.")
    for edge in graph.edges:
        if edge.source_id not in node_ids or edge.target_id not in node_ids:
            raise ValueError("PlanGraph edge references an unknown decision node.")
        if edge.relation != "depends_on":
            raise ValueError("PlanGraph edges must represent Planner-declared dependencies.")
    _assert_acyclic(graph.edges)
    actual = {(edge.source_id, edge.target_id, edge.relation) for edge in graph.edges}
    expected = {(edge.source_id, edge.target_id, edge.relation) for edge in expected_plan_edges(plan)}
    if actual != expected:
        raise ValueError("PlanGraph edges do not match the declared scientific dependencies.")


def _assert_acyclic(edges: Iterable[PlanEdge]) -> None:
    graph: dict[str, list[str]] = {}
    for edge in edges:
        graph.setdefault(edge.source_id, []).append(edge.target_id)
    visiting: set[str] = set()
    complete: set[str] = set()

    def visit(node: str) -> None:
        if node in complete:
            return
        if node in visiting:
            raise ValueError("PlanGraph scientific dependencies contain a cycle.")
        visiting.add(node)
        for child in graph.get(node, []):
            visit(child)
        visiting.remove(node)
        complete.add(node)

    for node in graph:
        visit(node)
