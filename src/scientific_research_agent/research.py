"""Flexible scientific execution over a component DAG."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from typing import Any, Protocol

from .aiq_grounding import EvidenceSourceRegistry, ScientificResearchNotes, ScientificResearchQuery, note_from_entries
from .contracts import EvidenceEntry, EvidenceStatus, FailureRecord, LocalExecutionGraph, ResearchAction, ScientificPlan, ScoutReport
from .adjudication import EvidenceAdjudicator
from .data_handles import effective_input_scope
from .data_tool import workspace_catalogue
from .evidence_contract import RAW_OBSERVABLES
from .ledger import EvidenceLedger
from .plan_graph import component_execution_epoch, evidence_input_component_ids
from .registry import DataRegistry, ExternalDataRegistry, ToolContext, ToolRegistry, ToolSpec, tool_consumes_observation_rows
from .resilience import RetryOutcome, RetryPolicy, classify_error, execute_with_retry, retry_metadata
from .reasoning import ReasoningUnavailable, StrictSchemaViolation
from .scope import ScopeCompiler
from .validation import ContractViolation, validate_actions, validate_local_execution_graph


class ResearchPolicy(Protocol):
    def choose(self, candidates: list[ToolSpec], existing_tags: set[str], limit: int) -> list[ToolSpec]: ...


class CapabilityResearchPolicy:
    """Replaceable policy. It chooses diverse methods, not a hard-coded scientific workflow."""

    def choose(self, candidates: list[ToolSpec], existing_tags: set[str], limit: int) -> list[ToolSpec]:
        selected: list[ToolSpec] = []
        for tool in candidates:
            if set(tool.output_tags).issubset(existing_tags):
                continue
            selected.append(tool)
            if len(selected) >= limit:
                break
        return selected


@dataclass
class LocalGraphAdvance:
    """One scheduler-visible transition of a component's local execution DAG."""

    graph: LocalExecutionGraph
    produced: list[EvidenceEntry]
    rescout_requested: bool = False
    graph_changed: bool = False


class FlexibleResearcher:
    """Researches existing observations. It never performs paper or table extraction."""

    def __init__(self, tools: ToolRegistry, sources: DataRegistry, external: ExternalDataRegistry, ledger: EvidenceLedger, policy: ResearchPolicy | None = None, *, action_advisor: Any | None = None, scope_compiler: ScopeCompiler | None = None, skills: Any | None = None, trace: Any | None = None, evidence_registry: EvidenceSourceRegistry | None = None, attempt_store: Any | None = None, run_id: str | None = None, execution_policy: RetryPolicy | None = None) -> None:
        self.tools, self.sources, self.external, self.ledger = tools, sources, external, ledger
        self.policy = policy or CapabilityResearchPolicy()
        self.action_advisor, self.skills, self.trace = action_advisor, skills, trace
        self.attempt_store, self.run_id = attempt_store, run_id
        self.execution_policy = execution_policy or RetryPolicy()
        self.scope_compiler = scope_compiler or ScopeCompiler()
        self.evidence_registry = evidence_registry or EvidenceSourceRegistry(ledger.entries)
        self.last_notes: list[ScientificResearchNotes] = []

    def _durable_action_fingerprint(self, action: ResearchAction, component: Any, graph: LocalExecutionGraph, context: ToolContext) -> str:
        declared = graph.action_input_fingerprints.get(action.action_id)
        if declared:
            return declared
        # Every action kind, including source and finish actions, must use the
        # same semantic input identity at proposal, scheduling, recovery and
        # execution boundaries.  The previous split implementation used a
        # second payload for non-tool actions, so a finish node could be
        # recorded under one fingerprint and compared under another after a
        # restart or local repair.
        return self._action_input_fingerprint(action, component, graph, context)

    def _record_failure(
        self,
        graph: LocalExecutionGraph,
        *,
        failure_kind: str,
        phase: str,
        error: BaseException,
        action: ResearchAction | None = None,
        category: str | None = None,
        retryable: bool | None = None,
    ) -> FailureRecord:
        classification = classify_error(error)
        history = (
            graph.tool_execution_failures if failure_kind == "tool_execution"
            else graph.model_action_proposal_failures if failure_kind == "model_action_proposal"
            else graph.contract_failures
        )
        record = FailureRecord(
            failure_id=f"F:{graph.component_id}:{graph.diagnostic_epoch}:{failure_kind}:{len(history) + 1}",
            failure_kind=failure_kind,
            phase=phase,
            error_type=type(error).__name__,
            category=category or classification.category,
            retryable=classification.retryable if retryable is None else retryable,
            message=str(error)[:2_000],
            component_id=graph.component_id,
            action_id=action.action_id if action is not None else "",
            graph_revision=graph.revision,
            diagnostic_epoch=graph.diagnostic_epoch,
            retry_metadata=retry_metadata(error),
        )
        history.append(record)
        if self.trace:
            self.trace(f"{failure_kind}_failed", {"failure": record.__dict__})
        return record

    def _recover_action(self, action: ResearchAction, component: Any, graph: LocalExecutionGraph, action_fingerprint: str) -> list[EvidenceEntry]:
        """Reconcile a pending graph node with durable action/evidence state."""
        if self.attempt_store is None or self.run_id is None:
            return []
        attempt = self.attempt_store.action_attempt(self.run_id, action.action_id, action_fingerprint)
        evidence = self.ledger.for_action(action.action_id, action_fingerprint)
        if not evidence and attempt and attempt.get("status") == "completed":
            by_id = {entry.evidence_id: entry for entry in self.ledger.entries}
            evidence = [by_id[item] for item in attempt.get("evidence_ids", []) if item in by_id]
        evidence = [entry for entry in evidence if self._evidence_can_complete_action(entry)]
        if not evidence:
            return []
        evidence_ids = [entry.evidence_id for entry in evidence]
        graph.action_evidence_ids[action.action_id] = evidence_ids
        graph.action_status[action.action_id] = "completed"
        if attempt and attempt.get("status") != "completed":
            self.attempt_store.complete_action(self.run_id, action.action_id, action_fingerprint, evidence_ids)
        if self.trace:
            self.trace("local_action_recovered", {"component_id": component.component_id, "action_id": action.action_id, "evidence_ids": evidence_ids, "reason": "durable evidence already exists"})
        return evidence

    @staticmethod
    def _evidence_can_complete_action(entry: EvidenceEntry) -> bool:
        """Only recover an action from a durable, usable operation result.

        A data-tool result can be durable diagnostic evidence while still
        being an invalid request, unavailable provider, or decision request.
        Such a record explains a failed action; it must never turn that action
        into a completed node after a restart.
        """

        if entry.status != "ok":
            return False
        operation = entry.findings.get("data_operation") if isinstance(entry.findings, dict) else None
        return not isinstance(operation, dict) or operation.get("status") == "completed"

    def _begin_action(self, action: ResearchAction, component: Any, action_fingerprint: str) -> None:
        if self.attempt_store is not None and self.run_id is not None:
            self.attempt_store.begin_action(self.run_id, action.action_id, action_fingerprint, component_id=component.component_id, kind=action.kind, tool_id=action.tool_id or "")

    def _complete_action(self, action: ResearchAction, action_fingerprint: str, evidence_ids: list[str]) -> None:
        if self.attempt_store is not None and self.run_id is not None:
            self.attempt_store.complete_action(self.run_id, action.action_id, action_fingerprint, evidence_ids)

    def _fail_action(self, action: ResearchAction, action_fingerprint: str, error: str) -> None:
        if self.attempt_store is not None and self.run_id is not None:
            self.attempt_store.fail_action(self.run_id, action.action_id, action_fingerprint, error)

    def _record_retry_outcome(self, action: ResearchAction, outcome: RetryOutcome) -> None:
        for event in outcome.retry_events:
            if self.trace:
                self.trace("action_retry_scheduled", {
                    "component_id": action.component_id,
                    "action_id": action.action_id,
                    **event,
                })

    def _execute_action(self, action: ResearchAction, operation: Any, *, emit_trace: bool = True) -> RetryOutcome:
        """Execute one local action under its durable retry policy."""

        outcome = execute_with_retry(
            operation,
            self.execution_policy.for_action(action),
            operation_name=action.action_id,
        )
        if emit_trace:
            self._record_retry_outcome(action, outcome)
        return outcome

    def _record_retry_failure(self, action: ResearchAction, error: BaseException) -> None:
        if not self.trace:
            return
        metadata = retry_metadata(error)
        if metadata:
            self.trace("action_retry_exhausted", {
                "component_id": action.component_id,
                "action_id": action.action_id,
                **metadata,
            })

    def ensure_execution_graph(self, plan: ScientificPlan, component_id: str, graph: LocalExecutionGraph | None = None) -> LocalExecutionGraph:
        """Create the durable local graph owned by the Researcher.

        A model-backed run starts empty so the Researcher chooses relevant
        sources and may expand dependent exploration/query actions. Offline
        fixtures seed registered sources only because they have no model.
        """

        if graph is not None:
            return validate_local_execution_graph(graph, plan, self.tools)
        component = plan.component(component_id)
        # A model-backed Researcher chooses which source to explore and may
        # append several dependent source queries.  Deterministic fixtures keep
        # the old all-source roots so they remain useful without a model.
        roots = [
            ResearchAction(
                action_id=f"source:{component_id}:{source.source_id}", component_id=component_id,
                kind="source_query", source_id=source.source_id,
                rationale="Acquire the registered observation source within the frozen component scope.",
                local_node_label=f"Acquire {source.source_id}",
            )
            for source in self.sources.all()
        ]
        if self.action_advisor is not None:
            roots = []
        graph = LocalExecutionGraph(graph_id=f"local:{component_id}", component_id=component_id)
        graph.add_actions(roots)
        return validate_local_execution_graph(graph, plan, self.tools)

    def advance_execution_graph(
        self,
        plan: ScientificPlan,
        scout: ScoutReport,
        graph: LocalExecutionGraph,
        *,
        max_actions: int = 4,
    ) -> LocalGraphAdvance:
        """Execute one ready local-DAG wave, or ask the Researcher to expand it.

        This method never changes the global ScientificPlan.  It only changes
        local action nodes and writes durable evidence.  A graph marked
        ``exhausted`` is a factual hand-off: the top-level Planner decides
        whether the scientific plan itself needs revision.
        """

        graph = validate_local_execution_graph(graph, plan, self.tools)
        component = plan.component(graph.component_id)
        if graph.scientific_status in {"supported", "contradicted"} or graph.execution_status in {"completed", "exhausted", "escalated"}:
            return LocalGraphAdvance(graph, [])
        # Reconcile durable results before scheduling.  This is what makes a
        # process restart a continuation rather than a blind replay.
        recovered: list[EvidenceEntry] = []
        for action in graph.actions:
            if graph.action_status.get(action.action_id) in {"completed", "failed"}:
                continue
            context = ToolContext(plan.task.question, plan.task, component, self._scope_for_local_graph(plan, graph, component), self.sources, self.external, list(self.ledger.entries), component.candidate_hypothesis_ids, plan, scout, action, self.action_advisor is not None)
            action_fingerprint = self._durable_action_fingerprint(action, component, graph, context)
            recovered.extend(self._recover_action(action, component, graph, action_fingerprint))
        ready = graph.ready_actions()
        if ready:
            advance = self._execute_ready_local_actions(plan, scout, component, graph, ready[:max_actions])
            if recovered:
                advance.produced = [*recovered, *advance.produced]
            return advance
        if recovered:
            graph.revision += 1
            return LocalGraphAdvance(graph, recovered, graph_changed=True)
        if any(status == "running" for status in graph.action_status.values()):
            return LocalGraphAdvance(graph, [])
        if any(status == "pending" for status in graph.action_status.values()):
            # Pending nodes with no schedulable predecessor are a graph state,
            # not a controller diagnosis.  Escalate the intact dossier.
            graph.execution_status = "exhausted"
            graph.escalation_note = "No pending local action has all declared inputs available."
            return LocalGraphAdvance(graph, [])
        return self._expand_execution_graph(plan, scout, component, graph, max_actions=max_actions)

    def _execute_ready_local_actions(
        self, plan: ScientificPlan, scout: ScoutReport, component: Any,
        graph: LocalExecutionGraph, actions: list[ResearchAction],
    ) -> LocalGraphAdvance:
        produced: list[EvidenceEntry] = []
        rescout_requested = False
        execution_epoch = component_execution_epoch(plan, component.component_id)
        action_fingerprints: dict[str, str] = {}
        pending_actions: list[ResearchAction] = []
        for action in actions:
            context = ToolContext(plan.task.question, plan.task, component, self._scope_for_local_graph(plan, graph, component), self.sources, self.external, list(self.ledger.entries), component.candidate_hypothesis_ids, plan, scout, action, self.action_advisor is not None)
            action_fingerprint = self._durable_action_fingerprint(action, component, graph, context)
            graph.action_input_fingerprints.setdefault(action.action_id, action_fingerprint)
            action_fingerprints[action.action_id] = action_fingerprint
            recovered = self._recover_action(action, component, graph, action_fingerprint)
            if recovered:
                produced.extend(recovered)
                continue
            self._begin_action(action, component, action_fingerprint)
            graph.action_status[action.action_id] = "running"
            pending_actions.append(action)
        actions = pending_actions
        if not actions:
            graph.revision += 1
            return LocalGraphAdvance(graph, produced, graph_changed=True)
        # Source acquisition is intentionally serial and directly auditable;
        # pure tool calls within the same dependency-free wave may run in
        # parallel but are committed to the Ledger in action order.
        for action in [item for item in actions if item.kind == "source_query"]:
            try:
                source = self.sources.get(action.source_id or "")
                descriptor = source.describe_source()
                resolution = self.scope_compiler.compile_one(plan.task, source, action.input_scope, component=component)
                scope = resolution.canonical_scope
                entry = self.ledger.source_query(
                    descriptor.source_id, component.component_id, scope, descriptor.version,
                    execution_epoch=execution_epoch,
                )
                if entry is None:
                    rows = self._execute_action(action, lambda: source.query(scope)).value
                    entry = self.ledger.record_source_query(
                        descriptor.source_id, [component.component_id], scope, rows, descriptor.version,
                        action_id=action.action_id,
                        action_input_fingerprint=action_fingerprints[action.action_id],
                        execution_epoch=execution_epoch,
                    )
                self.evidence_registry.capture(entry)
                graph.action_evidence_ids[action.action_id] = [entry.evidence_id]
                graph.action_status[action.action_id] = "completed"
                self._complete_action(action, action_fingerprints[action.action_id], [entry.evidence_id])
                produced.append(entry)
                if self.trace:
                    self.trace("local_action_completed", {"component_id": component.component_id, "action_id": action.action_id, "kind": action.kind, "evidence_ids": [entry.evidence_id], "source_scope_status": resolution.status, "source_scope": scope})
            except Exception as error:
                graph.action_status[action.action_id] = "failed"
                self._record_retry_failure(action, error)
                failure = self._record_failure(graph, failure_kind="tool_execution", phase="source_query", error=error, action=action)
                self._fail_action(action, action_fingerprints[action.action_id], failure.message)
        # A relational workspace operation is a registered tool, but it stays
        # serialized with source access: database-side aggregation and handle
        # creation must be durable before a downstream numerical wave starts.
        for action in [item for item in actions if item.kind == "data_tool"]:
            try:
                tool = self.tools.get("data.tool")
                local_scope = self._scope_for_local_graph(plan, graph, component)
                source = self.sources.get(action.source_id or "")
                action_scope = dict(action.input_scope or {})
                request_table_id = str((action.data_request or {}).get("table_id") or "")
                if request_table_id and not (action.data_request or {}).get("tables") and not (action.data_request or {}).get("relations"):
                    # The physical table named by the operation is authoritative.
                    # A component-level multi-table scope must never silently
                    # substitute the last table explored by this source.
                    action_scope["table_id"] = request_table_id
                resolution = self.scope_compiler.compile_one(plan.task, source, action_scope, component=component)
                if request_table_id and resolution.canonical_scope.get("table_id") != request_table_id:
                    raise ContractViolation(
                        f"data.tool scope table mismatch: request={request_table_id!r}, "
                        f"resolved={resolution.canonical_scope.get('table_id')!r}"
                    )
                # A workspace node is itself the first local inspection/query
                # route.  It needs an executable source-bound scope but does
                # not need a preceding raw-row source_query node.
                source_scopes = dict(local_scope.get("source_scopes", {}))
                source_scopes[source.source_id] = resolution.canonical_scope
                table_scopes = dict(local_scope.get("table_scopes", {}))
                resolved_table_id = str(resolution.canonical_scope.get("table_id") or "")
                if resolved_table_id:
                    table_scopes[f"{source.source_id}::{resolved_table_id}"] = resolution.canonical_scope
                local_scope = {**local_scope, "_scope_contract": "scope_bundle/v2", "status": resolution.status, "source_scopes": source_scopes, "table_scopes": table_scopes}
                context = ToolContext(
                    plan.task.question, plan.task, component, local_scope, self.sources,
                    self.external, list(self.ledger.entries), component.candidate_hypothesis_ids,
                    plan, scout, action, self.action_advisor is not None,
                )
                trace_id = self.trace("tool_started", {"component_id": component.component_id, "action_id": action.action_id, "tool_id": tool.tool_id, "source_id": action.source_id, "data_request": action.data_request, "source_scope": resolution.canonical_scope}) if self.trace else None
                raw_result = self._execute_action(action, lambda: tool.runner(context)).value
                try:
                    result = self.tools.validate_result(tool, raw_result)
                except Exception as error:
                    graph.action_status[action.action_id] = "failed"
                    failure = self._record_failure(graph, failure_kind="contract_validation", phase="tool_result_validation", error=error, action=action, category="schema_contract", retryable=False)
                    self._fail_action(action, action_fingerprints[action.action_id], failure.message)
                    continue
                evidence_scope = resolution.canonical_scope if request_table_id else local_scope
                entry = self.ledger.record_tool_call(
                    result, [component.component_id], evidence_scope,
                    parameters={"local_action_id": action.action_id, "action_input_fingerprint": action_fingerprints[action.action_id], "data_request": action.data_request},
                    trace_id=trace_id,
                    upstream_evidence_ids=self._upstream_evidence_for_action(graph, action),
                    execution_epoch=execution_epoch,
                )
                self.evidence_registry.capture(entry)
                graph.action_evidence_ids[action.action_id] = [entry.evidence_id]
                if self._evidence_can_complete_action(entry):
                    graph.action_status[action.action_id] = "completed"
                    self._complete_action(action, action_fingerprints[action.action_id], [entry.evidence_id])
                else:
                    operation = entry.findings.get("data_operation", {}) if isinstance(entry.findings, dict) else {}
                    operation_status = operation.get("status", "invalid_result") if isinstance(operation, dict) else "invalid_result"
                    graph.action_status[action.action_id] = "failed"
                    failure = self._record_failure(
                        graph, failure_kind="tool_execution", phase="data_tool_result",
                        error=RuntimeError(f"data tool returned {operation_status}"), action=action,
                        category="tool_result_failure", retryable=False,
                    )
                    self._fail_action(action, action_fingerprints[action.action_id], failure.message)
                produced.append(entry)
                if self.trace:
                    event = "local_action_completed" if graph.action_status[action.action_id] == "completed" else "local_action_failed"
                    self.trace(event, {"component_id": component.component_id, "action_id": action.action_id, "kind": action.kind, "evidence_ids": [entry.evidence_id]})
            except Exception as error:
                graph.action_status[action.action_id] = "failed"
                self._record_retry_failure(action, error)
                failure = self._record_failure(graph, failure_kind="tool_execution", phase="data_tool_runner", error=error, action=action)
                self._fail_action(action, action_fingerprints[action.action_id], failure.message)
        tool_actions = [item for item in actions if item.kind == "tool" and item.tool_id]
        jobs: list[tuple[ResearchAction, ToolSpec, ToolContext, Any, str | None]] = []
        if tool_actions:
            with ThreadPoolExecutor(max_workers=len(tool_actions)) as executor:
                for action in tool_actions:
                    tool = self.tools.get(action.tool_id or "")
                    local_scope = self._scope_for_local_graph(plan, graph, component)
                    context = ToolContext(plan.task.question, plan.task, component, local_scope, self.sources, self.external, list(self.ledger.entries), component.candidate_hypothesis_ids, plan, scout, action, self.action_advisor is not None)
                    trace_id = self.trace("tool_started", {"component_id": component.component_id, "action_id": action.action_id, "tool_id": tool.tool_id, "scope": local_scope}) if self.trace else None
                    jobs.append((action, tool, context, executor.submit(self._execute_action, action, lambda tool=tool, context=context: tool.runner(context), emit_trace=False), trace_id))
                for action, tool, context, future, trace_id in jobs:
                    try:
                        outcome = future.result()
                        self._record_retry_outcome(action, outcome)
                        try:
                            result = self.tools.validate_result(tool, outcome.value)
                        except Exception as error:
                            graph.action_status[action.action_id] = "failed"
                            failure = self._record_failure(graph, failure_kind="contract_validation", phase="tool_result_validation", error=error, action=action, category="schema_contract", retryable=False)
                            self._fail_action(action, action_fingerprints[action.action_id], failure.message)
                            continue
                        entry = self.ledger.record_tool_call(
                            result, [component.component_id], effective_input_scope(context),
                            parameters={"local_action_id": action.action_id, "action_input_fingerprint": action_fingerprints[action.action_id], "action_parameters": action.parameters},
                            trace_id=trace_id,
                            upstream_evidence_ids=self._upstream_evidence_for_action(graph, action),
                            execution_epoch=execution_epoch,
                        )
                        self.evidence_registry.capture(entry)
                        graph.action_evidence_ids[action.action_id] = [entry.evidence_id]
                        graph.action_status[action.action_id] = "completed"
                        self._complete_action(action, action_fingerprints[action.action_id], [entry.evidence_id])
                        produced.append(entry)
                        if self.trace:
                            self.trace("local_action_completed", {"component_id": component.component_id, "action_id": action.action_id, "kind": action.kind, "evidence_ids": [entry.evidence_id]})
                    except Exception as error:
                        graph.action_status[action.action_id] = "failed"
                        self._record_retry_failure(action, error)
                        failure = self._record_failure(graph, failure_kind="tool_execution", phase="tool_runner", error=error, action=action)
                        self._fail_action(action, action_fingerprints[action.action_id], failure.message)
        for action in [item for item in actions if item.kind == "local_rescout"]:
            graph.action_status[action.action_id] = "completed"
            self._complete_action(action, action_fingerprints[action.action_id], [])
            graph.execution_status = "escalated"
            graph.escalation_note = action.rationale or "Researcher requested a refreshed observation-world readout."
            rescout_requested = True
        for action in [item for item in actions if item.kind == "finish_component"]:
            graph.action_status[action.action_id] = "completed"
            self._complete_action(action, action_fingerprints[action.action_id], [])
            graph.execution_status = "exhausted"
            graph.escalation_note = action.rationale or "Researcher has no further legal local action."
        if produced:
            self._append_local_note(plan, component, produced)
        graph.revision += 1
        return LocalGraphAdvance(graph, produced, rescout_requested=rescout_requested, graph_changed=True)

    def execute_ready_tool_wave(self, plan: ScientificPlan, scout: ScoutReport, graphs: list[LocalExecutionGraph]) -> dict[str, LocalGraphAdvance]:
        """Run independent ready *tool* nodes across local graphs concurrently.

        Source reads, graph edits, and Ledger commits stay serialized.  Only
        pure registered computations whose local dependencies are already
        complete share a worker pool.  This gives true parallelism without
        allowing concurrent workers to mutate the durable research state.
        """

        jobs: list[tuple[LocalExecutionGraph, ResearchAction, Any, ToolSpec, str | None, list[str], list[Any], str]] = []
        advances = {graph.component_id: LocalGraphAdvance(graph, [], graph_changed=True) for graph in graphs}
        for graph in graphs:
            component = plan.component(graph.component_id)
            ready = graph.ready_actions()
            if not ready or any(action.kind != "tool" or not action.tool_id for action in ready):
                raise ContractViolation("A parallel local-tool wave may contain only ready registered tool actions")
            for action in ready:
                tool = self.tools.get(action.tool_id or "")
                local_scope = self._scope_for_local_graph(plan, graph, component)
                context = ToolContext(
                    plan.task.question, plan.task, component, local_scope, self.sources,
                    self.external, list(self.ledger.entries), component.candidate_hypothesis_ids,
                    plan, scout, action, self.action_advisor is not None,
                )
                action_fingerprint = self._durable_action_fingerprint(action, component, graph, context)
                graph.action_input_fingerprints.setdefault(action.action_id, action_fingerprint)
                recovered = self._recover_action(action, component, graph, action_fingerprint)
                if recovered:
                    advances[graph.component_id].produced.extend(recovered)
                    continue
                self._begin_action(action, component, action_fingerprint)
                graph.action_status[action.action_id] = "running"
                trace_id = self.trace("tool_started", {"component_id": component.component_id, "action_id": action.action_id, "tool_id": tool.tool_id, "scope": local_scope}) if self.trace else None
                jobs.append((graph, action, context, tool, trace_id, action_fingerprint))
        computed = []
        if jobs:
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                futures = [(item, executor.submit(self._execute_action, item[1], lambda item=item: item[3].runner(item[2]), emit_trace=False)) for item in jobs]
                for item, future in futures:
                    try:
                        outcome = future.result()
                        self._record_retry_outcome(item[1], outcome)
                        computed.append((item, outcome.value, None))
                    except Exception as error:
                        self._record_retry_failure(item[1], error)
                        computed.append((item, None, error))
        for (graph, action, _context, tool, trace_id, _action_fingerprint), raw_result, compute_error in computed:
            component = plan.component(graph.component_id)
            try:
                if compute_error is not None:
                    raise compute_error
                try:
                    result = self.tools.validate_result(tool, raw_result)
                except Exception as error:
                    graph.action_status[action.action_id] = "failed"
                    failure = self._record_failure(graph, failure_kind="contract_validation", phase="tool_result_validation", error=error, action=action, category="schema_contract", retryable=False)
                    self._fail_action(action, _action_fingerprint, failure.message)
                    continue
                entry = self.ledger.record_tool_call(
                    result, [component.component_id], effective_input_scope(_context),
                    parameters={"local_action_id": action.action_id, "action_input_fingerprint": _action_fingerprint, "action_parameters": action.parameters},
                    trace_id=trace_id,
                    upstream_evidence_ids=self._upstream_evidence_for_action(graph, action),
                    execution_epoch=component_execution_epoch(plan, component.component_id),
                )
                self.evidence_registry.capture(entry)
                graph.action_evidence_ids[action.action_id] = [entry.evidence_id]
                graph.action_status[action.action_id] = "completed"
                self._complete_action(action, _action_fingerprint, [entry.evidence_id])
                advances[graph.component_id].produced.append(entry)
                if self.trace:
                    self.trace("local_action_completed", {"component_id": component.component_id, "action_id": action.action_id, "kind": action.kind, "evidence_ids": [entry.evidence_id]})
            except Exception as error:
                graph.action_status[action.action_id] = "failed"
                failure = self._record_failure(graph, failure_kind="tool_execution", phase="tool_runner", error=error, action=action)
                self._fail_action(action, _action_fingerprint, failure.message)
        for graph in graphs:
            entries = advances[graph.component_id].produced
            if entries:
                component = plan.component(graph.component_id)
                self._append_local_note(plan, component, entries)
            graph.revision += 1
        return advances

    def _expand_execution_graph(
        self, plan: ScientificPlan, scout: ScoutReport, component: Any,
        graph: LocalExecutionGraph, *, max_actions: int,
    ) -> LocalGraphAdvance:
        existing = self._entries_for_component(plan, component.component_id)
        available = self.sources.capabilities()
        available.update(capability for dataset in scout.external_datasets if dataset.get("available") for capability in dataset.get("capabilities", []))
        needs = list(dict.fromkeys(need for need in component.evidence_needed if need not in RAW_OBSERVABLES))
        context = ToolContext(plan.task.question, plan.task, component, self._scope_for_local_graph(plan, graph, component), self.sources, self.external, list(self.ledger.entries), component.candidate_hypothesis_ids, plan, scout)
        graph.capability_coverage = self.tools.capability_coverage(needs, context)
        candidates = self.tools.select(needs, available, context, exposed_only=self.action_advisor is not None)
        proposals: list[ResearchAction] = []
        if self.action_advisor is not None:
            try:
                proposals = self.action_advisor.propose(
                    task=plan.task, local_graph=graph.to_local_view(self._scientific_neighbourhood(plan, component)),
                    component=component, scout=scout, skills=self.skills.all() if self.skills else [],
                    tool_catalogue=self.tools.catalogue(context), matching_tool_ids={tool.tool_id for tool in candidates},
                    workspace_catalogue=workspace_catalogue(self.sources),
                    evidence=list(self.ledger.entries), component_evidence=existing, limit=max_actions,
                )
            except StrictSchemaViolation as error:
                self._record_failure(
                    graph, failure_kind="contract_validation", phase="researcher_action_schema",
                    error=error, category="schema_contract", retryable=False,
                )
                proposals = []
            except Exception as error:
                self._record_retry_failure(
                    ResearchAction(f"proposal:{component.component_id}:{graph.revision}", component.component_id, "finish_component"),
                    error,
                )
                self._record_failure(
                    graph, failure_kind="model_action_proposal", phase="researcher_action_proposal",
                    error=error,
                )
                proposals = []
            try:
                proposals, rejection_changed = self._legal_local_actions(component, graph, proposals, context)
                legal_proposals = list(proposals)
                proposals = self._confirm_actions(
                    plan, component, proposals,
                    local_graph=graph.to_local_view(self._scientific_neighbourhood(plan, component)),
                    existing_actions=graph.actions, context=context,
                )
                confirmed_ids = {action.action_id for action in proposals}
                for rejected_action in legal_proposals:
                    if rejected_action.action_id in confirmed_ids:
                        continue
                    fingerprint = self._action_input_fingerprint(
                        rejected_action, component, graph,
                        replace(context, action=rejected_action, require_materialized_handle=True),
                    )
                    if fingerprint not in graph.rejected_action_fingerprints:
                        graph.rejected_action_fingerprints.append(fingerprint)
                        rejection_changed = True
                if not proposals and rejection_changed:
                    graph.revision += 1
                    return LocalGraphAdvance(graph, [], graph_changed=True)
            except (ContractViolation, StrictSchemaViolation, TypeError, ValueError, KeyError) as error:
                self._record_failure(
                    graph, failure_kind="contract_validation", phase="researcher_action_validation",
                    error=error, category="schema_contract", retryable=False,
                )
                proposals = []
            except ReasoningUnavailable as error:
                self._record_failure(
                    graph, failure_kind="model_action_proposal", phase="researcher_action_confirmation",
                    error=error,
                )
                proposals = []
            if not proposals and not graph.actions and graph.model_action_proposal_failures:
                proposals = [
                    ResearchAction(
                        action_id=f"source:{component.component_id}:{source.source_id}",
                        component_id=component.component_id,
                        kind="source_query",
                        source_id=source.source_id,
                        rationale="The Researcher model was unavailable; acquire registered read-only sources without inferring a method or conclusion.",
                        local_node_label=f"Explore {source.source_id}",
                    )
                    for source in self.sources.all()
                ]
        if not proposals and self.action_advisor is None:
            # Non-model fixtures retain a deterministic method policy.  A real
            # agent never fabricates a substitute calculation after an invalid
            # model proposal.
            existing_tags = {tag for entry in existing for tag in entry.evidence_tags}
            executed_tool_ids = {action.tool_id for action in graph.actions if action.kind == "tool" and action.tool_id}
            remaining_candidates = [tool for tool in candidates if tool.tool_id not in executed_tool_ids]
            proposals = [
                ResearchAction(
                    action_id=f"tool:{component.component_id}:{tool.tool_id}:{graph.revision}", component_id=component.component_id,
                    kind="tool", tool_id=tool.tool_id, rationale="Registered fallback policy selected a capability matching the current evidence need.",
                    expected_evidence_tags=list(tool.output_tags),
                    depends_on_action_ids=[action.action_id for action in graph.actions if action.kind == "source_query"],
                    local_node_label=tool.name,
                )
                for tool in self._policy_selection(plan, component, remaining_candidates, existing_tags, max_actions)
            ]
        if not proposals:
            graph.execution_status = "exhausted"
            absent = [need for need, item in graph.capability_coverage.items() if item.get("status") == "capability_absent"]
            blocked = [need for need, item in graph.capability_coverage.items() if item.get("status") == "input_not_ready"]
            graph.escalation_note = (
                f"No legal local execution path remains. Capabilities absent: {absent}; registered methods with unavailable inputs: {blocked}."
            )
            return LocalGraphAdvance(graph, [])
        existing_ids = {action.action_id for action in graph.actions}
        if any(action.action_id in existing_ids for action in proposals):
            raise ContractViolation("Researcher may append local actions but cannot overwrite a completed local node")
        candidate_graph = LocalExecutionGraph(
            graph_id=graph.graph_id, component_id=graph.component_id,
            actions=[*graph.actions, *proposals],
            # Validate the same state transition that ``add_actions`` will
            # commit below.  A candidate local DAG must never contain an
            # action without an explicit lifecycle state, even transiently.
            action_status={
                **dict(graph.action_status),
                **{action.action_id: "pending" for action in proposals},
            },
            action_evidence_ids=dict(graph.action_evidence_ids),
            action_input_fingerprints=dict(graph.action_input_fingerprints), rejected_action_fingerprints=list(graph.rejected_action_fingerprints),
            execution_status=graph.execution_status, scientific_status=graph.scientific_status, missing_evidence=list(graph.missing_evidence),
            capability_coverage=dict(graph.capability_coverage), repair_history=list(graph.repair_history),
            tool_execution_failures=list(graph.tool_execution_failures),
            model_action_proposal_failures=list(graph.model_action_proposal_failures),
            contract_failures=list(graph.contract_failures),
            escalation_note=graph.escalation_note, planner_directive=graph.planner_directive, revision=graph.revision,
            diagnostic_epoch=graph.diagnostic_epoch,
        )
        validate_local_execution_graph(candidate_graph, plan, self.tools)
        for action in proposals:
            graph.action_input_fingerprints[action.action_id] = self._action_input_fingerprint(
                action, component, graph,
                replace(context, action=action, require_materialized_handle=self.action_advisor is not None),
            )
        graph.add_actions(proposals)
        if self.trace:
            self.trace("local_graph_expanded", {"component_id": component.component_id, "graph_id": graph.graph_id, "actions": [{"action_id": action.action_id, "kind": action.kind, "tool_id": action.tool_id, "source_id": action.source_id, "depends_on": action.depends_on_action_ids} for action in proposals]})
        return LocalGraphAdvance(graph, [], graph_changed=True)

    def _legal_local_actions(self, component: Any, graph: LocalExecutionGraph, actions: list[ResearchAction], context: ToolContext) -> tuple[list[ResearchAction], bool]:
        """Enforce method readiness and no-op retry invariants for the local graph."""

        legal: list[ResearchAction] = []
        changed = False
        used = set(graph.action_input_fingerprints.values())
        rejected = set(graph.rejected_action_fingerprints)
        local_action_ids = {item.action_id for item in graph.actions}
        # The model may return a small local DAG in one proposal.  Its
        # producer and consumer are both new nodes, so the consumer's edge is
        # local even though the producer is not in ``graph.actions`` yet.
        # Only dependencies outside the union of the existing graph and this
        # proposal batch are foreign local-graph edges.
        proposal_action_ids = {item.action_id for item in actions}
        candidate_local_action_ids = local_action_ids | proposal_action_ids
        for proposed in actions:
            action = self._bind_unique_materialized_handle(proposed, context, local_action_ids=candidate_local_action_ids)
            foreign_dependencies = [
                dependency for dependency in action.depends_on_action_ids
                if dependency not in candidate_local_action_ids
            ]
            if foreign_dependencies:
                # A global component edge must cross the boundary as durable
                # evidence/artifact inputs, never as action ids from another
                # LocalExecutionGraph.  Resolve all foreign producers before
                # rejecting the action; this applies to finish nodes as well
                # as numerical tools.
                bridged = self._bridge_foreign_dependencies(action, context, foreign_dependencies)
                if bridged is None:
                    reason = (
                        "cross-local action dependencies require a registered "
                        f"data handle or evidence input: {sorted(foreign_dependencies)}"
                    )
                    if self.trace:
                        self.trace("local_action_rejected", {
                            "component_id": component.component_id,
                            "action_id": action.action_id,
                            "tool_id": action.tool_id,
                            "reason": reason,
                        })
                    changed = True
                    continue
                action = bridged
            action_context = replace(context, action=action, require_materialized_handle=self.action_advisor is not None)
            fingerprint = self._action_input_fingerprint(action, component, graph, action_context)
            reason = ""
            if action.kind in {"tool", "data_tool"} and action.tool_id:
                readiness = self.tools.assess_readiness(self.tools.get(action.tool_id), action_context)
                if readiness.status != "ready":
                    reason = f"tool inputs are not ready: {readiness.missing_requirement_ids}"
            # A finish action is a state transition, not a computation.  Its
            # input fingerprint intentionally ignores the cosmetic action id,
            # so a Researcher that proposes the same finish node again is
            # asking to close the same local graph, not asking to repeat work.
            # Reusing the already completed finish action is therefore the
            # idempotent success path.  Treating it as an ordinary duplicate
            # makes an exhausted graph reopen forever.
            if not reason and action.kind == "finish_component" and fingerprint in used:
                existing = self._completed_action_for_fingerprint(graph, fingerprint)
                if existing is not None:
                    graph.execution_status = "exhausted"
                    graph.escalation_note = "Equivalent finish action already completed; reused its terminal state."
                    changed = True
                    if self.trace:
                        self.trace("local_action_reused", {
                            "component_id": component.component_id,
                            "requested_action_id": action.action_id,
                            "reused_action_id": existing.action_id,
                            "kind": action.kind,
                            "input_fingerprint": fingerprint,
                            "reason": "equivalent finish action already completed",
                        })
                    continue
                # The same semantic finish request is still pending or was
                # recorded without completion.  It must follow the ordinary
                # duplicate path and remain rejected until its owner reaches
                # a durable terminal state; silently dropping it would hide a
                # real local-graph defect.
                reason = "equivalent finish action is already pending or incomplete"
            if not reason and fingerprint in used:
                reason = "equivalent action already ran with the same declared inputs"
            if not reason and fingerprint in rejected:
                reason = "equivalent action was already rejected under unchanged inputs"
            if reason:
                if fingerprint not in rejected:
                    graph.rejected_action_fingerprints.append(fingerprint)
                    changed = True
                if self.trace:
                    self.trace("local_action_rejected", {"component_id": component.component_id, "action_id": action.action_id, "tool_id": action.tool_id, "reason": reason, "input_fingerprint": fingerprint})
                continue
            legal.append(action)
        return legal, changed

    @staticmethod
    def _completed_action_for_fingerprint(graph: LocalExecutionGraph, fingerprint: str) -> ResearchAction | None:
        """Find a completed local action that already owns ``fingerprint``.

        Fingerprints are semantic input identities and deliberately exclude
        generated action ids.  Only a completed action may satisfy a repeated
        finish request; pending/running actions must remain schedulable and a
        rejected action must never be promoted to success.
        """

        for action in graph.actions:
            if (
                graph.action_status.get(action.action_id) == "completed"
                and graph.action_input_fingerprints.get(action.action_id) == fingerprint
            ):
                return action
        return None

    @staticmethod
    def _upstream_evidence_for_action(graph: LocalExecutionGraph, action: ResearchAction) -> list[str]:
        """Return evidence on the action's declared input edges only.

        A tool result must show which local predecessor supplied its data. It
        must not inherit every completed evidence item in the component.
        """

        evidence_ids: list[str] = []
        for upstream_action_id in action.depends_on_action_ids:
            evidence_ids.extend(graph.action_evidence_ids.get(upstream_action_id, []))
        if action.data_handle_evidence_id:
            evidence_ids.append(action.data_handle_evidence_id)
        evidence_ids.extend(action.input_evidence_ids)
        return list(dict.fromkeys(item for item in evidence_ids if item))

    @staticmethod
    def _bridge_foreign_dependencies(action: ResearchAction, context: ToolContext, foreign_dependencies: list[str]) -> ResearchAction | None:
        """Replace foreign local-action edges with durable Ledger evidence.

        Local action ids are scoped to one component graph.  The only legal
        cross-graph edge is an evidence id recorded by the completed producer
        action.  This is generic for data handles, relation artifacts and
        ordinary source/tool evidence; it does not invent a scientific join.
        """

        by_producer: dict[str, list[str]] = {}
        for entry in context.prior_evidence:
            producer = str((getattr(entry, "tool_parameters", {}) or {}).get("local_action_id") or "")
            evidence_id = str(getattr(entry, "evidence_id", "") or "")
            if producer and evidence_id:
                by_producer.setdefault(producer, []).append(evidence_id)
        resolved: list[str] = []
        for dependency in foreign_dependencies:
            candidates = list(dict.fromkeys(by_producer.get(dependency, [])))
            if len(candidates) != 1:
                return None
            resolved.append(candidates[0])
        return replace(
            action,
            depends_on_action_ids=[dependency for dependency in action.depends_on_action_ids if dependency not in foreign_dependencies],
            input_evidence_ids=list(dict.fromkeys([*action.input_evidence_ids, *resolved])),
        )

    def _bind_unique_materialized_handle(self, action: ResearchAction, context: ToolContext, *, local_action_ids: set[str] | None = None) -> ResearchAction:
        """Bind an unambiguous upstream workspace handle at the execution edge.

        The model may omit a handle even though the durable Ledger contains
        exactly one compatible materialization.  Requiring the model to copy
        an evidence id across every local subgraph is an avoidable interface
        failure.  When more than one handle is available, leave the action
        untouched so the Researcher must choose explicitly.
        """

        if action.kind != "tool" or not action.tool_id:
            return action
        local_scope_supplied = local_action_ids is not None
        local_ids = set(local_action_ids or [])
        dependencies = list(action.depends_on_action_ids)
        foreign_dependencies = [dependency for dependency in dependencies if local_scope_supplied and dependency not in local_ids]
        tool = self.tools.get(action.tool_id)
        if not tool_consumes_observation_rows(tool):
            return action
        candidates: list[str] = []
        candidate_entries = context.prior_evidence
        if context.plan is not None:
            try:
                allowed_components = evidence_input_component_ids(context.plan, context.component.component_id)
                candidate_entries = [entry for entry in context.prior_evidence if set(getattr(entry, "component_ids", [])).intersection(allowed_components)]
            except Exception:
                candidate_entries = context.prior_evidence
        for entry in candidate_entries:
            findings = getattr(entry, "findings", {}) or {}
            handle = findings.get("dataset_handle") if isinstance(findings, dict) else None
            if not isinstance(handle, dict) or handle.get("handle_contract") != "data_handle/v1":
                continue
            if action.source_id and str(handle.get("source_id") or "") != action.source_id:
                continue
            if foreign_dependencies:
                producer_action_id = str((getattr(entry, "tool_parameters", {}) or {}).get("local_action_id") or "")
                if producer_action_id not in foreign_dependencies:
                    continue
            candidates.append(str(getattr(entry, "evidence_id", "")))
        candidates = list(dict.fromkeys(item for item in candidates if item))
        evidence_id = action.data_handle_evidence_id
        if not evidence_id:
            if len(candidates) != 1:
                return action
            evidence_id = candidates[0]
        entry = next(
            (item for item in context.prior_evidence if getattr(item, "evidence_id", None) == evidence_id),
            None,
        )
        upstream_action_id = (
            str((getattr(entry, "tool_parameters", {}) or {}).get("local_action_id") or "")
            if entry is not None else ""
        )
        if foreign_dependencies:
            dependencies = [dependency for dependency in dependencies if dependency in local_ids]
        if upstream_action_id and (not local_scope_supplied or upstream_action_id in local_ids) and upstream_action_id not in dependencies:
            dependencies.append(upstream_action_id)
        return replace(
            action,
            data_handle_evidence_id=evidence_id,
            depends_on_action_ids=dependencies,
        )

    def _action_input_fingerprint(self, action: ResearchAction, component: Any, graph: LocalExecutionGraph, context: ToolContext) -> str:
        readiness = None
        if action.kind in {"tool", "data_tool"} and action.tool_id:
            readiness = self.tools.assess_readiness(self.tools.get(action.tool_id), context).input_fingerprint
        upstream_evidence = sorted(
            evidence_id
            for action_id in action.depends_on_action_ids
            for evidence_id in graph.action_evidence_ids.get(action_id, [])
        )
        payload = {
            "kind": action.kind,
            "tool_id": action.tool_id,
            "source_id": action.source_id,
            # Built-in runners currently consume the frozen component scope.
            # Only parameters declared by a tool card are computational input;
            # cosmetic free-form JSON cannot legalize a repeated calculation.
            "scope": context.scope,
            "parameters": {
                key: value for key, value in action.parameters.items()
                if action.tool_id and key in self.tools.get(action.tool_id).card.parameter_explanation
            },
            "data_request": action.data_request if action.kind == "data_tool" else {},
            "data_handle_evidence_id": action.data_handle_evidence_id,
            "input_evidence_ids": sorted(action.input_evidence_ids),
            "upstream_evidence_ids": upstream_evidence,
            "tool_input_state": readiness,
            "execution_epoch": component_execution_epoch(context.plan, component.component_id) if context.plan is not None else "",
        }
        return sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()

    def _append_local_note(self, plan: ScientificPlan, component: Any, entries: list[EvidenceEntry]) -> None:
        self.last_notes.append(self._note_for_component(plan, component, entries))

    @staticmethod
    def _scientific_neighbourhood(plan: ScientificPlan, component: Any) -> dict[str, Any]:
        """One-hop scientific context; unrelated global branches stay hidden."""

        upstream_ids = set(component.depends_on)
        downstream = [item for item in plan.required_components if component.component_id in item.depends_on]
        involved = [item for item in plan.required_components if item.component_id in upstream_ids or item.component_id == component.component_id or item in downstream]
        return {
            "scope": "current scientific node plus one-hop neighbours",
            "nodes": [{"node_id": item.component_id, "role": "current" if item.component_id == component.component_id else ("upstream" if item.component_id in upstream_ids else "downstream"), "claim": item.claim, "execution_status": item.execution_status, "scientific_status": item.scientific_status, "evidence_needed": item.evidence_needed, "failure_meaning": item.failure_meaning} for item in involved],
            "edges": ([{"from": upstream, "to": component.component_id, "meaning": "upstream conclusion is an input to the current task"} for upstream in component.depends_on] + [{"from": component.component_id, "to": item.component_id, "meaning": "current result is an input to the downstream task"} for item in downstream]),
        }

    def run(self, plan: ScientificPlan, scout: ScoutReport, *, max_tools_per_component: int = 4, component_ids: list[str] | None = None, max_components: int | None = None) -> list[EvidenceEntry]:
        """Execute only the current plan frontier when requested.

        The runtime normally supplies one component at a time so every actual
        calculation is independently assessed before the next research choice.
        Calling without ``component_ids`` keeps the public batch API useful for
        offline integration tests.
        """
        if component_ids is not None and len(component_ids) > 1 and max_components is None:
            return self._run_parallel_wave(plan, scout, component_ids, max_tools_per_component)
        produced: list[EvidenceEntry] = []
        self.last_notes = []
        available = self.sources.capabilities()
        available.update(capability for dataset in scout.external_datasets if dataset.get("available") for capability in dataset.get("capabilities", []))
        allowed_components = set(component_ids) if component_ids is not None else None
        completed_components = 0
        for component in plan.required_components:
            if allowed_components is not None and component.component_id not in allowed_components:
                continue
            if plan.component_is_terminal(component.component_id) or not self._dependencies_ready(plan, component.depends_on):
                continue
            if max_components is not None and completed_components >= max_components:
                break
            prior_entry_ids = {entry.evidence_id for entry in self._entries_for_component(plan, component.component_id)}
            source_entries = self._acquire_component_sources(plan, component)
            existing = self._entries_for_component(plan, component.component_id)
            existing_tags = {tag for entry in existing for tag in entry.evidence_tags}
            derived_need = list(dict.fromkeys(need for need in component.evidence_needed if need not in RAW_OBSERVABLES))
            candidates = self.tools.select(derived_need, available)
            # The model sees the complete live registry, just like a coding
            # agent sees its complete tool surface.  The matching subset is
            # advisory context, not an irreversible Planner-to-Researcher
            # compression.  The deterministic fallback below remains strict:
            # if no registered capability matches, it runs nothing rather than
            # substituting a merely similar calculation.
            visible_tools = self.tools.all()
            if self.action_advisor is not None:
                try:
                    actions = self.action_advisor.propose(task=plan.task, local_graph=self._local_graph(plan, component), component=component, scout=scout, skills=self.skills.all() if self.skills else [], tool_catalogue=self.tools.catalogue(), matching_tool_ids={tool.tool_id for tool in candidates}, evidence=list(self.ledger.entries), component_evidence=existing, limit=max_tools_per_component)
                    validated_actions = validate_actions(actions, plan, self.tools)
                    validated_actions = self._confirm_actions(plan, component, validated_actions)
                    if self.trace:
                        self.trace("model_actions_validated", {"component_id": component.component_id, "actions": [{"action_id": action.action_id, "kind": action.kind, "tool_id": action.tool_id, "source_id": action.source_id, "rationale": action.rationale} for action in validated_actions]})
                    selected = self._tools_from_actions(validated_actions)
                except (StrictSchemaViolation, ContractViolation, TypeError, ValueError, KeyError) as error:
                    # Invalid model action JSON must never become an invented
                    # operation or stop evidence collection when a registered,
                    # deterministic capability policy can continue safely.
                    if self.trace:
                        self.trace("contract_validation_failed", {"component_id": component.component_id, "phase": "legacy_researcher_action_validation", "error_type": type(error).__name__, "message": str(error)[:2_000]})
                    selected = self._policy_selection(plan, component, candidates, existing_tags, max_tools_per_component)
                except Exception as error:
                    if self.trace:
                        self.trace("model_action_proposal_failed", {"component_id": component.component_id, "phase": "legacy_researcher_action_proposal", "error_type": type(error).__name__, "message": str(error)[:2_000]})
                    selected = self._policy_selection(plan, component, candidates, existing_tags, max_tools_per_component)
            else:
                selected = self._policy_selection(plan, component, candidates, existing_tags, max_tools_per_component)
            component_entries = [*source_entries]
            for tool in selected:
                local_scope = self._scope_from_component_evidence(plan, component)
                context = ToolContext(plan.task.question, plan.task, component, local_scope, self.sources, self.external, self.ledger.entries, component.candidate_hypothesis_ids, plan, scout)
                trace_id = self.trace("tool_started", {"component_id": component.component_id, "tool_id": tool.tool_id, "scope": local_scope}) if self.trace else None
                retry_action = ResearchAction(
                    action_id=f"legacy:{component.component_id}:{tool.tool_id}",
                    component_id=component.component_id, kind="tool", tool_id=tool.tool_id,
                    rationale="Legacy batch execution retry identity.",
                )
                result = self.tools.validate_result(tool, self._execute_action(retry_action, lambda: tool.runner(context)).value)
                entry = self.ledger.record_tool_call(
                    result, [component.component_id], effective_input_scope(context),
                    parameters={}, trace_id=trace_id,
                    upstream_evidence_ids=[entry.evidence_id for entry in existing if entry.kind == "source_query" and entry.status == "ok"],
                    execution_epoch=component_execution_epoch(plan, component.component_id),
                )
                self.evidence_registry.capture(entry)
                produced.append(entry)
                existing.append(entry)
                component_entries.append(entry)
            # AI-Q persists one ResearchNotes artifact per actual evidence
            # execution. A local re-scout that adds no new source/tool result
            # must not manufacture an empty duplicate note.
            if any(entry.evidence_id not in prior_entry_ids for entry in component_entries):
                self.last_notes.append(self._note_for_component(plan, component, component_entries))
            completed_components += 1
        return produced

    def _run_parallel_wave(self, plan: ScientificPlan, scout: ScoutReport, component_ids: list[str], max_tools_per_component: int) -> list[EvidenceEntry]:
        """Run one dependency-free DAG wave concurrently, commit Ledger in order.

        Source acquisition, action validation, Ledger writes, and ResearchNotes
        remain serialized and durable. Only pure registered tool runners for
        mutually independent frontier nodes execute concurrently.
        """
        self.last_notes = []
        available = self.sources.capabilities()
        available.update(capability for dataset in scout.external_datasets if dataset.get("available") for capability in dataset.get("capabilities", []))
        selected_ids = set(component_ids)
        prepared: list[dict[str, Any]] = []
        for component in plan.required_components:
            if component.component_id not in selected_ids or plan.component_is_terminal(component.component_id) or not self._dependencies_ready(plan, component.depends_on):
                continue
            prior_entry_ids = {entry.evidence_id for entry in self._entries_for_component(plan, component.component_id)}
            source_entries = self._acquire_component_sources(plan, component)
            existing = self._entries_for_component(plan, component.component_id)
            existing_tags = {tag for entry in existing for tag in entry.evidence_tags}
            derived_need = list(dict.fromkeys(need for need in component.evidence_needed if need not in RAW_OBSERVABLES))
            candidates = self.tools.select(derived_need, available)
            visible_tools = self.tools.all()
            if self.action_advisor is not None:
                try:
                    actions = self.action_advisor.propose(task=plan.task, local_graph=self._local_graph(plan, component), component=component, scout=scout, skills=self.skills.all() if self.skills else [], tool_catalogue=self.tools.catalogue(), matching_tool_ids={tool.tool_id for tool in candidates}, evidence=list(self.ledger.entries), component_evidence=existing, limit=max_tools_per_component)
                    selected = self._tools_from_actions(self._confirm_actions(plan, component, validate_actions(actions, plan, self.tools)))
                except (StrictSchemaViolation, ContractViolation, TypeError, ValueError, KeyError) as error:
                    if self.trace:
                        self.trace("contract_validation_failed", {"component_id": component.component_id, "phase": "legacy_parallel_action_validation", "error_type": type(error).__name__, "message": str(error)[:2_000]})
                    selected = self._policy_selection(plan, component, candidates, existing_tags, max_tools_per_component)
                except Exception as error:
                    if self.trace:
                        self.trace("model_action_proposal_failed", {"component_id": component.component_id, "phase": "legacy_parallel_action_proposal", "error_type": type(error).__name__, "message": str(error)[:2_000]})
                    selected = self._policy_selection(plan, component, candidates, existing_tags, max_tools_per_component)
            else:
                selected = self._policy_selection(plan, component, candidates, existing_tags, max_tools_per_component)
            prepared.append({"component": component, "prior_entry_ids": prior_entry_ids, "source_entries": source_entries, "existing": existing, "selected": selected})
        if self.trace and prepared:
            self.trace("parallel_research_wave", {"component_ids": [item["component"].component_id for item in prepared], "tool_ids": [tool.tool_id for item in prepared for tool in item["selected"]]})
        jobs: list[tuple[dict[str, Any], Any, Any, str | None]] = []
        with ThreadPoolExecutor(max_workers=max(1, sum(len(item["selected"]) for item in prepared))) as executor:
            for item in prepared:
                component = item["component"]
                for tool in item["selected"]:
                    local_scope = self._scope_from_component_evidence(plan, component)
                    trace_id = self.trace("tool_started", {"component_id": component.component_id, "tool_id": tool.tool_id, "scope": local_scope}) if self.trace else None
                    context = ToolContext(plan.task.question, plan.task, component, local_scope, self.sources, self.external, list(self.ledger.entries), component.candidate_hypothesis_ids, plan, scout)
                    retry_action = ResearchAction(
                        action_id=f"legacy:{component.component_id}:{tool.tool_id}",
                        component_id=component.component_id, kind="tool", tool_id=tool.tool_id,
                        rationale="Legacy parallel execution retry identity.",
                    )
                    jobs.append((item, tool, executor.submit(self._execute_action, retry_action, lambda tool=tool, context=context: tool.runner(context), emit_trace=False), trace_id))
            computed = []
            for item, tool, future, trace_id in jobs:
                retry_action = ResearchAction(
                    action_id=f"legacy:{item['component'].component_id}:{tool.tool_id}",
                    component_id=item["component"].component_id, kind="tool", tool_id=tool.tool_id,
                    rationale="Legacy parallel execution retry identity.",
                )
                try:
                    outcome = future.result()
                    self._record_retry_outcome(retry_action, outcome)
                    computed.append((item, tool, self.tools.validate_result(tool, outcome.value), trace_id))
                except Exception as error:
                    self._record_retry_failure(retry_action, error)
                    raise
        produced: list[EvidenceEntry] = []
        for item in prepared:
            component_entries = [*item["source_entries"]]
            for prepared_item, tool, result, trace_id in computed:
                if prepared_item is not item:
                    continue
                entry = self.ledger.record_tool_call(
                    result, [item["component"].component_id], self._scope_from_component_evidence(plan, item["component"]),
                    parameters={}, trace_id=trace_id,
                    upstream_evidence_ids=[entry.evidence_id for entry in item["existing"] if entry.kind == "source_query" and entry.status == "ok"],
                    execution_epoch=component_execution_epoch(plan, item["component"].component_id),
                )
                self.evidence_registry.capture(entry)
                produced.append(entry); item["existing"].append(entry); component_entries.append(entry)
            if any(entry.evidence_id not in item["prior_entry_ids"] for entry in component_entries):
                self.last_notes.append(self._note_for_component(plan, item["component"], component_entries))
        return produced

    def _acquire_component_sources(self, plan: ScientificPlan, component: Any) -> list[EvidenceEntry]:
        """AI-Q equivalent of source-tool execution before ResearchNotes are emitted."""
        entries: list[EvidenceEntry] = []
        for source in self.sources.all():
            descriptor = source.describe_source()
            source_scope = self.scope_compiler.compile_one(plan.task, source, component=component).canonical_scope
            execution_epoch = component_execution_epoch(plan, component.component_id)
            existing = self.ledger.source_query(
                descriptor.source_id, component.component_id, source_scope, descriptor.version,
                execution_epoch=execution_epoch,
            )
            if existing is None:
                retry_action = ResearchAction(
                    action_id=f"legacy:source:{component.component_id}:{descriptor.source_id}",
                    component_id=component.component_id, kind="source_query", source_id=descriptor.source_id,
                    rationale="Legacy batch source acquisition retry identity.",
                )
                rows = self._execute_action(retry_action, lambda: source.query(source_scope)).value
                existing = self.ledger.record_source_query(
                    descriptor.source_id, [component.component_id], source_scope, rows, descriptor.version,
                    execution_epoch=execution_epoch,
                )
            self.evidence_registry.capture(existing)
            entries.append(existing)
        return entries

    def _scope_for_local_graph(self, plan: ScientificPlan, graph: LocalExecutionGraph, component: Any) -> dict[str, Any]:
        evidence_ids = {
            evidence_id
            for action in graph.actions if action.kind == "source_query"
            for evidence_id in graph.action_evidence_ids.get(action.action_id, [])
        }
        entries = [entry for entry in self._entries_for_component(plan, component.component_id) if entry.kind == "source_query"]
        entries.extend(
            entry for entry in self.ledger.entries
            if entry.evidence_id in evidence_ids and entry.kind == "source_query" and entry not in entries
        )
        return self._scope_bundle(component, entries)

    def _scope_from_component_evidence(self, plan: ScientificPlan, component: Any) -> dict[str, Any]:
        entries = [entry for entry in self._entries_for_component(plan, component.component_id) if entry.kind == "source_query"]
        return self._scope_bundle(component, entries)

    def _entries_for_component(self, plan: ScientificPlan, component_id: str) -> list[EvidenceEntry]:
        """Return the current node's own and graph-upstream evidence packet."""

        return EvidenceAdjudicator.entries_for_component(plan, plan.component(component_id), self.ledger)

    def _scope_bundle(self, component: Any, entries: list[EvidenceEntry]) -> dict[str, Any]:
        """Build the executable local query boundary from actual source reads.

        The global component retains only the user's rough planning scope.
        Source-specific predicates are local execution artifacts and therefore
        travel with the evidence that established them.
        """

        latest: dict[str, EvidenceEntry] = {}
        latest_by_table: dict[tuple[str, str], EvidenceEntry] = {}
        for entry in entries:
            latest[entry.subject_id] = entry
            table_id = str((entry.input_scope or {}).get("table_id") or "")
            if table_id:
                latest_by_table[(entry.subject_id, table_id)] = entry
        source_scopes: dict[str, dict[str, Any]] = {}
        for source in self.sources.all():
            if source.source_id in latest:
                source_scopes[source.source_id] = latest[source.source_id].input_scope
            else:
                source_scopes[source.source_id] = {
                    "_scope_contract": "observation_scope/v1",
                    "source_id": source.source_id,
                    "status": "scope_unresolvable",
                    "predicates": [],
                    "unresolved_terms": ["This source has not been explored in the current local graph."],
                    "provenance": {"compiler": "research_local_scope_bundle"},
                }
        table_scopes = {
            f"{source_id}::{table_id}": entry.input_scope
            for (source_id, table_id), entry in latest_by_table.items()
        }
        return {
            "_scope_contract": "scope_bundle/v2",
            "status": "resolved" if latest else "scope_unresolvable",
            "source_scopes": source_scopes,
            "table_scopes": table_scopes,
            "planning_scope": component.scope,
            "provenance": {"owner": "researcher_local_execution_graph"},
        }

    @staticmethod
    def _query_for_component(component: Any) -> ScientificResearchQuery:
        return ScientificResearchQuery(
            query_id=f"query_{component.component_id}",
            information_need="; ".join(component.evidence_needed),
            target_components=[component.component_id], scope=component.scope,
            preferred_sensor_kinds=[],
            preferred_capabilities=list(component.evidence_needed),
            fallback_sensor_kinds=[],
            rationale=component.failure_meaning,
        )

    def _note_for_component(self, plan: ScientificPlan, component: Any, entries: list[EvidenceEntry]) -> ScientificResearchNotes:
        language = plan.task_analysis.language
        return note_from_entries(self._query_for_component(component), entries, self.evidence_registry, language=language)

    def _tools_from_actions(self, actions: list[ResearchAction]) -> list[ToolSpec]:
        by_id = {action.action_id: action for action in actions}
        ordered: list[ResearchAction] = []
        done: set[str] = set()
        while len(done) < len(actions):
            ready = [action for action in actions if action.action_id not in done and set(action.depends_on_action_ids).issubset(done)]
            if not ready:  # validate_actions already guards cycles; retain a defensive stop.
                break
            for action in ready:
                ordered.append(action); done.add(action.action_id)
        return [self.tools.get(action.tool_id) for action in ordered if action.kind == "tool" and action.tool_id]

    def _confirm_actions(self, plan: ScientificPlan, component: Any, actions: list[ResearchAction], *, local_graph: dict[str, Any] | None = None, existing_actions: list[ResearchAction] | None = None, context: ToolContext | None = None) -> list[ResearchAction]:
        """初筛后才展开被选工具的完整能力卡；兼容测试替身。"""

        if not actions or not hasattr(self.action_advisor, "confirm_tool_actions"):
            return actions
        cards = [self.tools.capability_card(action.tool_id, context) for action in actions if action.kind == "tool" and action.tool_id]
        if not cards:
            return actions
        confirmed = self.action_advisor.confirm_tool_actions(task=plan.task, local_graph=local_graph or self._local_graph(plan, component), actions=actions, capability_cards=cards)
        # The confirmation model is a selector, not a second scientific
        # readiness gate.  A model may reject every action because it is
        # uncertain, but a registered action that already passes the
        # deterministic input contract must remain executable.  Otherwise a
        # valid KDE/GMM or bootstrap action can disappear without a tool call
        # and the Judge only sees a silent calculation gap.
        accepted_ids = {action.action_id for action in confirmed}
        readiness_context = context or ToolContext(
            plan.task.question, plan.task, component,
            self._scope_from_component_evidence(plan, component), self.sources,
            self.external, list(self.ledger.entries), component.candidate_hypothesis_ids,
            plan, None,
        )
        bound_actions: dict[str, ResearchAction] = {}
        candidate_local_action_ids = {
            item.action_id for item in (existing_actions or [])
        } | {item.action_id for item in actions}
        for action in actions:
            if action.kind != "tool" or not action.tool_id:
                continue
            bound = self._bind_unique_materialized_handle(
                action,
                readiness_context,
                local_action_ids=candidate_local_action_ids,
            )
            bound_actions[action.action_id] = bound
            readiness = self.tools.assess_readiness(
                self.tools.get(action.tool_id),
                replace(readiness_context, action=bound, require_materialized_handle=True),
            )
            if readiness.status == "ready":
                accepted_ids.add(action.action_id)
        confirmed = [
            bound_actions.get(action.action_id, action)
            for action in actions
            if action.kind != "tool" or action.action_id in accepted_ids
        ]
        # A later local-DAG wave may depend on completed nodes not present in
        # this proposal.  Validate the cumulative graph in that case.
        validate_actions([*(existing_actions or []), *confirmed], plan, self.tools)
        if self.trace:
            self.trace("tool_actions_confirmed", {"component_id": component.component_id, "action_ids": [action.action_id for action in confirmed]})
        return confirmed

    @staticmethod
    def _local_graph(plan: ScientificPlan, component: Any) -> dict[str, Any]:
        """One-hop local graph for the Researcher; exclude unrelated global nodes."""

        upstream_ids = set(component.depends_on)
        downstream = [item for item in plan.required_components if component.component_id in item.depends_on]
        involved = [item for item in plan.required_components if item.component_id in upstream_ids or item.component_id == component.component_id or item in downstream]
        nodes = [
            {
                "node_id": item.component_id,
                "role": "current" if item.component_id == component.component_id else ("upstream" if item.component_id in upstream_ids else "downstream"),
                "claim": item.claim,
                "execution_status": item.execution_status,
                "scientific_status": item.scientific_status,
                "evidence_needed": item.evidence_needed,
                "failure_meaning": item.failure_meaning,
            }
            for item in involved
        ]
        edges = []
        for upstream in component.depends_on:
            edges.append({"from": upstream, "to": component.component_id, "meaning": "upstream conclusion is an input to the current task"})
        for item in downstream:
            edges.append({"from": component.component_id, "to": item.component_id, "meaning": "current result is an input to the downstream task"})
        return {"scope": "current node plus one-hop neighbours", "nodes": nodes, "edges": edges}

    def _policy_selection(self, plan: ScientificPlan, component: Any, candidates: list[ToolSpec], existing_tags: set[str], limit: int) -> list[ToolSpec]:
        if hasattr(self.policy, "choose_for_component"):
            return self.policy.choose_for_component(plan.task.question, component, candidates, existing_tags, limit)
        return self.policy.choose(candidates, existing_tags, limit)

    @staticmethod
    def _dependencies_ready(plan: ScientificPlan, dependencies: list[str]) -> bool:
        return all(
            plan.component(component_id).scientific_status == "supported"
            for component_id in dependencies
        )
