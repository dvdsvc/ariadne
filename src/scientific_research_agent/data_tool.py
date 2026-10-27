"""Read-only relational operations behind the ordinary ``data.tool`` tool.

This is deliberately a *tool*, not another Scout or graph node.  It inspects
or transforms one registered source for one current scientific component.  The
request contains no SQL and no executable Python: sources validate a finite
operation vocabulary against their live schema and perform the work locally.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Any

from .contracts import ToolResult
from .data_handles import enrich_materialized_handle
from .data_operation_contract import DATA_ARTIFACT_CONTRACT, add_artifact_contract, operation_metadata, operation_status
from .evidence_contract import observable_manifest_from_rows
from .registry import ToolContext
from .relational_engine import DuckDBRelationalEngine


OPERATIONS = ("inspect", "profile", "sample", "query", "aggregate", "materialize", "relate")
AGGREGATIONS = ("count", "min", "max", "mean", "median", "std", "n_unique")
FILTER_OPERATORS = ("eq", "in", "between", "contains")


def is_relation_request(request: dict[str, Any]) -> bool:
    """Return whether a workspace request asks for a relational object.

    The distinction is semantic, not tied to the model's selected operation
    string.  Older model turns could put a table list or relation edges next
    to ``operation=materialize``; those requests must still be recognized as
    relational before their result is accepted as a plain table handle.
    """

    tables = request.get("tables")
    return bool(request.get("relations")) or (isinstance(tables, list) and len(tables) >= 2) or bool(request.get("related_table_id"))


def valid_materialized_output(request: dict[str, Any], findings: dict[str, Any]) -> bool:
    """Validate the durable data object promised by a materialize action."""

    handle = findings.get("dataset_handle")
    source_id = findings.get("source_id")
    if not isinstance(handle, dict) or handle.get("handle_contract") != "data_handle/v1":
        return False
    # New handles use one artifact contract regardless of whether their
    # physical operation was a table query, a join, or an aggregate.  Accept
    # old checkpoint handles without this field so long runs remain resumable.
    if handle.get("artifact_contract") not in {None, DATA_ARTIFACT_CONTRACT}:
        return False
    operation = findings.get("data_operation")
    if isinstance(operation, dict) and operation.get("status") != "completed":
        return False
    if handle.get("artifact_contract") == DATA_ARTIFACT_CONTRACT:
        provenance = handle.get("provenance")
        if not (
            handle.get("artifact_type") == "tabular_observation"
            and isinstance(provenance, dict)
            and provenance.get("source_id") in {None, "", source_id}
            and provenance.get("operation") in {"materialize", "relate"}
            and isinstance(handle.get("record_count"), int)
            and handle["record_count"] >= 0
        ):
            return False
        # A content-addressed artifact is not sufficient by itself.  It must
        # still describe the relation request that produced it, including its
        # bounded input scope.
        # ``add_artifact_contract`` is also used to upgrade older in-memory
        # handles which did not duplicate the request at top level.  The
        # provenance copy is authoritative for those handles.
        if provenance.get("request") != request or (handle.get("request") not in (None, {}) and handle.get("request") != request):
            return False
    if source_id and handle.get("source_id") not in {None, "", source_id}:
        return False
    if not isinstance(handle.get("record_count"), int) or handle["record_count"] < 0:
        return False
    if not is_relation_request(request):
        return True

    tables = request.get("tables")
    relations = request.get("relations")
    if isinstance(tables, list) and (len(tables) >= 2 or relations):
        request_table_ids = {str(item.get("table_id")) for item in tables if isinstance(item, dict) and item.get("table_id")}
        handle_table_ids = {str(item.get("table_id")) for item in (handle.get("tables") or []) if isinstance(item, dict) and item.get("table_id")}
        request_edges = {
            (str(item.get("left_table_id")), str(item.get("left_key")), str(item.get("right_table_id")), str(item.get("right_key")), str(item.get("join_type") or "inner"))
            for item in (relations or []) if isinstance(item, dict)
        }
        handle_edges = {
            (str(item.get("left_table_id")), str(item.get("left_key")), str(item.get("right_table_id")), str(item.get("right_key")), str(item.get("join_type") or "inner"))
            for item in (handle.get("relations") or []) if isinstance(item, dict)
        }
        return (
            handle.get("relation_contract") == "table_relation_graph/v1"
            and isinstance(handle.get("tables"), list) and len(handle["tables"]) >= 2
            and isinstance(handle.get("relations"), list) and bool(handle["relations"])
            and isinstance(handle.get("join_order"), list) and bool(handle["join_order"])
            and handle.get("relation_mode") == "materialize"
            and request_table_ids == handle_table_ids
            and request_edges == handle_edges
            and handle.get("request", {}).get("filters", []) == request.get("filters", [])
        )
    return (
        handle.get("relation_contract") == "table_relation/v1"
        and handle.get("table_id")
        and handle.get("related_table_id")
        and handle.get("left_key")
        and handle.get("right_key")
        and handle.get("relation_mode") == "materialize"
        and handle.get("table_id") == request.get("table_id")
        and handle.get("related_table_id") == request.get("related_table_id")
        and handle.get("left_key") == request.get("left_key")
        and handle.get("right_key") == request.get("right_key")
        and handle.get("request", {}).get("filters", []) == request.get("filters", [])
    )


def valid_relation_output(request: dict[str, Any], findings: dict[str, Any]) -> bool:
    """Verify that a relation result is the declared relation, not merely tagged so."""

    if not is_relation_request(request):
        return False
    handle = findings.get("dataset_handle")
    if isinstance(handle, dict):
        return valid_materialized_output(request, findings)
    tables, relations = request.get("tables"), request.get("relations")
    if isinstance(tables, list) and (len(tables) >= 2 or relations):
        requested_tables = {str(item.get("table_id")) for item in tables if isinstance(item, dict) and item.get("table_id")}
        actual_tables = {str(item.get("table_id")) for item in findings.get("tables", []) if isinstance(item, dict) and item.get("table_id")}
        requested_edges = {
            (str(item.get("left_table_id")), str(item.get("left_key")), str(item.get("right_table_id")), str(item.get("right_key")), str(item.get("join_type") or "inner"))
            for item in (relations or []) if isinstance(item, dict)
        }
        actual_edges = {
            (str(item.get("left_table_id")), str(item.get("left_key")), str(item.get("right_table_id")), str(item.get("right_key")), str(item.get("join_type") or "inner"))
            for item in findings.get("relations", []) if isinstance(item, dict)
        }
        if not (
            findings.get("relation_contract") == "table_relation_graph/v1"
            and findings.get("connected") is True and findings.get("acyclic") is True
            and bool(findings.get("edge_reports"))
            and requested_tables == actual_tables and requested_edges == actual_edges
        ):
            return False
    else:
        if not (
            findings.get("relation_contract") == "table_relation/v1"
            and findings.get("left_table_id") == request.get("table_id")
            and findings.get("right_table_id") == request.get("related_table_id")
            and findings.get("left_key") == request.get("left_key")
            and findings.get("right_key") == request.get("right_key")
        ):
            return False
    if request.get("filters"):
        scope = findings.get("relation_scope")
        return isinstance(scope, dict) and scope.get("filtered_table_id") == request.get("table_id") and scope.get("filters") == request.get("filters")
    return True


def workspace_catalogue(source_registry: Any) -> list[dict[str, Any]]:
    """Return live, source-owned fields usable by strict data actions.

    It purposely exposes physical fields for generic tables and canonical
    fields for a native scientific adapter.  No aliases or keyword matching is
    invented here.
    """

    items: list[dict[str, Any]] = []
    for source in source_registry.all():
        if hasattr(source, "workspace_catalogue"):
            raw = source.workspace_catalogue()
            if isinstance(raw, dict):
                items.append(raw)
                continue
        contract = source.describe_query_contract()
        fields = [item.get("field") for item in contract.get("fields", []) if isinstance(item, dict) and isinstance(item.get("field"), str)]
        items.append({
            "source_id": source.source_id,
            "data_operation_contract": "data_operation/v1",
            "decision_contract": "data_decision_required/v1",
            "tables": [{"table_id": "", "fields": fields}],
            # A generic adapter is only allowed to advertise operations it
            # can actually execute.  Relation probing belongs to a provider
            # with a multi-table contract; it is not inferred from a source's
            # mere existence.
            "operations": [item for item in OPERATIONS if item != "relate"],
            "aggregate_operations": list(AGGREGATIONS),
            "filter_operators": sorted({
                str(operator)
                for item in contract.get("fields", [])
                if isinstance(item, dict)
                for operator in item.get("operators", [])
                if isinstance(operator, str) and operator
            }) or list(FILTER_OPERATORS),
        })
    return items


def run_data_tool(context: ToolContext) -> ToolResult:
    """Execute one source-owned relational operation and return Ledger evidence."""

    action = context.action
    if action is None or not getattr(action, "source_id", None):
        # Direct registry audits deliberately execute every built-in runner
        # without a local DAG node.  A workspace cannot perform a meaningful
        # operation in that situation, so publish no scientific evidence.
        return ToolResult(
            result_id=f"data.tool:{context.component.component_id}", tool_id="data.tool",
            status="not_applicable", calculation_status="not_applicable",
            summary="No local data-tool action was supplied.", findings={}, evidence_tags=[],
            uncertainty={}, data_versions={}, limitations=["data.tool requires a source-bound local DAG action."],
        )
    request = dict(getattr(action, "data_request", {}) or {})
    operation = request.get("operation")
    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported data tool operation: {operation!r}")
    source = context.source_registry.get(action.source_id)
    requested_table_id = str(request.get("table_id") or "")
    input_scope = context.source_registry.scope_for(source.source_id, context.scope, requested_table_id or None)
    physical_rows = input_scope.get("projection_contract") == "physical_rows/v1"
    if requested_table_id and operation != "inspect" and (
        input_scope.get("_scope_contract") != "table_query/v1"
        or input_scope.get("table_id") != requested_table_id
        or input_scope.get("status") not in {"resolved", "partial", "unconstrained"}
        or (not physical_rows and not input_scope.get("bindings"))
    ):
        return ToolResult(
            result_id=f"data.tool:{context.component.component_id}:{getattr(action, 'action_id', '')}",
            tool_id="data.tool", status="error", calculation_status="not_computable",
            summary="The data operation request and compiled input scope name different physical tables.",
            findings={"operation": operation, "source_id": source.source_id, "request": _public_request(request), "contract_error": "input_scope_table_mismatch"},
            evidence_tags=[], uncertainty={}, data_versions={source.source_id: source.describe_source().version},
            limitations=["A multi-table source scope must be selected by source_id and table_id together."],
        )
    if hasattr(source, "workspace_execute"):
        response = source.workspace_execute(request)
    else:
        response = _generic_execute(source, request, context.source_registry.scope_for(source.source_id, context.scope, str(request.get("table_id") or "") or None))
    if not isinstance(response, dict):
        raise ValueError("Data tool provider returned a non-object response")
    findings = dict(response.get("findings") or {})
    findings.setdefault("operation", operation)
    findings.setdefault("source_id", source.source_id)
    findings.setdefault("request", _public_request(request))
    findings["data_operation"] = operation_metadata(
        source_id=source.source_id,
        operation=str(operation),
        request=_public_request(request),
        response=response,
        findings=findings,
        input_scope=input_scope,
    )
    if isinstance(findings.get("dataset_handle"), dict):
        findings["dataset_handle"] = add_artifact_contract(
            findings["dataset_handle"], source_id=source.source_id, operation=str(operation),
            request=_public_request(request), record_count=int(findings.get("record_count") or findings["dataset_handle"].get("record_count") or 0),
        )
        findings["data_operation"]["output_artifact_id"] = findings["dataset_handle"].get("artifact_id", "")
    if operation == "materialize" and is_relation_request(request) and not valid_materialized_output(request, findings):
        # Never publish a data_materialize tag for a relational request whose
        # provider returned only a plain table handle.  The model may retry or
        # replan, but the Ledger must not turn a broken data boundary into
        # scientific evidence.
        return ToolResult(
            result_id=f"data.tool:{context.component.component_id}:{getattr(action, 'action_id', '')}",
            tool_id="data.tool", status="error", calculation_status="not_computable",
            summary="The relational materialize request did not return a valid relation data handle.",
            findings={**findings, "contract_error": "relation_materialize_requires_relation_handle"},
            evidence_tags=[], uncertainty={},
            data_versions={source.source_id: source.describe_source().version},
            limitations=["A materialize request must return a data_handle/v1 with the common data_artifact/v1 contract and a valid operation-specific transformation."],
        )
    if operation == "materialize" and isinstance(findings.get("dataset_handle"), dict):
        findings["dataset_handle"] = enrich_materialized_handle(
            findings["dataset_handle"], None, source.source_id,
        )
        if isinstance(input_scope, dict) and input_scope.get("_scope_contract") == "table_query/v1" and input_scope.get("bindings"):
            findings["dataset_handle"]["scope"] = dict(input_scope)
            findings["dataset_handle"]["projection_contract"] = "canonical_rows/v1"
            findings["dataset_handle"]["semantic_fields"] = [
                str(item["canonical_field"])
                for item in input_scope["bindings"]
                if isinstance(item, dict) and item.get("canonical_field")
            ]
    if operation in {"profile", "sample", "query", "materialize"} and not findings.get("observables"):
        # This is direct-observable metadata, not scientific support. It lets
        # the Judge verify that the data operation supplied age/location/error
        # inputs without promoting the data read into a conclusion.
        try:
            findings["observables"] = source.describe_observables(input_scope)
        except Exception:
            sample = findings.get("sample")
            if isinstance(sample, list):
                findings["observables"] = observable_manifest_from_rows(sample)
    tag = f"data_{operation}"
    evidence_tags = [] if operation_status(response) in {"unavailable", "invalid_request"} else [tag]
    return ToolResult(
        result_id=f"data.tool:{context.component.component_id}:{getattr(action, 'action_id', '')}",
        tool_id="data.tool",
        status=("error" if operation_status(response) in {"unavailable", "invalid_request", "decision_required"} else str(response.get("status") or "ok")),
        calculation_status=("not_computable" if operation_status(response) in {"unavailable", "invalid_request", "decision_required"} else "computed"),
        summary=str(response.get("summary") or f"Completed read-only {operation} operation on {source.source_id}."),
        findings=findings,
        evidence_tags=evidence_tags,
        uncertainty=dict(response.get("uncertainty") or {}),
        data_versions={source.source_id: source.describe_source().version},
        limitations=list(response.get("limitations") or []),
        warnings=list(response.get("warnings") or []),
    )


def _generic_execute(source: Any, request: dict[str, Any], scope: dict[str, Any]) -> dict[str, Any]:
    """Safe fallback for simple in-memory adapters used by tests.

    Large or native sources should implement ``workspace_execute`` so they can
    aggregate in their own storage engine rather than materialising rows here.
    """

    operation = request["operation"]
    if operation == "inspect":
        return {"summary": f"Inspected {source.source_id}.", "findings": {"schema": source.describe_schema(), "query_contract": source.describe_query_contract()}}
    if operation == "profile":
        profile = source.profile_coverage(scope)
        return {"summary": f"Profiled {profile.matched_records} records from {source.source_id}.", "findings": {"matched_records": profile.matched_records, "total_records": profile.total_records, "missingness": profile.missingness, "resolution": profile.resolution, "uncertainty": profile.uncertainty}}
    rows = source.query(scope)
    fields = [
        str(item.get("field"))
        for item in source.describe_query_contract().get("fields", [])
        if isinstance(item, dict) and item.get("field")
    ]
    if not fields and rows:
        fields = list(rows[0])
    engine_result = DuckDBRelationalEngine().execute_table(rows, fields, request)
    if operation in {"sample", "query"}:
        return {
            "summary": f"Read {engine_result['returned_records']} of {engine_result['record_count']} available normalized records from {source.source_id} with DuckDB.",
            "findings": engine_result,
            "limitations": ["Generic fallback materialised rows in process; native large sources should provide database-side workspace operations."],
        }
    if operation == "materialize":
        return {
            "summary": f"Materialized {engine_result['record_count']} records from {source.source_id} with DuckDB.",
            "findings": {"record_count": engine_result["record_count"], "dataset_handle": _handle(source.source_id, request, engine_result["record_count"], scope=scope)},
        }
    return {"summary": f"Aggregated {engine_result['input_record_count']} records from {source.source_id} with DuckDB.", "findings": engine_result}


def _aggregate_rows(rows: list[dict[str, Any]], request: dict[str, Any], source_id: str) -> dict[str, Any]:
    filtered = [row for row in rows if _matches_all(row, request.get("filters") or [])]
    group_by = [field for field in request.get("group_by", []) if field]
    bin_field = str(request.get("bin_field") or "")
    width = float(request.get("bin_width") or 0)
    origin = float(request.get("bin_origin") or 0)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        key: list[Any] = [row.get(field) for field in group_by]
        if bin_field and width > 0:
            value = _number(row.get(bin_field))
            key.append(None if value is None else origin + int((value - origin) // width) * width)
        groups[tuple(key)].append(row)
    if not group_by and not (bin_field and width > 0):
        groups[()] = filtered
    aggregate_specs = request.get("aggregates") or [{"field": "", "operation": "count", "as": "count"}]
    result_rows = []
    for key, group in groups.items():
        item = {field: key[index] for index, field in enumerate(group_by)}
        if bin_field and width > 0:
            item[f"{bin_field}_bin_start"] = key[-1]
        for spec in aggregate_specs:
            name = str(spec.get("as") or f"{spec.get('operation')}_{spec.get('field')}")
            item[name] = _aggregate(group, str(spec.get("field") or ""), str(spec.get("operation") or "count"))
        result_rows.append(item)
    result_rows.sort(key=lambda item: tuple(str(value) for value in item.values()))
    limit = max(1, int(request.get("row_limit") or 500))
    return {
        "summary": f"Aggregated {len(filtered)} records from {source_id} into {len(result_rows)} group(s).",
        "findings": {"input_record_count": len(filtered), "group_count": len(result_rows), "rows": result_rows[:limit], "truncated": len(result_rows) > limit},
    }


def _matches_all(row: dict[str, Any], filters: list[dict[str, Any]]) -> bool:
    for predicate in filters:
        field, operator = str(predicate.get("field") or ""), str(predicate.get("operator") or "")
        value = row.get(field)
        if operator == "between":
            number = _number(value)
            lower, upper = _number(predicate.get("minimum_number")), _number(predicate.get("maximum_number"))
            if number is None or lower is None or upper is None or not min(lower, upper) <= number <= max(lower, upper):
                return False
        elif operator == "contains":
            values = [str(item).casefold() for item in predicate.get("text_values", [])]
            if not any(item in str(value or "").casefold() for item in values):
                return False
        elif operator in {"eq", "in"}:
            values = {str(item).casefold() for item in predicate.get("text_values", [])}
            if str(value).casefold() not in values:
                return False
        else:
            return False
    return True


def _aggregate(rows: list[dict[str, Any]], field: str, operation: str) -> Any:
    if operation == "count":
        return (
            len(rows)
            if not field
            else sum(row.get(field) not in (None, "") for row in rows)
        )
    values = [_number(row.get(field)) for row in rows]
    numeric = [value for value in values if value is not None]
    if operation == "n_unique":
        return len({str(row.get(field)) for row in rows if row.get(field) not in (None, "")})
    if not numeric:
        return None
    if operation == "min": return min(numeric)
    if operation == "max": return max(numeric)
    if operation == "mean": return mean(numeric)
    if operation == "median": return median(numeric)
    if operation == "std": return pstdev(numeric) if len(numeric) > 1 else 0.0
    raise ValueError(f"Unsupported aggregate operation: {operation}")


def _number(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _handle(source_id: str, request: dict[str, Any], record_count: int, *, scope: dict[str, Any] | None = None) -> dict[str, Any]:
    # The request describes how the source was queried; the compiled source
    # scope is the executable input for adapters whose physical columns do not
    # expose every semantic predicate (for example location_text).  Keeping
    # both makes the handle portable without asking downstream tools to
    # reconstruct a query from prose or aliases.
    handle = {
        "handle_contract": "data_handle/v1", "source_id": source_id,
        "request": _public_request(request), "scope": dict(scope or {}),
        "record_count": record_count,
    }
    return add_artifact_contract(handle, source_id=source_id, operation=str(request.get("operation") or "materialize"), request=_public_request(request), record_count=record_count)


def _public_request(request: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in request.items()
        if key in {
            "operation", "table_id", "related_table_id", "left_key", "right_key",
            "join_type", "relation_mode", "tables", "relations", "columns", "filters", "group_by",
            "aggregates", "bin_field", "bin_width", "bin_origin", "row_limit",
            "dataset_name", "input_artifact_ids", "semantic_authorization", "match_policy", "match_tolerance",
        }
    }
