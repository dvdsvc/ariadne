"""Durable references from a data.tool materialize node to local data."""

from __future__ import annotations

from typing import Any


def effective_input_scope(context: Any) -> dict[str, Any]:
    """Return the source scope actually consumed by a row-based tool.

    The Planner/local graph scope is a routing envelope. Once a tool is bound
    to a materialized dataset, the handle's compiled source scope is the
    authoritative input and must also be recorded in the Ledger.
    """

    action = getattr(context, "action", None)
    evidence_id = (
        getattr(action, "data_handle_evidence_id", "")
        or ((getattr(action, "parameters", {}) or {}).get("data_handle_evidence_id") if action else None)
    )
    if not evidence_id:
        return dict(getattr(context, "scope", {}) or {})
    entry = next((item for item in getattr(context, "prior_evidence", []) if getattr(item, "evidence_id", None) == evidence_id), None)
    findings = getattr(entry, "findings", {}) if entry is not None else {}
    handle = findings.get("dataset_handle") if isinstance(findings, dict) else None
    handle_scope = handle.get("scope") if isinstance(handle, dict) else None
    handle_table_id = str(handle.get("table_id") or (handle.get("request") or {}).get("table_id") or "") if isinstance(handle, dict) else ""
    invalid_table_projection = (
        isinstance(handle_scope, dict)
        and handle_scope.get("_scope_contract") == "table_query/v1"
        and handle_table_id
        and handle_scope.get("table_id") != handle_table_id
    )
    if not isinstance(handle_scope, dict) or invalid_table_projection:
        handle = enrich_materialized_handle(handle, entry, str(handle.get("source_id") or "")) if isinstance(handle, dict) else handle
        handle_scope = handle.get("scope") if isinstance(handle, dict) else None
    if not isinstance(handle_scope, dict):
        return dict(getattr(context, "scope", {}) or {})
    current = dict(getattr(context, "scope", {}) or {})
    if current.get("_scope_contract") in {"scope_bundle/v1", "scope_bundle/v2"}:
        source_scopes = dict(current.get("source_scopes") or {})
        source_id = str(handle.get("source_id") or "")
        source_scopes[source_id] = handle_scope
        current["source_scopes"] = source_scopes
        table_id = str(handle.get("table_id") or (handle.get("request") or {}).get("table_id") or handle_scope.get("table_id") or "")
        if table_id:
            table_scopes = dict(current.get("table_scopes") or {})
            table_scopes[f"{source_id}::{table_id}"] = handle_scope
            current["table_scopes"] = table_scopes
        current["status"] = "resolved"
        current["provenance"] = {**dict(current.get("provenance") or {}), "materialized_handle_evidence_id": evidence_id}
        return current
    return handle_scope


def _source_scope_from_entry(entry: Any, source_id: str, table_id: str = "") -> dict[str, Any] | None:
    """Recover the semantic table binding persisted beside an old handle."""

    input_scope = getattr(entry, "input_scope", {}) or {}
    if not isinstance(input_scope, dict):
        return None
    if input_scope.get("_scope_contract") == "table_query/v1":
        if input_scope.get("source_id") not in {None, "", source_id}:
            return None
        if table_id and input_scope.get("table_id") != table_id:
            return None
        return dict(input_scope)
    if input_scope.get("_scope_contract") not in {"scope_bundle/v1", "scope_bundle/v2"}:
        return None
    if table_id:
        table_scopes = input_scope.get("table_scopes")
        candidate = table_scopes.get(f"{source_id}::{table_id}") if isinstance(table_scopes, dict) else None
        if isinstance(candidate, dict) and candidate.get("table_id") == table_id:
            return dict(candidate)
        legacy = input_scope.get("source_scopes", {}).get(source_id) if isinstance(input_scope.get("source_scopes"), dict) else None
        if isinstance(legacy, dict) and legacy.get("table_id") == table_id:
            return dict(legacy)
        return None
    source_scopes = input_scope.get("source_scopes")
    candidate = source_scopes.get(source_id) if isinstance(source_scopes, dict) else None
    if not isinstance(candidate, dict) or candidate.get("_scope_contract") != "table_query/v1":
        return None
    return dict(candidate)


def enrich_materialized_handle(handle: dict[str, Any], entry: Any | None, source_id: str) -> dict[str, Any]:
    """Attach the canonical row projection to a persisted materialization.

    Older runs stored a valid raw-table handle but omitted the dynamic table
    navigation scope that described how physical columns map to canonical
    scientific fields.  The evidence entry already persisted that scope, so
    it is safe to recover it deterministically without asking the model or
    inventing aliases.  New handles are enriched at creation time; this path
    keeps checkpoint resume backward-compatible.
    """

    enriched = dict(handle or {})
    if enriched.get("source_id") not in {None, "", source_id}:
        return enriched
    current = enriched.get("scope")
    current_table_id = str(enriched.get("table_id") or (enriched.get("request") or {}).get("table_id") or "")
    current_is_native_scope = (
        isinstance(current, dict)
        and current.get("_scope_contract") != "table_query/v1"
    )
    current_is_valid_table_scope = (
        isinstance(current, dict)
        and current.get("_scope_contract") == "table_query/v1"
        and current.get("bindings")
        and (not current_table_id or current.get("table_id") == current_table_id)
    )
    if current_is_native_scope or current_is_valid_table_scope:
        scope = dict(current)
    else:
        scope = _source_scope_from_entry(entry, source_id, current_table_id)
    scope_is_native = isinstance(scope, dict) and scope.get("_scope_contract") != "table_query/v1"
    scope_has_valid_table_bindings = isinstance(scope, dict) and bool(scope.get("bindings"))
    if not isinstance(scope, dict) or (not scope_is_native and not scope_has_valid_table_bindings):
        # Preserve the handle identity but remove an invalid semantic
        # projection.  A stale projection must not survive resume and be
        # mistaken for a valid canonical input by downstream tools.
        enriched.pop("scope", None)
        enriched.pop("projection_contract", None)
        enriched.pop("semantic_fields", None)
        return enriched
    enriched["scope"] = scope
    enriched["projection_contract"] = "canonical_rows/v1"
    enriched["semantic_fields"] = [
        str(item["canonical_field"])
        for item in (scope.get("bindings") or [])
        if isinstance(item, dict) and item.get("canonical_field")
    ]
    return enriched


def resolve_rows(context: Any) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Load only a Researcher-declared materialized dataset.

    New analysis providers must use this function.  The fallback is retained
    solely for existing deterministic fixtures that predate data.tool;
    model-backed actions are rejected unless they name a Ledger handle.
    """

    action = getattr(context, "action", None)
    evidence_id = (getattr(action, "data_handle_evidence_id", "") or ((getattr(action, "parameters", {}) or {}).get("data_handle_evidence_id") if action else None))
    if not evidence_id:
        if getattr(context, "require_materialized_handle", False):
            raise ValueError("Analysis action requires data_handle_evidence_id from an upstream data.tool materialize result")
        return context.source_registry.query_all(context.scope), None
    entry = next((item for item in context.prior_evidence if getattr(item, "evidence_id", None) == evidence_id), None)
    if entry is None:
        raise ValueError(f"Unknown data handle evidence id: {evidence_id}")
    handle = getattr(entry, "findings", {}).get("dataset_handle")
    if not isinstance(handle, dict) or handle.get("handle_contract") != "data_handle/v1":
        raise ValueError(f"Evidence {evidence_id} is not a materialized data handle")
    handle = enrich_materialized_handle(handle, entry, str(handle.get("source_id") or ""))
    source = context.source_registry.get(str(handle.get("source_id") or ""))
    if not hasattr(source, "load_workspace_handle"):
        raise ValueError(f"Source {source.source_id} cannot load materialized workspace handles")
    return source.load_workspace_handle(handle), handle
