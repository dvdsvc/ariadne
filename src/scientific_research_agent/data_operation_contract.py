"""Shared contract for every Researcher-facing data operation.

The operation name may be select, relate, aggregate or materialize, but the
boundary is always the same: a deterministic result, a semantic decision
request, an unavailable source, or an invalid request.  Provider-specific
details stay below this contract and are never used as separate routing rules.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any


DATA_OPERATION_CONTRACT = "data_operation/v1"
DATA_ARTIFACT_CONTRACT = "data_artifact/v1"
DATA_DECISION_CONTRACT = "data_decision_required/v1"
DATA_OPERATION_STATUSES = frozenset({"completed", "decision_required", "unavailable", "invalid_request"})


def operation_status(response: dict[str, Any]) -> str:
    explicit = str(response.get("operation_status") or "")
    if explicit in DATA_OPERATION_STATUSES:
        return explicit
    provider_status = str(response.get("status") or "ok")
    if provider_status in {"data_unavailable", "unavailable"}:
        return "unavailable"
    if provider_status in {"error", "invalid", "invalid_request", "insufficient"}:
        return "invalid_request"
    if provider_status == "decision_required":
        return "decision_required"
    return "completed"


def operation_metadata(*, source_id: str, operation: str, request: dict[str, Any], response: dict[str, Any], findings: dict[str, Any], input_scope: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the one common operation envelope for every provider result."""

    status = operation_status(response)
    metadata = {
        "contract": DATA_OPERATION_CONTRACT,
        "status": status,
        "source_id": source_id,
        "operation": operation,
        "request": dict(request),
        "input_artifact_ids": list(request.get("input_artifact_ids") or []),
        "output_artifact_id": str(findings.get("dataset_handle", {}).get("artifact_id") or "") if isinstance(findings.get("dataset_handle"), dict) else "",
        "provenance": {
            "source_id": source_id,
            "operation": operation,
            "physical_inputs": _physical_inputs(request),
        },
    }
    if isinstance(input_scope, dict):
        metadata["provenance"]["input_scope"] = dict(input_scope)
    decision = findings.get("decision")
    if status == "decision_required":
        metadata["decision"] = decision if isinstance(decision, dict) else {
            "contract": DATA_DECISION_CONTRACT,
            "reason": "The provider found an ambiguity that can change the scientific meaning of the requested data relation.",
            "candidates": [],
            "evidence": [],
        }
    return metadata


def add_artifact_contract(handle: dict[str, Any], *, source_id: str, operation: str, request: dict[str, Any], record_count: int) -> dict[str, Any]:
    """Annotate every durable handle with the same artifact/provenance fields."""

    result = dict(handle)
    result.setdefault("handle_contract", "data_handle/v1")
    result["artifact_contract"] = DATA_ARTIFACT_CONTRACT
    fingerprint = sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]
    result.setdefault("artifact_id", f"data:{source_id}:{operation}:{fingerprint}")
    result["artifact_type"] = "tabular_observation"
    result["record_count"] = int(record_count)
    result["provenance"] = {
        "source_id": source_id,
        "operation": operation,
        "request": dict(request),
        "input_artifact_ids": list(request.get("input_artifact_ids") or []),
    }
    return result


def _physical_inputs(request: dict[str, Any]) -> dict[str, Any]:
    return {
        key: request[key]
        for key in ("table_id", "related_table_id", "tables", "relations", "columns", "filters", "group_by", "aggregates")
        if request.get(key) not in (None, "", [], {})
    }
