"""Bounded optional acquisition: useful when present, never a main evidence dependency."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import ToolResult
from .ledger import EvidenceLedger
from .sources import OpportunisticSensorSource


class OptionalAcquirer(Protocol):
    source_id: str
    def acquire(self, scope: dict[str, Any]) -> list[dict[str, Any]]: ...


@dataclass
class OpportunisticAcquisition:
    source: OpportunisticSensorSource | None
    result: ToolResult


class OpportunisticAcquisitionLayer:
    """Runs only on an explicit request; failure becomes auditable `data_unavailable`."""

    def acquire(self, acquirer: OptionalAcquirer, *, scope: dict[str, Any], component_id: str, ledger: EvidenceLedger) -> OpportunisticAcquisition:
        try:
            rows = acquirer.acquire(scope)
        except Exception as error:
            result = ToolResult(
                result_id=f"opportunistic.acquire:{component_id}",
                tool_id=f"opportunistic.{acquirer.source_id}",
                status="data_unavailable",
                calculation_status="not_computable",
                summary="Optional acquisition was unavailable; the primary evidence chain continues.",
                findings={"error_type": type(error).__name__},
                evidence_tags=["opportunistic_external"],
                uncertainty={},
                data_versions={},
                limitations=["Opportunity-only source cannot be promoted to a required main-chain sensor without validation."],
            )
            ledger.record_tool_call(result, [component_id], scope, parameters={"source_id": acquirer.source_id})
            return OpportunisticAcquisition(None, result)
        source = OpportunisticSensorSource(rows, source_id=acquirer.source_id, acquisition_status="ok")
        result = ToolResult(
            result_id=f"opportunistic.acquire:{component_id}",
            tool_id=f"opportunistic.{acquirer.source_id}",
            status="ok",
            calculation_status="computed",
            summary=f"Optional acquisition returned {len(rows)} rows; it remains non-authoritative until independently validated.",
            findings={"record_count": len(rows)},
            evidence_tags=["opportunistic_external"],
            uncertainty={},
            data_versions={acquirer.source_id: source.version},
            limitations=["This source is not merged into Event Clock evidence automatically."],
        )
        ledger.record_tool_call(result, [component_id], scope, parameters={"source_id": acquirer.source_id})
        return OpportunisticAcquisition(source, result)
