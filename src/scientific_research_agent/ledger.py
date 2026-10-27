from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from threading import RLock
from typing import Any

from .aiq_grounding import fingerprint
from .contracts import EvidenceEntry, ToolResult
from .evidence_contract import observable_manifest_from_rows


class EvidenceLedger:
    def __init__(self, *, sink: Any | None = None, run_id: str | None = None, entries: list[EvidenceEntry] | None = None) -> None:
        self.entries = entries or []
        self._counter = max((int(entry.evidence_id[1:]) for entry in self.entries if entry.evidence_id.startswith("E") and entry.evidence_id[1:].isdigit()), default=0)
        self.sink, self.run_id = sink, run_id
        self._lock = RLock()

    def _next(self) -> str:
        with self._lock:
            self._counter += 1
            return f"E{self._counter:04d}"

    def record_tool_call(self, result: ToolResult, component_ids: list[str], scope: dict[str, Any], *, parameters: dict[str, Any] | None = None, trace_id: str | None = None, upstream_evidence_ids: list[str] | None = None, execution_epoch: str = "") -> EvidenceEntry:
        kind = "forward_model" if result.tool_id.startswith("forward.") else "tool_call"
        upstream = list(upstream_evidence_ids or [])
        parameters = parameters or {}
        entry = EvidenceEntry(
            evidence_id=self._next(), kind=kind, subject_id=result.tool_id, component_ids=component_ids,
            input_scope=scope, result_summary=result.summary, findings=result.findings, evidence_tags=result.evidence_tags,
            uncertainty=result.uncertainty, data_versions=result.data_versions, calculation_status=result.calculation_status, status=result.status,
            tool_parameters=parameters, trace_id=trace_id, recorded_at=datetime.now(timezone.utc).isoformat(),
            limitations=result.limitations, warnings=result.warnings,
            upstream_evidence_ids=upstream,
            input_fingerprint=fingerprint({"scope": scope, "parameters": parameters, "upstream_evidence_ids": upstream, "data_versions": result.data_versions}),
            result_fingerprint=fingerprint({"findings": result.findings, "calculation_status": result.calculation_status, "status": result.status}),
            execution_epoch=execution_epoch,
        )
        self._append(entry)
        return entry

    def record_source_query(self, source_id: str, component_ids: list[str], scope: dict[str, Any], rows: list[dict[str, Any]], version: str, *, observables: dict[str, dict[str, Any]] | None = None, action_id: str | None = None, action_input_fingerprint: str | None = None, execution_epoch: str = "") -> EvidenceEntry:
        status = "ok" if rows else "data_unavailable"
        record_ids = [row.get("record_id") or row.get("sample_id") or row.get("event_id") for row in rows]
        tool_parameters: dict[str, Any] = {}
        if action_id:
            tool_parameters["local_action_id"] = action_id
        if action_input_fingerprint:
            tool_parameters["action_input_fingerprint"] = action_input_fingerprint
        entry = EvidenceEntry(
            evidence_id=self._next(), kind="source_query", subject_id=source_id, component_ids=component_ids,
            input_scope=scope, result_summary=f"Returned {len(rows)} normalized observation records.", findings={"record_count": len(rows), "sample": rows[:3], "observables": observables if observables is not None else observable_manifest_from_rows(rows)},
            evidence_tags=["source_query"], uncertainty={}, data_versions={source_id: version}, calculation_status="computed" if rows else "not_computable", status=status, recorded_at=datetime.now(timezone.utc).isoformat(),
            tool_parameters=tool_parameters,
            input_fingerprint=fingerprint({"scope": scope, "source_id": source_id, "version": version}),
            result_fingerprint=fingerprint({"record_count": len(rows), "record_ids": record_ids}),
            execution_epoch=execution_epoch,
        )
        self._append(entry)
        return entry

    def _append(self, entry: EvidenceEntry) -> None:
        with self._lock:
            self.entries.append(entry)
            if self.sink is not None and self.run_id is not None:
                self.sink.append_evidence(self.run_id, entry)

    def for_component(self, component_id: str) -> list[EvidenceEntry]:
        with self._lock:
            return [entry for entry in self.entries if component_id in entry.component_ids]

    def for_components(self, component_ids: set[str]) -> list[EvidenceEntry]:
        """Return original entries owned by any component in a DAG input set.

        This is intentionally a selection, not a copy or a relabelling step:
        an evidence record keeps the component that produced it and its full
        provenance while becoming visible to downstream graph nodes.
        """

        with self._lock:
            return [entry for entry in self.entries if component_ids.intersection(entry.component_ids)]

    def source_query(self, source_id: str, component_id: str, scope: dict[str, Any], version: str, *, execution_epoch: str = "") -> EvidenceEntry | None:
        """Avoid repeating an identical data acquisition within one component."""
        expected = fingerprint({"scope": scope, "source_id": source_id, "version": version})
        with self._lock:
            return next((
                entry for entry in self.entries
                if entry.kind == "source_query"
                and entry.subject_id == source_id
                and component_id in entry.component_ids
                and entry.input_fingerprint == expected
                and (not execution_epoch or entry.execution_epoch == execution_epoch)
            ), None)

    def for_action(self, action_id: str, action_input_fingerprint: str | None = None) -> list[EvidenceEntry]:
        """Return durable evidence already produced by one local action.

        This is the recovery lookup for the crash window between writing the
        Ledger and marking the action completed in the attempt table.
        """
        with self._lock:
            return [
                entry for entry in self.entries
                if entry.tool_parameters.get("local_action_id") == action_id
                and (action_input_fingerprint is None or entry.tool_parameters.get("action_input_fingerprint") == action_input_fingerprint)
            ]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(entry) for entry in self.entries]
