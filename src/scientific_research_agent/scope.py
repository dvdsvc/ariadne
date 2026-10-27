"""Bind Planner data requests to source-bound, executable observation scopes."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Protocol

from .contracts import QuestionTask, ScopeResolution
from .registry import DataRegistry, ObservationSource
from .scout import compact_profile


class ScopeAdvisor(Protocol):
    def resolve(self, *, question: str, source_contract: dict[str, Any], task_scope: dict[str, Any], patch_scope: dict[str, Any]) -> dict[str, Any]: ...


class ScopeCompiler:
    """The boundary between a planned data request and a concrete predicate.

    It may use a model to resolve observation-selection terms, but only a
    source's live query contract decides which fields/operators can execute.
    A planned claim such as ``is there migration`` is not a database filter and
    must never make an otherwise valid local query fail closed.
    """

    def __init__(self, advisor: ScopeAdvisor | None = None) -> None:
        self.advisor = advisor

    def compile(self, task: QuestionTask, sources: DataRegistry, patch_scope: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[ScopeResolution]]:
        patch = dict(patch_scope or {})
        resolutions = [self._compile_source(task, source, patch) for source in sources.all()]
        source_scopes = {resolution.source_id: resolution.canonical_scope for resolution in resolutions}
        fingerprint = hashlib.sha256(json.dumps(source_scopes, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]
        statuses = {resolution.status for resolution in resolutions}
        bundle_status = "resolved" if statuses == {"resolved"} else ("scope_unresolvable" if statuses == {"scope_unresolvable"} else "partial")
        table_scopes = {
            f"{resolution.source_id}::{resolution.canonical_scope['table_id']}": resolution.canonical_scope
            for resolution in resolutions
            if isinstance(resolution.canonical_scope, dict) and resolution.canonical_scope.get("table_id")
        }
        return {
            "_scope_contract": "scope_bundle/v2", "scope_id": f"scope:{fingerprint}",
            "status": bundle_status, "source_scopes": source_scopes, "table_scopes": table_scopes,
            "unresolved_terms": sorted({term for resolution in resolutions for term in resolution.unresolved_terms}),
            "non_filter_terms": sorted({term for resolution in resolutions for term in resolution.non_filter_terms}),
            "provenance": {"compiler": "scope_compiler", "task_scope": task.scope, "patch_scope": patch},
        }, resolutions

    def compile_one(self, task: QuestionTask, source: ObservationSource, patch_scope: dict[str, Any] | None = None, *, component: Any | None = None) -> ScopeResolution:
        """Resolve one exact source query inside a Researcher local graph."""

        if hasattr(source, "navigation_catalogue") and hasattr(source, "canonicalize_navigation"):
            catalogue = source.navigation_catalogue()
            if self.advisor is not None and hasattr(self.advisor, "navigate_table") and component is not None:
                requested_table_id = str((patch_scope or {}).get("table_id") or "")
                proposal = self.advisor.navigate_table(
                    question=task.question, component=component, source_catalogue=catalogue,
                    task_scope=task.scope, local_scope=dict(patch_scope or {}), requested_table_id=requested_table_id,
                )
                provenance = {"compiler": "researcher_dynamic_table_navigation", "question": task.question}
            else:
                canonical = source.identity_navigation({**task.scope, **dict(patch_scope or {})})
                status = canonical.get("status", "scope_unresolvable")
                requested_table_id = str((patch_scope or {}).get("table_id") or "")
                if requested_table_id and canonical.get("table_id") != requested_table_id:
                    canonical = {
                        "_scope_contract": "table_query/v1", "source_id": source.source_id,
                        "table_id": requested_table_id, "status": "scope_unresolvable", "bindings": [], "predicates": [],
                        "unresolved_terms": [f"Exact fallback selected table {canonical.get('table_id')!r}, but the operation declared {requested_table_id!r}."],
                        "non_filter_terms": [], "limit": getattr(source, "max_query_rows", 100_000),
                        "provenance": {"compiler": "exact_name_table_fallback", "error": "declared_table_mismatch"},
                    }
                    status = "scope_unresolvable"
                return ScopeResolution(
                    source_id=source.source_id, status=status, canonical_scope=canonical,
                    query_contract=source.describe_query_contract(),
                    unresolved_terms=list(canonical.get("unresolved_terms", [])),
                    non_filter_terms=list(canonical.get("non_filter_terms", [])),
                    warnings=[] if status in {"resolved", "unconstrained"} else ["Dynamic table navigation is not executable."],
                    provenance={"compiler": "exact_name_table_fallback"},
                )
            canonical = source.canonicalize_navigation(proposal)
            status = canonical.get("status", "scope_unresolvable")
            requested_table_id = str((patch_scope or {}).get("table_id") or "")
            if requested_table_id and canonical.get("table_id") != requested_table_id:
                canonical = {
                    "_scope_contract": "table_query/v1", "source_id": source.source_id,
                    "table_id": requested_table_id, "status": "scope_unresolvable", "bindings": [], "predicates": [],
                    "unresolved_terms": [f"Navigation selected table {canonical.get('table_id')!r}, but the operation declared {requested_table_id!r}."],
                    "non_filter_terms": [], "limit": getattr(source, "max_query_rows", 100_000),
                    "provenance": {"compiler": "researcher_dynamic_table_navigation", "error": "declared_table_mismatch"},
                }
                status = "scope_unresolvable"
            return ScopeResolution(
                source_id=source.source_id, status=status, canonical_scope=canonical,
                query_contract=source.describe_query_contract(),
                unresolved_terms=list(canonical.get("unresolved_terms", [])),
                non_filter_terms=list(canonical.get("non_filter_terms", [])),
                warnings=[] if status in {"resolved", "unconstrained"} else ["Dynamic table navigation is not fully executable."],
                provenance=provenance,
            )
        return self._compile_source(task, source, dict(patch_scope or {}))

    def compile_from_discovery(self, task: QuestionTask, scout: Any, patch_scope: dict[str, Any] | None = None) -> tuple[dict[str, Any], list[ScopeResolution]]:
        """Legacy compatibility for persisted tests with embedded contracts.

        Production Scout reports no longer carry exact query contracts, and
        the production Runtime resolves one live source at a time in Research.
        """

        catalogue = _DiscoveryCatalogue()
        for source_id, probe in getattr(scout, "probes", {}).items():
            if source_id.startswith("__") or not isinstance(probe, dict):
                continue
            contract = probe.get("describe_query_contract")
            descriptor = probe.get("describe_source", {})
            if not isinstance(contract, dict) or not isinstance(descriptor, dict):
                continue
            catalogue.register(_CatalogueSource(
                source_id=str(descriptor.get("source_id") or source_id),
                sensor_kind=str(descriptor.get("sensor_kind") or "opportunistic_sensor"),
                contract=contract,
            ))
        return self.compile(task, catalogue, patch_scope)

    def _compile_source(self, task: QuestionTask, source: ObservationSource, patch: dict[str, Any]) -> ScopeResolution:
        contract = source.describe_query_contract()
        explicit_scope = {**task.scope, **patch}
        if explicit_scope and not self.advisor:
            proposal = self._proposal_from_explicit_scope(
                source.source_id,
                contract,
                explicit_scope,
                source.describe_source().sensor_kind,
            )
            provenance = {"compiler": "explicit_scope_adapter", "input_scope": explicit_scope}
        elif self.advisor is not None:
            try:
                proposal = self.advisor.resolve(question=task.question, source_contract=contract, task_scope=task.scope, patch_scope=patch)
                provenance = {"compiler": "model_scope_advisor", "question": task.question}
            except Exception as error:
                proposal = {"status": "scope_unresolvable", "predicates": [], "unresolved_terms": [task.question], "research_terms_outside_query": [], "rationale": f"Scope advisor unavailable: {type(error).__name__}"}
                provenance = {"compiler": "scope_advisor_failure", "error_type": type(error).__name__}
        else:
            proposal = {"status": "unconstrained", "predicates": [], "unresolved_terms": [], "research_terms_outside_query": [], "rationale": "No explicit scope and no scope advisor are available."}
            provenance = {"compiler": "unconstrained_offline_probe"}
        return self._canonicalize(source.source_id, contract, proposal, provenance)

    @staticmethod
    def _proposal_from_explicit_scope(
        source_id: str,
        contract: dict[str, Any],
        scope: dict[str, Any],
        sensor_kind: str,
    ) -> dict[str, Any]:
        supported = {item["field"] for item in contract.get("fields", []) if isinstance(item, dict) and isinstance(item.get("field"), str)}
        predicates: list[dict[str, Any]] = []
        unresolved: list[str] = []
        interval = scope.get("age_range_ma")
        lower, upper = scope.get("min_age_ma"), scope.get("max_age_ma")
        if interval is not None and isinstance(interval, (list, tuple)) and len(interval) == 2:
            lower, upper = interval
        if lower is not None or upper is not None:
            if "age_ma" in supported and lower is not None and upper is not None:
                predicates.append({"field": "age_ma", "operator": "between", "value": [lower, upper]})
            elif sensor_kind == "event_clock_sensor":
                unresolved.append("event age interval")
        for scope_key, field in (("longitude_range", "longitude"), ("latitude_range", "latitude")):
            interval_value = scope.get(scope_key)
            if interval_value is None:
                continue
            if field in supported and isinstance(interval_value, (list, tuple)) and len(interval_value) == 2:
                predicates.append({"field": field, "operator": "between", "value": list(interval_value)})
            else:
                unresolved.append(scope_key)
        if scope.get("isotope_system") not in (None, ""):
            if "isotope_system" in supported:
                predicates.append({"field": "isotope_system", "operator": "eq", "value": scope["isotope_system"]})
            elif sensor_kind == "event_clock_sensor":
                unresolved.append(str(scope["isotope_system"]))
        locations: list[Any] = []
        for key in ("region", "regions", "province", "provinces", "basin", "basins", "locality", "localities", "city", "cities"):
            value = scope.get(key)
            locations.extend(value if isinstance(value, (list, tuple, set)) else [value])
        locations = [value for value in locations if value not in (None, "")]
        if locations:
            if "location_text" in supported:
                predicates.append({"field": "location_text", "operator": "contains_any", "value": locations})
            else:
                unresolved.extend(str(value) for value in locations)
        unknown = {key for key in scope if key not in {
            "min_age_ma", "max_age_ma", "age_range_ma", "isotope_system", "region", "regions", "province", "provinces", "basin", "basins", "locality", "localities", "city", "cities", "longitude_range", "latitude_range",
        }}
        unresolved.extend(sorted(unknown))
        return {
            "source_id": source_id,
            "status": "partial" if unresolved and predicates else ("scope_unresolvable" if unresolved else ("resolved" if predicates else "unconstrained")),
            "predicates": predicates, "unresolved_terms": unresolved,
            "research_terms_outside_query": [],
            "rationale": "Compiled from an already-structured task or local re-scout scope.",
        }

    @staticmethod
    def _canonicalize(source_id: str, contract: dict[str, Any], proposal: dict[str, Any], provenance: dict[str, Any]) -> ScopeResolution:
        fields = {item["field"]: item for item in contract.get("fields", []) if isinstance(item, dict) and isinstance(item.get("field"), str)}
        unresolved = [str(term) for term in proposal.get("unresolved_terms", []) if str(term).strip()]
        non_filter_terms = [str(term) for term in proposal.get("research_terms_outside_query", []) if str(term).strip()]
        canonical: list[dict[str, Any]] = []
        warnings: list[str] = []
        for raw in proposal.get("predicates", []):
            if not isinstance(raw, dict):
                warnings.append("Scope proposal contained a non-object predicate.")
                continue
            field, operator = raw.get("field"), raw.get("operator")
            if field not in fields or operator not in fields[field].get("operators", []):
                unresolved.append(f"unsupported predicate {field!r}/{operator!r}")
                continue
            # The strict provider schema carries native JSON values.  This
            # conversion is deliberately driven by the field/operator pair,
            # not question-specific parsing rules or a table-specific map.
            if field == "age_ma" and operator == "between":
                # ``value`` is retained only for persisted/offline callers
                # that already supplied a native list. Production model calls
                # use the strict typed minimum/maximum variant above.
                value = (
                    [raw.get("minimum_age_ma"), raw.get("maximum_age_ma")]
                    if "minimum_age_ma" in raw or "maximum_age_ma" in raw
                    else raw.get("value")
                )
            elif fields[field].get("value_type") == "number" and operator == "between":
                value = (
                    [raw.get("minimum_value"), raw.get("maximum_value")]
                    if "minimum_value" in raw or "maximum_value" in raw
                    else raw.get("value")
                )
            elif "values" in raw:
                value = raw.get("values")
            else:
                value = raw.get("value")
            valid, normalized, reason = ScopeCompiler._validate_value(fields[field], operator, value)
            if not valid:
                unresolved.append(reason)
                continue
            canonical.append({"field": field, "operator": operator, "value": normalized, "provenance": "scope_compiler"})
        declared = str(proposal.get("status", "scope_unresolvable"))
        if unresolved:
            status = "partial" if canonical else "scope_unresolvable"
        elif canonical:
            status = "resolved"
        elif declared == "unconstrained":
            status = "unconstrained"
        else:
            status = "scope_unresolvable"
            warnings.append("A constrained scope proposal contained no executable predicates.")
        canonical_scope = {
            "_scope_contract": "observation_scope/v1", "source_id": source_id,
            "status": status, "predicates": canonical,
            "unresolved_terms": sorted(dict.fromkeys(unresolved)),
            "non_filter_terms": sorted(dict.fromkeys(non_filter_terms)),
            "provenance": {**provenance, "proposal_rationale": str(proposal.get("rationale", ""))[:500]},
        }
        if status != "resolved":
            warnings.append("This source scope is not fully executable; source queries are fail-closed.")
        return ScopeResolution(
            source_id=source_id, status=status, canonical_scope=canonical_scope, query_contract=contract,
            unresolved_terms=sorted(dict.fromkeys(unresolved)), non_filter_terms=sorted(dict.fromkeys(non_filter_terms)),
            warnings=warnings, provenance=provenance,
        )

    @staticmethod
    def _validate_value(field: dict[str, Any], operator: str, value: Any) -> tuple[bool, Any, str]:
        name = field["field"]
        if operator == "between" and field.get("value_type") == "number":
            if not isinstance(value, list) or len(value) != 2:
                return False, value, f"{name} requires a two-number interval"
            try:
                return True, [float(value[0]), float(value[1])], ""
            except (TypeError, ValueError):
                return False, value, f"{name} interval contains a non-numeric value"
        if name == "location_text" and operator == "contains_any":
            values = value if isinstance(value, list) else [value]
            terms = [str(item).strip() for item in values if str(item).strip()]
            return (True, terms, "") if terms else (False, value, "location_text has no usable term")
        values = value if isinstance(value, list) else [value]
        known = {str(item).casefold() for item in field.get("known_values", [])}
        if not values or any(str(item).casefold() not in known for item in values):
            return False, value, f"{name} value is not declared by the source query contract"
        return True, values if operator == "in" else values[0], ""


class PlanScopeBinder:
    """Legacy binder retained only for old persisted runs and direct tests.

    New runs must use ``ScopeCompiler.compile_one`` from the Researcher's local
    execution graph.  The Runtime does not instantiate this class.
    """

    def __init__(self, compiler: ScopeCompiler) -> None:
        self.compiler = compiler

    def bind(self, task: QuestionTask, scout: Any, sources: DataRegistry, patch_scope: dict[str, Any] | None = None) -> Any:
        scope, resolutions = self.compiler.compile(task, sources, patch_scope)
        profiles = [compact_profile(profile) for profile in sources.profile_all(scope)]
        probes = deepcopy(scout.probes)
        for source in sources.all():
            source_id = source.source_id
            source_scope = sources.scope_for(source_id, scope)
            profile = next(item for item in profiles if item.source_id == source_id)
            probes.setdefault(source_id, {})["plan_scope_binding"] = {
                "scope_resolution": next(item.__dict__ for item in resolutions if item.source_id == source_id),
                "profile_coverage": profile.__dict__,
                "estimate_query": source.estimate_query(source_scope),
                "sample_records": profile.sample_records,
                "describe_resolution": source.describe_resolution(source_scope),
                "describe_uncertainty": source.describe_uncertainty(source_scope),
            }
        fingerprint_payload = {
            "discovery": scout.world_fingerprint, "scope": scope,
            "profiles": [profile.__dict__ for profile in profiles],
        }
        fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]
        warnings = [*scout.warnings, *[warning for resolution in resolutions for warning in resolution.warnings]]
        limits = list(scout.sensor_limits)
        for resolution in resolutions:
            if resolution.status != "resolved":
                limits.append(f"{resolution.source_id}: 计划取数范围状态为 {resolution.status}；未解析项：{', '.join(resolution.unresolved_terms) or '无'}。")
        return replace(
            scout, task_scope=scope, sources=profiles, sensor_limits=limits,
            world_fingerprint=fingerprint, warnings=warnings, probes=probes,
            scope_resolutions=resolutions,
        )

    def bind_from_discovery(self, task: QuestionTask, scout: Any, patch_scope: dict[str, Any] | None = None) -> Any:
        """Planner-only binding using Scout's read-only source contracts."""

        scope, resolutions = self.compiler.compile_from_discovery(task, scout, patch_scope)
        probes = deepcopy(scout.probes)
        for resolution in resolutions:
            probes.setdefault(resolution.source_id, {})["plan_scope_binding"] = {
                "scope_resolution": resolution.__dict__,
                "profile_coverage": "not_reprofiled_without_live_registry",
            }
        fingerprint = hashlib.sha256(json.dumps({"discovery": scout.world_fingerprint, "scope": scope}, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]
        limits = list(scout.sensor_limits)
        for resolution in resolutions:
            if resolution.status != "resolved":
                limits.append(f"{resolution.source_id}: 计划取数范围状态为 {resolution.status}；未解析项：{', '.join(resolution.unresolved_terms) or '无'}。")
        return replace(
            scout, task_scope=scope, sensor_limits=limits, world_fingerprint=fingerprint,
            probes=probes, scope_resolutions=resolutions,
        )


@dataclass
class _CatalogueSource:
    source_id: str
    sensor_kind: str
    contract: dict[str, Any]

    def describe_query_contract(self) -> dict[str, Any]:
        return self.contract

    def describe_source(self) -> Any:
        return SimpleNamespace(sensor_kind=self.sensor_kind)


class _DiscoveryCatalogue:
    def __init__(self) -> None:
        self._sources: list[_CatalogueSource] = []

    def register(self, source: _CatalogueSource) -> None:
        self._sources.append(source)

    def all(self) -> list[_CatalogueSource]:
        return list(self._sources)
