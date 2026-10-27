"""Registries are the agent's dynamic model of its observable world."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Callable, Literal, Protocol

from .contracts import CoverageProfile, SourceDescriptor, ToolResult
from .data_handles import enrich_materialized_handle
from .evidence_contract import DEFAULT_EVIDENCE_VOCABULARY, EvidenceVocabulary


ROW_INPUT_KINDS = frozenset({"minimum_records", "nested_collection", "minimum_distinct_values", "materialized_data_handle"})


def tool_consumes_observation_rows(spec: "ToolSpec") -> bool:
    """Whether a tool's declared contract consumes observation rows."""

    return any(
        isinstance(requirement, ToolInputRequirement) and requirement.kind in ROW_INPUT_KINDS
        for requirement in spec.input_requirements
    )


class ObservationSource(Protocol):
    source_id: str

    def describe_scout_summary(self) -> dict[str, Any]: ...
    def describe_source(self) -> SourceDescriptor: ...
    def describe_schema(self) -> dict[str, Any]: ...
    def describe_capabilities(self) -> list[str]: ...
    def describe_observables(self, scope: dict[str, Any]) -> dict[str, dict[str, Any]]: ...
    def audit_semantic_bindings(self) -> dict[str, Any]: ...
    def describe_query_contract(self) -> dict[str, Any]: ...
    def describe_discovery_coverage(self) -> dict[str, Any]: ...
    def profile_coverage(self, scope: dict[str, Any]) -> CoverageProfile: ...
    def estimate_query(self, scope: dict[str, Any]) -> dict[str, Any]: ...
    def sample_records(self, scope: dict[str, Any], n: int = 5) -> list[dict[str, Any]]: ...
    def describe_resolution(self, scope: dict[str, Any]) -> dict[str, Any]: ...
    def describe_uncertainty(self, scope: dict[str, Any]) -> dict[str, Any]: ...
    def query(self, scope: dict[str, Any]) -> list[dict[str, Any]]: ...


@dataclass
class ToolContext:
    """The shared research workspace visible to a scientific tool.

    ``component`` marks the immediate unit of work.  It is deliberately not
    the whole instruction: tools (and tool adapters that use an LLM) retain
    the unchanged user question, current plan and Scout readout so a Planner
    summary cannot silently replace a user-specified scientific condition.
    """

    question: str
    task: Any
    component: Any
    scope: dict[str, Any]
    source_registry: "DataRegistry"
    external_registry: "ExternalDataRegistry"
    prior_evidence: list[Any]
    hypothesis_ids: list[str] = field(default_factory=list)
    plan: Any | None = None
    scout: Any | None = None
    # The current local-DAG node is available to tool adapters.  Ordinary
    # numerical tools normally ignore it; data.tool uses it to read its
    # strict operation request without smuggling it through prose.
    action: Any | None = None
    # Fixture policies may still supply rows directly.  A model-backed run is
    # stricter: a numerical provider must cite the materialized dataset made
    # by an upstream data.tool node.
    require_materialized_handle: bool = False


ToolRunner = Callable[[ToolContext], ToolResult]


@dataclass
class ToolCapabilityCard:
    """Complete scientific tool documentation; selection sees only its summary."""

    purpose: str = ""
    input_objects: list[str] = field(default_factory=list)
    output_objects: list[str] = field(default_factory=list)
    reference_frame: str = ""
    usage_guidelines: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    parameter_explanation: dict[str, str] = field(default_factory=dict)
    # Method choices are part of the native tool contract, not an instruction
    # embedded in free-form JSON.  Strict model schemas read this field.
    method_ids: list[str] = field(default_factory=list)
    examples: list[dict[str, Any]] = field(default_factory=list)
    implementation: str = ""


ToolRequirementKind = Literal[
    "minimum_records",
    "nested_collection",
    "minimum_distinct_values",
    "registered_source",
    "materialized_data_handle",
    "external_capability",
    "prior_evidence_tag",
]


@dataclass(frozen=True)
class ToolInputRequirement:
    """A machine-checkable precondition for a scientific computation.

    Requirements describe data shapes and upstream evidence, not particular
    questions.  This lets a newly registered source or tool participate
    without adding aliases, keywords, or case-specific routing rules.
    """

    requirement_id: str
    kind: ToolRequirementKind
    description: str
    field_paths: tuple[str, ...] = ()
    minimum_count: int = 1
    minimum_distinct: int = 1
    capability: str = ""
    evidence_tag: str = ""
    producer_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequirementAssessment:
    requirement_id: str
    kind: str
    satisfied: bool
    observed: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class ToolReadiness:
    tool_id: str
    status: Literal["ready", "not_ready"]
    assessments: tuple[RequirementAssessment, ...]
    input_fingerprint: str

    @property
    def missing_requirement_ids(self) -> list[str]:
        return [item.requirement_id for item in self.assessments if not item.satisfied]


@dataclass
class ToolSpec:
    tool_id: str
    name: str
    capability_group: str
    description: str
    capabilities: list[str]
    input_requirements: list[ToolInputRequirement | str]
    output_tags: list[str]
    runner: ToolRunner
    cost: float = 1.0
    requires_external: bool = False
    card: ToolCapabilityCard = field(default_factory=ToolCapabilityCard)
    # Internal implementations can remain registered for reproducible legacy
    # traces without being offered as independent choices to a model.
    exposed_to_researcher: bool = True
    requires_materialized_handle: bool = False


class DataRegistry:
    def __init__(self) -> None:
        self._sources: dict[str, ObservationSource] = {}

    def register(self, source: ObservationSource) -> None:
        if source.source_id in self._sources:
            raise ValueError(f"duplicate source_id: {source.source_id}")
        self._sources[source.source_id] = source

    def get(self, source_id: str) -> ObservationSource:
        return self._sources[source_id]

    def all(self) -> list[ObservationSource]:
        return list(self._sources.values())

    def describe_all(self) -> list[SourceDescriptor]:
        return [source.describe_source() for source in self.all()]

    def profile_all(self, scope: dict[str, Any]) -> list[CoverageProfile]:
        profiles = []
        for source in self.all():
            profiles.append(source.profile_coverage(self.scope_for(source.source_id, scope)))
        return profiles

    @staticmethod
    def scope_for(source_id: str, scope: dict[str, Any], table_id: str | None = None) -> dict[str, Any]:
        """Select one source's compiled scope from a Scout scope bundle.

        A bundle lacking a source is deliberately unresolvable for that source;
        it must never degrade to an unfiltered query.
        """

        # Compatibility for direct library callers and low-level tool tests.
        # The production runtime never executes this rough scope: Researcher
        # replaces it with a source-specific local scope bundle first.
        if scope.get("_scope_contract") == "planning_scope/v1":
            requested = scope.get("requested_scope", {})
            return dict(requested) if isinstance(requested, dict) else {}
        if scope.get("_scope_contract") not in {"scope_bundle/v1", "scope_bundle/v2"}:
            return scope
        if table_id:
            table_scopes = scope.get("table_scopes", {})
            candidate = table_scopes.get(f"{source_id}::{table_id}") if isinstance(table_scopes, dict) else None
            if isinstance(candidate, dict) and candidate.get("table_id") == table_id:
                return candidate
            # A v1 checkpoint had no table_scopes map.  It is still safe to
            # recover its source-level scope only when the stored table id is
            # exactly the table requested by the operation.
            legacy = scope.get("source_scopes", {}).get(source_id) if isinstance(scope.get("source_scopes"), dict) else None
            if isinstance(legacy, dict) and legacy.get("table_id") == table_id:
                return legacy
            return {
                "_scope_contract": "table_query/v1", "source_id": source_id, "table_id": table_id,
                "status": "scope_unresolvable", "bindings": [], "predicates": [],
                "unresolved_terms": [f"No compiled scope exists for table {table_id} in source {source_id}."],
                "provenance": {"compiler": "scope_bundle_dispatch"},
            }
        source_scopes = scope.get("source_scopes", {})
        candidate = source_scopes.get(source_id) if isinstance(source_scopes, dict) else None
        if isinstance(candidate, dict):
            return candidate
        return {
            "_scope_contract": "observation_scope/v1", "source_id": source_id,
            "status": "scope_unresolvable", "predicates": [],
            "unresolved_terms": [f"No compiled scope exists for source {source_id}."],
            "provenance": {"compiler": "scope_bundle_dispatch"},
        }

    def query_all(self, scope: dict[str, Any]) -> list[dict[str, Any]]:
        event_rows: list[dict[str, Any]] = []
        spatial_constraints: dict[tuple[Any, Any], dict[str, Any]] = {}
        material_states: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
        for source in self.all():
            descriptor = source.describe_source()
            source_scope = self.scope_for(source.source_id, scope)
            source_rows = source.query(source_scope)
            if descriptor.sensor_kind == "spatial_constraint":
                for row in source_rows:
                    key = (row.get("paper_id") or row.get("source_id"), row.get("sample_id"))
                    previous = spatial_constraints.get(key)
                    if previous is None or self._position_rank(row) < self._position_rank(previous):
                        spatial_constraints[key] = row
                continue
            if descriptor.sensor_kind == "material_state_sensor":
                for row in source_rows:
                    key = (row.get("paper_id") or row.get("source_id"), row.get("sample_id"))
                    material_states.setdefault(key, []).append(row)
                continue
            if descriptor.sensor_kind != "event_clock_sensor":
                # An opportunistic source becomes part of the main chain only after
                # it has been validated and registered as an appropriate sensor.
                continue
            for row in source_rows:
                item = dict(row)
                item.setdefault("source_id", source.source_id)
                item.setdefault("data_version", descriptor.version)
                item.setdefault("sensor_kind", descriptor.sensor_kind)
                event_rows.append(item)
        for event in event_rows:
            key = (event.get("paper_id") or event.get("source_id"), event.get("sample_id"))
            constraint = spatial_constraints.get(key)
            if constraint is not None and self._position_rank(constraint) < self._position_rank(event):
                event["position"] = constraint["position"]
                event["latitude"], event["longitude"] = constraint["position"].get("latitude"), constraint["position"].get("longitude")
                event["spatial_constraint_source_id"] = constraint.get("source_id")
            if key in material_states:
                event["material_state_payload"] = material_states[key]
        return event_rows

    @staticmethod
    def _position_rank(row: dict[str, Any]) -> tuple[float, int]:
        position = row.get("position", {})
        uncertainty = position.get("uncertainty_km")
        resolution_rank = {"site": 0, "locality": 1, "county": 2, "regional": 3, "unresolved": 4}.get(position.get("resolution", "unresolved"), 5)
        return (float(uncertainty) if uncertainty is not None else float("inf"), resolution_rank)

    def capabilities(self) -> set[str]:
        return {capability for source in self.all() for capability in source.describe_capabilities()}

    def observable_terms(self) -> set[str]:
        """Direct observables actually present in the current registered world."""

        terms: set[str] = set()
        for source in self.all():
            for term, description in source.describe_observables({}).items():
                if description.get("available"):
                    terms.add(term)
        return terms


# Kept as an ergonomic compatibility name for callers that use "SourceRegistry".
SourceRegistry = DataRegistry


@dataclass
class ExternalDataset:
    dataset_id: str
    name: str
    version: str
    provider_kind: str
    capabilities: list[str]
    coverage: dict[str, Any]
    source: str | None = None
    license: str | None = None
    downloaded_at: str | None = None
    checksum: str | None = None
    local_uri: str | None = None
    available: bool = False
    uncertainty: dict[str, Any] = field(default_factory=dict)
    provider_config: dict[str, Any] = field(default_factory=dict)


class ExternalDataRegistry:
    def __init__(self) -> None:
        self._datasets: dict[str, ExternalDataset] = {}
        self._providers: dict[str, Any] = {}

    def register(self, dataset: ExternalDataset, provider: Any | None = None) -> None:
        self._datasets[dataset.dataset_id] = dataset
        if provider is not None:
            self._providers[dataset.dataset_id] = provider

    def describe_all(self) -> list[dict[str, Any]]:
        return [dataset.__dict__.copy() for dataset in self._datasets.values()]

    def available_for(self, capability: str) -> list[ExternalDataset]:
        return [dataset for dataset in self._datasets.values() if capability in dataset.capabilities and dataset.available]

    def resolve(self, capability: str) -> ExternalDataset | None:
        matches = self.available_for(capability)
        return matches[0] if matches else None

    def call(self, capability: str, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        dataset = self.resolve(capability)
        if dataset is None:
            return {"status": "data_unavailable", "required_capability": capability, "operation": operation}
        provider = self._providers.get(dataset.dataset_id)
        if provider is None:
            return {"status": "provider_unavailable", "dataset_id": dataset.dataset_id, "operation": operation}
        result = provider.call(operation, arguments)
        result.setdefault("dataset_id", dataset.dataset_id)
        result.setdefault("version", dataset.version)
        return result


class ToolRegistry:
    def __init__(self, vocabulary: EvidenceVocabulary | None = None) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self.vocabulary = vocabulary or DEFAULT_EVIDENCE_VOCABULARY

    def register(self, spec: ToolSpec) -> None:
        if spec.tool_id in self._tools:
            raise ValueError(f"duplicate tool_id: {spec.tool_id}")
        self.vocabulary.require([*spec.capabilities, *spec.output_tags], owner=f"ToolSpec {spec.tool_id}")
        if not spec.card.purpose:
            spec.card.purpose = spec.description
        self._tools[spec.tool_id] = spec

    def all(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def get(self, tool_id: str) -> ToolSpec:
        return self._tools[tool_id]

    def evidence_tags(self) -> set[str]:
        """All Ledger tags that the currently registered tools can emit.

        Tool capabilities express the scientific evidence a tool can provide;
        output tags additionally cover more specific Ledger descriptors such
        as ``centroid_shift``.  Both belong in the Planner's live contract.
        """

        return {
            tag
            for tool in self.all()
            for tag in [*tool.capabilities, *tool.output_tags]
            if isinstance(tag, str) and tag
        }

    def catalogue(self, context: ToolContext | None = None, *, exposed_only: bool = True) -> list[dict[str, Any]]:
        """Compact catalogue for selection without loading every long card."""
        catalogue = []
        for tool in self.all():
            if exposed_only and not tool.exposed_to_researcher:
                continue
            # A catalogue is a selection aid, not permission to scan every
            # registered source.  Readiness for row-consuming computations is
            # evaluated only after the Researcher has made a data tool
            # node; data.tool itself can be checked without reading rows.
            row_consuming = any(
                isinstance(requirement, ToolInputRequirement)
                and requirement.kind in {"minimum_records", "nested_collection", "minimum_distinct_values"}
                for requirement in tool.input_requirements
            )
            readiness = self.assess_readiness(tool, context) if context is not None and not row_consuming else None
            item = {
                "tool_id": tool.tool_id,
                "namespace": tool.capability_group,
                "purpose": tool.card.purpose,
                "input_objects": tool.card.input_objects,
                "output_objects": tool.card.output_objects,
                "reference_frame": tool.card.reference_frame,
                "provides": sorted(set(tool.capabilities).union(tool.output_tags)),
                "method_ids": list(tool.card.method_ids),
                "requires_materialized_handle": tool.requires_materialized_handle,
                "required_input_ids": [requirement.requirement_id if isinstance(requirement, ToolInputRequirement) else requirement for requirement in tool.input_requirements],
            }
            if readiness is not None:
                item["readiness"] = {
                    "status": readiness.status,
                    "missing_requirement_ids": readiness.missing_requirement_ids,
                    "input_fingerprint": readiness.input_fingerprint,
                }
            elif context is not None and row_consuming:
                item["readiness"] = {
                    "status": "deferred",
                    "missing_requirement_ids": [requirement.requirement_id for requirement in tool.input_requirements if isinstance(requirement, ToolInputRequirement)],
                    "input_fingerprint": "",
                }
            catalogue.append(item)
        return catalogue

    def capability_card(self, tool_id: str, context: ToolContext | None = None) -> dict[str, Any]:
        """Complete card used only to review a selected tool before execution."""
        tool = self.get(tool_id)
        card = {
            "tool_id": tool.tool_id,
            "name": tool.name,
            "purpose": tool.card.purpose,
            "input_objects": tool.card.input_objects,
            "output_objects": tool.card.output_objects,
            "reference_frame": tool.card.reference_frame,
            "usage_guidelines": tool.card.usage_guidelines,
            "limitations": tool.card.limitations,
            "parameter_explanation": tool.card.parameter_explanation,
            "method_ids": list(tool.card.method_ids),
            "validated_examples": tool.card.examples,
            "implementation": tool.card.implementation,
            "evidence_tags": tool.output_tags,
            "input_contract": [self._requirement_dict(requirement) for requirement in tool.input_requirements],
        }
        if context is not None:
            card["readiness"] = self._readiness_dict(self.assess_readiness(tool, context))
        return card

    def validate_result(self, spec: ToolSpec, result: ToolResult) -> ToolResult:
        """Reject a tool result that lies about its registered Ledger contract."""

        if result.tool_id != spec.tool_id:
            raise ValueError(f"Tool runner for {spec.tool_id} returned result for {result.tool_id}")
        self.vocabulary.require(result.evidence_tags, owner=f"ToolResult {spec.tool_id}")
        allowed = set(spec.capabilities).union(spec.output_tags)
        unknown = sorted(set(result.evidence_tags) - allowed)
        if unknown:
            raise ValueError(f"Tool {spec.tool_id} emitted unregistered evidence tags: {unknown}")
        if result.status in {"not_applicable", "data_unavailable", "error"} and result.evidence_tags:
            raise ValueError(f"Tool {spec.tool_id} emitted scientific evidence tags while status={result.status}")
        return result

    def select(self, evidence_needed: list[str], available_capabilities: set[str], context: ToolContext | None = None, *, exposed_only: bool = False) -> list[ToolSpec]:
        needs = set(evidence_needed)
        candidates: list[tuple[int, float, ToolSpec]] = []
        for tool in self.all():
            if exposed_only and not tool.exposed_to_researcher:
                continue
            # A component may require either a scientific capability or a
            # more specific Ledger output tag.  Both are declared by the
            # registered tool; do not fall back to every tool merely because
            # the Planner selected the precise output name.
            provided_tags = set(tool.capabilities).union(tool.output_tags)
            match = len(needs.intersection(provided_tags))
            legacy_requirements = {item for item in tool.input_requirements if isinstance(item, str)}
            requirements_met = legacy_requirements.issubset(available_capabilities | needs)
            if context is not None and requirements_met:
                requirements_met = self.assess_readiness(tool, context).status == "ready"
            if match and requirements_met:
                candidates.append((match, tool.cost, tool))
        candidates.sort(key=lambda item: (-item[0], item[1], item[2].tool_id))
        return [item[2] for item in candidates]

    def capability_coverage(self, evidence_needed: list[str], context: ToolContext) -> dict[str, Any]:
        """Separate absent methods from methods whose current inputs are not ready."""

        coverage: dict[str, Any] = {}
        for need in dict.fromkeys(evidence_needed):
            providers = [tool for tool in self.all() if need in set(tool.capabilities).union(tool.output_tags)]
            # Do not materialise the entire observation world merely to print
            # a tool catalogue.  Each provider evaluates only the data shape
            # it actually requires; data.tool itself has no row-read
            # prerequisite and can safely be the first local DAG node.
            readiness = [self.assess_readiness(tool, context) for tool in providers]
            coverage[need] = {
                "status": "capability_absent" if not providers else ("ready" if any(item.status == "ready" for item in readiness) else "input_not_ready"),
                "providers": [tool.tool_id for tool in providers],
                "provider_readiness": [self._readiness_dict(item) for item in readiness],
            }
        return coverage

    def assess_readiness(self, spec: ToolSpec, context: ToolContext | None, *, rows: list[dict[str, Any]] | None = None) -> ToolReadiness:
        if context is None:
            assessments = tuple(
                RequirementAssessment(
                    requirement.requirement_id if isinstance(requirement, ToolInputRequirement) else str(requirement),
                    requirement.kind if isinstance(requirement, ToolInputRequirement) else "legacy_source_capability",
                    False, {}, "A ToolContext is required to evaluate this input contract.",
                )
                for requirement in spec.input_requirements
            )
            return ToolReadiness(spec.tool_id, "not_ready", assessments, "context-unavailable")
        requires_handle = context.require_materialized_handle and tool_consumes_observation_rows(spec)
        requires_rows = any(
            not isinstance(requirement, ToolInputRequirement)
            or requirement.kind in {"minimum_records", "nested_collection", "minimum_distinct_values"}
            for requirement in spec.input_requirements
        )
        access_error = ""
        if rows is None and requires_handle:
            action = context.action
            evidence_id = str(getattr(action, "data_handle_evidence_id", "") or "")
            try:
                entry = next(item for item in context.prior_evidence if getattr(item, "evidence_id", None) == evidence_id)
                handle = getattr(entry, "findings", {}).get("dataset_handle")
                if not isinstance(handle, dict) or handle.get("handle_contract") != "data_handle/v1":
                    raise ValueError("referenced evidence is not a materialized data handle")
                handle = enrich_materialized_handle(handle, entry, str(handle.get("source_id") or ""))
                source = context.source_registry.get(str(handle.get("source_id") or ""))
                if not hasattr(source, "load_workspace_handle"):
                    raise ValueError("source does not support materialized handles")
                rows = source.load_workspace_handle(handle)
            except Exception as error:
                rows = []
                access_error = f"Materialized data handle is unavailable: {type(error).__name__}: {error}"
        elif rows is None and requires_rows:
            try:
                rows = context.source_registry.query_all(context.scope)
            except Exception as error:
                rows = []
                access_error = f"Data access is not ready for this computation: {type(error).__name__}: {error}"
        else:
            rows = rows or []
        assessments = tuple(self._assess_requirement(requirement, context, rows) for requirement in spec.input_requirements)
        if access_error:
            assessments = tuple(
                RequirementAssessment(item.requirement_id, item.kind, False, item.observed, access_error)
                if item.kind in {"minimum_records", "nested_collection", "minimum_distinct_values", "materialized_data_handle"} else item
                for item in assessments
            )
        fingerprint_payload = {
            "tool_id": spec.tool_id,
            "scope": context.scope,
            "requirements": [self._assessment_dict(item) for item in assessments],
            "data_versions": sorted(
                (source.describe_source().source_id, source.describe_source().version)
                for source in context.source_registry.all()
            ),
        }
        fingerprint = sha256(json.dumps(fingerprint_payload, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
        return ToolReadiness(spec.tool_id, "ready" if all(item.satisfied for item in assessments) else "not_ready", assessments, fingerprint)

    def _assess_requirement(self, requirement: ToolInputRequirement | str, context: ToolContext, rows: list[dict[str, Any]]) -> RequirementAssessment:
        if isinstance(requirement, str):
            satisfied = requirement in context.source_registry.capabilities()
            return RequirementAssessment(requirement, "legacy_source_capability", satisfied, {"available_source_capabilities": sorted(context.source_registry.capabilities())}, "Legacy source capability requirement.")
        if requirement.kind == "registered_source":
            source_ids = [source.source_id for source in context.source_registry.all()]
            return RequirementAssessment(requirement.requirement_id, requirement.kind, bool(source_ids), {"source_ids": source_ids}, requirement.description)
        if requirement.kind == "materialized_data_handle":
            evidence_id = str(getattr(context.action, "data_handle_evidence_id", "") or "")
            # Deterministic fixtures and direct local tool audits are allowed
            # to supply rows through the source contract.  A model-backed
            # Researcher sets require_materialized_handle=True, making the
            # durable data.tool handle mandatory there.
            if not context.require_materialized_handle:
                return RequirementAssessment(
                    requirement.requirement_id, requirement.kind, bool(rows),
                    {"evidence_id": evidence_id or None, "loaded_record_count": len(rows), "access_mode": "source_query_fixture"},
                    requirement.description,
                )
            return RequirementAssessment(
                requirement.requirement_id, requirement.kind, bool(evidence_id),
                {"evidence_id": evidence_id or None, "loaded_record_count": len(rows)}, requirement.description,
            )
        if requirement.kind == "minimum_records":
            matching = [row for row in rows if all(self._path_has_value(row, path) for path in requirement.field_paths)]
            minimum = requirement.minimum_count
            satisfied = len(matching) >= minimum
            return RequirementAssessment(requirement.requirement_id, requirement.kind, satisfied, {"matching_records": len(matching), "minimum_records": minimum, "field_paths": list(requirement.field_paths)}, requirement.description)
        if requirement.kind == "nested_collection":
            count = sum(len(value) for row in rows for path in requirement.field_paths for value in self._path_values(row, path) if isinstance(value, list))
            satisfied = count >= requirement.minimum_count
            return RequirementAssessment(requirement.requirement_id, requirement.kind, satisfied, {"nested_items": count, "minimum_items": requirement.minimum_count, "field_paths": list(requirement.field_paths)}, requirement.description)
        if requirement.kind == "minimum_distinct_values":
            values = {
                json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
                for row in rows for path in requirement.field_paths for value in self._path_values(row, path)
                if value not in (None, "", [], {})
            }
            satisfied = len(values) >= requirement.minimum_distinct
            return RequirementAssessment(requirement.requirement_id, requirement.kind, satisfied, {"distinct_values": len(values), "minimum_distinct": requirement.minimum_distinct, "field_paths": list(requirement.field_paths)}, requirement.description)
        if requirement.kind == "external_capability":
            dataset = context.external_registry.resolve(requirement.capability)
            satisfied = dataset is not None
            return RequirementAssessment(requirement.requirement_id, requirement.kind, satisfied, {"capability": requirement.capability, "dataset_id": dataset.dataset_id if dataset else None, "version": dataset.version if dataset else None}, requirement.description)
        if requirement.kind == "prior_evidence_tag":
            matching = [
                entry.evidence_id for entry in context.prior_evidence
                if requirement.evidence_tag in getattr(entry, "evidence_tags", [])
                and getattr(entry, "status", None) == "ok"
                and (not requirement.producer_ids or getattr(entry, "subject_id", None) in requirement.producer_ids)
            ]
            satisfied = bool(matching)
            return RequirementAssessment(requirement.requirement_id, requirement.kind, satisfied, {"evidence_tag": requirement.evidence_tag, "producer_ids": list(requirement.producer_ids), "matching_evidence_ids": matching}, requirement.description)
        raise ValueError(f"Unsupported tool requirement kind: {requirement.kind}")

    @classmethod
    def _path_has_value(cls, record: dict[str, Any], path: str) -> bool:
        return any(value not in (None, "", [], {}) for value in cls._path_values(record, path))

    @classmethod
    def _path_values(cls, value: Any, path: str) -> list[Any]:
        parts = path.split(".") if path else []
        current = [value]
        for part in parts:
            following: list[Any] = []
            for item in current:
                if isinstance(item, dict) and part in item:
                    following.append(item[part])
                elif isinstance(item, list):
                    for child in item:
                        if isinstance(child, dict) and part in child:
                            following.append(child[part])
            current = following
        return current

    @staticmethod
    def _requirement_dict(requirement: ToolInputRequirement | str) -> dict[str, Any]:
        if isinstance(requirement, str):
            return {"requirement_id": requirement, "kind": "legacy_source_capability", "description": requirement}
        return {
            "requirement_id": requirement.requirement_id, "kind": requirement.kind,
            "description": requirement.description, "field_paths": list(requirement.field_paths),
            "minimum_count": requirement.minimum_count, "minimum_distinct": requirement.minimum_distinct,
            "capability": requirement.capability, "evidence_tag": requirement.evidence_tag,
            "producer_ids": list(requirement.producer_ids),
        }

    @staticmethod
    def _assessment_dict(assessment: RequirementAssessment) -> dict[str, Any]:
        return {"requirement_id": assessment.requirement_id, "kind": assessment.kind, "satisfied": assessment.satisfied, "observed": assessment.observed, "reason": assessment.reason}

    @classmethod
    def _readiness_dict(cls, readiness: ToolReadiness) -> dict[str, Any]:
        return {
            "status": readiness.status, "input_fingerprint": readiness.input_fingerprint,
            "missing_requirement_ids": readiness.missing_requirement_ids,
            "assessments": [cls._assessment_dict(item) for item in readiness.assessments],
        }
