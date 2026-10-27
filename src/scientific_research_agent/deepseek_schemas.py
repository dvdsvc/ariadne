"""Static schemas submitted to DeepSeek strict tool mode.

These are API contracts, not Pydantic-derived post-hoc validators.  Dynamic
maps are represented at the provider boundary as JSON strings and decoded by
the adapters below; DeepSeek strict mode requires every object property to be
known, required, and closed.
"""

from __future__ import annotations

from typing import Any



def _closed(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


_STR = {"type": "string"}
_NON_EMPTY_STR = {"type": "string", "minLength": 1, "pattern": r"\S"}
_NUM = {"type": "number"}
_STRINGS = {"type": "array", "items": _STR}
_NON_EMPTY_STRINGS = {"type": "array", "items": _NON_EMPTY_STR}
_SENSOR_KIND = {"type": "string", "enum": ["event_clock_sensor", "spatial_constraint", "material_state_sensor", "external_boundary_condition", "forward_physical_model", "opportunistic_sensor"]}
_SENSOR_KINDS = {"type": "array", "items": _SENSOR_KIND}

_HYPOTHESIS = _closed({"hypothesis_id": _STR, "label": _STR, "description": _STR, "predictions": _STRINGS, "differentiating_observables": _STRINGS})


def research_brief_schema() -> dict[str, Any]:
    """Schema for the explicit Interpretation node output.

    The two scope objects remain semantic maps; they are not source fields or
    query predicates and are intentionally not enumerated by the schema.
    """

    return _closed({
        "research_object": _NON_EMPTY_STR,
        "spatial_scope": {"type": "object"},
        "temporal_scope": {"type": "object"},
        "core_phenomenon_or_relation": _NON_EMPTY_STR,
        "must_answer": _NON_EMPTY_STRINGS,
        "key_assumptions_and_unresolved_ambiguities": _STRINGS,
        "conclusion_level": _NON_EMPTY_STR,
    })


def planner_schema(*, scope_ids: list[str], evidence_vocabulary: list[str] | None = None, **_ignored_legacy_arguments: Any) -> dict[str, Any]:
    """Planner schema for scientific decisions, not an execution pipeline.

    Scope identifiers preserve the user's coarse boundary. Evidence
    requirements remain open strings: the Planner states what would falsify or
    resolve a claim, while the Researcher later chooses among live registered
    capabilities or reports that the requirement is unavailable.
    """

    values = list(dict.fromkeys(item for item in scope_ids if item))
    if not values:
        raise ValueError("planner schema requires at least one Scout scope id")
    scope_id = {"type": "string", "enum": values}
    component = _closed({
        "component_id": _STR, "claim": _STR, "depends_on": _STRINGS,
        "evidence_needed": _NON_EMPTY_STRINGS, "constraints_json": _STR,
        "failure_meaning": _STR, "scope_id": scope_id,
        "candidate_hypothesis_ids": _STRINGS, "skill_ids": _STRINGS,
    })
    return _closed({
        "primary_component": component,
        "additional_components": {"type": "array", "items": component},
        "candidate_hypotheses": {"type": "array", "items": _HYPOTHESIS},
    })


def scope_resolution_schema(source_contract: dict[str, Any]) -> dict[str, Any]:
    """Strict, typed scope-resolution output derived from a live source contract.

    Predicate values must remain JSON values at the provider boundary.  Earlier
    versions placed them in a ``value_json`` string, so strict tool mode could
    validate the envelope but not an age interval inside it.  This builder is
    source-generic: a source declares its fields/operators/value types and the
    resulting strict schema makes invalid query shapes impossible to emit.
    """

    source_id = str(source_contract.get("source_id") or "")
    fields = [item for item in source_contract.get("fields", []) if isinstance(item, dict) and isinstance(item.get("field"), str)]
    variants: list[dict[str, Any]] = []
    for item in fields:
        field = item["field"]
        for operator in item.get("operators", []):
            if isinstance(operator, str):
                variants.append(_typed_scope_predicate(item, operator))
    predicate_items: dict[str, Any] = {"anyOf": variants} if variants else _closed({"field": _STR, "operator": _STR, "value": _STR})
    return _closed({
        "source_id": {"type": "string", "enum": [source_id]},
        "status": {"type": "string", "enum": ["resolved", "partial", "scope_unresolvable", "unconstrained"]},
        "predicates": {"type": "array", "items": predicate_items},
        "unresolved_terms": _STRINGS,
        "research_terms_outside_query": _STRINGS,
        "rationale": _STR,
    })


def table_navigation_schema(catalogue: dict[str, Any], *, canonical_fields: list[str], requested_table_id: str | None = None) -> dict[str, Any]:
    """Strict per-run schema built from real table and column identifiers."""

    tables = [item for item in catalogue.get("tables", []) if isinstance(item, dict) and isinstance(item.get("table_id"), str)]
    if requested_table_id:
        tables = [item for item in tables if item.get("table_id") == requested_table_id]
        if not tables:
            raise ValueError(f"Requested table is absent from the live catalogue: {requested_table_id}")
    table_ids = [item["table_id"] for item in tables]
    columns = list(dict.fromkeys(
        column["column_id"]
        for table in tables
        for column in table.get("columns", [])
        if isinstance(column, dict) and isinstance(column.get("column_id"), str)
    ))
    if not table_ids or not columns or not canonical_fields:
        raise ValueError("table navigation schema requires live tables, columns, and canonical fields")
    binding = _closed({
        "canonical_field": {"type": "string", "enum": list(dict.fromkeys(canonical_fields))},
        "source_column": {"type": "string", "enum": columns},
    })
    variants: list[dict[str, Any]] = []
    profiles = {
        column["column_id"]: column
        for table in tables for column in table.get("columns", [])
        if isinstance(column, dict) and isinstance(column.get("column_id"), str)
    }
    for column_id, profile in profiles.items():
        scalar = _NUM if profile.get("inferred_type") == "number" else _STR
        base = {"source_column": {"type": "string", "enum": [column_id]}}
        variants.extend([
            _closed({**base, "operator": {"type": "string", "enum": ["eq"]}, "value": scalar}),
            _closed({**base, "operator": {"type": "string", "enum": ["in"]}, "values": {"type": "array", "items": scalar}}),
        ])
        if profile.get("inferred_type") == "number":
            variants.append(_closed({**base, "operator": {"type": "string", "enum": ["between"]}, "minimum_value": _NUM, "maximum_value": _NUM}))
        else:
            variants.append(_closed({**base, "operator": {"type": "string", "enum": ["contains"]}, "value": _STR}))
    return _closed({
        "status": {"type": "string", "enum": ["resolved", "partial", "scope_unresolvable", "unconstrained"]},
        "table_id": {"type": "string", "enum": table_ids},
        "bindings": {"type": "array", "items": binding},
        "predicates": {"type": "array", "items": {"anyOf": variants}},
        "unresolved_terms": _STRINGS,
        "research_terms_outside_query": _STRINGS,
        "rationale": _STR,
        "limit": {"type": "integer", "minimum": 1},
    })


def _typed_scope_predicate(field: dict[str, Any], operator: str) -> dict[str, Any]:
    """One closed variant with a native JSON value shape for one predicate."""

    name = field["field"]
    base = {
        "field": {"type": "string", "enum": [name]},
        "operator": {"type": "string", "enum": [operator]},
    }
    if name == "age_ma" and operator == "between":
        return _closed({**base, "minimum_age_ma": _NUM, "maximum_age_ma": _NUM})
    if field.get("value_type") == "number" and operator == "between":
        return _closed({**base, "minimum_value": _NUM, "maximum_value": _NUM})
    scalar = _typed_scope_scalar(field)
    if name == "location_text" and operator == "contains_any":
        return _closed({**base, "values": {"type": "array", "items": _STR}})
    if operator == "in":
        return _closed({**base, "values": {"type": "array", "items": scalar}})
    return _closed({**base, "value": scalar})


def _typed_scope_scalar(field: dict[str, Any]) -> dict[str, Any]:
    value_type = field.get("value_type")
    if value_type == "number":
        return _NUM
    known_values = [item for item in field.get("known_values", []) if isinstance(item, str) and item]
    if known_values:
        return {"type": "string", "enum": list(dict.fromkeys(known_values))}
    return _STR

def research_action_schema(
    *, tool_ids: list[str], component_ids: list[str], source_ids: list[str], workspace_catalogue: list[dict[str, Any]] | None = None,
    tool_methods: dict[str, list[str]] | None = None,
    data_handle_evidence_ids: list[str] | None = None, tool_requires_handle: dict[str, bool] | None = None,
    local_action_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build the strict action schema from the *live* runtime registries.

    DeepSeek's ``strict`` flag validates the static JSON shape supplied with a
    request.  Registry membership is dynamic, so using a plain ``string`` for
    a tool or component identifier would still let the model name something
    that cannot execute.  The empty value is deliberate: strict schemas cannot
    express the conditional ``tool_id is empty unless kind == tool`` rule, and
    the normal runtime validator continues to enforce that relation.
    """

    component_values = list(dict.fromkeys(item for item in component_ids if item))
    if not component_values:
        raise ValueError("research action schema requires at least one live component id")
    registered_tools = list(dict.fromkeys(item for item in tool_ids if item))
    registered_sources = list(dict.fromkeys(item for item in source_ids if item))
    # A Researcher may extend only its current component-scoped execution DAG.
    # This enum is deliberately bound to the already existing local nodes;
    # global component ids and guessed action labels must not cross this API.
    #
    # Do not add an empty-string sentinel here.  ``depends_on_action_ids`` is
    # an array, so its valid no-predecessor value is simply ``[]``.  Empty
    # sentinels are appropriate only for required scalar properties such as
    # ``tool_id`` in a non-tool action.  Putting ``""`` in this array makes a
    # fake dependency look API-valid and lets it reach the graph validator.
    local_dependency_values = list(dict.fromkeys(item for item in (local_action_ids or []) if item))

    def local_dependency_array() -> dict[str, Any]:
        if local_dependency_values:
            return {"type": "array", "items": {"type": "string", "enum": local_dependency_values}}
        # DeepSeek strict mode does not provide a portable way to express the
        # empty-domain array while keeping the property required.  At this
        # point no local action id exists, so the only semantically valid value
        # is []; the runtime validator remains authoritative if the model emits
        # a non-empty array.  Crucially, the schema never advertises a fake id.
        return {"type": "array", "items": {"type": "string"}}

    workspace_sources = list(workspace_catalogue or [])
    workspace_fields = sorted({
        str(field)
        for source in workspace_sources if isinstance(source, dict)
        for table in source.get("tables", []) if isinstance(table, dict)
        for field in table.get("fields", []) if isinstance(field, str) and field
    })
    workspace_tables = sorted({
        str(table.get("table_id"))
        for source in workspace_sources if isinstance(source, dict)
        for table in source.get("tables", []) if isinstance(table, dict) and isinstance(table.get("table_id"), str)
    })
    workspace_filter_operators = sorted({
        str(operator)
        for source in workspace_sources if isinstance(source, dict)
        for operator in source.get("filter_operators", [])
        if isinstance(operator, str) and operator
    }) or ["eq", "in", "between", "contains"]
    workspace_aggregate_operations = sorted({
        str(operation)
        for source in workspace_sources if isinstance(source, dict)
        for operation in source.get("aggregate_operations", [])
        if isinstance(operation, str) and operation
    }) or ["count"]
    workspace_operations = sorted({
        str(operation)
        for source in workspace_sources if isinstance(source, dict)
        for operation in source.get("operations", [])
        if isinstance(operation, str) and operation
    }) or ["inspect", "profile", "sample", "query", "aggregate", "materialize"]

    handle_ids = list(dict.fromkeys(["", *(item for item in (data_handle_evidence_ids or []) if item)]))

    def action_variant(kind: str, *, accepted_tools: list[str], accepted_sources: list[str], method_ids: list[str] | None = None, requires_handle: bool = False) -> dict[str, Any]:
        return _closed({
            "action_id": _STR,
            "component_id": {"type": "string", "enum": component_values},
            "kind": {"type": "string", "enum": [kind]},
            "tool_id": {"type": "string", "enum": accepted_tools},
            "source_id": {"type": "string", "enum": accepted_sources},
            "method": {"type": "string", "enum": method_ids or [""]},
            "data_handle_evidence_id": {"type": "string", "enum": [item for item in handle_ids if item] if requires_handle else [""]},
            "input_evidence_ids": {"type": "array", "items": {"type": "string"}},
            "input_scope_json": _STR,
            "parameters_json": _STR,
            "rationale": _STR,
            "expected_evidence_tags": _STRINGS,
            "depends_on_action_ids": local_dependency_array(),
            "local_node_label": _STR,
        })

    def workspace_action_variant() -> dict[str, Any]:
        if not workspace_sources or not workspace_fields:
            raise ValueError("data tool schema requires live source tables and fields")
        field = {"type": "string", "enum": workspace_fields}
        predicate = _closed({
            "field": field,
            "operator": {"type": "string", "enum": workspace_filter_operators},
            "text_values": _STRINGS,
            "minimum_number": _NUM,
            "maximum_number": _NUM,
        })
        aggregate = _closed({
            "field": {"anyOf": [field, {"type": "string", "enum": [""]}]},
            "operation": {"type": "string", "enum": workspace_aggregate_operations},
            "as": _STR,
        })
        relation_table = _closed({
            "table_id": {"type": "string", "enum": workspace_tables},
            "alias": _STR,
        })
        relation_edge = _closed({
            "left_table_id": {"type": "string", "enum": workspace_tables},
            "left_key": field,
            "right_table_id": {"type": "string", "enum": workspace_tables},
            "right_key": field,
            "join_type": {"type": "string", "enum": ["inner", "left"]},
            # The Researcher must state the semantic matching rule.  The
            # executor never silently upgrades an exact relation to fuzzy or
            # spatial/temporal matching.
            "match_policy": {"type": "string", "enum": ["exact", "normalized", "fuzzy", "spatial", "temporal"]},
            "match_tolerance": _NUM,
        })
        request_properties = {
            "operation": {"type": "string", "enum": workspace_operations},
            "table_id": {"type": "string", "enum": workspace_tables},
            "related_table_id": {"type": "string", "enum": ["", *workspace_tables]},
            "left_key": {"type": "string", "enum": ["", *workspace_fields]},
            "right_key": {"type": "string", "enum": ["", *workspace_fields]},
            "join_type": {"type": "string", "enum": ["", "inner", "left", "semi", "anti"]},
            "relation_mode": {"type": "string", "enum": ["", "probe", "materialize"]},
            "semantic_authorization": {"type": "string", "enum": ["", "explicit"]},
            "match_policy": {"type": "string", "enum": ["", "exact", "normalized", "fuzzy", "spatial", "temporal"]},
            "match_tolerance": _NUM,
            "tables": {"type": "array", "items": relation_table},
            "relations": {"type": "array", "items": relation_edge},
            "columns": {"type": "array", "items": field},
            "filters": {"type": "array", "items": predicate},
            "group_by": {"type": "array", "items": field},
            "aggregates": {"type": "array", "items": aggregate},
            "bin_field": {"anyOf": [field, {"type": "string", "enum": [""]}]},
            "bin_width": _NUM,
            "bin_origin": _NUM,
            "row_limit": {"type": "integer", "minimum": 1},
            "dataset_name": _STR,
        }
        # A relation is a different input object from a plain table
        # operation.  Encode that distinction in the API schema so a model
        # cannot legally send relation edges with operation=materialize and
        # rely on a post-hoc repair.  Empty arrays remain valid for the
        # non-relation variant because DeepSeek strict mode does not provide a
        # portable minItems/maxItems contract.
        simple = dict(request_properties)
        simple["operation"] = {"type": "string", "enum": [item for item in workspace_operations if item != "relate"]}
        simple["related_table_id"] = {"type": "string", "enum": [""]}
        simple["left_key"] = {"type": "string", "enum": [""]}
        simple["right_key"] = {"type": "string", "enum": [""]}
        simple["join_type"] = {"type": "string", "enum": [""]}
        simple["relation_mode"] = {"type": "string", "enum": [""]}
        simple["semantic_authorization"] = {"type": "string", "enum": [""]}
        simple["match_policy"] = {"type": "string", "enum": [""]}
        simple["match_tolerance"] = _NUM
        simple["tables"] = {"type": "array", "items": {"type": "string", "enum": [""]}}
        simple["relations"] = {"type": "array", "items": {"type": "string", "enum": [""]}}
        relation = dict(request_properties)
        relation["operation"] = {"type": "string", "enum": ["relate"]}
        relation["table_id"] = {"type": "string", "enum": ["", *workspace_tables]}
        relation["relation_mode"] = {"type": "string", "enum": ["probe", "materialize"]}
        relation["semantic_authorization"] = {"type": "string", "enum": ["explicit"]}
        relation["match_policy"] = {"type": "string", "enum": ["exact", "normalized", "fuzzy", "spatial", "temporal"]}
        relation["match_tolerance"] = _NUM
        relation["tables"] = {"type": "array", "items": relation_table}
        relation["relations"] = {"type": "array", "items": relation_edge}
        request = {"anyOf": [_closed(simple), _closed(relation)]} if "relate" in workspace_operations else _closed(simple)
        return _closed({
            "action_id": _STR,
            "component_id": {"type": "string", "enum": component_values},
            "kind": {"type": "string", "enum": ["data_tool"]},
            "tool_id": {"type": "string", "enum": ["data.tool"]},
            "source_id": {"type": "string", "enum": registered_sources},
            "method": {"type": "string", "enum": [""]},
            "data_handle_evidence_id": {"type": "string", "enum": [""]},
            "input_evidence_ids": {"type": "array", "items": {"type": "string"}},
            "input_scope_json": _STR,
            "parameters_json": _STR,
            "data_request": request,
            "rationale": _STR,
            "expected_evidence_tags": _STRINGS,
            "depends_on_action_ids": local_dependency_array(),
            "local_node_label": _STR,
        })

    # DeepSeek strict mode supports anyOf.  Use it to make the action kind and
    # the permitted identifiers one contract, rather than allowing a
    # finish_component action to carry a tool id and rejecting it later.
    variants = [
        action_variant(kind, accepted_tools=[""], accepted_sources=[""], method_ids=[""])
        for kind in ("local_rescout", "finish_component")
    ]
    if registered_tools:
        variants[0:0] = [
            action_variant(
                "tool", accepted_tools=[tool_id], accepted_sources=["", *registered_sources],
                method_ids=list((tool_methods or {}).get(tool_id) or [""]),
                requires_handle=bool((tool_requires_handle or {}).get(tool_id)),
            )
            for tool_id in registered_tools
            if tool_id != "data.tool" and (not bool((tool_requires_handle or {}).get(tool_id)) or any(handle_ids))
        ]
    if "data.tool" in registered_tools and registered_sources and workspace_sources and workspace_fields:
        variants.insert(1 if registered_tools else 0, workspace_action_variant())
    # Once the generic relational workspace is live, data access must travel
    # through that typed tool. Keep the old source_query variant only for
    # compatibility schemas that do not expose data.tool.
    if registered_sources and "data.tool" not in registered_tools:
        variants.insert(1 if registered_tools else 0, action_variant("source_query", accepted_tools=[""], accepted_sources=registered_sources, method_ids=[""]))
    # A local Researcher must either advance its graph or explicitly emit a
    # finish_component node.  DeepSeek cannot enforce minItems, therefore the
    # first action is an object and only later actions form an array.
    action = {"anyOf": variants}
    return _closed({"primary_action": action, "additional_actions": {"type": "array", "items": action}})


def tool_readiness_schema(*, action_ids: list[str]) -> dict[str, Any]:
    values = list(dict.fromkeys(item for item in action_ids if item))
    if not values:
        raise ValueError("tool readiness schema requires at least one selected action")
    return _closed({"accepted_action_ids": {"type": "array", "items": {"type": "string", "enum": values}}, "route_note": _STR})

_DIAGNOSIS = _closed({"kind": {"type": "string", "enum": ["calculation_missing", "wrong_scope_or_data_assumption", "missing_external_sensor", "missing_tool_capability", "tool_input_unavailable", "weak_optional_source", "fundamentally_unidentifiable", "resolution_limit"]}, "reason": _STR, "missing_observables": _STRINGS, "suggested_sensor_kinds": _SENSOR_KINDS, "expected_information_gain": {"type": "string", "enum": ["low", "medium", "high"]}, "patch_scope_json": _STR})
def planner_review_schema(*, component_ids: list[str], evidence_vocabulary: list[str] | None = None, scope_ids: list[str], **_ignored_legacy_arguments: Any) -> dict[str, Any]:
    """Strict global-map patch contract for the top-level Planner.

    It deliberately has no tool, query, or local-action field.  A Planner may
    add/supersede/reconnect scientific components, reopen a component under a
    new contract, or request a fresh observation-world readout.
    """

    ids = list(dict.fromkeys(item for item in component_ids if item))
    scopes = list(dict.fromkeys(item for item in scope_ids if item))
    if not ids or not scopes:
        raise ValueError("planner review schema requires live component and scope identifiers")
    component = _closed({
        "component_id": _STR, "claim": _STR, "depends_on": {"type": "array", "items": {"type": "string", "enum": ids}},
        "evidence_needed": _NON_EMPTY_STRINGS,
        "constraints_json": _STR, "failure_meaning": _STR,
        "scope_id": {"type": "string", "enum": scopes}, "candidate_hypothesis_ids": _STRINGS,
        "skill_ids": _STRINGS,
    })
    # Each variant contains only the information needed for that map change.
    # The model never receives a catch-all operation object, so a local tool
    # instruction cannot slip into a global planning patch.
    operation = {"anyOf": [
        _closed({"operation": {"type": "string", "enum": ["add_component"]}, "component": component, "reason": _STR}),
        _closed({"operation": {"type": "string", "enum": ["supersede_component"]}, "target_component_id": {"type": "string", "enum": ids}, "reason": _STR}),
        _closed({"operation": {"type": "string", "enum": ["reconnect_dependencies"]}, "target_component_id": {"type": "string", "enum": ids}, "depends_on": {"type": "array", "items": {"type": "string", "enum": ids}}, "reason": _STR}),
        _closed({"operation": {"type": "string", "enum": ["reopen_component"]}, "target_component_id": {"type": "string", "enum": ids}, "reason": _STR}),
        _closed({"operation": {"type": "string", "enum": ["rescout"]}, "target_component_id": {"type": "string", "enum": ids}, "rescout_scope_json": _STR, "reason": _STR}),
    ]}
    common = {
        "rationale": _STR,
        "affected_component_ids": {"type": "array", "items": {"type": "string", "enum": ids}},
        "diagnoses": {"type": "array", "items": _DIAGNOSIS},
    }
    # A patch route needs at least one operation, but minItems is unsupported.
    # Encode that fact in the object shape rather than accepting patch_graph
    # plus an empty operations array and detecting it only after the call.
    decision = {"anyOf": [
        _closed({"route": {"type": "string", "enum": ["repair_local"]}, **common}),
        _closed({"route": {"type": "string", "enum": ["formalize_gap"]}, **common}),
        _closed({"route": {"type": "string", "enum": ["patch_graph"]}, **common, "primary_operation": operation, "additional_operations": {"type": "array", "items": operation}}),
    ]}
    return _closed({"decision": decision})
CLAIM_ENTAILMENT_SCHEMA = _closed({"verdict": {"type": "string", "enum": ["entailed", "not_entailed"]}, "rationale": _STR})
# The R-layer Judge must make the same three-way decision as the global goal
# verifier.  A boolean ``entailed/not_entailed`` reviewer could only veto a
# deterministic pass and could not distinguish contradiction from insufficiency.
EVIDENCE_REVIEW_SCHEMA = _closed({
    "status": {"type": "string", "enum": ["supported", "contradicted", "insufficient"]},
    "confidence": _NUM,
    "rationale": _STR,
    "evidence_ids": _STRINGS,
    "missing_evidence": _STRINGS,
})
NODE_CONCLUSION_SCHEMA = _closed({
    "status": {"type": "string", "enum": ["supported", "contradicted", "insufficient"]},
    "conclusion": _STR,
    "observed_results": _STRINGS,
    "limitations": _STRINGS,
    "evidence_ids": _STRINGS,
})
WRITER_SUBGRAPH_SCHEMA = _closed({
    "root_component_id": _STR,
    "text": _STR,
    "component_ids": _STRINGS,
    "edge_ids": _STRINGS,
})
GOAL_VERIFICATION_SCHEMA = _closed({
    "status": {"type": "string", "enum": ["proven", "needs_plan_patch", "evidence_gap"]},
    "reason": _STR,
    "blocking_component_ids": _STRINGS,
    "missing_requirements": _STRINGS,
})
FINAL_ANSWER_SCHEMA = _closed({
    "status": {"type": "string", "enum": ["proven", "needs_plan_patch", "evidence_gap"]},
    "conclusion": _STR,
    "observed_pattern": _STRINGS,
    "supported_interpretation": _STRINGS,
    "alternative_explanations": _STRINGS,
    "evidence_gaps": _STRINGS,
    "next_actions": _STRINGS,
    "component_ids": _STRINGS,
})
