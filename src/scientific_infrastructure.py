"""Production scientific tools and read-only observation databases.

This module adapts the existing Source/Data/Tool registries to ordinary
LangChain tools.  The database is an external observable world: role nodes do
not read it from shared state, and only LangGraph's ToolNode can execute these
tools. Raw tool results are appended to ``objective_ledger`` for provenance;
data selected during one R episode remains an execution-scoped intermediate
value rather than an LLM-managed artifact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Annotated, Any, Literal, Mapping, Sequence, cast

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field, create_model
from typing_extensions import TypedDict

from scientific_research_agent.contracts import ToolResult
from scientific_research_agent.data_adapters import DynamicTableObservationSource
from scientific_research_agent.data_tool import run_data_tool
from scientific_research_agent.external import register_manifest
from scientific_research_agent.registry import (
    DataRegistry,
    ExternalDataRegistry,
    ToolContext,
    ToolRegistry,
    ToolSpec,
    tool_consumes_observation_rows,
)
from scientific_research_agent.sources import InMemoryObservationSource
from scientific_research_agent.tools_builtin import default_tool_registry


JSON = dict[str, Any]
Scalar = str | int | float | bool
ScientificFieldRole = Literal[
    "record_id",
    "paper_id",
    "sample_id",
    "age_ma",
    "age_error_ma",
    "latitude",
    "longitude",
    "position_uncertainty_km",
    "region",
    "province",
    "city",
    "basin",
    "locality",
    "rock_type",
    "formation",
    "group",
    "member",
    "isotope_system",
    "mineral",
    "technique",
    "statistical_model",
    "event_interpretation",
]


class ScientificFieldSelection(TypedDict):
    """One R-chosen source field for this scientific-tool call only."""

    role: ScientificFieldRole
    source_field: str


class TemporaryToolParameter(TypedDict):
    """One declared input in an auditable run-scoped Python operation."""

    name: str
    value_type: Literal[
        "integer",
        "number",
        "string",
        "boolean",
        "integer_array",
        "number_array",
        "string_array",
        "json",
    ]
    description: str
    required: bool


class EqualityOperator(str, Enum):
    EQ = "eq"


class RangeOperator(str, Enum):
    BETWEEN = "between"


class RelationMode(str, Enum):
    PROBE = "probe"


class SemanticAuthorization(str, Enum):
    EXPLICIT = "explicit"


class EqualityFilter(TypedDict):
    field: str
    operator: EqualityOperator
    value: Scalar


class SetFilter(TypedDict):
    field: str
    operator: Literal["in", "contains"]
    values: list[Scalar]


class RangeFilter(TypedDict):
    field: str
    operator: RangeOperator
    minimum_value: float
    maximum_value: float


class NullFilter(TypedDict):
    field: str
    operator: Literal["is_null", "not_null"]


WorkspaceFilter = EqualityFilter | SetFilter | RangeFilter | NullFilter


class WorkspaceAggregate(TypedDict):
    field: str
    operation: Literal["count", "min", "max", "mean", "median", "std", "n_unique"]
    alias: str


class WorkspaceTable(TypedDict):
    table_id: str
    alias: str


class WorkspaceRelation(TypedDict):
    left_table_id: str
    left_key: str
    right_table_id: str
    right_key: str
    join_type: Literal["inner", "left", "semi", "anti"]
    match_policy: Literal["exact", "normalized", "fuzzy", "spatial", "temporal"]
    match_tolerance: float


class WorkspaceInspectRequest(TypedDict):
    operation: Literal["inspect"]
    table_id: str


class WorkspaceReadRequest(TypedDict):
    operation: Literal["profile", "sample", "query"]
    table_id: str
    columns: list[str]
    filters: list[WorkspaceFilter]
    row_limit: int


class WorkspaceAggregateRequest(TypedDict):
    operation: Literal["aggregate"]
    table_id: str
    columns: list[str]
    filters: list[WorkspaceFilter]
    group_by: list[str]
    aggregates: list[WorkspaceAggregate]
    bin_field: str | None
    bin_width: float | None
    bin_origin: float | None
    row_limit: int


class WorkspacePairRelationRequest(TypedDict):
    operation: Literal["relate"]
    table_id: str
    related_table_id: str
    left_key: str
    right_key: str
    join_type: Literal["inner", "left", "semi", "anti"]
    match_policy: Literal["exact", "normalized", "fuzzy", "spatial", "temporal"]
    match_tolerance: float | None
    semantic_authorization: SemanticAuthorization | None
    row_limit: int


class WorkspaceGraphRelationRequest(TypedDict):
    operation: Literal["relate"]
    table_id: str
    tables: list[WorkspaceTable]
    relations: list[WorkspaceRelation]
    semantic_authorization: SemanticAuthorization | None
    row_limit: int


WorkspaceRequest = (
    WorkspaceInspectRequest
    | WorkspaceReadRequest
    | WorkspaceAggregateRequest
    | WorkspacePairRelationRequest
    | WorkspaceGraphRelationRequest
)


class _StrictToolModel(BaseModel):
    """Base for the small provider-facing contracts generated per tool."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)


_FIELD_PATH_TO_ROLE: dict[str, str] = {
    "age_ma": "age_ma",
    "age_error_ma": "age_error_ma",
    "position.latitude": "latitude",
    "position.longitude": "longitude",
    "position.uncertainty_km": "position_uncertainty_km",
    "isotope_system": "isotope_system",
    "observation_payload.spots": "observation_payload",
}

_TOOL_ROLE_OVERRIDES: dict[str, tuple[str, ...]] = {
    "sample.spot_quality": ("observation_payload",),
    "space.resolution": ("latitude", "longitude", "position_uncertainty_km"),
    # Provider tools need at least one selected physical row even when the
    # provider itself owns the chronometer-specific semantics.
    "chronology.isoplotr": ("record_id",),
}

_NUMERIC_SCIENTIFIC_ROLES = {
    "age_ma",
    "age_error_ma",
    "latitude",
    "longitude",
    "position_uncertainty_km",
}


def _scientific_roles(spec: ToolSpec) -> tuple[str, ...]:
    """Return only the scientific roles that this tool actually consumes."""

    overridden = _TOOL_ROLE_OVERRIDES.get(spec.tool_id)
    if overridden is not None:
        return overridden
    roles: list[str] = []
    for requirement in spec.input_requirements:
        for field_path in getattr(requirement, "field_paths", ()):
            role = _FIELD_PATH_TO_ROLE.get(str(field_path))
            if role and role not in roles:
                roles.append(role)
    return tuple(roles or ["record_id"])


def _scientific_args_schema(spec: ToolSpec) -> type[BaseModel]:
    """Build one minimal strict tool-call schema from the ToolSpec itself."""

    roles = _scientific_roles(spec)
    model_prefix = "".join(part.title() for part in _safe_tool_name(spec.tool_id).split("__"))
    source_fields_model = create_model(
        f"{model_prefix}SourceFields",
        __base__=_StrictToolModel,
        **{
            role: (
                str | None,
                Field(
                    description=(
                        f"Physical source field for scientific role {role}; use null "
                        "when supplying that role through inline_observations."
                    )
                ),
            )
            for role in roles
        },
    )
    inline_fields: dict[str, tuple[Any, Any]] = {}
    for role in roles:
        value_type: Any = float if role in _NUMERIC_SCIENTIFIC_ROLES else str
        inline_fields[role] = (value_type | None, None)
    inline_model = create_model(
        f"{model_prefix}InlineObservation",
        __base__=_StrictToolModel,
        **inline_fields,
    )
    argument_fields: dict[str, tuple[Any, Any]] = {
        "local_intent": (
            str,
            Field(
                description=(
                    "One brief statement of the immediate technical purpose; "
                    "recorded for observation only."
                )
            ),
        ),
        "input_from": (
            list[str],
            Field(
                description=(
                    "objective_record_id values from completed data-producing calls "
                    "visible in the current R context; use an empty list for direct values."
                )
            ),
        ),
        "source_fields": (
            source_fields_model,
            Field(
                description=(
                    "Map this tool's scientific roles to physical fields in input_from."
                )
            ),
        ),
        "inline_observations": (
            list[inline_model],
            Field(
                description=(
                    "Direct values already present in R's context; use an empty list "
                    "when consuming input_from."
                )
            ),
        ),
        # ToolRuntime is injected by LangGraph and automatically omitted from
        # the provider-facing schema, but must remain accepted by Pydantic at
        # the local ToolNode boundary.
        "runtime": (ToolRuntime, ...),
    }
    if spec.tool_id == "space.dbscan":
        argument_fields["eps_km"] = (
            float,
            Field(gt=0, description="Positive DBSCAN radius in kilometres."),
        )
    return create_model(
        f"{model_prefix}Args",
        __base__=_StrictToolModel,
        **argument_fields,
    )


_TEMPORARY_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_PYTHON_SANDBOX_TIMEOUT_SECONDS = 120
_PYTHON_SANDBOX_RUNNER = r'''
import ast
import contextlib
import io
import json
import math
import statistics
import sys

import numpy as np
import scipy
from scipy import optimize, signal, stats


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"non-JSON result value: {type(value).__name__}")


payload = json.loads(sys.stdin.read())
code = payload["code"]
tree = ast.parse(code, filename=f"temporary_tool:{payload['name']}", mode="exec")
ALLOWED_IMPORTS = {
    "math",
    "statistics",
    "numpy",
    "scipy",
    "scipy.optimize",
    "scipy.signal",
    "scipy.stats",
}
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        disallowed = [
            alias.name
            for alias in node.names
            if alias.name not in ALLOWED_IMPORTS
        ]
        if disallowed:
            raise ValueError(
                "only approved scientific imports are available: "
                + ", ".join(sorted(disallowed))
            )
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        if node.level or module not in ALLOWED_IMPORTS:
            raise ValueError(
                "only approved scientific imports are available: "
                + ("." * node.level + module)
            )
    if isinstance(node, ast.Name) and node.id.startswith("__"):
        raise ValueError("dunder names are unavailable")
    if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
        raise ValueError("private and dunder attributes are unavailable")

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level or name not in ALLOWED_IMPORTS:
        raise ImportError(f"import is not allowed: {name}")
    return __import__(name, globals, locals, fromlist, level)

safe_builtins = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "ValueError": ValueError,
    "zip": zip,
    "__import__": safe_import,
}
scope = {
    "__builtins__": safe_builtins,
    "arguments": payload["arguments"],
    "inputs": payload["inputs"],
    "math": math,
    "np": np,
    "numpy": np,
    "optimize": optimize,
    "scipy": scipy,
    "signal": signal,
    "statistics": statistics,
    "stats": stats,
}
stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    exec(compile(tree, filename=f"temporary_tool:{payload['name']}", mode="exec"), scope, scope)
if "result" not in scope:
    raise ValueError("temporary tool code must assign its JSON-serializable output to result")
print(json.dumps({
    "output": scope["result"],
    "stdout": stdout.getvalue(),
    "stderr": stderr.getvalue(),
}, ensure_ascii=False, default=json_default))
'''


def _validate_temporary_tool_arguments(
    parameter_schema: Sequence[TemporaryToolParameter],
    arguments_json: str,
) -> JSON:
    try:
        arguments = json.loads(arguments_json)
    except json.JSONDecodeError as exc:
        raise ValueError("arguments_json must contain one JSON object") from exc
    if not isinstance(arguments, dict):
        raise ValueError("arguments_json must contain one JSON object")

    schema_by_name: dict[str, TemporaryToolParameter] = {}
    for item in parameter_schema:
        name = str(item["name"])
        if not _TEMPORARY_TOOL_NAME.fullmatch(name):
            raise ValueError(f"invalid temporary-tool parameter name: {name!r}")
        if name in schema_by_name:
            raise ValueError(f"duplicate temporary-tool parameter: {name}")
        schema_by_name[name] = item
    missing = sorted(
        name
        for name, item in schema_by_name.items()
        if item["required"] and name not in arguments
    )
    extra = sorted(set(arguments) - set(schema_by_name))
    if missing:
        raise ValueError(f"missing required temporary-tool parameters: {missing}")
    if extra:
        raise ValueError(f"undeclared temporary-tool parameters: {extra}")

    def matches(value: Any, value_type: str) -> bool:
        if value_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if value_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if value_type == "string":
            return isinstance(value, str)
        if value_type == "boolean":
            return isinstance(value, bool)
        if value_type == "integer_array":
            return isinstance(value, list) and all(
                isinstance(item, int) and not isinstance(item, bool) for item in value
            )
        if value_type == "number_array":
            return isinstance(value, list) and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in value
            )
        if value_type == "string_array":
            return isinstance(value, list) and all(isinstance(item, str) for item in value)
        return True

    mismatches = [
        name
        for name, value in arguments.items()
        if not matches(value, schema_by_name[name]["value_type"])
    ]
    if mismatches:
        raise ValueError(f"temporary-tool parameter types do not match: {mismatches}")
    return cast(JSON, _plain(arguments))


def _macos_sandbox_profile(working_directory: Path) -> str:
    readable_roots = {
        Path("/System"),
        Path("/usr"),
        Path("/Library"),
        Path("/opt/homebrew"),
        Path(sys.prefix).resolve(),
        Path(sys.base_prefix).resolve(),
        working_directory.resolve(),
    }

    def quoted(path: Path) -> str:
        return str(path).replace("\\", "\\\\").replace('"', '\\"')

    read_rules = "\n".join(
        f'    (subpath "{quoted(path)}")' for path in sorted(readable_roots, key=str)
    )
    workdir = quoted(working_directory.resolve())
    return f'''(version 1)
(deny default)
(allow process*)
(allow sysctl-read)
(allow mach-lookup)
(allow file-read*
{read_rules})
(allow file-write* (subpath "{workdir}"))
'''


def _execute_temporary_python(
    *,
    name: str,
    code: str,
    arguments: JSON,
    inputs: list[JSON],
) -> tuple[Any, str, str, str | None, str]:
    payload = json.dumps(
        {"name": name, "code": code, "arguments": arguments, "inputs": inputs},
        ensure_ascii=False,
        default=str,
    )
    environment = {
        "HOME": "",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PATH": os.defpath,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
    }
    with tempfile.TemporaryDirectory(prefix="epistemic-python-") as raw_directory:
        working_directory = Path(raw_directory)
        base_command = [sys.executable, "-I", "-c", _PYTHON_SANDBOX_RUNNER]
        command = list(base_command)
        engine = "isolated_python_subprocess"
        sandbox_exec = shutil.which("sandbox-exec")
        if sys.platform == "darwin" and sandbox_exec:
            command = [
                sandbox_exec,
                "-p",
                _macos_sandbox_profile(working_directory),
                *command,
            ]
            engine = "macos_sandboxed_python_subprocess"
        def run(selected_command: list[str]) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                selected_command,
                input=payload,
                text=True,
                capture_output=True,
                cwd=working_directory,
                env=environment,
                timeout=_PYTHON_SANDBOX_TIMEOUT_SECONDS,
                check=False,
            )

        try:
            completed = run(command)
            if (
                engine == "macos_sandboxed_python_subprocess"
                and completed.returncode != 0
                and "sandbox_apply: Operation not permitted" in completed.stderr
            ):
                completed = run(base_command)
                engine = "isolated_python_subprocess_os_sandbox_unavailable"
        except subprocess.TimeoutExpired as exc:
            return None, exc.stdout or "", exc.stderr or "", "execution timed out", engine
        if completed.returncode != 0:
            error = completed.stderr.strip() or f"python exited with status {completed.returncode}"
            return None, completed.stdout, completed.stderr, error, engine
        try:
            envelope = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            return None, completed.stdout, completed.stderr, f"invalid sandbox output: {exc}", engine
        return (
            cast(JSON, _plain(envelope.get("output"))),
            str(envelope.get("stdout") or ""),
            str(envelope.get("stderr") or ""),
            None,
            engine,
        )


class ObjectiveObservation(TypedDict):
    record_id: str
    tool_call_id: str
    local_intent: str
    task_id: str
    item_id: str
    record_kind: Literal["data_result", "data_operation", "scientific_result"]
    tool_id: str
    source_id: str
    input_scope: JSON
    parameters: JSON
    upstream_record_ids: list[str]
    result: JSON


# Showcase configuration: explicit paths are supplied to from_database_paths.
# Legacy convenience defaults use a portable directory rather than personal paths.
_PUBLIC_DATA_DIR = Path(os.getenv("DEEPRESEARCH_DATA_DIR", "data/private"))
DEFAULT_DATABASE_PATHS = tuple(_PUBLIC_DATA_DIR / name for name in (
    "global_zircon_v2_1.sqlite", "usgs_v4_multitable.sqlite",
    "GPL1734_TS-2.xlsx.xls", "Xu2022_Table_S1_extracted.xlsx",
    "Table S2. Input detrital zircon U-Pb data for three-dimensional metric multidimensional scaling.xlsx",
    "Table S3. Input detrital zircon U-Pb data for mixing modeling.xlsx",
))

DEFAULT_SOURCE_IDS = {
    str(DEFAULT_DATABASE_PATHS[2]): "GPL1734_TS-2",
    str(DEFAULT_DATABASE_PATHS[3]): "Xu2022_Table_S1",
    str(DEFAULT_DATABASE_PATHS[4]): "D6XCR_Table_S2_MDS",
    str(DEFAULT_DATABASE_PATHS[5]): "D6XCR_Table_S3_Mixing",
}

DEFAULT_HEADER_ROWS = {
    str(DEFAULT_DATABASE_PATHS[4]): {
        "Fig. S2 Australia-Canning&Perth": 1,
        "Fig. S2 Canning basin": 1,
        "Fig. S2 Perth basin": 1,
        "Fig. S3 Himalaya": 1,
        "Fig. S4 Arabia & Iran": 1,
        "Fig.S5 India": 1,
        "Fig. S6 Africa": 1,
    },
    str(DEFAULT_DATABASE_PATHS[5]): {
        "Fig. 3a Lhasa-Cannning&Perth": 1,
        "Fig. 3b Lhasa-Himalaya&India": 1,
        "Fig. 3c Lhasa-H&I&A&Iran&Africa": 1,
        "Fig. 3d E. Lhasa-Canning": 1,
        "Fig. 3e W. Lhasa-Perth": 1,
        "Fig. 3f E. Lhasa-Canning&Perth": 1,
        "Fig. 3g W. Lhasa-Canning&Perth": 1,
    },
}


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if hasattr(value, "__dict__"):
        return _plain(vars(value))
    return value


def _clean_request(request: WorkspaceRequest) -> JSON:
    """Adapt the strict provider shape to the native workspace contract."""

    result: JSON = {"operation": request["operation"]}
    for key, value in request.items():
        if key in {"operation", "filters", "aggregates"}:
            continue
        if value not in (None, "", [], {}):
            result[key] = _plain(value)
    filters: list[JSON] = []
    for item in request.get("filters", []):
        raw_operator = item["operator"]
        operator = raw_operator.value if isinstance(raw_operator, Enum) else raw_operator
        adapted: JSON = {"field": item["field"], "operator": operator}
        if operator in {"is_null", "not_null"}:
            pass
        elif operator == "between":
            adapted["minimum_number"] = item["minimum_value"]
            adapted["maximum_number"] = item["maximum_value"]
        elif operator in {"in", "contains"}:
            values = list(item["values"])
            if not values:
                raise ValueError(f"{operator} filter requires non-empty values")
            adapted["text_values"] = values
        else:
            adapted["text_values"] = [item["value"]]
        filters.append(adapted)
    if filters:
        result["filters"] = filters
    aggregates = [
        {
            "field": item["field"],
            "operation": item["operation"],
            "as": item["alias"],
        }
        for item in request.get("aggregates", [])
    ]
    if aggregates:
        result["aggregates"] = aggregates
    raw_row_limit = request.get("row_limit")
    row_limit = 1_000 if raw_row_limit is None else int(raw_row_limit)
    if row_limit <= 0:
        raise ValueError("data.tool row_limit must be positive")
    result["row_limit"] = row_limit
    return result


def _navigation_scope(
    source: Any,
    *,
    table_id: str,
    filters: Sequence[WorkspaceFilter],
    row_limit: int,
) -> JSON:
    table_id = str(table_id or "")
    if not table_id:
        return {}
    predicates: list[JSON] = []
    for item in filters:
        raw_operator = item["operator"]
        operator = raw_operator.value if isinstance(raw_operator, Enum) else raw_operator
        predicate: JSON = {
            "source_column": item["field"],
            "operator": operator,
        }
        if operator == "between":
            predicate["minimum_value"] = item["minimum_value"]
            predicate["maximum_value"] = item["maximum_value"]
        elif operator in {"in", "contains"}:
            predicate["values"] = list(item["values"])
        elif operator == "eq":
            predicate["value"] = item["value"]
        predicates.append(predicate)
    return {
        "_scope_contract": "table_query/v1",
        "projection_contract": "physical_rows/v1",
        "source_id": source.source_id,
        "table_id": table_id,
        "status": "resolved",
        "predicates": predicates,
        "unresolved_terms": [],
        "non_filter_terms": [],
        "limit": int(row_limit),
        "provenance": {
            "owner": "researcher_data_navigation",
            "representation": "physical table fields",
        },
    }


def _scope_bundle(source_id: str, table_id: str, source_scope: JSON) -> JSON:
    if not source_scope:
        return {}
    return {
        "_scope_contract": "scope_bundle/v2",
        "status": source_scope.get("status", "resolved"),
        "source_scopes": {source_id: source_scope},
        "table_scopes": {f"{source_id}::{table_id}": source_scope},
        "provenance": {"owner": "scientific_infrastructure"},
    }


def _select_scientific_fields(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[ScientificFieldSelection],
    source_id: str,
) -> list[JSON]:
    """Apply R's explicit per-call field choices without inferring semantics."""

    fields = list(fields)
    if not fields:
        raise ValueError(
            "scientific_input.fields is required when consuming a raw data.tool result"
        )
    roles = [item["role"] for item in fields]
    if len(set(roles)) != len(roles):
        raise ValueError("scientific_input.fields contains duplicate scientific roles")
    available_fields = {
        str(key)
        for row in rows
        for key in row
    }
    missing_fields = sorted(
        {
            item["source_field"]
            for item in fields
            if item["source_field"] not in available_fields
        }
    )
    if missing_fields:
        raise ValueError(
            f"scientific_input.fields names unavailable source fields: {missing_fields}"
        )
    projected_rows: list[JSON] = []
    numeric_fields = {
        "age_ma",
        "age_error_ma",
        "latitude",
        "longitude",
        "position_uncertainty_km",
    }
    for index, row in enumerate(rows, start=1):
        projected: JSON = {
            item["role"]: row.get(item["source_field"])
            for item in fields
        }
        for field_name in numeric_fields:
            value = projected.get(field_name)
            if value not in (None, ""):
                try:
                    projected[field_name] = float(value)
                except (TypeError, ValueError) as exc:
                    source_field = next(
                        item["source_field"]
                        for item in fields
                        if item["role"] == field_name
                    )
                    raise ValueError(
                        f"source field {source_field!r} contains a non-numeric value "
                        f"for scientific role {field_name!r}"
                    ) from exc
        projected.setdefault("record_id", f"{source_id}:{index}")
        projected.setdefault("source_id", source_id)
        projected["_semantic_field_provenance"] = {
            item["role"]: [item["source_field"]] for item in fields
        }
        projected["position"] = {
            "latitude": projected.get("latitude"),
            "longitude": projected.get("longitude"),
            "uncertainty_km": projected.get("position_uncertainty_km"),
            "resolution": "unresolved",
            "provenance": "R-selected fields for this scientific tool call",
        }
        projected_rows.append(projected)
    return projected_rows


def _objective_records(state: Mapping[str, Any], item_id: str) -> list[ObjectiveObservation]:
    return [
        cast(ObjectiveObservation, item)
        for item in state.get("objective_ledger", [])
        if isinstance(item, dict) and item.get("item_id") == item_id
    ]


def _technical_item_id(technical_task: Mapping[str, Any]) -> str:
    return str(
        technical_task.get("target_epistemic_item")
        or technical_task.get("item_id")
        or ""
    )


def _visible_objective_record_ids(messages: Sequence[Any]) -> set[str]:
    """Return objective result references explicitly present in R's visible messages."""

    visible: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage) or not isinstance(message.content, str):
            continue
        try:
            payload = json.loads(message.content)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        record_id = payload.get("objective_record_id")
        if record_id:
            visible.add(str(record_id))
    return visible


def _prior_evidence(records: Sequence[ObjectiveObservation]) -> list[Any]:
    evidence = []
    for record in records:
        result = record.get("result", {})
        evidence.append(
            SimpleNamespace(
                evidence_id=record["record_id"],
                findings=dict(result.get("findings") or {}),
                input_scope=dict(record.get("input_scope") or {}),
                evidence_tags=list(result.get("evidence_tags") or []),
                status=str(result.get("status") or "error"),
                subject_id=record["tool_id"],
            )
        )
    return evidence


def _tool_context(
    *,
    state: Mapping[str, Any],
    source_registry: DataRegistry,
    external_registry: ExternalDataRegistry,
    source_id: str,
    input_scope: JSON,
    parameters: JSON,
    action_id: str,
    data_request: JSON | None = None,
    prior_record_ids: Sequence[str] | None = None,
) -> ToolContext:
    technical_task = state.get("technical_task")
    if not isinstance(technical_task, dict):
        raise ValueError("scientific tool execution requires the current technical task")
    item_id = _technical_item_id(technical_task)
    records = _objective_records(state, item_id)
    if prior_record_ids is not None:
        allowed_record_ids = {str(item) for item in prior_record_ids}
        records = [item for item in records if item["record_id"] in allowed_record_ids]
    component = SimpleNamespace(
        component_id=str(technical_task.get("task_id") or ""),
        constraints={key: value for key, value in parameters.items() if value is not None},
    )
    requested_scale = {}
    if parameters.get("spatial_claim_scale_km") is not None:
        requested_scale["spatial_claim_scale_km"] = parameters["spatial_claim_scale_km"]
    task = SimpleNamespace(
        question=str(technical_task.get("technical_question") or ""),
        scope=input_scope,
        requested_scale=requested_scale,
    )
    action = SimpleNamespace(
        action_id=action_id,
        tool_id="",
        component_id=component.component_id,
        source_id=source_id,
        data_request=dict(data_request or {}),
        parameters=dict(parameters),
    )
    return ToolContext(
        question=task.question,
        task=task,
        component=component,
        scope=input_scope,
        source_registry=source_registry,
        external_registry=external_registry,
        prior_evidence=_prior_evidence(records),
        hypothesis_ids=[],
        plan=None,
        scout=None,
        action=action,
    )


def _record_and_message(
    *,
    runtime: ToolRuntime[dict[str, Any], dict[str, Any]],
    local_intent: str,
    tool_name: str,
    native_tool_id: str,
    source_id: str,
    input_scope: JSON,
    parameters: JSON,
    upstream_record_ids: list[str],
    result: ToolResult,
    record_kind: Literal["data_result", "data_operation", "scientific_result"] | None = None,
    archive: bool = True,
    message_status: Literal["success", "error"] = "success",
) -> Command:
    technical_task = runtime.state.get("technical_task")
    if not isinstance(technical_task, dict):
        raise ValueError("tool result has no current technical task")
    payload = cast(JSON, _plain(result))
    if record_kind is None:
        record_kind = "data_operation" if native_tool_id == "data.tool" else "scientific_result"
    record = ObjectiveObservation(
        record_id=runtime.tool_call_id,
        tool_call_id=runtime.tool_call_id,
        local_intent=local_intent,
        task_id=str(technical_task.get("task_id") or ""),
        item_id=_technical_item_id(technical_task),
        record_kind=record_kind,
        tool_id=native_tool_id,
        source_id=source_id,
        input_scope=input_scope,
        parameters=parameters,
        upstream_record_ids=upstream_record_ids,
        result=payload,
    )
    content = json.dumps(
        {
            "objective_record_id": runtime.tool_call_id,
            "record_kind": record_kind,
            "upstream_record_ids": upstream_record_ids,
            "result": payload,
        },
        ensure_ascii=False,
        default=str,
    )
    update: JSON = {
        "objective_ledger": [record],
        "messages": [
            ToolMessage(
                content=content,
                name=tool_name,
                tool_call_id=runtime.tool_call_id,
                status=message_status,
            )
        ],
    }
    if archive:
        update["archive_index"] = [
            {
                "archive_id": f"tool:{runtime.tool_call_id}",
                "entry_kind": "tool_result",
                "item_id": record["item_id"],
                "task_id": record["task_id"],
                "reference_id": record["record_id"],
            }
        ]
    return Command(update=update)


def _safe_tool_name(tool_id: str) -> str:
    return tool_id.replace(".", "__").replace("-", "_")


def _react_capability_text(value: Any) -> str:
    """Translate legacy artifact vocabulary at the public R/D boundary."""

    text = str(value)
    replacements = (
        ("A completed upstream data.tool materialize result", "A completed data-producing call"),
        ("materialized_data_handle", "current_episode_canonical_data"),
        ("materialized_dataset", "current_episode_canonical_data"),
        ("materialized data handle", "current-episode canonical data"),
        ("materialized age-position observations", "current-episode age-position observations"),
        ("materialized age observations", "current-episode age observations"),
        ("materialized positions", "current-episode positions"),
        ("materialized coordinates", "current-episode coordinates"),
        ("materialized", "current-episode"),
    )
    for old, new in replacements:
        text = text.replace(old, new)
    return text


@dataclass(slots=True)
class ScientificInfrastructure:
    """The non-agent infrastructure behind R's standard LangChain tools."""

    source_registry: DataRegistry
    tool_registry: ToolRegistry
    external_registry: ExternalDataRegistry

    @classmethod
    def from_database_paths(
        cls,
        database_paths: Sequence[str | Path],
        *,
        source_ids: Mapping[str, str] | None = None,
        header_rows: Mapping[str, Mapping[str, int]] | None = None,
        external_manifest: str | Path | None = None,
        max_query_rows: int = 3_000_000,
    ) -> "ScientificInfrastructure":
        sources = DataRegistry()
        resolved_ids = dict(source_ids or {})
        for raw_path in database_paths:
            path = Path(raw_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            source_id = resolved_ids.get(str(path), path.stem)
            sources.register(
                DynamicTableObservationSource.from_path(
                    path,
                    source_id=source_id,
                    header_rows=(header_rows or {}).get(str(path)),
                    max_query_rows=max_query_rows,
                )
            )
        external = ExternalDataRegistry()
        if external_manifest is not None:
            register_manifest(external, external_manifest)
        return cls(sources, default_tool_registry(), external)

    @classmethod
    def from_local_databases(
        cls,
    ) -> "ScientificInfrastructure":
        """Open the registered local scientific data files read-only."""

        return cls.from_database_paths(
            DEFAULT_DATABASE_PATHS,
            source_ids=DEFAULT_SOURCE_IDS,
            header_rows=DEFAULT_HEADER_ROWS,
        )

    def registered_data_sources(self) -> list[JSON]:
        """Project source-owned coarse descriptions into R's read-only context."""

        catalogue: list[JSON] = []
        for source in self.source_registry.all():
            descriptor = cast(JSON, _plain(source.describe_source()))
            source_schema = cast(JSON, _plain(source.describe_schema()))
            raw_tables = source_schema.get("tables")
            if not isinstance(raw_tables, list):
                workspace_catalogue = (
                    source.workspace_catalogue()
                    if hasattr(source, "workspace_catalogue")
                    else {}
                )
                raw_tables = (
                    workspace_catalogue.get("tables", [])
                    if isinstance(workspace_catalogue, dict)
                    else []
                )
            tables: list[JSON] = []
            seen_table_ids: set[str] = set()
            for raw_table in raw_tables:
                if not isinstance(raw_table, dict):
                    continue
                table_id = str(raw_table.get("table_id") or raw_table.get("name") or "")
                if not table_id or table_id in seen_table_ids:
                    continue
                seen_table_ids.add(table_id)
                tables.append(
                    {
                        "table_id": table_id,
                        "description": str(raw_table.get("description") or ""),
                    }
                )
            catalogue.append(
                {
                    "source_id": source.source_id,
                    "name": str(descriptor.get("name") or ""),
                    "description": str(descriptor.get("description") or ""),
                    "tables": tables,
                }
            )
        return catalogue

    def _data_result_rows(self, record: ObjectiveObservation) -> list[JSON]:
        """Read physical rows from the same public data.tool result shown to R."""

        if record.get("tool_id") != "data.tool" or record.get("record_kind") != "data_result":
            raise ValueError("input_from must name a prior data.tool data result")
        findings = record.get("result", {}).get("findings", {})
        rows = [
            cast(JSON, _plain(dict(item)))
            for item in findings.get("records", [])
            if isinstance(item, dict)
        ]
        expected = record.get("result", {}).get("findings", {}).get("record_count")
        if not rows and expected:
            raise ValueError(
                "the prior data.tool result does not expose its physical records"
            )
        if isinstance(expected, int) and len(rows) != expected:
            raise ValueError(
                f"data.tool result is incomplete: expected {expected}, found {len(rows)} records"
            )
        return rows

    def langchain_tools(self) -> list[BaseTool]:
        tools = [
            self._data_tool(),
            self._archive_read_tool(),
        ]
        if os.getenv("MIX_DISABLE_SANDBOX_PYTHON", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            tools.append(self._python_sandbox_tool())
        tools.extend(
            self._scientific_tool(spec)
            for spec in self.tool_registry.all()
            if spec.tool_id != "data.tool" and spec.exposed_to_researcher
        )
        return tools

    def compile_graph(self, *, checkpointer: BaseCheckpointSaver):
        """Inject this objective world into a caller-owned durable graph."""

        from thin_epistemic_research_agent import build_graph

        return build_graph(
            self.langchain_tools(),
            checkpointer=checkpointer,
            registered_data_sources=self.registered_data_sources(),
        )

    def _archive_read_tool(self) -> BaseTool:
        """Build one ordinary, read-only peer tool over the flat task archive."""

        def read_archive(
            local_intent: Annotated[
                str,
                "One brief statement of why an older archived record is needed now; recorded for observation only.",
            ],
            entry_ids: list[str],
            entry_kinds: list[
                Literal["tool_result", "technical_packet", "department_report"]
            ],
            task_ids: list[str],
            limit: Annotated[
                int,
                Field(
                    ge=1,
                    le=20,
                    description="Required maximum number of archive entries to return; use an integer from 1 through 20.",
                ),
            ],
            runtime: ToolRuntime[dict[str, Any], dict[str, Any]],
        ) -> Command:
            """Read selected older records from the current epistemic item's flat archive."""

            if limit < 1 or limit > 20:
                raise ValueError("archive.read limit must be between 1 and 20")
            technical_task = runtime.state.get("technical_task")
            if not isinstance(technical_task, dict):
                raise ValueError("archive.read requires a current technical task")
            item_id = str(
                technical_task.get("target_epistemic_item")
                or technical_task.get("item_id")
                or ""
            )
            requested_ids = set(dict.fromkeys(item for item in entry_ids if item))
            requested_kinds = set(entry_kinds)
            requested_tasks = set(
                dict.fromkeys(item for item in task_ids if item)
            )

            objective_by_id = {
                str(item.get("record_id") or ""): item
                for item in runtime.state.get("objective_ledger", [])
                if isinstance(item, dict)
            }
            packet_by_id = {
                str(item.get("packet_id") or ""): item
                for item in runtime.state.get("evidence_ledger", [])
                if isinstance(item, dict)
            }
            report_by_id = {
                str(item.get("report_id") or ""): item
                for item in runtime.state.get("report_ledger", [])
                if isinstance(item, dict)
            }
            visible_index = [
                item
                for item in runtime.state.get("archive_index", [])
                if isinstance(item, dict)
                and str(item.get("item_id") or "") == item_id
                and (not requested_ids or str(item.get("archive_id") or "") in requested_ids)
                and (
                    not requested_kinds
                    or str(item.get("entry_kind") or "") in requested_kinds
                )
                and (
                    not requested_tasks
                    or str(item.get("task_id") or "") in requested_tasks
                )
            ]
            selected_index = visible_index[-limit:]
            entries: list[JSON] = []
            missing_references: list[str] = []
            for index_entry in selected_index:
                reference_id = str(index_entry.get("reference_id") or "")
                kind = str(index_entry.get("entry_kind") or "")
                if kind == "tool_result":
                    payload = objective_by_id.get(reference_id)
                elif kind == "technical_packet":
                    payload = packet_by_id.get(reference_id)
                else:
                    payload = report_by_id.get(reference_id)
                if payload is None:
                    missing_references.append(reference_id)
                    continue
                entries.append({**_plain(index_entry), "content": _plain(payload)})

            result = ToolResult(
                result_id=f"archive.read:{runtime.tool_call_id}",
                tool_id="archive.read",
                status="ok",
                calculation_status="computed",
                summary=(
                    f"Read {len(entries)} archived record(s) from epistemic item "
                    f"{item_id}."
                ),
                findings={
                    "entries": entries,
                    "matched_entry_count": len(visible_index),
                    "returned_entry_count": len(entries),
                    "missing_reference_ids": missing_references,
                    "archive_layout": "flat_append_only_index",
                },
                evidence_tags=[],
                uncertainty={},
                data_versions={},
                limitations=[
                    "Only records from the current epistemic item are visible.",
                    "Archive retrieval restores prior context; it does not create new scientific evidence.",
                ],
            )
            return _record_and_message(
                runtime=runtime,
                local_intent=local_intent,
                tool_name="archive__read",
                native_tool_id="archive.read",
                source_id="",
                input_scope={"item_id": item_id},
                parameters={
                    "entry_ids": list(entry_ids),
                    "entry_kinds": list(entry_kinds),
                    "task_ids": list(task_ids),
                    "limit": limit,
                },
                upstream_record_ids=[],
                result=result,
                record_kind="data_operation",
                archive=False,
            )

        description = (
            "Read older durable records from the current epistemic item's flat archive when the rolling "
            "R context no longer contains something needed for the current action. Filter by exact "
            "archive entry IDs, record kinds, or task IDs; empty filters select the latest matching "
            "entries. limit is required and must be an integer from 1 through 20. It may return tool "
            "results, TechnicalPackets, or DepartmentReports. This is an "
            "ordinary optional peer tool, never a prerequisite, and cannot read another epistemic item."
        )
        result = tool("archive__read", description=description)(read_archive)
        result.metadata = {
            "native_tool_id": "archive.read",
            "capability_group": "execution_archive",
            "capabilities": ["archived_context_retrieval"],
            "output_tags": ["archived_execution_records"],
            "input_objects": ["flat archive entry IDs, record kinds, or task IDs"],
            "output_objects": [
                "older tool results, TechnicalPackets, or DepartmentReports from the current epistemic item"
            ],
            "hard_requirements": [
                "Use only when the needed record is absent from the rolling context.",
                "limit is required and must be between 1 and 20 inclusive.",
                "Archive reads are contextual retrieval, not new scientific evidence.",
            ],
            "data_access_role": "current_agenda_archive_reader",
        }
        return result

    def _python_sandbox_tool(self) -> BaseTool:
        """Build one peer tool for auditable, run-scoped Python operations."""

        def execute_temporary_tool(
            local_intent: Annotated[
                str,
                "One brief statement of the immediate technical purpose of this call; recorded for observation only.",
            ],
            temporary_tool_name: Annotated[
                str,
                "A descriptive snake_case name for this one temporary operation.",
            ],
            purpose: Annotated[
                str,
                "The single orthogonal operation performed by this temporary tool.",
            ],
            parameter_schema: list[TemporaryToolParameter],
            arguments_json: Annotated[
                str,
                "One JSON object containing this invocation's declared parameter values; use {} when there are none.",
            ],
            input_from: list[str],
            code: Annotated[
                str,
                "Python code for this operation. Read arguments and inputs, then assign the JSON-serializable output to result.",
            ],
            runtime: ToolRuntime[dict[str, Any], dict[str, Any]],
        ) -> Command:
            """Create and immediately run one auditable temporary Python operation."""

            technical_task = runtime.state.get("technical_task")
            if not isinstance(technical_task, dict):
                raise ValueError("sandbox.python requires a current technical task")
            item_id = _technical_item_id(technical_task)
            task_id = str(technical_task.get("task_id") or "")
            requested_inputs = list(
                dict.fromkeys(str(item) for item in input_from if item)
            )
            visible_record_ids = _visible_objective_record_ids(
                runtime.state.get("messages", [])
            )
            records = [
                item
                for item in _objective_records(runtime.state, item_id)
                if (
                    str(item.get("task_id") or "") == task_id
                    or str(item.get("record_id") or "") in visible_record_ids
                )
            ]
            records_by_id = {str(item["record_id"]): item for item in records}
            definition: JSON = {
                "definition_id": f"temporary-tool:{runtime.tool_call_id}",
                "name": temporary_tool_name,
                "purpose": purpose,
                "parameter_schema": _plain(parameter_schema),
                "code": code,
                "lifetime": "current_agent_run",
                "registered_globally": False,
            }
            arguments: JSON = {}
            output: Any = None
            stdout = ""
            stderr = ""
            engine = "not_started"
            error: str | None = None
            try:
                if not _TEMPORARY_TOOL_NAME.fullmatch(temporary_tool_name):
                    raise ValueError(
                        "temporary_tool_name must be snake_case and contain 2 to 64 characters"
                    )
                if not purpose.strip():
                    raise ValueError("temporary tool purpose cannot be empty")
                if not code.strip():
                    raise ValueError("temporary tool code cannot be empty")
                arguments = _validate_temporary_tool_arguments(
                    parameter_schema,
                    arguments_json,
                )
                missing_ids = [
                    record_id
                    for record_id in requested_inputs
                    if record_id not in records_by_id
                ]
                if missing_ids:
                    raise ValueError(
                        "input_from must reference completed calls visible in the current R context: "
                        f"{missing_ids}"
                    )
                inputs = [
                    {
                        "objective_record_id": record_id,
                        "tool_id": records_by_id[record_id]["tool_id"],
                        "record_kind": records_by_id[record_id]["record_kind"],
                        "result": _plain(records_by_id[record_id]["result"]),
                    }
                    for record_id in requested_inputs
                ]
                output, stdout, stderr, error, engine = _execute_temporary_python(
                    name=temporary_tool_name,
                    code=code,
                    arguments=arguments,
                    inputs=inputs,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

            definition["runtime"] = {
                "engine": engine,
                "timeout_seconds": _PYTHON_SANDBOX_TIMEOUT_SECONDS,
                "network_access": (
                    "denied_by_os_sandbox"
                    if engine == "macos_sandboxed_python_subprocess"
                    else "not_os_enforced"
                ),
                "working_directory": "ephemeral",
                "preloaded_names": [
                    "math",
                    "statistics",
                    "np",
                    "numpy",
                    "scipy",
                    "stats",
                    "signal",
                    "optimize",
                ],
            }
            findings: JSON = {
                "temporary_tool": definition,
                "arguments": arguments,
                "input_from": requested_inputs,
                "output": _plain(output),
                "stdout": stdout,
                "stderr": stderr,
                "error": error,
            }
            succeeded = error is None
            result = ToolResult(
                result_id=f"sandbox.python:{runtime.tool_call_id}",
                tool_id="sandbox.python",
                status="ok" if succeeded else "error",
                calculation_status="computed" if succeeded else "not_computable",
                summary=(
                    f"Executed temporary operation {temporary_tool_name}."
                    if succeeded
                    else f"Temporary operation {temporary_tool_name} failed: {error}"
                ),
                findings=findings,
                evidence_tags=["run_scoped_python_operation"] if succeeded else [],
                uncertainty={},
                data_versions={},
                limitations=[
                    "The temporary definition is retained for audit but is not added to the global tool registry.",
                    "The operation is code-generated for this run and is not a substitute for later package-backed registration.",
                ],
            )
            return _record_and_message(
                runtime=runtime,
                local_intent=local_intent,
                tool_name="sandbox__python",
                native_tool_id="sandbox.python",
                source_id="",
                input_scope={"item_id": item_id, "task_id": task_id},
                parameters={
                    "temporary_tool": definition,
                    "arguments": arguments,
                    "arguments_json": arguments_json,
                    "input_from": requested_inputs,
                },
                upstream_record_ids=requested_inputs,
                result=result,
                record_kind="scientific_result",
                message_status="success" if succeeded else "error",
            )

        description = (
            "Create and immediately run one auditable, run-scoped Python operation only when the "
            "registered toolbox lacks the required atomic capability. Each call declares a temporary "
            "name, one purpose, parameter schema, arguments, prior result references, and code; all are "
            "retained with the output or error, while the global toolbox remains unchanged. The code "
            "receives arguments and inputs, has preloaded math/statistics/NumPy/SciPy names, and must "
            "assign result. The operation must be orthogonal: it may compute one statistic or transform. "
            "Good example: accept one numeric vector and return one dip statistic plus its p-value. "
            "Forbidden example: read and filter tables, run KDE, dip testing and bootstrap, choose an "
            "answer, and interpret it inside one temporary tool. This is an ordinary peer tool, never a node."
        )
        result = tool("sandbox__python", description=description)(execute_temporary_tool)
        result.metadata = {
            "native_tool_id": "sandbox.python",
            "capability_group": "temporary_python_operations",
            "capabilities": [
                "run_scoped_tool_creation",
                "auditable_python_execution",
                "prior_result_consumption",
            ],
            "output_tags": ["temporary_tool_definition", "temporary_tool_result"],
            "input_objects": [
                "one orthogonal temporary operation definition",
                "declared JSON parameters",
                "optional completed tool-result references",
            ],
            "output_objects": [
                "auditable temporary-tool definition, source code, parameters, output, and errors"
            ],
            "hard_requirements": [
                "Use only when an atomic capability is absent from the registered toolbox.",
                "One temporary tool performs one orthogonal operation.",
                "Never bundle data access, analysis sequences, answer selection, or interpretation.",
                "Temporary definitions remain run-scoped and never mutate the global ToolRegistry.",
            ],
            "data_access_role": "prior_call_or_inline_parameter_consumer",
        }
        return result

    def _data_tool(self) -> BaseTool:
        infrastructure = self
        spec = self.tool_registry.get("data.tool")

        def execute_data_tool(
            local_intent: Annotated[
                str,
                "One brief statement of the immediate technical purpose of this call; recorded for observation only.",
            ],
            source_id: str,
            request: WorkspaceRequest,
            runtime: ToolRuntime[dict[str, Any], dict[str, Any]],
        ) -> Command:
            """Read the registered objective world through one bounded operation."""

            operation = request["operation"]
            if not source_id:
                raise ValueError(
                    "data.tool requires a source_id from registered_data_sources"
                )
            if not request["table_id"]:
                raise ValueError(
                    "data.tool requires a table_id from registered_data_sources"
                )
            source = infrastructure.source_registry.get(source_id)
            cleaned = _clean_request(request)
            source_scope = (
                {}
                if operation == "inspect"
                else _navigation_scope(
                    source,
                    table_id=request["table_id"],
                    filters=list(request.get("filters", [])),
                    row_limit=int(request.get("row_limit") or 0),
                )
            )
            input_scope = _scope_bundle(source_id, request["table_id"], source_scope)
            context = _tool_context(
                state=runtime.state,
                source_registry=infrastructure.source_registry,
                external_registry=infrastructure.external_registry,
                source_id=source_id,
                input_scope=input_scope,
                parameters={},
                action_id=runtime.tool_call_id,
                data_request=cleaned,
            )
            context.action.tool_id = "data.tool"
            result = run_data_tool(context)
            if operation in {"query", "sample"}:
                native_findings = dict(result.findings)
                rows = [
                    cast(JSON, item)
                    for item in native_findings.get("records", [])
                    if isinstance(item, dict)
                ]
                physical_fields = sorted(
                    {
                        str(key)
                        for row in rows
                        for key in row
                    }
                )
                findings: JSON = {
                    "operation": operation,
                    "source_id": source_id,
                    "record_count": len(rows),
                    "available_record_count": int(native_findings.get("record_count") or len(rows)),
                    "truncated": bool(native_findings.get("truncated")),
                    "physical_fields": physical_fields,
                    "records": rows,
                }
                result.summary = (
                    f"Selected {len(rows)} physical row(s) from {source_id} as a "
                    "temporary result of this tool call."
                )
                result.findings = findings
                return _record_and_message(
                    runtime=runtime,
                    local_intent=local_intent,
                    tool_name="data__tool",
                    native_tool_id="data.tool",
                    source_id=source_id,
                    input_scope=input_scope,
                    parameters={"source_id": source_id, "request": cleaned},
                    upstream_record_ids=[],
                    result=infrastructure.tool_registry.validate_result(spec, result),
                    record_kind="data_result",
                )
            result = infrastructure.tool_registry.validate_result(spec, result)
            return _record_and_message(
                runtime=runtime,
                local_intent=local_intent,
                tool_name="data__tool",
                native_tool_id="data.tool",
                source_id=source_id,
                input_scope=input_scope,
                parameters={"source_id": source_id, "request": cleaned},
                upstream_record_ids=[],
                result=result,
            )

        description = (
            "Read registered tables and execute bounded, read-only SQL-style data operations and "
            "common descriptive statistics. It returns objective rows or compact computed results "
            "without assigning scientific meaning. This is an ordinary peer tool."
        )
        result = tool("data__tool", description=description)(execute_data_tool)
        result.metadata = {
            "native_tool_id": "data.tool",
            "capability_group": spec.capability_group,
            "capabilities": [
                "known_table_inspect",
                "known_table_profile",
                "known_table_read",
                "known_table_filter",
                "known_table_aggregate",
                "known_table_relate",
                "physical_row_read",
            ],
            "output_tags": [
                "data_inspect",
                "data_profile",
                "data_sample",
                "data_query",
                "data_aggregate",
                "data_relate",
            ],
            "input_objects": [
                "known source_id and table_id",
                "physical table columns and filters for data reads",
                "strict read or relational operation request",
            ],
            "output_objects": [
                "read-only data operation result",
                "execution-scoped physical rows from query/sample",
                "data decision request when scientific meaning is ambiguous",
            ],
            "hard_requirements": [
                "source_id and table_id must already be known",
                "inspect exposes only physical schema metadata",
                "query and sample preserve physical field names and values",
                "this tool never performs semantic table discovery",
            ],
            "data_access_role": "relational_data_operator",
        }
        return result

    def _scientific_tool(self, spec: ToolSpec) -> BaseTool:
        infrastructure = self
        tool_name = _safe_tool_name(spec.tool_id)
        args_schema = _scientific_args_schema(spec)

        def execute_scientific_tool(
            local_intent: Annotated[
                str,
                "One brief statement of the immediate technical purpose of this call; recorded for observation only.",
            ],
            input_from: list[str],
            source_fields: BaseModel | Mapping[str, str | None],
            inline_observations: list[BaseModel | Mapping[str, Any]],
            runtime: ToolRuntime[dict[str, Any], dict[str, Any]],
            **parameters: Any,
        ) -> Command:
            """Execute one registered scientific transformation."""

            technical_task = runtime.state.get("technical_task")
            if not isinstance(technical_task, dict):
                raise ValueError("scientific tool requires a current technical task")
            item_id = _technical_item_id(technical_task)
            task_id = str(technical_task.get("task_id") or "")
            visible_record_ids = _visible_objective_record_ids(
                runtime.state.get("messages", [])
            )
            records = [
                item
                for item in _objective_records(runtime.state, item_id)
                if (
                    str(item.get("task_id") or "") == task_id
                    or str(item.get("record_id") or "") in visible_record_ids
                )
            ]
            clean_parameters = {
                key: value for key, value in parameters.items() if value is not None
            }
            input_from = list(
                dict.fromkeys(str(item) for item in input_from if item)
            )
            plain_source_fields = cast(
                JSON,
                _plain(
                    source_fields.model_dump()
                    if isinstance(source_fields, BaseModel)
                    else dict(source_fields)
                ),
            )
            fields = cast(
                list[ScientificFieldSelection],
                [
                    {"role": role, "source_field": str(source_field)}
                    for role, source_field in plain_source_fields.items()
                    if source_field not in (None, "")
                ],
            )
            plain_inline_observations = [
                cast(
                    JSON,
                    _plain(
                        observation.model_dump()
                        if isinstance(observation, BaseModel)
                        else dict(observation)
                    ),
                )
                for observation in inline_observations
            ]

            records_by_id = {item["record_id"]: item for item in records}
            missing_ids = [item for item in input_from if item not in records_by_id]
            if missing_ids:
                raise ValueError(
                    "input_from must reference completed calls visible in the current R context: "
                    f"{missing_ids}"
                )
            selected_sources = DataRegistry()
            data_records = [
                records_by_id[record_id]
                for record_id in input_from
                if records_by_id[record_id].get("record_kind") == "data_result"
            ]
            working_rows = [
                row
                for record in data_records
                for row in _select_scientific_fields(
                    infrastructure._data_result_rows(record),
                    fields,
                    str(record.get("source_id") or ""),
                )
            ]
            working_rows.extend(
                {
                    key: value
                    for key, value in observation.items()
                    if value not in (None, "")
                }
                for observation in plain_inline_observations
            )
            if working_rows:
                source_ids = sorted(
                    {str(record.get("source_id") or "") for record in data_records}
                    - {""}
                )
                source_id = (
                    source_ids[0]
                    if len(source_ids) == 1 and not plain_inline_observations
                    else (
                        f"inline:{runtime.tool_call_id}"
                        if not data_records
                        else f"working-set:{hashlib.sha256('|'.join(input_from).encode()).hexdigest()[:12]}"
                    )
                )
                versions = sorted(
                    {
                        str(record.get("result", {}).get("data_versions", {}).get(item) or "")
                        for record in data_records
                        for item in record.get("result", {}).get("data_versions", {})
                    }
                    - {""}
                )
                version = versions[0] if len(versions) == 1 else runtime.tool_call_id
                selected_sources.register(
                    InMemoryObservationSource(
                        working_rows,
                        source_id=source_id,
                        version=version,
                        description=(
                            "Execution-scoped values selected by R from prior raw data.tool "
                            "results in the current ReAct episode."
                        ),
                    )
                )
                input_scope = {}
            else:
                source_id = ""
                input_scope = {}
                if tool_consumes_observation_rows(spec):
                    raise ValueError(
                        "this scientific tool needs observations; call any suitable "
                        "data-producing tool first and cite that completed call in input_from, "
                        "or provide inline observations already present in context"
                    )

            context = _tool_context(
                state=runtime.state,
                source_registry=selected_sources,
                external_registry=infrastructure.external_registry,
                source_id=source_id,
                input_scope=input_scope,
                parameters=clean_parameters,
                action_id=runtime.tool_call_id,
                prior_record_ids=input_from,
            )
            context.action.tool_id = spec.tool_id
            result = infrastructure.tool_registry.validate_result(spec, spec.runner(context))
            return _record_and_message(
                runtime=runtime,
                local_intent=local_intent,
                tool_name=tool_name,
                native_tool_id=spec.tool_id,
                source_id=source_id,
                input_scope=input_scope,
                parameters={
                    "input_from": input_from,
                    "source_fields": plain_source_fields,
                    "inline_observations": plain_inline_observations,
                    **clean_parameters,
                },
                upstream_record_ids=input_from,
                result=result,
            )

        requirements = [
            _react_capability_text(
                getattr(item, "description", None)
                or getattr(item, "requirement_id", None)
                or item
            )
            for item in spec.input_requirements
        ]
        description = (
            f"{spec.card.purpose or spec.description} "
            f"Scientific roles: {list(_scientific_roles(spec))}. "
            f"Requirements: {requirements}."
        )
        result = tool(
            tool_name,
            description=description,
            args_schema=args_schema,
        )(execute_scientific_tool)
        result.metadata = {
            "native_tool_id": spec.tool_id,
            "capability_group": spec.capability_group,
            "capabilities": list(spec.capabilities),
            "output_tags": list(spec.output_tags),
            "input_objects": [_react_capability_text(item) for item in spec.card.input_objects],
            "output_objects": [_react_capability_text(item) for item in spec.card.output_objects],
            "hard_requirements": [
                _react_capability_text(getattr(item, "description", item))
                for item in spec.input_requirements
            ],
            "data_access_role": "prior_call_or_inline_consumer",
        }
        return result


__all__ = [
    "DEFAULT_DATABASE_PATHS",
    "ObjectiveObservation",
    "ScientificFieldSelection",
    "ScientificInfrastructure",
    "TemporaryToolParameter",
    "WorkspaceRequest",
]
