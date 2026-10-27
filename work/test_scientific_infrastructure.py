from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
from operator import add
from typing import Annotated

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from scientific_infrastructure import ScientificInfrastructure


class ToolState(TypedDict):
    question: str
    technical_task: dict
    messages: Annotated[list, add_messages]
    objective_ledger: Annotated[list[dict], add]
    evidence_ledger: Annotated[list[dict], add]
    report_ledger: Annotated[list[dict], add]
    archive_index: Annotated[list[dict], add]


def workspace_request(operation: str, **overrides):
    request = {"operation": operation, "table_id": ""}
    if operation in {"profile", "sample", "query"}:
        request.update(columns=[], filters=[], row_limit=100)
    elif operation == "aggregate":
        request.update(
            columns=[],
            filters=[],
            group_by=[],
            aggregates=[],
            bin_field=None,
            bin_width=None,
            bin_origin=None,
            row_limit=100,
        )
    elif operation == "relate":
        request.update(
            related_table_id="",
            left_key="",
            right_key="",
            join_type="inner",
            match_policy="exact",
            match_tolerance=None,
            semantic_authorization=None,
            row_limit=100,
        )
    request.update(overrides)
    return request


def scientific_input(
    *,
    roles=("age_ma", "age_error_ma"),
    input_from=None,
    fields=None,
    inline_observations=None,
):
    field_map = {
        item["role"]: item["source_field"]
        for item in (fields or [])
    }
    return {
        "input_from": list(input_from or []),
        "source_fields": {role: field_map.get(role) for role in roles},
        "inline_observations": [
            {role: item.get(role) for role in roles}
            for item in (inline_observations or [])
        ],
    }


def inline_observation(record_id: str, age_ma: float, age_error_ma: float):
    return {
        "age_ma": age_ma,
        "age_error_ma": age_error_ma,
    }


def age_fields():
    return [
        {"role": "age_ma", "source_field": "event_age"},
        {"role": "age_error_ma", "source_field": "event_sigma"},
    ]


def call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": {"local_intent": f"test intent for {name}", **args},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


class ScientificInfrastructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary.name) / "observations.sqlite"
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute(
                "CREATE TABLE events "
                "(event_age REAL, event_sigma REAL, latitude REAL, longitude REAL, optional_age REAL)"
            )
            connection.executemany(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                [
                    (100.0, 1.0, 25.0, 110.0, 100.0),
                    (102.0, 1.0, 25.2, 110.2, 102.0),
                    (104.0, 2.0, 25.4, 110.4, None),
                    (106.0, 2.0, 25.6, 110.6, 106.0),
                    (108.0, 2.0, 25.8, 110.8, 108.0),
                    (110.0, 2.0, 26.0, 111.0, 110.0),
                ],
            )
            connection.execute(
                "CREATE TABLE localities "
                "(locality_name TEXT, province TEXT, latitude REAL, longitude REAL)"
            )
            connection.executemany(
                "INSERT INTO localities VALUES (?, ?, ?, ?)",
                [
                    ("Alpha", "Guangdong", 23.1, 113.3),
                    ("Beta", "Zhejiang", 30.3, 120.2),
                ],
            )
            connection.commit()
        self.infrastructure = ScientificInfrastructure.from_database_paths(
            [self.database],
            source_ids={str(self.database.resolve()): "observed_events"},
        )
        self.tools = self.infrastructure.langchain_tools()
        self.tools_by_name = {item.name: item for item in self.tools}
        self.node = ToolNode(self.tools)
        builder = StateGraph(ToolState)
        builder.add_node("tools", self.node)
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        self.graph = builder.compile()
        self.technical_task = {
            "task_id": "task-db-1",
            "item_id": "item-db-1",
            "technical_question": "计算已登记事件的加权年龄",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _invoke(self, state: dict, message: AIMessage) -> dict:
        state["messages"].append(message)
        before_messages = len(state["messages"])
        before_records = len(state["objective_ledger"])
        result = self.graph.invoke(state)
        state.clear()
        state.update(result)
        return {
            "messages": result["messages"][before_messages:],
            "objective_ledger": result["objective_ledger"][before_records:],
        }

    def test_strict_tools_expose_real_registry(self) -> None:
        # Package-backed providers are exposed only when their actual runtime
        # exists; the native data/statistical core is always present.
        self.assertGreaterEqual(len(self.tools), 24)
        self.assertIn("data__tool", self.tools_by_name)
        self.assertNotIn("data__table_retriever", self.tools_by_name)
        self.assertIn("archive__read", self.tools_by_name)
        self.assertIn("sandbox__python", self.tools_by_name)
        self.assertIn("time__weighted_age", self.tools_by_name)
        self.assertIn("time__dip", self.tools_by_name)
        for item in self.tools:
            schema = convert_to_openai_tool(item, strict=True)["function"]["parameters"]
            self.assertFalse(schema["additionalProperties"])
            self.assertIn("local_intent", schema["properties"])
            self.assertIn("local_intent", schema["required"])
            if item.name not in {
                "data__tool",
                "archive__read",
                "sandbox__python",
            }:
                self.assertEqual(
                    {
                        "local_intent",
                        "input_from",
                        "source_fields",
                        "inline_observations",
                    }
                    | ({"eps_km"} if item.name == "space__dbscan" else set()),
                    set(schema["properties"]),
                )
                serialized = json.dumps(schema)
                self.assertNotIn("registered_query", serialized)
                self.assertNotIn("source_id\"", serialized)
                self.assertNotIn("table_id", serialized)
                self.assertNotIn("filters", serialized)
                self.assertNotIn("bindings", serialized)
                self.assertNotIn("data_handle_record_id", schema["properties"])
                self.assertNotIn("change_point_penalty", serialized)
                self.assertNotIn("spatial_claim_scale_km", serialized)
            elif item.name == "sandbox__python":
                self.assertEqual(
                    set(schema["properties"]),
                    {
                        "local_intent",
                        "temporary_tool_name",
                        "purpose",
                        "parameter_schema",
                        "arguments_json",
                        "input_from",
                        "code",
                    },
                )
                self.assertEqual(
                    item.metadata["capability_group"],
                    "temporary_python_operations",
                )
                self.assertIn("global toolbox remains unchanged", item.description)
                self.assertIn("Good example", item.description)
                self.assertIn("Forbidden example", item.description)
            elif item.name == "data__tool":
                self.assertEqual(
                    item.description,
                    "Read registered tables and execute bounded, read-only SQL-style data operations and "
                    "common descriptive statistics. It returns objective rows or compact computed results "
                    "without assigning scientific meaning. This is an ordinary peer tool.",
                )
                self.assertEqual(
                    item.metadata["data_access_role"],
                    "relational_data_operator",
                )
                serialized = json.dumps(schema)
                self.assertNotIn("catalogue", serialized)
                self.assertNotIn("materialize", serialized)
                self.assertNotIn("dataset_name", serialized)
                self.assertNotIn("input_artifact_ids", serialized)
                request_variants = schema["properties"]["request"]["anyOf"]
                operations = {
                    operation
                    for variant in request_variants
                    for operation in (
                        [variant["properties"]["operation"]["const"]]
                        if "const" in variant["properties"]["operation"]
                        else variant["properties"]["operation"]["enum"]
                    )
                }
                self.assertEqual(
                    operations,
                    {"inspect", "profile", "sample", "query", "aggregate", "relate"},
                )
                inspect_variant = next(
                    item
                    for item in request_variants
                    if item["properties"]["operation"].get("const") == "inspect"
                )
                self.assertEqual(set(inspect_variant["properties"]), {"operation", "table_id"})
                read_variant = next(
                    item
                    for item in request_variants
                    if "query" in item["properties"]["operation"].get("enum", [])
                )
                self.assertEqual(
                    set(read_variant["properties"]),
                    {"operation", "table_id", "columns", "filters", "row_limit"},
                )
                aggregate_variant = next(
                    item
                    for item in request_variants
                    if item["properties"]["operation"].get("const") == "aggregate"
                )
                self.assertNotIn("related_table_id", aggregate_variant["properties"])
            else:
                self.assertEqual(
                    set(schema["properties"]),
                    {
                        "local_intent",
                        "entry_ids",
                        "entry_kinds",
                        "task_ids",
                        "limit",
                    },
                )
                limit_schema = schema["properties"]["limit"]
                self.assertEqual(limit_schema["minimum"], 1)
                self.assertEqual(limit_schema["maximum"], 20)
                self.assertIn("required", limit_schema["description"].lower())
                self.assertIn("1 through 20", item.description)

        self.assertNotIn(
            "sandbox.python",
            {spec.tool_id for spec in self.infrastructure.tool_registry.all()},
        )

        kde_schema = convert_to_openai_tool(
            self.tools_by_name["time__kde"], strict=True
        )["function"]["parameters"]
        self.assertEqual(
            set(kde_schema["properties"]["source_fields"]["properties"]),
            {"age_ma"},
        )
        self.assertEqual(
            set(
                kde_schema["properties"]["inline_observations"]["items"][
                    "properties"
                ]
            ),
            {"age_ma"},
        )
        weighted_schema = convert_to_openai_tool(
            self.tools_by_name["time__weighted_age"], strict=True
        )["function"]["parameters"]
        self.assertEqual(
            set(weighted_schema["properties"]["source_fields"]["properties"]),
            {"age_ma", "age_error_ma"},
        )
        dbscan_schema = convert_to_openai_tool(
            self.tools_by_name["space__dbscan"], strict=True
        )["function"]["parameters"]
        self.assertEqual(
            set(dbscan_schema["properties"])
            - {"local_intent", "input_from", "source_fields", "inline_observations"},
            {"eps_km"},
        )

    def test_python_sandbox_retains_many_definitions_without_mutating_registry(self) -> None:
        state = {
            "question": "计算两个临时统计量",
            "technical_task": {
                "task_id": "task-sandbox-1",
                "target_epistemic_item": "item-sandbox-1",
                "technical_objective": "计算两个缺失的原子统计量",
            },
            "messages": [],
            "objective_ledger": [],
            "archive_index": [],
        }
        native_registry_before = {
            spec.tool_id for spec in self.infrastructure.tool_registry.all()
        }
        first = self._invoke(
            state,
            call(
                "sandbox__python",
                {
                    "temporary_tool_name": "numeric_span",
                    "purpose": "Return the span of one numeric vector.",
                    "parameter_schema": [
                        {
                            "name": "values",
                            "value_type": "number_array",
                            "description": "Numeric observations.",
                            "required": True,
                        }
                    ],
                    "arguments_json": '{"values": [2, 5, 11]}',
                    "input_from": [],
                    "code": 'values = arguments["values"]\nresult = {"span": max(values) - min(values)}',
                },
                "call-sandbox-span",
            ),
        )
        second = self._invoke(
            state,
            call(
                "sandbox__python",
                {
                    "temporary_tool_name": "root_mean_square",
                    "purpose": "Return the root mean square of one numeric vector.",
                    "parameter_schema": [
                        {
                            "name": "values",
                            "value_type": "number_array",
                            "description": "Numeric observations.",
                            "required": True,
                        }
                    ],
                    "arguments_json": '{"values": [3, 4]}',
                    "input_from": [],
                    "code": 'values = np.asarray(arguments["values"], dtype=float)\nresult = {"rms": float(np.sqrt(np.mean(values ** 2)))}',
                },
                "call-sandbox-rms",
            ),
        )

        self.assertEqual(first["objective_ledger"][0]["item_id"], "item-sandbox-1")
        self.assertEqual(
            first["objective_ledger"][0]["result"]["findings"]["output"],
            {"span": 9},
        )
        self.assertEqual(
            second["objective_ledger"][0]["result"]["findings"]["output"],
            {"rms": 3.5355339059327378},
        )
        definitions = [
            record["parameters"]["temporary_tool"]
            for record in state["objective_ledger"]
            if record["tool_id"] == "sandbox.python"
        ]
        self.assertEqual(
            [definition["name"] for definition in definitions],
            ["numeric_span", "root_mean_square"],
        )
        self.assertTrue(all(definition["code"] for definition in definitions))
        self.assertTrue(
            all(definition["registered_globally"] is False for definition in definitions)
        )
        self.assertEqual(
            {spec.tool_id for spec in self.infrastructure.tool_registry.all()},
            native_registry_before,
        )

    def test_failed_temporary_python_definition_remains_auditable(self) -> None:
        state = {
            "question": "运行一个失败的临时计算",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
            "archive_index": [],
        }
        attempted_code = 'result = {"value": 1 / 0}'
        invoked = self._invoke(
            state,
            call(
                "sandbox__python",
                {
                    "temporary_tool_name": "failing_ratio",
                    "purpose": "Return one ratio.",
                    "parameter_schema": [],
                    "arguments_json": "{}",
                    "input_from": [],
                    "code": attempted_code,
                },
                "call-sandbox-failure",
            ),
        )

        record = invoked["objective_ledger"][0]
        self.assertEqual(record["result"]["status"], "error")
        self.assertEqual(record["parameters"]["temporary_tool"]["code"], attempted_code)
        self.assertIn("ZeroDivisionError", record["result"]["findings"]["error"])
        tool_message = next(
            message for message in invoked["messages"] if isinstance(message, ToolMessage)
        )
        self.assertEqual(tool_message.status, "error")

    def test_python_sandbox_consumes_prior_tool_results_by_reference(self) -> None:
        state = {
            "question": "读取后计算当前样本均值",
            "technical_task": {
                "task_id": "task-sandbox-data",
                "target_epistemic_item": "item-sandbox-data",
                "technical_objective": "计算当前样本均值",
            },
            "messages": [],
            "objective_ledger": [],
            "archive_index": [],
        }
        self._invoke(
            state,
            call(
                "data__tool",
                {
                    "source_id": "observed_events",
                    "request": workspace_request(
                        "query",
                        table_id="events",
                        columns=["event_age"],
                    ),
                },
                "call-sandbox-input-data",
            ),
        )
        computed = self._invoke(
            state,
            call(
                "sandbox__python",
                {
                    "temporary_tool_name": "event_age_mean",
                    "purpose": "Return the arithmetic mean of event_age.",
                    "parameter_schema": [],
                    "arguments_json": "{}",
                    "input_from": ["call-sandbox-input-data"],
                    "code": (
                        'rows = inputs[0]["result"]["findings"]["records"]\n'
                        'values = [row["event_age"] for row in rows]\n'
                        'result = {"mean": statistics.mean(values)}'
                    ),
                },
                "call-sandbox-input-mean",
            ),
        )

        record = computed["objective_ledger"][0]
        self.assertEqual(record["upstream_record_ids"], ["call-sandbox-input-data"])
        self.assertEqual(record["result"]["findings"]["output"], {"mean": 105.0})

    def test_registered_sources_project_only_coarse_source_owned_descriptions(self) -> None:
        catalogue = self.infrastructure.registered_data_sources()

        self.assertEqual(len(catalogue), 1)
        source = catalogue[0]
        self.assertEqual(source["source_id"], "observed_events")
        self.assertEqual(source["name"], "observations.sqlite")
        self.assertEqual(
            source["description"],
            "Dynamically navigated read-only tabular observations.",
        )
        self.assertEqual(
            source["tables"],
            [
                {"table_id": "events", "description": ""},
                {"table_id": "localities", "description": ""},
            ],
        )
        self.assertNotIn("fields", json.dumps(catalogue))
        self.assertNotIn("rows", json.dumps(catalogue))

    def test_known_table_inspect_returns_physical_schema(self) -> None:
        state = {
            "question": "events 表有哪些物理字段？",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        inspected = self._invoke(
            state,
            call(
                "data__tool",
                {
                    "source_id": "observed_events",
                    "request": workspace_request("inspect", table_id="events"),
                },
                "call-inspect-known-table",
            ),
        )
        record = inspected["objective_ledger"][0]
        self.assertEqual(record["tool_id"], "data.tool")
        self.assertEqual(record["record_kind"], "data_operation")
        self.assertEqual(record["input_scope"], {})
        table = record["result"]["findings"]["table"]
        self.assertEqual(table["table_id"], "events")
        self.assertEqual(
            {item["column_id"] for item in table["columns"]},
            {"event_age", "event_sigma", "latitude", "longitude", "optional_age"},
        )

    def test_data_read_returns_physical_rows_without_bindings(self) -> None:
        state = {
            "question": "读取 events 表",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        selected = self._invoke(
            state,
            call(
                "data__tool",
                {
                    "source_id": "observed_events",
                    "request": workspace_request(
                        "query",
                        table_id="events",
                        columns=["event_age", "event_sigma"],
                    ),
                },
                "call-physical-query",
            ),
        )
        findings = selected["objective_ledger"][0]["result"]["findings"]
        self.assertEqual(findings["physical_fields"], ["event_age", "event_sigma"])
        self.assertEqual(
            findings["records"][0],
            {"event_age": 100.0, "event_sigma": 1.0},
        )
        self.assertNotIn("age_ma", findings["records"][0])

    def test_count_field_counts_only_non_null_values(self) -> None:
        state = {
            "question": "optional_age 有多少个非空值？",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        aggregated = self._invoke(
            state,
            call(
                "data__tool",
                {
                    "source_id": "observed_events",
                    "request": workspace_request(
                        "aggregate",
                        table_id="events",
                        aggregates=[
                            {
                                "field": "optional_age",
                                "operation": "count",
                                "alias": "non_null_optional_age",
                            }
                        ],
                    ),
                },
                "call-count-non-null",
            ),
        )
        findings = aggregated["objective_ledger"][0]["result"]["findings"]
        self.assertEqual(findings["input_record_count"], 6)
        self.assertEqual(findings["rows"], [{"non_null_optional_age": 5}])
        self.assertEqual(
            aggregated["objective_ledger"][0]["parameters"]["request"]["aggregates"],
            [
                {
                    "field": "optional_age",
                    "operation": "count",
                    "as": "non_null_optional_age",
                }
            ],
        )

    def test_scientific_tool_never_infers_fields_from_raw_data(self) -> None:
        state = {
            "question": "计算 events 表的加权年龄",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        self._invoke(
            state,
            call(
                "data__tool",
                {
                    "source_id": "observed_events",
                    "request": workspace_request(
                        "query",
                        table_id="events",
                        columns=["event_age", "event_sigma"],
                    ),
                },
                "call-raw-without-field-choice",
            ),
        )
        before_records = len(state["objective_ledger"])
        with self.assertRaisesRegex(
            ValueError,
            "scientific_input.fields is required",
        ):
            self._invoke(
                state,
                call(
                    "time__weighted_age",
                    scientific_input(
                        input_from=["call-raw-without-field-choice"]
                    ),
                    "call-without-field-choice",
                ),
            )
        self.assertEqual(len(state["objective_ledger"]), before_records)

    def test_scientific_tool_accepts_inline_values_without_table_access(self) -> None:
        state = {
            "question": "数据库里的加权年龄是多少？",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        registered_source = self.infrastructure.source_registry.get("observed_events")
        original_query = registered_source.query

        def reject_registered_query(_: dict) -> list[dict]:
            raise AssertionError("inline scientific values must not query a registered table")

        registered_source.query = reject_registered_query
        try:
            calculation = self._invoke(
                state,
                call(
                    "time__weighted_age",
                    scientific_input(
                        inline_observations=[
                            inline_observation("inline-1", 100.0, 1.0),
                            inline_observation("inline-2", 102.0, 1.0),
                            inline_observation("inline-3", 104.0, 2.0),
                            inline_observation("inline-4", 106.0, 2.0),
                            inline_observation("inline-5", 108.0, 2.0),
                            inline_observation("inline-6", 110.0, 2.0),
                        ],
                    ),
                    "call-direct-weighted-age",
                ),
            )
        finally:
            registered_source.query = original_query
        self.assertEqual(len(calculation["objective_ledger"]), 1)
        calculation_record = calculation["objective_ledger"][0]
        result = calculation_record["result"]
        self.assertEqual(result["tool_id"], "time.weighted_age")
        self.assertEqual(result["calculation_status"], "computed")
        self.assertAlmostEqual(result["findings"]["weighted_age_ma"], 103.0, places=6)
        self.assertEqual(calculation_record["upstream_record_ids"], [])
        self.assertEqual(calculation_record["record_kind"], "scientific_result")
        self.assertTrue(calculation_record["source_id"].startswith("inline:"))

    def test_dip_tool_is_an_atomic_unimodality_test(self) -> None:
        state = {
            "question": "这些年龄是否偏离单峰分布？",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        ages = [100.0, 100.1, 100.2, 100.3, 109.7, 109.8, 109.9, 110.0]
        calculation = self._invoke(
            state,
            call(
                "time__dip",
                scientific_input(
                    roles=("age_ma",),
                    inline_observations=[
                        inline_observation(f"dip-{index}", age, 1.0)
                        for index, age in enumerate(ages)
                    ],
                ),
                "call-dip",
            ),
        )
        result = calculation["objective_ledger"][0]["result"]
        self.assertEqual(result["tool_id"], "time.dip")
        self.assertEqual(result["calculation_status"], "computed")
        self.assertEqual(result["findings"]["n"], len(ages))
        self.assertGreaterEqual(result["findings"]["p_value"], 0.0)
        self.assertLessEqual(result["findings"]["p_value"], 1.0)
        self.assertEqual(result["findings"]["library"], "diptest 0.11.0")
        self.assertNotIn("peak", result["findings"])
        self.assertNotIn("reject_null", result["findings"])

    def test_archive_read_is_optional_flat_current_agenda_retrieval(self) -> None:
        state = {
            "question": "回看当前事项此前的计算记录",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
            "evidence_ledger": [],
            "report_ledger": [],
            "archive_index": [],
        }
        self._invoke(
            state,
            call(
                "time__weighted_age",
                scientific_input(
                    inline_observations=[
                        inline_observation("archive-1", 100.0, 1.0),
                        inline_observation("archive-2", 102.0, 1.0),
                    ]
                ),
                "call-archived-weighted-age",
            ),
        )
        self.assertEqual(
            [item["archive_id"] for item in state["archive_index"]],
            ["tool:call-archived-weighted-age"],
        )
        retrieved = self._invoke(
            state,
            call(
                "archive__read",
                {
                    "entry_ids": [],
                    "entry_kinds": ["tool_result"],
                    "task_ids": [],
                    "limit": 10,
                },
                "call-read-archive",
            ),
        )
        result = retrieved["objective_ledger"][0]["result"]
        entries = result["findings"]["entries"]
        self.assertEqual(result["tool_id"], "archive.read")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["archive_id"], "tool:call-archived-weighted-age")
        self.assertEqual(entries[0]["content"]["tool_id"], "time.weighted_age")
        self.assertEqual(
            [item["archive_id"] for item in state["archive_index"]],
            ["tool:call-archived-weighted-age"],
        )

    def test_react_query_result_flows_into_later_scientific_calls(self) -> None:
        state = {
            "question": "数据库里的加权年龄是多少？",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        selected = self._invoke(
            state,
            call(
                "data__tool",
                {
                    "source_id": "observed_events",
                    "request": workspace_request(
                        "query",
                        table_id="events",
                        columns=["event_age", "event_sigma", "latitude", "longitude"],
                    ),
                },
                "call-query",
            ),
        )
        data_record = selected["objective_ledger"][0]
        self.assertEqual(data_record["result"]["findings"]["record_count"], 6)
        self.assertEqual(data_record["record_kind"], "data_result")
        data_message = selected["messages"][0].content
        self.assertNotIn("dataset_handle", data_message)
        public_rows = data_record["result"]["findings"]["records"]
        self.assertEqual(len(public_rows), 6)
        self.assertEqual(public_rows[0]["event_age"], 100.0)
        self.assertIn('"records"', data_message)
        self.assertIn('"event_age"', data_message)
        self.assertNotIn('"age_ma"', data_message)

        registered_source = self.infrastructure.source_registry.get("observed_events")
        original_query = registered_source.query

        def reject_second_table_read(_: dict) -> list[dict]:
            raise AssertionError("the scientific tool must consume the prior execution value")

        registered_source.query = reject_second_table_read
        try:
            calculation = self._invoke(
                state,
                call(
                    "time__weighted_age",
                    scientific_input(
                        input_from=["call-query"],
                        fields=age_fields(),
                    ),
                    "call-weighted-age",
                ),
            )
        finally:
            registered_source.query = original_query
        calculation_record = calculation["objective_ledger"][0]
        result = calculation_record["result"]
        self.assertEqual(result["tool_id"], "time.weighted_age")
        self.assertEqual(result["calculation_status"], "computed")
        self.assertAlmostEqual(result["findings"]["weighted_age_ma"], 103.0, places=6)
        self.assertEqual(calculation_record["upstream_record_ids"], ["call-query"])
        self.assertEqual(
            calculation_record["parameters"]["source_fields"],
            {"age_ma": "event_age", "age_error_ma": "event_sigma"},
        )
        payload = json.loads(calculation["messages"][0].content)
        self.assertEqual(payload["objective_record_id"], "call-weighted-age")
        self.assertEqual(payload["upstream_record_ids"], ["call-query"])

        kde = self._invoke(
            state,
            call(
                "time__kde",
                scientific_input(
                    roles=("age_ma",),
                    input_from=["call-query"],
                    fields=age_fields(),
                ),
                "call-kde",
            ),
        )
        kde_record = kde["objective_ledger"][0]
        self.assertEqual(kde_record["upstream_record_ids"], ["call-query"])
        self.assertEqual(kde_record["result"]["tool_id"], "time.kde")
        self.assertEqual(kde_record["result"]["calculation_status"], "computed")

        diagnostic = self._invoke(
            state,
            call(
                "time__bootstrap",
                scientific_input(
                    roles=("age_ma",),
                    input_from=["call-query", "call-weighted-age"],
                    fields=age_fields(),
                ),
                "call-bootstrap",
            ),
        )
        diagnostic_record = diagnostic["objective_ledger"][0]
        self.assertEqual(
            diagnostic_record["upstream_record_ids"],
            ["call-query", "call-weighted-age"],
        )
        self.assertEqual(
            diagnostic_record["result"]["calculation_status"],
            "computed",
        )

    def test_execution_value_is_replayed_from_public_tool_result(self) -> None:
        state = {
            "question": "数据库里的加权年龄是多少？",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        self._invoke(
            state,
            call(
                "data__tool",
                {
                    "source_id": "observed_events",
                    "request": workspace_request(
                        "query",
                        table_id="events",
                        columns=["event_age", "event_sigma"],
                    ),
                },
                "call-query-before-restart",
            ),
        )
        calculation = self._invoke(
            state,
            call(
                "time__weighted_age",
                scientific_input(
                    input_from=["call-query-before-restart"],
                    fields=age_fields(),
                ),
                "call-weighted-after-restart",
            ),
        )
        self.assertEqual(
            calculation["objective_ledger"][0]["result"]["calculation_status"],
            "computed",
        )

    def test_visible_prior_result_can_be_consumed_by_a_followup_r_question(self) -> None:
        state = {
            "question": "先读取数据，再在依赖它的新问题中计算",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        selected = self._invoke(
            state,
            call(
                "data__tool",
                {
                    "source_id": "observed_events",
                    "request": workspace_request(
                        "query",
                        table_id="events",
                        columns=["event_age", "event_sigma"],
                    ),
                },
                "call-visible-prior-query",
            ),
        )
        prior_record = selected["objective_ledger"][0]
        state["technical_task"] = {
            "task_id": "task-db-2",
            "item_id": self.technical_task["item_id"],
            "technical_question": "使用上一轮数据计算加权年龄",
        }

        calculation = self._invoke(
            state,
            call(
                "time__weighted_age",
                scientific_input(
                    input_from=["call-visible-prior-query"],
                    fields=age_fields(),
                ),
                "call-followup-weighted-age",
            ),
        )
        self.assertEqual(
            calculation["objective_ledger"][0]["result"]["calculation_status"],
            "computed",
        )

        hidden_state = {
            "question": "旧结果未出现在当前 R 上下文",
            "technical_task": {
                "task_id": "task-db-3",
                "item_id": self.technical_task["item_id"],
                "technical_question": "尝试使用不可见旧结果",
            },
            "messages": [],
            "objective_ledger": [prior_record],
        }
        with self.assertRaisesRegex(ValueError, "visible in the current R context"):
            self._invoke(
                hidden_state,
                call(
                    "time__weighted_age",
                    scientific_input(
                        input_from=["call-visible-prior-query"],
                        fields=age_fields(),
                    ),
                    "call-hidden-prior-weighted-age",
                ),
            )

    def test_provider_filter_shape_is_adapted_before_native_execution(self) -> None:
        state = {
            "question": "只提取 104 Ma 的记录",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        filtered = self._invoke(
            state,
            call(
                "data__tool",
                {
                    "source_id": "observed_events",
                    "request": workspace_request(
                        "query",
                        table_id="events",
                        filters=[
                            {
                                "field": "event_age",
                                "operator": "eq",
                                "value": 104.0,
                            },
                            {
                                "field": "event_age",
                                "operator": "not_null",
                            }
                        ],
                    ),
                },
                "call-filtered-query",
            ),
        )
        record = filtered["objective_ledger"][0]
        self.assertEqual(record["result"]["findings"]["record_count"], 1)
        self.assertEqual(
            record["parameters"]["request"]["filters"],
            [
                {"field": "event_age", "operator": "eq", "text_values": [104.0]},
                {"field": "event_age", "operator": "not_null"},
            ],
        )

    def test_sqlite_filters_are_pushed_down_before_full_table_rows_are_read(self) -> None:
        source = self.infrastructure.source_registry.get("observed_events")
        backend = source.backend
        original_rows = backend.rows

        def reject_full_table_read(_: str):
            raise AssertionError("filtered data.tool must not read the full SQLite table")

        backend.rows = reject_full_table_read
        state = {
            "question": "只读取 104 Ma 的记录",
            "technical_task": self.technical_task,
            "messages": [],
            "objective_ledger": [],
        }
        try:
            filtered = self._invoke(
                state,
                call(
                    "data__tool",
                    {
                        "source_id": "observed_events",
                        "request": workspace_request(
                            "query",
                            table_id="events",
                            columns=["event_age", "event_sigma"],
                            filters=[
                                {
                                    "field": "event_age",
                                    "operator": "eq",
                                    "value": 104.0,
                                },
                                {
                                    "field": "event_sigma",
                                    "operator": "not_null",
                                },
                            ],
                        ),
                    },
                    "call-pushed-down-filter",
                ),
            )
        finally:
            backend.rows = original_rows

        record = filtered["objective_ledger"][0]
        self.assertEqual(record["result"]["findings"]["record_count"], 1)
        self.assertEqual(
            record["result"]["findings"]["records"][0]["event_age"],
            104.0,
        )


if __name__ == "__main__":
    unittest.main()
