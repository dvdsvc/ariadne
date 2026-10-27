from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
sys.path.insert(0, str(OUTPUTS))

import thin_epistemic_research_agent as agent


RAW_QUESTION = "RAW-SECRET-PENDULUM: 振幅是否影响单摆周期？"
TOOL_CALLS: list[str] = []
REGISTERED_DATA_SOURCES = [
    {
        "source_id": "pendulum_observations",
        "name": "Pendulum observations",
        "description": "Registered read-only pendulum measurements.",
        "tables": [
            {
                "table_id": "period_measurements",
                "description": "Amplitude and period observations.",
            }
        ],
    }
]


@tool
def pendulum_compute(local_intent: str, mode: str) -> dict[str, Any]:
    """Compute a registered pendulum diagnostic."""

    TOOL_CALLS.append(mode)
    return {"ok": True, "mode": mode}


@tool
def data__tool(local_intent: str, operation: str) -> dict[str, Any]:
    """Read registered objective data when technically useful."""

    TOOL_CALLS.append(f"data:{operation}")
    return {"ok": True, "operation": operation}


pendulum_compute.metadata = {
    "capability_group": "pendulum_analysis",
    "input_objects": ["registered pendulum observations"],
    "output_objects": ["group period difference", "leave-one-out stability"],
    "hard_requirements": ["amplitude and period observations are required"],
    "data_access_role": "prior_call_or_inline_consumer",
}
data__tool.metadata = {
    "native_tool_id": "data.tool",
    "capability_group": "data_tool",
    "input_objects": ["registered objective data"],
    "output_objects": ["execution-scoped data result"],
    "hard_requirements": ["a declared data boundary is required"],
    "data_access_role": "relational_data_operator",
}


INTERPRETATION_CONSTITUTION = {
    "research_object": "给定单摆数据中的振幅与周期关系",
    "boundaries": [],
    "requested_output": "振幅是否影响单摆周期",
}


CONSTITUTION = {
    "question": RAW_QUESTION,
    **INTERPRETATION_CONSTITUTION,
}


OPEN_ITEM = {
    "item_id": "period-contrast",
    "question": "现有记录中的振幅与周期之间呈现什么关系？",
    "status": "open",
    "current_understanding": "尚未形成技术结果",
    "evidence_refs": [],
}


BOOTSTRAP_MAP = {
    "revision": 0,
    "items": [OPEN_ITEM],
    "overall_state": {
        "current_answer": "尚未回答",
        "claim_ceiling": "尚未形成技术认识",
    },
}


BOOTSTRAP_SNAPSHOT = {
    "constitution": CONSTITUTION,
    "decision": "continue",
    "epistemic_map": BOOTSTRAP_MAP,
}


QUESTION_1 = "量化低振幅与高振幅记录中的周期差"
QUESTION_2 = "确定观察到的周期差是否由单条记录驱动"


DEPARTMENT_REPORT = {
    "report_id": "report-1",
    "item_id": OPEN_ITEM["item_id"],
    "what_is_known": "现有技术结果显示周期随振幅略增，且方向不由单条记录驱动",
    "core_evidence": [
        {
            "statement": "高振幅组周期较长且不由单一样本驱动",
            "source": "已登记单摆观测的分组差和留一诊断",
            "evidence_refs": ["packet-1", "packet-2"],
        }
    ],
    "conclusion_boundary": "仅适用于本次登记样本和已观测振幅范围",
    "remaining_uncertainty": ["更高振幅范围尚未观测"],
    "cross_item_implications": [],
}


DEPARTMENT_RESULTS = {
    key: value
    for key, value in DEPARTMENT_REPORT.items()
    if key not in {"report_id", "item_id"}
}


REVISED_MAP = {
    "revision": 1,
    "items": [
        {
            **OPEN_ITEM,
            "status": "resolved",
            "current_understanding": "现有技术结果显示周期随振幅略增，且方向不由单条记录驱动",
            "evidence_refs": ["packet-1", "packet-2"],
        }
    ],
    "overall_state": {
        "current_answer": "现有数据支持周期随振幅略增",
        "claim_ceiling": "结论限于已观测振幅范围",
    },
}


REVISED_SNAPSHOT = {
    "constitution": CONSTITUTION,
    "decision": "close",
    "epistemic_map": REVISED_MAP,
}


INQUIRY_PACKAGE = {
    "item": OPEN_ITEM,
    "constitution": CONSTITUTION,
}


STRUCTURED_SCRIPT = [
    ("InterpretationOutput", {"constitution": INTERPRETATION_CONSTITUTION}),
    (
        "UpdaterOutput",
        {key: value for key, value in BOOTSTRAP_SNAPSHOT.items() if key != "constitution"},
    ),
    ("PlannerSelection", {"selected_item_id": OPEN_ITEM["item_id"]}),
    (
        "UpdaterOutput",
        {key: value for key, value in REVISED_SNAPSHOT.items() if key != "constitution"},
    ),
    ("WriterOutput", {"answer": "现有数据支持周期随振幅略增，但结论仅限已观测范围。"}),
]


def tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
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


def technical_results(results: list[str], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "TechnicalResults",
                "args": {"results": results},
                "id": call_id,
                "type": "tool_call",
            }
        ],
    )


D_TOOL_SCRIPT = [
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "delegate_to_researcher",
                "args": {
                    "technical_question": QUESTION_1,
                    "depends_on_previous_results": False,
                },
                "id": "delegate-1",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "delegate_to_researcher",
                "args": {
                    "technical_question": QUESTION_2,
                    "depends_on_previous_results": True,
                },
                "id": "delegate-2",
                "type": "tool_call",
            }
        ],
    ),
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "DepartmentResults",
                "args": DEPARTMENT_RESULTS,
                "id": "department-results-1",
                "type": "tool_call",
            }
        ],
    ),
]


R_TOOL_SCRIPT = [
    tool_call("pendulum_compute", {"mode": "group_difference"}, "tool-1"),
    technical_results(["高振幅组周期略长"], "technical-results-1"),
    tool_call("pendulum_compute", {"mode": "leave_one_out"}, "tool-3"),
    technical_results(["留一结果方向一致"], "technical-results-2"),
]


class BoundStructured:
    def __init__(self, adapter: "ScriptedAdapter", schema: type) -> None:
        self.adapter = adapter
        self.schema = schema

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        expected, result = self.adapter.structured_script.pop(0)
        if expected != self.schema.__name__:
            raise AssertionError(f"expected {expected}, got {self.schema.__name__}")
        self.adapter.structured_invocations.append(
            {"schema": self.schema.__name__, "messages": messages}
        )
        if isinstance(result, BaseException):
            raise result
        return result


class ScriptedReactModel(BaseChatModel):
    adapter: Any

    @property
    def _llm_type(self) -> str:
        return "scripted-react"

    def _generate(self, *_: Any, **__: Any) -> ChatResult:
        raise AssertionError("scripted ReAct model must be invoked through bind_tools")

    def bind_tools(
        self,
        tools: list[Any],
        *,
        tool_choice: str | None = None,
        **_: Any,
    ) -> RunnableLambda:
        allowed = {
            convert_to_openai_tool(item)["function"]["name"] for item in tools
        }
        agent_role = (
            "department" if "delegate_to_researcher" in allowed else "researcher"
        )

        def invoke(messages: list[Any]) -> AIMessage:
            script = (
                self.adapter.department_tool_script
                if agent_role == "department"
                else self.adapter.researcher_tool_script
            )
            response = script.pop(0)
            if isinstance(response, BaseException):
                raise response
            names = {item["name"] for item in response.tool_calls}
            if not names <= allowed:
                raise AssertionError(f"{names} not allowed by {allowed}")
            self.adapter.tool_invocations.append(
                {
                    "messages": messages,
                    "allowed": allowed,
                    "tool_choice": tool_choice,
                    "agent_role": agent_role,
                }
            )
            return response

        return RunnableLambda(invoke)


class ScriptedAdapter:
    def __init__(self) -> None:
        self.structured_script = list(STRUCTURED_SCRIPT)
        self.department_tool_script = list(D_TOOL_SCRIPT)
        self.researcher_tool_script = list(R_TOOL_SCRIPT)
        self.structured_invocations: list[dict[str, Any]] = []
        self.tool_invocations: list[dict[str, Any]] = []
        self.bind_tools_kwargs: dict[str, Any] = {}
        self.model = ScriptedReactModel(adapter=self)

    def structured(
        self,
        schema: type,
        *,
        cache_scope: str = "unscoped",
    ) -> BoundStructured:
        del cache_scope
        return BoundStructured(self, schema)


class StructuredEnvelopeModel:
    def __init__(self, envelopes: list[dict[str, Any]]) -> None:
        self.envelopes = list(envelopes)
        self.structured_kwargs: list[dict[str, Any]] = []
        self.invoke_count = 0

    def with_structured_output(self, schema: type, **kwargs: Any) -> RunnableLambda:
        self.structured_kwargs.append(kwargs)

        def invoke(_: Any) -> dict[str, Any]:
            self.invoke_count += 1
            return self.envelopes.pop(0)

        return RunnableLambda(invoke)


class ThinAgentTest(unittest.TestCase):
    def test_deepseek_adapter_uses_128k_output_limit(self) -> None:
        adapter = agent.make_deepseek_adapter(
            api_key="test-key",
            model="deepseek-v4-flash",
        )

        self.assertEqual(agent.MAX_OUTPUT_TOKENS, 131_072)
        self.assertEqual(adapter.model.max_tokens, 131_072)
        self.assertEqual(agent.MODEL_REQUEST_TIMEOUT_SECONDS, 120.0)
        self.assertEqual(adapter.model.request_timeout, 120.0)
        self.assertEqual(adapter.model.max_retries, 0)

    def setUp(self) -> None:
        TOOL_CALLS.clear()

    def test_r_structured_output_contains_only_results(self) -> None:
        schema = agent.api_schemas()["technical_results"]

        self.assertEqual(set(schema["properties"]), {"results"})

    def test_nested_loops_share_an_information_gain_stop_criterion(self) -> None:
        self.assertNotIn("Call at least one appropriate tool", agent.RESEARCHER_PROMPT)
        self.assertIn("data__tool operates on registered tables", agent.RESEARCHER_PROMPT)
        self.assertNotIn("Required scientific inputs", agent.RESEARCHER_PROMPT)
        self.assertIn("expected result could change", agent.RESEARCHER_PROMPT)
        self.assertIn("call TechnicalResults to complete", agent.RESEARCHER_PROMPT)
        self.assertIn("expected to change that understanding", agent.DEPARTMENT_PROMPT)
        self.assertIn("could materially change", agent.UPDATER_PROMPT)
        self.assertIn("mark it resolved", agent.UPDATER_PROMPT)
        self.assertIn("or limited", agent.UPDATER_PROMPT)

    def test_local_package_contains_only_selected_item_and_constitution(self) -> None:
        package = agent._materialize_local_epistemic_package(
            BOOTSTRAP_SNAPSHOT, OPEN_ITEM
        )

        self.assertEqual(set(package), {"item", "constitution"})
        self.assertEqual(package["item"], OPEN_ITEM)
        self.assertEqual(package["constitution"], CONSTITUTION)

    def test_r_memory_projection_keeps_evidence_and_removes_runtime_noise(self) -> None:
        request = {
            "operation": "aggregate",
            "table_id": "period_measurements",
            "filters": [{"field": "amplitude", "operator": "not_null"}],
        }
        provenance = {
            "source_id": "pendulum_observations",
            "physical_inputs": {"table_id": "period_measurements"},
        }
        messages = [
            HumanMessage(content="上一轮技术问题", id="technical-task:prior"),
            AIMessage(
                content="这里是无需带入下一轮的自由思考",
                id="provider-run-id",
                response_metadata={"finish_reason": "tool_calls"},
                usage_metadata={
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                },
                tool_calls=[
                    {
                        "name": "data__tool",
                        "args": {"local_intent": "count", "request": request},
                        "id": "tool-1",
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(
                content=json.dumps(
                    {
                        "objective_record_id": "tool-1",
                        "result": {
                            "findings": {
                                "request": request,
                                "data_operation": {
                                    "request": request,
                                    "provenance": provenance,
                                },
                                "provenance": provenance,
                                "record_count": 120,
                            }
                        },
                    }
                ),
                name="data__tool",
                tool_call_id="tool-1",
                id="tool-message-provider-id",
            ),
            technical_results(["得到 120 条记录"], "technical-results-1"),
            ToolMessage(
                content="Returning structured response",
                name="TechnicalResults",
                tool_call_id="technical-results-1",
            ),
        ]
        packet = {
            "packet_id": "packet-1",
            "task_id": "task-1",
            "item_id": OPEN_ITEM["item_id"],
            "results": ["得到 120 条记录"],
            "tool_call_ids": ["tool-1"],
        }

        projected = agent._project_researcher_messages(
            messages,
            completed_packet=packet,
        )

        self.assertEqual(len(projected), 4)
        projected_ai = next(item for item in projected if isinstance(item, AIMessage))
        self.assertEqual(projected_ai.content, "")
        self.assertIsNone(projected_ai.id)
        self.assertEqual(projected_ai.response_metadata, {})
        self.assertIsNone(projected_ai.usage_metadata)
        self.assertEqual(projected_ai.tool_calls[0]["args"]["request"], request)
        projected_tool = next(
            item for item in projected if isinstance(item, ToolMessage)
        )
        self.assertIsNone(projected_tool.id)
        tool_payload = json.loads(projected_tool.content)
        findings = tool_payload["result"]["findings"]
        self.assertEqual(findings["request"], request)
        self.assertEqual(findings["data_operation"]["provenance"], provenance)
        self.assertEqual(
            findings["data_operation"]["request"]["$ref"],
            "$.result.findings.request",
        )
        self.assertEqual(
            findings["provenance"]["$ref"],
            "$.result.findings.data_operation.provenance",
        )
        serialized = json.dumps(
            [item.model_dump() for item in projected],
            ensure_ascii=False,
        )
        self.assertNotIn("自由思考", serialized)
        self.assertNotIn("Returning structured response", serialized)
        self.assertNotIn("provider-run-id", serialized)
        self.assertIn("得到 120 条记录", projected[-1].content)

    def test_repetition_observation_counts_actions_and_compares_stable_results(self) -> None:
        def repeated_call(call_id: str, local_intent: str) -> AIMessage:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "data__tool",
                        "args": {
                            "local_intent": local_intent,
                            "source_id": "global_zircon_v2_1",
                            "request": {
                                "operation": "query",
                                "table_id": "zircon_records",
                                "filters": [
                                    {
                                        "field": "data_number",
                                        "operator": "eq",
                                        "text_values": [64201],
                                    }
                                ],
                            },
                        },
                        "id": call_id,
                        "type": "tool_call",
                    }
                ],
            )

        def repeated_result(call_id: str, value: str) -> ToolMessage:
            return ToolMessage(
                content=json.dumps(
                    {
                        "objective_record_id": call_id,
                        "result": {
                            "result_id": f"data.tool:{call_id}",
                            "status": "ok",
                            "findings": {"records": [{"data_number": 64201, "value": value}]},
                        },
                    }
                ),
                name="data__tool",
                tool_call_id=call_id,
            )

        unchanged_messages = [
            repeated_call("repeat-1", "first wording"),
            repeated_result("repeat-1", "same"),
            repeated_call("repeat-2", "different wording"),
            repeated_result("repeat-2", "same"),
        ]
        unchanged = agent._researcher_repetition_observation(unchanged_messages)
        self.assertEqual(
            unchanged["execution_repetition_observation"]["consecutive_repeat_count"],
            2,
        )
        self.assertFalse(
            unchanged["execution_repetition_observation"][
                "result_changed_across_repeats"
            ]
        )
        self.assertEqual(
            unchanged["execution_repetition_observation"][
                "new_information_from_latest_repeat"
            ],
            "none",
        )

        changed = agent._researcher_repetition_observation(
            [
                *unchanged_messages,
                repeated_call("repeat-3", "third wording"),
                repeated_result("repeat-3", "changed"),
            ]
        )
        self.assertEqual(
            changed["execution_repetition_observation"]["consecutive_repeat_count"],
            3,
        )
        self.assertTrue(
            changed["execution_repetition_observation"][
                "result_changed_across_repeats"
            ]
        )
        self.assertEqual(
            changed["execution_repetition_observation"][
                "new_information_from_latest_repeat"
            ],
            "result_changed",
        )

    def test_repetition_observation_is_injected_into_r_system_context(self) -> None:
        model = ScriptedAdapter()
        model.researcher_tool_script = [
            tool_call("pendulum_compute", {"mode": "same"}, "repeat-tool-1"),
            tool_call("pendulum_compute", {"mode": "same"}, "repeat-tool-2"),
            technical_results(["重复计算没有产生新结果"], "repeat-results"),
        ]
        researcher = agent._build_researcher_agent(
            [pendulum_compute],
            REGISTERED_DATA_SOURCES,
        )
        task = {
            "task_id": "task-repetition-observation",
            "item_id": OPEN_ITEM["item_id"],
            "technical_question": "计算一次即可回答的问题",
        }

        result = researcher.invoke(
            {
                "messages": [
                    HumanMessage(
                        id=f"technical-task:{task['task_id']}",
                        content=json.dumps({"technical_task": task}),
                    )
                ],
                "constitution": CONSTITUTION,
                "technical_task": task,
                "objective_ledger": [],
                "evidence_ledger": [],
                "report_ledger": [],
                "archive_index": [],
                "technical_packet": None,
            },
            context={"model": model},
            config={"recursion_limit": 20},
        )

        researcher_calls = [
            item
            for item in model.tool_invocations
            if item["agent_role"] == "researcher"
        ]
        self.assertEqual(len(researcher_calls), 3)
        self.assertNotIn(
            "execution_repetition_observation",
            researcher_calls[1]["messages"][0].content,
        )
        final_system_context = researcher_calls[2]["messages"][0].content
        self.assertIn("execution_repetition_observation", final_system_context)
        self.assertIn('"consecutive_repeat_count": 2', final_system_context)
        self.assertIn('"new_information_from_latest_repeat": "none"', final_system_context)
        self.assertEqual(
            result["technical_packet"]["results"],
            ["重复计算没有产生新结果"],
        )

    def test_fourth_identical_r_action_is_refused_without_running_the_tool(self) -> None:
        model = ScriptedAdapter()
        model.researcher_tool_script = [
            tool_call("pendulum_compute", {"mode": "same"}, "same-1"),
            tool_call("pendulum_compute", {"mode": "same"}, "same-2"),
            tool_call("pendulum_compute", {"mode": "same"}, "same-3"),
            tool_call("pendulum_compute", {"mode": "same"}, "same-blocked-1"),
            tool_call("pendulum_compute", {"mode": "different"}, "different-1"),
            tool_call("pendulum_compute", {"mode": "same"}, "same-blocked-2"),
            technical_results(["改用不同动作后完成当前技术认识"], "guard-results"),
        ]
        researcher = agent._build_researcher_agent(
            [pendulum_compute],
            REGISTERED_DATA_SOURCES,
        )
        task = {
            "task_id": "task-repetition-guard",
            "item_id": OPEN_ITEM["item_id"],
            "technical_question": "避免用完全相同的动作无限重复",
        }

        result = researcher.invoke(
            {
                "messages": [
                    HumanMessage(
                        id=f"technical-task:{task['task_id']}",
                        content=json.dumps({"technical_task": task}),
                    )
                ],
                "constitution": CONSTITUTION,
                "technical_task": task,
                "objective_ledger": [],
                "evidence_ledger": [],
                "report_ledger": [],
                "archive_index": [],
                "technical_packet": None,
            },
            context={"model": model},
            config={"recursion_limit": 30},
        )

        self.assertEqual(TOOL_CALLS, ["same", "same", "same", "different"])
        interruptions = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
            and message.additional_kwargs.get("control_interruption")
        ]
        self.assertEqual(len(interruptions), 2)
        self.assertTrue(all(message.status == "error" for message in interruptions))
        first_interruption = json.loads(interruptions[0].content)[
            "control_interruption"
        ]
        self.assertEqual(first_interruption["completed_repeat_count"], 3)
        self.assertIn("Do not execute it again", first_interruption["instruction"])
        self.assertEqual(
            result["technical_packet"]["tool_call_ids"],
            ["same-1", "same-2", "same-3", "different-1"],
        )

    def test_thin_context_boundaries_and_two_department_cycles(self) -> None:
        model = ScriptedAdapter()
        graph = agent.build_graph(
            [data__tool, pendulum_compute],
            checkpointer=InMemorySaver(),
            registered_data_sources=REGISTERED_DATA_SOURCES,
        )
        config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": "thin-context-test"},
        }
        result = graph.invoke(
            {"question": RAW_QUESTION},
            context={"model": model},
            config=config,
            durability="sync",
        )

        self.assertEqual(
            result,
            {"final_answer": "现有数据支持周期随振幅略增，但结论仅限已观测范围。"},
        )
        self.assertEqual(TOOL_CALLS, ["group_difference", "leave_one_out"])
        self.assertFalse(model.structured_script)
        self.assertFalse(model.department_tool_script)
        self.assertFalse(model.researcher_tool_script)

        schemas = [item["schema"] for item in model.structured_invocations]
        self.assertEqual(
            schemas,
            [
                "InterpretationOutput",
                "UpdaterOutput",
                "PlannerSelection",
                "UpdaterOutput",
                "WriterOutput",
            ],
        )

        updater_calls = [
            item for item in model.structured_invocations if item["schema"] == "UpdaterOutput"
        ]
        bootstrap_context = json.loads(updater_calls[0]["messages"][-1].content)
        revision_context = json.loads(updater_calls[1]["messages"][-1].content)
        self.assertEqual(
            set(bootstrap_context),
            {"mode", "constitution"},
        )
        self.assertEqual(
            set(revision_context),
            {"mode", "constitution", "current_u_snapshot", "department_report"},
        )
        self.assertEqual(revision_context["constitution"], CONSTITUTION)
        self.assertNotIn("constitution", revision_context["current_u_snapshot"])
        self.assertEqual(
            revision_context["current_u_snapshot"]["constitution_ref"],
            agent._constitution_digest(CONSTITUTION),
        )
        self.assertNotIn("technical_packets", json.dumps(revision_context))
        self.assertNotIn("tool", json.dumps(revision_context).lower())
        report = revision_context["department_report"]
        self.assertEqual(report["item_id"], OPEN_ITEM["item_id"])
        self.assertTrue(report["report_id"].startswith("report-"))
        self.assertEqual(
            {key: value for key, value in report.items() if key not in {"report_id", "item_id"}},
            DEPARTMENT_RESULTS,
        )

        planner_calls = [
            item for item in model.structured_invocations if item["schema"] == "PlannerSelection"
        ]
        for call in planner_calls:
            planner_context = json.loads(call["messages"][-1].content)
            self.assertEqual(set(planner_context), {"constitution", "u_snapshot"})
            self.assertEqual(planner_context["constitution"], CONSTITUTION)
            self.assertNotIn("constitution", planner_context["u_snapshot"])
            self.assertEqual(
                planner_context["u_snapshot"]["constitution_ref"],
                agent._constitution_digest(CONSTITUTION),
            )

        department_calls = [
            item
            for item in model.tool_invocations
            if item["agent_role"] == "department"
        ]
        researcher_calls = [
            item
            for item in model.tool_invocations
            if item["agent_role"] == "researcher"
        ]
        self.assertEqual(len(department_calls), 3)
        self.assertEqual(len(researcher_calls), 4)
        for call in department_calls:
            self.assertEqual(
                call["allowed"],
                {"capability__query", "delegate_to_researcher", "DepartmentResults"},
            )
            system_context = call["messages"][0].content
            self.assertIn(RAW_QUESTION, system_context)
            self.assertIn("SQL-style data operations", system_context)
            self.assertIn("mathematical or statistical computation", system_context)
            self.assertNotIn("pendulum_compute", system_context)
            self.assertNotIn("u_snapshot", system_context)
        self.assertFalse(
            D_TOOL_SCRIPT[0].tool_calls[0]["args"]["depends_on_previous_results"]
        )
        self.assertTrue(
            D_TOOL_SCRIPT[1].tool_calls[0]["args"]["depends_on_previous_results"]
        )
        self.assertTrue(
            any(
                isinstance(message, ToolMessage)
                and message.name == "delegate_to_researcher"
                for message in department_calls[1]["messages"]
            )
        )
        serialized_d_history = json.dumps(
            [str(message.content) for message in department_calls[-1]["messages"]],
            ensure_ascii=False,
        )
        self.assertNotIn("pendulum_compute", serialized_d_history)
        self.assertNotIn('"ok": true', serialized_d_history)

        for call in researcher_calls:
            self.assertEqual(
                call["allowed"],
                {"data__tool", "pendulum_compute", "TechnicalResults"},
            )
            self.assertEqual(call["tool_choice"], "any")
        first_r_messages = researcher_calls[0]["messages"]
        self.assertEqual(len(first_r_messages), 2)
        self.assertIn(RAW_QUESTION, first_r_messages[0].content)
        self.assertIn("pendulum_observations", first_r_messages[0].content)
        self.assertNotIn("u_snapshot", first_r_messages[0].content)
        first_task = json.loads(first_r_messages[-1].content)["technical_task"]
        self.assertEqual(first_task["technical_question"], QUESTION_1)
        self.assertEqual(first_task["item_id"], OPEN_ITEM["item_id"])
        second_r_messages = researcher_calls[1]["messages"]
        self.assertEqual(len(second_r_messages), 4)
        self.assertIsInstance(second_r_messages[-2], AIMessage)
        self.assertIsInstance(second_r_messages[-1], ToolMessage)
        self.assertEqual(second_r_messages[-1].tool_call_id, "tool-1")
        self.assertEqual(
            json.loads(second_r_messages[-1].content),
            {"ok": True, "mode": "group_difference"},
        )
        third_r_messages = researcher_calls[2]["messages"]
        self.assertTrue(
            any(
                isinstance(message, ToolMessage)
                and message.tool_call_id == "tool-1"
                for message in third_r_messages
            )
        )
        second_task = json.loads(third_r_messages[-1].content)["technical_task"]
        self.assertEqual(second_task["technical_question"], QUESTION_2)
        self.assertEqual(second_task["item_id"], OPEN_ITEM["item_id"])
        self.assertNotEqual(first_task["task_id"], second_task["task_id"])
        self.assertTrue(
            any(
                isinstance(message, HumanMessage)
                and "previous_technical_results" in str(message.content)
                and "高振幅组周期略长" in str(message.content)
                for message in third_r_messages
            )
        )
        self.assertFalse(
            any(
                isinstance(message, ToolMessage)
                and message.name == "TechnicalResults"
                for message in third_r_messages
            )
        )

        writer = next(
            item for item in model.structured_invocations if item["schema"] == "WriterOutput"
        )
        writer_context = writer["messages"][-1].content
        writer_payload = json.loads(writer_context)
        self.assertEqual(set(writer_payload), {"constitution", "u_snapshot"})
        self.assertNotIn("constitution", writer_payload["u_snapshot"])
        self.assertEqual(writer_payload["constitution"], CONSTITUTION)
        self.assertIn(RAW_QUESTION, writer_context)
        self.assertNotIn("pendulum_compute", writer_context)
        self.assertIn(REVISED_MAP["overall_state"]["current_answer"], writer_context)
        final_state = graph.get_state(config).values
        self.assertEqual(final_state["messages"], [])
        self.assertEqual(final_state["researcher_messages"], [])
        self.assertEqual(
            [packet["item_id"] for packet in final_state["evidence_ledger"]],
            [OPEN_ITEM["item_id"], OPEN_ITEM["item_id"]],
        )
        self.assertEqual(
            len({packet["packet_id"] for packet in final_state["evidence_ledger"]}),
            2,
        )

        graph_nodes = set(graph.get_graph().nodes)
        self.assertEqual(
            graph_nodes,
            {
                "__start__",
                "interpretation",
                "updater",
                "planner",
                "department",
                "writer",
                "__end__",
            },
        )

    def test_department_can_query_compact_capability_on_demand(self) -> None:
        model = ScriptedAdapter()
        model.department_tool_script.insert(
            0,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "capability__query",
                        "args": {"capability_question": "能否执行稳健数值比较？"},
                        "id": "capability-1",
                        "type": "tool_call",
                    }
                ],
            ),
        )
        graph = agent.build_graph(
            [data__tool, pendulum_compute],
            checkpointer=InMemorySaver(),
        )
        graph.invoke(
            {"question": RAW_QUESTION},
            context={"model": model},
            config={"configurable": {"thread_id": "capability-query-test"}},
        )

        department_calls = [
            call
            for call in model.tool_invocations
            if call["agent_role"] == "department"
        ]
        capability_message = next(
            message
            for message in department_calls[1]["messages"]
            if isinstance(message, ToolMessage) and message.name == "capability__query"
        )
        payload = json.loads(capability_message.content)
        self.assertEqual(
            {card["capability"] for card in payload["capabilities"]},
            {"data_tool", "pendulum_analysis"},
        )
        self.assertNotIn("parameters", payload)
        self.assertNotIn("pendulum_compute", capability_message.content)

    def test_constitution_projection_rejects_drift(self) -> None:
        drifted = {
            **BOOTSTRAP_SNAPSHOT,
            "constitution": {**CONSTITUTION, "question": "drift"},
        }
        with self.assertRaisesRegex(ValueError, "constitution drifted"):
            agent._project_constitution_reference(
                drifted,
                CONSTITUTION,
                owner="test snapshot",
            )

    def test_structured_adapter_preserves_raw_failure_without_private_retry(self) -> None:
        raw = AIMessage(content="plain response without a tool call")
        model = StructuredEnvelopeModel(
            [
                {"raw": raw, "parsed": None, "parsing_error": None},
            ]
        )
        adapter = agent.LangChainChatAdapter(model)

        with self.assertRaises(agent.StructuredOutputFailure) as captured:
            adapter.structured(agent.WriterOutput).invoke([])

        failure = captured.exception
        self.assertEqual(model.invoke_count, 1)
        self.assertTrue(model.structured_kwargs[0]["include_raw"])
        self.assertEqual(failure.category, "missing_tool_call")
        self.assertEqual(failure.schema_name, "WriterOutput")
        self.assertIs(failure.raw_response, raw)
        self.assertEqual(failure.as_record()["layer"], "structured_output_adapter")

    def test_structured_adapter_returns_official_parsed_envelope(self) -> None:
        raw = AIMessage(content="")
        expected = {"answer": "adapter-ok"}
        model = StructuredEnvelopeModel(
            [{"raw": raw, "parsed": expected, "parsing_error": None}]
        )
        adapter = agent.LangChainChatAdapter(model)

        result = adapter.structured(agent.WriterOutput).invoke([])

        self.assertEqual(result, expected)
        self.assertEqual(model.invoke_count, 1)
        self.assertTrue(model.structured_kwargs[0]["include_raw"])

    def test_structured_adapter_preserves_parser_error_layer(self) -> None:
        raw = AIMessage(
            content="",
            invalid_tool_calls=[
                {
                    "name": "WriterOutput",
                    "args": "{broken-json",
                    "id": "invalid-1",
                    "error": "invalid JSON",
                    "type": "invalid_tool_call",
                }
            ],
        )
        parser_error = ValueError("invalid JSON arguments")
        model = StructuredEnvelopeModel(
            [{"raw": raw, "parsed": None, "parsing_error": parser_error}]
        )
        adapter = agent.LangChainChatAdapter(model)

        with self.assertRaises(agent.StructuredOutputFailure) as captured:
            adapter.structured(agent.WriterOutput).invoke([])

        failure = captured.exception
        self.assertEqual(failure.category, "parsing_error")
        self.assertIs(failure.parsing_error, parser_error)
        self.assertIs(failure.raw_response, raw)
        self.assertEqual(
            failure.as_record()["parsing_error"],
            {"type": "ValueError", "message": "invalid JSON arguments"},
        )

    def test_outer_roles_retry_and_d_r_use_native_agent_loops(self) -> None:
        graph = agent.build_graph([pendulum_compute], checkpointer=InMemorySaver())

        for node_name in {
            "interpretation",
            "updater",
            "planner",
            "writer",
        }:
            self.assertEqual(
                graph.nodes[node_name].retry_policy,
                (agent.MODEL_RETRY_POLICY,),
            )
        self.assertIsNone(graph.nodes["department"].retry_policy)
        nested = graph.get_graph(xray=True)
        nested_nodes = set(nested.nodes)
        self.assertTrue(any(name.endswith(":model") for name in nested_nodes))
        self.assertTrue(any(name.endswith(":tools") for name in nested_nodes))

    def test_sqlite_checkpointer_resumes_failed_updater_without_replaying_work(self) -> None:
        model = ScriptedAdapter()
        model.structured_script = [
            *STRUCTURED_SCRIPT[:3],
            ("UpdaterOutput", RuntimeError("temporary updater failure")),
            *STRUCTURED_SCRIPT[3:],
        ]
        config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": "sqlite-updater-resume"},
        }

        with tempfile.TemporaryDirectory() as temporary:
            database = str(Path(temporary) / "checkpoints.sqlite")
            with SqliteSaver.from_conn_string(database) as checkpointer:
                graph = agent.build_graph(
                    [data__tool, pendulum_compute],
                    checkpointer=checkpointer,
                )
                with self.assertRaisesRegex(RuntimeError, "temporary updater failure"):
                    graph.invoke(
                        {"question": RAW_QUESTION},
                        context={"model": model},
                        config=config,
                        durability="sync",
                    )

            self.assertEqual(TOOL_CALLS, ["group_difference", "leave_one_out"])
            with SqliteSaver.from_conn_string(database) as checkpointer:
                graph = agent.build_graph(
                    [data__tool, pendulum_compute],
                    checkpointer=checkpointer,
                )
                self.assertEqual(graph.get_state(config).next, ("updater",))
                result = graph.invoke(
                    None,
                    context={"model": model},
                    config=config,
                    durability="sync",
                )

        self.assertEqual(
            result,
            {"final_answer": "现有数据支持周期随振幅略增，但结论仅限已观测范围。"},
        )
        self.assertEqual(TOOL_CALLS, ["group_difference", "leave_one_out"])
        self.assertEqual(
            [
                invocation["schema"]
                for invocation in model.structured_invocations
                if invocation["schema"] == "InterpretationOutput"
            ],
            ["InterpretationOutput"],
        )

    def test_sqlite_checkpointer_resumes_inside_researcher_episode(self) -> None:
        model = ScriptedAdapter()
        model.researcher_tool_script = [
            R_TOOL_SCRIPT[0],
            RuntimeError("temporary researcher model failure"),
            *R_TOOL_SCRIPT[1:],
        ]
        config = {
            "recursion_limit": 50,
            "configurable": {"thread_id": "sqlite-researcher-resume"},
        }

        with tempfile.TemporaryDirectory() as temporary:
            database = str(Path(temporary) / "checkpoints.sqlite")
            with SqliteSaver.from_conn_string(database) as checkpointer:
                graph = agent.build_graph(
                    [data__tool, pendulum_compute],
                    checkpointer=checkpointer,
                )
                with self.assertRaisesRegex(
                    RuntimeError, "temporary researcher model failure"
                ):
                    graph.invoke(
                        {"question": RAW_QUESTION},
                        context={"model": model},
                        config=config,
                        durability="sync",
                    )

            self.assertEqual(TOOL_CALLS, ["group_difference"])
            with SqliteSaver.from_conn_string(database) as checkpointer:
                graph = agent.build_graph(
                    [data__tool, pendulum_compute],
                    checkpointer=checkpointer,
                )
                result = graph.invoke(
                    None,
                    context={"model": model},
                    config=config,
                    durability="sync",
                )

        self.assertEqual(
            result,
            {"final_answer": "现有数据支持周期随振幅略增，但结论仅限已观测范围。"},
        )
        self.assertEqual(TOOL_CALLS, ["group_difference", "leave_one_out"])

    def test_generated_role_schemas_are_strict_objects(self) -> None:
        def walk(schema: dict[str, Any]) -> None:
            if schema.get("type") == "object":
                self.assertFalse(schema.get("additionalProperties", True))
                self.assertEqual(
                    set(schema.get("required", [])),
                    set(schema.get("properties", {})),
                )
                for child in schema.get("properties", {}).values():
                    if isinstance(child, dict):
                        walk(child)
            if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
                walk(schema["items"])
            for branch in schema.get("anyOf", []):
                if isinstance(branch, dict):
                    walk(branch)

        for schema in agent.api_schemas().values():
            walk(schema)

    def test_provider_schemas_have_clean_direct_object_descriptions(self) -> None:
        schemas = agent.api_schemas()
        serialized = json.dumps(schemas, ensure_ascii=False)
        self.assertNotIn("dict() -> new empty dictionary", serialized)

        interpretation = schemas["interpretation"]
        constitution = interpretation["properties"]["constitution"]
        self.assertEqual(constitution["type"], "object")
        self.assertNotIn("allOf", constitution)
        self.assertIn("never encode", constitution["description"])
        self.assertEqual(
            set(constitution["properties"]),
            {"research_object", "boundaries", "requested_output"},
        )
        self.assertNotIn("question", constitution["properties"])

        updater = schemas["updater"]
        self.assertEqual(set(updater["properties"]), {"decision", "epistemic_map"})
        self.assertEqual(
            set(updater["properties"]["decision"]["enum"]),
            {"close", "continue", "restructure"},
        )
        overall_state = updater["properties"]["epistemic_map"]["properties"][
            "overall_state"
        ]
        self.assertEqual(
            set(overall_state["properties"]),
            {"current_answer", "claim_ceiling"},
        )
        self.assertNotIn("status", overall_state["properties"])

    def test_department_report_is_a_thin_u_facing_handoff(self) -> None:
        schema = convert_to_openai_tool(agent.DepartmentReport, strict=True)["function"][
            "parameters"
        ]
        self.assertEqual(
            set(schema["properties"]),
            {
                "report_id",
                "item_id",
                "what_is_known",
                "core_evidence",
                "conclusion_boundary",
                "remaining_uncertainty",
                "cross_item_implications",
            },
        )
        serialized = json.dumps(schema)
        for internal_field in (
            "method_limits",
            "observability_updates",
            "unresolved_technical_issues",
            "constitutional_tensions",
            "evidence_packet_ids",
        ):
            self.assertNotIn(internal_field, serialized)

    def test_department_model_output_excludes_system_owned_identity_fields(self) -> None:
        schema = agent.api_schemas()["department"]

        self.assertEqual(
            set(schema["properties"]),
            {
                "what_is_known",
                "core_evidence",
                "conclusion_boundary",
                "remaining_uncertainty",
                "cross_item_implications",
            },
        )
        self.assertNotIn("report_id", schema["properties"])
        self.assertNotIn("item_id", schema["properties"])

    def test_u_decision_must_match_open_items(self) -> None:
        invalid_updates = [
            (
                "continue",
                {
                    "revision": 0,
                    "items": [{**OPEN_ITEM, "status": "resolved"}],
                    "overall_state": BOOTSTRAP_MAP["overall_state"],
                },
                "decision=continue requires at least one open",
            ),
            (
                "restructure",
                {
                    "revision": 0,
                    "items": [{**OPEN_ITEM, "status": "limited"}],
                    "overall_state": BOOTSTRAP_MAP["overall_state"],
                },
                "decision=restructure requires at least one open",
            ),
            (
                "close",
                {
                    "revision": 0,
                    "items": [OPEN_ITEM],
                    "overall_state": BOOTSTRAP_MAP["overall_state"],
                },
                "cannot close while an epistemic item remains open",
            ),
        ]
        for decision, epistemic_map, message in invalid_updates:
            with self.subTest(message=message):
                model = ScriptedAdapter()
                model.structured_script = [
                    (
                        "InterpretationOutput",
                        {"constitution": INTERPRETATION_CONSTITUTION},
                    ),
                    (
                        "UpdaterOutput",
                        {"decision": decision, "epistemic_map": epistemic_map},
                    ),
                ]
                model.department_tool_script = []
                model.researcher_tool_script = []
                graph = agent.build_graph([], checkpointer=InMemorySaver())
                with self.assertRaisesRegex(ValueError, message):
                    graph.invoke(
                        {"question": RAW_QUESTION},
                        context={"model": model},
                        config={
                            "configurable": {"thread_id": f"invalid-{message}"},
                        },
                    )


if __name__ == "__main__":
    unittest.main()
