from __future__ import annotations

import hashlib
import inspect
import json
import sys
import unittest
from pathlib import Path
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
sys.path.insert(0, str(OUTPUTS))

import mix_epistemic_research_agent as mix
from work.test_thin_epistemic_research_agent import (
    INTERPRETATION_CONSTITUTION,
    RAW_QUESTION,
    REGISTERED_DATA_SOURCES,
    ScriptedAdapter,
    TOOL_CALLS,
    pendulum_compute,
    tool_call,
)


CONSTITUTION = {
    "question": RAW_QUESTION,
    **INTERPRETATION_CONSTITUTION,
    "epistemic_policy": mix.BASE_EPISTEMIC_POLICY,
}
MODEL_INTERPRETATION_CONSTITUTION = {
    **INTERPRETATION_CONSTITUTION,
    "epistemic_policy": mix.BASE_EPISTEMIC_POLICY,
}
OPEN_ITEM_DRAFT = {
    "item_id": "period-contrast",
    "question": "现有记录中的振幅与周期呈现什么关系？",
    "status": "open",
    "supported": [],
    "challenged": [],
    "uncertain": ["尚未获得技术结果"],
    "remaining_gap": ["描述关系并检查基本稳定性"],
}
BOOTSTRAP_MAP_DRAFT = {
    "revision": 0,
    "items": [OPEN_ITEM_DRAFT],
    "overall_state": {
        "current_answer": "尚未回答",
        "claim_ceiling": "尚未形成技术认识",
    },
}
REVISED_MAP_DRAFT = {
    "revision": 1,
    "items": [
        {
            **OPEN_ITEM_DRAFT,
            "status": "resolved",
            "supported": ["主要描述性差异在留一检查中方向一致"],
            "uncertain": [],
            "remaining_gap": [],
        }
    ],
    "overall_state": {
        "current_answer": "当前样本支持周期随振幅略增",
        "claim_ceiling": "仅限已登记样本和已观测振幅范围",
    },
}
TASK_DRAFT = {
    "technical_objective": "确定当前样本中的周期差异及其基本稳定性",
    "scope": ["当前已登记单摆样本", "已观测振幅范围"],
    "required_outputs": ["主要差异", "基本稳健性", "方法与限制"],
    "acceptance_criteria": ["返回量化差异并完成一种基本敏感性检查"],
    "coarse_stages": ["量化主要差异", "检查结果是否由单条记录驱动"],
    "allowed_resources": ["登记数据", "数学统计工具"],
    "escalation_conditions": ["必要变量缺失或样本无法形成比较"],
}


def task_id_for(draft: dict[str, object], ordinal: int = 1) -> str:
    payload = {
        "item_id": OPEN_ITEM_DRAFT["item_id"],
        "episode_ordinal": ordinal,
        "task": draft,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return "task-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


TASK_ID = task_id_for(TASK_DRAFT)
REPAIR_TOOL_CALLS: list[int] = []


@tool
def bounded_lookup(
    local_intent: str,
    limit: Annotated[int, Field(ge=1, le=20)],
) -> dict[str, object]:
    """Return a bounded number of test records."""

    REPAIR_TOOL_CALLS.append(limit)
    return {"ok": True, "limit": limit}


class MixBoundedHierarchyTest(unittest.TestCase):
    def setUp(self) -> None:
        TOOL_CALLS.clear()
        REPAIR_TOOL_CALLS.clear()

    def test_openai_compatible_providers_reuse_langchain_chatopenai(self) -> None:
        expected = {
            "glm": (
                "glm-test",
                "https://open.bigmodel.cn/api/paas/v4/",
                False,
            ),
            "qwen": (
                "qwen-test",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                False,
            ),
            "mimo": ("mimo-test", "https://api.xiaomimimo.com/v1", False),
            "openai": ("gpt-test", "https://api.openai.com/v1", True),
        }

        for provider, (model_name, api_base, strict) in expected.items():
            with self.subTest(provider=provider):
                adapter = mix.make_model_adapter(
                    provider,
                    api_key="test-key",
                    model=model_name,
                )
                self.assertEqual(type(adapter.model).__name__, "ChatOpenAI")
                self.assertEqual(adapter.model.model_name, model_name)
                self.assertEqual(adapter.model.openai_api_base, api_base)
                self.assertEqual(
                    adapter.structured_output_kwargs,
                    {"method": "function_calling", "strict": strict},
                )
                self.assertEqual(adapter.bind_tools_kwargs, {"strict": strict})

        self.assertEqual(mix.resolve_provider_name("gpt"), "openai")
        self.assertEqual(mix.resolve_provider_name("chatglm"), "glm")
        strict_qwen = mix.make_model_adapter(
            "qwen",
            api_key="test-key",
            model="qwen-test",
            strict_function_calling=True,
        )
        self.assertEqual(strict_qwen.bind_tools_kwargs, {"strict": True})

    def test_d_task_boundary_and_r_claimed_finite_dag_complete(self) -> None:
        model = ScriptedAdapter()
        model.structured_script = [
            (
                "InterpretationOutput",
                {"constitution": MODEL_INTERPRETATION_CONSTITUTION},
            ),
            (
                "UpdaterOutput",
                {"epistemic_map": BOOTSTRAP_MAP_DRAFT},
            ),
            ("PlannerSelection", {"selected_item_id": OPEN_ITEM_DRAFT["item_id"]}),
            ("DepartmentTaskDraft", TASK_DRAFT),
            ("RTaskClaim", {"task_id": TASK_ID}),
            (
                "RPlanProposalDraft",
                {
                    "reason": "用两个相互依赖的原子操作覆盖任务要求",
                    "stages": [
                        {
                            "stage_id": "contrast",
                            "operation_objective": "量化两组周期差异",
                            "depends_on": [],
                        },
                        {
                            "stage_id": "robustness",
                            "operation_objective": "检查差异是否由单条记录驱动",
                            "depends_on": ["contrast"],
                        },
                    ],
                },
            ),
            (
                "DPlanReview",
                {"decision": "approve", "boundary_findings": []},
            ),
            (
                "RProgressDecision",
                {"decision": "continue", "reason": "继续执行已批准的稳健性节点"},
            ),
            (
                "TechnicalResults",
                {"results": ["高振幅组周期略长，留一后方向保持一致"]},
            ),
            (
                "DepartmentResults",
                {
                    "what_is_known": "当前样本中的周期差异方向具有基本稳定性",
                    "core_evidence": ["分组差异与留一检查方向一致"],
                    "conclusion_boundary": "仅适用于当前登记样本",
                    "remaining_uncertainty": [],
                    "cross_item_implications": [],
                },
            ),
            (
                "UpdaterOutput",
                {"epistemic_map": REVISED_MAP_DRAFT},
            ),
            ("WriterOutput", {"answer": "当前样本支持周期随振幅略增。"}),
        ]
        model.researcher_tool_script = [
            tool_call("pendulum_compute", {"mode": "group_difference"}, "mix-tool-1"),
            tool_call("pendulum_compute", {"mode": "leave_one_out"}, "mix-tool-2"),
        ]
        graph = mix.build_graph(
            [pendulum_compute],
            checkpointer=InMemorySaver(),
            registered_data_sources=REGISTERED_DATA_SOURCES,
        )
        config = {
            "recursion_limit": 64,
            "configurable": {"thread_id": "mix-bounded-hierarchy"},
        }

        result = graph.invoke(
            {"question": RAW_QUESTION},
            context={"model": model},
            config=config,
        )

        self.assertEqual(result, {"final_answer": "当前样本支持周期随振幅略增。"})
        self.assertEqual(TOOL_CALLS, ["group_difference", "leave_one_out"])
        schemas = [item["schema"] for item in model.structured_invocations]
        self.assertNotIn("DepartmentDisposition", schemas)
        self.assertEqual(schemas.count("DepartmentTaskDraft"), 1)
        self.assertEqual(schemas.count("RTaskClaim"), 1)
        self.assertEqual(schemas.count("RPlanProposalDraft"), 1)
        self.assertEqual(schemas.count("DPlanReview"), 1)
        self.assertEqual(schemas.count("RProgressDecision"), 1)
        self.assertEqual(schemas.count("TechnicalResults"), 1)

        self.assertEqual(len(model.tool_invocations), 2)
        second_execution_context = json.dumps(
            [message.model_dump() for message in model.tool_invocations[1]["messages"]],
            ensure_ascii=False,
        )
        self.assertIn("group_difference", second_execution_context)
        self.assertIn("mix-tool-1", second_execution_context)

        final_state = graph.get_state(config).values
        self.assertEqual(len(final_state["evidence_ledger"]), 1)
        packet = final_state["evidence_ledger"][0]
        self.assertEqual(packet["task_id"], TASK_ID)
        self.assertEqual(packet["tool_call_ids"], ["mix-tool-1", "mix-tool-2"])
        report = final_state["report_ledger"][0]
        self.assertEqual(report["core_evidence"][0]["evidence_refs"], [packet["packet_id"]])
        closed_item = final_state["u_snapshot"]["epistemic_map"]["items"][0]
        self.assertEqual(closed_item["evidence_refs"], [packet["packet_id"]])

    def test_additional_task_requires_a_new_u_p_cycle(self) -> None:
        second_draft = {
            **TASK_DRAFT,
            "technical_objective": "对第一项差异结果做独立的敏感性检查",
            "required_outputs": ["敏感性检查结果", "与第一项结果的关系"],
            "coarse_stages": ["实施独立敏感性检查"],
        }
        second_task_id = task_id_for(second_draft, ordinal=2)
        intermediate_map = {
            "revision": 1,
            "items": [
                {
                    **OPEN_ITEM_DRAFT,
                    "supported": ["已获得主要差异"],
                    "uncertain": ["基本稳定性尚未确定"],
                    "remaining_gap": ["差异是否对单条记录敏感"],
                }
            ],
            "overall_state": {
                "current_answer": "已经观察到主要差异，但稳定性尚未确定",
                "claim_ceiling": "目前只能陈述未经敏感性检查的样本差异",
            },
        }
        final_map = {
            **REVISED_MAP_DRAFT,
            "revision": 2,
        }
        model = ScriptedAdapter()
        model.structured_script = [
            (
                "InterpretationOutput",
                {"constitution": MODEL_INTERPRETATION_CONSTITUTION},
            ),
            (
                "UpdaterOutput",
                {"epistemic_map": BOOTSTRAP_MAP_DRAFT},
            ),
            ("PlannerSelection", {"selected_item_id": OPEN_ITEM_DRAFT["item_id"]}),
            ("DepartmentTaskDraft", TASK_DRAFT),
            ("RTaskClaim", {"task_id": TASK_ID}),
            (
                "RPlanProposalDraft",
                {
                    "reason": "先取得主要差异",
                    "stages": [
                        {
                            "stage_id": "contrast",
                            "operation_objective": "量化差异",
                            "depends_on": [],
                        }
                    ],
                },
            ),
            ("DPlanReview", {"decision": "approve", "boundary_findings": []}),
            ("TechnicalResults", {"results": ["获得主要差异"]}),
            (
                "DepartmentResults",
                {
                    "what_is_known": "已经获得当前样本的主要差异",
                    "core_evidence": ["第一项技术结果给出了量化差异"],
                    "conclusion_boundary": "尚未完成敏感性检查",
                    "remaining_uncertainty": ["差异是否由单条记录驱动"],
                    "cross_item_implications": [],
                },
            ),
            ("UpdaterOutput", {"epistemic_map": intermediate_map}),
            ("PlannerSelection", {"selected_item_id": OPEN_ITEM_DRAFT["item_id"]}),
            ("DepartmentTaskDraft", second_draft),
            ("RTaskClaim", {"task_id": second_task_id}),
            (
                "RPlanProposalDraft",
                {
                    "reason": "实施独立敏感性检查",
                    "stages": [
                        {
                            "stage_id": "sensitivity",
                            "operation_objective": "独立检查第一项结果",
                            "depends_on": [],
                        }
                    ],
                },
            ),
            ("DPlanReview", {"decision": "approve", "boundary_findings": []}),
            ("TechnicalResults", {"results": ["敏感性检查方向一致"]}),
            (
                "DepartmentResults",
                {
                    "what_is_known": "主要差异及独立检查方向一致",
                    "core_evidence": ["两项技术结果方向一致"],
                    "conclusion_boundary": "仅适用于当前登记样本",
                    "remaining_uncertainty": [],
                    "cross_item_implications": [],
                },
            ),
            (
                "UpdaterOutput",
                {"epistemic_map": final_map},
            ),
            ("WriterOutput", {"answer": "两项技术结果方向一致。"}),
        ]
        model.researcher_tool_script = [
            tool_call("pendulum_compute", {"mode": "group_difference"}, "task1-tool"),
            tool_call("pendulum_compute", {"mode": "leave_one_out"}, "task2-tool"),
        ]
        graph = mix.build_graph(
            [pendulum_compute],
            checkpointer=InMemorySaver(),
        )
        config = {
            "recursion_limit": 80,
            "configurable": {"thread_id": "mix-two-tasks"},
        }

        result = graph.invoke(
            {"question": RAW_QUESTION}, context={"model": model}, config=config
        )

        self.assertEqual(result["final_answer"], "两项技术结果方向一致。")
        final_state = graph.get_state(config).values
        self.assertEqual(
            [packet["task_id"] for packet in final_state["evidence_ledger"]],
            [TASK_ID, second_task_id],
        )
        packet_ids = [packet["packet_id"] for packet in final_state["evidence_ledger"]]
        self.assertEqual(
            [
                report["core_evidence"][0]["evidence_refs"]
                for report in final_state["report_ledger"]
            ],
            [[packet_ids[0]], [packet_ids[1]]],
        )
        schemas = [item["schema"] for item in model.structured_invocations]
        self.assertNotIn("DepartmentDisposition", schemas)
        self.assertEqual(schemas.count("UpdaterOutput"), 3)
        self.assertEqual(schemas.count("PlannerSelection"), 2)
        self.assertEqual(schemas.count("DepartmentTaskDraft"), 2)
        claim_calls = [
            item
            for item in model.structured_invocations
            if item["schema"] == "RTaskClaim"
        ]
        self.assertEqual(len(claim_calls), 2)

    def test_r_rejects_a_claim_for_another_task(self) -> None:
        model = ScriptedAdapter()
        model.structured_script = [("RTaskClaim", {"task_id": "wrong-task"})]
        task = {
            "task_id": TASK_ID,
            "target_epistemic_item": OPEN_ITEM_DRAFT["item_id"],
            **TASK_DRAFT,
        }
        researcher = mix._build_researcher_graph(
            [pendulum_compute], REGISTERED_DATA_SOURCES
        )

        with self.assertRaisesRegex(ValueError, "claim exactly the task_id"):
            researcher.invoke(
                {
                    "constitution": CONSTITUTION,
                    "technical_task": task,
                    "messages": [
                        HumanMessage(
                            id=f"technical-task:{TASK_ID}",
                            content=json.dumps({"technical_task": task}),
                        )
                    ],
                    "objective_ledger": [],
                    "evidence_ledger": [],
                    "report_ledger": [],
                    "archive_index": [],
                },
                context={"model": model},
                config={"recursion_limit": 12},
            )

    def test_r_proposes_patch_and_only_d_approval_changes_the_dag(self) -> None:
        model = ScriptedAdapter()
        model.structured_script = [
            ("RTaskClaim", {"task_id": TASK_ID}),
            (
                "RPlanProposalDraft",
                {
                    "reason": "先量化差异，再按结果实施后续检查",
                    "stages": [
                        {
                            "stage_id": "contrast",
                            "operation_objective": "量化两组周期差异",
                            "depends_on": [],
                        },
                        {
                            "stage_id": "initial_check",
                            "operation_objective": "检查初始差异",
                            "depends_on": ["contrast"],
                        },
                    ],
                },
            ),
            ("DPlanReview", {"decision": "approve", "boundary_findings": []}),
            (
                "RProgressDecision",
                {
                    "decision": "propose_patch",
                    "reason": "首项结果表明应改用留一敏感性检查",
                },
            ),
            (
                "RPlanProposalDraft",
                {
                    "reason": "尝试扩大到任务范围外的新实验",
                    "stages": [
                        {
                            "stage_id": "contrast",
                            "operation_objective": "量化两组周期差异",
                            "depends_on": [],
                        },
                        {
                            "stage_id": "external_experiment",
                            "operation_objective": "采集任务范围外的新实验数据",
                            "depends_on": ["contrast"],
                        },
                    ],
                },
            ),
            (
                "DPlanReview",
                {
                    "decision": "reject",
                    "boundary_findings": ["候选节点超出当前登记样本范围"],
                },
            ),
            (
                "RProgressDecision",
                {
                    "decision": "propose_patch",
                    "reason": "根据 D 的边界审核改为范围内的敏感性检查",
                },
            ),
            (
                "RPlanProposalDraft",
                {
                    "reason": "保留已完成节点，将未执行节点替换为留一检查",
                    "stages": [
                        {
                            "stage_id": "contrast",
                            "operation_objective": "量化两组周期差异",
                            "depends_on": [],
                        },
                        {
                            "stage_id": "robustness",
                            "operation_objective": "检查差异是否由单条记录驱动",
                            "depends_on": ["contrast"],
                        },
                    ],
                },
            ),
            ("DPlanReview", {"decision": "approve", "boundary_findings": []}),
            (
                "TechnicalResults",
                {"results": ["主要差异在留一检查后方向一致"]},
            ),
        ]
        model.researcher_tool_script = [
            tool_call(
                "pendulum_compute", {"mode": "group_difference"}, "patch-tool-1"
            ),
            tool_call(
                "pendulum_compute", {"mode": "leave_one_out"}, "patch-tool-2"
            ),
        ]
        task = {
            "task_id": TASK_ID,
            "target_epistemic_item": OPEN_ITEM_DRAFT["item_id"],
            **TASK_DRAFT,
        }
        researcher = mix._build_researcher_graph(
            [pendulum_compute], REGISTERED_DATA_SOURCES
        )

        result = researcher.invoke(
            {
                "constitution": CONSTITUTION,
                "technical_task": task,
                "messages": [
                    HumanMessage(
                        id=f"technical-task:{TASK_ID}",
                        content=json.dumps({"technical_task": task}),
                    )
                ],
                "objective_ledger": [],
                "evidence_ledger": [],
                "report_ledger": [],
                "archive_index": [],
            },
            context={"model": model},
            config={"recursion_limit": 40},
        )

        self.assertEqual(result["researcher_plan"]["revision"], 1)
        self.assertEqual(
            [stage["stage_id"] for stage in result["researcher_plan"]["stages"]],
            ["contrast", "robustness"],
        )
        self.assertEqual(TOOL_CALLS, ["group_difference", "leave_one_out"])
        plan_calls = [
            item
            for item in model.structured_invocations
            if item["schema"] == "RPlanProposalDraft"
        ]
        patch_context = json.dumps(
            [message.model_dump() for message in plan_calls[2]["messages"]],
            ensure_ascii=False,
        )
        self.assertIn("patch-tool-1", patch_context)
        review_calls = [
            item
            for item in model.structured_invocations
            if item["schema"] == "DPlanReview"
        ]
        second_review_context = json.dumps(
            [message.model_dump() for message in review_calls[1]["messages"]],
            ensure_ascii=False,
        )
        self.assertNotIn("patch-tool-1", second_review_context)
        self.assertNotIn("registered_capabilities", second_review_context)

    def test_one_coarse_dag_stage_can_execute_a_parallel_tool_batch(self) -> None:
        model = ScriptedAdapter()
        model.structured_script = [
            ("RTaskClaim", {"task_id": TASK_ID}),
            (
                "RPlanProposalDraft",
                {
                    "reason": "同一比较目标需要两个互补诊断",
                    "stages": [
                        {
                            "stage_id": "combined_diagnostic",
                            "operation_objective": "同时取得主要差异与留一诊断",
                            "depends_on": [],
                        }
                    ],
                },
            ),
            ("DPlanReview", {"decision": "approve", "boundary_findings": []}),
            (
                "TechnicalResults",
                {"results": ["主要差异和留一诊断已经同时取得"]},
            ),
        ]
        model.researcher_tool_script = [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "pendulum_compute",
                        "args": {
                            "local_intent": "量化主要差异",
                            "mode": "group_difference",
                        },
                        "id": "batch-tool-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "pendulum_compute",
                        "args": {
                            "local_intent": "实施留一诊断",
                            "mode": "leave_one_out",
                        },
                        "id": "batch-tool-2",
                        "type": "tool_call",
                    },
                ],
            )
        ]
        task = {
            "task_id": TASK_ID,
            "target_epistemic_item": OPEN_ITEM_DRAFT["item_id"],
            **TASK_DRAFT,
        }
        researcher = mix._build_researcher_graph(
            [pendulum_compute], REGISTERED_DATA_SOURCES
        )

        result = researcher.invoke(
            {
                "constitution": CONSTITUTION,
                "technical_task": task,
                "messages": [
                    HumanMessage(
                        id=f"technical-task:{TASK_ID}",
                        content=json.dumps({"technical_task": task}),
                    )
                ],
                "objective_ledger": [],
                "evidence_ledger": [],
                "report_ledger": [],
                "archive_index": [],
            },
            context={"model": model},
            config={"recursion_limit": 24},
        )

        self.assertEqual(TOOL_CALLS, ["group_difference", "leave_one_out"])
        self.assertEqual(
            result["technical_packet"]["tool_call_ids"],
            ["batch-tool-1", "batch-tool-2"],
        )
        self.assertEqual(
            result["researcher_completed_stage_ids"], ["combined_diagnostic"]
        )

    def test_tool_input_error_alone_triggers_same_stage_repair(self) -> None:
        model = ScriptedAdapter()
        model.structured_script = [
            ("RTaskClaim", {"task_id": TASK_ID}),
            (
                "RPlanProposalDraft",
                {
                    "reason": "读取完成当前技术目标所需的有限记录",
                    "stages": [
                        {
                            "stage_id": "bounded-read",
                            "operation_objective": "读取有限数量的记录",
                            "depends_on": [],
                        }
                    ],
                },
            ),
            ("DPlanReview", {"decision": "approve", "boundary_findings": []}),
            ("TechnicalResults", {"results": ["成功读取 10 条记录"]}),
        ]
        model.researcher_tool_script = [
            tool_call("bounded_lookup", {"limit": 0}, "invalid-limit"),
            tool_call("bounded_lookup", {"limit": 10}, "repaired-limit"),
        ]
        task = {
            "task_id": TASK_ID,
            "target_epistemic_item": OPEN_ITEM_DRAFT["item_id"],
            **TASK_DRAFT,
        }
        researcher = mix._build_researcher_graph([bounded_lookup], [])

        result = researcher.invoke(
            {
                "constitution": CONSTITUTION,
                "technical_task": task,
                "messages": [
                    HumanMessage(
                        id=f"technical-task:{TASK_ID}",
                        content=json.dumps({"technical_task": task}),
                    )
                ],
                "objective_ledger": [],
                "evidence_ledger": [],
                "report_ledger": [],
                "archive_index": [],
            },
            context={"model": model},
            config={"recursion_limit": 20},
        )

        self.assertEqual(REPAIR_TOOL_CALLS, [10])
        self.assertEqual(
            result["technical_packet"]["tool_call_ids"],
            ["repaired-limit"],
        )
        self.assertEqual(len(model.tool_invocations), 2)
        repair_context = json.dumps(
            [
                message.model_dump()
                for message in model.tool_invocations[1]["messages"]
            ],
            ensure_ascii=False,
        )
        self.assertIn('"status": "error"', repair_context)
        self.assertIn("limit", repair_context)

    def test_contracts_keep_authority_fields_out_of_d_model_output(self) -> None:
        schemas = mix.api_schemas()
        task_properties = set(schemas["department_task"]["properties"])
        self.assertEqual(
            task_properties,
            {
                "technical_objective",
                "scope",
                "required_outputs",
                "acceptance_criteria",
                "coarse_stages",
                "allowed_resources",
                "escalation_conditions",
            },
        )
        self.assertNotIn("task_id", task_properties)
        self.assertNotIn("target_epistemic_item", task_properties)
        self.assertNotIn("budget", task_properties)
        review_properties = set(
            schemas["department_plan_review"]["properties"]
        )
        self.assertEqual(review_properties, {"decision", "boundary_findings"})
        self.assertNotIn("stages", review_properties)
        updater_item = schemas["updater"]["properties"]["epistemic_map"]
        serialized = json.dumps(updater_item, ensure_ascii=False)
        self.assertIn("supported", serialized)
        self.assertIn("challenged", serialized)
        self.assertIn("uncertain", serialized)
        self.assertIn("remaining_gap", serialized)
        self.assertIn("overall_state", serialized)
        self.assertIn("current_answer", serialized)
        self.assertIn("claim_ceiling", serialized)
        self.assertNotIn("evidence_refs", serialized)
        self.assertNotIn("decision", schemas["updater"]["properties"])

    def test_common_structured_entry_rejects_missing_nested_required_fields(self) -> None:
        model = ScriptedAdapter()
        model.structured_script = [
            (
                "UpdaterOutput",
                {"epistemic_map": {"revision": 0, "items": []}},
            )
        ]

        with self.assertRaises(mix.StructuredOutputFailure) as raised:
            mix._structured(
                model,
                mix.UpdaterOutput,
                mix.UPDATER_BOOTSTRAP_PROMPT,
                {"constitution": CONSTITUTION},
                policy_role="U",
            )

        self.assertEqual(raised.exception.category, "schema_contract")
        self.assertIn("UpdaterOutput.epistemic_map.overall_state", str(raised.exception))

    def test_u_routing_decision_is_derived_from_the_epistemic_map(self) -> None:
        open_snapshot = mix._materialize_epistemic_snapshot(
            constitution=CONSTITUTION,
            update={"epistemic_map": BOOTSTRAP_MAP_DRAFT},
            prior=None,
            report=None,
        )
        self.assertEqual(open_snapshot["decision"], "continue")

        closed_snapshot = mix._materialize_epistemic_snapshot(
            constitution=CONSTITUTION,
            update={"epistemic_map": REVISED_MAP_DRAFT},
            prior=open_snapshot,
            report=None,
        )
        self.assertEqual(closed_snapshot["decision"], "close")

        restructured_map = {
            "revision": 1,
            "items": [
                {
                    "item_id": "new-limited-proposition",
                    "question": "A newly exposed proposition with no actionable gap",
                    "status": "limited",
                    "supported": [],
                    "challenged": [],
                    "uncertain": ["The available record cannot distinguish it"],
                    "remaining_gap": ["Requires unavailable observations"],
                }
            ],
            "overall_state": {
                "current_answer": "The new proposition remains observationally limited.",
                "claim_ceiling": "No stronger claim is identifiable from the available record.",
            },
        }
        restructured_snapshot = mix._materialize_epistemic_snapshot(
            constitution=CONSTITUTION,
            update={"epistemic_map": restructured_map},
            prior=open_snapshot,
            report=None,
        )
        self.assertEqual(restructured_snapshot["decision"], "restructure")

    def test_each_role_receives_one_shared_authority_example(self) -> None:
        prompts = [
            mix.INTERPRETATION_PROMPT,
            mix.UPDATER_BOOTSTRAP_PROMPT,
            mix.UPDATER_PROMPT,
            mix.PLANNER_PROMPT,
            mix.DEPARTMENT_TASK_PROMPT,
            mix.RESEARCHER_CLAIM_PROMPT,
            mix.RESEARCHER_PLAN_PROMPT,
            mix.D_PLAN_REVIEW_PROMPT,
            mix.RESEARCHER_EXECUTE_PROMPT,
            mix.RESEARCHER_PROGRESS_PROMPT,
            mix.RESEARCHER_FINALIZE_PROMPT,
            mix.DEPARTMENT_REPORT_PROMPT,
            mix.WRITER_PROMPT,
        ]
        for prompt in prompts:
            self.assertEqual(prompt.count("Shared boundary example"), 1)
            self.assertIn("U alone updates proposition status", prompt)
            self.assertIn("R claims that task_id", prompt)
            self.assertIn("Boundary counterexamples", prompt)

    def test_temporary_tool_policy_is_not_injected_into_role_prompts(self) -> None:
        for prompt in [
            mix.INTERPRETATION_PROMPT,
            mix.UPDATER_BOOTSTRAP_PROMPT,
            mix.UPDATER_PROMPT,
            mix.PLANNER_PROMPT,
            mix.DEPARTMENT_TASK_PROMPT,
            mix.RESEARCHER_CLAIM_PROMPT,
            mix.RESEARCHER_PLAN_PROMPT,
            mix.D_PLAN_REVIEW_PROMPT,
            mix.RESEARCHER_EXECUTE_PROMPT,
            mix.RESEARCHER_PROGRESS_PROMPT,
            mix.RESEARCHER_FINALIZE_PROMPT,
            mix.DEPARTMENT_REPORT_PROMPT,
            mix.WRITER_PROMPT,
        ]:
            self.assertNotIn("Temporary-tool example", prompt)
            self.assertNotIn("temporary tool", prompt.lower())

    def test_d_capability_summary_includes_registered_python_sandbox(self) -> None:
        @tool
        def sandbox__python(local_intent: str) -> dict[str, object]:
            """Execute one run-scoped Python operation."""

            return {"local_intent": local_intent}

        sandbox__python.metadata = {
            "native_tool_id": "sandbox.python",
            "capability_group": "temporary_python_operations",
        }
        envelope = mix._r_capability_envelope([sandbox__python])
        self.assertIn("auditable run-scoped Python operations", envelope["summary"])

    def test_r_rolling_messages_keep_science_but_omit_data_and_archive_reads(self) -> None:
        messages = [
            HumanMessage(content="technical task"),
            tool_call("data__tool", {"operation": "query"}, "read-data"),
            ToolMessage(
                content="raw table rows",
                name="data__tool",
                tool_call_id="read-data",
            ),
            tool_call("archive__read", {"limit": 5}, "read-archive"),
            ToolMessage(
                content="archived records",
                name="archive__read",
                tool_call_id="read-archive",
            ),
            tool_call("time__dip", {"input_from": ["read-data"]}, "run-dip"),
            ToolMessage(
                content="dip statistic and p-value",
                name="time__dip",
                tool_call_id="run-dip",
            ),
        ]

        projected = mix._project_researcher_messages(
            messages,
            {"data__tool", "archive__read"},
        )
        serialized = json.dumps(
            [message.model_dump() for message in projected],
            ensure_ascii=False,
        )
        self.assertIn("time__dip", serialized)
        self.assertIn("dip statistic and p-value", serialized)
        self.assertNotIn("data__tool", serialized)
        self.assertNotIn("raw table rows", serialized)
        self.assertNotIn("archive__read", serialized)
        self.assertNotIn("archived records", serialized)

    def test_data_and_archive_results_are_projected_only_by_dag_dependency(self) -> None:
        plan = {
            "task_id": TASK_ID,
            "revision": 0,
            "stages": [
                {"stage_id": "table", "operation_objective": "读取数据", "depends_on": []},
                {"stage_id": "archive", "operation_objective": "读取旧档案", "depends_on": []},
                {
                    "stage_id": "analyze",
                    "operation_objective": "计算统计量",
                    "depends_on": ["table"],
                },
            ],
        }
        messages = [
            tool_call("data__tool", {"operation": "query"}, "dependency-data"),
            ToolMessage(
                content="needed table rows",
                name="data__tool",
                tool_call_id="dependency-data",
            ),
            tool_call("archive__read", {"limit": 5}, "unrelated-archive"),
            ToolMessage(
                content="unrelated archived rows",
                name="archive__read",
                tool_call_id="unrelated-archive",
            ),
        ]
        dependency_scope = mix._dag_stage_closure(plan, ["analyze"])
        observations = mix._transient_stage_observations(
            messages,
            completed_stage_ids=["table", "archive"],
            active_stage_id=None,
            included_stage_ids=dependency_scope,
            transient_tool_names={"data__tool", "archive__read"},
        )

        self.assertEqual(
            [observation["tool_call_id"] for observation in observations],
            ["dependency-data"],
        )
        self.assertEqual(observations[0]["stage_id"], "table")
        self.assertEqual(observations[0]["result"], "needed table rows")

    def test_control_plane_has_no_research_budget_or_code_only_nodes(self) -> None:
        self.assertEqual(
            set(inspect.signature(mix.build_graph).parameters),
            {"technical_tools", "checkpointer", "registered_data_sources"},
        )
        self.assertNotIn("budget", mix.TechnicalTask.__annotations__)
        self.assertNotIn("department_disposition", mix.ROLE_CONTRACTS)
        self.assertNotIn("DepartmentTaskDocket", mix.__dict__)

        researcher = mix._build_researcher_graph([pendulum_compute], [])
        self.assertEqual(
            set(researcher.get_graph().nodes),
            {
                "__start__",
                "__end__",
                "claim_task",
                "plan_local_dag",
                "department_plan_review",
                "execute_stage",
                "tools",
                "review_progress",
                "finalize",
            },
        )


if __name__ == "__main__":
    unittest.main()
