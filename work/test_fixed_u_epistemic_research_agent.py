from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver


OUTPUTS = Path(__file__).resolve().parents[1] / "outputs"
sys.path.insert(0, str(OUTPUTS))

import fixed_u_epistemic_research_agent as fixed
from work.test_thin_epistemic_research_agent import (
    INTERPRETATION_CONSTITUTION,
    RAW_QUESTION,
    REGISTERED_DATA_SOURCES,
    ScriptedAdapter,
    TOOL_CALLS,
    pendulum_compute,
    tool_call,
)


COGNITIVE_ITEM = {
    "c_id": "c-period-contrast",
    "explanation_under_evaluation": "振幅本身导致当前样本的周期差异。",
    "assessment_question": "当前观测是否支持振幅效应，而非偶然单条记录造成的差异？",
    "scientific_role": "独立评估振幅效应的经验支持、反证和判别性。",
    "predicted_observations": ["高低振幅组出现方向稳定的周期差。"],
    "counterevidence_signals": ["分组差异在留一检查中消失或反转。"],
    "required_evidence": ["分组比较结果", "对单条记录敏感性的稳健性检查"],
    "decision_target": "形成当前样本中是否存在稳定振幅—周期关系的局部结论。",
}
SELECTION_ARTIFACT_ITEM = {
    "c_id": "c-selection-artifact",
    "explanation_under_evaluation": "样本选择结构而非振幅本身造成观察到的周期差异。",
    "assessment_question": "当前观测是否与样本选择伪差异的预期一致？",
    "scientific_role": "独立评估选择结构这一替代解释的支持、反证和判别性。",
    "predicted_observations": ["改变登记分组后差异显著改变。"],
    "counterevidence_signals": ["不同选择处理下差异方向保持稳定。"],
    "required_evidence": ["选择敏感性诊断"],
    "decision_target": "判断选择结构是否仍能解释当前观察。",
}
FIXED_INTERPRETATION_CONSTITUTION = {
    **INTERPRETATION_CONSTITUTION,
    "boundaries": ["仅讨论当前登记单摆样本和已观测振幅范围"],
}

TASK_MAIN = {
    "technical_objective": "量化当前登记样本中高低振幅组的周期差异",
    "scope": ["当前登记单摆样本", "已观测振幅范围"],
    "required_outputs": ["组间周期差", "可审查的分组定义"],
    "acceptance_criteria": ["返回方向和量化差异"],
    "coarse_stages": ["计算高低振幅组的周期差"],
    "allowed_resources": ["登记观测", "统计计算工具"],
    "escalation_conditions": ["无法形成可比较的两组"],
}
TASK_ROBUST = {
    "technical_objective": "检查观察到的周期差是否由单条记录驱动",
    "scope": ["当前登记单摆样本", "已观测振幅范围"],
    "required_outputs": ["留一检查的方向稳定性"],
    "acceptance_criteria": ["说明移除单条记录后差异方向是否改变"],
    "coarse_stages": ["执行留一敏感性检查"],
    "allowed_resources": ["登记观测", "统计计算工具"],
    "escalation_conditions": ["样本量不足以进行留一检查"],
}
DEPARTMENT_PLAN = {
    "technical_items": [
        {
            "t_id": "t-main-difference",
            "question": "高低振幅组的周期差是多少？",
            "scientific_role": "提供目标现象的主要量化证据。",
            "assessment_dimension": "support",
            "technical_task": TASK_MAIN,
        },
    ]
}
TECHNICAL_ITEMS = fixed._materialize_technical_items(COGNITIVE_ITEM, DEPARTMENT_PLAN)
TASK_IDS = [item["task_id"] for item in TECHNICAL_ITEMS]


class ContextualStructured:
    def __init__(self, adapter: "ParallelAssessmentAdapter", schema: type) -> None:
        self.adapter = adapter
        self.schema = schema

    def invoke(self, messages: list[Any]) -> dict[str, Any]:
        context = json.loads(str(messages[-1].content))
        self.adapter.structured_invocations.append({"schema": self.schema.__name__, "messages": messages})
        return self.adapter.response(self.schema.__name__, context)


class ParallelAssessmentAdapter(ScriptedAdapter):
    def structured(
        self,
        schema: type,
        *,
        cache_scope: str = "unscoped",
    ) -> ContextualStructured:
        del cache_scope
        return ContextualStructured(self, schema)

    def response(self, schema: str, context: dict[str, Any]) -> dict[str, Any]:
        if schema == "InterpretationOutput":
            return {"constitution": FIXED_INTERPRETATION_CONSTITUTION}
        if schema == "UPlanOutput":
            return {"cognitive_items": [COGNITIVE_ITEM, SELECTION_ARTIFACT_ITEM]}
        if schema == "DepartmentPlanOutput":
            item = context["cognitive_item"]
            task = TASK_MAIN if item["c_id"] == COGNITIVE_ITEM["c_id"] else TASK_ROBUST
            return {"technical_items": [{"t_id": f"t-{item['c_id']}", "question": item["assessment_question"], "scientific_role": item["scientific_role"], "assessment_dimension": "discrimination", "technical_task": task}]}
        if schema == "RTaskClaim":
            return {"task_id": context["technical_task"]["task_id"]}
        if schema == "RPlanProposalDraft":
            return {"reason": "执行该解释评估项目的单一可审查诊断", "stages": [{"stage_id": "diagnostic", "operation_objective": "执行已签发的判别诊断", "depends_on": []}]}
        if schema == "DPlanReview":
            return {"decision": "approve", "boundary_findings": []}
        if schema == "TechnicalResults":
            return {"results": [f"{context['technical_task']['target_epistemic_item']} 的登记诊断结果"]}
        if schema == "DepartmentConclusionOutput":
            item = context["cognitive_item"]
            t_id = context["technical_items"][0]["t_id"]
            return {"c_id": item["c_id"], "explanation_under_evaluation": item["explanation_under_evaluation"], "conclusion": "该解释获得有限且可审查的经验评估。", "technical_conclusions": [{"t_id": t_id, "conclusion": "完成一个登记诊断。", "evidence_basis": ["R 技术包"], "technical_basis": technical_basis("登记诊断统计量", "单一诊断"), "uncertainty": ["单一证据链"], "scope": ["当前样本"], "status": "completed"}], "supporting_evidence": ["登记诊断"], "counterevidence": [], "discriminative_evidence": ["该诊断改变候选解释的相对支持"], "assessment_status": "supported", "uncertainty": ["需与其他解释比较"], "scope": ["当前样本"]}
        if schema == "EpistemicConclusionOutput":
            rows = []
            for report in context["department_conclusions"]:
                rows.append({"c_id": report["c_id"], "explanation": report["explanation_under_evaluation"], "supporting_evidence": report["supporting_evidence"], "counterevidence": report["counterevidence"], "discriminative_evidence": report["discriminative_evidence"], "non_discriminative_evidence": [], "independent_evidence_chains": ["各 C 项目独立诊断"], "uncertainty": report["uncertainty"], "scope": report["scope"], "status": "supported"})
            return {"evidence_matrix": rows, "normalization_notes": ["两项诊断均限于当前样本。"], "cross_explanation_conflicts": [], "unresolved_items": ["需要更多独立证据链"]}
        if schema == "WriterOutput":
            return {"answer": "W 依据归一后的两行证据矩阵进行最终裁决。"}
        raise AssertionError(f"unexpected schema {schema}")


def technical_basis(statistic: str, check: str) -> dict[str, object]:
    return {
        "data_scope": ["当前登记单摆观测", "已观测振幅范围"],
        "selection_conditions": ["仅使用同时具有振幅与周期值的记录"],
        "sample_size": "N=24",
        "grouping_criteria": ["按预先登记的高低振幅分组"],
        "methods": ["分组差异计算", "留一敏感性诊断"],
        "key_parameters": ["分组边界按登记定义", "留一逐条剔除"],
        "thresholds": ["未使用额外显著性阈值"],
        "core_statistics": [statistic],
        "uncertainty": ["仅覆盖当前样本与观测范围"],
        "robustness_checks": [check],
    }


class FixedUArgumentTreeTest(unittest.TestCase):
    def setUp(self) -> None:
        TOOL_CALLS.clear()

    def test_fixed_c_and_t_trees_complete_before_structured_d_and_u_conclusions(self) -> None:
        model = ScriptedAdapter()
        model.structured_script = [
            ("InterpretationOutput", {"constitution": FIXED_INTERPRETATION_CONSTITUTION}),
            ("UPlanOutput", {"cognitive_items": [COGNITIVE_ITEM]}),
            ("DepartmentPlanOutput", DEPARTMENT_PLAN),
            ("RTaskClaim", {"task_id": TASK_IDS[0]}),
            (
                "RPlanProposalDraft",
                {"reason": "量化分组差异", "stages": [{"stage_id": "difference", "operation_objective": "计算组间周期差", "depends_on": []}]},
            ),
            ("DPlanReview", {"decision": "approve", "boundary_findings": []}),
            ("TechnicalResults", {"results": ["高振幅组平均周期较低振幅组长 0.03 s"]}),
            (
                "DepartmentConclusionOutput",
                {
                    "c_id": COGNITIVE_ITEM["c_id"],
                    "explanation_under_evaluation": COGNITIVE_ITEM["explanation_under_evaluation"],
                    "conclusion": "当前登记样本中，高振幅组周期略长，且方向不由单条记录驱动。",
                    "technical_conclusions": [
                        {
                            "t_id": "t-main-difference",
                            "conclusion": "高振幅组平均周期较长。",
                            "evidence_basis": ["分组差异计算返回 0.03 s 的方向性差异"],
                            "technical_basis": technical_basis("组间平均周期差=0.03 s", "留待独立 T 完成"),
                            "uncertainty": ["样本量有限"],
                            "scope": ["当前登记样本"],
                            "status": "completed",
                        },
                    ],
                    "supporting_evidence": ["分组差异为正"],
                    "counterevidence": [],
                    "discriminative_evidence": ["留一检查未改变差异方向"],
                    "assessment_status": "supported",
                    "uncertainty": ["当前证据不确定机制，也不覆盖更高振幅范围"],
                    "scope": ["当前登记单摆样本和已观测振幅范围"],
                },
            ),
            (
                "EpistemicConclusionOutput",
                {
                    "evidence_matrix": [
                        {
                            "c_id": COGNITIVE_ITEM["c_id"],
                            "explanation": COGNITIVE_ITEM["explanation_under_evaluation"],
                            "supporting_evidence": ["分组差异"],
                            "counterevidence": [],
                            "discriminative_evidence": ["留一检查"],
                            "non_discriminative_evidence": [],
                            "independent_evidence_chains": ["分组差异与留一诊断"],
                            "uncertainty": ["样本范围有限"],
                            "scope": ["当前登记样本与已观测振幅范围"],
                            "status": "supported",
                        }
                    ],
                    "normalization_notes": ["所有证据均限于当前样本，未作为机制证明。"],
                    "cross_explanation_conflicts": [],
                    "unresolved_items": ["未测量混杂因素"],
                },
            ),
            ("WriterOutput", {"answer": "当前登记样本支持小幅且稳健的振幅—周期关联，但不能据此确认机制。"}),
        ]
        model.researcher_tool_script = [
            tool_call("pendulum_compute", {"mode": "group_difference"}, "fixed-tool-1"),
        ]
        graph = fixed.build_graph(
            [pendulum_compute],
            checkpointer=InMemorySaver(),
            registered_data_sources=REGISTERED_DATA_SOURCES,
        )
        config = {"recursion_limit": 64, "configurable": {"thread_id": "fixed-u-tree"}}

        result = graph.invoke({"question": RAW_QUESTION}, context={"model": model}, config=config)

        self.assertEqual(result, {"final_answer": "当前登记样本支持小幅且稳健的振幅—周期关联，但不能据此确认机制。"})
        self.assertEqual(TOOL_CALLS, ["group_difference"])
        schemas = [item["schema"] for item in model.structured_invocations]
        self.assertEqual(schemas.count("UPlanOutput"), 1)
        self.assertEqual(schemas.count("DepartmentPlanOutput"), 1)
        self.assertEqual(schemas.count("DepartmentConclusionOutput"), 1)
        self.assertEqual(schemas.count("EpistemicConclusionOutput"), 1)
        self.assertNotIn("UpdaterOutput", schemas)
        self.assertNotIn("PlannerSelection", schemas)

        state = graph.get_state(config).values
        self.assertEqual(state["cognitive_items"], [COGNITIVE_ITEM])
        self.assertEqual([item["task_id"] for item in state["evidence_ledger"]], TASK_IDS)
        conclusion = state["department_conclusions"][0]
        self.assertEqual(
            [item["packet_refs"] for item in conclusion["technical_conclusions"]],
            [[packet["packet_id"]] for packet in state["evidence_ledger"]],
        )
        self.assertEqual(state["epistemic_conclusion"]["evidence_matrix"][0]["status"], "supported")

    def test_u_fans_out_competing_explanations_then_normalizes_one_matrix(self) -> None:
        model = ParallelAssessmentAdapter()
        model.researcher_tool_script = [
            tool_call("pendulum_compute", {"mode": "group_difference"}, "parallel-tool-1"),
            tool_call("pendulum_compute", {"mode": "leave_one_out"}, "parallel-tool-2"),
        ]
        graph = fixed.build_graph(
            [pendulum_compute],
            checkpointer=InMemorySaver(),
            registered_data_sources=REGISTERED_DATA_SOURCES,
        )
        config = {"recursion_limit": 96, "configurable": {"thread_id": "parallel-c-projects"}}

        result = graph.invoke({"question": RAW_QUESTION}, context={"model": model}, config=config)

        self.assertEqual(result["final_answer"], "W 依据归一后的两行证据矩阵进行最终裁决。")
        state = graph.get_state(config).values
        self.assertEqual(
            {report["c_id"] for report in state["department_conclusions"]},
            {COGNITIVE_ITEM["c_id"], SELECTION_ARTIFACT_ITEM["c_id"]},
        )
        self.assertEqual(
            {row["c_id"] for row in state["epistemic_conclusion"]["evidence_matrix"]},
            {COGNITIVE_ITEM["c_id"], SELECTION_ARTIFACT_ITEM["c_id"]},
        )
        schemas = [item["schema"] for item in model.structured_invocations]
        self.assertEqual(schemas.count("DepartmentPlanOutput"), 2)
        self.assertEqual(schemas.count("DepartmentConclusionOutput"), 2)
        self.assertEqual(schemas.count("EpistemicConclusionOutput"), 1)

    def test_r_fans_out_independent_stages_without_replaying_messages(self) -> None:
        model = ScriptedAdapter()
        model.structured_script = [
            ("RTaskClaim", {"task_id": TASK_IDS[0]}),
            (
                "RPlanProposalDraft",
                {
                    "reason": "两个无数据依赖的独立诊断并发执行",
                    "stages": [
                        {"stage_id": "support", "operation_objective": "计算主要差异", "depends_on": []},
                        {"stage_id": "robustness", "operation_objective": "执行独立稳健性检查", "depends_on": []},
                    ],
                },
            ),
            ("DPlanReview", {"decision": "approve", "boundary_findings": []}),
            ("TechnicalResults", {"results": ["两个独立诊断均已完成"]}),
        ]
        model.researcher_tool_script = [
            tool_call("pendulum_compute", {"mode": "group_difference"}, "parallel-r-1"),
            tool_call("pendulum_compute", {"mode": "leave_one_out"}, "parallel-r-2"),
        ]
        graph = fixed._build_researcher_graph(
            [pendulum_compute], REGISTERED_DATA_SOURCES
        )
        result = graph.invoke(
            {
                "constitution": {
                    **FIXED_INTERPRETATION_CONSTITUTION,
                    "question": RAW_QUESTION,
                    "epistemic_policy": fixed.BASE_EPISTEMIC_POLICY,
                },
                "technical_task": {
                    **TECHNICAL_ITEMS[0],
                    "target_epistemic_item": COGNITIVE_ITEM["c_id"],
                },
                "messages": [],
            },
            context={"model": model},
            config={"recursion_limit": 32},
        )

        self.assertEqual(result["technical_packet"]["results"], ["两个独立诊断均已完成"])
        self.assertEqual(set(TOOL_CALLS), {"group_difference", "leave_one_out"})
        researcher_calls = [
            call for call in model.tool_invocations if call["agent_role"] == "researcher"
        ]
        self.assertEqual(len(researcher_calls), 2)
        for call in researcher_calls:
            # Worker context consists only of role policy plus its DAG-local
            # input packet; no parent R rolling message history is replayed.
            self.assertEqual(len(call["messages"]), 2)
            packet = json.loads(str(call["messages"][-1].content))
            self.assertEqual(packet["dag_dependency_stage_results"], [])

    def test_r_releases_dependent_stage_only_after_its_parent_wave(self) -> None:
        model = ScriptedAdapter()
        model.structured_script = [
            ("RTaskClaim", {"task_id": TASK_IDS[0]}),
            (
                "RPlanProposalDraft",
                {
                    "reason": "先计算共享输入，再执行依赖诊断",
                    "stages": [
                        {"stage_id": "prepare", "operation_objective": "计算共享输入", "depends_on": []},
                        {"stage_id": "diagnose", "operation_objective": "使用共享输入执行诊断", "depends_on": ["prepare"]},
                    ],
                },
            ),
            ("DPlanReview", {"decision": "approve", "boundary_findings": []}),
            ("RProgressDecision", {"decision": "continue", "reason": "上游输入已获得，继续依赖诊断"}),
            ("TechnicalResults", {"results": ["依赖诊断已完成"]}),
        ]
        model.researcher_tool_script = [
            tool_call("pendulum_compute", {"mode": "group_difference"}, "dependent-r-1"),
            tool_call("pendulum_compute", {"mode": "leave_one_out"}, "dependent-r-2"),
        ]
        graph = fixed._build_researcher_graph(
            [pendulum_compute], REGISTERED_DATA_SOURCES
        )
        graph.invoke(
            {
                "constitution": {
                    **FIXED_INTERPRETATION_CONSTITUTION,
                    "question": RAW_QUESTION,
                    "epistemic_policy": fixed.BASE_EPISTEMIC_POLICY,
                },
                "technical_task": {
                    **TECHNICAL_ITEMS[0],
                    "target_epistemic_item": COGNITIVE_ITEM["c_id"],
                },
                "messages": [],
            },
            context={"model": model},
            config={"recursion_limit": 32},
        )

        researcher_calls = [
            call for call in model.tool_invocations if call["agent_role"] == "researcher"
        ]
        self.assertEqual(len(researcher_calls), 2)
        first = json.loads(str(researcher_calls[0]["messages"][-1].content))
        second = json.loads(str(researcher_calls[1]["messages"][-1].content))
        self.assertEqual(first["current_stage"]["stage_id"], "prepare")
        self.assertEqual(second["current_stage"]["stage_id"], "diagnose")
        self.assertEqual(
            [item["stage_id"] for item in second["dag_dependency_stage_results"]],
            ["prepare"],
        )


if __name__ == "__main__":
    unittest.main()
