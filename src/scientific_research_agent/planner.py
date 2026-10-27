"""Compile a scientific task into falsifiable claims, never a fixed tool sequence."""

from __future__ import annotations

import json
from typing import Any, Protocol

from .contracts import ComponentConclusion, EvidenceEntry, Hypothesis, JudgeDecision, LocalExecutionGraph, MissingDiagnosis, PlanPatchOperation, PlannerReview, RequiredComponent, ResearchBrief, ScientificAnswerStrategy, ScientificPlan, ScientificTaskAnalysis, ScoutReport
from .question import task_from_brief
from .skills import SkillRegistry, default_skill_registry
from .validation import ContractViolation, validate_plan


class PlannerModel(Protocol):
    def plan(self, task: object, scout: ScoutReport) -> ScientificPlan: ...


class ScientificPlanner:
    def __init__(self, skills: SkillRegistry | None = None, advisor: Any | None = None, review_advisor: Any | None = None, *, allow_test_rules: bool = False, evidence_vocabulary: list[str] | None = None, vocabulary: Any | None = None) -> None:
        self.skills = skills or default_skill_registry()
        self.advisor = advisor
        # A production caller that supplies the normal PlanAdvisor should not
        # silently fall back to fixture re-planning merely because it omitted
        # the companion reviewer.  Keep the import local to avoid the
        # planner/reasoning module cycle at import time.
        if review_advisor is None and advisor is not None and hasattr(advisor, "reasoner") and not allow_test_rules:
            from .reasoning import PlannerReviewAdvisor
            review_advisor = PlannerReviewAdvisor(advisor.reasoner)
        self.review_advisor = review_advisor
        self.allow_test_rules = allow_test_rules
        # Scientific evidence requirements are open semantic strings. The
        # live tool/source catalogue is a Researcher capability boundary, not
        # a Planner enum or prerequisite for accepting a scientific claim.

    def plan(self, task: object, scout: ScoutReport, *, research_brief: ResearchBrief | None = None) -> ScientificPlan:
        # The LangGraph Interpretation node normally supplies both objects,
        # but keep the public Planner boundary explicit for direct callers:
        # a Brief is normalized here, never reinterpreted as a legacy task.
        if isinstance(task, ResearchBrief):
            research_brief = research_brief or task
            task = task_from_brief(research_brief.core_phenomenon_or_relation, research_brief)
        # Scout supplies only a stable planning boundary.  Exact source fields,
        # predicates, samples, and trial queries are resolved later inside the
        # Researcher's local execution graph.
        if scout.task_scope.get("_scope_contract") != "planning_scope/v1":
            raise ContractViolation("Planner requires a lightweight Scout planning scope")
        if self.advisor is not None:
            proposal = self.advisor.propose(
                task, scout, self.skills.all(), evidence_vocabulary=None,
                research_brief=research_brief,
            )
            plan = self._plan_from_proposal(self._proposal_seed(task, scout, research_brief), proposal)
            plan.research_brief = research_brief
            return validate_plan(self._bind_scout_scope(plan, scout))
        if not self.allow_test_rules:
            raise ContractViolation(
                "A Planner Agent is required for scientific research. "
                "Fixed fixture rules are available only with allow_test_rules=True."
            )
        plan = self._bind_scout_scope(self._fixture_plan(task, scout), scout)
        plan.research_brief = research_brief
        return validate_plan(plan)

    def review_local_execution(self, plan: ScientificPlan, scout: ScoutReport, graph: LocalExecutionGraph, decision: JudgeDecision, *, component_conclusions: list[ComponentConclusion], component_evidence: list[EvidenceEntry], patch_rejection: dict[str, Any] | None = None) -> PlannerReview:
        """Let the top Planner, never the Judge or scheduler, choose re-route.

        The model gets the complete global argument graph plus the durable
        local execution dossier.  The deterministic branch is only for
        explicit fixture plans, where no Planner model is installed.
        """

        component = plan.component(graph.component_id)
        if self.review_advisor is not None:
            raw = self.review_advisor.review(
                question=plan.task.question, plan=plan, component=component,
                local_execution_graph=graph.to_local_view({}), judge_decision=decision, scout=scout,
                component_conclusions=component_conclusions, component_evidence=component_evidence,
                patch_rejection=patch_rejection,
            )
            diagnoses = self._review_diagnoses(component.component_id, raw.get("diagnoses", []))
            affected = [item for item in raw.get("affected_component_ids", []) if item in {node.component_id for node in plan.required_components}]
            route = raw.get("route")
            if route not in {"repair_local", "patch_graph", "formalize_gap"}:
                raise ContractViolation("Planner review returned an unknown route")
            operations = self._review_operations(raw.get("operations", []), plan, scout)
            if route == "repair_local" and operations:
                raise ContractViolation("Planner repair_local review cannot contain global map operations")
            if route == "patch_graph" and not operations:
                raise ContractViolation("Planner patch_graph review requires at least one map operation")
            return PlannerReview(route, component.component_id, str(raw.get("rationale", "")), diagnoses, affected or [component.component_id], operations)
        # Test-only policy: a direct local re-scout request refreshes the
        # observation world; otherwise preserve the Judge's evidence boundary
        # as an explicit gap rather than pretending to discover a new route.
        if graph.execution_status == "escalated" and "unchanged" not in graph.escalation_note.lower():
            return PlannerReview("patch_graph", component.component_id, graph.escalation_note, [], [component.component_id], [PlanPatchOperation("rescout", component.component_id, {"patch_scope": {}, "reason": graph.escalation_note})])
        diagnoses = [diagnosis for judgment in decision.component_judgments for diagnosis in judgment.diagnoses]
        # The Judge does not classify a route.  Its deterministic assessment
        # does retain the current packet's missing evidence and availability
        # state; this fixture-only Planner reads that dossier to choose a
        # route.  Model-backed Planner review performs this reasoning itself.
        if not diagnoses:
            for assessment in decision.assessments:
                if assessment.component_id != component.component_id or assessment.outcome != "insufficient" or not assessment.gap_kind:
                    continue
                suggested = {
                    "resolution_limit": ["spatial_constraint"],
                    "missing_external_sensor": ["external_boundary_condition"],
                    "fundamentally_unidentifiable": ["material_state_sensor"],
                    "wrong_scope_or_data_assumption": ["event_clock_sensor"],
                    "calculation_missing": [],
                }.get(assessment.gap_kind, [])
                # A frozen component scope is not a patch.  Passing the whole
                # compiled bundle back into Scout recursively recompiled its
                # own metadata and made an unchanged world look different.
                diagnoses.append(MissingDiagnosis(component.component_id, assessment.gap_kind, assessment.rationale, list(assessment.missing_evidence), suggested, "high" if suggested else "medium", {}))
        absent = [need for need, item in graph.capability_coverage.items() if item.get("status") == "capability_absent"]
        blocked = [need for need, item in graph.capability_coverage.items() if item.get("status") == "input_not_ready"]
        external_blocked = [
            need for need in blocked
            if any(
                assessment.get("kind") == "external_capability" and not assessment.get("satisfied")
                for provider in graph.capability_coverage[need].get("provider_readiness", [])
                for assessment in provider.get("assessments", [])
            )
        ]
        if absent:
            diagnoses.append(MissingDiagnosis(component.component_id, "missing_tool_capability", "The current Tool Registry has no method that provides part of the component's declared evidence need.", absent, [], "high", {}))
        if external_blocked:
            diagnoses.append(MissingDiagnosis(component.component_id, "missing_external_sensor", "Registered methods require versioned external datasets that are unavailable in the current observation world.", external_blocked, ["external_boundary_condition"], "high", {}))
        elif blocked:
            diagnoses.append(MissingDiagnosis(component.component_id, "tool_input_unavailable", "Registered methods exist, but their machine-readable input contracts are not satisfied in the current local graph.", blocked, [], "high", {}))
        if any(item.kind in {"resolution_limit", "wrong_scope_or_data_assumption"} for item in diagnoses):
            return PlannerReview("patch_graph", component.component_id, graph.escalation_note or "The Planner requests a refreshed observation-world readout before changing the route.", diagnoses, [component.component_id], [PlanPatchOperation("rescout", component.component_id, {"patch_scope": {}, "reason": graph.escalation_note})])
        return PlannerReview("formalize_gap", component.component_id, graph.escalation_note or "The bounded local execution graph is exhausted.", diagnoses, [component.component_id])

    def apply_global_review(self, plan: ScientificPlan, scout: ScoutReport, review: PlannerReview) -> ScientificPlan:
        """Commit a Planner-owned scientific-map patch."""

        from .evolver import PlanEvolver

        patched = PlanEvolver().apply_global_review(plan, scout, review)
        return validate_plan(patched)

    @staticmethod
    def _review_operations(raw_operations: list[dict[str, Any]], plan: ScientificPlan, scout: ScoutReport) -> list[PlanPatchOperation]:
        known = {item.component_id for item in plan.required_components}
        operations: list[PlanPatchOperation] = []
        for item in raw_operations:
            kind = item.get("operation")
            reason = str(item.get("reason", ""))
            if kind == "add_component":
                raw = item.get("component")
                if not isinstance(raw, dict):
                    raise ContractViolation("Planner add_component operation lacks a typed component")
                try:
                    constraints = json.loads(raw["constraints_json"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ContractViolation("Planner add_component constraints_json must be an object") from error
                if not isinstance(constraints, dict):
                    raise ContractViolation("Planner add_component constraints_json must decode to an object")
                component_id = raw.get("component_id")
                depends_on = list(raw.get("depends_on", []))
                evidence = list(raw.get("evidence_needed", []))
                if not isinstance(component_id, str) or not component_id or component_id in known or not set(depends_on).issubset(known):
                    raise ContractViolation("Planner add_component has an invalid id or dependency")
                if not evidence:
                    raise ContractViolation("Planner add_component has no evidence requirement")
                operations.append(PlanPatchOperation("add_component", component_id, {"component": {
                    "component_id": component_id, "claim": str(raw["claim"]), "depends_on": depends_on,
                    "evidence_needed": evidence, "constraints": constraints, "failure_meaning": str(raw["failure_meaning"]),
                    "candidate_hypothesis_ids": list(raw.get("candidate_hypothesis_ids", [])), "skill_ids": list(raw.get("skill_ids", [])),
                }, "reason": reason}))
            elif kind in {"supersede_component", "reopen_component"}:
                target = item.get("target_component_id")
                if target not in known:
                    raise ContractViolation("Planner operation targets an unknown component")
                operations.append(PlanPatchOperation("supersede" if kind == "supersede_component" else "reopen_component", target, {"reason": reason}))
            elif kind == "reconnect_dependencies":
                target = item.get("target_component_id")
                depends_on = list(item.get("depends_on", []))
                if target not in known or target in depends_on or not set(depends_on).issubset(known):
                    raise ContractViolation("Planner reconnect_dependencies has an invalid graph edge")
                operations.append(PlanPatchOperation("reconnect_dependencies", target, {"depends_on": depends_on, "reason": reason}))
            elif kind == "rescout":
                target = item.get("target_component_id")
                if target not in known:
                    raise ContractViolation("Planner rescout targets an unknown component")
                try:
                    patch_scope = json.loads(item.get("rescout_scope_json", "{}"))
                except (TypeError, ValueError) as error:
                    raise ContractViolation("Planner rescout_scope_json must be an object") from error
                if not isinstance(patch_scope, dict):
                    raise ContractViolation("Planner rescout_scope_json must decode to an object")
                operations.append(PlanPatchOperation("rescout", target, {"patch_scope": patch_scope, "reason": reason}))
            else:
                raise ContractViolation("Planner review contains an unknown map operation")
        return operations

    @staticmethod
    def _review_diagnoses(component_id: str, raw_diagnoses: list[dict[str, Any]]) -> list[MissingDiagnosis]:
        valid_kinds = {"calculation_missing", "wrong_scope_or_data_assumption", "missing_external_sensor", "missing_tool_capability", "tool_input_unavailable", "weak_optional_source", "fundamentally_unidentifiable", "resolution_limit"}
        valid_sensors = {"event_clock_sensor", "spatial_constraint", "material_state_sensor", "external_boundary_condition", "forward_physical_model", "opportunistic_sensor"}
        diagnoses: list[MissingDiagnosis] = []
        for item in raw_diagnoses:
            if item.get("kind") not in valid_kinds or not isinstance(item.get("reason"), str):
                continue
            sensors = [value for value in item.get("suggested_sensor_kinds", []) if value in valid_sensors]
            gain = item.get("expected_information_gain", "medium")
            diagnoses.append(MissingDiagnosis(component_id, item["kind"], item["reason"], [str(value) for value in item.get("missing_observables", [])], sensors, gain if gain in {"low", "medium", "high"} else "medium", item.get("patch_scope", {}) if isinstance(item.get("patch_scope", {}), dict) else {}))
        return diagnoses

    @staticmethod
    def _bind_scout_scope(plan: ScientificPlan, scout: ScoutReport) -> ScientificPlan:
        """Attach the user's rough boundary without compiling a data query."""

        if scout.task_scope.get("_scope_contract") != "planning_scope/v1":
            raise ContractViolation("Planner requires a Scout planning scope")
        for component in plan.required_components:
            component.scope = scout.task_scope
        return plan

    def _fixture_plan(self, task: object, scout: ScoutReport) -> ScientificPlan:
        """Deterministic test-only plan for synthetic integration fixtures.

        It accepts an explicitly built ``QuestionTask``; it must never receive
        or classify a natural-language question in a scientific run.
        """
        # Skills are a catalogue for the Planner Agent, not a keyword router.
        # This fixture path intentionally attaches none of them.
        matched_skills: list[str] = []
        migration = "identify_spatiotemporal_pattern" in task.scientific_goals
        mechanism = "compare_candidate_mechanisms" in task.scientific_goals
        components = [self._temporal_component(task, matched_skills)]
        hypotheses: list[Hypothesis] = []
        if migration:
            components.extend(self._migration_components(task, matched_skills, mechanism))
            hypotheses = self._migration_hypotheses()
        else:
            components.extend(self._relationship_components(task, matched_skills))
        if "multi_chronometer_thermal_history" in matched_skills:
            components.append(RequiredComponent(
                "multi_clock_ordering", "不同计时器是否在可比样品/区域中记录可解释的事件顺序或时滞？",
                ["temporal_event_structure"], ["multi_clock", "event_interpretation", "mineral", "age_uncertainty"],
                {"validate_clock_compatibility": True, "do_not_subtract_unmatched_ages": True},
                "若事件不具有物理可比性，禁止将简单年龄差写成热历史。", task.scope,
                skill_ids=["multi_chronometer_thermal_history"],
            ))
        known_gaps = list(scout.sensor_limits)
        analysis = ScientificTaskAnalysis(
            user_intent="synthetic contract-test task",
            explicit_requirements=list(task.scientific_goals) or ["explicit fixture task required"],
            implicit_requirements=["先确认传感器覆盖和分辨率", "量化不确定性与采样偏差", "区分观测模式与机制解释"],
            out_of_scope=["未注册来源的实时网页检索", "没有 Ledger 支持的地质断言"],
            language="fixture",
        )
        plan = ScientificPlan(
            task=task, task_analysis=analysis,
            scientific_goal="从带不确定性和采样偏差的观测中反演潜在时空过程，并在可辨识范围内比较机制。",
            required_components=components, candidate_hypotheses=hypotheses,
            current_sensor_limits=list(scout.sensor_limits), known_evidence_gaps=known_gaps, scout_fingerprint=scout.world_fingerprint,
            answer_strategy=ScientificAnswerStrategy("scientific_report", "Evidence-bound scientific inference", [component.component_id for component in components], ["区分 observed pattern、supported interpretation、alternative explanations 与 evidence gaps"]),
            constraints=["所有核心结论必须有 Ledger 证据", *scout.sensor_limits],
        )
        return validate_plan(plan)

    @staticmethod
    def _proposal_seed(task: object, scout: ScoutReport, research_brief: ResearchBrief | None = None) -> ScientificPlan:
        """Transport context only; model output supplies every scientific claim."""
        return ScientificPlan(
            task=task,
            task_analysis=ScientificTaskAnalysis(
                user_intent="pending planner-agent proposal",
                explicit_requirements=[], implicit_requirements=[], out_of_scope=[], language="unspecified",
            ),
            scientific_goal="pending planner-agent proposal",
            required_components=[], candidate_hypotheses=[],
            current_sensor_limits=list(scout.sensor_limits), known_evidence_gaps=list(scout.sensor_limits),
            scout_fingerprint=scout.world_fingerprint,
            answer_strategy=ScientificAnswerStrategy("custom", "pending planner-agent proposal", [], []),
            research_brief=research_brief,
        )

    @staticmethod
    def _plan_from_proposal(seed: ScientificPlan, proposal: dict[str, Any]) -> ScientificPlan:
        """DeepSeek has already server-validated the strict tool JSON Schema.

        This boundary converts its closed tool arguments into runtime contracts;
        ``validate_plan`` below remains the scientific DAG safety gate.
        """
        components: dict[str, RequiredComponent] = {}
        for item in proposal.get("required_components", []):
            try:
                if not all(isinstance(item.get(field), list) for field in ("depends_on", "evidence_needed", "candidate_hypothesis_ids", "skill_ids")) or not isinstance(item.get("constraints"), dict) or not isinstance(item.get("scope"), dict):
                    raise TypeError("component arrays, constraints and scope must have their declared JSON types")
                component = RequiredComponent(
                    component_id=item["component_id"], claim=item["claim"], depends_on=item["depends_on"],
                    evidence_needed=item["evidence_needed"], constraints=item["constraints"],
                    failure_meaning=item["failure_meaning"], scope=item["scope"],
                    candidate_hypothesis_ids=item.get("candidate_hypothesis_ids", []), skill_ids=item.get("skill_ids", []),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ContractViolation(f"Planner proposal has malformed component: {item}") from error
            components[component.component_id] = component
        hypotheses: list[Hypothesis] = []
        for item in proposal.get("candidate_hypotheses", []):
            try:
                hypotheses.append(Hypothesis(item["hypothesis_id"], item["label"], item["description"], item["predictions"], item["differentiating_observables"]))
            except (KeyError, TypeError) as error:
                raise ContractViolation(f"Planner proposal has malformed hypothesis: {item}") from error
        if not components:
            raise ContractViolation("Planner produced no required components")
        brief = seed.research_brief
        analysis = ScientificTaskAnalysis(
            user_intent=brief.core_phenomenon_or_relation if brief else str(getattr(seed.task, "question", "")),
            explicit_requirements=list(brief.must_answer) if brief else [],
            implicit_requirements=[],
            out_of_scope=[],
            language="unspecified",
            ambiguities=list(brief.key_assumptions_and_unresolved_ambiguities) if brief else [],
        )
        strategy = ScientificAnswerStrategy(
            "scientific_report", "Evidence-bound scientific inference",
            [component.component_id for component in components.values()],
            ["逐项处理 Research Brief 的认识缺口；不把研究动作当作答案要求。"],
        )
        constraints = []
        return ScientificPlan(
            task=seed.task, task_analysis=analysis, scientific_goal=brief.core_phenomenon_or_relation if brief else seed.scientific_goal,
            required_components=list(components.values()), candidate_hypotheses=hypotheses,
            current_sensor_limits=list(seed.current_sensor_limits), known_evidence_gaps=list(seed.known_evidence_gaps), scout_fingerprint=seed.scout_fingerprint,
            answer_strategy=strategy, constraints=constraints,
        )

    @staticmethod
    def _temporal_component(task: object, skills: list[str]) -> RequiredComponent:
        return RequiredComponent(
            "temporal_event_structure", "给定范围内是否存在稳定、可区分的事件时间结构，而不是少数离群值或单一来源贡献？",
            [], ["event_age", "age_uncertainty", "source_provenance", "temporal_structure"],
            {"bootstrap": True, "do_not_infer_unobserved_events": True},
            "无法识别时，后续空间或机制解释必须停止在 evidence gap，不能将年龄峰当作事实。", task.scope,
            skill_ids=skills,
        )

    @staticmethod
    def _migration_components(task: object, skills: list[str], mechanism: bool) -> list[RequiredComponent]:
        components = [
            RequiredComponent(
                "spatial_observability", "当前位置观测及其不确定性是否足以区分用户要求尺度上的空间变化？",
                [], ["spatial_location", "spatial_uncertainty", "resolution_assessment"],
                {"respect_position_provenance": True, "minimum_coordinate_coverage": .5},
                "若空间分辨率不足，允许报告区域级结构，但禁止推断精细迁移路径。", task.scope, skill_ids=skills,
            ),
            RequiredComponent(
                "spatiotemporal_pattern", "在事件阶段与空间可观测性成立时，活动中心、密度面或前缘是否发生系统且定向的变化？",
                ["temporal_event_structure", "spatial_observability"], ["event_age", "spatial_location", "spatiotemporal_pattern"],
                {"report_effect_size": True, "test_multiple_spatial_summaries": True},
                "若不同统计表征不一致或移动幅度低于分辨率，迁移命题为 insufficient。", task.scope,
                candidate_hypothesis_ids=["H1_continuous_migration", "H2_two_stationary_centers", "H3_coordinate_artifact", "H4_sampling_artifact"], skill_ids=skills,
            ),
            RequiredComponent(
                "robustness_and_bias", "观察到的空间—时间模式在年龄/空间误差传播、留一来源、重采样与零模型下是否仍存在？",
                ["spatiotemporal_pattern"], ["uncertainty_propagation", "sampling_bias", "null_model_test", "alternative_specification"],
                {"monte_carlo": True, "leave_one_source_out": True, "permutation": True},
                "若模式依赖单一来源、阈值或采样格局，则不能称为潜在过程的证据。", task.scope,
                candidate_hypothesis_ids=["H4_sampling_artifact"], skill_ids=skills,
            ),
        ]
        if mechanism:
            components.extend([
                RequiredComponent(
                    "paleogeographic_persistence", "在可用古地理恢复模型下，模式是否仍然存在，而不是现代坐标框架造成的表观迁移？",
                    ["robustness_and_bias"], ["plate_rotation", "reconstruct_paleolocation"],
                    {"preserve_reconstruction_uncertainty": True},
                    "若恢复后趋势消失，现代坐标解释优先，不能进入构造驱动断言。", task.scope,
                    candidate_hypothesis_ids=["H3_coordinate_artifact"], skill_ids=skills,
                ),
                RequiredComponent(
                    "external_boundary_compatibility", "外部边界条件变化是否与模式在时间、空间和方向上兼容？",
                    ["paleogeographic_persistence"], ["external_boundary_condition", "distance_to_trench", "plate_velocity"],
                    {"require_versioned_local_dataset": True},
                    "缺少覆盖期的版本化外部边界数据时，机制关联为 evidence gap，而非网页检索替代。", task.scope,
                    candidate_hypothesis_ids=["H5_external_boundary_driven"], skill_ids=skills,
                ),
                RequiredComponent(
                    "mechanism_discrimination", "候选机制是否产生可区分预测，且当前传感器能否排除至少一个替代机制？",
                    ["external_boundary_compatibility"], ["mechanism_prediction", "forward_model", "model_comparison", "material_state"],
                    {"compare_alternatives": True, "allow_unidentifiable": True},
                    "若候选模型生成近似相同观测，结论必须是 association supported / causality unresolved。", task.scope,
                    candidate_hypothesis_ids=["H1_continuous_migration", "H2_two_stationary_centers", "H3_coordinate_artifact", "H4_sampling_artifact", "H5_external_boundary_driven"], skill_ids=skills,
                ),
            ])
        return components

    @staticmethod
    def _relationship_components(task: object, skills: list[str]) -> list[RequiredComponent]:
        return [
            RequiredComponent(
                "target_relationship", "用户要求的事件关系是否能由当前传感器直接观测、量化且表达不确定性？",
                ["temporal_event_structure"], ["event_age", "target_relationship", "age_uncertainty"],
                {"report_effect_size": True}, "若关系不可观测，领域常识不能代替数据结论。", task.scope, skill_ids=skills,
            ),
            RequiredComponent(
                "robustness_and_bias", "该关系是否通过误差传播和采样偏差检验？",
                ["target_relationship"], ["uncertainty_propagation", "sampling_bias", "null_model_test"],
                {"monte_carlo": True, "leave_one_source_out": True}, "若不稳健，只能报告为暂定模式。", task.scope, skill_ids=skills,
            ),
        ]

    @staticmethod
    def _migration_hypotheses() -> list[Hypothesis]:
        return [
            Hypothesis("H1_continuous_migration", "连续活动源迁移", "潜在活动源连续移动。", ["年龄与位置存在连续关系", "可拟合稳定传播方向/速度"], ["migration_velocity", "continuous_front_fit"]),
            Hypothesis("H2_two_stationary_centers", "两个独立静态中心", "观测来自不同时间脉冲的两个稳定事件中心。", ["空间分布更接近离散簇", "连续前缘模型不优于双中心模型"], ["spatial_clustering", "stationary_center_forward_model"]),
            Hypothesis("H3_coordinate_artifact", "现代坐标表观迁移", "板块运动或坐标框架使稳定过程显得迁移。", ["古地理恢复后趋势减弱或消失"], ["reconstruct_paleolocation"]),
            Hypothesis("H4_sampling_artifact", "采样/发表偏差", "空间与时间覆盖不均产生伪趋势。", ["重采样、置换或留一来源后趋势消失"], ["spatial_resampling", "permutation", "leave_one_source_out"]),
            Hypothesis("H5_external_boundary_driven", "外部构造边界驱动", "外部边界的迁移或运动与活动中心变化一致。", ["时间、空间、方向与边界变化匹配", "独立物质状态传感器给出兼容预测"], ["distance_to_trench", "plate_velocity", "material_state"]),
        ]

    def patch_plan(self, plan: ScientificPlan, scout: ScoutReport, diagnoses: list[MissingDiagnosis]) -> ScientificPlan:
        """Compatibility facade for explicit local repairs.

        The old implementation reset mutable component status.  The PlanEvolver
        instead records an append-only revision with the triggering diagnosis;
        existing Ledger evidence and prior assessments are retained.
        """
        from .evolver import PlanEvolver

        return validate_plan(PlanEvolver().apply_diagnoses(plan, scout, diagnoses))
