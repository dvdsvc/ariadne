"""Final synthesis bounded strictly by plan, ledger, and judge output."""

from __future__ import annotations

from typing import Any

from .aiq_grounding import ClaimEvidenceValidator, EvidenceSourceRegistry, ReportClaim, verify_report_citations
from .contracts import (
    ComponentConclusion, ComponentJudgment, DAGSynthesis, EvidenceGap, EvidenceStatus,
    FinalAnswer, GoalAssessment, ResearchReport, ScientificPlan, ScoutReport,
    WriterSection, WriterSubgraphPacket,
)
from .ledger import EvidenceLedger
from .plan_graph import evidence_input_component_ids


class EvidenceBoundWriter:
    def __init__(self, *, claim_advisor: Any | None = None) -> None:
        self.claim_advisor = claim_advisor

    def write(self, plan: ScientificPlan, scouts: list[ScoutReport], ledger: EvidenceLedger, judgments: list[ComponentJudgment], gaps: list[EvidenceGap], *, research_notes: list[Any] | None = None, evidence_registry: EvidenceSourceRegistry | None = None) -> ResearchReport:
        research_notes = research_notes or []
        full_registry = evidence_registry or EvidenceSourceRegistry(ledger.entries)
        registry = full_registry.compact_for_notes(research_notes)
        claims = self._claims(plan, judgments, ledger)
        manifest = ClaimEvidenceValidator(registry, self.claim_advisor).validate(claims)
        claim_by_component = {claim.component_id: claim for claim in manifest}
        cited_evidence_ids = [evidence_id for claim in manifest if claim.status == "verified" for evidence_id in claim.evidence_ids]
        sources = registry.sources_for(cited_evidence_ids)
        citation_number = {source.locator: index for index, source in enumerate(sources, 1)}
        observed, interpretations, alternatives, unresolved = [], [], [], []
        by_id = {judgment.component_id: judgment for judgment in judgments}
        for component in plan.required_components:
            judgment = by_id[component.component_id]
            claim = claim_by_component[component.component_id]
            if claim.status == "rejected":
                unresolved.append(f"- **{component.component_id}**：结论未发布；{claim.validation_reason or 'evidence validation failed'}。")
                continue
            references = " ".join(f"[{citation_number[evidence_id]}]" for evidence_id in claim.evidence_ids if evidence_id in citation_number) or "（explicit evidence gap）"
            line = f"- **{component.component_id}**：{judgment.status}；{judgment.rationale} {references}"
            category = self._component_category(plan, component, ledger, judgment.allowed_claim_level)
            if category == "observed_pattern":
                observed.append(line)
            elif category == "interpretation" and judgment.status == "supported":
                interpretations.append(line)
            elif category in {"interpretation", "alternative"}:
                alternatives.append(line)
            if judgment.status == "insufficient":
                unresolved.append(f"- **{component.component_id}**：{', '.join(judgment.missing_evidence) or 'evidence incomplete'}。")
        next_observations = self._next_observations(gaps)
        strategy_title = plan.answer_strategy.title if plan.answer_strategy else "Evidence-bound scientific inference"
        lines = [f"# {strategy_title}\n\n**问题**：{plan.task.question}\n", "## Observed Pattern\n", "\n".join(observed) or "- 当前没有由 ledger 支持的观测模式。", "\n## Supported Interpretation\n", "\n".join(interpretations) or "- 当前没有能唯一成立的机制解释。", "\n## Alternative Explanations\n", "\n".join(alternatives) or "- 替代机制尚未被充分区分。", "\n## Evidence Gaps\n", "\n".join(unresolved) or "- 无未决 component。"]
        if plan.candidate_hypotheses:
            lines.append("\n## Candidate mechanism status\n")
            for hypothesis in plan.candidate_hypotheses:
                lines.append(f"- **{hypothesis.hypothesis_id} / {hypothesis.label}**：{hypothesis.status}；可区分观测：{', '.join(hypothesis.differentiating_observables)}。")
        if gaps:
            lines.append("\n### Structured gaps\n")
            for gap in gaps:
                lines.append(f"- **{gap.gap_id} / {gap.target_component}**：缺 `{gap.missing_observable}`；候选传感器 `{gap.candidate_sensor or 'none'}`；预期信息增益 {gap.expected_information_gain}；结论上限：{gap.conclusion_limit}。")
        lines.extend(["\n## Next observation priorities\n", "\n".join(f"- {item}" for item in next_observations) or "- 当前没有明确的新观测建议。", "\n## Data and uncertainty audit\n"])
        for scout in scouts:
            for profile in scout.sources:
                lines.append(f"- `{profile.source_id}` v`{profile.data_version}`：匹配 {profile.matched_records}/{profile.total_records}；空间覆盖 {profile.resolution.get('coordinate_coverage', 0):.1%}；中位空间 σ {profile.resolution.get('median_spatial_uncertainty_km')} km；中位年龄误差 {profile.uncertainty.get('median_age_error_ma')} Ma。")
            for dataset in scout.external_datasets:
                lines.append(f"- 外部数据 `{dataset.get('dataset_id')}` v`{dataset.get('version')}`：{'available' if dataset.get('available') else 'data_unavailable'}。")
        lines.append("\n## Evidence Ledger\n")
        for entry in ledger.entries:
            lines.append(f"- **{entry.evidence_id}** `{entry.subject_id}`：{entry.result_summary}；计算状态={entry.calculation_status}；版本 {entry.data_versions}；不确定性 {entry.uncertainty or '未提供'}。")
        lines.append("\n## Sources\n")
        for index, source in enumerate(sources, 1):
            lines.append(f"[{index}] {source.title}; evidence_id={source.locator}; versions={source.data_versions}; scope={source.input_scope}")
        markdown, removed_citations = verify_report_citations("\n".join(lines), registry)
        if removed_citations:
            markdown += "\n\n- Citation verification removed unregistered source references: " + ", ".join(str(number) for number in removed_citations) + "."
        return ResearchReport(plan.task.question, markdown, plan.task, plan, scouts, ledger.entries.copy(), judgments, gaps, next_observations, research_notes, manifest, sources, research_brief=plan.research_brief)

    @staticmethod
    def _claims(plan: ScientificPlan, judgments: list[ComponentJudgment], ledger: EvidenceLedger | None = None) -> list[ReportClaim]:
        claims: list[ReportClaim] = []
        for judgment in judgments:
            component = plan.component(judgment.component_id)
            category = EvidenceBoundWriter._component_category(plan, component, ledger, judgment.allowed_claim_level)
            if judgment.status == "insufficient" or judgment.allowed_claim_level == "none":
                category = "gap"
            claims.append(ReportClaim(
                claim_id=f"claim_{judgment.component_id}", component_id=judgment.component_id,
                category=category, text=f"{judgment.claim} {judgment.rationale}",
                evidence_ids=list(judgment.evidence_ids), status="pending",
            ))
        return claims

    @staticmethod
    def _component_category(plan: ScientificPlan, component: Any, ledger: EvidenceLedger | None, allowed_claim_level: str = "component_only") -> str:
        # The isolated adjudicator controls the highest semantic layer Writer
        # can use.  Evidence tags may choose wording within that layer, never
        # promote a pattern result to a mechanism claim.
        if allowed_claim_level in {"none", "pattern_only", "component_only"}:
            return "observed_pattern"
        entries = ledger.for_components(evidence_input_component_ids(plan, component.component_id)) if ledger else []
        tags = {tag for entry in entries for tag in entry.evidence_tags}
        if {"external_boundary_condition", "distance_to_trench", "plate_velocity", "model_comparison", "mechanism_prediction"}.intersection(tags):
            return "interpretation"
        if {"source_provenance", "spatial_resampling", "sampling_density_normalization", "publication_contribution"}.intersection(tags):
            return "alternative"
        return "observed_pattern"

    @staticmethod
    def _next_observations(gaps: list[EvidenceGap]) -> list[str]:
        recommendations: list[str] = []
        for gap in gaps:
            if gap.candidate_sensor == "material_state_sensor":
                recommendations.append("优先接入 Hf/Nd 等物质来源状态传感器，以区分时间—空间关联与深部驱动机制。")
            elif gap.candidate_sensor == "spatial_constraint":
                recommendations.append("优先补充带 provenance 的 paper GPS 或 locality geocoding，并保留空间 σ，提升空间命题分辨率。")
            elif gap.candidate_sensor == "external_boundary_condition":
                recommendations.append("注册覆盖目标时段、具有版本/许可/校验和的本地板块旋转与古海沟几何数据。")
        return list(dict.fromkeys(recommendations))


class LocalSubgraphWriter:
    """Translate one isolated conclusion-only packet into prose.

    The local writer never receives the question, raw ledger, Scout report,
    Researcher history, or the global DAG.  It can only rewrite the
    conclusions and edges that the global assembler placed in this packet.
    """

    def __init__(self, advisor: Any | None = None) -> None:
        self.advisor = advisor

    @staticmethod
    def _fallback_section(packet: WriterSubgraphPacket) -> WriterSection:
        by_component = {item.component_id: item for item in packet.conclusions}
        root = by_component.get(packet.root_component_id)
        if root is None:
            text = f"节点 {packet.root_component_id} 尚未形成局部结论；该分支保留为证据缺口。"
            status: EvidenceStatus = "insufficient"
            claim_level = "none"
        else:
            prerequisites = [
                f"{component_id}={by_component[component_id].status}"
                for component_id in packet.component_ids
                if component_id != packet.root_component_id and component_id in by_component
            ]
            prefix = f"前置节点状态：{', '.join(prerequisites)}。" if prerequisites else "该节点无已声明的前置结论。"
            text = f"{prefix} 节点 {root.component_id}：{root.conclusion}"
            status = root.status
            claim_level = root.allowed_claim_level
        return WriterSection(
            section_id=f"writer:{packet.root_component_id}",
            root_component_id=packet.root_component_id,
            component_ids=list(packet.component_ids),
            node_ids=list(packet.node_ids),
            edge_ids=list(packet.edge_ids),
            text=text,
            status=status,
            allowed_claim_level=claim_level,
        )

    def compile(self, packet: WriterSubgraphPacket) -> WriterSection:
        """Compile one packet; an advisor may only rewrite its local text."""

        section = self._fallback_section(packet)
        if self.advisor is None:
            return section
        try:
            proposal = self.advisor.write_subgraph(packet)
            text = proposal.get("text")
            proposed_components = [item for item in proposal.get("component_ids", []) if isinstance(item, str)]
            proposed_edges = [item for item in proposal.get("edge_ids", []) if isinstance(item, str)]
            if (
                isinstance(text, str) and text.strip()
                and proposal.get("root_component_id") == packet.root_component_id
                and set(proposed_components).issubset(set(packet.component_ids))
                and set(proposed_edges).issubset(set(packet.edge_ids))
            ):
                root = next((item for item in packet.conclusions if item.component_id == packet.root_component_id), None)
                return WriterSection(
                    section_id=f"writer:{packet.root_component_id}",
                    root_component_id=packet.root_component_id,
                    component_ids=proposed_components or list(packet.component_ids),
                    node_ids=list(packet.node_ids),
                    edge_ids=proposed_edges or list(packet.edge_ids),
                    text=text.strip(),
                    status=root.status if root else "insufficient",
                    allowed_claim_level=root.allowed_claim_level if root else "none",
                )
        except Exception:
            # Wording failure never changes evidence status or research state.
            pass
        return section


class DAGBoundWriter:
    """Compile conclusion-only local subgraphs into a final report.

    This class is deliberately not a Judge.  It never decides whether a
    result supports a claim and never changes the Plan.  The Plan DAG selects
    the local subgraph, its topological order, and its dependency edges; the
    optional advisor only translates that packet into natural language.
    """

    def __init__(self, subgraph_advisor: Any | None = None) -> None:
        self.subgraph_advisor = subgraph_advisor
        self.local_writer = LocalSubgraphWriter(subgraph_advisor)

    @staticmethod
    def _topological_components(plan: ScientificPlan) -> list[str]:
        """Return a stable topological order from the global component DAG."""

        component_ids = [item.component_id for item in plan.required_components]
        known = set(component_ids)
        dependencies = {
            item.component_id: [dependency for dependency in item.depends_on if dependency in known]
            for item in plan.required_components
        }
        order: list[str] = []
        remaining = set(component_ids)
        while remaining:
            ready = [item for item in component_ids if item in remaining and not (set(dependencies[item]) & remaining)]
            if not ready:
                raise ValueError("Writer cannot compile a cyclic component DAG")
            order.extend(ready)
            remaining.difference_update(ready)
        return order

    @staticmethod
    def _packet_for(plan: ScientificPlan, root_component_id: str, conclusions: dict[str, ComponentConclusion], order: list[str]) -> WriterSubgraphPacket:
        component_ids = {root_component_id}
        stack = [root_component_id]
        while stack:
            current = stack.pop()
            component = plan.component(current)
            for dependency in component.depends_on:
                if dependency not in component_ids:
                    component_ids.add(dependency)
                    stack.append(dependency)
        ordered_components = [item for item in order if item in component_ids]
        node_ids = [f"component:{component_id}" for component_id in ordered_components]
        graph = plan.graph
        edge_ids: list[str] = []
        if graph is not None:
            nodes = {node.node_id: node for node in graph.nodes}
            for edge in graph.edges:
                source = nodes.get(edge.source_id)
                target = nodes.get(edge.target_id)
                if source is None or target is None:
                    continue
                if source.component_id in component_ids and target.component_id in component_ids:
                    edge_ids.append(f"{edge.source_id}->{edge.target_id}")
        return WriterSubgraphPacket(
            root_component_id=root_component_id,
            component_ids=ordered_components,
            node_ids=node_ids,
            edge_ids=edge_ids,
            conclusions=[conclusions[item] for item in ordered_components if item in conclusions],
        )

    @staticmethod
    def _fallback_section(packet: WriterSubgraphPacket) -> WriterSection:
        by_component = {item.component_id: item for item in packet.conclusions}
        root = by_component.get(packet.root_component_id)
        if root is None:
            text = f"节点 {packet.root_component_id} 尚未形成局部结论；该分支保留为证据缺口。"
            status: EvidenceStatus = "insufficient"
            claim_level = "none"
        else:
            prerequisites = [
                f"{component_id}={by_component[component_id].status}"
                for component_id in packet.component_ids
                if component_id != packet.root_component_id and component_id in by_component
            ]
            prefix = f"前置节点状态：{', '.join(prerequisites)}。" if prerequisites else "该节点无已声明的前置结论。"
            text = f"{prefix} 节点 {root.component_id}：{root.conclusion}"
            status = root.status
            claim_level = root.allowed_claim_level
        return WriterSection(
            section_id=f"writer:{packet.root_component_id}",
            root_component_id=packet.root_component_id,
            component_ids=list(packet.component_ids),
            node_ids=list(packet.node_ids),
            edge_ids=list(packet.edge_ids),
            text=text,
            status=status,
            allowed_claim_level=claim_level,
        )

    def compile_sections(self, plan: ScientificPlan, conclusions: list[ComponentConclusion], synthesis: DAGSynthesis) -> list[WriterSection]:
        """Translate every active scientific decision subgraph in DAG order."""

        order = self._topological_components(plan)
        by_component = {item.component_id: item for item in conclusions}
        answer_ids = {
            item.component_id for item in plan.required_components
            if item.execution_status != "superseded"
        }
        sections: list[WriterSection] = []
        for component_id in order:
            if component_id not in answer_ids:
                continue
            packet = self._packet_for(plan, component_id, by_component, order)
            # The global writer only chooses packet boundaries and topological
            # order.  Natural-language compilation is delegated to a local
            # writer that cannot see the rest of the run.
            sections.append(self.local_writer.compile(packet))
        return sections

    def write(self, plan: ScientificPlan, scouts: list[ScoutReport], conclusions: list[ComponentConclusion], synthesis: DAGSynthesis, goal: GoalAssessment, final_answer: FinalAnswer, *, evidence_registry: EvidenceSourceRegistry, writer_sections: list[WriterSection] | None = None) -> ResearchReport:
        by_component = {item.component_id: item for item in conclusions}
        writer_sections = writer_sections if writer_sections is not None else self.compile_sections(plan, conclusions, synthesis)
        source_ids = [evidence_id for item in conclusions for evidence_id in item.evidence_ids]
        sources = evidence_registry.sources_for(list(dict.fromkeys(source_ids)))
        numbers = {source.locator: index for index, source in enumerate(sources, 1)}
        observed, interpretations, alternatives, gaps = [], [], [], []
        claims: list[ReportClaim] = []
        for component in plan.required_components:
            conclusion = by_component.get(component.component_id)
            if conclusion is None:
                if component.component_id in synthesis.terminal_component_ids:
                    alternatives.append(f"- **{component.component_id}**：因上游前提已被反驳，DAG 正式剪枝，未执行该节点。")
                else:
                    gaps.append(f"- **{component.component_id}**：没有形成隔离节点结论。")
                continue
            citations = " ".join(f"[{numbers[item]}]" for item in conclusion.evidence_ids if item in numbers) or "（无可发布证据）"
            line = f"- **{component.component_id}**：{conclusion.status}；{conclusion.conclusion} {citations}"
            claim_category = "interpretation" if conclusion.allowed_claim_level == "mechanism_association" else "observed_pattern"
            claim_status = "rejected" if conclusion.writer_verdict == "rejected" else ("verified" if conclusion.status != "insufficient" else "gap")
            claims.append(ReportClaim(
                claim_id=f"claim_{component.component_id}", component_id=component.component_id,
                category=claim_category if conclusion.status != "insufficient" else "gap",
                text=conclusion.conclusion, evidence_ids=conclusion.evidence_ids,
                status=claim_status,
                validation_reason=conclusion.writer_verdict,
            ))
            if conclusion.status == "insufficient":
                gaps.append(f"- **{component.component_id}**：结论未发布；{conclusion.conclusion} {citations}")
            elif conclusion.allowed_claim_level == "mechanism_association":
                interpretations.append(line)
            elif conclusion.status == "contradicted":
                alternatives.append(line)
            else:
                observed.append(line)
        if goal.status != "proven":
            gaps.append(f"- **问题级验证**：{goal.reason}；阻断节点：{', '.join(goal.blocking_component_ids) or 'none'}。")
        lines = [
            f"# {plan.answer_strategy.title if plan.answer_strategy else 'Evidence-bound scientific inference'}\n\n**问题**：{plan.task.question}\n",
            "## Final Answer\n", final_answer.conclusion,
            "\n### Answer status\n", f"- `{final_answer.status}`; presentation artifact `{final_answer.answer_node_id}`.",
            "\n### Next action\n", "\n".join(f"- {item}" for item in final_answer.next_actions) or "- No additional action is required by the final answer contract.",
            "\n## DAG-compiled local sections\n",
            "\n".join(f"### {section.root_component_id}\n{section.text}" for section in writer_sections) or "- 没有可编译的局部子图。",
            "## Observed Pattern\n", "\n".join(observed) or "- 无可发布的观测结论。",
            "\n## Supported Interpretation\n", "\n".join(interpretations) or "- 当前没有经 DAG 放行的机制解释。",
            "\n## Alternative Explanations\n", "\n".join(alternatives) or "- 没有已证伪的替代节点。",
            "\n## Evidence Gaps\n", "\n".join(gaps) or "- 无未决证据缺口。",
            "\n## DAG verification\n", *[f"- {item}" for item in synthesis.dependency_summary],
            f"\n- Root verification: `{goal.status}` — {goal.reason}",
            "\n## Sources\n", *[f"[{index}] {source.title}; evidence_id={source.locator}; versions={source.data_versions}; scope={source.input_scope}" for index, source in enumerate(sources, 1)],
        ]
        markdown, removed = verify_report_citations("\n".join(lines), evidence_registry)
        if removed:
            markdown += "\n\n- Citation verification removed unregistered source references: " + ", ".join(str(item) for item in removed) + "."
        return ResearchReport(plan.task.question, markdown, plan.task, plan, scouts, [], [], [], [], [], claims, sources, conclusions, synthesis, goal, final_answer, writer_sections, plan.research_brief)
