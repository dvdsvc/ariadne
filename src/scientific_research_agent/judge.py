"""R-Judge boundary for scientific result packets.

The deterministic runtime only assembles the current component packet. It
does not apply scientific thresholds, count rules, or provider support labels.
The isolated R-Judge is the sole producer of a scientific component verdict.
"""

from __future__ import annotations

from .adjudication import EvidenceAdjudicator
from .contracts import ComponentJudgment, EvidenceGap, JudgeDecision, MissingDiagnosis, ScientificPlan, ScoutReport
from .ledger import EvidenceLedger


class SufficiencyJudge:
    def __init__(self, adjudicator: EvidenceAdjudicator | None = None) -> None:
        self.adjudicator = adjudicator or EvidenceAdjudicator()

    def evaluate(
        self,
        plan: ScientificPlan,
        scout: ScoutReport,
        ledger: EvidenceLedger,
        *,
        outer_round: int,
        max_outer_rounds: int,
        component_ids: list[str] | None = None,
    ) -> JudgeDecision:
        selected = [
            component
            for component in plan.required_components
            if (component_ids is None or component.component_id in set(component_ids))
            and component.scientific_status not in {"supported", "contradicted"}
        ]
        judgments: list[ComponentJudgment] = []
        assessments = []
        for component in selected:
            judgment, component_assessments = self._judge_component(plan, component, ledger)
            judgments.append(judgment)
            assessments.extend(component_assessments)

        judgment_by_id = {item.component_id: item.status for item in judgments}
        terminal = all(
            judgment_by_id.get(component.component_id, component.scientific_status) in {"supported", "contradicted"}
            for component in plan.required_components
        )
        if terminal:
            return JudgeDecision("write", judgments, [], [], "All required components now have R-Judge scientific verdicts.", assessments=assessments)
        return JudgeDecision(
            "continue_research",
            judgments,
            [],
            [],
            "R-Judge assessed the supplied results; unresolved claims remain available for further execution.",
            [component.component_id for component in selected],
            assessments,
        )

    def _judge_component(self, plan: ScientificPlan, component: object, ledger: EvidenceLedger) -> tuple[ComponentJudgment, list[Any]]:
        entries = self.adjudicator.entries_for_component(plan, component, ledger)
        assessments = self.adjudicator.assess_component(plan, component, ledger)
        primary = assessments[0] if assessments else None
        if primary is None:
            raise RuntimeError(f"Component {component.component_id} has no result packet for R-Judge")
        missing = list(primary.missing_evidence)
        judgment = ComponentJudgment(
            component.component_id,
            component.claim,
            primary.outcome,
            primary.confidence,
            [entry.evidence_id for entry in entries],
            primary.rationale,
            missing,
            [],
            [assessment.assessment_id for assessment in assessments],
            primary.allowed_claim_level,
        )
        return judgment, assessments

    @staticmethod
    def _gaps(plan: ScientificPlan, diagnoses: list[MissingDiagnosis]) -> list[EvidenceGap]:
        gaps: list[EvidenceGap] = []
        for index, diagnosis in enumerate(diagnoses, 1):
            component = plan.component(diagnosis.component_id)
            candidate = diagnosis.suggested_sensor_kinds[0] if diagnosis.suggested_sensor_kinds else None
            limit = "association supported; causality unresolved" if diagnosis.kind == "fundamentally_unidentifiable" else "当前问题在现有观测分辨率或传感器覆盖下不可可靠判断"
            gaps.append(EvidenceGap(
                f"G{index:03d}",
                diagnosis.component_id,
                component.candidate_hypothesis_ids,
                diagnosis.reason,
                ", ".join(diagnosis.missing_observables) or "additional observable",
                candidate,
                diagnosis.expected_information_gain,
                limit,
            ))
        return gaps
