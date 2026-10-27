# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Scientific adaptation of AI-Q's ResearchQuery → ResearchNotes → verified-source chain.

The contracts and control flow are deliberately modelled on AI-Q's
``subagent_contracts.py`` and ``citation_verification.py``.  The only semantic
extension is that a source locator may be a durable Ledger evidence id rather
than a URL: structured database queries and numerical calculations are the
primary citable sources in this system.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contracts import EvidenceEntry


class _StrictContract(BaseModel):
    """AI-Q's strict structured-response boundary: unknown model fields fail."""

    model_config = ConfigDict(extra="forbid")


class ScientificResearchQuery(_StrictContract):
    """AI-Q ResearchQuery specialized from a web query to an evidence need."""

    query_id: str = Field(min_length=1)
    information_need: str = Field(min_length=1)
    target_components: list[str] = Field(min_length=1)
    scope: dict[str, Any] = Field(default_factory=dict)
    preferred_sensor_kinds: list[str] = Field(default_factory=list)
    preferred_capabilities: list[str] = Field(default_factory=list)
    fallback_sensor_kinds: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)


class PlannerTaskAnalysis(_StrictContract):
    user_intent: str
    explicit_requirements: list[str]
    implicit_requirements: list[str]
    out_of_scope: list[str]
    language: str
    ambiguities: list[str] = Field(default_factory=list)


class PlannerAnswerStrategy(_StrictContract):
    answer_type: Literal["scientific_report", "comparative_inference", "hypothesis_test", "observation_design", "custom"]
    title: str
    required_component_ids: list[str] = Field(min_length=1)
    synthesis_requirements: list[str] = Field(min_length=1)


class PlannerComponent(_StrictContract):
    component_id: str
    claim: str
    depends_on: list[str]
    evidence_needed: list[str] = Field(min_length=1)
    constraints: dict[str, Any]
    failure_meaning: str
    scope: dict[str, Any]
    candidate_hypothesis_ids: list[str] = Field(default_factory=list)
    skill_ids: list[str] = Field(default_factory=list)


class PlannerHypothesis(_StrictContract):
    hypothesis_id: str
    label: str
    description: str
    predictions: list[str]
    differentiating_observables: list[str]


class ScientificPlannerProposal(_StrictContract):
    """AI-Q-style strict planner response, before conversion to runtime dataclasses."""

    task_analysis: PlannerTaskAnalysis
    scientific_goal: str
    answer_strategy: PlannerAnswerStrategy
    constraints: list[str]
    required_components: list[PlannerComponent] = Field(min_length=1)
    candidate_hypotheses: list[PlannerHypothesis] = Field(default_factory=list)


class ResearchSource(_StrictContract):
    """AI-Q ResearchSource with a Ledger locator and full scientific lineage."""

    id: int = Field(ge=1)
    title: str = Field(min_length=1)
    source_type: Literal["observation_query", "tool_result", "external_dataset", "forward_model"]
    locator: str = Field(min_length=1, description="Durable EvidenceEntry id, never model-invented.")
    data_versions: dict[str, str] = Field(default_factory=dict)
    input_scope: dict[str, Any] = Field(default_factory=dict)
    uncertainty: dict[str, Any] = Field(default_factory=dict)
    input_fingerprint: str | None = None
    result_fingerprint: str | None = None


class ResearchFinding(_StrictContract):
    """AI-Q's atomic finding: claim → evidence text → exact sources."""

    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    source_ids: list[int] = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: Literal["low", "medium", "high"]
    caveats: list[str] = Field(default_factory=list)


class ResearchGap(_StrictContract):
    description: str
    impact: str


class EvidenceJudgment(_StrictContract):
    relevance_score: int = Field(ge=0, le=100)
    confidence: Literal["low", "medium", "high"]
    rationale: str


class ScientificResearchNotes(_StrictContract):
    """Direct scientific counterpart of AI-Q's schema-validated ResearchNotes."""

    query_id: str
    query_topic: str
    target_components: list[str] = Field(min_length=1)
    summary: str
    findings: list[ResearchFinding]
    gaps: list[ResearchGap] = Field(default_factory=list)
    sources: list[ResearchSource] = Field(default_factory=list)
    narrative_notes: str
    language: str
    evidence_judgment: EvidenceJudgment | None = None

    @model_validator(mode="after")
    def findings_must_reference_own_verified_sources(self) -> "ScientificResearchNotes":
        source_ids = {source.id for source in self.sources}
        source_locators = {source.id: source.locator for source in self.sources}
        if len(source_ids) != len(self.sources):
            raise ValueError("ResearchNotes.sources ids must be unique")
        for finding in self.findings:
            if not set(finding.source_ids).issubset(source_ids):
                raise ValueError("ResearchFinding.source_ids must reference sources in the same ResearchNotes")
            locators = {source_locators[source_id] for source_id in finding.source_ids}
            if not set(finding.evidence_ids).issubset(locators):
                raise ValueError("ResearchFinding.evidence_ids must be supplied by its linked ResearchSources")
        return self


class ReportClaim(_StrictContract):
    """A final-report assertion that must survive deterministic and LLM evidence checks."""

    claim_id: str
    component_id: str
    category: Literal["observed_pattern", "interpretation", "alternative", "gap"]
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    status: Literal["pending", "verified", "rejected", "gap"] = "pending"
    validation_reason: str | None = None


class ClaimEntailmentAdvisor(Protocol):
    """Narrow LLM task: assess supplied claim against supplied evidence, never author facts."""

    def verify_claim(self, *, claim: ReportClaim, evidence: list[EvidenceEntry]) -> dict[str, Any]: ...


def fingerprint(value: Any) -> str:
    """Stable provenance digest; it is intentionally not an opaque model assertion."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class EvidenceSourceRegistry:
    """AI-Q SourceRegistry adapted to whitelist real Ledger evidence entries."""

    def __init__(self, entries: Iterable[EvidenceEntry] = ()) -> None:
        self._entries: dict[str, EvidenceEntry] = {}
        for entry in entries:
            self.capture(entry)

    def capture(self, entry: EvidenceEntry) -> bool:
        """Only successful, versioned executions are eligible as writer-facing sources."""
        if entry.status != "ok" or not entry.data_versions:
            return False
        self._entries[entry.evidence_id] = entry
        return True

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self._entries

    def get(self, evidence_id: str) -> EvidenceEntry | None:
        return self._entries.get(evidence_id)

    def sources_for(self, evidence_ids: Iterable[str]) -> list[ResearchSource]:
        sources: list[ResearchSource] = []
        seen: set[str] = set()
        for evidence_id in evidence_ids:
            if evidence_id in seen:
                continue
            entry = self.get(evidence_id)
            if entry is None:
                continue
            seen.add(evidence_id)
            source_type = {
                "source_query": "observation_query", "forward_model": "forward_model",
                "external_data": "external_dataset",
            }.get(entry.kind, "tool_result")
            sources.append(ResearchSource(
                id=len(sources) + 1,
                title=f"{entry.subject_id} ({entry.evidence_id})", source_type=source_type,
                locator=entry.evidence_id, data_versions=entry.data_versions,
                input_scope=entry.input_scope, uncertainty=entry.uncertainty,
                input_fingerprint=entry.input_fingerprint, result_fingerprint=entry.result_fingerprint,
            ))
        return sources

    def compact_for_notes(self, notes: Iterable[ScientificResearchNotes]) -> "EvidenceSourceRegistry":
        """AI-Q compact mode: only sources workers carried into ResearchNotes reach Writer."""
        locators = {source.locator for note in notes for source in note.sources}
        return EvidenceSourceRegistry(entry for evidence_id, entry in self._entries.items() if evidence_id in locators)

    def validate_claim(self, claim: ReportClaim) -> tuple[bool, str, list[EvidenceEntry]]:
        if claim.category == "gap":
            return True, "explicit gap", []
        if not claim.evidence_ids:
            return False, "material claim has no evidence ids", []
        evidence = [self.get(evidence_id) for evidence_id in claim.evidence_ids]
        if any(item is None for item in evidence):
            return False, "claim references evidence outside the verified source registry", []
        valid = [item for item in evidence if item is not None]
        if any(claim.component_id not in item.component_ids for item in valid):
            return False, "claim evidence is not mapped to its required component", valid
        return True, "verified registry lineage", valid


class ClaimEvidenceValidator:
    """Fail-closed final gate, patterned after AI-Q citation verification.

    Deterministic lineage checks happen first.  The optional advisor is a
    constrained LLM entailment check and can only reject a claim; it cannot
    promote an unsupported claim.
    """

    def __init__(self, registry: EvidenceSourceRegistry, advisor: ClaimEntailmentAdvisor | None = None) -> None:
        self.registry, self.advisor = registry, advisor

    def validate(self, claims: list[ReportClaim]) -> list[ReportClaim]:
        validated: list[ReportClaim] = []
        for claim in claims:
            allowed, reason, evidence = self.registry.validate_claim(claim)
            if not allowed:
                validated.append(claim.model_copy(update={"status": "rejected", "validation_reason": reason}))
                continue
            if self.advisor is not None and claim.category != "gap":
                try:
                    verdict = self.advisor.verify_claim(claim=claim, evidence=evidence)
                    if verdict.get("verdict") != "entailed":
                        reason = str(verdict.get("rationale") or "LLM evidence comparison did not establish entailment")
                        validated.append(claim.model_copy(update={"status": "rejected", "validation_reason": reason}))
                        continue
                except Exception:
                    validated.append(claim.model_copy(update={"status": "rejected", "validation_reason": "LLM evidence comparison unavailable; fail closed"}))
                    continue
            status = "gap" if claim.category == "gap" else "verified"
            validated.append(claim.model_copy(update={"status": status, "validation_reason": reason}))
        return validated


def note_from_entries(query: ScientificResearchQuery, entries: list[EvidenceEntry], registry: EvidenceSourceRegistry, *, language: str) -> ScientificResearchNotes:
    """Persist only execution-derived notes; no free-form model synthesis enters this boundary."""
    eligible = [entry for entry in entries if registry.has(entry.evidence_id)]
    all_evidence_ids = [entry.evidence_id for entry in eligible]
    sources = registry.sources_for(all_evidence_ids)
    source_id_by_evidence = {source.locator: source.id for source in sources}
    findings: list[ResearchFinding] = []
    for entry in eligible:
        if entry.kind == "source_query":
            continue
        linked = [entry.evidence_id, *entry.upstream_evidence_ids]
        linked = [evidence_id for evidence_id in linked if evidence_id in source_id_by_evidence]
        if not linked:
            continue
        evidence_text = json.dumps({"findings": entry.findings, "uncertainty": entry.uncertainty, "versions": entry.data_versions}, ensure_ascii=False, default=str)
        confidence: Literal["low", "medium", "high"] = "high" if entry.calculation_status == "computed" else "medium"
        findings.append(ResearchFinding(
            claim=entry.result_summary, evidence=evidence_text,
            source_ids=[source_id_by_evidence[evidence_id] for evidence_id in linked], evidence_ids=linked,
            confidence=confidence, caveats=[*entry.limitations, *entry.warnings],
        ))
    gaps = [] if findings else [ResearchGap(description="No verified computational finding was produced for this evidence query.", impact="The target component remains insufficient.")]
    return ScientificResearchNotes(
        query_id=query.query_id, query_topic=query.information_need,
        target_components=query.target_components,
        summary=f"{len(findings)} execution-derived findings for {', '.join(query.target_components)}.",
        findings=findings, gaps=gaps, sources=sources,
        narrative_notes="This note was constructed from verified Ledger entries only.", language=language,
        evidence_judgment=EvidenceJudgment(relevance_score=80 if findings else 0, confidence="high" if findings else "low", rationale="Derived from successful versioned executions." if findings else "No successful versioned execution."),
    )


_INLINE_CITATION = re.compile(r"\[(\d+)\]")
_SOURCE_LINE = re.compile(r"^\[(\d+)\]\s+.*?\b(E\d+)\b.*$", re.MULTILINE)


def verify_report_citations(report: str, registry: EvidenceSourceRegistry) -> tuple[str, list[int]]:
    """Small scientific specialization of AI-Q's post-writer citation verification."""
    marker = "## Sources"
    if marker not in report:
        return report, []
    body, source_section = report.split(marker, 1)
    valid_numbers: set[int] = set()
    valid_lines: list[str] = []
    removed: list[int] = []
    for line in source_section.splitlines():
        match = _SOURCE_LINE.match(line)
        if match is None:
            valid_lines.append(line)
            continue
        number, evidence_id = int(match.group(1)), match.group(2)
        if registry.has(evidence_id):
            valid_numbers.add(number)
            valid_lines.append(line)
        else:
            removed.append(number)
    cleaned_body = _INLINE_CITATION.sub(lambda match: match.group(0) if int(match.group(1)) in valid_numbers else "", body)
    return cleaned_body.rstrip() + "\n\n## Sources\n" + "\n".join(valid_lines).strip(), removed
