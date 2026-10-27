"""Registered evidence tags for Tool Registry, Ledger and runtime diagnostics.

The vocabulary is deliberately machine-readable at the provider boundary. A
scientific Planner claim remains open language; only a registered tool result
may emit a tag into the Ledger. This is not a Planner enum or a prose-to-tag
heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal


# Values supplied directly by a successful source query.  They are assessed
# against sampled normalized observations rather than tool-result tags.
RAW_OBSERVABLES = frozenset({
    "event_age", "age_uncertainty", "source_provenance", "spatial_location",
    "spatial_uncertainty", "event_interpretation", "mineral", "multi_clock",
    "material_state",
})


# Baseline for direct Planner use and strict-schema construction.  A runtime
# replaces/extends this with its live Tool Registry before planning.  Keep the
# list here (rather than in a prompt) so a plan can persist its exact contract.
BUILTIN_DERIVED_EVIDENCE_TAGS = frozenset({
    "data_tool", "data_inspect", "data_profile", "data_sample", "data_query",
    "data_aggregate", "data_materialize", "data_relate",
    "weighted_age", "concordia", "detrital_distribution", "temporal_structure", "change_point", "peak_detection",
    "bootstrap", "median_age_stability", "temporal_clustering", "kde", "gmm",
    "dip_test", "unimodality_test", "spot_quality",
    "discordance_concordance", "outlier_detection", "inheritance_detection",
    "sample_event_reconstruction", "resolution_assessment", "spatiotemporal_pattern",
    "migration_direction", "migration_velocity", "centroid_shift",
    "spatial_regression", "spatial_clustering", "density_surface",
    "front_detection", "spatial_geometry", "sampling_density_normalization",
    "uncertainty_propagation", "alternative_specification", "sampling_bias",
    "spatial_resampling", "publication_contribution", "null_model_test",
    "plate_rotation", "reconstruct_paleolocation", "external_boundary_condition",
    "distance_to_trench", "plate_velocity", "forward_model", "model_comparison",
    "mechanism_prediction", "simulate_event_field", "likelihood_score",
    "clock_offset", "lag_distribution",
})

# These tags prove that a data operation completed. They are execution
# metadata, not scientific support for the claim under investigation.
OPERATIONAL_EVIDENCE_TAGS = frozenset({
    "data_tool", "data_inspect", "data_profile", "data_sample",
    "data_query", "data_aggregate", "data_materialize", "data_relate",
})


@dataclass(frozen=True)
class EvidenceTerm:
    """One stable identifier accepted by a registered tool result."""

    term_id: str
    kind: Literal["raw_observable", "derived_evidence"]


class EvidenceVocabulary:
    """The authority for evidence identifiers emitted by one runtime's tools.

    A tool may *produce* a term but cannot create one.  Adding a plugin term
    therefore requires an explicit vocabulary extension before tool
    registration; the Planner may still express requirements that are not
    currently observable and let the Researcher/R-Judge report that boundary.
    """

    def __init__(self, terms: Iterable[EvidenceTerm]) -> None:
        items = list(terms)
        table = {term.term_id: term for term in items if term.term_id}
        if len(table) != len(items):
            # The public constructor normally receives a short list.  Keeping
            # duplicate detection explicit avoids silent vocabulary shadowing.
            raise ValueError("EvidenceVocabulary has duplicate or empty term ids")
        self._terms = table

    def ids(self) -> list[str]:
        return sorted(self._terms)

    def raw_ids(self) -> set[str]:
        return {term_id for term_id, term in self._terms.items() if term.kind == "raw_observable"}

    def require(self, term_ids: Iterable[str], *, owner: str) -> None:
        unknown = sorted({term_id for term_id in term_ids if term_id not in self._terms})
        if unknown:
            raise ValueError(f"{owner} refers to evidence terms absent from the shared vocabulary: {unknown}")

    def extended(self, terms: Iterable[EvidenceTerm]) -> "EvidenceVocabulary":
        """Create an explicit new vocabulary for a plugin/runtime extension."""

        return EvidenceVocabulary([*self._terms.values(), *terms])


DEFAULT_EVIDENCE_VOCABULARY = EvidenceVocabulary([
    *(EvidenceTerm(term_id, "raw_observable") for term_id in sorted(RAW_OBSERVABLES)),
    *(EvidenceTerm(term_id, "derived_evidence") for term_id in sorted(BUILTIN_DERIVED_EVIDENCE_TAGS)),
])


def default_evidence_vocabulary() -> list[str]:
    """Return all baseline requirement values in stable order."""

    return DEFAULT_EVIDENCE_VOCABULARY.ids()


def normalize_evidence_vocabulary(values: Iterable[str] | None) -> list[str]:
    """Keep only non-empty controlled identifiers; preserve deterministic order."""

    return sorted({value for value in (values or []) if isinstance(value, str) and value})


def observable_manifest_from_rows(rows: Iterable[dict[str, Any]], *, field_provenance: dict[str, list[str]] | None = None) -> dict[str, dict[str, Any]]:
    """Describe direct scientific observables in canonical row semantics.

    This is deliberately a *semantic* manifest, not a list of database column
    names.  A source adapter is responsible for projecting its native fields
    into the canonical row representation before this function is called.
    Every downstream layer consumes this same manifest rather than maintaining
    a second alias table.
    """

    materialized = list(rows)
    count = len(materialized)
    present = {
        "event_age": lambda row: row.get("age_ma") is not None,
        "age_uncertainty": lambda row: row.get("age_error_ma") is not None,
        "source_provenance": lambda row: bool(row.get("source_id") or row.get("paper_id")),
        "spatial_location": lambda row: isinstance(row.get("position"), dict) and row["position"].get("latitude") is not None and row["position"].get("longitude") is not None,
        "spatial_uncertainty": lambda row: isinstance(row.get("position"), dict) and row["position"].get("uncertainty_km") is not None,
        "event_interpretation": lambda row: row.get("event_interpretation") not in (None, ""),
        "mineral": lambda row: row.get("mineral") not in (None, ""),
        # One populated isotope-system value is an event-clock label, not
        # evidence of a multi-clock comparison.  The latter needs >=2 systems.
        "multi_clock": lambda row: row.get("isotope_system") not in (None, ""),
        "material_state": lambda row: bool(row.get("material_state_payload")),
    }
    manifest: dict[str, dict[str, Any]] = {}
    for observable, predicate in present.items():
        matched = sum(1 for row in materialized if predicate(row))
        if observable == "multi_clock":
            systems = {str(row.get("isotope_system")) for row in materialized if row.get("isotope_system") not in (None, "")}
            available = len(systems) >= 2
            detail: dict[str, Any] = {"clock_systems": sorted(systems), "system_count": len(systems)}
        else:
            available = matched > 0
            detail = {}
        manifest[observable] = {
            "available": available,
            "coverage": round(matched / count, 4) if count else 0.0,
            "record_count": matched,
            "source_fields": list((field_provenance or {}).get(observable, [])),
            **detail,
        }
    return manifest
