"""Cheap Planner support over the current observation world.

Scout is not a database agent and is not an independent research layer.  It
reads only source-level catalogue facts needed by the Planner: what sources
exist, their broad coverage, and obvious resolution limits.  Exact schema
exploration, semantic binding, query construction, and trial queries belong to
the Researcher's local execution graph.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any, Protocol

from .contracts import CoverageProfile, QuestionTask, ResearchBrief, ScoutReport
from .question import task_from_brief
from .registry import DataRegistry, ExternalDataRegistry


class OpenWorldScoutProvider(Protocol):
    """Optional adapter for an open information space.

    A provider may fuzzy-search a web, literature index or remote catalogue.
    Its output is a discovery hint only: it cannot become main-chain evidence
    until a returned dataset is separately registered as an observation source.
    """

    provider_id: str

    def probe(self, task: QuestionTask) -> dict[str, Any]: ...


@dataclass(frozen=True)
class UnavailableOpenWorldProvider:
    """Explicit placeholder for an unconfigured open-space connector."""

    provider_id: str
    reason: str = "No fuzzy-search connector is configured."

    def probe(self, task: QuestionTask) -> dict[str, Any]:
        return {"status": "provider_unavailable", "reason": self.reason, "question_received": bool(task.question)}


def compact_profile(profile: CoverageProfile) -> CoverageProfile:
    return replace(profile, sample_records=[compact_record(record) for record in profile.sample_records])


def compact_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep identity and uncertainty, exclude bulk native payloads from LLM context."""

    fields = (
        "record_id", "paper_id", "source_id", "sample_id", "age_ma", "age_error_ma",
        "isotope_system", "mineral", "technique", "statistical_model", "event_interpretation",
        "rock_type", "formation", "group", "member", "region", "province", "city",
        "basin", "locality", "latitude", "longitude", "origin", "value_error_scale",
    )
    compact: dict[str, Any] = {}
    for field in fields:
        value = record.get(field)
        if isinstance(value, str):
            compact[field] = value[:240]
        elif value is None or isinstance(value, (bool, int, float)):
            compact[field] = value
    position = record.get("position")
    if isinstance(position, dict):
        compact["position"] = {
            key: position.get(key)
            for key in ("latitude", "longitude", "uncertainty_km", "resolution", "provenance")
        }
    payload = record.get("observation_payload")
    if isinstance(payload, dict):
        compact["observation_payload"] = {
            "spot_count": payload.get("spot_count"),
            "fields": list(payload.get("fields", []))[:80],
        }
    return compact


class Scout:
    def __init__(self, sources: DataRegistry, external: ExternalDataRegistry, open_world_providers: list[OpenWorldScoutProvider] | None = None) -> None:
        self.sources, self.external = sources, external
        self._open_world_providers = list(open_world_providers or [])

    def register_open_world_provider(self, provider: OpenWorldScoutProvider) -> None:
        if any(item.provider_id == provider.provider_id for item in self._open_world_providers):
            raise ValueError(f"duplicate open-world Scout provider: {provider.provider_id}")
        self._open_world_providers.append(provider)

    def run(self, task: QuestionTask | ResearchBrief, patch_scope: dict[str, object] | None = None) -> ScoutReport:
        if isinstance(task, ResearchBrief):
            task = task_from_brief(task.core_phenomenon_or_relation, task)
        # ``patch_scope`` remains accepted for durable-run compatibility, but
        # exact local repair is a Researcher concern.  Scout only refreshes the
        # registered catalogue and broad source coverage.
        probes: dict[str, dict[str, Any]] = {}
        profiles: list[CoverageProfile] = []
        for source in self.sources.all():
            if hasattr(source, "describe_scout_summary"):
                summary = source.describe_scout_summary()
            else:  # compatibility for third-party sources registered before this contract
                descriptor = source.describe_source()
                summary = {
                    "source_id": descriptor.source_id, "name": descriptor.name,
                    "sensor_kind": descriptor.sensor_kind, "version": descriptor.version,
                    "description": descriptor.description, "variables": list(descriptor.variables),
                    "capabilities": list(descriptor.capabilities), "authoritative": descriptor.authoritative,
                }
            discovery_coverage = source.describe_discovery_coverage()
            record_count = int(discovery_coverage.get("record_count") or 0)
            resolution = {
                key: discovery_coverage[key]
                for key in ("coordinate_coverage", "median_spatial_uncertainty_km", "minimum_resolvable_migration_km")
                if isinstance(discovery_coverage.get(key), (int, float))
            }
            missingness = {}
            if isinstance(discovery_coverage.get("age_error_coverage"), (int, float)):
                missingness["age_error_ma"] = round(1.0 - float(discovery_coverage["age_error_coverage"]), 4)
            profile = CoverageProfile(
                source_id=summary["source_id"],
                matched_records=record_count,
                total_records=record_count,
                dimensions={
                    "variables": list(summary.get("variables", [])),
                    "capabilities": list(summary.get("capabilities", [])),
                    "broad_coverage": discovery_coverage,
                },
                missingness=missingness,
                resolution=resolution,
                uncertainty={},
                estimated_query_cost={"catalogue_only": True},
                sample_records=[],
                warnings=[],
                data_version=summary["version"],
            )
            profiles.append(profile)
            probes[summary["source_id"]] = {
                "probe_kind": "local_catalogue",
                "describe_source": summary,
                "describe_capabilities": list(summary.get("capabilities", [])),
                "discovery_coverage": discovery_coverage,
            }

        open_world: dict[str, dict[str, Any]] = {}
        for provider in self._open_world_providers:
            try:
                open_world[provider.provider_id] = provider.probe(task)
            except Exception as error:
                open_world[provider.provider_id] = {"status": "provider_unavailable", "reason": type(error).__name__}
        probes["__scout_providers__"] = {
            "local_catalogue": {"status": "ok", "source_ids": [profile.source_id for profile in profiles]},
            "open_world": open_world,
        }

        external = self.external.describe_all()
        limits = self._sensor_limits(profiles, task)
        if "mechanism" in task.inference_level and not any(dataset.get("available") and {"plate_rotation", "paleo_trench_geometry"}.intersection(dataset.get("capabilities", [])) for dataset in external):
            limits.append("当前没有覆盖研究所需时段的版本化本地板块旋转或古海沟边界条件。")
        fingerprint_payload = {"profiles": [profile.__dict__ for profile in profiles], "external": external, "probes": probes}
        fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True, default=str).encode()).hexdigest()[:16]
        warnings = [warning for profile in profiles for warning in profile.warnings]
        if patch_scope:
            warnings.append("Scout 忽略精确 patch scope；该范围由 Researcher 的局部数据导航重新绑定。")
        if not profiles:
            warnings.append("当前没有注册 observation source。")
        requested_scope = dict(task.scope)
        scope_fingerprint = hashlib.sha256(json.dumps(requested_scope, sort_keys=True, default=str).encode()).hexdigest()[:16]
        discovery = {
            "_scope_contract": "planning_scope/v1",
            "scope_id": f"planning-scope:{scope_fingerprint}",
            "requested_scope": requested_scope,
            "status": "unbound",
            "provenance": {"scout": "catalogue_support_for_planner", "requested_patch_scope": dict(patch_scope or {})},
        }
        return ScoutReport(datetime.now(timezone.utc).isoformat(), discovery, profiles, external, limits, fingerprint, warnings, probes, [])

    @staticmethod
    def _sensor_limits(profiles: list[CoverageProfile], task: QuestionTask) -> list[str]:
        limits: list[str] = []
        requested = task.requested_scale.get("spatial_claim_scale_km")
        for profile in profiles:
            resolution = profile.resolution
            minimum = resolution.get("minimum_resolvable_migration_km")
            if requested and minimum and minimum > requested:
                limits.append(f"{profile.source_id}: 用户要求 {requested:g} km 尺度，但当前空间不确定性只支持约 {minimum:g} km 以上的迁移判断。")
            if "age_error_ma" in profile.missingness and profile.missingness["age_error_ma"] > .5:
                limits.append(f"{profile.source_id}: 超过一半年代记录没有误差，时间结构必须保守。")
            if "coordinate_coverage" in resolution and resolution["coordinate_coverage"] < .5:
                limits.append(f"{profile.source_id}: 坐标覆盖低于 50%，空间命题可能不可辨识。")
        return limits
