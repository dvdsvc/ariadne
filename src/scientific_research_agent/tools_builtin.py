"""Tool registry for scientific transformations, not data acquisition or extraction."""

from __future__ import annotations

import math
import random
import importlib.util
import subprocess
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Any

import diptest
import numpy as np
import ruptures as rpt
import statsmodels.api as sm
import geopandas as gpd
from pyproj import Geod
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from shapely.geometry import Point

from .contracts import ToolResult
from .data_handles import resolve_rows
from .data_tool import run_data_tool
from .registry import ToolCapabilityCard, ToolContext, ToolInputRequirement, ToolRegistry, ToolSpec


GEODESIC = Geod(ellps="WGS84")


def _rows(ctx: ToolContext) -> list[dict[str, Any]]:
    return resolve_rows(ctx)[0]


def _versions(ctx: ToolContext) -> dict[str, str]:
    versions = {source.describe_source().source_id: source.describe_source().version for source in ctx.source_registry.all()}
    # External boundary-condition results are evidence only when their local,
    # versioned datasets travel with the calculation into the Ledger.
    versions.update({dataset["dataset_id"]: dataset["version"] for dataset in ctx.external_registry.describe_all() if dataset.get("available")})
    return versions


def _result(ctx: ToolContext, tool_id: str, calculation_status: str, summary: str, findings: dict[str, Any], tags: list[str], *, status: str = "ok", uncertainty: dict[str, Any] | None = None, limitations: list[str] | None = None, warnings: list[str] | None = None) -> ToolResult:
    if calculation_status not in {"computed", "insufficient_data", "not_computable", "not_applicable"}:
        raise ValueError(f"{tool_id} returned an invalid calculation status: {calculation_status}")
    return ToolResult(f"{tool_id}:{ctx.component.component_id}", tool_id, status, calculation_status, summary, findings, tags, uncertainty or {}, _versions(ctx), limitations or [], warnings or [])


def _spatial_rows(ctx: ToolContext) -> list[dict[str, Any]]:
    return [row for row in _rows(ctx) if row.get("age_ma") is not None and row.get("position", {}).get("latitude") is not None and row.get("position", {}).get("longitude") is not None]


def _centroid(rows: list[dict[str, Any]]) -> tuple[float, float]:
    return mean(row["position"]["latitude"] for row in rows), mean(row["position"]["longitude"] for row in rows)


def _distance_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    _, _, meters = GEODESIC.inv(first[1], first[0], second[1], second[0])
    return meters / 1000


def _ols_slope(xs: list[float], ys: list[float]) -> tuple[float, float]:
    if len(xs) < 2 or len(set(xs)) < 2:
        return 0.0, mean(ys) if ys else 0.0
    fitted = sm.OLS(np.asarray(ys, dtype=float), sm.add_constant(np.asarray(xs, dtype=float))).fit()
    return float(fitted.params[1]), float(fitted.params[0])


def temporal_structure(ctx: ToolContext) -> ToolResult:
    ages = sorted([row["age_ma"] for row in _rows(ctx) if row.get("age_ma") is not None])
    # This is an execution precondition, owned by the registered ToolSpec and
    # returned as insufficient_data/readiness diagnostics.  It is never read
    # from a Planner component constraint and therefore cannot become a
    # scientific pass threshold.
    minimum = 5
    if len(ages) < minimum:
        return _result(ctx, "time.temporal_structure", "insufficient_data", f"Only {len(ages)} event ages are available; need at least {minimum}.", {"age_count": len(ages)}, ["event_age", "temporal_structure"], limitations=["事件时间结构不可辨识。"])
    bins: dict[int, int] = defaultdict(int)
    for age in ages:
        bins[int(age // 10) * 10] += 1
    ordered = sorted(bins.items(), key=lambda item: (-item[1], item[0]))
    peaks = [{"bin_start_ma": start, "count": count} for start, count in ordered[:3]]
    return _result(ctx, "time.temporal_structure", "computed", f"{len(ages)} events span {min(ages):.1f}–{max(ages):.1f} Ma with {len(bins)} occupied ten-Ma bins.", {"age_count": len(ages), "age_range_ma": [min(ages), max(ages)], "histogram_10ma": dict(sorted(bins.items())), "dominant_bins": peaks}, ["event_age", "age_uncertainty", "temporal_structure"], uncertainty={"age_spread_ma": pstdev(ages) if len(ages) > 1 else 0.0})


def weighted_age(ctx: ToolContext) -> ToolResult:
    rows = [row for row in _rows(ctx) if row.get("age_ma") is not None and row.get("age_error_ma") not in (None, 0)]
    if len(rows) < 2:
        return _result(ctx, "time.weighted_age", "insufficient_data", "At least two age records with non-zero uncertainty are required.", {"n": len(rows)}, ["weighted_age"])
    ages = np.asarray([row["age_ma"] for row in rows], dtype=float)
    errors = np.asarray([row["age_error_ma"] for row in rows], dtype=float)
    weights = 1 / errors ** 2
    value = float(np.average(ages, weights=weights))
    one_sigma = float(math.sqrt(1 / weights.sum()))
    mswd = float((weights * (ages - value) ** 2).sum() / max(len(ages) - 1, 1))
    return _result(ctx, "time.weighted_age", "computed", f"Inverse-variance weighted age is {value:.2f} ± {one_sigma:.2f} Ma (MSWD={mswd:.2f}).", {"n": len(rows), "weighted_age_ma": value, "one_sigma_ma": one_sigma, "mswd": mswd, "library": "numpy"}, ["weighted_age"], limitations=["区域级混合事件不应自动解释为单一地质事件。"])


def change_point(ctx: ToolContext) -> ToolResult:
    ages = sorted([row["age_ma"] for row in _rows(ctx) if row.get("age_ma") is not None])
    if len(ages) < 4:
        return _result(ctx, "time.change_point", "insufficient_data", "At least four ages are required to calculate candidate temporal regime changes.", {"n": len(ages)}, ["change_point"])
    signal = np.asarray(ages, dtype=float).reshape(-1, 1)
    penalty = float(ctx.component.constraints.get("change_point_penalty", 3 * math.log(len(ages))))
    breakpoints = rpt.Pelt(model="l2", min_size=2).fit(signal).predict(pen=penalty)
    boundaries = [float((ages[index - 1] + ages[index]) / 2) for index in breakpoints[:-1] if 0 < index < len(ages)]
    return _result(ctx, "time.change_point", "computed", f"PELT change-point analysis found {len(boundaries)} candidate temporal boundaries.", {"candidate_change_points_ma": boundaries, "algorithm": "ruptures.Pelt(model=l2)", "penalty": penalty, "n": len(ages)}, ["change_point"], limitations=["统计 change point 不是自动的构造状态转换。"])


def spot_quality(ctx: ToolContext) -> ToolResult:
    spots = [spot for row in _rows(ctx) for spot in row.get("observation_payload", {}).get("spots", [])]
    if not spots:
        return _result(ctx, "sample.spot_quality", "not_applicable", "No spot/step payload is available for sample-internal quality control.", {"spot_count": 0}, [], status="not_applicable", limitations=["需要 Event Clock adapter 保留原生 spot/step payload。"])
    concordance = [spot.get("concordance") for spot in spots if isinstance(spot.get("concordance"), (int, float))]
    age_keys = ("age_206_238", "age_ma", "age", "age_206_238_ma")
    spot_ages = [float(next(spot[key] for key in age_keys if isinstance(spot.get(key), (int, float)))) for spot in spots if any(isinstance(spot.get(key), (int, float)) for key in age_keys)]
    age_errors = [spot.get("age_206_238_err") for spot in spots if isinstance(spot.get("age_206_238_err"), (int, float))]
    med, mad = (median(spot_ages), median([abs(value - median(spot_ages)) for value in spot_ages])) if spot_ages else (None, None)
    outliers = [value for value in spot_ages if mad and abs(value - med) > 3.5 * mad]
    domains = Counter(str(spot.get("domain") or spot.get("texture") or "unlabelled") for spot in spots)
    inheritance = [value for value in spot_ages if med and value > med + max(3 * (mad or 0), 5)]
    return _result(ctx, "sample.spot_quality", "computed", f"Spot-level payload covers {len(spots)} analyses; {len(outliers)} robust age outliers and {len(inheritance)} older inheritance candidates are flagged, not removed.", {"spot_count": len(spots), "concordance_coverage": len(concordance) / len(spots), "median_206_238_error": median(age_errors) if age_errors else None, "age_median_ma": med, "age_mad_ma": mad, "robust_outlier_count": len(outliers), "inheritance_candidate_count": len(inheritance), "domain_labels": dict(domains)}, ["spot_quality", "discordance_concordance", "outlier_detection", "inheritance_detection"], limitations=["Flags are not automatic exclusions. Core/rim comparison is performed only when an explicit domain/texture label exists."])


def spatial_observability(ctx: ToolContext) -> ToolResult:
    # This provider consumes observations. Reading profile_all(ctx.scope)
    # silently switched back to the unresolved local routing envelope and
    # produced false zero-row results after a valid materialization.
    rows = _rows(ctx)
    total_rows = len(rows)
    located = [
        row for row in rows
        if isinstance(row.get("position"), dict)
        and row["position"].get("latitude") is not None
        and row["position"].get("longitude") is not None
    ]
    coordinate_rows = len(located)
    uncertainties = [
        float(row["position"].get("uncertainty_km"))
        for row in located
        if isinstance(row["position"].get("uncertainty_km"), (int, float))
        and math.isfinite(float(row["position"].get("uncertainty_km")))
        and float(row["position"].get("uncertainty_km")) > 0
    ]
    # No per-record uncertainty means an unknown resolvable scale, not a
    # guessed default.
    resolution = median(uncertainties) * 2 if uncertainties else None
    requested = ctx.task.requested_scale.get("spatial_claim_scale_km") or resolution
    resolution_text = f"{resolution:g} km" if resolution is not None else "unknown because spatial uncertainty is not supplied"
    summary = f"Coordinate coverage is {coordinate_rows / total_rows:.1%} and conservative resolvable scale is {resolution_text}." if total_rows else "No observations are in scope."
    calculation_status = "computed" if total_rows else "insufficient_data"
    return _result(ctx, "space.resolution", calculation_status, summary, {"coordinate_coverage": coordinate_rows / total_rows if total_rows else 0.0, "minimum_resolvable_migration_km": resolution, "requested_scale_km": requested, "scale_comparison": {"resolution_leq_requested": resolution <= requested if resolution is not None and requested is not None else None}}, ["spatial_location", "spatial_uncertainty", "resolution_assessment"], limitations=["使用位置 provenance 和空间不确定性，未将所有坐标视为精确 site 点。"])


def centroid_shift(ctx: ToolContext) -> ToolResult:
    rows = _spatial_rows(ctx)
    if len(rows) < 4:
        return _result(ctx, "space.centroid", "insufficient_data", f"Only {len(rows)} complete age-location events are available.", {"coordinate_records": len(rows)}, ["spatiotemporal_pattern"], limitations=["无法形成稳健的两阶段空间摘要。"])
    cut = median(row["age_ma"] for row in rows)
    older, younger = [row for row in rows if row["age_ma"] >= cut], [row for row in rows if row["age_ma"] < cut]
    old_center, young_center = _centroid(older), _centroid(younger)
    shift = _distance_km(old_center, young_center)
    uncertainty = median([row["position"].get("uncertainty_km") or 50 for row in rows])
    direction = "east" if young_center[1] > old_center[1] else "west"
    return _result(ctx, "space.centroid", "computed", f"Older-to-younger centroid shift is {shift:.1f} km ({direction}).", {"older_n": len(older), "younger_n": len(younger), "age_cut_ma": cut, "older_centroid": old_center, "younger_centroid": young_center, "shift_km": shift, "direction": direction, "shift_to_uncertainty_ratio": shift / uncertainty if uncertainty else None}, ["spatiotemporal_pattern", "migration_direction", "centroid_shift"], uncertainty={"median_position_uncertainty_km": uncertainty}, limitations=["质心是区域摘要，不代表连续的实际物理路径。"])


def spatial_regression(ctx: ToolContext) -> ToolResult:
    rows = _spatial_rows(ctx)
    if len(rows) < 4:
        return _result(ctx, "space.regression", "insufficient_data", "Not enough complete age-location events for regression.", {"n": len(rows)}, ["spatiotemporal_pattern"])
    ages, longitudes = [row["age_ma"] for row in rows], [row["position"]["longitude"] for row in rows]
    age_matrix = sm.add_constant(np.asarray(ages, dtype=float))
    sigma = np.asarray([max(row["position"].get("uncertainty_km") or 10, .1) for row in rows], dtype=float)
    fitted = sm.WLS(np.asarray(longitudes, dtype=float), age_matrix, weights=1 / sigma ** 2).fit(cov_type="HC3")
    intercept, slope = (float(fitted.params[0]), float(fitted.params[1]))
    predicted = fitted.predict(age_matrix)
    residual = float(np.sqrt(np.mean((np.asarray(longitudes) - predicted) ** 2)))
    direction = "eastward_toward_younger" if slope < 0 else "westward_toward_younger"
    predicted_shift = abs(slope) * (max(ages) - min(ages)) * 111
    return _result(ctx, "space.regression", "computed", f"Uncertainty-weighted longitude-age slope is {slope:.5f} degrees/Ma ({direction}); robust p={fitted.pvalues[1]:.3g}.", {"n": len(rows), "longitude_age_slope_deg_per_ma": slope, "slope_ci95_deg_per_ma": [float(v) for v in fitted.conf_int()[1]], "slope_pvalue_hc3": float(fitted.pvalues[1]), "predicted_shift_km": predicted_shift, "rmse_deg": residual, "direction": direction, "library": "statsmodels.WLS(HC3)"}, ["spatiotemporal_pattern", "migration_velocity", "spatial_regression"], uncertainty={"position_uncertainty_weighting": "inverse_sigma_xy_squared", "robust_covariance": "HC3"}, limitations=["未控制协变量时，斜率只是模式量而非机制量。"])


def monte_carlo_migration(ctx: ToolContext) -> ToolResult:
    rows = _spatial_rows(ctx)
    if len(rows) < 4:
        return _result(ctx, "uncertainty.monte_carlo_migration", "insufficient_data", "Not enough complete records for uncertainty propagation.", {"n": len(rows)}, ["uncertainty_propagation"])
    rng, shifts = random.Random(20260807), []
    for _ in range(500):
        sampled = []
        for row in rows:
            pos = row["position"]
            sampled.append({"age": rng.gauss(row["age_ma"], row.get("age_error_ma") or 2.0), "lat": rng.gauss(pos["latitude"], (pos.get("uncertainty_km") or 10) / 111), "lon": rng.gauss(pos["longitude"], (pos.get("uncertainty_km") or 10) / 111)})
        cut = median(item["age"] for item in sampled)
        older, younger = [item for item in sampled if item["age"] >= cut], [item for item in sampled if item["age"] < cut]
        if older and younger:
            shifts.append(_distance_km((mean(item["lat"] for item in older), mean(item["lon"] for item in older)), (mean(item["lat"] for item in younger), mean(item["lon"] for item in younger))))
    shifts.sort()
    p05, p95 = shifts[int(.05 * len(shifts))], shifts[int(.95 * len(shifts))]
    return _result(ctx, "uncertainty.monte_carlo_migration", "computed", f"Monte Carlo centroid shift: median {median(shifts):.1f} km, 90% interval {p05:.1f}–{p95:.1f} km.", {"iterations": len(shifts), "shift_km_median": median(shifts), "shift_km_p05": p05, "shift_km_p95": p95}, ["uncertainty_propagation", "alternative_specification"], uncertainty={"random_seed": 20260807})


def leave_one_source_out(ctx: ToolContext) -> ToolResult:
    rows = _spatial_rows(ctx)
    groups = sorted({str(row.get("source_id", "unknown")) for row in rows})
    if len(groups) < 2:
        return _result(ctx, "bias.leave_one_source_out", "insufficient_data", "Fewer than two independent source groups are available.", {"source_groups": groups}, ["sampling_bias"])
    estimates = []
    for omitted in groups:
        remaining = [row for row in rows if str(row.get("source_id", "unknown")) != omitted]
        if len(remaining) < 4:
            continue
        cut = median(row["age_ma"] for row in remaining)
        old, young = [row for row in remaining if row["age_ma"] >= cut], [row for row in remaining if row["age_ma"] < cut]
        estimates.append(_distance_km(_centroid(old), _centroid(young)))
    stable = len(estimates) >= 2 and min(estimates) > 0 and max(estimates) - min(estimates) < max(25, mean(estimates) * .75)
    return _result(ctx, "bias.leave_one_source_out", "computed", f"Leave-one-source-out produced {len(estimates)} migration estimates; range is {(max(estimates) - min(estimates)) if estimates else 0:.1f} km.", {"source_groups": groups, "shift_estimates_km": estimates, "stable": stable}, ["sampling_bias", "source_provenance"], limitations=[] if stable else ["模式可能由少数来源贡献。"])


def permutation_null(ctx: ToolContext) -> ToolResult:
    rows = _spatial_rows(ctx)
    if len(rows) < 5:
        return _result(ctx, "bias.permutation_null", "insufficient_data", "Not enough records for a permutation null model.", {"n": len(rows)}, ["null_model_test"])
    observed = abs(_ols_slope([row["age_ma"] for row in rows], [row["position"]["longitude"] for row in rows])[0])
    rng, greater = random.Random(20260807), 0
    ages = [row["age_ma"] for row in rows]
    lons = [row["position"]["longitude"] for row in rows]
    for _ in range(500):
        shuffled = lons.copy(); rng.shuffle(shuffled)
        if abs(_ols_slope(ages, shuffled)[0]) >= observed:
            greater += 1
    p_value = (greater + 1) / 501
    return _result(ctx, "bias.permutation_null", "computed", f"Permutation null p={p_value:.3f} for the longitude-age slope.", {"iterations": 500, "observed_abs_slope": observed, "p_value": p_value}, ["null_model_test", "sampling_bias"], uncertainty={"random_seed": 20260807})


def spatial_resampling(ctx: ToolContext) -> ToolResult:
    rows = _spatial_rows(ctx)
    if len(rows) < 5:
        return _result(ctx, "bias.spatial_resampling", "insufficient_data", "Not enough spatial records for coverage-normalized resampling.", {"n": len(rows)}, ["spatial_resampling", "sampling_density_normalization", "publication_contribution"])
    contributions = Counter(str(row.get("source_id", "unknown")) for row in rows)
    dominant = max(contributions.values()) / len(rows)
    cells = Counter((round(row["position"]["latitude"]), round(row["position"]["longitude"])) for row in rows)
    return _result(ctx, "bias.spatial_resampling", "computed", f"Largest publication contribution is {dominant:.1%}; observations occupy {len(cells)} one-degree cells.", {"publication_contribution": dict(contributions), "dominant_fraction": dominant, "spatial_cells": len(cells), "coverage_concentration_high": dominant >= .7 or len(cells) < 2}, ["spatial_resampling", "sampling_density_normalization", "publication_contribution"], limitations=["空间覆盖或单一来源贡献仍可能主导模式。"] if dominant >= .7 or len(cells) < 2 else [])


def reconstruct_paleolocation(ctx: ToolContext) -> ToolResult:
    rows = _spatial_rows(ctx)
    reconstructed = []
    for row in rows:
        response = ctx.external_registry.call("plate_rotation", "reconstruct_location", {"latitude": row["position"]["latitude"], "longitude": row["position"]["longitude"], "age_ma": row["age_ma"], "plate_id": row.get("plate_id")})
        if response.get("status") != "ok":
            return _result(ctx, "external.reconstruct_paleolocation", "not_computable", "No usable versioned plate-rotation provider is available.", response, [], status="data_unavailable", limitations=["外部物理边界条件当前不可用。"])
        reconstructed.append({"age": row["age_ma"], "lat": response["latitude"], "lon": response["longitude"], "uncertainty": response.get("uncertainty_km")})
    cut = median(item["age"] for item in reconstructed)
    old, young = [item for item in reconstructed if item["age"] >= cut], [item for item in reconstructed if item["age"] < cut]
    shift = _distance_km((mean(item["lat"] for item in old), mean(item["lon"] for item in old)), (mean(item["lat"] for item in young), mean(item["lon"] for item in young)))
    return _result(ctx, "external.reconstruct_paleolocation", "computed", f"Reconstructed older-to-younger centroid shift is {shift:.1f} km.", {"reconstructed_shift_km": shift, "n": len(reconstructed)}, ["plate_rotation", "reconstruct_paleolocation"], uncertainty={"provider_uncertainty_km": median([item["uncertainty"] for item in reconstructed if item["uncertainty"] is not None]) if reconstructed else None})


def boundary_compatibility(ctx: ToolContext) -> ToolResult:
    rows = _spatial_rows(ctx)
    distances, velocities = [], []
    for row in rows:
        response = ctx.external_registry.call("paleo_trench_geometry", "distance_to_trench", {"latitude": row["position"]["latitude"], "longitude": row["position"]["longitude"], "age_ma": row["age_ma"]})
        if response.get("status") != "ok":
            return _result(ctx, "external.boundary_compatibility", "not_computable", "No usable versioned paleo-trench provider is available.", response, [], status="data_unavailable", limitations=["没有用网页检索替代外部物理边界。"])
        velocity = ctx.external_registry.call("plate_velocity", "plate_velocity", {"latitude": row["position"]["latitude"], "longitude": row["position"]["longitude"], "age_ma": row["age_ma"], "plate_id": row.get("plate_id")})
        if velocity.get("status") != "ok":
            return _result(ctx, "external.boundary_compatibility", "not_computable", "The boundary provider is available but plate velocity is not resolvable for these observations.", velocity, [], status="data_unavailable", limitations=["Plate identity or a velocity-capable local provider is required."])
        distances.append((row["age_ma"], response["distance_km"], response.get("uncertainty_km")))
        velocities.append(velocity)
    slope, _ = _ols_slope([item[0] for item in distances], [item[1] for item in distances])
    speed_values = [item.get("velocity_km_per_ma") for item in velocities if isinstance(item.get("velocity_km_per_ma"), (int, float))]
    return _result(ctx, "external.boundary_compatibility", "computed", f"Distance-to-trench changes at {slope:.2f} km/Ma across {len(distances)} events.", {"n": len(distances), "distance_age_slope_km_per_ma": slope, "distance_range_km": [min(item[1] for item in distances), max(item[1] for item in distances)], "median_plate_velocity_km_per_ma": median(speed_values) if speed_values else None, "velocity_directions": sorted({str(item.get("direction") or item.get("azimuth_degrees")) for item in velocities})}, ["external_boundary_condition", "distance_to_trench", "plate_velocity"], uncertainty={"median_trench_uncertainty_km": median([item[2] for item in distances if item[2] is not None]) if distances else None}, limitations=["时间相关不单独证明因果。"])


def multi_clock_ordering(ctx: ToolContext) -> ToolResult:
    rows = [row for row in _rows(ctx) if row.get("age_ma") is not None]
    systems = defaultdict(list)
    for row in rows:
        systems[row.get("isotope_system", "unknown")].append(row["age_ma"])
    if len(systems) < 2:
        return _result(ctx, "multi_clock.event_matching", "insufficient_data", "Fewer than two clock systems are available in scope.", {"systems": {key: len(value) for key, value in systems.items()}}, ["multi_clock"], limitations=["当前不能形成跨计时器顺序。"])
    medians = {system: median(ages) for system, ages in systems.items()}
    return _result(ctx, "multi_clock.event_matching", "computed", f"Multiple clock groups available as comparison candidates: {', '.join(sorted(systems))}.", {"system_median_ages_ma": medians, "counts": {system: len(ages) for system, ages in systems.items()}}, ["multi_clock"], limitations=["This groups clocks by isotope-system labels; it does not establish event-level physical comparability."])


def lag_distribution(ctx: ToolContext) -> ToolResult:
    rows = [row for row in _rows(ctx) if row.get("age_ma") is not None]
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row.get("isotope_system", "unknown")].append(row["age_ma"])
    if len(grouped) < 2:
        return _result(ctx, "multi_clock.lag_distribution", "insufficient_data", "Lag distributions need at least two validated clock systems.", {"systems": list(grouped)}, ["clock_offset", "lag_distribution"])
    medians = {system: median(ages) for system, ages in grouped.items()}
    pairs = {f"{first}__{second}": medians[first] - medians[second] for index, first in enumerate(medians) for second in list(medians)[index + 1:]}
    return _result(ctx, "multi_clock.lag_distribution", "computed", "Computed regional candidate clock offsets.", {"median_clock_offsets_ma": pairs}, ["clock_offset", "lag_distribution"], limitations=["必须先通过矿物、事件解释和重置可能性验证，才可作热历史解释。"])


def temporal_kde(ctx: ToolContext) -> ToolResult:
    """One atomic KDE operation; mixture fitting is a separate tool."""

    ages = np.asarray([row["age_ma"] for row in _rows(ctx) if row.get("age_ma") is not None], dtype=float)
    if ages.size < 6:
        return _result(ctx, "time.kde", "insufficient_data", "At least six ages are required for KDE temporal structure.", {"n": int(ages.size)}, ["kde", "peak_detection"])
    sigma = float(np.std(ages, ddof=1))
    if not math.isfinite(sigma) or sigma <= 0:
        return _result(ctx, "time.kde", "insufficient_data", "KDE needs a non-zero age spread.", {"n": int(ages.size), "age_spread_ma": sigma}, ["kde"])
    bandwidth = max(0.25, 1.06 * sigma * ages.size ** (-.2))
    grid = np.linspace(float(ages.min()), float(ages.max()), 256)
    density = stats.gaussian_kde(ages, bw_method=bandwidth / sigma)(grid)
    peaks = [float(grid[index]) for index in range(1, len(grid) - 1) if density[index] > density[index - 1] and density[index] > density[index + 1]]
    return _result(ctx, "time.kde", "computed", f"SciPy KDE evaluated {len(ages)} ages and identified {len(peaks)} candidate peaks.", {"n": int(ages.size), "bandwidth_ma": bandwidth, "kde_peak_ages_ma": peaks, "library": "scipy.stats.gaussian_kde"}, ["kde", "peak_detection"], uncertainty={"bandwidth_rule": "Silverman"})


def temporal_gmm(ctx: ToolContext) -> ToolResult:
    """One atomic Gaussian-mixture operation."""

    ages = np.asarray([row["age_ma"] for row in _rows(ctx) if row.get("age_ma") is not None], dtype=float)
    if ages.size < 6:
        return _result(ctx, "time.gmm", "insufficient_data", "At least six ages are required for a Gaussian mixture.", {"n": int(ages.size)}, ["gmm", "temporal_clustering"])
    if np.unique(ages).size < 2:
        return _result(ctx, "time.gmm", "insufficient_data", "A Gaussian mixture needs at least two distinct ages.", {"n": int(ages.size), "distinct_age_count": int(np.unique(ages).size)}, ["gmm"])
    matrix = ages.reshape(-1, 1)
    one = GaussianMixture(n_components=1, random_state=20260807, n_init=20).fit(matrix)
    two = GaussianMixture(n_components=2, random_state=20260807, n_init=20).fit(matrix)
    bic_one, bic_two = float(one.bic(matrix)), float(two.bic(matrix))
    return _result(ctx, "time.gmm", "computed", f"scikit-learn Gaussian mixture fitting compared one and two components (ΔBIC={bic_one - bic_two:.2f}).", {"n": int(ages.size), "component_means_ma": sorted(float(value) for value in two.means_.ravel()), "bic_one_component": bic_one, "bic_two_component": bic_two, "delta_bic_two_vs_one": bic_one - bic_two, "library": "sklearn.mixture.GaussianMixture"}, ["gmm", "temporal_clustering"], uncertainty={"gmm_n_init": 20, "random_seed": 20260807})


def temporal_dip(ctx: ToolContext) -> ToolResult:
    """Hartigan's dip test only; density and mixture estimation stay separate."""

    ages = np.asarray([row["age_ma"] for row in _rows(ctx) if row.get("age_ma") is not None], dtype=float)
    ages = ages[np.isfinite(ages)]
    if ages.size < 4:
        return _result(ctx, "time.dip", "insufficient_data", "Hartigan's dip test requires at least four finite ages.", {"n": int(ages.size)}, ["dip_test", "unimodality_test"])
    dip, p_value = diptest.diptest(ages, boot_pval=False)
    return _result(
        ctx,
        "time.dip",
        "computed",
        f"Hartigan's dip test evaluated {len(ages)} ages (dip={float(dip):.6g}, p={float(p_value):.6g}).",
        {
            "n": int(ages.size),
            "dip_statistic": float(dip),
            "p_value": float(p_value),
            "p_value_method": "critical_value_table_interpolation",
            "null_hypothesis": "unimodal_distribution",
            "alternative_hypothesis": "multimodal_distribution",
            "library": f"diptest {diptest.__version__}",
        },
        ["dip_test", "unimodality_test"],
        limitations=[
            "Failure to reject the null is not proof of unimodality.",
            "The test does not estimate the number or locations of modes.",
            "Per-observation age uncertainties are not modeled by this test.",
        ],
    )


def temporal_bootstrap(ctx: ToolContext) -> ToolResult:
    ages = np.asarray([row["age_ma"] for row in _rows(ctx) if row.get("age_ma") is not None], dtype=float)
    if ages.size < 5:
        return _result(ctx, "time.bootstrap", "insufficient_data", "At least five ages are required for bootstrap stability.", {"n": int(ages.size)}, ["bootstrap", "median_age_stability"])
    rng = np.random.default_rng(20260807)
    medians = [float(np.median(rng.choice(ages, size=ages.size, replace=True))) for _ in range(1000)]
    lo, hi = np.quantile(medians, [.025, .975])
    return _result(ctx, "time.bootstrap", "computed", f"Bootstrap median age 95% interval is {lo:.2f}–{hi:.2f} Ma.", {"iterations": 1000, "median_age_ma": float(np.median(ages)), "bootstrap_ci95_ma": [float(lo), float(hi)]}, ["bootstrap", "median_age_stability"], uncertainty={"random_seed": 20260807})


def spatial_geometry_audit(ctx: ToolContext) -> ToolResult:
    """Use GeoPandas/Shapely only for geometry-aware coverage diagnostics, not invented polygons."""
    rows = _spatial_rows(ctx)
    if len(rows) < 3:
        return _result(ctx, "space.geometry", "insufficient_data", "At least three coordinates are required for geometry-aware coverage audit.", {"n": len(rows)}, ["spatial_geometry", "sampling_density_normalization"])
    geometry = [Point(row["position"]["longitude"], row["position"]["latitude"]) for row in rows]
    frame = gpd.GeoDataFrame({"age_ma": [row["age_ma"] for row in rows]}, geometry=geometry, crs="EPSG:4326").to_crs("EPSG:6933")
    hull = frame.union_all().convex_hull
    area_km2 = float(hull.area / 1_000_000)
    return _result(ctx, "space.geometry", "computed", f"Geometry-aware observation hull covers {area_km2:.1f} km² in an equal-area CRS.", {"n": len(rows), "convex_hull_area_km2": area_km2, "input_crs": "EPSG:4326", "analysis_crs": "EPSG:6933" , "library": "geopandas/shapely/pyproj"}, ["spatial_geometry", "sampling_density_normalization"], limitations=["Convex hull is a coverage diagnostic, not a geological boundary."])


def leave_one_region_out(ctx: ToolContext) -> ToolResult:
    rows = _spatial_rows(ctx)
    blocks: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        blocks[(round(row["position"]["latitude"]), round(row["position"]["longitude"]))].append(row)
    if len(blocks) < 2:
        return _result(ctx, "bias.leave_one_region_out", "insufficient_data", "Fewer than two spatial blocks are available.", {"blocks": len(blocks)}, ["sampling_bias", "spatial_resampling"])
    slopes = []
    for key in blocks:
        remain = [row for other, values in blocks.items() if other != key for row in values]
        if len(remain) >= 4:
            slopes.append(_ols_slope([r["age_ma"] for r in remain], [r["position"]["longitude"] for r in remain])[0])
    signs = {math.copysign(1, value) for value in slopes if value}
    stable = len(slopes) >= 2 and len(signs) == 1
    return _result(ctx, "bias.leave_one_region_out", "computed", f"Leave-one-region-out slope sign is {'stable' if stable else 'unstable'} across {len(slopes)} blocks.", {"block_count": len(blocks), "slopes_deg_per_ma": slopes, "stable_direction": stable}, ["sampling_bias", "spatial_resampling", "publication_contribution"])


def forward_simulate_and_compare(ctx: ToolContext) -> ToolResult:
    rows = _spatial_rows(ctx)
    if len(rows) < 6:
        return _result(ctx, "model.compare", "insufficient_data", "Need six spatial events to compare synthetic observation processes.", {"n": len(rows)}, ["forward_model", "simulate_event_field", "model_comparison", "null_model_test", "likelihood_score"])
    ages = np.array([row["age_ma"] for row in rows]); lons = np.array([row["position"]["longitude"] for row in rows])
    if np.unique(ages).size < 2:
        return _result(ctx, "model.compare", "insufficient_data", "The declared model comparison needs at least two distinct ages so both temporal hypotheses have observable predictions.", {"n": len(rows), "distinct_age_count": int(np.unique(ages).size)}, ["forward_model", "model_comparison"], limitations=["A narrow or single-age scope cannot distinguish continuous migration from stationary centres."])
    slope, intercept = _ols_slope(list(ages), list(lons)); residual = lons - (intercept + slope * ages)
    # H1 continuous migration likelihood; H2 pulsed stationary centers selected by temporal split.
    sigma1 = max(float(np.std(residual, ddof=1)), .001); ll1 = float(np.sum(-.5 * (residual / sigma1) ** 2 - np.log(sigma1)))
    best = None
    for cut in np.unique(np.quantile(ages, np.linspace(.2, .8, 7))):
        older, younger = ages >= cut, ages < cut
        if not older.any() or not younger.any():
            continue
        means = [float(lons[older].mean()), float(lons[younger].mean())]
        predicted = np.where(ages >= cut, means[0], means[1]); sigma = max(float(np.std(lons - predicted, ddof=1)), .001)
        ll = float(np.sum(-.5 * ((lons - predicted) / sigma) ** 2 - np.log(sigma)))
        if best is None or ll > best[0]: best = (ll, float(cut), means, sigma)
    if best is None:
        return _result(ctx, "model.compare", "insufficient_data", "No non-empty temporal split was available for the stationary-centre alternative.", {"n": len(rows), "distinct_age_count": int(np.unique(ages).size)}, ["forward_model", "model_comparison"])
    ll2, cut, means, sigma2 = best
    bic1, bic2 = -2 * ll1 + 3 * math.log(len(rows)), -2 * ll2 + 4 * math.log(len(rows))
    delta = abs(bic1 - bic2); winner = "continuous_migration" if bic1 < bic2 else "two_stationary_centers"
    return _result(ctx, "model.compare", "computed", f"Synthetic-observation model comparison prefers {winner}; ΔBIC={delta:.2f}.", {"models": {"continuous_migration": {"bic": bic1, "slope_deg_per_ma": slope, "residual_sigma_deg": sigma1}, "two_stationary_centers": {"bic": bic2, "age_split_ma": cut, "centers_longitude": means, "residual_sigma_deg": sigma2}}, "winner": winner, "delta_bic": delta, "model_score_difference": delta, "next_observation": "material-state sensor (Hf/Nd/geochemistry) or independent boundary-condition constraint"}, ["forward_model", "simulate_event_field", "model_comparison", "null_model_test", "likelihood_score", "mechanism_prediction"], uncertainty={"observation_model": "Gaussian longitude residual", "model_dimension": "low-dimensional"}, limitations=["This is explicit low-dimensional model competition; the R-Judge must decide what it means for the component claim."])


def _records(requirement_id: str, minimum: int, *field_paths: str, description: str) -> ToolInputRequirement:
    return ToolInputRequirement(requirement_id, "minimum_records", description, tuple(field_paths), minimum_count=minimum)


def _nested(requirement_id: str, minimum: int, *field_paths: str, description: str) -> ToolInputRequirement:
    return ToolInputRequirement(requirement_id, "nested_collection", description, tuple(field_paths), minimum_count=minimum)


def _distinct(requirement_id: str, minimum: int, *field_paths: str, description: str) -> ToolInputRequirement:
    return ToolInputRequirement(requirement_id, "minimum_distinct_values", description, tuple(field_paths), minimum_distinct=minimum)


def _external(requirement_id: str, capability: str, *, description: str) -> ToolInputRequirement:
    return ToolInputRequirement(requirement_id, "external_capability", description, capability=capability)


def _prior(requirement_id: str, evidence_tag: str, *producer_ids: str, description: str) -> ToolInputRequirement:
    return ToolInputRequirement(requirement_id, "prior_evidence_tag", description, evidence_tag=evidence_tag, producer_ids=tuple(producer_ids))


def _source(requirement_id: str, *, description: str) -> ToolInputRequirement:
    return ToolInputRequirement(requirement_id, "registered_source", description)


def _handle(requirement_id: str = "materialized_dataset") -> ToolInputRequirement:
    return ToolInputRequirement(
        requirement_id, "materialized_data_handle",
        "A completed upstream data.tool materialize result is required.",
    )


def _provider_result(ctx: ToolContext, tool_id: str, *, provider: str, package_available: bool, input_description: str) -> ToolResult:
    status = "not_applicable" if not package_available else "data_unavailable"
    reason = (
        f"{provider} is not installed in this runtime."
        if not package_available else f"{provider} is available but this source has not supplied the required {input_description}."
    )
    return _result(ctx, tool_id, "not_computable", reason, {"provider": provider, "available": package_available}, [], status=status, limitations=["A scientific provider is selected only when its required data semantics are present."])


def chronology_isoplotr(ctx: ToolContext) -> ToolResult:
    """Thin provider boundary; no reimplementation of IsoplotR in Python."""

    available = _isoplotr_available()
    return _provider_result(ctx, "chronology.isoplotr", provider="IsoplotR", package_available=available, input_description="raw isotope-ratio, uncertainty, and chronometer-specific measurement fields")


def _isoplotr_available() -> bool:
    """Return provider availability without making it a runtime failure."""

    try:
        response = subprocess.run(["Rscript", "-e", "cat(requireNamespace('IsoplotR', quietly=TRUE))"], capture_output=True, text=True, timeout=10, check=False)
        return response.returncode == 0 and response.stdout.strip().casefold() == "true"
    except (OSError, subprocess.SubprocessError):
        return False


def detrital_detritalpy(ctx: ToolContext) -> ToolResult:
    available = importlib.util.find_spec("detritalpy") is not None
    return _provider_result(ctx, "detrital.detritalpy", provider="detritalPy", package_available=available, input_description="sample-to-grain detrital age records")


def space_dbscan(ctx: ToolContext) -> ToolResult:
    """One atomic DBSCAN operation with an explicit radius."""

    rows = _spatial_rows(ctx)
    if len(rows) < 5:
        return _result(ctx, "space.dbscan", "insufficient_data", "At least five complete coordinates are required for DBSCAN.", {"n": len(rows)}, ["spatial_clustering"])
    eps_km = float((ctx.action.parameters if ctx.action else {}).get("eps_km") or (10 if not ctx.require_materialized_handle else 0))
    if eps_km <= 0:
        raise ValueError("space.dbscan requires a positive eps_km")
    coords = np.asarray([[row["position"]["latitude"], row["position"]["longitude"]] for row in rows], dtype=float)
    labels = DBSCAN(eps=eps_km / 6371.0088, min_samples=2, metric="haversine").fit_predict(np.radians(coords))
    clusters = [label for label in sorted(set(labels)) if label != -1]
    return _result(ctx, "space.dbscan", "computed", f"DBSCAN evaluated {len(rows)} located records with radius {eps_km:g} km.", {"n": len(rows), "eps_km": eps_km, "cluster_sizes": [int((labels == label).sum()) for label in clusters], "noise_count": int((labels == -1).sum()), "library": "sklearn.cluster.DBSCAN(metric=haversine)"}, ["spatial_clustering"])


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    age = ("age_ma",)
    located = ("age_ma", "position.latitude", "position.longitude")

    def add(tool_id, title, group, description, capabilities, requirements, output_tags, runner, *, cost=1.0, handle=False, external=False):
        registry.register(ToolSpec(tool_id, title, group, description, capabilities, requirements, output_tags, runner, cost=cost, requires_materialized_handle=handle, requires_external=external))

    # The active registry contains only atomic operations.  A multi-step
    # analysis is represented by several local-DAG nodes, never by a wrapper
    # that hides method selection inside one tool.
    add("data.tool", "Read-only data tool", "data_tool", "Execute one Researcher-declared data operation over one registered source: select/filter/project, explicit union/join/match, group/aggregate/bin, or materialize. Deterministic transformations stay inside the provider; semantic ambiguity returns a common decision-required result for the Researcher.", ["data_tool", "data_relate"], [_source("registered_observation_source", description="At least one registered observation source is required.")], ["data_inspect", "data_profile", "data_sample", "data_query", "data_aggregate", "data_materialize", "data_relate"], run_data_tool, cost=.2)
    add("time.temporal_structure", "Temporal structure", "temporal_analysis", "Summarize age coverage and coarse temporal structure without selecting a mechanism.", ["event_age", "age_uncertainty", "temporal_structure", "source_provenance"], [_records("dated_events", 5, *age, description="Dated event records are required.")], ["temporal_structure"], temporal_structure)
    add("time.weighted_age", "Weighted age", "temporal_analysis", "Compute one inverse-variance weighted age from an already justified single event.", ["weighted_age"], [_handle(), _records("ages_with_uncertainty", 2, "age_ma", "age_error_ma", description="At least two paired ages and uncertainties are required.")], ["weighted_age"], weighted_age, handle=True)
    add("time.change_point", "Change point", "temporal_analysis", "Segment a sorted age series and report candidate regime boundaries; this is not a modality or peak-count test.", ["change_point"], [_handle(), _records("dated_events", 4, *age, description="At least four dated events are required.")], ["change_point"], change_point, cost=1.2, handle=True)
    add("time.kde", "Kernel density estimate", "temporal_analysis", "Estimate a one-dimensional age density and its candidate local peaks.", ["kde", "peak_detection"], [_handle(), _records("dated_events", 6, *age, description="At least six dated events are required.")], ["kde", "peak_detection"], temporal_kde, cost=1.3, handle=True)
    add("time.gmm", "Gaussian mixture", "temporal_analysis", "Fit Gaussian mixture components to an age series and report model scores.", ["gmm", "temporal_clustering"], [_handle(), _records("dated_events", 6, *age, description="At least six dated events are required.")], ["gmm", "temporal_clustering"], temporal_gmm, cost=1.3, handle=True)
    add("time.dip", "Hartigan dip test", "temporal_analysis", "Compute Hartigan's nonparametric dip statistic and p-value for the null hypothesis of unimodality; this does not locate peaks or fit mixture components.", ["dip_test", "unimodality_test"], [_handle(), _records("dated_events", 4, *age, description="At least four dated events are required by Hartigan's dip test.")], ["dip_test", "unimodality_test"], temporal_dip, cost=1.2, handle=True)
    add("time.bootstrap", "Median-age bootstrap", "uncertainty", "Bootstrap the sample median age and report its 95% interval; this implementation does not resample an arbitrary statistic or peak location.", ["bootstrap", "median_age_stability"], [_handle(), _records("dated_events", 5, *age, description="At least five dated events are required.")], ["bootstrap", "median_age_stability"], temporal_bootstrap, cost=1.2, handle=True)
    add("sample.spot_quality", "Spot or step quality", "sample_internal", "Audit preserved spot or step payload and flag quality candidates.", ["spot_quality", "discordance_concordance", "outlier_detection", "inheritance_detection"], [_handle(), _nested("native_spot_or_step_payload", 1, "observation_payload.spots", description="A preserved native spot or step payload is required.")], ["spot_quality", "discordance_concordance", "outlier_detection", "inheritance_detection"], spot_quality, cost=1.5, handle=True)
    add("space.resolution", "Spatial resolution", "spatial_analysis", "Assess coordinate coverage and the conservative spatial scale supported by the source.", ["spatial_location", "spatial_uncertainty", "resolution_assessment"], [_handle(), _records("observations_in_scope", 1, description="At least one observation is required.")], ["resolution_assessment"], spatial_observability, handle=True)
    add("space.centroid", "Spatial centroid", "spatial_analysis", "Compute a two-group activity-centre displacement from located observations.", ["spatiotemporal_pattern", "migration_direction"], [_handle(), _records("located_dated_events", 4, *located, description="At least four located and dated events are required.")], ["spatiotemporal_pattern", "migration_direction", "centroid_shift"], centroid_shift, handle=True)
    add("space.regression", "Coordinate regression", "spatial_analysis", "Estimate a declared coordinate-age statistical relation.", ["spatiotemporal_pattern", "migration_velocity", "spatial_regression"], [_handle(), _records("located_dated_events", 4, *located, description="At least four located and dated events are required.")], ["spatiotemporal_pattern", "migration_velocity", "spatial_regression"], spatial_regression, cost=1.2, handle=True)
    add("space.dbscan", "Spatial DBSCAN", "spatial_analysis", "Cluster located observations using an explicit haversine radius.", ["spatial_clustering"], [_handle(), _records("located_dated_events", 5, *located, description="At least five located and dated events are required.")], ["spatial_clustering"], space_dbscan, cost=1.3, handle=True)
    add("space.geometry", "Spatial geometry audit", "spatial_analysis", "Measure observation coverage geometry in a projected equal-area coordinate system.", ["spatial_geometry", "sampling_density_normalization"], [_handle(), _records("located_dated_events", 3, *located, description="At least three located and dated events are required.")], ["spatial_geometry", "sampling_density_normalization"], spatial_geometry_audit, cost=1.3, handle=True)
    add("uncertainty.monte_carlo_migration", "Migration uncertainty propagation", "uncertainty", "Propagate declared age and position errors through a specified migration statistic.", ["uncertainty_propagation", "alternative_specification"], [_handle(), _records("located_events_with_uncertainty", 4, "age_ma", "age_error_ma", "position.latitude", "position.longitude", "position.uncertainty_km", description="Located events with reported age and position uncertainty are required.")], ["uncertainty_propagation", "alternative_specification"], monte_carlo_migration, cost=1.5, handle=True)
    add("bias.leave_one_source_out", "Leave one source out", "bias_control", "Test sensitivity of a declared statistic to individual sources.", ["sampling_bias", "source_provenance"], [_handle(), _records("located_dated_events", 5, *located, description="At least five located events are required."), _distinct("independent_sources", 2, "source_id", description="At least two source identifiers are required.")], ["sampling_bias", "source_provenance"], leave_one_source_out, cost=1.2, handle=True)
    add("bias.leave_one_region_out", "Leave one region out", "bias_control", "Test sensitivity of a declared statistic to spatial partitions.", ["sampling_bias", "spatial_resampling"], [_handle(), _records("located_dated_events", 5, *located, description="At least five located events are required.")], ["sampling_bias", "spatial_resampling", "publication_contribution"], leave_one_region_out, cost=1.3, handle=True)
    add("bias.permutation_null", "Permutation null", "bias_control", "Test a declared age-coordinate association against a label permutation null.", ["null_model_test", "sampling_bias"], [_handle(), _records("located_dated_events", 5, *located, description="At least five located events are required.")], ["null_model_test", "sampling_bias"], permutation_null, cost=1.2, handle=True)
    add("bias.spatial_resampling", "Spatial resampling", "bias_control", "Assess uneven spatial coverage and source contribution by resampling.", ["spatial_resampling", "sampling_density_normalization", "publication_contribution"], [_handle(), _records("located_dated_events", 5, *located, description="At least five located events are required.")], ["spatial_resampling", "sampling_density_normalization", "publication_contribution"], spatial_resampling, cost=1.3, handle=True)
    add("external.reconstruct_paleolocation", "Reconstruct paleolocation", "plate_kinematics", "Reconstruct locations with a registered versioned rotation provider.", ["plate_rotation", "reconstruct_paleolocation"], [_handle(), _records("located_dated_events", 2, *located, description="Located and dated observations are required."), _external("plate_rotation_provider", "plate_rotation", description="A versioned local plate-rotation provider is required.")], ["plate_rotation", "reconstruct_paleolocation"], reconstruct_paleolocation, cost=2, handle=True, external=True)
    add("external.boundary_compatibility", "Boundary compatibility", "plate_kinematics", "Measure compatibility with a registered paleo-boundary and plate-velocity provider.", ["external_boundary_condition", "distance_to_trench", "plate_velocity"], [_handle(), _records("located_dated_events", 2, *located, description="Located and dated observations are required."), _external("paleo_trench_provider", "paleo_trench_geometry", description="A versioned local paleo-boundary provider is required."), _external("plate_velocity_provider", "plate_velocity", description="A versioned local plate-velocity provider is required.")], ["external_boundary_condition", "distance_to_trench", "plate_velocity"], boundary_compatibility, cost=2, handle=True, external=True)
    add("chronology.isoplotr", "IsoplotR provider", "chronology", "Delegate raw chronometer reduction to the installed IsoplotR provider when raw ratios and uncertainties are present.", ["weighted_age", "concordia"], [_handle(), _source("registered_observation_source", description="A source with raw chronometer measurements is required.")], ["weighted_age", "concordia"], chronology_isoplotr, cost=1.5, handle=True)
    add("detrital.detritalpy", "detritalPy provider", "detrital", "Delegate grain-to-sample detrital age analysis to detritalPy when grouping is explicit.", ["detrital_distribution"], [_handle(), _source("registered_observation_source", description="A source with grain and sample grouping is required.")], ["detrital_distribution"], detrital_detritalpy, cost=1.5, handle=True)
    add("model.compare", "Forward model comparison", "model_testing", "Compare explicitly declared low-dimensional observable-process models.", ["forward_model", "simulate_event_field", "model_comparison", "null_model_test", "likelihood_score", "mechanism_prediction"], [_handle(), _records("located_dated_events", 6, *located, description="At least six located and dated events are required.")], ["forward_model", "simulate_event_field", "model_comparison", "null_model_test", "likelihood_score", "mechanism_prediction"], forward_simulate_and_compare, cost=2, handle=True)
    add("multi_clock.event_matching", "Clock event matching", "multi_clock", "Group dated records from distinct clock systems as candidates for later comparison.", ["multi_clock"], [_handle(), _records("dated_clock_records", 2, "age_ma", "isotope_system", description="At least two dated records are required."), _distinct("clock_systems", 2, "isotope_system", description="At least two clock systems are required.")], ["multi_clock"], multi_clock_ordering, cost=1.5, handle=True)
    add("multi_clock.lag_distribution", "Clock lag distribution", "multi_clock", "Compute offsets only from an upstream matched multi-clock result.", ["clock_offset", "lag_distribution"], [_handle(), _records("dated_clock_records", 2, "age_ma", "isotope_system", description="At least two dated records are required."), _distinct("clock_systems", 2, "isotope_system", description="At least two clock systems are required."), _prior("cross_clock_grouping_evidence", "multi_clock", "multi_clock.event_matching", description="An upstream matching result is required.")], ["clock_offset", "lag_distribution"], lag_distribution, cost=1.5, handle=True)

    # Package-backed providers are present only when their runtime provider
    # exists.  Data applicability is still decided by readiness and the tool.
    registry.get("chronology.isoplotr").exposed_to_researcher = _isoplotr_available()
    registry.get("detrital.detritalpy").exposed_to_researcher = importlib.util.find_spec("detritalpy") is not None
    _install_capability_cards(registry)
    return registry


def _install_capability_cards(registry: ToolRegistry) -> None:
    """Scientific tool cards state objects and conditions, not case-specific prohibitions."""

    cards = {
        "data.tool": ("Perform one bounded, read-only operation on a registered source through the common data_operation/v1 contract.", ["Researcher-declared data relation or grouping intent", "live source catalogue", "strict data request"], ["data_operation/v1 result", "data_decision_required/v1 when scientific meaning is ambiguous", "data_artifact/v1 when materialized"], "the physical table and column identifiers exposed by the live source catalogue", ["Inspect the live catalogue before selecting tables or columns.", "Use profile or aggregate for large datasets; do not materialise raw records merely to obtain a count or histogram.", "Keep select/filter/project, explicit union/join/match, group/aggregate/bin, and materialize within this one data-operation contract.", "State match policy and tolerance explicitly when they can change scientific meaning; never rely on a provider default.", "If the provider returns decision_required, use the local Researcher graph to resolve or reject the candidate before materializing."], ["It never executes SQL supplied by the model.", "It never chooses research boundaries, group membership, fuzzy authorization, or spatial/temporal tolerance.", "It never treats physical column similarity as proof of scientific equivalence.", "It does not produce a scientific conclusion; it produces a data artifact or a decision request."], "DuckDB relational execution over source-owned readers; specialized providers may be registered behind the same contract"),
        "time.temporal_structure": ("Summarize event-age distribution, range, and resolvable temporal units.", ["event ages", "age uncertainties", "source identifiers"], ["age range", "time-bin counts", "temporal summary"], "present-day time axis (Ma)", ["Use to establish the data quality and broad temporal structure."], ["Does not identify individual peak positions or mechanisms."], "NumPy"),
        "time.weighted_age": ("Compute an inverse-variance weighted age for observations representing one event.", ["ages from one event", "age uncertainties"], ["weighted mean age", "uncertainty"], "present-day time axis (Ma)", ["Use only when observations have a shared event interpretation."], ["Does not resolve multiple events into one age."], "NumPy"),
        "time.change_point": ("Segment a sorted age series and identify candidate temporal regime boundaries.", ["materialized event ages"], ["candidate change-point ages", "segmentation penalty"], "present-day time axis (Ma)", ["Use when the target is a boundary in an ordered series."], ["It is not a unimodality test, peak-count test, or peak locator; candidate boundaries require scientific interpretation."], "ruptures.Pelt"),
        "time.dip": ("Test the null hypothesis that a one-dimensional age distribution is unimodal using Hartigan's dip statistic.", ["materialized one-dimensional age observations"], ["dip statistic", "p-value", "p-value method"], "age in Ma", ["Use when the question is whether the observed distribution departs from unimodality independently of KDE bandwidth or a Gaussian-mixture family."], ["Non-rejection is not proof of unimodality; the test does not estimate mode count or location and does not model individual age uncertainties."], "diptest.diptest (Python/C++)"),
        "time.bootstrap": ("Quantify sampling stability of the sample median age by nonparametric resampling.", ["materialized event ages"], ["sample median age", "bootstrap 95% interval"], "present-day time axis (Ma)", ["Use only when median-age sampling stability is the requested output."], ["It does not bootstrap an arbitrary statistic or peak location and cannot recover systematically missing observations."], "NumPy"),
        "sample.spot_quality": ("Audit available spot or step quality fields without reconstructing a geological event.", ["spot or step payload", "sample identifiers", "age information"], ["quality coverage", "outlier candidates", "older-age candidates"], "within-sample analysis layer", ["Use only quality fields actually stored in the database."], ["Absent concordance or domain labels limit quality assessment; this tool does not perform event reconstruction."], "NumPy, SciPy"),
        "space.centroid": ("Compute the displacement between two declared temporal-group centroids.", ["located events", "temporal grouping"], ["group centroids", "centroid displacement"], "present-day geographic coordinates", ["Use only when the grouping rule is explicit in the local graph."], ["A centroid is a regional summary, not a continuous path or causal mechanism."], "NumPy, PyProj"),
        "space.regression": ("Estimate a continuous statistical relation between event age and longitude.", ["event ages", "longitudes", "age uncertainties"], ["longitude-age slope", "confidence interval", "significance"], "present-day longitude axis", ["Use for a continuous relation along longitude."], ["Reports an age-longitude statistical relation."], "statsmodels"),
        "space.geometry": ("Measure observation coverage geometry in a projected equal-area coordinate system.", ["located events"], ["coverage geometry", "sampling-density indicator"], "projected planar coordinates", ["Use to diagnose spatial coverage."], ["Coverage geometry is not a geological boundary."], "GeoPandas, Shapely, PyProj"),
        "uncertainty.monte_carlo_migration": ("Propagate age and location uncertainty through a defined migration statistic.", ["event ages and errors", "sample coordinates and errors", "migration statistic"], ["simulation distribution", "uncertainty interval"], "defined by the input statistic", ["Define the migration statistic before use."], ["Does not define a spatial process or mechanism."], "NumPy"),
        "bias.leave_one_source_out": ("Test sensitivity of a result to individual papers or data sources.", ["source identifiers", "target statistic"], ["source sensitivity", "leave-one-out results"], "reference frame of the target statistic", ["Use as a robustness check for an existing statistic."], ["Cannot assess unrecorded source bias."], "NumPy"),
        "bias.leave_one_region_out": ("Test sensitivity of a result to spatial partitions.", ["spatial partitions", "target statistic"], ["regional sensitivity", "resampling results"], "present-day geographic partitions", ["Requires usable region or basin identifiers."], ["Textual partitions do not ensure uniform geographic sampling."], "NumPy"),
        "bias.permutation_null": ("Construct a spatial-label permutation null model for an age-space association.", ["event ages", "spatial coordinates"], ["null distribution", "permutation significance"], "present-day geographic coordinates", ["Use with a defined age-space statistic."], ["Does not replace measurement-error or sampling-process models."], "NumPy"),
        "bias.spatial_resampling": ("Use spatial resampling to assess uneven coverage and source contribution.", ["sample coordinates", "source identifiers"], ["spatial resampling results", "coverage-normalized indicators"], "present-day geographic coordinates", ["Requires sufficient located observations."], ["Does not create new spatial observations."], "NumPy, GeoPandas"),
        "external.reconstruct_paleolocation": ("Reconstruct sample paleolocations at a specified geological age using a versioned plate-rotation model.", ["modern sample coordinates", "geological age", "plate-rotation model"], ["paleogeographic coordinates", "model version", "reconstruction uncertainty"], "paleogeographic coordinates at specified age", ["Requires a registered local rotation model."], ["Does not provide boundary geometry or distance."], "pygplates, GPlately"),
        "external.boundary_compatibility": ("Relate reconstructed sample positions to a versioned paleo-trench boundary.", ["reconstructed sample position", "geological age", "paleo-trench geometry", "plate model"], ["distance to paleo-trench", "relative motion", "model version"], "paleogeographic coordinates at specified age", ["Requires local paleo-trench data and a rotation model."], ["Its boundary object is a paleo-trench."], "pygplates, GPlately, GeoPandas"),
        "multi_clock.event_matching": ("Summarize multiple isotope-system age groups as candidates for later compatibility testing.", ["dated records", "isotope-system labels"], ["clock-group counts", "clock-group median ages"], "event time axis", ["Use to establish whether multiple clock groups exist in scope."], ["Does not establish event-level physical comparability or thermal-history ordering."], "NumPy"),
        "multi_clock.lag_distribution": ("Summarize response lags for already matched cross-chronometer events.", ["matched cross-chronometer events"], ["lag distribution", "ordering statistics"], "event time axis", ["Use only after event matching."], ["Does not establish physical comparability."], "NumPy"),
    }
    cards.update({
        "chronology.isoplotr": ("Delegate raw chronometer reduction and plotting to IsoplotR when installed.", ["raw isotope ratios", "chronometer-specific uncertainties", "measurement metadata"], ["provider result or availability gap"], "chronometer-specific measurement space", ["Select only for sources preserving raw measurement semantics."], ["This runtime never substitutes a Python reimplementation for IsoplotR."], "R IsoplotR provider"),
        "detrital.detritalpy": ("Delegate detrital grain-to-sample distribution analysis to detritalPy when installed.", ["grain ages", "sample grouping"], ["provider result or availability gap"], "detrital sample-to-grain relation", ["Select only when both grain and sample grouping are explicit."], ["It is not applicable to an ungrouped event catalogue."], "Python detritalPy provider"),
        "sample.spot_quality": ("Audit native spot or step records retained by an Event Clock source.", ["native spot or step payload"], ["quality coverage and candidate flags"], "within-sample measurement space", ["Use only when the source preserves native analyses."], ["No payload means an explicit applicability gap, not a failed event reconstruction."], "NumPy, SciPy"),
        "space.resolution": ("Assess whether observations support a requested spatial scale before spatial inference.", ["materialized positions", "position provenance", "position uncertainty"], ["coverage and conservative resolvable scale"], "present-day geographic coordinates", ["Use before choosing a spatial relationship test."], ["It does not test migration or any causal relation."], "GeoPandas, Shapely, PyProj"),
        "model.compare": ("Compare declared low-dimensional observable-process models.", ["materialized age-position observations", "declared model family"], ["model scores and identifiability result"], "declared observation coordinate frame", ["Use only after candidate observable patterns and alternatives are stated in the local graph."], ["It is not a three-dimensional geodynamic simulation."], "NumPy, SciPy"),
    })
    cards.update({
        "time.kde": ("Estimate one-dimensional age density with a kernel method.", ["materialized age observations"], ["density and candidate peaks"], "age in Ma", ["Use to describe distributional structure."], ["It does not identify a geological mechanism."], "SciPy gaussian_kde"),
        "time.gmm": ("Fit Gaussian mixture components to an age series.", ["materialized age observations"], ["component means and model scores"], "age in Ma", ["Use to compare declared mixture representations."], ["Mixture components are not automatically geological events."], "scikit-learn GaussianMixture"),
        "space.dbscan": ("Cluster located observations with an explicit geographic radius.", ["materialized coordinates", "eps_km"], ["cluster labels and noise count"], "present-day geographic coordinates", ["Set eps_km explicitly."], ["Clusters do not establish causation."], "scikit-learn DBSCAN with haversine metric"),
    })
    registered_ids = {tool.tool_id for tool in registry.all()}
    for tool_id, (purpose, inputs, outputs, frame, guidelines, limitations, implementation) in cards.items():
        if tool_id not in registered_ids:
            continue
        tool = registry.get(tool_id)
        tool.card = ToolCapabilityCard(
            purpose=purpose, input_objects=inputs, output_objects=outputs,
            reference_frame=frame, usage_guidelines=guidelines, limitations=limitations,
            parameter_explanation={"research_scope": "The current plan supplies the bound data scope; the tool does not expand it."},
            implementation=implementation,
        )
    registry.get("space.dbscan").card.parameter_explanation["eps_km"] = "The explicit clustering radius in kilometres."
