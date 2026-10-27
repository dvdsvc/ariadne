"""Versioned, local-cache external physical boundary conditions.

No research tool scrapes the web. Acquisition can populate a cache separately;
the research loop calls stable physical operations against registered versions.
"""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path
from typing import Any, Protocol

from pyproj import Geod

from .registry import ExternalDataset, ExternalDataRegistry


_GEOD = Geod(ellps="WGS84")


def _geodesic_km(first: tuple[float, float], second: tuple[float, float]) -> float:
    _, _, meters = _GEOD.inv(first[1], first[0], second[1], second[0])
    return meters / 1000


def _geodesic_azimuth(first: tuple[float, float], second: tuple[float, float]) -> float:
    azimuth, _, _ = _GEOD.inv(first[1], first[0], second[1], second[0])
    return azimuth


class ExternalProvider(Protocol):
    def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


class CachedExternalProvider:
    def __init__(self, dataset: ExternalDataset) -> None:
        self.dataset = dataset

    def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if not self.dataset.available or not self.dataset.local_uri:
            return {"status": "data_unavailable", "dataset_id": self.dataset.dataset_id, "operation": operation}
        return {"status": "data_unavailable", "dataset_id": self.dataset.dataset_id, "version": self.dataset.version, "operation": operation, "reason": "A dataset exists but no provider adapter implements this physical operation."}


class GPlatesProvider:
    """Adapter boundary for locally cached GPlates/GPlately models.

    It deliberately does not download or scrape. A deployment supplies the local
    adapter callable backed by gplately/pygplates, while research code sees only
    reconstruct_location, distance_to_trench and plate_velocity.
    """

    supported_operations = {"reconstruct_location", "distance_to_trench", "plate_velocity", "relative_plate_motion"}

    def __init__(self, dataset: ExternalDataset, adapter: Any | None = None) -> None:
        self.dataset, self.adapter = dataset, adapter

    def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if operation not in self.supported_operations:
            return {"status": "not_applicable", "operation": operation}
        if self.adapter is None:
            return {"status": "data_unavailable", "dataset_id": self.dataset.dataset_id, "operation": operation, "reason": "Local GPlates provider has not been configured for this runtime."}
        result = self.adapter(operation, arguments)
        result.setdefault("status", "ok")
        return result


class PyGPlatesLocalProvider:
    """Direct pygplates adapter for a locally cached rotation model.

    A plate id is deliberately required: guessing a plate at a location would
    create a false boundary condition. Trench distance needs a deployment
    topology/geometry adapter and therefore returns data_unavailable until one
    is supplied.
    """

    def __init__(self, dataset: ExternalDataset) -> None:
        self.dataset = dataset
        import pygplates
        self.pygplates = pygplates
        self.rotation_model = pygplates.RotationModel(dataset.local_uri)

    def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if operation == "reconstruct_location":
            plate_id = arguments.get("plate_id")
            if plate_id is None:
                return {"status": "data_unavailable", "reason": "reconstruct_location requires an observed or polygon-resolved plate_id"}
            time = float(arguments["age_ma"])
            point = self.pygplates.PointOnSphere(float(arguments["latitude"]), float(arguments["longitude"]))
            rotation = self.rotation_model.get_rotation(time, int(plate_id))
            latitude, longitude = (rotation * point).to_lat_lon()
            return {"status": "ok", "latitude": latitude, "longitude": longitude, "reconstruction_time_ma": time, "plate_id": int(plate_id), "uncertainty_km": self.dataset.uncertainty.get("reconstruction_uncertainty_km")}
        if operation == "plate_velocity":
            plate_id = arguments.get("plate_id")
            if plate_id is None:
                return {"status": "data_unavailable", "reason": "plate_velocity requires plate_id"}
            time = float(arguments["age_ma"])
            point = self.pygplates.PointOnSphere(float(arguments["latitude"]), float(arguments["longitude"]))
            first = (self.rotation_model.get_rotation(time, int(plate_id)) * point).to_lat_lon()
            second = (self.rotation_model.get_rotation(time + 1.0, int(plate_id)) * point).to_lat_lon()
            distance = abs(_geodesic_km(first, second))
            azimuth = _geodesic_azimuth(first, second)
            return {"status": "ok", "velocity_km_per_ma": distance, "azimuth_degrees": azimuth, "method": "pygplates finite rotation over 1 Ma", "plate_id": int(plate_id), "uncertainty_km": self.dataset.uncertainty.get("velocity_uncertainty_km")}
        if operation in {"distance_to_trench", "relative_plate_motion"}:
            return {"status": "data_unavailable", "reason": "This pygplates rotation-only adapter needs locally cached topology/trench geometry before this operation is valid."}
        return {"status": "not_applicable", "operation": operation}


class GPlatelyTopologyProvider(PyGPlatesLocalProvider):
    """Local GPlately topology environment layered on the direct pygplates adapter.

    It validates that the supplied rotation/topology files form a real GPlately
    reconstruction context. Operations remain conservative until a deployment
    configures which topology features represent trenches/subduction zones.
    """

    def __init__(self, dataset: ExternalDataset) -> None:
        super().__init__(dataset)
        import gplately
        topology_uris = dataset.provider_config.get("topology_uris", [])
        topology_files = [self.pygplates.FeatureCollection(str(path)) for path in topology_uris]
        self.reconstruction = gplately.PlateReconstruction(self.rotation_model, topology_features=topology_files or None, anchor_plate_id=dataset.provider_config.get("anchor_plate_id"))

    def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if operation == "distance_to_trench" and not self.dataset.provider_config.get("trench_feature_types"):
            return {"status": "data_unavailable", "reason": "GPlately topology context is loaded, but no locally declared trench/subduction feature selector is configured."}
        return super().call(operation, arguments)


class SimpleKinematicJsonProvider:
    """Deterministic low-dimensional cached provider used for tests and simple boundary models.

    It intentionally has the same stable surface as a later GPlates/GPlately
    provider: reconstruct_location, distance_to_trench, and plate_velocity.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.model = json.loads(self.path.read_text(encoding="utf-8"))

    def call(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        age = float(arguments.get("age_ma", 0.0))
        rotation_rate = float(self.model.get("rotation_deg_per_ma", 0.0))
        if operation == "reconstruct_location":
            return {"status": "ok", "latitude": float(arguments["latitude"]), "longitude": float(arguments["longitude"]) - rotation_rate * age, "uncertainty_km": self.model.get("reconstruction_uncertainty_km", 25.0)}
        if operation == "distance_to_trench":
            trench_at_zero = float(self.model.get("trench_longitude_at_0", 0.0))
            trench_rate = float(self.model.get("trench_deg_per_ma", 0.0))
            trench_lon = trench_at_zero + trench_rate * age
            longitude, latitude = float(arguments["longitude"]), float(arguments["latitude"])
            distance = abs(longitude - trench_lon) * 111.0 * math.cos(math.radians(latitude))
            return {"status": "ok", "distance_km": distance, "trench_longitude": trench_lon, "uncertainty_km": self.model.get("trench_uncertainty_km", 50.0)}
        if operation == "plate_velocity":
            return {"status": "ok", "velocity_km_per_ma": abs(rotation_rate) * 111.0, "direction": "west" if rotation_rate > 0 else "east"}
        return {"status": "not_applicable", "operation": operation}


def register_manifest(registry: ExternalDataRegistry, manifest_path: str | Path) -> list[ExternalDataset]:
    """Register pre-downloaded datasets and metadata; this function never touches the network."""
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload if isinstance(payload, list) else payload.get("datasets", [])
    registered: list[ExternalDataset] = []
    for item in items:
        local_uri = item.get("local_uri")
        resolved = (Path(local_uri).expanduser() if Path(local_uri).is_absolute() else path.parent / local_uri) if local_uri else None
        required = {"dataset_id", "version", "source", "license", "coverage", "uncertainty", "downloaded_at", "checksum"}
        missing = sorted(field for field in required if field not in item)
        if missing:
            raise ValueError(f"External dataset manifest {item.get('dataset_id', '<unknown>')} lacks required provenance fields: {missing}")
        available = bool(resolved and resolved.exists())
        if available and item.get("checksum"):
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
            if digest != item["checksum"]:
                available = False
        dataset = ExternalDataset(
            item["dataset_id"], item.get("name", item["dataset_id"]), item.get("version", "unknown"), item.get("provider_kind", "versioned_local"),
            item.get("capabilities", []), item.get("coverage", {}), item.get("source"), item.get("license"), item.get("downloaded_at"), item.get("checksum"),
            str(resolved) if resolved else None, available, item.get("uncertainty", {}), item.get("provider_config", {}),
        )
        provider: ExternalProvider | None = None
        if available and item.get("provider_kind") == "simple_kinematic_json":
            provider = SimpleKinematicJsonProvider(resolved)
        elif available and item.get("provider_kind") == "pygplates_rotation":
            provider = PyGPlatesLocalProvider(dataset)
        elif available and item.get("provider_kind") == "gplately_topology":
            provider = GPlatelyTopologyProvider(dataset)
        elif available and item.get("provider_kind") in {"gplates", "gplately"}:
            provider = GPlatesProvider(dataset)
        elif available:
            provider = CachedExternalProvider(dataset)
        registry.register(dataset, provider)
        registered.append(dataset)
    return registered
