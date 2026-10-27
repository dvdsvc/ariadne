"""Adapters for already-existing observation databases.

This namespace is deliberately outside the research engine. It does not read
papers, infer geological events, or extract scientific claims; it only exposes
rows from an existing source through the ObservationSource contract.
"""

from ..sources import InMemoryObservationSource, JsonObservationSource, MaterialStateSensorSource, OpportunisticSensorSource, SemanticBinding, SpatialConstraintSource, TotalTreeXlsxSource, load_semantic_manifest
from ..global_zircon import GlobalZirconSQLiteSource, build_global_zircon_cache
from ..tabular import DynamicTableObservationSource, InMemoryTableBackend, backend_from_path

__all__ = ["DynamicTableObservationSource", "InMemoryTableBackend", "backend_from_path", "GlobalZirconSQLiteSource", "build_global_zircon_cache", "InMemoryObservationSource", "JsonObservationSource", "MaterialStateSensorSource", "OpportunisticSensorSource", "SemanticBinding", "SpatialConstraintSource", "TotalTreeXlsxSource", "load_semantic_manifest"]
