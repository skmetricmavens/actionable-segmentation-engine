"""
Data module for the segmentation engine.

Contains data loading, schemas, synthetic generation, and customer ID merging.
Supports flexible schema mapping for different client data structures.
"""

from src.data.local_loader import (
    LocalDataLoader,
    LoadResult,
    load_local_data,
    load_events_only,
)
from src.data.field_mapping import (
    ClientSchemaConfig,
    EventTypeMapping,
    SemanticFieldType,
    FieldMapping,
    EventTypeConfig,
    create_bloomreach_config,
    create_ga4_config,
    create_segment_config,
    create_generic_config,
    extract_field,
    extract_with_alternatives,
)

__all__ = [
    # Data loading
    "LocalDataLoader",
    "LoadResult",
    "load_local_data",
    "load_events_only",
    # Schema configuration
    "ClientSchemaConfig",
    "EventTypeMapping",
    "SemanticFieldType",
    "FieldMapping",
    "EventTypeConfig",
    # Preset configs
    "create_bloomreach_config",
    "create_ga4_config",
    "create_segment_config",
    "create_generic_config",
    # Field extraction helpers
    "extract_field",
    "extract_with_alternatives",
]
