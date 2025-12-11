"""
BigQuery data adapter for CDP-style event data.

Handles:
- Separate tables per event type with varying schemas
- Customer ID merge/resolution tables
- Customer property tables with nested structures
- External ID mapping tables
- Automatic schema discovery with LLM inference
- Sample data extraction for local testing
"""

from src.data.bigquery.config import (
    BigQueryConfig,
    EventTableConfig,
    CustomerTableConfig,
    MergeTableConfig,
    FieldMapping,
    EventType,
)
from src.data.bigquery.adapter import (
    BigQueryAdapter,
    load_from_bigquery,
    create_config_from_tables,
)
from src.data.bigquery.schema_discovery import (
    SchemaDiscovery,
    DiscoveryResult,
    TableRelevanceFilter,
    discover_schema,
    print_discovery_report,
)
from src.data.bigquery.sample_extractor import (
    SampleExtractor,
    ExtractionConfig,
    ExtractionResult,
    LocalDataLoader,
    extract_sample_data,
    load_local_sample,
)

__all__ = [
    # Config
    "BigQueryConfig",
    "EventTableConfig",
    "CustomerTableConfig",
    "MergeTableConfig",
    "FieldMapping",
    "EventType",
    # Adapter
    "BigQueryAdapter",
    "load_from_bigquery",
    "create_config_from_tables",
    # Schema Discovery
    "SchemaDiscovery",
    "DiscoveryResult",
    "TableRelevanceFilter",
    "discover_schema",
    "print_discovery_report",
    # Sample Extraction
    "SampleExtractor",
    "ExtractionConfig",
    "ExtractionResult",
    "LocalDataLoader",
    "extract_sample_data",
    "load_local_sample",
]
