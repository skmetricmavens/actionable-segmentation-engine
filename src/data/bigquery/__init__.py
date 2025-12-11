"""
BigQuery data adapter for CDP-style event data.

Handles:
- Separate tables per event type with varying schemas
- Customer ID merge/resolution tables
- Customer property tables with nested structures
- External ID mapping tables
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

__all__ = [
    "BigQueryConfig",
    "EventTableConfig",
    "CustomerTableConfig",
    "MergeTableConfig",
    "FieldMapping",
    "EventType",
    "BigQueryAdapter",
    "load_from_bigquery",
    "create_config_from_tables",
]
