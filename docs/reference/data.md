# Data Layer Reference

The data layer provides flexible loading, schema mapping, and ID resolution for customer event data.

## Data Loading

### LocalDataLoader

The primary way to load data from parquet files with flexible schema mapping:

```python
from src.data import LocalDataLoader, create_bloomreach_config

# Default (Bloomreach-style data)
loader = LocalDataLoader("data/samples")
result = loader.load()

# With custom schema config
loader = LocalDataLoader("data/samples", schema_config=create_bloomreach_config())
result = loader.load()
```

### Convenience Functions

```python
from src.data import load_local_data, load_events_only

# Load full result with statistics
result = load_local_data("data/samples")
print(f"Loaded {len(result.events)} events")
print(f"Tables: {result.tables_loaded}")

# Load just events and ID history for pipeline
events, id_history = load_events_only("data/samples")
```

::: src.data.local_loader
    options:
      show_root_heading: true
      show_source: false
      members:
        - LocalDataLoader
        - LoadResult
        - load_local_data
        - load_events_only

## Schema Configuration

### Client Schema Config

Configure field mappings for different data sources:

```python
from src.data import ClientSchemaConfig

config = ClientSchemaConfig(
    client_name="my_company",
    customer_id_field="user_id",
    timestamp_field="event_timestamp",
    properties_field="event_data",

    # ID merge configuration
    id_history_table="user_id_mapping",
    past_id_field="old_id",
    canonical_id_field="current_id",

    # Flexible field alternatives
    category_fields=["category", "product_category", "item_type"],
    revenue_fields=["total", "amount", "value", "revenue"],

    # Device detection values
    mobile_device_values=["mobile", "iOS", "Android"],
)
```

### Preset Configurations

Pre-built configurations for common platforms:

```python
from src.data import (
    create_bloomreach_config,  # Bloomreach Engagement
    create_ga4_config,         # Google Analytics 4
    create_segment_config,     # Segment.com
    create_generic_config,     # Minimal assumptions
)

# Each returns a ClientSchemaConfig instance
config = create_ga4_config()
```

::: src.data.field_mapping
    options:
      show_root_heading: true
      show_source: false
      members:
        - ClientSchemaConfig
        - EventTypeConfig
        - FieldMapping
        - SemanticFieldType
        - EventTypeMapping
        - create_bloomreach_config
        - create_ga4_config
        - create_segment_config
        - create_generic_config

## Schemas

Core data models used throughout the pipeline:

::: src.data.schemas
    options:
      show_root_heading: true
      show_source: true

## ID Resolution

Customer ID merging and unification:

::: src.data.joiner
    options:
      show_root_heading: true
      show_source: true

## Synthetic Data Generation

Generate synthetic data for testing:

::: src.data.synthetic_generator
    options:
      show_root_heading: true
      show_source: true

## BigQuery Integration

### Sample Extraction

Extract sample data from BigQuery for local testing:

```python
from src.data.bigquery import SampleExtractor, ExtractionConfig

config = ExtractionConfig(
    project_id="my-project",
    dataset_id="my_dataset",
    months_back=3,
    max_customers=10000,
    output_dir="data/samples",
)

extractor = SampleExtractor(config)
result = extractor.extract()
print(f"Extracted {result.total_rows} rows")
```

### Schema Discovery

Auto-discover BigQuery schema and generate configuration:

```python
from src.data.bigquery import SchemaDiscovery, BigQueryConfig

discovery = SchemaDiscovery(
    project_id="my-project",
    dataset_id="my_dataset",
)

# Analyze tables and suggest field mappings
config = discovery.discover_config()
```

::: src.data.bigquery
    options:
      show_root_heading: true
      show_source: false
