#!/usr/bin/env python3
"""
Example: Loading Local Data with Flexible Schema Configuration

This example demonstrates how to load data from local parquet files
with different schema configurations for various data sources.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    load_local_data,
    load_events_only,
    LocalDataLoader,
    ClientSchemaConfig,
    create_bloomreach_config,
    create_ga4_config,
    create_segment_config,
)


def example_default_loading():
    """Load data with default (Bloomreach) configuration."""
    print("=" * 60)
    print("Example 1: Default Loading")
    print("=" * 60)

    # Simple one-liner
    events, id_history = load_events_only("data/samples")

    print(f"Loaded {len(events):,} events")
    print(f"Loaded {len(id_history):,} ID history records")

    # Show sample event
    if events:
        e = events[0]
        print(f"\nSample event:")
        print(f"  Type: {e.event_type.value}")
        print(f"  Customer: {e.internal_customer_id}")
        print(f"  Timestamp: {e.timestamp}")


def example_full_statistics():
    """Load data with full statistics."""
    print("\n" + "=" * 60)
    print("Example 2: Loading with Statistics")
    print("=" * 60)

    result = load_local_data("data/samples")

    print(f"Tables loaded: {result.tables_loaded}")
    print(f"Total events: {len(result.events):,}")
    print(f"ID history: {len(result.id_history):,}")
    print(f"Customer properties: {len(result.customer_properties):,}")
    print(f"Unique customers: {result.unique_customers:,}")
    print(f"Load time: {result.load_duration_ms:.1f}ms")

    print("\nEvents by type:")
    for event_type, count in sorted(result.events_by_type.items()):
        print(f"  {event_type}: {count:,}")


def example_custom_config():
    """Load data with custom schema configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Schema Configuration")
    print("=" * 60)

    # Create custom configuration
    custom_config = ClientSchemaConfig(
        client_name="my_company",
        customer_id_field="internal_customer_id",  # Your field name
        timestamp_field="timestamp",
        properties_field="properties",  # Nested properties (or None for flat)

        # ID merge configuration
        id_history_table="customers_id_history",
        past_id_field="past_id",
        canonical_id_field="internal_customer_id",

        # Field alternatives - tries each until a value is found
        category_fields=["category_level_1", "category", "product_category", "type"],
        revenue_fields=["total_price", "amount", "revenue", "value"],

        # Device detection
        mobile_device_values=["mobile", "iOS", "Android", "tablet"],
    )

    # Load with custom config
    events, id_history = load_events_only("data/samples", schema_config=custom_config)

    print(f"Using config: {custom_config.client_name}")
    print(f"Customer ID field: {custom_config.customer_id_field}")
    print(f"Loaded {len(events):,} events")


def example_preset_configs():
    """Show available preset configurations."""
    print("\n" + "=" * 60)
    print("Example 4: Preset Configurations")
    print("=" * 60)

    configs = {
        "Bloomreach": create_bloomreach_config(),
        "GA4": create_ga4_config(),
        "Segment": create_segment_config(),
    }

    for name, config in configs.items():
        print(f"\n{name} Config:")
        print(f"  Customer ID: {config.customer_id_field}")
        print(f"  Timestamp: {config.timestamp_field}")
        print(f"  Properties: {config.properties_field}")
        print(f"  Revenue fields: {config.revenue_fields[:3]}...")


def example_filtered_loading():
    """Load specific tables only."""
    print("\n" + "=" * 60)
    print("Example 5: Filtered Loading")
    print("=" * 60)

    # Load only purchase and view events
    loader = LocalDataLoader(
        "data/samples",
        include_tables=["purchase", "view_item", "customers_id_history"],
    )

    result = loader.load()

    print(f"Tables loaded: {result.tables_loaded}")
    print(f"Events: {len(result.events):,}")


if __name__ == "__main__":
    # Check if data exists
    if not Path("data/samples").exists():
        print("Note: data/samples/ not found.")
        print("Run BigQuery sample extraction first, or adjust the path.")
        print("\nShowing preset configuration examples only...\n")
        example_preset_configs()
    else:
        example_default_loading()
        example_full_statistics()
        example_custom_config()
        example_preset_configs()
        example_filtered_loading()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
