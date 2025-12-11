"""
Example: Using the BigQuery adapter to load real CDP data.

This shows how to configure the pipeline to read from your BigQuery tables
with their specific schemas.
"""

from datetime import datetime, timezone

from src.data.bigquery import (
    BigQueryConfig,
    EventTableConfig,
    CustomerTableConfig,
    MergeTableConfig,
    FieldMapping,
)
from src.data.bigquery.config import EventType
from src.pipeline import PipelineConfig, run_pipeline, format_pipeline_summary
from src.analysis.integrated_analysis import format_integrated_report


def create_loavies_config() -> BigQueryConfig:
    """
    Example configuration for a fashion e-commerce CDP.

    Adjust table names and field mappings to match your actual schema.
    """

    # Define event table configurations based on your schema
    event_tables = [
        # Purchase events
        EventTableConfig(
            table_name="eu1_xxx_views.purchase",  # Replace with actual table
            event_type=EventType.PURCHASE,
            customer_id_field="internal_customer_id",
            timestamp_field="timestamp",
            property_mappings={
                "order_id": FieldMapping("properties.purchase_id", "order_id"),
                "order_total": FieldMapping("properties.total_price", "order_total", transform="decimal"),
                "product_list": FieldMapping("properties.product_list", "product_list", transform="json_parse"),
                "channel": FieldMapping("properties.channel", "channel"),
                "payment_method": FieldMapping("properties.payment_method", "payment_method"),
                "shipping_country": FieldMapping("properties.shipping_country", "shipping_country"),
                "device": FieldMapping("properties.device", "device"),
            },
        ),

        # Page view events
        EventTableConfig(
            table_name="eu1_xxx_views.page_view",  # Replace with actual table
            event_type=EventType.PAGE_VIEW,
            customer_id_field="internal_customer_id",
            timestamp_field="timestamp",
            property_mappings={
                "page_url": FieldMapping("properties.page_url", "page_url"),
                "page_title": FieldMapping("properties.page_title", "page_title"),
                "device": FieldMapping("properties.device", "device"),
            },
        ),

        # View item events (product views)
        EventTableConfig(
            table_name="eu1_xxx_views.view_item",  # Replace with actual table
            event_type=EventType.VIEW_ITEM,
            customer_id_field="internal_customer_id",
            timestamp_field="timestamp",
            product_category_field="properties.product_category",
            property_mappings={
                "product_id": FieldMapping("properties.product_id", "product_id"),
                "product_name": FieldMapping("properties.product_name", "product_name"),
                "product_category": FieldMapping("properties.product_category", "product_category"),
                "product_price": FieldMapping("properties.product_price", "product_price", transform="decimal"),
            },
        ),

        # Add to cart events
        EventTableConfig(
            table_name="eu1_xxx_views.add_to_cart",  # Replace with actual table
            event_type=EventType.ADD_TO_CART,
            customer_id_field="internal_customer_id",
            timestamp_field="timestamp",
            product_category_field="properties.product_category",
            property_mappings={
                "product_id": FieldMapping("properties.product_id", "product_id"),
                "product_name": FieldMapping("properties.product_name", "product_name"),
                "product_category": FieldMapping("properties.product_category", "product_category"),
                "product_price": FieldMapping("properties.product_price", "product_price", transform="decimal"),
                "quantity": FieldMapping("properties.quantity", "quantity", transform="int", default=1),
            },
        ),
    ]

    # Customer properties table config
    customer_table = CustomerTableConfig(
        table_name="eu1_xxx_views.customers",  # Replace with actual table
        customer_id_field="internal_customer_id",
        property_mappings={
            "email": FieldMapping("properties.email", "email"),
            "first_name": FieldMapping("properties.first_name", "first_name"),
            "last_name": FieldMapping("properties.last_name", "last_name"),
            "country": FieldMapping("properties.country", "country"),
            "language": FieldMapping("properties.language", "language"),
            "gender": FieldMapping("properties.gender", "gender"),
            "birth_date": FieldMapping("properties.birth_date", "birth_date"),
            # RFM and engagement from your CDP
            "rfm_segment": FieldMapping("properties.rfm_simplified_today", "rfm_segment"),
            "engagement_level": FieldMapping("properties.engagement_level", "engagement_level"),
            # Consent fields
            "newsletter": FieldMapping("properties.newsletter", "newsletter", transform="bool"),
            "double_optin": FieldMapping("properties.double_optin", "double_optin", transform="bool"),
        },
        auto_detect_fields=True,  # Will also try to find other common fields
    )

    # ID merge table config
    merge_table = MergeTableConfig(
        table_name="eu1_xxx_views.customer_id_merge",  # Replace with actual table
        current_id_field="internal_customer_id",
        past_id_field="past_id",
    )

    return BigQueryConfig(
        project_id="your-gcp-project",  # Replace with your project
        dataset_id="your_dataset",  # Replace with your dataset
        event_tables=event_tables,
        customer_table=customer_table,
        merge_table=merge_table,
        # Date range for analysis
        start_date="2024-01-01",
        end_date="2024-12-31",
        # Sampling for testing (remove for production)
        # sample_rate=0.1,  # 10% sample
        # limit_per_table=10000,  # Max rows per table for testing
    )


def run_with_bigquery():
    """Run the segmentation pipeline with BigQuery data."""

    # Create BigQuery config
    bq_config = create_loavies_config()

    # Create pipeline config
    pipeline_config = PipelineConfig(
        data_source="bigquery",
        bigquery_config=bq_config,
        # Clustering settings
        n_clusters=6,
        auto_select_k=True,
        k_range=(4, 10),
        # Analysis settings
        run_sensitivity=True,
        run_integrated_analysis=True,
        verbose=True,
    )

    # Run pipeline
    print("Starting segmentation pipeline with BigQuery data...")
    print(f"Project: {bq_config.project_id}")
    print(f"Dataset: {bq_config.dataset_id}")
    print(f"Event tables: {[t.table_name for t in bq_config.event_tables]}")
    print()

    result = run_pipeline(config=pipeline_config)

    # Print results
    print(format_pipeline_summary(result))

    if result.integrated_analysis:
        print()
        print(format_integrated_report(result.integrated_analysis))

    return result


# =============================================================================
# QUICK CONFIG HELPER
# =============================================================================


def quick_bigquery_config(
    project_id: str,
    dataset_id: str,
    table_prefix: str = "",
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    sample_rate: float | None = None,
    limit: int | None = None,
) -> BigQueryConfig:
    """
    Quick helper to create a BigQuery config with standard table naming.

    Assumes tables are named: {prefix}purchase, {prefix}page_view, etc.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_prefix: Prefix for table names (e.g., "eu1_xxx_views.")
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        sample_rate: Sampling rate (0.0-1.0) for testing
        limit: Row limit per table for testing

    Returns:
        BigQueryConfig ready for use
    """
    from src.data.bigquery import create_config_from_tables

    return create_config_from_tables(
        project_id=project_id,
        dataset_id=dataset_id,
        purchase_table=f"{table_prefix}purchase",
        page_view_table=f"{table_prefix}page_view",
        view_item_table=f"{table_prefix}view_item",
        add_to_cart_table=f"{table_prefix}add_to_cart",
        customer_table=f"{table_prefix}customers",
        merge_table=f"{table_prefix}customer_id_merge",
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )


if __name__ == "__main__":
    # Example usage - update with your actual GCP credentials and table names
    # run_with_bigquery()

    print("BigQuery configuration example.")
    print()
    print("To use:")
    print("1. Update project_id, dataset_id, and table names in create_loavies_config()")
    print("2. Ensure GCP credentials are configured (GOOGLE_APPLICATION_CREDENTIALS)")
    print("3. Run: python -m examples.bigquery_config_example")
    print()
    print("Quick start with standard table naming:")
    print("""
    from examples.bigquery_config_example import quick_bigquery_config
    from src.pipeline import PipelineConfig, run_pipeline

    bq_config = quick_bigquery_config(
        project_id="my-project",
        dataset_id="my_dataset",
        table_prefix="cdp_views.",
        start_date="2024-01-01",
        limit=10000,  # For testing
    )

    result = run_pipeline(config=PipelineConfig(
        data_source="bigquery",
        bigquery_config=bq_config,
    ))
    """)
