"""
Configuration for BigQuery data sources.

Defines how to map BigQuery tables with varying schemas to our internal data model.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(str, Enum):
    """Supported event types from BigQuery tables."""

    PURCHASE = "purchase"
    PAGE_VIEW = "page_view"
    VIEW_ITEM = "view_item"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    BEGIN_CHECKOUT = "begin_checkout"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    WISHLIST_ADD = "wishlist_add"
    SEARCH = "search"
    EMAIL_OPEN = "email_open"
    EMAIL_CLICK = "email_click"
    CUSTOM = "custom"


@dataclass
class FieldMapping:
    """Maps a BigQuery field to an internal field."""

    source_field: str  # Field path in BQ (e.g., "properties.total_price")
    target_field: str  # Our internal field name
    transform: str | None = None  # Optional transform: "decimal", "int", "datetime", "json_parse"
    default: Any = None  # Default value if missing


@dataclass
class EventTableConfig:
    """Configuration for a single event table."""

    # Table identification
    table_name: str  # Full table name: project.dataset.table or just table
    event_type: EventType

    # Required fields (must exist in schema)
    customer_id_field: str = "internal_customer_id"
    timestamp_field: str = "timestamp"
    event_type_field: str | None = "type"  # If None, use event_type from config

    # Properties mapping - maps BQ fields to our EventProperties
    # Keys are our internal field names, values are BQ field paths
    property_mappings: dict[str, FieldMapping] = field(default_factory=dict)

    # For purchase events - product category field path
    product_category_field: str | None = None  # e.g., "properties.product_category" or from product_list

    # Filter conditions (WHERE clause additions)
    filters: list[str] = field(default_factory=list)

    # Date range filter field
    date_field: str = "timestamp"

    @classmethod
    def purchase_table(
        cls,
        table_name: str,
        *,
        product_list_field: str = "properties.product_list",
        total_price_field: str = "properties.total_price",
        order_id_field: str = "properties.purchase_id",
    ) -> "EventTableConfig":
        """Create config for a purchase event table."""
        return cls(
            table_name=table_name,
            event_type=EventType.PURCHASE,
            property_mappings={
                "order_id": FieldMapping(order_id_field, "order_id"),
                "order_total": FieldMapping(total_price_field, "order_total", transform="decimal"),
                "product_list": FieldMapping(product_list_field, "product_list", transform="json_parse"),
            },
        )

    @classmethod
    def page_view_table(
        cls,
        table_name: str,
        *,
        page_url_field: str = "properties.page_url",
        page_title_field: str = "properties.page_title",
    ) -> "EventTableConfig":
        """Create config for a page view event table."""
        return cls(
            table_name=table_name,
            event_type=EventType.PAGE_VIEW,
            property_mappings={
                "page_url": FieldMapping(page_url_field, "page_url"),
                "page_title": FieldMapping(page_title_field, "page_title"),
            },
        )

    @classmethod
    def view_item_table(
        cls,
        table_name: str,
        *,
        product_id_field: str = "properties.product_id",
        product_name_field: str = "properties.product_name",
        product_category_field: str = "properties.product_category",
        product_price_field: str = "properties.product_price",
    ) -> "EventTableConfig":
        """Create config for a view_item event table."""
        return cls(
            table_name=table_name,
            event_type=EventType.VIEW_ITEM,
            product_category_field=product_category_field,
            property_mappings={
                "product_id": FieldMapping(product_id_field, "product_id"),
                "product_name": FieldMapping(product_name_field, "product_name"),
                "product_category": FieldMapping(product_category_field, "product_category"),
                "product_price": FieldMapping(product_price_field, "product_price", transform="decimal"),
            },
        )

    @classmethod
    def add_to_cart_table(
        cls,
        table_name: str,
        *,
        product_id_field: str = "properties.product_id",
        product_name_field: str = "properties.product_name",
        product_category_field: str = "properties.product_category",
        product_price_field: str = "properties.product_price",
        quantity_field: str = "properties.quantity",
    ) -> "EventTableConfig":
        """Create config for an add_to_cart event table."""
        return cls(
            table_name=table_name,
            event_type=EventType.ADD_TO_CART,
            product_category_field=product_category_field,
            property_mappings={
                "product_id": FieldMapping(product_id_field, "product_id"),
                "product_name": FieldMapping(product_name_field, "product_name"),
                "product_category": FieldMapping(product_category_field, "product_category"),
                "product_price": FieldMapping(product_price_field, "product_price", transform="decimal"),
                "quantity": FieldMapping(quantity_field, "quantity", transform="int", default=1),
            },
        )


@dataclass
class CustomerTableConfig:
    """Configuration for customer properties table."""

    table_name: str
    customer_id_field: str = "internal_customer_id"

    # Properties to extract (from properties or raw_properties nested structs)
    # Maps our internal field name to BQ field path
    property_mappings: dict[str, FieldMapping] = field(default_factory=dict)

    # Known useful fields to auto-detect
    auto_detect_fields: bool = True

    # Fields to look for (if auto_detect_fields=True)
    known_fields: list[str] = field(default_factory=lambda: [
        "email", "first_name", "last_name", "country", "language",
        "gender", "birth_date", "phone",
        "rfm_today", "rfm_simplified_today", "engagement_level",
        "newsletter", "double_optin",
        "cart_value", "last_viewed_items", "last_kept_items",
    ])

    @classmethod
    def default(cls, table_name: str) -> "CustomerTableConfig":
        """Create default customer table config with common field mappings."""
        return cls(
            table_name=table_name,
            property_mappings={
                "email": FieldMapping("properties.email", "email"),
                "first_name": FieldMapping("properties.first_name", "first_name"),
                "last_name": FieldMapping("properties.last_name", "last_name"),
                "country": FieldMapping("properties.country", "country"),
                "language": FieldMapping("properties.language", "language"),
                "gender": FieldMapping("properties.gender", "gender"),
                "birth_date": FieldMapping("properties.birth_date", "birth_date"),
                "rfm": FieldMapping("properties.rfm_simplified_today", "rfm_segment"),
                "engagement_level": FieldMapping("properties.engagement_level", "engagement_level"),
            },
        )


@dataclass
class MergeTableConfig:
    """Configuration for customer ID merge table."""

    table_name: str
    current_id_field: str = "internal_customer_id"
    past_id_field: str = "past_id"


@dataclass
class ExternalIdTableConfig:
    """Configuration for external ID mapping table."""

    table_name: str
    customer_id_field: str = "internal_customer_id"
    id_name_field: str = "id_name"
    id_value_field: str = "id_value"


@dataclass
class BigQueryConfig:
    """Complete configuration for BigQuery data source."""

    # GCP project and dataset
    project_id: str
    dataset_id: str

    # Event tables (one per event type)
    event_tables: list[EventTableConfig] = field(default_factory=list)

    # Customer properties table
    customer_table: CustomerTableConfig | None = None

    # ID merge table
    merge_table: MergeTableConfig | None = None

    # External IDs table
    external_ids_table: ExternalIdTableConfig | None = None

    # Date range for queries
    start_date: str | None = None  # YYYY-MM-DD
    end_date: str | None = None    # YYYY-MM-DD

    # Query limits (for testing)
    limit_per_table: int | None = None

    # Sampling (for large datasets)
    sample_rate: float | None = None  # 0.0-1.0

    def get_full_table_name(self, table_name: str) -> str:
        """Get fully qualified table name."""
        if "." in table_name:
            return table_name
        return f"{self.project_id}.{self.dataset_id}.{table_name}"

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "BigQueryConfig":
        """Create config from dictionary (e.g., from YAML/JSON)."""
        event_tables = []
        for et_config in config.get("event_tables", []):
            event_tables.append(EventTableConfig(
                table_name=et_config["table_name"],
                event_type=EventType(et_config["event_type"]),
                customer_id_field=et_config.get("customer_id_field", "internal_customer_id"),
                timestamp_field=et_config.get("timestamp_field", "timestamp"),
                property_mappings={
                    k: FieldMapping(**v) if isinstance(v, dict) else FieldMapping(v, k)
                    for k, v in et_config.get("property_mappings", {}).items()
                },
            ))

        customer_table = None
        if "customer_table" in config:
            ct_config = config["customer_table"]
            customer_table = CustomerTableConfig(
                table_name=ct_config["table_name"],
                customer_id_field=ct_config.get("customer_id_field", "internal_customer_id"),
            )

        merge_table = None
        if "merge_table" in config:
            mt_config = config["merge_table"]
            merge_table = MergeTableConfig(
                table_name=mt_config["table_name"],
                current_id_field=mt_config.get("current_id_field", "internal_customer_id"),
                past_id_field=mt_config.get("past_id_field", "past_id"),
            )

        return cls(
            project_id=config["project_id"],
            dataset_id=config["dataset_id"],
            event_tables=event_tables,
            customer_table=customer_table,
            merge_table=merge_table,
            start_date=config.get("start_date"),
            end_date=config.get("end_date"),
            limit_per_table=config.get("limit_per_table"),
            sample_rate=config.get("sample_rate"),
        )
