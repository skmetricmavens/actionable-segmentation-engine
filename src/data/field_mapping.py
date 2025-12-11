"""
Flexible field mapping for client-agnostic data ingestion.

This module provides a configuration-based approach to mapping diverse
client data schemas to the internal canonical schema. Instead of hardcoding
field names, clients can provide their own mapping configurations.
"""

from dataclasses import dataclass, field
from typing import Any, Callable
from enum import Enum


class SemanticFieldType(str, Enum):
    """Semantic field types that can be mapped from any source field name."""

    # Identity fields
    CUSTOMER_ID = "customer_id"
    EVENT_ID = "event_id"
    SESSION_ID = "session_id"

    # Temporal fields
    TIMESTAMP = "timestamp"
    DATE = "date"

    # Event type
    EVENT_TYPE = "event_type"

    # Product fields
    PRODUCT_ID = "product_id"
    PRODUCT_NAME = "product_name"
    PRODUCT_SKU = "product_sku"
    PRODUCT_CATEGORY = "product_category"
    PRODUCT_PRICE = "product_price"
    PRODUCT_LIST = "product_list"

    # Transaction fields
    ORDER_ID = "order_id"
    ORDER_TOTAL = "order_total"
    QUANTITY = "quantity"
    DISCOUNT = "discount"
    CURRENCY = "currency"

    # Engagement fields
    PAGE_URL = "page_url"
    PAGE_TITLE = "page_title"
    SEARCH_QUERY = "search_query"

    # Device/Channel fields
    DEVICE_TYPE = "device_type"
    CHANNEL = "channel"
    BROWSER = "browser"
    OS = "os"
    REFERRER = "referrer"

    # Location fields
    COUNTRY = "country"
    CITY = "city"

    # Customer properties
    EMAIL = "email"
    FIRST_NAME = "first_name"
    LAST_NAME = "last_name"
    PHONE = "phone"

    # Merge/Identity
    PAST_ID = "past_id"
    CANONICAL_ID = "canonical_id"


class EventTypeMapping(str, Enum):
    """Canonical event types that source events can map to."""

    PURCHASE = "purchase"
    PURCHASE_ITEM = "purchase_item"
    VIEW_ITEM = "view_item"
    VIEW_CATEGORY = "view_category"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    CHECKOUT = "checkout"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    PAGE_VIEW = "page_view"
    SEARCH = "search"
    REFUND = "refund"
    WISHLIST_ADD = "wishlist_add"
    EMAIL_OPEN = "email_open"
    EMAIL_CLICK = "email_click"
    CUSTOM = "custom"


@dataclass
class FieldMapping:
    """
    Mapping from a source field to a semantic field type.

    Attributes:
        source_field: The field name in the source data (supports dot notation)
        semantic_type: The canonical semantic meaning of this field
        transform: Optional transformation function (e.g., str, int, decimal)
        default: Default value if source field is missing
        alternatives: Alternative source field names to try
    """
    source_field: str
    semantic_type: SemanticFieldType
    transform: str | Callable | None = None
    default: Any = None
    alternatives: list[str] = field(default_factory=list)

    def get_source_fields(self) -> list[str]:
        """Get all possible source field names (primary + alternatives)."""
        return [self.source_field] + self.alternatives


@dataclass
class EventTypeConfig:
    """
    Configuration for mapping source events to canonical event types.

    Attributes:
        source_table: Name of the source table (e.g., "purchase", "page_visit")
        source_type_value: Value in the event_type field (if applicable)
        canonical_type: The canonical event type this maps to
        is_transactional: Whether this event involves monetary transactions
        is_engagement: Whether this event represents user engagement
        contributes_to_revenue: Whether to include in revenue calculations
        field_mappings: Field mappings specific to this event type
    """
    source_table: str
    canonical_type: EventTypeMapping
    source_type_value: str | None = None
    is_transactional: bool = False
    is_engagement: bool = True
    contributes_to_revenue: bool = False
    field_mappings: list[FieldMapping] = field(default_factory=list)


@dataclass
class ClientSchemaConfig:
    """
    Complete schema configuration for a client's data.

    This is the main configuration object that defines how a client's
    data should be interpreted and mapped to the canonical schema.

    Attributes:
        client_name: Identifier for this client configuration
        description: Description of the client's data source

        # Core field mappings (apply to all events)
        customer_id_field: Field name for customer ID
        timestamp_field: Field name for event timestamp
        properties_field: Field name for nested properties (if any)

        # Event configuration
        event_types: List of event type configurations

        # ID merge configuration
        id_history_table: Table name for ID merge history
        past_id_field: Field name for historical/past customer ID
        canonical_id_field: Field name for current canonical ID

        # Customer properties configuration
        customer_properties_table: Table name for customer attributes
        customer_property_mappings: Field mappings for customer properties

        # Device type detection
        mobile_device_values: Values that indicate mobile device
        desktop_device_values: Values that indicate desktop device

        # Category field alternatives
        category_fields: Possible field names for product category

        # Revenue fields
        revenue_fields: Possible field names for transaction amount
    """
    client_name: str
    description: str = ""

    # Core fields
    customer_id_field: str = "internal_customer_id"
    timestamp_field: str = "timestamp"
    properties_field: str | None = "properties"

    # Event types
    event_types: list[EventTypeConfig] = field(default_factory=list)

    # ID merge
    id_history_table: str = "customers_id_history"
    past_id_field: str = "past_id"
    canonical_id_field: str = "internal_customer_id"

    # Customer properties
    customer_properties_table: str = "customers_properties"
    customer_property_mappings: list[FieldMapping] = field(default_factory=list)

    # Device detection (flexible values)
    mobile_device_values: list[str] = field(
        default_factory=lambda: ["mobile", "tablet", "ios", "android", "Mobile", "iOS", "Android"]
    )
    desktop_device_values: list[str] = field(
        default_factory=lambda: ["desktop", "web", "Web", "Desktop"]
    )

    # Field name alternatives
    category_fields: list[str] = field(
        default_factory=lambda: [
            "product_category", "category", "category_level_1",
            "category_name", "item_category", "product_type"
        ]
    )
    revenue_fields: list[str] = field(
        default_factory=lambda: [
            "total_price", "order_total", "total_amount", "revenue",
            "transaction_total", "purchase_amount", "amount"
        ]
    )

    def get_event_type_config(self, table_name: str) -> EventTypeConfig | None:
        """Get event type configuration for a table name."""
        for config in self.event_types:
            if config.source_table == table_name:
                return config
        return None

    def get_canonical_event_type(self, table_name: str) -> EventTypeMapping:
        """Get the canonical event type for a source table."""
        config = self.get_event_type_config(table_name)
        if config:
            return config.canonical_type
        # Default mapping by table name
        return _default_table_to_event_type(table_name)


def _default_table_to_event_type(table_name: str) -> EventTypeMapping:
    """Default mapping from common table names to event types."""
    table_lower = table_name.lower()

    mappings = {
        "purchase": EventTypeMapping.PURCHASE,
        "purchases": EventTypeMapping.PURCHASE,
        "order": EventTypeMapping.PURCHASE,
        "orders": EventTypeMapping.PURCHASE,
        "transaction": EventTypeMapping.PURCHASE,
        "transactions": EventTypeMapping.PURCHASE,

        "purchase_item": EventTypeMapping.PURCHASE_ITEM,
        "order_item": EventTypeMapping.PURCHASE_ITEM,
        "line_item": EventTypeMapping.PURCHASE_ITEM,

        "view_item": EventTypeMapping.VIEW_ITEM,
        "product_view": EventTypeMapping.VIEW_ITEM,
        "item_view": EventTypeMapping.VIEW_ITEM,
        "pdp_view": EventTypeMapping.VIEW_ITEM,

        "view_category": EventTypeMapping.VIEW_CATEGORY,
        "category_view": EventTypeMapping.VIEW_CATEGORY,
        "plp_view": EventTypeMapping.VIEW_CATEGORY,

        "cart_update": EventTypeMapping.ADD_TO_CART,
        "add_to_cart": EventTypeMapping.ADD_TO_CART,
        "cart_add": EventTypeMapping.ADD_TO_CART,
        "cart": EventTypeMapping.ADD_TO_CART,

        "checkout": EventTypeMapping.CHECKOUT,
        "begin_checkout": EventTypeMapping.CHECKOUT,

        "session_start": EventTypeMapping.SESSION_START,
        "session_begin": EventTypeMapping.SESSION_START,
        "visit_start": EventTypeMapping.SESSION_START,

        "session_end": EventTypeMapping.SESSION_END,
        "visit_end": EventTypeMapping.SESSION_END,

        "page_visit": EventTypeMapping.PAGE_VIEW,
        "page_view": EventTypeMapping.PAGE_VIEW,
        "pageview": EventTypeMapping.PAGE_VIEW,

        "search": EventTypeMapping.SEARCH,
        "site_search": EventTypeMapping.SEARCH,

        "refund": EventTypeMapping.REFUND,
        "return": EventTypeMapping.REFUND,
    }

    return mappings.get(table_lower, EventTypeMapping.CUSTOM)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================


def create_bloomreach_config() -> ClientSchemaConfig:
    """Create configuration for Bloomreach EBQ data."""
    return ClientSchemaConfig(
        client_name="bloomreach",
        description="Bloomreach Engagement BigQuery export (EBQ)",
        customer_id_field="internal_customer_id",
        timestamp_field="timestamp",
        properties_field="properties",
        event_types=[
            EventTypeConfig(
                source_table="purchase",
                canonical_type=EventTypeMapping.PURCHASE,
                is_transactional=True,
                contributes_to_revenue=True,
                field_mappings=[
                    FieldMapping("properties.total_price", SemanticFieldType.ORDER_TOTAL, "decimal"),
                    FieldMapping("properties.purchase_id", SemanticFieldType.ORDER_ID),
                    FieldMapping("properties.total_quantity", SemanticFieldType.QUANTITY, "int"),
                ],
            ),
            EventTypeConfig(
                source_table="view_item",
                canonical_type=EventTypeMapping.VIEW_ITEM,
                is_engagement=True,
                field_mappings=[
                    FieldMapping("properties.product_id", SemanticFieldType.PRODUCT_ID),
                    FieldMapping("properties.title", SemanticFieldType.PRODUCT_NAME),
                    FieldMapping("properties.category_level_1", SemanticFieldType.PRODUCT_CATEGORY),
                    FieldMapping("properties.price", SemanticFieldType.PRODUCT_PRICE, "decimal"),
                ],
            ),
            EventTypeConfig(
                source_table="cart_update",
                canonical_type=EventTypeMapping.ADD_TO_CART,
                is_engagement=True,
            ),
            EventTypeConfig(
                source_table="session_start",
                canonical_type=EventTypeMapping.SESSION_START,
                is_engagement=True,
            ),
            EventTypeConfig(
                source_table="search",
                canonical_type=EventTypeMapping.SEARCH,
                is_engagement=True,
                field_mappings=[
                    FieldMapping("properties.query", SemanticFieldType.SEARCH_QUERY),
                ],
            ),
        ],
        id_history_table="customers_id_history",
        past_id_field="past_id",
        canonical_id_field="internal_customer_id",
        category_fields=["category_level_1", "category_level_2", "product_category"],
        revenue_fields=["total_price", "total_price_local_currency"],
    )


def create_ga4_config() -> ClientSchemaConfig:
    """Create configuration for Google Analytics 4 BigQuery export."""
    return ClientSchemaConfig(
        client_name="ga4",
        description="Google Analytics 4 BigQuery export",
        customer_id_field="user_pseudo_id",
        timestamp_field="event_timestamp",
        properties_field="event_params",
        event_types=[
            EventTypeConfig(
                source_table="events",
                source_type_value="purchase",
                canonical_type=EventTypeMapping.PURCHASE,
                is_transactional=True,
                contributes_to_revenue=True,
            ),
            EventTypeConfig(
                source_table="events",
                source_type_value="view_item",
                canonical_type=EventTypeMapping.VIEW_ITEM,
                is_engagement=True,
            ),
            EventTypeConfig(
                source_table="events",
                source_type_value="add_to_cart",
                canonical_type=EventTypeMapping.ADD_TO_CART,
                is_engagement=True,
            ),
            EventTypeConfig(
                source_table="events",
                source_type_value="session_start",
                canonical_type=EventTypeMapping.SESSION_START,
                is_engagement=True,
            ),
        ],
        mobile_device_values=["mobile", "tablet"],
        desktop_device_values=["desktop"],
        revenue_fields=["value", "transaction_revenue"],
    )


def create_segment_config() -> ClientSchemaConfig:
    """Create configuration for Segment data warehouse."""
    return ClientSchemaConfig(
        client_name="segment",
        description="Segment data warehouse (tracks/identifies)",
        customer_id_field="user_id",
        timestamp_field="timestamp",
        properties_field=None,  # Flat structure
        event_types=[
            EventTypeConfig(
                source_table="tracks",
                canonical_type=EventTypeMapping.CUSTOM,
            ),
            EventTypeConfig(
                source_table="order_completed",
                canonical_type=EventTypeMapping.PURCHASE,
                is_transactional=True,
                contributes_to_revenue=True,
            ),
            EventTypeConfig(
                source_table="product_viewed",
                canonical_type=EventTypeMapping.VIEW_ITEM,
                is_engagement=True,
            ),
            EventTypeConfig(
                source_table="product_added",
                canonical_type=EventTypeMapping.ADD_TO_CART,
                is_engagement=True,
            ),
        ],
        id_history_table="identifies",
        past_id_field="anonymous_id",
        canonical_id_field="user_id",
        revenue_fields=["total", "revenue", "value"],
    )


def create_generic_config(
    customer_id_field: str = "customer_id",
    timestamp_field: str = "timestamp",
) -> ClientSchemaConfig:
    """Create a generic configuration with minimal assumptions."""
    return ClientSchemaConfig(
        client_name="generic",
        description="Generic configuration - customize as needed",
        customer_id_field=customer_id_field,
        timestamp_field=timestamp_field,
        properties_field=None,
    )


# =============================================================================
# FIELD EXTRACTION HELPERS
# =============================================================================


def extract_field(
    data: dict[str, Any],
    field_path: str,
    default: Any = None,
) -> Any:
    """
    Extract a field value from nested data using dot notation.

    Args:
        data: Dictionary to extract from
        field_path: Dot-separated path (e.g., "properties.total_price")
        default: Default value if not found

    Returns:
        The extracted value or default
    """
    parts = field_path.split(".")
    current = data

    for part in parts:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return default

    if current is None:
        return default

    # Handle numpy/pandas types
    if hasattr(current, "item"):
        return current.item()

    return current


def extract_with_alternatives(
    data: dict[str, Any],
    field_names: list[str],
    default: Any = None,
) -> Any:
    """
    Try multiple field names and return the first non-None value.

    Args:
        data: Dictionary to extract from
        field_names: List of field names/paths to try
        default: Default value if none found

    Returns:
        The first non-None value found, or default
    """
    for field_name in field_names:
        value = extract_field(data, field_name)
        if value is not None:
            return value
    return default


def is_mobile_device(
    device_value: str | None,
    mobile_values: list[str],
) -> bool:
    """Check if a device value indicates a mobile device."""
    if not device_value:
        return False
    device_lower = str(device_value).lower()
    return any(mv.lower() in device_lower for mv in mobile_values)
