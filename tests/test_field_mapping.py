"""
Tests for flexible field mapping configuration.
"""

import pytest

from src.data.field_mapping import (
    ClientSchemaConfig,
    EventTypeConfig,
    EventTypeMapping,
    FieldMapping,
    SemanticFieldType,
    create_bloomreach_config,
    create_ga4_config,
    create_generic_config,
    create_segment_config,
    extract_field,
    extract_with_alternatives,
    is_mobile_device,
    _default_table_to_event_type,
)


# =============================================================================
# SEMANTIC FIELD TYPE TESTS
# =============================================================================


class TestSemanticFieldType:
    """Tests for SemanticFieldType enum."""

    def test_identity_fields_exist(self) -> None:
        """Test identity field types exist."""
        assert SemanticFieldType.CUSTOMER_ID == "customer_id"
        assert SemanticFieldType.EVENT_ID == "event_id"
        assert SemanticFieldType.SESSION_ID == "session_id"

    def test_temporal_fields_exist(self) -> None:
        """Test temporal field types exist."""
        assert SemanticFieldType.TIMESTAMP == "timestamp"
        assert SemanticFieldType.DATE == "date"

    def test_product_fields_exist(self) -> None:
        """Test product field types exist."""
        assert SemanticFieldType.PRODUCT_ID == "product_id"
        assert SemanticFieldType.PRODUCT_NAME == "product_name"
        assert SemanticFieldType.PRODUCT_CATEGORY == "product_category"
        assert SemanticFieldType.PRODUCT_PRICE == "product_price"

    def test_transaction_fields_exist(self) -> None:
        """Test transaction field types exist."""
        assert SemanticFieldType.ORDER_ID == "order_id"
        assert SemanticFieldType.ORDER_TOTAL == "order_total"
        assert SemanticFieldType.QUANTITY == "quantity"

    def test_customer_fields_exist(self) -> None:
        """Test customer field types exist."""
        assert SemanticFieldType.EMAIL == "email"
        assert SemanticFieldType.FIRST_NAME == "first_name"
        assert SemanticFieldType.PHONE == "phone"


# =============================================================================
# EVENT TYPE MAPPING TESTS
# =============================================================================


class TestEventTypeMapping:
    """Tests for EventTypeMapping enum."""

    def test_purchase_types_exist(self) -> None:
        """Test purchase event types exist."""
        assert EventTypeMapping.PURCHASE == "purchase"
        assert EventTypeMapping.PURCHASE_ITEM == "purchase_item"
        assert EventTypeMapping.REFUND == "refund"

    def test_engagement_types_exist(self) -> None:
        """Test engagement event types exist."""
        assert EventTypeMapping.VIEW_ITEM == "view_item"
        assert EventTypeMapping.VIEW_CATEGORY == "view_category"
        assert EventTypeMapping.ADD_TO_CART == "add_to_cart"
        assert EventTypeMapping.CHECKOUT == "checkout"

    def test_session_types_exist(self) -> None:
        """Test session event types exist."""
        assert EventTypeMapping.SESSION_START == "session_start"
        assert EventTypeMapping.SESSION_END == "session_end"
        assert EventTypeMapping.PAGE_VIEW == "page_view"

    def test_custom_type_exists(self) -> None:
        """Test custom event type exists."""
        assert EventTypeMapping.CUSTOM == "custom"


# =============================================================================
# FIELD MAPPING TESTS
# =============================================================================


class TestFieldMapping:
    """Tests for FieldMapping dataclass."""

    def test_basic_field_mapping(self) -> None:
        """Test creating a basic field mapping."""
        mapping = FieldMapping(
            source_field="properties.total_price",
            semantic_type=SemanticFieldType.ORDER_TOTAL,
        )

        assert mapping.source_field == "properties.total_price"
        assert mapping.semantic_type == SemanticFieldType.ORDER_TOTAL
        assert mapping.transform is None
        assert mapping.default is None
        assert mapping.alternatives == []

    def test_field_mapping_with_transform(self) -> None:
        """Test field mapping with transform."""
        mapping = FieldMapping(
            source_field="price",
            semantic_type=SemanticFieldType.PRODUCT_PRICE,
            transform="decimal",
        )

        assert mapping.transform == "decimal"

    def test_field_mapping_with_alternatives(self) -> None:
        """Test field mapping with alternative field names."""
        mapping = FieldMapping(
            source_field="category",
            semantic_type=SemanticFieldType.PRODUCT_CATEGORY,
            alternatives=["product_category", "item_category"],
        )

        assert mapping.alternatives == ["product_category", "item_category"]

    def test_get_source_fields(self) -> None:
        """Test getting all source field names."""
        mapping = FieldMapping(
            source_field="category",
            semantic_type=SemanticFieldType.PRODUCT_CATEGORY,
            alternatives=["product_category", "item_category"],
        )

        fields = mapping.get_source_fields()
        assert fields == ["category", "product_category", "item_category"]

    def test_get_source_fields_no_alternatives(self) -> None:
        """Test getting source fields with no alternatives."""
        mapping = FieldMapping(
            source_field="price",
            semantic_type=SemanticFieldType.PRODUCT_PRICE,
        )

        fields = mapping.get_source_fields()
        assert fields == ["price"]


# =============================================================================
# EVENT TYPE CONFIG TESTS
# =============================================================================


class TestEventTypeConfig:
    """Tests for EventTypeConfig dataclass."""

    def test_basic_event_type_config(self) -> None:
        """Test creating basic event type config."""
        config = EventTypeConfig(
            source_table="purchase",
            canonical_type=EventTypeMapping.PURCHASE,
        )

        assert config.source_table == "purchase"
        assert config.canonical_type == EventTypeMapping.PURCHASE
        assert config.source_type_value is None
        assert config.is_transactional is False
        assert config.is_engagement is True
        assert config.contributes_to_revenue is False
        assert config.field_mappings == []

    def test_transactional_event_config(self) -> None:
        """Test transactional event configuration."""
        config = EventTypeConfig(
            source_table="orders",
            canonical_type=EventTypeMapping.PURCHASE,
            is_transactional=True,
            contributes_to_revenue=True,
        )

        assert config.is_transactional is True
        assert config.contributes_to_revenue is True

    def test_event_config_with_field_mappings(self) -> None:
        """Test event config with field mappings."""
        field_mappings = [
            FieldMapping("total", SemanticFieldType.ORDER_TOTAL),
            FieldMapping("order_id", SemanticFieldType.ORDER_ID),
        ]

        config = EventTypeConfig(
            source_table="purchase",
            canonical_type=EventTypeMapping.PURCHASE,
            field_mappings=field_mappings,
        )

        assert len(config.field_mappings) == 2
        assert config.field_mappings[0].source_field == "total"


# =============================================================================
# CLIENT SCHEMA CONFIG TESTS
# =============================================================================


class TestClientSchemaConfig:
    """Tests for ClientSchemaConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ClientSchemaConfig(client_name="test")

        assert config.client_name == "test"
        assert config.customer_id_field == "internal_customer_id"
        assert config.timestamp_field == "timestamp"
        assert config.properties_field == "properties"
        assert config.id_history_table == "customers_id_history"
        assert config.past_id_field == "past_id"
        assert config.canonical_id_field == "internal_customer_id"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ClientSchemaConfig(
            client_name="my_company",
            customer_id_field="user_id",
            timestamp_field="event_time",
            properties_field=None,
        )

        assert config.customer_id_field == "user_id"
        assert config.timestamp_field == "event_time"
        assert config.properties_field is None

    def test_default_mobile_device_values(self) -> None:
        """Test default mobile device values."""
        config = ClientSchemaConfig(client_name="test")

        assert "mobile" in config.mobile_device_values
        assert "tablet" in config.mobile_device_values
        assert "ios" in config.mobile_device_values
        assert "android" in config.mobile_device_values

    def test_default_category_fields(self) -> None:
        """Test default category field alternatives."""
        config = ClientSchemaConfig(client_name="test")

        assert "product_category" in config.category_fields
        assert "category" in config.category_fields
        assert "category_level_1" in config.category_fields

    def test_default_revenue_fields(self) -> None:
        """Test default revenue field alternatives."""
        config = ClientSchemaConfig(client_name="test")

        assert "total_price" in config.revenue_fields
        assert "order_total" in config.revenue_fields
        assert "revenue" in config.revenue_fields

    def test_get_event_type_config_found(self) -> None:
        """Test getting event type config when it exists."""
        event_config = EventTypeConfig(
            source_table="orders",
            canonical_type=EventTypeMapping.PURCHASE,
        )
        config = ClientSchemaConfig(
            client_name="test",
            event_types=[event_config],
        )

        result = config.get_event_type_config("orders")
        assert result is not None
        assert result.source_table == "orders"

    def test_get_event_type_config_not_found(self) -> None:
        """Test getting event type config when it doesn't exist."""
        config = ClientSchemaConfig(client_name="test")

        result = config.get_event_type_config("unknown_table")
        assert result is None

    def test_get_canonical_event_type_from_config(self) -> None:
        """Test getting canonical event type from config."""
        event_config = EventTypeConfig(
            source_table="orders",
            canonical_type=EventTypeMapping.PURCHASE,
        )
        config = ClientSchemaConfig(
            client_name="test",
            event_types=[event_config],
        )

        result = config.get_canonical_event_type("orders")
        assert result == EventTypeMapping.PURCHASE

    def test_get_canonical_event_type_default(self) -> None:
        """Test getting canonical event type uses default mapping."""
        config = ClientSchemaConfig(client_name="test")

        # Should use default mapping
        result = config.get_canonical_event_type("purchase")
        assert result == EventTypeMapping.PURCHASE


# =============================================================================
# DEFAULT TABLE TO EVENT TYPE MAPPING TESTS
# =============================================================================


class TestDefaultTableToEventType:
    """Tests for _default_table_to_event_type function."""

    def test_purchase_tables(self) -> None:
        """Test purchase table mappings."""
        assert _default_table_to_event_type("purchase") == EventTypeMapping.PURCHASE
        assert _default_table_to_event_type("purchases") == EventTypeMapping.PURCHASE
        assert _default_table_to_event_type("order") == EventTypeMapping.PURCHASE
        assert _default_table_to_event_type("orders") == EventTypeMapping.PURCHASE
        assert _default_table_to_event_type("transaction") == EventTypeMapping.PURCHASE

    def test_purchase_item_tables(self) -> None:
        """Test purchase item table mappings."""
        assert _default_table_to_event_type("purchase_item") == EventTypeMapping.PURCHASE_ITEM
        assert _default_table_to_event_type("order_item") == EventTypeMapping.PURCHASE_ITEM
        assert _default_table_to_event_type("line_item") == EventTypeMapping.PURCHASE_ITEM

    def test_view_item_tables(self) -> None:
        """Test view item table mappings."""
        assert _default_table_to_event_type("view_item") == EventTypeMapping.VIEW_ITEM
        assert _default_table_to_event_type("product_view") == EventTypeMapping.VIEW_ITEM
        assert _default_table_to_event_type("pdp_view") == EventTypeMapping.VIEW_ITEM

    def test_cart_tables(self) -> None:
        """Test cart table mappings."""
        assert _default_table_to_event_type("cart_update") == EventTypeMapping.ADD_TO_CART
        assert _default_table_to_event_type("add_to_cart") == EventTypeMapping.ADD_TO_CART
        assert _default_table_to_event_type("cart") == EventTypeMapping.ADD_TO_CART

    def test_session_tables(self) -> None:
        """Test session table mappings."""
        assert _default_table_to_event_type("session_start") == EventTypeMapping.SESSION_START
        assert _default_table_to_event_type("session_end") == EventTypeMapping.SESSION_END

    def test_page_view_tables(self) -> None:
        """Test page view table mappings."""
        assert _default_table_to_event_type("page_visit") == EventTypeMapping.PAGE_VIEW
        assert _default_table_to_event_type("page_view") == EventTypeMapping.PAGE_VIEW
        assert _default_table_to_event_type("pageview") == EventTypeMapping.PAGE_VIEW

    def test_search_tables(self) -> None:
        """Test search table mappings."""
        assert _default_table_to_event_type("search") == EventTypeMapping.SEARCH
        assert _default_table_to_event_type("site_search") == EventTypeMapping.SEARCH

    def test_refund_tables(self) -> None:
        """Test refund table mappings."""
        assert _default_table_to_event_type("refund") == EventTypeMapping.REFUND
        assert _default_table_to_event_type("return") == EventTypeMapping.REFUND

    def test_unknown_table(self) -> None:
        """Test unknown table returns CUSTOM."""
        assert _default_table_to_event_type("unknown_table") == EventTypeMapping.CUSTOM
        assert _default_table_to_event_type("my_custom_events") == EventTypeMapping.CUSTOM

    def test_case_insensitive(self) -> None:
        """Test mapping is case insensitive."""
        assert _default_table_to_event_type("PURCHASE") == EventTypeMapping.PURCHASE
        assert _default_table_to_event_type("Purchase") == EventTypeMapping.PURCHASE
        assert _default_table_to_event_type("VIEW_ITEM") == EventTypeMapping.VIEW_ITEM


# =============================================================================
# PRESET CONFIGURATION TESTS
# =============================================================================


class TestPresetConfigurations:
    """Tests for preset configuration functions."""

    def test_bloomreach_config(self) -> None:
        """Test Bloomreach configuration."""
        config = create_bloomreach_config()

        assert config.client_name == "bloomreach"
        assert config.customer_id_field == "internal_customer_id"
        assert config.timestamp_field == "timestamp"
        assert config.properties_field == "properties"
        assert config.id_history_table == "customers_id_history"
        assert "category_level_1" in config.category_fields
        assert "total_price" in config.revenue_fields
        assert len(config.event_types) > 0

    def test_bloomreach_event_types(self) -> None:
        """Test Bloomreach event type configurations."""
        config = create_bloomreach_config()

        # Check purchase event
        purchase_config = config.get_event_type_config("purchase")
        assert purchase_config is not None
        assert purchase_config.is_transactional is True
        assert purchase_config.contributes_to_revenue is True

        # Check view_item event
        view_config = config.get_event_type_config("view_item")
        assert view_config is not None
        assert view_config.is_engagement is True

    def test_ga4_config(self) -> None:
        """Test Google Analytics 4 configuration."""
        config = create_ga4_config()

        assert config.client_name == "ga4"
        assert config.customer_id_field == "user_pseudo_id"
        assert config.timestamp_field == "event_timestamp"
        assert config.properties_field == "event_params"
        assert "mobile" in config.mobile_device_values
        assert "value" in config.revenue_fields

    def test_ga4_event_types(self) -> None:
        """Test GA4 event type configurations."""
        config = create_ga4_config()

        # GA4 uses source_type_value for event filtering
        event_types = config.event_types
        assert len(event_types) > 0

        purchase_events = [e for e in event_types if e.canonical_type == EventTypeMapping.PURCHASE]
        assert len(purchase_events) > 0

    def test_segment_config(self) -> None:
        """Test Segment configuration."""
        config = create_segment_config()

        assert config.client_name == "segment"
        assert config.customer_id_field == "user_id"
        assert config.timestamp_field == "timestamp"
        assert config.properties_field is None  # Flat structure
        assert config.id_history_table == "identifies"
        assert config.past_id_field == "anonymous_id"
        assert "total" in config.revenue_fields

    def test_segment_event_types(self) -> None:
        """Test Segment event type configurations."""
        config = create_segment_config()

        order_completed = config.get_event_type_config("order_completed")
        assert order_completed is not None
        assert order_completed.canonical_type == EventTypeMapping.PURCHASE

    def test_generic_config(self) -> None:
        """Test generic configuration."""
        config = create_generic_config()

        assert config.client_name == "generic"
        assert config.customer_id_field == "customer_id"
        assert config.timestamp_field == "timestamp"
        assert config.properties_field is None

    def test_generic_config_custom_fields(self) -> None:
        """Test generic configuration with custom fields."""
        config = create_generic_config(
            customer_id_field="user_id",
            timestamp_field="event_time",
        )

        assert config.customer_id_field == "user_id"
        assert config.timestamp_field == "event_time"


# =============================================================================
# FIELD EXTRACTION HELPER TESTS
# =============================================================================


class TestExtractField:
    """Tests for extract_field function."""

    def test_extract_simple_field(self) -> None:
        """Test extracting simple field."""
        data = {"name": "John", "age": 30}

        assert extract_field(data, "name") == "John"
        assert extract_field(data, "age") == 30

    def test_extract_nested_field(self) -> None:
        """Test extracting nested field with dot notation."""
        data = {
            "properties": {
                "total_price": 99.99,
                "product": {
                    "name": "Widget",
                },
            },
        }

        assert extract_field(data, "properties.total_price") == 99.99
        assert extract_field(data, "properties.product.name") == "Widget"

    def test_extract_missing_field(self) -> None:
        """Test extracting missing field returns default."""
        data = {"name": "John"}

        assert extract_field(data, "missing") is None
        assert extract_field(data, "missing", default="N/A") == "N/A"

    def test_extract_missing_nested_field(self) -> None:
        """Test extracting missing nested field."""
        data = {"properties": {"name": "test"}}

        assert extract_field(data, "properties.missing") is None
        assert extract_field(data, "missing.nested") is None

    def test_extract_from_none(self) -> None:
        """Test extracting from None in path."""
        data = {"properties": None}

        assert extract_field(data, "properties.field") is None

    def test_extract_handles_numpy_types(self) -> None:
        """Test extraction handles numpy-like types with .item() method."""
        class NumpyLike:
            def item(self):
                return 42

        data = {"value": NumpyLike()}

        assert extract_field(data, "value") == 42

    def test_extract_non_dict_intermediate(self) -> None:
        """Test extracting through a non-dict intermediate returns default.

        When traversing a path like 'a.b.c', if 'a' is not a dict (e.g., a string
        or list), should return the default value.
        """
        # Case: intermediate is a string
        data = {"name": "John"}
        assert extract_field(data, "name.first") is None

        # Case: intermediate is a list
        data = {"items": [1, 2, 3]}
        assert extract_field(data, "items.length") is None

        # Case: intermediate is an integer
        data = {"count": 42}
        assert extract_field(data, "count.value") is None

        # Custom default
        assert extract_field(data, "count.value", default="N/A") == "N/A"


class TestExtractWithAlternatives:
    """Tests for extract_with_alternatives function."""

    def test_extract_first_match(self) -> None:
        """Test extracting first matching field."""
        data = {"category_level_1": "Clothing", "category": "Apparel"}

        result = extract_with_alternatives(
            data, ["category_level_1", "category", "product_type"]
        )
        assert result == "Clothing"

    def test_extract_second_match(self) -> None:
        """Test extracting second field when first is missing."""
        data = {"category": "Apparel"}

        result = extract_with_alternatives(
            data, ["category_level_1", "category", "product_type"]
        )
        assert result == "Apparel"

    def test_extract_none_when_no_match(self) -> None:
        """Test returning None when no field matches."""
        data = {"other_field": "value"}

        result = extract_with_alternatives(
            data, ["category_level_1", "category"]
        )
        assert result is None

    def test_extract_with_default(self) -> None:
        """Test returning default when no field matches."""
        data = {"other_field": "value"}

        result = extract_with_alternatives(
            data, ["missing1", "missing2"], default="Unknown"
        )
        assert result == "Unknown"

    def test_extract_skips_none_values(self) -> None:
        """Test extraction skips None values and finds next."""
        data = {"category_level_1": None, "category": "Apparel"}

        result = extract_with_alternatives(
            data, ["category_level_1", "category"]
        )
        assert result == "Apparel"


class TestIsMobileDevice:
    """Tests for is_mobile_device function."""

    def test_mobile_device(self) -> None:
        """Test mobile device detection."""
        mobile_values = ["mobile", "tablet", "ios", "android"]

        assert is_mobile_device("mobile", mobile_values) is True
        assert is_mobile_device("Mobile", mobile_values) is True
        assert is_mobile_device("MOBILE", mobile_values) is True

    def test_tablet_device(self) -> None:
        """Test tablet detection."""
        mobile_values = ["mobile", "tablet", "ios", "android"]

        assert is_mobile_device("tablet", mobile_values) is True
        assert is_mobile_device("Tablet", mobile_values) is True

    def test_ios_device(self) -> None:
        """Test iOS detection."""
        mobile_values = ["mobile", "tablet", "ios", "android"]

        assert is_mobile_device("iOS", mobile_values) is True
        assert is_mobile_device("ios", mobile_values) is True

    def test_android_device(self) -> None:
        """Test Android detection."""
        mobile_values = ["mobile", "tablet", "ios", "android"]

        assert is_mobile_device("Android", mobile_values) is True
        assert is_mobile_device("android", mobile_values) is True

    def test_desktop_device(self) -> None:
        """Test desktop is not mobile."""
        mobile_values = ["mobile", "tablet", "ios", "android"]

        assert is_mobile_device("desktop", mobile_values) is False
        assert is_mobile_device("web", mobile_values) is False
        assert is_mobile_device("Desktop", mobile_values) is False

    def test_none_device(self) -> None:
        """Test None device returns False."""
        mobile_values = ["mobile", "tablet"]

        assert is_mobile_device(None, mobile_values) is False

    def test_empty_string_device(self) -> None:
        """Test empty string returns False."""
        mobile_values = ["mobile", "tablet"]

        assert is_mobile_device("", mobile_values) is False

    def test_partial_match(self) -> None:
        """Test partial match works."""
        mobile_values = ["mobile"]

        # "mobile" is contained in "mobile_app"
        assert is_mobile_device("mobile_app", mobile_values) is True
