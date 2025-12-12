"""
Tests for src/data/bigquery/adapter.py

Comprehensive test suite for BigQuery adapter functionality.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.data.bigquery.adapter import (
    BQ_TO_INTERNAL_EVENT_TYPE,
    LoadResult,
    transform_value,
    get_nested_value,
    BigQueryAdapter,
    load_from_bigquery,
    create_config_from_tables,
)
from src.data.bigquery.config import (
    BigQueryConfig,
    EventTableConfig,
    CustomerTableConfig,
    MergeTableConfig,
    FieldMapping,
    EventType,
)
from src.data.schemas import EventType as InternalEventType


# =============================================================================
# TESTS: Event Type Mapping
# =============================================================================


class TestEventTypeMapping:
    """Tests for BQ_TO_INTERNAL_EVENT_TYPE mapping."""

    def test_purchase_mapping(self) -> None:
        """PURCHASE should map to internal PURCHASE."""
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.PURCHASE] == InternalEventType.PURCHASE

    def test_page_view_mapping(self) -> None:
        """PAGE_VIEW should map to VIEW_ITEM."""
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.PAGE_VIEW] == InternalEventType.VIEW_ITEM

    def test_view_item_mapping(self) -> None:
        """VIEW_ITEM should map to VIEW_ITEM."""
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.VIEW_ITEM] == InternalEventType.VIEW_ITEM

    def test_cart_mappings(self) -> None:
        """Cart events should map to ADD_TO_CART."""
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.ADD_TO_CART] == InternalEventType.ADD_TO_CART
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.REMOVE_FROM_CART] == InternalEventType.ADD_TO_CART

    def test_checkout_mapping(self) -> None:
        """BEGIN_CHECKOUT should map to CHECKOUT."""
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.BEGIN_CHECKOUT] == InternalEventType.CHECKOUT

    def test_session_mappings(self) -> None:
        """Session events should map to SESSION_START."""
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.SESSION_START] == InternalEventType.SESSION_START
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.SESSION_END] == InternalEventType.SESSION_START

    def test_engagement_mappings(self) -> None:
        """Engagement events should map appropriately."""
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.WISHLIST_ADD] == InternalEventType.VIEW_ITEM
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.SEARCH] == InternalEventType.VIEW_CATEGORY
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.EMAIL_OPEN] == InternalEventType.SESSION_START
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.EMAIL_CLICK] == InternalEventType.VIEW_ITEM

    def test_custom_mapping(self) -> None:
        """CUSTOM should map to VIEW_ITEM."""
        assert BQ_TO_INTERNAL_EVENT_TYPE[EventType.CUSTOM] == InternalEventType.VIEW_ITEM

    def test_all_event_types_mapped(self) -> None:
        """All EventType values should be mapped."""
        for event_type in EventType:
            assert event_type in BQ_TO_INTERNAL_EVENT_TYPE


# =============================================================================
# TESTS: LoadResult
# =============================================================================


class TestLoadResult:
    """Tests for LoadResult dataclass."""

    def test_default_values(self) -> None:
        """LoadResult should have sensible defaults."""
        result = LoadResult()

        assert result.events == []
        assert result.id_history == []
        assert result.customer_properties == {}
        assert result.tables_loaded == 0
        assert result.total_rows == 0
        assert result.events_by_type == {}
        assert result.unique_customers == 0
        assert result.load_duration_ms == 0.0
        assert result.errors == []

    def test_custom_values(self) -> None:
        """LoadResult should accept custom values."""
        result = LoadResult(
            tables_loaded=5,
            total_rows=10000,
            unique_customers=500,
            errors=["Error 1"],
        )

        assert result.tables_loaded == 5
        assert result.total_rows == 10000
        assert result.unique_customers == 500
        assert len(result.errors) == 1


# =============================================================================
# TESTS: transform_value
# =============================================================================


class TestTransformValue:
    """Tests for transform_value function."""

    def test_none_value(self) -> None:
        """None value should return None regardless of transform."""
        assert transform_value(None, "decimal") is None
        assert transform_value(None, "int") is None
        assert transform_value(None, None) is None

    def test_no_transform(self) -> None:
        """No transform should return value as-is."""
        assert transform_value("hello", None) == "hello"
        assert transform_value(42, None) == 42
        assert transform_value({"key": "value"}, None) == {"key": "value"}

    def test_decimal_transform(self) -> None:
        """Decimal transform should convert to Decimal."""
        assert transform_value("99.99", "decimal") == Decimal("99.99")
        assert transform_value(100, "decimal") == Decimal("100")
        assert transform_value(49.95, "decimal") == Decimal("49.95")

    def test_decimal_transform_invalid(self) -> None:
        """Invalid decimal should return None."""
        assert transform_value("invalid", "decimal") is None
        assert transform_value([], "decimal") is None

    def test_int_transform(self) -> None:
        """Int transform should convert to int."""
        assert transform_value("42", "int") == 42
        assert transform_value(42.9, "int") == 42
        assert transform_value(100, "int") == 100

    def test_int_transform_invalid(self) -> None:
        """Invalid int should return None."""
        assert transform_value("invalid", "int") is None
        assert transform_value({}, "int") is None

    def test_float_transform(self) -> None:
        """Float transform should convert to float."""
        assert transform_value("3.14", "float") == 3.14
        assert transform_value(42, "float") == 42.0

    def test_float_transform_invalid(self) -> None:
        """Invalid float should return None."""
        assert transform_value("invalid", "float") is None

    def test_datetime_transform_datetime_input(self) -> None:
        """Datetime transform with datetime input should return as-is."""
        dt = datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc)
        assert transform_value(dt, "datetime") == dt

    def test_datetime_transform_string(self) -> None:
        """Datetime transform should parse ISO string."""
        result = transform_value("2024-01-15T10:30:00Z", "datetime")
        assert isinstance(result, datetime)
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_datetime_transform_invalid(self) -> None:
        """Invalid datetime should return None."""
        assert transform_value("invalid", "datetime") is None

    def test_json_parse_dict_input(self) -> None:
        """JSON parse with dict input should return as-is."""
        data = {"key": "value"}
        assert transform_value(data, "json_parse") == data

    def test_json_parse_list_input(self) -> None:
        """JSON parse with list input should return as-is."""
        data = [1, 2, 3]
        assert transform_value(data, "json_parse") == data

    def test_json_parse_string(self) -> None:
        """JSON parse should parse JSON string."""
        result = transform_value('{"key": "value"}', "json_parse")
        assert result == {"key": "value"}

    def test_json_parse_invalid(self) -> None:
        """Invalid JSON should return original value."""
        result = transform_value("not json", "json_parse")
        assert result == "not json"

    def test_bool_transform_bool_input(self) -> None:
        """Bool transform with bool input should return as-is."""
        assert transform_value(True, "bool") is True
        assert transform_value(False, "bool") is False

    def test_bool_transform_string(self) -> None:
        """Bool transform should parse string values."""
        assert transform_value("true", "bool") is True
        assert transform_value("True", "bool") is True
        assert transform_value("1", "bool") is True
        assert transform_value("yes", "bool") is True

        assert transform_value("false", "bool") is False
        assert transform_value("0", "bool") is False
        assert transform_value("no", "bool") is False

    def test_unknown_transform(self) -> None:
        """Unknown transform should return value as-is."""
        assert transform_value("test", "unknown") == "test"


# =============================================================================
# TESTS: get_nested_value
# =============================================================================


class TestGetNestedValue:
    """Tests for get_nested_value function."""

    def test_simple_field(self) -> None:
        """Get simple top-level field."""
        row = {"name": "John", "age": 30}

        assert get_nested_value(row, "name") == "John"
        assert get_nested_value(row, "age") == 30

    def test_nested_field(self) -> None:
        """Get nested field with dot notation."""
        row = {
            "properties": {
                "total_price": 99.99,
                "product": {
                    "name": "Widget",
                },
            },
        }

        assert get_nested_value(row, "properties.total_price") == 99.99
        assert get_nested_value(row, "properties.product.name") == "Widget"

    def test_missing_field(self) -> None:
        """Missing field should return None."""
        row = {"name": "John"}

        assert get_nested_value(row, "missing") is None
        assert get_nested_value(row, "properties.missing") is None

    def test_none_intermediate(self) -> None:
        """None intermediate value should return None."""
        row = {"properties": None}

        assert get_nested_value(row, "properties.field") is None

    def test_non_dict_intermediate(self) -> None:
        """Non-dict intermediate should return None."""
        row = {"name": "John"}

        assert get_nested_value(row, "name.first") is None

    def test_empty_path(self) -> None:
        """Empty path parts should work."""
        row = {"field": "value"}

        assert get_nested_value(row, "field") == "value"


# =============================================================================
# TESTS: BigQueryAdapter
# =============================================================================


class TestBigQueryAdapter:
    """Tests for BigQueryAdapter class."""

    @pytest.fixture
    def basic_config(self) -> BigQueryConfig:
        """Create a basic BigQuery config for testing."""
        return BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            event_tables=[
                EventTableConfig(
                    table_name="purchases",
                    event_type=EventType.PURCHASE,
                ),
            ],
        )

    def test_init(self, basic_config: BigQueryConfig) -> None:
        """Test adapter initialization."""
        adapter = BigQueryAdapter(basic_config)

        assert adapter.config == basic_config
        assert adapter._client is None
        assert adapter._bigquery_module is None

    def test_get_bigquery_import_error(self, basic_config: BigQueryConfig) -> None:
        """Test that ImportError is raised when BigQuery not installed."""
        adapter = BigQueryAdapter(basic_config)

        with patch.dict("sys.modules", {"google.cloud": None, "google.cloud.bigquery": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="google-cloud-bigquery is required"):
                    adapter._get_bigquery()

    def test_client_property_creates_client(self, basic_config: BigQueryConfig) -> None:
        """Test client property creates BigQuery client."""
        adapter = BigQueryAdapter(basic_config)

        mock_bigquery = MagicMock()
        mock_client = MagicMock()
        mock_bigquery.Client.return_value = mock_client

        with patch.object(adapter, "_get_bigquery", return_value=mock_bigquery):
            client = adapter.client

            mock_bigquery.Client.assert_called_once_with(project="test-project")
            assert client == mock_client

    def test_client_property_caches_client(self, basic_config: BigQueryConfig) -> None:
        """Test client property caches the client."""
        adapter = BigQueryAdapter(basic_config)

        mock_bigquery = MagicMock()
        mock_client = MagicMock()
        mock_bigquery.Client.return_value = mock_client

        with patch.object(adapter, "_get_bigquery", return_value=mock_bigquery):
            client1 = adapter.client
            client2 = adapter.client

            # Should only create client once
            assert mock_bigquery.Client.call_count == 1
            assert client1 is client2

    def test_build_event_query_basic(self, basic_config: BigQueryConfig) -> None:
        """Test basic event query building."""
        adapter = BigQueryAdapter(basic_config)
        table_config = basic_config.event_tables[0]

        query = adapter._build_event_query(
            table_config,
            "test-project.test_dataset.purchases",
        )

        assert "SELECT * FROM `test-project.test_dataset.purchases`" in query
        assert "ORDER BY timestamp" in query

    def test_build_event_query_with_date_range(self) -> None:
        """Test query building with date range."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            start_date="2024-01-01",
            end_date="2024-12-31",
            event_tables=[
                EventTableConfig(table_name="events", event_type=EventType.PURCHASE),
            ],
        )
        adapter = BigQueryAdapter(config)

        query = adapter._build_event_query(
            config.event_tables[0],
            "project.dataset.events",
        )

        assert "timestamp >= '2024-01-01'" in query
        assert "timestamp <= '2024-12-31'" in query

    def test_build_event_query_with_sampling(self) -> None:
        """Test query building with sampling."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            sample_rate=0.1,
            event_tables=[
                EventTableConfig(table_name="events", event_type=EventType.PURCHASE),
            ],
        )
        adapter = BigQueryAdapter(config)

        query = adapter._build_event_query(
            config.event_tables[0],
            "project.dataset.events",
        )

        assert "RAND() < 0.1" in query

    def test_build_event_query_with_limit(self) -> None:
        """Test query building with limit."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            limit_per_table=1000,
            event_tables=[
                EventTableConfig(table_name="events", event_type=EventType.PURCHASE),
            ],
        )
        adapter = BigQueryAdapter(config)

        query = adapter._build_event_query(
            config.event_tables[0],
            "project.dataset.events",
        )

        assert "LIMIT 1000" in query

    def test_build_event_query_with_filters(self) -> None:
        """Test query building with custom filters."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            event_tables=[
                EventTableConfig(
                    table_name="events",
                    event_type=EventType.PURCHASE,
                    filters=["status = 'completed'", "amount > 0"],
                ),
            ],
        )
        adapter = BigQueryAdapter(config)

        query = adapter._build_event_query(
            config.event_tables[0],
            "project.dataset.events",
        )

        assert "status = 'completed'" in query
        assert "amount > 0" in query

    def test_row_to_event_basic(self, basic_config: BigQueryConfig) -> None:
        """Test converting a row to EventRecord."""
        adapter = BigQueryAdapter(basic_config)
        table_config = basic_config.event_tables[0]

        row = {
            "internal_customer_id": "cust_001",
            "timestamp": datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
        }

        event = adapter._row_to_event(row, table_config, InternalEventType.PURCHASE)

        assert event is not None
        assert event.internal_customer_id == "cust_001"
        assert event.event_type == InternalEventType.PURCHASE
        assert event.timestamp.year == 2024

    def test_row_to_event_missing_customer_id(self, basic_config: BigQueryConfig) -> None:
        """Test row without customer ID returns None."""
        adapter = BigQueryAdapter(basic_config)
        table_config = basic_config.event_tables[0]

        row = {
            "internal_customer_id": None,
            "timestamp": datetime.now(tz=timezone.utc),
        }

        event = adapter._row_to_event(row, table_config, InternalEventType.PURCHASE)
        assert event is None

    def test_row_to_event_missing_timestamp(self, basic_config: BigQueryConfig) -> None:
        """Test row without timestamp returns None."""
        adapter = BigQueryAdapter(basic_config)
        table_config = basic_config.event_tables[0]

        row = {
            "internal_customer_id": "cust_001",
            "timestamp": None,
        }

        event = adapter._row_to_event(row, table_config, InternalEventType.PURCHASE)
        assert event is None

    def test_row_to_event_string_timestamp(self, basic_config: BigQueryConfig) -> None:
        """Test row with string timestamp."""
        adapter = BigQueryAdapter(basic_config)
        table_config = basic_config.event_tables[0]

        row = {
            "internal_customer_id": "cust_001",
            "timestamp": "2024-01-15T10:30:00Z",
        }

        event = adapter._row_to_event(row, table_config, InternalEventType.PURCHASE)

        assert event is not None
        assert event.timestamp.year == 2024

    def test_row_to_event_with_property_mappings(self) -> None:
        """Test row conversion with property mappings."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            event_tables=[
                EventTableConfig(
                    table_name="purchases",
                    event_type=EventType.PURCHASE,
                    property_mappings={
                        "order_total": FieldMapping(
                            "properties.total_price",
                            "order_total",
                            transform="decimal",
                        ),
                        "order_id": FieldMapping(
                            "properties.purchase_id",
                            "order_id",
                        ),
                    },
                ),
            ],
        )
        adapter = BigQueryAdapter(config)

        row = {
            "internal_customer_id": "cust_001",
            "timestamp": datetime.now(tz=timezone.utc),
            "properties": {
                "total_price": "99.99",
                "purchase_id": "order_123",
            },
        }

        event = adapter._row_to_event(
            row,
            config.event_tables[0],
            InternalEventType.PURCHASE,
        )

        assert event is not None
        assert event.properties.order_total == Decimal("99.99")
        assert event.properties.order_id == "order_123"

    def test_row_to_event_with_default_value(self) -> None:
        """Test row conversion uses default values."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            event_tables=[
                EventTableConfig(
                    table_name="cart",
                    event_type=EventType.ADD_TO_CART,
                    property_mappings={
                        "quantity": FieldMapping(
                            "properties.qty",
                            "quantity",
                            transform="int",
                            default=1,
                        ),
                    },
                ),
            ],
        )
        adapter = BigQueryAdapter(config)

        row = {
            "internal_customer_id": "cust_001",
            "timestamp": datetime.now(tz=timezone.utc),
            "properties": {},  # No quantity
        }

        event = adapter._row_to_event(
            row,
            config.event_tables[0],
            InternalEventType.ADD_TO_CART,
        )

        assert event is not None
        assert event.properties.quantity == 1

    def test_row_to_event_with_product_category_field(self) -> None:
        """Test row conversion extracts product category."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            event_tables=[
                EventTableConfig(
                    table_name="views",
                    event_type=EventType.VIEW_ITEM,
                    product_category_field="properties.category",
                ),
            ],
        )
        adapter = BigQueryAdapter(config)

        row = {
            "internal_customer_id": "cust_001",
            "timestamp": datetime.now(tz=timezone.utc),
            "properties": {
                "category": "Electronics",
            },
        }

        event = adapter._row_to_event(
            row,
            config.event_tables[0],
            InternalEventType.VIEW_ITEM,
        )

        assert event is not None
        assert event.properties.product_category == "Electronics"

    def test_row_to_event_extracts_categories_from_product_list(self) -> None:
        """Test row conversion extracts categories from product_list."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            event_tables=[
                EventTableConfig(
                    table_name="purchases",
                    event_type=EventType.PURCHASE,
                    property_mappings={
                        "product_list": FieldMapping(
                            "properties.products",
                            "product_list",
                        ),
                    },
                ),
            ],
        )
        adapter = BigQueryAdapter(config)

        row = {
            "internal_customer_id": "cust_001",
            "timestamp": datetime.now(tz=timezone.utc),
            "properties": {
                "products": [
                    {"name": "Product A", "category": "Electronics"},
                    {"name": "Product B", "product_category": "Clothing"},
                ],
            },
        }

        event = adapter._row_to_event(
            row,
            config.event_tables[0],
            InternalEventType.PURCHASE,
        )

        assert event is not None
        assert event.properties.custom_properties is not None
        assert "product_categories" in event.properties.custom_properties

    def test_row_to_event_handles_exception(self, basic_config: BigQueryConfig) -> None:
        """Test row conversion handles exceptions gracefully."""
        adapter = BigQueryAdapter(basic_config)
        table_config = basic_config.event_tables[0]

        # Row that will cause an error (timestamp not a string or datetime)
        row = {
            "internal_customer_id": "cust_001",
            "timestamp": object(),  # Will fail to convert
        }

        event = adapter._row_to_event(row, table_config, InternalEventType.PURCHASE)
        assert event is None

    def test_auto_detect_properties(self, basic_config: BigQueryConfig) -> None:
        """Test auto-detection of known fields."""
        adapter = BigQueryAdapter(basic_config)
        known_fields = ["email", "first_name", "country"]
        props: dict[str, Any] = {}

        row = {
            "email": "test@example.com",
            "properties": {
                "first_name": "John",
                "country": "US",
            },
        }

        adapter._auto_detect_properties(row, props, known_fields)

        assert props["email"] == "test@example.com"
        assert props["first_name"] == "John"
        assert props["country"] == "US"

    def test_auto_detect_properties_with_suffix(self, basic_config: BigQueryConfig) -> None:
        """Test auto-detection handles field suffixes."""
        adapter = BigQueryAdapter(basic_config)
        known_fields = ["email"]
        props: dict[str, Any] = {}

        row = {
            "properties": {
                "email__hash123": "test@example.com",
            },
        }

        adapter._auto_detect_properties(row, props, known_fields)

        assert props["email"] == "test@example.com"

    def test_auto_detect_skips_existing(self, basic_config: BigQueryConfig) -> None:
        """Test auto-detection doesn't overwrite existing values."""
        adapter = BigQueryAdapter(basic_config)
        known_fields = ["email"]
        props: dict[str, Any] = {"email": "existing@example.com"}

        row = {
            "email": "new@example.com",
        }

        adapter._auto_detect_properties(row, props, known_fields)

        assert props["email"] == "existing@example.com"


# =============================================================================
# TESTS: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_load_from_bigquery(self) -> None:
        """Test load_from_bigquery creates adapter and loads."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
        )

        with patch.object(BigQueryAdapter, "load") as mock_load:
            mock_load.return_value = LoadResult()

            result = load_from_bigquery(config)

            mock_load.assert_called_once()
            assert isinstance(result, LoadResult)

    def test_create_config_from_tables_minimal(self) -> None:
        """Test create_config_from_tables with minimal args."""
        config = create_config_from_tables(
            project_id="my-project",
            dataset_id="my_dataset",
        )

        assert config.project_id == "my-project"
        assert config.dataset_id == "my_dataset"
        assert config.event_tables == []

    def test_create_config_from_tables_with_purchase(self) -> None:
        """Test create_config_from_tables with purchase table."""
        config = create_config_from_tables(
            project_id="project",
            dataset_id="dataset",
            purchase_table="purchases",
        )

        assert len(config.event_tables) == 1
        assert config.event_tables[0].event_type == EventType.PURCHASE
        assert config.event_tables[0].table_name == "purchases"

    def test_create_config_from_tables_with_all_event_tables(self) -> None:
        """Test create_config_from_tables with all event table types."""
        config = create_config_from_tables(
            project_id="project",
            dataset_id="dataset",
            purchase_table="purchases",
            page_view_table="page_views",
            view_item_table="view_items",
            add_to_cart_table="cart_events",
        )

        assert len(config.event_tables) == 4

        event_types = {t.event_type for t in config.event_tables}
        assert EventType.PURCHASE in event_types
        assert EventType.PAGE_VIEW in event_types
        assert EventType.VIEW_ITEM in event_types
        assert EventType.ADD_TO_CART in event_types

    def test_create_config_from_tables_with_customer_table(self) -> None:
        """Test create_config_from_tables with customer table."""
        config = create_config_from_tables(
            project_id="project",
            dataset_id="dataset",
            customer_table="customers",
        )

        assert config.customer_table is not None
        assert config.customer_table.table_name == "customers"

    def test_create_config_from_tables_with_merge_table(self) -> None:
        """Test create_config_from_tables with merge table."""
        config = create_config_from_tables(
            project_id="project",
            dataset_id="dataset",
            merge_table="id_history",
        )

        assert config.merge_table is not None
        assert config.merge_table.table_name == "id_history"

    def test_create_config_from_tables_with_dates_and_limit(self) -> None:
        """Test create_config_from_tables with date range and limit."""
        config = create_config_from_tables(
            project_id="project",
            dataset_id="dataset",
            start_date="2024-01-01",
            end_date="2024-12-31",
            limit=10000,
        )

        assert config.start_date == "2024-01-01"
        assert config.end_date == "2024-12-31"
        assert config.limit_per_table == 10000

    def test_create_config_from_tables_complete(self) -> None:
        """Test create_config_from_tables with all options."""
        config = create_config_from_tables(
            project_id="production",
            dataset_id="cdp_data",
            purchase_table="purchases",
            page_view_table="page_views",
            view_item_table="product_views",
            add_to_cart_table="cart",
            customer_table="customer_profiles",
            merge_table="identity_graph",
            start_date="2024-01-01",
            end_date="2024-06-30",
            limit=50000,
        )

        assert config.project_id == "production"
        assert len(config.event_tables) == 4
        assert config.customer_table is not None
        assert config.merge_table is not None
        assert config.start_date == "2024-01-01"
        assert config.limit_per_table == 50000


# =============================================================================
# TESTS: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for BigQueryAdapter."""

    def test_load_with_mocked_client(self) -> None:
        """Test full load flow with mocked BigQuery client."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            event_tables=[
                EventTableConfig(
                    table_name="purchases",
                    event_type=EventType.PURCHASE,
                ),
            ],
            merge_table=MergeTableConfig(table_name="id_history"),
            customer_table=CustomerTableConfig(table_name="customers"),
        )

        adapter = BigQueryAdapter(config)

        # Mock the BigQuery client
        mock_client = MagicMock()

        # Mock event query results
        mock_event_row = MagicMock()
        mock_event_row.__iter__ = lambda self: iter(["internal_customer_id", "timestamp"])
        mock_event_row.keys = lambda: ["internal_customer_id", "timestamp"]
        mock_event_row.values = lambda: ["cust_001", datetime.now(tz=timezone.utc)]
        mock_event_row.items = lambda: [
            ("internal_customer_id", "cust_001"),
            ("timestamp", datetime.now(tz=timezone.utc)),
        ]

        def dict_from_row(row):
            return {
                "internal_customer_id": "cust_001",
                "timestamp": datetime.now(tz=timezone.utc),
            }

        # Mock query job
        mock_query_job = MagicMock()
        mock_query_job.result.return_value = []  # Empty results for simplicity

        mock_client.query.return_value = mock_query_job

        # Patch the client property
        with patch.object(BigQueryAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            result = adapter.load()

            assert isinstance(result, LoadResult)
            assert result.load_duration_ms > 0

    def test_load_handles_error(self) -> None:
        """Test load handles errors gracefully."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            event_tables=[
                EventTableConfig(
                    table_name="purchases",
                    event_type=EventType.PURCHASE,
                ),
            ],
        )

        adapter = BigQueryAdapter(config)

        with patch.object(BigQueryAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client = MagicMock()
            mock_client.query.side_effect = Exception("Connection failed")
            mock_client_prop.return_value = mock_client

            result = adapter.load()

            assert len(result.errors) == 1
            assert "Connection failed" in result.errors[0]

    def test_load_event_table(self) -> None:
        """Test _load_event_table method."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            event_tables=[
                EventTableConfig(
                    table_name="purchases",
                    event_type=EventType.PURCHASE,
                ),
            ],
        )

        adapter = BigQueryAdapter(config)

        # Create mock row that can be converted to dict
        class MockRow(dict):
            """Mock BigQuery row that behaves like a dict."""
            pass

        mock_row = MockRow({
            "internal_customer_id": "cust_001",
            "timestamp": datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
        })

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = [mock_row]

        mock_client = MagicMock()
        mock_client.query.return_value = mock_query_job

        with patch.object(BigQueryAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            events, count = adapter._load_event_table(config.event_tables[0])

            assert count == 1
            assert len(events) == 1
            assert events[0].internal_customer_id == "cust_001"

    def test_load_merge_table(self) -> None:
        """Test _load_merge_table method."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            merge_table=MergeTableConfig(table_name="id_history"),
        )

        adapter = BigQueryAdapter(config)

        # Create mock rows that behave like dicts
        class MockRow(dict):
            """Mock BigQuery row that behaves like a dict."""
            pass

        mock_rows = [
            MockRow({"current_id": "cust_001", "past_id": "old_001"}),
            MockRow({"current_id": "cust_002", "past_id": "old_002"}),
            MockRow({"current_id": "cust_003", "past_id": None}),  # Should be skipped
        ]

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_rows

        mock_client = MagicMock()
        mock_client.query.return_value = mock_query_job

        with patch.object(BigQueryAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            id_history = adapter._load_merge_table(config.merge_table)

            assert len(id_history) == 2
            assert id_history[0].internal_customer_id == "cust_001"
            assert id_history[0].past_id == "old_001"

    def test_load_merge_table_with_limit(self) -> None:
        """Test _load_merge_table with limit."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            merge_table=MergeTableConfig(table_name="id_history"),
            limit_per_table=100,
        )

        adapter = BigQueryAdapter(config)

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = []

        mock_client = MagicMock()
        mock_client.query.return_value = mock_query_job

        with patch.object(BigQueryAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            adapter._load_merge_table(config.merge_table)

            # Check that LIMIT was added to query
            query_call = mock_client.query.call_args[0][0]
            assert "LIMIT 100" in query_call

    def test_load_customer_properties(self) -> None:
        """Test _load_customer_properties method."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            customer_table=CustomerTableConfig(
                table_name="customers",
                property_mappings={
                    "email": FieldMapping("properties.email", "email"),
                },
            ),
        )

        adapter = BigQueryAdapter(config)

        # Create mock rows that behave like dicts
        class MockRow(dict):
            """Mock BigQuery row that behaves like a dict."""
            pass

        mock_rows = [
            MockRow({
                "internal_customer_id": "cust_001",
                "properties": {"email": "test@example.com"},
            }),
            MockRow({
                "internal_customer_id": None,  # Should be skipped
                "properties": {"email": "skip@example.com"},
            }),
        ]

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_rows

        mock_client = MagicMock()
        mock_client.query.return_value = mock_query_job

        with patch.object(BigQueryAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            props = adapter._load_customer_properties(config.customer_table)

            assert "cust_001" in props
            assert props["cust_001"]["email"] == "test@example.com"

    def test_load_customer_properties_with_limit(self) -> None:
        """Test _load_customer_properties with limit."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            customer_table=CustomerTableConfig(table_name="customers"),
            limit_per_table=500,
        )

        adapter = BigQueryAdapter(config)

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = []

        mock_client = MagicMock()
        mock_client.query.return_value = mock_query_job

        with patch.object(BigQueryAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            adapter._load_customer_properties(config.customer_table)

            query_call = mock_client.query.call_args[0][0]
            assert "LIMIT 500" in query_call

    def test_load_customer_properties_auto_detect(self) -> None:
        """Test _load_customer_properties with auto-detect enabled."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            customer_table=CustomerTableConfig(
                table_name="customers",
                auto_detect_fields=True,
            ),
        )

        adapter = BigQueryAdapter(config)

        # Create mock rows that behave like dicts
        class MockRow(dict):
            """Mock BigQuery row that behaves like a dict."""
            pass

        mock_rows = [
            MockRow({
                "internal_customer_id": "cust_001",
                "email": "direct@example.com",  # Top-level field
                "properties": {
                    "first_name": "John",
                    "country": "US",
                },
            }),
        ]

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_rows

        mock_client = MagicMock()
        mock_client.query.return_value = mock_query_job

        with patch.object(BigQueryAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            props = adapter._load_customer_properties(config.customer_table)

            assert "cust_001" in props
            assert props["cust_001"]["email"] == "direct@example.com"
            assert props["cust_001"]["first_name"] == "John"

    def test_get_bigquery_successful_import(self) -> None:
        """Test _get_bigquery with successful import."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
        )

        adapter = BigQueryAdapter(config)

        # Mock the import
        mock_bigquery = MagicMock()

        with patch.dict("sys.modules", {"google.cloud.bigquery": mock_bigquery, "google.cloud": MagicMock()}):
            with patch("builtins.__import__") as mock_import:
                def import_side_effect(name, *args, **kwargs):
                    if name == "google.cloud":
                        mock_module = MagicMock()
                        mock_module.bigquery = mock_bigquery
                        return mock_module
                    raise ImportError(f"No module named '{name}'")

                # Actually test the caching behavior
                adapter._bigquery_module = mock_bigquery

                result = adapter._get_bigquery()
                assert result == mock_bigquery

    def test_get_bigquery_fresh_import(self) -> None:
        """Test _get_bigquery performs actual import when module is None."""
        import sys

        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
        )

        adapter = BigQueryAdapter(config)

        # Ensure module is None
        adapter._bigquery_module = None

        # Create a mock bigquery module
        mock_bigquery = MagicMock()
        mock_google_cloud = MagicMock()
        mock_google_cloud.bigquery = mock_bigquery

        # Save original modules
        original_google = sys.modules.get("google", None)
        original_google_cloud = sys.modules.get("google.cloud", None)
        original_bigquery = sys.modules.get("google.cloud.bigquery", None)

        try:
            # Set up the mock import
            sys.modules["google"] = MagicMock()
            sys.modules["google.cloud"] = mock_google_cloud
            sys.modules["google.cloud.bigquery"] = mock_bigquery

            result = adapter._get_bigquery()

            # Should have stored and returned the module
            assert result == mock_bigquery
            assert adapter._bigquery_module == mock_bigquery
        finally:
            # Restore original modules
            if original_google is None:
                sys.modules.pop("google", None)
            else:
                sys.modules["google"] = original_google
            if original_google_cloud is None:
                sys.modules.pop("google.cloud", None)
            else:
                sys.modules["google.cloud"] = original_google_cloud
            if original_bigquery is None:
                sys.modules.pop("google.cloud.bigquery", None)
            else:
                sys.modules["google.cloud.bigquery"] = original_bigquery

    def test_load_customer_properties_with_default_values(self) -> None:
        """Test _load_customer_properties uses default values when value is None."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            customer_table=CustomerTableConfig(
                table_name="customers",
                property_mappings={
                    "country": FieldMapping(
                        "properties.country",
                        "country",
                        default="Unknown",  # Default value
                    ),
                    "segment": FieldMapping(
                        "properties.segment",
                        "segment",
                        default="Default",
                    ),
                },
            ),
        )

        adapter = BigQueryAdapter(config)

        # Create mock rows that behave like dicts
        class MockRow(dict):
            """Mock BigQuery row that behaves like a dict."""
            pass

        mock_rows = [
            MockRow({
                "internal_customer_id": "cust_001",
                "properties": {},  # No country or segment - should use defaults
            }),
            MockRow({
                "internal_customer_id": "cust_002",
                "properties": {"country": "US"},  # Has country, should use it
            }),
        ]

        mock_query_job = MagicMock()
        mock_query_job.result.return_value = mock_rows

        mock_client = MagicMock()
        mock_client.query.return_value = mock_query_job

        with patch.object(BigQueryAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            props = adapter._load_customer_properties(config.customer_table)

            # cust_001 should have default values
            assert "cust_001" in props
            assert props["cust_001"]["country"] == "Unknown"
            assert props["cust_001"]["segment"] == "Default"

            # cust_002 should have actual country, default segment
            assert "cust_002" in props
            assert props["cust_002"]["country"] == "US"
            assert props["cust_002"]["segment"] == "Default"

    def test_full_load_with_all_tables(self) -> None:
        """Test full load with events, merge, and customer tables."""
        config = BigQueryConfig(
            project_id="test-project",
            dataset_id="test_dataset",
            event_tables=[
                EventTableConfig(
                    table_name="purchases",
                    event_type=EventType.PURCHASE,
                ),
            ],
            merge_table=MergeTableConfig(table_name="id_history"),
            customer_table=CustomerTableConfig(table_name="customers"),
        )

        adapter = BigQueryAdapter(config)

        # Create mock rows that behave like dicts
        class MockRow(dict):
            """Mock BigQuery row that behaves like a dict."""
            pass

        # Event rows
        event_rows = [
            MockRow({
                "internal_customer_id": "cust_001",
                "timestamp": datetime(2024, 1, 15, tzinfo=timezone.utc),
            }),
        ]

        # Merge rows
        merge_rows = [
            MockRow({"current_id": "cust_001", "past_id": "old_001"}),
        ]

        # Customer rows
        customer_rows = [
            MockRow({
                "internal_customer_id": "cust_001",
                "properties": {"email": "test@example.com"},
            }),
        ]

        call_count = [0]

        def mock_query(query):
            mock_job = MagicMock()
            if "purchases" in query:
                mock_job.result.return_value = event_rows
            elif "id_history" in query:
                mock_job.result.return_value = merge_rows
            elif "customers" in query:
                mock_job.result.return_value = customer_rows
            else:
                mock_job.result.return_value = []
            call_count[0] += 1
            return mock_job

        mock_client = MagicMock()
        mock_client.query = mock_query

        with patch.object(BigQueryAdapter, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            result = adapter.load()

            assert result.tables_loaded == 1
            assert len(result.events) == 1
            assert len(result.id_history) == 1
            assert "cust_001" in result.customer_properties
            assert result.unique_customers == 1
