"""
Tests for src/data/local_loader.py

Comprehensive test suite for LocalDataLoader and related functionality.
"""

import json
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.data.local_loader import (
    CANONICAL_TO_INTERNAL_EVENT_TYPE,
    TABLE_TO_EVENT_TYPE,
    NON_EVENT_TABLES,
    LoadResult,
    LocalDataLoader,
    load_local_data,
    load_events_only,
)
from src.data.field_mapping import (
    ClientSchemaConfig,
    EventTypeMapping,
    create_bloomreach_config,
    create_ga4_config,
)
from src.data.schemas import EventType, EventRecord, CustomerIdHistory


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_pandas():
    """Mock pandas module for testing without actual parquet files."""
    with patch.dict("sys.modules", {"pandas": MagicMock()}):
        yield


@pytest.fixture
def sample_event_df():
    """Create a mock DataFrame with sample event data."""
    mock_df = MagicMock()
    mock_df.__len__ = MagicMock(return_value=2)

    # Create mock rows
    row1 = {
        "internal_customer_id": "cust_001",
        "timestamp": datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
        "properties": {
            "product_id": "prod_123",
            "title": "Test Product",
            "price": "29.99",
            "category_level_1": "Electronics",
        }
    }
    row2 = {
        "internal_customer_id": "cust_002",
        "timestamp": "2024-01-16T14:00:00Z",
        "properties": {
            "product_id": "prod_456",
            "title": "Another Product",
            "price": "49.99",
        }
    }

    mock_df.iterrows = MagicMock(return_value=iter([
        (0, MagicMock(**{"get": lambda k, d=None: row1.get(k, d)})),
        (1, MagicMock(**{"get": lambda k, d=None: row2.get(k, d)})),
    ]))

    return mock_df


@pytest.fixture
def sample_id_history_df():
    """Create a mock DataFrame with ID history data."""
    mock_df = MagicMock()
    mock_df.__len__ = MagicMock(return_value=2)

    row1 = {"internal_customer_id": "cust_001", "past_id": "old_001"}
    row2 = {"internal_customer_id": "cust_002", "past_id": "old_002"}

    mock_df.iterrows = MagicMock(return_value=iter([
        (0, MagicMock(**{"get": lambda k, d=None: row1.get(k, d)})),
        (1, MagicMock(**{"get": lambda k, d=None: row2.get(k, d)})),
    ]))

    return mock_df


# =============================================================================
# TESTS: CONSTANTS AND MAPPINGS
# =============================================================================


class TestEventTypeMappings:
    """Tests for event type mapping constants."""

    def test_canonical_to_internal_event_type_completeness(self):
        """All EventTypeMapping values should map to internal EventType."""
        for mapping in EventTypeMapping:
            assert mapping in CANONICAL_TO_INTERNAL_EVENT_TYPE
            assert isinstance(CANONICAL_TO_INTERNAL_EVENT_TYPE[mapping], EventType)

    def test_table_to_event_type_purchase(self):
        """Purchase table should map to PURCHASE event type."""
        assert TABLE_TO_EVENT_TYPE["purchase"] == EventType.PURCHASE

    def test_table_to_event_type_view_item(self):
        """view_item table should map to VIEW_ITEM event type."""
        assert TABLE_TO_EVENT_TYPE["view_item"] == EventType.VIEW_ITEM

    def test_table_to_event_type_cart_update(self):
        """cart_update table should map to ADD_TO_CART event type."""
        assert TABLE_TO_EVENT_TYPE["cart_update"] == EventType.ADD_TO_CART

    def test_table_to_event_type_session_events(self):
        """Session tables should map to SESSION_START event type."""
        assert TABLE_TO_EVENT_TYPE["session_start"] == EventType.SESSION_START
        assert TABLE_TO_EVENT_TYPE["session_end"] == EventType.SESSION_START

    def test_non_event_tables_contains_expected(self):
        """NON_EVENT_TABLES should contain known non-event tables."""
        expected = {"customers_properties", "customers_id_history", "customers_external_ids", "merge"}
        assert NON_EVENT_TABLES == expected


# =============================================================================
# TESTS: LoadResult
# =============================================================================


class TestLoadResult:
    """Tests for LoadResult dataclass."""

    def test_default_values(self):
        """LoadResult should have sensible defaults."""
        result = LoadResult()

        assert result.events == []
        assert result.id_history == []
        assert result.customer_properties == {}
        assert result.tables_loaded == []
        assert result.events_by_type == {}
        assert result.total_rows == 0
        assert result.unique_customers == 0
        assert result.load_duration_ms == 0.0
        assert result.errors == []

    def test_can_set_values(self):
        """LoadResult should accept custom values."""
        result = LoadResult(
            total_rows=100,
            unique_customers=50,
            tables_loaded=["purchase", "view_item"],
        )

        assert result.total_rows == 100
        assert result.unique_customers == 50
        assert result.tables_loaded == ["purchase", "view_item"]


# =============================================================================
# TESTS: LocalDataLoader Initialization
# =============================================================================


class TestLocalDataLoaderInit:
    """Tests for LocalDataLoader initialization."""

    def test_init_with_string_path(self, temp_data_dir):
        """Should accept string path."""
        loader = LocalDataLoader(str(temp_data_dir))
        assert loader.data_dir == temp_data_dir

    def test_init_with_path_object(self, temp_data_dir):
        """Should accept Path object."""
        loader = LocalDataLoader(temp_data_dir)
        assert loader.data_dir == temp_data_dir

    def test_default_schema_config(self, temp_data_dir):
        """Should use Bloomreach config by default."""
        loader = LocalDataLoader(temp_data_dir)
        assert loader.schema_config.client_name == "bloomreach"

    def test_custom_schema_config(self, temp_data_dir):
        """Should accept custom schema config."""
        ga4_config = create_ga4_config()
        loader = LocalDataLoader(temp_data_dir, schema_config=ga4_config)
        assert loader.schema_config.client_name == "ga4"

    def test_include_tables(self, temp_data_dir):
        """Should store include_tables as set."""
        loader = LocalDataLoader(
            temp_data_dir,
            include_tables=["purchase", "view_item"]
        )
        assert loader.include_tables == {"purchase", "view_item"}

    def test_exclude_tables(self, temp_data_dir):
        """Should store exclude_tables as set."""
        loader = LocalDataLoader(
            temp_data_dir,
            exclude_tables=["session_start"]
        )
        assert loader.exclude_tables == {"session_start"}

    def test_none_include_tables(self, temp_data_dir):
        """Should handle None include_tables."""
        loader = LocalDataLoader(temp_data_dir, include_tables=None)
        assert loader.include_tables is None


# =============================================================================
# TESTS: LocalDataLoader._get_pandas
# =============================================================================


class TestGetPandas:
    """Tests for lazy pandas import."""

    def test_imports_pandas_on_first_call(self, temp_data_dir):
        """Should import pandas lazily."""
        loader = LocalDataLoader(temp_data_dir)
        assert loader._pandas_module is None

        pd = loader._get_pandas()
        assert pd is not None
        assert loader._pandas_module is not None

    def test_caches_pandas_module(self, temp_data_dir):
        """Should cache pandas module after first import."""
        loader = LocalDataLoader(temp_data_dir)

        pd1 = loader._get_pandas()
        pd2 = loader._get_pandas()

        assert pd1 is pd2

    def test_raises_import_error_without_pandas(self, temp_data_dir):
        """Should raise ImportError with helpful message if pandas unavailable."""
        loader = LocalDataLoader(temp_data_dir)

        with patch.dict("sys.modules", {"pandas": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module named 'pandas'")):
                with pytest.raises(ImportError, match="pandas is required"):
                    loader._get_pandas()


# =============================================================================
# TESTS: LocalDataLoader.load
# =============================================================================


class TestLocalDataLoaderLoad:
    """Tests for LocalDataLoader.load method."""

    def test_returns_error_for_nonexistent_directory(self):
        """Should return error when directory doesn't exist."""
        loader = LocalDataLoader("/nonexistent/path")
        result = loader.load()

        assert len(result.errors) == 1
        assert "not found" in result.errors[0]

    def test_empty_directory(self, temp_data_dir):
        """Should handle empty directory gracefully."""
        loader = LocalDataLoader(temp_data_dir)
        result = loader.load()

        assert result.events == []
        assert result.id_history == []
        assert result.tables_loaded == []
        assert result.errors == []

    def test_include_tables_filter(self, temp_data_dir):
        """Should only load included tables."""
        # Create mock parquet files
        (temp_data_dir / "purchase.parquet").touch()
        (temp_data_dir / "view_item.parquet").touch()
        (temp_data_dir / "session_start.parquet").touch()

        loader = LocalDataLoader(
            temp_data_dir,
            include_tables=["purchase"]
        )

        with patch.object(loader, "_get_pandas") as mock_pd:
            mock_pd.return_value.read_parquet.return_value = MagicMock(
                __len__=lambda x: 0,
                iterrows=lambda: iter([])
            )

            result = loader.load()

            # Should only process purchase table
            assert "purchase" in result.tables_loaded
            assert "view_item" not in result.tables_loaded

    def test_exclude_tables_filter(self, temp_data_dir):
        """Should skip excluded tables."""
        (temp_data_dir / "purchase.parquet").touch()
        (temp_data_dir / "session_start.parquet").touch()

        loader = LocalDataLoader(
            temp_data_dir,
            exclude_tables=["session_start"]
        )

        with patch.object(loader, "_get_pandas") as mock_pd:
            mock_pd.return_value.read_parquet.return_value = MagicMock(
                __len__=lambda x: 0,
                iterrows=lambda: iter([])
            )

            result = loader.load()

            assert "session_start" not in result.tables_loaded

    def test_records_load_duration(self, temp_data_dir):
        """Should record load duration."""
        loader = LocalDataLoader(temp_data_dir)
        result = loader.load()

        assert result.load_duration_ms >= 0


# =============================================================================
# TESTS: LocalDataLoader._row_to_event
# =============================================================================


class TestRowToEvent:
    """Tests for converting rows to EventRecord."""

    def test_missing_customer_id_returns_none(self, temp_data_dir):
        """Should return None if customer_id is missing."""
        loader = LocalDataLoader(temp_data_dir)

        row = MagicMock()
        row.get = MagicMock(side_effect=lambda k, d=None: {
            "internal_customer_id": None,
            "timestamp": datetime.now(tz=timezone.utc),
        }.get(k, d))

        result = loader._row_to_event(row, EventType.PURCHASE)
        assert result is None

    def test_missing_timestamp_returns_none(self, temp_data_dir):
        """Should return None if timestamp is missing."""
        loader = LocalDataLoader(temp_data_dir)

        row = MagicMock()
        row.get = MagicMock(side_effect=lambda k, d=None: {
            "internal_customer_id": "cust_001",
            "timestamp": None,
        }.get(k, d))

        result = loader._row_to_event(row, EventType.PURCHASE)
        assert result is None

    def test_parses_string_timestamp(self, temp_data_dir):
        """Should parse ISO format string timestamp."""
        loader = LocalDataLoader(temp_data_dir)

        row = MagicMock()
        row.get = MagicMock(side_effect=lambda k, d=None: {
            "internal_customer_id": "cust_001",
            "timestamp": "2024-01-15T10:30:00Z",
            "properties": {},
        }.get(k, d))

        result = loader._row_to_event(row, EventType.VIEW_ITEM)

        assert result is not None
        assert result.timestamp.year == 2024
        assert result.timestamp.month == 1
        assert result.timestamp.day == 15

    def test_handles_pandas_timestamp(self, temp_data_dir):
        """Should handle pandas Timestamp objects."""
        loader = LocalDataLoader(temp_data_dir)

        mock_timestamp = MagicMock()
        mock_timestamp.to_pydatetime = MagicMock(
            return_value=datetime(2024, 1, 15, 10, 30)
        )

        row = MagicMock()
        row.get = MagicMock(side_effect=lambda k, d=None: {
            "internal_customer_id": "cust_001",
            "timestamp": mock_timestamp,
            "properties": {},
        }.get(k, d))

        result = loader._row_to_event(row, EventType.VIEW_ITEM)

        assert result is not None
        assert result.timestamp.tzinfo is not None  # Should be timezone aware

    def test_adds_utc_to_naive_timestamp(self, temp_data_dir):
        """Should add UTC timezone to naive datetime."""
        loader = LocalDataLoader(temp_data_dir)

        naive_dt = datetime(2024, 1, 15, 10, 30)

        row = MagicMock()
        row.get = MagicMock(side_effect=lambda k, d=None: {
            "internal_customer_id": "cust_001",
            "timestamp": naive_dt,
            "properties": {},
        }.get(k, d))

        result = loader._row_to_event(row, EventType.VIEW_ITEM)

        assert result is not None
        assert result.timestamp.tzinfo == timezone.utc

    def test_flat_structure_without_properties_field(self, temp_data_dir):
        """Should use entire row as dict when properties_field is None."""
        config = ClientSchemaConfig(
            client_name="flat",
            customer_id_field="customer_id",
            properties_field=None,
        )
        loader = LocalDataLoader(temp_data_dir, schema_config=config)

        row_data = {
            "customer_id": "cust_001",
            "timestamp": datetime.now(tz=timezone.utc),
            "product_id": "prod_123",
            "price": "29.99",
        }

        # Create a proper mock that acts like a dict
        class DictLikeRow(dict):
            def get(self, key, default=None):
                return row_data.get(key, default)

        row = DictLikeRow(row_data)

        result = loader._row_to_event(row, EventType.VIEW_ITEM)

        assert result is not None
        assert result.internal_customer_id == "cust_001"

    def test_generates_uuid_for_event_id(self, temp_data_dir):
        """Should generate UUID for event_id."""
        loader = LocalDataLoader(temp_data_dir)

        row = MagicMock()
        row.get = MagicMock(side_effect=lambda k, d=None: {
            "internal_customer_id": "cust_001",
            "timestamp": datetime.now(tz=timezone.utc),
            "properties": {},
        }.get(k, d))

        result = loader._row_to_event(row, EventType.VIEW_ITEM)

        assert result is not None
        assert len(result.event_id) == 36  # UUID format


# =============================================================================
# TESTS: LocalDataLoader._build_event_properties
# =============================================================================


class TestBuildEventProperties:
    """Tests for building EventProperties from raw data."""

    def test_extracts_product_fields(self, temp_data_dir):
        """Should extract product-related fields."""
        loader = LocalDataLoader(temp_data_dir)

        props = {
            "product_id": "prod_123",
            "title": "Test Product",
            "price": "29.99",
            "category_level_1": "Electronics",
        }

        result = loader._build_event_properties(props, EventType.VIEW_ITEM)

        assert result.product_id == "prod_123"
        assert result.product_name == "Test Product"
        assert result.product_price == Decimal("29.99")
        assert result.product_category == "Electronics"

    def test_extracts_order_fields(self, temp_data_dir):
        """Should extract order-related fields."""
        loader = LocalDataLoader(temp_data_dir)

        props = {
            "purchase_id": "order_789",
            "total_price": "199.99",
            "total_quantity": "3",
        }

        result = loader._build_event_properties(props, EventType.PURCHASE)

        assert result.order_id == "order_789"
        assert result.order_total == Decimal("199.99")
        assert result.quantity == 3

    def test_extracts_search_query(self, temp_data_dir):
        """Should extract search query."""
        loader = LocalDataLoader(temp_data_dir)

        props = {"query": "laptop"}

        result = loader._build_event_properties(props, EventType.VIEW_CATEGORY)

        assert result.search_query == "laptop"

    def test_extracts_device_type(self, temp_data_dir):
        """Should extract device type."""
        loader = LocalDataLoader(temp_data_dir)

        props = {"device": "mobile"}

        result = loader._build_event_properties(props, EventType.SESSION_START)

        assert result.device_type == "mobile"

    def test_extracts_page_info(self, temp_data_dir):
        """Should extract page URL and title."""
        loader = LocalDataLoader(temp_data_dir)

        props = {
            "page_url": "https://example.com/products",
            "page_title": "Products Page",
        }

        result = loader._build_event_properties(props, EventType.VIEW_ITEM)

        assert result.page_url == "https://example.com/products"
        assert result.page_title == "Products Page"

    def test_extracts_session_id(self, temp_data_dir):
        """Should extract session ID."""
        loader = LocalDataLoader(temp_data_dir)

        props = {"session_id": "sess_123"}

        result = loader._build_event_properties(props, EventType.SESSION_START)

        assert result.session_id == "sess_123"

    def test_handles_invalid_price(self, temp_data_dir):
        """Should handle non-numeric price values gracefully."""
        loader = LocalDataLoader(temp_data_dir)

        props = {"price": "invalid"}

        result = loader._build_event_properties(props, EventType.VIEW_ITEM)

        assert result.product_price is None

    def test_handles_invalid_quantity(self, temp_data_dir):
        """Should handle non-numeric quantity values."""
        loader = LocalDataLoader(temp_data_dir)

        props = {"quantity": "invalid"}

        result = loader._build_event_properties(props, EventType.PURCHASE)

        assert result.quantity is None

    def test_handles_numpy_types(self, temp_data_dir):
        """Should handle numpy scalar types."""
        loader = LocalDataLoader(temp_data_dir)

        # Mock numpy int64
        mock_value = MagicMock()
        mock_value.item = MagicMock(return_value=42)

        props = {"quantity": mock_value}

        result = loader._build_event_properties(props, EventType.PURCHASE)

        assert result.quantity == 42

    def test_stores_custom_properties(self, temp_data_dir):
        """Should store unknown fields in custom_properties."""
        loader = LocalDataLoader(temp_data_dir)

        props = {
            "product_id": "prod_123",
            "custom_field": "custom_value",
            "another_field": 42,
        }

        result = loader._build_event_properties(props, EventType.VIEW_ITEM)

        assert result.custom_properties is not None
        assert "custom_field" in result.custom_properties
        assert result.custom_properties["custom_field"] == "custom_value"

    def test_alternative_field_names_for_product_id(self, temp_data_dir):
        """Should try alternative field names for product_id."""
        loader = LocalDataLoader(temp_data_dir)

        props = {"item_id": "item_123"}  # Alternative name

        result = loader._build_event_properties(props, EventType.VIEW_ITEM)

        assert result.product_id == "item_123"

    def test_alternative_field_names_for_product_name(self, temp_data_dir):
        """Should try alternative field names for product_name."""
        loader = LocalDataLoader(temp_data_dir)

        props = {"product_name": "My Product"}  # Alternative name

        result = loader._build_event_properties(props, EventType.VIEW_ITEM)

        assert result.product_name == "My Product"

    def test_uses_config_category_fields(self, temp_data_dir):
        """Should use category_fields from config for category extraction."""
        config = create_bloomreach_config()
        loader = LocalDataLoader(temp_data_dir, schema_config=config)

        props = {"category_level_1": "Clothing"}

        result = loader._build_event_properties(props, EventType.VIEW_ITEM)

        assert result.product_category == "Clothing"

    def test_uses_config_revenue_fields(self, temp_data_dir):
        """Should use revenue_fields from config for order_total extraction."""
        config = create_bloomreach_config()
        loader = LocalDataLoader(temp_data_dir, schema_config=config)

        props = {"total_price": "99.99"}

        result = loader._build_event_properties(props, EventType.PURCHASE)

        assert result.order_total == Decimal("99.99")


# =============================================================================
# TESTS: LocalDataLoader._load_id_history
# =============================================================================


class TestLoadIdHistory:
    """Tests for loading ID history."""

    def test_loads_id_history_records(self, temp_data_dir):
        """Should load CustomerIdHistory records."""
        loader = LocalDataLoader(temp_data_dir)

        mock_df = MagicMock()
        rows = [
            {"internal_customer_id": "cust_001", "past_id": "old_001"},
            {"internal_customer_id": "cust_002", "past_id": "old_002"},
        ]

        mock_df.iterrows = MagicMock(return_value=iter([
            (0, MagicMock(**{"get": lambda k, d=None, r=rows[0]: r.get(k, d)})),
            (1, MagicMock(**{"get": lambda k, d=None, r=rows[1]: r.get(k, d)})),
        ]))

        seen: set[str] = set()
        result = loader._load_id_history(mock_df, seen)

        assert len(result) == 2
        assert all(isinstance(r, CustomerIdHistory) for r in result)

    def test_deduplicates_by_past_id(self, temp_data_dir):
        """Should skip duplicate past_ids."""
        loader = LocalDataLoader(temp_data_dir)

        mock_df = MagicMock()
        rows = [
            {"internal_customer_id": "cust_001", "past_id": "old_001"},
            {"internal_customer_id": "cust_002", "past_id": "old_001"},  # Duplicate
        ]

        mock_rows = [
            MagicMock(**{"get": lambda k, d=None, r=rows[0]: r.get(k, d)}),
            MagicMock(**{"get": lambda k, d=None, r=rows[1]: r.get(k, d)}),
        ]
        mock_df.iterrows = MagicMock(return_value=iter(enumerate(mock_rows)))

        seen: set[str] = set()
        result = loader._load_id_history(mock_df, seen)

        assert len(result) == 1
        assert "old_001" in seen

    def test_skips_missing_fields(self, temp_data_dir):
        """Should skip rows with missing required fields."""
        loader = LocalDataLoader(temp_data_dir)

        mock_df = MagicMock()
        rows = [
            {"internal_customer_id": "cust_001", "past_id": None},
            {"internal_customer_id": None, "past_id": "old_002"},
        ]

        mock_rows = [
            MagicMock(**{"get": lambda k, d=None, r=rows[0]: r.get(k, d)}),
            MagicMock(**{"get": lambda k, d=None, r=rows[1]: r.get(k, d)}),
        ]
        mock_df.iterrows = MagicMock(return_value=iter(enumerate(mock_rows)))

        seen: set[str] = set()
        result = loader._load_id_history(mock_df, seen)

        assert len(result) == 0

    def test_uses_config_field_names(self, temp_data_dir):
        """Should use field names from schema config."""
        config = ClientSchemaConfig(
            client_name="custom",
            canonical_id_field="user_id",
            past_id_field="old_user_id",
        )
        loader = LocalDataLoader(temp_data_dir, schema_config=config)

        row_data = {"user_id": "user_001", "old_user_id": "old_user_001"}

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        seen: set[str] = set()
        result = loader._load_id_history(mock_df, seen)

        assert len(result) == 1
        assert result[0].internal_customer_id == "user_001"
        assert result[0].past_id == "old_user_001"


# =============================================================================
# TESTS: LocalDataLoader._load_merge_events
# =============================================================================


class TestLoadMergeEvents:
    """Tests for loading merge events."""

    def test_loads_merge_events_with_list_source_ids(self, temp_data_dir):
        """Should load merge events with list source_internal_ids."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": {
                "source_internal_ids": ["old_001", "old_002"]
            }
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_merge_events(mock_df)

        assert len(result) == 2

    def test_loads_merge_events_with_json_string_source_ids(self, temp_data_dir):
        """Should parse JSON string source_internal_ids."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": {
                "source_internal_ids": '["old_001", "old_002"]'
            }
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_merge_events(mock_df)

        assert len(result) == 2

    def test_handles_non_json_string_source_id(self, temp_data_dir):
        """Should handle non-JSON string as single source ID."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": {
                "source_internal_ids": "old_001"  # Plain string, not JSON
            }
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_merge_events(mock_df)

        assert len(result) == 1
        assert result[0].past_id == "old_001"

    def test_skips_same_id_as_current(self, temp_data_dir):
        """Should skip source_id that equals current_id."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": {
                "source_internal_ids": ["cust_001", "old_001"]  # First equals current
            }
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_merge_events(mock_df)

        assert len(result) == 1
        assert result[0].past_id == "old_001"

    def test_deduplicates_across_calls(self, temp_data_dir):
        """Should deduplicate using shared seen set."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": {
                "source_internal_ids": ["old_001"]
            }
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        seen: set[str] = {"old_001"}  # Already seen
        result = loader._load_merge_events(mock_df, seen)

        assert len(result) == 0

    def test_handles_missing_properties(self, temp_data_dir):
        """Should skip rows with missing properties."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": None
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_merge_events(mock_df)

        assert len(result) == 0

    def test_handles_missing_source_ids(self, temp_data_dir):
        """Should skip when source_internal_ids is missing."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": {}  # No source_internal_ids
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_merge_events(mock_df)

        assert len(result) == 0

    def test_flat_structure_without_properties_field(self, temp_data_dir):
        """Should use dict(row) when properties_field is None."""
        config = ClientSchemaConfig(
            client_name="flat",
            properties_field=None,
        )
        loader = LocalDataLoader(temp_data_dir, schema_config=config)

        row_data = {
            "internal_customer_id": "cust_001",
            "source_internal_ids": ["old_001"],
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))

        # Make dict(row) work
        mock_row.__iter__ = MagicMock(return_value=iter(row_data.keys()))
        mock_row.keys = MagicMock(return_value=row_data.keys())
        mock_row.values = MagicMock(return_value=row_data.values())
        mock_row.items = MagicMock(return_value=row_data.items())

        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        # Patch dict to return row_data when called with mock_row
        original_dict = dict
        def patched_dict(x):
            if hasattr(x, 'get') and x is mock_row:
                return row_data
            return original_dict(x)

        with patch("builtins.dict", patched_dict):
            result = loader._load_merge_events(mock_df)

        assert len(result) == 1


# =============================================================================
# TESTS: LocalDataLoader._load_customer_properties
# =============================================================================


class TestLoadCustomerProperties:
    """Tests for loading customer properties."""

    def test_loads_customer_properties(self, temp_data_dir):
        """Should load customer properties."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": {
                "email": "test@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "country": "US",
            }
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_customer_properties(mock_df)

        assert "cust_001" in result
        assert result["cust_001"]["email"] == "test@example.com"
        assert result["cust_001"]["first_name"] == "John"

    def test_maps_property_names(self, temp_data_dir):
        """Should map Bloomreach property names to canonical names."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": {
                "newsletter__f84ca2e7": True,
                "rfm_today__b3103f82": "VIP",
            }
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_customer_properties(mock_df)

        assert "newsletter" in result["cust_001"]
        assert "rfm_segment" in result["cust_001"]

    def test_handles_internal_id_field_alternative(self, temp_data_dir):
        """Should try internal_id as alternative to internal_customer_id."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_id": "cust_001",  # Alternative field name
            "properties": {"email": "test@example.com"}
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_customer_properties(mock_df)

        assert "cust_001" in result

    def test_skips_empty_properties(self, temp_data_dir):
        """Should skip rows with empty extracted properties."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": {"unknown_field": "value"}  # No known properties
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_customer_properties(mock_df)

        assert "cust_001" not in result

    def test_skips_none_values(self, temp_data_dir):
        """Should skip properties with None values."""
        loader = LocalDataLoader(temp_data_dir)

        row_data = {
            "internal_customer_id": "cust_001",
            "properties": {
                "email": "test@example.com",
                "first_name": None,  # Should be skipped
            }
        }

        mock_df = MagicMock()
        mock_row = MagicMock()
        mock_row.get = MagicMock(side_effect=lambda k, d=None: row_data.get(k, d))
        mock_df.iterrows = MagicMock(return_value=iter([(0, mock_row)]))

        result = loader._load_customer_properties(mock_df)

        assert "first_name" not in result["cust_001"]


# =============================================================================
# TESTS: LocalDataLoader.load_metadata
# =============================================================================


class TestLoadMetadata:
    """Tests for loading metadata."""

    def test_loads_metadata_file(self, temp_data_dir):
        """Should load metadata.json if present."""
        metadata = {"extraction_date": "2024-01-15", "sample_size": 10000}

        with open(temp_data_dir / "metadata.json", "w") as f:
            json.dump(metadata, f)

        loader = LocalDataLoader(temp_data_dir)
        result = loader.load_metadata()

        assert result["extraction_date"] == "2024-01-15"
        assert result["sample_size"] == 10000

    def test_returns_empty_dict_without_metadata(self, temp_data_dir):
        """Should return empty dict if metadata.json doesn't exist."""
        loader = LocalDataLoader(temp_data_dir)
        result = loader.load_metadata()

        assert result == {}


# =============================================================================
# TESTS: LocalDataLoader.list_tables
# =============================================================================


class TestListTables:
    """Tests for listing tables."""

    def test_lists_parquet_files(self, temp_data_dir):
        """Should list parquet files without extension."""
        (temp_data_dir / "purchase.parquet").touch()
        (temp_data_dir / "view_item.parquet").touch()
        (temp_data_dir / "not_parquet.txt").touch()

        loader = LocalDataLoader(temp_data_dir)
        tables = loader.list_tables()

        assert "purchase" in tables
        assert "view_item" in tables
        assert "not_parquet" not in tables

    def test_empty_directory(self, temp_data_dir):
        """Should return empty list for empty directory."""
        loader = LocalDataLoader(temp_data_dir)
        tables = loader.list_tables()

        assert tables == []


# =============================================================================
# TESTS: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_load_local_data_creates_loader(self, temp_data_dir):
        """load_local_data should create LocalDataLoader and call load()."""
        result = load_local_data(temp_data_dir)

        assert isinstance(result, LoadResult)

    def test_load_local_data_with_schema_config(self, temp_data_dir):
        """load_local_data should accept schema_config."""
        ga4_config = create_ga4_config()
        result = load_local_data(temp_data_dir, schema_config=ga4_config)

        assert isinstance(result, LoadResult)

    def test_load_events_only_returns_tuple(self, temp_data_dir):
        """load_events_only should return tuple of (events, id_history)."""
        events, id_history = load_events_only(temp_data_dir)

        assert isinstance(events, list)
        assert isinstance(id_history, list)

    def test_load_events_only_with_schema_config(self, temp_data_dir):
        """load_events_only should accept schema_config."""
        ga4_config = create_ga4_config()
        events, id_history = load_events_only(temp_data_dir, schema_config=ga4_config)

        assert isinstance(events, list)
        assert isinstance(id_history, list)


# =============================================================================
# TESTS: Integration
# =============================================================================


class TestIntegration:
    """Integration tests with real (mocked) data flow."""

    def test_full_load_flow(self, temp_data_dir):
        """Test complete load flow with mocked pandas."""
        # Create dummy parquet files
        (temp_data_dir / "purchase.parquet").touch()
        (temp_data_dir / "customers_id_history.parquet").touch()

        loader = LocalDataLoader(temp_data_dir)

        # Mock pandas and its read_parquet method
        with patch.object(loader, "_get_pandas") as mock_get_pandas:
            mock_pd = MagicMock()
            mock_get_pandas.return_value = mock_pd

            # Setup purchase DataFrame
            purchase_row = {
                "internal_customer_id": "cust_001",
                "timestamp": datetime(2024, 1, 15, 10, 30, tzinfo=timezone.utc),
                "properties": {"total_price": "99.99", "purchase_id": "order_123"}
            }

            purchase_df = MagicMock()
            purchase_df.__len__ = MagicMock(return_value=1)
            purchase_mock_row = MagicMock()
            purchase_mock_row.get = MagicMock(side_effect=lambda k, d=None: purchase_row.get(k, d))
            purchase_df.iterrows = MagicMock(return_value=iter([(0, purchase_mock_row)]))

            # Setup ID history DataFrame
            id_history_row = {"internal_customer_id": "cust_001", "past_id": "old_001"}
            id_history_df = MagicMock()
            id_history_df.__len__ = MagicMock(return_value=1)
            id_mock_row = MagicMock()
            id_mock_row.get = MagicMock(side_effect=lambda k, d=None: id_history_row.get(k, d))
            id_history_df.iterrows = MagicMock(return_value=iter([(0, id_mock_row)]))

            def read_parquet_side_effect(path):
                if "purchase" in str(path):
                    return purchase_df
                elif "id_history" in str(path):
                    return id_history_df
                raise ValueError(f"Unexpected path: {path}")

            mock_pd.read_parquet = MagicMock(side_effect=read_parquet_side_effect)

            result = loader.load()

            assert len(result.tables_loaded) == 2
            assert result.total_rows == 2

    def test_loads_merge_table(self, temp_data_dir):
        """Test loading merge events table."""
        (temp_data_dir / "merge.parquet").touch()

        loader = LocalDataLoader(temp_data_dir)

        with patch.object(loader, "_get_pandas") as mock_get_pandas:
            mock_pd = MagicMock()
            mock_get_pandas.return_value = mock_pd

            # Setup merge DataFrame
            merge_row = {
                "internal_customer_id": "cust_001",
                "properties": {"source_internal_ids": ["old_001", "old_002"]}
            }

            merge_df = MagicMock()
            merge_df.__len__ = MagicMock(return_value=1)
            merge_mock_row = MagicMock()
            merge_mock_row.get = MagicMock(side_effect=lambda k, d=None: merge_row.get(k, d))
            merge_df.iterrows = MagicMock(return_value=iter([(0, merge_mock_row)]))

            mock_pd.read_parquet.return_value = merge_df

            result = loader.load()

            assert "merge" in result.tables_loaded
            assert len(result.id_history) == 2

    def test_loads_customers_properties_table(self, temp_data_dir):
        """Test loading customers_properties table."""
        (temp_data_dir / "customers_properties.parquet").touch()

        loader = LocalDataLoader(temp_data_dir)

        with patch.object(loader, "_get_pandas") as mock_get_pandas:
            mock_pd = MagicMock()
            mock_get_pandas.return_value = mock_pd

            # Setup customer properties DataFrame
            props_row = {
                "internal_customer_id": "cust_001",
                "properties": {
                    "email": "test@example.com",
                    "first_name": "John",
                }
            }

            props_df = MagicMock()
            props_df.__len__ = MagicMock(return_value=1)
            props_mock_row = MagicMock()
            props_mock_row.get = MagicMock(side_effect=lambda k, d=None: props_row.get(k, d))
            props_df.iterrows = MagicMock(return_value=iter([(0, props_mock_row)]))

            mock_pd.read_parquet.return_value = props_df

            result = loader.load()

            assert "customers_properties" in result.tables_loaded
            assert "cust_001" in result.customer_properties
            assert result.customer_properties["cust_001"]["email"] == "test@example.com"

    def test_handles_invalid_order_total(self, temp_data_dir):
        """Test that invalid order_total values are handled gracefully."""
        loader = LocalDataLoader(temp_data_dir)

        props = {"total_price": "not_a_number"}

        result = loader._build_event_properties(props, EventType.PURCHASE)

        assert result.order_total is None


# =============================================================================
# TESTS: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_handles_parquet_read_error(self, temp_data_dir):
        """Should handle errors when reading parquet files."""
        (temp_data_dir / "corrupt.parquet").touch()

        loader = LocalDataLoader(temp_data_dir)

        with patch.object(loader, "_get_pandas") as mock_get_pandas:
            mock_pd = MagicMock()
            mock_get_pandas.return_value = mock_pd
            mock_pd.read_parquet.side_effect = Exception("Corrupt file")

            result = loader.load()

            assert len(result.errors) == 1
            assert "corrupt" in result.errors[0].lower()

    def test_handles_row_conversion_error(self, temp_data_dir):
        """Should handle errors when converting rows to events."""
        (temp_data_dir / "purchase.parquet").touch()

        loader = LocalDataLoader(temp_data_dir)

        with patch.object(loader, "_get_pandas") as mock_get_pandas:
            mock_pd = MagicMock()
            mock_get_pandas.return_value = mock_pd

            # Mock DataFrame that raises error during iteration
            mock_df = MagicMock()
            mock_df.__len__ = MagicMock(return_value=1)

            # First row raises, second row succeeds
            row1 = MagicMock()
            row1.get = MagicMock(side_effect=Exception("Conversion error"))

            row2_data = {
                "internal_customer_id": "cust_001",
                "timestamp": datetime.now(tz=timezone.utc),
                "properties": {}
            }
            row2 = MagicMock()
            row2.get = MagicMock(side_effect=lambda k, d=None: row2_data.get(k, d))

            mock_df.iterrows = MagicMock(return_value=iter([(0, row1), (1, row2)]))
            mock_pd.read_parquet.return_value = mock_df

            result = loader.load()

            # Should continue despite first row error
            assert "purchase" in result.tables_loaded

    def test_handles_none_properties(self, temp_data_dir):
        """Should handle None properties gracefully."""
        loader = LocalDataLoader(temp_data_dir)

        row = MagicMock()
        row.get = MagicMock(side_effect=lambda k, d=None: {
            "internal_customer_id": "cust_001",
            "timestamp": datetime.now(tz=timezone.utc),
            "properties": None,
        }.get(k, d))

        result = loader._row_to_event(row, EventType.VIEW_ITEM)

        assert result is not None
        assert result.properties is not None
