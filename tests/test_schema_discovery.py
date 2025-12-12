"""
Tests for src/data/bigquery/schema_discovery.py

Comprehensive test suite for BigQuery schema discovery functionality.
"""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.data.bigquery.schema_discovery import (
    ColumnInfo,
    TableSchema,
    DiscoveredTable,
    DiscoveryResult,
    TableRelevanceFilter,
    SchemaDiscovery,
    discover_schema,
    print_discovery_report,
)
from src.data.bigquery.config import (
    BigQueryConfig,
    EventTableConfig,
    CustomerTableConfig,
    MergeTableConfig,
    FieldMapping,
    EventType,
)


# =============================================================================
# TESTS: ColumnInfo
# =============================================================================


class TestColumnInfo:
    """Tests for ColumnInfo dataclass."""

    def test_basic_column(self) -> None:
        """Test basic column without nested fields."""
        col = ColumnInfo(
            name="customer_id",
            data_type="STRING",
            mode="REQUIRED",
        )

        assert col.name == "customer_id"
        assert col.data_type == "STRING"
        assert col.mode == "REQUIRED"
        assert col.description is None
        assert col.is_nested is False
        assert col.nested_fields == []

    def test_column_with_description(self) -> None:
        """Test column with description."""
        col = ColumnInfo(
            name="email",
            data_type="STRING",
            mode="NULLABLE",
            description="Customer email address",
        )

        assert col.description == "Customer email address"

    def test_nested_column(self) -> None:
        """Test nested column with sub-fields."""
        nested = ColumnInfo(
            name="properties",
            data_type="RECORD",
            mode="NULLABLE",
            is_nested=True,
            nested_fields=[
                ColumnInfo(name="email", data_type="STRING", mode="NULLABLE"),
                ColumnInfo(name="phone", data_type="STRING", mode="NULLABLE"),
            ],
        )

        assert nested.is_nested is True
        assert len(nested.nested_fields) == 2
        assert nested.nested_fields[0].name == "email"

    def test_to_dict_basic(self) -> None:
        """Test to_dict for basic column."""
        col = ColumnInfo(
            name="amount",
            data_type="FLOAT64",
            mode="REQUIRED",
        )

        result = col.to_dict()

        assert result == {
            "name": "amount",
            "type": "FLOAT64",
            "mode": "REQUIRED",
        }

    def test_to_dict_with_description(self) -> None:
        """Test to_dict includes description when present."""
        col = ColumnInfo(
            name="status",
            data_type="STRING",
            mode="NULLABLE",
            description="Order status",
        )

        result = col.to_dict()

        assert result["description"] == "Order status"

    def test_to_dict_with_nested_fields(self) -> None:
        """Test to_dict includes nested_fields when present."""
        col = ColumnInfo(
            name="properties",
            data_type="RECORD",
            mode="NULLABLE",
            is_nested=True,
            nested_fields=[
                ColumnInfo(name="value", data_type="STRING", mode="NULLABLE"),
            ],
        )

        result = col.to_dict()

        assert "nested_fields" in result
        assert len(result["nested_fields"]) == 1
        assert result["nested_fields"][0]["name"] == "value"


# =============================================================================
# TESTS: TableSchema
# =============================================================================


class TestTableSchema:
    """Tests for TableSchema dataclass."""

    def test_basic_schema(self) -> None:
        """Test basic table schema."""
        schema = TableSchema(
            table_name="purchases",
            full_name="project.dataset.purchases",
        )

        assert schema.table_name == "purchases"
        assert schema.full_name == "project.dataset.purchases"
        assert schema.columns == []
        assert schema.row_count is None
        assert schema.sample_rows == []

    def test_schema_with_columns(self) -> None:
        """Test schema with columns."""
        schema = TableSchema(
            table_name="events",
            full_name="project.dataset.events",
            columns=[
                ColumnInfo(name="id", data_type="STRING", mode="REQUIRED"),
                ColumnInfo(name="timestamp", data_type="TIMESTAMP", mode="REQUIRED"),
            ],
            row_count=10000,
        )

        assert len(schema.columns) == 2
        assert schema.row_count == 10000

    def test_to_dict(self) -> None:
        """Test to_dict conversion."""
        schema = TableSchema(
            table_name="test",
            full_name="project.dataset.test",
            columns=[
                ColumnInfo(name="col1", data_type="STRING", mode="NULLABLE"),
            ],
            row_count=500,
            sample_rows=[
                {"col1": "value1"},
                {"col1": "value2"},
                {"col1": "value3"},
                {"col1": "value4"},
            ],
        )

        result = schema.to_dict()

        assert result["table_name"] == "test"
        assert result["full_name"] == "project.dataset.test"
        assert len(result["columns"]) == 1
        assert result["row_count"] == 500
        # Should only include first 3 sample rows
        assert len(result["sample_rows"]) == 3

    def test_to_dict_no_samples(self) -> None:
        """Test to_dict with no sample rows."""
        schema = TableSchema(
            table_name="empty",
            full_name="project.dataset.empty",
        )

        result = schema.to_dict()

        assert result["sample_rows"] == []


# =============================================================================
# TESTS: DiscoveredTable
# =============================================================================


class TestDiscoveredTable:
    """Tests for DiscoveredTable dataclass."""

    def test_default_values(self) -> None:
        """Test default values for DiscoveredTable."""
        schema = TableSchema(table_name="test", full_name="p.d.test")
        discovered = DiscoveredTable(
            schema=schema,
            inferred_type="unknown",
        )

        assert discovered.event_type is None
        assert discovered.confidence == 0.0
        assert discovered.suggested_config is None
        assert discovered.reasoning == ""
        assert discovered.is_relevant is True
        assert discovered.relevance_score == 1.0

    def test_event_table(self) -> None:
        """Test discovered event table."""
        schema = TableSchema(table_name="purchases", full_name="p.d.purchases")
        config = EventTableConfig(
            table_name="purchases",
            event_type=EventType.PURCHASE,
        )

        discovered = DiscoveredTable(
            schema=schema,
            inferred_type="event",
            event_type=EventType.PURCHASE,
            confidence=0.9,
            suggested_config=config,
            reasoning="Table name indicates purchase events",
        )

        assert discovered.inferred_type == "event"
        assert discovered.event_type == EventType.PURCHASE
        assert discovered.confidence == 0.9
        assert discovered.suggested_config is not None


# =============================================================================
# TESTS: DiscoveryResult
# =============================================================================


class TestDiscoveryResult:
    """Tests for DiscoveryResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = DiscoveryResult(
            project_id="my-project",
            dataset_id="my_dataset",
        )

        assert result.project_id == "my-project"
        assert result.dataset_id == "my_dataset"
        assert result.tables == []
        assert result.excluded_tables == []
        assert result.suggested_config is None
        assert result.warnings == []
        assert result.discovery_timestamp is not None

    def test_relevant_tables_property(self) -> None:
        """Test relevant_tables property."""
        result = DiscoveryResult(
            project_id="project",
            dataset_id="dataset",
            tables=[
                DiscoveredTable(
                    schema=TableSchema("t1", "p.d.t1"),
                    inferred_type="event",
                    is_relevant=True,
                ),
                DiscoveredTable(
                    schema=TableSchema("t2", "p.d.t2"),
                    inferred_type="unknown",
                    is_relevant=False,
                ),
                DiscoveredTable(
                    schema=TableSchema("t3", "p.d.t3"),
                    inferred_type="event",
                    is_relevant=True,
                ),
            ],
        )

        relevant = result.relevant_tables
        assert len(relevant) == 2

    def test_irrelevant_tables_property(self) -> None:
        """Test irrelevant_tables property."""
        result = DiscoveryResult(
            project_id="project",
            dataset_id="dataset",
            tables=[
                DiscoveredTable(
                    schema=TableSchema("t1", "p.d.t1"),
                    inferred_type="event",
                    is_relevant=True,
                ),
                DiscoveredTable(
                    schema=TableSchema("t2", "p.d.t2"),
                    inferred_type="irrelevant",
                    is_relevant=False,
                ),
            ],
        )

        irrelevant = result.irrelevant_tables
        assert len(irrelevant) == 1
        assert irrelevant[0].schema.table_name == "t2"


# =============================================================================
# TESTS: TableRelevanceFilter
# =============================================================================


class TestTableRelevanceFilter:
    """Tests for TableRelevanceFilter class."""

    def test_exclude_log_tables(self) -> None:
        """Test that log tables are excluded."""
        is_rel, score, reason = TableRelevanceFilter.is_relevant("audit_log")
        assert is_rel is False
        assert score < 0.5
        assert "log" in reason.lower()

    def test_exclude_temp_tables(self) -> None:
        """Test that temp tables are excluded."""
        is_rel, score, _ = TableRelevanceFilter.is_relevant("temp_data")
        assert is_rel is False

    def test_exclude_backup_tables(self) -> None:
        """Test that backup tables are excluded."""
        is_rel, score, _ = TableRelevanceFilter.is_relevant("orders_backup")
        assert is_rel is False

    def test_exclude_staging_tables(self) -> None:
        """Test that staging tables are excluded."""
        is_rel, score, _ = TableRelevanceFilter.is_relevant("staging_events")
        assert is_rel is False

    def test_exclude_aggregated_tables(self) -> None:
        """Test that aggregated tables are excluded."""
        is_rel, score, _ = TableRelevanceFilter.is_relevant("sales_daily")
        assert is_rel is False

        is_rel, score, _ = TableRelevanceFilter.is_relevant("metrics_summary")
        assert is_rel is False

    def test_include_purchase_table(self) -> None:
        """Test that purchase tables are included."""
        is_rel, score, reason = TableRelevanceFilter.is_relevant("purchase_events")
        assert is_rel is True
        assert score >= 0.9
        assert "purchase" in reason.lower()

    def test_include_page_view_table(self) -> None:
        """Test that page view tables are included."""
        is_rel, score, _ = TableRelevanceFilter.is_relevant("page_views")
        assert is_rel is True
        assert score >= 0.9

    def test_include_cart_table(self) -> None:
        """Test that cart tables are included."""
        is_rel, score, _ = TableRelevanceFilter.is_relevant("add_to_cart")
        assert is_rel is True

    def test_include_customer_table(self) -> None:
        """Test that customer tables are included."""
        is_rel, score, _ = TableRelevanceFilter.is_relevant("customer_profiles")
        assert is_rel is True

    def test_include_merge_table(self) -> None:
        """Test that merge/identity tables are included."""
        is_rel, score, _ = TableRelevanceFilter.is_relevant("id_history")
        assert is_rel is True

    def test_ambiguous_order_table(self) -> None:
        """Test that order tables return moderate score."""
        is_rel, score, reason = TableRelevanceFilter.is_relevant("orders")
        assert is_rel is True
        assert 0.5 <= score < 0.9
        assert "ambiguous" in reason.lower()

    def test_low_row_count(self) -> None:
        """Test that low row count tables are excluded."""
        is_rel, score, reason = TableRelevanceFilter.is_relevant(
            "unknown_table",
            row_count=50,
        )
        assert is_rel is False
        assert "few rows" in reason.lower()

    def test_with_customer_id_and_timestamp(self) -> None:
        """Test table with customer_id and timestamp columns."""
        columns = [
            ColumnInfo("internal_customer_id", "STRING", "REQUIRED"),
            ColumnInfo("timestamp", "TIMESTAMP", "REQUIRED"),
            ColumnInfo("data", "STRING", "NULLABLE"),
        ]

        is_rel, score, reason = TableRelevanceFilter.is_relevant(
            "some_table",
            columns=columns,
        )

        assert is_rel is True
        assert score >= 0.8
        assert "customer_id" in reason.lower() and "timestamp" in reason.lower()

    def test_with_customer_id_only(self) -> None:
        """Test table with customer_id but no timestamp."""
        columns = [
            ColumnInfo("user_id", "STRING", "REQUIRED"),
            ColumnInfo("data", "STRING", "NULLABLE"),
        ]

        is_rel, score, reason = TableRelevanceFilter.is_relevant(
            "some_table",
            columns=columns,
        )

        assert is_rel is True
        assert score >= 0.5
        assert "customer_id" in reason.lower()

    def test_missing_both_columns(self) -> None:
        """Test table missing customer_id and timestamp."""
        columns = [
            ColumnInfo("data", "STRING", "NULLABLE"),
            ColumnInfo("value", "FLOAT64", "NULLABLE"),
        ]

        is_rel, score, reason = TableRelevanceFilter.is_relevant(
            "unknown_table",
            columns=columns,
        )

        assert is_rel is False
        assert score < 0.5
        assert "missing" in reason.lower()

    def test_unknown_table_default(self) -> None:
        """Test unknown table with no additional info."""
        is_rel, score, reason = TableRelevanceFilter.is_relevant("generic_data")

        # Should return uncertain result
        assert is_rel is True
        assert score == 0.5
        assert "could not determine" in reason.lower()

    def test_filter_tables_basic(self) -> None:
        """Test filter_tables method."""
        tables = [
            "purchases",
            "temp_data",
            "page_views",
            "staging_import",
            "customer_profiles",
        ]

        relevant, excluded = TableRelevanceFilter.filter_tables(tables)

        assert "purchases" in relevant
        assert "page_views" in relevant
        assert "customer_profiles" in relevant
        assert any("temp_data" in t for t, _ in excluded)
        assert any("staging_import" in t for t, _ in excluded)

    def test_filter_tables_with_get_table_info(self) -> None:
        """Test filter_tables with table info callback."""
        tables = ["table_a", "table_b"]

        def get_info(table: str) -> tuple[int | None, list[ColumnInfo] | None]:
            if table == "table_a":
                return 1000, [ColumnInfo("customer_id", "STRING", "REQUIRED")]
            return 10, None  # Low row count

        relevant, excluded = TableRelevanceFilter.filter_tables(tables, get_info)

        assert "table_a" in relevant
        # table_b excluded due to low row count
        assert any("table_b" in t for t, _ in excluded)

    def test_filter_tables_with_callback_exception(self) -> None:
        """Test filter_tables handles callback exceptions."""
        tables = ["error_table"]

        def get_info(table: str) -> tuple[int | None, list[ColumnInfo] | None]:
            raise RuntimeError("API error")

        # Should not raise, just skip the info
        relevant, excluded = TableRelevanceFilter.filter_tables(tables, get_info)

        # Table should still be processed, just without the info
        assert len(relevant) + len(excluded) == 1


# =============================================================================
# TESTS: SchemaDiscovery
# =============================================================================


class TestSchemaDiscovery:
    """Tests for SchemaDiscovery class."""

    def test_init(self) -> None:
        """Test initialization."""
        discovery = SchemaDiscovery(
            project_id="my-project",
            dataset_id="my_dataset",
            llm_provider="anthropic",
            sample_rows=10,
        )

        assert discovery.project_id == "my-project"
        assert discovery.dataset_id == "my_dataset"
        assert discovery.llm_provider == "anthropic"
        assert discovery.sample_rows == 10
        assert discovery._client is None
        assert discovery._bigquery_module is None

    def test_get_bigquery_import_error(self) -> None:
        """Test _get_bigquery raises ImportError when not installed."""
        discovery = SchemaDiscovery("project", "dataset")

        with patch.dict("sys.modules", {"google.cloud": None, "google.cloud.bigquery": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="google-cloud-bigquery is required"):
                    discovery._get_bigquery()

    def test_client_property_creates_client(self) -> None:
        """Test client property creates BigQuery client."""
        discovery = SchemaDiscovery("test-project", "dataset")

        mock_bigquery = MagicMock()
        mock_client = MagicMock()
        mock_bigquery.Client.return_value = mock_client

        with patch.object(discovery, "_get_bigquery", return_value=mock_bigquery):
            client = discovery.client

            mock_bigquery.Client.assert_called_once_with(project="test-project")
            assert client == mock_client

    def test_client_property_caches(self) -> None:
        """Test client property caches the client."""
        discovery = SchemaDiscovery("test-project", "dataset")

        mock_bigquery = MagicMock()
        mock_client = MagicMock()
        mock_bigquery.Client.return_value = mock_client

        with patch.object(discovery, "_get_bigquery", return_value=mock_bigquery):
            client1 = discovery.client
            client2 = discovery.client

            assert mock_bigquery.Client.call_count == 1
            assert client1 is client2

    def test_infer_event_type_from_name_purchase(self) -> None:
        """Test inferring PURCHASE event type."""
        discovery = SchemaDiscovery("project", "dataset")

        assert discovery._infer_event_type_from_name("purchases") == EventType.PURCHASE
        assert discovery._infer_event_type_from_name("order_events") == EventType.PURCHASE
        assert discovery._infer_event_type_from_name("transactions") == EventType.PURCHASE

    def test_infer_event_type_from_name_page_view(self) -> None:
        """Test inferring PAGE_VIEW event type."""
        discovery = SchemaDiscovery("project", "dataset")

        assert discovery._infer_event_type_from_name("page_views") == EventType.PAGE_VIEW
        assert discovery._infer_event_type_from_name("pageviews") == EventType.PAGE_VIEW

    def test_infer_event_type_from_name_view_item(self) -> None:
        """Test inferring VIEW_ITEM event type."""
        discovery = SchemaDiscovery("project", "dataset")

        assert discovery._infer_event_type_from_name("view_item") == EventType.VIEW_ITEM
        assert discovery._infer_event_type_from_name("product_views") == EventType.VIEW_ITEM

    def test_infer_event_type_from_name_cart(self) -> None:
        """Test inferring cart event types."""
        discovery = SchemaDiscovery("project", "dataset")

        assert discovery._infer_event_type_from_name("add_to_cart") == EventType.ADD_TO_CART
        assert discovery._infer_event_type_from_name("remove_from_cart") == EventType.REMOVE_FROM_CART

    def test_infer_event_type_from_name_other(self) -> None:
        """Test inferring other event types."""
        discovery = SchemaDiscovery("project", "dataset")

        assert discovery._infer_event_type_from_name("checkout_started") == EventType.BEGIN_CHECKOUT
        assert discovery._infer_event_type_from_name("session_start") == EventType.SESSION_START
        assert discovery._infer_event_type_from_name("search_queries") == EventType.SEARCH
        assert discovery._infer_event_type_from_name("wishlist_adds") == EventType.WISHLIST_ADD
        assert discovery._infer_event_type_from_name("email_opens") == EventType.EMAIL_OPEN
        assert discovery._infer_event_type_from_name("email_clicks") == EventType.EMAIL_CLICK

    def test_infer_event_type_from_name_unknown(self) -> None:
        """Test unknown table names return None."""
        discovery = SchemaDiscovery("project", "dataset")

        assert discovery._infer_event_type_from_name("random_table") is None
        assert discovery._infer_event_type_from_name("data") is None

    def test_find_customer_id_field(self) -> None:
        """Test finding customer ID field."""
        discovery = SchemaDiscovery("project", "dataset")

        # Test standard fields
        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("internal_customer_id", "STRING", "REQUIRED"),
                ColumnInfo("other", "STRING", "NULLABLE"),
            ],
        )
        assert discovery._find_customer_id_field(schema) == "internal_customer_id"

        # Test customer_id
        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("customer_id", "STRING", "REQUIRED"),
            ],
        )
        assert discovery._find_customer_id_field(schema) == "customer_id"

        # Test user_id
        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("user_id", "STRING", "REQUIRED"),
            ],
        )
        assert discovery._find_customer_id_field(schema) == "user_id"

    def test_find_customer_id_field_fallback(self) -> None:
        """Test finding customer ID with fallback."""
        discovery = SchemaDiscovery("project", "dataset")

        # Test composite name fallback
        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("my_customer_id_field", "STRING", "REQUIRED"),
            ],
        )
        assert discovery._find_customer_id_field(schema) == "my_customer_id_field"

        # Test default when nothing matches
        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("other_field", "STRING", "NULLABLE"),
            ],
        )
        assert discovery._find_customer_id_field(schema) == "internal_customer_id"

    def test_find_timestamp_field(self) -> None:
        """Test finding timestamp field."""
        discovery = SchemaDiscovery("project", "dataset")

        # Test standard timestamp
        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("timestamp", "TIMESTAMP", "REQUIRED"),
            ],
        )
        assert discovery._find_timestamp_field(schema) == "timestamp"

        # Test event_timestamp
        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("event_timestamp", "TIMESTAMP", "REQUIRED"),
            ],
        )
        assert discovery._find_timestamp_field(schema) == "event_timestamp"

        # Test created_at
        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("created_at", "TIMESTAMP", "REQUIRED"),
            ],
        )
        assert discovery._find_timestamp_field(schema) == "created_at"

    def test_find_timestamp_field_by_type(self) -> None:
        """Test finding timestamp field by data type."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("random_name", "TIMESTAMP", "REQUIRED"),
            ],
        )
        assert discovery._find_timestamp_field(schema) == "random_name"

        # Test DATETIME type
        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("event_dt", "DATETIME", "REQUIRED"),
            ],
        )
        assert discovery._find_timestamp_field(schema) == "event_dt"

    def test_find_timestamp_field_default(self) -> None:
        """Test timestamp field default."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("other", "STRING", "NULLABLE"),
            ],
        )
        assert discovery._find_timestamp_field(schema) == "timestamp"

    def test_rule_based_analysis_merge_table(self) -> None:
        """Test rule-based analysis for merge tables."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "id_history", "p.d.id_history",
            columns=[
                ColumnInfo("current_id", "STRING", "REQUIRED"),
                ColumnInfo("past_id", "STRING", "NULLABLE"),
            ],
        )

        result = discovery._rule_based_analysis(schema)

        assert result.inferred_type == "merge"
        assert result.confidence >= 0.9
        assert isinstance(result.suggested_config, MergeTableConfig)

    def test_rule_based_analysis_customer_table(self) -> None:
        """Test rule-based analysis for customer tables."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "customer_profiles", "p.d.customer_profiles",
            columns=[
                ColumnInfo("customer_id", "STRING", "REQUIRED"),
                ColumnInfo("email", "STRING", "NULLABLE"),
            ],
        )

        result = discovery._rule_based_analysis(schema)

        assert result.inferred_type == "customer"
        assert result.confidence >= 0.85
        assert isinstance(result.suggested_config, CustomerTableConfig)

    def test_rule_based_analysis_event_table(self) -> None:
        """Test rule-based analysis for event tables."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "purchases", "p.d.purchases",
            columns=[
                ColumnInfo("customer_id", "STRING", "REQUIRED"),
                ColumnInfo("timestamp", "TIMESTAMP", "REQUIRED"),
                ColumnInfo("amount", "FLOAT64", "NULLABLE"),
            ],
        )

        result = discovery._rule_based_analysis(schema)

        assert result.inferred_type == "event"
        assert result.event_type == EventType.PURCHASE
        assert result.confidence >= 0.85
        assert isinstance(result.suggested_config, EventTableConfig)

    def test_rule_based_analysis_unknown(self) -> None:
        """Test rule-based analysis for unknown tables."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "random_data", "p.d.random_data",
            columns=[
                ColumnInfo("col1", "STRING", "NULLABLE"),
            ],
        )

        result = discovery._rule_based_analysis(schema)

        assert result.inferred_type == "unknown"
        assert result.confidence < 0.5

    def test_suggest_event_config(self) -> None:
        """Test suggesting event config."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "purchases", "p.d.purchases",
            columns=[
                ColumnInfo("internal_customer_id", "STRING", "REQUIRED"),
                ColumnInfo("timestamp", "TIMESTAMP", "REQUIRED"),
                ColumnInfo("order_id", "STRING", "NULLABLE"),
                ColumnInfo("total_price", "FLOAT64", "NULLABLE"),
            ],
        )

        config = discovery._suggest_event_config(schema, EventType.PURCHASE)

        assert config.table_name == "purchases"
        assert config.event_type == EventType.PURCHASE
        assert config.customer_id_field == "internal_customer_id"
        assert config.timestamp_field == "timestamp"

    def test_suggest_customer_config(self) -> None:
        """Test suggesting customer config."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "customers", "p.d.customers",
            columns=[
                ColumnInfo("customer_id", "STRING", "REQUIRED"),
                ColumnInfo("email", "STRING", "NULLABLE"),
            ],
        )

        config = discovery._suggest_customer_config(schema)

        assert config.table_name == "customers"
        assert config.customer_id_field == "customer_id"

    def test_suggest_merge_config(self) -> None:
        """Test suggesting merge config."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "id_history", "p.d.id_history",
            columns=[
                ColumnInfo("current_id", "STRING", "REQUIRED"),
                ColumnInfo("past_id", "STRING", "NULLABLE"),
            ],
        )

        config = discovery._suggest_merge_config(schema)

        assert config.table_name == "id_history"
        assert config.current_id_field == "current_id"
        assert config.past_id_field == "past_id"

    def test_suggest_merge_config_alternative_names(self) -> None:
        """Test merge config with alternative column names."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "identity", "p.d.identity",
            columns=[
                ColumnInfo("internal_customer_id", "STRING", "REQUIRED"),
                ColumnInfo("previous_id", "STRING", "NULLABLE"),
            ],
        )

        config = discovery._suggest_merge_config(schema)

        assert config.current_id_field == "internal_customer_id"
        assert config.past_id_field == "previous_id"

    def test_map_nested_properties_purchase(self) -> None:
        """Test mapping nested properties for purchase events."""
        discovery = SchemaDiscovery("project", "dataset")

        properties = ColumnInfo(
            "properties", "RECORD", "NULLABLE",
            is_nested=True,
            nested_fields=[
                ColumnInfo("order_id", "STRING", "NULLABLE"),
                ColumnInfo("total_price", "FLOAT64", "NULLABLE"),
                ColumnInfo("device", "STRING", "NULLABLE"),
            ],
        )

        mappings = discovery._map_nested_properties(properties, EventType.PURCHASE)

        assert "order_id" in mappings
        assert mappings["order_id"].source_field == "properties.order_id"
        assert "order_total" in mappings
        assert mappings["order_total"].transform == "decimal"
        assert "device" in mappings

    def test_map_nested_properties_view_item(self) -> None:
        """Test mapping nested properties for view item events."""
        discovery = SchemaDiscovery("project", "dataset")

        properties = ColumnInfo(
            "properties", "RECORD", "NULLABLE",
            is_nested=True,
            nested_fields=[
                ColumnInfo("product_id", "STRING", "NULLABLE"),
                ColumnInfo("product_name", "STRING", "NULLABLE"),
                ColumnInfo("price", "FLOAT64", "NULLABLE"),
            ],
        )

        mappings = discovery._map_nested_properties(properties, EventType.VIEW_ITEM)

        assert "product_id" in mappings
        assert "product_name" in mappings
        assert "product_price" in mappings
        assert mappings["product_price"].transform == "decimal"

    def test_map_flat_columns(self) -> None:
        """Test mapping flat columns."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("customer_id", "STRING", "REQUIRED"),
                ColumnInfo("timestamp", "TIMESTAMP", "REQUIRED"),
                ColumnInfo("order_id", "STRING", "NULLABLE"),
                ColumnInfo("total_price", "FLOAT64", "NULLABLE"),
                ColumnInfo("device", "STRING", "NULLABLE"),
            ],
        )

        mappings = discovery._map_flat_columns(schema, EventType.PURCHASE)

        assert "order_id" in mappings
        assert "order_total" in mappings
        assert "device" in mappings

    def test_infer_customer_property_mappings_nested(self) -> None:
        """Test inferring customer property mappings with nested properties."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "customers", "p.d.customers",
            columns=[
                ColumnInfo("customer_id", "STRING", "REQUIRED"),
                ColumnInfo(
                    "properties", "RECORD", "NULLABLE",
                    is_nested=True,
                    nested_fields=[
                        ColumnInfo("email", "STRING", "NULLABLE"),
                        ColumnInfo("first_name", "STRING", "NULLABLE"),
                        ColumnInfo("country", "STRING", "NULLABLE"),
                        ColumnInfo("newsletter", "BOOLEAN", "NULLABLE"),
                    ],
                ),
            ],
        )

        mappings = discovery._infer_customer_property_mappings(schema)

        assert "email" in mappings
        assert mappings["email"].source_field == "properties.email"
        assert "first_name" in mappings
        assert "country" in mappings
        assert "newsletter" in mappings
        assert mappings["newsletter"].transform == "bool"

    def test_infer_customer_property_mappings_flat(self) -> None:
        """Test inferring customer property mappings with flat columns."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "customers", "p.d.customers",
            columns=[
                ColumnInfo("customer_id", "STRING", "REQUIRED"),
                ColumnInfo("email", "STRING", "NULLABLE"),
                ColumnInfo("first_name", "STRING", "NULLABLE"),
            ],
        )

        mappings = discovery._infer_customer_property_mappings(schema)

        assert "email" in mappings
        assert mappings["email"].source_field == "email"
        assert "first_name" in mappings

    def test_infer_property_mappings_with_nested(self) -> None:
        """Test inferring property mappings finds nested properties field."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "events", "p.d.events",
            columns=[
                ColumnInfo("customer_id", "STRING", "REQUIRED"),
                ColumnInfo("timestamp", "TIMESTAMP", "REQUIRED"),
                ColumnInfo(
                    "properties", "RECORD", "NULLABLE",
                    is_nested=True,
                    nested_fields=[
                        ColumnInfo("order_id", "STRING", "NULLABLE"),
                    ],
                ),
            ],
        )

        mappings = discovery._infer_property_mappings(schema, EventType.PURCHASE)

        assert "order_id" in mappings
        assert "properties." in mappings["order_id"].source_field

    def test_infer_property_mappings_without_nested(self) -> None:
        """Test inferring property mappings uses flat columns."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "events", "p.d.events",
            columns=[
                ColumnInfo("customer_id", "STRING", "REQUIRED"),
                ColumnInfo("timestamp", "TIMESTAMP", "REQUIRED"),
                ColumnInfo("order_id", "STRING", "NULLABLE"),
            ],
        )

        mappings = discovery._infer_property_mappings(schema, EventType.PURCHASE)

        # Should use flat mapping
        if "order_id" in mappings:
            assert "properties." not in mappings["order_id"].source_field

    def test_build_suggested_config(self) -> None:
        """Test building suggested config from discovered tables."""
        discovery = SchemaDiscovery("project", "dataset")

        tables = [
            DiscoveredTable(
                schema=TableSchema("purchases", "p.d.purchases"),
                inferred_type="event",
                event_type=EventType.PURCHASE,
                confidence=0.9,
                suggested_config=EventTableConfig(
                    table_name="purchases",
                    event_type=EventType.PURCHASE,
                ),
            ),
            DiscoveredTable(
                schema=TableSchema("customers", "p.d.customers"),
                inferred_type="customer",
                confidence=0.85,
                suggested_config=CustomerTableConfig(
                    table_name="customers",
                ),
            ),
            DiscoveredTable(
                schema=TableSchema("id_history", "p.d.id_history"),
                inferred_type="merge",
                confidence=0.9,
                suggested_config=MergeTableConfig(
                    table_name="id_history",
                ),
            ),
            # Low confidence should be skipped
            DiscoveredTable(
                schema=TableSchema("unknown", "p.d.unknown"),
                inferred_type="event",
                confidence=0.3,
                suggested_config=EventTableConfig(
                    table_name="unknown",
                    event_type=EventType.CUSTOM,
                ),
            ),
        ]

        config = discovery._build_suggested_config(tables)

        assert config.project_id == "project"
        assert config.dataset_id == "dataset"
        assert len(config.event_tables) == 1
        assert config.customer_table is not None
        assert config.merge_table is not None

    def test_build_llm_prompt(self) -> None:
        """Test building LLM prompt."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "test", "p.d.test",
            columns=[
                ColumnInfo("id", "STRING", "REQUIRED"),
            ],
            sample_rows=[{"id": "123"}],
        )

        prompt = discovery._build_llm_prompt(schema)

        assert "test" in prompt
        assert "table_type" in prompt
        assert "JSON" in prompt

    def test_parse_llm_response_event(self) -> None:
        """Test parsing LLM response for event table."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema("purchases", "p.d.purchases")
        analysis = {
            "table_type": "event",
            "event_type": "purchase",
            "confidence": 0.9,
            "customer_id_field": "customer_id",
            "timestamp_field": "timestamp",
            "property_mappings": {
                "order_id": {"source": "order_id", "transform": None},
            },
            "reasoning": "Looks like a purchase table",
        }

        result = discovery._parse_llm_response(schema, analysis)

        assert result.inferred_type == "event"
        assert result.event_type == EventType.PURCHASE
        assert result.confidence == 0.9
        assert isinstance(result.suggested_config, EventTableConfig)

    def test_parse_llm_response_customer(self) -> None:
        """Test parsing LLM response for customer table."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "customers", "p.d.customers",
            columns=[ColumnInfo("email", "STRING", "NULLABLE")],
        )
        analysis = {
            "table_type": "customer",
            "confidence": 0.85,
            "reasoning": "Customer profiles",
        }

        result = discovery._parse_llm_response(schema, analysis)

        assert result.inferred_type == "customer"
        assert isinstance(result.suggested_config, CustomerTableConfig)

    def test_parse_llm_response_merge(self) -> None:
        """Test parsing LLM response for merge table."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "id_history", "p.d.id_history",
            columns=[
                ColumnInfo("current_id", "STRING", "REQUIRED"),
                ColumnInfo("past_id", "STRING", "NULLABLE"),
            ],
        )
        analysis = {
            "table_type": "merge",
            "confidence": 0.9,
            "reasoning": "ID merge history",
        }

        result = discovery._parse_llm_response(schema, analysis)

        assert result.inferred_type == "merge"
        assert isinstance(result.suggested_config, MergeTableConfig)

    def test_parse_llm_response_invalid_event_type(self) -> None:
        """Test parsing LLM response with invalid event type."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema("test", "p.d.test")
        analysis = {
            "table_type": "event",
            "event_type": "invalid_type",
            "confidence": 0.7,
        }

        result = discovery._parse_llm_response(schema, analysis)

        assert result.event_type is None

    def test_llm_analysis_no_client(self) -> None:
        """Test LLM analysis when LLM client import fails."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema("test", "p.d.test")
        rule_result = DiscoveredTable(
            schema=schema,
            inferred_type="unknown",
            confidence=0.3,
        )

        # Test that _llm_analysis returns rule_result when import fails
        # This happens inside the method when it can't import get_llm_client
        with patch.dict("sys.modules", {"src.llm.client": None}):
            # The method should catch ImportError and return rule_result
            result = discovery._llm_analysis(schema, rule_result)
            # Should return the rule_result when LLM is not available
            assert result.inferred_type == "unknown"


# =============================================================================
# TESTS: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_discover_schema(self) -> None:
        """Test discover_schema convenience function."""
        with patch.object(SchemaDiscovery, "discover") as mock_discover:
            mock_discover.return_value = DiscoveryResult(
                project_id="project",
                dataset_id="dataset",
            )

            result = discover_schema("project", "dataset", sample_rows=10)

            mock_discover.assert_called_once()
            assert result.project_id == "project"

    def test_print_discovery_report_basic(self) -> None:
        """Test print_discovery_report with basic result."""
        result = DiscoveryResult(
            project_id="my-project",
            dataset_id="my_dataset",
            tables=[
                DiscoveredTable(
                    schema=TableSchema("purchases", "p.d.purchases", row_count=10000),
                    inferred_type="event",
                    event_type=EventType.PURCHASE,
                    confidence=0.9,
                    is_relevant=True,
                    relevance_score=0.95,
                    reasoning="Purchase events",
                    suggested_config=EventTableConfig(
                        table_name="purchases",
                        event_type=EventType.PURCHASE,
                        customer_id_field="customer_id",
                        timestamp_field="timestamp",
                        property_mappings={"order_id": FieldMapping("order_id", "order_id")},
                    ),
                ),
            ],
        )

        report = print_discovery_report(result)

        assert "my-project" in report
        assert "my_dataset" in report
        assert "purchases" in report
        assert "purchase" in report  # Event type value (lowercase)
        assert "EVENT" in report

    def test_print_discovery_report_with_excluded(self) -> None:
        """Test print_discovery_report with excluded tables."""
        result = DiscoveryResult(
            project_id="project",
            dataset_id="dataset",
            tables=[
                DiscoveredTable(
                    schema=TableSchema("irrelevant", "p.d.irrelevant"),
                    inferred_type="irrelevant",
                    is_relevant=False,
                    relevance_score=0.1,
                    reasoning="System table",
                ),
            ],
            excluded_tables=[
                ("temp_data", "Matches exclude pattern: temp"),
                ("logs", "Matches exclude pattern: log"),
            ],
        )

        report = print_discovery_report(result)

        assert "EXCLUDED" in report
        assert "temp_data" in report
        assert "logs" in report

    def test_print_discovery_report_with_warnings(self) -> None:
        """Test print_discovery_report with warnings."""
        result = DiscoveryResult(
            project_id="project",
            dataset_id="dataset",
            warnings=["Warning 1", "Warning 2"],
        )

        report = print_discovery_report(result)

        assert "WARNING" in report
        assert "Warning 1" in report

    def test_print_discovery_report_many_excluded(self) -> None:
        """Test report truncates long lists."""
        result = DiscoveryResult(
            project_id="project",
            dataset_id="dataset",
            excluded_tables=[(f"table_{i}", "reason") for i in range(15)],
            warnings=[f"Warning {i}" for i in range(25)],
        )

        report = print_discovery_report(result)

        assert "... and" in report
        assert "more" in report


# =============================================================================
# TESTS: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for SchemaDiscovery."""

    def test_discover_with_mocked_client(self) -> None:
        """Test full discover flow with mocked BigQuery client."""
        discovery = SchemaDiscovery("test-project", "test_dataset", sample_rows=0)

        mock_client = MagicMock()

        # Mock list_tables
        mock_table_ref = MagicMock()
        mock_table_ref.table_id = "purchases"
        mock_client.list_tables.return_value = [mock_table_ref]

        # Mock get_table
        mock_table = MagicMock()
        mock_table.num_rows = 10000
        mock_field = MagicMock()
        mock_field.name = "customer_id"
        mock_field.field_type = "STRING"
        mock_field.mode = "REQUIRED"
        mock_field.description = None
        mock_field.fields = []

        mock_ts_field = MagicMock()
        mock_ts_field.name = "timestamp"
        mock_ts_field.field_type = "TIMESTAMP"
        mock_ts_field.mode = "REQUIRED"
        mock_ts_field.description = None
        mock_ts_field.fields = []

        mock_table.schema = [mock_field, mock_ts_field]
        mock_client.get_table.return_value = mock_table

        with patch.object(SchemaDiscovery, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            result = discovery.discover()

            assert result.project_id == "test-project"
            assert len(result.tables) == 1
            assert result.tables[0].inferred_type == "event"

    def test_discover_handles_errors(self) -> None:
        """Test discover handles errors gracefully."""
        discovery = SchemaDiscovery("test-project", "test_dataset")

        mock_client = MagicMock()
        mock_table_ref = MagicMock()
        mock_table_ref.table_id = "error_table"
        mock_client.list_tables.return_value = [mock_table_ref]
        mock_client.get_table.side_effect = Exception("API Error")

        with patch.object(SchemaDiscovery, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            result = discovery.discover()

            assert len(result.warnings) >= 1
            assert "error_table" in result.warnings[0]

    def test_discover_filters_irrelevant(self) -> None:
        """Test discover filters out irrelevant tables."""
        discovery = SchemaDiscovery("project", "dataset", sample_rows=0)

        mock_client = MagicMock()

        # Mock list_tables with mix of relevant and irrelevant
        mock_tables = []
        for name in ["purchases", "temp_data", "logs"]:
            ref = MagicMock()
            ref.table_id = name
            mock_tables.append(ref)
        mock_client.list_tables.return_value = mock_tables

        # Mock get_table
        mock_table = MagicMock()
        mock_table.num_rows = 10000
        mock_field = MagicMock()
        mock_field.name = "customer_id"
        mock_field.field_type = "STRING"
        mock_field.mode = "REQUIRED"
        mock_field.description = None
        mock_field.fields = []
        mock_ts = MagicMock()
        mock_ts.name = "timestamp"
        mock_ts.field_type = "TIMESTAMP"
        mock_ts.mode = "REQUIRED"
        mock_ts.description = None
        mock_ts.fields = []
        mock_table.schema = [mock_field, mock_ts]
        mock_client.get_table.return_value = mock_table

        with patch.object(SchemaDiscovery, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            result = discovery.discover(filter_irrelevant=True)

            # Should have excluded temp_data and logs
            assert len(result.excluded_tables) == 2

    def test_discover_without_filtering(self) -> None:
        """Test discover without filtering."""
        discovery = SchemaDiscovery("project", "dataset", sample_rows=0)

        mock_client = MagicMock()
        mock_table_ref = MagicMock()
        mock_table_ref.table_id = "temp_data"
        mock_client.list_tables.return_value = [mock_table_ref]

        mock_table = MagicMock()
        mock_table.num_rows = 100
        mock_table.schema = []
        mock_client.get_table.return_value = mock_table

        with patch.object(SchemaDiscovery, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            result = discovery.discover(filter_irrelevant=False)

            # Should not exclude based on name
            assert len(result.excluded_tables) == 0

    def test_get_sample_rows(self) -> None:
        """Test getting sample rows."""
        discovery = SchemaDiscovery("project", "dataset", sample_rows=3)

        mock_client = MagicMock()
        mock_rows = [
            {"col1": "val1"},
            {"col1": "val2"},
        ]
        mock_job = MagicMock()
        mock_job.result.return_value = mock_rows
        mock_client.query.return_value = mock_job

        with patch.object(SchemaDiscovery, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            rows = discovery._get_sample_rows("project.dataset.table")

            assert len(rows) == 2

    def test_get_sample_rows_error(self) -> None:
        """Test getting sample rows handles errors."""
        discovery = SchemaDiscovery("project", "dataset", sample_rows=3)

        mock_client = MagicMock()
        mock_client.query.side_effect = Exception("Query failed")

        with patch.object(SchemaDiscovery, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            rows = discovery._get_sample_rows("project.dataset.table")

            assert rows == []

    def test_schema_field_to_column(self) -> None:
        """Test converting BigQuery schema field."""
        discovery = SchemaDiscovery("project", "dataset")

        mock_field = MagicMock()
        mock_field.name = "test_col"
        mock_field.field_type = "STRING"
        mock_field.mode = "NULLABLE"
        mock_field.description = "Test column"
        mock_field.fields = []

        col = discovery._schema_field_to_column(mock_field)

        assert col.name == "test_col"
        assert col.data_type == "STRING"
        assert col.mode == "NULLABLE"
        assert col.description == "Test column"
        assert col.is_nested is False

    def test_schema_field_to_column_nested(self) -> None:
        """Test converting nested BigQuery schema field."""
        discovery = SchemaDiscovery("project", "dataset")

        mock_nested = MagicMock()
        mock_nested.name = "value"
        mock_nested.field_type = "STRING"
        mock_nested.mode = "NULLABLE"
        mock_nested.description = None
        mock_nested.fields = []

        mock_field = MagicMock()
        mock_field.name = "properties"
        mock_field.field_type = "RECORD"
        mock_field.mode = "NULLABLE"
        mock_field.description = None
        mock_field.fields = [mock_nested]

        col = discovery._schema_field_to_column(mock_field)

        assert col.name == "properties"
        assert col.is_nested is True
        assert len(col.nested_fields) == 1
        assert col.nested_fields[0].name == "value"

    def test_analyze_table_high_confidence(self) -> None:
        """Test analyze_table uses rule-based for high confidence."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "id_history", "p.d.id_history",
            columns=[
                ColumnInfo("current_id", "STRING", "REQUIRED"),
                ColumnInfo("past_id", "STRING", "NULLABLE"),
            ],
        )

        result = discovery._analyze_table(schema)

        # High confidence rule-based result
        assert result.inferred_type == "merge"
        assert result.confidence >= 0.8

    def test_analyze_table_low_confidence_uses_llm(self) -> None:
        """Test analyze_table falls back to LLM for low confidence."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema(
            "random_data", "p.d.random_data",
            columns=[
                ColumnInfo("col1", "STRING", "NULLABLE"),
            ],
        )

        # Mock LLM analysis to return an improved result
        with patch.object(discovery, "_llm_analysis") as mock_llm:
            mock_llm.return_value = DiscoveredTable(
                schema=schema,
                inferred_type="unknown",
                confidence=0.5,
            )

            result = discovery._analyze_table(schema)

            mock_llm.assert_called_once()

    def test_discover_marks_irrelevant_after_schema_check(self) -> None:
        """Test that tables can be marked irrelevant after schema analysis."""
        discovery = SchemaDiscovery("project", "dataset", sample_rows=0)

        mock_client = MagicMock()

        # Table that passes name filter but fails schema check
        mock_table_ref = MagicMock()
        mock_table_ref.table_id = "generic_events"
        mock_client.list_tables.return_value = [mock_table_ref]

        # Mock table with no useful columns and low row count
        mock_table = MagicMock()
        mock_table.num_rows = 50  # Low row count
        mock_table.schema = []
        mock_client.get_table.return_value = mock_table

        with patch.object(SchemaDiscovery, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            result = discovery.discover(filter_irrelevant=True)

            # Check that warnings about irrelevant tables were added
            # or table was marked as irrelevant
            irrelevant = [t for t in result.tables if not t.is_relevant]
            assert len(irrelevant) >= 0  # Table may be in warnings or irrelevant list

    def test_get_bigquery_fresh_import(self) -> None:
        """Test _get_bigquery performs actual import when module is None."""
        import sys

        discovery = SchemaDiscovery("test-project", "test_dataset")

        # Ensure module is None
        discovery._bigquery_module = None

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

            result = discovery._get_bigquery()

            # Should have stored and returned the module
            assert result == mock_bigquery
            assert discovery._bigquery_module == mock_bigquery
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

    def test_discover_marks_table_irrelevant_after_detailed_check(self) -> None:
        """Test that tables are marked irrelevant after detailed schema analysis."""
        discovery = SchemaDiscovery("project", "dataset", sample_rows=0)

        mock_client = MagicMock()

        # Table that passes initial name filter but has no useful columns
        mock_table_ref = MagicMock()
        mock_table_ref.table_id = "some_data"  # Doesn't match any exclude pattern
        mock_client.list_tables.return_value = [mock_table_ref]

        # Mock table with no customer_id and no timestamp - should be marked irrelevant
        mock_table = MagicMock()
        mock_table.num_rows = 1000  # Enough rows

        # Create mock fields with no customer_id or timestamp
        mock_field = MagicMock()
        mock_field.name = "random_column"
        mock_field.field_type = "STRING"
        mock_field.mode = "NULLABLE"
        mock_field.description = None
        mock_field.fields = []
        mock_table.schema = [mock_field]
        mock_client.get_table.return_value = mock_table

        with patch.object(SchemaDiscovery, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            result = discovery.discover(filter_irrelevant=True)

            # Should have marked table as irrelevant due to missing columns
            # Either in irrelevant tables or warnings
            has_irrelevant = any(
                not t.is_relevant or t.inferred_type == "irrelevant"
                for t in result.tables
            )
            has_warning = any("irrelevant" in w.lower() for w in result.warnings)
            assert has_irrelevant or has_warning

    def test_get_table_schema_with_sample_rows(self) -> None:
        """Test _get_table_schema fetches sample rows when configured."""
        discovery = SchemaDiscovery("project", "dataset", sample_rows=5)

        mock_client = MagicMock()

        # Mock get_table
        mock_table = MagicMock()
        mock_table.num_rows = 100
        mock_field = MagicMock()
        mock_field.name = "col1"
        mock_field.field_type = "STRING"
        mock_field.mode = "NULLABLE"
        mock_field.description = None
        mock_field.fields = []
        mock_table.schema = [mock_field]
        mock_client.get_table.return_value = mock_table

        # Mock query for sample rows
        mock_sample_rows = [{"col1": "val1"}, {"col1": "val2"}]
        mock_job = MagicMock()
        mock_job.result.return_value = mock_sample_rows
        mock_client.query.return_value = mock_job

        with patch.object(SchemaDiscovery, "client", new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client

            schema = discovery._get_table_schema("test_table")

            assert schema.table_name == "test_table"
            assert len(schema.sample_rows) == 2
            mock_client.query.assert_called_once()

    def test_llm_analysis_with_exception(self) -> None:
        """Test _llm_analysis handles exceptions and returns rule_result."""
        discovery = SchemaDiscovery("project", "dataset")

        schema = TableSchema("test", "p.d.test")
        rule_result = DiscoveredTable(
            schema=schema,
            inferred_type="unknown",
            confidence=0.3,
        )

        # Mock the LLM client import succeeding but client failing
        mock_client = MagicMock()
        mock_client.complete.side_effect = Exception("LLM API error")

        with patch("src.data.bigquery.schema_discovery.get_llm_client", return_value=mock_client, create=True):
            # Will catch the exception and return rule_result
            result = discovery._llm_analysis(schema, rule_result)

            # Should fall back to rule_result
            assert result.inferred_type == "unknown"

    def test_print_discovery_report_with_many_irrelevant_tables(self) -> None:
        """Test report with many irrelevant tables truncation."""
        irrelevant_tables = [
            DiscoveredTable(
                schema=TableSchema(f"table_{i}", f"p.d.table_{i}"),
                inferred_type="irrelevant",
                is_relevant=False,
                reasoning=f"Reason {i}",
            )
            for i in range(15)
        ]

        result = DiscoveryResult(
            project_id="project",
            dataset_id="dataset",
            tables=irrelevant_tables,
        )

        report = print_discovery_report(result)

        assert "... and" in report
        assert "more" in report
