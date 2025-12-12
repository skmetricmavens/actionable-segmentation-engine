"""
Tests for src/data/bigquery/config.py

Comprehensive test suite for BigQuery configuration classes.
"""

import pytest
from typing import Any

from src.data.bigquery.config import (
    EventType,
    FieldMapping,
    EventTableConfig,
    CustomerTableConfig,
    MergeTableConfig,
    ExternalIdTableConfig,
    BigQueryConfig,
)


# =============================================================================
# TESTS: EventType Enum
# =============================================================================


class TestEventType:
    """Tests for EventType enum."""

    def test_purchase_event(self) -> None:
        """Test PURCHASE event type."""
        assert EventType.PURCHASE.value == "purchase"

    def test_page_view_event(self) -> None:
        """Test PAGE_VIEW event type."""
        assert EventType.PAGE_VIEW.value == "page_view"

    def test_view_item_event(self) -> None:
        """Test VIEW_ITEM event type."""
        assert EventType.VIEW_ITEM.value == "view_item"

    def test_cart_events(self) -> None:
        """Test cart-related event types."""
        assert EventType.ADD_TO_CART.value == "add_to_cart"
        assert EventType.REMOVE_FROM_CART.value == "remove_from_cart"
        assert EventType.BEGIN_CHECKOUT.value == "begin_checkout"

    def test_session_events(self) -> None:
        """Test session event types."""
        assert EventType.SESSION_START.value == "session_start"
        assert EventType.SESSION_END.value == "session_end"

    def test_engagement_events(self) -> None:
        """Test engagement event types."""
        assert EventType.WISHLIST_ADD.value == "wishlist_add"
        assert EventType.SEARCH.value == "search"
        assert EventType.EMAIL_OPEN.value == "email_open"
        assert EventType.EMAIL_CLICK.value == "email_click"

    def test_custom_event(self) -> None:
        """Test CUSTOM event type."""
        assert EventType.CUSTOM.value == "custom"

    def test_all_event_types_are_strings(self) -> None:
        """All event types should be string enums."""
        for event_type in EventType:
            assert isinstance(event_type.value, str)


# =============================================================================
# TESTS: FieldMapping
# =============================================================================


class TestFieldMapping:
    """Tests for FieldMapping dataclass."""

    def test_basic_field_mapping(self) -> None:
        """Test basic field mapping without transform."""
        mapping = FieldMapping(
            source_field="properties.total_price",
            target_field="order_total",
        )

        assert mapping.source_field == "properties.total_price"
        assert mapping.target_field == "order_total"
        assert mapping.transform is None
        assert mapping.default is None

    def test_field_mapping_with_transform(self) -> None:
        """Test field mapping with transformation."""
        mapping = FieldMapping(
            source_field="properties.price",
            target_field="product_price",
            transform="decimal",
        )

        assert mapping.transform == "decimal"

    def test_field_mapping_with_default(self) -> None:
        """Test field mapping with default value."""
        mapping = FieldMapping(
            source_field="properties.quantity",
            target_field="quantity",
            transform="int",
            default=1,
        )

        assert mapping.default == 1

    def test_supported_transforms(self) -> None:
        """Test various transform types are accepted."""
        transforms = ["decimal", "int", "datetime", "json_parse", None]

        for transform in transforms:
            mapping = FieldMapping(
                source_field="field",
                target_field="target",
                transform=transform,
            )
            assert mapping.transform == transform


# =============================================================================
# TESTS: EventTableConfig
# =============================================================================


class TestEventTableConfig:
    """Tests for EventTableConfig dataclass."""

    def test_basic_config(self) -> None:
        """Test basic event table configuration."""
        config = EventTableConfig(
            table_name="my_project.my_dataset.purchases",
            event_type=EventType.PURCHASE,
        )

        assert config.table_name == "my_project.my_dataset.purchases"
        assert config.event_type == EventType.PURCHASE
        assert config.customer_id_field == "internal_customer_id"
        assert config.timestamp_field == "timestamp"

    def test_custom_field_names(self) -> None:
        """Test custom field name configuration."""
        config = EventTableConfig(
            table_name="events",
            event_type=EventType.PAGE_VIEW,
            customer_id_field="user_id",
            timestamp_field="event_timestamp",
            event_type_field="event_name",
        )

        assert config.customer_id_field == "user_id"
        assert config.timestamp_field == "event_timestamp"
        assert config.event_type_field == "event_name"

    def test_property_mappings(self) -> None:
        """Test property mappings configuration."""
        config = EventTableConfig(
            table_name="events",
            event_type=EventType.VIEW_ITEM,
            property_mappings={
                "product_id": FieldMapping("props.pid", "product_id"),
                "price": FieldMapping("props.price", "price", "decimal"),
            },
        )

        assert "product_id" in config.property_mappings
        assert "price" in config.property_mappings

    def test_filters(self) -> None:
        """Test filter conditions."""
        config = EventTableConfig(
            table_name="events",
            event_type=EventType.PURCHASE,
            filters=["status = 'completed'", "amount > 0"],
        )

        assert len(config.filters) == 2
        assert "status = 'completed'" in config.filters

    def test_purchase_table_factory(self) -> None:
        """Test purchase_table class method."""
        config = EventTableConfig.purchase_table(
            "purchases",
            product_list_field="props.products",
            total_price_field="props.total",
            order_id_field="props.order_num",
        )

        assert config.table_name == "purchases"
        assert config.event_type == EventType.PURCHASE
        assert "order_id" in config.property_mappings
        assert "order_total" in config.property_mappings
        assert "product_list" in config.property_mappings

        # Check transforms
        assert config.property_mappings["order_total"].transform == "decimal"
        assert config.property_mappings["product_list"].transform == "json_parse"

    def test_page_view_table_factory(self) -> None:
        """Test page_view_table class method."""
        config = EventTableConfig.page_view_table(
            "page_views",
            page_url_field="props.url",
            page_title_field="props.title",
        )

        assert config.table_name == "page_views"
        assert config.event_type == EventType.PAGE_VIEW
        assert "page_url" in config.property_mappings
        assert "page_title" in config.property_mappings

    def test_view_item_table_factory(self) -> None:
        """Test view_item_table class method."""
        config = EventTableConfig.view_item_table(
            "product_views",
            product_id_field="props.id",
            product_name_field="props.name",
            product_category_field="props.category",
            product_price_field="props.price",
        )

        assert config.table_name == "product_views"
        assert config.event_type == EventType.VIEW_ITEM
        assert config.product_category_field == "props.category"
        assert "product_id" in config.property_mappings
        assert "product_price" in config.property_mappings
        assert config.property_mappings["product_price"].transform == "decimal"

    def test_add_to_cart_table_factory(self) -> None:
        """Test add_to_cart_table class method."""
        config = EventTableConfig.add_to_cart_table(
            "cart_events",
            product_id_field="props.product_id",
            quantity_field="props.qty",
        )

        assert config.table_name == "cart_events"
        assert config.event_type == EventType.ADD_TO_CART
        assert "quantity" in config.property_mappings
        assert config.property_mappings["quantity"].transform == "int"
        assert config.property_mappings["quantity"].default == 1


# =============================================================================
# TESTS: CustomerTableConfig
# =============================================================================


class TestCustomerTableConfig:
    """Tests for CustomerTableConfig dataclass."""

    def test_basic_config(self) -> None:
        """Test basic customer table configuration."""
        config = CustomerTableConfig(table_name="customers")

        assert config.table_name == "customers"
        assert config.customer_id_field == "internal_customer_id"
        assert config.auto_detect_fields is True

    def test_custom_customer_id_field(self) -> None:
        """Test custom customer ID field."""
        config = CustomerTableConfig(
            table_name="users",
            customer_id_field="user_id",
        )

        assert config.customer_id_field == "user_id"

    def test_property_mappings(self) -> None:
        """Test property mappings."""
        config = CustomerTableConfig(
            table_name="customers",
            property_mappings={
                "email": FieldMapping("props.email", "email"),
            },
        )

        assert "email" in config.property_mappings

    def test_known_fields_default(self) -> None:
        """Test default known fields."""
        config = CustomerTableConfig(table_name="customers")

        assert "email" in config.known_fields
        assert "first_name" in config.known_fields
        assert "country" in config.known_fields
        assert "rfm_today" in config.known_fields

    def test_default_factory_method(self) -> None:
        """Test default class method creates common mappings."""
        config = CustomerTableConfig.default("customer_properties")

        assert config.table_name == "customer_properties"
        assert "email" in config.property_mappings
        assert "first_name" in config.property_mappings
        assert "country" in config.property_mappings
        assert "rfm" in config.property_mappings

        # Check source paths
        assert config.property_mappings["email"].source_field == "properties.email"

    def test_disable_auto_detect(self) -> None:
        """Test disabling auto-detect."""
        config = CustomerTableConfig(
            table_name="customers",
            auto_detect_fields=False,
        )

        assert config.auto_detect_fields is False


# =============================================================================
# TESTS: MergeTableConfig
# =============================================================================


class TestMergeTableConfig:
    """Tests for MergeTableConfig dataclass."""

    def test_basic_config(self) -> None:
        """Test basic merge table configuration."""
        config = MergeTableConfig(table_name="id_history")

        assert config.table_name == "id_history"
        assert config.current_id_field == "internal_customer_id"
        assert config.past_id_field == "past_id"

    def test_custom_field_names(self) -> None:
        """Test custom field names."""
        config = MergeTableConfig(
            table_name="identity_graph",
            current_id_field="canonical_id",
            past_id_field="historical_id",
        )

        assert config.current_id_field == "canonical_id"
        assert config.past_id_field == "historical_id"


# =============================================================================
# TESTS: ExternalIdTableConfig
# =============================================================================


class TestExternalIdTableConfig:
    """Tests for ExternalIdTableConfig dataclass."""

    def test_basic_config(self) -> None:
        """Test basic external ID table configuration."""
        config = ExternalIdTableConfig(table_name="external_ids")

        assert config.table_name == "external_ids"
        assert config.customer_id_field == "internal_customer_id"
        assert config.id_name_field == "id_name"
        assert config.id_value_field == "id_value"

    def test_custom_field_names(self) -> None:
        """Test custom field names."""
        config = ExternalIdTableConfig(
            table_name="user_identities",
            customer_id_field="user_id",
            id_name_field="identity_type",
            id_value_field="identity_value",
        )

        assert config.customer_id_field == "user_id"
        assert config.id_name_field == "identity_type"
        assert config.id_value_field == "identity_value"


# =============================================================================
# TESTS: BigQueryConfig
# =============================================================================


class TestBigQueryConfig:
    """Tests for BigQueryConfig dataclass."""

    def test_basic_config(self) -> None:
        """Test basic BigQuery configuration."""
        config = BigQueryConfig(
            project_id="my-project",
            dataset_id="my_dataset",
        )

        assert config.project_id == "my-project"
        assert config.dataset_id == "my_dataset"
        assert config.event_tables == []
        assert config.customer_table is None
        assert config.merge_table is None

    def test_with_event_tables(self) -> None:
        """Test config with event tables."""
        event_config = EventTableConfig(
            table_name="purchases",
            event_type=EventType.PURCHASE,
        )

        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            event_tables=[event_config],
        )

        assert len(config.event_tables) == 1
        assert config.event_tables[0].event_type == EventType.PURCHASE

    def test_with_all_tables(self) -> None:
        """Test config with all table types."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            event_tables=[
                EventTableConfig.purchase_table("purchases"),
            ],
            customer_table=CustomerTableConfig.default("customers"),
            merge_table=MergeTableConfig(table_name="id_history"),
            external_ids_table=ExternalIdTableConfig(table_name="external_ids"),
        )

        assert config.customer_table is not None
        assert config.merge_table is not None
        assert config.external_ids_table is not None

    def test_date_range(self) -> None:
        """Test date range configuration."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

        assert config.start_date == "2024-01-01"
        assert config.end_date == "2024-12-31"

    def test_query_limits(self) -> None:
        """Test query limit configuration."""
        config = BigQueryConfig(
            project_id="project",
            dataset_id="dataset",
            limit_per_table=1000,
            sample_rate=0.1,
        )

        assert config.limit_per_table == 1000
        assert config.sample_rate == 0.1

    def test_get_full_table_name_simple(self) -> None:
        """Test get_full_table_name with simple table name."""
        config = BigQueryConfig(
            project_id="my-project",
            dataset_id="my_dataset",
        )

        full_name = config.get_full_table_name("purchases")
        assert full_name == "my-project.my_dataset.purchases"

    def test_get_full_table_name_already_qualified(self) -> None:
        """Test get_full_table_name with already qualified name."""
        config = BigQueryConfig(
            project_id="my-project",
            dataset_id="my_dataset",
        )

        full_name = config.get_full_table_name("other-project.other_dataset.table")
        assert full_name == "other-project.other_dataset.table"

    def test_from_dict_minimal(self) -> None:
        """Test from_dict with minimal config."""
        config_dict = {
            "project_id": "test-project",
            "dataset_id": "test_dataset",
        }

        config = BigQueryConfig.from_dict(config_dict)

        assert config.project_id == "test-project"
        assert config.dataset_id == "test_dataset"
        assert config.event_tables == []

    def test_from_dict_with_event_tables(self) -> None:
        """Test from_dict with event tables."""
        config_dict = {
            "project_id": "project",
            "dataset_id": "dataset",
            "event_tables": [
                {
                    "table_name": "purchases",
                    "event_type": "purchase",
                    "customer_id_field": "user_id",
                    "timestamp_field": "event_time",
                    "property_mappings": {
                        "order_total": {
                            "source_field": "props.total",
                            "target_field": "order_total",
                            "transform": "decimal",
                        },
                    },
                },
            ],
        }

        config = BigQueryConfig.from_dict(config_dict)

        assert len(config.event_tables) == 1
        assert config.event_tables[0].event_type == EventType.PURCHASE
        assert config.event_tables[0].customer_id_field == "user_id"
        assert "order_total" in config.event_tables[0].property_mappings

    def test_from_dict_with_simple_property_mapping(self) -> None:
        """Test from_dict with simple string property mapping."""
        config_dict = {
            "project_id": "project",
            "dataset_id": "dataset",
            "event_tables": [
                {
                    "table_name": "events",
                    "event_type": "view_item",
                    "property_mappings": {
                        "product_id": "props.pid",  # Simple string, not dict
                    },
                },
            ],
        }

        config = BigQueryConfig.from_dict(config_dict)

        mapping = config.event_tables[0].property_mappings["product_id"]
        assert mapping.source_field == "props.pid"
        assert mapping.target_field == "product_id"

    def test_from_dict_with_customer_table(self) -> None:
        """Test from_dict with customer table."""
        config_dict = {
            "project_id": "project",
            "dataset_id": "dataset",
            "customer_table": {
                "table_name": "customers",
                "customer_id_field": "cust_id",
            },
        }

        config = BigQueryConfig.from_dict(config_dict)

        assert config.customer_table is not None
        assert config.customer_table.table_name == "customers"
        assert config.customer_table.customer_id_field == "cust_id"

    def test_from_dict_with_merge_table(self) -> None:
        """Test from_dict with merge table."""
        config_dict = {
            "project_id": "project",
            "dataset_id": "dataset",
            "merge_table": {
                "table_name": "id_merges",
                "current_id_field": "new_id",
                "past_id_field": "old_id",
            },
        }

        config = BigQueryConfig.from_dict(config_dict)

        assert config.merge_table is not None
        assert config.merge_table.table_name == "id_merges"
        assert config.merge_table.current_id_field == "new_id"
        assert config.merge_table.past_id_field == "old_id"

    def test_from_dict_with_dates_and_limits(self) -> None:
        """Test from_dict with date range and limits."""
        config_dict = {
            "project_id": "project",
            "dataset_id": "dataset",
            "start_date": "2024-01-01",
            "end_date": "2024-06-30",
            "limit_per_table": 5000,
            "sample_rate": 0.25,
        }

        config = BigQueryConfig.from_dict(config_dict)

        assert config.start_date == "2024-01-01"
        assert config.end_date == "2024-06-30"
        assert config.limit_per_table == 5000
        assert config.sample_rate == 0.25

    def test_from_dict_full_config(self) -> None:
        """Test from_dict with complete configuration."""
        config_dict = {
            "project_id": "production-project",
            "dataset_id": "cdp_data",
            "event_tables": [
                {
                    "table_name": "purchase_events",
                    "event_type": "purchase",
                },
                {
                    "table_name": "view_events",
                    "event_type": "view_item",
                },
            ],
            "customer_table": {
                "table_name": "customer_profiles",
            },
            "merge_table": {
                "table_name": "identity_history",
            },
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "limit_per_table": 10000,
        }

        config = BigQueryConfig.from_dict(config_dict)

        assert config.project_id == "production-project"
        assert len(config.event_tables) == 2
        assert config.customer_table is not None
        assert config.merge_table is not None
        assert config.start_date == "2024-01-01"
