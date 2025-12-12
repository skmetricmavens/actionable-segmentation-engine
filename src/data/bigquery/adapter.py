"""
BigQuery adapter for loading CDP-style event data.

Converts BigQuery tables with varying schemas into our internal EventRecord
and CustomerIdHistory structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, TYPE_CHECKING
import json
import logging
import uuid

# Lazy import for BigQuery - only needed when actually loading data
if TYPE_CHECKING:
    from google.cloud import bigquery

from src.data.bigquery.config import (
    BigQueryConfig,
    EventTableConfig,
    CustomerTableConfig,
    MergeTableConfig,
    FieldMapping,
    EventType,
)
from src.data.schemas import (
    EventRecord,
    EventType as InternalEventType,
    EventProperties,
    CustomerIdHistory,
)


logger = logging.getLogger(__name__)


# =============================================================================
# EVENT TYPE MAPPING
# =============================================================================

# Map BigQuery event types to our internal types
# Internal EventType has: SESSION_START, VIEW_CATEGORY, VIEW_ITEM, ADD_TO_CART,
# CHECKOUT, PURCHASE, PURCHASE_ITEM, REFUND, REFUND_ITEM, DELIVERY_EVENTS
BQ_TO_INTERNAL_EVENT_TYPE: dict[EventType, InternalEventType] = {
    EventType.PURCHASE: InternalEventType.PURCHASE,
    EventType.PAGE_VIEW: InternalEventType.VIEW_ITEM,  # Map page views to view_item
    EventType.VIEW_ITEM: InternalEventType.VIEW_ITEM,
    EventType.ADD_TO_CART: InternalEventType.ADD_TO_CART,
    EventType.REMOVE_FROM_CART: InternalEventType.ADD_TO_CART,  # Track as cart activity
    EventType.BEGIN_CHECKOUT: InternalEventType.CHECKOUT,
    EventType.SESSION_START: InternalEventType.SESSION_START,
    EventType.SESSION_END: InternalEventType.SESSION_START,  # No direct mapping
    EventType.WISHLIST_ADD: InternalEventType.VIEW_ITEM,  # Track as engagement
    EventType.SEARCH: InternalEventType.VIEW_CATEGORY,  # Search is category exploration
    EventType.EMAIL_OPEN: InternalEventType.SESSION_START,  # Track as engagement start
    EventType.EMAIL_CLICK: InternalEventType.VIEW_ITEM,  # Track as item interest
    EventType.CUSTOM: InternalEventType.VIEW_ITEM,  # Default to view_item
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class LoadResult:
    """Result of loading data from BigQuery."""

    events: list[EventRecord] = field(default_factory=list)
    id_history: list[CustomerIdHistory] = field(default_factory=list)
    customer_properties: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Statistics
    tables_loaded: int = 0
    total_rows: int = 0
    events_by_type: dict[str, int] = field(default_factory=dict)
    unique_customers: int = 0
    load_duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


# =============================================================================
# VALUE TRANSFORMERS
# =============================================================================


def transform_value(value: Any, transform: str | None) -> Any:
    """Apply transformation to a value."""
    if value is None:
        return None

    if transform is None:
        return value

    if transform == "decimal":
        try:
            return Decimal(str(value))
        except (ValueError, TypeError, ArithmeticError):
            # ArithmeticError catches decimal.InvalidOperation
            return None

    if transform == "int":
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    if transform == "float":
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    if transform == "datetime":
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None

    if transform == "json_parse":
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except (json.JSONDecodeError, TypeError):
            return value  # Return as-is if not valid JSON

    if transform == "bool":
        if isinstance(value, bool):
            return value
        return str(value).lower() in ("true", "1", "yes")

    return value


def get_nested_value(row: dict[str, Any], field_path: str) -> Any:
    """
    Get a value from a nested dictionary using dot notation.

    Args:
        row: Dictionary (BigQuery row)
        field_path: Dot-separated path (e.g., "properties.total_price")

    Returns:
        Value at path or None if not found
    """
    parts = field_path.split(".")
    current = row

    for part in parts:
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None

    return current


# =============================================================================
# BIGQUERY ADAPTER
# =============================================================================


class BigQueryAdapter:
    """
    Adapter for loading data from BigQuery CDP tables.

    Handles:
    - Multiple event tables with different schemas
    - Customer property tables with nested structures
    - ID merge tables for identity resolution
    """

    def __init__(self, config: BigQueryConfig):
        """
        Initialize adapter with configuration.

        Args:
            config: BigQuery configuration

        Raises:
            ImportError: If google-cloud-bigquery is not installed
        """
        self.config = config
        self._client: Any = None  # bigquery.Client
        self._bigquery_module: Any = None

    def _get_bigquery(self) -> Any:
        """Lazy import of BigQuery module."""
        if self._bigquery_module is None:
            try:
                from google.cloud import bigquery
                self._bigquery_module = bigquery
            except ImportError:
                raise ImportError(
                    "google-cloud-bigquery is required for BigQuery data loading. "
                    "Install it with: pip install google-cloud-bigquery"
                )
        return self._bigquery_module

    @property
    def client(self) -> Any:
        """Get or create BigQuery client."""
        if self._client is None:
            bigquery = self._get_bigquery()
            self._client = bigquery.Client(project=self.config.project_id)
        return self._client

    def load(self) -> LoadResult:
        """
        Load all configured data from BigQuery.

        Returns:
            LoadResult with events, ID history, and customer properties
        """
        import time
        start_time = time.perf_counter()

        result = LoadResult()

        try:
            # Load events from each table
            for event_config in self.config.event_tables:
                events, count = self._load_event_table(event_config)
                result.events.extend(events)
                result.tables_loaded += 1
                result.total_rows += count
                result.events_by_type[event_config.event_type.value] = count
                logger.info(f"Loaded {count} events from {event_config.table_name}")

            # Load ID merge history
            if self.config.merge_table:
                result.id_history = self._load_merge_table(self.config.merge_table)
                logger.info(f"Loaded {len(result.id_history)} ID merges")

            # Load customer properties
            if self.config.customer_table:
                result.customer_properties = self._load_customer_properties(
                    self.config.customer_table
                )
                logger.info(f"Loaded properties for {len(result.customer_properties)} customers")

            # Calculate unique customers
            customer_ids = {e.internal_customer_id for e in result.events}
            result.unique_customers = len(customer_ids)

        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Error loading data: {e}")

        result.load_duration_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _load_event_table(
        self,
        table_config: EventTableConfig,
    ) -> tuple[list[EventRecord], int]:
        """Load events from a single BigQuery table."""

        full_table = self.config.get_full_table_name(table_config.table_name)

        # Build query
        query = self._build_event_query(table_config, full_table)
        logger.debug(f"Event query: {query}")

        # Execute query
        query_job = self.client.query(query)
        rows = list(query_job.result())

        # Convert rows to EventRecords
        events: list[EventRecord] = []
        internal_event_type = BQ_TO_INTERNAL_EVENT_TYPE.get(
            table_config.event_type,
            InternalEventType.VIEW_ITEM,  # Default fallback
        )

        for row in rows:
            row_dict = dict(row)
            event = self._row_to_event(row_dict, table_config, internal_event_type)
            if event:
                events.append(event)

        return events, len(rows)

    def _build_event_query(
        self,
        table_config: EventTableConfig,
        full_table: str,
    ) -> str:
        """Build SQL query for event table."""

        # Select all fields (we'll extract what we need)
        query_parts = [f"SELECT * FROM `{full_table}`"]

        # Add WHERE conditions
        where_conditions = []

        # Date range filter
        if self.config.start_date:
            where_conditions.append(
                f"{table_config.date_field} >= '{self.config.start_date}'"
            )
        if self.config.end_date:
            where_conditions.append(
                f"{table_config.date_field} <= '{self.config.end_date}'"
            )

        # Custom filters
        where_conditions.extend(table_config.filters)

        # Sampling
        if self.config.sample_rate and self.config.sample_rate < 1.0:
            where_conditions.append(
                f"RAND() < {self.config.sample_rate}"
            )

        if where_conditions:
            query_parts.append("WHERE " + " AND ".join(where_conditions))

        # Order by timestamp
        query_parts.append(f"ORDER BY {table_config.timestamp_field}")

        # Limit
        if self.config.limit_per_table:
            query_parts.append(f"LIMIT {self.config.limit_per_table}")

        return " ".join(query_parts)

    def _row_to_event(
        self,
        row: dict[str, Any],
        table_config: EventTableConfig,
        event_type: InternalEventType,
    ) -> EventRecord | None:
        """Convert a BigQuery row to an EventRecord."""

        try:
            # Get required fields
            customer_id = row.get(table_config.customer_id_field)
            timestamp = row.get(table_config.timestamp_field)

            if not customer_id or not timestamp:
                return None

            # Ensure timestamp is datetime
            if not isinstance(timestamp, datetime):
                timestamp = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))

            # Build event properties from mappings
            properties_dict: dict[str, Any] = {}

            for target_field, mapping in table_config.property_mappings.items():
                value = get_nested_value(row, mapping.source_field)
                if value is None and mapping.default is not None:
                    value = mapping.default
                value = transform_value(value, mapping.transform)
                if value is not None:
                    properties_dict[target_field] = value

            # Handle product_category for purchase events
            if table_config.product_category_field:
                cat_value = get_nested_value(row, table_config.product_category_field)
                if cat_value:
                    properties_dict["product_category"] = cat_value

            # For purchase events, try to extract categories from product_list
            if event_type == InternalEventType.PURCHASE and "product_list" in properties_dict:
                product_list = properties_dict.get("product_list")
                if isinstance(product_list, list):
                    categories = set()
                    for product in product_list:
                        if isinstance(product, dict):
                            cat = product.get("category") or product.get("product_category")
                            if cat:
                                categories.add(cat)
                    if categories:
                        properties_dict["product_categories"] = list(categories)

            # Create EventProperties
            event_properties = EventProperties(
                page_url=properties_dict.get("page_url"),
                page_title=properties_dict.get("page_title"),
                product_id=properties_dict.get("product_id"),
                product_name=properties_dict.get("product_name"),
                product_category=properties_dict.get("product_category"),
                product_price=properties_dict.get("product_price"),
                quantity=properties_dict.get("quantity"),
                order_id=properties_dict.get("order_id"),
                order_total=properties_dict.get("order_total"),
                search_query=properties_dict.get("search_query"),
                device_type=properties_dict.get("device") or properties_dict.get("device_type"),
                session_id=properties_dict.get("session_id"),
                custom_properties={
                    k: v for k, v in properties_dict.items()
                    if k not in {
                        "page_url", "page_title", "product_id", "product_name",
                        "product_category", "product_price", "quantity",
                        "order_id", "order_total", "search_query", "device_type",
                        "session_id", "device",
                    }
                } or None,
            )

            return EventRecord(
                event_id=str(uuid.uuid4()),  # Generate ID since BQ may not have one
                internal_customer_id=str(customer_id),
                event_type=event_type,
                timestamp=timestamp,
                properties=event_properties,
            )

        except Exception as e:
            logger.warning(f"Error converting row to event: {e}")
            return None

    def _load_merge_table(
        self,
        merge_config: MergeTableConfig,
    ) -> list[CustomerIdHistory]:
        """Load ID merge history from BigQuery.

        Returns list of CustomerIdHistory objects, one per merge.
        """

        full_table = self.config.get_full_table_name(merge_config.table_name)

        query = f"""
        SELECT
            {merge_config.current_id_field} as current_id,
            {merge_config.past_id_field} as past_id
        FROM `{full_table}`
        WHERE {merge_config.past_id_field} IS NOT NULL
        """

        if self.config.limit_per_table:
            query += f" LIMIT {self.config.limit_per_table}"

        query_job = self.client.query(query)
        rows = list(query_job.result())

        id_history_list: list[CustomerIdHistory] = []
        for row in rows:
            row_dict = dict(row)
            current_id = row_dict.get("current_id")
            past_id = row_dict.get("past_id")
            if current_id and past_id:
                id_history_list.append(CustomerIdHistory(
                    internal_customer_id=str(current_id),
                    past_id=str(past_id),
                    merge_timestamp=datetime.now(),  # BQ table may not have timestamp
                ))

        return id_history_list

    def _load_customer_properties(
        self,
        customer_config: CustomerTableConfig,
    ) -> dict[str, dict[str, Any]]:
        """Load customer properties from BigQuery."""

        full_table = self.config.get_full_table_name(customer_config.table_name)

        query = f"""
        SELECT *
        FROM `{full_table}`
        """

        if self.config.limit_per_table:
            query += f" LIMIT {self.config.limit_per_table}"

        query_job = self.client.query(query)
        rows = list(query_job.result())

        customer_properties: dict[str, dict[str, Any]] = {}

        for row in rows:
            row_dict = dict(row)
            customer_id = row_dict.get(customer_config.customer_id_field)

            if not customer_id:
                continue

            # Extract mapped properties
            props: dict[str, Any] = {}

            for target_field, mapping in customer_config.property_mappings.items():
                value = get_nested_value(row_dict, mapping.source_field)
                if value is None and mapping.default is not None:
                    value = mapping.default
                value = transform_value(value, mapping.transform)
                if value is not None:
                    props[target_field] = value

            # Auto-detect known fields if enabled
            if customer_config.auto_detect_fields:
                self._auto_detect_properties(row_dict, props, customer_config.known_fields)

            customer_properties[str(customer_id)] = props

        return customer_properties

    def _auto_detect_properties(
        self,
        row: dict[str, Any],
        props: dict[str, Any],
        known_fields: list[str],
    ) -> None:
        """Auto-detect and extract known fields from row."""

        # Check top-level fields
        for field_name in known_fields:
            if field_name in row and row[field_name] is not None:
                if field_name not in props:
                    props[field_name] = row[field_name]

        # Check nested properties
        for struct_name in ["properties", "raw_properties"]:
            struct = row.get(struct_name)
            if isinstance(struct, dict):
                for field_name in known_fields:
                    # Try exact match and variations
                    for variant in [field_name, field_name.replace("_", "")]:
                        if variant in struct and struct[variant] is not None:
                            if field_name not in props:
                                props[field_name] = struct[variant]
                            break

                        # Try with common suffixes (like __hash)
                        for key in struct:
                            if key.startswith(variant) and struct[key] is not None:
                                if field_name not in props:
                                    props[field_name] = struct[key]
                                break


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def load_from_bigquery(config: BigQueryConfig) -> LoadResult:
    """
    Load data from BigQuery using configuration.

    Args:
        config: BigQuery configuration

    Returns:
        LoadResult with events, ID history, and customer properties
    """
    adapter = BigQueryAdapter(config)
    return adapter.load()


def create_config_from_tables(
    project_id: str,
    dataset_id: str,
    *,
    purchase_table: str | None = None,
    page_view_table: str | None = None,
    view_item_table: str | None = None,
    add_to_cart_table: str | None = None,
    customer_table: str | None = None,
    merge_table: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
) -> BigQueryConfig:
    """
    Create a BigQuery config from table names with sensible defaults.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        purchase_table: Name of purchase events table
        page_view_table: Name of page view events table
        view_item_table: Name of view item events table
        add_to_cart_table: Name of add to cart events table
        customer_table: Name of customer properties table
        merge_table: Name of ID merge table
        start_date: Start date filter (YYYY-MM-DD)
        end_date: End date filter (YYYY-MM-DD)
        limit: Row limit per table

    Returns:
        BigQueryConfig ready for loading
    """
    event_tables = []

    if purchase_table:
        event_tables.append(EventTableConfig.purchase_table(purchase_table))

    if page_view_table:
        event_tables.append(EventTableConfig.page_view_table(page_view_table))

    if view_item_table:
        event_tables.append(EventTableConfig.view_item_table(view_item_table))

    if add_to_cart_table:
        event_tables.append(EventTableConfig.add_to_cart_table(add_to_cart_table))

    return BigQueryConfig(
        project_id=project_id,
        dataset_id=dataset_id,
        event_tables=event_tables,
        customer_table=CustomerTableConfig.default(customer_table) if customer_table else None,
        merge_table=MergeTableConfig(merge_table) if merge_table else None,
        start_date=start_date,
        end_date=end_date,
        limit_per_table=limit,
    )
