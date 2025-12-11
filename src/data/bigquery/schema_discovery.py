"""
BigQuery schema discovery with LLM-based field mapping.

Automatically scans BigQuery tables and uses an LLM to infer:
- Event types from table names and schemas
- Field mappings (customer_id, timestamp, properties)
- Data transformations needed
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, TYPE_CHECKING

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

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ColumnInfo:
    """Information about a BigQuery column."""

    name: str
    data_type: str
    mode: str  # NULLABLE, REQUIRED, REPEATED
    description: str | None = None
    is_nested: bool = False
    nested_fields: list["ColumnInfo"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM prompt."""
        result = {
            "name": self.name,
            "type": self.data_type,
            "mode": self.mode,
        }
        if self.description:
            result["description"] = self.description
        if self.nested_fields:
            result["nested_fields"] = [f.to_dict() for f in self.nested_fields]
        return result


@dataclass
class TableSchema:
    """Schema information for a BigQuery table."""

    table_name: str
    full_name: str  # project.dataset.table
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: int | None = None
    sample_rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for LLM prompt."""
        return {
            "table_name": self.table_name,
            "full_name": self.full_name,
            "columns": [c.to_dict() for c in self.columns],
            "row_count": self.row_count,
            "sample_rows": self.sample_rows[:3] if self.sample_rows else [],
        }


@dataclass
class DiscoveredTable:
    """Result of analyzing a single table."""

    schema: TableSchema
    inferred_type: str  # "event", "customer", "merge", "unknown", "irrelevant"
    event_type: EventType | None = None
    confidence: float = 0.0
    suggested_config: EventTableConfig | CustomerTableConfig | MergeTableConfig | None = None
    reasoning: str = ""
    is_relevant: bool = True  # False for system/internal tables
    relevance_score: float = 1.0  # 0-1, higher = more relevant


# =============================================================================
# TABLE RELEVANCE FILTER
# =============================================================================


class TableRelevanceFilter:
    """
    Filters out irrelevant tables based on naming patterns and content.

    Identifies tables that are:
    - System/internal tables (logs, metadata, temp)
    - ETL/pipeline artifacts (staging, backup, archive)
    - Derived/aggregated tables (not raw events)
    - Domain-irrelevant tables (delivery tracking, etc.)
    """

    # Tables to exclude based on name patterns
    EXCLUDE_PATTERNS = [
        # System/internal tables
        "log", "logs", "_log",
        "metadata", "_meta",
        "temp", "tmp", "_temp",
        "backup", "_backup", "_bak",
        "archive", "_archive",
        "staging", "_staging", "_stg",
        "test", "_test",

        # ETL artifacts
        "etl_", "_etl",
        "sync_", "_sync",
        "import_", "_import",
        "export_", "_export",
        "migration_",

        # Aggregated/derived tables (we want raw events)
        "_agg", "_aggregated",
        "_summary", "_stats",
        "_daily", "_weekly", "_monthly",
        "_rollup",

        # Delivery/logistics (usually not customer behavior)
        "delivery", "shipping", "logistics",
        "warehouse", "inventory",
        "fulfillment",
        "delayer",  # CDP internal tables

        # System tables
        "schema_", "_schema",
        "config_", "_config",
        "settings_",

        # CDP internal/system tables
        "job_", "_job",
        "task_", "_task",
        "queue_", "_queue",
        "notification",
        "audit_",
    ]

    # Tables to include - these patterns indicate relevant customer data
    # Note: These are checked FIRST, so be specific to avoid false positives
    INCLUDE_PATTERNS = [
        "purchase", "transaction",
        "page_view", "pageview", "view_item",
        "add_to_cart", "cart",
        "session_start", "session_end", "visit",
        "customer", "user", "profile",
        "search", "click",
        "email_open", "email_click", "campaign",
        "wishlist", "favorite",
        "checkout", "begin_checkout",
        "merge", "id_history", "identity",
        # Be more specific with "order" to avoid "delayer_order"
    ]

    # Patterns that are ambiguous - need additional context
    AMBIGUOUS_PATTERNS = [
        "order",  # Could be purchase order or system order
        "event",  # Could be behavioral event or system event
    ]

    # Minimum row count to be considered relevant
    MIN_ROW_COUNT = 100

    @classmethod
    def is_relevant(
        cls,
        table_name: str,
        row_count: int | None = None,
        columns: list[ColumnInfo] | None = None,
    ) -> tuple[bool, float, str]:
        """
        Determine if a table is relevant for customer segmentation.

        Args:
            table_name: Name of the table
            row_count: Number of rows (if known)
            columns: Table columns (if known)

        Returns:
            Tuple of (is_relevant, relevance_score, reason)
        """
        table_lower = table_name.lower()

        # Check exclude patterns FIRST (high priority exclusion)
        for pattern in cls.EXCLUDE_PATTERNS:
            if pattern in table_lower:
                return False, 0.1, f"Matches exclude pattern: {pattern}"

        # Check explicit include patterns (known relevant tables)
        for pattern in cls.INCLUDE_PATTERNS:
            if pattern in table_lower:
                return True, 0.9, f"Matches relevant pattern: {pattern}"

        # Check ambiguous patterns - return moderate score
        for pattern in cls.AMBIGUOUS_PATTERNS:
            if pattern in table_lower:
                return True, 0.6, f"Matches ambiguous pattern: {pattern} (needs review)"

        # Check row count
        if row_count is not None and row_count < cls.MIN_ROW_COUNT:
            return False, 0.2, f"Too few rows ({row_count} < {cls.MIN_ROW_COUNT})"

        # Check for required columns (customer_id, timestamp)
        if columns:
            has_customer_id = any(
                col.name.lower() in ("internal_customer_id", "customer_id", "user_id")
                for col in columns
            )
            has_timestamp = any(
                col.name.lower() in ("timestamp", "event_timestamp", "created_at", "time")
                or col.data_type in ("TIMESTAMP", "DATETIME")
                for col in columns
            )

            if has_customer_id and has_timestamp:
                return True, 0.8, "Has customer_id and timestamp columns"
            elif has_customer_id:
                return True, 0.6, "Has customer_id column"
            elif not has_customer_id and not has_timestamp:
                return False, 0.3, "Missing customer_id and timestamp columns"

        # Default: uncertain, include with moderate score
        return True, 0.5, "Could not determine relevance"

    @classmethod
    def filter_tables(
        cls,
        tables: list[str],
        get_table_info: Callable[[str], tuple[int | None, list[ColumnInfo] | None]] | None = None,
    ) -> tuple[list[str], list[tuple[str, str]]]:
        """
        Filter list of tables to only relevant ones.

        Args:
            tables: List of table names
            get_table_info: Optional callable to get (row_count, columns) for a table

        Returns:
            Tuple of (relevant_tables, excluded_tables_with_reasons)
        """
        relevant = []
        excluded = []

        for table in tables:
            row_count = None
            columns = None

            if get_table_info:
                try:
                    row_count, columns = get_table_info(table)
                except Exception:
                    pass

            is_rel, score, reason = cls.is_relevant(table, row_count, columns)

            if is_rel and score >= 0.5:
                relevant.append(table)
            else:
                excluded.append((table, reason))

        return relevant, excluded


@dataclass
class DiscoveryResult:
    """Complete result of schema discovery."""

    project_id: str
    dataset_id: str
    tables: list[DiscoveredTable] = field(default_factory=list)
    excluded_tables: list[tuple[str, str]] = field(default_factory=list)  # (name, reason)
    suggested_config: BigQueryConfig | None = None
    warnings: list[str] = field(default_factory=list)
    discovery_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def relevant_tables(self) -> list[DiscoveredTable]:
        """Get only relevant tables."""
        return [t for t in self.tables if t.is_relevant]

    @property
    def irrelevant_tables(self) -> list[DiscoveredTable]:
        """Get irrelevant tables."""
        return [t for t in self.tables if not t.is_relevant]


# =============================================================================
# SCHEMA DISCOVERY
# =============================================================================


class SchemaDiscovery:
    """
    Discovers and analyzes BigQuery schemas using LLM inference.

    Usage:
        discovery = SchemaDiscovery(project_id="my-project", dataset_id="my_dataset")
        result = discovery.discover()
        config = result.suggested_config
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        *,
        llm_provider: str = "anthropic",
        sample_rows: int = 5,
    ):
        """
        Initialize schema discovery.

        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
            llm_provider: LLM provider for inference ("anthropic" or "openai")
            sample_rows: Number of sample rows to fetch per table
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.llm_provider = llm_provider
        self.sample_rows = sample_rows
        self._client: Any = None
        self._bigquery_module: Any = None

    def _get_bigquery(self) -> Any:
        """Lazy import of BigQuery module."""
        if self._bigquery_module is None:
            try:
                from google.cloud import bigquery
                self._bigquery_module = bigquery
            except ImportError:
                raise ImportError(
                    "google-cloud-bigquery is required. "
                    "Install with: pip install google-cloud-bigquery"
                )
        return self._bigquery_module

    @property
    def client(self) -> Any:
        """Get or create BigQuery client."""
        if self._client is None:
            bigquery = self._get_bigquery()
            self._client = bigquery.Client(project=self.project_id)
        return self._client

    def discover(self, filter_irrelevant: bool = True) -> DiscoveryResult:
        """
        Discover all tables in dataset and infer their types/mappings.

        Args:
            filter_irrelevant: If True, skip tables that appear irrelevant
                              (system tables, ETL artifacts, etc.)

        Returns:
            DiscoveryResult with analyzed tables and suggested config
        """
        result = DiscoveryResult(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
        )

        # List all tables in dataset
        all_tables = self._list_tables()
        logger.info(f"Found {len(all_tables)} tables in {self.dataset_id}")

        # Pre-filter tables based on naming patterns
        if filter_irrelevant:
            tables, excluded = TableRelevanceFilter.filter_tables(all_tables)
            result.excluded_tables = excluded
            for table_name, reason in excluded:
                logger.debug(f"Skipped {table_name}: {reason}")
            logger.info(
                f"After relevance filter: {len(tables)} relevant, "
                f"{len(excluded)} excluded"
            )
        else:
            tables = all_tables

        # Analyze each table
        for table_name in tables:
            try:
                schema = self._get_table_schema(table_name)

                # Additional relevance check with schema info
                if filter_irrelevant:
                    is_rel, rel_score, rel_reason = TableRelevanceFilter.is_relevant(
                        table_name,
                        row_count=schema.row_count,
                        columns=schema.columns,
                    )
                    if not is_rel or rel_score < 0.5:
                        result.tables.append(DiscoveredTable(
                            schema=schema,
                            inferred_type="irrelevant",
                            confidence=rel_score,
                            reasoning=rel_reason,
                            is_relevant=False,
                            relevance_score=rel_score,
                        ))
                        result.warnings.append(f"Marked {table_name} as irrelevant: {rel_reason}")
                        continue

                discovered = self._analyze_table(schema)

                # Set relevance info
                is_rel, rel_score, _ = TableRelevanceFilter.is_relevant(
                    table_name, schema.row_count, schema.columns
                )
                discovered.is_relevant = is_rel
                discovered.relevance_score = rel_score

                result.tables.append(discovered)
            except Exception as e:
                logger.warning(f"Error analyzing table {table_name}: {e}")
                result.warnings.append(f"Failed to analyze {table_name}: {str(e)}")

        # Generate suggested config from discovered tables (only relevant ones)
        relevant_tables = [t for t in result.tables if t.is_relevant]
        result.suggested_config = self._build_suggested_config(relevant_tables)

        return result

    def _list_tables(self) -> list[str]:
        """List all tables in the dataset."""
        dataset_ref = f"{self.project_id}.{self.dataset_id}"
        tables = list(self.client.list_tables(dataset_ref))
        return [t.table_id for t in tables]

    def _get_table_schema(self, table_name: str) -> TableSchema:
        """Get schema information for a table."""
        full_name = f"{self.project_id}.{self.dataset_id}.{table_name}"
        table = self.client.get_table(full_name)

        columns = []
        for field in table.schema:
            col = self._schema_field_to_column(field)
            columns.append(col)

        schema = TableSchema(
            table_name=table_name,
            full_name=full_name,
            columns=columns,
            row_count=table.num_rows,
        )

        # Get sample rows
        if self.sample_rows > 0:
            schema.sample_rows = self._get_sample_rows(full_name)

        return schema

    def _schema_field_to_column(self, field: Any) -> ColumnInfo:
        """Convert BigQuery schema field to ColumnInfo."""
        nested_fields = []
        is_nested = field.field_type in ("RECORD", "STRUCT")

        if is_nested and field.fields:
            nested_fields = [
                self._schema_field_to_column(f) for f in field.fields
            ]

        return ColumnInfo(
            name=field.name,
            data_type=field.field_type,
            mode=field.mode,
            description=field.description,
            is_nested=is_nested,
            nested_fields=nested_fields,
        )

    def _get_sample_rows(self, full_table: str) -> list[dict[str, Any]]:
        """Get sample rows from table."""
        query = f"SELECT * FROM `{full_table}` LIMIT {self.sample_rows}"
        try:
            rows = list(self.client.query(query).result())
            return [dict(row) for row in rows]
        except Exception as e:
            logger.warning(f"Could not get sample rows: {e}")
            return []

    def _analyze_table(self, schema: TableSchema) -> DiscoveredTable:
        """Analyze a table schema using LLM inference."""
        # First, do rule-based analysis
        rule_result = self._rule_based_analysis(schema)

        # If confident, use rule-based result
        if rule_result.confidence >= 0.8:
            return rule_result

        # Otherwise, use LLM for more sophisticated inference
        llm_result = self._llm_analysis(schema, rule_result)
        return llm_result

    def _rule_based_analysis(self, schema: TableSchema) -> DiscoveredTable:
        """Apply rule-based heuristics to identify table type."""
        table_lower = schema.table_name.lower()
        column_names = {c.name.lower() for c in schema.columns}

        # Check for merge/identity table
        if any(kw in table_lower for kw in ["merge", "id_history", "identity"]):
            if "past_id" in column_names or "previous_id" in column_names:
                return DiscoveredTable(
                    schema=schema,
                    inferred_type="merge",
                    confidence=0.9,
                    suggested_config=self._suggest_merge_config(schema),
                    reasoning="Table name and columns suggest ID merge table",
                )

        # Check for customer properties table
        if any(kw in table_lower for kw in ["customer", "user", "profile"]):
            if "properties" in column_names or "email" in column_names:
                return DiscoveredTable(
                    schema=schema,
                    inferred_type="customer",
                    confidence=0.85,
                    suggested_config=self._suggest_customer_config(schema),
                    reasoning="Table name and columns suggest customer properties",
                )

        # Check for event tables
        event_type = self._infer_event_type_from_name(table_lower)
        if event_type:
            has_timestamp = any(
                kw in column_names
                for kw in ["timestamp", "created_at", "event_time", "time"]
            )
            has_customer_id = any(
                kw in column_names
                for kw in ["customer_id", "user_id", "internal_customer_id"]
            )

            if has_timestamp and has_customer_id:
                return DiscoveredTable(
                    schema=schema,
                    inferred_type="event",
                    event_type=event_type,
                    confidence=0.85,
                    suggested_config=self._suggest_event_config(schema, event_type),
                    reasoning=f"Table name suggests {event_type.value} event type",
                )

        # Unknown table type
        return DiscoveredTable(
            schema=schema,
            inferred_type="unknown",
            confidence=0.3,
            reasoning="Could not determine table type from name/columns",
        )

    def _infer_event_type_from_name(self, table_name: str) -> EventType | None:
        """Infer event type from table name."""
        name_to_type = {
            "purchase": EventType.PURCHASE,
            "order": EventType.PURCHASE,
            "transaction": EventType.PURCHASE,
            "page_view": EventType.PAGE_VIEW,
            "pageview": EventType.PAGE_VIEW,
            "view_item": EventType.VIEW_ITEM,
            "product_view": EventType.VIEW_ITEM,
            "item_view": EventType.VIEW_ITEM,
            "add_to_cart": EventType.ADD_TO_CART,
            "cart_add": EventType.ADD_TO_CART,
            "add_cart": EventType.ADD_TO_CART,
            "remove_from_cart": EventType.REMOVE_FROM_CART,
            "cart_remove": EventType.REMOVE_FROM_CART,
            "checkout": EventType.BEGIN_CHECKOUT,
            "begin_checkout": EventType.BEGIN_CHECKOUT,
            "session_start": EventType.SESSION_START,
            "session": EventType.SESSION_START,
            "search": EventType.SEARCH,
            "wishlist": EventType.WISHLIST_ADD,
            "email_open": EventType.EMAIL_OPEN,
            "email_click": EventType.EMAIL_CLICK,
        }

        for pattern, event_type in name_to_type.items():
            if pattern in table_name:
                return event_type

        return None

    def _suggest_event_config(
        self,
        schema: TableSchema,
        event_type: EventType,
    ) -> EventTableConfig:
        """Suggest EventTableConfig based on schema."""
        # Find customer ID field
        customer_id_field = self._find_customer_id_field(schema)

        # Find timestamp field
        timestamp_field = self._find_timestamp_field(schema)

        # Build property mappings based on event type
        property_mappings = self._infer_property_mappings(schema, event_type)

        return EventTableConfig(
            table_name=schema.table_name,
            event_type=event_type,
            customer_id_field=customer_id_field,
            timestamp_field=timestamp_field,
            property_mappings=property_mappings,
        )

    def _suggest_customer_config(self, schema: TableSchema) -> CustomerTableConfig:
        """Suggest CustomerTableConfig based on schema."""
        customer_id_field = self._find_customer_id_field(schema)
        property_mappings = self._infer_customer_property_mappings(schema)

        return CustomerTableConfig(
            table_name=schema.table_name,
            customer_id_field=customer_id_field,
            property_mappings=property_mappings,
        )

    def _suggest_merge_config(self, schema: TableSchema) -> MergeTableConfig:
        """Suggest MergeTableConfig based on schema."""
        column_names = {c.name.lower(): c.name for c in schema.columns}

        # Find current ID field
        current_id = "internal_customer_id"
        for candidate in ["internal_customer_id", "customer_id", "current_id", "id"]:
            if candidate in column_names:
                current_id = column_names[candidate]
                break

        # Find past ID field
        past_id = "past_id"
        for candidate in ["past_id", "previous_id", "old_id", "merged_from"]:
            if candidate in column_names:
                past_id = column_names[candidate]
                break

        return MergeTableConfig(
            table_name=schema.table_name,
            current_id_field=current_id,
            past_id_field=past_id,
        )

    def _find_customer_id_field(self, schema: TableSchema) -> str:
        """Find the customer ID field in schema."""
        candidates = [
            "internal_customer_id",
            "customer_id",
            "user_id",
            "client_id",
            "member_id",
        ]

        column_names = {c.name.lower(): c.name for c in schema.columns}

        for candidate in candidates:
            if candidate in column_names:
                return column_names[candidate]

        # Fallback to first column with "id" in name
        for col in schema.columns:
            if "customer" in col.name.lower() and "id" in col.name.lower():
                return col.name

        return "internal_customer_id"  # Default

    def _find_timestamp_field(self, schema: TableSchema) -> str:
        """Find the timestamp field in schema."""
        candidates = [
            "timestamp",
            "event_timestamp",
            "created_at",
            "event_time",
            "time",
            "date",
        ]

        column_names = {c.name.lower(): c.name for c in schema.columns}

        for candidate in candidates:
            if candidate in column_names:
                return column_names[candidate]

        # Look for TIMESTAMP type columns
        for col in schema.columns:
            if col.data_type in ("TIMESTAMP", "DATETIME"):
                return col.name

        return "timestamp"  # Default

    def _infer_property_mappings(
        self,
        schema: TableSchema,
        event_type: EventType,
    ) -> dict[str, FieldMapping]:
        """Infer property mappings based on schema and event type."""
        mappings: dict[str, FieldMapping] = {}

        # Check if there's a nested 'properties' field
        properties_field = None
        for col in schema.columns:
            if col.name.lower() == "properties" and col.is_nested:
                properties_field = col
                break

        if properties_field:
            # Map from nested properties
            mappings = self._map_nested_properties(properties_field, event_type)
        else:
            # Map from flat columns
            mappings = self._map_flat_columns(schema, event_type)

        return mappings

    def _map_nested_properties(
        self,
        properties_field: ColumnInfo,
        event_type: EventType,
    ) -> dict[str, FieldMapping]:
        """Map nested properties to our internal fields."""
        mappings: dict[str, FieldMapping] = {}

        nested_names = {f.name.lower(): f for f in properties_field.nested_fields}

        # Common mappings for all event types
        common_mappings = {
            "device": ("device", None),
            "device_type": ("device", None),
            "channel": ("channel", None),
            "session_id": ("session_id", None),
        }

        # Event-specific mappings
        event_mappings: dict[EventType, dict[str, tuple[str, str | None]]] = {
            EventType.PURCHASE: {
                "purchase_id": ("order_id", None),
                "order_id": ("order_id", None),
                "total_price": ("order_total", "decimal"),
                "total_amount": ("order_total", "decimal"),
                "product_list": ("product_list", "json_parse"),
                "products": ("product_list", "json_parse"),
                "payment_method": ("payment_method", None),
                "shipping_country": ("shipping_country", None),
            },
            EventType.PAGE_VIEW: {
                "page_url": ("page_url", None),
                "url": ("page_url", None),
                "page_title": ("page_title", None),
                "title": ("page_title", None),
            },
            EventType.VIEW_ITEM: {
                "product_id": ("product_id", None),
                "item_id": ("product_id", None),
                "product_name": ("product_name", None),
                "item_name": ("product_name", None),
                "product_category": ("product_category", None),
                "category": ("product_category", None),
                "product_price": ("product_price", "decimal"),
                "price": ("product_price", "decimal"),
            },
            EventType.ADD_TO_CART: {
                "product_id": ("product_id", None),
                "item_id": ("product_id", None),
                "product_name": ("product_name", None),
                "product_category": ("product_category", None),
                "product_price": ("product_price", "decimal"),
                "quantity": ("quantity", "int"),
            },
            EventType.SEARCH: {
                "query": ("search_query", None),
                "search_query": ("search_query", None),
                "search_term": ("search_query", None),
            },
        }

        # Apply common mappings
        for source, (target, transform) in common_mappings.items():
            if source in nested_names:
                mappings[target] = FieldMapping(
                    source_field=f"properties.{nested_names[source].name}",
                    target_field=target,
                    transform=transform,
                )

        # Apply event-specific mappings
        if event_type in event_mappings:
            for source, (target, transform) in event_mappings[event_type].items():
                if source in nested_names and target not in mappings:
                    mappings[target] = FieldMapping(
                        source_field=f"properties.{nested_names[source].name}",
                        target_field=target,
                        transform=transform,
                    )

        return mappings

    def _map_flat_columns(
        self,
        schema: TableSchema,
        event_type: EventType,
    ) -> dict[str, FieldMapping]:
        """Map flat columns to our internal fields."""
        mappings: dict[str, FieldMapping] = {}
        column_names = {c.name.lower(): c for c in schema.columns}

        # Similar logic but for flat columns
        flat_mappings = {
            "order_id": ("order_id", None),
            "purchase_id": ("order_id", None),
            "total_price": ("order_total", "decimal"),
            "total_amount": ("order_total", "decimal"),
            "product_id": ("product_id", None),
            "product_name": ("product_name", None),
            "product_category": ("product_category", None),
            "category": ("product_category", None),
            "product_price": ("product_price", "decimal"),
            "price": ("product_price", "decimal"),
            "quantity": ("quantity", "int"),
            "page_url": ("page_url", None),
            "url": ("page_url", None),
            "page_title": ("page_title", None),
            "search_query": ("search_query", None),
            "query": ("search_query", None),
            "device": ("device", None),
            "device_type": ("device", None),
        }

        for source, (target, transform) in flat_mappings.items():
            if source in column_names and target not in mappings:
                col = column_names[source]
                mappings[target] = FieldMapping(
                    source_field=col.name,
                    target_field=target,
                    transform=transform,
                )

        return mappings

    def _infer_customer_property_mappings(
        self,
        schema: TableSchema,
    ) -> dict[str, FieldMapping]:
        """Infer customer property mappings from schema."""
        mappings: dict[str, FieldMapping] = {}

        # Check for nested properties
        properties_field = None
        for col in schema.columns:
            if col.name.lower() == "properties" and col.is_nested:
                properties_field = col
                break

        customer_fields = {
            "email": ("email", None),
            "first_name": ("first_name", None),
            "last_name": ("last_name", None),
            "country": ("country", None),
            "language": ("language", None),
            "gender": ("gender", None),
            "birth_date": ("birth_date", None),
            "phone": ("phone", None),
            "newsletter": ("newsletter", "bool"),
            "double_optin": ("double_optin", "bool"),
            "rfm_today": ("rfm_segment", None),
            "rfm_simplified_today": ("rfm_segment", None),
            "engagement_level": ("engagement_level", None),
        }

        if properties_field:
            nested_names = {f.name.lower(): f for f in properties_field.nested_fields}
            for source, (target, transform) in customer_fields.items():
                if source in nested_names:
                    mappings[target] = FieldMapping(
                        source_field=f"properties.{nested_names[source].name}",
                        target_field=target,
                        transform=transform,
                    )
        else:
            column_names = {c.name.lower(): c for c in schema.columns}
            for source, (target, transform) in customer_fields.items():
                if source in column_names:
                    mappings[target] = FieldMapping(
                        source_field=column_names[source].name,
                        target_field=target,
                        transform=transform,
                    )

        return mappings

    def _llm_analysis(
        self,
        schema: TableSchema,
        rule_result: DiscoveredTable,
    ) -> DiscoveredTable:
        """Use LLM for more sophisticated schema analysis."""
        try:
            from src.llm.client import get_llm_client
        except ImportError:
            logger.warning("LLM client not available, using rule-based result")
            return rule_result

        prompt = self._build_llm_prompt(schema)

        try:
            client = get_llm_client(self.llm_provider)
            response = client.complete(
                prompt=prompt,
                system="You are a data engineer analyzing BigQuery schemas. "
                       "Respond with valid JSON only.",
                max_tokens=1000,
            )

            analysis = json.loads(response)
            return self._parse_llm_response(schema, analysis)

        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}, using rule-based result")
            return rule_result

    def _build_llm_prompt(self, schema: TableSchema) -> str:
        """Build prompt for LLM analysis."""
        schema_json = json.dumps(schema.to_dict(), indent=2, default=str)

        return f"""Analyze this BigQuery table schema and determine:
1. Table type: "event" (behavioral events), "customer" (customer properties), "merge" (ID merge/history), or "unknown"
2. If event table, what event type: purchase, page_view, view_item, add_to_cart, search, session_start, etc.
3. Key field mappings:
   - customer_id_field: which column contains the customer identifier
   - timestamp_field: which column contains the event timestamp
   - property_mappings: map source fields to standard names (order_id, product_id, etc.)

Schema:
{schema_json}

Respond with JSON in this format:
{{
    "table_type": "event|customer|merge|unknown",
    "event_type": "purchase|page_view|view_item|add_to_cart|search|session_start|null",
    "confidence": 0.0-1.0,
    "customer_id_field": "field_name",
    "timestamp_field": "field_name",
    "property_mappings": {{
        "target_field": {{"source": "source_field", "transform": "decimal|int|bool|null"}}
    }},
    "reasoning": "explanation"
}}"""

    def _parse_llm_response(
        self,
        schema: TableSchema,
        analysis: dict[str, Any],
    ) -> DiscoveredTable:
        """Parse LLM response into DiscoveredTable."""
        table_type = analysis.get("table_type", "unknown")
        event_type_str = analysis.get("event_type")
        confidence = analysis.get("confidence", 0.5)
        reasoning = analysis.get("reasoning", "")

        event_type = None
        if event_type_str:
            try:
                event_type = EventType(event_type_str)
            except ValueError:
                pass

        # Build suggested config based on LLM analysis
        suggested_config = None

        if table_type == "event" and event_type:
            property_mappings = {}
            for target, mapping in analysis.get("property_mappings", {}).items():
                if isinstance(mapping, dict):
                    property_mappings[target] = FieldMapping(
                        source_field=mapping.get("source", target),
                        target_field=target,
                        transform=mapping.get("transform"),
                    )

            suggested_config = EventTableConfig(
                table_name=schema.table_name,
                event_type=event_type,
                customer_id_field=analysis.get("customer_id_field", "internal_customer_id"),
                timestamp_field=analysis.get("timestamp_field", "timestamp"),
                property_mappings=property_mappings,
            )

        elif table_type == "customer":
            suggested_config = self._suggest_customer_config(schema)

        elif table_type == "merge":
            suggested_config = self._suggest_merge_config(schema)

        return DiscoveredTable(
            schema=schema,
            inferred_type=table_type,
            event_type=event_type,
            confidence=confidence,
            suggested_config=suggested_config,
            reasoning=reasoning,
        )

    def _build_suggested_config(
        self,
        tables: list[DiscoveredTable],
    ) -> BigQueryConfig:
        """Build complete BigQueryConfig from discovered tables."""
        event_tables = []
        customer_table = None
        merge_table = None

        for table in tables:
            if table.confidence < 0.5:
                continue

            if table.inferred_type == "event" and isinstance(
                table.suggested_config, EventTableConfig
            ):
                event_tables.append(table.suggested_config)

            elif table.inferred_type == "customer" and isinstance(
                table.suggested_config, CustomerTableConfig
            ):
                customer_table = table.suggested_config

            elif table.inferred_type == "merge" and isinstance(
                table.suggested_config, MergeTableConfig
            ):
                merge_table = table.suggested_config

        return BigQueryConfig(
            project_id=self.project_id,
            dataset_id=self.dataset_id,
            event_tables=event_tables,
            customer_table=customer_table,
            merge_table=merge_table,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def discover_schema(
    project_id: str,
    dataset_id: str,
    *,
    sample_rows: int = 5,
) -> DiscoveryResult:
    """
    Discover BigQuery schema and suggest configuration.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        sample_rows: Number of sample rows to fetch per table

    Returns:
        DiscoveryResult with analyzed tables and suggested config
    """
    discovery = SchemaDiscovery(
        project_id=project_id,
        dataset_id=dataset_id,
        sample_rows=sample_rows,
    )
    return discovery.discover()


def print_discovery_report(result: DiscoveryResult) -> str:
    """Generate human-readable discovery report."""
    relevant_count = len(result.relevant_tables)
    excluded_count = len(result.excluded_tables) + len(result.irrelevant_tables)

    lines = [
        "=" * 60,
        "BigQuery Schema Discovery Report",
        "=" * 60,
        f"Project: {result.project_id}",
        f"Dataset: {result.dataset_id}",
        f"Tables found: {len(result.tables) + len(result.excluded_tables)}",
        f"Relevant tables: {relevant_count}",
        f"Excluded tables: {excluded_count}",
        "",
    ]

    # Group relevant tables by type
    by_type: dict[str, list[DiscoveredTable]] = {}
    for table in result.relevant_tables:
        by_type.setdefault(table.inferred_type, []).append(table)

    lines.append("RELEVANT TABLES")
    lines.append("=" * 40)

    for table_type, tables in by_type.items():
        lines.append(f"\n{table_type.upper()} ({len(tables)}):")
        lines.append("-" * 40)

        for table in tables:
            lines.append(f"\n  {table.schema.table_name}")
            lines.append(f"    Relevance: {table.relevance_score:.0%}")
            lines.append(f"    Confidence: {table.confidence:.0%}")
            if table.event_type:
                lines.append(f"    Event Type: {table.event_type.value}")
            lines.append(f"    Reasoning: {table.reasoning}")
            if table.schema.row_count:
                lines.append(f"    Row Count: {table.schema.row_count:,}")

            if table.suggested_config:
                if isinstance(table.suggested_config, EventTableConfig):
                    lines.append(f"    Customer ID: {table.suggested_config.customer_id_field}")
                    lines.append(f"    Timestamp: {table.suggested_config.timestamp_field}")
                    lines.append(f"    Mappings: {list(table.suggested_config.property_mappings.keys())}")

    # Show excluded tables (collapsed)
    if result.excluded_tables or result.irrelevant_tables:
        lines.append("\n\nEXCLUDED TABLES")
        lines.append("=" * 40)

        # Pre-filtered tables
        if result.excluded_tables:
            lines.append("\nFiltered by naming pattern:")
            for table_name, reason in result.excluded_tables[:10]:
                lines.append(f"  - {table_name}: {reason}")
            if len(result.excluded_tables) > 10:
                lines.append(f"  ... and {len(result.excluded_tables) - 10} more")

        # Tables marked irrelevant after schema analysis
        if result.irrelevant_tables:
            lines.append("\nMarked irrelevant after analysis:")
            for table in result.irrelevant_tables[:10]:
                lines.append(f"  - {table.schema.table_name}: {table.reasoning}")
            if len(result.irrelevant_tables) > 10:
                lines.append(f"  ... and {len(result.irrelevant_tables) - 10} more")

    if result.warnings:
        lines.append("\n\nWARNINGS:")
        for warning in result.warnings[:20]:
            lines.append(f"  - {warning}")
        if len(result.warnings) > 20:
            lines.append(f"  ... and {len(result.warnings) - 20} more")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)
