"""
Local data loader for parquet sample data.

Converts locally stored parquet files (extracted from BigQuery)
into EventRecord and CustomerIdHistory objects for the segmentation pipeline.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.data.schemas import (
    EventRecord,
    EventType,
    EventProperties,
    CustomerIdHistory,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TABLE TO EVENT TYPE MAPPING
# =============================================================================

TABLE_TO_EVENT_TYPE: dict[str, EventType] = {
    "purchase": EventType.PURCHASE,
    "purchase_item": EventType.PURCHASE_ITEM,
    "view_item": EventType.VIEW_ITEM,
    "view_category": EventType.VIEW_CATEGORY,
    "cart_update": EventType.ADD_TO_CART,
    "add_to_cart": EventType.ADD_TO_CART,
    "checkout": EventType.CHECKOUT,
    "session_start": EventType.SESSION_START,
    "session_end": EventType.SESSION_START,  # Map to session for engagement
    "page_visit": EventType.VIEW_ITEM,  # Page visits are engagement
    "search": EventType.VIEW_CATEGORY,  # Search is category exploration
}

# Tables that are NOT events (customer data tables)
NON_EVENT_TABLES = {
    "customers_properties",
    "customers_id_history",
    "customers_external_ids",
    "merge",
}


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class LoadResult:
    """Result of loading local sample data."""

    events: list[EventRecord] = field(default_factory=list)
    id_history: list[CustomerIdHistory] = field(default_factory=list)
    customer_properties: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Statistics
    tables_loaded: list[str] = field(default_factory=list)
    events_by_type: dict[str, int] = field(default_factory=dict)
    total_rows: int = 0
    unique_customers: int = 0
    load_duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)


# =============================================================================
# LOCAL DATA LOADER
# =============================================================================


class LocalDataLoader:
    """
    Load sample data from local parquet files.

    Converts parquet files (extracted from BigQuery) into EventRecord
    and CustomerIdHistory objects for use with the segmentation pipeline.

    Usage:
        loader = LocalDataLoader("data/samples")
        result = loader.load()

        # Use with pipeline
        from src.pipeline import run_pipeline, PipelineConfig
        pipeline_result = run_pipeline(
            events=result.events,
            id_history=result.id_history,
        )
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        include_tables: list[str] | None = None,
        exclude_tables: list[str] | None = None,
    ):
        """
        Initialize loader.

        Args:
            data_dir: Directory containing parquet files
            include_tables: Only load these tables (default: all)
            exclude_tables: Skip these tables
        """
        self.data_dir = Path(data_dir)
        self.include_tables = set(include_tables) if include_tables else None
        self.exclude_tables = set(exclude_tables) if exclude_tables else set()
        self._pandas_module: Any = None

    def _get_pandas(self) -> Any:
        """Lazy import of pandas."""
        if self._pandas_module is None:
            try:
                import pandas as pd
                self._pandas_module = pd
            except ImportError:
                raise ImportError(
                    "pandas is required: pip install pandas pyarrow"
                )
        return self._pandas_module

    def load(self) -> LoadResult:
        """
        Load all data from local parquet files.

        Returns:
            LoadResult with events, id_history, and statistics
        """
        import time
        start_time = time.perf_counter()

        result = LoadResult()
        pd = self._get_pandas()

        if not self.data_dir.exists():
            result.errors.append(f"Data directory not found: {self.data_dir}")
            return result

        # Track seen past_ids across both ID history sources to avoid duplicates
        seen_past_ids: set[str] = set()

        # Get list of parquet files
        parquet_files = list(self.data_dir.glob("*.parquet"))

        for path in parquet_files:
            table_name = path.stem

            # Check include/exclude
            if self.include_tables and table_name not in self.include_tables:
                continue
            if table_name in self.exclude_tables:
                continue

            try:
                df = pd.read_parquet(path)
                result.tables_loaded.append(table_name)
                result.total_rows += len(df)

                if table_name == "customers_id_history":
                    # Load ID history with deduplication
                    id_history = self._load_id_history(df, seen_past_ids)
                    result.id_history.extend(id_history)
                    logger.info(f"Loaded {len(id_history)} ID history records")

                elif table_name == "merge":
                    # Merge events contain ID merge info in properties
                    id_history = self._load_merge_events(df, seen_past_ids)
                    result.id_history.extend(id_history)
                    logger.info(f"Loaded {len(id_history)} merge records")

                elif table_name == "customers_properties":
                    # Load customer properties
                    props = self._load_customer_properties(df)
                    result.customer_properties.update(props)
                    logger.info(f"Loaded properties for {len(props)} customers")

                elif table_name not in NON_EVENT_TABLES:
                    # Load as events
                    events = self._load_events(df, table_name)
                    result.events.extend(events)
                    result.events_by_type[table_name] = len(events)
                    logger.info(f"Loaded {len(events)} events from {table_name}")

            except Exception as e:
                error_msg = f"Error loading {table_name}: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Calculate unique customers
        customer_ids = {e.internal_customer_id for e in result.events}
        result.unique_customers = len(customer_ids)

        result.load_duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            f"Loaded {len(result.events)} events, "
            f"{len(result.id_history)} ID merges, "
            f"{result.unique_customers} unique customers "
            f"in {result.load_duration_ms:.1f}ms"
        )

        return result

    def _load_events(
        self,
        df: Any,  # pd.DataFrame
        table_name: str,
    ) -> list[EventRecord]:
        """Convert DataFrame rows to EventRecord objects."""
        events: list[EventRecord] = []

        event_type = TABLE_TO_EVENT_TYPE.get(table_name, EventType.VIEW_ITEM)

        for _, row in df.iterrows():
            try:
                event = self._row_to_event(row, event_type)
                if event:
                    events.append(event)
            except Exception as e:
                logger.debug(f"Error converting row: {e}")
                continue

        return events

    def _row_to_event(
        self,
        row: Any,  # pd.Series
        event_type: EventType,
    ) -> EventRecord | None:
        """Convert a single row to EventRecord."""

        # Get required fields
        customer_id = row.get("internal_customer_id")
        timestamp = row.get("timestamp")

        if not customer_id or timestamp is None:
            return None

        # Convert timestamp
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        elif hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        # Ensure timezone awareness
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Get properties (nested dict from BigQuery)
        props = row.get("properties", {})
        if props is None:
            props = {}

        # Build EventProperties
        event_properties = self._build_event_properties(props, event_type)

        return EventRecord(
            event_id=str(uuid.uuid4()),
            internal_customer_id=str(customer_id),
            event_type=event_type,
            timestamp=timestamp,
            properties=event_properties,
        )

    def _build_event_properties(
        self,
        props: dict[str, Any],
        event_type: EventType,
    ) -> EventProperties:
        """Build EventProperties from raw properties dict."""

        # Helper to safely get value (handles numpy types)
        def safe_get(key: str, *alt_keys: str) -> Any:
            for k in [key, *alt_keys]:
                val = props.get(k)
                if val is not None:
                    # Handle numpy/pandas types
                    if hasattr(val, "item"):
                        return val.item()
                    return val
            return None

        # Extract common fields
        product_id = safe_get("product_id")
        product_name = safe_get("title", "product_name")
        product_category = safe_get("category_level_1", "product_category", "category")

        # Price handling
        product_price = None
        price_val = safe_get("price", "product_price")
        if price_val is not None:
            try:
                product_price = Decimal(str(price_val))
            except (ValueError, TypeError):
                pass

        # Order total for purchase events
        order_total = None
        total_val = safe_get("total_price", "order_total")
        if total_val is not None:
            try:
                order_total = Decimal(str(total_val))
            except (ValueError, TypeError):
                pass

        # Quantity
        quantity = None
        qty_val = safe_get("quantity", "total_quantity")
        if qty_val is not None:
            try:
                quantity = int(float(qty_val))
            except (ValueError, TypeError):
                pass

        # Device/channel
        device_type = safe_get("device", "device_type", "channel")

        # Search query
        search_query = safe_get("query", "search_query")

        # Page info
        page_url = safe_get("page_url", "url")
        page_title = safe_get("page_title", "title")

        # Order ID for purchases
        order_id = safe_get("purchase_id", "order_id")

        # Session ID
        session_id = safe_get("session_id")

        # Build custom properties for anything else
        known_fields = {
            "product_id", "title", "product_name", "category_level_1",
            "product_category", "category", "price", "product_price",
            "total_price", "order_total", "quantity", "total_quantity",
            "device", "device_type", "channel", "query", "search_query",
            "page_url", "url", "page_title", "purchase_id", "order_id",
            "session_id", "browser", "os", "location", "timestamp",
            "ingest_timestamp", "type", "internal_customer_id",
        }

        custom = {
            k: v for k, v in props.items()
            if k not in known_fields and v is not None
        }

        return EventProperties(
            product_id=str(product_id) if product_id else None,
            product_name=str(product_name) if product_name else None,
            product_category=str(product_category) if product_category else None,
            product_price=product_price,
            quantity=quantity,
            order_id=str(order_id) if order_id else None,
            order_total=order_total,
            search_query=str(search_query) if search_query else None,
            device_type=str(device_type) if device_type else None,
            session_id=str(session_id) if session_id else None,
            page_url=str(page_url) if page_url else None,
            page_title=str(page_title) if page_title else None,
            custom_properties=custom if custom else None,
        )

    def _load_id_history(
        self,
        df: Any,
        seen: set[str],
    ) -> list[CustomerIdHistory]:
        """Load CustomerIdHistory from customers_id_history table."""
        id_history: list[CustomerIdHistory] = []

        for _, row in df.iterrows():
            current_id = row.get("internal_customer_id")
            past_id = row.get("past_id")

            if current_id and past_id:
                past_id_str = str(past_id)
                # Skip duplicates - keep first occurrence
                if past_id_str in seen:
                    continue
                seen.add(past_id_str)

                id_history.append(CustomerIdHistory(
                    internal_customer_id=str(current_id),
                    past_id=past_id_str,
                    merge_timestamp=datetime.now(tz=timezone.utc),
                ))

        return id_history

    def _load_merge_events(
        self,
        df: Any,
        seen: set[str] | None = None,
    ) -> list[CustomerIdHistory]:
        """Load CustomerIdHistory from merge events table."""
        id_history: list[CustomerIdHistory] = []
        if seen is None:
            seen = set()

        for _, row in df.iterrows():
            current_id = row.get("internal_customer_id")
            props = row.get("properties", {})

            if not current_id or not props:
                continue

            # Source IDs are the ones being merged into current
            source_ids = props.get("source_internal_ids")

            if source_ids:
                # Parse if string
                if isinstance(source_ids, str):
                    try:
                        source_ids = json.loads(source_ids)
                    except json.JSONDecodeError:
                        source_ids = [source_ids]

                # Create history entry for each source
                for source_id in source_ids:
                    if source_id and source_id != current_id:
                        source_id_str = str(source_id)
                        # Skip duplicates
                        if source_id_str in seen:
                            continue
                        seen.add(source_id_str)

                        id_history.append(CustomerIdHistory(
                            internal_customer_id=str(current_id),
                            past_id=source_id_str,
                            merge_timestamp=datetime.now(tz=timezone.utc),
                        ))

        return id_history

    def _load_customer_properties(self, df: Any) -> dict[str, dict[str, Any]]:
        """Load customer properties from customers_properties table."""
        customer_props: dict[str, dict[str, Any]] = {}

        for _, row in df.iterrows():
            customer_id = row.get("internal_id") or row.get("internal_customer_id")
            props = row.get("properties", {})

            if customer_id and props:
                # Extract useful properties
                extracted = {}

                # Common customer properties
                property_map = {
                    "email": "email",
                    "first_name": "first_name",
                    "last_name": "last_name",
                    "country": "country",
                    "language": "language",
                    "gender": "gender",
                    "birth_date": "birth_date",
                    "phone": "phone",
                    "newsletter__f84ca2e7": "newsletter",
                    "double_optin": "double_optin",
                    "rfm_today__b3103f82": "rfm_segment",
                    "rfm_simplified_today__da2681cd": "rfm_simplified",
                    "engagement_level": "engagement_level",
                }

                for source_key, target_key in property_map.items():
                    if source_key in props and props[source_key] is not None:
                        extracted[target_key] = props[source_key]

                if extracted:
                    customer_props[str(customer_id)] = extracted

        return customer_props

    def load_metadata(self) -> dict[str, Any]:
        """Load extraction metadata if available."""
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}

    def list_tables(self) -> list[str]:
        """List available tables in data directory."""
        return [p.stem for p in self.data_dir.glob("*.parquet")]


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def load_local_data(
    data_dir: str | Path = "data/samples",
) -> LoadResult:
    """
    Load sample data from local parquet files.

    Args:
        data_dir: Directory containing parquet files

    Returns:
        LoadResult with events, id_history, and statistics

    Example:
        result = load_local_data("data/samples")
        print(f"Loaded {len(result.events)} events")
    """
    loader = LocalDataLoader(data_dir)
    return loader.load()


def load_events_only(
    data_dir: str | Path = "data/samples",
) -> tuple[list[EventRecord], list[CustomerIdHistory]]:
    """
    Load just events and ID history for pipeline use.

    Args:
        data_dir: Directory containing parquet files

    Returns:
        Tuple of (events, id_history)

    Example:
        events, id_history = load_events_only("data/samples")

        from src.pipeline import run_pipeline
        result = run_pipeline(events=events, id_history=id_history)
    """
    result = load_local_data(data_dir)
    return result.events, result.id_history
