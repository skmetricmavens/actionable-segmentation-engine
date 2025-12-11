"""
Extract sample data from BigQuery and store locally for testing.

Pulls a configurable time range of data and saves as Parquet files
for fast local iteration without needing BigQuery access.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud import bigquery
    import pandas as pd

from src.data.bigquery.config import BigQueryConfig, EventTableConfig

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ExtractionConfig:
    """Configuration for sample data extraction."""

    # BigQuery source
    project_id: str
    dataset_id: str

    # Time range (default: last 3 months)
    start_date: str | None = None  # YYYY-MM-DD
    end_date: str | None = None  # YYYY-MM-DD
    months_back: int = 3  # Used if start_date not specified

    # Sampling
    sample_rate: float | None = None  # 0.0-1.0, None = no sampling
    max_rows_per_table: int | None = 100000  # Limit per table
    max_customers: int | None = 10000  # Limit unique customers

    # Output
    output_dir: str = "data/samples"
    output_format: str = "parquet"  # "parquet" or "json"

    # Tables to extract (None = auto-discover)
    table_names: list[str] | None = None

    def get_date_range(self) -> tuple[str, str]:
        """Get start and end dates."""
        if self.end_date:
            end = datetime.strptime(self.end_date, "%Y-%m-%d")
        else:
            end = datetime.now()

        if self.start_date:
            start = datetime.strptime(self.start_date, "%Y-%m-%d")
        else:
            start = end - timedelta(days=self.months_back * 30)

        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


@dataclass
class ExtractionResult:
    """Result of sample data extraction."""

    output_dir: Path
    tables_extracted: list[str] = field(default_factory=list)
    rows_per_table: dict[str, int] = field(default_factory=dict)
    unique_customers: int = 0
    total_rows: int = 0
    extraction_time_seconds: float = 0.0
    config_saved: bool = False
    errors: list[str] = field(default_factory=list)


# =============================================================================
# SAMPLE EXTRACTOR
# =============================================================================


class SampleExtractor:
    """
    Extract sample data from BigQuery for local testing.

    Usage:
        extractor = SampleExtractor(ExtractionConfig(
            project_id="my-project",
            dataset_id="my_dataset",
            months_back=3,
            max_customers=10000,
        ))
        result = extractor.extract()

        # Later, load locally:
        from src.data.bigquery import LocalDataLoader
        events, id_history = LocalDataLoader("data/samples").load()
    """

    def __init__(self, config: ExtractionConfig):
        """Initialize extractor with configuration."""
        self.config = config
        self._client: Any = None
        self._bigquery_module: Any = None
        self._pandas_module: Any = None

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

    def _get_pandas(self) -> Any:
        """Lazy import of pandas."""
        if self._pandas_module is None:
            try:
                import pandas as pd
                self._pandas_module = pd
            except ImportError:
                raise ImportError(
                    "pandas is required for sample extraction. "
                    "Install with: pip install pandas pyarrow"
                )
        return self._pandas_module

    @property
    def client(self) -> Any:
        """Get or create BigQuery client."""
        if self._client is None:
            bigquery = self._get_bigquery()
            self._client = bigquery.Client(project=self.config.project_id)
        return self._client

    def extract(self) -> ExtractionResult:
        """
        Extract sample data from BigQuery.

        Returns:
            ExtractionResult with extraction details
        """
        import time
        start_time = time.perf_counter()

        pd = self._get_pandas()
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = ExtractionResult(output_dir=output_dir)
        start_date, end_date = self.config.get_date_range()

        logger.info(f"Extracting data from {start_date} to {end_date}")

        # Get tables to extract
        tables = self._get_tables()
        logger.info(f"Found {len(tables)} tables to extract")

        # First pass: identify customers to sample
        customer_ids = self._sample_customers(tables, start_date, end_date)
        result.unique_customers = len(customer_ids)
        logger.info(f"Sampled {len(customer_ids)} unique customers")

        # Save customer IDs for reference
        self._save_customer_ids(output_dir, customer_ids)

        # Extract each table
        for table_name in tables:
            try:
                rows = self._extract_table(
                    table_name,
                    customer_ids,
                    start_date,
                    end_date,
                )
                if rows:
                    self._save_table(output_dir, table_name, rows)
                    result.tables_extracted.append(table_name)
                    result.rows_per_table[table_name] = len(rows)
                    result.total_rows += len(rows)
                    logger.info(f"Extracted {len(rows)} rows from {table_name}")
            except Exception as e:
                error_msg = f"Error extracting {table_name}: {str(e)}"
                logger.error(error_msg)
                result.errors.append(error_msg)

        # Save extraction config and metadata
        self._save_metadata(output_dir, result, start_date, end_date)
        result.config_saved = True

        result.extraction_time_seconds = time.perf_counter() - start_time
        logger.info(
            f"Extraction complete: {result.total_rows} rows in "
            f"{result.extraction_time_seconds:.1f}s"
        )

        return result

    def _get_tables(self) -> list[str]:
        """Get list of tables to extract."""
        if self.config.table_names:
            return self.config.table_names

        # Auto-discover tables
        dataset_ref = f"{self.config.project_id}.{self.config.dataset_id}"
        tables = list(self.client.list_tables(dataset_ref))
        return [t.table_id for t in tables]

    def _sample_customers(
        self,
        tables: list[str],
        start_date: str,
        end_date: str,
    ) -> set[str]:
        """Sample customer IDs from tables."""
        customer_ids: set[str] = set()

        # Look for customer ID field in tables
        customer_id_fields = [
            "internal_customer_id",
            "customer_id",
            "user_id",
        ]

        for table_name in tables:
            if self.config.max_customers and len(customer_ids) >= self.config.max_customers:
                break

            full_table = f"{self.config.project_id}.{self.config.dataset_id}.{table_name}"

            # Try to find customer ID field
            for id_field in customer_id_fields:
                query = f"""
                SELECT DISTINCT {id_field}
                FROM `{full_table}`
                WHERE {id_field} IS NOT NULL
                """

                # Add date filter if table has timestamp
                if self._table_has_timestamp(table_name):
                    query += f"""
                    AND timestamp >= '{start_date}'
                    AND timestamp <= '{end_date}'
                    """

                # Add sampling
                if self.config.sample_rate:
                    query += f" AND RAND() < {self.config.sample_rate}"

                # Limit
                remaining = (
                    self.config.max_customers - len(customer_ids)
                    if self.config.max_customers
                    else 10000
                )
                query += f" LIMIT {remaining}"

                try:
                    rows = list(self.client.query(query).result())
                    for row in rows:
                        customer_ids.add(str(dict(row)[id_field]))
                    break  # Found the right field
                except Exception:
                    continue  # Try next field name

        return customer_ids

    def _table_has_timestamp(self, table_name: str) -> bool:
        """Check if table has a timestamp field."""
        full_table = f"{self.config.project_id}.{self.config.dataset_id}.{table_name}"
        try:
            table = self.client.get_table(full_table)
            for field in table.schema:
                if field.name.lower() in ("timestamp", "event_timestamp", "created_at"):
                    return True
        except Exception:
            pass
        return False

    def _extract_table(
        self,
        table_name: str,
        customer_ids: set[str],
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Extract rows from a single table for sampled customers."""
        full_table = f"{self.config.project_id}.{self.config.dataset_id}.{table_name}"

        # Find customer ID field
        customer_id_field = self._get_customer_id_field(table_name)
        if not customer_id_field:
            logger.warning(f"No customer ID field found in {table_name}")
            return []

        # Build query
        # For large customer sets, use a temp table or batch
        if len(customer_ids) > 1000:
            return self._extract_table_batched(
                full_table,
                customer_id_field,
                customer_ids,
                start_date,
                end_date,
            )

        customer_list = ", ".join(f"'{cid}'" for cid in customer_ids)

        query = f"""
        SELECT *
        FROM `{full_table}`
        WHERE {customer_id_field} IN ({customer_list})
        """

        if self._table_has_timestamp(table_name):
            query += f"""
            AND timestamp >= '{start_date}'
            AND timestamp <= '{end_date}'
            """

        if self.config.max_rows_per_table:
            query += f" LIMIT {self.config.max_rows_per_table}"

        rows = list(self.client.query(query).result())
        return [self._row_to_dict(row) for row in rows]

    def _extract_table_batched(
        self,
        full_table: str,
        customer_id_field: str,
        customer_ids: set[str],
        start_date: str,
        end_date: str,
    ) -> list[dict[str, Any]]:
        """Extract table in batches for large customer sets."""
        all_rows: list[dict[str, Any]] = []
        customer_list = list(customer_ids)
        batch_size = 500

        for i in range(0, len(customer_list), batch_size):
            batch = customer_list[i:i + batch_size]
            batch_str = ", ".join(f"'{cid}'" for cid in batch)

            query = f"""
            SELECT *
            FROM `{full_table}`
            WHERE {customer_id_field} IN ({batch_str})
            """

            if self._table_has_timestamp(full_table.split(".")[-1]):
                query += f"""
                AND timestamp >= '{start_date}'
                AND timestamp <= '{end_date}'
                """

            rows = list(self.client.query(query).result())
            all_rows.extend(self._row_to_dict(row) for row in rows)

            if self.config.max_rows_per_table and len(all_rows) >= self.config.max_rows_per_table:
                return all_rows[:self.config.max_rows_per_table]

        return all_rows

    def _get_customer_id_field(self, table_name: str) -> str | None:
        """Get customer ID field name for a table."""
        full_table = f"{self.config.project_id}.{self.config.dataset_id}.{table_name}"
        try:
            table = self.client.get_table(full_table)
            for field in table.schema:
                if field.name.lower() in (
                    "internal_customer_id",
                    "customer_id",
                    "user_id",
                ):
                    return field.name
        except Exception:
            pass
        return None

    def _row_to_dict(self, row: Any) -> dict[str, Any]:
        """Convert BigQuery row to dictionary, handling nested structures."""
        result = dict(row)

        # Convert nested records to dicts
        for key, value in result.items():
            if hasattr(value, "_asdict"):
                result[key] = dict(value._asdict())
            elif hasattr(value, "items"):
                # Already a dict-like object
                pass
            elif isinstance(value, datetime):
                result[key] = value.isoformat()

        return result

    def _save_table(
        self,
        output_dir: Path,
        table_name: str,
        rows: list[dict[str, Any]],
    ) -> None:
        """Save table data to file."""
        pd = self._get_pandas()

        if self.config.output_format == "parquet":
            file_path = output_dir / f"{table_name}.parquet"
            df = pd.DataFrame(rows)
            df.to_parquet(file_path, index=False)
        else:
            file_path = output_dir / f"{table_name}.json"
            with open(file_path, "w") as f:
                json.dump(rows, f, default=str, indent=2)

    def _save_customer_ids(self, output_dir: Path, customer_ids: set[str]) -> None:
        """Save sampled customer IDs."""
        file_path = output_dir / "customer_ids.json"
        with open(file_path, "w") as f:
            json.dump(list(customer_ids), f)

    def _save_metadata(
        self,
        output_dir: Path,
        result: ExtractionResult,
        start_date: str,
        end_date: str,
    ) -> None:
        """Save extraction metadata."""
        metadata = {
            "project_id": self.config.project_id,
            "dataset_id": self.config.dataset_id,
            "start_date": start_date,
            "end_date": end_date,
            "extraction_timestamp": datetime.now().isoformat(),
            "tables": result.tables_extracted,
            "rows_per_table": result.rows_per_table,
            "unique_customers": result.unique_customers,
            "total_rows": result.total_rows,
            "sample_rate": self.config.sample_rate,
            "max_customers": self.config.max_customers,
        }

        file_path = output_dir / "metadata.json"
        with open(file_path, "w") as f:
            json.dump(metadata, f, indent=2)


# =============================================================================
# LOCAL DATA LOADER
# =============================================================================


class LocalDataLoader:
    """
    Load sample data from local files.

    Usage:
        loader = LocalDataLoader("data/samples")
        events, id_history = loader.load()
    """

    def __init__(self, data_dir: str | Path):
        """Initialize loader with data directory."""
        self.data_dir = Path(data_dir)
        self._pandas_module: Any = None

    def _get_pandas(self) -> Any:
        """Lazy import of pandas."""
        if self._pandas_module is None:
            try:
                import pandas as pd
                self._pandas_module = pd
            except ImportError:
                raise ImportError("pandas is required: pip install pandas pyarrow")
        return self._pandas_module

    def load_metadata(self) -> dict[str, Any]:
        """Load extraction metadata."""
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata found at {metadata_path}")

        with open(metadata_path) as f:
            return json.load(f)

    def load_table(self, table_name: str) -> "pd.DataFrame":
        """Load a single table as DataFrame."""
        pd = self._get_pandas()

        parquet_path = self.data_dir / f"{table_name}.parquet"
        json_path = self.data_dir / f"{table_name}.json"

        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif json_path.exists():
            return pd.read_json(json_path)
        else:
            raise FileNotFoundError(f"No data file found for table {table_name}")

    def list_tables(self) -> list[str]:
        """List available tables."""
        tables = []
        for path in self.data_dir.glob("*.parquet"):
            tables.append(path.stem)
        for path in self.data_dir.glob("*.json"):
            if path.stem not in ["metadata", "customer_ids"]:
                tables.append(path.stem)
        return list(set(tables))

    def load_all(self) -> dict[str, "pd.DataFrame"]:
        """Load all tables as DataFrames."""
        return {table: self.load_table(table) for table in self.list_tables()}

    def to_events(
        self,
        config: BigQueryConfig | None = None,
    ) -> tuple[list[Any], list[Any]]:
        """
        Convert local data to EventRecord and CustomerIdHistory objects.

        Args:
            config: Optional BigQueryConfig for field mappings.
                    If not provided, uses auto-detected mappings.

        Returns:
            Tuple of (events, id_history)
        """
        from src.data.bigquery.adapter import BigQueryAdapter, LoadResult

        # Load all tables
        tables = self.load_all()

        # If no config provided, try to infer from metadata
        if config is None:
            config = self._infer_config_from_metadata()

        # Use adapter's conversion logic
        # This is a simplified version - the full adapter handles more cases
        events: list[Any] = []
        id_history: list[Any] = []

        # Process each event table
        for event_config in config.event_tables:
            table_name = event_config.table_name.split(".")[-1]
            if table_name in tables:
                df = tables[table_name]
                table_events = self._dataframe_to_events(df, event_config)
                events.extend(table_events)

        # Process merge table
        if config.merge_table:
            table_name = config.merge_table.table_name.split(".")[-1]
            if table_name in tables:
                df = tables[table_name]
                id_history = self._dataframe_to_id_history(df, config.merge_table)

        return events, id_history

    def _infer_config_from_metadata(self) -> BigQueryConfig:
        """Infer BigQueryConfig from metadata and available tables."""
        from src.data.bigquery.schema_discovery import SchemaDiscovery

        metadata = self.load_metadata()

        # Create a minimal config
        # In practice, you'd want to run schema discovery on the local files
        return BigQueryConfig(
            project_id=metadata.get("project_id", "local"),
            dataset_id=metadata.get("dataset_id", "sample"),
            event_tables=[],
        )

    def _dataframe_to_events(
        self,
        df: "pd.DataFrame",
        config: EventTableConfig,
    ) -> list[Any]:
        """Convert DataFrame to EventRecord objects."""
        from src.data.bigquery.adapter import (
            BQ_TO_INTERNAL_EVENT_TYPE,
            get_nested_value,
            transform_value,
        )
        from src.data.schemas import EventRecord, EventProperties
        import uuid

        events = []
        internal_event_type = BQ_TO_INTERNAL_EVENT_TYPE.get(config.event_type)

        if internal_event_type is None:
            return events

        for _, row in df.iterrows():
            row_dict = row.to_dict()

            customer_id = row_dict.get(config.customer_id_field)
            timestamp = row_dict.get(config.timestamp_field)

            if not customer_id or not timestamp:
                continue

            # Parse timestamp
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

            # Build properties
            properties_dict = {}
            for target_field, mapping in config.property_mappings.items():
                value = get_nested_value(row_dict, mapping.source_field)
                if value is None and mapping.default is not None:
                    value = mapping.default
                value = transform_value(value, mapping.transform)
                if value is not None:
                    properties_dict[target_field] = value

            try:
                event = EventRecord(
                    event_id=str(uuid.uuid4()),
                    internal_customer_id=str(customer_id),
                    event_type=internal_event_type,
                    timestamp=timestamp,
                    properties=EventProperties(**{
                        k: v for k, v in properties_dict.items()
                        if k in EventProperties.model_fields
                    }),
                )
                events.append(event)
            except Exception as e:
                logger.warning(f"Error creating event: {e}")

        return events

    def _dataframe_to_id_history(
        self,
        df: "pd.DataFrame",
        config: Any,  # MergeTableConfig
    ) -> list[Any]:
        """Convert DataFrame to CustomerIdHistory objects."""
        from src.data.schemas import CustomerIdHistory

        history = []
        for _, row in df.iterrows():
            current_id = row.get(config.current_id_field)
            past_id = row.get(config.past_id_field)

            if current_id and past_id:
                try:
                    history.append(CustomerIdHistory(
                        internal_customer_id=str(current_id),
                        past_id=str(past_id),
                        merge_timestamp=datetime.now(),
                    ))
                except Exception as e:
                    logger.warning(f"Error creating ID history: {e}")

        return history


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def extract_sample_data(
    project_id: str,
    dataset_id: str,
    *,
    months_back: int = 3,
    max_customers: int = 10000,
    output_dir: str = "data/samples",
) -> ExtractionResult:
    """
    Extract sample data from BigQuery for local testing.

    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        months_back: Number of months of data to extract
        max_customers: Maximum number of customers to sample
        output_dir: Directory to save extracted data

    Returns:
        ExtractionResult with extraction details

    Example:
        result = extract_sample_data(
            project_id="my-project",
            dataset_id="cdp_data",
            months_back=3,
            max_customers=10000,
        )
        print(f"Extracted {result.total_rows} rows")
    """
    config = ExtractionConfig(
        project_id=project_id,
        dataset_id=dataset_id,
        months_back=months_back,
        max_customers=max_customers,
        output_dir=output_dir,
    )

    extractor = SampleExtractor(config)
    return extractor.extract()


def load_local_sample(
    data_dir: str = "data/samples",
) -> tuple[list[Any], list[Any]]:
    """
    Load sample data from local files.

    Args:
        data_dir: Directory containing extracted sample data

    Returns:
        Tuple of (events, id_history)

    Example:
        events, id_history = load_local_sample("data/samples")
        print(f"Loaded {len(events)} events")
    """
    loader = LocalDataLoader(data_dir)
    return loader.to_events()
