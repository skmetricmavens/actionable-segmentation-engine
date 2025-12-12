"""
Tests for src/data/bigquery/sample_extractor.py

Comprehensive test suite for sample data extraction functionality.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from src.data.bigquery.sample_extractor import (
    ExtractionConfig,
    ExtractionResult,
    SampleExtractor,
    LocalDataLoader,
    extract_sample_data,
    load_local_sample,
)
from src.data.bigquery.config import (
    BigQueryConfig,
    EventTableConfig,
    MergeTableConfig,
    FieldMapping,
    EventType,
)


# =============================================================================
# TESTS: ExtractionConfig
# =============================================================================


class TestExtractionConfig:
    """Tests for ExtractionConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        config = ExtractionConfig(
            project_id="project",
            dataset_id="dataset",
        )

        assert config.project_id == "project"
        assert config.dataset_id == "dataset"
        assert config.start_date is None
        assert config.end_date is None
        assert config.months_back == 3
        assert config.sample_rate is None
        assert config.max_rows_per_table == 100000
        assert config.max_customers == 10000
        assert config.output_dir == "data/samples"
        assert config.output_format == "parquet"
        assert config.table_names is None

    def test_custom_values(self) -> None:
        """Test custom values."""
        config = ExtractionConfig(
            project_id="my-project",
            dataset_id="my_dataset",
            start_date="2024-01-01",
            end_date="2024-06-30",
            months_back=6,
            sample_rate=0.5,
            max_rows_per_table=50000,
            max_customers=5000,
            output_dir="/tmp/data",
            output_format="json",
            table_names=["table1", "table2"],
        )

        assert config.start_date == "2024-01-01"
        assert config.end_date == "2024-06-30"
        assert config.sample_rate == 0.5
        assert config.max_customers == 5000
        assert config.output_format == "json"
        assert len(config.table_names) == 2

    def test_get_date_range_with_explicit_dates(self) -> None:
        """Test get_date_range with explicit start and end dates."""
        config = ExtractionConfig(
            project_id="project",
            dataset_id="dataset",
            start_date="2024-01-15",
            end_date="2024-06-20",
        )

        start, end = config.get_date_range()

        assert start == "2024-01-15"
        assert end == "2024-06-20"

    def test_get_date_range_default(self) -> None:
        """Test get_date_range with default values."""
        config = ExtractionConfig(
            project_id="project",
            dataset_id="dataset",
            months_back=3,
        )

        start, end = config.get_date_range()

        # End date should be today
        today = datetime.now()
        assert end == today.strftime("%Y-%m-%d")

        # Start date should be roughly 3 months ago
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        expected_start = today - timedelta(days=90)
        # Allow for some variance
        assert abs((start_dt - expected_start).days) < 5

    def test_get_date_range_with_end_only(self) -> None:
        """Test get_date_range with only end date specified."""
        config = ExtractionConfig(
            project_id="project",
            dataset_id="dataset",
            end_date="2024-06-30",
            months_back=2,
        )

        start, end = config.get_date_range()

        assert end == "2024-06-30"
        # Start should be 2 months before end date
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        expected_start = datetime(2024, 6, 30) - timedelta(days=60)
        assert abs((start_dt - expected_start).days) < 5


# =============================================================================
# TESTS: ExtractionResult
# =============================================================================


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        result = ExtractionResult(output_dir=Path("/tmp"))

        assert result.output_dir == Path("/tmp")
        assert result.tables_extracted == []
        assert result.rows_per_table == {}
        assert result.unique_customers == 0
        assert result.total_rows == 0
        assert result.extraction_time_seconds == 0.0
        assert result.config_saved is False
        assert result.errors == []

    def test_custom_values(self) -> None:
        """Test custom values."""
        result = ExtractionResult(
            output_dir=Path("/data"),
            tables_extracted=["purchases", "events"],
            rows_per_table={"purchases": 1000, "events": 5000},
            unique_customers=500,
            total_rows=6000,
            extraction_time_seconds=30.5,
            config_saved=True,
            errors=["Warning: rate limited"],
        )

        assert len(result.tables_extracted) == 2
        assert result.rows_per_table["purchases"] == 1000
        assert result.unique_customers == 500
        assert result.total_rows == 6000
        assert result.extraction_time_seconds == 30.5


# =============================================================================
# TESTS: SampleExtractor
# =============================================================================


class TestSampleExtractor:
    """Tests for SampleExtractor class."""

    def test_init(self) -> None:
        """Test initialization."""
        config = ExtractionConfig(
            project_id="project",
            dataset_id="dataset",
        )
        extractor = SampleExtractor(config)

        assert extractor.config == config
        assert extractor._client is None
        assert extractor._bigquery_module is None
        assert extractor._pandas_module is None

    def test_get_bigquery_import_error(self) -> None:
        """Test _get_bigquery raises ImportError when not installed."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        with patch.dict("sys.modules", {"google.cloud": None, "google.cloud.bigquery": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="google-cloud-bigquery is required"):
                    extractor._get_bigquery()

    def test_get_pandas_import_error(self) -> None:
        """Test _get_pandas raises ImportError when not installed."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        with patch.dict("sys.modules", {"pandas": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="pandas is required"):
                    extractor._get_pandas()

    def test_get_bigquery_caches_module(self) -> None:
        """Test _get_bigquery caches the module."""
        import sys

        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        mock_bigquery = MagicMock()
        mock_google_cloud = MagicMock()
        mock_google_cloud.bigquery = mock_bigquery

        original_google = sys.modules.get("google", None)
        original_google_cloud = sys.modules.get("google.cloud", None)
        original_bigquery = sys.modules.get("google.cloud.bigquery", None)

        try:
            sys.modules["google"] = MagicMock()
            sys.modules["google.cloud"] = mock_google_cloud
            sys.modules["google.cloud.bigquery"] = mock_bigquery

            result1 = extractor._get_bigquery()
            result2 = extractor._get_bigquery()

            assert result1 == result2
            assert extractor._bigquery_module == mock_bigquery
        finally:
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

    def test_client_property_creates_client(self) -> None:
        """Test client property creates BigQuery client."""
        config = ExtractionConfig(project_id="test-project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        mock_bigquery = MagicMock()
        mock_client = MagicMock()
        mock_bigquery.Client.return_value = mock_client

        with patch.object(extractor, "_get_bigquery", return_value=mock_bigquery):
            client = extractor.client

            mock_bigquery.Client.assert_called_once_with(project="test-project")
            assert client == mock_client

    def test_client_property_caches_client(self) -> None:
        """Test client property caches the client."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        mock_bigquery = MagicMock()
        mock_client = MagicMock()
        mock_bigquery.Client.return_value = mock_client

        with patch.object(extractor, "_get_bigquery", return_value=mock_bigquery):
            client1 = extractor.client
            client2 = extractor.client

            assert mock_bigquery.Client.call_count == 1
            assert client1 is client2

    def test_get_tables_from_config(self) -> None:
        """Test _get_tables returns tables from config."""
        config = ExtractionConfig(
            project_id="project",
            dataset_id="dataset",
            table_names=["table1", "table2"],
        )
        extractor = SampleExtractor(config)

        tables = extractor._get_tables()

        assert tables == ["table1", "table2"]

    def test_get_tables_auto_discover(self) -> None:
        """Test _get_tables auto-discovers tables."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        mock_client = MagicMock()
        mock_table_refs = [
            MagicMock(table_id="purchases"),
            MagicMock(table_id="events"),
        ]
        mock_client.list_tables.return_value = mock_table_refs

        with patch.object(SampleExtractor, "client", new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_client

            tables = extractor._get_tables()

            assert tables == ["purchases", "events"]

    def test_table_has_timestamp_true(self) -> None:
        """Test _table_has_timestamp returns True when timestamp field exists."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_field = MagicMock()
        mock_field.name = "timestamp"
        mock_table.schema = [mock_field]
        mock_client.get_table.return_value = mock_table

        with patch.object(SampleExtractor, "client", new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_client

            result = extractor._table_has_timestamp("test_table")

            assert result is True

    def test_table_has_timestamp_false(self) -> None:
        """Test _table_has_timestamp returns False when no timestamp field."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_field = MagicMock()
        mock_field.name = "other_field"
        mock_table.schema = [mock_field]
        mock_client.get_table.return_value = mock_table

        with patch.object(SampleExtractor, "client", new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_client

            result = extractor._table_has_timestamp("test_table")

            assert result is False

    def test_table_has_timestamp_error(self) -> None:
        """Test _table_has_timestamp returns False on error."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        mock_client = MagicMock()
        mock_client.get_table.side_effect = Exception("API Error")

        with patch.object(SampleExtractor, "client", new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_client

            result = extractor._table_has_timestamp("test_table")

            assert result is False

    def test_get_customer_id_field(self) -> None:
        """Test _get_customer_id_field finds customer ID field."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_field = MagicMock()
        mock_field.name = "internal_customer_id"
        mock_table.schema = [mock_field]
        mock_client.get_table.return_value = mock_table

        with patch.object(SampleExtractor, "client", new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_client

            result = extractor._get_customer_id_field("test_table")

            assert result == "internal_customer_id"

    def test_get_customer_id_field_not_found(self) -> None:
        """Test _get_customer_id_field returns None when not found."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_field = MagicMock()
        mock_field.name = "other_field"
        mock_table.schema = [mock_field]
        mock_client.get_table.return_value = mock_table

        with patch.object(SampleExtractor, "client", new_callable=PropertyMock) as mock_prop:
            mock_prop.return_value = mock_client

            result = extractor._get_customer_id_field("test_table")

            assert result is None

    def test_row_to_dict_basic(self) -> None:
        """Test _row_to_dict converts basic row."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        class MockRow(dict):
            pass

        row = MockRow({"col1": "value1", "col2": 123})

        result = extractor._row_to_dict(row)

        assert result["col1"] == "value1"
        assert result["col2"] == 123

    def test_row_to_dict_with_datetime(self) -> None:
        """Test _row_to_dict converts datetime to ISO string."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        dt = datetime(2024, 1, 15, 10, 30)

        class MockRow(dict):
            pass

        row = MockRow({"timestamp": dt})

        result = extractor._row_to_dict(row)

        assert result["timestamp"] == dt.isoformat()

    def test_row_to_dict_with_named_tuple(self) -> None:
        """Test _row_to_dict handles named tuples (nested records)."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        from collections import namedtuple

        Properties = namedtuple("Properties", ["email", "name"])
        props = Properties(email="test@example.com", name="Test")

        class MockRow(dict):
            pass

        row = MockRow({"properties": props})

        result = extractor._row_to_dict(row)

        assert result["properties"]["email"] == "test@example.com"


# =============================================================================
# TESTS: LocalDataLoader
# =============================================================================


class TestLocalDataLoader:
    """Tests for LocalDataLoader class."""

    def test_init(self) -> None:
        """Test initialization."""
        loader = LocalDataLoader("/tmp/data")

        assert loader.data_dir == Path("/tmp/data")
        assert loader._pandas_module is None

    def test_init_with_path(self) -> None:
        """Test initialization with Path object."""
        loader = LocalDataLoader(Path("/tmp/data"))

        assert loader.data_dir == Path("/tmp/data")

    def test_get_pandas_import_error(self) -> None:
        """Test _get_pandas raises ImportError when not installed."""
        loader = LocalDataLoader("/tmp")

        with patch.dict("sys.modules", {"pandas": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with pytest.raises(ImportError, match="pandas is required"):
                    loader._get_pandas()

    def test_load_metadata(self) -> None:
        """Test load_metadata loads JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {
                "project_id": "test-project",
                "dataset_id": "test_dataset",
                "total_rows": 1000,
            }

            metadata_path = Path(tmpdir) / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f)

            loader = LocalDataLoader(tmpdir)
            result = loader.load_metadata()

            assert result["project_id"] == "test-project"
            assert result["total_rows"] == 1000

    def test_load_metadata_not_found(self) -> None:
        """Test load_metadata raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = LocalDataLoader(tmpdir)

            with pytest.raises(FileNotFoundError):
                loader.load_metadata()

    def test_list_tables_parquet(self) -> None:
        """Test list_tables finds parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create parquet files (empty)
            (Path(tmpdir) / "purchases.parquet").touch()
            (Path(tmpdir) / "events.parquet").touch()
            (Path(tmpdir) / "metadata.json").touch()

            loader = LocalDataLoader(tmpdir)
            tables = loader.list_tables()

            assert "purchases" in tables
            assert "events" in tables

    def test_list_tables_json(self) -> None:
        """Test list_tables finds JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create JSON files (valid JSON)
            (Path(tmpdir) / "purchases.json").write_text("[]")
            (Path(tmpdir) / "metadata.json").write_text("{}")
            (Path(tmpdir) / "customer_ids.json").write_text("[]")

            loader = LocalDataLoader(tmpdir)
            tables = loader.list_tables()

            assert "purchases" in tables
            # metadata and customer_ids should be excluded
            assert "metadata" not in tables
            assert "customer_ids" not in tables

    def test_list_tables_mixed(self) -> None:
        """Test list_tables deduplicates tables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create both parquet and json for same table
            (Path(tmpdir) / "purchases.parquet").touch()
            (Path(tmpdir) / "purchases.json").write_text("[]")

            loader = LocalDataLoader(tmpdir)
            tables = loader.list_tables()

            # Should only appear once
            assert tables.count("purchases") == 1

    def test_load_table_parquet(self) -> None:
        """Test load_table loads parquet file."""
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a parquet file
            df = pd.DataFrame({"col1": [1, 2, 3]})
            df.to_parquet(Path(tmpdir) / "test_table.parquet", index=False)

            loader = LocalDataLoader(tmpdir)
            result = loader.load_table("test_table")

            assert len(result) == 3
            assert list(result["col1"]) == [1, 2, 3]

    def test_load_table_json(self) -> None:
        """Test load_table loads JSON file."""
        pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a JSON file
            data = [{"col1": 1}, {"col1": 2}]
            with open(Path(tmpdir) / "test_table.json", "w") as f:
                json.dump(data, f)

            loader = LocalDataLoader(tmpdir)
            result = loader.load_table("test_table")

            assert len(result) == 2

    def test_load_table_not_found(self) -> None:
        """Test load_table raises FileNotFoundError."""
        pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = LocalDataLoader(tmpdir)

            with pytest.raises(FileNotFoundError):
                loader.load_table("nonexistent")

    def test_load_all(self) -> None:
        """Test load_all loads all tables."""
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create parquet files
            pd.DataFrame({"col": [1]}).to_parquet(
                Path(tmpdir) / "table1.parquet", index=False
            )
            pd.DataFrame({"col": [2]}).to_parquet(
                Path(tmpdir) / "table2.parquet", index=False
            )

            loader = LocalDataLoader(tmpdir)
            result = loader.load_all()

            assert "table1" in result
            assert "table2" in result

    def test_infer_config_from_metadata(self) -> None:
        """Test _infer_config_from_metadata creates config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = {
                "project_id": "test-project",
                "dataset_id": "test_dataset",
            }
            with open(Path(tmpdir) / "metadata.json", "w") as f:
                json.dump(metadata, f)

            loader = LocalDataLoader(tmpdir)
            config = loader._infer_config_from_metadata()

            assert config.project_id == "test-project"
            assert config.dataset_id == "test_dataset"


# =============================================================================
# TESTS: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_extract_sample_data(self) -> None:
        """Test extract_sample_data creates extractor and runs."""
        with patch.object(SampleExtractor, "extract") as mock_extract:
            mock_extract.return_value = ExtractionResult(
                output_dir=Path("/tmp"),
            )

            result = extract_sample_data(
                project_id="project",
                dataset_id="dataset",
                months_back=2,
                max_customers=5000,
                output_dir="/tmp/output",
            )

            mock_extract.assert_called_once()
            assert isinstance(result, ExtractionResult)

    def test_load_local_sample(self) -> None:
        """Test load_local_sample creates loader and converts."""
        with patch.object(LocalDataLoader, "to_events") as mock_to_events:
            mock_to_events.return_value = ([], [])

            events, id_history = load_local_sample("/tmp/data")

            mock_to_events.assert_called_once()
            assert events == []
            assert id_history == []


# =============================================================================
# TESTS: Integration
# =============================================================================


class TestIntegration:
    """Integration tests for SampleExtractor."""

    def test_save_customer_ids(self) -> None:
        """Test _save_customer_ids saves JSON file."""
        config = ExtractionConfig(project_id="project", dataset_id="dataset")
        extractor = SampleExtractor(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            customer_ids = {"cust_001", "cust_002", "cust_003"}

            extractor._save_customer_ids(Path(tmpdir), customer_ids)

            # Verify file was created
            file_path = Path(tmpdir) / "customer_ids.json"
            assert file_path.exists()

            with open(file_path) as f:
                saved_ids = json.load(f)

            assert set(saved_ids) == customer_ids

    def test_save_metadata(self) -> None:
        """Test _save_metadata saves extraction info."""
        config = ExtractionConfig(
            project_id="project",
            dataset_id="dataset",
            sample_rate=0.5,
            max_customers=1000,
        )
        extractor = SampleExtractor(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            result = ExtractionResult(
                output_dir=Path(tmpdir),
                tables_extracted=["purchases"],
                rows_per_table={"purchases": 500},
                unique_customers=100,
                total_rows=500,
            )

            extractor._save_metadata(
                Path(tmpdir),
                result,
                "2024-01-01",
                "2024-06-30",
            )

            # Verify file was created
            file_path = Path(tmpdir) / "metadata.json"
            assert file_path.exists()

            with open(file_path) as f:
                metadata = json.load(f)

            assert metadata["project_id"] == "project"
            assert metadata["start_date"] == "2024-01-01"
            assert metadata["end_date"] == "2024-06-30"
            assert metadata["total_rows"] == 500
            assert metadata["unique_customers"] == 100

    def test_save_table_parquet(self) -> None:
        """Test _save_table saves parquet file."""
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        import pandas as pd

        config = ExtractionConfig(
            project_id="project",
            dataset_id="dataset",
            output_format="parquet",
        )
        extractor = SampleExtractor(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"col1": "a", "col2": 1},
                {"col1": "b", "col2": 2},
            ]

            extractor._save_table(Path(tmpdir), "test_table", rows)

            # Verify file was created
            file_path = Path(tmpdir) / "test_table.parquet"
            assert file_path.exists()

            # Verify contents
            df = pd.read_parquet(file_path)
            assert len(df) == 2

    def test_save_table_json(self) -> None:
        """Test _save_table saves JSON file."""
        config = ExtractionConfig(
            project_id="project",
            dataset_id="dataset",
            output_format="json",
        )
        extractor = SampleExtractor(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            rows = [
                {"col1": "a", "col2": 1},
                {"col1": "b", "col2": 2},
            ]

            extractor._save_table(Path(tmpdir), "test_table", rows)

            # Verify file was created
            file_path = Path(tmpdir) / "test_table.json"
            assert file_path.exists()

            # Verify contents
            with open(file_path) as f:
                data = json.load(f)
            assert len(data) == 2

    def test_extract_full_flow(self) -> None:
        """Test full extract flow with mocked BigQuery."""
        pytest.importorskip("pandas")

        config = ExtractionConfig(
            project_id="project",
            dataset_id="dataset",
            table_names=["purchases"],
            max_customers=10,
            output_format="json",
        )
        extractor = SampleExtractor(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_dir = tmpdir

            mock_client = MagicMock()

            # Mock table schema
            mock_table = MagicMock()
            mock_customer_field = MagicMock()
            mock_customer_field.name = "customer_id"
            mock_ts_field = MagicMock()
            mock_ts_field.name = "timestamp"
            mock_table.schema = [mock_customer_field, mock_ts_field]
            mock_client.get_table.return_value = mock_table

            # Mock query results
            class MockRow(dict):
                pass

            customer_rows = [MockRow({"customer_id": f"cust_{i}"}) for i in range(5)]
            event_rows = [
                MockRow({"customer_id": "cust_0", "timestamp": "2024-01-15", "amount": 99.99}),
                MockRow({"customer_id": "cust_1", "timestamp": "2024-01-16", "amount": 50.00}),
            ]

            call_count = [0]

            def mock_query(query):
                mock_job = MagicMock()
                call_count[0] += 1
                if "DISTINCT" in query:
                    mock_job.result.return_value = customer_rows
                else:
                    mock_job.result.return_value = event_rows
                return mock_job

            mock_client.query = mock_query

            with patch.object(SampleExtractor, "client", new_callable=PropertyMock) as mock_prop:
                mock_prop.return_value = mock_client

                result = extractor.extract()

                assert result.unique_customers == 5
                assert "purchases" in result.tables_extracted
                assert result.config_saved is True
                assert result.extraction_time_seconds > 0

    def test_dataframe_to_events(self) -> None:
        """Test _dataframe_to_events converts DataFrame to events."""
        pytest.importorskip("pandas")

        import pandas as pd

        loader = LocalDataLoader("/tmp")

        df = pd.DataFrame([
            {"internal_customer_id": "cust_001", "timestamp": "2024-01-15T10:00:00"},
            {"internal_customer_id": "cust_002", "timestamp": "2024-01-16T11:00:00"},
        ])

        config = EventTableConfig(
            table_name="purchases",
            event_type=EventType.PURCHASE,
            customer_id_field="internal_customer_id",
            timestamp_field="timestamp",
        )

        events = loader._dataframe_to_events(df, config)

        assert len(events) == 2
        assert events[0].internal_customer_id == "cust_001"

    def test_dataframe_to_id_history(self) -> None:
        """Test _dataframe_to_id_history converts DataFrame to history."""
        pytest.importorskip("pandas")

        import pandas as pd

        loader = LocalDataLoader("/tmp")

        df = pd.DataFrame([
            {"current_id": "cust_001", "past_id": "old_001"},
            {"current_id": "cust_002", "past_id": "old_002"},
            {"current_id": "cust_003", "past_id": None},  # Should be skipped
        ])

        config = MergeTableConfig(
            table_name="id_history",
            current_id_field="current_id",
            past_id_field="past_id",
        )

        history = loader._dataframe_to_id_history(df, config)

        assert len(history) == 2
        assert history[0].internal_customer_id == "cust_001"
        assert history[0].past_id == "old_001"

    def test_to_events_with_config(self) -> None:
        """Test to_events with provided config."""
        pytest.importorskip("pandas")
        pytest.importorskip("pyarrow")

        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create metadata
            metadata = {
                "project_id": "project",
                "dataset_id": "dataset",
            }
            with open(Path(tmpdir) / "metadata.json", "w") as f:
                json.dump(metadata, f)

            # Create test data
            df = pd.DataFrame([
                {
                    "internal_customer_id": "cust_001",
                    "timestamp": "2024-01-15T10:00:00",
                },
            ])
            df.to_parquet(Path(tmpdir) / "purchases.parquet", index=False)

            config = BigQueryConfig(
                project_id="project",
                dataset_id="dataset",
                event_tables=[
                    EventTableConfig(
                        table_name="purchases",
                        event_type=EventType.PURCHASE,
                    ),
                ],
            )

            loader = LocalDataLoader(tmpdir)
            events, id_history = loader.to_events(config)

            assert len(events) == 1
            assert events[0].internal_customer_id == "cust_001"
