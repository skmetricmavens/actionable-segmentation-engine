# Examples

This directory contains example scripts demonstrating how to use the Actionable Segmentation Engine.

## Available Examples

| Example | Description |
|---------|-------------|
| `quick_start.py` | Basic pipeline usage with default settings |
| `load_local_data.py` | Load data from local parquet files with flexible schema config |
| `custom_config.py` | Advanced configuration with custom validation criteria |
| `export_results.py` | Export segmentation results to JSON files |
| `visualizations.py` | Create and save visualization charts |

## Running Examples

Make sure you're in the project root directory and have installed dependencies:

```bash
# Install dependencies
pip install -r requirements.txt

# Run examples
python examples/quick_start.py
python examples/load_local_data.py
python examples/custom_config.py
python examples/export_results.py
python examples/visualizations.py
```

## Using Local Data

To run the pipeline on your own data:

```python
from src.data import load_events_only
from src.pipeline import run_pipeline, PipelineConfig

# Load from parquet files
events, id_history = load_events_only("data/samples")

# Configure and run
config = PipelineConfig(n_clusters=5, run_sensitivity=True)
result = run_pipeline(config, events=events, id_history=id_history)
```

### Different Data Sources

```python
from src.data import (
    load_events_only,
    create_ga4_config,      # Google Analytics 4
    create_segment_config,  # Segment.com
    ClientSchemaConfig,     # Custom configuration
)

# For GA4 data
events, _ = load_events_only("data/ga4", schema_config=create_ga4_config())

# For custom data structure
custom = ClientSchemaConfig(
    client_name="my_company",
    customer_id_field="user_id",
    timestamp_field="event_time",
    revenue_fields=["total", "amount"],
)
events, _ = load_events_only("data/custom", schema_config=custom)
```

## Output

- `export_results.py` creates files in `output/`
- `visualizations.py` creates charts in `output/charts/`

## Interactive Examples

For interactive exploration, see the Jupyter notebooks in `notebooks/`:

| Notebook | Description |
|----------|-------------|
| `01_quick_start_demo.ipynb` | Interactive quick start |
| `02_local_data_pipeline.ipynb` | Working with local data |
| `03_custom_schema_config.ipynb` | Configure for your data source |

## Command-Line Interface

Run the pipeline from the command line:

```bash
# Run on local sample data
python scripts/run_local_pipeline.py --clusters 5 --verbose

# Save results to JSON
python scripts/run_local_pipeline.py --clusters 5 -o outputs/results.json

# Skip sensitivity analysis (faster)
python scripts/run_local_pipeline.py --clusters 5 --no-sensitivity
```
