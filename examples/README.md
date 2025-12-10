# Examples

This directory contains example scripts demonstrating how to use the Actionable Segmentation Engine.

## Available Examples

| Example | Description |
|---------|-------------|
| `quick_start.py` | Basic pipeline usage with default settings |
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
python examples/custom_config.py
python examples/export_results.py
python examples/visualizations.py
```

## Output

- `export_results.py` creates files in `output/`
- `visualizations.py` creates charts in `output/charts/`

## Interactive Examples

For interactive exploration, see the Jupyter notebooks in `notebooks/`:

- `01_quick_start_demo.ipynb` - Interactive quick start
- `02_custom_configuration.ipynb` - Configuration options
- `03_visualization_gallery.ipynb` - All chart types
