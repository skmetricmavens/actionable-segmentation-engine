# Installation

This guide covers installation of the Actionable Segmentation Engine.

## Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.11+ |
| pip | Latest |
| Git | 2.0+ |

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/skmetricmavens/actionable-segmentation-engine.git
cd actionable-segmentation-engine
```

### 2. Create Virtual Environment

=== "venv"

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

=== "conda"

    ```bash
    conda create -n segmentation python=3.11
    conda activate segmentation
    ```

=== "uv"

    ```bash
    uv venv
    source .venv/bin/activate
    ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```python
from src.pipeline import quick_segmentation

result = quick_segmentation(n_customers=100, n_clusters=3)
print(f"Success! Generated {len(result.segments)} segments")
```

## Optional: LLM Integration

To use real LLM capabilities (instead of mock responses):

### Set Up Claude API

1. Get an API key from [Anthropic Console](https://console.anthropic.com/)

2. Set the environment variable:

    ```bash
    export ANTHROPIC_API_KEY="your-api-key-here"
    ```

3. Enable LLM in pipeline config:

    ```python
    config = PipelineConfig(use_llm=True)
    ```

## Development Installation

For contributing to the project:

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest --cov=src
```

## Troubleshooting

??? question "ModuleNotFoundError: No module named 'src'"

    Ensure you're running from the project root directory, or add the project to your Python path:

    ```bash
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"
    ```

??? question "Import errors with numpy/sklearn"

    Try reinstalling scientific packages:

    ```bash
    pip install --upgrade numpy scikit-learn
    ```

## Next Steps

- [Quick Start Guide](quickstart.md) - Run your first pipeline
- [Configuration](configuration.md) - Customize pipeline behavior
