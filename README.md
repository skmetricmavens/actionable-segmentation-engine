# Actionable Segmentation Engine

ML + LLM-driven customer segmentation with robustness validation for Bloomreach EBQ data.

## Overview

This POC transforms raw Bloomreach EBQ data into commercially exploitable customer insights using a hybrid ML + LLM architecture. It discovers actionable customer segments tied to revenue, retention, and satisfaction with specific business plays.

## Features

- **Synthetic Bloomreach EBQ data generation** - Test pipeline without real customer data
- **Customer ID merge resolution** - Correctly handles cross-device customer unification
- **Actionable trait extraction** - No PCA/embeddings, only business-interpretable traits
- **ML-based segmentation** - KMeans clustering with sensitivity analysis
- **Robustness validation** - Feature and time window sensitivity tests
- **LLM-powered actionability filtering** - Rejects non-actionable segments
- **Business-language insights** - Confidence levels and recommended plays

## Installation

```bash
# Clone the repository
git clone https://github.com/skmetricmavens/actionable-segmentation-engine.git
cd actionable-segmentation-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

## Quick Start

```python
from src.pipeline import run_pipeline

# Run end-to-end pipeline with synthetic data
report = run_pipeline(n_customers=1000, seed=42)

# Access segments
for segment in report.segments:
    print(f"Segment: {segment.name}")
    print(f"Size: {segment.size}")
    print(f"Confidence: {segment.confidence_level}")
    print(f"Recommended Action: {segment.recommended_action}")
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed architecture documentation.

## Testing

```bash
# Run all tests with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_synthetic_generator.py

# Type checking
mypy src/ --strict
```

## Project Structure

```
actionable-segmentation-engine/
├── config/          # Configuration settings
├── src/
│   ├── data/        # Data loading, schemas, synthetic generation
│   ├── features/    # Profile building, trait extraction
│   ├── segmentation/# Clustering, validation, hypothesis generation
│   ├── validation/  # Sensitivity analysis
│   ├── llm/         # LLM integration
│   └── reporting/   # Report generation
├── tests/           # Unit and integration tests
├── notebooks/       # Jupyter notebook examples
└── examples/        # Sample outputs
```

## License

MIT
