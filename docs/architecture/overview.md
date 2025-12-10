# System Overview

High-level architecture of the Actionable Segmentation Engine.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PIPELINE ORCHESTRATOR                              │
│                              (src/pipeline.py)                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
    ┌──────────────────────────────────┼──────────────────────────────────┐
    │                                  │                                   │
    ▼                                  ▼                                   ▼
┌────────────┐                  ┌────────────┐                    ┌────────────┐
│   DATA     │                  │  FEATURES  │                    │ SEGMENTATION│
│ LAYER      │                  │   LAYER    │                    │   LAYER     │
│            │                  │            │                    │             │
│ • schemas  │                  │ • profile  │                    │ • clusterer │
│ • joiner   │ ─────────────▶  │   builder  │ ────────────────▶  │ • validator │
│ • synthetic│                  │ • aggregat │                    │ • sensitiv. │
│   generator│                  │            │                    │             │
└────────────┘                  └────────────┘                    └────────────┘
                                                                        │
                              ┌─────────────────────────────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │     LLM LAYER      │
                    │                    │
                    │ • actionability    │
                    │   filter           │
                    │ • segment          │
                    │   explainer        │
                    └────────────────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │   REPORTING LAYER  │
                    │                    │
                    │ • segment_reporter │
                    │ • visuals          │
                    └────────────────────┘
```

## Design Principles

### 1. Modular Pipeline

Each stage is independent and testable:

- **Loose coupling** - Stages communicate via well-defined interfaces
- **Single responsibility** - Each module does one thing well
- **Composable** - Stages can be skipped or replaced

### 2. Type Safety

Strict typing throughout:

- **Pydantic models** - Validation at data boundaries
- **Type hints** - Full coverage with mypy strict
- **Runtime checks** - Pydantic validates at runtime

### 3. Testability

Designed for comprehensive testing:

- **Pure functions** - Core logic is side-effect free
- **Dependency injection** - External services are injectable
- **Mock support** - LLM layer has deterministic mock

### 4. Robustness First

Segments must prove stability:

- **Sensitivity testing** - Multiple stability tests
- **Validation criteria** - Business rules enforcement
- **Quality metrics** - Quantified robustness scores

## Layer Responsibilities

### Data Layer

**Purpose:** Data ingestion, validation, and ID resolution

| Module | Responsibility |
|--------|---------------|
| `schemas.py` | Pydantic models for all data types |
| `joiner.py` | Customer ID merge resolution |
| `synthetic_generator.py` | Test data generation |

### Features Layer

**Purpose:** Transform events into customer profiles

| Module | Responsibility |
|--------|---------------|
| `profile_builder.py` | Orchestrate profile creation |
| `aggregators.py` | Individual aggregation functions |

### Segmentation Layer

**Purpose:** Cluster customers and validate segments

| Module | Responsibility |
|--------|---------------|
| `clusterer.py` | KMeans clustering with auto-k |
| `sensitivity.py` | Robustness testing |
| `segment_validator.py` | Business validation |

### LLM Layer

**Purpose:** AI-powered segment evaluation

| Module | Responsibility |
|--------|---------------|
| `actionability_filter.py` | Evaluate actionability dimensions |
| `segment_explainer.py` | Generate business explanations |

### Reporting Layer

**Purpose:** Output generation and visualization

| Module | Responsibility |
|--------|---------------|
| `segment_reporter.py` | Report generation |
| `visuals.py` | Matplotlib/seaborn charts |

## Configuration

Pipeline behavior is controlled by `PipelineConfig`:

```python
@dataclass
class PipelineConfig:
    # Data generation
    n_customers: int = 1000
    data_seed: int = 42

    # Clustering
    n_clusters: int | None = None
    auto_select_k: bool = True

    # Analysis
    run_sensitivity: bool = True

    # Output
    generate_report: bool = True
    use_llm: bool = False
```

## Extensibility Points

### Custom Data Sources

Replace synthetic generator with real data loader:

```python
from src.pipeline import run_pipeline_on_dataset

dataset = load_your_data()  # Your data loading logic
result = run_pipeline_on_dataset(dataset)
```

### Custom Clustering

Implement `ClustererProtocol`:

```python
class MyClusterer:
    def cluster(self, profiles: list[CustomerProfile]) -> ClusteringResult:
        # Your clustering logic
        pass
```

### Real LLM Integration

Enable with `use_llm=True`:

```python
config = PipelineConfig(use_llm=True)
# Requires ANTHROPIC_API_KEY environment variable
```
