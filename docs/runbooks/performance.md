# Performance Tuning

Optimize the segmentation pipeline for your workload.

## Current Benchmarks

| Stage | Time (500 customers) | Time (5000 customers) |
|-------|---------------------|----------------------|
| Data Generation | ~300ms | ~2s |
| Profile Building | ~100ms | ~500ms |
| Clustering | ~30ms | ~200ms |
| Sensitivity Analysis | ~200ms | ~2s |
| LLM Evaluation | ~10ms (mock) | ~10ms (mock) |
| **Total** | **~800ms** | **~5s** |

## Quick Wins

### 1. Skip Sensitivity Analysis

For exploratory analysis or quick iterations:

```python
config = PipelineConfig(
    run_sensitivity=False,  # Saves ~200ms per run
)
```

!!! warning "Trade-off"
    Without sensitivity analysis, you won't know if segments are stable.

### 2. Skip Sampling Stability

Sampling stability is the slowest sensitivity test:

```python
config = PipelineConfig(
    run_sensitivity=True,
    include_sampling_stability=False,  # Saves ~100ms
)
```

### 3. Use Fixed k

Auto k-selection tests multiple k values:

```python
config = PipelineConfig(
    n_clusters=5,
    auto_select_k=False,  # Saves ~50ms
)
```

### 4. Skip Report Generation

```python
config = PipelineConfig(
    generate_report=False,  # Saves ~20ms
)
```

## Scaling Guidelines

### 1,000 - 10,000 Customers

Standard configuration works well:

```python
config = PipelineConfig(
    n_customers=5000,
    run_sensitivity=True,
)
# Expected time: 3-5 seconds
```

### 10,000 - 50,000 Customers

Consider sampling for sensitivity analysis:

```python
config = PipelineConfig(
    n_customers=25000,
    run_sensitivity=True,
    include_sampling_stability=False,  # Most expensive test
)
# Expected time: 10-20 seconds
```

### 50,000+ Customers

Use incremental or batch processing:

```python
# Option 1: Sample for clustering
import random
sampled_profiles = random.sample(profiles, 10000)
result = run_pipeline_on_profiles(sampled_profiles)

# Option 2: Skip sensitivity entirely
config = PipelineConfig(
    run_sensitivity=False,
)
```

## Memory Optimization

### Profile Memory Usage

Each `CustomerProfile` uses approximately:
- ~500 bytes base
- +200 bytes per event aggregation

For 100,000 customers: ~50-70 MB

### Reduce Memory

1. **Process in batches:**
   ```python
   batch_size = 10000
   for i in range(0, len(all_events), batch_size):
       batch = all_events[i:i+batch_size]
       # Process batch
   ```

2. **Clear intermediate results:**
   ```python
   del profiles  # Free memory after clustering
   gc.collect()
   ```

## Profiling

### Time Profiling

```python
import time

start = time.perf_counter()
result = run_pipeline(config)
print(f"Total time: {time.perf_counter() - start:.2f}s")

# Per-stage timing available in result
for stage in result.stage_results:
    print(f"  {stage.stage_name}: {stage.duration_ms:.1f}ms")
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()
result = run_pipeline(config)
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

## Caching Strategies

### Cache Profiles

If running multiple clustering experiments on same data:

```python
# Generate profiles once
from src.data.synthetic_generator import generate_small_dataset
from src.features.profile_builder import ProfileBuilder

dataset = generate_small_dataset(seed=42)
builder = ProfileBuilder()
profiles = builder.build_profiles(dataset.events)

# Run multiple clustering experiments
for k in [3, 5, 7, 9]:
    result = run_pipeline_on_profiles(profiles, n_clusters=k)
```

### Cache Sensitivity Results

Sensitivity results are deterministic for same input:

```python
# Compute once, reuse for different validation criteria
sensitivity_result = analyzer.analyze_segments(profiles, segments)

for robustness_threshold in [0.3, 0.5, 0.7]:
    criteria = ValidationCriteria(min_overall_robustness=robustness_threshold)
    # Apply different criteria to same sensitivity results
```

## Real LLM Optimization

When using real Claude API (`use_llm=True`):

### Batch Requests

```python
# Current: Sequential calls (slow)
for segment in segments:
    evaluation = filter.evaluate(segment)

# Future: Batch API (when available)
evaluations = filter.evaluate_batch(segments)
```

### Cache LLM Responses

```python
import hashlib
import json

def cached_evaluate(segment, cache):
    key = hashlib.md5(json.dumps(segment.model_dump()).encode()).hexdigest()
    if key in cache:
        return cache[key]
    result = filter.evaluate(segment)
    cache[key] = result
    return result
```

## Configuration Presets

### Development (Fast)

```python
DEV_CONFIG = PipelineConfig(
    n_customers=100,
    n_clusters=3,
    auto_select_k=False,
    run_sensitivity=False,
    generate_report=False,
)
```

### Testing (Balanced)

```python
TEST_CONFIG = PipelineConfig(
    n_customers=500,
    n_clusters=5,
    run_sensitivity=True,
    include_sampling_stability=False,
)
```

### Production (Full)

```python
PROD_CONFIG = PipelineConfig(
    n_customers=10000,
    auto_select_k=True,
    k_range=(5, 15),
    run_sensitivity=True,
    include_sampling_stability=True,
    generate_report=True,
)
```
