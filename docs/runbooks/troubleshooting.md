# Troubleshooting Guide

Common issues and solutions for the Actionable Segmentation Engine.

## Pipeline Errors

### InsufficientDataError

**Symptom:**
```
InsufficientDataError: Need at least 30 customers for clustering, got 15
```

**Cause:** Not enough customers to form meaningful clusters.

**Solution:**

1. Increase `n_customers` in configuration:
   ```python
   config = PipelineConfig(n_customers=500)
   ```

2. Or reduce `n_clusters`:
   ```python
   config = PipelineConfig(n_clusters=3)  # Fewer clusters need fewer customers
   ```

3. Rule of thumb: Need at least `k * 10` customers per cluster.

---

### ClusteringError

**Symptom:**
```
ClusteringError: KMeans failed to converge
```

**Cause:** Data has issues that prevent clustering.

**Solution:**

1. Check for constant features (no variance):
   ```python
   import numpy as np
   features = clusterer._extract_features(profiles)
   print(f"Variance: {np.var(features, axis=0)}")
   ```

2. Remove or fix features with zero variance

3. Try different random seed:
   ```python
   config = PipelineConfig(cluster_seed=123)
   ```

---

### ValidationError (Pydantic)

**Symptom:**
```
pydantic.ValidationError: 1 validation error for CustomerProfile
total_revenue: Input should be a valid decimal
```

**Cause:** Input data doesn't match expected schema.

**Solution:**

1. Check input data types
2. Ensure numeric fields are `Decimal` not `float`:
   ```python
   from decimal import Decimal
   profile = CustomerProfile(total_revenue=Decimal("100.50"))
   ```

---

## Segment Quality Issues

### All Segments Rejected

**Symptom:** `result.valid_segments` is empty but `result.segments` has items.

**Cause:** Segments don't meet validation criteria.

**Solution:**

1. Check rejection reasons:
   ```python
   for seg in result.segments:
       validation = result.validation_results.get(seg.segment_id)
       if validation and not validation.is_valid:
           print(f"{seg.name}: {validation.rejection_reasons}")
   ```

2. Lower thresholds:
   ```python
   criteria = ValidationCriteria(
       min_segment_size=5,           # Lower from 10
       min_overall_robustness=0.2,   # Lower from 0.4
   )
   ```

---

### Low Robustness Scores

**Symptom:** All segments have `overall_robustness < 0.3`.

**Cause:** Segments are unstable and change with minor data variations.

**Solution:**

1. **Add more data** - More customers = more stable clusters

2. **Simplify features** - Too many features can cause instability:
   ```python
   # Use fewer features in profile building
   ```

3. **Use fixed k** - Auto k-selection may choose suboptimal k:
   ```python
   config = PipelineConfig(n_clusters=4, auto_select_k=False)
   ```

4. **Accept lower robustness** for exploratory analysis:
   ```python
   criteria = ValidationCriteria(min_overall_robustness=0.2)
   ```

---

### Segments Too Similar

**Symptom:** All segments have similar traits and CLV.

**Cause:** Data lacks variance or clustering isn't capturing differences.

**Solution:**

1. Check feature distributions:
   ```python
   for profile in profiles:
       print(f"{profile.customer_id}: revenue={profile.total_revenue}")
   ```

2. Try more clusters:
   ```python
   config = PipelineConfig(n_clusters=8)
   ```

3. Ensure synthetic data has diverse archetypes

---

## LLM Integration Issues

### LLMIntegrationError

**Symptom:**
```
LLMIntegrationError: Failed to call Claude API
```

**Cause:** Claude API issues.

**Solution:**

1. **Check API key:**
   ```bash
   echo $ANTHROPIC_API_KEY  # Should show key
   ```

2. **Verify API status:** Check [Anthropic Status](https://status.anthropic.com/)

3. **Fall back to mock:**
   ```python
   config = PipelineConfig(use_llm=False)  # Use deterministic mock
   ```

---

### Rate Limit Exceeded

**Symptom:**
```
Rate limit exceeded. Please retry after X seconds.
```

**Cause:** Too many API calls in short period.

**Solution:**

1. Reduce segment count or batch size
2. Add delays between calls
3. Use mock LLM for testing

---

## Performance Issues

### Pipeline Too Slow

**Symptom:** Pipeline takes > 10 seconds for 1000 customers.

**Cause:** Usually sensitivity analysis.

**Solution:**

1. **Skip sensitivity for quick runs:**
   ```python
   config = PipelineConfig(run_sensitivity=False)
   ```

2. **Skip sampling stability:**
   ```python
   config = PipelineConfig(include_sampling_stability=False)
   ```

3. **Reduce customer count for testing:**
   ```python
   config = PipelineConfig(n_customers=100)
   ```

See [Performance Tuning](performance.md) for detailed optimization.

---

## Import Errors

### ModuleNotFoundError

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Solution:**

1. Run from project root:
   ```bash
   cd actionable-segmentation-engine
   python -c "from src.pipeline import quick_segmentation"
   ```

2. Or add to PYTHONPATH:
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

---

## Getting Help

If you can't resolve an issue:

1. Search [existing issues](https://github.com/skmetricmavens/actionable-segmentation-engine/issues)
2. Check the error message carefully for hints
3. Open a new issue with:
   - Full error traceback
   - Configuration used
   - Steps to reproduce
   - Python and package versions
