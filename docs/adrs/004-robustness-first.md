# ADR-004: Robustness-First Segmentation

## Status

Accepted

## Context

Customer segmentation can produce unstable results that:
- Change significantly with minor data variations
- Disappear when a single feature is removed
- Differ between time periods

Segments used for marketing campaigns must be **stable and reliable**. We can't target customers based on segments that might not exist next week.

## Decision

Implement **robustness validation** as a required step before segment approval:

### 1. Feature Drop Stability

Remove each feature one at a time and re-cluster. Stable segments should persist.

```python
feature_stability = analyzer.test_feature_sensitivity(profiles, segments)
# Returns: 0.0 (unstable) to 1.0 (stable)
```

### 2. Time Window Stability

Cluster different time periods. Stable segments should appear consistently.

```python
time_consistency = analyzer.test_time_window_sensitivity(profiles, segments)
# Returns: 0.0 (varies by time) to 1.0 (consistent)
```

### 3. Bootstrap Sampling Stability

Re-sample customers with replacement. Stable segments should be reproducible.

```python
sampling_stability = analyzer.test_sampling_stability(profiles, segments)
# Returns: 0.0 (sample-dependent) to 1.0 (robust)
```

### Overall Robustness Score

```python
@dataclass
class RobustnessScore:
    segment_id: str
    feature_stability: float
    time_consistency: float
    sampling_stability: float | None
    overall_robustness: float  # Weighted average
    robustness_tier: RobustnessTier  # HIGH, MEDIUM, LOW
```

### Validation Thresholds

```python
criteria = ValidationCriteria(
    min_feature_stability=0.3,
    min_overall_robustness=0.4,
)
```

Segments below thresholds are marked as **not production-ready**.

## Consequences

### Positive

- **Reliable segments** - Only stable segments reach production
- **Confidence** - Quantified robustness enables informed decisions
- **Transparency** - Clear metrics explain segment quality
- **Early warning** - Unstable segments caught before deployment

### Negative

- **Longer pipeline** - Sensitivity analysis adds ~200ms
- **Fewer segments** - Some clusters won't pass validation
- **Complexity** - More concepts for users to understand

### Neutral

- Thresholds are configurable per use case
- Users can skip sensitivity analysis if needed (`run_sensitivity=False`)

## Alternatives Considered

### Skip Robustness Validation

Trust clustering results directly. Rejected because:
- No way to know if segments are meaningful
- Risk of targeting non-existent customer groups
- Wastes marketing budget on unstable segments

### Cross-Validation Only

Use k-fold cross-validation. Rejected because:
- Doesn't test feature importance
- Doesn't test temporal stability
- Less interpretable than our approach

### Manual Review Only

Have analysts review each segment. Rejected because:
- Doesn't scale
- Subjective assessments
- No quantified metrics

## References

- [Cluster Stability Analysis](https://en.wikipedia.org/wiki/Cluster_analysis#Evaluation_and_assessment)
- [Bootstrap Methods](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))
