# Working with Segments

Learn how to analyze, filter, and understand your customer segments.

## Segment Structure

Each `Segment` contains:

```python
@dataclass
class Segment:
    segment_id: str           # Unique identifier
    name: str                 # Human-readable name
    size: int                 # Number of customers
    customer_ids: list[str]   # List of customer IDs
    total_clv: Decimal        # Sum of customer CLVs
    avg_clv: Decimal          # Average CLV
    avg_order_value: Decimal  # Average order value
    defining_traits: list[str] # Key characteristics
    cluster_center: list[float] # Cluster centroid
```

## Accessing Segments

### All Segments

```python
result = run_pipeline(config)

for segment in result.segments:
    print(f"{segment.name}: {segment.size} customers")
```

### Valid Segments

Segments that pass validation criteria:

```python
for segment in result.valid_segments:
    validation = result.validation_results[segment.segment_id]
    print(f"{segment.name}: valid={validation.is_valid}")
```

### Actionable Segments

Segments with clear actionability dimensions:

```python
for segment in result.actionable_segments:
    eval = result.actionability_evaluations[segment.segment_id]
    print(f"{segment.name}: {eval.recommended_action}")
```

### Production-Ready Segments

Segments that are both valid AND actionable:

```python
for segment in result.production_ready_segments:
    robustness = result.robustness_scores[segment.segment_id]
    print(f"{segment.name}: robustness={robustness.overall_robustness:.2f}")
```

## Filtering Segments

### By Size

```python
large_segments = [s for s in result.segments if s.size >= 50]
```

### By CLV

```python
from decimal import Decimal

high_value = [s for s in result.segments if s.avg_clv >= Decimal("500")]
```

### By Robustness

```python
stable_segments = [
    s for s in result.segments
    if result.robustness_scores[s.segment_id].overall_robustness >= 0.6
]
```

### By Robustness Tier

```python
from src.data.schemas import RobustnessTier

high_confidence = [
    s for s in result.segments
    if result.robustness_scores[s.segment_id].robustness_tier == RobustnessTier.HIGH
]
```

## Segment Analysis

### View Defining Traits

```python
segment = result.segments[0]
print(f"Traits for {segment.name}:")
for trait in segment.defining_traits:
    print(f"  - {trait}")
```

### Compare Segments

```python
import pandas as pd

data = [{
    'name': s.name,
    'size': s.size,
    'avg_clv': float(s.avg_clv),
    'robustness': result.robustness_scores[s.segment_id].overall_robustness,
} for s in result.segments]

df = pd.DataFrame(data)
print(df.sort_values('avg_clv', ascending=False))
```

### Robustness Breakdown

```python
segment = result.segments[0]
robustness = result.robustness_scores[segment.segment_id]

print(f"Robustness for {segment.name}:")
print(f"  Feature stability: {robustness.feature_stability:.2f}")
print(f"  Time consistency: {robustness.time_consistency:.2f}")
if robustness.sampling_stability:
    print(f"  Sampling stability: {robustness.sampling_stability:.2f}")
print(f"  Overall: {robustness.overall_robustness:.2f}")
print(f"  Tier: {robustness.robustness_tier.value}")
```

### Actionability Dimensions

```python
segment = result.segments[0]
eval = result.actionability_evaluations[segment.segment_id]

print(f"Actionability for {segment.name}:")
print(f"  Is actionable: {eval.is_actionable}")
print(f"  WHAT: {eval.what_dimension}")
print(f"  WHEN: {eval.when_dimension}")
print(f"  HOW: {eval.how_dimension}")
print(f"  WHO: {eval.who_dimension}")
print(f"  Recommended: {eval.recommended_action}")
```

### Business Explanation

```python
segment = result.segments[0]
explanation = result.explanations[segment.segment_id]

print(f"Explanation for {segment.name}:")
print(f"  Summary: {explanation.executive_summary}")
print(f"  Campaign: {explanation.recommended_campaign}")
print(f"  ROI: {explanation.expected_roi}")
print(f"  Confidence: {explanation.confidence_level.value}")
```

## Segment Customers

### Get Customer IDs

```python
segment = result.segments[0]
customer_ids = segment.customer_ids

print(f"Customers in {segment.name}: {len(customer_ids)}")
```

### Find Customer's Segment

```python
def find_segment(customer_id: str, segments: list[Segment]) -> Segment | None:
    for segment in segments:
        if customer_id in segment.customer_ids:
            return segment
    return None

segment = find_segment("customer_123", result.segments)
```

### Match Customers to Profiles

```python
segment = result.segments[0]
segment_profiles = [
    p for p in result.profiles
    if p.customer_id in segment.customer_ids
]
```

## Validation Details

### Check Rejection Reasons

```python
for segment in result.segments:
    validation = result.validation_results.get(segment.segment_id)
    if validation and not validation.is_valid:
        print(f"{segment.name} rejected:")
        for reason in validation.rejection_reasons:
            print(f"  - {reason}")
```

### Viability Assessment

```python
for segment in result.segments:
    viability = result.viabilities.get(segment.segment_id)
    if viability:
        print(f"{segment.name}:")
        print(f"  Is viable: {viability.is_viable}")
        print(f"  Expected ROI: {viability.expected_roi:.1%}")
        print(f"  Strategic impact: {viability.strategic_impact}")
```

## Best Practices

!!! tip "Start with production_ready_segments"
    These segments have passed all quality checks and are safe to use for campaigns.

!!! tip "Review rejection reasons"
    Understanding why segments were rejected helps refine your approach.

!!! tip "Compare across runs"
    Run with different seeds to see which segments consistently appear.
