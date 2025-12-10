# ADR-005: Functional Core with Class Wrappers

## Status

Accepted

## Context

The codebase needs to balance:

- **Testability** - Easy to unit test individual functions
- **Usability** - Convenient API for users
- **Maintainability** - Clear separation of concerns
- **Flexibility** - Support different configuration patterns

## Decision

Adopt a **functional core, imperative shell** architecture:

### Functional Core

Pure functions for business logic with no side effects:

```python
# src/features/aggregators.py
def aggregate_purchases(events: list[EventRecord]) -> PurchaseMetrics:
    """Pure function - same input always produces same output."""
    total = sum(e.value for e in events if e.event_type == "purchase")
    count = len([e for e in events if e.event_type == "purchase"])
    return PurchaseMetrics(total=total, count=count, average=total/count)

def calculate_clv_estimate(
    revenue: Decimal,
    frequency: float,
    tenure_days: int,
) -> Decimal:
    """Pure function - no external dependencies."""
    annual_value = revenue * Decimal(str(frequency)) * Decimal("365") / tenure_days
    return annual_value * Decimal("3")  # 3-year projection
```

### Class Wrappers

Classes for configuration, orchestration, and stateful operations:

```python
# src/features/profile_builder.py
class ProfileBuilder:
    """Class wrapper provides convenient API and configuration."""

    def __init__(self, config: ProfileConfig | None = None):
        self.config = config or ProfileConfig()

    def build_profiles(self, events: list[EventRecord]) -> list[CustomerProfile]:
        """Orchestrates pure functions."""
        grouped = self._group_by_customer(events)
        return [
            CustomerProfile(
                customer_id=cid,
                purchases=aggregate_purchases(customer_events),
                sessions=aggregate_sessions(customer_events),
                clv=calculate_clv_estimate(...),
            )
            for cid, customer_events in grouped.items()
        ]
```

### Testing Strategy

```python
# Test pure functions directly
def test_aggregate_purchases():
    events = [EventRecord(event_type="purchase", value=100), ...]
    result = aggregate_purchases(events)
    assert result.total == 100

# Test class orchestration
def test_profile_builder():
    builder = ProfileBuilder()
    profiles = builder.build_profiles(events)
    assert len(profiles) == expected_count
```

## Consequences

### Positive

- **Easy testing** - Pure functions are trivial to unit test
- **Composability** - Functions can be combined in different ways
- **Clarity** - Clear separation between logic and orchestration
- **Reusability** - Functions can be reused outside class context

### Negative

- **More files** - Separate modules for functions and classes
- **Indirection** - Class methods call functions, not implement directly
- **Learning curve** - Pattern may be unfamiliar to some developers

### Neutral

- Common pattern in functional programming communities
- Well-suited for data processing pipelines

## Alternatives Considered

### Pure OOP

All logic in class methods. Rejected because:
- Harder to test methods with dependencies
- Encourages mocking over pure testing
- Less composable

### Pure Functional

No classes, only functions. Rejected because:
- Configuration becomes awkward (many parameters)
- No natural place for initialization logic
- Less familiar to Python developers

### Service Layer Pattern

Separate service classes for each domain. Rejected because:
- Overkill for our scope
- More boilerplate
- Less direct than functional approach

## References

- [Functional Core, Imperative Shell](https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell)
- [Parse, Don't Validate](https://lexi-lambda.github.io/blog/2019/11/05/parse-don-t-validate/)
