# ADR-001: Use Pydantic v2 for Data Models

## Status

Accepted

## Context

The segmentation engine needs robust data validation at system boundaries. Customer event data can be malformed, missing fields, or contain invalid values. We need:

- Runtime type validation
- Clear error messages for invalid data
- Serialization/deserialization support
- IDE autocompletion and type hints

## Decision

Use **Pydantic v2** for all data models throughout the system:

- `EventRecord` - Raw event data
- `CustomerProfile` - Aggregated customer metrics
- `Segment` - Customer segment with traits
- `RobustnessScore` - Segment stability metrics
- `ActionabilityEvaluation` - LLM evaluation result

```python
from pydantic import BaseModel, Field
from decimal import Decimal

class Segment(BaseModel):
    segment_id: str
    name: str
    size: int = Field(ge=1)
    total_clv: Decimal
    defining_traits: list[str]
```

## Consequences

### Positive

- **Type safety** - Validation at runtime catches errors early
- **Clear errors** - Pydantic provides detailed validation messages
- **Serialization** - Built-in JSON serialization via `.model_dump()`
- **Documentation** - Models serve as living documentation
- **IDE support** - Full autocompletion and type checking

### Negative

- **Learning curve** - Team must learn Pydantic v2 syntax
- **Performance** - Slight overhead for validation (negligible for our scale)
- **Dependency** - Adds external dependency

### Neutral

- Pydantic v2 is widely adopted and well-maintained
- Migration from v1 required syntax changes

## Alternatives Considered

### dataclasses

Python's built-in dataclasses are simpler but lack runtime validation. We would need to add manual validation, defeating the purpose.

### attrs

Similar to Pydantic but less popular in data science ecosystem. Pydantic has better serialization support and wider adoption.

### TypedDict

Only provides static type hints without runtime validation. Not suitable for external data boundaries.

## References

- [Pydantic v2 Documentation](https://docs.pydantic.dev/)
- [Pydantic v2 Migration Guide](https://docs.pydantic.dev/latest/migration/)
