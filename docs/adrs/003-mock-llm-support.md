# ADR-003: Mock LLM Support for Testing

## Status

Accepted

## Context

The engine uses LLM (Claude) for:
- Actionability evaluation of segments
- Business-language explanation generation

However, real LLM calls are:
- **Slow** - 1-3 seconds per call
- **Expensive** - API costs per token
- **Non-deterministic** - Different responses each time
- **Require API keys** - Not available in all environments

We need to test the LLM integration layer without these constraints.

## Decision

Implement a **mock LLM interface** that:

1. Returns deterministic responses based on input patterns
2. Matches the real LLM interface (protocol/ABC)
3. Is enabled by default (`use_llm=False`)
4. Can be swapped for real LLM via configuration

```python
from src.pipeline import PipelineConfig

# Mock LLM (default) - fast, deterministic, no API key needed
config = PipelineConfig(use_llm=False)

# Real LLM - requires ANTHROPIC_API_KEY
config = PipelineConfig(use_llm=True)
```

Mock implementation generates realistic responses:

```python
class MockActionabilityFilter:
    def evaluate(self, segment: Segment) -> ActionabilityEvaluation:
        # Deterministic logic based on segment properties
        is_actionable = segment.size >= 10 and segment.total_clv > 1000
        return ActionabilityEvaluation(
            segment_id=segment.segment_id,
            is_actionable=is_actionable,
            dimensions=self._generate_dimensions(segment),
            confidence_level=ConfidenceLevel.MEDIUM,
        )
```

## Consequences

### Positive

- **Fast tests** - Full test suite runs in seconds
- **Deterministic** - Same input produces same output
- **No API costs** - Free to run unlimited tests
- **Offline capable** - Works without internet
- **CI/CD friendly** - No secrets needed in pipelines

### Negative

- **Two code paths** - Must maintain mock and real implementations
- **Mock drift** - Mock may diverge from real LLM behavior
- **Limited coverage** - Can't test actual LLM edge cases

### Neutral

- Mock provides reasonable baseline for business logic testing
- Real LLM integration should be tested separately (integration tests)

## Alternatives Considered

### VCR/Cassette Recording

Record real LLM responses and replay them. Rejected because:
- Recordings become stale as prompts change
- Large response files in repo
- Still non-deterministic for new scenarios

### LLM Mocking Library (responses, httpretty)

Mock at HTTP level. Rejected because:
- Still need to construct realistic responses
- Tight coupling to API response format
- Our approach is more semantic

### Skip LLM Tests Entirely

Only test with real LLM in integration. Rejected because:
- Can't test actionability logic without LLM responses
- Would leave gaps in test coverage

## References

- [Testing LLM Applications](https://www.anthropic.com/research/testing-llm-applications)
- [Mock Object Pattern](https://en.wikipedia.org/wiki/Mock_object)
