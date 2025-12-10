# Architecture Decision Records

Architecture Decision Records (ADRs) document significant architectural decisions made during the development of this project.

## What is an ADR?

An ADR captures an important architectural decision along with its context and consequences. ADRs help:

- **Document reasoning** - Capture why decisions were made
- **Enable onboarding** - Help new team members understand the architecture
- **Support evolution** - Provide context when revisiting decisions
- **Promote discussion** - Create a record of alternatives considered

## ADR Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [ADR-001](001-pydantic-v2-models.md) | Use Pydantic v2 for Data Models | Accepted | 2024-12-01 |
| [ADR-002](002-kmeans-clustering.md) | KMeans for Customer Clustering | Accepted | 2024-12-01 |
| [ADR-003](003-mock-llm-support.md) | Mock LLM Support for Testing | Accepted | 2024-12-02 |
| [ADR-004](004-robustness-first.md) | Robustness-First Segmentation | Accepted | 2024-12-03 |
| [ADR-005](005-functional-core.md) | Functional Core with Class Wrappers | Accepted | 2024-12-04 |

## ADR Format

Each ADR follows this structure:

```markdown
# ADR-XXX: Title

## Status
Proposed | Accepted | Deprecated | Superseded

## Context
What is the issue that we're seeing that is motivating this decision?

## Decision
What is the change that we're proposing and/or doing?

## Consequences
What becomes easier or more difficult because of this change?
```

## Creating a New ADR

1. Copy `_template.md` to a new file (e.g., `006-your-decision.md`)
2. Fill in all sections
3. Submit for review via pull request
4. Update this index once accepted

## References

- [ADR GitHub Organization](https://adr.github.io/)
- [Documenting Architecture Decisions - Michael Nygard](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
