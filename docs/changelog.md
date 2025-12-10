# Changelog

All notable changes to the Actionable Segmentation Engine.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Spotify-style docs-like-code documentation
- Backstage catalog-info.yaml for service catalog integration
- Architecture Decision Records (ADRs)
- Operational runbooks
- Comprehensive MkDocs documentation site

## [0.1.0] - 2024-12-10

### Added
- Initial release of Actionable Segmentation Engine
- End-to-end pipeline orchestration (`src/pipeline.py`)
- Synthetic data generation for testing
- Customer ID merge resolution
- Profile building with aggregated metrics
- KMeans clustering with automatic k-selection
- Sensitivity analysis (feature drop, time window, bootstrap)
- Segment validation with business criteria
- LLM integration for actionability evaluation
- Mock LLM support for testing without API
- Segment explanation generation
- Comprehensive reporting and visualization
- Jupyter notebook examples
- 492 tests with 94% coverage
- Full type safety with mypy strict mode

### Pipeline Stages
1. Data Acquisition
2. ID Resolution
3. Profile Building
4. Clustering
5. Sensitivity Analysis
6. Robustness Scoring
7. Validation
8. Viability Assessment
9. Actionability Evaluation
10. Explanation Generation
11. Report Generation

### Documentation
- README with quick start guide
- Architecture documentation
- API reference (auto-generated)

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2024-12-10 | Initial release |

## Upgrade Guide

### From Pre-release to 0.1.0

This is the initial release. No migration needed.

### Future Versions

Check this changelog for breaking changes and migration instructions.
