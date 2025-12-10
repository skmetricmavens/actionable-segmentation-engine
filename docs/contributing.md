# Contributing

Thank you for your interest in contributing to the Actionable Segmentation Engine!

## Getting Started

### Prerequisites

- Python 3.11+
- Git
- A GitHub account

### Development Setup

1. **Fork and clone:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/actionable-segmentation-engine.git
   cd actionable-segmentation-engine
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

4. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Verify setup:**
   ```bash
   pytest --cov=src
   mypy src/ --strict
   ```

## Development Workflow

### Branch Naming

Use descriptive branch names:

- `feature/add-new-clustering-algorithm`
- `fix/handle-empty-profiles`
- `docs/update-quickstart`
- `refactor/simplify-pipeline`

### Making Changes

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Make changes** following our code standards

3. **Run tests:**
   ```bash
   pytest --cov=src
   ```

4. **Run type checking:**
   ```bash
   mypy src/ --strict
   ```

5. **Run linting:**
   ```bash
   ruff check src/
   ruff format src/
   ```

6. **Commit:**
   ```bash
   git commit -m "feat: add your feature"
   ```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `refactor` | Code refactoring |
| `test` | Adding tests |
| `chore` | Maintenance tasks |

**Examples:**
- `feat: add DBSCAN clustering option`
- `fix: handle empty customer list`
- `docs: update API reference`

### Pull Requests

1. Push your branch:
   ```bash
   git push -u origin feature/your-feature
   ```

2. Open a PR against `main`

3. Fill in the PR template

4. Wait for review

## Code Standards

### Type Hints

All code must have complete type hints:

```python
def process_customers(
    profiles: list[CustomerProfile],
    config: ProcessConfig | None = None,
) -> ProcessResult:
    """Process customer profiles."""
    ...
```

### Docstrings

Use Google-style docstrings:

```python
def calculate_clv(
    revenue: Decimal,
    frequency: float,
    tenure_days: int,
) -> Decimal:
    """Calculate customer lifetime value estimate.

    Args:
        revenue: Total historical revenue
        frequency: Purchase frequency (purchases per day)
        tenure_days: Days since first purchase

    Returns:
        Estimated 3-year CLV

    Raises:
        ValueError: If tenure_days is zero
    """
    ...
```

### Testing

- Write tests for all new code
- Maintain 90%+ coverage
- Use pytest fixtures for shared setup
- Test edge cases and error conditions

```python
def test_calculate_clv_normal():
    result = calculate_clv(Decimal("1000"), 0.1, 365)
    assert result > 0

def test_calculate_clv_zero_tenure():
    with pytest.raises(ValueError):
        calculate_clv(Decimal("1000"), 0.1, 0)
```

### Error Handling

Use domain-specific exceptions:

```python
from src.exceptions import InsufficientDataError

if len(profiles) < MIN_PROFILES:
    raise InsufficientDataError(
        f"Need at least {MIN_PROFILES} profiles",
        context={"actual": len(profiles)}
    )
```

## Architecture Guidelines

### Adding a New Feature

1. **Check ADRs** - Review existing decisions
2. **Write ADR** - If making architectural changes
3. **Follow patterns** - Match existing code structure
4. **Add tests** - Comprehensive coverage
5. **Update docs** - Keep documentation current

### Adding a New Module

1. Create module in appropriate layer
2. Add `__init__.py` exports
3. Write comprehensive tests
4. Add to API reference docs
5. Update architecture docs if needed

## Documentation

### Building Docs Locally

```bash
pip install mkdocs mkdocs-material mkdocstrings[python]
mkdocs serve
```

Visit http://localhost:8000

### Documentation Standards

- Clear, concise writing
- Code examples for all features
- Keep examples runnable
- Update when code changes

## Questions?

- Open a [GitHub Issue](https://github.com/skmetricmavens/actionable-segmentation-engine/issues)
- Check existing issues first
- Provide context and examples

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
