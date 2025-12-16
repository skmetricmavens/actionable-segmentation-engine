"""
Load and apply manual narrative overrides from YAML.

This module handles loading user-provided narrative overrides and applying
them to auto-generated narratives. Users can override headlines, insights,
and callouts for any chart while keeping the rest auto-generated.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from src.essays.base import ChartNarrative

logger = logging.getLogger(__name__)


# =============================================================================
# OVERRIDE LOADING
# =============================================================================


def load_overrides(path: Path | str) -> dict[str, dict[str, str | None]]:
    """Load narrative overrides from a YAML file.

    Expected YAML format:
    ```yaml
    narratives:
      loyalty_sankey:
        headline: "Custom headline here"
        insight: null  # Use auto-generated
        callout: "Custom callout"
      champion_scatter:
        headline: "Another custom headline"
    ```

    Args:
        path: Path to the YAML override file

    Returns:
        Dictionary mapping chart_id to override fields

    Raises:
        FileNotFoundError: If the override file doesn't exist
        yaml.YAMLError: If the YAML is invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Override file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    # Extract narratives section
    narratives = data.get("narratives", {})

    if not isinstance(narratives, dict):
        logger.warning("Invalid 'narratives' section in override file, expected dict")
        return {}

    # Normalize the overrides
    result: dict[str, dict[str, str | None]] = {}

    for chart_id, overrides in narratives.items():
        if not isinstance(overrides, dict):
            logger.warning(f"Invalid overrides for {chart_id}, expected dict")
            continue

        result[chart_id] = {
            "headline": overrides.get("headline"),
            "insight": overrides.get("insight"),
            "callout": overrides.get("callout"),
        }

    return result


def load_overrides_safe(path: Path | str | None) -> dict[str, dict[str, str | None]]:
    """Load overrides, returning empty dict on any error.

    Args:
        path: Path to override file, or None

    Returns:
        Override dictionary, or empty dict if path is None or file can't be loaded
    """
    if path is None:
        return {}

    try:
        return load_overrides(path)
    except FileNotFoundError:
        logger.info(f"No override file found at {path}")
        return {}
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML in override file {path}: {e}")
        return {}
    except Exception as e:
        logger.warning(f"Error loading overrides from {path}: {e}")
        return {}


# =============================================================================
# OVERRIDE APPLICATION
# =============================================================================


def apply_overrides(
    narratives: dict[str, ChartNarrative],
    overrides: dict[str, dict[str, str | None]],
) -> dict[str, ChartNarrative]:
    """Apply manual overrides to auto-generated narratives.

    Args:
        narratives: Dictionary of auto-generated ChartNarrative objects
        overrides: Dictionary of override values from YAML

    Returns:
        Updated narratives dictionary with overrides applied
    """
    for chart_id, narrative in narratives.items():
        if chart_id not in overrides:
            continue

        chart_overrides = overrides[chart_id]

        # Apply each override if not None
        if chart_overrides.get("headline") is not None:
            narrative.override_headline = chart_overrides["headline"]

        if chart_overrides.get("insight") is not None:
            narrative.override_insight = chart_overrides["insight"]

        if chart_overrides.get("callout") is not None:
            narrative.override_callout = chart_overrides["callout"]

    return narratives


# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================


def export_narratives_to_yaml(
    narratives: dict[str, ChartNarrative],
    path: Path | str,
    include_auto: bool = True,
) -> None:
    """Export narratives to YAML format for editing.

    Args:
        narratives: Dictionary of ChartNarrative objects
        path: Path to write the YAML file
        include_auto: If True, include auto-generated values as comments
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build the export structure
    export_data: dict[str, Any] = {
        "# Essay Narrative Overrides": None,
        "# Set any field to override the auto-generated text": None,
        "# Set to null or remove to use auto-generated text": None,
        "narratives": {},
    }

    for chart_id, narrative in narratives.items():
        chart_data: dict[str, str | None] = {}

        if include_auto:
            # Include auto-generated as default, with comment
            chart_data["headline"] = narrative.auto_headline
            chart_data["insight"] = narrative.auto_insight
            chart_data["callout"] = narrative.auto_callout
        else:
            # Just include structure with nulls
            chart_data["headline"] = None
            chart_data["insight"] = None
            chart_data["callout"] = None

        export_data["narratives"][chart_id] = chart_data

    # Write YAML with nice formatting
    with open(path, "w") as f:
        f.write("# Essay Narrative Overrides\n")
        f.write("# Edit any field to override the auto-generated text\n")
        f.write("# Set to null or delete to use auto-generated text\n")
        f.write("#\n")
        f.write("# Available charts:\n")
        for chart_id in narratives:
            f.write(f"#   - {chart_id}\n")
        f.write("#\n\n")

        yaml.dump(
            {"narratives": export_data["narratives"]},
            f,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
            width=100,
        )

    logger.info(f"Exported narratives to {path}")


def create_template_override_file(path: Path | str) -> None:
    """Create a template override file with examples.

    Args:
        path: Path to write the template file
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    template = """# Essay Narrative Overrides
# Edit any field to override the auto-generated text
# Set to null or delete the line to use auto-generated text
#
# Example: Override just the headline for the loyalty Sankey chart:
#
# narratives:
#   loyalty_sankey:
#     headline: "The path to loyalty is narrower than you think"
#     insight: null  # Uses auto-generated
#     callout: null  # Uses auto-generated

narratives:
  # Essay 1: How Customers Actually Become Loyal
  loyalty_sankey:
    headline: null
    insight: null
    callout: null

  loyalty_threshold:
    headline: null
    insight: null
    callout: null

  # Essay 2: Not All Champions Are Safe
  champion_scatter:
    headline: null
    insight: null
    callout: null

  champion_thermometer:
    headline: null
    insight: null
    callout: null

  # Essay 3: The Illusion of New Customers
  new_customer_funnel:
    headline: null
    insight: null
    callout: null

  quality_distribution:
    headline: null
    insight: null
    callout: null

  # Essay 4: What Predicts Churn Before It Happens
  churn_timeline:
    headline: null
    insight: null
    callout: null

  churn_radar:
    headline: null
    insight: null
    callout: null
"""

    with open(path, "w") as f:
        f.write(template)

    logger.info(f"Created template override file at {path}")
