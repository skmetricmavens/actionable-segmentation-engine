"""
Visualization pattern matcher for Pudding.cool-style essays.

This module provides pattern matching logic to select the best
visualization type for a given data story, using keyword matching
and scoring against the visualization catalog.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VizPattern:
    """A visualization pattern from the catalog."""

    id: str
    type: str
    name: str
    description: str
    use_when: list[str]
    avoid_when: list[str]
    data_shape: dict[str, list[str]]
    scroll_interactions: list[dict[str, str]]
    animation: dict[str, int]
    pudding_examples: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VizPattern":
        """Create VizPattern from dictionary."""
        return cls(
            id=data["id"],
            type=data["type"],
            name=data["name"],
            description=data["description"],
            use_when=data["use_when"],
            avoid_when=data["avoid_when"],
            data_shape=data["data_shape"],
            scroll_interactions=data.get("scroll_interactions", []),
            animation=data.get("animation", {}),
            pudding_examples=data.get("pudding_examples", []),
        )

    def match_score(self, query: str, data_fields: list[str] | None = None) -> float:
        """Calculate match score for a query string.

        Args:
            query: Description of the data story / visualization need
            data_fields: Optional list of data fields available

        Returns:
            Score from 0.0 (no match) to 1.0 (perfect match)
        """
        query_lower = query.lower()
        score = 0.0

        # Match against use_when phrases (positive)
        use_when_matches = sum(
            1 for phrase in self.use_when
            if any(word in query_lower for word in phrase.lower().split())
        )
        score += use_when_matches * 0.2

        # Match against description
        desc_words = self.description.lower().split()
        query_words = set(query_lower.split())
        desc_matches = len(set(desc_words) & query_words)
        score += min(desc_matches * 0.1, 0.3)

        # Penalty for avoid_when matches (negative)
        avoid_matches = sum(
            1 for phrase in self.avoid_when
            if any(word in query_lower for word in phrase.lower().split())
        )
        score -= avoid_matches * 0.15

        # Bonus for matching data shape
        if data_fields:
            required = set(self.data_shape.get("required", []))
            available = set(f.lower() for f in data_fields)
            if required and required.issubset(available):
                score += 0.3

        return max(0.0, min(1.0, score))


@dataclass
class MatchResult:
    """Result of pattern matching."""

    pattern: VizPattern
    score: float
    rationale: str


class VizCatalog:
    """Catalog of visualization patterns with matching capability."""

    def __init__(self, catalog_path: Path | None = None):
        """Initialize the catalog.

        Args:
            catalog_path: Path to patterns.json. If None, uses default.
        """
        if catalog_path is None:
            catalog_path = Path(__file__).parent / "patterns.json"

        with open(catalog_path) as f:
            data = json.load(f)

        self.patterns = [
            VizPattern.from_dict(p) for p in data["patterns"]
        ]
        self.recommendations = data.get("essay_recommendations", {})
        self._pattern_index = {p.id: p for p in self.patterns}

    def get_pattern(self, pattern_id: str) -> VizPattern | None:
        """Get a pattern by ID."""
        return self._pattern_index.get(pattern_id)

    def get_recommendation(self, essay_id: str) -> dict[str, Any] | None:
        """Get visualization recommendation for an essay.

        Args:
            essay_id: Essay identifier (e.g., "loyalty_journey")

        Returns:
            Dict with 'primary', 'supporting', and 'rationale' keys
        """
        return self.recommendations.get(essay_id)

    def match(
        self,
        query: str,
        data_fields: list[str] | None = None,
        top_k: int = 3,
    ) -> list[MatchResult]:
        """Find best matching visualization patterns for a query.

        Args:
            query: Natural language description of what to visualize
            data_fields: List of available data fields
            top_k: Number of top results to return

        Returns:
            List of MatchResult sorted by score descending
        """
        results = []

        for pattern in self.patterns:
            score = pattern.match_score(query, data_fields)
            if score > 0:
                # Generate rationale
                matched_uses = [
                    phrase for phrase in pattern.use_when
                    if any(word in query.lower() for word in phrase.lower().split())
                ]
                rationale = (
                    f"Matches: {', '.join(matched_uses[:2])}"
                    if matched_uses else pattern.description
                )

                results.append(MatchResult(
                    pattern=pattern,
                    score=score,
                    rationale=rationale,
                ))

        # Sort by score descending
        results.sort(key=lambda r: r.score, reverse=True)

        return results[:top_k]


# Module-level convenience function
_catalog: VizCatalog | None = None


def get_catalog() -> VizCatalog:
    """Get the singleton catalog instance."""
    global _catalog
    if _catalog is None:
        _catalog = VizCatalog()
    return _catalog


def match_visualization(
    query: str,
    data_fields: list[str] | None = None,
    top_k: int = 3,
) -> list[MatchResult]:
    """Match visualization patterns to a query.

    Convenience function that uses the default catalog.

    Args:
        query: Natural language description of what to visualize
        data_fields: List of available data fields
        top_k: Number of top results to return

    Returns:
        List of MatchResult sorted by score descending

    Example:
        >>> results = match_visualization(
        ...     "Show customer progression from new to champion",
        ...     data_fields=["source", "target", "value"]
        ... )
        >>> results[0].pattern.type
        'sankey'
    """
    return get_catalog().match(query, data_fields, top_k)


def get_essay_recommendation(essay_id: str) -> dict[str, Any] | None:
    """Get recommended visualizations for an essay.

    Args:
        essay_id: Essay identifier (e.g., "loyalty_journey", "champion_fragility")

    Returns:
        Dict with primary, supporting vizs and rationale, or None if not found

    Example:
        >>> rec = get_essay_recommendation("loyalty_journey")
        >>> rec["primary"]
        'sankey_flow'
    """
    return get_catalog().get_recommendation(essay_id)
