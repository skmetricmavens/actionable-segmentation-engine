"""
Module: whitespace

Purpose: Whitespace analysis for cross-sell opportunity detection.

Uses FAISS for scalable similarity search to find lookalike customers:
- Identifies customers who engage like buyers but haven't purchased in a category
- Scores opportunities using nearest neighbor similarity to actual buyers
- Supports category and subcategory level analysis

Architecture Notes:
- FAISS IndexFlatIP for exact cosine similarity (small/medium datasets)
- FAISS IndexIVFFlat for approximate search (large datasets 100K+)
- Engagement features used for similarity (not purchase features)
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import faiss
import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from src.data.schemas import CustomerProfile, CategoryAffinity


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class LookalikeCandidate:
    """A customer identified as similar to category buyers."""

    customer_id: str
    similarity_score: float  # 0-1, higher = more similar to buyers
    nearest_buyer_ids: list[str]  # IDs of most similar buyers
    nearest_buyer_distances: list[float]  # Distances to nearest buyers
    engagement_in_category: dict[str, Any]  # Their engagement metrics in this category


@dataclass
class CategoryWhitespace:
    """Whitespace opportunity for a single category."""

    category: str
    subcategory: str | None = None

    # Seed (buyer) statistics
    n_buyers: int = 0
    buyer_avg_clv: Decimal = Decimal("0")
    buyer_avg_purchases: float = 0.0

    # Candidate statistics
    n_candidates: int = 0  # Engaged but non-buyers
    n_lookalikes: int = 0  # High-similarity candidates

    # Opportunity metrics
    total_opportunity_value: Decimal = Decimal("0")  # Estimated if converted
    avg_similarity_score: float = 0.0

    # Top opportunities
    top_candidates: list[LookalikeCandidate] = field(default_factory=list)


@dataclass
class WhitespaceAnalysisResult:
    """Complete whitespace analysis results."""

    # Per-category opportunities
    category_whitespaces: dict[str, CategoryWhitespace] = field(default_factory=dict)

    # Global statistics
    total_customers_analyzed: int = 0
    total_opportunities: int = 0
    total_opportunity_value: Decimal = Decimal("0")

    # Top opportunities across all categories
    top_opportunities: list[tuple[str, LookalikeCandidate]] = field(default_factory=list)

    def get_top_categories(self, n: int = 5) -> list[CategoryWhitespace]:
        """Get top N categories by opportunity value."""
        sorted_cats = sorted(
            self.category_whitespaces.values(),
            key=lambda x: x.total_opportunity_value,
            reverse=True,
        )
        return sorted_cats[:n]


# =============================================================================
# FEATURE EXTRACTION FOR LOOKALIKE
# =============================================================================


def extract_engagement_features(
    profiles: list[CustomerProfile],
) -> tuple[NDArray[np.float32], list[str]]:
    """
    Extract engagement-only features for lookalike modeling.

    We exclude purchase/revenue features because we want to find
    customers who ENGAGE like buyers, not who already buy.

    Args:
        profiles: Customer profiles

    Returns:
        Tuple of (feature matrix, feature names)
    """
    feature_names = [
        "total_sessions",
        "total_page_views",
        "total_items_viewed",
        "total_cart_additions",
        "cart_abandonment_rate",
        "mobile_session_ratio",
        "days_active",  # tenure
        "session_frequency",  # sessions per day active
    ]

    rows: list[list[float]] = []

    for profile in profiles:
        # Calculate derived features
        days_active = (profile.last_seen - profile.first_seen).days + 1
        session_frequency = profile.total_sessions / max(days_active, 1)

        row = [
            float(profile.total_sessions),
            float(profile.total_page_views),
            float(profile.total_items_viewed),
            float(profile.total_cart_additions),
            profile.cart_abandonment_rate,
            profile.mobile_session_ratio,
            float(days_active),
            session_frequency,
        ]
        rows.append(row)

    return np.array(rows, dtype=np.float32), feature_names


def extract_category_engagement_features(
    profiles: list[CustomerProfile],
    category: str,
) -> tuple[NDArray[np.float32], list[str]]:
    """
    Extract category-specific engagement features.

    Args:
        profiles: Customer profiles
        category: Category to extract features for

    Returns:
        Tuple of (feature matrix, feature names)
    """
    feature_names = [
        "category_view_count",
        "category_engagement_score",
        "total_sessions",
        "total_page_views",
        "cart_abandonment_rate",
        "mobile_session_ratio",
    ]

    rows: list[list[float]] = []

    for profile in profiles:
        # Find category affinity
        cat_affinity = next(
            (a for a in profile.category_affinities if a.category == category),
            None,
        )

        row = [
            float(cat_affinity.view_count) if cat_affinity else 0.0,
            cat_affinity.engagement_score if cat_affinity else 0.0,
            float(profile.total_sessions),
            float(profile.total_page_views),
            profile.cart_abandonment_rate,
            profile.mobile_session_ratio,
        ]
        rows.append(row)

    return np.array(rows, dtype=np.float32), feature_names


# =============================================================================
# FAISS INDEX MANAGEMENT
# =============================================================================


class SimilarityIndex:
    """
    FAISS-based similarity index for lookalike modeling.

    Automatically selects index type based on dataset size:
    - < 10K: IndexFlatIP (exact search)
    - >= 10K: IndexIVFFlat (approximate search)
    """

    def __init__(
        self,
        use_approximate: bool | None = None,
        n_clusters: int = 100,
        n_probe: int = 10,
    ):
        """
        Initialize SimilarityIndex.

        Args:
            use_approximate: Force approximate (True) or exact (False) search.
                            None = auto-select based on data size.
            n_clusters: Number of clusters for IVF index
            n_probe: Number of clusters to search (trade-off speed vs accuracy)
        """
        self.use_approximate = use_approximate
        self.n_clusters = n_clusters
        self.n_probe = n_probe

        self._index: faiss.Index | None = None
        self._scaler: StandardScaler | None = None
        self._ids: list[str] = []
        self._dimension: int = 0

    def build(
        self,
        features: NDArray[np.float32],
        ids: list[str],
    ) -> None:
        """
        Build the FAISS index from features.

        Args:
            features: Feature matrix (n_samples, n_features)
            ids: Customer IDs corresponding to rows
        """
        n_samples, n_features = features.shape
        self._dimension = n_features
        self._ids = ids

        # Standardize features
        self._scaler = StandardScaler()
        normalized = self._scaler.fit_transform(features).astype(np.float32)

        # L2 normalize for cosine similarity via inner product
        faiss.normalize_L2(normalized)

        # Select index type
        use_approx = self.use_approximate
        if use_approx is None:
            use_approx = n_samples >= 10000

        if use_approx and n_samples >= self.n_clusters:
            # IVF index for large datasets
            quantizer = faiss.IndexFlatIP(n_features)
            n_clusters = min(self.n_clusters, n_samples // 10)
            self._index = faiss.IndexIVFFlat(
                quantizer, n_features, n_clusters, faiss.METRIC_INNER_PRODUCT
            )
            self._index.train(normalized)
            self._index.add(normalized)
            self._index.nprobe = self.n_probe
        else:
            # Exact search for smaller datasets
            self._index = faiss.IndexFlatIP(n_features)
            self._index.add(normalized)

    def search(
        self,
        query_features: NDArray[np.float32],
        k: int = 10,
    ) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
        """
        Search for k nearest neighbors.

        Args:
            query_features: Query feature matrix
            k: Number of neighbors to return

        Returns:
            Tuple of (distances, indices)
        """
        if self._index is None or self._scaler is None:
            raise ValueError("Index not built. Call build() first.")

        # Normalize query using same scaler
        normalized = self._scaler.transform(query_features).astype(np.float32)
        faiss.normalize_L2(normalized)

        # Search
        k = min(k, self._index.ntotal)
        distances, indices = self._index.search(normalized, k)

        return distances, indices

    def get_ids(self, indices: NDArray[np.int64]) -> list[list[str]]:
        """Convert indices to customer IDs."""
        return [[self._ids[i] for i in row if i >= 0] for row in indices]

    @property
    def n_vectors(self) -> int:
        """Number of vectors in index."""
        return self._index.ntotal if self._index else 0


# =============================================================================
# WHITESPACE ANALYZER
# =============================================================================


class WhitespaceAnalyzer:
    """
    Analyzes whitespace opportunities using FAISS-based lookalike modeling.

    For each category:
    1. Identifies "seed" customers (buyers in that category)
    2. Identifies "candidates" (engaged but non-buyers)
    3. Uses FAISS to find candidates most similar to seeds
    4. Scores and ranks opportunities
    """

    def __init__(
        self,
        *,
        min_seed_size: int = 5,
        min_candidate_engagement: int = 1,  # Min views in category
        similarity_threshold: float = 0.5,
        top_k_neighbors: int = 5,
        top_candidates_per_category: int = 100,
    ):
        """
        Initialize WhitespaceAnalyzer.

        Args:
            min_seed_size: Minimum buyers needed to analyze a category
            min_candidate_engagement: Minimum views for a candidate
            similarity_threshold: Min similarity to be considered a lookalike
            top_k_neighbors: Number of nearest buyers to consider
            top_candidates_per_category: Max candidates to return per category
        """
        self.min_seed_size = min_seed_size
        self.min_candidate_engagement = min_candidate_engagement
        self.similarity_threshold = similarity_threshold
        self.top_k_neighbors = top_k_neighbors
        self.top_candidates_per_category = top_candidates_per_category

    def analyze(
        self,
        profiles: list[CustomerProfile],
        categories: list[str] | None = None,
    ) -> WhitespaceAnalysisResult:
        """
        Run whitespace analysis across categories.

        Args:
            profiles: All customer profiles
            categories: Specific categories to analyze (None = all detected)

        Returns:
            WhitespaceAnalysisResult with opportunities
        """
        # Detect categories if not provided
        if categories is None:
            categories = self._detect_categories(profiles)

        result = WhitespaceAnalysisResult(
            total_customers_analyzed=len(profiles),
        )

        # Build profile lookup
        profile_lookup = {p.internal_customer_id: p for p in profiles}

        # Analyze each category
        for category in categories:
            whitespace = self._analyze_category(profiles, category, profile_lookup)
            if whitespace and whitespace.n_lookalikes > 0:
                result.category_whitespaces[category] = whitespace
                result.total_opportunities += whitespace.n_lookalikes
                result.total_opportunity_value += whitespace.total_opportunity_value

                # Add to global top opportunities
                for candidate in whitespace.top_candidates[:10]:
                    result.top_opportunities.append((category, candidate))

        # Sort global opportunities by similarity
        result.top_opportunities.sort(
            key=lambda x: x[1].similarity_score, reverse=True
        )
        result.top_opportunities = result.top_opportunities[:100]

        return result

    def _detect_categories(self, profiles: list[CustomerProfile]) -> list[str]:
        """Detect all categories from profile affinities."""
        categories: set[str] = set()
        for profile in profiles:
            for affinity in profile.category_affinities:
                categories.add(affinity.category)
        return sorted(categories)

    def _analyze_category(
        self,
        profiles: list[CustomerProfile],
        category: str,
        profile_lookup: dict[str, CustomerProfile],
    ) -> CategoryWhitespace | None:
        """
        Analyze whitespace for a single category.

        Args:
            profiles: All profiles
            category: Category to analyze
            profile_lookup: ID to profile mapping

        Returns:
            CategoryWhitespace or None if insufficient data
        """
        # Separate buyers and non-buyers for this category
        buyers: list[CustomerProfile] = []
        candidates: list[CustomerProfile] = []

        for profile in profiles:
            cat_affinity = next(
                (a for a in profile.category_affinities if a.category == category),
                None,
            )

            if cat_affinity and cat_affinity.purchase_count > 0:
                buyers.append(profile)
            elif cat_affinity and cat_affinity.view_count >= self.min_candidate_engagement:
                # Engaged but not purchased
                candidates.append(profile)

        # Check minimum requirements
        if len(buyers) < self.min_seed_size:
            return None

        if not candidates:
            return CategoryWhitespace(
                category=category,
                n_buyers=len(buyers),
                buyer_avg_clv=sum(b.clv_estimate for b in buyers) / len(buyers),
                buyer_avg_purchases=sum(b.total_purchases for b in buyers) / len(buyers),
                n_candidates=0,
                n_lookalikes=0,
            )

        # Extract features for buyers (seeds)
        buyer_features, _ = extract_category_engagement_features(buyers, category)
        buyer_ids = [b.internal_customer_id for b in buyers]

        # Build FAISS index on buyers
        index = SimilarityIndex()
        index.build(buyer_features, buyer_ids)

        # Extract features for candidates
        candidate_features, _ = extract_category_engagement_features(candidates, category)
        candidate_ids = [c.internal_customer_id for c in candidates]

        # Find nearest buyers for each candidate
        k = min(self.top_k_neighbors, len(buyers))
        distances, indices = index.search(candidate_features, k)

        # Score candidates
        lookalike_candidates: list[LookalikeCandidate] = []

        for i, (cand_id, dists, idxs) in enumerate(zip(candidate_ids, distances, indices)):
            # Average similarity to top-k buyers (distances are actually similarities for IP)
            avg_similarity = float(np.mean(dists[dists > 0])) if np.any(dists > 0) else 0.0

            if avg_similarity >= self.similarity_threshold:
                nearest_ids = [buyer_ids[int(idx)] for idx in idxs if idx >= 0]
                nearest_dists = [float(d) for d in dists if d > 0]

                # Get category engagement info
                candidate_profile = profile_lookup[cand_id]
                cat_affinity = next(
                    (a for a in candidate_profile.category_affinities if a.category == category),
                    None,
                )

                lookalike_candidates.append(
                    LookalikeCandidate(
                        customer_id=cand_id,
                        similarity_score=avg_similarity,
                        nearest_buyer_ids=nearest_ids[:5],
                        nearest_buyer_distances=nearest_dists[:5],
                        engagement_in_category={
                            "view_count": cat_affinity.view_count if cat_affinity else 0,
                            "engagement_score": cat_affinity.engagement_score if cat_affinity else 0,
                        },
                    )
                )

        # Sort by similarity
        lookalike_candidates.sort(key=lambda x: x.similarity_score, reverse=True)
        top_candidates = lookalike_candidates[: self.top_candidates_per_category]

        # Calculate opportunity value (estimated CLV if converted)
        buyer_avg_clv = sum(b.clv_estimate for b in buyers) / len(buyers)
        total_opportunity = buyer_avg_clv * len(lookalike_candidates)

        return CategoryWhitespace(
            category=category,
            n_buyers=len(buyers),
            buyer_avg_clv=buyer_avg_clv,
            buyer_avg_purchases=sum(b.total_purchases for b in buyers) / len(buyers),
            n_candidates=len(candidates),
            n_lookalikes=len(lookalike_candidates),
            total_opportunity_value=total_opportunity,
            avg_similarity_score=float(np.mean([c.similarity_score for c in lookalike_candidates]))
            if lookalike_candidates
            else 0.0,
            top_candidates=top_candidates,
        )

    def analyze_cross_category(
        self,
        profiles: list[CustomerProfile],
        source_category: str,
        target_category: str,
    ) -> CategoryWhitespace | None:
        """
        Find customers who buy in source_category but not target_category,
        who are similar to target_category buyers.

        Useful for: "Electronics buyers who might like Sports equipment"

        Args:
            profiles: All profiles
            source_category: Category customers already buy from
            target_category: Category to find opportunities in

        Returns:
            CategoryWhitespace for cross-sell opportunity
        """
        # Find source buyers who don't buy target
        source_buyers_not_target: list[CustomerProfile] = []
        target_buyers: list[CustomerProfile] = []

        for profile in profiles:
            source_affinity = next(
                (a for a in profile.category_affinities if a.category == source_category),
                None,
            )
            target_affinity = next(
                (a for a in profile.category_affinities if a.category == target_category),
                None,
            )

            has_source_purchase = source_affinity and source_affinity.purchase_count > 0
            has_target_purchase = target_affinity and target_affinity.purchase_count > 0
            has_target_engagement = target_affinity and target_affinity.view_count > 0

            if has_target_purchase:
                target_buyers.append(profile)
            elif has_source_purchase and has_target_engagement:
                # Buys source, engaged with target, but doesn't buy target
                source_buyers_not_target.append(profile)

        if len(target_buyers) < self.min_seed_size:
            return None

        if not source_buyers_not_target:
            return None

        # Build index on target buyers
        target_features, _ = extract_engagement_features(target_buyers)
        target_ids = [t.internal_customer_id for t in target_buyers]

        index = SimilarityIndex()
        index.build(target_features, target_ids)

        # Find similar source buyers
        candidate_features, _ = extract_engagement_features(source_buyers_not_target)
        candidate_ids = [c.internal_customer_id for c in source_buyers_not_target]

        k = min(self.top_k_neighbors, len(target_buyers))
        distances, indices = index.search(candidate_features, k)

        # Score
        lookalikes: list[LookalikeCandidate] = []
        profile_lookup = {p.internal_customer_id: p for p in profiles}

        for cand_id, dists, idxs in zip(candidate_ids, distances, indices):
            avg_sim = float(np.mean(dists[dists > 0])) if np.any(dists > 0) else 0.0

            if avg_sim >= self.similarity_threshold:
                nearest_ids = [target_ids[int(idx)] for idx in idxs if idx >= 0]

                target_affinity = next(
                    (a for a in profile_lookup[cand_id].category_affinities if a.category == target_category),
                    None,
                )

                lookalikes.append(
                    LookalikeCandidate(
                        customer_id=cand_id,
                        similarity_score=avg_sim,
                        nearest_buyer_ids=nearest_ids[:5],
                        nearest_buyer_distances=[float(d) for d in dists[:5] if d > 0],
                        engagement_in_category={
                            "source_category": source_category,
                            "target_view_count": target_affinity.view_count if target_affinity else 0,
                        },
                    )
                )

        lookalikes.sort(key=lambda x: x.similarity_score, reverse=True)

        buyer_avg_clv = sum(t.clv_estimate for t in target_buyers) / len(target_buyers)

        return CategoryWhitespace(
            category=f"{source_category} -> {target_category}",
            n_buyers=len(target_buyers),
            buyer_avg_clv=buyer_avg_clv,
            n_candidates=len(source_buyers_not_target),
            n_lookalikes=len(lookalikes),
            total_opportunity_value=buyer_avg_clv * len(lookalikes),
            avg_similarity_score=float(np.mean([l.similarity_score for l in lookalikes]))
            if lookalikes
            else 0.0,
            top_candidates=lookalikes[: self.top_candidates_per_category],
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def find_category_whitespace(
    profiles: list[CustomerProfile],
    category: str,
    *,
    min_buyers: int = 5,
    similarity_threshold: float = 0.5,
) -> CategoryWhitespace | None:
    """
    Quick function to find whitespace in a single category.

    Args:
        profiles: Customer profiles
        category: Category to analyze
        min_buyers: Minimum buyers required
        similarity_threshold: Minimum similarity for lookalikes

    Returns:
        CategoryWhitespace or None
    """
    analyzer = WhitespaceAnalyzer(
        min_seed_size=min_buyers,
        similarity_threshold=similarity_threshold,
    )
    result = analyzer.analyze(profiles, categories=[category])
    return result.category_whitespaces.get(category)


def find_all_whitespace(
    profiles: list[CustomerProfile],
    *,
    similarity_threshold: float = 0.5,
    top_n_categories: int = 10,
) -> WhitespaceAnalysisResult:
    """
    Find whitespace opportunities across all categories.

    Args:
        profiles: Customer profiles
        similarity_threshold: Minimum similarity for lookalikes
        top_n_categories: Max categories to include in result

    Returns:
        WhitespaceAnalysisResult
    """
    analyzer = WhitespaceAnalyzer(similarity_threshold=similarity_threshold)
    return analyzer.analyze(profiles)


def get_whitespace_summary(result: WhitespaceAnalysisResult) -> dict[str, Any]:
    """
    Generate human-readable summary of whitespace analysis.

    Args:
        result: WhitespaceAnalysisResult

    Returns:
        Summary dictionary
    """
    top_cats = result.get_top_categories(5)

    return {
        "total_customers": result.total_customers_analyzed,
        "total_opportunities": result.total_opportunities,
        "total_opportunity_value": float(result.total_opportunity_value),
        "categories_analyzed": len(result.category_whitespaces),
        "top_categories": [
            {
                "category": cat.category,
                "n_lookalikes": cat.n_lookalikes,
                "opportunity_value": float(cat.total_opportunity_value),
                "avg_similarity": cat.avg_similarity_score,
            }
            for cat in top_cats
        ],
    }
