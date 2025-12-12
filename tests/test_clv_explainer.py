"""Tests for CLV explainer module."""

from datetime import datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from src.data.schemas import BehaviorType, CategoryAffinity, CustomerProfile

# Check if SHAP is available
try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


# Skip all tests if SHAP not available
pytestmark = pytest.mark.skipif(
    not SHAP_AVAILABLE, reason="SHAP not installed"
)


def make_profile(
    internal_customer_id: str,
    total_revenue: Decimal,
    last_seen: datetime,
    total_purchases: int = 5,
) -> CustomerProfile:
    """Create a test CustomerProfile."""
    return CustomerProfile(
        internal_customer_id=internal_customer_id,
        first_seen=last_seen - timedelta(days=180),
        last_seen=last_seen,
        total_purchases=total_purchases,
        total_revenue=total_revenue,
        avg_order_value=total_revenue / total_purchases if total_purchases > 0 else Decimal("0"),
        total_sessions=20,
        total_page_views=100,
        total_items_viewed=50,
        total_cart_additions=15,
        days_since_last_purchase=30,
        purchase_frequency_per_month=0.83,
        cart_abandonment_rate=0.3,
        total_refunds=Decimal("0.00"),
        refund_rate=0.0,
        churn_risk_score=0.2,
        preferred_day_of_week=2,
        preferred_hour_of_day=14,
        mobile_session_ratio=0.4,
        category_affinities=[
            CategoryAffinity(
                category="Electronics",
                level=1,
                engagement_score=0.8,
                purchase_count=5,
                view_count=20,
            ),
        ],
        behavior_type=BehaviorType.REGULAR,
    )


def create_profiles(n: int = 50) -> list[CustomerProfile]:
    """Create a list of profiles for testing."""
    profiles = []
    base_date = datetime(2024, 1, 1)

    for i in range(n):
        revenue = Decimal(str(100 + i * 50 + np.random.randint(-20, 20)))
        purchases = max(1, 5 + i // 10)
        last_seen = base_date + timedelta(days=i * 3)

        profiles.append(
            make_profile(
                internal_customer_id=f"cust_{i:03d}",
                total_revenue=revenue,
                last_seen=last_seen,
                total_purchases=purchases,
            )
        )

    return profiles


@pytest.fixture
def trained_model():
    """Create a trained model for testing."""
    from sklearn.ensemble import RandomForestRegressor

    from src.features.clv.features import CLVFeatureBuilder

    profiles = create_profiles(50)
    builder = CLVFeatureBuilder()
    X, _ = builder.build_feature_matrix(profiles)
    y = np.array([float(p.total_revenue) for p in profiles])

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)

    return model


@pytest.fixture
def feature_builder():
    """Create a feature builder for testing."""
    from src.features.clv.features import CLVFeatureBuilder

    return CLVFeatureBuilder()


class TestCLVExplainer:
    """Tests for CLVExplainer class."""

    def test_create_explainer(self, trained_model, feature_builder):
        """Test creating an explainer."""
        from src.features.clv.explainer import CLVExplainer

        explainer = CLVExplainer(trained_model, feature_builder)
        assert explainer is not None

    def test_fit_explainer(self, trained_model, feature_builder):
        """Test fitting an explainer."""
        from src.features.clv.explainer import CLVExplainer

        profiles = create_profiles(30)
        explainer = CLVExplainer(trained_model, feature_builder)
        explainer.fit(profiles)

        assert explainer._is_fitted

    def test_explain_single_profile(self, trained_model, feature_builder):
        """Test explaining a single profile."""
        from src.features.clv.explainer import CLVExplainer

        profiles = create_profiles(30)
        explainer = CLVExplainer(trained_model, feature_builder)
        explainer.fit(profiles)

        explanation = explainer.explain(profiles[0])

        assert explanation.customer_id == profiles[0].internal_customer_id
        assert len(explanation.contributions) > 0
        assert explanation.base_value is not None
        assert explanation.predicted_value is not None

    def test_explain_without_fit_raises(self, trained_model, feature_builder):
        """Test that explain without fit raises error."""
        from src.features.clv.explainer import CLVExplainer

        profiles = create_profiles(10)
        explainer = CLVExplainer(trained_model, feature_builder)

        with pytest.raises(RuntimeError, match="not fitted"):
            explainer.explain(profiles[0])

    def test_explain_batch(self, trained_model, feature_builder):
        """Test batch explanation."""
        from src.features.clv.explainer import CLVExplainer

        profiles = create_profiles(30)
        explainer = CLVExplainer(trained_model, feature_builder)
        explainer.fit(profiles)

        explanations = explainer.explain_batch(profiles[:5])

        assert len(explanations) == 5
        for exp in explanations:
            assert len(exp.contributions) > 0

    def test_get_feature_importance(self, trained_model, feature_builder):
        """Test global feature importance calculation."""
        from src.features.clv.explainer import CLVExplainer

        profiles = create_profiles(30)
        explainer = CLVExplainer(trained_model, feature_builder)
        explainer.fit(profiles)

        importance = explainer.get_feature_importance(profiles)

        assert len(importance) > 0
        assert all(v >= 0 for v in importance.values())


class TestCLVExplanation:
    """Tests for CLVExplanation dataclass."""

    def test_top_positive_features(self, trained_model, feature_builder):
        """Test getting top positive contributing features."""
        from src.features.clv.explainer import CLVExplainer

        profiles = create_profiles(30)
        explainer = CLVExplainer(trained_model, feature_builder)
        explainer.fit(profiles)

        explanation = explainer.explain(profiles[0])
        top_pos = explanation.top_positive_features

        if top_pos:  # May be empty if all contributions are negative
            assert all(c.shap_value > 0 for c in top_pos)
            # Should be sorted in descending order
            shap_values = [c.shap_value for c in top_pos]
            assert shap_values == sorted(shap_values, reverse=True)

    def test_top_negative_features(self, trained_model, feature_builder):
        """Test getting top negative contributing features."""
        from src.features.clv.explainer import CLVExplainer

        profiles = create_profiles(30)
        explainer = CLVExplainer(trained_model, feature_builder)
        explainer.fit(profiles)

        explanation = explainer.explain(profiles[0])
        top_neg = explanation.top_negative_features

        if top_neg:  # May be empty if all contributions are positive
            assert all(c.shap_value < 0 for c in top_neg)
            # Should be sorted in ascending order (most negative first)
            shap_values = [c.shap_value for c in top_neg]
            assert shap_values == sorted(shap_values)

    def test_to_dict(self, trained_model, feature_builder):
        """Test converting explanation to dictionary."""
        from src.features.clv.explainer import CLVExplainer

        profiles = create_profiles(30)
        explainer = CLVExplainer(trained_model, feature_builder)
        explainer.fit(profiles)

        explanation = explainer.explain(profiles[0])
        result = explanation.to_dict()

        assert "customer_id" in result
        assert "base_value" in result
        assert "predicted_value" in result
        assert "contributions" in result
        assert isinstance(result["contributions"], list)


class TestConvenienceFunction:
    """Tests for create_explainer convenience function."""

    def test_create_explainer_function(self, trained_model):
        """Test create_explainer convenience function."""
        from src.features.clv.explainer import create_explainer

        explainer = create_explainer(trained_model)
        assert explainer is not None
