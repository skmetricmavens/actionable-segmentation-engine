"""
Essay section implementations.

Each module creates a complete EssaySection for one of the seven essays:
- loyalty_journey: Essay 1 - How Customers Actually Become Loyal
- champion_fragility: Essay 2 - Not All Champions Are Safe
- new_customer_quality: Essay 3 - The Illusion of New Customers
- churn_prediction: Essay 4 - What Predicts Churn Before It Happens
- whitespace_opportunities: Essay 5 - Hidden Cross-Sell Opportunities
- trait_insights: Essay 6 - What Makes Customers Different
- segment_robustness: Essay 7 - How Robust Are Your Segments
"""

from src.essays.sections.champion_fragility import create_champion_fragility_section
from src.essays.sections.churn_prediction import create_churn_prediction_section
from src.essays.sections.loyalty_journey import create_loyalty_journey_section
from src.essays.sections.new_customer_quality import create_new_customer_quality_section
from src.essays.sections.segment_robustness import create_segment_robustness_section
from src.essays.sections.trait_insights import create_trait_insights_section
from src.essays.sections.whitespace_opportunities import create_whitespace_opportunities_section

__all__ = [
    "create_loyalty_journey_section",
    "create_champion_fragility_section",
    "create_new_customer_quality_section",
    "create_churn_prediction_section",
    "create_whitespace_opportunities_section",
    "create_trait_insights_section",
    "create_segment_robustness_section",
]
