"""
Essay 7: How Robust Are Your Segments

This essay examines segment stability - which segments are reliable enough to build
campaigns on, and which might change with small data shifts.

Key visualizations:
- Robustness Gauge: Tier distribution (HIGH/MEDIUM/LOW)
- Feature Stability: Which features most affect segment boundaries
- Time Consistency: How stable segments are across time windows
"""

from src.essays.base import ChartNarrative, ChartSpec, EssaySection, ScrollyStep
from src.essays.data_queries import SensitivityData
from src.essays.narratives import (
    generate_feature_stability_narrative,
    generate_robustness_gauge_narrative,
    generate_time_consistency_narrative,
)


def create_segment_robustness_section(
    data: SensitivityData,
    narratives: dict[str, ChartNarrative] | None = None,
) -> EssaySection:
    """Create the Segment Robustness essay section.

    Args:
        data: SensitivityData with metrics and chart data
        narratives: Optional pre-generated narratives (with possible overrides)

    Returns:
        Complete EssaySection for the segment robustness essay
    """
    # Use provided narratives or generate fresh ones
    if narratives is None:
        narratives = {}

    gauge_narrative = (
        narratives.get("robustness_gauge")
        or generate_robustness_gauge_narrative(data)
    )
    stability_narrative = (
        narratives.get("feature_stability")
        or generate_feature_stability_narrative(data)
    )
    consistency_narrative = (
        narratives.get("time_consistency")
        or generate_time_consistency_narrative(data)
    )

    # Create chart specs
    gauge_chart = create_robustness_gauge_spec("robustness_gauge", data)
    stability_chart = create_feature_stability_spec("feature_stability", data)
    consistency_chart = create_time_consistency_spec("time_consistency", data)

    # Build scrollytelling steps
    total_segments = data.total_segments
    high_pct = data.high_robustness_percentage
    high_count = data.high_robustness_count
    most_critical = data.most_critical_feature.replace("_", " ").title()
    avg_stability = data.avg_feature_stability

    scrolly_steps = [
        ScrollyStep(
            step_number=1,
            narrative_text=(
                f"Here's the thing about customer segments: some are rock solid, "
                f"others shift like sand. We tested all {total_segments} segments "
                f"to see which ones you can actually rely on for campaigns."
            ),
            chart_action="showOverview",
            action_params={},
        ),
        ScrollyStep(
            step_number=2,
            narrative_text=(
                f"Good news: {high_pct:.0f}% of your segments are stable. "
                f"These {high_count} segments will have the same customers next week, "
                f"next month. Build your automation around these."
            ),
            chart_action="highlightTier",
            action_params={"tier": "HIGH"},
        ),
        ScrollyStep(
            step_number=3,
            narrative_text=(
                f"Watch out for '{most_critical}'. When this changes for a customer, "
                f"they often move to a different segment. If you see big seasonal "
                f"swings here, expect segment sizes to shift too."
            ),
            chart_action="highlightFeature",
            action_params={"feature": data.most_critical_feature},
        ),
        ScrollyStep(
            step_number=4,
            narrative_text=(
                f"Overall stability is {avg_stability:.0%}. That's decent - most "
                f"customers stay in their segments. But check segment sizes monthly "
                f"to catch any major shifts early."
            ),
            chart_action="showStability",
            action_params={},
        ),
    ]

    # Build content layers
    executive_summary = (
        f"{high_pct:.0f}% of segments are highly robust. "
        f"'{most_critical}' is the most critical segmentation feature. "
        f"Average stability: {avg_stability:.0%}."
    )

    marketing_narrative = f"""
**Can You Trust Your Segments?**

Think of it like this: if you build a campaign for "High-Value Regulars" but those
customers keep jumping in and out of that segment, your targeting is unreliable.

**The Traffic Light System**

{_format_tier_distribution(data)}

**🟢 HIGH (Green Light)**

Use freely. These segments are stable - the same customers will be in them tomorrow.
Perfect for: Automated campaigns, loyalty programs, always-on ads.

**🟡 MEDIUM (Yellow Light)**

Use with awareness. Segment membership might shift 10-20% month over month.
Perfect for: Monthly campaigns, promotional bursts, seasonal offers.

**🔴 LOW (Red Light)**

Use carefully. Customer membership is volatile.
Best for: One-time campaigns, test-and-learn, manual review before each use.

**The Stability Trigger**

These features have the most influence on segment boundaries:

{_format_critical_features(data)}

When these features change (e.g., seasonal effects on purchase frequency), expect
corresponding shifts in segment composition.

**Recommendations**

1. Base primary campaigns on HIGH robustness segments only
2. For MEDIUM segments, use broader targeting to account for boundary uncertainty
3. Re-run segmentation quarterly for LOW robustness segments
4. Monitor '{most_critical}' as an early indicator of segment drift
"""

    technical_details = f"""
**Methodology**

Sensitivity analysis evaluates segment robustness through:

1. **Feature Perturbation**: Add noise to each feature and measure segment changes
2. **Time Window Testing**: Run segmentation on different time windows and compare
3. **Bootstrap Sampling**: Resample customers and measure consistency

**Robustness Scoring**

Segments are assigned to robustness tiers based on:

| Metric | HIGH | MEDIUM | LOW |
|--------|------|--------|-----|
| ARI Score | ≥0.8 | 0.6-0.8 | <0.6 |
| NMI Score | ≥0.75 | 0.5-0.75 | <0.5 |
| Feature Stability | ≥85% | 70-85% | <70% |

**ARI (Adjusted Rand Index)**: Measures similarity between original and perturbed
segment assignments, adjusted for chance.

**NMI (Normalized Mutual Information)**: Information-theoretic measure of
clustering agreement.

**Segment Stability Details**

{_format_stability_table(data)}

**Time Consistency Details**

{_format_consistency_table(data)}

**Feature Impact Rankings**

{_format_feature_impact_table(data)}
"""

    # Key metrics for this section
    key_metrics = {
        "total_segments": total_segments,
        "high_robustness_pct": high_pct,
        "high_robustness_count": high_count,
        "avg_feature_stability": avg_stability,
        "most_critical_feature": data.most_critical_feature,
    }

    return EssaySection(
        section_id="segment-robustness",
        title="How Robust Are Your Segments",
        chart=gauge_chart,
        narrative=gauge_narrative,
        executive_summary=executive_summary,
        marketing_narrative=marketing_narrative,
        technical_details=technical_details,
        scrolly_steps=scrolly_steps,
        supporting_charts=[stability_chart, consistency_chart],
        key_metrics=key_metrics,
        appendix_data={
            "robustness_tiers": data.robustness_tiers,
            "feature_stability": data.feature_stability,
            "critical_features": data.critical_features,
            "time_consistency": data.time_consistency,
            "stability_metrics": data.stability_metrics,
        },
    )


# =============================================================================
# CHART SPECS
# =============================================================================


def create_robustness_gauge_spec(chart_id: str, data: SensitivityData) -> ChartSpec:
    """Create gauge/donut chart for robustness tier distribution."""
    tier_data = [
        {"tier": tier, "count": count, "color": _tier_color(tier)}
        for tier, count in data.robustness_tiers.items()
    ]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="donut",
        data=tier_data,
        config={
            "value": "count",
            "label": "tier",
            "colors": [d["color"] for d in tier_data],
            "title": "Segment Robustness Distribution",
            "centerText": f"{data.high_robustness_percentage:.0f}%",
            "centerSubtext": "High Robustness",
        },
    )


def create_feature_stability_spec(chart_id: str, data: SensitivityData) -> ChartSpec:
    """Create horizontal bar chart for feature stability rankings."""
    stability_data = [
        {
            "feature": item["feature"].replace("_", " ").title(),
            "stability": item["avg_stability"] * 100,
            "segment_count": item["segment_count"],
        }
        for item in data.feature_stability[:10]
    ]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="horizontal_bar",
        data=stability_data,
        config={
            "x": "stability",
            "y": "feature",
            "colorScale": ["#e74c3c", "#f1c40f", "#27ae60"],
            "colorThresholds": [60, 80],
            "title": "Feature Stability (%)",
            "xLabel": "Stability Score",
            "yLabel": "Feature",
            "xDomain": [0, 100],
        },
    )


def create_time_consistency_spec(chart_id: str, data: SensitivityData) -> ChartSpec:
    """Create grouped bar chart for time window consistency."""
    consistency_data = [
        {
            "segment": item["segment"],
            "consistency": item["consistency"] * 100,
            "tier": item["robustness"],
        }
        for item in data.time_consistency
    ]

    return ChartSpec(
        chart_id=chart_id,
        chart_type="bar",
        data=consistency_data,
        config={
            "x": "segment",
            "y": "consistency",
            "colorBy": "tier",
            "colors": {"HIGH": "#27ae60", "MEDIUM": "#f1c40f", "LOW": "#e74c3c"},
            "title": "Time Window Consistency (%)",
            "xLabel": "Segment",
            "yLabel": "Consistency (%)",
            "yDomain": [0, 100],
            "rotateLabels": True,
        },
    )


def _tier_color(tier: str) -> str:
    """Get color for robustness tier."""
    colors = {
        "HIGH": "#27ae60",
        "MEDIUM": "#f1c40f",
        "LOW": "#e74c3c",
    }
    return colors.get(tier.upper(), "#95a5a6")


# =============================================================================
# FORMATTING HELPERS
# =============================================================================


def _format_tier_distribution(data: SensitivityData) -> str:
    """Format tier distribution as a bullet list."""
    lines = []
    tier_labels = {"HIGH": "🟢 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🔴 LOW"}
    for tier, count in data.robustness_tiers.items():
        label = tier_labels.get(tier.upper(), tier)
        pct = count / data.total_segments * 100 if data.total_segments > 0 else 0
        lines.append(f"- {label}: {count} segments ({pct:.0f}%)")
    return "\n".join(lines)


def _format_critical_features(data: SensitivityData) -> str:
    """Format critical features as a bullet list."""
    lines = []
    for item in data.critical_features[:5]:
        features = ", ".join(f.replace("_", " ").title() for f in item["features"][:3])
        lines.append(f"- **{item['segment']}**: {features}")
    return "\n".join(lines)


def _format_stability_table(data: SensitivityData) -> str:
    """Format stability metrics as a text table."""
    lines = ["| Segment | ARI | NMI | Tier |"]
    lines.append("|---------|-----|-----|------|")
    for item in data.stability_metrics:
        lines.append(
            f"| {item['segment'][:20]} | {item['ari']:.2f} | {item['nmi']:.2f} | {item['tier']} |"
        )
    return "\n".join(lines)


def _format_consistency_table(data: SensitivityData) -> str:
    """Format time consistency as a text table."""
    lines = ["| Segment | Consistency | Robustness |"]
    lines.append("|---------|-------------|------------|")
    for item in data.time_consistency:
        lines.append(
            f"| {item['segment'][:20]} | {item['consistency']:.0%} | {item['robustness']} |"
        )
    return "\n".join(lines)


def _format_feature_impact_table(data: SensitivityData) -> str:
    """Format feature impact as a text table."""
    lines = ["| Feature | Avg Stability | # Segments |"]
    lines.append("|---------|---------------|------------|")
    for item in data.feature_stability[:10]:
        name = item["feature"].replace("_", " ").title()[:20]
        lines.append(
            f"| {name} | {item['avg_stability']:.0%} | {item['segment_count']} |"
        )
    return "\n".join(lines)
