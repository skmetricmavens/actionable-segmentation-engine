# Runbooks

Operational runbooks for managing and troubleshooting the Actionable Segmentation Engine.

## What are Runbooks?

Runbooks provide step-by-step procedures for common operational tasks. They help:

- **Standardize operations** - Consistent procedures across team members
- **Speed up troubleshooting** - Quick reference during incidents
- **Enable self-service** - Team members can resolve issues independently
- **Document tribal knowledge** - Capture expertise in written form

## Available Runbooks

<div class="grid cards" markdown>

-   :material-bug:{ .lg .middle } **Troubleshooting**

    ---

    Common issues and their solutions

    [:octicons-arrow-right-24: Troubleshooting Guide](troubleshooting.md)

-   :material-speedometer:{ .lg .middle } **Performance Tuning**

    ---

    Optimize pipeline performance for your workload

    [:octicons-arrow-right-24: Performance Guide](performance.md)

</div>

## Quick Reference

### Pipeline Not Producing Results

1. Check input data format
2. Verify minimum customer count (need at least `k * 10` customers)
3. Review validation criteria thresholds

### Segments All Rejected

1. Lower `min_overall_robustness` threshold
2. Increase `n_customers` for more data
3. Check if data has enough variance for clustering

### LLM Integration Failing

1. Verify `ANTHROPIC_API_KEY` is set
2. Check API rate limits
3. Fall back to mock LLM (`use_llm=False`)

## On-Call Checklist

For on-call engineers handling segmentation issues:

- [ ] Check pipeline logs for errors
- [ ] Verify input data quality
- [ ] Review recent configuration changes
- [ ] Check external service status (Claude API)
- [ ] Escalate if issue persists > 30 minutes
