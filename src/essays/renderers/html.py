"""
HTML renderer for visual essays using Jinja2 templates.

Renders Essay objects to complete HTML documents with:
- Scrollytelling structure
- Embedded chart data as JSON
- D3 and Scrollama script inclusion
- Responsive CSS
"""

import json
import logging
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

from src.essays.base import Essay, EssayConfig, EssaySection

logger = logging.getLogger(__name__)


# =============================================================================
# TEMPLATE LOADING
# =============================================================================


def get_template_env(template_dir: Path | None = None) -> Environment:
    """Get Jinja2 environment with templates.

    Args:
        template_dir: Optional custom template directory

    Returns:
        Jinja2 Environment configured for essay templates
    """
    if template_dir and template_dir.exists():
        loader = FileSystemLoader(str(template_dir))
    else:
        # Use inline templates for portability
        loader = FileSystemLoader(
            str(Path(__file__).parent / "templates"),
            encoding="utf-8",
        )

    env = Environment(
        loader=loader,
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    # Add custom filters
    env.filters["tojson"] = lambda x: json.dumps(x, default=str, indent=2)
    env.filters["format_number"] = _format_number
    env.filters["format_currency"] = _format_currency
    env.filters["format_percent"] = _format_percent

    return env


def _format_number(value: float | int, decimals: int = 0) -> str:
    """Format number with thousand separators."""
    if isinstance(value, int) or decimals == 0:
        return f"{int(value):,}"
    return f"{value:,.{decimals}f}"


def _format_currency(value: float | int, decimals: int = 0) -> str:
    """Format as currency."""
    return f"${_format_number(value, decimals)}"


def _format_percent(value: float, decimals: int = 1) -> str:
    """Format as percentage."""
    return f"{value * 100:.{decimals}f}%"


# =============================================================================
# HTML RENDERING
# =============================================================================


def render_essay_html(
    essay: Essay,
    config: EssayConfig | None = None,
    *,
    embed_assets: bool = False,
    cdn_base: str = "https://cdn.jsdelivr.net/npm",
) -> str:
    """Render an essay to complete HTML.

    Args:
        essay: The Essay object to render
        config: Optional essay configuration
        embed_assets: If True, embed JS/CSS inline; if False, use CDN links
        cdn_base: Base URL for CDN assets

    Returns:
        Complete HTML document as string
    """
    config = config or EssayConfig()

    # Build template context
    context = {
        "essay": essay,
        "config": config,
        "sections": essay.sections,
        "key_insights": essay.key_insights,
        "toc": essay.toc_entries,
        "embed_assets": embed_assets,
        "cdn_base": cdn_base,
        # Asset URLs
        "d3_url": f"{cdn_base}/d3@7/dist/d3.min.js",
        "scrollama_url": f"{cdn_base}/scrollama@3/build/scrollama.min.js",
        # Chart data (JSON-serialized)
        "chart_data": _extract_chart_data(essay),
        # Embedded assets
        "essay_css": _get_essay_css(),
        "essay_js": _get_essay_js(),
    }

    # Get template (use inline template for now)
    template_html = _get_main_template()

    # Render
    env = get_template_env()
    template = env.from_string(template_html)

    return template.render(**context)


def _extract_chart_data(essay: Essay) -> dict[str, Any]:
    """Extract all chart data from essay sections.

    Returns a dictionary mapping chart_id to chart spec data.
    """
    chart_data = {}

    for section in essay.sections:
        # Primary chart
        chart_data[section.chart.chart_id] = section.chart.to_dict()

        # Supporting charts
        for chart in section.supporting_charts:
            chart_data[chart.chart_id] = chart.to_dict()

    return chart_data


def render_section_html(section: EssaySection, audience: str = "all") -> str:
    """Render a single section to HTML.

    Args:
        section: The section to render
        audience: Target audience layer (executive, marketing, technical, all)

    Returns:
        Section HTML fragment
    """
    env = get_template_env()
    template = env.from_string(_get_section_template())

    return template.render(
        section=section,
        audience=audience,
        chart_data=section.chart.to_dict(),
        narrative=section.narrative.to_dict(),
    )


# =============================================================================
# INLINE TEMPLATES
# =============================================================================


def _get_main_template() -> str:
    """Get the main essay HTML template."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ essay.title }}</title>

    <!-- Styles -->
    <style>
{{ essay_css | safe }}
    </style>
</head>
<body>
    <!-- Executive Summary Banner -->
    <header class="executive-banner">
        <div class="banner-content">
            <h1>{{ essay.title }}</h1>
            <p class="subtitle">{{ essay.subtitle }}</p>

            {% if key_insights %}
            <div class="key-insights">
                {% for insight in key_insights[:3] %}
                <div class="insight-card">
                    <span class="insight-value">{{ insight.metric_value }}</span>
                    <span class="insight-label">{{ insight.metric_label }}</span>
                </div>
                {% endfor %}
            </div>
            {% endif %}

            <div class="metadata">
                <span>{{ essay.customer_count | format_number }} customers analyzed</span>
                <span class="separator">|</span>
                <span>Generated {{ essay.generated_at.strftime('%B %d, %Y') }}</span>
            </div>
        </div>
    </header>

    <!-- Table of Contents -->
    <nav class="toc">
        <h2>Contents</h2>
        <ol>
            {% for entry in toc %}
            <li><a href="#{{ entry.id }}">{{ entry.title }}</a></li>
            {% endfor %}
        </ol>
    </nav>

    <!-- Main Content -->
    <main class="essay-content">
        {% for section in sections %}
        <article id="{{ section.section_id }}" class="essay-section">
            <div class="section-header">
                <h2>{{ section.title }}</h2>
                <p class="executive-summary">{{ section.executive_summary }}</p>
            </div>

            <!-- Scrollytelling Container -->
            <div class="scrolly-container">
                <div class="scrolly-graphic">
                    <div class="chart-container" id="chart-{{ section.chart.chart_id }}">
                        <!-- D3 chart renders here -->
                    </div>
                </div>

                <div class="scrolly-text">
                    <!-- Headline -->
                    <div class="step" data-step="0">
                        <div class="step-content headline">
                            <h3>{{ section.narrative.headline }}</h3>
                        </div>
                    </div>

                    <!-- Scrolly steps -->
                    {% for step in section.scrolly_steps %}
                    <div class="step" data-step="{{ step.step_number }}">
                        <div class="step-content">
                            <p>{{ step.narrative_text }}</p>
                        </div>
                    </div>
                    {% endfor %}

                    <!-- Insight & Callout -->
                    <div class="step" data-step="insight">
                        <div class="step-content insight">
                            <p>{{ section.narrative.insight }}</p>
                        </div>
                    </div>

                    <div class="step" data-step="callout">
                        <div class="step-content callout">
                            <p><strong>Action:</strong> {{ section.narrative.callout }}</p>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Marketing Narrative -->
            {% if config.audience in ['marketing', 'all'] and section.marketing_narrative %}
            <div class="marketing-narrative">
                {{ section.marketing_narrative | safe }}
            </div>
            {% endif %}

            <!-- Technical Appendix -->
            {% if config.include_appendix and section.technical_details %}
            <details class="technical-appendix">
                <summary>Technical Details & Methodology</summary>
                <div class="appendix-content">
                    {{ section.technical_details | safe }}

                    {% if section.key_metrics %}
                    <h4>Key Metrics</h4>
                    <table class="metrics-table">
                        {% for key, value in section.key_metrics.items() %}
                        <tr>
                            <td>{{ key }}</td>
                            <td>{{ value }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                    {% endif %}
                </div>
            </details>
            {% endif %}
        </article>
        {% endfor %}
    </main>

    <!-- Footer -->
    <footer class="essay-footer">
        <p>Generated by Actionable Segmentation Engine</p>
    </footer>

    <!-- Scripts -->
    <script src="{{ d3_url }}"></script>
    <script src="{{ scrollama_url }}"></script>

    <!-- Chart Data -->
    <script>
        window.ESSAY_DATA = {{ chart_data | tojson | safe }};
    </script>

    <!-- Essay JavaScript -->
    <script>
{{ essay_js | safe }}
    </script>
</body>
</html>'''


def _get_section_template() -> str:
    """Get the section HTML template."""
    return '''<article id="{{ section.section_id }}" class="essay-section">
    <div class="section-header">
        <h2>{{ section.title }}</h2>
        <p class="executive-summary">{{ section.executive_summary }}</p>
    </div>

    <div class="chart-container" id="chart-{{ section.chart.chart_id }}"></div>

    <div class="narrative">
        <h3>{{ narrative.headline }}</h3>
        <p>{{ narrative.insight }}</p>
        <p class="callout"><strong>Action:</strong> {{ narrative.callout }}</p>
    </div>
</article>'''


def _get_essay_css() -> str:
    """Get embedded CSS for essays."""
    return '''
/* Reset & Base */
*, *::before, *::after { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6;
    color: #1a202c;
    background: #f7fafc;
    margin: 0;
    padding: 0;
}

/* Executive Banner */
.executive-banner {
    background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
    color: white;
    padding: 4rem 2rem;
    text-align: center;
}
.executive-banner h1 {
    font-size: 2.5rem;
    margin: 0 0 0.5rem;
    font-weight: 700;
}
.executive-banner .subtitle {
    font-size: 1.25rem;
    opacity: 0.9;
    margin: 0 0 2rem;
}
.key-insights {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin: 2rem 0;
}
.insight-card {
    background: rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.5rem 2rem;
    min-width: 180px;
}
.insight-value {
    display: block;
    font-size: 2rem;
    font-weight: 700;
    color: #68d391;
}
.insight-label {
    display: block;
    font-size: 0.9rem;
    opacity: 0.8;
    margin-top: 0.25rem;
}
.metadata {
    font-size: 0.875rem;
    opacity: 0.7;
}
.metadata .separator { margin: 0 0.5rem; }

/* Table of Contents */
.toc {
    max-width: 800px;
    margin: 2rem auto;
    padding: 1.5rem 2rem;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
}
.toc h2 {
    font-size: 1.25rem;
    margin: 0 0 1rem;
    color: #4a5568;
}
.toc ol {
    margin: 0;
    padding-left: 1.5rem;
}
.toc li {
    margin: 0.5rem 0;
}
.toc a {
    color: #4299e1;
    text-decoration: none;
}
.toc a:hover { text-decoration: underline; }

/* Essay Content */
.essay-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}
.essay-section {
    margin: 4rem 0;
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    overflow: hidden;
}
.section-header {
    padding: 2rem;
    border-bottom: 1px solid #e2e8f0;
}
.section-header h2 {
    font-size: 1.75rem;
    margin: 0 0 0.5rem;
    color: #2d3748;
}
.executive-summary {
    font-size: 1.125rem;
    color: #4a5568;
    margin: 0;
}

/* Scrollytelling */
.scrolly-container {
    display: flex;
    position: relative;
}
.scrolly-graphic {
    flex: 0 0 60%;
    position: sticky;
    top: 0;
    height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f7fafc;
}
.chart-container {
    width: 100%;
    max-width: 700px;
    padding: 2rem;
}
.scrolly-text {
    flex: 0 0 40%;
    padding: 2rem;
}
.step {
    min-height: 80vh;
    display: flex;
    align-items: center;
    padding: 2rem 0;
}
.step-content {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.step-content.headline h3 {
    font-size: 1.5rem;
    margin: 0;
    color: #2d3748;
}
.step-content.insight {
    border-left: 4px solid #4299e1;
}
.step-content.callout {
    background: #ebf8ff;
    border-left: 4px solid #4299e1;
}
.step-content p { margin: 0; }

/* Marketing Narrative */
.marketing-narrative {
    padding: 2rem;
    background: #f7fafc;
}

/* Technical Appendix */
.technical-appendix {
    margin: 0;
    border-top: 1px solid #e2e8f0;
}
.technical-appendix summary {
    padding: 1rem 2rem;
    cursor: pointer;
    font-weight: 600;
    color: #4a5568;
    background: #f7fafc;
}
.technical-appendix summary:hover { background: #edf2f7; }
.appendix-content {
    padding: 2rem;
    font-size: 0.9rem;
    color: #4a5568;
}
.metrics-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 1rem;
}
.metrics-table td {
    padding: 0.5rem;
    border-bottom: 1px solid #e2e8f0;
}
.metrics-table td:first-child {
    font-weight: 600;
    color: #2d3748;
}

/* Footer */
.essay-footer {
    text-align: center;
    padding: 2rem;
    color: #718096;
    font-size: 0.875rem;
}

/* Responsive */
@media (max-width: 1024px) {
    .scrolly-container {
        flex-direction: column;
    }
    .scrolly-graphic {
        position: relative;
        height: auto;
        min-height: 400px;
    }
    .scrolly-text {
        padding: 1rem;
    }
    .step {
        min-height: auto;
        padding: 1rem 0;
    }
}
'''


def _get_essay_js() -> str:
    """Get embedded JavaScript for essays."""
    return '''
// Initialize Scrollama
document.addEventListener('DOMContentLoaded', function() {
    const scroller = scrollama();

    // Setup scrollama instance
    scroller
        .setup({
            step: '.step',
            offset: 0.5,
            progress: true,
        })
        .onStepEnter(response => {
            const step = response.element.dataset.step;
            const section = response.element.closest('.essay-section');
            const sectionId = section ? section.id : null;

            // Highlight active step
            document.querySelectorAll('.step').forEach(s => s.classList.remove('is-active'));
            response.element.classList.add('is-active');

            // Trigger chart animation if available
            if (sectionId && window.chartAnimations && window.chartAnimations[sectionId]) {
                window.chartAnimations[sectionId](step);
            }
        });

    // Resize handling
    window.addEventListener('resize', scroller.resize);

    // Initialize charts
    initCharts();
});

// Chart initialization - dispatch to appropriate renderer
function initCharts() {
    if (!window.ESSAY_DATA) return;

    Object.keys(window.ESSAY_DATA).forEach(chartId => {
        const container = document.getElementById('chart-' + chartId);
        if (!container) return;

        const spec = window.ESSAY_DATA[chartId];

        // Route to appropriate chart renderer
        switch (spec.chart_type) {
            case 'sankey':
                renderSankeyChart(container, spec);
                break;
            case 'scatter':
                renderScatterChart(container, spec);
                break;
            case 'funnel':
                renderFunnelChart(container, spec);
                break;
            case 'timeline':
                renderTimelineChart(container, spec);
                break;
            case 'radar':
                renderRadarChart(container, spec);
                break;
            case 'bar':
                renderBarChart(container, spec);
                break;
            case 'horizontal_bar':
                renderHorizontalBarChart(container, spec);
                break;
            case 'grouped_bar':
                renderGroupedBarChart(container, spec);
                break;
            case 'heatmap':
                renderHeatmapChart(container, spec);
                break;
            case 'donut':
                renderDonutChart(container, spec);
                break;
            case 'line':
                renderLineChart(container, spec);
                break;
            default:
                renderPlaceholderChart(container, spec);
        }
    });
}

// =============================================================================
// CHART RENDERERS
// =============================================================================

function renderBarChart(container, spec) {
    const margin = {top: 40, right: 30, bottom: 60, left: 60};
    const width = (spec.config?.width || 600) - margin.left - margin.right;
    const height = (spec.config?.height || 400) - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = spec.data || [];
    const xField = spec.config?.x || 'x';
    const yField = spec.config?.y || 'y';
    const color = spec.config?.color || '#4299e1';

    // X scale
    const x = d3.scaleBand()
        .domain(data.map(d => d[xField]))
        .range([0, width])
        .padding(0.2);

    // Y scale
    const yMax = d3.max(data, d => d[yField]) || 100;
    const y = d3.scaleLinear()
        .domain([0, yMax * 1.1])
        .range([height, 0]);

    // Bars
    svg.selectAll('rect')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', d => x(d[xField]))
        .attr('y', d => y(d[yField]))
        .attr('width', x.bandwidth())
        .attr('height', d => height - y(d[yField]))
        .attr('fill', color)
        .attr('rx', 4);

    // X axis
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x))
        .selectAll('text')
        .attr('transform', spec.config?.rotateLabels ? 'rotate(-45)' : '')
        .style('text-anchor', spec.config?.rotateLabels ? 'end' : 'middle');

    // Y axis
    svg.append('g').call(d3.axisLeft(y));

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || '');
}

function renderHorizontalBarChart(container, spec) {
    const margin = {top: 40, right: 30, bottom: 40, left: 150};
    const width = (spec.config?.width || 600) - margin.left - margin.right;
    const height = (spec.config?.height || 400) - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = spec.data || [];
    const xField = spec.config?.x || 'x';
    const yField = spec.config?.y || 'y';

    // Y scale (categorical)
    const y = d3.scaleBand()
        .domain(data.map(d => d[yField]))
        .range([0, height])
        .padding(0.2);

    // X scale (numerical)
    const xDomain = spec.config?.xDomain || [0, d3.max(data, d => d[xField]) * 1.1];
    const x = d3.scaleLinear()
        .domain(xDomain)
        .range([0, width]);

    // Bars
    svg.selectAll('rect')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', 0)
        .attr('y', d => y(d[yField]))
        .attr('width', d => x(d[xField]))
        .attr('height', y.bandwidth())
        .attr('fill', d => {
            if (spec.config?.colorBy && spec.config?.colors) {
                return spec.config.colors[String(d[spec.config.colorBy])] || '#4299e1';
            }
            return '#4299e1';
        })
        .attr('rx', 4);

    // X axis
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));

    // Y axis
    svg.append('g').call(d3.axisLeft(y));

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || '');
}

function renderScatterChart(container, spec) {
    const margin = {top: 40, right: 30, bottom: 60, left: 60};
    const width = (spec.config?.width || 600) - margin.left - margin.right;
    const height = (spec.config?.height || 400) - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = spec.data?.points || spec.data || [];
    const xField = spec.config?.x || 'x';
    const yField = spec.config?.y || 'y';

    // X scale
    const xExtent = d3.extent(data, d => d[xField]);
    const x = d3.scaleLinear()
        .domain([xExtent[0] * 0.9, xExtent[1] * 1.1])
        .range([0, width]);

    // Y scale
    const yExtent = d3.extent(data, d => d[yField]);
    const y = d3.scaleLinear()
        .domain([0, yExtent[1] * 1.1])
        .range([height, 0]);

    // Color scale
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Points
    svg.selectAll('circle')
        .data(data)
        .enter()
        .append('circle')
        .attr('cx', d => x(d[xField]))
        .attr('cy', d => y(d[yField]))
        .attr('r', d => Math.sqrt(d.size || 50))
        .attr('fill', d => d.is_fragile ? '#e53e3e' : '#4299e1')
        .attr('opacity', 0.7);

    // Axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));

    svg.append('g').call(d3.axisLeft(y));

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || '');
}

function renderFunnelChart(container, spec) {
    const margin = {top: 40, right: 30, bottom: 40, left: 120};
    const width = (spec.config?.width || 600) - margin.left - margin.right;
    const height = (spec.config?.height || 400) - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const stages = spec.data?.stages || spec.data || [];
    const maxValue = d3.max(stages, d => d.count) || 100;

    const barHeight = height / stages.length - 10;
    const colorScale = d3.scaleSequential(d3.interpolateBlues)
        .domain([stages.length, 0]);

    stages.forEach((stage, i) => {
        const barWidth = (stage.count / maxValue) * width;
        const xOffset = (width - barWidth) / 2;

        svg.append('rect')
            .attr('x', xOffset)
            .attr('y', i * (barHeight + 10))
            .attr('width', barWidth)
            .attr('height', barHeight)
            .attr('fill', colorScale(i))
            .attr('rx', 4);

        svg.append('text')
            .attr('x', -10)
            .attr('y', i * (barHeight + 10) + barHeight / 2)
            .attr('text-anchor', 'end')
            .attr('dominant-baseline', 'middle')
            .attr('font-size', '12px')
            .text(stage.stage);

        svg.append('text')
            .attr('x', width / 2)
            .attr('y', i * (barHeight + 10) + barHeight / 2)
            .attr('text-anchor', 'middle')
            .attr('dominant-baseline', 'middle')
            .attr('fill', 'white')
            .attr('font-weight', 'bold')
            .text(`${stage.count.toLocaleString()} (${(stage.percentage * 100).toFixed(0)}%)`);
    });

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || 'Funnel');
}

function renderSankeyChart(container, spec) {
    const margin = {top: 40, right: 30, bottom: 40, left: 30};
    const width = (spec.config?.width || 700) - margin.left - margin.right;
    const height = (spec.config?.height || 500) - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const nodes = spec.data?.nodes || [];
    const links = spec.data?.links || [];

    if (nodes.length === 0) {
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', height / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#a0aec0')
            .text('No Sankey data available');
        return;
    }

    // Simple force-directed layout for Sankey-like effect
    const nodeMap = new Map(nodes.map((n, i) => [n.id, {...n, index: i}]));
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Position nodes in columns
    const nodesByType = d3.group(nodes, d => d.group || 0);
    const numCols = nodesByType.size || 1;
    const colWidth = width / numCols;

    let colIndex = 0;
    nodesByType.forEach((colNodes, group) => {
        const rowHeight = height / (colNodes.length + 1);
        colNodes.forEach((node, i) => {
            node.x = colIndex * colWidth + colWidth / 2;
            node.y = (i + 1) * rowHeight;
        });
        colIndex++;
    });

    // Draw links
    svg.selectAll('.link')
        .data(links)
        .enter()
        .append('path')
        .attr('class', 'link')
        .attr('d', d => {
            const source = nodeMap.get(d.source);
            const target = nodeMap.get(d.target);
            if (!source || !target) return '';
            return `M${source.x},${source.y} C${(source.x + target.x) / 2},${source.y} ${(source.x + target.x) / 2},${target.y} ${target.x},${target.y}`;
        })
        .attr('fill', 'none')
        .attr('stroke', '#cbd5e0')
        .attr('stroke-width', d => Math.max(2, Math.sqrt(d.value)))
        .attr('opacity', 0.5);

    // Draw nodes
    svg.selectAll('.node')
        .data(nodes)
        .enter()
        .append('circle')
        .attr('class', 'node')
        .attr('cx', d => d.x)
        .attr('cy', d => d.y)
        .attr('r', d => Math.sqrt(d.value || 100))
        .attr('fill', (d, i) => spec.config?.nodeColors?.[d.id] || colorScale(i));

    // Node labels
    svg.selectAll('.node-label')
        .data(nodes)
        .enter()
        .append('text')
        .attr('x', d => d.x)
        .attr('y', d => d.y + Math.sqrt(d.value || 100) + 15)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .text(d => d.label || d.id);

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || 'Customer Journey');
}

function renderTimelineChart(container, spec) {
    const margin = {top: 40, right: 30, bottom: 60, left: 60};
    const width = (spec.config?.width || 600) - margin.left - margin.right;
    const height = (spec.config?.height || 400) - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const indicators = spec.data?.indicators || spec.data || [];

    // X scale (days before churn)
    const x = d3.scaleLinear()
        .domain([d3.max(indicators, d => d.avg_days_before_churn) || 90, 0])
        .range([0, width]);

    // Y scale (importance)
    const y = d3.scaleLinear()
        .domain([0, 1])
        .range([height, 0]);

    // Line
    const line = d3.line()
        .x(d => x(d.avg_days_before_churn))
        .y(d => y(d.importance))
        .curve(d3.curveMonotoneX);

    svg.append('path')
        .datum(indicators.sort((a, b) => b.avg_days_before_churn - a.avg_days_before_churn))
        .attr('fill', 'none')
        .attr('stroke', '#e53e3e')
        .attr('stroke-width', 2)
        .attr('d', line);

    // Points
    svg.selectAll('circle')
        .data(indicators)
        .enter()
        .append('circle')
        .attr('cx', d => x(d.avg_days_before_churn))
        .attr('cy', d => y(d.importance))
        .attr('r', 6)
        .attr('fill', '#e53e3e');

    // Axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x).tickFormat(d => d + ' days'));

    svg.append('g').call(d3.axisLeft(y).tickFormat(d => (d * 100) + '%'));

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || 'Warning Signs Timeline');
}

function renderRadarChart(container, spec) {
    const width = spec.config?.width || 400;
    const height = spec.config?.height || 400;
    const radius = Math.min(width, height) / 2 - 40;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .append('g')
        .attr('transform', `translate(${width / 2},${height / 2})`);

    const axes = spec.data?.axes || [];
    const series = spec.data?.series || [];

    if (axes.length === 0) {
        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('fill', '#a0aec0')
            .text('No radar data');
        return;
    }

    const angleSlice = (Math.PI * 2) / axes.length;
    const rScale = d3.scaleLinear().domain([0, 1]).range([0, radius]);

    // Draw circular grid
    [0.25, 0.5, 0.75, 1].forEach(level => {
        svg.append('circle')
            .attr('r', rScale(level))
            .attr('fill', 'none')
            .attr('stroke', '#e2e8f0')
            .attr('stroke-dasharray', '3,3');
    });

    // Draw axes
    axes.forEach((axis, i) => {
        const angle = angleSlice * i - Math.PI / 2;
        svg.append('line')
            .attr('x1', 0)
            .attr('y1', 0)
            .attr('x2', rScale(1) * Math.cos(angle))
            .attr('y2', rScale(1) * Math.sin(angle))
            .attr('stroke', '#cbd5e0');

        svg.append('text')
            .attr('x', (rScale(1) + 15) * Math.cos(angle))
            .attr('y', (rScale(1) + 15) * Math.sin(angle))
            .attr('text-anchor', 'middle')
            .attr('font-size', '11px')
            .text(axis);
    });

    // Draw series
    const colors = ['#4299e1', '#e53e3e'];
    series.forEach((s, si) => {
        const points = s.values.map((v, i) => {
            const angle = angleSlice * i - Math.PI / 2;
            return [rScale(v) * Math.cos(angle), rScale(v) * Math.sin(angle)];
        });

        svg.append('polygon')
            .attr('points', points.map(p => p.join(',')).join(' '))
            .attr('fill', colors[si])
            .attr('fill-opacity', 0.3)
            .attr('stroke', colors[si])
            .attr('stroke-width', 2);
    });

    // Title
    svg.append('text')
        .attr('y', -height / 2 + 20)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || '');
}

function renderHeatmapChart(container, spec) {
    const margin = {top: 40, right: 30, bottom: 60, left: 120};
    const width = (spec.config?.width || 600) - margin.left - margin.right;
    const height = (spec.config?.height || 400) - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = spec.data || [];
    const xField = spec.config?.x || 'dimension';
    const yField = spec.config?.y || 'trait';
    const valueField = spec.config?.value || 'value';

    const xValues = [...new Set(data.map(d => d[xField]))];
    const yValues = [...new Set(data.map(d => d[yField]))];

    const x = d3.scaleBand().domain(xValues).range([0, width]).padding(0.05);
    const y = d3.scaleBand().domain(yValues).range([0, height]).padding(0.05);

    const colorScale = d3.scaleSequential(d3.interpolateYlOrRd)
        .domain([0, 1]);

    // Cells
    svg.selectAll('rect')
        .data(data)
        .enter()
        .append('rect')
        .attr('x', d => x(d[xField]))
        .attr('y', d => y(d[yField]))
        .attr('width', x.bandwidth())
        .attr('height', y.bandwidth())
        .attr('fill', d => colorScale(d[valueField]))
        .attr('rx', 4);

    // Cell values
    svg.selectAll('.cell-text')
        .data(data)
        .enter()
        .append('text')
        .attr('x', d => x(d[xField]) + x.bandwidth() / 2)
        .attr('y', d => y(d[yField]) + y.bandwidth() / 2)
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'middle')
        .attr('fill', d => d[valueField] > 0.5 ? 'white' : '#1a202c')
        .attr('font-size', '12px')
        .text(d => (d[valueField] * 100).toFixed(0) + '%');

    // Axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));

    svg.append('g').call(d3.axisLeft(y));

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || '');
}

function renderDonutChart(container, spec) {
    const width = spec.config?.width || 400;
    const height = spec.config?.height || 400;
    const radius = Math.min(width, height) / 2 - 20;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`)
        .append('g')
        .attr('transform', `translate(${width / 2},${height / 2})`);

    const data = spec.data || [];
    const valueField = spec.config?.value || 'count';
    const labelField = spec.config?.label || 'tier';

    const pie = d3.pie().value(d => d[valueField]).sort(null);
    const arc = d3.arc().innerRadius(radius * 0.6).outerRadius(radius);

    const colors = spec.config?.colors || d3.schemeCategory10;

    // Arcs
    svg.selectAll('path')
        .data(pie(data))
        .enter()
        .append('path')
        .attr('d', arc)
        .attr('fill', (d, i) => d.data.color || colors[i % colors.length])
        .attr('stroke', 'white')
        .attr('stroke-width', 2);

    // Labels
    svg.selectAll('.label')
        .data(pie(data))
        .enter()
        .append('text')
        .attr('transform', d => `translate(${arc.centroid(d)})`)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', 'white')
        .attr('font-weight', 'bold')
        .text(d => d.data[labelField]);

    // Center text
    if (spec.config?.centerText) {
        svg.append('text')
            .attr('text-anchor', 'middle')
            .attr('font-size', '2rem')
            .attr('font-weight', 'bold')
            .text(spec.config.centerText);

        if (spec.config?.centerSubtext) {
            svg.append('text')
                .attr('y', 25)
                .attr('text-anchor', 'middle')
                .attr('font-size', '0.9rem')
                .attr('fill', '#718096')
                .text(spec.config.centerSubtext);
        }
    }

    // Title
    svg.append('text')
        .attr('y', -height / 2 + 20)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || '');
}

function renderGroupedBarChart(container, spec) {
    const margin = {top: 40, right: 100, bottom: 60, left: 60};
    const width = (spec.config?.width || 600) - margin.left - margin.right;
    const height = (spec.config?.height || 400) - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = spec.data || [];
    const xField = spec.config?.x || 'metric';
    const groups = spec.config?.groups || ['Buyers', 'Lookalikes'];
    const colors = spec.config?.colors || ['#4299e1', '#e53e3e'];

    // X scale
    const x0 = d3.scaleBand()
        .domain(data.map(d => d[xField]))
        .range([0, width])
        .padding(0.2);

    const x1 = d3.scaleBand()
        .domain(groups)
        .range([0, x0.bandwidth()])
        .padding(0.05);

    // Y scale
    const yMax = d3.max(data, d => d3.max(groups, g => d[g])) || 10;
    const y = d3.scaleLinear()
        .domain([0, yMax * 1.1])
        .range([height, 0]);

    // Bars
    data.forEach(d => {
        groups.forEach((group, gi) => {
            svg.append('rect')
                .attr('x', x0(d[xField]) + x1(group))
                .attr('y', y(d[group] || 0))
                .attr('width', x1.bandwidth())
                .attr('height', height - y(d[group] || 0))
                .attr('fill', colors[gi])
                .attr('rx', 2);
        });
    });

    // Axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x0));

    svg.append('g').call(d3.axisLeft(y));

    // Legend
    groups.forEach((group, i) => {
        svg.append('rect')
            .attr('x', width + 10)
            .attr('y', i * 20)
            .attr('width', 15)
            .attr('height', 15)
            .attr('fill', colors[i]);

        svg.append('text')
            .attr('x', width + 30)
            .attr('y', i * 20 + 12)
            .attr('font-size', '12px')
            .text(group);
    });

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || '');
}

function renderLineChart(container, spec) {
    const margin = {top: 40, right: 30, bottom: 60, left: 60};
    const width = (spec.config?.width || 600) - margin.left - margin.right;
    const height = (spec.config?.height || 400) - margin.top - margin.bottom;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height + margin.top + margin.bottom)
        .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
        .append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const data = spec.data || [];
    const xField = spec.config?.x || 'x';
    const yField = spec.config?.y || 'y';

    const x = d3.scaleLinear()
        .domain(d3.extent(data, d => d[xField]))
        .range([0, width]);

    const y = d3.scaleLinear()
        .domain([0, d3.max(data, d => d[yField]) * 1.1])
        .range([height, 0]);

    const line = d3.line()
        .x(d => x(d[xField]))
        .y(d => y(d[yField]))
        .curve(d3.curveMonotoneX);

    svg.append('path')
        .datum(data)
        .attr('fill', 'none')
        .attr('stroke', spec.config?.color || '#4299e1')
        .attr('stroke-width', 2)
        .attr('d', line);

    // Axes
    svg.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(x));

    svg.append('g').call(d3.axisLeft(y));

    // Title
    svg.append('text')
        .attr('x', width / 2)
        .attr('y', -10)
        .attr('text-anchor', 'middle')
        .attr('font-weight', 'bold')
        .text(spec.config?.title || '');
}

function renderPlaceholderChart(container, spec) {
    const width = spec.config?.width || 600;
    const height = spec.config?.height || 400;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    svg.append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#a0aec0')
        .attr('font-size', '1.5rem')
        .text(`${spec.chart_type} chart: ${spec.chart_id}`);

    svg.append('rect')
        .attr('x', 10)
        .attr('y', 10)
        .attr('width', width - 20)
        .attr('height', height - 20)
        .attr('fill', 'none')
        .attr('stroke', '#e2e8f0')
        .attr('stroke-width', 2)
        .attr('rx', 8);
}

// Store for chart animations
window.chartAnimations = {};
'''


# =============================================================================
# FILE OUTPUT
# =============================================================================


def save_essay_html(html: str, output_path: Path | str) -> None:
    """Save rendered HTML to file.

    Args:
        html: The rendered HTML string
        output_path: Path to write HTML file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info(f"Saved essay to {output_path}")
