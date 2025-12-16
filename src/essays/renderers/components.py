"""
Reusable HTML/CSS/JS components for essay rendering.

Provides template snippets and utility functions for building essay HTML.
"""

from typing import Any


# =============================================================================
# HTML COMPONENTS
# =============================================================================


def insight_card_html(value: str, label: str, color: str = "#68d391") -> str:
    """Generate HTML for an insight card.

    Args:
        value: The metric value (e.g., "23%")
        label: The metric label (e.g., "of Champions are fragile")
        color: Accent color for the value

    Returns:
        HTML string for the insight card
    """
    return f'''<div class="insight-card">
    <span class="insight-value" style="color: {color}">{value}</span>
    <span class="insight-label">{label}</span>
</div>'''


def metric_row_html(label: str, value: Any, format_type: str = "text") -> str:
    """Generate HTML for a metric row in a table.

    Args:
        label: Metric label
        value: Metric value
        format_type: How to format the value (text, number, currency, percent)

    Returns:
        HTML table row
    """
    formatted = _format_value(value, format_type)
    return f'''<tr>
    <td class="metric-label">{label}</td>
    <td class="metric-value">{formatted}</td>
</tr>'''


def _format_value(value: Any, format_type: str) -> str:
    """Format a value based on type."""
    if format_type == "number":
        return f"{int(value):,}" if isinstance(value, (int, float)) else str(value)
    elif format_type == "currency":
        return f"${float(value):,.0f}" if isinstance(value, (int, float)) else str(value)
    elif format_type == "percent":
        return f"{float(value) * 100:.1f}%" if isinstance(value, (int, float)) else str(value)
    return str(value)


def callout_box_html(text: str, style: str = "info") -> str:
    """Generate HTML for a callout box.

    Args:
        text: Callout text content
        style: Visual style (info, warning, success, action)

    Returns:
        HTML for callout box
    """
    colors = {
        "info": "#4299e1",
        "warning": "#f6ad55",
        "success": "#68d391",
        "action": "#805ad5",
    }
    color = colors.get(style, colors["info"])

    return f'''<div class="callout-box" style="border-left-color: {color}">
    <p>{text}</p>
</div>'''


def chart_placeholder_html(chart_id: str, chart_type: str, width: int = 600, height: int = 400) -> str:
    """Generate HTML placeholder for a chart.

    Args:
        chart_id: Unique chart identifier
        chart_type: Type of chart
        width: Chart width
        height: Chart height

    Returns:
        HTML for chart container
    """
    return f'''<div class="chart-container" id="chart-{chart_id}" data-chart-type="{chart_type}"
    style="width: 100%; max-width: {width}px; height: {height}px;">
    <div class="chart-loading">Loading {chart_type} chart...</div>
</div>'''


def progress_indicator_html(current: int, total: int, label: str = "") -> str:
    """Generate HTML for a progress indicator.

    Args:
        current: Current value
        total: Total value
        label: Optional label

    Returns:
        HTML for progress indicator
    """
    percentage = (current / total * 100) if total > 0 else 0
    return f'''<div class="progress-indicator">
    <div class="progress-bar">
        <div class="progress-fill" style="width: {percentage:.1f}%"></div>
    </div>
    <div class="progress-label">
        <span>{current:,} / {total:,}</span>
        {f'<span class="progress-text">{label}</span>' if label else ''}
    </div>
</div>'''


# =============================================================================
# CSS COMPONENTS
# =============================================================================


def get_callout_css() -> str:
    """Get CSS for callout boxes."""
    return '''
.callout-box {
    background: #f7fafc;
    border-left: 4px solid #4299e1;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    border-radius: 0 8px 8px 0;
}
.callout-box p { margin: 0; }
'''


def get_progress_css() -> str:
    """Get CSS for progress indicators."""
    return '''
.progress-indicator {
    margin: 1rem 0;
}
.progress-bar {
    height: 8px;
    background: #e2e8f0;
    border-radius: 4px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #4299e1, #68d391);
    border-radius: 4px;
    transition: width 0.5s ease;
}
.progress-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
    color: #718096;
    margin-top: 0.5rem;
}
'''


def get_chart_loading_css() -> str:
    """Get CSS for chart loading states."""
    return '''
.chart-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100%;
    color: #a0aec0;
    font-size: 1rem;
}
.chart-loading::before {
    content: '';
    width: 24px;
    height: 24px;
    border: 3px solid #e2e8f0;
    border-top-color: #4299e1;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-right: 0.75rem;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
'''


# =============================================================================
# JS COMPONENTS
# =============================================================================


def get_scroll_progress_js() -> str:
    """Get JavaScript for scroll progress indicator."""
    return '''
function initScrollProgress() {
    const progress = document.createElement('div');
    progress.className = 'scroll-progress';
    progress.innerHTML = '<div class="scroll-progress-bar"></div>';
    document.body.appendChild(progress);

    window.addEventListener('scroll', () => {
        const scrolled = (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100;
        document.querySelector('.scroll-progress-bar').style.width = scrolled + '%';
    });
}

// CSS for scroll progress
const style = document.createElement('style');
style.textContent = `
.scroll-progress {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    z-index: 1000;
}
.scroll-progress-bar {
    height: 100%;
    background: linear-gradient(90deg, #4299e1, #68d391);
    width: 0;
    transition: width 0.1s ease;
}
`;
document.head.appendChild(style);

document.addEventListener('DOMContentLoaded', initScrollProgress);
'''


def get_tooltip_js() -> str:
    """Get JavaScript for chart tooltips."""
    return '''
function createTooltip() {
    const tooltip = d3.select('body')
        .append('div')
        .attr('class', 'chart-tooltip')
        .style('opacity', 0);

    return {
        show: function(content, event) {
            tooltip
                .html(content)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px')
                .transition()
                .duration(200)
                .style('opacity', 1);
        },
        hide: function() {
            tooltip
                .transition()
                .duration(200)
                .style('opacity', 0);
        }
    };
}

// Tooltip CSS
const tooltipStyle = document.createElement('style');
tooltipStyle.textContent = `
.chart-tooltip {
    position: absolute;
    background: #1a202c;
    color: white;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    font-size: 0.875rem;
    pointer-events: none;
    z-index: 100;
    box-shadow: 0 4px 6px rgba(0,0,0,0.2);
}
.chart-tooltip::after {
    content: '';
    position: absolute;
    left: -6px;
    top: 50%;
    transform: translateY(-50%);
    border: 6px solid transparent;
    border-right-color: #1a202c;
}
`;
document.head.appendChild(tooltipStyle);
'''


def get_animation_helpers_js() -> str:
    """Get JavaScript animation helper functions."""
    return '''
const Animations = {
    // Fade in element
    fadeIn: function(selection, duration = 500) {
        return selection
            .style('opacity', 0)
            .transition()
            .duration(duration)
            .style('opacity', 1);
    },

    // Draw line path
    drawLine: function(path, duration = 1000) {
        const length = path.node().getTotalLength();
        return path
            .attr('stroke-dasharray', length)
            .attr('stroke-dashoffset', length)
            .transition()
            .duration(duration)
            .ease(d3.easeLinear)
            .attr('stroke-dashoffset', 0);
    },

    // Grow bar from zero
    growBar: function(selection, targetHeight, duration = 500) {
        return selection
            .attr('height', 0)
            .transition()
            .duration(duration)
            .attr('height', targetHeight);
    },

    // Stagger animation for multiple elements
    stagger: function(selection, callback, delay = 100) {
        selection.each(function(d, i) {
            d3.select(this)
                .transition()
                .delay(i * delay)
                .call(callback);
        });
    },

    // Count up number
    countUp: function(element, targetValue, duration = 2000, format = d => d.toLocaleString()) {
        const obj = { value: 0 };
        d3.select(obj)
            .transition()
            .duration(duration)
            .tween('text', function() {
                const i = d3.interpolateNumber(0, targetValue);
                return function(t) {
                    element.textContent = format(Math.round(i(t)));
                };
            });
    }
};
'''


# =============================================================================
# TEMPLATE FRAGMENTS
# =============================================================================


def get_header_fragment() -> str:
    """Get HTML fragment for essay header."""
    return '''
<header class="essay-header">
    <div class="header-content">
        <h1>{{ title }}</h1>
        {% if subtitle %}<p class="subtitle">{{ subtitle }}</p>{% endif %}
    </div>
</header>
'''


def get_footer_fragment() -> str:
    """Get HTML fragment for essay footer."""
    return '''
<footer class="essay-footer">
    <div class="footer-content">
        <p>Generated by Actionable Segmentation Engine</p>
        <p class="timestamp">{{ generated_at.strftime('%Y-%m-%d %H:%M') }}</p>
    </div>
</footer>
'''


def get_toc_fragment() -> str:
    """Get HTML fragment for table of contents."""
    return '''
<nav class="toc">
    <h2>In This Essay</h2>
    <ol>
        {% for entry in toc_entries %}
        <li>
            <a href="#{{ entry.id }}">
                <span class="toc-number">{{ loop.index }}</span>
                <span class="toc-title">{{ entry.title }}</span>
            </a>
        </li>
        {% endfor %}
    </ol>
</nav>
'''
