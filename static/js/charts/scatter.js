/**
 * D3 Scatter plot implementation for quadrant visualization.
 * Used in Essay 2: Not All Champions Are Safe
 */

function renderScatter(container, spec) {
    const { data, config } = spec;
    const { points } = data;

    const width = config.width || 700;
    const height = config.height || 500;
    const margin = { top: 40, right: 40, bottom: 60, left: 70 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    d3.select(container).selectAll('*').remove();

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
        .domain(config.xDomain || [0, d3.max(points, d => d.x) * 1.1])
        .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
        .domain(config.yDomain || [0, 1])
        .range([innerHeight, 0]);

    // Quadrant backgrounds
    if (config.quadrants) {
        const xThresh = xScale(config.quadrants.x_threshold);
        const yThresh = yScale(config.quadrants.y_threshold);
        const colors = config.quadrants.quadrant_colors;

        // Top-left: Low value, high risk
        g.append('rect')
            .attr('x', 0).attr('y', 0)
            .attr('width', xThresh).attr('height', yThresh)
            .attr('fill', colors.top_left);

        // Top-right: High value, high risk (danger)
        g.append('rect')
            .attr('x', xThresh).attr('y', 0)
            .attr('width', innerWidth - xThresh).attr('height', yThresh)
            .attr('fill', colors.top_right);

        // Bottom-left: Low value, stable
        g.append('rect')
            .attr('x', 0).attr('y', yThresh)
            .attr('width', xThresh).attr('height', innerHeight - yThresh)
            .attr('fill', colors.bottom_left);

        // Bottom-right: High value, stable (ideal)
        g.append('rect')
            .attr('x', xThresh).attr('y', yThresh)
            .attr('width', innerWidth - xThresh).attr('height', innerHeight - yThresh)
            .attr('fill', colors.bottom_right);

        // Quadrant labels
        const labels = config.quadrants.quadrant_labels;
        g.append('text').attr('x', xThresh / 2).attr('y', 20).attr('text-anchor', 'middle')
            .attr('fill', '#718096').attr('font-size', '11px').text(labels.top_left);
        g.append('text').attr('x', xThresh + (innerWidth - xThresh) / 2).attr('y', 20)
            .attr('text-anchor', 'middle').attr('fill', '#e53e3e').attr('font-size', '11px')
            .attr('font-weight', 'bold').text(labels.top_right);
    }

    // Axes
    g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(6).tickFormat(d => `$${d3.format(',.0f')(d)}`))
        .selectAll('text').attr('fill', '#4a5568');

    g.append('g')
        .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.0%')))
        .selectAll('text').attr('fill', '#4a5568');

    // Axis labels
    g.append('text')
        .attr('x', innerWidth / 2).attr('y', innerHeight + 45)
        .attr('text-anchor', 'middle').attr('fill', '#4a5568')
        .text(config.xLabel || 'Revenue');

    g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2).attr('y', -50)
        .attr('text-anchor', 'middle').attr('fill', '#4a5568')
        .text(config.yLabel || 'Fragility Score');

    // Points
    const dots = g.selectAll('.dot')
        .data(points)
        .join('circle')
        .attr('class', 'dot')
        .attr('cx', d => xScale(d.x))
        .attr('cy', d => yScale(d.y))
        .attr('r', config.pointRadius || 6)
        .attr('fill', d => d.color || (d.is_fragile ? '#fc8181' : '#68d391'))
        .attr('opacity', config.pointOpacity || 0.7)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('r', (config.pointRadius || 6) * 1.5);
            showTooltip(event, `Revenue: $${d.x.toLocaleString()}<br>Fragility: ${(d.y * 100).toFixed(0)}%`);
        })
        .on('mouseout', function() {
            d3.select(this).attr('r', config.pointRadius || 6);
            hideTooltip();
        });

    return {
        highlightFragile: function() {
            dots.attr('opacity', d => d.is_fragile ? 1 : 0.2);
        },
        highlightQuadrant: function(quadrant) {
            // Highlight specific quadrant
            dots.attr('opacity', d => {
                const inHighRisk = d.y > (config.quadrants?.y_threshold || 0.5);
                const inHighValue = d.x > (config.quadrants?.x_threshold || 500);
                if (quadrant === 'top_right') return (inHighRisk && inHighValue) ? 1 : 0.2;
                return 0.7;
            });
        },
        showAll: function() {
            dots.attr('opacity', config.pointOpacity || 0.7);
        }
    };
}

if (typeof window !== 'undefined') {
    window.renderScatter = renderScatter;
}
