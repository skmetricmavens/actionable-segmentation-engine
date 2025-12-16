/**
 * D3 Funnel chart implementation for conversion visualization.
 * Used in Essay 3: The Illusion of New Customers
 */

function renderFunnel(container, spec) {
    const { data, config } = spec;
    const { stages } = data;

    const width = config.width || 600;
    const height = config.height || 400;
    const margin = { top: 30, right: 100, bottom: 30, left: 30 };
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

    // Calculate funnel dimensions
    const stageHeight = innerHeight / stages.length;
    const maxWidth = innerWidth * 0.8;

    // Scale for funnel width based on value
    const maxValue = d3.max(stages, d => d.value);
    const widthScale = d3.scaleLinear()
        .domain([0, maxValue])
        .range([maxWidth * 0.3, maxWidth]);

    // Draw funnel stages
    const stageGroups = g.selectAll('.funnel-stage')
        .data(stages)
        .join('g')
        .attr('class', 'funnel-stage')
        .attr('transform', (d, i) => `translate(0, ${i * stageHeight})`);

    // Trapezoid paths
    stageGroups.append('path')
        .attr('d', (d, i) => {
            const currentWidth = widthScale(d.value);
            const nextWidth = i < stages.length - 1 ? widthScale(stages[i + 1].value) : currentWidth * 0.8;
            const centerX = innerWidth / 2;
            const h = stageHeight - 5;

            return `
                M ${centerX - currentWidth / 2} 0
                L ${centerX + currentWidth / 2} 0
                L ${centerX + nextWidth / 2} ${h}
                L ${centerX - nextWidth / 2} ${h}
                Z
            `;
        })
        .attr('fill', d => d.color)
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .attr('opacity', 0.9);

    // Stage labels (left side)
    stageGroups.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', stageHeight / 2)
        .attr('dy', '0.35em')
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-weight', '600')
        .attr('font-size', '14px')
        .text(d => d.label);

    // Values and percentages (right side)
    stageGroups.append('text')
        .attr('x', innerWidth + 10)
        .attr('y', stageHeight / 2 - 8)
        .attr('fill', '#4a5568')
        .attr('font-size', '14px')
        .attr('font-weight', '600')
        .text(d => d.value.toLocaleString());

    stageGroups.append('text')
        .attr('x', innerWidth + 10)
        .attr('y', stageHeight / 2 + 10)
        .attr('fill', '#718096')
        .attr('font-size', '12px')
        .text(d => `${d.percentage.toFixed(0)}%`);

    // Drop-off indicators
    if (config.showDropoffs) {
        stageGroups.filter((d, i) => i > 0 && d.drop_off_pct)
            .append('text')
            .attr('x', -10)
            .attr('y', 0)
            .attr('text-anchor', 'end')
            .attr('fill', config.dropoffColor || '#fc8181')
            .attr('font-size', '11px')
            .text(d => `↓ ${d.drop_off_pct.toFixed(0)}%`);
    }

    return {
        highlightStage: function(stageId) {
            stageGroups.selectAll('path')
                .attr('opacity', d => d.id === stageId ? 1 : 0.4);
        },
        highlightDropoff: function(fromId, toId) {
            stageGroups.selectAll('path')
                .attr('opacity', d => (d.id === fromId || d.id === toId) ? 1 : 0.4);
        },
        showAll: function() {
            stageGroups.selectAll('path').attr('opacity', 0.9);
        },
        animateIn: function(duration = 800) {
            stageGroups.selectAll('path')
                .attr('opacity', 0)
                .transition()
                .delay((d, i) => i * 200)
                .duration(duration)
                .attr('opacity', 0.9);
        }
    };
}

if (typeof window !== 'undefined') {
    window.renderFunnel = renderFunnel;
}
