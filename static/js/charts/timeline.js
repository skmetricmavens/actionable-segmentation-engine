/**
 * D3 Timeline chart for leading indicators visualization.
 * Used in Essay 4: What Predicts Churn Before It Happens
 */

function renderTimeline(container, spec) {
    const { data, config } = spec;
    const { indicators, average_warning_days } = data;

    const width = config.width || 700;
    const height = config.height || 400;
    const margin = { top: 40, right: 40, bottom: 60, left: 60 };
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

    // X scale (days before churn - reversed)
    const xScale = d3.scaleLinear()
        .domain([90, 0])
        .range([0, innerWidth]);

    // Y scale (importance)
    const yScale = d3.scaleLinear()
        .domain([0, d3.max(indicators, d => d.importance) * 1.2])
        .range([innerHeight, 0]);

    // Critical zone background
    g.append('rect')
        .attr('x', xScale(14))
        .attr('y', 0)
        .attr('width', xScale(0) - xScale(14))
        .attr('height', innerHeight)
        .attr('fill', 'rgba(252, 129, 129, 0.15)');

    g.append('text')
        .attr('x', xScale(7))
        .attr('y', 15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fc8181')
        .attr('font-size', '11px')
        .text('Critical Window');

    // Average warning line
    g.append('line')
        .attr('x1', xScale(average_warning_days))
        .attr('x2', xScale(average_warning_days))
        .attr('y1', 0)
        .attr('y2', innerHeight)
        .attr('stroke', '#f6ad55')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5');

    g.append('text')
        .attr('x', xScale(average_warning_days) - 5)
        .attr('y', innerHeight - 10)
        .attr('text-anchor', 'end')
        .attr('fill', '#f6ad55')
        .attr('font-size', '11px')
        .text(`Avg: ${average_warning_days.toFixed(0)} days`);

    // Axes
    g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(6).tickFormat(d => `${d} days`))
        .selectAll('text').attr('fill', '#4a5568');

    g.append('g')
        .call(d3.axisLeft(yScale).ticks(5).tickFormat(d3.format('.0%')))
        .selectAll('text').attr('fill', '#4a5568');

    // Axis labels
    g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 45)
        .attr('text-anchor', 'middle')
        .attr('fill', '#4a5568')
        .text('Days Before Churn');

    g.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -innerHeight / 2)
        .attr('y', -45)
        .attr('text-anchor', 'middle')
        .attr('fill', '#4a5568')
        .text(config.importanceLabel || 'Signal Strength');

    // Indicator bubbles
    const bubbles = g.selectAll('.indicator')
        .data(indicators)
        .join('g')
        .attr('class', 'indicator')
        .attr('transform', d => `translate(${xScale(d.days_before)},${yScale(d.importance)})`);

    bubbles.append('circle')
        .attr('r', d => d.radius || 15)
        .attr('fill', d => d.color)
        .attr('opacity', 0.8)
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

    // Bubble labels
    bubbles.append('text')
        .attr('y', d => (d.radius || 15) + 15)
        .attr('text-anchor', 'middle')
        .attr('fill', '#4a5568')
        .attr('font-size', '10px')
        .each(function(d) {
            const words = d.name.split(' ');
            const text = d3.select(this);
            words.forEach((word, i) => {
                text.append('tspan')
                    .attr('x', 0)
                    .attr('dy', i === 0 ? 0 : '1.1em')
                    .text(word);
            });
        });

    // Connecting line
    const line = d3.line()
        .x(d => xScale(d.days_before))
        .y(d => yScale(d.importance))
        .curve(d3.curveMonotoneX);

    g.append('path')
        .datum(indicators.sort((a, b) => b.days_before - a.days_before))
        .attr('fill', 'none')
        .attr('stroke', '#a0aec0')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '3,3')
        .attr('d', line);

    return {
        highlightIndicator: function(name) {
            bubbles.selectAll('circle')
                .attr('opacity', d => d.name === name ? 1 : 0.3);
        },
        showAll: function() {
            bubbles.selectAll('circle').attr('opacity', 0.8);
        },
        animateIn: function() {
            bubbles
                .attr('opacity', 0)
                .transition()
                .delay((d, i) => i * 200)
                .duration(500)
                .attr('opacity', 1);
        }
    };
}

if (typeof window !== 'undefined') {
    window.renderTimeline = renderTimeline;
}
