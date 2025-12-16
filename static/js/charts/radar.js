/**
 * D3 Radar chart for multi-dimensional risk profiles.
 * Used in Essay 4: What Predicts Churn Before It Happens
 */

function renderRadar(container, spec) {
    const { data, config } = spec;
    const { axes, series } = data;

    const width = config.width || 500;
    const height = config.height || 500;
    const levels = config.levels || 5;
    const maxValue = config.maxValue || 1;
    const labelFactor = config.labelFactor || 1.15;

    const margin = { top: 50, right: 80, bottom: 50, left: 80 };
    const radius = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom) / 2;

    d3.select(container).selectAll('*').remove();

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    const g = svg.append('g')
        .attr('transform', `translate(${width / 2},${height / 2})`);

    // Angle calculation
    const angleSlice = (Math.PI * 2) / axes.length;

    // Scale for radius
    const rScale = d3.scaleLinear()
        .domain([0, maxValue])
        .range([0, radius]);

    // Draw circular grid
    const gridGroup = g.append('g').attr('class', 'grid');

    for (let level = 1; level <= levels; level++) {
        const levelRadius = radius * level / levels;

        gridGroup.append('circle')
            .attr('r', levelRadius)
            .attr('fill', 'none')
            .attr('stroke', '#e2e8f0')
            .attr('stroke-width', 1);

        // Level labels
        gridGroup.append('text')
            .attr('x', 5)
            .attr('y', -levelRadius)
            .attr('fill', '#a0aec0')
            .attr('font-size', '10px')
            .text(`${(level * maxValue / levels * 100).toFixed(0)}%`);
    }

    // Draw axis lines
    const axisGroup = g.append('g').attr('class', 'axes');

    axes.forEach((axis, i) => {
        const angle = angleSlice * i - Math.PI / 2;

        axisGroup.append('line')
            .attr('x1', 0)
            .attr('y1', 0)
            .attr('x2', radius * Math.cos(angle))
            .attr('y2', radius * Math.sin(angle))
            .attr('stroke', '#e2e8f0')
            .attr('stroke-width', 1);

        // Axis labels
        axisGroup.append('text')
            .attr('x', radius * labelFactor * Math.cos(angle))
            .attr('y', radius * labelFactor * Math.sin(angle))
            .attr('text-anchor', angle < Math.PI ? 'start' : 'end')
            .attr('dominant-baseline', 'middle')
            .attr('fill', '#4a5568')
            .attr('font-size', '12px')
            .text(axis.name);
    });

    // Radar area line generator
    const radarLine = d3.lineRadial()
        .radius(d => rScale(d))
        .angle((d, i) => i * angleSlice)
        .curve(d3.curveLinearClosed);

    // Draw series
    const seriesGroups = g.selectAll('.series')
        .data(series)
        .join('g')
        .attr('class', 'series');

    // Series areas
    seriesGroups.append('path')
        .attr('d', d => radarLine(d.values))
        .attr('fill', d => d.color)
        .attr('fill-opacity', d => d.fillOpacity || 0.2)
        .attr('stroke', d => d.color)
        .attr('stroke-width', config.strokeWidth || 2);

    // Series dots
    seriesGroups.each(function(seriesData) {
        d3.select(this).selectAll('.dot')
            .data(seriesData.values)
            .join('circle')
            .attr('class', 'dot')
            .attr('cx', (d, i) => rScale(d) * Math.cos(angleSlice * i - Math.PI / 2))
            .attr('cy', (d, i) => rScale(d) * Math.sin(angleSlice * i - Math.PI / 2))
            .attr('r', config.dotRadius || 4)
            .attr('fill', seriesData.color)
            .attr('stroke', '#fff')
            .attr('stroke-width', 1);
    });

    // Legend
    if (config.showLegend) {
        const legend = svg.append('g')
            .attr('transform', `translate(${width - 100}, 20)`);

        series.forEach((s, i) => {
            const legendRow = legend.append('g')
                .attr('transform', `translate(0, ${i * 20})`);

            legendRow.append('rect')
                .attr('width', 12)
                .attr('height', 12)
                .attr('fill', s.color)
                .attr('opacity', 0.7);

            legendRow.append('text')
                .attr('x', 18)
                .attr('y', 10)
                .attr('fill', '#4a5568')
                .attr('font-size', '11px')
                .text(s.name);
        });
    }

    return {
        highlightSeries: function(name) {
            seriesGroups.selectAll('path')
                .attr('fill-opacity', d => d.name === name ? 0.4 : 0.1)
                .attr('stroke-width', d => d.name === name ? 3 : 1);
        },
        showAll: function() {
            seriesGroups.selectAll('path')
                .attr('fill-opacity', d => d.fillOpacity || 0.2)
                .attr('stroke-width', config.strokeWidth || 2);
        },
        animateIn: function() {
            seriesGroups.selectAll('path')
                .attr('stroke-dasharray', function() {
                    return this.getTotalLength();
                })
                .attr('stroke-dashoffset', function() {
                    return this.getTotalLength();
                })
                .transition()
                .duration(1500)
                .attr('stroke-dashoffset', 0);
        }
    };
}

if (typeof window !== 'undefined') {
    window.renderRadar = renderRadar;
}
