/**
 * D3 Sankey diagram implementation for customer flow visualization.
 * Used in Essay 1: How Customers Actually Become Loyal
 */

function renderSankey(container, spec) {
    const { data, config } = spec;
    const { nodes, links } = data;

    const width = config.width || 800;
    const height = config.height || 500;
    const nodeWidth = config.nodeWidth || 20;
    const nodePadding = config.nodePadding || 10;

    // Clear container
    d3.select(container).selectAll('*').remove();

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    // Create sankey generator
    const sankey = d3.sankey()
        .nodeWidth(nodeWidth)
        .nodePadding(nodePadding)
        .nodeAlign(d3.sankeyJustify)
        .extent([[20, 20], [width - 20, height - 20]]);

    // Generate sankey data
    const sankeyData = sankey({
        nodes: nodes.map(d => Object.assign({}, d)),
        links: links.map(d => Object.assign({}, d))
    });

    // Color scale
    const colors = config.colors || {
        'new': '#4299e1',
        'one_time': '#fc8181',
        'irregular': '#f6ad55',
        'regular': '#68d391',
        'long_cycle': '#b794f4'
    };

    // Draw links
    const link = svg.append('g')
        .attr('class', 'links')
        .attr('fill', 'none')
        .attr('stroke-opacity', config.linkOpacity || 0.5)
        .selectAll('path')
        .data(sankeyData.links)
        .join('path')
        .attr('class', 'sankey-link')
        .attr('d', d3.sankeyLinkHorizontal())
        .attr('stroke', d => colors[d.source.id] || '#a0aec0')
        .attr('stroke-width', d => Math.max(1, d.width))
        .on('mouseover', function(event, d) {
            d3.select(this)
                .attr('stroke-opacity', config.linkOpacityHover || 0.8);
            showTooltip(event, `${d.source.name} → ${d.target.name}: ${d.value.toLocaleString()}`);
        })
        .on('mouseout', function() {
            d3.select(this)
                .attr('stroke-opacity', config.linkOpacity || 0.5);
            hideTooltip();
        });

    // Draw nodes
    const node = svg.append('g')
        .attr('class', 'nodes')
        .selectAll('g')
        .data(sankeyData.nodes)
        .join('g')
        .attr('class', 'sankey-node');

    node.append('rect')
        .attr('x', d => d.x0)
        .attr('y', d => d.y0)
        .attr('height', d => d.y1 - d.y0)
        .attr('width', d => d.x1 - d.x0)
        .attr('fill', d => colors[d.id] || '#a0aec0')
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .attr('rx', 4);

    // Node labels
    node.append('text')
        .attr('x', d => d.x0 < width / 2 ? d.x1 + 6 : d.x0 - 6)
        .attr('y', d => (d.y1 + d.y0) / 2)
        .attr('dy', '0.35em')
        .attr('text-anchor', d => d.x0 < width / 2 ? 'start' : 'end')
        .attr('font-size', '12px')
        .attr('fill', '#4a5568')
        .text(d => `${d.name} (${d.value?.toLocaleString() || 0})`);

    // Animation helpers
    return {
        highlightNode: function(nodeId) {
            node.selectAll('rect')
                .attr('opacity', d => d.id === nodeId ? 1 : 0.3);
            link.attr('stroke-opacity', d =>
                d.source.id === nodeId || d.target.id === nodeId
                    ? 0.8 : 0.1
            );
        },
        highlightFlow: function(sourceId, targetId) {
            link.attr('stroke-opacity', d =>
                d.source.id === sourceId && d.target.id === targetId
                    ? 0.8 : 0.1
            );
        },
        showAll: function() {
            node.selectAll('rect').attr('opacity', 1);
            link.attr('stroke-opacity', config.linkOpacity || 0.5);
        }
    };
}

// Tooltip helpers
function showTooltip(event, content) {
    let tooltip = d3.select('.chart-tooltip');
    if (tooltip.empty()) {
        tooltip = d3.select('body')
            .append('div')
            .attr('class', 'chart-tooltip')
            .style('position', 'absolute')
            .style('background', '#1a202c')
            .style('color', 'white')
            .style('padding', '8px 12px')
            .style('border-radius', '6px')
            .style('font-size', '14px')
            .style('pointer-events', 'none')
            .style('z-index', '100');
    }
    tooltip
        .html(content)
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 10) + 'px')
        .style('opacity', 1);
}

function hideTooltip() {
    d3.select('.chart-tooltip').style('opacity', 0);
}

// Export for use in essay.js
if (typeof window !== 'undefined') {
    window.renderSankey = renderSankey;
}
