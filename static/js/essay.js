/**
 * Main essay JavaScript - Scrollama orchestration and chart management.
 *
 * This file coordinates:
 * - Scrollama initialization for scrollytelling
 * - Chart rendering based on ESSAY_DATA
 * - Scroll-triggered animations
 */

// Chart instances storage
const chartInstances = {};

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', initEssay);

function initEssay() {
    console.log('Initializing essay...');

    // Initialize charts
    initCharts();

    // Initialize Scrollama
    initScrollama();

    // Initialize progress indicator
    initProgressIndicator();

    console.log('Essay initialized');
}

// =============================================================================
// CHART INITIALIZATION
// =============================================================================

function initCharts() {
    if (!window.ESSAY_DATA) {
        console.warn('No ESSAY_DATA found');
        return;
    }

    Object.entries(window.ESSAY_DATA).forEach(([chartId, spec]) => {
        const container = document.getElementById(`chart-${chartId}`);
        if (!container) {
            console.warn(`Container not found for chart: ${chartId}`);
            return;
        }

        try {
            const instance = renderChart(container, spec);
            if (instance) {
                chartInstances[chartId] = instance;
            }
        } catch (e) {
            console.error(`Error rendering chart ${chartId}:`, e);
            renderErrorState(container, chartId, e);
        }
    });
}

function renderChart(container, spec) {
    const { chart_type } = spec;

    // Map chart types to render functions
    const renderers = {
        'sankey': window.renderSankey,
        'scatter': window.renderScatter,
        'funnel': window.renderFunnel,
        'timeline': window.renderTimeline,
        'radar': window.renderRadar,
        'bar': renderBar,
        'bar_horizontal': renderBarHorizontal,
        'line': renderLine,
        'donut': renderDonut,
        'thermometer': renderThermometer,
        'metrics': renderMetrics,
    };

    const renderer = renderers[chart_type];
    if (renderer) {
        return renderer(container, spec);
    } else {
        console.warn(`Unknown chart type: ${chart_type}`);
        return renderPlaceholder(container, spec);
    }
}

function renderErrorState(container, chartId, error) {
    container.innerHTML = `
        <div class="chart-error">
            <p>Error loading chart: ${chartId}</p>
            <small>${error.message}</small>
        </div>
    `;
}

function renderPlaceholder(container, spec) {
    const { chart_id, chart_type, config } = spec;
    const width = config?.width || 600;
    const height = config?.height || 400;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    svg.append('rect')
        .attr('x', 10).attr('y', 10)
        .attr('width', width - 20).attr('height', height - 20)
        .attr('fill', '#f7fafc')
        .attr('stroke', '#e2e8f0')
        .attr('stroke-width', 2)
        .attr('rx', 8);

    svg.append('text')
        .attr('x', width / 2).attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#a0aec0')
        .attr('font-size', '16px')
        .text(`${chart_type}: ${chart_id}`);

    return null;
}

// =============================================================================
// SIMPLE CHART RENDERERS
// =============================================================================

function renderBar(container, spec) {
    const { data, config } = spec;
    const { bars } = data;
    const width = config.width || 600;
    const height = config.height || 300;
    const margin = { top: 20, right: 20, bottom: 40, left: 60 };

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const x = d3.scaleBand()
        .domain(bars.map(d => d.label))
        .range([0, innerWidth])
        .padding(0.2);

    const y = d3.scaleLinear()
        .domain([0, d3.max(bars, d => d.value) * 1.1])
        .range([innerHeight, 0]);

    g.selectAll('rect')
        .data(bars)
        .join('rect')
        .attr('x', d => x(d.label))
        .attr('y', d => y(d.value))
        .attr('width', x.bandwidth())
        .attr('height', d => innerHeight - y(d.value))
        .attr('fill', config.color || '#4299e1')
        .attr('rx', 4);

    g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x));

    g.append('g')
        .call(d3.axisLeft(y).ticks(5));

    return null;
}

function renderBarHorizontal(container, spec) {
    const { data, config } = spec;
    const { bars } = data;
    const width = config.width || 500;
    const height = config.height || 300;
    const margin = { top: 20, right: 40, bottom: 30, left: 150 };

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const y = d3.scaleBand()
        .domain(bars.map(d => d.label))
        .range([0, innerHeight])
        .padding(0.2);

    const x = d3.scaleLinear()
        .domain([0, d3.max(bars, d => d.value) * 1.1])
        .range([0, innerWidth]);

    g.selectAll('rect')
        .data(bars)
        .join('rect')
        .attr('x', 0)
        .attr('y', d => y(d.label))
        .attr('width', d => x(d.value))
        .attr('height', y.bandwidth())
        .attr('fill', d => d.color || config.color || '#4299e1')
        .attr('rx', 4);

    g.append('g')
        .call(d3.axisLeft(y));

    return null;
}

function renderLine(container, spec) {
    const { data, config } = spec;
    const { points } = data;
    const width = config.width || 600;
    const height = config.height || 300;
    const margin = { top: 20, right: 20, bottom: 40, left: 60 };

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const x = d3.scaleLinear()
        .domain(d3.extent(points, d => d.x))
        .range([0, innerWidth]);

    const y = d3.scaleLinear()
        .domain([0, d3.max(points, d => d.y) * 1.1])
        .range([innerHeight, 0]);

    const line = d3.line()
        .x(d => x(d.x))
        .y(d => y(d.y))
        .curve(d3.curveMonotoneX);

    if (config.showArea) {
        const area = d3.area()
            .x(d => x(d.x))
            .y0(innerHeight)
            .y1(d => y(d.y))
            .curve(d3.curveMonotoneX);

        g.append('path')
            .datum(points)
            .attr('fill', config.color || '#68d391')
            .attr('fill-opacity', config.areaOpacity || 0.2)
            .attr('d', area);
    }

    g.append('path')
        .datum(points)
        .attr('fill', 'none')
        .attr('stroke', config.color || '#68d391')
        .attr('stroke-width', config.strokeWidth || 3)
        .attr('d', line);

    g.selectAll('circle')
        .data(points)
        .join('circle')
        .attr('cx', d => x(d.x))
        .attr('cy', d => y(d.y))
        .attr('r', 4)
        .attr('fill', config.color || '#68d391');

    g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).ticks(6));

    g.append('g')
        .call(d3.axisLeft(y).ticks(5));

    return null;
}

function renderDonut(container, spec) {
    const { data, config } = spec;
    const { segments, total } = data;
    const width = config.width || 400;
    const height = config.height || 400;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    const g = svg.append('g')
        .attr('transform', `translate(${width / 2},${height / 2})`);

    const pie = d3.pie()
        .value(d => d.value)
        .sort(null);

    const arc = d3.arc()
        .innerRadius(config.innerRadius || 60)
        .outerRadius(config.outerRadius || 120);

    g.selectAll('path')
        .data(pie(segments))
        .join('path')
        .attr('d', arc)
        .attr('fill', d => d.data.color)
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);

    if (config.centerText) {
        g.append('text')
            .attr('text-anchor', 'middle')
            .attr('dy', '-0.2em')
            .attr('font-size', '24px')
            .attr('font-weight', 'bold')
            .attr('fill', '#2d3748')
            .text(config.centerText);

        if (config.centerSubtext) {
            g.append('text')
                .attr('text-anchor', 'middle')
                .attr('dy', '1.2em')
                .attr('font-size', '12px')
                .attr('fill', '#718096')
                .text(config.centerSubtext);
        }
    }

    return null;
}

function renderThermometer(container, spec) {
    const { data, config } = spec;
    const { buckets, total } = data;
    const width = config.width || 200;
    const height = config.height || 300;

    const svg = d3.select(container)
        .append('svg')
        .attr('width', '100%')
        .attr('height', height)
        .attr('viewBox', `0 0 ${width} ${height}`);

    const barWidth = 60;
    const barHeight = height - 80;
    const x = (width - barWidth) / 2;
    const y = 40;

    // Background
    svg.append('rect')
        .attr('x', x).attr('y', y)
        .attr('width', barWidth).attr('height', barHeight)
        .attr('fill', '#e2e8f0')
        .attr('rx', barWidth / 2);

    // Fill segments
    let currentY = y + barHeight;
    buckets.forEach(bucket => {
        const segmentHeight = (bucket.count / total) * barHeight;
        currentY -= segmentHeight;

        svg.append('rect')
            .attr('x', x).attr('y', currentY)
            .attr('width', barWidth).attr('height', segmentHeight)
            .attr('fill', bucket.color)
            .attr('rx', segmentHeight === barHeight ? barWidth / 2 : 0);
    });

    return null;
}

function renderMetrics(container, spec) {
    const { data, config } = spec;
    const { metrics } = data;

    const wrapper = d3.select(container)
        .append('div')
        .attr('class', 'metrics-wrapper')
        .style('display', 'flex')
        .style('gap', '2rem')
        .style('justify-content', 'center')
        .style('flex-wrap', 'wrap');

    metrics.forEach(metric => {
        const card = wrapper.append('div')
            .attr('class', 'metric-card')
            .style('text-align', 'center')
            .style('padding', '1.5rem');

        card.append('div')
            .attr('class', 'metric-value')
            .style('font-size', '2rem')
            .style('font-weight', 'bold')
            .style('color', metric.color)
            .text(`${metric.prefix || ''}${metric.value.toLocaleString()}${metric.suffix || ''}`);

        card.append('div')
            .attr('class', 'metric-label')
            .style('color', '#718096')
            .style('font-size', '0.875rem')
            .text(metric.label);
    });

    return null;
}

// =============================================================================
// SCROLLAMA
// =============================================================================

function initScrollama() {
    if (typeof scrollama === 'undefined') {
        console.warn('Scrollama not loaded');
        return;
    }

    const scroller = scrollama();

    scroller
        .setup({
            step: '.step',
            offset: 0.5,
            progress: true,
        })
        .onStepEnter(handleStepEnter)
        .onStepExit(handleStepExit);

    window.addEventListener('resize', scroller.resize);
}

function handleStepEnter(response) {
    const { element, index, direction } = response;
    const step = element.dataset.step;

    // Highlight active step
    document.querySelectorAll('.step').forEach(s => s.classList.remove('is-active'));
    element.classList.add('is-active');

    // Get section and trigger chart animation
    const section = element.closest('.essay-section');
    if (section) {
        const sectionId = section.id;
        triggerChartAnimation(sectionId, step, direction);
    }
}

function handleStepExit(response) {
    // Optional: handle step exit
}

function triggerChartAnimation(sectionId, step, direction) {
    // Find chart spec for this section
    if (!window.ESSAY_DATA) return;

    Object.entries(window.ESSAY_DATA).forEach(([chartId, spec]) => {
        if (!chartId.includes(sectionId.replace('section-', ''))) return;

        const instance = chartInstances[chartId];
        if (!instance) return;

        // Find matching trigger
        const triggers = spec.scrolly_triggers || [];
        const trigger = triggers.find(t => String(t.step) === String(step));

        if (trigger && instance[trigger.action]) {
            instance[trigger.action](trigger.params);
        }
    });
}

// =============================================================================
// PROGRESS INDICATOR
// =============================================================================

function initProgressIndicator() {
    const progress = document.createElement('div');
    progress.className = 'scroll-progress';
    progress.innerHTML = '<div class="scroll-progress-bar"></div>';
    document.body.appendChild(progress);

    window.addEventListener('scroll', () => {
        const scrolled = (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100;
        document.querySelector('.scroll-progress-bar').style.width = `${scrolled}%`;
    });
}
