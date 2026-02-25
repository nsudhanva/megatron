/**
 * D3 Benchmark Visualizer
 * Renders bar charts and matrix split diagrams for GPU benchmark results.
 */

const BenchViz = {
    colors: {
        single: '#00b4d8',
        half: '#f4a261',
        twoGpu: '#f4a261',
        comm: '#e94560',
        compute: '#9b5de5',
        speedup: '#76b900',
        bg: '#1a1a28',
        grid: 'rgba(255,255,255,0.06)',
        text: '#8888a4',
        textBright: '#f0f0f5',
    },

    /**
     * Render single-GPU benchmark: bar chart + matrix split diagram.
     */
    renderSingleGpu(container, data) {
        const el = document.getElementById(container);
        el.innerHTML = '';

        const wrapper = document.createElement('div');
        wrapper.style.width = '100%';
        el.appendChild(wrapper);

        // ── Bar Chart: Full vs Half Matmul ──
        const chartDiv = document.createElement('div');
        chartDiv.className = 'bench-viz-chart';
        wrapper.appendChild(chartDiv);

        const margin = { top: 30, right: 30, bottom: 60, left: 70 };
        const width = 500 - margin.left - margin.right;
        const height = 250 - margin.top - margin.bottom;

        const svg = d3.select(chartDiv).append('svg')
            .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const bars = [
            { label: `Full (${data.hidden_size}x${data.hidden_size})`, value: data.full_matmul_ms, color: this.colors.single },
            { label: `Half (${data.hidden_size}x${data.hidden_size / 2})`, value: data.half_matmul_ms, color: this.colors.half },
        ];

        const x = d3.scaleBand().domain(bars.map(b => b.label)).range([0, width]).padding(0.4);
        const y = d3.scaleLinear().domain([0, d3.max(bars, b => b.value) * 1.2]).range([height, 0]);

        // Grid lines
        svg.selectAll('.grid-line')
            .data(y.ticks(5))
            .join('line')
            .attr('x1', 0).attr('x2', width)
            .attr('y1', d => y(d)).attr('y2', d => y(d))
            .attr('stroke', this.colors.grid);

        // Y axis
        svg.append('g')
            .call(d3.axisLeft(y).ticks(5).tickFormat(d => d.toFixed(2) + 'ms'))
            .selectAll('text').style('fill', this.colors.text).style('font-size', '10px');
        svg.selectAll('.domain, .tick line').attr('stroke', this.colors.grid);

        // Bars
        svg.selectAll('.bar')
            .data(bars)
            .join('rect')
            .attr('x', d => x(d.label))
            .attr('y', d => y(d.value))
            .attr('width', x.bandwidth())
            .attr('height', d => height - y(d.value))
            .attr('fill', d => d.color)
            .attr('rx', 4)
            .attr('opacity', 0.9);

        // Value labels
        svg.selectAll('.bar-label')
            .data(bars)
            .join('text')
            .attr('x', d => x(d.label) + x.bandwidth() / 2)
            .attr('y', d => y(d.value) - 8)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.textBright)
            .attr('font-size', '12px')
            .attr('font-weight', '600')
            .attr('font-family', "'JetBrains Mono', monospace")
            .text(d => d.value.toFixed(3) + 'ms');

        // X axis labels
        svg.selectAll('.x-label')
            .data(bars)
            .join('text')
            .attr('x', d => x(d.label) + x.bandwidth() / 2)
            .attr('y', height + 20)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.text)
            .attr('font-size', '10px')
            .text(d => d.label);

        // Title
        svg.append('text')
            .attr('x', width / 2).attr('y', -10)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.textBright)
            .attr('font-size', '13px')
            .attr('font-weight', '600')
            .text('Matmul Timing');

        // ── Stats Cards ──
        const statsDiv = document.createElement('div');
        statsDiv.className = 'bench-grid';
        statsDiv.style.marginTop = '24px';
        wrapper.appendChild(statsDiv);

        statsDiv.innerHTML = `
            <div class="bench-card highlight">
                <div class="bench-value">${data.theoretical_speedup.toFixed(2)}x</div>
                <div class="bench-label">Theoretical TP Speedup</div>
            </div>
            <div class="bench-card">
                <div class="bench-value" style="color: ${this.colors.single}">${data.gpu_name}</div>
                <div class="bench-label">${data.num_iterations} iterations</div>
            </div>
        `;

        // ── Matrix Split Diagram ──
        const matDiv = document.createElement('div');
        matDiv.className = 'bench-viz-chart';
        matDiv.style.marginTop = '24px';
        wrapper.appendChild(matDiv);

        this._renderMatrixSplit(matDiv, data.hidden_size);
    },

    /**
     * Render multi-GPU benchmark: comparison bar chart + communication breakdown.
     */
    renderMultiGpu(container, data) {
        const el = document.getElementById(container);
        el.innerHTML = '';

        const wrapper = document.createElement('div');
        wrapper.style.width = '100%';
        el.appendChild(wrapper);

        // ── Comparison Bar Chart ──
        const chartDiv = document.createElement('div');
        chartDiv.className = 'bench-viz-chart';
        wrapper.appendChild(chartDiv);

        const margin = { top: 30, right: 30, bottom: 60, left: 70 };
        const width = 500 - margin.left - margin.right;
        const height = 250 - margin.top - margin.bottom;

        const svg = d3.select(chartDiv).append('svg')
            .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const bars = [
            { label: '1x GPU', value: data.single_gpu_ms, color: this.colors.single },
            { label: '2x GPU (TP)', value: data.two_gpu_tp_ms, color: this.colors.twoGpu },
        ];

        const x = d3.scaleBand().domain(bars.map(b => b.label)).range([0, width]).padding(0.4);
        const maxVal = d3.max(bars, b => b.value) * 1.2;
        const y = d3.scaleLinear().domain([0, maxVal]).range([height, 0]);

        svg.selectAll('.grid-line')
            .data(y.ticks(5))
            .join('line')
            .attr('x1', 0).attr('x2', width)
            .attr('y1', d => y(d)).attr('y2', d => y(d))
            .attr('stroke', this.colors.grid);

        svg.append('g')
            .call(d3.axisLeft(y).ticks(5).tickFormat(d => d.toFixed(1) + 'ms'))
            .selectAll('text').style('fill', this.colors.text).style('font-size', '10px');
        svg.selectAll('.domain, .tick line').attr('stroke', this.colors.grid);

        svg.selectAll('.bar')
            .data(bars)
            .join('rect')
            .attr('x', d => x(d.label))
            .attr('y', d => y(d.value))
            .attr('width', x.bandwidth())
            .attr('height', d => height - y(d.value))
            .attr('fill', d => d.color)
            .attr('rx', 4)
            .attr('opacity', 0.9);

        svg.selectAll('.bar-label')
            .data(bars)
            .join('text')
            .attr('x', d => x(d.label) + x.bandwidth() / 2)
            .attr('y', d => y(d.value) - 8)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.textBright)
            .attr('font-size', '12px')
            .attr('font-weight', '600')
            .attr('font-family', "'JetBrains Mono', monospace")
            .text(d => d.value.toFixed(3) + 'ms');

        svg.selectAll('.x-label')
            .data(bars)
            .join('text')
            .attr('x', d => x(d.label) + x.bandwidth() / 2)
            .attr('y', height + 20)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.text)
            .attr('font-size', '11px')
            .text(d => d.label);

        // Speedup annotation
        if (bars[0].value > bars[1].value) {
            const arrowY = y(bars[1].value) - 30;
            svg.append('text')
                .attr('x', width / 2).attr('y', arrowY)
                .attr('text-anchor', 'middle')
                .attr('fill', this.colors.speedup)
                .attr('font-size', '14px')
                .attr('font-weight', '700')
                .attr('font-family', "'JetBrains Mono', monospace")
                .text(`${data.actual_speedup.toFixed(2)}x faster`);
        }

        svg.append('text')
            .attr('x', width / 2).attr('y', -10)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.textBright)
            .attr('font-size', '13px')
            .attr('font-weight', '600')
            .text(`${data.hidden_size}x${data.hidden_size} Matmul: 1 GPU vs 2-GPU TP`);

        // ── Communication Breakdown Stacked Bar ──
        const breakDiv = document.createElement('div');
        breakDiv.className = 'bench-viz-chart';
        breakDiv.style.marginTop = '24px';
        wrapper.appendChild(breakDiv);

        this._renderCommBreakdown(breakDiv, data);

        // ── Stats Grid ──
        const statsDiv = document.createElement('div');
        statsDiv.className = 'bench-grid';
        statsDiv.style.marginTop = '24px';
        statsDiv.style.gridTemplateColumns = 'repeat(3, 1fr)';
        wrapper.appendChild(statsDiv);

        statsDiv.innerHTML = `
            <div class="bench-card highlight">
                <div class="bench-value">${data.actual_speedup.toFixed(2)}x</div>
                <div class="bench-label">Actual Speedup</div>
            </div>
            <div class="bench-card">
                <div class="bench-value" style="color: ${this.colors.compute}">${data.parallel_efficiency.toFixed(1)}%</div>
                <div class="bench-label">Parallel Efficiency</div>
            </div>
            <div class="bench-card">
                <div class="bench-value" style="color: ${this.colors.comm}">${data.communication_ms.toFixed(3)}</div>
                <div class="bench-unit">ms</div>
                <div class="bench-label">Communication Cost</div>
            </div>
        `;

        // GPU label
        const gpuLabel = document.createElement('div');
        gpuLabel.className = 'bench-gpu-name';
        gpuLabel.textContent = `${data.gpu_names.join(' + ')} -- ${data.num_iterations} iters -- column-parallel TP`;
        wrapper.appendChild(gpuLabel);
    },

    /**
     * Stacked horizontal bar: compute vs communication time.
     */
    _renderCommBreakdown(container, data) {
        const margin = { top: 30, right: 30, bottom: 40, left: 100 };
        const width = 500 - margin.left - margin.right;
        const height = 100 - margin.top - margin.bottom;

        const svg = d3.select(container).append('svg')
            .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const computeMs = data.breakdown.compute_per_gpu_ms;
        const commMs = data.breakdown.communication_ms;
        const total = computeMs + commMs;

        const x = d3.scaleLinear().domain([0, total]).range([0, width]);

        // Compute bar
        svg.append('rect')
            .attr('x', 0).attr('y', 0)
            .attr('width', x(computeMs)).attr('height', height)
            .attr('fill', this.colors.compute).attr('rx', 4).attr('opacity', 0.9);

        // Communication bar
        svg.append('rect')
            .attr('x', x(computeMs)).attr('y', 0)
            .attr('width', x(commMs)).attr('height', height)
            .attr('fill', this.colors.comm).attr('rx', 4).attr('opacity', 0.9);

        // Labels inside bars
        if (computeMs > total * 0.15) {
            svg.append('text')
                .attr('x', x(computeMs) / 2).attr('y', height / 2 + 4)
                .attr('text-anchor', 'middle')
                .attr('fill', '#fff').attr('font-size', '10px').attr('font-weight', '600')
                .text(`Compute: ${computeMs.toFixed(2)}ms`);
        }
        if (commMs > total * 0.1) {
            svg.append('text')
                .attr('x', x(computeMs) + x(commMs) / 2).attr('y', height / 2 + 4)
                .attr('text-anchor', 'middle')
                .attr('fill', '#fff').attr('font-size', '10px').attr('font-weight', '600')
                .text(`Comms: ${commMs.toFixed(2)}ms`);
        }

        // Label
        svg.append('text')
            .attr('x', -10).attr('y', height / 2 + 4)
            .attr('text-anchor', 'end')
            .attr('fill', this.colors.text)
            .attr('font-size', '10px')
            .text('2-GPU Time');

        svg.append('text')
            .attr('x', width / 2).attr('y', -10)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.textBright)
            .attr('font-size', '13px')
            .attr('font-weight', '600')
            .text('Time Breakdown: Compute vs Communication');
    },

    /**
     * Matrix split diagram showing column parallel.
     */
    _renderMatrixSplit(container, hiddenSize) {
        const margin = { top: 30, right: 20, bottom: 30, left: 20 };
        const width = 500 - margin.left - margin.right;
        const height = 140 - margin.top - margin.bottom;

        const svg = d3.select(container).append('svg')
            .attr('viewBox', `0 0 ${width + margin.left + margin.right} ${height + margin.top + margin.bottom}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);

        const matW = 100, matH = 80;
        const gap = 30;

        // Full matrix
        const x0 = 40;
        svg.append('rect')
            .attr('x', x0).attr('y', 0)
            .attr('width', matW).attr('height', matH)
            .attr('fill', this.colors.single).attr('rx', 4).attr('opacity', 0.3)
            .attr('stroke', this.colors.single).attr('stroke-width', 1.5);

        svg.append('text')
            .attr('x', x0 + matW / 2).attr('y', matH / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.textBright).attr('font-size', '10px').attr('font-weight', '600')
            .text(`W [${hiddenSize}x${hiddenSize}]`);

        svg.append('text')
            .attr('x', x0 + matW / 2).attr('y', matH + 16)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.text).attr('font-size', '9px')
            .text('Full matrix');

        // Arrow
        const arrowX = x0 + matW + gap;
        svg.append('line')
            .attr('x1', arrowX - 10).attr('y1', matH / 2)
            .attr('x2', arrowX + 15).attr('y2', matH / 2)
            .attr('stroke', this.colors.text).attr('stroke-width', 1.5)
            .attr('marker-end', 'url(#arrowhead)');

        svg.append('text')
            .attr('x', arrowX + 3).attr('y', matH / 2 - 8)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.speedup).attr('font-size', '9px').attr('font-weight', '600')
            .text('SPLIT');

        // GPU 0 shard
        const x1 = arrowX + gap + 10;
        svg.append('rect')
            .attr('x', x1).attr('y', 0)
            .attr('width', matW / 2).attr('height', matH)
            .attr('fill', '#76b900').attr('rx', 4).attr('opacity', 0.4)
            .attr('stroke', '#76b900').attr('stroke-width', 1.5);

        svg.append('text')
            .attr('x', x1 + matW / 4).attr('y', matH / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.textBright).attr('font-size', '9px').attr('font-weight', '600')
            .text(`GPU 0`);

        svg.append('text')
            .attr('x', x1 + matW / 4).attr('y', matH + 16)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.text).attr('font-size', '8px')
            .text(`[${hiddenSize}x${hiddenSize / 2}]`);

        // GPU 1 shard
        const x2 = x1 + matW / 2 + 8;
        svg.append('rect')
            .attr('x', x2).attr('y', 0)
            .attr('width', matW / 2).attr('height', matH)
            .attr('fill', '#00b4d8').attr('rx', 4).attr('opacity', 0.4)
            .attr('stroke', '#00b4d8').attr('stroke-width', 1.5);

        svg.append('text')
            .attr('x', x2 + matW / 4).attr('y', matH / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.textBright).attr('font-size', '9px').attr('font-weight', '600')
            .text(`GPU 1`);

        svg.append('text')
            .attr('x', x2 + matW / 4).attr('y', matH + 16)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.text).attr('font-size', '8px')
            .text(`[${hiddenSize}x${hiddenSize / 2}]`);

        // Title
        svg.append('text')
            .attr('x', width / 2).attr('y', -10)
            .attr('text-anchor', 'middle')
            .attr('fill', this.colors.textBright)
            .attr('font-size', '13px')
            .attr('font-weight', '600')
            .text('Column-Parallel Weight Split');
    },
};
