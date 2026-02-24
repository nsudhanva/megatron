/**
 * Megatron Parallelism Visualizer — D3.js + GSAP rendering engine.
 *
 * Creates stunning, animated visualizations of GPU parallelism strategies.
 */

const GPU_COLORS = [
    '#76b900', '#00b4d8', '#e94560', '#f4a261',
    '#9b5de5', '#00f5d4', '#fee440', '#f15bb5',
];

function gpuColor(i) {
    return GPU_COLORS[i % GPU_COLORS.length];
}

function gpuColorAlpha(i, alpha) {
    const hex = GPU_COLORS[i % GPU_COLORS.length];
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/* ════════════════════════════════════════════
   TENSOR PARALLELISM VISUALIZATION
   ════════════════════════════════════════════ */

function renderTensorParallel(containerId, hiddenSize, numGpus, type, animate = false) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    const width = container.clientWidth;
    const height = Math.max(500, container.clientHeight);
    const margin = { top: 50, right: 40, bottom: 50, left: 40 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const svg = d3.select(`#${containerId}`)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    // Defs for gradients and arrows
    const defs = svg.append('defs');

    // Arrowhead marker
    defs.append('marker')
        .attr('id', 'arrowhead')
        .attr('viewBox', '0 0 10 10')
        .attr('refX', 8)
        .attr('refY', 5)
        .attr('markerWidth', 6)
        .attr('markerHeight', 6)
        .attr('orient', 'auto-start-reverse')
        .append('path')
        .attr('d', 'M 0 0 L 10 5 L 0 10 z')
        .attr('fill', '#76b900');

    // Glow filter
    const glow = defs.append('filter').attr('id', 'glow');
    glow.append('feGaussianBlur').attr('stdDeviation', '3').attr('result', 'coloredBlur');
    const merge = glow.append('feMerge');
    merge.append('feMergeNode').attr('in', 'coloredBlur');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Layout: 3 sections horizontally
    // [Original Matrix] → [Shards on GPUs] → [Output]
    const sectionW = w / 3;

    // ── Section 1: Original Matrix ──
    const matrixSize = Math.min(sectionW - 40, h - 80, 200);
    const cellCount = Math.min(hiddenSize, 16); // visual cells
    const cellSize = matrixSize / cellCount;

    const matrixGroup = g.append('g')
        .attr('transform', `translate(${sectionW / 2 - matrixSize / 2}, ${h / 2 - matrixSize / 2})`);

    // Title
    g.append('text')
        .attr('x', sectionW / 2)
        .attr('y', -20)
        .attr('text-anchor', 'middle')
        .attr('class', 'axis-label')
        .text(`Weight Matrix W (${hiddenSize}×${hiddenSize})`);

    // Draw matrix cells with GPU color preview
    const shardVisualSize = cellCount / numGpus;
    for (let row = 0; row < cellCount; row++) {
        for (let col = 0; col < cellCount; col++) {
            let gpuIdx;
            if (type === 'column') {
                gpuIdx = Math.floor(col / shardVisualSize);
            } else {
                gpuIdx = Math.floor(row / shardVisualSize);
            }
            gpuIdx = Math.min(gpuIdx, numGpus - 1);

            matrixGroup.append('rect')
                .attr('x', col * cellSize)
                .attr('y', row * cellSize)
                .attr('width', cellSize - 1)
                .attr('height', cellSize - 1)
                .attr('rx', 2)
                .attr('fill', gpuColorAlpha(gpuIdx, 0.6))
                .attr('class', 'matrix-cell')
                .attr('data-gpu', gpuIdx);
        }
    }

    // Matrix border
    matrixGroup.append('rect')
        .attr('width', matrixSize)
        .attr('height', matrixSize)
        .attr('fill', 'none')
        .attr('stroke', 'rgba(255,255,255,0.15)')
        .attr('stroke-width', 2)
        .attr('rx', 4);

    // ── Section 2: GPU Shards ──
    const shardSection = g.append('g')
        .attr('transform', `translate(${sectionW}, 0)`);

    g.append('text')
        .attr('x', sectionW + sectionW / 2)
        .attr('y', -20)
        .attr('text-anchor', 'middle')
        .attr('class', 'axis-label')
        .text(`${numGpus} GPU Shards (${type} split)`);

    const shardGap = 12;
    const availH = h - 40;
    const shardH = (availH - (numGpus - 1) * shardGap) / numGpus;
    const shardW = type === 'column'
        ? matrixSize / numGpus
        : matrixSize;

    const scaledShardW = Math.min(sectionW - 60, type === 'column' ? 40 : matrixSize);
    const scaledShardH = Math.min(shardH, type === 'row' ? 30 : shardH);

    for (let gpu = 0; gpu < numGpus; gpu++) {
        const sy = 20 + gpu * (scaledShardH + shardGap);
        const shardGroup = shardSection.append('g')
            .attr('transform', `translate(${sectionW / 2 - scaledShardW / 2}, ${sy})`);

        // Shard rectangle
        shardGroup.append('rect')
            .attr('width', scaledShardW)
            .attr('height', scaledShardH)
            .attr('rx', 6)
            .attr('fill', gpuColorAlpha(gpu, 0.25))
            .attr('stroke', gpuColor(gpu))
            .attr('stroke-width', 1.5)
            .style('filter', 'url(#glow)');

        // GPU label
        shardGroup.append('text')
            .attr('x', scaledShardW + 10)
            .attr('y', scaledShardH / 2 + 4)
            .attr('class', 'gpu-label')
            .attr('fill', gpuColor(gpu))
            .text(`GPU ${gpu}`);

        // Shard label inside
        const shardSize = hiddenSize / numGpus;
        const dimLabel = type === 'column'
            ? `${hiddenSize}×${shardSize}`
            : `${shardSize}×${hiddenSize}`;

        shardGroup.append('text')
            .attr('x', scaledShardW / 2)
            .attr('y', scaledShardH / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('font-family', "'JetBrains Mono', monospace")
            .attr('font-size', '10px')
            .attr('fill', 'rgba(255,255,255,0.7)')
            .text(dimLabel);
    }

    // ── Arrows: Matrix → Shards ──
    const arrow1 = g.append('g');
    arrow1.append('line')
        .attr('x1', sectionW / 2 + matrixSize / 2 + 10)
        .attr('y1', h / 2)
        .attr('x2', sectionW + sectionW / 2 - scaledShardW / 2 - 15)
        .attr('y2', h / 2)
        .attr('stroke', '#76b900')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrowhead)')
        .attr('opacity', 0.6);

    arrow1.append('text')
        .attr('x', (sectionW / 2 + matrixSize / 2 + sectionW + sectionW / 2 - scaledShardW / 2) / 2)
        .attr('y', h / 2 - 10)
        .attr('text-anchor', 'middle')
        .attr('font-size', '11px')
        .attr('fill', '#76b900')
        .attr('font-weight', '600')
        .text('SPLIT');

    // ── Section 3: Communication + Output ──
    const commSection = g.append('g')
        .attr('transform', `translate(${sectionW * 2}, 0)`);

    const commType = type === 'column' ? 'All-Gather' : 'All-Reduce';

    g.append('text')
        .attr('x', sectionW * 2 + sectionW / 2)
        .attr('y', -20)
        .attr('text-anchor', 'middle')
        .attr('class', 'axis-label')
        .text(`Communication: ${commType}`);

    // Communication visualization — ring topology
    const ringR = Math.min(sectionW / 2 - 30, h / 2 - 60, 100);
    const ringCx = sectionW / 2;
    const ringCy = h / 2;

    // Draw ring connections
    for (let i = 0; i < numGpus; i++) {
        const a1 = (2 * Math.PI * i) / numGpus - Math.PI / 2;
        const a2 = (2 * Math.PI * ((i + 1) % numGpus)) / numGpus - Math.PI / 2;

        commSection.append('line')
            .attr('x1', ringCx + Math.cos(a1) * ringR)
            .attr('y1', ringCy + Math.sin(a1) * ringR)
            .attr('x2', ringCx + Math.cos(a2) * ringR)
            .attr('y2', ringCy + Math.sin(a2) * ringR)
            .attr('stroke', 'rgba(118, 185, 0, 0.3)')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '4 3');
    }

    // Draw GPU nodes in ring
    for (let i = 0; i < numGpus; i++) {
        const angle = (2 * Math.PI * i) / numGpus - Math.PI / 2;
        const nx = ringCx + Math.cos(angle) * ringR;
        const ny = ringCy + Math.sin(angle) * ringR;

        commSection.append('circle')
            .attr('cx', nx)
            .attr('cy', ny)
            .attr('r', 20)
            .attr('fill', gpuColorAlpha(i, 0.3))
            .attr('stroke', gpuColor(i))
            .attr('stroke-width', 2)
            .style('filter', 'url(#glow)');

        commSection.append('text')
            .attr('x', nx)
            .attr('y', ny + 4)
            .attr('text-anchor', 'middle')
            .attr('font-family', "'JetBrains Mono', monospace")
            .attr('font-size', '10px')
            .attr('fill', '#fff')
            .attr('font-weight', '600')
            .text(`G${i}`);
    }

    // Communication label
    commSection.append('text')
        .attr('x', ringCx)
        .attr('y', ringCy + ringR + 40)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#76b900')
        .attr('font-weight', '600')
        .text(commType);

    // Arrow from shards to comm
    g.append('line')
        .attr('x1', sectionW * 1.5 + scaledShardW / 2 + 15)
        .attr('y1', h / 2)
        .attr('x2', sectionW * 2 + ringCx - ringR - 25)
        .attr('y2', h / 2)
        .attr('stroke', '#76b900')
        .attr('stroke-width', 2)
        .attr('marker-end', 'url(#arrowhead)')
        .attr('opacity', 0.6);

    // ── GSAP Animation ──
    if (animate) {
        const tl = gsap.timeline();

        // Fade in matrix
        tl.from(matrixGroup.node(), { opacity: 0, scale: 0.8, duration: 0.6, ease: 'back.out(1.7)' });

        // Highlight splits with wave effect
        tl.to('.matrix-cell', {
            opacity: 1,
            duration: 0.4,
            stagger: { each: 0.01, from: type === 'column' ? 'start' : 'edges' },
        }, '-=0.2');

        // Slide in shards
        shardSection.selectAll('g').each(function (_, i) {
            tl.from(this, {
                x: -50, opacity: 0, duration: 0.4, ease: 'power2.out',
            }, `-=${i > 0 ? 0.25 : 0}`);
        });

        // Animate comm ring
        tl.from(commSection.node(), { opacity: 0, scale: 0.5, duration: 0.6, ease: 'back.out(1.7)' });

        // Pulse ring nodes
        commSection.selectAll('circle').each(function (_, i) {
            tl.to(this, {
                attr: { r: 24 },
                duration: 0.3,
                yoyo: true,
                repeat: 1,
                ease: 'sine.inOut',
            }, `-=${i > 0 ? 0.15 : 0}`);
        });
    }

    // Update info cards
    renderTPInfoCards(hiddenSize, numGpus, type);
}


function renderTPInfoCards(hiddenSize, numGpus, type) {
    const container = document.getElementById('tp-info');
    const steps = Simulator.tensorParallel(hiddenSize, numGpus, type);

    container.innerHTML = steps.map((step, i) => `
        <div class="info-card">
            <span class="step-number">${i + 1}</span>
            <h4>${step.label}</h4>
            <p>${step.description}</p>
            ${step.shards && step.shards.length > 0 ? `
                <div class="gpu-list">
                    ${step.shards.map(s => `
                        <div class="gpu-item" style="border-left: 3px solid ${gpuColor(s.gpu)}">
                            GPU ${s.gpu}: ${s.label}
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        </div>
    `).join('');
}


/* ════════════════════════════════════════════
   PIPELINE PARALLELISM VISUALIZATION
   ════════════════════════════════════════════ */

function renderPipelineParallel(containerId, numLayers, numGpus, numMicrobatches, animate = false) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    const data = Simulator.pipelineParallel(numLayers, numGpus, numMicrobatches);
    const { grid, maxTime, metrics } = data;

    const width = container.clientWidth;
    const height = Math.max(500, container.clientHeight);
    const margin = { top: 60, right: 40, bottom: 60, left: 80 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const svg = d3.select(`#${containerId}`)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const defs = svg.append('defs');
    const glow = defs.append('filter').attr('id', 'pp-glow');
    glow.append('feGaussianBlur').attr('stdDeviation', '2').attr('result', 'coloredBlur');
    const merge = glow.append('feMerge');
    merge.append('feMergeNode').attr('in', 'coloredBlur');
    merge.append('feMergeNode').attr('in', 'SourceGraphic');

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Cell dimensions
    const cellW = Math.min(w / maxTime, 50);
    const cellH = Math.min(h / numGpus, 60);
    const gap = 3;

    // Title
    g.append('text')
        .attr('x', (maxTime * cellW) / 2)
        .attr('y', -35)
        .attr('text-anchor', 'middle')
        .attr('class', 'axis-label')
        .text('Pipeline Schedule (1F1B)');

    // Y-axis labels (GPU stages)
    for (let s = 0; s < numGpus; s++) {
        g.append('text')
            .attr('x', -15)
            .attr('y', s * cellH + cellH / 2 + 4)
            .attr('text-anchor', 'end')
            .attr('class', 'gpu-label')
            .attr('fill', gpuColor(s))
            .text(`Stage ${s}`);

        // Layers info
        const layersPerStage = Math.floor(numLayers / numGpus);
        g.append('text')
            .attr('x', -15)
            .attr('y', s * cellH + cellH / 2 + 16)
            .attr('text-anchor', 'end')
            .attr('font-size', '9px')
            .attr('fill', 'rgba(255,255,255,0.3)')
            .text(`L${s * layersPerStage}-${(s + 1) * layersPerStage - 1}`);
    }

    // X-axis label
    g.append('text')
        .attr('x', (maxTime * cellW) / 2)
        .attr('y', numGpus * cellH + 35)
        .attr('text-anchor', 'middle')
        .attr('class', 'axis-label')
        .text('Time Steps →');

    // Time step ticks
    for (let t = 0; t < maxTime; t += Math.max(1, Math.floor(maxTime / 20))) {
        g.append('text')
            .attr('x', t * cellW + cellW / 2)
            .attr('y', numGpus * cellH + 18)
            .attr('text-anchor', 'middle')
            .attr('font-size', '9px')
            .attr('fill', 'rgba(255,255,255,0.3)')
            .attr('font-family', "'JetBrains Mono', monospace")
            .text(t);
    }

    // Draw grid cells
    grid.forEach((cell, idx) => {
        const x = cell.time * cellW;
        const y = cell.stage * cellH;

        if (cell.isBubble) {
            g.append('rect')
                .attr('x', x + gap / 2)
                .attr('y', y + gap / 2)
                .attr('width', cellW - gap)
                .attr('height', cellH - gap)
                .attr('class', 'bubble-cell');
        } else {
            const isForward = cell.phase === 'forward';
            const mbColor = gpuColor(cell.microbatch % GPU_COLORS.length);
            const alpha = isForward ? 0.7 : 0.4;

            const rect = g.append('rect')
                .attr('x', x + gap / 2)
                .attr('y', y + gap / 2)
                .attr('width', cellW - gap)
                .attr('height', cellH - gap)
                .attr('class', 'pipeline-cell')
                .attr('fill', gpuColorAlpha(cell.microbatch % GPU_COLORS.length, alpha))
                .attr('stroke', mbColor)
                .attr('stroke-opacity', 0.5);

            // Label inside cell
            if (cellW > 20 && cellH > 20) {
                g.append('text')
                    .attr('x', x + cellW / 2)
                    .attr('y', y + cellH / 2 - 2)
                    .attr('text-anchor', 'middle')
                    .attr('font-size', '9px')
                    .attr('font-weight', '600')
                    .attr('fill', '#fff')
                    .text(isForward ? 'F' : 'B');

                g.append('text')
                    .attr('x', x + cellW / 2)
                    .attr('y', y + cellH / 2 + 10)
                    .attr('text-anchor', 'middle')
                    .attr('font-family', "'JetBrains Mono', monospace")
                    .attr('font-size', '8px')
                    .attr('fill', 'rgba(255,255,255,0.6)')
                    .text(`m${cell.microbatch}`);
            }
        }
    });

    // Legend
    const legend = g.append('g')
        .attr('transform', `translate(0, ${numGpus * cellH + 45})`);

    const legendItems = [
        { label: 'Forward', fill: gpuColorAlpha(0, 0.7), stroke: gpuColor(0) },
        { label: 'Backward', fill: gpuColorAlpha(0, 0.4), stroke: gpuColor(0) },
        { label: 'Bubble (idle)', fill: 'rgba(255,255,255,0.03)', stroke: 'rgba(255,255,255,0.15)', dashed: true },
    ];

    legendItems.forEach((item, i) => {
        legend.append('rect')
            .attr('x', i * 120)
            .attr('y', 0)
            .attr('width', 14)
            .attr('height', 14)
            .attr('rx', 3)
            .attr('fill', item.fill)
            .attr('stroke', item.stroke)
            .attr('stroke-dasharray', item.dashed ? '3 2' : 'none');

        legend.append('text')
            .attr('x', i * 120 + 20)
            .attr('y', 11)
            .attr('font-size', '11px')
            .attr('fill', 'rgba(255,255,255,0.6)')
            .text(item.label);
    });

    // Animation
    if (animate) {
        const activeCells = g.selectAll('.pipeline-cell').nodes();
        gsap.from(activeCells, {
            opacity: 0,
            scale: 0.3,
            duration: 0.3,
            stagger: { each: 0.02, from: 'start' },
            ease: 'back.out(1.5)',
        });
    }

    // Metrics
    renderPPMetrics(metrics, numLayers, numGpus);
}


function renderPPMetrics(metrics, numLayers, numGpus) {
    const container = document.getElementById('pp-metrics');
    container.innerHTML = `
        <div class="metric-tile">
            <div class="metric-value">${metrics.layersPerStage}</div>
            <div class="metric-label">Layers / Stage</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">${(metrics.bubbleRatio * 100).toFixed(1)}%</div>
            <div class="metric-label">Bubble Overhead</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">${metrics.activeSlots}</div>
            <div class="metric-label">Active Slots</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">${metrics.bubbleSlots}</div>
            <div class="metric-label">Idle (Bubble) Slots</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">${metrics.totalSlots}</div>
            <div class="metric-label">Total Slots</div>
        </div>
    `;
}


/* ════════════════════════════════════════════
   DATA PARALLELISM VISUALIZATION
   ════════════════════════════════════════════ */

function renderDataParallel(containerId, batchSize, numGpus, accumSteps, animate = false) {
    const container = document.getElementById(containerId);
    container.innerHTML = '';

    const data = Simulator.dataParallel(batchSize, numGpus, accumSteps);
    const { phases, metrics } = data;

    const width = container.clientWidth;
    const height = Math.max(500, container.clientHeight);
    const margin = { top: 60, right: 40, bottom: 60, left: 60 };
    const w = width - margin.left - margin.right;
    const h = height - margin.top - margin.bottom;

    const svg = d3.select(`#${containerId}`)
        .append('svg')
        .attr('width', width)
        .attr('height', height);

    const defs = svg.append('defs');
    const glow = defs.append('filter').attr('id', 'dp-glow');
    glow.append('feGaussianBlur').attr('stdDeviation', '2').attr('result', 'coloredBlur');
    const feMerge = glow.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Layout: 4 sections vertically
    // [Full Batch] → [Split Batches per GPU] → [Forward/Backward] → [All-Reduce]
    const sectionH = h / 4;
    const gpuCardW = Math.min((w - (numGpus - 1) * 16) / numGpus, 180);
    const gpuCardH = sectionH - 30;
    const totalGpuW = numGpus * gpuCardW + (numGpus - 1) * 16;
    const startX = (w - totalGpuW) / 2;

    // ── 1. Full Batch ──
    g.append('text')
        .attr('x', w / 2)
        .attr('y', -25)
        .attr('text-anchor', 'middle')
        .attr('class', 'axis-label')
        .text('Data Parallel Training Flow');

    const batchGroup = g.append('g');
    const batchW = Math.min(w * 0.7, 500);
    const batchH = 40;
    const batchX = (w - batchW) / 2;

    batchGroup.append('rect')
        .attr('x', batchX)
        .attr('y', 10)
        .attr('width', batchW)
        .attr('height', batchH)
        .attr('rx', 8)
        .attr('fill', 'rgba(118, 185, 0, 0.15)')
        .attr('stroke', '#76b900')
        .attr('stroke-width', 1.5);

    batchGroup.append('text')
        .attr('x', w / 2)
        .attr('y', 35)
        .attr('text-anchor', 'middle')
        .attr('font-family', "'JetBrains Mono', monospace")
        .attr('font-size', '12px')
        .attr('fill', '#76b900')
        .attr('font-weight', '600')
        .text(`Full Batch: ${batchSize} samples`);

    // Draw colored segments for each GPU's micro-batch
    const microBatch = batchSize / numGpus;
    const segW = batchW / numGpus;
    for (let i = 0; i < numGpus; i++) {
        batchGroup.append('rect')
            .attr('x', batchX + i * segW + 1)
            .attr('y', 11)
            .attr('width', segW - 2)
            .attr('height', batchH - 2)
            .attr('rx', i === 0 ? 7 : (i === numGpus - 1 ? 7 : 0))
            .attr('fill', gpuColorAlpha(i, 0.3))
            .attr('stroke', 'none');
    }

    // ── 2. Split arrows ──
    const splitY = sectionH;
    for (let i = 0; i < numGpus; i++) {
        const fromX = batchX + i * segW + segW / 2;
        const toX = startX + i * (gpuCardW + 16) + gpuCardW / 2;

        g.append('path')
            .attr('d', `M ${fromX} ${batchH + 15} Q ${(fromX + toX) / 2} ${splitY - 10}, ${toX} ${splitY}`)
            .attr('fill', 'none')
            .attr('stroke', gpuColorAlpha(i, 0.4))
            .attr('stroke-width', 2);
    }

    // ── 3. GPU Cards — Forward & Backward ──
    const gpuSection = g.append('g')
        .attr('transform', `translate(0, ${splitY})`);

    for (let i = 0; i < numGpus; i++) {
        const cx = startX + i * (gpuCardW + 16);
        const card = gpuSection.append('g')
            .attr('transform', `translate(${cx}, 0)`);

        // Card background
        card.append('rect')
            .attr('width', gpuCardW)
            .attr('height', gpuCardH)
            .attr('rx', 10)
            .attr('fill', gpuColorAlpha(i, 0.1))
            .attr('stroke', gpuColor(i))
            .attr('stroke-width', 1.5)
            .style('filter', 'url(#dp-glow)');

        // GPU header
        card.append('text')
            .attr('x', gpuCardW / 2)
            .attr('y', 22)
            .attr('text-anchor', 'middle')
            .attr('font-size', '12px')
            .attr('font-weight', '700')
            .attr('fill', gpuColor(i))
            .text(`GPU ${i}`);

        // Micro-batch info
        card.append('text')
            .attr('x', gpuCardW / 2)
            .attr('y', 40)
            .attr('text-anchor', 'middle')
            .attr('font-family', "'JetBrains Mono', monospace")
            .attr('font-size', '10px')
            .attr('fill', 'rgba(255,255,255,0.5)')
            .text(`${microBatch} samples`);

        // Forward bar
        const barY = 55;
        const barH = 18;
        card.append('rect')
            .attr('x', 10)
            .attr('y', barY)
            .attr('width', gpuCardW - 20)
            .attr('height', barH)
            .attr('rx', 4)
            .attr('fill', gpuColorAlpha(i, 0.5));

        card.append('text')
            .attr('x', gpuCardW / 2)
            .attr('y', barY + 13)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('font-weight', '600')
            .attr('fill', '#fff')
            .text('Forward →');

        // Backward bar
        card.append('rect')
            .attr('x', 10)
            .attr('y', barY + barH + 6)
            .attr('width', gpuCardW - 20)
            .attr('height', barH)
            .attr('rx', 4)
            .attr('fill', gpuColorAlpha(i, 0.3));

        card.append('text')
            .attr('x', gpuCardW / 2)
            .attr('y', barY + barH + 6 + 13)
            .attr('text-anchor', 'middle')
            .attr('font-size', '10px')
            .attr('font-weight', '600')
            .attr('fill', '#fff')
            .text('← Backward');

        // Model label
        card.append('text')
            .attr('x', gpuCardW / 2)
            .attr('y', gpuCardH - 10)
            .attr('text-anchor', 'middle')
            .attr('font-size', '9px')
            .attr('fill', 'rgba(255,255,255,0.3)')
            .text('Full Model Copy');
    }

    // ── 4. All-Reduce ──
    const arY = splitY + gpuCardH + 30;
    const arGroup = g.append('g')
        .attr('transform', `translate(0, ${arY})`);

    // Connection lines between all GPUs (mesh)
    for (let i = 0; i < numGpus; i++) {
        for (let j = i + 1; j < numGpus; j++) {
            const x1 = startX + i * (gpuCardW + 16) + gpuCardW / 2;
            const x2 = startX + j * (gpuCardW + 16) + gpuCardW / 2;
            arGroup.append('line')
                .attr('x1', x1)
                .attr('y1', 0)
                .attr('x2', x2)
                .attr('y2', 0)
                .attr('class', 'sync-line');
        }
    }

    // GPU dots on the sync line
    for (let i = 0; i < numGpus; i++) {
        const x = startX + i * (gpuCardW + 16) + gpuCardW / 2;
        arGroup.append('circle')
            .attr('cx', x)
            .attr('cy', 0)
            .attr('r', 8)
            .attr('fill', gpuColor(i))
            .style('filter', 'url(#dp-glow)');
    }

    arGroup.append('text')
        .attr('x', w / 2)
        .attr('y', 25)
        .attr('text-anchor', 'middle')
        .attr('font-size', '12px')
        .attr('fill', '#76b900')
        .attr('font-weight', '600')
        .text('All-Reduce: Synchronize Gradients');

    // Animation
    if (animate) {
        const tl = gsap.timeline();

        tl.from(batchGroup.node(), { opacity: 0, y: -20, duration: 0.5, ease: 'power2.out' });

        gpuSection.selectAll('g').each(function (_, i) {
            tl.from(this, { opacity: 0, y: 30, duration: 0.4, ease: 'back.out(1.5)' }, `-=${i > 0 ? 0.2 : 0}`);
        });

        tl.from(arGroup.node(), { opacity: 0, scale: 0.8, duration: 0.5, ease: 'power2.out' });

        // Pulse sync dots
        arGroup.selectAll('circle').each(function (_, i) {
            tl.to(this, { attr: { r: 12 }, duration: 0.2, yoyo: true, repeat: 1 }, `-=${i > 0 ? 0.1 : 0}`);
        });
    }

    // Metrics
    renderDPMetrics(metrics);
}

function renderDPMetrics(metrics) {
    const container = document.getElementById('dp-metrics');
    container.innerHTML = `
        <div class="metric-tile">
            <div class="metric-value">${metrics.microBatchSize}</div>
            <div class="metric-label">Micro-batch / GPU</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">${metrics.effectiveBatchSize}</div>
            <div class="metric-label">Effective Batch Size</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">${metrics.accumSteps}</div>
            <div class="metric-label">Accumulation Steps</div>
        </div>
        <div class="metric-tile">
            <div class="metric-value">${metrics.communicationRounds}</div>
            <div class="metric-label">Comm. Rounds</div>
        </div>
    `;
}
