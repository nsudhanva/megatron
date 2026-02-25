/**
 * Main Application — Wires controls, tabs, cloud buttons, and benchmark.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ── Tab Navigation ──
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const target = tab.dataset.tab;
            tabs.forEach(t => t.classList.remove('active'));
            panels.forEach(p => p.classList.remove('active'));
            tab.classList.add('active');
            document.getElementById(`panel-${target}`).classList.add('active');
        });
    });

    // ── Slider Value Display ──
    document.querySelectorAll('input[type="range"]').forEach(slider => {
        const valSpan = document.getElementById(`${slider.id}-val`);
        if (valSpan) {
            slider.addEventListener('input', () => { valSpan.textContent = slider.value; });
        }
    });

    // ── Toggle Groups ──
    document.querySelectorAll('.toggle-group').forEach(group => {
        group.querySelectorAll('.toggle').forEach(btn => {
            btn.addEventListener('click', () => {
                group.querySelectorAll('.toggle').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
            });
        });
    });

    // ── Initial Renders ──
    function renderTP() {
        const hs = parseInt(document.getElementById('tp-hidden-size').value);
        const gpus = parseInt(document.getElementById('tp-num-gpus').value);
        const type = document.querySelector('#panel-tensor .toggle.active').dataset.value;
        const steps = Simulator.tensorParallel(hs, gpus, type);
        Visualizer.tensorParallel('#tp-viz', steps, gpus);
        renderTPInfo(steps);
    }

    function renderPP() {
        const layers = parseInt(document.getElementById('pp-layers').value);
        const gpus = parseInt(document.getElementById('pp-gpus').value);
        const mb = parseInt(document.getElementById('pp-microbatches').value);
        const result = Simulator.pipelineParallel(layers, gpus, mb);
        Visualizer.pipelineParallel('#pp-viz', result, gpus);
        renderPPMetrics(result.metrics);
    }

    function renderDP() {
        const batch = parseInt(document.getElementById('dp-batch').value);
        const gpus = parseInt(document.getElementById('dp-gpus').value);
        const accum = parseInt(document.getElementById('dp-accum').value);
        const result = Simulator.dataParallel(batch, gpus, accum);
        Visualizer.dataParallel('#dp-viz', result, gpus);
        renderDPMetrics(result.metrics);
    }

    function renderTPInfo(steps) {
        const container = document.getElementById('tp-info');
        container.innerHTML = steps.map(s => `
            <div class="info-card">
                <div class="step-number">${s.id}</div>
                <h4>${s.label}</h4>
                <p>${s.description}</p>
                ${s.shards.length ? `<div class="gpu-list">${s.shards.map((sh, i) => `<div class="gpu-item" style="border-left: 3px solid var(--gpu-${i})">GPU ${i}: ${sh.rows}×${sh.cols}</div>`).join('')
                }</div>` : ''}
            </div>
        `).join('');
    }

    function renderPPMetrics(m) {
        document.getElementById('pp-metrics').innerHTML = [
            { label: 'Layers / Stage', value: m.layersPerStage },
            { label: 'Bubble Ratio', value: (m.bubbleRatio * 100).toFixed(1) + '%' },
            { label: 'Active Slots', value: m.activeSlots },
            { label: 'Bubble Slots', value: m.bubbleSlots },
        ].map(t => `
            <div class="metric-tile">
                <div class="metric-value">${t.value}</div>
                <div class="metric-label">${t.label}</div>
            </div>
        `).join('');
    }

    function renderDPMetrics(m) {
        document.getElementById('dp-metrics').innerHTML = [
            { label: 'Micro-batch Size', value: m.microBatchSize },
            { label: 'Effective Batch', value: m.effectiveBatchSize },
            { label: 'Comm Rounds', value: m.communicationRounds },
        ].map(t => `
            <div class="metric-tile">
                <div class="metric-value">${t.value}</div>
                <div class="metric-label">${t.label}</div>
            </div>
        `).join('');
    }

    // ── Slider Change → Re-render ──
    ['tp-hidden-size', 'tp-num-gpus'].forEach(id =>
        document.getElementById(id).addEventListener('input', renderTP)
    );
    document.querySelectorAll('#panel-tensor .toggle').forEach(btn =>
        btn.addEventListener('click', renderTP)
    );
    ['pp-layers', 'pp-gpus', 'pp-microbatches'].forEach(id =>
        document.getElementById(id).addEventListener('input', renderPP)
    );
    ['dp-batch', 'dp-gpus', 'dp-accum'].forEach(id =>
        document.getElementById(id).addEventListener('input', renderDP)
    );

    // ── Animate Buttons ──
    document.getElementById('tp-animate').addEventListener('click', () => {
        renderTP();
        gsap.from('#tp-viz .matrix-cell', {
            scale: 0, opacity: 0, duration: 0.4, stagger: 0.01, ease: 'back.out(1.7)',
        });
    });

    document.getElementById('pp-animate').addEventListener('click', () => {
        renderPP();
        gsap.from('#pp-viz .pipeline-cell, #pp-viz .bubble-cell', {
            scaleY: 0, opacity: 0, duration: 0.3, stagger: 0.02, ease: 'power2.out',
        });
    });

    document.getElementById('dp-animate').addEventListener('click', () => {
        renderDP();
        gsap.from('#dp-viz .data-batch', {
            scale: 0, opacity: 0, duration: 0.5, stagger: 0.05, ease: 'elastic.out(1, 0.5)',
        });
    });

    // ── Cloud Buttons ──
    async function cloudCall(btnId, responseDivId, timeSpanId, bodyPreId, apiFn) {
        const btn = document.getElementById(btnId);
        const responseDiv = document.getElementById(responseDivId);
        const timeSpan = document.getElementById(timeSpanId);
        const bodyPre = document.getElementById(bodyPreId);

        btn.classList.add('loading');
        btn.innerHTML = '<span>⟳</span> Calling API…';
        responseDiv.style.display = 'none';

        try {
            const result = await apiFn();
            timeSpan.textContent = `${result.elapsed}ms`;
            bodyPre.textContent = JSON.stringify(result.data, null, 2);
            responseDiv.style.display = 'block';
            gsap.from(responseDiv, { opacity: 0, y: 20, duration: 0.5, ease: 'power2.out' });
        } catch (err) {
            timeSpan.textContent = 'Error';
            bodyPre.textContent = `Error: ${err.message}`;
            responseDiv.style.display = 'block';
        } finally {
            btn.classList.remove('loading');
            btn.innerHTML = '<span>☁</span> Run on Cloud';
        }
    }

    document.getElementById('tp-cloud').addEventListener('click', () => {
        const hs = parseInt(document.getElementById('tp-hidden-size').value);
        const gpus = parseInt(document.getElementById('tp-num-gpus').value);
        const type = document.querySelector('#panel-tensor .toggle.active').dataset.value;
        cloudCall('tp-cloud', 'tp-response', 'tp-response-time', 'tp-response-body',
            () => ModalAPI.tensorParallel(hs, gpus, type));
    });

    document.getElementById('pp-cloud').addEventListener('click', () => {
        const layers = parseInt(document.getElementById('pp-layers').value);
        const gpus = parseInt(document.getElementById('pp-gpus').value);
        const mb = parseInt(document.getElementById('pp-microbatches').value);
        cloudCall('pp-cloud', 'pp-response', 'pp-response-time', 'pp-response-body',
            () => ModalAPI.pipelineParallel(layers, gpus, mb));
    });

    document.getElementById('dp-cloud').addEventListener('click', () => {
        const batch = parseInt(document.getElementById('dp-batch').value);
        const gpus = parseInt(document.getElementById('dp-gpus').value);
        const accum = parseInt(document.getElementById('dp-accum').value);
        cloudCall('dp-cloud', 'dp-response', 'dp-response-time', 'dp-response-body',
            () => ModalAPI.dataParallel(batch, gpus, accum));
    });

    // ── GPU Benchmark ──
    document.getElementById('bench-run').addEventListener('click', async () => {
        const btn = document.getElementById('bench-run');
        const resultsDiv = document.getElementById('bench-results');
        const hs = parseInt(document.getElementById('bench-hidden').value);
        const iters = parseInt(document.getElementById('bench-iters').value);

        btn.classList.add('loading');
        btn.innerHTML = '<span>⟳</span> Running on T4 GPU…';
        resultsDiv.innerHTML = `
            <div class="bench-placeholder">
                <p>⟳ Running ${iters} iterations of ${hs}×${hs} matmul on T4 GPU…</p>
                <p class="bench-note">This may take ~30s on first run (GPU cold start).</p>
            </div>
        `;

        try {
            const { data, elapsed } = await ModalAPI.gpuBenchmark(hs, iters);

            resultsDiv.innerHTML = `
                <div style="width: 100%">
                    <div class="bench-grid">
                        <div class="bench-card">
                            <div class="bench-value" style="color: #00b4d8">${data.full_matmul_ms.toFixed(3)}</div>
                            <div class="bench-unit">ms</div>
                            <div class="bench-label">Full Matmul (${hs}×${hs})</div>
                        </div>
                        <div class="bench-card">
                            <div class="bench-value" style="color: #f4a261">${data.half_matmul_ms.toFixed(3)}</div>
                            <div class="bench-unit">ms</div>
                            <div class="bench-label">Half Matmul (${hs}×${hs / 2})</div>
                        </div>
                        <div class="bench-card highlight">
                            <div class="bench-value">${data.theoretical_speedup.toFixed(2)}×</div>
                            <div class="bench-label">Theoretical TP Speedup</div>
                        </div>
                        <div class="bench-card">
                            <div class="bench-value" style="color: #9b5de5">${elapsed}</div>
                            <div class="bench-unit">ms</div>
                            <div class="bench-label">API Round-trip</div>
                        </div>
                    </div>
                    <div class="bench-gpu-name">
                        ${data.gpu_name} · ${data.num_iterations} iters · via Modal Cloud
                    </div>
                </div>
            `;

            gsap.from('.bench-card', {
                scale: 0.8, opacity: 0, duration: 0.6, stagger: 0.1, ease: 'back.out(1.7)',
            });
        } catch (err) {
            resultsDiv.innerHTML = `
                <div class="bench-placeholder">
                    <p style="color: #e94560">❌ ${err.message}</p>
                    <p class="bench-note">Make sure the app is deployed: uv run modal deploy src/megatron_viz/app.py</p>
                </div>
            `;
        } finally {
            btn.classList.remove('loading');
            btn.innerHTML = '<span>⚡</span> Run on T4 GPU';
        }
    });

    // ── API Health Check ──
    (async () => {
        const badge = document.getElementById('api-status');
        try {
            const ok = await ModalAPI.healthCheck();
            if (ok) {
                badge.textContent = '☁ API: Online';
                badge.classList.add('online');
            } else {
                badge.textContent = '☁ API: Offline';
                badge.classList.add('offline');
            }
        } catch {
            badge.textContent = '☁ API: Offline';
            badge.classList.add('offline');
        }
    })();

    // ── Initial Render ──
    renderTP();
    renderPP();
    renderDP();
});
