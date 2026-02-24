/**
 * Megatron Parallelism Visualizer — Main Application
 *
 * Wires tabs, controls, and visualizers together.
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

            // Re-render current viz on tab switch
            if (target === 'tensor') renderCurrentTP();
            if (target === 'pipeline') renderCurrentPP();
            if (target === 'data') renderCurrentDP();
        });
    });

    // ── Slider Bindings ──

    function bindSlider(id, onChange) {
        const slider = document.getElementById(id);
        const valDisplay = document.getElementById(`${id}-val`);
        slider.addEventListener('input', () => {
            valDisplay.textContent = slider.value;
            onChange();
        });
    }

    // ── Toggle Bindings ──

    let tpType = 'column';
    document.querySelectorAll('#panel-tensor .toggle').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#panel-tensor .toggle').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            tpType = btn.dataset.value;
            renderCurrentTP();
        });
    });

    // ── Tensor Parallelism ──

    function renderCurrentTP(animate = false) {
        const hiddenSize = parseInt(document.getElementById('tp-hidden-size').value);
        const numGpus = parseInt(document.getElementById('tp-num-gpus').value);
        renderTensorParallel('tp-viz', hiddenSize, numGpus, tpType, animate);
    }

    bindSlider('tp-hidden-size', () => renderCurrentTP());
    bindSlider('tp-num-gpus', () => renderCurrentTP());

    document.getElementById('tp-animate').addEventListener('click', () => {
        renderCurrentTP(true);
    });

    // ── Pipeline Parallelism ──

    function renderCurrentPP(animate = false) {
        const layers = parseInt(document.getElementById('pp-layers').value);
        const gpus = parseInt(document.getElementById('pp-gpus').value);
        const mbs = parseInt(document.getElementById('pp-microbatches').value);
        renderPipelineParallel('pp-viz', layers, gpus, mbs, animate);
    }

    bindSlider('pp-layers', () => renderCurrentPP());
    bindSlider('pp-gpus', () => renderCurrentPP());
    bindSlider('pp-microbatches', () => renderCurrentPP());

    document.getElementById('pp-animate').addEventListener('click', () => {
        renderCurrentPP(true);
    });

    // ── Data Parallelism ──

    function renderCurrentDP(animate = false) {
        const batch = parseInt(document.getElementById('dp-batch').value);
        const gpus = parseInt(document.getElementById('dp-gpus').value);
        const accum = parseInt(document.getElementById('dp-accum').value);
        renderDataParallel('dp-viz', batch, gpus, accum, animate);
    }

    bindSlider('dp-batch', () => renderCurrentDP());
    bindSlider('dp-gpus', () => renderCurrentDP());
    bindSlider('dp-accum', () => renderCurrentDP());

    document.getElementById('dp-animate').addEventListener('click', () => {
        renderCurrentDP(true);
    });

    // ── Initial Render ──
    renderCurrentTP();

    // ── Resize Handler ──
    let resizeTimer;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            const activeTab = document.querySelector('.tab.active').dataset.tab;
            if (activeTab === 'tensor') renderCurrentTP();
            if (activeTab === 'pipeline') renderCurrentPP();
            if (activeTab === 'data') renderCurrentDP();
        }, 250);
    });
});
