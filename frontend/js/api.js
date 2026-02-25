/**
 * Modal API Client
 * Connects the frontend to the deployed Modal endpoints.
 */

const API_BASE = 'https://nsudhanva--megatron-viz';

const ModalAPI = {
    endpoints: {
        tensorParallel: `${API_BASE}-api-tensor-parallel.modal.run`,
        pipelineParallel: `${API_BASE}-api-pipeline-parallel.modal.run`,
        dataParallel: `${API_BASE}-api-data-parallel.modal.run`,
        gpuBenchmark: `${API_BASE}-api-gpu-benchmark.modal.run`,
    },

    /**
     * Fetch from a Modal endpoint and return JSON + timing info.
     */
    async fetch(endpoint, params = {}) {
        const url = new URL(endpoint);
        Object.entries(params).forEach(([k, v]) => url.searchParams.set(k, v));

        const start = performance.now();
        const res = await fetch(url.toString());
        const elapsed = performance.now() - start;

        if (!res.ok) {
            throw new Error(`API returned ${res.status}: ${res.statusText}`);
        }

        const data = await res.json();
        return { data, elapsed: Math.round(elapsed) };
    },

    async tensorParallel(hiddenSize, numGpus, type) {
        return this.fetch(this.endpoints.tensorParallel, {
            hidden_size: hiddenSize,
            num_gpus: numGpus,
            parallelism_type: type,
        });
    },

    async pipelineParallel(numLayers, numGpus, numMicrobatches) {
        return this.fetch(this.endpoints.pipelineParallel, {
            num_layers: numLayers,
            num_gpus: numGpus,
            num_microbatches: numMicrobatches,
        });
    },

    async dataParallel(batchSize, numGpus, accumSteps) {
        return this.fetch(this.endpoints.dataParallel, {
            batch_size: batchSize,
            num_gpus: numGpus,
            gradient_accumulation_steps: accumSteps,
        });
    },

    async gpuBenchmark(hiddenSize, numIterations) {
        return this.fetch(this.endpoints.gpuBenchmark, {
            hidden_size: hiddenSize,
            num_iterations: numIterations,
        });
    },

    /**
     * Quick health check — hit TP endpoint with minimal params.
     */
    async healthCheck() {
        try {
            await this.fetch(this.endpoints.tensorParallel, {
                hidden_size: 128,
                num_gpus: 2,
            });
            return true;
        } catch {
            return false;
        }
    },
};
