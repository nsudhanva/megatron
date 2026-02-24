/**
 * Client-side parallelism simulator.
 * Mirrors the Python simulator logic for instant interactions.
 */

const Simulator = {
    /**
     * Simulate tensor parallelism — splitting a weight matrix across GPUs.
     */
    tensorParallel(hiddenSize, numGpus, type = 'column') {
        const shardSize = Math.floor(hiddenSize / numGpus);
        const steps = [];

        // Step 0: Original matrix
        steps.push({
            id: 0,
            label: 'Original Weight Matrix',
            description: `Full weight matrix W (${hiddenSize} × ${hiddenSize})`,
            phase: 'original',
            matrix: { rows: hiddenSize, cols: hiddenSize },
            shards: [],
        });

        // Step 1: Split
        const shards = [];
        for (let g = 0; g < numGpus; g++) {
            if (type === 'column') {
                shards.push({
                    gpu: g,
                    rows: hiddenSize,
                    cols: shardSize,
                    label: `W[:, ${g * shardSize}:${(g + 1) * shardSize}]`,
                });
            } else {
                shards.push({
                    gpu: g,
                    rows: shardSize,
                    cols: hiddenSize,
                    label: `W[${g * shardSize}:${(g + 1) * shardSize}, :]`,
                });
            }
        }
        steps.push({
            id: 1,
            label: `${type === 'column' ? 'Column' : 'Row'} Split`,
            description: `Split into ${numGpus} shards across GPUs`,
            phase: 'split',
            shards,
        });

        // Step 2: Parallel compute
        steps.push({
            id: 2,
            label: 'Parallel MatMul',
            description: 'Each GPU computes its shard independently',
            phase: 'compute',
            shards: shards.map(s => ({
                ...s,
                label: type === 'column'
                    ? `Y_${s.gpu} = X @ W_${s.gpu}`
                    : `Y_${s.gpu} = X_${s.gpu} @ W_${s.gpu}`,
            })),
        });

        // Step 3: Communication
        const commType = type === 'column' ? 'All-Gather' : 'All-Reduce';
        steps.push({
            id: 3,
            label: 'Communication',
            description: type === 'column'
                ? 'Concatenate partial outputs along columns'
                : 'Sum partial outputs across all GPUs',
            phase: 'communicate',
            communication: commType,
            shards,
        });

        // Step 4: Output
        steps.push({
            id: 4,
            label: 'Final Output',
            description: `Complete output Y (batch × ${hiddenSize})`,
            phase: 'output',
            matrix: { rows: 1, cols: hiddenSize },
            shards: [],
        });

        return steps;
    },

    /**
     * Simulate pipeline parallelism with a simplified 1F1B schedule.
     */
    pipelineParallel(numLayers, numGpus, numMicrobatches) {
        const layersPerStage = Math.floor(numLayers / numGpus);
        const schedule = [];

        // Build a grid: time × stage
        const totalTime = numMicrobatches + numGpus - 1;
        const grid = [];

        // Forward passes
        for (let mb = 0; mb < numMicrobatches; mb++) {
            for (let stage = 0; stage < numGpus; stage++) {
                const t = mb + stage;
                grid.push({
                    time: t,
                    stage,
                    microbatch: mb,
                    phase: 'forward',
                    isBubble: false,
                });
            }
        }

        // Backward passes
        for (let mb = 0; mb < numMicrobatches; mb++) {
            for (let stage = numGpus - 1; stage >= 0; stage--) {
                const t = totalTime + mb + (numGpus - 1 - stage);
                grid.push({
                    time: t,
                    stage,
                    microbatch: mb,
                    phase: 'backward',
                    isBubble: false,
                });
            }
        }

        // Find max time
        const maxTime = Math.max(...grid.map(g => g.time)) + 1;

        // Mark bubbles
        const occupied = new Set(grid.map(g => `${g.time}-${g.stage}`));
        for (let t = 0; t < maxTime; t++) {
            for (let s = 0; s < numGpus; s++) {
                if (!occupied.has(`${t}-${s}`)) {
                    grid.push({
                        time: t,
                        stage: s,
                        microbatch: -1,
                        phase: 'bubble',
                        isBubble: true,
                    });
                }
            }
        }

        const totalSlots = maxTime * numGpus;
        const activeSlots = grid.filter(g => !g.isBubble).length;
        const bubbleSlots = totalSlots - activeSlots;

        return {
            grid,
            maxTime,
            metrics: {
                layersPerStage,
                bubbleRatio: bubbleSlots / totalSlots,
                activeSlots,
                bubbleSlots,
                totalSlots,
            },
        };
    },

    /**
     * Simulate data parallelism with batch splitting and gradient sync.
     */
    dataParallel(batchSize, numGpus, accumSteps = 1) {
        const microBatch = Math.floor(batchSize / numGpus);
        const phases = [];

        for (let acc = 0; acc < accumSteps; acc++) {
            // Forward
            for (let gpu = 0; gpu < numGpus; gpu++) {
                phases.push({
                    gpu,
                    phase: 'forward',
                    accumStep: acc,
                    batchStart: gpu * microBatch,
                    batchEnd: (gpu + 1) * microBatch,
                });
            }
            // Backward
            for (let gpu = 0; gpu < numGpus; gpu++) {
                phases.push({
                    gpu,
                    phase: 'backward',
                    accumStep: acc,
                    batchStart: gpu * microBatch,
                    batchEnd: (gpu + 1) * microBatch,
                });
            }
        }

        // All-reduce
        for (let gpu = 0; gpu < numGpus; gpu++) {
            phases.push({
                gpu,
                phase: 'all_reduce',
                accumStep: -1,
                batchStart: 0,
                batchEnd: batchSize,
            });
        }

        return {
            phases,
            metrics: {
                microBatchSize: microBatch,
                effectiveBatchSize: batchSize * accumSteps,
                communicationRounds: 1,
                accumSteps,
            },
        };
    },
};
