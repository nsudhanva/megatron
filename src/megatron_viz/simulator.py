"""Pure-Python simulation engine for parallelism strategies.

Computes theoretical splits, communication costs, and timing data
for Tensor, Pipeline, and Data parallelism without requiring a GPU.
"""

from dataclasses import dataclass, field


@dataclass
class TensorParallelStep:
    """A single step in the tensor parallelism simulation."""

    step_id: int
    label: str
    description: str
    gpu_assignments: dict[int, str]
    communication: str | None = None
    matrix_shape: tuple[int, int] | None = None
    split_shapes: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class PipelineParallelStep:
    """A single step in the pipeline parallelism simulation."""

    microbatch_id: int
    stage_id: int
    phase: str  # "forward" or "backward"
    time_slot: int
    is_bubble: bool = False


@dataclass
class DataParallelStep:
    """A single step in the data parallelism simulation."""

    step_id: int
    label: str
    gpu_id: int
    batch_slice: tuple[int, int]
    phase: str  # "forward", "backward", "all_reduce"


def simulate_tensor_parallel(
    hidden_size: int,
    num_gpus: int,
    parallelism_type: str = "column",
) -> list[TensorParallelStep]:
    """Simulate tensor parallelism for a linear layer.

    In Megatron-style tensor parallelism:
    - Column parallel: splits weight matrix W along columns, each GPU gets W[:,k]
    - Row parallel: splits weight matrix W along rows, each GPU gets W[k,:]

    Args:
        hidden_size: Size of the hidden dimension.
        num_gpus: Number of GPUs to split across.
        parallelism_type: Either "column" or "row".

    Returns:
        List of simulation steps showing the split and communication.
    """
    shard_size = hidden_size // num_gpus
    steps: list[TensorParallelStep] = []

    # Step 1: Original matrix
    steps.append(
        TensorParallelStep(
            step_id=0,
            label="Original Weight Matrix",
            description=f"Full weight matrix W of shape ({hidden_size}, {hidden_size})",
            gpu_assignments={0: "Full matrix on single GPU"},
            matrix_shape=(hidden_size, hidden_size),
        )
    )

    # Step 2: Split across GPUs
    gpu_assignments: dict[int, str] = {}
    split_shapes: list[tuple[int, int]] = []

    for gpu_id in range(num_gpus):
        if parallelism_type == "column":
            shape = (hidden_size, shard_size)
            gpu_assignments[gpu_id] = (
                f"W[:, {gpu_id * shard_size}:{(gpu_id + 1) * shard_size}]"
            )
        else:
            shape = (shard_size, hidden_size)
            gpu_assignments[gpu_id] = (
                f"W[{gpu_id * shard_size}:{(gpu_id + 1) * shard_size}, :]"
            )
        split_shapes.append(shape)

    steps.append(
        TensorParallelStep(
            step_id=1,
            label=f"{'Column' if parallelism_type == 'column' else 'Row'} Split",
            description=(
                f"Weight matrix split into {num_gpus} shards "
                f"({'columns' if parallelism_type == 'column' else 'rows'})"
            ),
            gpu_assignments=gpu_assignments,
            split_shapes=split_shapes,
        )
    )

    # Step 3: Parallel computation
    compute_assignments: dict[int, str] = {}
    for gpu_id in range(num_gpus):
        if parallelism_type == "column":
            compute_assignments[gpu_id] = f"Y_{gpu_id} = X @ W_{gpu_id}"
        else:
            compute_assignments[gpu_id] = f"Y_{gpu_id} = X_{gpu_id} @ W_{gpu_id}"

    steps.append(
        TensorParallelStep(
            step_id=2,
            label="Parallel MatMul",
            description="Each GPU computes its shard independently",
            gpu_assignments=compute_assignments,
        )
    )

    # Step 4: Communication
    if parallelism_type == "column":
        comm_type = "All-Gather"
        comm_desc = "Concatenate partial outputs Y_k along columns"
    else:
        comm_type = "All-Reduce (sum)"
        comm_desc = "Sum partial outputs Y_k across all GPUs"

    steps.append(
        TensorParallelStep(
            step_id=3,
            label="Communication",
            description=comm_desc,
            gpu_assignments={
                gpu_id: f"Participates in {comm_type}" for gpu_id in range(num_gpus)
            },
            communication=comm_type,
        )
    )

    # Step 5: Final output
    steps.append(
        TensorParallelStep(
            step_id=4,
            label="Final Output",
            description=f"Complete output Y of shape (batch, {hidden_size})",
            gpu_assignments={0: "Full output reconstructed"},
            matrix_shape=(1, hidden_size),
        )
    )

    return steps


def simulate_pipeline_parallel(
    num_layers: int,
    num_gpus: int,
    num_microbatches: int,
) -> tuple[list[PipelineParallelStep], dict[str, float]]:
    """Simulate pipeline parallelism with 1F1B schedule.

    In Megatron's pipeline parallelism:
    - Model layers are split evenly across GPU stages
    - Uses 1F1B (one forward, one backward) interleaving
    - "Bubbles" represent idle GPU time

    Args:
        num_layers: Total number of transformer layers.
        num_gpus: Number of pipeline stages.
        num_microbatches: Number of micro-batches to schedule.

    Returns:
        Tuple of (schedule steps, metrics dict with bubble ratio etc).
    """
    layers_per_stage = num_layers // num_gpus
    steps: list[PipelineParallelStep] = []
    time_slot = 0

    # Warmup phase: fill the pipeline with forward passes
    for mb in range(min(num_microbatches, num_gpus)):
        for stage in range(num_gpus):
            steps.append(
                PipelineParallelStep(
                    microbatch_id=mb,
                    stage_id=stage,
                    phase="forward",
                    time_slot=time_slot + stage,
                )
            )
        time_slot += 1

    # Steady state: 1F1B
    for mb in range(num_gpus, num_microbatches):
        # One backward pass for an earlier microbatch
        backward_mb = mb - num_gpus
        for stage in reversed(range(num_gpus)):
            steps.append(
                PipelineParallelStep(
                    microbatch_id=backward_mb,
                    stage_id=stage,
                    phase="backward",
                    time_slot=time_slot,
                )
            )
            time_slot += 1

        # One forward pass for the current microbatch
        for stage in range(num_gpus):
            steps.append(
                PipelineParallelStep(
                    microbatch_id=mb,
                    stage_id=stage,
                    phase="forward",
                    time_slot=time_slot,
                )
            )
            time_slot += 1

    # Cooldown: remaining backward passes
    for mb in range(max(0, num_microbatches - num_gpus), num_microbatches):
        for stage in reversed(range(num_gpus)):
            steps.append(
                PipelineParallelStep(
                    microbatch_id=mb,
                    stage_id=stage,
                    phase="backward",
                    time_slot=time_slot,
                )
            )
            time_slot += 1

    # Calculate bubble ratio
    total_slots = time_slot * num_gpus
    active_slots = len(steps)
    bubble_slots = total_slots - active_slots
    bubble_ratio = bubble_slots / total_slots if total_slots > 0 else 0.0

    metrics = {
        "total_time_slots": time_slot,
        "layers_per_stage": layers_per_stage,
        "bubble_ratio": bubble_ratio,
        "active_slots": active_slots,
        "bubble_slots": bubble_slots,
        "total_slots": total_slots,
    }

    # Mark bubble slots
    occupied: set[tuple[int, int]] = set()
    for step in steps:
        occupied.add((step.time_slot, step.stage_id))

    for t in range(time_slot):
        for s in range(num_gpus):
            if (t, s) not in occupied:
                steps.append(
                    PipelineParallelStep(
                        microbatch_id=-1,
                        stage_id=s,
                        phase="bubble",
                        time_slot=t,
                        is_bubble=True,
                    )
                )

    return steps, metrics


def simulate_data_parallel(
    batch_size: int,
    num_gpus: int,
    gradient_accumulation_steps: int = 1,
) -> tuple[list[DataParallelStep], dict[str, float]]:
    """Simulate data parallelism.

    In data parallelism:
    - Each GPU holds a full copy of the model
    - Training batch is split evenly across GPUs
    - After backward pass, gradients are all-reduced

    Args:
        batch_size: Total batch size.
        num_gpus: Number of data-parallel replicas.
        gradient_accumulation_steps: Number of accumulation steps before sync.

    Returns:
        Tuple of (simulation steps, metrics dict).
    """
    micro_batch = batch_size // num_gpus
    steps: list[DataParallelStep] = []
    step_counter = 0

    for accum_step in range(gradient_accumulation_steps):
        # Forward pass on each GPU
        for gpu_id in range(num_gpus):
            start = gpu_id * micro_batch + accum_step * batch_size
            end = start + micro_batch
            steps.append(
                DataParallelStep(
                    step_id=step_counter,
                    label=f"Forward (accum {accum_step})",
                    gpu_id=gpu_id,
                    batch_slice=(start, end),
                    phase="forward",
                )
            )
            step_counter += 1

        # Backward pass on each GPU
        for gpu_id in range(num_gpus):
            start = gpu_id * micro_batch + accum_step * batch_size
            end = start + micro_batch
            steps.append(
                DataParallelStep(
                    step_id=step_counter,
                    label=f"Backward (accum {accum_step})",
                    gpu_id=gpu_id,
                    batch_slice=(start, end),
                    phase="backward",
                )
            )
            step_counter += 1

    # All-reduce after accumulation
    for gpu_id in range(num_gpus):
        steps.append(
            DataParallelStep(
                step_id=step_counter,
                label="All-Reduce Gradients",
                gpu_id=gpu_id,
                batch_slice=(0, batch_size),
                phase="all_reduce",
            )
        )
        step_counter += 1

    communication_volume = batch_size * num_gpus  # simplified metric
    metrics = {
        "micro_batch_size": micro_batch,
        "total_steps": step_counter,
        "communication_volume": communication_volume,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
    }

    return steps, metrics
