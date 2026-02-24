"""Tests for the parallelism simulator."""

from megatron_viz.simulator import (
    simulate_data_parallel,
    simulate_pipeline_parallel,
    simulate_tensor_parallel,
)


class TestTensorParallel:
    def test_column_parallel_step_count(self):
        steps = simulate_tensor_parallel(512, num_gpus=4, parallelism_type="column")
        assert len(steps) == 5  # original, split, compute, comm, output

    def test_row_parallel_step_count(self):
        steps = simulate_tensor_parallel(512, num_gpus=4, parallelism_type="row")
        assert len(steps) == 5

    def test_column_parallel_split_shapes(self):
        steps = simulate_tensor_parallel(512, num_gpus=4, parallelism_type="column")
        split_step = steps[1]
        assert len(split_step.split_shapes) == 4
        for shape in split_step.split_shapes:
            assert shape == (512, 128)  # 512 / 4 = 128

    def test_row_parallel_split_shapes(self):
        steps = simulate_tensor_parallel(512, num_gpus=4, parallelism_type="row")
        split_step = steps[1]
        for shape in split_step.split_shapes:
            assert shape == (128, 512)

    def test_column_uses_all_gather(self):
        steps = simulate_tensor_parallel(256, num_gpus=2, parallelism_type="column")
        comm_step = steps[3]
        assert comm_step.communication == "All-Gather"

    def test_row_uses_all_reduce(self):
        steps = simulate_tensor_parallel(256, num_gpus=2, parallelism_type="row")
        comm_step = steps[3]
        assert comm_step.communication == "All-Reduce (sum)"

    def test_gpu_assignment_count(self):
        steps = simulate_tensor_parallel(1024, num_gpus=8)
        split_step = steps[1]
        assert len(split_step.gpu_assignments) == 8


class TestPipelineParallel:
    def test_basic_schedule(self):
        _steps, metrics = simulate_pipeline_parallel(
            num_layers=8, num_gpus=2, num_microbatches=4
        )
        assert metrics["layers_per_stage"] == 4

    def test_bubble_ratio_positive(self):
        _, metrics = simulate_pipeline_parallel(
            num_layers=16, num_gpus=4, num_microbatches=8
        )
        assert 0 <= metrics["bubble_ratio"] < 1

    def test_more_microbatches_produces_valid_ratio(self):
        _, metrics_few = simulate_pipeline_parallel(16, 4, num_microbatches=4)
        _, metrics_many = simulate_pipeline_parallel(16, 4, num_microbatches=16)
        # Both should produce valid bubble ratios between 0 and 1
        assert 0 <= metrics_few["bubble_ratio"] < 1
        assert 0 <= metrics_many["bubble_ratio"] < 1

    def test_has_forward_and_backward(self):
        steps, _ = simulate_pipeline_parallel(8, 2, 4)
        phases = {s.phase for s in steps}
        assert "forward" in phases
        assert "backward" in phases


class TestDataParallel:
    def test_micro_batch_size(self):
        _, metrics = simulate_data_parallel(256, num_gpus=4)
        assert metrics["micro_batch_size"] == 64

    def test_gradient_accumulation(self):
        _, metrics = simulate_data_parallel(
            128, num_gpus=2, gradient_accumulation_steps=4
        )
        assert metrics["effective_batch_size"] == 512

    def test_has_all_reduce(self):
        steps, _ = simulate_data_parallel(64, num_gpus=2)
        phases = {s.phase for s in steps}
        assert "all_reduce" in phases

    def test_all_gpus_participate(self):
        steps, _ = simulate_data_parallel(256, num_gpus=8)
        gpu_ids = {s.gpu_id for s in steps}
        assert gpu_ids == set(range(8))
