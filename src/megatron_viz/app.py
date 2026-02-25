"""Modal app for the Megatron Parallelism Visualizer.

Serves the frontend and provides GPU-powered demo endpoints.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import modal

app = modal.App("megatron-viz")

frontend_path = Path(__file__).parent.parent.parent / "frontend"
pkg_path = Path(__file__).parent  # src/megatron_viz/

# Base image for CPU endpoints — add_local_python_source goes LAST
base_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source(
    "megatron_viz"
)

# GPU image: need copy=True since pip_install runs after source addition
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch>=2.6", "numpy>=2.2")
    .add_local_python_source("megatron_viz")
)


def _serialize_steps(steps: list[Any]) -> list[dict[str, Any]]:
    """Convert dataclass steps to JSON-serializable dicts."""
    result = []
    for step in steps:
        d = asdict(step)
        # Convert tuple values to lists for JSON
        for key, val in d.items():
            if isinstance(val, tuple):
                d[key] = list(val)
            elif isinstance(val, list):
                d[key] = [list(v) if isinstance(v, tuple) else v for v in val]
            elif isinstance(val, dict):
                d[key] = {
                    str(k): list(v) if isinstance(v, tuple) else v
                    for k, v in val.items()
                }
        result.append(d)
    return result


@app.function(image=base_image)
@modal.fastapi_endpoint(method="GET")
def api_tensor_parallel(
    hidden_size: int = 512,
    num_gpus: int = 4,
    parallelism_type: str = "column",
) -> dict[str, Any]:
    """Simulate tensor parallelism and return visualization data."""
    from megatron_viz.simulator import simulate_tensor_parallel

    steps = simulate_tensor_parallel(hidden_size, num_gpus, parallelism_type)
    return {
        "strategy": "tensor_parallel",
        "config": {
            "hidden_size": hidden_size,
            "num_gpus": num_gpus,
            "parallelism_type": parallelism_type,
        },
        "steps": _serialize_steps(steps),
    }


@app.function(image=base_image)
@modal.fastapi_endpoint(method="GET")
def api_pipeline_parallel(
    num_layers: int = 24,
    num_gpus: int = 4,
    num_microbatches: int = 8,
) -> dict[str, Any]:
    """Simulate pipeline parallelism and return visualization data."""
    from megatron_viz.simulator import simulate_pipeline_parallel

    steps, metrics = simulate_pipeline_parallel(num_layers, num_gpus, num_microbatches)
    return {
        "strategy": "pipeline_parallel",
        "config": {
            "num_layers": num_layers,
            "num_gpus": num_gpus,
            "num_microbatches": num_microbatches,
        },
        "steps": _serialize_steps(steps),
        "metrics": metrics,
    }


@app.function(image=base_image)
@modal.fastapi_endpoint(method="GET")
def api_data_parallel(
    batch_size: int = 256,
    num_gpus: int = 4,
    gradient_accumulation_steps: int = 1,
) -> dict[str, Any]:
    """Simulate data parallelism and return visualization data."""
    from megatron_viz.simulator import simulate_data_parallel

    steps, metrics = simulate_data_parallel(
        batch_size, num_gpus, gradient_accumulation_steps
    )
    return {
        "strategy": "data_parallel",
        "config": {
            "batch_size": batch_size,
            "num_gpus": num_gpus,
            "gradient_accumulation_steps": gradient_accumulation_steps,
        },
        "steps": _serialize_steps(steps),
        "metrics": metrics,
    }


@app.function(image=gpu_image, gpu="T4")
@modal.fastapi_endpoint(method="GET")
def api_gpu_benchmark(
    hidden_size: int = 1024,
    num_iterations: int = 100,
) -> dict[str, Any]:
    """Run actual matrix operations on a T4 GPU and return timing data.

    This uses real GPU compute via Modal to demonstrate
    the actual speedup from parallelism.
    """
    import time

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Single GPU matmul
    x = torch.randn(hidden_size, hidden_size, device=device)
    w = torch.randn(hidden_size, hidden_size, device=device)

    # Warmup
    for _ in range(10):
        _ = x @ w
    torch.cuda.synchronize()

    # Benchmark full matmul
    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = x @ w
    torch.cuda.synchronize()
    full_time = (time.perf_counter() - start) / num_iterations

    # Benchmark "simulated shard" (half-size matmul)
    x_half = torch.randn(hidden_size, hidden_size // 2, device=device)
    w_half = torch.randn(hidden_size // 2, hidden_size, device=device)

    for _ in range(10):
        _ = x_half @ w_half
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = x_half @ w_half
    torch.cuda.synchronize()
    half_time = (time.perf_counter() - start) / num_iterations

    return {
        "device": str(device),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        ),
        "hidden_size": hidden_size,
        "num_iterations": num_iterations,
        "full_matmul_ms": full_time * 1000,
        "half_matmul_ms": half_time * 1000,
        "theoretical_speedup": full_time / half_time if half_time > 0 else 0,
        "note": (
            "Half-size matmul simulates tensor parallelism with 2 GPUs. "
            "Real TP would add communication overhead."
        ),
    }


@app.function(image=gpu_image, gpu="T4:2")
@modal.fastapi_endpoint(method="GET")
def api_multi_gpu_benchmark(
    hidden_size: int = 2048,
    num_iterations: int = 50,
) -> dict[str, Any]:
    """Run real tensor parallelism across 2 T4 GPUs.

    Performs column-parallel matmul: each GPU computes half the output columns,
    then results are gathered. Compares against single-GPU baseline.
    """
    import time

    import torch

    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]

    # ── Single GPU baseline ──
    device_0 = torch.device("cuda:0")
    x = torch.randn(hidden_size, hidden_size, device=device_0)
    w = torch.randn(hidden_size, hidden_size, device=device_0)

    for _ in range(10):
        _ = x @ w
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = x @ w
    torch.cuda.synchronize()
    single_gpu_time = (time.perf_counter() - start) / num_iterations

    # ── Real 2-GPU Tensor Parallel (column split) ──
    # Split weight matrix columns across GPUs
    half = hidden_size // 2
    w0 = w[:, :half].to("cuda:0")  # GPU 0 gets left columns
    w1 = w[:, half:].to("cuda:1")  # GPU 1 gets right columns
    x1 = x.to("cuda:1")  # input replicated on GPU 1

    # Warmup
    for _ in range(10):
        y0 = x @ w0  # GPU 0: partial matmul
        y1 = x1 @ w1  # GPU 1: partial matmul
        y1_on_0 = y1.to("cuda:0")  # Gather to GPU 0
        _ = torch.cat([y0, y1_on_0], dim=1)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        y0 = x @ w0
        y1 = x1 @ w1
        torch.cuda.synchronize()  # wait for both GPUs
        y1_on_0 = y1.to("cuda:0")  # communication: GPU1 -> GPU0
        _ = torch.cat([y0, y1_on_0], dim=1)
    torch.cuda.synchronize()
    two_gpu_time = (time.perf_counter() - start) / num_iterations

    # ── Communication overhead measurement ──
    dummy = torch.randn(hidden_size, half, device="cuda:1")
    for _ in range(10):
        _ = dummy.to("cuda:0")
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_iterations):
        _ = dummy.to("cuda:1").to("cuda:0")
    torch.cuda.synchronize()
    comm_time = (time.perf_counter() - start) / num_iterations

    speedup = single_gpu_time / two_gpu_time if two_gpu_time > 0 else 0
    ideal_speedup = 2.0
    efficiency = (speedup / ideal_speedup) * 100

    return {
        "num_gpus": num_gpus,
        "gpu_names": gpu_names,
        "hidden_size": hidden_size,
        "num_iterations": num_iterations,
        "single_gpu_ms": round(single_gpu_time * 1000, 4),
        "two_gpu_tp_ms": round(two_gpu_time * 1000, 4),
        "communication_ms": round(comm_time * 1000, 4),
        "actual_speedup": round(speedup, 3),
        "ideal_speedup": ideal_speedup,
        "parallel_efficiency": round(efficiency, 1),
        "breakdown": {
            "compute_per_gpu_ms": round((two_gpu_time - comm_time) * 1000, 4),
            "communication_ms": round(comm_time * 1000, 4),
            "communication_pct": round(
                (comm_time / two_gpu_time) * 100 if two_gpu_time > 0 else 0, 1
            ),
        },
    }


@app.local_entrypoint()
def main() -> None:
    """Run a local simulation demo."""
    from rich.console import Console
    from rich.table import Table

    from megatron_viz.simulator import (
        simulate_data_parallel,
        simulate_pipeline_parallel,
        simulate_tensor_parallel,
    )

    console = Console()

    # Tensor Parallel demo
    console.print("\n[bold cyan]═══ Tensor Parallelism ═══[/bold cyan]\n")
    tp_steps = simulate_tensor_parallel(hidden_size=512, num_gpus=4)
    for step in tp_steps:
        console.print(f"  [green]Step {step.step_id}:[/green] {step.label}")
        console.print(f"    {step.description}")
        for gpu_id, assignment in step.gpu_assignments.items():
            console.print(f"    [dim]GPU {gpu_id}:[/dim] {assignment}")

    # Pipeline Parallel demo
    console.print("\n[bold magenta]═══ Pipeline Parallelism ═══[/bold magenta]\n")
    _pp_steps, pp_metrics = simulate_pipeline_parallel(
        num_layers=16, num_gpus=4, num_microbatches=8
    )
    table = Table(title="Pipeline Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, val in pp_metrics.items():
        table.add_row(key, f"{val:.2f}" if isinstance(val, float) else str(val))
    console.print(table)

    # Data Parallel demo
    console.print("\n[bold yellow]═══ Data Parallelism ═══[/bold yellow]\n")
    _dp_steps, dp_metrics = simulate_data_parallel(
        batch_size=256, num_gpus=4, gradient_accumulation_steps=2
    )
    table = Table(title="Data Parallel Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    for key, val in dp_metrics.items():
        table.add_row(key, f"{val:.2f}" if isinstance(val, float) else str(val))
    console.print(table)

    console.print(
        "\n[bold green]All simulations complete![/bold green] "
        "Deploy with: [cyan]modal deploy src/megatron_viz/app.py[/cyan]\n"
    )
