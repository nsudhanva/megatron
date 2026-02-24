# Megatron Parallelism Visualizer

Interactive visualizer for NVIDIA Megatron-Core parallelism strategies — **Tensor, Pipeline, and Data Parallelism**.

Built to understand how large language models are distributed across GPUs during training.

## Architecture

```mermaid
graph LR
    subgraph "Your Mac (Local)"
        A["Frontend<br/>HTML / D3.js / GSAP"] -->|instant| B["Client Simulator<br/>JavaScript"]
        A -->|"API call<br/>(optional)"| C
    end
    subgraph "Modal Cloud (T4 GPU)"
        C["FastAPI Endpoints"] --> D["Tensor Parallel Demo"]
        C --> E["Pipeline Parallel Demo"]
        C --> F["Data Parallel Demo"]
        C --> G["GPU Benchmark<br/>(real matmul timing)"]
    end
    style A fill:#1a1a2e,stroke:#76b900,color:#fff
    style B fill:#1a1a2e,stroke:#00b4d8,color:#fff
    style C fill:#16213e,stroke:#e94560,color:#fff
    style D fill:#0f3460,stroke:#76b900,color:#fff
    style E fill:#0f3460,stroke:#76b900,color:#fff
    style F fill:#0f3460,stroke:#76b900,color:#fff
    style G fill:#0f3460,stroke:#e94560,color:#fff
```

## How Megatron Parallelism Works

### Tensor Parallelism (TP)

Splits individual **weight matrices** across GPUs. Each GPU computes a partial result, then they communicate to reconstruct the full output.

```mermaid
graph LR
    W["Weight Matrix W<br/>(H × H)"] -->|Column Split| G0["GPU 0<br/>W[:, :H/4]"]
    W -->|Column Split| G1["GPU 1<br/>W[:, H/4:H/2]"]
    W -->|Column Split| G2["GPU 2<br/>W[:, H/2:3H/4]"]
    W -->|Column Split| G3["GPU 3<br/>W[:, 3H/4:]"]

    G0 -->|"Y₀ = X @ W₀"| AR["All-Gather<br/>Concatenate"]
    G1 -->|"Y₁ = X @ W₁"| AR
    G2 -->|"Y₂ = X @ W₂"| AR
    G3 -->|"Y₃ = X @ W₃"| AR

    AR --> Y["Full Output Y"]

    style W fill:#2d2d44,stroke:#76b900,color:#fff
    style AR fill:#2d2d44,stroke:#e94560,color:#fff
    style Y fill:#2d2d44,stroke:#76b900,color:#fff
    style G0 fill:#1a3a1a,stroke:#76b900,color:#fff
    style G1 fill:#1a2a3a,stroke:#00b4d8,color:#fff
    style G2 fill:#3a1a1a,stroke:#e94560,color:#fff
    style G3 fill:#3a2a1a,stroke:#f4a261,color:#fff
```

### Pipeline Parallelism (PP)

Distributes transformer **layers** across GPU stages. Uses micro-batch scheduling (1F1B) to minimize idle "bubble" time.

```mermaid
graph TD
    subgraph "Stage 0 (GPU 0)"
        L0["Layers 0-5"]
    end
    subgraph "Stage 1 (GPU 1)"
        L1["Layers 6-11"]
    end
    subgraph "Stage 2 (GPU 2)"
        L2["Layers 12-17"]
    end
    subgraph "Stage 3 (GPU 3)"
        L3["Layers 18-23"]
    end

    L0 -->|"activations"| L1
    L1 -->|"activations"| L2
    L2 -->|"activations"| L3
    L3 -->|"gradients"| L2
    L2 -->|"gradients"| L1
    L1 -->|"gradients"| L0

    style L0 fill:#1a3a1a,stroke:#76b900,color:#fff
    style L1 fill:#1a2a3a,stroke:#00b4d8,color:#fff
    style L2 fill:#3a1a1a,stroke:#e94560,color:#fff
    style L3 fill:#3a2a1a,stroke:#f4a261,color:#fff
```

#### 1F1B Schedule

```mermaid
gantt
    title Pipeline 1F1B Schedule (4 stages, 6 micro-batches)
    dateFormat X
    axisFormat %s

    section Stage 0
    F0 :f00, 0, 1
    F1 :f01, 1, 2
    F2 :f02, 2, 3
    F3 :f03, 3, 4
    B0 :crit, b00, 4, 5
    B1 :crit, b01, 5, 6

    section Stage 1
    Bubble :done, bub10, 0, 1
    F0 :f10, 1, 2
    F1 :f11, 2, 3
    F2 :f12, 3, 4
    B0 :crit, b10, 5, 6
    B1 :crit, b11, 6, 7

    section Stage 2
    Bubble :done, bub20, 0, 2
    F0 :f20, 2, 3
    F1 :f21, 3, 4
    B0 :crit, b20, 6, 7
    B1 :crit, b21, 7, 8

    section Stage 3
    Bubble :done, bub30, 0, 3
    F0 :f30, 3, 4
    B0 :crit, b30, 7, 8
    B1 :crit, b31, 8, 9
```

### Data Parallelism (DP)

Each GPU holds a **full copy** of the model. The training batch is split evenly, and gradients are synchronized via All-Reduce.

```mermaid
graph TD
    B["Training Batch<br/>(256 samples)"] --> S["Split"]

    S --> G0["GPU 0<br/>64 samples<br/>Full Model"]
    S --> G1["GPU 1<br/>64 samples<br/>Full Model"]
    S --> G2["GPU 2<br/>64 samples<br/>Full Model"]
    S --> G3["GPU 3<br/>64 samples<br/>Full Model"]

    G0 --> FW0["Forward + Backward"]
    G1 --> FW1["Forward + Backward"]
    G2 --> FW2["Forward + Backward"]
    G3 --> FW3["Forward + Backward"]

    FW0 --> AR["All-Reduce<br/>∇W = avg(∇W₀, ∇W₁, ∇W₂, ∇W₃)"]
    FW1 --> AR
    FW2 --> AR
    FW3 --> AR

    AR --> U["Update Weights<br/>(identical on all GPUs)"]

    style B fill:#2d2d44,stroke:#76b900,color:#fff
    style S fill:#2d2d44,stroke:#76b900,color:#fff
    style AR fill:#2d2d44,stroke:#e94560,color:#fff
    style U fill:#2d2d44,stroke:#76b900,color:#fff
    style G0 fill:#1a3a1a,stroke:#76b900,color:#fff
    style G1 fill:#1a2a3a,stroke:#00b4d8,color:#fff
    style G2 fill:#3a1a1a,stroke:#e94560,color:#fff
    style G3 fill:#3a2a1a,stroke:#f4a261,color:#fff
```

### How They Combine in Megatron

In practice, Megatron uses **all three** simultaneously. This is called **3D parallelism**:

```mermaid
graph TD
    subgraph "3D Parallelism"
        direction TB
        DP["Data Parallelism<br/>Split BATCHES across replicas"] --> PP
        PP["Pipeline Parallelism<br/>Split LAYERS across stages"] --> TP
        TP["Tensor Parallelism<br/>Split MATRICES within layers"]
    end

    subgraph "Example: 64 GPUs"
        direction LR
        D["DP = 4 replicas"] --> P["PP = 4 stages each"]
        P --> T["TP = 4 GPUs per layer"]
    end

    style DP fill:#1a3a1a,stroke:#76b900,color:#fff
    style PP fill:#1a2a3a,stroke:#00b4d8,color:#fff
    style TP fill:#3a1a1a,stroke:#e94560,color:#fff
    style D fill:#1a3a1a,stroke:#76b900,color:#fff
    style P fill:#1a2a3a,stroke:#00b4d8,color:#fff
    style T fill:#3a1a1a,stroke:#e94560,color:#fff
```

## Quick Start

```bash
# Prerequisites: uv (https://docs.astral.sh/uv/)
uv sync                        # Install all dependencies
open frontend/index.html       # Open the interactive visualizer
```

## Modal Commands

The project uses [Modal](https://modal.com) for serverless GPU compute. You need a Modal account and token (`uv run modal token set`).

### `uv run modal run src/megatron_viz/app.py`

Runs the **local entrypoint** (`main()` function). This executes the simulation locally on your machine and prints a rich CLI demo with tables and metrics. No GPU needed, no cloud costs — the simulator is pure Python.

### `uv run modal deploy src/megatron_viz/app.py`

Deploys the app to Modal's cloud. This creates **persistent HTTPS endpoints** that run on Modal's infrastructure. The CPU endpoints serve simulation data (cold start ~2s); the GPU endpoint runs real matrix operations on a T4 GPU (cold start ~30s).

### Cost Breakdown

| Command | What runs | Cost |
| --- | --- | --- |
| `modal run` | Local entrypoint only | **Free** (runs on your Mac) |
| `modal deploy` | Creates cloud endpoints | **Free** until called |
| CPU endpoints | Simulation on Modal CPU | ~$0.000014/sec (~free) |
| GPU benchmark | T4 GPU matmul timing | ~$0.59/hr ($30/mo free tier) |

> **Note:** Modal's starter tier gives you $30/month in free credits. The CPU simulation endpoints cost virtually nothing. Only the GPU benchmark endpoint consumes meaningful credits, and each call takes <1 second.

## Project Structure

```mermaid
graph TD
    Root["megatron/"] --> Src["src/megatron_viz/"]
    Root --> FE["frontend/"]
    Root --> Tests["tests/"]
    Root --> Config["pyproject.toml"]

    Src --> Sim["simulator.py<br/>Pure-Python simulation engine"]
    Src --> App["app.py<br/>Modal endpoints + CLI"]

    FE --> HTML["index.html"]
    FE --> CSS["css/style.css<br/>Dark glassmorphism theme"]
    FE --> JS["js/"]

    JS --> AppJS["app.js<br/>Tab/control wiring"]
    JS --> SimJS["simulator.js<br/>Client-side simulation"]
    JS --> VizJS["visualizer.js<br/>D3.js + GSAP rendering"]

    Tests --> TestSim["test_simulator.py<br/>15 unit tests"]

    style Root fill:#2d2d44,stroke:#76b900,color:#fff
    style Sim fill:#1a3a1a,stroke:#76b900,color:#fff
    style App fill:#1a3a1a,stroke:#76b900,color:#fff
    style VizJS fill:#1a2a3a,stroke:#00b4d8,color:#fff
    style TestSim fill:#3a2a1a,stroke:#f4a261,color:#fff
```

## Development

```bash
# Lint
uv run ruff check src/ tests/

# Format
uv run ruff format src/ tests/

# Test
uv run pytest tests/ -v
```

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.13 |
| Package Manager | uv |
| GPU Compute | Modal (T4, starter tier) |
| Visualization | D3.js + GSAP |
| Linting | Ruff |
| Testing | pytest |
