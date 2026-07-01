# Spatial Gossip PD

LLM agents play iterated Prisoner's Dilemma on a 2D grid, with reputation information spreading through space as decaying, distortable gossip. Submitted to **CSoNet 2026**.

## Overview

30 LLM-controlled agents are placed at fixed positions on a 6×6 grid. Over 10 macro rounds, every pair of agents (435 matches per round) plays one round of Prisoner's Dilemma. After each match, nearby agents may overhear the outcome — but reception probability decays with distance, and overheard information may be distorted the further it travels.

We compare three information regimes to isolate the effect of reputation systems on cooperation:

| Mode | Description |
|---|---|
| `no_info` | Baseline — agents only remember their own direct play history with an opponent |
| `gossip` | Agents learn about others only through spatially-decaying, distortable rumours |
| `reputation` | Agents have access to a global, accurate, common-knowledge reputation table |

### Gossip model

Reception probability for a third-party agent at distance `d` from a match:

```
P(receive) = exp(-λ · d)
```

If received, the report is distorted with probability:

```
P(distort | received) = 1 - exp(-μ · d)
```

Distortion flips one of the two reported actions (Binary Symmetric Channel). Combined, the probability of receiving an *accurate* report decays as `exp(-(λ+μ)·d)`.

## Research questions

- **RQ1** — Does any reputation mechanism (gossip or global) raise cooperation above the no-information baseline?
- **RQ2** — How much does accurate global reputation outperform noisy, distance-decayed gossip?
- **RQ3** — Does spatial structure produce visible cooperative clustering on the grid?

## Repository structure

```
config.py            Simulation constants (grid size, payoff matrix, decay params, seeds)
data_structures.py   Dataclasses, SimulationState definition, initialize()
gossip.py            Spatial gossip propagation (reception + distortion)
llm_agent.py         Prompt construction, Ollama API calls, action parsing
graph.py             Simulation loop, round-level checkpointing, stats computation
runner.py            Multi-seed experiment runner with seed-level result persistence
plotting.py          All matplotlib figures (cooperation rate, payoff, isolation,
                      Gini coefficient, gossip accuracy vs. distance, spatial heatmaps)
main.py               CLI entry point — run one (model, mode) configuration at a time
smoke_test.py         4-phase pre-flight check before running the full experiment
spatial_gossip_colab.py   Cell-by-cell script for running the experiment on Colab
```

## Setup

### Requirements

- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- `numpy`, `matplotlib`, `requests` (install via `pip install -r requirements.txt`)

### Models

```bash
ollama pull mistral:7b-instruct
ollama pull llama3.1:8b
```

### Install

```bash
git clone <repo-url>
cd spatial-gossip-pd
pip install -r requirements.txt
ollama serve   # in a separate terminal
```

## Usage

### 1. Smoke test (run this first)

```bash
python smoke_test.py
```

Runs a patched 2-round × 5-match version of the full pipeline (~3–8 minutes) to verify Ollama connectivity, output parsing, and the simulation loop before committing to a full run.

### 2. Run experiments — one configuration at a time

Each `(model, mode)` pair is run as an independent process invocation, so that any interruption only affects the configuration currently running.

```bash
python main.py            # show status of all configurations
python main.py 1          # run configuration #1
python main.py 2          # run configuration #2
...
python main.py plot       # generate all figures from saved results
```

| # | Model | Mode |
|---|---|---|
| 1 | mistral:7b-instruct | no_info |
| 2 | mistral:7b-instruct | gossip |
| 3 | mistral:7b-instruct | reputation |
| 4 | llama3.1:8b | no_info |
| 5 | llama3.1:8b | gossip |
| 6 | llama3.1:8b | reputation |

### 3. Resuming an interrupted run

The pipeline is checkpointed at two levels:

- **Round-level** (`graph.py`) — state is pickled after every macro round to `checkpoints/`. Re-running the same configuration resumes from the last completed round.
- **Seed-level** (`runner.py`) — each completed seed is saved to `results/seeds/*.json`. If a configuration is interrupted partway through its 5 seeds, completed seeds are loaded from disk on the next run instead of being recomputed.

Both recovery mechanisms are automatic — re-running the same `python main.py N` command is sufficient.

### 4. Running on Google Colab

For users without a sufficiently capable local GPU, `spatial_gossip_colab.py` contains a cell-by-cell script that installs Ollama, mounts Google Drive for persistent model/checkpoint storage, and runs the same `main.py` CLI. See inline comments for setup.

Two settings differ between local and cloud execution:

| Setting | Location | Local | Colab |
|---|---|---|---|
| `NUM_GPU_LAYERS` | `llm_agent.py` | `0` (CPU inference) | `99` (full GPU offload) |
| `COOLING_PAUSE_SECONDS` | `graph.py` | configurable delay between rounds | `0` |

## Output

Results are written to:

```
results/
├── seeds/
│   └── <model>__<mode>__seed<N>.json   per-seed raw output
└── <model>__<mode>.json                aggregated mean ± std across seeds

plots/
├── cooperation_rate.png
├── payoff_distribution.png
├── defector_isolation.png
├── gini_coefficient.png
├── gossip_accuracy.png
└── spatial_heatmaps.png
```

## Limitations

- **Distortion model**: gossip content distortion is a simple Binary Symmetric Channel (one action flipped) rather than a richer corruption model.
- **Pairing**: match pairing is independent of grid position — only gossip propagation is spatially constrained.
- **Reception/belief conflation**: agents have no way to distinguish an accurate report from a distorted one; both are recorded identically in memory.

## Citation

If you use this code, please cite the corresponding CSoNet 2026 submission (details to be added upon acceptance).