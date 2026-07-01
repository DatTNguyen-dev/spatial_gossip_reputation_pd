"""Experiment runner and aggregation utilities."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from config import (
    MODE_GOSSIP,
    MODE_NO_INFO,
    MODE_REPUTATION,
    MODELS,
    N_AGENTS,
    N_MACRO,
    SEED_PLACEMENT,
)
from data_structures import (
    RoundStats,
    SimulationState,
    initialize,
    validate_initial_state,
)
from graph import run_simulation

ModeName      = str
ModelName     = str
ExperimentKey = Tuple[ModelName, ModeName]

# Directory for individual seed results
SEED_RESULTS_DIR = Path("results") / "seeds"


# ==============================================================================
#  SEED-LEVEL PERSISTENCE
#  Save each seed result immediately after completion.
#  If the process is interrupted at seed 3, re-running resumes from seed 3 without re-running seeds 0, 1, and 2.
# ==============================================================================

def _seed_path(model_name: str, mode: str, seed: int) -> Path:
    """Return the canonical path for a single seed result file."""
    safe = model_name.replace(":", "_").replace("/", "_")
    return SEED_RESULTS_DIR / f"{safe}__{mode}__seed{seed}.json"


def _save_seed_result(result: dict[str, Any], path: Path) -> None:
    """
    Serialize one seed result to JSON, skipping final_memory (too large).
    Preserves everything aggregate() needs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # RoundStats is a dataclass — convert to dict via asdict()
    history_dicts = []
    for stats in result["history"]:
        d = asdict(stats)
        # JSON requires string keys — convert int agent IDs to str
        d["spatial_coop_map"] = {
            str(k): v for k, v in d["spatial_coop_map"].items()
        }
        history_dicts.append(d)

    payload = {
        "history":         history_dicts,
        "gossip_accuracy": result["gossip_accuracy"],
        "positions":       {str(k): list(v)
                            for k, v in result["positions"].items()},
        "sim_seed":        result["sim_seed"],
        "mode":            result["mode"],
        "model_name":      result["model_name"],
        # final_memory omitted — not needed for aggregation or plotting
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_seed_result(path: Path) -> dict[str, Any]:
    """
    Load a seed result from JSON and restore the correct Python types.
    RoundStats objects are reconstructed so aggregate() works normally.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    # Reconstruct RoundStats objects (aggregate() uses getattr on them)
    history = [
        RoundStats(
            macro_round=s["macro_round"],
            cooperation_rate=s["cooperation_rate"],
            mutual_coop_rate=s["mutual_coop_rate"],
            mutual_defect_rate=s["mutual_defect_rate"],
            avg_payoff=s["avg_payoff"],
            gini_coefficient=s["gini_coefficient"],
            defector_isolation=s["defector_isolation"],
            # Restore int keys (JSON encodes dict keys as strings)
            spatial_coop_map={int(k): v
                              for k, v in s["spatial_coop_map"].items()},
        )
        for s in data["history"]
    ]

    return {
        "history":         history,
        "gossip_accuracy": data["gossip_accuracy"],
        "positions":       {int(k): tuple(v)
                            for k, v in data["positions"].items()},
        "sim_seed":        data["sim_seed"],
        "mode":            data["mode"],
        "model_name":      data["model_name"],
        "final_memory":    None,  # not saved to disk, not needed for aggregation
    }


# ==============================================================================
#  AGGREGATION HELPERS
# ==============================================================================

def _round_stats_to_array(history: List[RoundStats], field: str) -> np.ndarray:
    """Extract one scalar field from a list of RoundStats into a 1-D array."""
    return np.array([getattr(stats, field) for stats in history], dtype=float)


def _aggregate_scalar_series(
    runs: List[dict[str, Any]],
    field: str,
) -> dict[str, list[float]]:
    """
    Stack the per-seed time series for `field` and return mean ± std.

    Raises ValueError if seeds produced different numbers of rounds,
    which indicates an early-termination bug in graph.py.
    """
    arrays  = [_round_stats_to_array(run["history"], field) for run in runs]
    lengths = [len(a) for a in arrays]

    if len(set(lengths)) > 1:
        raise ValueError(
            f"Runs have different history lengths: {lengths}. "
            f"Some runs may have terminated early — check graph.py."
        )

    series = np.stack(arrays, axis=0)   # shape (n_seeds, n_rounds)
    return {
        "mean": np.mean(series, axis=0).tolist(),
        "std":  np.std(series,  axis=0).tolist(),
    }


def _aggregate_spatial_maps(runs: List[dict[str, Any]]) -> dict[int, float]:
    """
    Average the final-round spatial cooperation map across seeds.

    All seeds share the same agent positions (seed_placement is fixed),
    so averaging per-agent cooperation rates across seeds is meaningful.
    """
    per_run = np.array(
        [
            [run["history"][-1].spatial_coop_map[i] for i in range(N_AGENTS)]
            for run in runs
        ],
        dtype=float,
    )   # shape (n_seeds, N_AGENTS)
    return {i: float(np.mean(per_run[:, i])) for i in range(N_AGENTS)}


def _aggregate_gossip_accuracy(runs: List[dict[str, Any]]) -> dict[str, Any]:
    """
    Aggregate gossip accuracy statistics across seeds.

    accuracy is None for MODE_NO_INFO and MODE_REPUTATION (no gossip entries).
    n_entries is always a float (0.0 when there is no gossip).
    Checking the value rather than the key correctly handles MODE_NO_INFO,
    where the key always exists but accuracy is None.
    """
    accuracies = [run["gossip_accuracy"]["accuracy"] for run in runs]
    n_entries  = [run["gossip_accuracy"]["n_entries"] for run in runs]
    return {
        "gossip_accuracy_mean": float(np.mean(accuracies))
                                if all(a is not None for a in accuracies)
                                else None,
        "gossip_entries_mean":  float(np.mean(n_entries)),
    }


# ==============================================================================
#  AGGREGATE
# ==============================================================================

def aggregate(runs: List[dict[str, Any]]) -> dict[str, Any]:
    """Combine repeated runs into mean/std summaries for plotting."""
    if not runs:
        raise ValueError("aggregate() requires at least one completed run.")

    scalar_fields = [
        "cooperation_rate",
        "mutual_coop_rate",
        "mutual_defect_rate",
        "avg_payoff",
        "gini_coefficient",
        "defector_isolation",
    ]

    aggregated: dict[str, Any] = {
        field: _aggregate_scalar_series(runs, field)
        for field in scalar_fields
    }
    aggregated["macro_rounds"]          = list(range(1, N_MACRO + 1))
    aggregated["spatial_coop_map_mean"] = _aggregate_spatial_maps(runs)
    aggregated.update(_aggregate_gossip_accuracy(runs))

    # All seeds share the same positions (seed_placement is fixed),
    # so using the first run's positions is valid for all seeds.
    aggregated["positions"]        = runs[0]["positions"]
    aggregated["history_examples"] = [asdict(s) for s in runs[0]["history"]]

    return aggregated


# ==============================================================================
#  SINGLE EXPERIMENT
# ==============================================================================

def run_single_experiment(
    mode: str,
    model_name: str,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """
    Run one full simulation and return its final_output dict.

    seed_placement is fixed across seeds so that agent positions are
    identical in every run — this makes spatial averaging valid.
    sim_seed varies per seed to randomise gossip and pairing order.
    """
    state: SimulationState = initialize(
        mode=mode,
        model_name=model_name,
        seed_placement=SEED_PLACEMENT,  # fixed — same positions every seed
        sim_seed=seed_offset,           # varies — different RNG streams
    )
    validate_initial_state(state)
    final_state = run_simulation(state)
    return final_state["final_output"]


# ==============================================================================
#  RUN CONFIGURATION  —  with seed-level persistence
# ==============================================================================

def run_configuration(
    model_name: str,
    mode: str,
    n_seeds: int = 5,
) -> tuple[dict[str, Any], List[dict[str, Any]]]:
    """
    Run one (model, mode) configuration across n_seeds independent seeds.

    Each seed result is saved to results/seeds/ immediately after completion.
    On re-run, completed seeds are loaded from disk instead of re-executed.
    Within each seed, round-level checkpoints (graph.py) handle crash recovery.

    Interruption recovery
    ---------------------
    Interrupted mid-round   → graph.py checkpoint resumes from last completed round
    Interrupted mid-config  → completed seeds load from disk, failed seed re-runs
    Interrupted before JSON → all seeds reload from disk, aggregate runs instantly
    """
    runs: List[dict[str, Any]] = []

    for seed in range(n_seeds):
        seed_path = _seed_path(model_name, mode, seed)

        # Resume: this seed already completed — load from disk
        if seed_path.exists():
            try:
                result = _load_seed_result(seed_path)
                runs.append(result)
                coop = result["history"][-1].cooperation_rate
                print(f"  ↩  seed={seed} loaded from disk  "
                      f"(final coop={coop:.3f})")
                continue
            except Exception as exc:
                # Corrupt file — delete and re-run this seed
                print(f"  ⚠  seed={seed} file corrupt, re-running: {exc}")
                seed_path.unlink(missing_ok=True)

        # Run this seed from scratch (or resume via graph.py checkpoint)
        try:
            result = run_single_experiment(
                mode=mode,
                model_name=model_name,
                seed_offset=seed,
            )

            # Save immediately — if the next seed fails, this result is already on disk
            _save_seed_result(result, seed_path)
            coop = result["history"][-1].cooperation_rate
            print(f"  ✓  seed={seed} done  "
                  f"(final coop={coop:.3f})  "
                  f"→ {seed_path.name}")

            runs.append(result)

        except Exception as exc:
            print(f"  ✗  seed={seed} FAILED: {exc}")
            continue

    if not runs:
        raise RuntimeError(
            f"All {n_seeds} seeds failed for {model_name}/{mode}. "
            f"Check that Ollama is running: ollama serve"
        )
    if len(runs) < n_seeds:
        print(f"  ⚠  Only {len(runs)}/{n_seeds} seeds succeeded — "
              f"std estimates will have higher variance.")

    return aggregate(runs), runs


# ==============================================================================
#  RUN ALL EXPERIMENTS
# ==============================================================================

def run_all_experiments(
    models: Iterable[str] = MODELS,
    modes: Iterable[str] = (MODE_NO_INFO, MODE_GOSSIP, MODE_REPUTATION),
    n_seeds: int = 5,
) -> Dict[ExperimentKey, dict[str, Any]]:
    """Run every (model, mode) combination and return aggregated results."""
    results: Dict[ExperimentKey, dict[str, Any]] = {}

    for model_name in models:
        for mode in modes:
            print(f"\n{'='*55}")
            print(f"  Model : {model_name}")
            print(f"  Mode  : {mode}")
            print(f"{'='*55}")

            aggregated, _ = run_configuration(
                model_name=model_name,
                mode=mode,
                n_seeds=n_seeds,
            )
            results[(model_name, mode)] = aggregated
            print(f"  → Final coop: "
                  f"{aggregated['cooperation_rate']['mean'][-1]:.3f}")

    return results