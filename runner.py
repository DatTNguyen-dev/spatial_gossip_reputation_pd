"""Experiment runner and aggregation utilities."""

from __future__ import annotations

from dataclasses import asdict
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
from data_structures import RoundStats, SimulationState, initialize, validate_initial_state
from graph import run_simulation

ModeName     = str
ModelName    = str
ExperimentKey = Tuple[ModelName, ModeName]


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

    Raises ValueError if seeds produced different numbers of rounds
    (indicates an early-termination bug in graph.py).
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
    so averaging per-agent cooperation rates is meaningful.
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
    """
    accuracies = [run["gossip_accuracy"]["accuracy"] for run in runs]
    n_entries  = [run["gossip_accuracy"]["n_entries"] for run in runs]

    # Only compute mean accuracy when gossip was actually active.
    # Checking the value (not the key) correctly handles MODE_NO_INFO,
    # where the key exists but accuracy is None.
    if all(a is not None for a in accuracies):
        accuracy_mean = float(np.mean(accuracies))
    else:
        accuracy_mean = None

    return {
        "gossip_accuracy_mean": accuracy_mean,
        "gossip_entries_mean":  float(np.mean(n_entries)),
    }


# ==============================================================================
#  AGGREGATE
# ==============================================================================

def aggregate(runs: List[dict[str, Any]]) -> dict[str, Any]:
    """
    Combine repeated runs into mean/std summaries for plotting.

    Parameters
    ----------
    runs : list of final_output dicts, one per seed.

    Returns
    -------
    aggregated dict with time-series stats, spatial map, and gossip metrics.
    """
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

    # Keep first-seed positions and a sample history for debugging.
    # All seeds share the same positions (seed_placement is fixed).
    aggregated["positions"]       = runs[0]["positions"]
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
    sim_seed varies per seed to randomise gossip/pairing order.
    """
    state: SimulationState = initialize(
        mode=mode,
        model_name=model_name,
        seed_placement=SEED_PLACEMENT,   # fixed — same positions every seed
        sim_seed=seed_offset,            # varies — different RNG streams
    )
    validate_initial_state(state)
    final_state = run_simulation(state)
    return final_state["final_output"]


# ==============================================================================
#  RUN CONFIGURATION  (one model + mode, multiple seeds)
# ==============================================================================

def run_configuration(
    model_name: str,
    mode: str,
    n_seeds: int = 5,
) -> tuple[dict[str, Any], List[dict[str, Any]]]:
    """
    Run one (model, mode) configuration across n_seeds independent seeds.

    Failed seeds are skipped with a warning rather than crashing the run.
    If all seeds fail, RuntimeError is raised so the caller can decide
    whether to abort or continue with the remaining configurations.

    Returns
    -------
    (aggregated_dict, list_of_raw_run_dicts)
    """
    runs: List[dict[str, Any]] = []

    for seed in range(n_seeds):
        try:
            result = run_single_experiment(
                mode=mode,
                model_name=model_name,
                seed_offset=seed,
            )
            runs.append(result)
            print(f"  ✓ seed={seed} completed ({len(runs)}/{n_seeds})")

        except Exception as exc:
            print(f"  ✗ seed={seed} FAILED: {exc}")
            continue

    if not runs:
        raise RuntimeError(
            f"All {n_seeds} seeds failed for {model_name} / {mode}. "
            f"Check that Ollama is running: ollama serve"
        )
    if len(runs) < n_seeds:
        print(
            f"  ⚠  Only {len(runs)}/{n_seeds} seeds succeeded — "
            f"std estimates will have higher variance."
        )

    return aggregate(runs), runs


# ==============================================================================
#  RUN ALL EXPERIMENTS  (all models × all modes)
# ==============================================================================

def run_all_experiments(
    models: Iterable[str] = MODELS,
    modes: Iterable[str] = (MODE_NO_INFO, MODE_GOSSIP, MODE_REPUTATION),
    n_seeds: int = 5,
) -> Dict[ExperimentKey, dict[str, Any]]:
    """
    Run every (model, mode) combination and return aggregated results.

    Results are collected in-memory.  For long runs, prefer calling
    run_configuration() directly inside a loop so you can save each
    result to disk before moving to the next configuration.
    """
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

            final_coop = aggregated["cooperation_rate"]["mean"][-1]
            print(f"  → Final cooperation rate: {final_coop:.3f}")

    return results