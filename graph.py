"""
Sequential simulation runner for the spatial gossip PD experiment.
 
Match-level node functions (selection, prompting, LLM call, recording,
gossip propagation, reputation update) are orchestrated by a plain
Python loop rather than a graph-execution framework, keeping per-call
overhead bounded across the thousands of sequential LLM calls in a
full run.
 
Simulation state is checkpointed after every completed macro round
and deleted automatically once a run finishes successfully, allowing
an interrupted run to resume from the last completed round.
"""

from __future__ import annotations

import gc
import pickle
import time
from pathlib import Path
from typing import List

import numpy as np

from config import (
    MATCHES_PER_ROUND,
    MODE_REPUTATION,
    N_AGENTS,
    N_MACRO,
    SEED_PAIRING,
)
from data_structures import (
    MatchResult,
    PersonalEntry,
    ReputationEntry,
    RoundStats,
    SimulationState,
    update_reputation_summary,
)
from gossip import node_broadcast_gossip
from llm_agent import node_build_prompt, node_call_llm

# ── Execution settings ─────────────────────────────────────────────────────
# Optional delay (seconds) inserted between macro rounds. Configurable
# per execution environment; see README for recommended values on
# local hardware versus cloud GPU runtimes.
COOLING_PAUSE_SECONDS: int = 0
 
# Directory for round-level checkpoint files.
CHECKPOINT_DIR = Path("checkpoints")

# ==============================================================================
#  CHECKPOINT HELPERS
# ==============================================================================

def _ckpt_path(state: SimulationState, round_num: int) -> Path:
    model = state.get("model_name", "unknown").replace(":", "_").replace("/", "_")
    mode  = state.get("mode", "unknown")
    seed  = state.get("sim_seed", 0)
    return CHECKPOINT_DIR / f"{model}__{mode}__seed{seed}__round{round_num:02d}.pkl"


def _save_checkpoint(state: SimulationState, round_num: int) -> None:
    """Pickle the full state after completing round_num."""
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    path = _ckpt_path(state, round_num)
    try:
        with open(path, "wb") as f:
            pickle.dump(dict(state), f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  [ckpt] Saved  → {path.name}")
    except Exception as exc:
        # A failed write does not interrupt the simulation; the run continues without a checkpoint for this round.
        print(f"  [ckpt] WARNING: could not save checkpoint: {exc}")


def _load_latest_checkpoint(
    state: SimulationState,
) -> tuple[SimulationState | None, int]:
    """
    Locate the most recent valid checkpoint matching this state's
    (model, mode, seed) identity.
 
    Returns (loaded_state, last_completed_round), or (None, 0) if no
    checkpoint exists for this configuration.
    """
    if not CHECKPOINT_DIR.exists():
        return None, 0

    model = state.get("model_name", "unknown").replace(":", "_").replace("/", "_")
    mode  = state.get("mode", "unknown")
    seed  = state.get("sim_seed", 0)
    pattern = f"{model}__{mode}__seed{seed}__round*.pkl"

    candidates = sorted(CHECKPOINT_DIR.glob(pattern), reverse=True)
    for path in candidates:
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)
            round_num = int(path.stem.split("__round")[-1])
            print(f"  [ckpt] Resuming from {path.name}  (round {round_num} done)")
            return loaded, round_num
        except Exception as exc:
            print(f"  [ckpt] Corrupt checkpoint {path.name}, skipping: {exc}")
            path.unlink(missing_ok=True)

    return None, 0


def _delete_checkpoints(state: SimulationState) -> None:
    """Remove all checkpoints for this (model, mode, seed) after a successful run."""
    model = state.get("model_name", "unknown").replace(":", "_").replace("/", "_")
    mode  = state.get("mode", "unknown")
    seed  = state.get("sim_seed", 0)
    pattern = f"{model}__{mode}__seed{seed}__round*.pkl"
    for path in CHECKPOINT_DIR.glob(pattern):
        path.unlink(missing_ok=True)


# ==============================================================================
#  MATCH-LEVEL OPERATIONS
#  Each function performs one stage of a single micro-match: scheduling,
#  prompting, querying the LLM, recording results, propagating gossip,
#  and updating the global reputation table.
# ==============================================================================

def node_start_macro_round(state: SimulationState) -> SimulationState:
    rng_pair = np.random.default_rng(
        SEED_PAIRING
        + state.get("sim_seed", 0) * 100_000
        + state["macro_round"]
    )
    all_pairs = [(i, j) for i in range(N_AGENTS) for j in range(i + 1, N_AGENTS)]
    shuffled  = all_pairs.copy()
    rng_pair.shuffle(shuffled)

    state["match_schedule"]        = shuffled
    state["match_index"]           = 0
    state["current_round_results"] = []
    return state


def node_select_next_pair(state: SimulationState) -> SimulationState:
    """Set the current match to the next pair in this round's schedule."""
    state["current_pair"] = state["match_schedule"][state["match_index"]]
    return state


def node_record_result(state: SimulationState) -> SimulationState:
    """Write the outcome of the current match into both agents' personal logs."""
    result = state["last_result"]
    if result is None:
        return state

    a_mem       = state["agent_memory"][result.agent_A]
    b_mem       = state["agent_memory"][result.agent_B]
    macro_round = state["macro_round"]

    a_mem.personal_log.append(PersonalEntry(
        macro_round=result.macro_round,
        match_index=result.match_index,
        opponent_id=result.agent_B,
        my_action=result.action_A,
        opp_action=result.action_B,
        my_payoff=result.payoff_A,
        opp_payoff=result.payoff_B,
    ))
    b_mem.personal_log.append(PersonalEntry(
        macro_round=result.macro_round,
        match_index=result.match_index,
        opponent_id=result.agent_A,
        my_action=result.action_B,
        opp_action=result.action_A,
        my_payoff=result.payoff_B,
        opp_payoff=result.payoff_A,
    ))

    update_reputation_summary(a_mem, result.agent_B, result.action_B, macro_round)
    update_reputation_summary(b_mem, result.agent_A, result.action_A, macro_round)
    state["current_round_results"].append(result)
    return state


def _update_global_entry(
    table: dict[int, ReputationEntry],
    agent_id: int,
    action: str,
    macro_round: int,
) -> None:
    entry = table.get(agent_id, ReputationEntry())
    if action == "C":
        entry.total_C += 1
    else:
        entry.total_D += 1
    entry.last_seen_round = macro_round
    table[agent_id] = entry


def node_update_reputation(state: SimulationState) -> SimulationState:
    """Update the global reputation table with the current match's outcome.
 
    No-op outside MODE_REPUTATION, where reputation is instead inferred
    from gossip or direct play history.
    """
    if state["mode"] != MODE_REPUTATION:
        return state
    result = state["last_result"]
    if result is None:
        return state
    table = state.setdefault("global_reputation_table", {})
    _update_global_entry(table, result.agent_A, result.action_A, state["macro_round"])
    _update_global_entry(table, result.agent_B, result.action_B, state["macro_round"])
    return state


# ==============================================================================
#  STATS
# ==============================================================================

def _gini_coefficient(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    s = np.sort(values)
    n = s.size
    i = np.arange(1, n + 1)
    return float(
        (2 * np.sum(i * s) - (n + 1) * np.sum(s)) / (n * np.sum(s))
        if np.sum(s) > 0 else 0.0
    )


def _compute_defector_isolation(
    state: SimulationState,
    matches: List[MatchResult],
) -> float:
    """
    Fraction of (Defect, Cooperate) outcomes in which the exploited
    cooperator already held a negative running-reputation impression
    of the defector prior to the match.
    """
    warned_and_exploited = 0
    total_dc = 0
    for m in matches:
        dc_pairs = []
        if m.action_A == "D" and m.action_B == "C":
            dc_pairs.append((m.agent_A, m.agent_B))
        if m.action_B == "D" and m.action_A == "C":
            dc_pairs.append((m.agent_B, m.agent_A))
        for defector_id, cooperator_id in dc_pairs:
            total_dc += 1
            rep = state["agent_memory"][cooperator_id].reputation_summary.get(defector_id)
            if rep and rep.total_D > rep.total_C:
                warned_and_exploited += 1
    return warned_and_exploited / total_dc if total_dc else 0.0


def _compute_spatial_coop_map(matches: List[MatchResult]) -> dict[int, float]:
    """Per-agent cooperation rate for this round, keyed by agent id."""
    coop   = {i: 0 for i in range(N_AGENTS)}
    played = {i: 0 for i in range(N_AGENTS)}
    for m in matches:
        for agent_id, action in ((m.agent_A, m.action_A), (m.agent_B, m.action_B)):
            played[agent_id] += 1
            if action == "C":
                coop[agent_id] += 1
    return {
        i: coop[i] / played[i] if played[i] > 0 else 0.0
        for i in range(N_AGENTS)
    }


def compute_round_stats(state: SimulationState) -> RoundStats:
    """Aggregate all per-match outcomes from the current round into RoundStats."""
    matches = state["current_round_results"]
    total   = len(matches)
    n_cc = sum(1 for m in matches if m.action_A == "C" and m.action_B == "C")
    n_cd = sum(1 for m in matches if m.action_A == "C" and m.action_B == "D")
    n_dc = sum(1 for m in matches if m.action_A == "D" and m.action_B == "C")
    n_dd = sum(1 for m in matches if m.action_A == "D" and m.action_B == "D")
    all_payoffs = [p for m in matches for p in (m.payoff_A, m.payoff_B)]
    return RoundStats(
        macro_round=state["macro_round"],
        cooperation_rate=(2*n_cc + n_cd + n_dc) / (2*total) if total else 0.0,
        mutual_coop_rate=n_cc / total if total else 0.0,
        mutual_defect_rate=n_dd / total if total else 0.0,
        avg_payoff=float(np.mean(all_payoffs)) if all_payoffs else 0.0,
        gini_coefficient=_gini_coefficient(np.array(all_payoffs)),
        defector_isolation=_compute_defector_isolation(state, matches),
        spatial_coop_map=_compute_spatial_coop_map(matches),
    )


def node_end_macro_round(state: SimulationState) -> SimulationState:
    """Compute and store this round's statistics, then advance the round counter."""
    state["history"].append(compute_round_stats(state))
    state["macro_round"] += 1
    return state


def _compute_gossip_accuracy(state: SimulationState) -> dict:
    """Fraction of received gossip reports that were not distorted in transit."""
    correct = total = 0
    for memory in state["agent_memory"].values():
        for entry in memory.gossip_log:
            total += 1
            if not entry.is_distorted:
                correct += 1
    return {
        "accuracy":  (correct / total) if total > 0 else None,
        "n_entries": float(total),
    }


def node_finalize(state: SimulationState) -> SimulationState:
    """Assemble the final per-run output dict consumed by runner.aggregate()."""
    state["final_output"] = {
        "history":         state["history"],
        "final_memory":    state["agent_memory"],
        "gossip_accuracy": _compute_gossip_accuracy(state),
        "positions":       state["positions"],
        "sim_seed":        state.get("sim_seed", 0),
        "mode":            state["mode"],
        "model_name":      state["model_name"],
    }
    return state


# ==============================================================================
#  SIMULATION RUNNER
# ==============================================================================

def _run_one_micro_match(state: SimulationState) -> SimulationState:
    """Execute one micro-match: select pair, prompt, query LLM, record, propagate."""
    state = node_select_next_pair(state)
    state = node_build_prompt(state)
    state = node_call_llm(state)
    state = node_record_result(state)
    state = node_broadcast_gossip(state)
    state = node_update_reputation(state)
    state["match_index"] += 1
    return state


def _run_one_macro_round(state: SimulationState) -> SimulationState:
    """Run every match scheduled for the current macro round."""
    state = node_start_macro_round(state)
    t0    = time.time()

    for k in range(MATCHES_PER_ROUND):
        state = _run_one_micro_match(state)
        if (k + 1) % 50 == 0 or (k + 1) == MATCHES_PER_ROUND:
            elapsed = time.time() - t0
            eta     = elapsed / (k + 1) * (MATCHES_PER_ROUND - k - 1)
            print(
                f"    match {k+1:>3}/{MATCHES_PER_ROUND}"
                f"  elapsed {elapsed/60:.1f}m"
                f"  ETA {eta/60:.1f}m",
                end="\r",
            )

    print()
    state = node_end_macro_round(state)
    return state


def run_simulation(state: SimulationState) -> SimulationState:
    """
    Run the full N_MACRO-round simulation for a single (model, mode, seed).
 
    If a checkpoint exists for this configuration, execution resumes
    from the round immediately following the last one completed;
    otherwise the simulation starts from round 1. State is checkpointed
    after every round and all checkpoints for this configuration are
    removed once the run finishes successfully.
    """
    model = state.get("model_name", "?")
    mode  = state.get("mode",       "?")
    seed  = state.get("sim_seed",    0)
    print(f"  [{model}  |  {mode}  |  seed={seed}]")

    # ── Resume from the most recent checkpoint, if one exists ─────────────────
    loaded_state, last_done = _load_latest_checkpoint(state)
    if loaded_state is not None:
        state = loaded_state

    start_round = last_done + 1
    if start_round > N_MACRO:
        print(f"  All {N_MACRO} rounds already completed in checkpoint.")
        return node_finalize(state)

    # ── Run remaining rounds ───────────────────────────────────────────────────
    for r in range(start_round, N_MACRO + 1):
        t_round = time.time()
        print(f"\n  ── Macro round {r}/{N_MACRO} ──")

        state   = _run_one_macro_round(state)
        elapsed = time.time() - t_round
        stats   = state["history"][-1]

        print(
            f"  Round {r} done in {elapsed/60:.1f}m  "
            f"coop={stats.cooperation_rate:.3f}  "
            f"avg_payoff={stats.avg_payoff:.2f}"
        )

        # ── Persist state immediately after each round ─────────────────────────
        _save_checkpoint(state, r)

        # ── Optional inter-round pause (skipped after the final round) ─────────
        if r < N_MACRO and COOLING_PAUSE_SECONDS > 0:
            print(f"  Cooling pause {COOLING_PAUSE_SECONDS}s ...", end="\r")
            time.sleep(COOLING_PAUSE_SECONDS)
            print(f"  Cooling done.{' ' * 30}")

        # ── Release per-round allocations before starting the next round ───────
        gc.collect()

    # ── Remove checkpoints now that the run has completed successfully ────────
    _delete_checkpoints(state)
    print("  Checkpoints cleaned up.")

    return node_finalize(state)