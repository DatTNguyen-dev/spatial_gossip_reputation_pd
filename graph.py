"""
LangGraph node functions + plain-Python simulation runner.

Crash protection:
  - Saves a checkpoint after every macro round (pickle)
  - If laptop crashes at round 6, next run resumes from round 7
  - Configurable cooling pause between rounds (default 90s)

Checkpoint files are saved to checkpoints/ and deleted automatically
after a successful full simulation.
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

# ── Thermal protection settings ───────────────────────────────────────────────
# Increase COOLING_PAUSE if laptop still crashes.
# 90s pause × 9 rounds × 9 configs = ~2h total idle — worth it to avoid crashes.
COOLING_PAUSE_SECONDS: int = 0

# Where to save round-level checkpoints
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
        # Checkpoint failure is non-fatal — just warn and continue.
        print(f"  [ckpt] WARNING: could not save checkpoint: {exc}")


def _load_latest_checkpoint(
    state: SimulationState,
) -> tuple[SimulationState | None, int]:
    """
    Find the most recent valid checkpoint for (model, mode, seed).

    Returns (loaded_state, last_completed_round) or (None, 0) if none found.
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
#  NODE FUNCTIONS  (unchanged logic — only orchestration differs)
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
    state["current_pair"] = state["match_schedule"][state["match_index"]]
    return state


def node_record_result(state: SimulationState) -> SimulationState:
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
    state["history"].append(compute_round_stats(state))
    state["macro_round"] += 1
    return state


def _compute_gossip_accuracy(state: SimulationState) -> dict:
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
    """One micro-match: select → prompt → LLM → record → gossip → reputation."""
    state = node_select_next_pair(state)
    state = node_build_prompt(state)
    state = node_call_llm(state)
    state = node_record_result(state)
    state = node_broadcast_gossip(state)
    state = node_update_reputation(state)
    state["match_index"] += 1
    return state


def _run_one_macro_round(state: SimulationState) -> SimulationState:
    """Run all MATCHES_PER_ROUND micro-matches, printing progress every 50."""
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
    Full simulation with checkpoint + cooling pause between rounds.

    On every crash:
      → Re-run the same command (python main.py N)
      → Automatically resumes from the last completed round

    Tune COOLING_PAUSE_SECONDS at the top of this file if still crashing.
    """
    model = state.get("model_name", "?")
    mode  = state.get("mode",       "?")
    seed  = state.get("sim_seed",    0)
    print(f"  [{model}  |  {mode}  |  seed={seed}]")

    # ── Try to resume from a previous crash ───────────────────────────────────
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

        # ── Save checkpoint immediately after each round ───────────────────────
        _save_checkpoint(state, r)

        # ── Cooling pause (skip after the last round) ──────────────────────────
        if r < N_MACRO and COOLING_PAUSE_SECONDS > 0:
            print(f"  Cooling pause {COOLING_PAUSE_SECONDS}s ...", end="\r")
            time.sleep(COOLING_PAUSE_SECONDS)
            print(f"  Cooling done.{' ' * 30}")

        # ── Free memory ───────────────────────────────────────────────────────
        gc.collect()

    # ── Clean up checkpoints on success ───────────────────────────────────────
    _delete_checkpoints(state)
    print("  Checkpoints cleaned up.")

    return node_finalize(state)