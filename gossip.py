"""Spatial gossip propagation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from config import LAMBDA, MODE_GOSSIP, MU, N_AGENTS, SEED_GOSSIP
from data_structures import (
    Action,
    GossipEntry,
    MatchResult,
    SimulationState,
    update_reputation_summary,
)

if TYPE_CHECKING:
    from numpy.random import Generator


def flip(action: Action) -> Action:
    """Flip cooperate/defect (Binary Symmetric Channel)."""
    return "D" if action == "C" else "C"


def _euclidean_distances(
    pos_c: np.ndarray,
    pos_a: np.ndarray,
    pos_b: np.ndarray,
) -> np.ndarray:
    dist_ca = np.linalg.norm(pos_c - pos_a, axis=1)
    dist_cb = np.linalg.norm(pos_c - pos_b, axis=1)
    return np.minimum(dist_ca, dist_cb)


def broadcast_gossip(
    state: SimulationState,
    last_result: MatchResult,
    rng: Generator,
) -> SimulationState:
    """
    Propagate match outcome to non-participant agents.

    Reception: P_receive = exp(-LAMBDA * d_C), Bernoulli sample.
    Distortion: epsilon = 1 - exp(-MU * d_C); with prob epsilon flip one
    reported action (randomly chosen participant) via Binary Symmetric Channel.
    """
    if state["mode"] != MODE_GOSSIP:
        return state

    agent_a = last_result.agent_A
    agent_b = last_result.agent_B
    positions = state["positions"]
    agent_memory = state["agent_memory"]
    macro_round = last_result.macro_round

    pos_a = np.array(positions[agent_a], dtype=float)
    pos_b = np.array(positions[agent_b], dtype=float)

    others = [i for i in range(N_AGENTS) if i not in (agent_a, agent_b)]
    if not others:
        return state

    pos_others = np.array([positions[i] for i in others], dtype=float)
    d_c = _euclidean_distances(pos_others, pos_a, pos_b)

    # Step 1: reception
    p_receive = np.exp(-LAMBDA * d_c)
    u1 = rng.random(len(others))
    received_mask = u1 <= p_receive

    if not np.any(received_mask):
        return state

    received_indices = np.where(received_mask)[0]
    received_d = d_c[received_mask]

    # Step 2: distortion
    epsilon = 1.0 - np.exp(-MU * received_d)
    u2 = rng.random(len(received_indices))
    distorted_mask = u2 <= epsilon

    for local_idx, agent_idx in enumerate(received_indices):
        agent_c_id = others[agent_idx]
        d_val = float(received_d[local_idx])
        is_distorted = bool(distorted_mask[local_idx])

        reported_action_a: Action = last_result.action_A
        reported_action_b: Action = last_result.action_B

        if is_distorted:
            flip_target = int(rng.choice([agent_a, agent_b]))
            if flip_target == agent_a:
                reported_action_a = flip(reported_action_a)
            else:
                reported_action_b = flip(reported_action_b)

        # Step 3: write to listener memory
        c_mem = agent_memory[agent_c_id]
        c_mem.gossip_log.append(
            GossipEntry(
                macro_round=last_result.macro_round,
                match_index=last_result.match_index,
                source_A=agent_a,
                source_B=agent_b,
                reported_action_A=reported_action_a,
                reported_action_B=reported_action_b,
                is_distorted=is_distorted,
                distance_received=d_val,
            )
        )

        update_reputation_summary(
            c_mem, agent_a, reported_action_a, macro_round
        )
        update_reputation_summary(
            c_mem, agent_b, reported_action_b, macro_round
        )

    return state


def node_broadcast_gossip(state: SimulationState) -> SimulationState:
    """LangGraph node wrapper for gossip propagation."""
    last_result = state.get("last_result")
    if last_result is None:
        return state

    rng = np.random.default_rng(
        SEED_GOSSIP
        + state["sim_seed"] * 100_000
        + last_result.macro_round * 1000
        + last_result.match_index
    )
    return broadcast_gossip(state, last_result, rng)
