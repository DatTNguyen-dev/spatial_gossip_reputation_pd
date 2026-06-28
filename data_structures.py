"""Core data structures for the LangGraph simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np

from config import GRID_H, GRID_W, N_AGENTS

Action = Literal["C", "D"]
Position = Tuple[int, int]


@dataclass
class PersonalEntry:
    macro_round: int
    match_index: int
    opponent_id: int
    my_action: Action
    opp_action: Action
    my_payoff: float
    opp_payoff: float


@dataclass
class GossipEntry:
    macro_round: int
    match_index: int
    source_A: int
    source_B: int
    reported_action_A: Action
    reported_action_B: Action
    is_distorted: bool
    distance_received: float


@dataclass
class ReputationEntry:
    total_C: int = 0
    total_D: int = 0
    last_seen_round: int = 0


@dataclass
class AgentMemory:
    agent_id: int
    personal_log: List[PersonalEntry] = field(default_factory=list)
    gossip_log: List[GossipEntry] = field(default_factory=list)
    reputation_summary: Dict[int, ReputationEntry] = field(default_factory=dict)


@dataclass
class MatchResult:
    macro_round: int
    match_index: int
    agent_A: int
    agent_B: int
    action_A: Action
    action_B: Action
    payoff_A: float
    payoff_B: float


@dataclass
class RoundStats:
    macro_round: int
    cooperation_rate: float
    mutual_coop_rate: float
    mutual_defect_rate: float
    avg_payoff: float
    gini_coefficient: float
    defector_isolation: float
    spatial_coop_map: Dict[int, float]


class SimulationState(TypedDict, total=False):
    sim_seed: int
    positions: Dict[int, Position]
    mode: str
    model_name: str
    macro_round: int
    match_index: int
    match_schedule: List[Tuple[int, int]]
    current_pair: Tuple[int, int]
    last_result: Optional[MatchResult]
    agent_memory: Dict[int, AgentMemory]
    global_reputation_table: Dict[int, ReputationEntry]
    history: List[RoundStats]
    current_round_results: List[MatchResult]
    prompt_A: str
    prompt_B: str
    final_output: dict


def empty_agent_memory(agent_id: int) -> AgentMemory:
    return AgentMemory(agent_id=agent_id)


def update_reputation_summary(
    memory: AgentMemory,
    other_id: int,
    other_action: Action,
    current_round: int,
) -> None:
    entry = memory.reputation_summary.get(other_id, ReputationEntry())
    if other_action == "C":
        entry.total_C += 1
    else:
        entry.total_D += 1
    entry.last_seen_round = current_round
    memory.reputation_summary[other_id] = entry


def initialize(
    mode: str,
    model_name: str,
    seed_placement: int = 42,
    sim_seed: int = 0,
) -> SimulationState:
    rng_place = np.random.default_rng(seed_placement)

    all_cells = [(x, y) for x in range(GRID_W) for y in range(GRID_H)]
    chosen_indices = rng_place.choice(len(all_cells), size=N_AGENTS, replace=False)
    chosen = [all_cells[i] for i in chosen_indices]
    positions = {agent_id: chosen[agent_id] for agent_id in range(N_AGENTS)}

    agent_memory = {i: empty_agent_memory(i) for i in range(N_AGENTS)}

    return SimulationState(
        sim_seed=sim_seed,
        positions=positions,
        mode=mode,
        model_name=model_name,
        macro_round=1,
        match_index=0,
        match_schedule=[],
        current_pair=(0, 1),
        last_result=None,
        agent_memory=agent_memory,
        global_reputation_table={},
        history=[],
        current_round_results=[],
        prompt_A="",
        prompt_B="",
    )

_REQUIRED_FIELDS = {
    "positions", "mode", "model_name", "macro_round",
    "match_index", "match_schedule", "agent_memory",
    "global_reputation_table", "history", "current_round_results",
}


def validate_initial_state(state: SimulationState) -> None:
    missing = _REQUIRED_FIELDS - state.keys()
    if missing:
        raise ValueError(f"Missing required fields in initial state: {missing}")