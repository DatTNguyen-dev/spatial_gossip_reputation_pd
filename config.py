"""Simulation constants."""

GRID_W = 6
GRID_H = 6
N_AGENTS = 30
N_MACRO = 10
MATCHES_PER_ROUND = 435  # C(30, 2)

PAYOFF = {
    ("C", "C"): (3, 3),
    ("C", "D"): (0, 5),
    ("D", "C"): (5, 0),
    ("D", "D"): (1, 1),
}

# Gossip parameters (ablate LAMBDA: 0.1, 0.3, 1.0)
LAMBDA = 0.3
MU = 0.15  # 0.5 * LAMBDA when LAMBDA = 0.3

MODE_NO_INFO = "no_info"
MODE_GOSSIP = "gossip"
MODE_REPUTATION = "reputation"

MODELS = ["mistral:7b-instruct", "llama3.1:8b", "qwen3:8b"]

SEED_PLACEMENT = 42
SEED_PAIRING = 100
SEED_GOSSIP = 200
SEED_LLM = 300
