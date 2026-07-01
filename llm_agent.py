"""LLM prompt construction and Ollama API calls."""

from __future__ import annotations

import os
import re
import shutil
import subprocess as _subprocess
import time
from typing import List

import requests

from config import MODE_GOSSIP, MODE_NO_INFO, MODE_REPUTATION, N_MACRO, PAYOFF
from data_structures import (
    Action,
    GossipEntry,
    MatchResult,
    PersonalEntry,
    ReputationEntry,
    SimulationState,
)

# ==============================================================================
#  INFERENCE SETTINGS
# ==============================================================================

# Number of model layers offloaded to GPU.
# 0  = CPU-only inference
# 99 = full GPU offload (requires sufficient VRAM)
NUM_GPU_LAYERS: int = 99

# Optional delay (seconds) inserted between successive Ollama calls.
SLEEP_BETWEEN_CALLS: float = 0.0


# ==============================================================================
#  CONTEXT FORMATTERS
# ==============================================================================

def summarize_direct_history(entries: List[PersonalEntry]) -> str:
    lines = ["Your direct play history with this agent (most recent 5 matches):"]
    for e in entries[-5:]:
        lines.append(
            f"  Round {e.macro_round}: "
            f"you={e.my_action}, opponent={e.opp_action} "
            f"(your payoff: {int(e.my_payoff)})"
        )
    return "\n".join(lines)


def format_gossip_context(
    summary: ReputationEntry | None,
    relevant: List[GossipEntry],
    max_entries: int = 5,
) -> str:
    parts: List[str] = []
    if summary is not None:
        total = summary.total_C + summary.total_D
        rate  = f"{summary.total_C / total:.0%}" if total else "0%"
        parts.append(
            f"Reputation (from rumours): cooperated {summary.total_C}/{total} times "
            f"({rate} rate, last seen round {summary.last_seen_round})."
        )
    if relevant:
        parts.append("Recent rumours involving this agent:")
        for e in relevant[-max_entries:]:
            parts.append(
                f"  Round {e.macro_round}: heard that Agent {e.source_A} "
                f"and Agent {e.source_B} played "
                f"({e.reported_action_A}, {e.reported_action_B})."
            )
    return "\n".join(parts) if parts else "No rumours available about this agent."


def format_global_rep(entry: ReputationEntry) -> str:
    total = entry.total_C + entry.total_D
    rate  = f"{entry.total_C / total:.0%}" if total else "0%"
    return (
        f"Public record: cooperated {entry.total_C}/{total} times "
        f"({rate} cooperation rate, last updated round {entry.last_seen_round})."
    )


# ==============================================================================
#  CONTEXT BUILDER
# ==============================================================================

def build_context_for_agent(
    self_id: int,
    opponent_id: int,
    state: SimulationState,
) -> str:
    memory = state["agent_memory"][self_id]
    mode   = state["mode"]

    if mode == MODE_NO_INFO:
        direct = [e for e in memory.personal_log if e.opponent_id == opponent_id]
        if not direct:
            return f"You have never played against Agent {opponent_id}."
        return summarize_direct_history(direct)

    if mode == MODE_GOSSIP:
        relevant = [
            e for e in memory.gossip_log
            if opponent_id in (e.source_A, e.source_B)
        ]
        summary = memory.reputation_summary.get(opponent_id)
        if summary is None and not relevant:
            return f"You have heard nothing about Agent {opponent_id}."
        return format_gossip_context(summary, relevant)

    if mode == MODE_REPUTATION:
        global_rep = state.get("global_reputation_table", {}).get(opponent_id)
        if global_rep is None:
            return f"Agent {opponent_id} has no recorded history yet."
        return format_global_rep(global_rep)

    return f"No information about Agent {opponent_id}."


# ==============================================================================
#  PROMPT BUILDER
# ==============================================================================

def _build_agent_prompt(
    self_id: int,
    opponent_id: int,
    state: SimulationState,
) -> str:
    context    = build_context_for_agent(self_id, opponent_id, state)
    model_name = state.get("model_name", "")

    return (
        f"You are Agent {self_id} in a social network experiment.\n"
        f"You are about to play ONE round of Prisoner's Dilemma "
        f"with Agent {opponent_id}.\n"
        f"This game repeats for {N_MACRO} rounds total — "
        f"your reputation and others' reputations carry across rounds.\n"
        f"\n"
        f"=== PAYOFF MATRIX ===\n"
        f"- Both Cooperate    (C, C): you +3, opponent +3\n"
        f"- You C, they D    (C, D): you +0, opponent +5\n"
        f"- You D, they C    (D, C): you +5, opponent +0\n"
        f"- Both Defect       (D, D): you +1, opponent +1\n"
        f"\n"
        f"=== INFORMATION ABOUT AGENT {opponent_id} ===\n"
        f"{context}\n"
        f"\n"
        f"=== YOUR DECISION ===\n"
        f"Reply with ONLY one letter on a single line: C or D"
    )


def node_build_prompt(state: SimulationState) -> SimulationState:
    a_id, b_id = state["current_pair"]
    state["prompt_A"] = _build_agent_prompt(a_id, b_id, state)
    state["prompt_B"] = _build_agent_prompt(b_id, a_id, state)
    return state


# ==============================================================================
#  OLLAMA SERVER MANAGEMENT
#  These utilities ensure the Ollama server is reachable before each API call.
#  On a local machine, _ensure_server_running() is a no-op when the server is
#  already running. On notebook environments (e.g. Colab), the server process
#  may be killed after a period of inactivity; these functions detect and
#  transparently recover from that condition.
# ==============================================================================

def _find_ollama() -> str:
    """Return the path to the ollama executable."""
    for candidate in ["/usr/bin/ollama", "/usr/local/bin/ollama"]:
        if os.path.exists(candidate):
            return candidate
    found = shutil.which("ollama")
    if found:
        return found
    raise FileNotFoundError("ollama binary not found on this system.")


def _is_server_up() -> bool:
    """Return True if the Ollama server is reachable on port 11434."""
    try:
        r = requests.get("http://localhost:11434", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _ensure_server_running() -> None:
    """
    Ensure the Ollama server is reachable, starting it if necessary.

    On a local machine this is typically a no-op. On notebook runtimes
    where background processes may be terminated after idle periods,
    this function detects the condition and restarts the server.
    """
    if _is_server_up():
        return

    print("[INFO] Ollama server is not responding — restarting...")
    ollama_path = _find_ollama()
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "127.0.0.1:11434"
    _subprocess.Popen(
        [ollama_path, "serve"],
        env=env,
        stdout=_subprocess.DEVNULL,
        stderr=_subprocess.DEVNULL,
    )

    # Poll until the server is ready, up to 30 seconds.
    for i in range(30):
        time.sleep(1)
        if _is_server_up():
            print(f"[INFO] Ollama server restarted successfully after {i + 1}s ✅")
            return

    raise RuntimeError("Ollama server failed to start within 30 seconds.")


# ==============================================================================
#  OLLAMA API CALL
# ==============================================================================

def call_ollama(
    model_name: str,
    prompt: str,
    max_retries: int = 5,
) -> str:
    """Send a prompt to Ollama and return the raw text response.

    Automatically restarts the server if a connection error is detected
    (Colab frequently kills background processes after idle periods).

    Timeout tuple: (connect_timeout, read_timeout)
      - connect_timeout=10s : maximum time to establish the TCP connection
      - read_timeout=600s   : maximum time to wait for generation to complete
                              (first call may take 1-2 min while loading the
                              model into VRAM)

    Retry backoff: 5s, 10s, 20s, 40s, 80s (exponential, base 2, factor 5).
    """
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            # Transparently restart server if it was killed by Colab.
            _ensure_server_running()

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model" : model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_gpu": NUM_GPU_LAYERS,  # 0 = CPU only, 99 = full GPU
                    },
                },
                timeout=(10, 600),  # (connect_timeout, read_timeout)
            )
            response.raise_for_status()

            if SLEEP_BETWEEN_CALLS > 0:
                time.sleep(SLEEP_BETWEEN_CALLS)

            return response.json()["response"]

        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            wait = 5 * (2 ** attempt)  # 5s, 10s, 20s, 40s, 80s
            print(
                f"[WARN] Ollama attempt {attempt + 1}/{max_retries} failed: {exc}. "
                f"Retrying in {wait}s..."
            )
            time.sleep(wait)

        except Exception as exc:
            raise RuntimeError(f"Ollama non-retryable error: {exc}") from exc

    raise RuntimeError(
        f"Ollama failed after {max_retries} retries. "
        f"Last error: {last_error}. "
        f"Verify Ollama is running: ollama serve"
    )


# ==============================================================================
#  ACTION PARSER
# ==============================================================================

def parse_action(raw_text: str) -> Action:
    text = raw_text.strip().upper()
    if re.search(r"\bC\b", text):      return "C"
    if re.search(r"\bD\b", text):      return "D"
    if re.search(r"\bCOOPERAT", text): return "C"
    if re.search(r"\bDEFECT",   text): return "D"
    print(f"[WARN] parse_action: unexpected output — defaulting to C. Raw: {raw_text!r}")
    return "C"


# ==============================================================================
#  SIMULATION NODE
# ==============================================================================

def node_call_llm(state: SimulationState) -> SimulationState:
    raw_a = call_ollama(state["model_name"], state["prompt_A"])
    raw_b = call_ollama(state["model_name"], state["prompt_B"])

    action_a = parse_action(raw_a)
    action_b = parse_action(raw_b)
    payoff_a, payoff_b = PAYOFF[(action_a, action_b)]

    a_id, b_id = state["current_pair"]
    state["last_result"] = MatchResult(
        macro_round=state["macro_round"],
        match_index=state["match_index"],
        agent_A=a_id,
        agent_B=b_id,
        action_A=action_a,
        action_B=action_b,
        payoff_A=float(payoff_a),
        payoff_B=float(payoff_b),
    )
    return state