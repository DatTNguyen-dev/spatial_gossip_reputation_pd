"""
main.py

USAGE
─────
    python main.py          → show status of all 9 configs
    python main.py 1        → run config #1
    python main.py 2        → run config #2
    python main.py plot     → generate plots from saved results

CONFIG TABLE
────────────
    #1  mistral:7b-instruct  |  no_info
    #2  mistral:7b-instruct  |  gossip
    #3  mistral:7b-instruct  |  reputation
    #4  llama3.1:8b          |  no_info
    #5  llama3.1:8b          |  gossip
    #6  llama3.1:8b          |  reputation

RECOMMENDED WORKFLOW
────────────────────
    1. Run python main.py       to see which configs are done/pending
    2. Run python main.py 1     wait until finished
    3. Run python main.py 2     wait until finished
    ...
    8. Run python main.py plot to generate all charts
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import requests

from config import LAMBDA, MODE_GOSSIP, MODE_NO_INFO, MODE_REPUTATION, MODELS, MU
from runner import run_configuration

# ── Directories ───────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
PLOTS_DIR   = Path("plots")

# ── How many seeds per config ─────────────────────────────────────────────────
# Number of independent seeds per (model, mode) configuration.
N_SEEDS = 5

# ── All 6 configurations in fixed order ───────────────────────────────────────
MODES = [MODE_NO_INFO, MODE_GOSSIP, MODE_REPUTATION]
CONFIGS: list[tuple[str, str]] = [
    (model, mode)
    for model in MODELS
    for mode in MODES
]


ExperimentKey = Tuple[str, str]


# ==============================================================================
#  JSON HELPERS  (int keys get converted to str by JSON — restore them here)
# ==============================================================================

def _save(model_name: str, mode: str, aggregated: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = _result_path(model_name, mode)
    slim = {k: v for k, v in aggregated.items() if k != "final_memory"}
    path.write_text(json.dumps(slim, indent=2, default=str), encoding="utf-8")
    return path


def _load(model_name: str, mode: str) -> dict[str, Any] | None:
    path = _result_path(model_name, mode)
    if not path.exists():
        return None
    data: dict = json.loads(path.read_text(encoding="utf-8"))
    if "spatial_coop_map_mean" in data:
        data["spatial_coop_map_mean"] = {
            int(k): v for k, v in data["spatial_coop_map_mean"].items()
        }
    if "positions" in data:
        data["positions"] = {
            int(k): tuple(v) for k, v in data["positions"].items()
        }
    return data


def _result_path(model_name: str, mode: str) -> Path:
    safe = model_name.replace(":", "_").replace("/", "_")
    return RESULTS_DIR / f"{safe}_{mode}.json"


# ==============================================================================
#  OLLAMA MODEL UNLOADER
# ==============================================================================

def _unload_model(model_name: str) -> None:
    """
    Tell Ollama to evict the model from VRAM immediately.
    keep_alive=0 means: unload as soon as this request is done.
    """
    try:
        requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model_name, "prompt": "", "keep_alive": 0},
            timeout=30,
        )
        print(f"  VRAM freed — {model_name} unloaded.")
    except Exception as exc:
        print(f"  [WARN] Could not unload model from VRAM: {exc}")


# ==============================================================================
#  STATUS TABLE
# ==============================================================================

def _print_status() -> None:
    """Show which configs are done, pending, or skipped."""
    print()
    print(f"  {'#':<4} {'Model':<25} {'Mode':<12} {'Status'}")
    print(f"  {'─'*4} {'─'*25} {'─'*12} {'─'*10}")

    for idx, (model, mode) in enumerate(CONFIGS, 1):
        done = _result_path(model, mode).exists()
        status = "✓  done" if done else "·  pending"
        print(f"  {idx:<4} {model:<25} {mode:<12} {status}")

    total  = len(CONFIGS)
    n_done = sum(1 for m, mo in CONFIGS if _result_path(m, mo).exists())
    print()
    print(f"  Progress: {n_done}/{total} completed")

    if n_done < total:
        next_idx = next(
            i + 1 for i, (m, mo) in enumerate(CONFIGS)
            if not _result_path(m, mo).exists()
        )
        print(f"  Next    : python main.py {next_idx}")
    else:
        print("  All done! Run: python main.py plot")
    print()


# ==============================================================================
#  RUN ONE CONFIG
# ==============================================================================

def _run_config(idx: int) -> None:
    """Run configuration number `idx` (1-based)."""
    if not (1 <= idx <= len(CONFIGS)):
        print(f"  Invalid config number: {idx}  (must be 1–{len(CONFIGS)})")
        sys.exit(1)

    model_name, mode = CONFIGS[idx - 1]
    save_path = _result_path(model_name, mode)

    print()
    print(f"  Config #{idx}:  {model_name}  |  {mode}")
    print(f"  Seeds   :  {N_SEEDS}")
    print()

    # ── Skip if already done ──────────────────────────────────────────────────
    if save_path.exists():
        print(f"  ↩  Already completed ({save_path.name}).")
        print(f"     Delete the file and re-run to redo this config.")
        _print_next(idx)
        return

    # ── Run ───────────────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        aggregated, _ = run_configuration(
            model_name=model_name,
            mode=mode,
            n_seeds=N_SEEDS,
        )
    except RuntimeError as exc:
        print(f"\n  [ERROR] {exc}")
        print(f"  Config #{idx} failed. Fix the issue and re-run: python main.py {idx}")
        _unload_model(model_name)
        sys.exit(1)

    elapsed = time.time() - t0

    # ── Save ──────────────────────────────────────────────────────────────────
    saved_path = _save(model_name, mode, aggregated)
    coop = aggregated["cooperation_rate"]["mean"][-1]

    print()
    print(f"  ✓  Config #{idx} done in {elapsed/60:.1f} min")
    print(f"     Final cooperation rate : {coop:.3f}")
    print(f"     Saved to               : {saved_path}")

    # ── Unload model from VRAM before returning ───────────────────────────────
    _unload_model(model_name)

    _print_next(idx)


def _print_next(current_idx: int) -> None:
    """Print what the user should run next."""
    print()
    remaining = [
        i + 1 for i, (m, mo) in enumerate(CONFIGS)
        if not _result_path(m, mo).exists()
    ]
    if remaining:
        print(f"  Next: python main.py {remaining[0]}")
    else:
        print(f"  All {len(CONFIGS)} configs done!  →  python main.py plot")
    print()


# ==============================================================================
#  PLOT
# ==============================================================================

def _plot() -> None:
    """Load all saved results and generate plots."""
    from plotting import plot_all, save_all

    results: Dict[ExperimentKey, dict[str, Any]] = {}
    for model, mode in CONFIGS:
        data = _load(model, mode)
        if data is not None:
            results[(model, mode)] = data
        else:
            print(f"  [SKIP] {model} | {mode} — not found")

    if not results:
        print("  No results found. Run some configs first.")
        sys.exit(1)

    n = len(results)
    print(f"\n  Plotting {n}/{len(CONFIGS)} completed configurations ...")

    positions = next(iter(results.values())).get("positions")

    PLOTS_DIR.mkdir(exist_ok=True)
    figures = plot_all(
        results=results,
        positions=positions,
        lambda_value=LAMBDA,
        mu_value=MU,
    )
    save_all(figures, output_dir=str(PLOTS_DIR))
    print(f"  Plots saved to {PLOTS_DIR}/")
    print()


# ==============================================================================
#  ENTRY POINT
# ==============================================================================

def main() -> None:
    args = sys.argv[1:]

    if not args:
        # No argument → show status
        _print_status()

    elif args[0] == "plot":
        _plot()

    elif args[0].isdigit():
        _run_config(int(args[0]))

    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()