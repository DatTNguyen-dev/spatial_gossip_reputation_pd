"""Matplotlib plotting utilities for the spatial gossip PD simulation."""

from __future__ import annotations

from math import ceil
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from config import GRID_H, GRID_W, LAMBDA, MODE_GOSSIP, MODE_NO_INFO, MODE_REPUTATION, MU

ExperimentKey = Tuple[str, str]

# Consistent color per model across all plots
_MODEL_COLORS = {
    "mistral:7b-instruct": "tab:blue",
    "llama3.1:8b":         "tab:orange",
}

# Line style per mode for cooperation-rate plot
_MODE_LINESTYLE = {
    MODE_NO_INFO:    "dotted",
    MODE_GOSSIP:     "dashed",
    MODE_REPUTATION: "solid",
}


def _mode_title(mode: str) -> str:
    return {
        MODE_NO_INFO:    "No Information",
        MODE_GOSSIP:     "Gossip",
        MODE_REPUTATION: "Reputation",
    }.get(mode, mode)


def _model_color(model_name: str) -> str:
    return _MODEL_COLORS.get(model_name, "tab:gray")


def _extract_metric(
    results: Dict[ExperimentKey, dict[str, Any]],
    metric: str,
):
    """Yield (model_name, mode, rounds, mean_list, std_list) for every config."""
    for (model_name, mode), aggregated in results.items():
        rounds = aggregated["macro_rounds"]
        mean   = aggregated[metric]["mean"]
        std    = aggregated[metric]["std"]
        yield model_name, mode, rounds, mean, std


# ==============================================================================
#  PLOT 1 — Cooperation rate over macro rounds
# ==============================================================================

def plot_cooperation_rate(
    results: Dict[ExperimentKey, dict[str, Any]],
) -> plt.Figure:
    """
    Line chart: cooperation rate per macro round.
    One subplot per information mode, one line per model, ±std shaded.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    mode_to_ax = {
        MODE_NO_INFO:    axes[0],
        MODE_GOSSIP:     axes[1],
        MODE_REPUTATION: axes[2],
    }

    for model_name, mode, rounds, mean, std in _extract_metric(
        results, "cooperation_rate"
    ):
        ax       = mode_to_ax[mode]
        mean_arr = np.array(mean)
        std_arr  = np.array(std)
        color    = _model_color(model_name)

        ax.plot(rounds, mean_arr, label=model_name, color=color)
        ax.fill_between(
            rounds,
            mean_arr - std_arr,
            mean_arr + std_arr,
            alpha=0.2,
            color=color,
        )
        ax.set_title(_mode_title(mode))
        ax.set_xlabel("Macro round")
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(range(1, len(rounds) + 1))

    axes[0].set_ylabel("Cooperation rate")

    # Figure-level legend — collects handles from all subplots, deduplicates
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=fig.transFigure,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    return fig


# ==============================================================================
#  PLOT 2 — Average payoff (final round) per configuration
# ==============================================================================

def plot_payoff_distribution(
    results: Dict[ExperimentKey, dict[str, Any]],
) -> plt.Figure:
    """
    Bar chart: mean ± std of average payoff in the final macro round.
    One bar per (model, mode) configuration.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    labels, means, errs, colors = [], [], [], []

    for (model_name, mode), aggregated in results.items():
        labels.append(f"{model_name}\n{_mode_title(mode)}")
        means.append(aggregated["avg_payoff"]["mean"][-1])
        errs.append(aggregated["avg_payoff"]["std"][-1])
        colors.append(_model_color(model_name))

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=errs, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Average payoff (final macro round)")
    ax.set_title("Payoff comparison across configurations (mean ± std over seeds)")
    ax.set_ylim(0, 5)       # PD payoff range: [0, T=5]
    ax.axhline(3, color="gray", linestyle="--", linewidth=0.8, label="Mutual coop (R=3)")
    ax.axhline(1, color="red",  linestyle=":",  linewidth=0.8, label="Mutual defect (P=1)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


# ==============================================================================
#  PLOT 3 — Defector isolation rate (final round)
# ==============================================================================

def plot_defector_isolation(
    results: Dict[ExperimentKey, dict[str, Any]],
) -> plt.Figure:
    """
    Bar chart: fraction of (D,C) outcomes where the cooperator already held
    negative reputation info on the defector — in the final macro round.
    """
    fig, ax = plt.subplots(figsize=(13, 5))
    labels, bars, errs, colors = [], [], [], []

    for (model_name, mode), aggregated in results.items():
        labels.append(f"{model_name}\n{_mode_title(mode)}")
        bars.append(aggregated["defector_isolation"]["mean"][-1])
        errs.append(aggregated["defector_isolation"]["std"][-1])
        colors.append(_model_color(model_name))

    x = np.arange(len(labels))
    ax.bar(x, bars, yerr=errs, capsize=4, color=colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")     # backward-compatible
    ax.set_ylabel("Isolation rate")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "Defector isolation in final macro round\n"
        "(fraction of exploitations where cooperator had prior warning)"
    )
    fig.tight_layout()
    return fig


# ==============================================================================
#  PLOT 4 — Gossip accuracy vs. distance
# ==============================================================================

def plot_gossip_accuracy_vs_distance(
    final_memory: dict[int, Any],
    lambda_value: float = LAMBDA,
    mu_value: float = MU,
    n_bins: int = 10,
) -> plt.Figure:
    """
    Scatter + empirical bin accuracy + theoretical curve exp(-(λ+μ)·d).
    Verifies that the gossip distortion model matches observed accuracy.
    """
    distances, correctness = [], []

    for memory in final_memory.values():
        for entry in memory.gossip_log:
            distances.append(entry.distance_received)
            correctness.append(0 if entry.is_distorted else 1)

    fig, ax = plt.subplots(figsize=(8, 5))

    if not distances:
        ax.text(0.5, 0.5, "No gossip entries (MODE_NO_INFO or MODE_REPUTATION run)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Gossip accuracy vs distance")
        return fig

    d_arr = np.array(distances,    dtype=float)
    c_arr = np.array(correctness,  dtype=float)

    # Hexbin instead of raw scatter — avoids overplotting ~120k points
    hb = ax.hexbin(d_arr, c_arr, gridsize=30, cmap="Blues",
                   mincnt=1, alpha=0.7, label="Sample density")
    fig.colorbar(hb, ax=ax, label="Count")

    # Empirical accuracy per distance bin
    bins    = np.linspace(d_arr.min(), d_arr.max(), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    empirical = [
        float(np.mean(c_arr[(d_arr >= lo) & (d_arr < hi)]))
        if np.any((d_arr >= lo) & (d_arr < hi)) else np.nan
        for lo, hi in zip(bins[:-1], bins[1:])
    ]
    ax.plot(centers, empirical,
            color="tab:orange", linewidth=2, marker="o", markersize=4,
            label="Empirical accuracy (binned)")

    # Theoretical curve: P_accurate = exp(-(λ+μ)·d)
    x_th = np.linspace(d_arr.min(), d_arr.max(), 200)
    y_th = np.exp(-(lambda_value + mu_value) * x_th)
    ax.plot(x_th, y_th,
            linestyle="--", color="black", linewidth=1.5,
            label=f"Theory: exp(−(λ+μ)d), λ={lambda_value}, μ={mu_value}")

    ax.set_xlabel("Distance from match (Euclidean)")
    ax.set_ylabel("Accuracy (1 = undistorted)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper right")
    ax.set_title("Gossip accuracy vs distance")
    fig.tight_layout()
    return fig


# ==============================================================================
#  PLOT 5 — Spatial cooperation heatmap (6×6 grid)
# ==============================================================================

def plot_spatial_heatmaps(
    results: Dict[ExperimentKey, dict[str, Any]],
    positions: dict[int, tuple[int, int]],
) -> plt.Figure:
    """
    One 6×6 heatmap per configuration.
    Cell colour = mean cooperation rate of the agent occupying that cell.
    Empty cells (no agent) shown as white.
    """
    n_plots = len(results)
    n_cols  = 3
    n_rows  = ceil(n_plots / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        squeeze=False,          # always return 2D array
    )
    axes_arr = axes.reshape(-1)

    for ax, ((model_name, mode), aggregated) in zip(axes_arr, results.items()):
        heatmap  = np.full((GRID_H, GRID_W), np.nan)
        coop_map = aggregated["spatial_coop_map_mean"]

        for agent_id, (x, y) in positions.items():
            heatmap[y, x] = coop_map[agent_id]

        image = ax.imshow(
            heatmap, vmin=0.0, vmax=1.0,
            cmap="viridis", origin="lower",
        )
        ax.set_title(f"{model_name}\n{_mode_title(mode)}")
        ax.set_xticks(range(GRID_W))
        ax.set_yticks(range(GRID_H))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Coop rate")

    # Turn off unused subplots
    for ax in axes_arr[n_plots:]:
        ax.axis("off")

    fig.suptitle("Spatial cooperation map (mean over seeds, final round)", y=1.01)
    fig.tight_layout()
    return fig

# ==============================================================================
#  PLOT 6 — Payoff inequality (Gini coefficient)
# ==============================================================================

def plot_gini_coefficient(
    results: Dict[ExperimentKey, dict[str, Any]],
) -> plt.Figure:
    """
    Line chart: Gini coefficient of per-agent payoff per macro round.
    One subplot per information mode, one line per model, ±std shaded.

    Measures wealth stratification among agents: 0 = perfectly equal
    payoff distribution, higher values = a subset of agents
    accumulating disproportionately more payoff than the rest.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    mode_to_ax = {
        MODE_NO_INFO:    axes[0],
        MODE_GOSSIP:     axes[1],
        MODE_REPUTATION: axes[2],
    }

    for model_name, mode, rounds, mean, std in _extract_metric(
        results, "gini_coefficient"
    ):
        ax       = mode_to_ax[mode]
        mean_arr = np.array(mean)
        std_arr  = np.array(std)
        color    = _model_color(model_name)

        ax.plot(rounds, mean_arr, label=model_name, color=color)
        ax.fill_between(
            rounds,
            mean_arr - std_arr,
            mean_arr + std_arr,
            alpha=0.2,
            color=color,
        )
        ax.set_title(_mode_title(mode))
        ax.set_xlabel("Macro round")
        ax.set_xticks(range(1, len(rounds) + 1))

    axes[0].set_ylabel("Gini coefficient (payoff)")
    # Start at 0 to avoid a truncated-axis distortion; let matplotlib pick the upper bound automatically since Gini values here stay well under 1.0 (unlike cooperation_rate, which naturally spans the full [0,1] range).
    for ax in axes:
        ax.set_ylim(bottom=0.0)

    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(
        handles, labels,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=fig.transFigure,
    )
    fig.suptitle(
        "Payoff inequality across macro rounds (mean ± std over seeds)", y=1.04
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    return fig

# ==============================================================================
#  PLOT ALL — convenience wrapper
# ==============================================================================

def plot_all(
    results: Dict[ExperimentKey, dict[str, Any]],
    positions: Optional[dict[int, tuple[int, int]]] = None,
    final_memory: Optional[dict[int, Any]] = None,
    lambda_value: Optional[float] = None,
    mu_value: Optional[float] = None,
) -> Dict[str, plt.Figure]:
    """
    Generate all available figures and return them as a dict.

    Parameters
    ----------
    results       : output of runner.run_all_experiments()
    positions     : agent_id -> (x, y); auto-extracted from results if None
    final_memory  : agent_id -> AgentMemory; needed for gossip accuracy plot
    lambda_value  : gossip reception decay (defaults to config.LAMBDA)
    mu_value      : gossip distortion decay (defaults to config.MU)

    Returns
    -------
    Dict mapping figure name -> matplotlib Figure object.
    Call fig.savefig(...) or plt.show() on each.
    """
    figures: Dict[str, plt.Figure] = {}

    figures["cooperation_rate"]   = plot_cooperation_rate(results)
    figures["payoff_distribution"] = plot_payoff_distribution(results)
    figures["defector_isolation"] = plot_defector_isolation(results)
    figures["gini_coefficient"] = plot_gini_coefficient(results)

    # Gossip accuracy plot — only meaningful for MODE_GOSSIP runs
    if final_memory is not None:
        figures["gossip_accuracy"] = plot_gossip_accuracy_vs_distance(
            final_memory,
            lambda_value=lambda_value if lambda_value is not None else LAMBDA,
            mu_value=mu_value if mu_value is not None else MU,
        )

    # Spatial heatmap — extract positions from results if not provided
    if positions is None and results:
        positions = next(iter(results.values())).get("positions")
    if positions is not None:
        figures["spatial_heatmaps"] = plot_spatial_heatmaps(results, positions)

    return figures


# ==============================================================================
#  SAVE HELPER
# ==============================================================================

def save_all(
    figures: Dict[str, plt.Figure],
    output_dir: str = ".",
    dpi: int = 150,
) -> None:
    """Save every figure returned by plot_all() to <output_dir>/<name>.png."""
    from pathlib import Path
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        path = out / f"{name}.png"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        print(f"Saved {path}")