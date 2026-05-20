"""Neural-activity visualization for Phase 0 (AC4).

Provides a heatmap of (T × N) activity plus optional category bands so
sensory/inter/motor structure is visible at a glance.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


CATEGORY_COLORS = {
    "pharyngeal": "#888888",
    "sensory": "#1f77b4",
    "interneuron": "#2ca02c",
    "motor": "#d62728",
    "sex_specific": "#9467bd",
    "other_neuron": "#bcbd22",
}


def plot_activity_matrix(
    activity_history: np.ndarray,
    neuron_names: Sequence[str],
    *,
    categories: Sequence[str] | None = None,
    sort_by_category: bool = True,
    title: str = "Neural Activity",
    figsize: tuple[float, float] = (15, 8),
):
    """Heatmap of neuron activity over time.

    Args:
        activity_history: (T, N) float array with values in [-1, 1].
        neuron_names: length-N list of neuron names.
        categories: optional length-N list of category labels matching
            `CATEGORY_COLORS`. Used to colour-band the y-axis if present.
        sort_by_category: reorder neurons so each category is contiguous
            (makes structural patterns easier to see).
        title: figure title.
        figsize: matplotlib figure size in inches.

    Returns:
        The created matplotlib Figure.
    """
    T, N = activity_history.shape
    assert len(neuron_names) == N
    if categories is not None:
        assert len(categories) == N

    if sort_by_category and categories is not None:
        order = sorted(range(N), key=lambda i: (
            list(CATEGORY_COLORS.keys()).index(categories[i])
            if categories[i] in CATEGORY_COLORS else 999,
            neuron_names[i],
        ))
    else:
        order = list(range(N))

    activity_sorted = activity_history[:, order]
    sorted_categories = [categories[i] for i in order] if categories else None
    sorted_names = [neuron_names[i] for i in order]

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=[0.02, 1], wspace=0.02)
    ax_band = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1], sharey=ax_band)

    im = ax.imshow(
        activity_sorted.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
        origin="lower",
    )
    ax.set_xlabel("tick")
    ax.set_ylabel("neuron (sorted by category)" if sort_by_category else "neuron")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("V")

    # Category band (vertical strip on the left)
    if sorted_categories is not None:
        ax_band.set_xlim(0, 1)
        ax_band.set_ylim(-0.5, N - 0.5)
        for i, cat in enumerate(sorted_categories):
            color = CATEGORY_COLORS.get(cat, "#cccccc")
            ax_band.add_patch(mpatches.Rectangle((0, i - 0.5), 1, 1, color=color))
        ax_band.set_xticks([])
        ax_band.tick_params(axis="y", labelleft=False)
        # Legend
        handles = [
            mpatches.Patch(color=c, label=k)
            for k, c in CATEGORY_COLORS.items()
            if k in set(sorted_categories)
        ]
        ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=8)
    else:
        ax_band.axis("off")

    return fig


def plot_trace(
    activity_history: np.ndarray,
    neuron_names: Sequence[str],
    trace_names: Sequence[str],
    name_to_idx: dict,
    *,
    title: str = "Neuron traces",
    figsize: tuple[float, float] = (12, 5),
):
    """Line plot of a small set of named neurons over time."""
    fig, ax = plt.subplots(figsize=figsize)
    for name in trace_names:
        idx = name_to_idx[name]
        ax.plot(activity_history[:, idx], label=name, linewidth=1.0)
    ax.set_xlabel("tick")
    ax.set_ylabel("V")
    ax.set_title(title)
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="grey", linewidth=0.5, alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
