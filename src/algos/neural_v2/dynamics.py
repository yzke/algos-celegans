"""LIF dynamics for Phase 1.0 — graph-native, event-driven.

Per docs/phase1_design.md §2.1, the standard node update is:

    V(t) = V(t-1) × (1 - 1/tau) + net_input
    if V(t) > threshold:
        emit spike
        V(t) = v_reset

with `net_input` formed from:

    net_input = arriving_signals + gap_input + sensory_input + noise

where ``arriving_signals`` is whatever the SignalQueue says lands at
this tick, ``gap_input`` is the synchronous double-sided membrane
coupling (§3.2), and ``sensory_input`` is external drive.

This module is intentionally numpy-only and vectorized: one Euler step
over all ``N`` neurons in O(N) plus the chemical/gap matvecs (handled
by the caller). Per-neuron refractory periods are supported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Numerical safety net: if V somehow blows up, clip rather than NaN out.
# The LIF reset is the dominant nonlinearity; this clip is only a guard
# against pathological inputs.
V_HARD_CEIL: float = 50.0
V_HARD_FLOOR: float = -50.0


@dataclass(frozen=True)
class LIFParams:
    """Per-population (or per-neuron) LIF parameters.

    All fields are NumPy arrays of length N. The dynamics step is fully
    vectorized; heterogeneity in tau/threshold/etc. is free.
    """

    threshold: np.ndarray         # (N,) spike threshold
    tau: np.ndarray               # (N,) membrane time constant in ticks
    v_reset: np.ndarray           # (N,) post-spike reset potential
    refractory_ticks: np.ndarray  # (N,) int — ticks held at v_reset after spike

    def __post_init__(self) -> None:
        shapes = {a.shape for a in (
            self.threshold, self.tau, self.v_reset, self.refractory_ticks
        )}
        if len(shapes) != 1:
            raise ValueError(
                f"LIFParams arrays must share shape; got {sorted(shapes)}"
            )


def lif_step(
    V: np.ndarray,
    refractory: np.ndarray,
    total_input: np.ndarray,
    params: LIFParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance LIF state by one tick.

    Args:
        V:            (N,) current membrane potential.
        refractory:   (N,) int — ticks remaining in refractory period.
        total_input:  (N,) sum of chemical-arrivals + gap + sensory + noise.
        params:       per-neuron LIF parameters.

    Returns:
        (V_new, refractory_new, spike_mask) where:
            V_new is the updated (N,) membrane potential
            refractory_new is the updated (N,) int refractory countdown
            spike_mask is a (N,) bool array — True wherever a spike fired.
    """
    # Euler leak + input. Form from §2.1.
    V_after_leak = V * (1.0 - 1.0 / params.tau) + total_input

    # Refractory neurons are pinned to v_reset (no integration).
    refr_mask = refractory > 0
    V_candidate = np.where(refr_mask, params.v_reset, V_after_leak)

    # Spike detection — only non-refractory neurons can fire.
    spike_mask = (V_candidate >= params.threshold) & ~refr_mask

    # Post-spike reset.
    V_new = np.where(spike_mask, params.v_reset, V_candidate)

    # Refractory countdown: spikers get full refractory_ticks; others
    # decrement (floored at 0).
    refractory_new = np.where(
        spike_mask,
        params.refractory_ticks,
        np.maximum(refractory - 1, 0),
    ).astype(refractory.dtype)

    # Numerical guard rail.
    np.clip(V_new, V_HARD_FLOOR, V_HARD_CEIL, out=V_new)

    return V_new, refractory_new, spike_mask


__all__ = ["LIFParams", "lif_step", "V_HARD_CEIL", "V_HARD_FLOOR"]
