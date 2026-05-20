"""CTRNN dynamics — the heart of the Phase 0 neural skeleton.

Implements design.md §3.3 (v0.3):

    chem_input[i] = sum_j W_chem[i,j] * tanh(beta * V[j])
    gap_input[i]  = sum_j W_gap[i,j] * (V[j] - V[i])
    dV/dt         = (-V + chem_input + gap_input + sensory_input + noise) / tau

The v0.3 design doc switched the chemical-synapse activation from a centered
logistic (`sigmoid(V) - 0.5` — Phase 0's patch) to `tanh(beta * V)`. They are
the same family of centered nonlinearities (`tanh(x) = 2*sigmoid(2x) - 1`); the
tanh form is the standard CTRNN convention and removes the `-0.5` magic
constant from the equation.

Phase 0 deliberately omits modulator drive (`S @ c`) and plasticity. They are
introduced in phases 3 and 4.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from algos.config import CTRNNDefaults
from algos.connectome import ConnectomeData
from algos.neural.state import NeuralState


@dataclass(frozen=True)
class CTRNNParams:
    """CTRNN hyperparameters — design.md §3.3 and §3.6."""

    tau: float = CTRNNDefaults.tau
    beta: float = CTRNNDefaults.beta
    noise_level: float = CTRNNDefaults.noise_level


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def sigmoid(V: np.ndarray, beta: float) -> np.ndarray:
    """Standard logistic in [0, 1] with `beta` controlling steepness.

    Retained for backwards-compatible imports and `test_sigmoid_basic`. The
    Phase-0.5+ dynamics use `tanh(beta * V)` instead (see `neural_step`).
    """
    return 1.0 / (1.0 + np.exp(-beta * V))


# ---------------------------------------------------------------------------
# Single-tick update
# ---------------------------------------------------------------------------


def neural_step(
    state: NeuralState,
    connectome: ConnectomeData,
    sensory_input: np.ndarray,
    params: CTRNNParams,
    rng: np.random.Generator,
) -> NeuralState:
    """Advance the neural state by one tick.

    Args:
        state: Current `NeuralState`. Not mutated; a new object is returned.
        connectome: Loaded connectome supplying `W_chem` and `W_gap`.
        sensory_input: (N,) external drive. For Phase 0 (no body) the caller
            chooses which sensory neurons to stimulate; typical values are 0
            for most entries.
        params: CTRNN hyperparameters.
        rng: Random generator used for the Gaussian noise term.

    Returns:
        New `NeuralState` at tick `state.tick + 1`.
    """
    V = state.V
    W_chem = connectome.W_chem
    W_gap = connectome.W_gap

    # Chemical synaptic input: tanh-rectified pre-synaptic activity.
    # tanh is naturally centered (tanh(0)=0), so V=0 is a true fixed point of
    # the un-driven dynamics with no magic constants needed. See design.md
    # §3.3 (v0.3) and DECISIONS.md "[Phase 0.5] design doc bumped to v0.3" for
    # the discussion of why we moved away from `sigmoid(V) - 0.5`.
    chem_input = W_chem @ np.tanh(params.beta * V)

    # Electrical (gap-junction) input — Laplacian form, equivalent to summing
    # W_gap[i,j] * (V[j] - V[i]) over j.
    gap_input = W_gap @ V - V * W_gap.sum(axis=1)

    # Gaussian noise.
    if params.noise_level > 0.0:
        noise = rng.standard_normal(V.shape[0]) * params.noise_level
    else:
        noise = 0.0

    # CTRNN forward Euler step.
    dV = (-V + chem_input + gap_input + sensory_input + noise) / params.tau
    V_new = np.clip(V + dV, -1.0, 1.0)

    return NeuralState(V=V_new, tick=state.tick + 1)


__all__ = ["CTRNNParams", "sigmoid", "neural_step"]
