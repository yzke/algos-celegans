"""CTRNN dynamics — the heart of the Phase 0 neural skeleton.

Implements design.md §3.3:

    chem_input[i] = sum_j W_chem[i,j] * sigmoid(V[j])
    gap_input[i]  = sum_j W_gap[i,j] * (V[j] - V[i])
    dV/dt         = (-V + chem_input + gap_input + sensory_input + noise) / tau

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
    """Centered logistic in [0, 1], with `beta` controlling steepness.

    We center on V=0 with output 0.5 — i.e. the same form as the design doc.
    `numpy.exp` on `-beta * V` with |V| <= 1 cannot overflow for any practical
    `beta`, so we don't need a manual clamp.
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

    # Chemical synaptic input: sigmoid-rectified pre-synaptic activity.
    # We subtract the resting offset of 0.5 so that V=0 is a true fixed point
    # under zero input. Without this correction the logistic sigmoid's value
    # at V=0 (=0.5) produces a constant positive bias through W_chem and the
    # network saturates within a few ticks. See DECISIONS.md "Centered chem
    # input" for the justification.
    chem_input = W_chem @ (sigmoid(V, params.beta) - 0.5)

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
