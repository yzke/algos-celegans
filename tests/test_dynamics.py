"""AC2: dynamics correctness (phase0.md §3.2).

Phase 0 had to relax the literal `max|V| < 0.1` zero-input threshold because
the centered-logistic activation (`sigmoid(V) - 0.5 = 0.5*tanh(βV/2)`) gave
the recurrent network a non-trivial fixed point at max|V| ≈ 0.35.

Phase 0.5 (v0.3 of the design doc) switched the chemical-synapse activation
to `tanh(β*V)` with β = 1.0. At that gain V=0 is the unique stable fixed
point of the un-driven dynamics on our per-row-L1-normalized Cook 2019
connectome, so the original strict `max|V| < 0.1` assertion holds again.
This file therefore restores the literal phase0.md §3.2 thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from algos.neural import CTRNNParams, NeuralState, neural_step


def test_sigmoid_basic():
    from algos.neural.dynamics import sigmoid
    V = np.array([-1.0, 0.0, 1.0])
    out = sigmoid(V, beta=5.0)
    assert np.isclose(out[1], 0.5)
    assert 0.0 <= out[0] < 0.5
    assert 0.5 < out[2] <= 1.0


def _run(state, connectome, sens, n, params=None, seed=42):
    params = params or CTRNNParams(noise_level=0.0)
    rng = np.random.default_rng(seed)
    for _ in range(n):
        state = neural_step(state, connectome, sens, params, rng)
    return state


def test_zero_input_decays_to_zero(connectome):
    """Phase0 §3.2 literal: zero input → max|V| < 0.1.

    With the v0.3 tanh(β=1) chemical activation, V=0 is the unique stable
    fixed point of the un-driven dynamics on the per-row-L1-normalized Cook
    2019 connectome. The original strict phase0.md §3.2 bound holds, but the
    relaxation timescale is ~350 ticks (β=1 sits just below the bifurcation),
    so we let the system run for 5000 ticks before checking.
    """
    state = NeuralState.initialize(302, seed=42)
    state.V[:] += 0.5
    state = _run(state, connectome, np.zeros(302), 5000)
    V5000 = state.V.copy()
    state = _run(state, connectome, np.zeros(302), 200)
    diff = np.max(np.abs(state.V - V5000))
    assert diff < 1e-5, f"not converged: max diff {diff}"
    assert np.max(np.abs(state.V)) <= 1.0
    # phase0.md §3.2 literal threshold now holds:
    assert np.max(np.abs(state.V)) < 0.1, (
        f"un-driven network did not decay near zero: max|V|={np.max(np.abs(state.V)):.4f}"
    )


def test_constant_input_reaches_steady_state(connectome):
    """Phase0 §3.2: convergence under constant sensory input.

    With tanh(β=1) activation the relaxation timescale is ~350 ticks, so we
    run 3000 ticks before checking for convergence — well past 5τ.
    """
    state = NeuralState.initialize(302, seed=42)
    sens = np.zeros(302)
    sens[connectome.idx("ASEL")] = 0.5
    state = _run(state, connectome, sens, 3000)
    V3000 = state.V.copy()
    state = _run(state, connectome, sens, 100)
    diff = np.max(np.abs(state.V - V3000))
    assert diff < 1e-4, f"not converged: max diff {diff}"
    # The steady-state ASEL activation should be non-zero with constant drive.
    assert np.max(np.abs(state.V)) > 0.05, (
        f"constant input produced ~zero response: max|V|={np.max(np.abs(state.V)):.4f}"
    )


def test_pulse_then_decay(connectome):
    """Pulse response: state moves during stimulus, returns to V=0 after.

    With the v0.3 tanh(β=1) dynamics the un-driven attractor is V=0 itself, so
    the "return to baseline" check is "decay back to zero". We pre-equilibrate
    the network (5000 ticks of zero input → V ≈ 0), apply an 800-tick ASEL
    pulse, then release and verify the state decays back near zero.
    """
    rng = np.random.default_rng(42)
    params = CTRNNParams(noise_level=0.0)

    # Pre-equilibrate to the V=0 attractor.
    state = NeuralState.initialize(302, seed=42)
    for _ in range(5000):
        state = neural_step(state, connectome, np.zeros(302), params, rng)
    V_baseline = state.V.copy()
    assert np.max(np.abs(V_baseline)) < 1e-3, "pre-equilibration failed"

    sens = np.zeros(302)
    sens[connectome.idx("ASEL")] = 0.8
    for _ in range(800):
        state = neural_step(state, connectome, sens, params, rng)
    V_pulsed = state.V.copy()

    # Release pulse and let the system decay.
    for _ in range(5000):
        state = neural_step(state, connectome, np.zeros(302), params, rng)
    V_after = state.V.copy()

    move = np.linalg.norm(V_pulsed - V_baseline)
    recovery = np.max(np.abs(V_after))
    assert move > 1e-2, f"pulse barely moved the state: ||ΔV||={move:.6f}"
    assert recovery < 1e-3, (
        f"state did not decay back near zero after pulse: max|V_after|={recovery:.6f}"
    )


def test_bounded_within_clip(connectome):
    """Even with adversarial unit-norm sensory input, V stays in [-1, 1]."""
    state = NeuralState.initialize(302, seed=42)
    rng = np.random.default_rng(42)
    params = CTRNNParams()
    for t in range(500):
        sens = np.ones(302) * 0.9 if t % 50 < 25 else -np.ones(302) * 0.9
        state = neural_step(state, connectome, sens, params, rng)
        assert np.max(np.abs(state.V)) <= 1.0 + 1e-12
