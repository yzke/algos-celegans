"""Phase 0.8.1 — heterogeneous-step-function architecture tests.

Two acceptance properties:

  1. **Numerical equivalence.** When every neuron uses `ctrnn_default`,
     the heterogeneous network is bit-close (< 1e-6) to the Phase 0.7
     `neural_step` path under the same seed sequence.
  2. **Dispatch correctness.** Neurons can be assigned different step
     functions; the per-group dispatch routes each function to its
     neurons.
"""

from __future__ import annotations

import numpy as np
import pytest

from algos.neural import CTRNNParams, NeuralState, neural_step
from algos.neural.heterogeneous import (
    STEP_LIBRARY,
    HeterogeneousNetwork,
)


def test_homogeneous_equivalence(connectome):
    """All neurons → `ctrnn_default` must match Phase 0.7 `neural_step` < 1e-6."""
    # Phase 0.7 path.
    state_hom = NeuralState.initialize(302, seed=42)
    params = CTRNNParams()
    rng_hom = np.random.default_rng(42)
    for _ in range(100):
        sens = rng_hom.standard_normal(302) * 0.05
        state_hom = neural_step(state_hom, connectome, sens, params, rng_hom)

    # Heterogeneous path — same seed, same sensory schedule.
    network = HeterogeneousNetwork.from_homogeneous_ctrnn(connectome)
    state_het = network.initial_state(seed=42)
    rng_het = np.random.default_rng(42)
    for _ in range(100):
        sens = rng_het.standard_normal(302) * 0.05
        state_het = network.step(state_het, sens, rng_het)

    diff = np.max(np.abs(state_hom.V - state_het.V))
    assert diff < 1e-6, (
        f"heterogeneous drifted from Phase 0.7: max |ΔV| = {diff:.3e}"
    )


def test_homogeneous_equivalence_no_noise(connectome):
    """With noise off, ctrnn_default vs Phase 0.7 should match to machine precision."""
    state_hom = NeuralState.initialize(302, seed=42)
    params = CTRNNParams(noise_level=0.0)
    rng_hom = np.random.default_rng(42)
    for _ in range(50):
        sens = np.zeros(302)
        sens[connectome.idx("ASEL")] = 0.5
        state_hom = neural_step(state_hom, connectome, sens, params, rng_hom)

    network = HeterogeneousNetwork.from_homogeneous_ctrnn(
        connectome, noise_level=0.0
    )
    state_het = network.initial_state(seed=42)
    rng_het = np.random.default_rng(42)
    for _ in range(50):
        sens = np.zeros(302)
        sens[connectome.idx("ASEL")] = 0.5
        state_het = network.step(state_het, sens, rng_het)

    diff = np.max(np.abs(state_hom.V - state_het.V))
    assert diff < 1e-12, (
        f"deterministic divergence — implementations differ structurally: "
        f"max |ΔV| = {diff:.3e}"
    )


def test_can_assign_per_neuron_functions(connectome):
    """Different neurons can use different step functions; dispatch correct."""

    def constant_zero(V_current, total_input, V_history, params):
        return np.zeros_like(V_current)

    library = dict(STEP_LIBRARY)
    library["constant_zero"] = constant_zero

    n = connectome.n_neurons
    forced_idx = list(range(10))
    assignment = ["ctrnn_default"] * n
    for i in forced_idx:
        assignment[i] = "constant_zero"

    network = HeterogeneousNetwork(
        connectome=connectome,
        function_assignment=assignment,
        function_params={"tau": np.full(n, 10.0)},
        function_library=library,
    )
    state = network.initial_state(seed=42)
    rng = np.random.default_rng(42)
    sens_idx = connectome.get_neuron_indices_by_category("sensory")
    sens = np.zeros(n)
    for _ in range(50):
        sens[:] = 0.0
        sens[sens_idx] = 0.5
        state = network.step(state, sens, rng)

    forced = state.V[forced_idx]
    assert np.allclose(forced, 0.0), (
        f"constant_zero neurons should be 0; max |V|={np.max(np.abs(forced))}"
    )
    other = state.V[10:]
    assert np.max(np.abs(other)) > 1e-3, (
        "ctrnn_default neurons unexpectedly silent — dispatch broken?"
    )


def test_history_buffer_advances_correctly(connectome):
    """V_history[-1] is current; V_history[-2] is one tick ago."""
    network = HeterogeneousNetwork.from_homogeneous_ctrnn(connectome)
    state = network.initial_state(seed=42)
    rng = np.random.default_rng(42)
    sens = np.zeros(connectome.n_neurons)

    prev_v_history_last = state.V_history[-1].copy()
    state = network.step(state, sens, rng)
    # V_history[-2] should now equal what was V_history[-1] before the step.
    assert np.array_equal(state.V_history[-2], prev_v_history_last)
    # V_history[-1] is the new V.
    assert np.array_equal(state.V_history[-1], state.V)


def test_unknown_function_rejected(connectome):
    """Assigning a function not in the library raises KeyError."""
    n = connectome.n_neurons
    with pytest.raises(KeyError):
        HeterogeneousNetwork(
            connectome=connectome,
            function_assignment=["nonexistent_function"] * n,
            function_params={"tau": np.full(n, 10.0)},
        )


def test_category_defaults_factory_assigns_correctly(connectome):
    """from_category_defaults assigns the right function per category."""
    from algos.neural import from_category_defaults, DEFAULT_CATEGORY_ASSIGNMENT

    net = from_category_defaults(connectome)
    for i, name in enumerate(net.function_assignment):
        cat = connectome.category[i]
        expected_func, expected_params = DEFAULT_CATEGORY_ASSIGNMENT.get(
            cat, ("ctrnn_default", {"tau": 10.0, "beta": 1.0})
        )
        assert name == expected_func, (
            f"neuron {i} ({connectome.neuron_names[i]}, cat={cat}): "
            f"expected {expected_func}, got {name}"
        )
        assert net.function_params["tau"][i] == expected_params["tau"], (
            f"neuron {i}: tau mismatch"
        )


def test_category_default_network_runs_stably(connectome):
    """Category-default network runs 2000 ticks without NaN/Inf."""
    from algos.neural import from_category_defaults

    net = from_category_defaults(connectome)
    state = net.initial_state(seed=42)
    rng = np.random.default_rng(42)
    sens = np.zeros(connectome.n_neurons)
    for t in range(2000):
        sens[:] = 0.0
        sens[connectome.get_neuron_indices_by_category("sensory")] = (
            rng.standard_normal(83) * 0.05
        )
        state = net.step(state, sens, rng)
    assert np.all(np.isfinite(state.V))
    assert np.max(np.abs(state.V)) <= 1.0 + 1e-12
