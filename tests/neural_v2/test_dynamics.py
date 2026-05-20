"""Tests for the LIF dynamics + signal-propagation runtime."""

from __future__ import annotations

import numpy as np
import pytest

from algos.graph import load_connectome_into_graph
from algos.neural_v2 import (
    DelayBucket,
    GraphNeuralState,
    GraphSimulator,
    LIFParams,
    SignalQueue,
    SimulatorConfig,
    build_delay_buckets,
    build_gap_matrix,
    lif_step,
)


# ---------------------------------------------------------------------------
# LIF unit tests
# ---------------------------------------------------------------------------


def _basic_params(n: int, *, threshold=1.0, tau=10.0, v_reset=0.0, refr=2):
    return LIFParams(
        threshold=np.full(n, threshold),
        tau=np.full(n, tau),
        v_reset=np.full(n, v_reset),
        refractory_ticks=np.full(n, refr, dtype=np.int64),
    )


def test_lif_no_input_decays_to_zero():
    n = 3
    V = np.array([0.5, -0.3, 0.0])
    refr = np.zeros(n, dtype=np.int64)
    params = _basic_params(n, tau=10.0)
    for _ in range(200):
        V, refr, _ = lif_step(V, refr, np.zeros(n), params)
    assert np.max(np.abs(V)) < 1e-3


def test_lif_constant_input_reaches_steady_state():
    """With V = V*(1-1/tau) + input, steady-state V = input * tau."""
    n = 1
    input_val = 0.05  # tau=10 → steady = 0.5, below threshold=1.0
    V = np.zeros(1)
    refr = np.zeros(1, dtype=np.int64)
    params = _basic_params(n, threshold=10.0, tau=10.0)  # threshold high enough not to spike
    inp = np.full(1, input_val)
    for _ in range(500):
        V, refr, _ = lif_step(V, refr, inp, params)
    assert V[0] == pytest.approx(input_val * params.tau[0], rel=1e-3)


def test_lif_spike_then_reset_and_refractory():
    n = 1
    V = np.zeros(1)
    refr = np.zeros(1, dtype=np.int64)
    params = _basic_params(n, threshold=1.0, tau=10.0, v_reset=0.0, refr=3)
    # Big input → spike on first step.
    V, refr, sp = lif_step(V, refr, np.full(1, 5.0), params)
    assert sp[0]
    assert V[0] == pytest.approx(0.0)
    assert refr[0] == 3
    # Next steps: refractory pins V even with big input.
    for expected_refr in (2, 1, 0):
        V, refr, sp = lif_step(V, refr, np.full(1, 5.0), params)
        assert not sp[0] or expected_refr == 0  # at expected_refr==0 can spike again
        if expected_refr > 0:
            assert V[0] == pytest.approx(0.0)
            assert refr[0] == expected_refr


def test_lif_clamps_runaway_input():
    """Pathological huge input should clip, not NaN."""
    n = 2
    V = np.zeros(2)
    refr = np.zeros(2, dtype=np.int64)
    params = _basic_params(n, threshold=1.0, tau=10.0, refr=0)
    V, refr, sp = lif_step(V, refr, np.array([1e6, -1e6]), params)
    assert np.all(np.isfinite(V))


# ---------------------------------------------------------------------------
# SignalQueue
# ---------------------------------------------------------------------------


def test_signal_queue_delivers_after_delay():
    n = 4
    # One edge: 0 → 2 with delay 3, signed weight +0.5.
    W_d3 = np.zeros((n, n))
    W_d3[2, 0] = 0.5
    q = SignalQueue.from_delay_buckets(
        [DelayBucket(delay=3, W=W_d3)], n_neurons=n,
    )
    # Spike at tick 0 from neuron 0.
    spike = np.array([1, 0, 0, 0])
    # tick=0: read arrivals (nothing), schedule, advance.
    arr = q.arrivals(); assert arr.sum() == 0
    q.schedule(spike); q.advance()
    # ticks 1, 2: nothing arrives.
    for _ in range(2):
        arr = q.arrivals(); assert arr.sum() == 0
        q.schedule(np.zeros(n)); q.advance()
    # tick 3: the delayed signal lands.
    arr = q.arrivals()
    assert arr[2] == pytest.approx(0.5)
    assert arr.sum() == pytest.approx(0.5)


def test_signal_queue_handles_multiple_delays():
    n = 3
    W_d1 = np.zeros((n, n)); W_d1[1, 0] = 0.2
    W_d2 = np.zeros((n, n)); W_d2[2, 0] = 0.3
    q = SignalQueue.from_delay_buckets(
        [DelayBucket(delay=1, W=W_d1), DelayBucket(delay=2, W=W_d2)],
        n_neurons=n,
    )
    spike = np.array([1, 0, 0])
    q.arrivals(); q.schedule(spike); q.advance()  # tick 0
    arr1 = q.arrivals(); q.schedule(np.zeros(n)); q.advance()  # tick 1
    arr2 = q.arrivals()
    assert arr1[1] == pytest.approx(0.2)
    assert arr2[2] == pytest.approx(0.3)


def test_signal_queue_reset_clears_buffer():
    n = 2
    W = np.zeros((n, n)); W[1, 0] = 1.0
    q = SignalQueue.from_delay_buckets([DelayBucket(delay=1, W=W)], n_neurons=n)
    q.schedule(np.array([1, 0]))
    q.reset()
    assert q.tick == 0
    arr = q.arrivals()
    assert arr.sum() == 0


# ---------------------------------------------------------------------------
# Integration: full simulator
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simulator():
    g = load_connectome_into_graph()
    return GraphSimulator(g, config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2))


def test_simulator_short_run_finite(simulator: GraphSimulator):
    """100 ticks → all V finite, no NaN, no extreme values."""
    out = simulator.run(100, seed=42)
    V = out["V"]
    assert np.all(np.isfinite(V))
    # Per docs, threshold=1.0 and v_reset=0; spikes reset to 0. After reset
    # the maximum |V| sustainable is dominated by leak+input balance.
    # Empirically with tau≈8-12 and noise≈0.005 we stay well below 5.
    assert np.max(np.abs(V)) < 10.0


def test_simulator_10k_ticks_stable(simulator: GraphSimulator):
    """The Phase 1.0.2 acceptance criterion: 10^4 ticks, no NaN, V bounded."""
    out = simulator.run(10_000, seed=7, record_V=False, record_rate=False)
    final = out["final_state"]
    assert np.all(np.isfinite(final.V))
    assert np.max(np.abs(final.V)) < 10.0
    # With sensory_noise=0.2 the sensory pool fires regularly and the
    # event-driven queue should be ticking over — verify spikes happened.
    assert final.spike_count.sum() > 0


def test_simulator_event_driven_propagates(simulator: GraphSimulator):
    """A sensory neuron with elevated drive should make downstream
    neurons (via chemical edges) spike too — i.e. the SignalQueue
    actually delivers signals beyond the source pool."""
    sim = GraphSimulator(
        simulator.graph,
        config=SimulatorConfig(noise_level=0.0, sensory_noise=0.0),
    )
    g = sim.graph
    # Pick a sensory neuron with outgoing chemical edges.
    src = None
    for name in g.neuron_names():
        if g.node(name).category != "sensory":
            continue
        n_out = sum(1 for _ in g.out_edges(name, "chemical"))
        if n_out >= 3:
            src = name
            break
    assert src is not None
    src_idx = g.index_of(src)
    direct_targets = {
        g.index_of(e.target) for e in g.out_edges(src, "chemical")
    }
    # Drive the chosen neuron hard until something downstream spikes too.
    state = sim.initial_state(seed=0, spread=0.0)
    rng = np.random.default_rng(0)
    sens = np.zeros(sim.n)
    for _ in range(300):
        sens[:] = 0.0
        sens[src_idx] = 0.5  # well above the steady-state threshold
        state = sim.step(state, sens, rng)
    # Source should have spiked.
    assert state.spike_count[src_idx] > 0
    # And at least one direct chemical target should also have spiked or
    # accumulated significant V from the propagated signals.
    other_spikes = sum(
        int(state.spike_count[i]) for i in direct_targets if i != src_idx
    )
    other_V = np.array([state.V[i] for i in direct_targets if i != src_idx])
    assert other_spikes > 0 or np.max(np.abs(other_V)) > 0.05, (
        "SignalQueue is not delivering downstream"
    )


def test_simulator_zero_input_decays(simulator: GraphSimulator):
    """With sensory_noise=0 and an unmodulated network, activity should
    eventually quiet (the bare graph has no autonomous oscillator strong
    enough to sustain itself at the default LIF thresholds)."""
    sim = GraphSimulator(
        simulator.graph,
        config=SimulatorConfig(noise_level=0.0, sensory_noise=0.0),
    )
    state = sim.initial_state(seed=0, spread=0.05)
    rng = np.random.default_rng(0)
    sens = np.zeros(sim.n)
    for _ in range(2000):
        state = sim.step(state, sens, rng)
    # After 2000 ticks of no input, all V should be near zero (decayed).
    assert np.max(np.abs(state.V)) < 0.05


def test_simulator_gap_input_couples_neighbors(simulator: GraphSimulator):
    """Setting one neuron to high V should drive its gap-coupled neighbors."""
    sim = GraphSimulator(
        simulator.graph,
        config=SimulatorConfig(noise_level=0.0, sensory_noise=0.0),
    )
    # Find a neuron with at least one gap neighbor.
    g = simulator.graph
    test_name = None
    for n in g.neuron_names():
        n_gap = sum(1 for _ in g.out_edges(n, "electrical"))
        if n_gap > 2:
            test_name = n
            break
    assert test_name is not None, "no neuron has gap junctions?"

    state = sim.initial_state(seed=0, spread=0.0)
    state.V[g.index_of(test_name)] = 0.5
    rng = np.random.default_rng(0)
    sens = np.zeros(sim.n)
    # One step: gap coupling moves toward the high V at neighbors.
    initial_V = state.V.copy()
    state = sim.step(state, sens, rng)
    # At least one neighbor should have moved positive.
    neighbors = [g.index_of(e.target) for e in g.out_edges(test_name, "electrical")]
    delta = state.V[neighbors] - initial_V[neighbors]
    assert (delta > 0).any(), f"no positive gap coupling to neighbors of {test_name}"
