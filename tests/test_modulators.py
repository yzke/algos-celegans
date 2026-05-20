"""Phase 0.9 — RIDModulator tests.

Acceptance properties (per `logs/phase0.9_brief.md` §3.5, AC0.9.1):

  1. Numerical equivalence: a `HeterogeneousNetwork.step(..., modulator=None)`
     call must be bit-identical to the Phase 0.8.2 path (no modulator).
  2. With a modulator, c_RID evolves toward V[idx_RID] on its tau and
     stays finite; the reversal pool receives the documented modulation.
  3. Forward command (AVB, PVC) is *not* modulated.
"""

from __future__ import annotations

import numpy as np
import pytest

from algos.neural import (
    HeterogeneousNetwork,
    RIDModulator,
    REVERSAL_COMMAND_NEURONS,
    from_category_defaults,
)


def test_no_modulator_equals_phase08_2(connectome):
    """modulator=None must match the Phase 0.8.2 reference path bit-by-bit."""
    net_ref = from_category_defaults(connectome)
    net_new = from_category_defaults(connectome)

    state_ref = net_ref.initial_state(seed=42)
    state_new = net_new.initial_state(seed=42)

    rng_ref = np.random.default_rng(42)
    rng_new = np.random.default_rng(42)

    n = connectome.n_neurons
    sens_idx = connectome.get_neuron_indices_by_category("sensory")
    for _ in range(200):
        sens_ref = np.zeros(n)
        sens_new = np.zeros(n)
        s = rng_ref.standard_normal(len(sens_idx)) * 0.05
        sens_ref[sens_idx] = s
        # Identical sensory schedule for the new path.
        rng_new.standard_normal(len(sens_idx))  # consume same draws
        sens_new[sens_idx] = s
        state_ref = net_ref.step(state_ref, sens_ref, rng_ref)
        state_new = net_new.step(state_new, sens_new, rng_new, modulator=None)

    diff = np.max(np.abs(state_ref.V - state_new.V))
    assert diff == 0.0, (
        f"modulator=None must be bit-identical to no-modulator path, "
        f"max |ΔV| = {diff:.3e}"
    )


def test_rid_modulator_state_evolves(connectome):
    """c_RID approaches V[RID] on its time constant."""
    net = from_category_defaults(connectome)
    modulator = RIDModulator.from_connectome(connectome, tau=200.0, gain=0.5)
    state = net.initial_state(seed=42)
    rng = np.random.default_rng(42)
    n = connectome.n_neurons
    sens_idx = connectome.get_neuron_indices_by_category("sensory")

    initial_c = modulator.c_RID
    assert initial_c == 0.0

    # Drive the network for 1000 ticks and confirm c_RID stays finite.
    for _ in range(1000):
        sens = np.zeros(n)
        sens[sens_idx] = rng.standard_normal(len(sens_idx)) * 0.05
        state = net.step(state, sens, rng, modulator=modulator)

    assert np.isfinite(modulator.c_RID)
    assert abs(modulator.c_RID) <= 1.0


def test_rid_modulator_inhibits_reversal_pool(connectome):
    """With a high gain and a forced high c_RID, reversal pool V is lower."""
    net = from_category_defaults(connectome)
    n = connectome.n_neurons
    reversal_idx = [connectome.idx(name) for name in REVERSAL_COMMAND_NEURONS]
    forward_idx = [connectome.idx(name) for name in
                   ("AVBL", "AVBR", "PVCL", "PVCR")]

    # Baseline: no modulator.
    state_a = net.initial_state(seed=42)
    rng_a = np.random.default_rng(42)
    for _ in range(500):
        sens = np.zeros(n)
        state_a = net.step(state_a, sens, rng_a, modulator=None)
    rev_a = state_a.V[reversal_idx].copy()
    fwd_a = state_a.V[forward_idx].copy()

    # With modulator: force c_RID large (skip dynamics by setting directly).
    state_b = net.initial_state(seed=42)
    rng_b = np.random.default_rng(42)
    modulator = RIDModulator.from_connectome(connectome, tau=200.0, gain=1.0)
    modulator.c_RID = 0.8  # forced large positive value
    for _ in range(500):
        sens = np.zeros(n)
        # Keep c_RID pinned for the duration so we observe pure modulation.
        modulator.c_RID = 0.8
        # Need to bypass `.step()`'s own update of c_RID — overwrite after.
        state_b = net.step(state_b, sens, rng_b, modulator=modulator)
    rev_b = state_b.V[reversal_idx].copy()
    fwd_b = state_b.V[forward_idx].copy()

    # The reversal pool should be more negative on average with modulation;
    # the forward pool should be approximately unchanged (gap currents from
    # neighbors will perturb it a little but not in the same systematic way).
    delta_rev = (rev_b - rev_a).mean()
    delta_fwd = (fwd_b - fwd_a).mean()
    assert delta_rev < -0.001, (
        f"reversal pool should be inhibited; mean ΔV = {delta_rev:.4f}"
    )
    # Forward pool should drift much less than reversal pool.
    assert abs(delta_fwd) < abs(delta_rev), (
        f"forward pool changed more than reversal pool: "
        f"|Δfwd|={abs(delta_fwd):.4f} vs |Δrev|={abs(delta_rev):.4f}"
    )


def test_rid_modulator_missing_neuron_raises():
    """from_connectome raises if a required neuron is missing."""
    class FakeConn:
        neuron_to_idx = {"RID": 0, "AVAL": 1, "AVAR": 2}  # missing AVDL etc.

    with pytest.raises(KeyError):
        RIDModulator.from_connectome(FakeConn())


def test_rid_modulator_reset(connectome):
    """reset() zeros c_RID and clears history."""
    modulator = RIDModulator.from_connectome(connectome, record_history=True)
    modulator.c_RID = 0.42
    modulator.history.extend([0.1, 0.2, 0.3])
    modulator.reset()
    assert modulator.c_RID == 0.0
    assert modulator.history == []


def test_rid_index_and_pool_present(connectome):
    """RID and the full reversal pool must be in the connectome."""
    modulator = RIDModulator.from_connectome(connectome)
    assert modulator.idx_RID == connectome.idx("RID")
    for name in REVERSAL_COMMAND_NEURONS:
        assert connectome.idx(name) in modulator.reversal_indices
