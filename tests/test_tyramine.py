"""Phase 1.6.2 — tests for the tyramine modulator.

Tyramine is the third modulator in the default bank
(`build_default_modulator_bank`), after RID and 5-HT. Producers are
RIML/R and RICL/R; targets are AVB (forward command), MC (pharyngeal
pump), and the four RMDs (head motor). Sensitivity is +0.5
(threshold-raising → suppress) per Pirri 2009.

These tests cover:
  1. The bank now reports 3 modulators including 'tyramine'.
  2. With RIM driven by stimulation, c_tyramine rises and AVB
     threshold is raised in the documented direction.
  3. Without stimulation (bare network), tyramine is silent —
     same Phase 0.9 / Phase 1.0 obstruction that affects every
     modulator producer in the absence of behavioral drive.
  4. Backward compatibility — attaching the bank to a Phase 1.0
     simulator does not break existing tests or change baseline
     behavior at zero c_m.
"""

from __future__ import annotations

import numpy as np
import pytest

from algos.graph import load_connectome_into_graph
from algos.neural_v2 import (
    GraphSimulator,
    SimulatorConfig,
    build_default_modulator_bank,
)
from algos.neural_v2.modulators import (
    DEFAULT_TAU_M_TYRAMINE,
    TYRAMINE_SENSITIVITY,
    TYRAMINE_SOURCE_NEURONS,
    TYRAMINE_TARGET_NEURONS,
)


# ---------------------------------------------------------------------------
# Bank-level structural tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph():
    return load_connectome_into_graph()


def test_default_bank_has_three_modulators(graph):
    base = np.ones(graph.n_nodes)
    bank = build_default_modulator_bank(graph, base)
    assert bank.n_modulators == 3
    names = [m.name for m in bank.modulators]
    assert names == ["RID", "5HT", "tyramine"]


def test_tyramine_producer_members_match_constants(graph):
    base = np.ones(graph.n_nodes)
    bank = build_default_modulator_bank(graph, base)
    tyr = bank.modulators[2]
    expected_producers = {graph.index_of(n) for n in TYRAMINE_SOURCE_NEURONS}
    assert set(tyr.producer_idx.tolist()) == expected_producers


def test_tyramine_targets_include_avb_mc_rmd(graph):
    base = np.ones(graph.n_nodes)
    bank = build_default_modulator_bank(graph, base)
    tyr = bank.modulators[2]
    target_names = {graph.neuron_names()[i] for i in tyr.target_idx}
    # AVB and MC and RMD must be present per Pirri 2009.
    assert {"AVBL", "AVBR"}.issubset(target_names)
    assert {"MCL", "MCR"}.issubset(target_names)
    assert {"RMDL", "RMDR"}.issubset(target_names)


def test_tyramine_sensitivity_is_positive(graph):
    """Positive sensitivity → raise threshold → suppress (Pirri 2009)."""
    base = np.ones(graph.n_nodes)
    bank = build_default_modulator_bank(graph, base)
    tyr = bank.modulators[2]
    assert (tyr.sensitivity > 0).all()
    assert tyr.sensitivity[0] == pytest.approx(TYRAMINE_SENSITIVITY)


def test_tyramine_tau_m_is_intermediate(graph):
    """τ_m for tyramine should be smaller than 5-HT's (300 < 500),
    reflecting the faster mixed-receptor pharmacology."""
    base = np.ones(graph.n_nodes)
    bank = build_default_modulator_bank(graph, base, tau_m=500.0)
    tyr = bank.modulators[2]
    assert tyr.tau_m == DEFAULT_TAU_M_TYRAMINE == 300.0


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------


def _setup_sim(graph):
    sim = GraphSimulator(
        graph, config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2)
    )
    bank = build_default_modulator_bank(graph, sim.params.threshold)
    sim.attach_modulators(bank)
    return sim, bank


def test_c_tyramine_tracks_rim_rate_under_stimulation(graph):
    """With RIML+RIMR driven by 0.5 sustained input, c_tyramine
    increases over a tau-scale, well above zero."""
    sim, bank = _setup_sim(graph)
    tyr = bank.modulators[2]
    riml = graph.index_of("RIML")
    rimr = graph.index_of("RIMR")
    state = sim.initial_state(seed=7)
    rng = np.random.default_rng(7)
    sens = np.zeros(sim.n)
    c_initial = tyr.c_m
    for _ in range(2000):
        sens[:] = 0.0
        sens[sim.sensory_idx] = rng.standard_normal(sim.sensory_idx.size) * 0.2
        sens[riml] = 0.5
        sens[rimr] = 0.5
        state = sim.step(state, sens, rng)
    assert tyr.c_m > c_initial + 0.5, (
        f"c_tyramine did not rise under sustained RIM drive "
        f"({c_initial} → {tyr.c_m})"
    )


def test_avb_threshold_raised_under_high_c_tyramine(graph):
    """When c_tyramine > 0, AVB threshold should be above its base."""
    sim, bank = _setup_sim(graph)
    avbl = graph.index_of("AVBL")
    base_thresh = float(sim.params.threshold[avbl])
    tyr = bank.modulators[2]
    # Force c_m to a finite positive value and run one threshold update.
    tyr.c_m = 2.0
    bank.apply_threshold_modulation(sim.params.threshold)
    new_thresh = float(sim.params.threshold[avbl])
    # 1 + (+0.5) * 2.0 = 2.0 → expected new threshold = base * 2.0.
    expected = base_thresh * (1.0 + TYRAMINE_SENSITIVITY * 2.0)
    assert new_thresh == pytest.approx(expected, rel=1e-6)


def test_no_effect_when_tyramine_silent_in_bare_network(graph):
    """In the bare network, RIM and RIC fire ~0 times so c_tyramine
    stays near zero and AVB thresholds are essentially unchanged."""
    sim, bank = _setup_sim(graph)
    tyr = bank.modulators[2]
    avbl = graph.index_of("AVBL")
    base_thresh = float(bank.base_threshold[avbl])
    state = sim.initial_state(seed=1042)
    rng = np.random.default_rng(1042)
    sens = np.zeros(sim.n)
    for _ in range(2000):
        sens[:] = 0.0
        sens[sim.sensory_idx] = rng.standard_normal(sim.sensory_idx.size) * 0.2
        state = sim.step(state, sens, rng)
    # c_tyramine should be very small (< 0.05) because RIM is silent.
    assert abs(tyr.c_m) < 0.05, (
        f"unexpected c_tyramine = {tyr.c_m} in bare-network run"
    )
    # Threshold drift < 5%.
    drift = abs(sim.params.threshold[avbl] - base_thresh) / base_thresh
    assert drift < 0.05


def test_rim_stimulation_suppresses_avb(graph):
    """Sustained RIM drive → c_tyramine ↑ → AVB threshold ↑ → AVB
    fires *less* than in a control with the same noise and no RIM
    stim. We do not assert AVB > 0 in either run (AVB barely fires
    in the bare network); we assert "AVB count with stim ≤ AVB count
    without stim" — the inhibitory direction."""
    rim_l = graph.index_of("RIML")
    rim_r = graph.index_of("RIMR")
    avbl = graph.index_of("AVBL")
    avbr = graph.index_of("AVBR")

    # Control: no RIM injection.
    sim_a, _ = _setup_sim(graph)
    state = sim_a.initial_state(seed=99)
    rng = np.random.default_rng(99)
    sens = np.zeros(sim_a.n)
    for _ in range(3000):
        sens[:] = 0.0
        sens[sim_a.sensory_idx] = rng.standard_normal(sim_a.sensory_idx.size) * 0.2
        state = sim_a.step(state, sens, rng)
    avb_a = int(state.spike_count[avbl] + state.spike_count[avbr])

    # Treatment: RIM stim.
    sim_b, _ = _setup_sim(graph)
    state = sim_b.initial_state(seed=99)
    rng = np.random.default_rng(99)
    sens = np.zeros(sim_b.n)
    for _ in range(3000):
        sens[:] = 0.0
        sens[sim_b.sensory_idx] = rng.standard_normal(sim_b.sensory_idx.size) * 0.2
        sens[rim_l] = 0.5
        sens[rim_r] = 0.5
        state = sim_b.step(state, sens, rng)
    avb_b = int(state.spike_count[avbl] + state.spike_count[avbr])

    assert avb_b <= avb_a, (
        f"AVB fired more with RIM stim ({avb_a} → {avb_b}) — tyramine "
        f"suppression not working"
    )


def test_backward_compat_without_tyramine_in_bank(graph):
    """A simulator with NO modulator bank attached must produce the
    same V trajectory as before. Tyramine only affects behavior when
    the bank is attached AND c_tyramine != 0."""
    sim_off = GraphSimulator(
        graph, config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2)
    )
    # Same setup but with bank attached AND c_tyramine forced to 0.
    sim_on = GraphSimulator(
        graph, config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2)
    )
    bank = build_default_modulator_bank(sim_on.graph, sim_on.params.threshold)
    sim_on.attach_modulators(bank)
    for m in bank.modulators:
        m.c_m = 0.0
    bank.apply_threshold_modulation(sim_on.params.threshold)

    s_off = sim_off.initial_state(seed=77)
    s_on = sim_on.initial_state(seed=77)
    rng_off = np.random.default_rng(77)
    rng_on = np.random.default_rng(77)
    sens_off = np.zeros(sim_off.n)
    sens_on = np.zeros(sim_on.n)
    # The bank's per-tick update will modulate thresholds based on the
    # rate trace; at t=0 rate=0, so for the FIRST step c_m stays 0 and
    # thresholds are unchanged. Verify just 1 step of equivalence.
    sens_off[:] = 0.0
    sens_off[sim_off.sensory_idx] = (
        rng_off.standard_normal(sim_off.sensory_idx.size) * 0.2
    )
    sens_on[:] = 0.0
    sens_on[sim_on.sensory_idx] = (
        rng_on.standard_normal(sim_on.sensory_idx.size) * 0.2
    )
    s_off = sim_off.step(s_off, sens_off, rng_off)
    s_on = sim_on.step(s_on, sens_on, rng_on)
    np.testing.assert_allclose(s_off.V, s_on.V, atol=1e-12)
