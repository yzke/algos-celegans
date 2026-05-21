"""Tests for the Hebbian plasticity rule + modulator bank."""

from __future__ import annotations

import numpy as np
import pytest

from algos.graph import load_connectome_into_graph
from algos.neural_v2 import (
    GraphSimulator,
    HebbianRule,
    Modulator,
    ModulatorBank,
    SimulatorConfig,
    build_default_modulator_bank,
)


# ---------------------------------------------------------------------------
# HebbianRule
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph():
    return load_connectome_into_graph()


def test_hebbian_default_rule_50_to_100_edges(graph):
    rule = HebbianRule.from_graph(graph)
    assert 50 <= rule.n_edges <= 100


def test_hebbian_marks_underlying_edges_as_plastic(graph):
    g2 = load_connectome_into_graph()
    rule = HebbianRule.from_graph(g2)
    for e in rule.edges:
        assert e.is_plastic
        assert e.plasticity_rule == "hebbian"


def test_hebbian_zero_activity_decays_weights(graph):
    """With no pre/post activity, weights drift downward by λ·W."""
    rule = HebbianRule.from_graph(graph, eta=1e-3, lam=1e-3)
    initial = rule.weights.copy()
    n = graph.n_nodes
    zero_rate = np.zeros(n)
    for _ in range(50):
        rule.step(zero_rate)
    assert (rule.weights < initial - 1e-6).all()


def test_hebbian_high_correlated_activity_grows_weights(graph):
    """When pre and post both have rate=1, η·1·1 > λ·W for a while."""
    rule = HebbianRule.from_graph(graph, eta=5e-2, lam=1e-5, max_edges=10)
    initial = rule.weights.copy()
    n = graph.n_nodes
    rate = np.zeros(n)
    rate[rule.pre_idx] = 1.0
    rate[rule.post_idx] = 1.0
    for _ in range(20):
        rule.step(rate)
    assert (rule.weights > initial).all()


def test_hebbian_weights_stay_bounded(graph):
    """Even with explosive input, weights stay in [w_min, w_max]."""
    rule = HebbianRule.from_graph(graph, eta=10.0, lam=0.0, max_edges=5)
    n = graph.n_nodes
    rate = np.ones(n)
    for _ in range(100):
        rule.step(rate)
    assert (rule.weights <= rule.w_max + 1e-9).all()
    assert (rule.weights >= rule.w_min - 1e-9).all()


def test_hebbian_write_back_updates_graph_edges(graph):
    g2 = load_connectome_into_graph()
    rule = HebbianRule.from_graph(g2, max_edges=5)
    rule.weights[:] = 0.123
    rule.write_back_to_graph()
    for e in rule.edges:
        assert e.weight == pytest.approx(0.123)


# ---------------------------------------------------------------------------
# Modulator bank
# ---------------------------------------------------------------------------


def test_default_modulator_bank_has_three_modulators(graph):
    """Phase 1.6.2 added tyramine as the third modulator. RID + 5-HT
    were the Phase 1.0.4 baseline (tyramine: Pirri 2009, RIM-driven
    AVB/MC/RMD suppression — see DECISIONS.md Phase 1.6.2)."""
    base = np.ones(graph.n_nodes)
    bank = build_default_modulator_bank(graph, base)
    assert bank.n_modulators == 3
    names = {m.name for m in bank.modulators}
    assert names == {"RID", "5HT", "tyramine"}


def test_modulator_concentration_responds_to_rate(graph):
    base = np.ones(graph.n_nodes)
    bank = build_default_modulator_bank(graph, base, tau_m=10.0)
    n = graph.n_nodes
    rate = np.zeros(n)
    # Drive RID to a constant high rate.
    rid = bank.modulators[0]
    rate[rid.producer_idx] = 1.0
    for _ in range(100):
        bank.step_concentrations(rate)
    # c_m should have approached the mean producer rate (= 1.0).
    assert bank.modulators[0].c_m > 0.9
    # 5-HT should be near zero (no producer activity).
    assert abs(bank.modulators[1].c_m) < 1e-6


def test_modulator_threshold_modulation_direction(graph):
    base = np.full(graph.n_nodes, 1.0)
    bank = build_default_modulator_bank(graph, base, tau_m=1.0)
    rid = bank.modulators[0]
    rid.c_m = 1.0  # max concentration
    out = base.copy()
    bank.apply_threshold_modulation(out)
    # RID targets should have *lower* thresholds (sensitivity is negative).
    for idx in rid.target_idx:
        assert out[idx] < base[idx]
    # Non-target neurons should keep base threshold (if 5-HT c_m == 0).
    untouched = np.setdiff1d(
        np.arange(graph.n_nodes),
        np.concatenate([m.target_idx for m in bank.modulators]),
    )
    np.testing.assert_allclose(out[untouched], base[untouched])


def test_modulator_threshold_clamp_prevents_zero(graph):
    base = np.full(graph.n_nodes, 1.0)
    bank = build_default_modulator_bank(graph, base, tau_m=1.0)
    rid = bank.modulators[0]
    # Force a huge concentration → threshold * (1 - 100*0.5) = -49
    rid.c_m = 100.0
    out = base.copy()
    bank.apply_threshold_modulation(out)
    # Clamp should keep them >= 0.1 * base.
    assert out[rid.target_idx[0]] >= 0.1


# ---------------------------------------------------------------------------
# Simulator integration with plasticity + modulators
# ---------------------------------------------------------------------------


def test_simulator_with_plasticity_and_modulators_runs(graph):
    sim = GraphSimulator(
        graph, config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2),
    )
    rule = HebbianRule.from_graph(graph)
    sim.attach_plasticity(rule)
    bank = build_default_modulator_bank(graph, sim.params.threshold)
    sim.attach_modulators(bank)
    out = sim.run(2000, seed=42, record_V=False, record_rate=False)
    final = out["final_state"]
    assert np.all(np.isfinite(final.V))
    assert np.max(np.abs(final.V)) < 10.0
    # Plasticity should have moved weights — at least *some*.
    assert (rule.weights != rule.weights_initial).any()


def test_simulator_modulators_change_thresholds(graph):
    """Threshold array should differ from base after modulator runs."""
    sim = GraphSimulator(
        graph, config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2),
    )
    base_thresh = sim.params.threshold.copy()
    bank = build_default_modulator_bank(graph, base_thresh)
    sim.attach_modulators(bank)
    sim.run(2000, seed=42, record_V=False, record_rate=False)
    # At least one of the modulator-target thresholds should have moved.
    moved = (sim.params.threshold != base_thresh).any()
    assert moved


def test_simulator_plasticity_refresh_updates_queue(graph):
    """After refresh, the queue's DelayBucket W should reflect plastic changes."""
    sim = GraphSimulator(graph, config=SimulatorConfig(noise_level=0.0, sensory_noise=0.0))
    rule = HebbianRule.from_graph(graph, max_edges=20, eta=0.0, lam=0.0)
    sim.attach_plasticity(rule)
    # Manually overwrite plastic weights to 0.999 and force a refresh.
    rule.weights[:] = 0.999
    sim._refresh_chemical_matrices()
    # The queue bucket at delay=1 should now have 0.999 at every
    # (post_idx, pre_idx) for the plastic edges (signed).
    bucket = sim.queue.buckets[0]
    for pre, post, sign in zip(rule.pre_idx, rule.post_idx, rule.signs):
        assert bucket.W[post, pre] == pytest.approx(0.999 * sign)
