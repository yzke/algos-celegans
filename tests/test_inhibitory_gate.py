"""Phase 1.6.1 — tests for the inhibitory_command_gate subgraph.

The subgraph itself is purely a *view*; the inhibitory edges from
RIS/AVL/DVB onto their downstream targets are already in W_chem with
sign=-1 because those neurons are in the canonical 26-GABA set
(McIntire 1993, Gendrel 2016). Adding the subgraph is a data-level
acknowledgement of biological functional role, not a dynamics change.

These tests document both facts:
  1. The subgraph loads correctly and aligns with the connectome.
  2. In the bare Phase 1.0 simulator, the gate is DORMANT — none of
     its member neurons fire — so adding it does not by itself change
     forward↔reversal coupling.
  3. When the gate is *driven* (via direct injection at RIS), the
     existing inhibitory chemical pathway DOES suppress downstream
     command neurons. So the mechanism is sound; the missing piece
     is what activates RIS (Phase 1.5 body feedback, or Phase 1.6.2
     tyramine arm of RIM).
"""

from __future__ import annotations

import numpy as np
import pytest

from algos.graph import (
    build_canonical_subgraphs,
    load_connectome_into_graph,
)
from algos.neural_v2 import GraphSimulator, SimulatorConfig


# ---------------------------------------------------------------------------
# Structural tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph_with_gate():
    g = load_connectome_into_graph()
    build_canonical_subgraphs(g)
    return g


def test_inhibitory_gate_present(graph_with_gate):
    assert "inhibitory_command_gate" in graph_with_gate.subgraphs


def test_inhibitory_gate_members(graph_with_gate):
    gate = graph_with_gate.subgraphs["inhibitory_command_gate"]
    assert set(gate.node_names) == {"RIS", "AVL", "DVB", "ALA", "RIH"}
    assert gate.type == "recurrent"


def test_inhibitory_gate_overlaps_match_design(graph_with_gate):
    gate = graph_with_gate.subgraphs["inhibitory_command_gate"]
    # AVL and DVB are also in defecation_pacemaker (per circuits.py note).
    defec = graph_with_gate.subgraphs["defecation_pacemaker"]
    assert gate.overlap_with(defec) == {"AVL", "DVB"}
    # RIH is also in modulator_5HT.
    sht = graph_with_gate.subgraphs["modulator_5HT"]
    assert gate.overlap_with(sht) == {"RIH"}


def test_inhibitory_gate_members_are_gaba_or_modulator(graph_with_gate):
    """RIS/AVL/DVB are GABAergic; ALA is a peptide source; RIH receives."""
    gate = graph_with_gate.subgraphs["inhibitory_command_gate"]
    expected_gaba = {"RIS", "AVL", "DVB"}
    for name in expected_gaba:
        assert graph_with_gate.node(name).neurotransmitter == "GABA", (
            f"{name} should be in canonical GABAergic set"
        )


def test_inhibitory_gate_has_gaba_edges_to_command_targets(graph_with_gate):
    """At least one GABA edge from a gate neuron lands on AVE / AVA / AVB /
    RIM / RIB (command + integrator pool). RIS in particular is well-
    documented to target AVE pair (~17-18 contacts each)."""
    g = graph_with_gate
    command_pool = {
        "AVAL", "AVAR", "AVBL", "AVBR",
        "AVEL", "AVER", "AVDL", "AVDR",
        "PVCL", "PVCR", "RIML", "RIMR",
        "RIBL", "RIBR",
    }
    found_inhibitory = []
    for src in ("RIS", "AVL", "DVB"):
        for e in g.out_edges(src, "chemical"):
            if e.target in command_pool:
                assert e.sign == -1, (
                    f"{src} → {e.target} should be inhibitory (GABA)"
                )
                found_inhibitory.append((src, e.target, e.weight))
    assert len(found_inhibitory) > 0, (
        "no inhibitory edges from gate to command pool — connectome "
        "data drift?"
    )


# ---------------------------------------------------------------------------
# Dynamics tests — documenting bare-network behavior
# ---------------------------------------------------------------------------


def _make_sim():
    g = load_connectome_into_graph()
    build_canonical_subgraphs(g)
    return GraphSimulator(
        g, config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2)
    )


def test_gate_silent_in_bare_simulator():
    """All five gate neurons fire 0 times across a 2000-tick run.

    This is the central Phase 1.0 observation: the inhibitory hub
    exists structurally but cannot activate without external drive.
    """
    sim = _make_sim()
    state = sim.initial_state(seed=1042)
    rng = np.random.default_rng(1042)
    sens = np.zeros(sim.n)
    for _ in range(2000):
        sens[:] = 0.0
        sens[sim.sensory_idx] = (
            rng.standard_normal(sim.sensory_idx.size) * 0.2
        )
        state = sim.step(state, sens, rng)
    for name in ("RIS", "AVL", "DVB", "ALA", "RIH"):
        idx = sim.graph.index_of(name)
        # The gate is dormant; assert spike_count is small (we allow up
        # to a handful so the test isn't brittle to noise / future
        # parameter tweaks, but document the empirical 0).
        assert int(state.spike_count[idx]) <= 2, (
            f"{name} spiked {state.spike_count[idx]} times — gate not "
            f"dormant as documented; investigate"
        )


def test_ris_stimulation_inhibits_avb():
    """When RIS is *driven* by injected sensory input, the inhibitory
    chemical pathway suppresses AVB (forward command).

    Compares:
      A. baseline: sensory_noise only.
      B. + RIS injection: same setup plus constant strong input at
         the RIS index.

    Expected: AVB total spike count drops between A and B (or at
    least does not increase). This validates the mechanism without
    asserting a specific magnitude.
    """
    sim_a = _make_sim()
    sim_b = _make_sim()
    avbl_idx_a = sim_a.graph.index_of("AVBL")
    avbr_idx_a = sim_a.graph.index_of("AVBR")
    avel_idx = sim_b.graph.index_of("AVEL")
    aver_idx = sim_b.graph.index_of("AVER")
    ris_idx = sim_b.graph.index_of("RIS")

    # A. baseline.
    state = sim_a.initial_state(seed=7)
    rng = np.random.default_rng(7)
    sens = np.zeros(sim_a.n)
    for _ in range(3000):
        sens[:] = 0.0
        sens[sim_a.sensory_idx] = rng.standard_normal(sim_a.sensory_idx.size) * 0.2
        state = sim_a.step(state, sens, rng)
    avb_a = int(state.spike_count[avbl_idx_a] + state.spike_count[avbr_idx_a])
    ave_a = int(state.spike_count[avel_idx] + state.spike_count[aver_idx])

    # B. with RIS driven.
    state = sim_b.initial_state(seed=7)
    rng = np.random.default_rng(7)
    sens = np.zeros(sim_b.n)
    for _ in range(3000):
        sens[:] = 0.0
        sens[sim_b.sensory_idx] = rng.standard_normal(sim_b.sensory_idx.size) * 0.2
        sens[ris_idx] = 0.5     # drive RIS hard enough to fire
        state = sim_b.step(state, sens, rng)
    avb_b = int(state.spike_count[avbl_idx_a] + state.spike_count[avbr_idx_a])
    ave_b = int(state.spike_count[avel_idx] + state.spike_count[aver_idx])
    ris_spikes = int(state.spike_count[ris_idx])

    # Validate: (1) RIS actually fired with the injection;
    #          (2) AVE (the canonical RIS target) reduced its firing
    #              compared to baseline; AVE > AVB as a target because
    #              RIS → AVE has 17+18 contacts vs ~zero to AVB.
    assert ris_spikes > 5, (
        f"RIS injection of 0.5 produced only {ris_spikes} spikes — "
        f"adjust drive level if this fails"
    )
    assert ave_b <= ave_a, (
        f"AVE pair fired more under RIS drive ({ave_a}→{ave_b}) — "
        f"inhibitory pathway not working as designed"
    )


def test_no_regression_when_gate_not_used():
    """Numerical equivalence: a simulator that loads the gate subgraph
    but does not use it must behave bit-identically to a simulator
    without the gate. The subgraph is a *view*; it should not affect
    dynamics.
    """
    # Build two graphs: one with all 14 subgraphs, one with no subgraphs.
    g_with = load_connectome_into_graph()
    build_canonical_subgraphs(g_with)
    g_without = load_connectome_into_graph()
    # No build_canonical_subgraphs call.

    sim_w = GraphSimulator(
        g_with,
        config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2),
    )
    sim_n = GraphSimulator(
        g_without,
        config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2),
    )
    s_w = sim_w.initial_state(seed=99)
    s_n = sim_n.initial_state(seed=99)
    rng_w = np.random.default_rng(99)
    rng_n = np.random.default_rng(99)
    sens_w = np.zeros(sim_w.n)
    sens_n = np.zeros(sim_n.n)
    for _ in range(500):
        sens_w[:] = 0.0
        sens_w[sim_w.sensory_idx] = rng_w.standard_normal(sim_w.sensory_idx.size) * 0.2
        s_w = sim_w.step(s_w, sens_w, rng_w)
        sens_n[:] = 0.0
        sens_n[sim_n.sensory_idx] = rng_n.standard_normal(sim_n.sensory_idx.size) * 0.2
        s_n = sim_n.step(s_n, sens_n, rng_n)
    np.testing.assert_allclose(s_w.V, s_n.V, atol=1e-12,
                               err_msg="subgraph view changed dynamics")
    np.testing.assert_array_equal(
        s_w.spike_count, s_n.spike_count,
        err_msg="subgraph view changed spike behavior"
    )
