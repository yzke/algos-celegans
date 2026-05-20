"""Tests for the canonical functional-circuit definitions."""

from __future__ import annotations

import numpy as np
import pytest

from algos.graph import (
    CIRCUIT_SPECS,
    NeuralGraph,
    Subgraph,
    build_canonical_subgraphs,
    load_connectome_into_graph,
    summarize_subgraphs,
)


# ---------------------------------------------------------------------------
# Coverage / membership integrity
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def graph_with_subs() -> NeuralGraph:
    g = load_connectome_into_graph()
    build_canonical_subgraphs(g)
    return g


def test_at_least_eight_subgraphs(graph_with_subs: NeuralGraph):
    """Phase 1.0.3 acceptance: ≥8 functional subgraphs."""
    assert len(graph_with_subs.subgraphs) >= 8


def test_specs_contain_the_required_circuit_families():
    """Each of the families listed in docs/phase1.0.md must appear."""
    names = {s.name for s in CIRCUIT_SPECS}
    required = {
        "reversal_command",      # backward command
        "forward_command",       # forward command
        "anterior_touch",        # touch reflex (paired with posterior)
        "posterior_touch",
        "chemosensory_amphid",   # chemical sense
        "thermosensory",         # temperature
        "head_motor_cpg",        # head CPG
        "pharyngeal_cpg",        # feeding CPG
    }
    missing = required - names
    assert not missing, f"missing required circuit families: {missing}"


def test_modulator_subgraph_present(graph_with_subs: NeuralGraph):
    """Task §1.0.3 calls for a modulator subgraph (8th item)."""
    assert "modulator_RID" in graph_with_subs.subgraphs


def test_all_members_resolve_to_connectome_nodes(graph_with_subs: NeuralGraph):
    """Every spec member should be present in the 302-neuron set."""
    for spec in CIRCUIT_SPECS:
        for member in spec.members:
            assert graph_with_subs.has_node(member), (
                f"{spec.name}: member {member!r} not in connectome"
            )


def test_subgraph_internal_edge_counts_positive(graph_with_subs: NeuralGraph):
    """Every functional circuit should have at least *some* internal
    chemical or gap edges — otherwise the circuit is just a list."""
    for name, sg in graph_with_subs.subgraphs.items():
        n_internal = int((sg.W_chem != 0).sum()) + int((sg.W_gap != 0).sum())
        assert n_internal > 0, f"{name}: no internal edges"


# ---------------------------------------------------------------------------
# Overlap is non-trivial (the §4.4 design intent)
# ---------------------------------------------------------------------------


def test_overlap_nodes_share_V_across_subgraphs(graph_with_subs: NeuralGraph):
    """A shared node's V is read identically by both subgraphs (the §4.4
    promise: the shared node is the same Node object in the parent)."""
    g = graph_with_subs
    avbl = g.node("AVBL")
    avbl.V = 0.42
    fwd = g.subgraphs["forward_command"]
    post = g.subgraphs["posterior_touch"]
    # AVBL is in both.
    assert "AVBL" in fwd.node_names and "AVBL" in post.node_names
    # Pulling V via either subgraph yields the same number.
    V_global = g.collect_V()
    fwd_V = fwd.gather_V(V_global)
    post_V = post.gather_V(V_global)
    assert fwd_V[fwd.local_index("AVBL")] == pytest.approx(0.42)
    assert post_V[post.local_index("AVBL")] == pytest.approx(0.42)
    g.reset_dynamic_state()


def test_command_circuit_overlap_with_touch(graph_with_subs: NeuralGraph):
    """Anterior touch should share AVA/AVD with the reversal command —
    this is the canonical overlap (touch sensors hand off to the same
    command interneurons that the reversal circuit uses)."""
    rev = graph_with_subs.subgraphs["reversal_command"]
    ant = graph_with_subs.subgraphs["anterior_touch"]
    shared = rev.overlap_with(ant)
    assert {"AVAL", "AVAR", "AVDL", "AVDR"}.issubset(shared)


def test_chemo_thermo_share_ria(graph_with_subs: NeuralGraph):
    """RIA is the design's named overlap point between chemo and thermo paths."""
    chemo = graph_with_subs.subgraphs["chemosensory_amphid"]
    thermo = graph_with_subs.subgraphs["thermosensory"]
    shared = chemo.overlap_with(thermo)
    assert {"RIAL", "RIAR"}.issubset(shared)


# ---------------------------------------------------------------------------
# Subgraph dynamics-side smoke tests
# ---------------------------------------------------------------------------


def test_recurrent_subgraph_has_no_topological_order(graph_with_subs: NeuralGraph):
    """Recurrent subgraphs must contain cycles (sometimes)."""
    rev = graph_with_subs.subgraphs["reversal_command"]
    # We declare it recurrent → topological_order returns None by contract
    # (regardless of whether internal chemical edges happen to form a cycle).
    assert rev.topological_order() is None


def test_subgraph_matrix_dims(graph_with_subs: NeuralGraph):
    for name, sg in graph_with_subs.subgraphs.items():
        n = sg.n_nodes
        assert sg.W_chem.shape == (n, n), name
        assert sg.W_gap.shape == (n, n), name
        # Gap matrix should remain symmetric.
        np.testing.assert_allclose(sg.W_gap, sg.W_gap.T, atol=1e-12)


def test_summary_includes_all_subgraphs(graph_with_subs: NeuralGraph):
    summaries = summarize_subgraphs(graph_with_subs)
    assert {s["name"] for s in summaries} == set(graph_with_subs.subgraphs)
