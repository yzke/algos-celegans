"""Tests for the Phase 1.0 graph-native primitives."""

from __future__ import annotations

import numpy as np
import pytest

from algos.connectome import ConnectomeData
from algos.graph import (
    DEFAULT_MODULATOR_NEURONS,
    Edge,
    NeuralGraph,
    Node,
    Subgraph,
    load_connectome_into_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded_graph(connectome: ConnectomeData) -> NeuralGraph:
    return load_connectome_into_graph(connectome)


# ---------------------------------------------------------------------------
# Node / Edge contract
# ---------------------------------------------------------------------------


def test_node_construction_defaults():
    n = Node.from_connectome_row(
        name="AVAL", category="interneuron", neurotransmitter="default"
    )
    assert n.id == "AVAL"
    assert n.category == "interneuron"
    assert n.threshold > 0 and n.tau > 0
    assert n.V == 0.0
    assert n.refractory == 0
    assert not n.is_modulator
    assert n.f_name == "lif_standard"


def test_node_modulator_uses_modulator_defaults():
    n = Node.from_connectome_row(
        name="RID", category="interneuron", neurotransmitter="default",
        is_modulator=True,
    )
    # Modulator nodes get a longer tau and higher threshold (slow integrator).
    assert n.is_modulator
    assert n.tau > 20.0


def test_edge_rejects_invalid_type():
    with pytest.raises(ValueError):
        Edge(source="A", target="B", type="foobar", weight=1.0, sign=+1, delay=1)


def test_edge_rejects_negative_weight():
    with pytest.raises(ValueError):
        Edge(source="A", target="B", type="chemical", weight=-0.1, sign=+1, delay=1)


def test_edge_signed_weight():
    e = Edge(source="A", target="B", type="chemical", weight=0.7, sign=-1, delay=2)
    assert e.signed_weight == pytest.approx(-0.7)
    assert e.key == "chem"


# ---------------------------------------------------------------------------
# NeuralGraph
# ---------------------------------------------------------------------------


def test_graph_n_nodes_is_302(loaded_graph: NeuralGraph):
    assert loaded_graph.n_nodes == 302


def test_graph_chemical_count_matches_connectome(
    loaded_graph: NeuralGraph, connectome: ConnectomeData
):
    expected = int((connectome.W_chem_raw != 0).sum())
    assert loaded_graph.n_edges_of_type("chemical") == expected


def test_graph_electrical_count_is_symmetric_pair(
    loaded_graph: NeuralGraph, connectome: ConnectomeData
):
    # Gap junctions are stored as one directed edge *per direction*.
    # Cook's symmetric sheet has 2 * (unique unordered pairs) nonzero entries.
    n_gap_directed = int((connectome.W_gap_raw != 0).sum())
    assert loaded_graph.n_edges_of_type("electrical") == n_gap_directed


def test_graph_modulator_nodes_marked(loaded_graph: NeuralGraph):
    present = {n for n in DEFAULT_MODULATOR_NEURONS if loaded_graph.has_node(n)}
    for name in present:
        assert loaded_graph.node(name).is_modulator, name
    # And non-modulators stay un-marked.
    assert not loaded_graph.node("AVAL").is_modulator


def test_graph_index_stability(loaded_graph: NeuralGraph):
    names = loaded_graph.neuron_names()
    assert names == sorted(names)
    assert len(set(names)) == len(names) == 302
    for i, name in enumerate(names):
        assert loaded_graph.index_of(name) == i


def test_graph_collect_scatter_V_roundtrip(loaded_graph: NeuralGraph):
    V0 = np.random.default_rng(0).standard_normal(loaded_graph.n_nodes)
    loaded_graph.scatter_V(V0.astype(np.float64))
    V1 = loaded_graph.collect_V()
    np.testing.assert_allclose(V0, V1, atol=1e-12)
    loaded_graph.reset_dynamic_state()


def test_graph_edge_signs_match_neurotransmitter(loaded_graph: NeuralGraph):
    """GABAergic source → sign=-1; default source → sign=+1."""
    for e in loaded_graph.edges("chemical"):
        src_node = loaded_graph.node(e.source)
        if src_node.neurotransmitter == "GABA":
            assert e.sign == -1, e
        else:
            assert e.sign == +1, e


def test_graph_summary_self_consistent(loaded_graph: NeuralGraph):
    s = loaded_graph.summary()
    assert s["n_chemical"] + s["n_electrical"] + s["n_modulatory"] == s["n_edges"]
    assert sum(s["by_category"].values()) == s["n_nodes"]


# ---------------------------------------------------------------------------
# Subgraph
# ---------------------------------------------------------------------------


def test_subgraph_basic_view(loaded_graph: NeuralGraph):
    sg = Subgraph(
        name="ava_pair",
        type="recurrent",
        node_names=["AVAL", "AVAR"],
        parent=loaded_graph,
    )
    sg.materialize()
    assert sg.n_nodes == 2
    assert sg.W_chem.shape == (2, 2)
    assert sg.W_gap.shape == (2, 2)
    # Indices should be the parent's alphabetic positions.
    idx = sg.node_indices()
    assert idx.tolist() == [
        loaded_graph.index_of("AVAL"), loaded_graph.index_of("AVAR")
    ]


def test_subgraph_rejects_unknown_node(loaded_graph: NeuralGraph):
    with pytest.raises(KeyError):
        Subgraph(
            name="bad",
            type="recurrent",
            node_names=["AVAL", "NOT_A_REAL_NEURON"],
            parent=loaded_graph,
        )


def test_subgraph_rejects_unknown_type(loaded_graph: NeuralGraph):
    with pytest.raises(ValueError):
        Subgraph(
            name="bad", type="foo", node_names=["AVAL"], parent=loaded_graph
        )


def test_subgraph_overlap(loaded_graph: NeuralGraph):
    a = Subgraph(
        name="reversal",
        type="recurrent",
        node_names=["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"],
        parent=loaded_graph,
    )
    b = Subgraph(
        name="touch_reflex",
        type="feedforward",
        node_names=["ALML", "ALMR", "AVM", "AVDL", "AVDR"],
        parent=loaded_graph,
    )
    assert a.overlap_with(b) == {"AVDL", "AVDR"}


def test_subgraph_topological_order_only_for_feedforward(
    loaded_graph: NeuralGraph,
):
    rec = Subgraph(
        name="rec",
        type="recurrent",
        node_names=["AVAL", "AVAR"],
        parent=loaded_graph,
    )
    assert rec.topological_order() is None
    ff = Subgraph(
        name="ff",
        type="feedforward",
        node_names=["ALML", "AVDL", "AVAL"],
        parent=loaded_graph,
    )
    order = ff.topological_order()
    # Either we get a valid order or None (if no edges form a chain).
    assert order is None or set(order) == {"ALML", "AVDL", "AVAL"}


# ---------------------------------------------------------------------------
# Loader-level invariants
# ---------------------------------------------------------------------------


def test_loader_chemical_signs_match_W_chem(
    loaded_graph: NeuralGraph, connectome: ConnectomeData
):
    """For every nonzero raw chem entry, the graph edge's signed weight has
    the same sign as connectome.W_chem at that position."""
    W_raw = connectome.W_chem_raw
    W_signed = connectome.W_chem  # signed, normalized
    # Pick a few rows to keep this O(N*sparse).
    rng = np.random.default_rng(0)
    sample_post = rng.choice(connectome.n_neurons, size=20, replace=False)
    for i_post in sample_post:
        for j_pre in range(connectome.n_neurons):
            if W_raw[i_post, j_pre] == 0:
                continue
            pre = connectome.neuron_names[j_pre]
            post = connectome.neuron_names[i_post]
            e = loaded_graph.get_edge(pre, post, "chemical")
            assert e is not None
            assert np.sign(e.signed_weight) == np.sign(W_signed[i_post, j_pre])


def test_loader_no_modulatory_edges_yet(loaded_graph: NeuralGraph):
    # Phase 1.0.1 builds the structure; modulator edges are added in 1.0.4.
    assert loaded_graph.n_edges_of_type("modulatory") == 0
