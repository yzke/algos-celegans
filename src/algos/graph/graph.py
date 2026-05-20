"""NeuralGraph — the graph-native container for the Phase 1.0 system.

Per docs/phase1_design.md §1.1:

  Graph:
    nodes: 节点集合
    edges: 边集合
    subgraphs: 子图集合(对子集 nodes 和 edges 的引用)
    图是 first-class 对象。所有运算都通过图接口完成。

The underlying store is a ``networkx.MultiDiGraph``: directed (chemical
synapses have a polarity; electrical synapses are stored as a mirrored
pair) with multi-edge support keyed by edge type ('chem' | 'gap' |
'mod') so the same ordered pair can hold both a chemical synapse and a
gap junction simultaneously.

Convenience: ``index_of(name)`` provides a stable position for each
node (alphabetic by id), so subgraphs can yield NumPy index arrays that
align with the parent graph for fast scatter/gather.

This module does not load the connectome itself — see
``graph.loader.load_connectome_into_graph``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

import networkx as nx
import numpy as np

from algos.graph.node import Node
from algos.graph.edge import Edge, edge_key_for_type


@dataclass
class NeuralGraph:
    """First-class graph container for the Phase 1.0 neural system."""

    G: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    # `subgraphs` is intentionally a plain dict — a Subgraph is a *view*
    # onto this graph (see subgraph.py), not a separate object.
    subgraphs: dict[str, "Subgraph"] = field(default_factory=dict)
    # Frozen alphabetic ordering of node ids — `index_of[name]` gives the
    # column of `name` in the canonical NumPy state vector. Rebuilt on
    # demand via `rebuild_index()`.
    _index_of: dict[str, int] = field(default_factory=dict, init=False)
    _names_by_index: list[str] = field(default_factory=list, init=False)

    # ----------------------------------------------------------- queries

    @property
    def n_nodes(self) -> int:
        return self.G.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.G.number_of_edges()

    def n_edges_of_type(self, edge_type: str) -> int:
        key = edge_key_for_type(edge_type)
        return sum(1 for _, _, k in self.G.edges(keys=True) if k == key)

    def has_node(self, name: str) -> bool:
        return self.G.has_node(name)

    def node(self, name: str) -> Node:
        return self.G.nodes[name]["node"]

    def nodes(self) -> Iterator[Node]:
        for _, data in self.G.nodes(data=True):
            yield data["node"]

    def edges(self, edge_type: str | None = None) -> Iterator[Edge]:
        """Yield every Edge, optionally filtered by type."""
        if edge_type is None:
            target_key = None
        else:
            target_key = edge_key_for_type(edge_type)
        for _, _, k, data in self.G.edges(keys=True, data=True):
            if target_key is None or k == target_key:
                yield data["edge"]

    def get_edge(
        self, source: str, target: str, edge_type: str
    ) -> Edge | None:
        """Look up a specific edge by (source, target, type)."""
        key = edge_key_for_type(edge_type)
        if self.G.has_edge(source, target, key=key):
            return self.G[source][target][key]["edge"]
        return None

    def out_edges(self, source: str, edge_type: str | None = None) -> Iterator[Edge]:
        """All edges leaving `source`, optionally filtered by type."""
        target_key = edge_key_for_type(edge_type) if edge_type else None
        for _, _, k, data in self.G.out_edges(source, keys=True, data=True):
            if target_key is None or k == target_key:
                yield data["edge"]

    def in_edges(self, target: str, edge_type: str | None = None) -> Iterator[Edge]:
        """All edges arriving at `target`, optionally filtered by type."""
        target_key = edge_key_for_type(edge_type) if edge_type else None
        for _, _, k, data in self.G.in_edges(target, keys=True, data=True):
            if target_key is None or k == target_key:
                yield data["edge"]

    def neuron_names(self) -> list[str]:
        """Stable alphabetic list of all neuron names."""
        if not self._names_by_index:
            self.rebuild_index()
        return list(self._names_by_index)

    def index_of(self, name: str) -> int:
        """Position of `name` in the canonical NumPy state vector."""
        if not self._index_of:
            self.rebuild_index()
        return self._index_of[name]

    # ---------------------------------------------------------- mutation

    def add_node(self, node: Node) -> None:
        self.G.add_node(node.id, node=node)
        self._invalidate_index()

    def add_edge(self, edge: Edge) -> None:
        if not self.G.has_node(edge.source):
            raise KeyError(f"Edge source {edge.source!r} not in graph")
        if not self.G.has_node(edge.target):
            raise KeyError(f"Edge target {edge.target!r} not in graph")
        self.G.add_edge(edge.source, edge.target, key=edge.key, edge=edge)

    def rebuild_index(self) -> None:
        """Recompute the canonical alphabetic name → index map."""
        self._names_by_index = sorted(self.G.nodes())
        self._index_of = {n: i for i, n in enumerate(self._names_by_index)}

    def _invalidate_index(self) -> None:
        self._names_by_index.clear()
        self._index_of.clear()

    # ------------------------------------------------------------ state

    def collect_V(self) -> np.ndarray:
        """Gather V from every Node into a (N,) array in canonical order."""
        names = self.neuron_names()
        return np.array([self.node(n).V for n in names], dtype=np.float64)

    def scatter_V(self, V: np.ndarray) -> None:
        """Write a (N,) V vector back into Node fields."""
        names = self.neuron_names()
        if V.shape != (len(names),):
            raise ValueError(
                f"V shape {V.shape} != ({len(names)},)"
            )
        for name, v in zip(names, V):
            self.node(name).V = float(v)

    def reset_dynamic_state(self) -> None:
        """Reset every Node's dynamic state (V, refractory). Static params kept."""
        for n in self.nodes():
            n.reset_state()

    # ------------------------------------------------------ subgraph mgmt

    def register_subgraph(self, subgraph: "Subgraph") -> None:
        if subgraph.name in self.subgraphs:
            raise KeyError(f"Subgraph {subgraph.name!r} already registered")
        self.subgraphs[subgraph.name] = subgraph

    def subgraph_membership(self) -> dict[str, list[str]]:
        """For each node name, which subgraphs contain it (debug helper)."""
        out: dict[str, list[str]] = {n.id: [] for n in self.nodes()}
        for sname, sg in self.subgraphs.items():
            for nm in sg.node_names:
                out[nm].append(sname)
        return out

    # ----------------------------------------------------------- summary

    def summary(self) -> dict[str, int | dict[str, int]]:
        """Useful counts for diagnostics."""
        from collections import Counter
        cats = Counter(n.category for n in self.nodes())
        nts = Counter(n.neurotransmitter for n in self.nodes())
        mods = sum(1 for n in self.nodes() if n.is_modulator)
        plastic_nodes = sum(1 for n in self.nodes() if n.is_plastic)
        plastic_edges = sum(1 for e in self.edges() if e.is_plastic)
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_chemical": self.n_edges_of_type("chemical"),
            "n_electrical": self.n_edges_of_type("electrical"),
            "n_modulatory": self.n_edges_of_type("modulatory"),
            "by_category": dict(cats),
            "by_neurotransmitter": dict(nts),
            "n_modulator_nodes": mods,
            "n_plastic_nodes": plastic_nodes,
            "n_plastic_edges": plastic_edges,
            "n_subgraphs": len(self.subgraphs),
        }


# Late import to avoid circular dependency — Subgraph references NeuralGraph.
from algos.graph.subgraph import Subgraph  # noqa: E402

__all__ = ["NeuralGraph"]
