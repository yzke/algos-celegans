"""Subgraph — a named functional circuit view onto a NeuralGraph.

Per docs/phase1_design.md §1.4 / §4:

  Subgraph:
    name: 例如 'forward_command', 'chemotaxis_loop', 'feeding_CPG'
    type: 'feedforward' | 'recurrent'
    nodes: 这个子图包含的节点引用
    edges: 这个子图包含的边引用
    state: 这个子图的内部状态(如果是 recurrent 类型)
    matrix_view: 实时从图提取的矩阵视图(缓存)

The Subgraph is a *view*, not a copy. The authoritative state still
lives on the Nodes of the parent graph; the Subgraph just provides:

  1. A scoped index set (which node positions belong here).
  2. A cached matrix view of edges restricted to this scope (built once
     at construction; can be rebuilt via `materialize`).
  3. A handle for the runtime to call subgraph-specific dynamics, if
     any (Phase 1.0.3 leaves these "informational" — the dynamics in
     1.0.2 already operate on the global graph; subgraphs are used for
     diagnostics, anti-correlation tests, and Phase 1.0.4 will use them
     for scoped plasticity / modulator targeting).

Two flavors:
  * feedforward: signal has a clear entry → exit pathway (touch reflex)
  * recurrent:   reciprocal connectivity producing persistent activity
                 (command circuit, head CPG)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable

import numpy as np
import networkx as nx

if TYPE_CHECKING:
    from algos.graph.graph import NeuralGraph


VALID_SUBGRAPH_TYPES: frozenset[str] = frozenset({"feedforward", "recurrent"})


@dataclass
class Subgraph:
    """A named view onto a subset of NeuralGraph nodes + edges."""

    name: str
    type: str                                # 'feedforward' | 'recurrent'
    node_names: list[str]                    # node ids in this subgraph
    parent: "NeuralGraph" = field(repr=False)
    # Optional per-subgraph state (for recurrent CPGs to carry a phase
    # variable across ticks, etc.). Kept generic.
    state: dict = field(default_factory=dict)
    # Cached matrix view, built on demand.
    _W_chem_view: np.ndarray | None = field(default=None, init=False, repr=False)
    _W_gap_view: np.ndarray | None = field(default=None, init=False, repr=False)
    _node_indices: np.ndarray | None = field(default=None, init=False, repr=False)
    _local_to_global: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.type not in VALID_SUBGRAPH_TYPES:
            raise ValueError(
                f"Subgraph type {self.type!r} not in {sorted(VALID_SUBGRAPH_TYPES)}"
            )
        # Deduplicate while keeping order (declared order matters for
        # diagnostics readability).
        seen: set[str] = set()
        dedup: list[str] = []
        for n in self.node_names:
            if n in seen:
                continue
            if not self.parent.has_node(n):
                raise KeyError(
                    f"Subgraph {self.name!r}: node {n!r} not in parent graph"
                )
            seen.add(n)
            dedup.append(n)
        self.node_names = dedup

    # ---------------------------------------------------------- indices

    @property
    def n_nodes(self) -> int:
        return len(self.node_names)

    def node_indices(self) -> np.ndarray:
        """NumPy int array of parent-graph positions for these nodes."""
        if self._node_indices is None:
            self._node_indices = np.array(
                [self.parent.index_of(n) for n in self.node_names],
                dtype=np.int64,
            )
        return self._node_indices

    def local_index(self, name: str) -> int:
        return self.node_names.index(name)

    # ------------------------------------------------------ matrix view

    def materialize(self) -> None:
        """Build cached (n × n) chemical and gap matrices over this subgraph.

        ``W_chem_view[i, j]`` = signed_weight of the chemical edge from
        local node j (pre) to local node i (post). Convention matches the
        global ConnectomeData (rows = post, cols = pre) — see
        ``algos.connectome``.
        """
        n = len(self.node_names)
        local_idx = {n_name: i for i, n_name in enumerate(self.node_names)}
        W_chem = np.zeros((n, n), dtype=np.float64)
        W_gap = np.zeros((n, n), dtype=np.float64)
        for e in self.parent.edges("chemical"):
            if e.source in local_idx and e.target in local_idx:
                i_post = local_idx[e.target]
                j_pre = local_idx[e.source]
                W_chem[i_post, j_pre] = e.signed_weight
        for e in self.parent.edges("electrical"):
            if e.source in local_idx and e.target in local_idx:
                W_gap[local_idx[e.target], local_idx[e.source]] = e.signed_weight
        # Gap symmetry sanity: take elementwise max with transpose. The
        # loader emits both directions but we guard against partial
        # rebuilds.
        W_gap = np.maximum(W_gap, W_gap.T)
        self._W_chem_view = W_chem
        self._W_gap_view = W_gap

    @property
    def W_chem(self) -> np.ndarray:
        if self._W_chem_view is None:
            self.materialize()
        return self._W_chem_view  # type: ignore[return-value]

    @property
    def W_gap(self) -> np.ndarray:
        if self._W_gap_view is None:
            self.materialize()
        return self._W_gap_view  # type: ignore[return-value]

    def gather_V(self, V_global: np.ndarray) -> np.ndarray:
        """Pull the V values for this subgraph from a global state vector."""
        return V_global[self.node_indices()]

    # ------------------------------------------------------------- info

    def topological_order(self) -> list[str] | None:
        """For feedforward subgraphs, return a topological order; else None.

        Returns None for recurrent subgraphs (cycles → no topological order).
        Only chemical edges between subgraph members are considered.
        """
        if self.type != "feedforward":
            return None
        H = nx.DiGraph()
        H.add_nodes_from(self.node_names)
        for e in self.parent.edges("chemical"):
            if e.source in H and e.target in H:
                H.add_edge(e.source, e.target)
        try:
            return list(nx.topological_sort(H))
        except nx.NetworkXUnfeasible:
            # If the user labeled it feedforward but there's a real cycle
            # in chemical edges, surface that.
            return None

    def overlap_with(self, other: "Subgraph") -> set[str]:
        """Set of node names shared with another subgraph (the §4.4 overlaps)."""
        return set(self.node_names) & set(other.node_names)


__all__ = ["Subgraph", "VALID_SUBGRAPH_TYPES"]
