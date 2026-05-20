"""Hebbian plasticity for Phase 1.0.4.

Per docs/phase1_design.md §6:

  对每条 plastic 边 (i → j):
    ΔW = η × pre.V × post.V - λ × W
    W_new = W + ΔW
    W_new 被夹在 [W_min, W_max] 范围内

For a spiking network the per-tick V is mostly zero (or v_reset),
which would starve the Hebbian term of signal. We instead drive the
rule with the leaky-integrator ``rate`` trace from the runner — this
is the same quantity used as the comparison-metric observable
(``state.rate``), and it's continuous + bounded.

The HebbianRule keeps weights non-negative (plastic edges retain their
original *sign* from the pre-synaptic NT; the weight magnitude is what
the rule updates). Update is once per simulation tick; cost is O(k)
where k = number of plastic edges (target 50-100).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from algos.graph import NeuralGraph
from algos.graph.edge import Edge


# Default plastic-neuron set — well-attested learning hubs in C. elegans
# (Phase 0.8 review): AWC chemosensory learning, AIY/AIB/AIZ associative
# memory, RIA premotor integrator, RIM, AFD thermal memory, PLM
# mechanical habituation, AVD touch-driven reversal command.
DEFAULT_PLASTIC_PRE_NEURONS: tuple[str, ...] = (
    "AWCL", "AWCR",
    "AIYL", "AIYR",
    "AIBL", "AIBR",
    "AIZL", "AIZR",
    "RIAL", "RIAR",
    "RIML", "RIMR",
    "AFDL", "AFDR",
    "PLML", "PLMR",
    "AVDL", "AVDR",
)

DEFAULT_ETA: float = 5e-4
DEFAULT_LAMBDA: float = 5e-5
DEFAULT_MAX_PLASTIC_EDGES: int = 100


@dataclass
class HebbianRule:
    """Stateful Hebbian + decay rule over a fixed plastic edge set.

    The rule operates on **edge magnitudes** (always non-negative).
    The signed_weight that the dynamics layer sees is ``sign * weight``;
    we never flip the sign — that would require structural plasticity,
    which is outside Phase 1.0.

    Construct via ``HebbianRule.from_graph(graph)``.
    """

    pre_idx: np.ndarray         # (E,) int — pre-synaptic positions
    post_idx: np.ndarray        # (E,) int — post-synaptic positions
    signs: np.ndarray           # (E,) ±1
    weights: np.ndarray         # (E,) current magnitudes
    w_min: np.ndarray           # (E,) per-edge lower bound
    w_max: np.ndarray           # (E,) per-edge upper bound
    eta: float = DEFAULT_ETA
    lam: float = DEFAULT_LAMBDA
    # The Edge objects in graph order — used to write back updated weights.
    edges: list[Edge] = field(default_factory=list)
    # Initial weights, for diagnostics (delta over time).
    weights_initial: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.weights_initial = self.weights.copy()

    @property
    def n_edges(self) -> int:
        return int(self.pre_idx.size)

    @classmethod
    def from_graph(
        cls,
        graph: NeuralGraph,
        *,
        plastic_pre_neurons: Iterable[str] = DEFAULT_PLASTIC_PRE_NEURONS,
        eta: float = DEFAULT_ETA,
        lam: float = DEFAULT_LAMBDA,
        max_edges: int = DEFAULT_MAX_PLASTIC_EDGES,
    ) -> "HebbianRule":
        """Promote the top ``max_edges`` outgoing chemical edges of every
        neuron in ``plastic_pre_neurons`` to plastic, sort by weight.

        Limits the count to ``max_edges`` to keep the per-tick cost
        bounded and to honor the design's "50-100 plastic edges"
        guidance (§6.1).
        """
        candidate_edges: list[tuple[float, Edge]] = []
        for pre_name in plastic_pre_neurons:
            if not graph.has_node(pre_name):
                continue
            for e in graph.out_edges(pre_name, "chemical"):
                candidate_edges.append((e.weight, e))
        # Sort by weight descending → keep the strongest connections.
        candidate_edges.sort(key=lambda t: -t[0])
        kept = [e for _, e in candidate_edges[:max_edges]]

        for e in kept:
            e.is_plastic = True
            e.plasticity_rule = "hebbian"

        pre_idx = np.array(
            [graph.index_of(e.source) for e in kept], dtype=np.int64
        )
        post_idx = np.array(
            [graph.index_of(e.target) for e in kept], dtype=np.int64
        )
        signs = np.array([e.sign for e in kept], dtype=np.float64)
        weights = np.array([e.weight for e in kept], dtype=np.float64)
        w_min = np.array([e.w_min for e in kept], dtype=np.float64)
        w_max = np.array([e.w_max for e in kept], dtype=np.float64)

        return cls(
            pre_idx=pre_idx, post_idx=post_idx, signs=signs,
            weights=weights, w_min=w_min, w_max=w_max,
            eta=eta, lam=lam, edges=kept,
        )

    # ---------------------------------------------------------- update

    def step(self, rate: np.ndarray) -> None:
        """Apply one Hebbian tick using the leaky-rate trace."""
        if self.n_edges == 0:
            return
        pre = rate[self.pre_idx]
        post = rate[self.post_idx]
        dW = self.eta * pre * post - self.lam * self.weights
        np.add(self.weights, dW, out=self.weights)
        np.clip(self.weights, self.w_min, self.w_max, out=self.weights)

    def write_back_to_graph(self) -> None:
        """Push current magnitudes onto the underlying Edge objects.

        Useful for diagnostics and for re-building delay buckets after
        the run. Note: the runner caches per-delay W matrices at
        construction time, so plasticity updates do NOT take effect in
        the running simulator unless the runner pulls fresh matrices.
        See ``GraphSimulator.refresh_chemical_matrices()``.
        """
        for e, w in zip(self.edges, self.weights):
            e.weight = float(w)

    # ----------------------------------------------------- diagnostics

    def weight_summary(self) -> dict[str, float]:
        if self.n_edges == 0:
            return {"n_edges": 0}
        delta = self.weights - self.weights_initial
        return {
            "n_edges": self.n_edges,
            "weight_mean":    float(self.weights.mean()),
            "weight_min":     float(self.weights.min()),
            "weight_max":     float(self.weights.max()),
            "delta_mean":     float(delta.mean()),
            "delta_max_abs":  float(np.max(np.abs(delta))),
            "n_grew":         int((delta > 0).sum()),
            "n_shrank":       int((delta < 0).sum()),
        }


__all__ = [
    "HebbianRule",
    "DEFAULT_PLASTIC_PRE_NEURONS",
    "DEFAULT_ETA",
    "DEFAULT_LAMBDA",
    "DEFAULT_MAX_PLASTIC_EDGES",
]
