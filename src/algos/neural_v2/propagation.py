"""Event-driven signal propagation queue for Phase 1.0.

Per docs/phase1_design.md §3.1 and §5:

  当源节点 i 在时刻 t 发出 spike:
    在时刻 (t + delay[i,j]) 信号到达节点 j
    到达时的信号值 = sign[i,j] × weight[i,j]
    这个值被加入节点 j 的 incoming_signals 队列

The queue is implemented as a ring buffer of N-vectors indexed by
``tick mod (max_delay + 1)``. When a spike fires, the delivery is
scattered into the appropriate future slot by adding
``signed_weight * spike_indicator`` to that row.

For uniform-delay edges (Phase 1.0.2's default, delay = 1) the implementation
collapses to a single next-tick matvec. The class still exposes the
ring-buffer interface so per-edge delays in Phase 1.0.4+ slot in
without API changes.

Gap junctions are NOT routed through the queue (their delay = 0 and
they couple V directly, §3.2). The runner adds them at every tick
through ``W_gap @ V - V * W_gap.sum(axis=1)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class DelayBucket:
    """Sparse chemical matrix grouped by edge delay.

    For each unique edge delay ``d`` in the graph, we cache a dense
    (N × N) NumPy array ``W_d`` such that
    ``W_d[i, j] = signed_weight`` if the edge (j → i) has delay ``d``,
    else 0. Dispatch is just ``W_d @ spike_vec``.

    For 302 neurons a dense (302×302) per delay bucket is ~1 MB —
    trivial. If the per-edge delay distribution grows wide we can swap
    in scipy.sparse later.
    """

    delay: int
    W: np.ndarray  # (N, N) signed weights for this delay only


@dataclass
class SignalQueue:
    """Ring-buffer of arriving chemical signals.

    Construct via ``SignalQueue.from_delay_buckets(buckets, N)``.
    """

    n_neurons: int
    max_delay: int
    # Ring buffer: rows indexed by (tick + d) mod (max_delay + 1).
    _buf: np.ndarray = field(init=False)
    _tick: int = field(default=0, init=False)
    # Pre-grouped per-delay W blocks for fast dispatch.
    buckets: list[DelayBucket] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._buf = np.zeros(
            (self.max_delay + 1, self.n_neurons), dtype=np.float64
        )

    # -------------------------------------------------------------- build

    @classmethod
    def from_delay_buckets(
        cls,
        buckets: Iterable[DelayBucket],
        n_neurons: int,
    ) -> "SignalQueue":
        blist = list(buckets)
        if not blist:
            # No chemical edges — degenerate case. Build a length-1 buffer.
            return cls(n_neurons=n_neurons, max_delay=0, buckets=[])
        max_d = max(b.delay for b in blist)
        # Guard against pathological delays.
        if max_d > 1000:
            raise ValueError(
                f"max chemical delay {max_d} exceeds 1000 ticks — sanity check"
            )
        return cls(n_neurons=n_neurons, max_delay=max_d, buckets=blist)

    # ---------------------------------------------------------- runtime

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def buffer_size(self) -> int:
        return self.max_delay + 1

    def arrivals(self) -> np.ndarray:
        """Read (and consume) the signals arriving at the current tick.

        Returns the (N,) vector of arrivals; the slot is then zeroed for
        reuse on the next pass through the ring.
        """
        slot = self._tick % self.buffer_size
        arr = self._buf[slot].copy()
        self._buf[slot] = 0.0
        return arr

    def schedule(self, spike_mask: np.ndarray) -> None:
        """Schedule outgoing signals from a fresh spike pattern.

        Args:
            spike_mask: (N,) boolean or 0/1 vector of who spiked this tick.

        For each delay bucket d, add ``W_d @ spike_mask`` to the slot at
        ``(tick + d) mod buffer_size``. ``W_d`` is post×pre so the
        natural matvec lands the signed contribution at the correct
        post-synaptic row.
        """
        sv = spike_mask.astype(np.float64)
        if not self.buckets:
            return
        for bucket in self.buckets:
            target_slot = (self._tick + bucket.delay) % self.buffer_size
            self._buf[target_slot] += bucket.W @ sv

    def advance(self) -> None:
        """Move to the next tick. Call AFTER arrivals + schedule for tick t."""
        self._tick += 1

    def reset(self) -> None:
        """Zero the buffer and the tick counter."""
        self._buf[:] = 0.0
        self._tick = 0


# ---------------------------------------------------------------------------
# Helpers — turn a NeuralGraph into delay buckets + a gap matrix.
# ---------------------------------------------------------------------------


def build_delay_buckets(graph) -> list[DelayBucket]:
    """Group every chemical edge by its delay and return dense (N,N) buckets."""
    n = graph.n_nodes
    by_delay: dict[int, np.ndarray] = {}
    for e in graph.edges("chemical"):
        d = e.delay
        if d not in by_delay:
            by_delay[d] = np.zeros((n, n), dtype=np.float64)
        i_post = graph.index_of(e.target)
        j_pre = graph.index_of(e.source)
        by_delay[d][i_post, j_pre] += e.signed_weight
    return [DelayBucket(delay=d, W=W) for d, W in sorted(by_delay.items())]


def build_gap_matrix(graph) -> np.ndarray:
    """Materialize the symmetric, non-negative gap-junction matrix (N,N).

    Each electrical edge contributes its weight at (post, pre); since the
    loader mirrors both directions, the result is automatically symmetric.
    """
    n = graph.n_nodes
    W_gap = np.zeros((n, n), dtype=np.float64)
    for e in graph.edges("electrical"):
        i = graph.index_of(e.target)
        j = graph.index_of(e.source)
        W_gap[i, j] = e.weight  # gap weights are always +
    # Defensive symmetry (mirror loader should already guarantee this).
    return np.maximum(W_gap, W_gap.T)


__all__ = [
    "DelayBucket",
    "SignalQueue",
    "build_delay_buckets",
    "build_gap_matrix",
]
