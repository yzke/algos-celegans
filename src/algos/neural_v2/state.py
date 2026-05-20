"""Per-tick state container for the graph-native LIF runner.

Lightweight: just the arrays that change every tick. The persistent
identity (Node attrs, parameters) lives on the NeuralGraph; this
struct is what the inner loop touches.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GraphNeuralState:
    """V, refractory, spike trace, and tick counter — vectorized over N."""

    V: np.ndarray                   # (N,) membrane potential
    refractory: np.ndarray          # (N,) int — ticks remaining post-spike
    rate: np.ndarray                # (N,) leaky-integrator readout of spikes;
                                    # the closest analog to GCaMP/ΔF.
    tick: int = 0
    # Cumulative spike count per neuron — handy for sanity checks.
    spike_count: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    # Optional last-spike-tick (for STDP, debugging).
    last_spike_tick: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    def __post_init__(self) -> None:
        n = self.V.shape[0]
        if self.refractory.shape != (n,):
            raise ValueError("V and refractory shape mismatch")
        if self.rate.shape != (n,):
            raise ValueError("V and rate shape mismatch")
        if self.spike_count.size == 0:
            self.spike_count = np.zeros(n, dtype=np.int64)
        if self.last_spike_tick.size == 0:
            self.last_spike_tick = np.full(n, -1_000_000, dtype=np.int64)

    @classmethod
    def zeros(cls, n: int) -> "GraphNeuralState":
        return cls(
            V=np.zeros(n, dtype=np.float64),
            refractory=np.zeros(n, dtype=np.int64),
            rate=np.zeros(n, dtype=np.float64),
        )

    @classmethod
    def initial(
        cls, n: int, *, seed: int | None = None, spread: float = 0.01
    ) -> "GraphNeuralState":
        """Initialise with small uniform noise on V to break symmetry."""
        rng = np.random.default_rng(seed)
        V0 = rng.uniform(-spread, spread, n).astype(np.float64)
        return cls(
            V=V0,
            refractory=np.zeros(n, dtype=np.int64),
            rate=np.zeros(n, dtype=np.float64),
        )


__all__ = ["GraphNeuralState"]
