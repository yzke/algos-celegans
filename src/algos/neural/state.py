"""Neural state container.

Phase 0 keeps state minimal: a single (N,) activity vector and a tick counter.
Modulator and plasticity state will be added by phases 3 and 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NeuralState:
    """Per-tick neural activity. `V[i]` is bounded to [-1, 1]."""

    V: np.ndarray
    tick: int = 0

    @classmethod
    def initialize(
        cls,
        n_neurons: int,
        *,
        seed: int | None = None,
        spread: float = 0.01,
    ) -> "NeuralState":
        """Initialize with small uniform noise around 0.

        The default `spread=0.01` keeps the system well inside the sigmoid's
        linear regime so the first few ticks don't pin into a saturation
        attractor before the dynamics have a chance to settle.
        """
        rng = np.random.default_rng(seed)
        V = rng.uniform(-spread, spread, n_neurons).astype(np.float64)
        return cls(V=V, tick=0)
