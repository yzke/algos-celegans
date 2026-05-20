"""Heterogeneous-step-function neural architecture — Phase 0.8.1.

Generalizes the Phase 0.7 single-step `neural_step` into a per-neuron
step-function dispatch. The connectome matrices (W_chem, W_gap) are
preserved unchanged; what changes is the per-neuron "what do I compute
at each tick" definition.

Architecture:

  - `STEP_LIBRARY` — a registry mapping step-function name → callable.
    The default function `ctrnn_default` reproduces Phase 0.7 behavior
    exactly.
  - `HeterogeneousNetwork` — wraps a `ConnectomeData` plus a per-neuron
    function-name assignment plus per-neuron params. Groups neurons by
    assignment so each function is called once per tick on a vectorized
    sub-array — no Python-level per-neuron loop.
  - `HeterogeneousState` — wraps the last `history_len` ticks of V, plus
    a tick counter. History is a circular-buffer slot (most recent at
    [-1]).

Numerical equivalence with Phase 0.7's `neural_step` is enforced when
every neuron uses `ctrnn_default`; verified by
`tests/test_heterogeneous.py::test_homogeneous_equivalence`.

Step function signature:

    step_func(V_current, total_input, V_history, params) -> V_new
       V_current:   (k,)   current V at the k neurons in this group
       total_input: (k,)   precomputed (chem + gap + sensory + noise) for them
       V_history:   (H, k) recent V trajectory (H = history_len; [-1] = current)
       params:      dict   per-group ndarrays of length k (e.g. {'tau': ...})
       returns:     (k,)   new V at the group neurons (clipped to [-1, 1])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from algos.config import CTRNNDefaults
from algos.connectome import ConnectomeData

StepFunction = Callable[
    [np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]],
    np.ndarray,
]


# ---------------------------------------------------------------------------
# Default step function — Phase 0.7 / v0.3 dynamics, exactly.
# ---------------------------------------------------------------------------


def ctrnn_default(V_current, total_input, V_history, params):
    """Exact Phase 0.7 / v0.3 dynamics, vectorized over a group.

    `V_history` is unused here; this is the pure CTRNN form:
        dV   = (-V + total_input) / tau
        V_new = clip(V + dV, -1, 1)
    """
    dV = (-V_current + total_input) / params["tau"]
    return np.clip(V_current + dV, -1.0, 1.0)


# Registry — Phase 0.8.2 / 0.8.3 functions register themselves into this.
STEP_LIBRARY: dict[str, StepFunction] = {
    "ctrnn_default": ctrnn_default,
}


def register_step_function(name: str, func: StepFunction) -> None:
    """Add a step function to the global library (idempotent)."""
    STEP_LIBRARY[name] = func


# ---------------------------------------------------------------------------
# State + network
# ---------------------------------------------------------------------------


@dataclass
class HeterogeneousState:
    """V history buffer + tick counter.

    `V_history[-1]` is the current (most-recent) V; `V_history[-2]` is one
    tick ago; etc. `V_history[0]` is the oldest tick still in the buffer.
    """

    V_history: np.ndarray
    tick: int = 0

    @property
    def V(self) -> np.ndarray:
        return self.V_history[-1]


@dataclass
class HeterogeneousNetwork:
    """Connectome + per-neuron step-function assignment.

    Build examples:
        # Phase 0.7 equivalent (every neuron uses ctrnn_default).
        net = HeterogeneousNetwork.from_homogeneous_ctrnn(connectome)
        # Per-neuron assignment.
        net = HeterogeneousNetwork(
            connectome=connectome,
            function_assignment=["fast_filter"] * 50 + ["ctrnn_default"] * 252,
            function_params={"tau": np.array([...])},
            function_library=lib,
        )
    """

    connectome: ConnectomeData
    function_assignment: list[str]
    function_params: dict[str, np.ndarray]
    function_library: dict[str, StepFunction] = field(
        default_factory=lambda: dict(STEP_LIBRARY)
    )
    history_len: int = 5
    global_noise_level: float = CTRNNDefaults.noise_level
    global_beta: float = CTRNNDefaults.beta
    function_groups: dict[str, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self):
        n = self.connectome.n_neurons
        if len(self.function_assignment) != n:
            raise ValueError(
                f"function_assignment length {len(self.function_assignment)} "
                f"!= n_neurons {n}"
            )
        unknown = set(self.function_assignment) - set(self.function_library)
        if unknown:
            raise KeyError(
                f"step functions not in library: {sorted(unknown)}"
            )
        for k, arr in self.function_params.items():
            if len(arr) != n:
                raise ValueError(
                    f"function_params['{k}'] length {len(arr)} != n_neurons {n}"
                )
        # Build group → indices map (one fancy-index per group).
        groups: dict[str, list[int]] = {}
        for i, name in enumerate(self.function_assignment):
            groups.setdefault(name, []).append(i)
        self.function_groups = {
            k: np.array(v, dtype=np.int64) for k, v in groups.items()
        }

    # ----------------------------------------------------------------- build

    @classmethod
    def from_homogeneous_ctrnn(
        cls,
        connectome: ConnectomeData,
        *,
        tau: float | None = None,
        beta: float | None = None,
        noise_level: float | None = None,
    ) -> "HeterogeneousNetwork":
        """Build a network where every neuron uses `ctrnn_default`."""
        d = CTRNNDefaults()
        tau = d.tau if tau is None else tau
        beta = d.beta if beta is None else beta
        noise_level = d.noise_level if noise_level is None else noise_level
        n = connectome.n_neurons
        return cls(
            connectome=connectome,
            function_assignment=["ctrnn_default"] * n,
            function_params={"tau": np.full(n, tau, dtype=np.float64)},
            global_noise_level=noise_level,
            global_beta=beta,
        )

    def initial_state(
        self, *, seed: int | None = None, spread: float = 0.01
    ) -> HeterogeneousState:
        """Initialize V to small uniform noise; history filled with V0."""
        rng = np.random.default_rng(seed)
        V0 = rng.uniform(
            -spread, spread, self.connectome.n_neurons
        ).astype(np.float64)
        V_history = np.tile(V0[np.newaxis, :], (self.history_len, 1))
        return HeterogeneousState(V_history=V_history, tick=0)

    # ------------------------------------------------------------------ step

    def step(
        self,
        state: HeterogeneousState,
        sensory_input: np.ndarray,
        rng: np.random.Generator,
        modulator: "RIDModulator | None" = None,
    ) -> HeterogeneousState:
        """Advance one tick. Returns a new HeterogeneousState.

        If `modulator` is provided, its internal state is advanced from the
        current V *before* per-group dispatch and its modulation term is
        subtracted from `total_input` at the target indices. With
        `modulator=None` this is bit-equivalent to Phase 0.8.2.
        """
        V = state.V
        N = V.shape[0]

        # Global pre-computation — shared across all groups.
        chem_input = self.connectome.W_chem @ np.tanh(self.global_beta * V)
        gap_input = (
            self.connectome.W_gap @ V - V * self.connectome.W_gap.sum(axis=1)
        )
        if self.global_noise_level > 0.0:
            noise = rng.standard_normal(N) * self.global_noise_level
        else:
            noise = 0.0
        total_input = chem_input + gap_input + sensory_input + noise

        # Phase 0.9 — apply optional modulator (e.g. RID).
        if modulator is not None:
            modulator.step(V)
            modulator.apply_modulation(total_input)

        # Per-group dispatch.
        V_new = np.empty(N, dtype=np.float64)
        for func_name, indices in self.function_groups.items():
            func = self.function_library[func_name]
            V_new[indices] = func(
                V[indices],
                total_input[indices],
                state.V_history[:, indices],
                {k: arr[indices] for k, arr in self.function_params.items()},
            )

        # Roll history forward.
        new_history = np.empty_like(state.V_history)
        new_history[:-1] = state.V_history[1:]
        new_history[-1] = V_new
        return HeterogeneousState(V_history=new_history, tick=state.tick + 1)


__all__ = [
    "HeterogeneousNetwork",
    "HeterogeneousState",
    "STEP_LIBRARY",
    "StepFunction",
    "ctrnn_default",
    "register_step_function",
]
