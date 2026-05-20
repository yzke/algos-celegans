"""GraphSimulator — drives the full per-frame loop from docs/phase1_design.md §5.

For each tick:
  1. arrivals = signal_queue.arrivals()
  2. gap_input = W_gap @ V - V * W_gap.sum(axis=1)
  3. total_input = arrivals + gap_input + sensory_input + noise
  4. (V', refr', spike) = lif_step(V, refr, total_input, params)
  5. rate' = rate * (1 - 1/tau_rate) + spike (exponential filter)
  6. signal_queue.schedule(spike)
  7. signal_queue.advance()

The simulator pulls per-neuron parameters (tau, threshold, v_reset,
refractory_ticks) once from the graph's Nodes at construction time and
caches them in NumPy arrays. Parameter mutations on the Nodes are not
re-read mid-simulation; call ``refresh_params()`` to pull them again
(used by Phase 1.0.4 when modulators want to dynamically scale a
threshold).

The simulator is also where Phase 1.0.4 will plug in the Hebbian rule
and modulator concentration updates; for 1.0.2 those hooks are stubbed
out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from algos.graph import NeuralGraph
from algos.neural_v2.dynamics import LIFParams, lif_step
from algos.neural_v2.propagation import (
    DelayBucket,
    SignalQueue,
    build_delay_buckets,
    build_gap_matrix,
)
from algos.neural_v2.state import GraphNeuralState


# Tau for the rate readout — a slow LP filter on spike trains, the
# closest analog to GCaMP. 30 ticks is in the range Phase 0 used.
DEFAULT_TAU_RATE: float = 30.0


@dataclass
class SimulatorConfig:
    """Hyperparameters shared across the run."""

    noise_level: float = 0.01          # σ on per-tick Gaussian input noise
    tau_rate: float = DEFAULT_TAU_RATE
    sensory_noise: float = 0.0         # σ on sensory drive (Phase 1.0.2 baseline)


@dataclass
class GraphSimulator:
    """Stateful runner: graph + cached matrices + queue + per-neuron params."""

    graph: NeuralGraph
    config: SimulatorConfig = field(default_factory=SimulatorConfig)

    # Built in __post_init__ from the graph.
    n: int = field(init=False)
    W_gap: np.ndarray = field(init=False)
    gap_row_sum: np.ndarray = field(init=False)
    queue: SignalQueue = field(init=False)
    params: LIFParams = field(init=False)
    sensory_idx: np.ndarray = field(init=False)
    rate_decay: float = field(init=False)

    def __post_init__(self) -> None:
        self.graph.rebuild_index()
        self.n = self.graph.n_nodes
        # Cache per-neuron parameter arrays in canonical order.
        names = self.graph.neuron_names()
        tau = np.array([self.graph.node(n).tau for n in names], dtype=np.float64)
        thr = np.array(
            [self.graph.node(n).threshold for n in names], dtype=np.float64
        )
        vrst = np.array(
            [self.graph.node(n).v_reset for n in names], dtype=np.float64
        )
        rtk = np.array(
            [self.graph.node(n).refractory_ticks for n in names], dtype=np.int64
        )
        self.params = LIFParams(
            threshold=thr, tau=tau, v_reset=vrst, refractory_ticks=rtk,
        )
        # Matrix views.
        self.W_gap = build_gap_matrix(self.graph)
        self.gap_row_sum = self.W_gap.sum(axis=1)
        self.queue = SignalQueue.from_delay_buckets(
            build_delay_buckets(self.graph), self.n
        )
        # Sensory indices (we drive a random sensory bath each tick).
        sens_names = [
            n for n in names if self.graph.node(n).category == "sensory"
        ]
        self.sensory_idx = np.array(
            [self.graph.index_of(n) for n in sens_names], dtype=np.int64
        )
        self.rate_decay = 1.0 - 1.0 / self.config.tau_rate

    # ------------------------------------------------------------- reset

    def refresh_params(self) -> None:
        """Re-pull per-neuron params from the graph (e.g. after modulation)."""
        names = self.graph.neuron_names()
        self.params = LIFParams(
            threshold=np.array(
                [self.graph.node(n).threshold for n in names], dtype=np.float64
            ),
            tau=np.array([self.graph.node(n).tau for n in names], dtype=np.float64),
            v_reset=np.array(
                [self.graph.node(n).v_reset for n in names], dtype=np.float64
            ),
            refractory_ticks=np.array(
                [self.graph.node(n).refractory_ticks for n in names], dtype=np.int64
            ),
        )

    def initial_state(
        self, *, seed: int | None = None, spread: float = 0.01
    ) -> GraphNeuralState:
        self.queue.reset()
        return GraphNeuralState.initial(self.n, seed=seed, spread=spread)

    # -------------------------------------------------------------- step

    def step(
        self,
        state: GraphNeuralState,
        sensory_input: np.ndarray,
        rng: np.random.Generator,
    ) -> GraphNeuralState:
        """Advance one tick. Returns a new GraphNeuralState."""
        # 1. arrivals from the queue (consumed). Chemical signals are
        #    impulse inputs in V-units — added directly to total_input.
        arrivals = self.queue.arrivals()

        # 2. gap-junction input. Synchronous, instantaneous (§3.2). The
        #    Laplacian's max eigenvalue is ≤ 2*max_row_sum ≈ 2; with the
        #    impulse-form LIF update (V *= (1 - 1/tau) + input), an
        #    unscaled gap term can drive |eig| > 1 on some modes when
        #    tau is small (~8) and that mode's eigenvalue approaches the
        #    Laplacian's upper bound. Scaling by 1/tau treats the gap
        #    junction as a slow continuous conductance (the same way
        #    Phase 0's CTRNN did) and keeps the system contractive
        #    while preserving the diffusion-toward-mean behavior.
        gap_input = self.W_gap @ state.V - state.V * self.gap_row_sum
        gap_input_scaled = gap_input / self.params.tau

        # 3. noise (synaptic-input noise on the input side).
        if self.config.noise_level > 0.0:
            noise = rng.standard_normal(self.n) * self.config.noise_level
        else:
            noise = 0.0

        total_input = arrivals + gap_input_scaled + sensory_input + noise

        # 4. LIF step.
        V_new, refr_new, spike_mask = lif_step(
            state.V, state.refractory, total_input, self.params
        )

        # 5. Schedule outgoing signals and advance the queue.
        self.queue.schedule(spike_mask)
        self.queue.advance()

        # 6. Update the rate readout (leaky integrator on spikes).
        rate_new = state.rate * self.rate_decay + spike_mask.astype(np.float64)

        # 7. Bookkeeping.
        spike_count_new = state.spike_count + spike_mask.astype(np.int64)
        last_spike_new = np.where(
            spike_mask, state.tick, state.last_spike_tick
        ).astype(np.int64)

        return GraphNeuralState(
            V=V_new,
            refractory=refr_new,
            rate=rate_new,
            tick=state.tick + 1,
            spike_count=spike_count_new,
            last_spike_tick=last_spike_new,
        )

    # ----------------------------------------------- helpers for tests

    def run(
        self,
        n_ticks: int,
        *,
        seed: int | None = None,
        sensory_fn=None,
        record_V: bool = True,
        record_rate: bool = True,
        record_spikes: bool = False,
    ) -> dict:
        """Run for ``n_ticks`` and return the recorded traces.

        Args:
            sensory_fn: optional callable (tick, rng) → (N,) sensory vector.
                If None, a zero-mean Gaussian bath of width
                ``config.sensory_noise`` is applied to sensory neurons only.
            record_*: which arrays to capture per tick.
        """
        state = self.initial_state(seed=seed)
        rng = np.random.default_rng(seed)
        V_hist = (
            np.zeros((n_ticks, self.n), dtype=np.float32) if record_V else None
        )
        rate_hist = (
            np.zeros((n_ticks, self.n), dtype=np.float32) if record_rate else None
        )
        spike_hist = (
            np.zeros((n_ticks, self.n), dtype=np.uint8) if record_spikes else None
        )
        sens = np.zeros(self.n, dtype=np.float64)
        for t in range(n_ticks):
            if sensory_fn is None:
                sens[:] = 0.0
                if self.config.sensory_noise > 0:
                    sens[self.sensory_idx] = (
                        rng.standard_normal(self.sensory_idx.size)
                        * self.config.sensory_noise
                    )
            else:
                sens[:] = sensory_fn(t, rng)
            state = self.step(state, sens, rng)
            if V_hist is not None:
                V_hist[t] = state.V
            if rate_hist is not None:
                rate_hist[t] = state.rate
            if spike_hist is not None:
                spike_hist[t] = spike_mask_from_state(state)

        return {
            "V": V_hist,
            "rate": rate_hist,
            "spikes": spike_hist,
            "final_state": state,
            "names": self.graph.neuron_names(),
        }


def spike_mask_from_state(state: GraphNeuralState) -> np.ndarray:
    """Approximate spike mask for this tick (last_spike_tick == tick - 1)."""
    return (state.last_spike_tick == (state.tick - 1)).astype(np.uint8)


__all__ = [
    "GraphSimulator",
    "SimulatorConfig",
    "DEFAULT_TAU_RATE",
    "spike_mask_from_state",
]
