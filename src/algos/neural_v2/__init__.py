"""Phase 1.0 graph-native neural runtime.

Parallel to ``algos.neural`` (Phase 0 CTRNN). The two coexist; Phase 1.0
work targets ``neural_v2`` exclusively. The Phase 0 path stays for
backwards-compatible comparison reports.
"""

from algos.neural_v2.dynamics import LIFParams, lif_step
from algos.neural_v2.modulators import (
    DEFAULT_TAU_M,
    Modulator,
    ModulatorBank,
    build_default_modulator_bank,
)
from algos.neural_v2.plasticity import (
    DEFAULT_ETA,
    DEFAULT_LAMBDA,
    DEFAULT_MAX_PLASTIC_EDGES,
    DEFAULT_PLASTIC_PRE_NEURONS,
    HebbianRule,
)
from algos.neural_v2.propagation import (
    DelayBucket,
    SignalQueue,
    build_delay_buckets,
    build_gap_matrix,
)
from algos.neural_v2.runner import (
    DEFAULT_TAU_RATE,
    GraphSimulator,
    SimulatorConfig,
    spike_mask_from_state,
)
from algos.neural_v2.state import GraphNeuralState

__all__ = [
    "LIFParams",
    "lif_step",
    "DelayBucket",
    "SignalQueue",
    "build_delay_buckets",
    "build_gap_matrix",
    "GraphSimulator",
    "SimulatorConfig",
    "GraphNeuralState",
    "DEFAULT_TAU_RATE",
    "spike_mask_from_state",
    "HebbianRule",
    "DEFAULT_ETA",
    "DEFAULT_LAMBDA",
    "DEFAULT_MAX_PLASTIC_EDGES",
    "DEFAULT_PLASTIC_PRE_NEURONS",
    "Modulator",
    "ModulatorBank",
    "build_default_modulator_bank",
    "DEFAULT_TAU_M",
]
