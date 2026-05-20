"""CTRNN dynamics and neural state.

Phase 0.7 and earlier: homogeneous CTRNN — every neuron uses the same
step function, exposed via `neural_step(state, connectome, sensory, params, rng)`.

Phase 0.8.1 onwards: a parallel heterogeneous architecture is available in
`algos.neural.heterogeneous`; the two paths coexist. When every neuron uses
the `ctrnn_default` step function in the heterogeneous network, results
match `neural_step` numerically (< 1e-6 over 100-tick runs).
"""

from algos.neural.dynamics import CTRNNParams, neural_step, sigmoid
from algos.neural.heterogeneous import (
    STEP_LIBRARY,
    HeterogeneousNetwork,
    HeterogeneousState,
    StepFunction,
    ctrnn_default,
    register_step_function,
)
from algos.neural.state import NeuralState
# Side-effect import: registers Phase 0.8.2 step functions into STEP_LIBRARY.
from algos.neural import step_library as _step_library  # noqa: F401
from algos.neural.step_library import (
    DEFAULT_CATEGORY_ASSIGNMENT,
    fast_filter,
    from_category_defaults,
    integrator,
    slow_persistent,
)

__all__ = [
    "CTRNNParams",
    "NeuralState",
    "neural_step",
    "sigmoid",
    "HeterogeneousNetwork",
    "HeterogeneousState",
    "STEP_LIBRARY",
    "StepFunction",
    "ctrnn_default",
    "register_step_function",
    "fast_filter",
    "integrator",
    "slow_persistent",
    "DEFAULT_CATEGORY_ASSIGNMENT",
    "from_category_defaults",
]
