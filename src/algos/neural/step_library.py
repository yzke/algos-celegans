"""Phase 0.8.2 — category-default heterogeneous step functions.

Three new step functions and a factory that assigns them by neuron category.
The three functions are genuinely distinct in their computational form
(not just different tau values):

  - `fast_filter` — relaxation toward a *saturated* target
    `tanh(beta · total_input)`. Sharp, fast response. Sensory default.
  - `integrator` — pure leaky integration `dV = (-V + total_input)/tau`.
    No additional nonlinearity at the unit. Interneuron default.
  - `slow_persistent` — ctrnn-like leak plus a *momentum* term that
    carries the recent change in V forward (`+0.2 · (V[t-1] − V[t-2])`).
    Slow, with state inertia. Motor default.

All three honor the (V_current, total_input, V_history, params) contract
and return a clipped V_new in [-1, 1]. Registration into STEP_LIBRARY
happens at import time.

The category-default factory is `from_category_defaults(connectome)`. It
returns a `HeterogeneousNetwork` whose `function_assignment` follows
`DEFAULT_CATEGORY_ASSIGNMENT` (sensory → fast_filter etc.), with per-neuron
`tau` (and `beta`) arrays built from the same table.
"""

from __future__ import annotations

import numpy as np

from algos.connectome import ConnectomeData
from algos.neural.heterogeneous import (
    HeterogeneousNetwork,
    register_step_function,
)


# ---------------------------------------------------------------------------
# Three new step functions
# ---------------------------------------------------------------------------


def fast_filter(V_current, total_input, V_history, params):
    """Relaxation toward a saturated target — sensory default.

    Computes a "target" by passing the input through a sharp tanh
    (`tanh(beta · total_input)`) and relaxes V toward that target on a
    short timescale `tau`. Because the target is already saturated in
    [-1, 1], the unit responds quickly and clamps fast inputs.
    """
    target = np.tanh(params["beta"] * total_input)
    dV = (target - V_current) / params["tau"]
    return np.clip(V_current + dV, -1.0, 1.0)


def integrator(V_current, total_input, V_history, params):
    """Pure leaky integration — interneuron default.

    Same form as `ctrnn_default` (`dV = (-V + input)/tau`) but provided
    as a separate function so the dispatch can route interneurons here
    explicitly. With tau larger than the sensory tau, this unit
    smooths/averages its drive over time.
    """
    dV = (-V_current + total_input) / params["tau"]
    return np.clip(V_current + dV, -1.0, 1.0)


def slow_persistent(V_current, total_input, V_history, params):
    """CTRNN-like leak + momentum — motor default.

    Adds a momentum term `+momentum · (V[t-1] − V[t-2])` that propagates
    the recent change in V forward. This encodes persistence: motor
    commands tend to continue once started. Momentum coefficient is
    fixed at 0.2 (small enough to keep tau-driven dynamics dominant).
    """
    if V_history.shape[0] >= 2:
        momentum_term = 0.2 * (V_history[-1] - V_history[-2])
    else:
        momentum_term = 0.0
    dV = (-V_current + total_input) / params["tau"] + momentum_term
    return np.clip(V_current + dV, -1.0, 1.0)


# Register at import time so any importer immediately gets them.
register_step_function("fast_filter", fast_filter)
register_step_function("integrator", integrator)
register_step_function("slow_persistent", slow_persistent)


# ---------------------------------------------------------------------------
# Category → step function defaults
# ---------------------------------------------------------------------------


# (function_name, params_dict) per category. Params are merged into the
# network's length-N params arrays; categories with the same param key share
# the same array slot but different values.
DEFAULT_CATEGORY_ASSIGNMENT: dict[str, tuple[str, dict[str, float]]] = {
    "sensory":      ("fast_filter",     {"tau":  5.0, "beta": 5.0}),
    "interneuron":  ("integrator",      {"tau": 20.0, "beta": 1.0}),
    "motor":        ("slow_persistent", {"tau": 50.0, "beta": 1.0}),
    "pharyngeal":   ("ctrnn_default",   {"tau": 15.0, "beta": 1.0}),
    "sex_specific": ("ctrnn_default",   {"tau": 20.0, "beta": 1.0}),
    "other_neuron": ("ctrnn_default",   {"tau": 20.0, "beta": 1.0}),
}


def from_category_defaults(
    connectome: ConnectomeData,
    *,
    category_assignment: dict[str, tuple[str, dict[str, float]]] | None = None,
    fallback: tuple[str, dict[str, float]] = ("ctrnn_default", {"tau": 10.0, "beta": 1.0}),
) -> HeterogeneousNetwork:
    """Build a HeterogeneousNetwork with per-category step function defaults.

    Args:
        connectome: loaded `ConnectomeData`.
        category_assignment: mapping category → (function_name, params dict).
            Defaults to `DEFAULT_CATEGORY_ASSIGNMENT`.
        fallback: assignment for categories not in the table.

    Returns:
        A `HeterogeneousNetwork` whose every neuron uses the function and
        params dictated by its category in the assignment table.
    """
    if category_assignment is None:
        category_assignment = DEFAULT_CATEGORY_ASSIGNMENT

    n = connectome.n_neurons
    assignment: list[str] = []
    tau_arr = np.empty(n, dtype=np.float64)
    beta_arr = np.empty(n, dtype=np.float64)
    for i, cat in enumerate(connectome.category):
        func_name, p = category_assignment.get(cat, fallback)
        assignment.append(func_name)
        tau_arr[i] = p.get("tau", 10.0)
        beta_arr[i] = p.get("beta", 1.0)
    return HeterogeneousNetwork(
        connectome=connectome,
        function_assignment=assignment,
        function_params={"tau": tau_arr, "beta": beta_arr},
    )


__all__ = [
    "fast_filter",
    "integrator",
    "slow_persistent",
    "DEFAULT_CATEGORY_ASSIGNMENT",
    "from_category_defaults",
]
