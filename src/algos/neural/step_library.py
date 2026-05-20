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
# Phase 0.8.3 — specialized step functions for known key neurons
# ---------------------------------------------------------------------------
#
# These are *minimal mathematical sketches*, not biologically-faithful
# models. Each captures one computational property reported in the C.
# elegans literature for the relevant neuron class:
#
#   - change_detector (ASE)         — respond to input derivative
#   - setpoint_deviation (AFD)      — respond to |input − setpoint|
#   - threshold_accumulator (AVA)   — integrate; latch above threshold
#   - bistable_switch (RIM)         — push toward ±1 attractor
#
# Per-neuron params (polarity, setpoint, gain, threshold, persistence,
# self_gain) are stored as length-N ndarrays in HeterogeneousNetwork's
# `function_params`. Other step functions ignore these extra keys.


def change_detector(V_current, total_input, V_history, params):
    """Respond to mismatch between input and current state — derivative-like.

    The signal `polarity · (total_input − V_current)` is large when the
    input is changing relative to what V has caught up to. Saturated by
    tanh and integrated with `tau`. Used for ASE pair:
      - ASEL: polarity = +1 (responds to rising drive)
      - ASER: polarity = −1 (responds to falling drive)
    """
    gain = params.get("gain", np.ones_like(V_current) * 5.0)
    polarity = params["polarity"]
    signal = polarity * (total_input - V_current) * gain
    target = np.tanh(signal)
    dV = (target - V_current) / params["tau"]
    return np.clip(V_current + dV, -1.0, 1.0)


def setpoint_deviation(V_current, total_input, V_history, params):
    """Respond to deviation from a setpoint (AFD pair).

    Target = tanh(gain · polarity · (input − setpoint)). Used for AFD,
    where the "setpoint" is the cultivation temperature. In the bare
    network we set the setpoint to 0 so it functions as a signed
    deviation detector.
    """
    setpoint = params["setpoint"]
    polarity = params["polarity"]
    gain = params.get("gain", np.ones_like(V_current) * 5.0)
    target = np.tanh(gain * polarity * (total_input - setpoint))
    dV = (target - V_current) / params["tau"]
    return np.clip(V_current + dV, -1.0, 1.0)


def threshold_accumulator(V_current, total_input, V_history, params):
    """Leaky integrator with a soft latch above a threshold (AVA/AVD/AVB).

    Once V_current exceeds `threshold`, a positive `persistence` term is
    added to the drive that keeps V elevated. Below threshold the unit
    behaves like a CTRNN. This produces "command-neuron-like" behavior:
    inputs accumulate, threshold crossing produces a persistent active
    state.
    """
    threshold = params["threshold"]
    persistence = params["persistence"]
    latched = (V_current > threshold).astype(np.float64) * persistence
    dV = (-V_current + total_input + latched) / params["tau"]
    return np.clip(V_current + dV, -1.0, 1.0)


def bistable_switch(V_current, total_input, V_history, params):
    """Bistable unit with attractors near ±1 (RIM).

    Adds a self-amplifying term `self_gain · tanh(3·V_current)` to the
    drive, biasing V toward whichever sign it's currently in. Strong
    input can flip the state; weak input lets the unit stay in its
    current attractor.
    """
    self_gain = params["self_gain"]
    drive = total_input + self_gain * np.tanh(3.0 * V_current)
    dV = (-V_current + drive) / params["tau"]
    return np.clip(V_current + dV, -1.0, 1.0)


register_step_function("change_detector", change_detector)
register_step_function("setpoint_deviation", setpoint_deviation)
register_step_function("threshold_accumulator", threshold_accumulator)
register_step_function("bistable_switch", bistable_switch)


# ---------------------------------------------------------------------------
# Key-neuron specialization factory
# ---------------------------------------------------------------------------


# Per-neuron specialization table. Keys: neuron name. Value: (function_name,
# params dict). Params are merged into the network's length-N arrays at
# the neuron's index.
KEY_NEURON_SPECIALIZATIONS: dict[str, tuple[str, dict[str, float]]] = {
    # ASE pair — chemosensory, derivative-like response.
    "ASEL": ("change_detector",      {"tau":  8.0, "polarity": +1.0, "gain": 5.0}),
    "ASER": ("change_detector",      {"tau":  8.0, "polarity": -1.0, "gain": 5.0}),
    # AFD pair — thermosensory setpoint deviation.
    "AFDL": ("setpoint_deviation",   {"tau":  8.0, "polarity": +1.0,
                                       "setpoint": 0.0, "gain": 5.0}),
    "AFDR": ("setpoint_deviation",   {"tau":  8.0, "polarity": -1.0,
                                       "setpoint": 0.0, "gain": 5.0}),
    # Backward command pool — threshold accumulator with persistence.
    "AVAL": ("threshold_accumulator", {"tau": 25.0, "threshold": 0.20,
                                        "persistence": 0.30}),
    "AVAR": ("threshold_accumulator", {"tau": 25.0, "threshold": 0.20,
                                        "persistence": 0.30}),
    "AVDL": ("threshold_accumulator", {"tau": 25.0, "threshold": 0.15,
                                        "persistence": 0.25}),
    "AVDR": ("threshold_accumulator", {"tau": 25.0, "threshold": 0.15,
                                        "persistence": 0.25}),
    # Forward command pool.
    "AVBL": ("threshold_accumulator", {"tau": 25.0, "threshold": 0.20,
                                        "persistence": 0.30}),
    "AVBR": ("threshold_accumulator", {"tau": 25.0, "threshold": 0.20,
                                        "persistence": 0.30}),
    "PVCL": ("threshold_accumulator", {"tau": 25.0, "threshold": 0.15,
                                        "persistence": 0.25}),
    "PVCR": ("threshold_accumulator", {"tau": 25.0, "threshold": 0.15,
                                        "persistence": 0.25}),
    # RIM — bistable state switch.
    "RIML": ("bistable_switch",      {"tau": 20.0, "self_gain": 0.40}),
    "RIMR": ("bistable_switch",      {"tau": 20.0, "self_gain": 0.40}),
}


def from_key_neuron_specialization(
    connectome: ConnectomeData,
    *,
    base_category_assignment: dict | None = None,
    specializations: dict[str, tuple[str, dict[str, float]]] | None = None,
) -> HeterogeneousNetwork:
    """Build a network using category defaults + per-neuron overrides.

    Starts with `from_category_defaults` and overrides the assignment +
    params for any neuron in `specializations` (default =
    `KEY_NEURON_SPECIALIZATIONS`).
    """
    if specializations is None:
        specializations = KEY_NEURON_SPECIALIZATIONS

    # Begin with the category-default network, then mutate its lists/arrays.
    base = from_category_defaults(
        connectome, category_assignment=base_category_assignment
    )
    n = connectome.n_neurons
    assignment = list(base.function_assignment)
    params: dict[str, np.ndarray] = {
        k: arr.copy() for k, arr in base.function_params.items()
    }

    # Initialize new param keys to default values (length-N ndarrays).
    extra_param_defaults = {
        "polarity": 1.0,
        "setpoint": 0.0,
        "gain": 5.0,
        "threshold": 0.2,
        "persistence": 0.3,
        "self_gain": 0.4,
    }
    for key, default in extra_param_defaults.items():
        if key not in params:
            params[key] = np.full(n, default, dtype=np.float64)

    # Apply each specialization.
    for neuron_name, (func_name, p) in specializations.items():
        if neuron_name not in connectome.neuron_to_idx:
            continue  # Silently skip neurons absent from this connectome.
        idx = connectome.idx(neuron_name)
        assignment[idx] = func_name
        for key, val in p.items():
            if key not in params:
                params[key] = np.full(n, val, dtype=np.float64)
            else:
                params[key][idx] = val

    return HeterogeneousNetwork(
        connectome=connectome,
        function_assignment=assignment,
        function_params=params,
    )


__all__ = [
    "fast_filter",
    "integrator",
    "slow_persistent",
    "change_detector",
    "setpoint_deviation",
    "threshold_accumulator",
    "bistable_switch",
    "DEFAULT_CATEGORY_ASSIGNMENT",
    "KEY_NEURON_SPECIALIZATIONS",
    "from_category_defaults",
    "from_key_neuron_specialization",
]


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
