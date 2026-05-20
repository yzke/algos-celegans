"""Node (neuron) data structure for Phase 1.0 graph-native neural system.

A `Node` carries the per-neuron *static* identity (`id`, `category`,
`neurotransmitter`, `is_plastic`), the *dynamic* state used by the LIF
dynamics (`V`, `refractory`, `last_spike_tick`), and the per-neuron
*processing parameters* (`threshold`, `tau`, `v_reset`, `f_name`).

These objects are attached to NetworkX graph nodes via
``G.add_node(id, node=Node(...))`` — see ``graph.NeuralGraph``.

Design ref: docs/phase1_design.md §1.2 and §2.1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Default LIF parameters. The 5–50 tick range for tau and a unit-magnitude
# threshold sit inside the "neuron tau 5–50 tick" guidance from §8.1.
DEFAULT_THRESHOLD: float = 1.0
DEFAULT_TAU: float = 10.0
DEFAULT_V_RESET: float = 0.0
DEFAULT_V_INIT: float = 0.0
DEFAULT_REFRACTORY_TICKS: int = 2

# Per-category tau/threshold defaults. Pharyngeal CPG-style neurons are
# slightly snappier; modulator neurons are slow integrators. These numbers
# are deliberately mild — we want the default behavior to match the
# Phase 0 homogeneous baseline closely, then let category/subgraph rules
# add heterogeneity on top.
CATEGORY_PARAM_DEFAULTS: dict[str, dict[str, float]] = {
    "sensory":       {"tau": 6.0,  "threshold": 0.8},
    "interneuron":   {"tau": 12.0, "threshold": 1.0},
    "motor":         {"tau": 8.0,  "threshold": 0.9},
    "pharyngeal":    {"tau": 5.0,  "threshold": 0.8},
    "modulator":     {"tau": 40.0, "threshold": 1.2},
    "other_neuron":  {"tau": 10.0, "threshold": 1.0},
    "sex_specific":  {"tau": 10.0, "threshold": 1.0},
}


@dataclass
class Node:
    """A single neuron in the graph.

    The Node is the *authoritative* per-neuron container. Dynamics code
    pulls Node fields into numpy arrays once at the start of each frame
    (for vectorized subgraph computation) and writes the results back
    into the Node at the end of the frame (§5 in the design doc).
    """

    # ---- Static identity ------------------------------------------------
    id: str
    category: str                       # 'sensory' | 'interneuron' | 'motor' | ...
    neurotransmitter: str               # 'GABA' | 'default' | 'neuropeptide' | ...
    is_plastic: bool = False            # whether this node's *outgoing* edges
                                        # are eligible to be plastic by default
    is_modulator: bool = False          # if True, output feeds modulator pool

    # ---- Dynamic state --------------------------------------------------
    V: float = DEFAULT_V_INIT
    refractory: int = 0                 # ticks remaining before next spike allowed
    last_spike_tick: int = -1_000_000   # for diagnostics / STDP later

    # ---- Processing parameters -----------------------------------------
    # All of these are *frozen* by default; specific nodes can be promoted
    # to plastic per the design (§6.1).
    threshold: float = DEFAULT_THRESHOLD
    tau: float = DEFAULT_TAU
    v_reset: float = DEFAULT_V_RESET
    refractory_ticks: int = DEFAULT_REFRACTORY_TICKS
    f_name: str = "lif_standard"        # registry key in neural_v2.step_library

    # Bag for per-node bookkeeping the dynamics might need (e.g. running
    # average for adaptation, last sensory input, etc.). Not used by the
    # baseline LIF step.
    internal_state: dict[str, Any] = field(default_factory=dict)

    # ---- Construction helpers ------------------------------------------

    @classmethod
    def from_connectome_row(
        cls,
        name: str,
        category: str,
        neurotransmitter: str,
        *,
        is_modulator: bool = False,
        is_plastic: bool = False,
    ) -> "Node":
        """Construct a Node using category-default LIF parameters."""
        params = CATEGORY_PARAM_DEFAULTS.get(
            category, CATEGORY_PARAM_DEFAULTS["interneuron"]
        )
        # Modulator nodes get the modulator default regardless of category.
        if is_modulator:
            params = CATEGORY_PARAM_DEFAULTS["modulator"]
        return cls(
            id=name,
            category=category,
            neurotransmitter=neurotransmitter,
            is_plastic=is_plastic,
            is_modulator=is_modulator,
            threshold=params["threshold"],
            tau=params["tau"],
        )

    def reset_state(self, V: float = DEFAULT_V_INIT) -> None:
        """Zero dynamic state. Static identity and parameters untouched."""
        self.V = V
        self.refractory = 0
        self.last_spike_tick = -1_000_000
        self.internal_state.clear()


__all__ = [
    "Node",
    "CATEGORY_PARAM_DEFAULTS",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TAU",
    "DEFAULT_V_RESET",
    "DEFAULT_V_INIT",
    "DEFAULT_REFRACTORY_TICKS",
]
