"""Phase 1.0 — graph-native neural-system primitives.

The package exposes four core types:

  * Node              — neuron, with static identity + dynamic state +
                        processing parameters.
  * Edge              — chemical / electrical / modulatory connection.
  * NeuralGraph       — first-class graph container (nx.MultiDiGraph
                        underneath), with subgraph registry and the
                        canonical name→index mapping.
  * Subgraph          — named view onto a subset of nodes + edges.

Plus the connectome loader:

  * load_connectome_into_graph(connectome=None) → NeuralGraph
"""

from algos.graph.circuits import (
    CIRCUIT_SPECS,
    CircuitSpec,
    build_canonical_subgraphs,
    summarize_subgraphs,
)
from algos.graph.edge import (
    DEFAULT_CHEMICAL_DELAY,
    DEFAULT_ELECTRICAL_DELAY,
    Edge,
    edge_key_for_type,
)
from algos.graph.graph import NeuralGraph
from algos.graph.loader import (
    DEFAULT_MODULATOR_NEURONS,
    load_connectome_into_graph,
)
from algos.graph.node import (
    CATEGORY_PARAM_DEFAULTS,
    DEFAULT_TAU,
    DEFAULT_THRESHOLD,
    DEFAULT_V_INIT,
    DEFAULT_V_RESET,
    Node,
)
from algos.graph.subgraph import Subgraph, VALID_SUBGRAPH_TYPES

__all__ = [
    "Node",
    "Edge",
    "NeuralGraph",
    "Subgraph",
    "load_connectome_into_graph",
    "DEFAULT_MODULATOR_NEURONS",
    "DEFAULT_CHEMICAL_DELAY",
    "DEFAULT_ELECTRICAL_DELAY",
    "DEFAULT_TAU",
    "DEFAULT_THRESHOLD",
    "DEFAULT_V_INIT",
    "DEFAULT_V_RESET",
    "CATEGORY_PARAM_DEFAULTS",
    "VALID_SUBGRAPH_TYPES",
    "edge_key_for_type",
    "CircuitSpec",
    "CIRCUIT_SPECS",
    "build_canonical_subgraphs",
    "summarize_subgraphs",
]
