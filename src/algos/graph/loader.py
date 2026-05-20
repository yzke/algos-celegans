"""Load the Cook 2019 connectome into the Phase 1.0 graph representation.

This module is the bridge between the Phase 0 matrix world
(``algos.connectome.ConnectomeData``) and the Phase 1.0 graph world
(``algos.graph.NeuralGraph``). Conventions to remember:

  * Phase 0 ``W_chem[i, j]`` = synapse from j (pre) to i (post), with
    the sign of the *pre* neuron's neurotransmitter baked in.
  * Phase 0 ``W_chem_raw[i, j]`` = the *unsigned* serial-section count
    (same row/col convention).
  * Phase 0 ``W_gap`` is symmetric, non-negative.

In the graph, each chemical synapse becomes ONE directed Edge from
pre → post. Each gap-junction *pair* becomes TWO mirrored Edges (one
per direction). Modulator edges are NOT added by the loader — they
arrive in Phase 1.0.4, alongside the modulator-node promotion.

The graph node ids are the neuron names (e.g. 'AVAL'); the static
metadata (category, neurotransmitter) is copied verbatim from
``ConnectomeData``.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from algos.connectome import ConnectomeData
from algos.graph.edge import (
    DEFAULT_CHEMICAL_DELAY,
    DEFAULT_ELECTRICAL_DELAY,
    Edge,
)
from algos.graph.graph import NeuralGraph
from algos.graph.node import Node


# Neurons we promote to `is_modulator=True` based on docs/phase1_design.md
# §7 (RID example; NSM/RIC for the broader modulator pool). The exact set
# is conservative — only the well-attested neuropeptide/aminergic
# sources. Phase 1.0.4 will use this list when promoting modulator edges.
DEFAULT_MODULATOR_NEURONS: tuple[str, ...] = (
    "RID",                       # FLP-14 neuropeptide; forward-promoting
    "NSML", "NSMR",              # serotonin (5-HT) → feeding/exploration
    "RICL", "RICR",              # tyramine → reverse / suppress forward
    "ADFL", "ADFR",              # serotonin (chemosensory pair)
    "HSNL", "HSNR",              # serotonin → egg-laying motor
    "RIH",                       # serotonin (head)
    "AVKL", "AVKR",              # FLP-1 neuropeptide → muscle tone
    "PVT",                       # FLP-1 contributor
    "DVA",                       # neuropeptide hub
)


# Edge-delay heuristics. Phase 1.0 keeps these uniform within type; the
# field exists so plasticity / future biology work can refine.
def _chemical_delay_for(_pre: str, _post: str) -> int:
    return DEFAULT_CHEMICAL_DELAY


def _electrical_delay() -> int:
    return DEFAULT_ELECTRICAL_DELAY


def load_connectome_into_graph(
    connectome: ConnectomeData | None = None,
    *,
    modulator_neurons: Iterable[str] = DEFAULT_MODULATOR_NEURONS,
    plastic_neurons: Iterable[str] = (),
) -> NeuralGraph:
    """Build a NeuralGraph from a loaded ConnectomeData.

    Args:
        connectome: optional pre-loaded ConnectomeData (uses the cache
            by default).
        modulator_neurons: neuron names to promote to ``is_modulator=True``.
            The default set comes from ``DEFAULT_MODULATOR_NEURONS``.
            Names absent from the connectome are silently skipped.
        plastic_neurons: neuron names whose *outgoing* chemical edges
            should be marked ``is_plastic=True`` with rule="hebbian".
            Phase 1.0.1 default is empty — the actual plasticity wiring
            happens in Phase 1.0.4.

    Returns:
        A fully populated NeuralGraph (302 nodes; ~5800 edges from
        Cook 2019; no modulator edges yet — those are added in 1.0.4).
    """
    if connectome is None:
        connectome = ConnectomeData.load()

    mod_set = {n for n in modulator_neurons if n in connectome.neuron_to_idx}
    plastic_set = {n for n in plastic_neurons if n in connectome.neuron_to_idx}

    g = NeuralGraph()

    # ---- Nodes ----------------------------------------------------------
    for name, cat, nt in zip(
        connectome.neuron_names,
        connectome.category,
        connectome.neurotransmitter,
    ):
        is_mod = name in mod_set
        node = Node.from_connectome_row(
            name=name,
            category=cat,
            neurotransmitter=nt,
            is_modulator=is_mod,
            is_plastic=(name in plastic_set),
        )
        g.add_node(node)

    g.rebuild_index()

    # ---- Chemical edges -------------------------------------------------
    # Iterate the *raw* counts so we keep an unsigned magnitude on the Edge
    # and store the sign separately. This matches the design's principle
    # that the edge sign is a property of the synapse (driven by pre-NT).
    W_raw = connectome.W_chem_raw
    if W_raw is None:
        # Reconstruct unsigned magnitudes from W_chem if cache predated
        # raw storage — this is fine because |signed| == raw / scale.
        W_raw = np.abs(connectome.W_chem)
    max_chem = max(1.0, float(np.max(np.abs(W_raw))))
    for i_post in range(connectome.n_neurons):
        for j_pre in range(connectome.n_neurons):
            w = W_raw[i_post, j_pre]
            if w == 0.0:
                continue
            pre = connectome.neuron_names[j_pre]
            post = connectome.neuron_names[i_post]
            sign = -1 if connectome.neurotransmitter[j_pre] == "GABA" else +1
            edge = Edge(
                source=pre,
                target=post,
                type="chemical",
                weight=float(w) / max_chem,
                sign=sign,
                delay=_chemical_delay_for(pre, post),
                is_plastic=(pre in plastic_set),
                plasticity_rule="hebbian" if pre in plastic_set else "frozen",
            )
            g.add_edge(edge)

    # ---- Electrical edges (mirror both directions) ----------------------
    W_gap_raw = connectome.W_gap_raw
    if W_gap_raw is None:
        W_gap_raw = connectome.W_gap.copy()
    max_gap = max(1.0, float(W_gap_raw.max()))
    for i in range(connectome.n_neurons):
        for j in range(i + 1, connectome.n_neurons):
            w = W_gap_raw[i, j]
            if w == 0.0:
                continue
            a = connectome.neuron_names[i]
            b = connectome.neuron_names[j]
            for src, dst in ((a, b), (b, a)):
                edge = Edge(
                    source=src,
                    target=dst,
                    type="electrical",
                    weight=float(w) / max_gap,
                    sign=+1,
                    delay=_electrical_delay(),
                    is_plastic=False,
                )
                g.add_edge(edge)

    return g


__all__ = [
    "load_connectome_into_graph",
    "DEFAULT_MODULATOR_NEURONS",
]
