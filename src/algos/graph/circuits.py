"""Canonical C. elegans functional circuits as Phase 1.0 Subgraphs.

Each entry below is a named subgraph the literature treats as a
coherent functional unit. The membership lists are drawn from:

  - White et al. 1986 (original connectome anatomy)
  - Chalfie et al. 1985 (touch reflex; ALM/AVM/PLM → command)
  - Bargmann 2012 review (chemosensation: ASE/AWC/AWA → AIY/AIZ/RIA)
  - Mori & Ohshima 1995 (thermotaxis: AFD → AIY/AIZ)
  - Gray, Hill & Bargmann 2005 (forward/reversal command anatomy)
  - Avery & Horvitz 1989, Trojanowski et al. 2014 (pharyngeal CPG)
  - Schafer 2005 (egg-laying: HSN, VC)
  - Liu & Sternberg 1995, Wang et al. 2013 (defecation: DVA/DVB/AVL)
  - Faumont et al. 2011, Kawano et al. 2011 (head motor CPG: RMD, SMD,
    SAA, OLQ — the "head wiggle" oscillator)

Each subgraph names the neurons explicitly; the Subgraph object pulls
the relevant chemical and electrical edges from the parent graph at
construction time. Subgraphs overlap (AVA/AVD belong to both
``reversal_command`` and ``anterior_touch``, AVB/PVC belong to both
``forward_command`` and ``posterior_touch`` — see Subgraph.overlap_with).

This file is data + a single builder function; no dynamics live here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from algos.graph.graph import NeuralGraph
from algos.graph.subgraph import Subgraph


# ---------------------------------------------------------------------------
# Canonical circuit specifications
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CircuitSpec:
    name: str
    type: str                 # 'feedforward' | 'recurrent'
    members: tuple[str, ...]
    note: str = ""


CIRCUIT_SPECS: tuple[CircuitSpec, ...] = (
    CircuitSpec(
        name="reversal_command",
        type="recurrent",
        members=("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"),
        note="Backward-locomotion command interneurons (Gray 2005, "
             "Chalfie 1985). All three pairs feed the A-class motor "
             "pool that drives backward waves.",
    ),
    CircuitSpec(
        name="forward_command",
        type="recurrent",
        members=(
            "AVBL", "AVBR",       # main forward command
            "PVCL", "PVCR",       # forward command, posterior dominant
            "RIBL", "RIBR",       # forward-bias integrator
            "AIBL", "AIBR",       # arousal/turn-bias
            "RIML", "RIMR",       # reversal-bias counterpart, included
                                  # because it forms a competitive ring
                                  # with AVB/AIB (Wang et al. 2020).
        ),
        note="Forward-locomotion command + tightly coupled integrators "
             "(Gray 2005). RIM lives here too because it forms the "
             "winner-take-all loop with AIB/AVB that picks "
             "forward vs. reversal.",
    ),
    CircuitSpec(
        name="anterior_touch",
        type="feedforward",
        members=("ALML", "ALMR", "AVM", "AVDL", "AVDR", "AVAL", "AVAR"),
        note="Gentle anterior touch → backward escape. ALM/AVM "
             "(sensory) → AVD (command) → A-class motor neurons. "
             "AVA included as the downstream command end "
             "(Chalfie 1985).",
    ),
    CircuitSpec(
        name="posterior_touch",
        type="feedforward",
        members=(
            "PLML", "PLMR", "PVM", "PVDL", "PVDR",
            "AVBL", "AVBR", "PVCL", "PVCR",
        ),
        note="Posterior touch → forward acceleration. PLM/PVD/PVM "
             "(sensory) → AVB/PVC (command). PVD also reports harsh "
             "touch (Way & Chalfie 1989).",
    ),
    CircuitSpec(
        name="chemosensory_amphid",
        type="feedforward",
        members=(
            "ASEL", "ASER",       # salt taste
            "AWCL", "AWCR",       # volatile attractants
            "AWAL", "AWAR",       # volatile attractants (second class)
            "ASHL", "ASHR",       # nociception
            "ASKL", "ASKR",       # adaptation / hub
            "AIYL", "AIYR",       # 1st-order interneurons
            "AIZL", "AIZR",       # 2nd-order interneurons
            "AIBL", "AIBR",       # arousal
            "RIAL", "RIAR",       # premotor integrator
        ),
        note="Amphid chemosensory feedforward (Bargmann 2012). Sensory "
             "→ AIY/AIZ/AIB → RIA → head motor / command. The same "
             "AIY/AIZ pair appears in chemo + thermo paths.",
    ),
    CircuitSpec(
        name="thermosensory",
        type="feedforward",
        members=("AFDL", "AFDR", "AIYL", "AIYR", "AIZL", "AIZR",
                 "RIAL", "RIAR"),
        note="AFD-driven thermotaxis (Mori & Ohshima 1995). Shares "
             "AIY/AIZ/RIA with chemosensory — this is one of the "
             "design's intended overlap points (§4.4).",
    ),
    CircuitSpec(
        name="head_motor_cpg",
        type="recurrent",
        members=(
            "RMDL", "RMDR", "RMDDL", "RMDDR", "RMDVL", "RMDVR",
            "SMDDL", "SMDDR", "SMDVL", "SMDVR",
            "SAADL", "SAADR", "SAAVL", "SAAVR",
            "OLQDL", "OLQDR", "OLQVL", "OLQVR",
            "RIAL", "RIAR",
            "RMED", "RMEV", "RMEL", "RMER",
        ),
        note="Head-bending CPG (Faumont 2011, Kawano 2011). Mutually "
             "inhibitory dorsal/ventral RMD pairs produce the head "
             "wiggle that steers chemotaxis; RMEs gate amplitude. "
             "Shares RIA with chemosensory.",
    ),
    CircuitSpec(
        name="pharyngeal_cpg",
        type="recurrent",
        members=(
            "M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5",
            "MCL", "MCR", "MI",
            "I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6",
            "NSML", "NSMR",
        ),
        note="Pharyngeal feeding CPG (Avery & Horvitz 1989, "
             "Trojanowski 2014). M-class motorneurons + I-class "
             "interneurons + NSM serotonergic. Mostly anatomically "
             "isolated from the somatic nervous system but coupled "
             "via NSM → behavior axis.",
    ),
    CircuitSpec(
        name="ventral_cord_motor",
        type="recurrent",
        members=tuple(
            [f"DA{i:02d}" for i in range(1, 10)] +
            [f"DB{i:02d}" for i in range(1, 8)] +
            [f"VA{i:02d}" for i in range(1, 13)] +
            [f"VB{i:02d}" for i in range(1, 12)] +
            [f"DD{i:02d}" for i in range(1, 7)] +
            [f"VD{i:02d}" for i in range(1, 14)]
        ),
        note="Ventral nerve cord motor pool: A-class (backward), "
             "B-class (forward), D-class (cross-inhibitory GABAergic). "
             "The traveling wave is implemented by their gap-junction "
             "ladder + intra-class chemical recurrence.",
    ),
    CircuitSpec(
        name="modulator_RID",
        type="feedforward",
        members=("RID", "AVBL", "AVBR", "PVCL", "PVCR"),
        note="RID (neuropeptide source) → forward command pool. "
             "Subject of Phase 0.9 / 0.9a experiments. Modulator edges "
             "for the slow-modulation effect are added in Phase 1.0.4.",
    ),
    CircuitSpec(
        name="modulator_5HT",
        type="feedforward",
        members=(
            "NSML", "NSMR",          # primary 5-HT source
            "ADFL", "ADFR",          # secondary 5-HT
            "HSNL", "HSNR",          # 5-HT, egg-laying motor
            "RIH",                   # 5-HT receiver in head
            "M3L", "M3R",            # pharyngeal targets
        ),
        note="Serotonergic modulator subnetwork. NSM/ADF/HSN are the "
             "well-attested 5-HT sources; M3 and RIH are downstream "
             "targets with strong 5-HT-driven changes in firing "
             "(Sze 2000, Tanis 2008).",
    ),
    CircuitSpec(
        name="egg_laying",
        type="feedforward",
        members=("HSNL", "HSNR",
                 "VC01", "VC02", "VC03", "VC04", "VC05", "VC06"),
        note="HSN (5-HT command) + VC1-6 cholinergic vulval motor "
             "neurons (Schafer 2005). Vulval muscles themselves are "
             "outside the 302; this subgraph captures the neural "
             "command unit.",
    ),
    CircuitSpec(
        name="defecation_pacemaker",
        type="recurrent",
        members=("DVA", "DVB", "AVL"),
        note="Tri-neuron rhythmic command for the ~50-second "
             "defecation cycle (Liu & Sternberg 1995, Wang 2013). "
             "Smallest functional CPG in the worm.",
    ),
    CircuitSpec(
        name="inhibitory_command_gate",
        type="recurrent",
        members=(
            "RIS",           # master FLP-11 + GABA sleep / global inhibitor
            "AVL",           # GABA, also in defecation_pacemaker (shared)
            "DVB",           # GABA, also in defecation_pacemaker (shared)
            "ALA",           # FLP-13 stress-induced sleep peptide source
            "RIH",           # 5-HT / DA receiver, head-side modulator hub
        ),
        note="Inhibitory hub identified in Phase 1.5+.2 §A.1. RIS, AVL, "
             "DVB are the three GABAergic interneurons not already in "
             "command/touch/CPG subgraphs; ALA is the FLP-13 quiescence "
             "neuron with the connectome's strongest peptidergic output "
             "(75 contacts each onto PVDL/R); RIH is a 5-HT/DA-receiving "
             "head modulator hub. References: Steuer Costa 2019 (RIS), "
             "Nelson 2014 (ALA-FLP13), Hill 2014, Turek 2013. The "
             "inhibitory chemical edges from RIS/AVL/DVB onto command "
             "neurons are ALREADY in W_chem with sign=-1 (these "
             "neurons are in the 26-GABA list) — adding this subgraph "
             "is a data-level acknowledgement of their functional "
             "role, not a dynamics change. The gate is dormant in "
             "the bare network (member neurons fire ~0 times) and "
             "becomes active when something drives RIS/AVL/ALA "
             "(stimulation, Phase 1.5 body feedback, or "
             "Phase 1.6.2 tyramine arm of RIM).",
    ),
)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def build_canonical_subgraphs(
    graph: NeuralGraph,
    *,
    specs: Iterable[CircuitSpec] = CIRCUIT_SPECS,
    skip_missing: bool = False,
) -> dict[str, Subgraph]:
    """Instantiate every subgraph in ``specs`` and register them.

    Returns the mapping ``{name: Subgraph}``. Each subgraph is also
    inserted into ``graph.subgraphs`` so consumers can find them by
    name via ``graph.subgraphs['reversal_command']``.

    Args:
        skip_missing: if True, silently drop neurons absent from
            ``graph`` (useful for connectome variants). If False
            (default), raise the underlying KeyError so we notice
            data drift.
    """
    out: dict[str, Subgraph] = {}
    for spec in specs:
        members = list(spec.members)
        if skip_missing:
            members = [m for m in members if graph.has_node(m)]
        sg = Subgraph(
            name=spec.name,
            type=spec.type,
            node_names=members,
            parent=graph,
        )
        sg.materialize()
        graph.register_subgraph(sg)
        out[spec.name] = sg
    return out


def summarize_subgraphs(graph: NeuralGraph) -> list[dict]:
    """Return a per-subgraph summary: name, type, |nodes|, |chem|, |gap|."""
    summaries = []
    for name, sg in graph.subgraphs.items():
        chem_internal = int((sg.W_chem != 0).sum())
        gap_internal = int((sg.W_gap != 0).sum())
        # Overlap stats: how many of these nodes also belong to another
        # registered subgraph.
        overlaps = {}
        for other_name, other in graph.subgraphs.items():
            if other_name == name:
                continue
            shared = sg.overlap_with(other)
            if shared:
                overlaps[other_name] = sorted(shared)
        summaries.append({
            "name": name,
            "type": sg.type,
            "n_nodes": sg.n_nodes,
            "n_internal_chem": chem_internal,
            "n_internal_gap": gap_internal,
            "overlaps_with": overlaps,
        })
    return summaries


__all__ = [
    "CircuitSpec",
    "CIRCUIT_SPECS",
    "build_canonical_subgraphs",
    "summarize_subgraphs",
]
