"""Per-neuron neurotransmitter assignments for C. elegans hermaphrodite.

Sources (summarized — see `docs/data_audit.md` once it is fleshed out):

- GABAergic set: McIntire et al. 1993 *Nature*; Schuske et al. 2004 *J Neurosci*;
  Gendrel et al. 2016 *eLife* extended the canonical list.
- Glutamatergic and cholinergic assignments: Serrano-Saiz et al. 2013 *Cell*
  (glutamate) and Pereira et al. 2015 *eLife* (acetylcholine). For Phase 0 we
  only need to *correctly mark inhibitory neurons*; cholinergic, glutamatergic,
  serotonergic, dopaminergic, and unknown all keep the positive sign as
  prescribed by design.md §3.1 + phase0.md §1.4.

The GABAergic list below is the conservative canonical set: pharyngeal RIS,
the four RMEs, AVL, DVB, and the 19 D-type ventral cord motor neurons. This
totals 26 neurons.

If a neuron name is not in `GABAERGIC` it is treated as excitatory (sign +1).
"""

from __future__ import annotations

# Canonical GABAergic hermaphrodite neurons.
GABAERGIC: frozenset[str] = frozenset({
    # Pharyngeal / head GABAergic
    "AVL", "DVB", "RIS",
    # RMEs (all four are GABAergic per Gendrel 2016)
    "RMED", "RMEV", "RMEL", "RMER",
    # D-type ventral cord motor neurons (Cook 2019 uses zero-padded names)
    "DD01", "DD02", "DD03", "DD04", "DD05", "DD06",
    "VD01", "VD02", "VD03", "VD04", "VD05", "VD06",
    "VD07", "VD08", "VD09", "VD10", "VD11", "VD12", "VD13",
})


def neurotransmitter_sign(neuron_name: str) -> int:
    """Return +1 (excitatory / default) or -1 (GABAergic).

    Design choice: any neuron not on the GABAergic list is treated as
    excitatory. This matches phase0.md §1.4 ("mixed or unknown: keep positive,
    log it").
    """
    return -1 if neuron_name in GABAERGIC else 1
