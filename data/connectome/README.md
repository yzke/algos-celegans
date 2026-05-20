# Connectome data

Phase 0 uses the Cook et al. 2019 hermaphrodite connectome, corrected
July 2020.

## Required files

| File | Source | Purpose |
|------|--------|---------|
| `SI5_corrected.xlsx` | WormWiring SI 5 (corrected July 2020) | Chemical + gap-junction adjacency matrices |
| `SI4_cells.xlsx` *(optional)* | WormWiring SI 4 | Cell-list metadata (only used for spot-checks) |
| `connectome.npz` *(generated)* | Built by `ConnectomeData.load()` from the xlsx | Fast-load cache |

## Download

The corrected SI 5 workbook is available directly from WormWiring:

```bash
cd data/connectome
curl -L -o SI5_corrected.xlsx \
    "https://wormwiring.org/si/SI%205%20Connectome%20adjacency%20matrices,%20corrected%20July%202020.xlsx"
```

Same source serves the original (non-corrected) 2019 version at
`SI 5 Connectome adjacency matrices.xlsx`. We use the corrected version.

If WormWiring is unreachable, the matrices are reproduced as supplementary
material to:

> Cook, S. J., Jarrell, T. A., Brittin, C. A., et al. (2019).
> Whole-animal connectomes of both *Caenorhabditis elegans* sexes.
> *Nature*, 571(7763), 63–71.

## Sheet structure (for reference)

`SI5_corrected.xlsx` exposes seven sheets. We read only:

- **`hermaphrodite chemical`** — pre × post weighted by EM serial-section
  count. 300 pre-synaptic rows (every neuron with a chemical out-synapse) ×
  454 post-synaptic columns (neurons + muscles + glia + glands).
- **`hermaphrodite gap jn symmetric`** — 469 × 469 weighted symmetric.
  Contains 14 self-loop entries which we zero on load (see
  `src/algos/connectome.py` for the rationale — `gap_input` algebraically
  cancels them anyway).

Row labels live in column C starting at row 4; column labels live in row 3
starting at column D; ganglion section headers (`PHARYNX`, `SENSORY
NEURONS`, etc.) appear in column A / row 1.

## 302-neuron list construction

The canonical 302 hermaphrodite neurons are derived directly from the
spreadsheet:

1. Take every chemical row label → 300 neurons (every neuron that emits at
   least one chemical synapse).
2. Add `CANL`, `CANR` from the gap sheet (these have no chemical out-
   synapses but are part of the 302; Cook 2019 files them in the gap
   sheet's *MUSCLES* section).

The full neuron list is therefore reconstructible from a single
spreadsheet — no separate "302 neurons" file is required.

## Neurotransmitter signs

`W_chem` is signed: GABAergic neurons' output columns are flipped to
negative. The 26 GABAergic neurons (AVL, DVB, RIS, four RMEs, DD01–06,
VD01–13) are listed in `src/algos/neurotransmitters.py` based on McIntire
et al. 1993, Schuske et al. 2004, and Gendrel et al. 2016.
