# Reference electrophysiology data

This directory holds real C. elegans whole-brain calcium-imaging datasets
used for Phase 0.5 validation (AC0.5.1 / AC0.5.2). The neural-skeleton
implementation under `src/algos/` is compared against these data to measure
how closely the digital worm's activity resembles the real worm's.

## Primary source: Atanas et al. 2023 (Cell)

Paper: Atanas, Kim, Wang, Bueno, Becker, Kang, Park, Kramer, Wan, Baskoylu,
Dag, Kalogeropoulou, Gomes, Estrem, Cohen, Mansinghka, Flavell.
"Brain-wide representations of behavior spanning multiple timescales and
states in C. elegans." *Cell* 186(19): 4134-4151.e31, 2023.
<https://www.cell.com/cell/fulltext/S0092-8674(23)00850-4>

Code: <https://github.com/flavell-lab/AtanasKim-Cell2023>
Interactive: <https://wormwideweb.org>

Bulk data on Zenodo: <https://doi.org/10.5281/zenodo.8150514>
(Current record at fetch time: <https://zenodo.org/records/19388374>.)

### Files we use

The full Zenodo deposit is 33 GB. Phase 0.5 needs only a subset.

| File | Size (bz2) | Used for | Path |
|---|---:|---|---|
| `neuropal_label.json.bz2`        |  16 kB | neuron-ID labels per dataset                 | `data/reference/` |
| `neuron_categorization.h5.bz2`   | 107 kB | per-behavior neuron categorization (fwd/rev) | `data/reference/` |
| `encoding_changes_corrected.h5.bz2` | 55 kB | encoding-strength deltas (cross-state)    | `data/reference/` |
| `fit_ranges.h5.bz2`              |   2 kB | fit-range metadata                            | `data/reference/` |
| `processed_h5.tar.bz2`           | 569 MB | calcium time traces + behavior streams        | `data/reference/` |

All `*.bz2` files are committed to `.gitignore`; the loader decompresses them
on first use into the same directory.

### Download

The four small files (~180 kB total) are fetched in one batch by:

```bash
cd data/reference
for f in neuropal_label.json.bz2 neuron_categorization.h5.bz2 \
         encoding_changes_corrected.h5.bz2 fit_ranges.h5.bz2; do
  curl -sL --max-time 60 -o "$f" \
    "https://zenodo.org/api/records/19388374/files/$f/content"
done
```

The large `processed_h5.tar.bz2` (569 MB) is the time-trace tarball:

```bash
curl -sL --max-time 600 -o processed_h5.tar.bz2 \
  "https://zenodo.org/api/records/19388374/files/processed_h5.tar.bz2/content"
tar -xjf processed_h5.tar.bz2     # ~2.5 GB uncompressed, per-recording .h5
```

If a file is missing the loader will raise `FileNotFoundError` and reference
this README.

### Format notes

`neuropal_label.json` structure (top-level keys: `data`, `metadata`):
```
data[<recording_id>]["idx_neuron-label"][<idx_str>] = {
    "label": "RMDL",                # name, may include "?" suffix for low-confidence
    "neuron_class": "RMD",
    "LR": "L" | "R" | "undefined",
    "region": "Lateral Ganglion",
    "DV": "D" | "V" | "undefined",
    "roi_id": [<int>, ...],
    "confidence": float (1.0 - 5.0),
}
```

`neuron_categorization.h5` per-recording shape:
```
data/<recording_id>/<channel>/<behavior>/<category>  →  int64 array of indices
  behavior ∈ {"v", "θh", "P"}             # velocity, head-angle, feeding
  category ∈ {"all", "fwd", "rev", "rev_slope_neg", …}
```

The per-recording neuron indices are 1-based and *index into that recording's
NeuroPAL panel* — they are not global. To map a Phase 0.5 result against
real data, intersect via the `neuropal_label.json` labels.

## Fallback: Kato et al. 2015

If the Atanas 2023 data is ever inaccessible, Kato 2015 is the secondary
reference (whole-brain imaging, 5 worms). Not yet wired up; consult the
paper's supplementary materials.

## Notes for future phases

- The Atanas data is *behaviorally-conditioned* — it records freely behaving
  worms. Phase 0.5 (no body) can compare against the *steady-state activation
  patterns* in each behavioral epoch, but cannot reproduce the dynamics of
  freely-moving behavior. That work belongs to Phase 1+.
- For time-trace comparisons (AC0.5.2 metrics) we use the trace files in
  `processed_h5.tar.bz2`. They contain per-recording activity matrices with
  the same row indexing as `neuron_categorization.h5`.
