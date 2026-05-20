"""AC0.5.1 — Loader for the real C. elegans whole-brain calcium imaging data.

Primary source: Atanas et al. 2023, *Cell* — see `data/reference/README.md`.

The Atanas archive contains:

  - `neuropal_label.json` — per-recording mapping `int idx → {"label": "AVAL", ...}`.
    A label string can carry a `?` suffix for low-confidence calls; we filter
    those out by default.
  - `processed_h5/<recording_id>-data.h5` — per-recording HDF5 with calcium
    traces (`gcamp/traces_array_F_Fmean`, shape `(T, n_recorded_neurons)`)
    and behavioral covariates (`behavior/velocity`, `behavior/reversal_vec`
    etc., shape `(T,)`).

The two index spaces (JSON `idx` and HDF5 column) are 1-based / 0-based
respectively and live in the same numerical order — `idx=k` in the JSON
points at column `k-1` in the trace matrix. This module hides that
conversion behind `Recording.trace_by_name(...)`.
"""

from __future__ import annotations

import bz2
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from algos.config import DATA_DIR


REFERENCE_DIR: Path = DATA_DIR / "reference"
NEUROPAL_LABEL_JSON: Path = REFERENCE_DIR / "neuropal_label.json"
NEUROPAL_LABEL_BZ2: Path = REFERENCE_DIR / "neuropal_label.json.bz2"
PROCESSED_H5_DIR: Path = REFERENCE_DIR / "processed_h5"


# ---------------------------------------------------------------------------
# Per-recording container
# ---------------------------------------------------------------------------


@dataclass
class Recording:
    """One Atanas-2023 recording: aligned calcium + behavior."""

    recording_id: str
    traces: np.ndarray            # (T, n_recorded_neurons), ΔF/F_mean
    velocity: np.ndarray          # (T,)
    reversal: np.ndarray          # (T,) int — 1 during reversal events
    labels: dict[int, str] = field(default_factory=dict)
    # `labels[col_idx]` is the high-confidence neuron name (no `?` suffix),
    # where `col_idx` is the 0-based column into `traces`. Columns without a
    # high-confidence label are absent.

    @property
    def n_timepoints(self) -> int:
        return int(self.traces.shape[0])

    @property
    def n_recorded(self) -> int:
        return int(self.traces.shape[1])

    @property
    def n_labeled(self) -> int:
        return len(self.labels)

    def trace_by_name(self, neuron_name: str) -> np.ndarray | None:
        """Return the ΔF/F trace for a neuron name if labeled, else None."""
        for col_idx, name in self.labels.items():
            if name == neuron_name:
                return self.traces[:, col_idx]
        return None

    def labeled_trace_matrix(self, neuron_names: Iterable[str]) -> tuple[
        np.ndarray, list[str]
    ]:
        """Return the sub-matrix of traces matching `neuron_names`.

        Only neurons that are *both* in `neuron_names` and in `self.labels`
        appear. Returned matrix shape is `(T, n_matched)`; the parallel list
        gives the names in column order.
        """
        cols, found = [], []
        for col, name in self.labels.items():
            if name in neuron_names:
                cols.append(col)
                found.append(name)
        if not cols:
            return np.zeros((self.n_timepoints, 0)), []
        return self.traces[:, cols], found


# ---------------------------------------------------------------------------
# Dataset (collection of recordings)
# ---------------------------------------------------------------------------


@dataclass
class ReferenceDataset:
    """Uniform interface to the full collection of recordings."""

    recordings: list[Recording]

    @property
    def n_recordings(self) -> int:
        return len(self.recordings)

    def by_id(self, recording_id: str) -> Recording:
        for r in self.recordings:
            if r.recording_id == recording_id:
                return r
        raise KeyError(f"No recording with id {recording_id!r}")

    def neuron_label_frequency(self) -> dict[str, int]:
        """How often each neuron name is labeled across recordings."""
        counts: dict[str, int] = {}
        for r in self.recordings:
            for n in r.labels.values():
                counts[n] = counts.get(n, 0) + 1
        return counts

    # -------------------------------------------------------------- loading

    @classmethod
    def from_atanas2023(
        cls,
        *,
        reference_dir: Path | None = None,
        recording_ids: Iterable[str] | None = None,
        drop_low_confidence: bool = True,
        max_recordings: int | None = None,
    ) -> "ReferenceDataset":
        """Load Atanas 2023 recordings from `data/reference/`.

        Args:
            reference_dir: override the default `data/reference/` location.
            recording_ids: if given, load only these recording IDs.
            drop_low_confidence: skip labels with the `?` suffix (default True).
            max_recordings: cap on the number of recordings to load (useful
                for fast tests). None means load all available.
        """
        ref_dir = Path(reference_dir) if reference_dir else REFERENCE_DIR
        labels = _load_neuropal_labels(ref_dir, drop_low_confidence=drop_low_confidence)

        # Discover recordings on disk and intersect with the label set.
        h5_dir = ref_dir / "processed_h5"
        if not h5_dir.exists():
            raise FileNotFoundError(
                f"{h5_dir} does not exist. See data/reference/README.md for the "
                "download instructions for processed_h5.tar.bz2."
            )

        # Sort recordings by label coverage (most-labeled first) so
        # `max_recordings=N` gives the N best-annotated recordings rather than
        # whatever happens to come first alphabetically.
        candidates: list[tuple[int, str, Path]] = []
        for h5_path in sorted(h5_dir.glob("*-data.h5")):
            rid = h5_path.name.removesuffix("-data.h5")
            if recording_ids is not None and rid not in recording_ids:
                continue
            n_labels = len(labels.get(rid, {}))
            if recording_ids is None and n_labels == 0:
                # Skip recordings with no high-confidence labels — they can't
                # be matched against our digital neurons by name.
                continue
            candidates.append((n_labels, rid, h5_path))
        candidates.sort(key=lambda t: -t[0])

        recordings: list[Recording] = []
        for _, rid, h5_path in candidates:
            rec = _load_recording(rid, h5_path, labels.get(rid, {}))
            recordings.append(rec)
            if max_recordings is not None and len(recordings) >= max_recordings:
                break

        return cls(recordings=recordings)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _load_neuropal_labels(
    ref_dir: Path,
    *,
    drop_low_confidence: bool,
) -> dict[str, dict[int, str]]:
    """Return `{recording_id: {col_idx: neuron_name}}`."""
    if (ref_dir / "neuropal_label.json").exists():
        with (ref_dir / "neuropal_label.json").open() as f:
            payload = json.load(f)
    elif (ref_dir / "neuropal_label.json.bz2").exists():
        with bz2.open(ref_dir / "neuropal_label.json.bz2", "rt") as f:
            payload = json.load(f)
    else:
        raise FileNotFoundError(
            f"neuropal_label.json (or .bz2) not in {ref_dir}. "
            "See data/reference/README.md."
        )

    raw = payload["data"]  # {rec_id: {"idx_neuron-label": {idx_str: {...}}}}
    out: dict[str, dict[int, str]] = {}
    for rec_id, body in raw.items():
        per_rec: dict[int, str] = {}
        for idx_str, info in body.get("idx_neuron-label", {}).items():
            name = info.get("label", "")
            if not name:
                continue
            if drop_low_confidence and name.endswith("?"):
                continue
            # JSON idx is 1-based; convert to 0-based column index.
            col_idx = int(idx_str) - 1
            per_rec[col_idx] = name
        out[rec_id] = per_rec
    return out


def _load_recording(rid: str, h5_path: Path, labels: dict[int, str]) -> Recording:
    with h5py.File(h5_path, "r") as f:
        # ΔF/F normalized to baseline mean — this is what the paper's CePNEM
        # model fits.
        traces = np.asarray(f["gcamp/traces_array_F_Fmean"][...])
        velocity = np.asarray(f["behavior/velocity"][...])
        reversal = np.asarray(f["behavior/reversal_vec"][...])

    # Sanitize: traces sometimes carry NaNs around recording dropouts. Replace
    # them with zero so downstream correlations don't propagate NaN.
    traces = np.nan_to_num(traces, nan=0.0)

    return Recording(
        recording_id=rid,
        traces=traces,
        velocity=velocity,
        reversal=reversal,
        labels=labels,
    )


__all__ = ["ReferenceDataset", "Recording"]
