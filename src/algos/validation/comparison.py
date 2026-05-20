"""AC0.5.2 — Three similarity metrics between digital and real activity matrices.

The brief specifies three families:

  1. **Temporal correlation** — per-matched-neuron Pearson r between digital
     and real ΔF/F traces. Reduced to a single scalar (mean across neurons)
     and a vector (per-neuron) for diagnostics.

  2. **Functional connectivity similarity** — compute the pairwise (neuron×
     neuron) Pearson correlation matrix on both digital and real traces,
     restricted to neurons present in both, then compare the upper-triangle
     entries with a Pearson correlation. A score of 1.0 means the two
     functional-connectivity matrices have identical structure.

  3. **PCA structure similarity** — top-K principal components of both
     activity matrices. Compares (a) the explained-variance spectrum (cosine
     between the two length-K vectors) and (b) the alignment of the principal
     subspaces (Grassmann distance via singular values of the PC-loading
     overlap).

All three metrics work on *matched neuron sets*: only neurons appearing in
both the digital and real traces (by name) contribute.

The digital-vs-real comparison is *not* expected to produce r ≈ 1. The
digital worm has no body, no neuromodulators, and no behavior-driven
sensory stream. What the metrics report is how much of the real worm's
observed structure is already explained by the bare connectome topology
plus the v0.3 dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ComparisonResult:
    name: str
    score: float                                # primary scalar, in [-1, 1] or [0, 1]
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        return f"{self.name:42s}  score={self.score:+.4f}"


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def match_matrices(
    digital: np.ndarray,
    digital_names: list[str],
    real: np.ndarray,
    real_names: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Restrict both matrices to neurons present in both, in shared order.

    Shapes: `(T_digital, N_digital) -> (T_digital, N_matched)`, same for real.
    The time-axis lengths can differ — that's fine for FC and PCA which take
    means/correlations over time; temporal correlation handles it by
    resampling.
    """
    common = [n for n in digital_names if n in set(real_names)]
    d_idx = [digital_names.index(n) for n in common]
    r_idx = [real_names.index(n) for n in common]
    return digital[:, d_idx], real[:, r_idx], common


def _zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Z-score along an axis, replacing constant rows/columns with zero."""
    mu = x.mean(axis=axis, keepdims=True)
    sd = x.std(axis=axis, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd


def _resample_to(x: np.ndarray, n: int) -> np.ndarray:
    """Linear resample a (T, N) matrix along its first axis to length `n`."""
    if x.shape[0] == n:
        return x
    src = np.linspace(0.0, 1.0, x.shape[0])
    dst = np.linspace(0.0, 1.0, n)
    out = np.empty((n, x.shape[1]), dtype=x.dtype)
    for i in range(x.shape[1]):
        out[:, i] = np.interp(dst, src, x[:, i])
    return out


# ---------------------------------------------------------------------------
# Metric 1: temporal correlation
# ---------------------------------------------------------------------------


def temporal_correlation(
    digital: np.ndarray,
    digital_names: list[str],
    real: np.ndarray,
    real_names: list[str],
) -> ComparisonResult:
    """Mean Pearson r across matched neurons between digital and real traces.

    If trace lengths differ, the digital trace is linearly resampled to the
    real length first. This is a coarse alignment — we don't have a body so
    there's no meaningful temporal correspondence, but the metric is still
    informative as a sanity check (an r near zero is expected).
    """
    d, r, names = match_matrices(digital, digital_names, real, real_names)
    if d.shape[1] == 0:
        return ComparisonResult(
            name="temporal_correlation",
            score=float("nan"),
            details={"n_matched": 0},
        )
    d = _resample_to(d, r.shape[0])
    per_neuron = []
    for k in range(d.shape[1]):
        a, b = d[:, k], r[:, k]
        if a.std() < 1e-12 or b.std() < 1e-12:
            per_neuron.append(0.0)
            continue
        per_neuron.append(float(np.corrcoef(a, b)[0, 1]))
    per_neuron = np.array(per_neuron)
    return ComparisonResult(
        name="temporal_correlation",
        score=float(per_neuron.mean()),
        details={
            "n_matched": int(d.shape[1]),
            "per_neuron_mean": float(per_neuron.mean()),
            "per_neuron_median": float(np.median(per_neuron)),
            "per_neuron_max": float(per_neuron.max()),
            "per_neuron_min": float(per_neuron.min()),
            "per_neuron_abs_mean": float(np.abs(per_neuron).mean()),
            "per_neuron_dict": {names[k]: float(per_neuron[k])
                                for k in range(len(names))},
        },
    )


# ---------------------------------------------------------------------------
# Metric 2: functional connectivity similarity
# ---------------------------------------------------------------------------


def functional_connectivity_similarity(
    digital: np.ndarray,
    digital_names: list[str],
    real: np.ndarray,
    real_names: list[str],
) -> ComparisonResult:
    """Pearson correlation of the off-diagonal FC entries.

    Builds the (N_matched × N_matched) Pearson correlation matrix on each
    activity set, then computes the Pearson r between the upper-triangle
    entries (excluding the diagonal).
    """
    d, r, names = match_matrices(digital, digital_names, real, real_names)
    n = d.shape[1]
    if n < 4:
        return ComparisonResult(
            name="functional_connectivity_similarity",
            score=float("nan"),
            details={"n_matched": n, "reason": "fewer than 4 matched neurons"},
        )
    fc_d = np.corrcoef(d, rowvar=False)
    fc_r = np.corrcoef(r, rowvar=False)
    iu = np.triu_indices(n, k=1)
    a, b = fc_d[iu], fc_r[iu]
    a = np.nan_to_num(a)
    b = np.nan_to_num(b)
    score = float(np.corrcoef(a, b)[0, 1])
    # Also compute on absolute values — useful when sign flips between
    # populations are uninformative.
    score_abs = float(np.corrcoef(np.abs(a), np.abs(b))[0, 1])
    return ComparisonResult(
        name="functional_connectivity_similarity",
        score=score,
        details={
            "n_matched": int(n),
            "score_unsigned": score_abs,
            "fc_digital_mean_abs": float(np.abs(a).mean()),
            "fc_real_mean_abs": float(np.abs(b).mean()),
        },
    )


# ---------------------------------------------------------------------------
# Metric 3: PCA structure similarity
# ---------------------------------------------------------------------------


def pca_structure_similarity(
    digital: np.ndarray,
    digital_names: list[str],
    real: np.ndarray,
    real_names: list[str],
    *,
    n_components: int = 10,
) -> ComparisonResult:
    """Two-part similarity of top-K PCA structure.

    Reports:
      - `explained_variance_cos`: cosine between the top-K explained-variance
        ratios of digital and real. 1.0 = same spectrum, 0.0 = orthogonal.
      - `subspace_alignment`: mean of the singular values of `D^T R` where D
        and R are the K-dim orthonormal PC bases. This is the average
        principal angle cosine; 1.0 = identical subspaces, 0.0 = orthogonal.

    `score` is the average of the two (in [0, 1]).
    """
    d, r, names = match_matrices(digital, digital_names, real, real_names)
    n = d.shape[1]
    K = min(n_components, n - 1)
    if K < 2:
        return ComparisonResult(
            name="pca_structure_similarity",
            score=float("nan"),
            details={"n_matched": n, "reason": "fewer than 3 matched neurons"},
        )

    # Z-score per neuron across time before PCA so the scale of each variable
    # doesn't dominate (digital/real have different unit conventions).
    dz = _zscore(d, axis=0)
    rz = _zscore(r, axis=0)

    # SVD-based PCA on the (T x N) matrices.
    _, Sd, Vtd = np.linalg.svd(dz, full_matrices=False)
    _, Sr, Vtr = np.linalg.svd(rz, full_matrices=False)
    var_d = (Sd ** 2)[:K] / (Sd ** 2).sum()
    var_r = (Sr ** 2)[:K] / (Sr ** 2).sum()

    cos_var = float(
        np.dot(var_d, var_r) / (np.linalg.norm(var_d) * np.linalg.norm(var_r))
    )

    # Subspace alignment: each Vt has shape (n_pcs, N). The top-K basis is
    # Vtd[:K] (K, N). The principal angles between the two subspaces are the
    # arccos of the singular values of Vtd[:K] @ Vtr[:K].T (K, K).
    overlap = Vtd[:K] @ Vtr[:K].T
    s = np.linalg.svd(overlap, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    align = float(s.mean())

    score = 0.5 * (cos_var + align)
    return ComparisonResult(
        name="pca_structure_similarity",
        score=score,
        details={
            "n_matched": int(n),
            "n_components": int(K),
            "explained_variance_cos": cos_var,
            "subspace_alignment": align,
            "var_digital": var_d.tolist(),
            "var_real": var_r.tolist(),
        },
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def run_all_metrics(
    digital: np.ndarray,
    digital_names: list[str],
    real: np.ndarray,
    real_names: list[str],
    *,
    n_components: int = 10,
) -> list[ComparisonResult]:
    """Run the three metrics and return their results."""
    return [
        temporal_correlation(digital, digital_names, real, real_names),
        functional_connectivity_similarity(digital, digital_names, real, real_names),
        pca_structure_similarity(
            digital, digital_names, real, real_names, n_components=n_components
        ),
    ]


__all__ = [
    "ComparisonResult",
    "match_matrices",
    "temporal_correlation",
    "functional_connectivity_similarity",
    "pca_structure_similarity",
    "run_all_metrics",
]
