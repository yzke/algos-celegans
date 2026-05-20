"""Phase 0.7 — real-vs-real individual-variation baseline.

Compute three similarity metrics across pairs of real C. elegans worms
(Atanas 2023 recordings) and within each recording (split-half) so we
know what the *real* worm-to-worm variation looks like. Then compute
the same metrics digital-vs-real and report where the digital worm
sits in the real distributions.

Three metrics (Phase 0.5 / Phase 0.6 lineage):
  - subspace_alignment           — Phase 0.6 confirmed as the
                                   defensible sub-component
  - temporal_correlation         — mean per-neuron Pearson r
  - functional_connectivity_similarity  — Pearson r of FC upper-triangle entries

For each metric we compute three distributions:
  1. Real cross-worm pairs (N choose 2)
  2. Real within-worm split-halves (first half vs second half)
  3. Digital-vs-real (digital sim matched in length to each recording)

Bootstrap 95% confidence intervals on the mean and 5%/95% percentiles
(B = 1000 resamples each).

Outputs:
  output/individual_variation_results.json
  output/individual_variation_report.txt
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from algos.config import OUTPUT_DIR
from algos.connectome import ConnectomeData
from algos.neural import CTRNNParams, NeuralState, neural_step
from algos.validation.comparison import (
    functional_connectivity_similarity,
    pca_structure_similarity,
    temporal_correlation,
)
from algos.validation.reference_data import ReferenceDataset


N_WORMS = 10                # how many recordings to use
N_BOOTSTRAP = 1000
PRE_EQ_TICKS = 2000
SENSORY_NOISE = 0.1
N_PCA_COMPONENTS = 10


# ---------------------------------------------------------------------------
# Helpers — convert one Recording into (traces, names) for the metric API
# ---------------------------------------------------------------------------


def recording_matrix(rec) -> tuple[np.ndarray, list[str]]:
    """Return (T, n_labeled) trace matrix and the parallel list of names."""
    cols, names = [], []
    for c, n in rec.labels.items():
        cols.append(c)
        names.append(n)
    return rec.traces[:, cols], names


def half_split(traces: np.ndarray, names: list[str]):
    """Split a recording's trace matrix at the midpoint along time."""
    T = traces.shape[0]
    mid = T // 2
    return (
        (traces[:mid], names),
        (traces[mid:], names),
    )


def metric_triple(a: np.ndarray, names_a: list[str],
                  b: np.ndarray, names_b: list[str]) -> dict:
    """Run all three metrics on a pair of matrices.

    `pca_structure_similarity` returns the combined score; we extract
    `subspace_alignment` from its details (Phase 0.6 finding).
    """
    pca = pca_structure_similarity(a, names_a, b, names_b,
                                   n_components=N_PCA_COMPONENTS)
    fc = functional_connectivity_similarity(a, names_a, b, names_b)
    tc = temporal_correlation(a, names_a, b, names_b)

    align = pca.details.get("subspace_alignment", float("nan"))
    return {
        "subspace_alignment": float(align),
        "temporal_correlation": float(tc.score),
        "fc_similarity": float(fc.score),
        "n_matched": int(pca.details.get("n_matched", 0)),
    }


def bootstrap_ci(values: np.ndarray, *, B: int = N_BOOTSTRAP,
                 percentile: float | None = None,
                 ci: tuple[float, float] = (2.5, 97.5),
                 rng: np.random.Generator | None = None):
    """Bootstrap CI on either the mean (default) or a given percentile."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = values.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    estimates = np.empty(B)
    for k in range(B):
        sample = values[rng.integers(0, n, n)]
        if percentile is None:
            estimates[k] = sample.mean()
        else:
            estimates[k] = np.percentile(sample, percentile)
    lo, hi = np.percentile(estimates, ci)
    return float(lo), float(hi)


def describe(values: np.ndarray, *, rng: np.random.Generator | None = None) -> dict:
    """Mean / median / std / 5%/95% percentiles, with 95% bootstrap CIs."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if v.size == 0:
        return {"n": 0}
    if rng is None:
        rng = np.random.default_rng(0)
    mean_ci = bootstrap_ci(v, rng=rng)
    p5_ci = bootstrap_ci(v, percentile=5, rng=rng)
    p95_ci = bootstrap_ci(v, percentile=95, rng=rng)
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "mean_ci95": list(mean_ci),
        "median": float(np.median(v)),
        "std": float(v.std()),
        "p5": float(np.percentile(v, 5)),
        "p5_ci95": list(p5_ci),
        "p95": float(np.percentile(v, 95)),
        "p95_ci95": list(p95_ci),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def percentile_of(value: float, distribution: np.ndarray) -> float:
    """Where does `value` sit in `distribution`? Empirical CDF."""
    d = np.asarray(distribution, dtype=float)
    d = d[~np.isnan(d)]
    if d.size == 0 or np.isnan(value):
        return float("nan")
    return float(100.0 * np.mean(d <= value))


# ---------------------------------------------------------------------------
# Digital simulation
# ---------------------------------------------------------------------------


def run_digital(connectome: ConnectomeData, n_ticks: int, *,
                seed: int) -> tuple[np.ndarray, list[str]]:
    """Bare CTRNN with random Gaussian sensory drive (Phase 0.5 protocol A)."""
    n = connectome.n_neurons
    state = NeuralState.initialize(n, seed=seed)
    params = CTRNNParams()
    rng = np.random.default_rng(seed)
    sens_idx = connectome.get_neuron_indices_by_category("sensory")
    zero = np.zeros(n)
    for _ in range(PRE_EQ_TICKS):
        state = neural_step(state, connectome, zero, params, rng)
    history = np.zeros((n_ticks, n), dtype=np.float32)
    sens = np.zeros(n)
    for t in range(n_ticks):
        sens[:] = 0.0
        sens[sens_idx] = rng.standard_normal(len(sens_idx)) * SENSORY_NOISE
        state = neural_step(state, connectome, sens, params, rng)
        history[t] = state.V
    return history, connectome.neuron_names


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"loading top-{N_WORMS} Atanas 2023 recordings…", flush=True)
    reference = ReferenceDataset.from_atanas2023(max_recordings=N_WORMS)
    recordings = reference.recordings
    print(f"  loaded {len(recordings)} recordings")
    for r in recordings:
        print(f"    {r.recording_id}: T={r.n_timepoints} labeled={r.n_labeled}")

    # Pre-compute the (matrix, names) view of each recording.
    rec_views = [recording_matrix(r) for r in recordings]

    # --- (1) Real cross-worm pairwise ---------------------------------------
    pair_results = []
    pairs = list(combinations(range(len(recordings)), 2))
    print(f"\ncomputing {len(pairs)} cross-worm pairs…", flush=True)
    for i, j in pairs:
        a, na = rec_views[i]
        b, nb = rec_views[j]
        m = metric_triple(a, na, b, nb)
        m["pair"] = [recordings[i].recording_id, recordings[j].recording_id]
        pair_results.append(m)

    # --- (2) Real within-worm split-half ------------------------------------
    self_results = []
    print(f"computing {len(recordings)} split-half self-pairs…", flush=True)
    for r, (mat, names) in zip(recordings, rec_views):
        (a, na), (b, nb) = half_split(mat, names)
        m = metric_triple(a, na, b, nb)
        m["recording_id"] = r.recording_id
        self_results.append(m)

    # --- (3) Digital-vs-real ------------------------------------------------
    print(f"computing {len(recordings)} digital-vs-real comparisons…", flush=True)
    connectome = ConnectomeData.load()
    digital_results = []
    for r, (mat, names) in zip(recordings, rec_views):
        seed = hash(r.recording_id) & 0x7FFFFFFF
        dh, dn = run_digital(connectome, r.n_timepoints, seed=seed)
        m = metric_triple(dh, dn, mat, names)
        m["recording_id"] = r.recording_id
        digital_results.append(m)

    elapsed = time.time() - t0
    print(f"\ntotal runtime: {elapsed:.1f}s")

    # --- Aggregate distributions --------------------------------------------
    metric_keys = ["subspace_alignment", "temporal_correlation", "fc_similarity"]

    def collect(rs: list[dict], key: str) -> np.ndarray:
        return np.array([r[key] for r in rs], dtype=float)

    rng_boot = np.random.default_rng(12345)
    distributions = {}
    for cat, rs in [("cross_worm", pair_results),
                    ("within_worm_self", self_results),
                    ("digital_vs_real", digital_results)]:
        distributions[cat] = {}
        for m in metric_keys:
            distributions[cat][m] = describe(collect(rs, m), rng=rng_boot)

    # Where does the digital_vs_real sit in the real distributions?
    digital_position = {}
    for m in metric_keys:
        dvr = collect(digital_results, m)
        cross = collect(pair_results, m)
        self_arr = collect(self_results, m)
        digital_position[m] = {
            "digital_mean": float(np.mean(dvr)),
            "digital_median": float(np.median(dvr)),
            "digital_min": float(np.min(dvr)),
            "digital_max": float(np.max(dvr)),
            "percentile_in_cross_worm": percentile_of(np.mean(dvr), cross),
            "percentile_in_within_worm": percentile_of(np.mean(dvr), self_arr),
            "cross_worm_mean": float(np.mean(cross)),
            "within_worm_mean": float(np.mean(self_arr)),
            "gap_to_cross_worm_p5": float(np.percentile(cross, 5) - np.mean(dvr)),
            "gap_to_cross_worm_mean": float(np.mean(cross) - np.mean(dvr)),
        }

    # --- JSON output --------------------------------------------------------
    json_path = OUTPUT_DIR / "individual_variation_results.json"
    payload = {
        "settings": {
            "n_worms": N_WORMS,
            "n_bootstrap": N_BOOTSTRAP,
            "pca_components": N_PCA_COMPONENTS,
            "pre_eq_ticks": PRE_EQ_TICKS,
            "sensory_noise": SENSORY_NOISE,
        },
        "recordings": [
            {"recording_id": r.recording_id,
             "n_timepoints": r.n_timepoints,
             "n_labeled": r.n_labeled}
            for r in recordings
        ],
        "distributions": distributions,
        "digital_position": digital_position,
        "trials": {
            "cross_worm": pair_results,
            "within_worm_self": self_results,
            "digital_vs_real": digital_results,
        },
        "runtime_sec": elapsed,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    # --- Text report --------------------------------------------------------
    report_path = OUTPUT_DIR / "individual_variation_report.txt"
    with report_path.open("w") as f:
        f.write("Phase 0.7 — individual-variation baseline\n")
        f.write("=" * 78 + "\n")
        f.write(f"recordings: {len(recordings)}\n")
        f.write(f"cross-worm pairs: {len(pairs)}\n")
        f.write(f"runtime: {elapsed:.1f}s\n\n")

        for m in metric_keys:
            f.write(f"Metric: {m}\n")
            f.write("-" * 78 + "\n")
            f.write(f"  {'condition':18s}  {'n':>3s} {'mean':>9s}±{'sd':>5s}   "
                    f"{'95%CI(mean)':>17s}   "
                    f"{'p5':>7s}   {'p95':>7s}\n")
            for cat in ("within_worm_self", "cross_worm", "digital_vs_real"):
                d = distributions[cat][m]
                f.write(
                    f"  {cat:18s}  {d['n']:3d} "
                    f"{d['mean']:+8.4f}±{d['std']:.3f}   "
                    f"[{d['mean_ci95'][0]:+.4f}, {d['mean_ci95'][1]:+.4f}]   "
                    f"{d['p5']:+6.3f}   {d['p95']:+6.3f}\n"
                )
            dp = digital_position[m]
            f.write(
                f"  → digital vs cross-worm: digital_mean={dp['digital_mean']:+.4f}  "
                f"cross_worm_p5={distributions['cross_worm'][m]['p5']:+.4f}  "
                f"gap_to_p5={dp['gap_to_cross_worm_p5']:+.4f}  "
                f"percentile={dp['percentile_in_cross_worm']:.1f}%\n\n"
            )

    print(f"wrote {report_path}")
    print(f"wrote {json_path}\n")
    with report_path.open() as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
