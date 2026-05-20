"""Phase 0.8.2 — heterogeneous vs homogeneous on the Phase 0.7 metrics.

Compares two digital network configurations on the same 10 Atanas 2023
recordings:

  A. **homogeneous** — all neurons use `ctrnn_default` (= Phase 0.7's
     `neural_step`, verified bit-equivalent in Phase 0.8.1).
  B. **category heterogeneous** — sensory → fast_filter, interneuron →
     integrator, motor → slow_persistent, pharyngeal → ctrnn_default
     (the `DEFAULT_CATEGORY_ASSIGNMENT` from `step_library.py`).

For each recording and each configuration, we compute:
  - subspace_alignment
  - temporal_correlation
  - functional_connectivity_similarity

We then compare the two configurations' digital-vs-real scores and
report where each lands relative to the Phase 0.7 cross-worm baseline
distribution (from `output/individual_variation_results.json`).

Outputs:
  output/phase0_8_2_comparison_report.txt
  output/phase0_8_2_comparison_results.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from algos.config import OUTPUT_DIR
from algos.connectome import ConnectomeData
from algos.neural import (
    HeterogeneousNetwork,
    from_category_defaults,
)
from algos.validation.comparison import (
    functional_connectivity_similarity,
    pca_structure_similarity,
    temporal_correlation,
)
from algos.validation.reference_data import ReferenceDataset


N_WORMS = 10
PRE_EQ_TICKS = 2000
SENSORY_NOISE = 0.1
N_PCA_COMPONENTS = 10


def simulate_network(network: HeterogeneousNetwork, n_ticks: int, *,
                     seed: int) -> tuple[np.ndarray, list[str]]:
    """Run a heterogeneous network for `n_ticks` with random sensory drive."""
    n = network.connectome.n_neurons
    state = network.initial_state(seed=seed)
    rng = np.random.default_rng(seed)
    sens_idx = network.connectome.get_neuron_indices_by_category("sensory")
    zero = np.zeros(n)
    for _ in range(PRE_EQ_TICKS):
        state = network.step(state, zero, rng)
    history = np.zeros((n_ticks, n), dtype=np.float32)
    sens = np.zeros(n)
    for t in range(n_ticks):
        sens[:] = 0.0
        sens[sens_idx] = rng.standard_normal(len(sens_idx)) * SENSORY_NOISE
        state = network.step(state, sens, rng)
        history[t] = state.V
    return history, network.connectome.neuron_names


def metric_triple(digital, digital_names, real, real_names):
    pca = pca_structure_similarity(digital, digital_names, real, real_names,
                                   n_components=N_PCA_COMPONENTS)
    fc = functional_connectivity_similarity(digital, digital_names, real,
                                            real_names)
    tc = temporal_correlation(digital, digital_names, real, real_names)
    return {
        "subspace_alignment":
            float(pca.details.get("subspace_alignment", float("nan"))),
        "temporal_correlation": float(tc.score),
        "fc_similarity": float(fc.score),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"loading top-{N_WORMS} Atanas 2023 recordings…", flush=True)
    reference = ReferenceDataset.from_atanas2023(max_recordings=N_WORMS)
    recordings = reference.recordings
    connectome = ConnectomeData.load()

    # Build the two networks once.
    net_hom = HeterogeneousNetwork.from_homogeneous_ctrnn(connectome)
    net_het = from_category_defaults(connectome)
    print("homogeneous group counts:", {k: len(v) for k, v in
                                         net_hom.function_groups.items()})
    print("heterogeneous group counts:", {k: len(v) for k, v in
                                           net_het.function_groups.items()})

    # Per-recording real matrices.
    real_views = []
    for r in recordings:
        cols, names = [], []
        for c, n in r.labels.items():
            cols.append(c)
            names.append(n)
        real_views.append((r.traces[:, cols], names))

    results = {"homogeneous": [], "heterogeneous": []}

    for label, net in [("homogeneous", net_hom), ("heterogeneous", net_het)]:
        print(f"\nrunning {label} sims…")
        for r, (real_mat, real_names) in zip(recordings, real_views):
            seed = hash(r.recording_id) & 0x7FFFFFFF
            digital, digital_names = simulate_network(
                net, r.n_timepoints, seed=seed
            )
            metrics = metric_triple(digital, digital_names, real_mat, real_names)
            metrics["recording_id"] = r.recording_id
            metrics["digital_max_abs_V"] = float(np.max(np.abs(digital)))
            metrics["digital_mean_abs_V"] = float(np.mean(np.abs(digital)))
            results[label].append(metrics)

    elapsed = time.time() - t0
    print(f"\ntotal runtime: {elapsed:.1f}s")

    # --- Aggregate -----------------------------------------------------------
    metric_keys = ["subspace_alignment", "temporal_correlation", "fc_similarity"]
    aggregate = {}
    for cond, rs in results.items():
        aggregate[cond] = {}
        for m in metric_keys:
            v = np.array([r[m] for r in rs])
            aggregate[cond][m] = {
                "n": int(v.size),
                "mean": float(v.mean()),
                "std": float(v.std()),
                "min": float(v.min()),
                "max": float(v.max()),
            }

    # --- Load Phase 0.7 cross-worm baselines for percentile context --------
    cross_worm_baseline = {}
    try:
        with open(OUTPUT_DIR / "individual_variation_results.json") as f:
            iv = json.load(f)
        for m in metric_keys:
            cross_worm_baseline[m] = iv["distributions"]["cross_worm"][m]
    except FileNotFoundError:
        print("warning: Phase 0.7 baseline JSON not found.")

    # Where do hom/het lands relative to cross-worm distribution?
    def percentile_in(value, dist_array):
        d = np.asarray(dist_array, dtype=float)
        d = d[~np.isnan(d)]
        if d.size == 0:
            return float("nan")
        return float(100.0 * np.mean(d <= value))

    pair_results = {"homogeneous": [], "heterogeneous": []}
    # Recover raw cross_worm distribution from the same JSON.
    try:
        cross_pair_arrays = {m: np.array([t[m] for t in iv["trials"]["cross_worm"]])
                             for m in metric_keys}
    except (KeyError, NameError):
        cross_pair_arrays = {}

    summary = {}
    for cond in results:
        summary[cond] = {}
        for m in metric_keys:
            mean = aggregate[cond][m]["mean"]
            pct = (percentile_in(mean, cross_pair_arrays[m])
                   if m in cross_pair_arrays else float("nan"))
            summary[cond][m] = {
                "mean": mean,
                "std": aggregate[cond][m]["std"],
                "percentile_in_cross_worm": pct,
                "cross_worm_mean": cross_worm_baseline.get(m, {}).get("mean"),
                "cross_worm_p5": cross_worm_baseline.get(m, {}).get("p5"),
            }

    # --- JSON output --------------------------------------------------------
    json_path = OUTPUT_DIR / "phase0_8_2_comparison_results.json"
    json_path.write_text(json.dumps({
        "settings": {
            "n_worms": N_WORMS,
            "pre_eq_ticks": PRE_EQ_TICKS,
            "sensory_noise": SENSORY_NOISE,
        },
        "summary": summary,
        "per_recording": results,
        "runtime_sec": elapsed,
    }, indent=2))

    # --- Text report --------------------------------------------------------
    report_path = OUTPUT_DIR / "phase0_8_2_comparison_report.txt"
    with report_path.open("w") as f:
        f.write("Phase 0.8.2 — homogeneous vs category-heterogeneous\n")
        f.write("=" * 78 + "\n")
        f.write(f"recordings:  {N_WORMS}\n")
        f.write(f"runtime:     {elapsed:.1f}s\n\n")

        f.write("Mean digital-vs-real score (n=10 recordings):\n")
        f.write(f"  {'metric':30s}  {'homogeneous':>14s}   {'heterogeneous':>14s}   "
                f"{'delta':>8s}\n")
        for m in metric_keys:
            h = summary["homogeneous"][m]["mean"]
            x = summary["heterogeneous"][m]["mean"]
            f.write(f"  {m:30s}  {h:+14.4f}   {x:+14.4f}   {x-h:+8.4f}\n")
        f.write("\n")

        f.write("Where they sit in the cross-worm distribution:\n")
        f.write(f"  {'metric':30s}  cross-worm  hom%   het%\n")
        for m in metric_keys:
            cw_mean = summary["homogeneous"][m]["cross_worm_mean"]
            cw_p5 = summary["homogeneous"][m]["cross_worm_p5"]
            h_pct = summary["homogeneous"][m]["percentile_in_cross_worm"]
            x_pct = summary["heterogeneous"][m]["percentile_in_cross_worm"]
            cw_str = (f"mean={cw_mean:+.3f}, p5={cw_p5:+.3f}"
                      if cw_mean is not None else "—")
            f.write(f"  {m:30s}  {cw_str:30s}  {h_pct:>4.1f}%  {x_pct:>4.1f}%\n")
        f.write("\n")

        f.write("Per-recording detail (digital_vs_real for each metric):\n")
        for r in recordings:
            f.write(f"--- {r.recording_id} (labeled={r.n_labeled}) ---\n")
            h_m = next(x for x in results["homogeneous"]
                       if x["recording_id"] == r.recording_id)
            x_m = next(x for x in results["heterogeneous"]
                       if x["recording_id"] == r.recording_id)
            for m in metric_keys:
                f.write(f"  {m:30s}  hom={h_m[m]:+.4f}   het={x_m[m]:+.4f}   "
                        f"Δ={x_m[m]-h_m[m]:+.4f}\n")
            f.write("\n")

    print(f"wrote {report_path}")
    print(f"wrote {json_path}\n")
    with report_path.open() as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
