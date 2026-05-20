"""Phase 0.8.3 — homogeneous vs category vs key-neuron specialized.

Three-way comparison on the same 10 best-labeled Atanas 2023 recordings:

  A. **homogeneous** — all neurons ctrnn_default (= Phase 0.7).
  B. **category** — Phase 0.8.2 defaults (fast_filter / integrator /
     slow_persistent / ctrnn_default by category).
  C. **key-neuron** — Phase 0.8.3: B plus per-neuron overrides for
     ASE, AFD, AVA/AVD, AVB/PVC, RIM (14 specialized neurons).

Metrics: subspace_alignment, temporal_correlation, fc_similarity.
Outputs: output/phase0_8_3_comparison_{report.txt,results.json}.
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
    from_key_neuron_specialization,
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


def simulate_network(network, n_ticks: int, *, seed: int):
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
    fc = functional_connectivity_similarity(digital, digital_names, real, real_names)
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

    # Build the three networks once.
    networks = {
        "homogeneous":  HeterogeneousNetwork.from_homogeneous_ctrnn(connectome),
        "category":     from_category_defaults(connectome),
        "key_neuron":   from_key_neuron_specialization(connectome),
    }
    for label, net in networks.items():
        groups = {k: len(v) for k, v in net.function_groups.items()}
        print(f"  {label:12s} groups: {groups}")

    # Per-recording real matrices.
    real_views = []
    for r in recordings:
        cols, names = [], []
        for c, n in r.labels.items():
            cols.append(c)
            names.append(n)
        real_views.append((r.traces[:, cols], names))

    results = {k: [] for k in networks}
    for label, net in networks.items():
        print(f"\nrunning {label} sims…", flush=True)
        # Stable seeds: enumerate index gives reproducible numbers across runs
        # (Python's hash() is PYTHONHASHSEED-randomized per-process).
        for k_idx, (r, (real_mat, real_names)) in enumerate(zip(recordings, real_views)):
            seed = 1000 + k_idx
            digital, digital_names = simulate_network(net, r.n_timepoints, seed=seed)
            metrics = metric_triple(digital, digital_names, real_mat, real_names)
            metrics["recording_id"] = r.recording_id
            metrics["digital_max_abs_V"] = float(np.max(np.abs(digital)))
            metrics["digital_mean_abs_V"] = float(np.mean(np.abs(digital)))
            results[label].append(metrics)

    elapsed = time.time() - t0
    print(f"\ntotal runtime: {elapsed:.1f}s")

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

    # Phase 0.7 baselines.
    cross_worm = {}
    cross_pair_arrays = {}
    try:
        with open(OUTPUT_DIR / "individual_variation_results.json") as f:
            iv = json.load(f)
        for m in metric_keys:
            cross_worm[m] = iv["distributions"]["cross_worm"][m]
            cross_pair_arrays[m] = np.array([t[m] for t in iv["trials"]["cross_worm"]])
    except FileNotFoundError:
        print("warning: Phase 0.7 baseline JSON not found.")

    def percentile_in(value, dist_array):
        d = np.asarray(dist_array, dtype=float)
        d = d[~np.isnan(d)]
        if d.size == 0:
            return float("nan")
        return float(100.0 * np.mean(d <= value))

    # --- JSON output --------------------------------------------------------
    json_path = OUTPUT_DIR / "phase0_8_3_comparison_results.json"
    json_path.write_text(json.dumps({
        "settings": {
            "n_worms": N_WORMS,
            "pre_eq_ticks": PRE_EQ_TICKS,
            "sensory_noise": SENSORY_NOISE,
        },
        "aggregate": aggregate,
        "per_recording": results,
        "cross_worm_baseline": cross_worm,
        "runtime_sec": elapsed,
    }, indent=2))

    # --- Text report --------------------------------------------------------
    report_path = OUTPUT_DIR / "phase0_8_3_comparison_report.txt"
    with report_path.open("w") as f:
        f.write("Phase 0.8.3 — homogeneous / category / key-neuron specialized\n")
        f.write("=" * 78 + "\n")
        f.write(f"recordings: {N_WORMS}\n")
        f.write(f"runtime:    {elapsed:.1f}s\n\n")

        f.write("Mean digital-vs-real score (n=10):\n")
        f.write(f"  {'metric':30s}  {'homogeneous':>11s}  {'category':>10s}  "
                f"{'key_neuron':>11s}  {'Δ(key vs cat)':>14s}\n")
        for m in metric_keys:
            h = aggregate["homogeneous"][m]["mean"]
            c = aggregate["category"][m]["mean"]
            k = aggregate["key_neuron"][m]["mean"]
            f.write(f"  {m:30s}  {h:+11.4f}  {c:+10.4f}  {k:+11.4f}  "
                    f"{k - c:+14.4f}\n")
        f.write("\n")

        if cross_worm:
            f.write("Percentile in cross-worm distribution (Phase 0.7 baseline):\n")
            f.write(f"  {'metric':30s}  {'cross_worm':>22s}  hom%   cat%   key%\n")
            for m in metric_keys:
                cw_mean = cross_worm[m]["mean"]
                cw_p5 = cross_worm[m]["p5"]
                cw_str = f"mean={cw_mean:+.3f}, p5={cw_p5:+.3f}"
                h_pct = percentile_in(aggregate["homogeneous"][m]["mean"],
                                       cross_pair_arrays[m])
                c_pct = percentile_in(aggregate["category"][m]["mean"],
                                       cross_pair_arrays[m])
                k_pct = percentile_in(aggregate["key_neuron"][m]["mean"],
                                       cross_pair_arrays[m])
                f.write(f"  {m:30s}  {cw_str:>22s}  {h_pct:>4.1f}%  "
                        f"{c_pct:>4.1f}%  {k_pct:>4.1f}%\n")
        f.write("\n")

        f.write("Per-recording detail:\n")
        for r in recordings:
            f.write(f"--- {r.recording_id} (labeled={r.n_labeled}) ---\n")
            h_m = next(x for x in results["homogeneous"]
                       if x["recording_id"] == r.recording_id)
            c_m = next(x for x in results["category"]
                       if x["recording_id"] == r.recording_id)
            k_m = next(x for x in results["key_neuron"]
                       if x["recording_id"] == r.recording_id)
            for m in metric_keys:
                f.write(f"  {m:30s}  hom={h_m[m]:+.4f}  cat={c_m[m]:+.4f}  "
                        f"key={k_m[m]:+.4f}\n")
            f.write("\n")

    print(f"wrote {report_path}")
    print(f"wrote {json_path}\n")
    with report_path.open() as f:
        # Print first 30 lines of report for visibility.
        for _, line in zip(range(40), f):
            print(line, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
