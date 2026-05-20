"""Phase 0.9 — category baseline vs RID-modulated network.

Comparison on the same 10 best-labeled Atanas 2023 recordings used in
Phase 0.7/0.8, with stable seeds (`1000 + index`) per Phase 0.8.3:

  A. **category** — Phase 0.8.2 defaults (fast_filter / integrator /
     slow_persistent / ctrnn_default by category). This is the Phase 0.9
     baseline since the brief stipulates we add the modulator on top of
     the 0.8.2 architecture.
  B. **rid_gain_0.5** — same network, plus the default RID modulator
     (tau=200, gain=0.5).
  C. **rid_gain_0.2** — sensitivity sweep, weaker modulation.
  D. **rid_gain_1.0** — sensitivity sweep, stronger modulation.

Metrics: subspace_alignment, temporal_correlation, fc_similarity.
Outputs: output/phase0_9_comparison_{report.txt,results.json}.

Per the brief §5 (engineering discipline): we do **not** tune gain to
make P1 hold; we run a small sweep (0.2 / 0.5 / 1.0) to characterize the
parameter neighborhood and report all three. The default-gain (0.5)
number is the headline.
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
    RIDModulator,
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


def simulate_network(
    network: HeterogeneousNetwork,
    n_ticks: int,
    *,
    seed: int,
    modulator: RIDModulator | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Run the network for `n_ticks` with random sensory drive.

    If `modulator` is provided it is reset() at the start so each recording
    starts from c_RID = 0 (clean comparison across recordings).
    """
    n = network.connectome.n_neurons
    state = network.initial_state(seed=seed)
    rng = np.random.default_rng(seed)
    sens_idx = network.connectome.get_neuron_indices_by_category("sensory")
    if modulator is not None:
        modulator.reset()
    zero = np.zeros(n)
    for _ in range(PRE_EQ_TICKS):
        state = network.step(state, zero, rng, modulator=modulator)
    history = np.zeros((n_ticks, n), dtype=np.float32)
    sens = np.zeros(n)
    for t in range(n_ticks):
        sens[:] = 0.0
        sens[sens_idx] = rng.standard_normal(len(sens_idx)) * SENSORY_NOISE
        state = network.step(state, sens, rng, modulator=modulator)
        history[t] = state.V
    return history, network.connectome.neuron_names


def metric_triple(digital, digital_names, real, real_names):
    pca = pca_structure_similarity(
        digital, digital_names, real, real_names,
        n_components=N_PCA_COMPONENTS,
    )
    fc = functional_connectivity_similarity(
        digital, digital_names, real, real_names
    )
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

    # Build one shared network (category defaults) and three modulators.
    net = from_category_defaults(connectome)
    groups = {k: len(v) for k, v in net.function_groups.items()}
    print(f"  category groups: {groups}")

    configurations: dict[str, RIDModulator | None] = {
        "category":      None,
        "rid_gain_0.2":  RIDModulator.from_connectome(connectome, tau=200.0, gain=0.2),
        "rid_gain_0.5":  RIDModulator.from_connectome(connectome, tau=200.0, gain=0.5),
        "rid_gain_1.0":  RIDModulator.from_connectome(connectome, tau=200.0, gain=1.0),
    }
    print(f"  configurations: {list(configurations)}")

    real_views = []
    for r in recordings:
        cols, names = [], []
        for c, n in r.labels.items():
            cols.append(c)
            names.append(n)
        real_views.append((r.traces[:, cols], names))

    results = {k: [] for k in configurations}
    for label, modulator in configurations.items():
        print(f"\nrunning {label} sims…", flush=True)
        for k_idx, (r, (real_mat, real_names)) in enumerate(
            zip(recordings, real_views)
        ):
            seed = 1000 + k_idx
            digital, digital_names = simulate_network(
                net, r.n_timepoints, seed=seed, modulator=modulator
            )
            m = metric_triple(digital, digital_names, real_mat, real_names)
            m["recording_id"] = r.recording_id
            m["digital_max_abs_V"] = float(np.max(np.abs(digital)))
            m["digital_mean_abs_V"] = float(np.mean(np.abs(digital)))
            if modulator is not None:
                m["final_c_RID"] = float(modulator.c_RID)
            results[label].append(m)

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
    cross_worm: dict = {}
    cross_pair_arrays: dict = {}
    try:
        with open(OUTPUT_DIR / "individual_variation_results.json") as f:
            iv = json.load(f)
        for m in metric_keys:
            cross_worm[m] = iv["distributions"]["cross_worm"][m]
            cross_pair_arrays[m] = np.array(
                [t[m] for t in iv["trials"]["cross_worm"]]
            )
    except FileNotFoundError:
        print("warning: Phase 0.7 baseline JSON not found.")

    def percentile_in(value, dist_array):
        d = np.asarray(dist_array, dtype=float)
        d = d[~np.isnan(d)]
        if d.size == 0:
            return float("nan")
        return float(100.0 * np.mean(d <= value))

    json_path = OUTPUT_DIR / "phase0_9_comparison_results.json"
    json_path.write_text(json.dumps({
        "settings": {
            "n_worms": N_WORMS,
            "pre_eq_ticks": PRE_EQ_TICKS,
            "sensory_noise": SENSORY_NOISE,
            "tau_rid": 200.0,
            "gains_tested": [0.2, 0.5, 1.0],
        },
        "aggregate": aggregate,
        "per_recording": results,
        "cross_worm_baseline": cross_worm,
        "runtime_sec": elapsed,
    }, indent=2))

    report_path = OUTPUT_DIR / "phase0_9_comparison_report.txt"
    with report_path.open("w") as f:
        f.write("Phase 0.9 — category baseline vs RID-modulated networks\n")
        f.write("=" * 78 + "\n")
        f.write(f"recordings: {N_WORMS}\n")
        f.write(f"runtime:    {elapsed:.1f}s\n")
        f.write(f"tau_rid:    200.0\n")
        f.write(f"gains:      0.2, 0.5, 1.0  (default is 0.5)\n\n")

        f.write("Mean digital-vs-real score (n=10):\n")
        f.write(
            f"  {'metric':30s}  {'category':>10s}  "
            f"{'g=0.2':>10s}  {'g=0.5':>10s}  {'g=1.0':>10s}  "
            f"{'Δ(g=0.5 vs cat)':>16s}\n"
        )
        for m in metric_keys:
            c = aggregate["category"][m]["mean"]
            v2 = aggregate["rid_gain_0.2"][m]["mean"]
            v5 = aggregate["rid_gain_0.5"][m]["mean"]
            v10 = aggregate["rid_gain_1.0"][m]["mean"]
            f.write(
                f"  {m:30s}  {c:+10.4f}  "
                f"{v2:+10.4f}  {v5:+10.4f}  {v10:+10.4f}  "
                f"{v5 - c:+16.4f}\n"
            )
        f.write("\n")

        if cross_worm:
            f.write("Percentile in cross-worm distribution (Phase 0.7 baseline):\n")
            f.write(
                f"  {'metric':30s}  {'cross_worm':>22s}  "
                f"cat%   g0.2%  g0.5%  g1.0%\n"
            )
            for m in metric_keys:
                cw_mean = cross_worm[m]["mean"]
                cw_p5 = cross_worm[m]["p5"]
                cw_str = f"mean={cw_mean:+.3f}, p5={cw_p5:+.3f}"
                c_pct = percentile_in(
                    aggregate["category"][m]["mean"], cross_pair_arrays[m]
                )
                p2 = percentile_in(
                    aggregate["rid_gain_0.2"][m]["mean"], cross_pair_arrays[m]
                )
                p5 = percentile_in(
                    aggregate["rid_gain_0.5"][m]["mean"], cross_pair_arrays[m]
                )
                p10 = percentile_in(
                    aggregate["rid_gain_1.0"][m]["mean"], cross_pair_arrays[m]
                )
                f.write(
                    f"  {m:30s}  {cw_str:>22s}  "
                    f"{c_pct:>4.1f}%  {p2:>4.1f}%  {p5:>4.1f}%  {p10:>4.1f}%\n"
                )
        f.write("\n")

        # P1 verdict — explicit, per brief §8 question 1.
        delta_fc = (
            aggregate["rid_gain_0.5"]["fc_similarity"]["mean"]
            - aggregate["category"]["fc_similarity"]["mean"]
        )
        f.write("P1 verdict (default gain 0.5):\n")
        f.write(f"  Δfc_similarity = {delta_fc:+.4f}\n")
        if delta_fc >= 0.10:
            verdict = "SUPPORTED   (Δ ≥ +0.10)"
        elif delta_fc >= 0.03:
            verdict = "PARTIAL     (+0.03 ≤ Δ < +0.10)"
        else:
            verdict = "UNSUPPORTED (Δ < +0.03)"
        f.write(f"  P1: {verdict}\n\n")

        f.write("Per-recording detail:\n")
        for r in recordings:
            f.write(f"--- {r.recording_id} (labeled={r.n_labeled}) ---\n")
            row = {
                lbl: next(x for x in results[lbl]
                          if x["recording_id"] == r.recording_id)
                for lbl in results
            }
            for m in metric_keys:
                f.write(
                    f"  {m:30s}  "
                    f"cat={row['category'][m]:+.4f}  "
                    f"g0.2={row['rid_gain_0.2'][m]:+.4f}  "
                    f"g0.5={row['rid_gain_0.5'][m]:+.4f}  "
                    f"g1.0={row['rid_gain_1.0'][m]:+.4f}\n"
                )
            f.write(
                f"  final c_RID (g=0.5): {row['rid_gain_0.5']['final_c_RID']:+.4f}\n\n"
            )

    print(f"wrote {report_path}")
    print(f"wrote {json_path}\n")
    with report_path.open() as f:
        for _, line in zip(range(60), f):
            print(line, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
