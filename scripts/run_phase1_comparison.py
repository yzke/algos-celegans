"""Phase 1.0.5 — full validation against Atanas 2023.

Mirrors scripts/run_phase0_9_comparison.py for the new graph-native
runtime so the headline numbers are directly comparable.

For each of the top-10 best-labeled Atanas 2023 recordings, runs the
GraphSimulator for r.n_timepoints ticks (seed = 1000 + idx), extracts
the per-neuron rate trace, and computes the three Phase 0.7+ metrics
against the recording:

  - subspace_alignment   (pca_structure_similarity.details)
  - temporal_correlation
  - fc_similarity

Two configurations per recording:

  A. baseline_v2 — LIF + subgraphs (no plasticity, no modulators).
  B. full_v2     — baseline + Hebbian (100 plastic edges) + RID + 5-HT.

Outputs phase1_comparison_{results.json, report.txt} into
output/phase1.0/. The report writes a side-by-side block against
Phase 0.9's headline numbers from PHASE0.9_REPORT.md.
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
from algos.graph import build_canonical_subgraphs, load_connectome_into_graph
from algos.neural_v2 import (
    GraphSimulator,
    HebbianRule,
    SimulatorConfig,
    build_default_modulator_bank,
)
from algos.validation.comparison import (
    functional_connectivity_similarity,
    pca_structure_similarity,
    temporal_correlation,
)
from algos.validation.reference_data import ReferenceDataset


N_WORMS = 10
PRE_EQ_TICKS = 2000
SENSORY_NOISE = 0.2
NOISE_LEVEL = 0.005
N_PCA_COMPONENTS = 10


def simulate_phase1(
    config_name: str,
    *,
    n_ticks: int,
    seed: int,
) -> tuple[np.ndarray, list[str], dict]:
    """Run the Phase 1.0 simulator for n_ticks. Returns rate trace + names + diagnostics."""
    g = load_connectome_into_graph()
    build_canonical_subgraphs(g)
    sim = GraphSimulator(
        g, config=SimulatorConfig(noise_level=NOISE_LEVEL, sensory_noise=SENSORY_NOISE),
    )
    rule = None
    bank = None
    if "plast" in config_name:
        rule = HebbianRule.from_graph(g)
        sim.attach_plasticity(rule)
    if "mod" in config_name:
        bank = build_default_modulator_bank(g, sim.params.threshold)
        sim.attach_modulators(bank)

    state = sim.initial_state(seed=seed)
    rng = np.random.default_rng(seed)
    sens = np.zeros(sim.n)
    for _ in range(PRE_EQ_TICKS):
        sens[:] = 0.0
        sens[sim.sensory_idx] = (
            rng.standard_normal(sim.sensory_idx.size) * SENSORY_NOISE
        )
        state = sim.step(state, sens, rng)

    rate_hist = np.zeros((n_ticks, sim.n), dtype=np.float32)
    for t in range(n_ticks):
        sens[:] = 0.0
        sens[sim.sensory_idx] = (
            rng.standard_normal(sim.sensory_idx.size) * SENSORY_NOISE
        )
        state = sim.step(state, sens, rng)
        rate_hist[t] = state.rate

    diag = {
        "total_spikes": int(state.spike_count.sum()),
        "active_neurons": int((state.spike_count > 0).sum()),
        "max_rate": float(state.rate.max()),
    }
    if rule is not None:
        diag.update({f"plast_{k}": v for k, v in rule.weight_summary().items()})
    if bank is not None:
        diag["c_RID_final"] = float(bank.modulators[0].c_m)
        diag["c_5HT_final"] = float(bank.modulators[1].c_m)

    return rate_hist, sim.graph.neuron_names(), diag


def metric_triple(digital, digital_names, real, real_names):
    pca = pca_structure_similarity(
        digital, digital_names, real, real_names,
        n_components=N_PCA_COMPONENTS,
    )
    fc = functional_connectivity_similarity(
        digital, digital_names, real, real_names,
    )
    tc = temporal_correlation(
        digital, digital_names, real, real_names,
    )
    return {
        "subspace_alignment":
            float(pca.details.get("subspace_alignment", float("nan"))),
        "temporal_correlation": float(tc.score),
        "fc_similarity": float(fc.score),
        "n_matched": int(pca.details.get("n_matched", 0)),
    }


def main() -> int:
    phase1_dir = OUTPUT_DIR / "phase1.0"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading top-{N_WORMS} Atanas 2023 recordings…")
    reference = ReferenceDataset.from_atanas2023(max_recordings=N_WORMS)
    recordings = reference.recordings
    real_views = []
    for r in recordings:
        cols, names = [], []
        for c, n in r.labels.items():
            cols.append(c)
            names.append(n)
        real_views.append((r.traces[:, cols], names))

    configurations = ("baseline_v2", "plast_mod_v2")
    results: dict[str, list[dict]] = {cn: [] for cn in configurations}

    t_start = time.time()
    for cn in configurations:
        print(f"\n=== {cn} ===", flush=True)
        for k_idx, (r, (real_mat, real_names)) in enumerate(
            zip(recordings, real_views)
        ):
            seed = 1000 + k_idx
            t0 = time.time()
            rate_hist, dig_names, diag = simulate_phase1(
                cn, n_ticks=r.n_timepoints, seed=seed,
            )
            m = metric_triple(rate_hist, dig_names, real_mat, real_names)
            m["recording_id"] = r.recording_id
            m["seed"] = seed
            m["runtime_s"] = time.time() - t0
            m.update(diag)
            results[cn].append(m)
            print(
                f"  rec={r.recording_id[:16]:16s} "
                f"runtime={m['runtime_s']:5.1f}s  "
                f"matched={m['n_matched']:3d}  "
                f"sub={m['subspace_alignment']:+.4f}  "
                f"tc={m['temporal_correlation']:+.4f}  "
                f"fc={m['fc_similarity']:+.4f}",
                flush=True,
            )

    elapsed = time.time() - t_start
    print(f"\ntotal runtime: {elapsed:.1f}s")

    metric_keys = ("subspace_alignment", "temporal_correlation", "fc_similarity")
    aggregate = {}
    for cn, rs in results.items():
        aggregate[cn] = {}
        for m in metric_keys:
            v = np.array([r[m] for r in rs])
            aggregate[cn][m] = {
                "n": int(v.size),
                "mean": float(v.mean()),
                "std": float(v.std()),
                "min": float(v.min()),
                "max": float(v.max()),
            }

    # Phase 0.9 reference numbers (from PHASE0.9_REPORT.md).
    phase09_ref = {
        "category": {
            "subspace_alignment":   +0.3532,
            "temporal_correlation": -0.0140,
            "fc_similarity":        +0.0606,
        },
        "rid_gain_0.5": {
            "subspace_alignment":   +0.3526,
            "temporal_correlation": -0.0138,
            "fc_similarity":        +0.0616,
        },
    }

    out_json = phase1_dir / "phase1_comparison_results.json"
    out_json.write_text(json.dumps({
        "n_worms": N_WORMS,
        "pre_eq_ticks": PRE_EQ_TICKS,
        "sensory_noise": SENSORY_NOISE,
        "noise_level": NOISE_LEVEL,
        "aggregate": aggregate,
        "per_recording": results,
        "phase09_reference": phase09_ref,
        "total_runtime_s": elapsed,
    }, indent=2))

    out_txt = phase1_dir / "phase1_comparison_report.txt"
    with out_txt.open("w") as f:
        f.write("Phase 1.0.5 — graph-native simulator vs Atanas 2023\n")
        f.write("=" * 78 + "\n")
        f.write(f"recordings:      {N_WORMS}\n")
        f.write(f"pre_eq_ticks:    {PRE_EQ_TICKS}\n")
        f.write(f"sensory_noise:   {SENSORY_NOISE}\n")
        f.write(f"runtime:         {elapsed:.1f}s\n\n")

        f.write("Mean digital-vs-real score (n=10):\n\n")
        f.write(
            f"  {'metric':28s}  {'Phase 0.9 cat':>14s}  "
            f"{'Phase 0.9 +RID':>15s}  {'1.0 baseline':>13s}  "
            f"{'1.0 full':>10s}  {'Δ (1.0_full − 0.9_cat)':>22s}\n"
        )
        for m in metric_keys:
            cat09 = phase09_ref["category"][m]
            rid09 = phase09_ref["rid_gain_0.5"][m]
            base10 = aggregate["baseline_v2"][m]["mean"]
            full10 = aggregate["plast_mod_v2"][m]["mean"]
            f.write(
                f"  {m:28s}  {cat09:+14.4f}  {rid09:+15.4f}  "
                f"{base10:+13.4f}  {full10:+10.4f}  {full10 - cat09:+22.4f}\n"
            )
        f.write("\n")

        f.write("Per-recording detail (Phase 1.0 plast_mod_v2):\n")
        for r, m in zip(recordings, results["plast_mod_v2"]):
            row = m
            f.write(
                f"  {r.recording_id:24s} labeled={r.n_labeled:3d} "
                f"matched={row['n_matched']:3d} | "
                f"sub={row['subspace_alignment']:+.4f}  "
                f"tc={row['temporal_correlation']:+.4f}  "
                f"fc={row['fc_similarity']:+.4f}  "
                f"spikes={row['total_spikes']:>5d}\n"
            )
        f.write("\n")

        f.write("Baseline vs full (plasticity + modulators) deltas, per recording:\n")
        for r, b, fu in zip(recordings, results["baseline_v2"], results["plast_mod_v2"]):
            d_sub = fu["subspace_alignment"] - b["subspace_alignment"]
            d_tc = fu["temporal_correlation"] - b["temporal_correlation"]
            d_fc = fu["fc_similarity"] - b["fc_similarity"]
            f.write(
                f"  {r.recording_id:24s} Δsub={d_sub:+.4f} "
                f"Δtc={d_tc:+.4f} Δfc={d_fc:+.4f}\n"
            )

    print(f"wrote {out_txt}")
    print(f"wrote {out_json}\n")
    with out_txt.open() as f:
        for line in f:
            print(line, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
