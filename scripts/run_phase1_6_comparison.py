"""Phase 1.6.3 — comprehensive comparison vs Phase 1.0.

Runs every Atanas-2023 recording (top-10 best labeled) under two
configurations:

  A. phase1.0 — RID + 5-HT (Phase 1.0.4 wiring), plasticity on,
                Phase 1.0 13 subgraphs.
  B. phase1.6 — same plus tyramine modulator (1.6.2) + AIY/M4/AIM
                added to 5-HT targets (1.6.3) + inhibitory_command_gate
                subgraph registered (1.6.1; no dynamics effect since
                gate is dormant in bare network, but registered for
                completeness).

Reports the three Phase 0.7+ metrics, the fraction of pairs below
each FC threshold, the forward↔reversal subgraph correlation, and
each silent-subgraph mean rate.
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
    Modulator,
    ModulatorBank,
    SimulatorConfig,
    build_default_modulator_bank,
)
from algos.neural_v2.modulators import (
    DEFAULT_TAU_M,
    RID_SENSITIVITY,
    RID_TARGET_NEURONS,
    SHT_SENSITIVITY_FORWARD,
    SHT_SENSITIVITY_PHARYNX,
    SHT_SOURCE_NEURONS,
    SHT_TARGET_FORWARD,
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
FC_THRESHOLDS = (-0.05, -0.10, -0.20, -0.30)


def _idx_array(graph, names):
    return np.array(
        [graph.index_of(n) for n in names if graph.has_node(n)],
        dtype=np.int64,
    )


def build_phase1_0_bank(graph, base_threshold):
    """Reconstruct the Phase 1.0.4 modulator bank: RID + 5-HT only,
    with the *original* 5-HT pharynx targets (no M4, no AIY/AIM)."""
    rid = Modulator(
        name="RID",
        producer_idx=_idx_array(graph, ["RID"]),
        target_idx=_idx_array(graph, RID_TARGET_NEURONS),
        sensitivity=np.full(len(RID_TARGET_NEURONS), RID_SENSITIVITY),
        tau_m=DEFAULT_TAU_M,
    )
    sht_fwd = _idx_array(graph, SHT_TARGET_FORWARD)
    # Phase 1.0.4's pharynx list: M3, MI, I1 (no M4).
    sht_phx_old = _idx_array(graph, ("M3L", "M3R", "MI", "I1L", "I1R"))
    sht = Modulator(
        name="5HT",
        producer_idx=_idx_array(graph, SHT_SOURCE_NEURONS),
        target_idx=np.concatenate([sht_fwd, sht_phx_old]),
        sensitivity=np.concatenate([
            np.full(sht_fwd.shape, SHT_SENSITIVITY_FORWARD),
            np.full(sht_phx_old.shape, SHT_SENSITIVITY_PHARYNX),
        ]),
        tau_m=DEFAULT_TAU_M,
    )
    return ModulatorBank(modulators=[rid, sht], base_threshold=base_threshold.copy())


def simulate(config_name, n_ticks, seed):
    g = load_connectome_into_graph()
    build_canonical_subgraphs(g)
    sim = GraphSimulator(
        g, config=SimulatorConfig(noise_level=NOISE_LEVEL, sensory_noise=SENSORY_NOISE),
    )
    sim.attach_plasticity(HebbianRule.from_graph(g))
    if config_name == "phase1.0":
        bank = build_phase1_0_bank(g, sim.params.threshold)
    elif config_name == "phase1.6":
        bank = build_default_modulator_bank(g, sim.params.threshold)
    else:
        raise ValueError(config_name)
    sim.attach_modulators(bank)
    state = sim.initial_state(seed=seed)
    rng = np.random.default_rng(seed)
    sens = np.zeros(sim.n)
    for _ in range(PRE_EQ_TICKS):
        sens[:] = 0.0
        sens[sim.sensory_idx] = rng.standard_normal(sim.sensory_idx.size) * SENSORY_NOISE
        state = sim.step(state, sens, rng)
    rate_hist = np.zeros((n_ticks, sim.n), dtype=np.float32)
    for t in range(n_ticks):
        sens[:] = 0.0
        sens[sim.sensory_idx] = rng.standard_normal(sim.sensory_idx.size) * SENSORY_NOISE
        state = sim.step(state, sens, rng)
        rate_hist[t] = state.rate

    diag = {
        "total_spikes": int(state.spike_count.sum()),
        "active_neurons": int((state.spike_count > 0).sum()),
        "c_RID": float(bank.modulators[0].c_m),
        "c_5HT": float(bank.modulators[1].c_m),
    }
    if config_name == "phase1.6":
        diag["c_tyramine"] = float(bank.modulators[2].c_m)
    return rate_hist, g.neuron_names(), diag, g


def metric_triple(digital, dig_names, real, real_names):
    pca = pca_structure_similarity(digital, dig_names, real, real_names,
                                   n_components=N_PCA_COMPONENTS)
    fc = functional_connectivity_similarity(digital, dig_names, real, real_names)
    tc = temporal_correlation(digital, dig_names, real, real_names)
    return {
        "subspace_alignment":
            float(pca.details.get("subspace_alignment", float("nan"))),
        "temporal_correlation": float(tc.score),
        "fc_similarity": float(fc.score),
        "n_matched": int(pca.details.get("n_matched", 0)),
    }


def fc_neg_fractions(trace):
    keep = trace.std(axis=0) > 1e-9
    if int(keep.sum()) < 4:
        return {float(t): float("nan") for t in FC_THRESHOLDS}
    active = trace[:, keep]
    fc = np.corrcoef(active, rowvar=False)
    iu = np.triu_indices(fc.shape[0], k=1)
    upper = np.nan_to_num(fc[iu])
    return {float(t): float(np.mean(upper < t)) for t in FC_THRESHOLDS}


def main():
    out_dir = OUTPUT_DIR / "phase1.6"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading top-{N_WORMS} Atanas 2023 recordings…")
    reference = ReferenceDataset.from_atanas2023(max_recordings=N_WORMS)
    recordings = reference.recordings
    real_views = []
    for r in recordings:
        cols, names = [], []
        for c, n in r.labels.items():
            cols.append(c); names.append(n)
        real_views.append((r.traces[:, cols], names))

    results = {"phase1.0": [], "phase1.6": []}
    fc_fractions = {"phase1.0": [], "phase1.6": []}
    fwd_bwd_r = {"phase1.0": [], "phase1.6": []}
    silent_subgraph_rates = {"phase1.0": [], "phase1.6": []}

    t_start = time.time()
    for config in ("phase1.0", "phase1.6"):
        print(f"\n=== {config} ===", flush=True)
        for k_idx, (r, (real_mat, real_names)) in enumerate(zip(recordings, real_views)):
            seed = 1000 + k_idx
            t0 = time.time()
            rate_hist, dig_names, diag, g = simulate(config, r.n_timepoints, seed)
            m = metric_triple(rate_hist, dig_names, real_mat, real_names)
            m["recording_id"] = r.recording_id
            m["seed"] = seed
            m["runtime_s"] = time.time() - t0
            m.update(diag)
            results[config].append(m)
            # FC negative fractions
            fcs = fc_neg_fractions(rate_hist)
            fcs["recording_id"] = r.recording_id
            fc_fractions[config].append(fcs)
            # fwd↔bwd subgraph correlation
            fwd_idx = g.subgraphs["forward_command"].node_indices()
            bwd_idx = g.subgraphs["reversal_command"].node_indices()
            fwd_tr = rate_hist[:, fwd_idx].mean(axis=1)
            bwd_tr = rate_hist[:, bwd_idx].mean(axis=1)
            if fwd_tr.std() < 1e-9 or bwd_tr.std() < 1e-9:
                r_fwd_bwd = float("nan")
            else:
                r_fwd_bwd = float(np.corrcoef(fwd_tr, bwd_tr)[0, 1])
            fwd_bwd_r[config].append(r_fwd_bwd)
            # Silent subgraph mean rates
            silent = {}
            for name in ("pharyngeal_cpg", "ventral_cord_motor", "egg_laying"):
                idx = g.subgraphs[name].node_indices()
                silent[name] = float(rate_hist[:, idx].mean())
            silent_subgraph_rates[config].append(silent)
            print(
                f"  rec={r.recording_id[:16]:16s} runtime={m['runtime_s']:5.1f}s "
                f"sub={m['subspace_alignment']:+.4f} tc={m['temporal_correlation']:+.4f} "
                f"fc={m['fc_similarity']:+.4f} fwd↔bwd={r_fwd_bwd:+.3f}",
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
                "mean": float(v.mean()),
                "std": float(v.std()),
            }
        v = np.array(fwd_bwd_r[cn])
        v = v[~np.isnan(v)]
        aggregate[cn]["fwd_bwd_r"] = {
            "mean": float(v.mean()), "std": float(v.std())
        }
        for thr in FC_THRESHOLDS:
            vals = np.array([f[float(thr)] for f in fc_fractions[cn]])
            aggregate[cn][f"fc_below_{thr}"] = {
                "mean": float(vals.mean()), "std": float(vals.std())
            }
        # Silent subgraph averages
        for name in ("pharyngeal_cpg", "ventral_cord_motor", "egg_laying"):
            vals = np.array([s[name] for s in silent_subgraph_rates[cn]])
            aggregate[cn][f"silent_{name}"] = {
                "mean": float(vals.mean()), "std": float(vals.std())
            }

    # Phase 0.9 reference (for the absolute floor).
    phase09_ref = {
        "category": {
            "subspace_alignment": +0.3532,
            "temporal_correlation": -0.0140,
            "fc_similarity": +0.0606,
        },
    }

    out_json = out_dir / "comparison_results.json"
    out_json.write_text(json.dumps({
        "n_worms": N_WORMS,
        "pre_eq_ticks": PRE_EQ_TICKS,
        "sensory_noise": SENSORY_NOISE,
        "noise_level": NOISE_LEVEL,
        "aggregate": aggregate,
        "per_recording": results,
        "fc_fractions": fc_fractions,
        "fwd_bwd_r": fwd_bwd_r,
        "silent_subgraph_rates": silent_subgraph_rates,
        "phase09_reference": phase09_ref,
        "total_runtime_s": elapsed,
    }, indent=2))

    out_txt = out_dir / "comparison_report.txt"
    with out_txt.open("w") as f:
        f.write("Phase 1.6.3 — Phase 1.0 vs Phase 1.6 comparison\n")
        f.write("=" * 78 + "\n")
        f.write(f"recordings:     {N_WORMS}\n")
        f.write(f"pre_eq_ticks:   {PRE_EQ_TICKS}\n")
        f.write(f"sensory_noise:  {SENSORY_NOISE}\n")
        f.write(f"runtime:        {elapsed:.1f}s\n\n")

        f.write("Mean digital-vs-real metrics (n=10):\n\n")
        f.write(
            f"  {'metric':28s}  {'P0.9 cat':>9s}  {'P1.0':>9s}  {'P1.6':>9s}  Δ(1.6−1.0)\n"
        )
        for m in metric_keys:
            cat = phase09_ref["category"][m]
            v10 = aggregate["phase1.0"][m]["mean"]
            v16 = aggregate["phase1.6"][m]["mean"]
            f.write(
                f"  {m:28s}  {cat:+9.4f}  {v10:+9.4f}  {v16:+9.4f}  {v16-v10:+9.4f}\n"
            )
        f.write("\n")

        f.write("Forward↔reversal subgraph Pearson r (mean ± std):\n")
        for cn in ("phase1.0", "phase1.6"):
            d = aggregate[cn]["fwd_bwd_r"]
            f.write(f"  {cn}:  {d['mean']:+.4f} ± {d['std']:.4f}\n")
        f.write("\n")

        f.write("Fraction of off-diagonal FC pairs below each threshold (mean):\n\n")
        f.write(f"  {'threshold':>10s}  {'P1.0':>10s}  {'P1.6':>10s}  Δ\n")
        for thr in FC_THRESHOLDS:
            v10 = aggregate["phase1.0"][f"fc_below_{thr}"]["mean"]
            v16 = aggregate["phase1.6"][f"fc_below_{thr}"]["mean"]
            f.write(
                f"  FC < {thr:+.2f}   {v10*100:8.3f}%  {v16*100:8.3f}%  "
                f"{(v16-v10)*100:+7.3f}%\n"
            )
        f.write("\n")

        f.write("Silent-subgraph mean rate (Phase 1.0 had these = 0.000):\n\n")
        f.write(f"  {'subgraph':28s}  {'P1.0':>10s}  {'P1.6':>10s}\n")
        for name in ("pharyngeal_cpg", "ventral_cord_motor", "egg_laying"):
            v10 = aggregate["phase1.0"][f"silent_{name}"]["mean"]
            v16 = aggregate["phase1.6"][f"silent_{name}"]["mean"]
            f.write(f"  {name:28s}  {v10:10.4f}  {v16:10.4f}\n")
        f.write("\n")

        f.write("Modulator concentration distributions (Phase 1.6 final):\n")
        for cn in ("phase1.0", "phase1.6"):
            cs_rid = np.array([r.get("c_RID", float("nan")) for r in results[cn]])
            cs_5ht = np.array([r.get("c_5HT", float("nan")) for r in results[cn]])
            f.write(f"  {cn}:  c_RID mean={cs_rid.mean():+.4f} std={cs_rid.std():.4f}  "
                    f"c_5HT mean={cs_5ht.mean():+.4f} std={cs_5ht.std():.4f}")
            if cn == "phase1.6":
                cs_tyr = np.array([r.get("c_tyramine", float("nan")) for r in results[cn]])
                f.write(f"  c_tyramine mean={cs_tyr.mean():+.4f} std={cs_tyr.std():.4f}")
            f.write("\n")

    print(f"\nwrote {out_txt}")
    print(f"wrote {out_json}")
    print()
    with out_txt.open() as f:
        for line in f:
            print(line, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
