"""Phase 1.0.3 — anti-correlation diagnostic.

Phase 0.9's central finding (PHASE0.9_REPORT.md §3) was that the
homogeneous CTRNN produced **0% of neuron pairs at FC < −0.1**, vs.
17.5% in the Atanas 2023 recordings. Every Phase 0 architecture
variant tried so far has the same flaw.

Phase 1.0.3 asks: does the new graph-native LIF architecture
*structurally* unblock this? (We have not yet wired plasticity or
modulators — those are Phase 1.0.4 — so this is a measurement of the
spiking + subgraph-overlap topology alone, with no learned inhibition.)

Methodology:

  * Simulate the full network for 4000 ticks after a 1000-tick warmup.
  * Use sensory_noise=0.2 (the same drive level as the 10k stability
    test) and the runtime's default noise_level.
  * Compute the pairwise FC on the ``rate`` trace.
  * Report % of off-diagonal pairs at FC < −0.05 / −0.1 / −0.2 / −0.3,
    and the same for the Atanas reference recordings if available.

The result is written to ``output/phase1.0/anticorrelation.txt``.
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
from algos.neural_v2 import GraphSimulator, SimulatorConfig


WARMUP_TICKS = 1000
SAMPLE_TICKS = 4000
THRESHOLDS = (-0.05, -0.1, -0.2, -0.3)
SEEDS = (1000, 1001, 1002)


def fc_negative_fractions(trace: np.ndarray) -> dict[float, float]:
    """Fraction of off-diagonal FC entries below each threshold."""
    fc = np.corrcoef(trace, rowvar=False)
    fc = np.nan_to_num(fc)
    iu = np.triu_indices(fc.shape[0], k=1)
    upper = fc[iu]
    out = {}
    for t in THRESHOLDS:
        out[float(t)] = float(np.mean(upper < t))
    return out


def real_anticorrelation_baseline(max_recordings: int = 10) -> dict[float, float] | None:
    """Compute fraction-of-anti-correlated-pairs on Atanas 2023 directly.

    Pools every labeled neuron's trace across the top-N most-labeled
    recordings, computes per-recording FC, and reports the mean fraction
    of pairs below each threshold. This matches Phase 0.9's reference
    number of ≈17.5% pairs below FC = −0.1.
    """
    try:
        from algos.validation.reference_data import ReferenceDataset
    except Exception:
        return None
    try:
        dataset = ReferenceDataset.from_atanas2023(max_recordings=max_recordings)
    except Exception as exc:
        print(f"  (real baseline unavailable: {exc})")
        return None
    sums = {float(t): 0.0 for t in THRESHOLDS}
    n_records = 0
    for rec in dataset.recordings:
        if not rec.labels:
            continue
        cols = list(rec.labels.keys())
        traces = rec.traces[:, cols]
        if traces.shape[1] < 4 or traces.shape[0] < 10:
            continue
        fcs = fc_negative_fractions(traces)
        for t in THRESHOLDS:
            sums[float(t)] += fcs[float(t)]
        n_records += 1
    if n_records == 0:
        return None
    return {t: sums[t] / n_records for t in sums}


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    phase1_dir = OUTPUT_DIR / "phase1.0"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    print("loading connectome → graph…")
    g = load_connectome_into_graph()
    subs = build_canonical_subgraphs(g)
    print(f"  built {len(subs)} subgraphs")

    per_seed = []
    for seed in SEEDS:
        t0 = time.time()
        sim = GraphSimulator(
            g, config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2),
        )
        # Warmup.
        state = sim.initial_state(seed=seed)
        rng = np.random.default_rng(seed)
        sens = np.zeros(sim.n)
        for _ in range(WARMUP_TICKS):
            sens[:] = 0.0
            sens[sim.sensory_idx] = (
                rng.standard_normal(sim.sensory_idx.size) * 0.2
            )
            state = sim.step(state, sens, rng)
        # Record.
        rate_hist = np.zeros((SAMPLE_TICKS, sim.n), dtype=np.float32)
        for t in range(SAMPLE_TICKS):
            sens[:] = 0.0
            sens[sim.sensory_idx] = (
                rng.standard_normal(sim.sensory_idx.size) * 0.2
            )
            state = sim.step(state, sens, rng)
            rate_hist[t] = state.rate
        # Drop constant neurons (rate==0 throughout would give NaN corr).
        keep = rate_hist.std(axis=0) > 1e-9
        active = rate_hist[:, keep]
        n_active = int(keep.sum())

        fcs = fc_negative_fractions(active)
        elapsed = time.time() - t0
        per_seed.append({
            "seed": seed,
            "n_active": n_active,
            "runtime_s": elapsed,
            "fractions": fcs,
        })
        thresh_str = "  ".join(f"<{t:+.2f}: {fcs[t]*100:5.2f}%" for t in THRESHOLDS)
        print(
            f"  seed={seed} active={n_active}/302 runtime={elapsed:.1f}s | "
            f"{thresh_str}"
        )

    # Aggregate across seeds (mean fraction).
    agg = {}
    for t in THRESHOLDS:
        vals = [s["fractions"][float(t)] for s in per_seed]
        agg[float(t)] = float(np.mean(vals))

    # Compare with the real-recording baseline if available.
    real_b = real_anticorrelation_baseline()

    out_txt = phase1_dir / "anticorrelation.txt"
    out_json = phase1_dir / "anticorrelation.json"
    out_json.write_text(json.dumps({
        "warmup_ticks": WARMUP_TICKS,
        "sample_ticks": SAMPLE_TICKS,
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "aggregate": agg,
        "real_baseline": real_b,
    }, indent=2))

    with out_txt.open("w") as f:
        f.write("Phase 1.0.3 — anti-correlation diagnostic (no plasticity, no modulators)\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"warmup_ticks:  {WARMUP_TICKS}\n")
        f.write(f"sample_ticks:  {SAMPLE_TICKS}\n")
        f.write(f"seeds:         {SEEDS}\n\n")
        f.write("Fraction of off-diagonal FC entries below each threshold "
                "(rate trace, active neurons only):\n\n")
        header = "  threshold  " + " ".join(f"seed={s['seed']:>4}" for s in per_seed) + "   mean"
        f.write(header + "\n")
        f.write("  " + "-" * (len(header) - 2) + "\n")
        for t in THRESHOLDS:
            row = f"  FC < {t:+.2f}  "
            for s in per_seed:
                row += f"  {s['fractions'][float(t)]*100:7.3f}%"
            row += f"  {agg[float(t)]*100:7.3f}%"
            f.write(row + "\n")
        f.write("\n")
        if real_b:
            f.write("Real-recording baseline (Atanas 2023, computed in-band "
                    "from the labeled-neuron sub-matrix of the top-10 "
                    "most-labeled recordings):\n")
            for t in THRESHOLDS:
                if t in real_b:
                    f.write(f"  FC < {t:+.2f}  {real_b[t]*100:.3f}%\n")
        else:
            f.write("Real-recording baseline: not available "
                    "(data/reference/ is missing — see "
                    "data/reference/README.md)\n")
        f.write("\n")
        f.write("Reference benchmarks from PHASE0.9_REPORT.md §3:\n")
        f.write("  Phase 0.9 (CTRNN, all variants): FC < -0.1 ≈ 0.0%\n")
        f.write("  Real worm (Atanas 2023):         FC < -0.1 ≈ 17.5%\n")

    print(f"\nwrote {out_txt}")
    print(f"wrote {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
