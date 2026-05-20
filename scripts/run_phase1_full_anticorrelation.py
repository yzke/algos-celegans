"""Phase 1.0.4 — anti-correlation diagnostic with plasticity + modulators on.

Counterpart to scripts/run_phase1_anticorrelation.py. Adds:
  * Hebbian plasticity on 100 edges (default HebbianRule.from_graph).
  * Two modulators (RID + 5-HT) from build_default_modulator_bank().
  * Longer warmup so plasticity has time to take effect (3000 ticks).

Reports the same FC < {-0.05, -0.1, -0.2, -0.3} fractions and a
side-by-side diff against the no-plasticity / no-modulator baseline
written by run_phase1_anticorrelation.py.
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


WARMUP_TICKS = 3000
SAMPLE_TICKS = 4000
THRESHOLDS = (-0.05, -0.1, -0.2, -0.3)
SEEDS = (1000, 1001, 1002)


def fc_negative_fractions(trace: np.ndarray) -> dict[float, float]:
    fc = np.corrcoef(trace, rowvar=False)
    fc = np.nan_to_num(fc)
    iu = np.triu_indices(fc.shape[0], k=1)
    upper = fc[iu]
    return {float(t): float(np.mean(upper < t)) for t in THRESHOLDS}


def run_condition(
    label: str,
    *,
    plasticity: bool,
    modulators: bool,
) -> list[dict]:
    g = load_connectome_into_graph()
    build_canonical_subgraphs(g)
    per_seed = []
    for seed in SEEDS:
        sim = GraphSimulator(
            g, config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2),
        )
        if plasticity:
            rule = HebbianRule.from_graph(g)
            sim.attach_plasticity(rule)
        if modulators:
            bank = build_default_modulator_bank(g, sim.params.threshold)
            sim.attach_modulators(bank)

        state = sim.initial_state(seed=seed)
        rng = np.random.default_rng(seed)
        sens = np.zeros(sim.n)
        t0 = time.time()
        for _ in range(WARMUP_TICKS):
            sens[:] = 0.0
            sens[sim.sensory_idx] = (
                rng.standard_normal(sim.sensory_idx.size) * 0.2
            )
            state = sim.step(state, sens, rng)
        rate_hist = np.zeros((SAMPLE_TICKS, sim.n), dtype=np.float32)
        for t in range(SAMPLE_TICKS):
            sens[:] = 0.0
            sens[sim.sensory_idx] = (
                rng.standard_normal(sim.sensory_idx.size) * 0.2
            )
            state = sim.step(state, sens, rng)
            rate_hist[t] = state.rate

        keep = rate_hist.std(axis=0) > 1e-9
        active = rate_hist[:, keep]
        fcs = fc_negative_fractions(active)
        elapsed = time.time() - t0

        info = {
            "seed": seed,
            "n_active": int(keep.sum()),
            "runtime_s": elapsed,
            "fractions": fcs,
        }
        if plasticity:
            info["plastic_weight_summary"] = rule.weight_summary()
        if modulators:
            info["c_RID_final"] = float(bank.modulators[0].c_m)
            info["c_5HT_final"] = float(bank.modulators[1].c_m)
        per_seed.append(info)
        rid_str = f" c_RID={info.get('c_RID_final', 0):+.3f}" if modulators else ""
        sht_str = f" c_5HT={info.get('c_5HT_final', 0):+.3f}" if modulators else ""
        thresh_str = "  ".join(
            f"<{t:+.2f}: {fcs[t]*100:5.2f}%" for t in THRESHOLDS
        )
        print(
            f"  [{label}] seed={seed} active={info['n_active']:3d} "
            f"runtime={elapsed:.1f}s | {thresh_str}{rid_str}{sht_str}"
        )
    return per_seed


def aggregate(per_seed: list[dict]) -> dict[float, float]:
    return {
        float(t): float(np.mean([s["fractions"][float(t)] for s in per_seed]))
        for t in THRESHOLDS
    }


def main() -> int:
    phase1_dir = OUTPUT_DIR / "phase1.0"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("BASELINE (LIF only, no plasticity, no modulators)")
    print("=" * 60)
    baseline = run_condition("baseline", plasticity=False, modulators=False)

    print("=" * 60)
    print("MODULATORS only (RID + 5-HT)")
    print("=" * 60)
    mod_only = run_condition("modulators", plasticity=False, modulators=True)

    print("=" * 60)
    print("PLASTICITY only (100 Hebbian edges)")
    print("=" * 60)
    plast_only = run_condition("plasticity", plasticity=True, modulators=False)

    print("=" * 60)
    print("FULL (plasticity + modulators)")
    print("=" * 60)
    full = run_condition("full", plasticity=True, modulators=True)

    conditions = {
        "baseline":   baseline,
        "modulators": mod_only,
        "plasticity": plast_only,
        "full":       full,
    }
    agg = {k: aggregate(v) for k, v in conditions.items()}

    out_json = phase1_dir / "full_anticorrelation.json"
    out_json.write_text(json.dumps({
        "warmup_ticks": WARMUP_TICKS,
        "sample_ticks": SAMPLE_TICKS,
        "seeds": list(SEEDS),
        "per_seed_by_condition": conditions,
        "aggregate_by_condition": agg,
    }, indent=2))

    out_txt = phase1_dir / "full_anticorrelation.txt"
    with out_txt.open("w") as f:
        f.write("Phase 1.0.4 — anti-correlation with plasticity + modulators\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"warmup_ticks:  {WARMUP_TICKS}\n")
        f.write(f"sample_ticks:  {SAMPLE_TICKS}\n")
        f.write(f"seeds:         {SEEDS}\n\n")
        f.write("Mean fraction (off-diagonal pairs):\n\n")
        f.write(
            f"  {'threshold':>10s}   {'baseline':>10s}   "
            f"{'+mods':>10s}   {'+plast':>10s}   {'full':>10s}   "
            f"{'Δ(full−base)':>14s}\n"
        )
        f.write("  " + "-" * 76 + "\n")
        for t in THRESHOLDS:
            b = agg["baseline"][t]
            m = agg["modulators"][t]
            p = agg["plasticity"][t]
            full_v = agg["full"][t]
            f.write(
                f"  FC < {t:+.2f}  {b*100:9.3f}%  {m*100:9.3f}%  "
                f"{p*100:9.3f}%  {full_v*100:9.3f}%  {(full_v-b)*100:+13.3f}%\n"
            )
        f.write("\n")
        # Plasticity stats from the full condition.
        f.write("Plasticity weight changes (full condition, per seed):\n")
        for s in full:
            ws = s["plastic_weight_summary"]
            f.write(
                f"  seed={s['seed']}: n={ws['n_edges']} "
                f"mean={ws['weight_mean']:.4f} delta_mean={ws['delta_mean']:+.4f} "
                f"grew={ws['n_grew']:>3} shrank={ws['n_shrank']:>3}\n"
            )
        f.write("\n")
        f.write("Modulator final concentrations (full condition, per seed):\n")
        for s in full:
            f.write(
                f"  seed={s['seed']}: c_RID={s['c_RID_final']:+.4f}  "
                f"c_5HT={s['c_5HT_final']:+.4f}\n"
            )

    print(f"\nwrote {out_txt}")
    print(f"wrote {out_json}")
    print()
    with out_txt.open() as f:
        for line in f:
            print(line, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
