"""Phase 1.0.5 — qualitative subgraph behavior probe.

For each of a small set of named circuit pairs, computes the
inter-subgraph correlation of the mean rate trace over a single
long run. This is the cleanest behavior-level read on whether the
graph-native architecture produces the *kind* of structure that
matters (mutual inhibition between forward and reversal command;
chemo/thermo sharing a common premotor pool through RIA; etc.).

Output: output/phase1.0/subgraph_behavior.txt
"""

from __future__ import annotations

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


N_TICKS = 8000
WARMUP_TICKS = 1000
SEED = 1042


def mean_subgraph_rate(rate_hist: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return rate_hist[:, indices].mean(axis=1)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    phase1_dir = OUTPUT_DIR / "phase1.0"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    print("loading + building graph…")
    g = load_connectome_into_graph()
    subs = build_canonical_subgraphs(g)
    sim = GraphSimulator(
        g, config=SimulatorConfig(noise_level=0.005, sensory_noise=0.2),
    )
    rule = HebbianRule.from_graph(g)
    sim.attach_plasticity(rule)
    bank = build_default_modulator_bank(g, sim.params.threshold)
    sim.attach_modulators(bank)

    print(f"warmup ({WARMUP_TICKS}) + sample ({N_TICKS})…", flush=True)
    state = sim.initial_state(seed=SEED)
    rng = np.random.default_rng(SEED)
    sens = np.zeros(sim.n)
    for _ in range(WARMUP_TICKS):
        sens[:] = 0.0
        sens[sim.sensory_idx] = rng.standard_normal(sim.sensory_idx.size) * 0.2
        state = sim.step(state, sens, rng)
    rate_hist = np.zeros((N_TICKS, sim.n), dtype=np.float32)
    t0 = time.time()
    for t in range(N_TICKS):
        sens[:] = 0.0
        sens[sim.sensory_idx] = rng.standard_normal(sim.sensory_idx.size) * 0.2
        state = sim.step(state, sens, rng)
        rate_hist[t] = state.rate
    print(f"  sample runtime: {time.time() - t0:.2f}s")

    # Build per-subgraph mean rate trace.
    sub_traces: dict[str, np.ndarray] = {}
    for name, sg in subs.items():
        sub_traces[name] = mean_subgraph_rate(rate_hist, sg.node_indices())

    pairs_of_interest = [
        ("reversal_command", "forward_command",
         "Should be ANTI-correlated (winner-take-all)."),
        ("chemosensory_amphid", "thermosensory",
         "Should be POSITIVELY correlated (share AIY/AIZ/RIA)."),
        ("anterior_touch", "reversal_command",
         "Should be POSITIVELY correlated (touch → reversal)."),
        ("posterior_touch", "forward_command",
         "Should be POSITIVELY correlated (touch → forward)."),
        ("head_motor_cpg", "forward_command",
         "Loose: head wiggle modulated by locomotion command."),
        ("pharyngeal_cpg", "modulator_5HT",
         "5-HT drives feeding — positive correlation expected."),
        ("modulator_RID", "forward_command",
         "RID promotes forward — positive correlation expected."),
        ("defecation_pacemaker", "forward_command",
         "Loose: defecation interrupts forward locomotion."),
    ]

    out_txt = phase1_dir / "subgraph_behavior.txt"
    print("Subgraph correlations:")
    with out_txt.open("w") as f:
        f.write("Phase 1.0.5 — subgraph behavior probe\n")
        f.write("=" * 78 + "\n")
        f.write(f"n_ticks: {N_TICKS} (after {WARMUP_TICKS} warmup)\n")
        f.write(f"seed:    {SEED}\n\n")

        f.write("Per-subgraph activity (mean of mean-rate trace):\n")
        for name, tr in sub_traces.items():
            f.write(
                f"  {name:25s} mean={tr.mean():.4f}  std={tr.std():.4f}  "
                f"max={tr.max():.4f}\n"
            )
        f.write("\n")

        f.write("Inter-subgraph pairwise correlations (Pearson r):\n\n")
        f.write(f"  {'pair':50s}  {'r':>8s}  expectation\n")
        f.write("  " + "-" * 76 + "\n")
        for a, b, note in pairs_of_interest:
            r = pearson(sub_traces[a], sub_traces[b])
            f.write(f"  {a:25s} ↔ {b:25s}  {r:+7.3f}  {note}\n")
            print(f"  {a:25s} ↔ {b:25s}  r = {r:+.3f}")
        f.write("\n")

        # Full pairwise matrix
        f.write("Full inter-subgraph correlation matrix:\n\n")
        names = list(sub_traces.keys())
        f.write("                            " + " ".join(
            f"{n[:7]:>8s}" for n in names
        ) + "\n")
        for ai, na in enumerate(names):
            row = f"  {na:25s}  "
            for nb in names:
                row += f"{pearson(sub_traces[na], sub_traces[nb]):+8.3f}"
            f.write(row + "\n")
        f.write("\n")

        # Top anti-correlated subgraph pairs (the §0.9 diagnostic).
        anti = []
        for i, na in enumerate(names):
            for j in range(i + 1, len(names)):
                nb = names[j]
                anti.append((pearson(sub_traces[na], sub_traces[nb]), na, nb))
        anti.sort()
        f.write("Top 5 most ANTI-correlated subgraph pairs:\n")
        for r, na, nb in anti[:5]:
            f.write(f"  {na:25s} ↔ {nb:25s}  r = {r:+.4f}\n")
        f.write("\nTop 5 most POSITIVELY correlated subgraph pairs:\n")
        for r, na, nb in anti[-5:][::-1]:
            f.write(f"  {na:25s} ↔ {nb:25s}  r = {r:+.4f}\n")

    print(f"\nwrote {out_txt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
