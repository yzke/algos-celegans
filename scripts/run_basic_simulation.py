"""Run a basic Phase 0 simulation and produce the AC4 figures.

The schedule is:
  ticks 0–999:    zero input, system equilibrates from random init
  ticks 1000–1999: stimulate ASEL (chemosensory)
  ticks 2000–2999: zero input again
  ticks 3000–3999: stimulate AVAL (command interneuron for backward locomotion)
  ticks 4000–4999: zero input, return to baseline

Outputs:
  output/basic_simulation_heatmap.png — full (5000, 302) activity heatmap.
  output/basic_simulation_traces.png  — line traces for a handful of named neurons.
  output/basic_simulation_summary.txt — numeric summary of the run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from algos.config import OUTPUT_DIR
from algos.connectome import ConnectomeData
from algos.neural import CTRNNParams, NeuralState, neural_step
from algos.viz.activity import plot_activity_matrix, plot_trace


N_TICKS = 5000
STIM_INTENSITY = 0.5


def build_sensory(t: int, n: int, connectome: ConnectomeData) -> np.ndarray:
    """Schedule sensory input as a function of tick."""
    sens = np.zeros(n)
    if 1000 <= t < 2000:
        sens[connectome.idx("ASEL")] = STIM_INTENSITY
    elif 3000 <= t < 4000:
        sens[connectome.idx("AVAL")] = STIM_INTENSITY
    return sens


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    connectome = ConnectomeData.load()
    state = NeuralState.initialize(connectome.n_neurons, seed=42)
    params = CTRNNParams()
    rng = np.random.default_rng(42)

    history = np.zeros((N_TICKS, connectome.n_neurons), dtype=np.float32)

    t0 = time.time()
    for t in range(N_TICKS):
        sens = build_sensory(t, connectome.n_neurons, connectome)
        state = neural_step(state, connectome, sens, params, rng)
        history[t] = state.V
    elapsed = time.time() - t0

    # Sanity checks
    finite = bool(np.all(np.isfinite(history)))
    max_abs = float(np.max(np.abs(history)))

    # Activity heatmap
    fig = plot_activity_matrix(
        history,
        connectome.neuron_names,
        categories=connectome.category,
        title=f"Phase 0 — ASEL pulse 1000-2000, AVAL pulse 3000-4000 ({N_TICKS} ticks)",
    )
    heatmap_path = OUTPUT_DIR / "basic_simulation_heatmap.png"
    fig.savefig(heatmap_path, dpi=150)

    # Traces of selected neurons
    traces = ["ASEL", "ASER", "AVAL", "AVAR", "AVBL", "AVBR", "PLML", "RIS"]
    fig2 = plot_trace(
        history,
        connectome.neuron_names,
        traces,
        connectome.neuron_to_idx,
        title="Selected neuron traces",
    )
    traces_path = OUTPUT_DIR / "basic_simulation_traces.png"
    fig2.savefig(traces_path, dpi=150)

    # Numeric summary
    summary = OUTPUT_DIR / "basic_simulation_summary.txt"
    with summary.open("w") as f:
        f.write("ALGOS-Celegans Phase 0 basic simulation\n")
        f.write("=" * 50 + "\n")
        f.write(f"ticks:       {N_TICKS}\n")
        f.write(f"neurons:     {connectome.n_neurons}\n")
        f.write(f"runtime:     {elapsed:.2f}s ({elapsed * 1000 / N_TICKS:.3f} ms/tick)\n")
        f.write(f"all finite:  {finite}\n")
        f.write(f"max |V|:     {max_abs:.4f}\n")
        f.write("\nNeuron categories:\n")
        from collections import Counter
        for cat, n in Counter(connectome.category).items():
            f.write(f"  {cat}: {n}\n")
        f.write("\nFinal-state extrema:\n")
        order = np.argsort(state.V)
        for idx in order[:5]:
            f.write(f"  V[{connectome.neuron_names[idx]:6s}] = {state.V[idx]: .4f}\n")
        f.write("  ...\n")
        for idx in order[-5:]:
            f.write(f"  V[{connectome.neuron_names[idx]:6s}] = {state.V[idx]: .4f}\n")

    print(f"Wrote {heatmap_path}")
    print(f"Wrote {traces_path}")
    print(f"Wrote {summary}")
    print(f"runtime: {elapsed:.2f}s ({elapsed * 1000 / N_TICKS:.3f} ms/tick) | finite={finite} | max|V|={max_abs:.4f}")


if __name__ == "__main__":
    main()
