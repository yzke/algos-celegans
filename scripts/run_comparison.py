"""AC0.5.2 — Run digital sim + load Atanas 2023 + compute 3 similarity metrics.

The digital protocol is the simplest interpretable baseline: drive every
sensory neuron with independent Gaussian noise at each tick, run the
bare CTRNN for the same number of timesteps as the real recording, and
record V at every neuron. Functional connectivity then reflects only the
connectome topology and the v0.3 dynamics — no body, no behavior, no
neuromodulators.

The metrics are computed per real recording (against a single digital run
matched in length), then averaged.

Outputs:
  output/comparison_report.txt — human-readable summary across recordings.
  output/comparison_results.json — full per-metric, per-recording numbers.
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
from algos.neural import CTRNNParams, NeuralState, neural_step
from algos.validation.comparison import (
    ComparisonResult,
    run_all_metrics,
)
from algos.validation.reference_data import ReferenceDataset


# Default: 6 best-labeled Atanas 2023 recordings. Override with CLI arg.
N_RECORDINGS = 6
SENSORY_NOISE_LEVEL = 0.1
RNG_SEED = 42


def _sensory_neuron_indices(connectome: ConnectomeData) -> list[int]:
    """All neurons categorized 'sensory'."""
    return connectome.get_neuron_indices_by_category("sensory")


# Reversal/forward command neuron sets (mirror of neuron_specificity.py).
_BACKWARD_COMMAND = ("AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER")
_FORWARD_COMMAND = ("AVBL", "AVBR", "PVCL", "PVCR")


def run_digital_simulation(
    connectome: ConnectomeData,
    n_ticks: int,
    *,
    sensory_neuron_indices: list[int],
    noise_level: float = SENSORY_NOISE_LEVEL,
    seed: int = RNG_SEED,
    pre_eq_ticks: int = 2000,
) -> tuple[np.ndarray, list[str]]:
    """Run the v0.3 CTRNN with random sensory drive.

    Returns:
        history: (n_ticks, N) activity matrix.
        names: list of N neuron names (same order as columns).
    """
    state = NeuralState.initialize(connectome.n_neurons, seed=seed)
    params = CTRNNParams()       # default β=1 tanh
    rng = np.random.default_rng(seed)

    # Pre-equilibrate without input.
    zero_in = np.zeros(connectome.n_neurons)
    for _ in range(pre_eq_ticks):
        state = neural_step(state, connectome, zero_in, params, rng)

    # Random Gaussian drive at sensory neurons only.
    history = np.zeros((n_ticks, connectome.n_neurons), dtype=np.float32)
    sens = np.zeros(connectome.n_neurons)
    for t in range(n_ticks):
        # Independent draw per sensory neuron at each tick.
        sens[:] = 0.0
        sens[sensory_neuron_indices] = (
            rng.standard_normal(len(sensory_neuron_indices)) * noise_level
        )
        state = neural_step(state, connectome, sens, params, rng)
        history[t] = state.V
    return history, connectome.neuron_names


def run_behavior_conditioned_simulation(
    connectome: ConnectomeData,
    reversal_vec: np.ndarray,
    *,
    drive_strength: float = 0.5,
    seed: int = RNG_SEED,
    pre_eq_ticks: int = 2000,
    sensory_noise_level: float = SENSORY_NOISE_LEVEL,
) -> tuple[np.ndarray, list[str]]:
    """Drive command neurons in time with the real recording's behavior.

    Whenever `reversal_vec[t] == 1` (worm reversing), drive the backward
    command neurons. Otherwise drive the forward command neurons. A small
    amount of random Gaussian drive is also added at every sensory neuron
    to keep the rest of the network alive. This isolates "does the
    connectome propagate a forward-vs-backward command into the same
    activity patterns we see in real worms?" from the question of whether
    we have the right sensory inputs.
    """
    n = connectome.n_neurons
    state = NeuralState.initialize(n, seed=seed)
    params = CTRNNParams()
    rng = np.random.default_rng(seed)
    zero_in = np.zeros(n)
    for _ in range(pre_eq_ticks):
        state = neural_step(state, connectome, zero_in, params, rng)

    sens_idx = _sensory_neuron_indices(connectome)
    back_idx = [connectome.idx(name) for name in _BACKWARD_COMMAND]
    fwd_idx = [connectome.idx(name) for name in _FORWARD_COMMAND]

    n_ticks = int(reversal_vec.shape[0])
    history = np.zeros((n_ticks, n), dtype=np.float32)
    drive = np.zeros(n)
    for t in range(n_ticks):
        drive[:] = 0.0
        drive[sens_idx] = rng.standard_normal(len(sens_idx)) * sensory_noise_level
        if reversal_vec[t]:
            for i in back_idx:
                drive[i] = drive_strength
        else:
            for i in fwd_idx:
                drive[i] = drive_strength
        state = neural_step(state, connectome, drive, params, rng)
        history[t] = state.V
    return history, connectome.neuron_names


def compare_one_recording(
    digital_history: np.ndarray,
    digital_names: list[str],
    real_recording,
) -> dict:
    """Compute the 3 metrics for one real recording."""
    # Build the (T_real, n_labeled) real matrix and the parallel name list.
    real_cols = []
    real_names = []
    for col_idx, name in real_recording.labels.items():
        real_cols.append(col_idx)
        real_names.append(name)
    real_matrix = real_recording.traces[:, real_cols]

    results = run_all_metrics(
        digital_history, digital_names, real_matrix, real_names,
    )
    return {
        "recording_id": real_recording.recording_id,
        "n_real_labeled": int(real_recording.n_labeled),
        "metrics": {r.name: {"score": r.score, "details": r.details}
                    for r in results},
    }


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    n_recordings = (
        int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit()
        else N_RECORDINGS
    )

    connectome = ConnectomeData.load()
    print(f"loading top-{n_recordings} Atanas 2023 recordings…", flush=True)
    reference = ReferenceDataset.from_atanas2023(max_recordings=n_recordings)

    sensory_idx = _sensory_neuron_indices(connectome)
    print(f"sensory neurons in connectome: {len(sensory_idx)}")

    # Two digital protocols per recording:
    #   A) random sensory noise only (pure topology baseline)
    #   B) behavior-conditioned (drive AVA when real worm reverses, AVB else)
    # Protocol B isolates "is the right command-neuron drive enough?".
    per_recording_baseline = []
    per_recording_behavior = []
    t0 = time.time()
    for r in reference.recordings:
        print(f"  digital runs for {r.recording_id} (T={r.n_timepoints})…",
              flush=True)
        seed = hash(r.recording_id) & 0x7FFFFFFF

        digital_a, names_a = run_digital_simulation(
            connectome, r.n_timepoints,
            sensory_neuron_indices=sensory_idx,
            seed=seed,
        )
        per_recording_baseline.append(
            compare_one_recording(digital_a, names_a, r)
        )

        digital_b, names_b = run_behavior_conditioned_simulation(
            connectome, r.reversal.astype(int),
            seed=seed,
        )
        per_recording_behavior.append(
            compare_one_recording(digital_b, names_b, r)
        )
    elapsed = time.time() - t0

    # Aggregate per metric, per protocol.
    metric_names = list(per_recording_baseline[0]["metrics"].keys())

    def _aggregate(per_rec: list[dict]) -> dict:
        out = {}
        for m in metric_names:
            scores = [
                rec["metrics"][m]["score"]
                for rec in per_rec
                if not np.isnan(rec["metrics"][m]["score"])
            ]
            out[m] = {
                "n_recordings": len(scores),
                "mean": float(np.mean(scores)) if scores else float("nan"),
                "median": float(np.median(scores)) if scores else float("nan"),
                "min": float(np.min(scores)) if scores else float("nan"),
                "max": float(np.max(scores)) if scores else float("nan"),
            }
        return out

    agg_baseline = _aggregate(per_recording_baseline)
    agg_behavior = _aggregate(per_recording_behavior)

    # ---- Text report --------------------------------------------------------
    report_path = OUTPUT_DIR / "comparison_report.txt"
    with report_path.open("w") as f:
        f.write("AC0.5.2 — Digital vs. real (Atanas 2023) similarity report\n")
        f.write("=" * 64 + "\n")
        f.write(f"recordings:        {len(per_recording_baseline)}\n")
        f.write(f"sensory neurons:   {len(sensory_idx)}\n")
        f.write(f"digital noise σ:   {SENSORY_NOISE_LEVEL}\n")
        f.write(f"runtime:           {elapsed:.1f}s\n\n")

        f.write("Protocol A — random sensory drive (pure topology baseline):\n")
        for m in metric_names:
            a = agg_baseline[m]
            f.write(
                f"  {m:42s}  mean={a['mean']:+.4f}  "
                f"median={a['median']:+.4f}  "
                f"range=[{a['min']:+.4f}, {a['max']:+.4f}]\n"
            )
        f.write("\n")

        f.write("Protocol B — behavior-conditioned drive (AVA in real reversals,\n")
        f.write("             AVB otherwise; isolates 'is command drive enough?'):\n")
        for m in metric_names:
            a = agg_behavior[m]
            f.write(
                f"  {m:42s}  mean={a['mean']:+.4f}  "
                f"median={a['median']:+.4f}  "
                f"range=[{a['min']:+.4f}, {a['max']:+.4f}]\n"
            )
        f.write("\n")

        f.write("Per-recording details (protocol A then B):\n")
        for ra, rb in zip(per_recording_baseline, per_recording_behavior):
            assert ra["recording_id"] == rb["recording_id"]
            f.write(f"--- {ra['recording_id']}  (labeled={ra['n_real_labeled']}) ---\n")
            for m in metric_names:
                f.write(
                    f"  {m:42s}  A={ra['metrics'][m]['score']:+.4f}   "
                    f"B={rb['metrics'][m]['score']:+.4f}\n"
                )
            f.write("\n")

    # ---- JSON ---------------------------------------------------------------
    json_path = OUTPUT_DIR / "comparison_results.json"
    json_payload = {
        "settings": {
            "n_recordings": len(per_recording_baseline),
            "sensory_noise_level": SENSORY_NOISE_LEVEL,
            "rng_seed": RNG_SEED,
        },
        "protocol_A_random_drive": {
            "aggregate": agg_baseline,
            "per_recording": per_recording_baseline,
        },
        "protocol_B_behavior_conditioned": {
            "aggregate": agg_behavior,
            "per_recording": per_recording_behavior,
        },
    }

    def _to_python(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, dict):
            return {k: _to_python(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_to_python(v) for v in o]
        return o

    json_path.write_text(json.dumps(_to_python(json_payload), indent=2))

    print(f"\nwrote {report_path}")
    print(f"wrote {json_path}\n")
    print(f"Aggregate scores ({len(per_recording_baseline)} recordings):")
    for m in metric_names:
        a, b = agg_baseline[m], agg_behavior[m]
        print(f"  {m:42s}  A_mean={a['mean']:+.4f}  B_mean={b['mean']:+.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
