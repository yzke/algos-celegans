"""Phase 0.6 — internal audit of the PCA-structure similarity score.

Runs four conditions on a single Atanas 2023 recording (the
best-annotated one, 2022-08-02-01) and reports per-condition
distributions of PCA-similarity scores.

Conditions:
  - `real`     — true Cook 2019 connectome + v0.3 tanh dynamics
  - `shuffle`  — connectome with W_chem / W_gap entries shuffled to
                 random positions but with original values preserved
  - `transpose`— W_chem.T (W_gap unchanged because it's symmetric)
  - `relu`     — true connectome, but chem_input uses ReLU(V) instead
                 of tanh(β·V)

Outputs:
  output/pca_audit_results.json   — full per-trial scores
  output/pca_audit_report.txt     — human-readable summary
  PHASE0.6_AUDIT.md §§4–5         — filled in by this script
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from algos.config import OUTPUT_DIR
from algos.connectome import ConnectomeData
from algos.neural import CTRNNParams, NeuralState, neural_step
from algos.neural.dynamics import sigmoid  # noqa: F401  (kept for parity)
from algos.validation.reference_data import ReferenceDataset
from algos.validation.comparison import (
    match_matrices,
    pca_structure_similarity,
)


N_TRIALS_DIST = 50         # null + reference distributions
N_TRIALS_POINT = 10        # transpose + ReLU point estimates
PRE_EQ_TICKS = 2000
SENSORY_NOISE = 0.1
RECORDING_ID = "2022-08-02-01"   # most labeled
N_PCA_COMPONENTS = 10


# ---------------------------------------------------------------------------
# Connectome variants
# ---------------------------------------------------------------------------


def _renormalize_per_row(W: np.ndarray) -> np.ndarray:
    row_l1 = np.abs(W).sum(axis=1, keepdims=True)
    return W / np.maximum(row_l1, 1.0)


def shuffled_connectome(orig: ConnectomeData, seed: int) -> ConnectomeData:
    """Return a copy whose W_chem / W_gap have nonzero positions randomly
    re-placed but values preserved (in shuffled order).

    Total nnz, value distribution, and the per-row-L1 normalization are
    preserved on the *post-shuffle* matrix.
    """
    rng = np.random.default_rng(seed)
    new = copy.deepcopy(orig)
    N = orig.n_neurons

    # ---- W_chem -----------------------------------------------------------
    flat_idx = np.flatnonzero(orig.W_chem != 0.0)
    values = orig.W_chem.ravel()[flat_idx]
    rng.shuffle(values)
    # New positions: random sample without replacement from off-diagonal cells.
    off_diag_mask = ~np.eye(N, dtype=bool)
    off_diag_idx = np.flatnonzero(off_diag_mask)
    chosen = rng.choice(off_diag_idx, size=len(flat_idx), replace=False)
    new_W = np.zeros_like(orig.W_chem)
    new_W.ravel()[chosen] = values
    new.W_chem = _renormalize_per_row(new_W)

    # ---- W_gap (symmetric) -------------------------------------------------
    upper = np.triu(orig.W_gap, k=1)
    upper_flat_idx = np.flatnonzero(upper != 0.0)
    upper_values = upper.ravel()[upper_flat_idx]
    rng.shuffle(upper_values)
    # Sample upper-triangle off-diagonal positions.
    triu_off_diag_idx = np.flatnonzero(np.triu(np.ones_like(orig.W_gap), k=1))
    chosen_up = rng.choice(triu_off_diag_idx, size=len(upper_flat_idx),
                           replace=False)
    new_upper = np.zeros_like(orig.W_gap)
    new_upper.ravel()[chosen_up] = upper_values
    new_gap = new_upper + new_upper.T
    new.W_gap = _renormalize_per_row(new_gap)
    # Re-symmetrize after independent per-row L1.
    new.W_gap = 0.5 * (new.W_gap + new.W_gap.T)
    np.fill_diagonal(new.W_gap, 0.0)
    return new


def transposed_connectome(orig: ConnectomeData) -> ConnectomeData:
    """W_chem becomes W_chem.T (then re-normalized per row). W_gap unchanged."""
    new = copy.deepcopy(orig)
    new.W_chem = _renormalize_per_row(orig.W_chem.T)
    # W_gap stays as-is (symmetric).
    return new


# ---------------------------------------------------------------------------
# Dynamics variants
# ---------------------------------------------------------------------------


def step_tanh(state: NeuralState, connectome: ConnectomeData,
              sensory: np.ndarray, params: CTRNNParams,
              rng: np.random.Generator) -> NeuralState:
    """Standard v0.3 tanh(β·V) step — same as algos.neural.neural_step."""
    return neural_step(state, connectome, sensory, params, rng)


def step_relu(state: NeuralState, connectome: ConnectomeData,
              sensory: np.ndarray, params: CTRNNParams,
              rng: np.random.Generator) -> NeuralState:
    """Step using ReLU(V) for the chem activation instead of tanh(β·V)."""
    V = state.V
    chem_input = connectome.W_chem @ np.maximum(V, 0.0)
    gap_input = connectome.W_gap @ V - V * connectome.W_gap.sum(axis=1)
    if params.noise_level > 0.0:
        noise = rng.standard_normal(V.shape[0]) * params.noise_level
    else:
        noise = 0.0
    dV = (-V + chem_input + gap_input + sensory + noise) / params.tau
    V_new = np.clip(V + dV, -1.0, 1.0)
    return NeuralState(V=V_new, tick=state.tick + 1)


# ---------------------------------------------------------------------------
# One trial
# ---------------------------------------------------------------------------


def run_one_trial(
    connectome: ConnectomeData,
    real_recording,
    *,
    sim_seed: int,
    step_fn=step_tanh,
    pre_eq_ticks: int = PRE_EQ_TICKS,
    noise_level: float = SENSORY_NOISE,
) -> dict:
    """Run digital sim + compute PCA-structure similarity vs the recording."""
    n = connectome.n_neurons
    state = NeuralState.initialize(n, seed=sim_seed)
    params = CTRNNParams()
    rng = np.random.default_rng(sim_seed)

    sens_idx = connectome.get_neuron_indices_by_category("sensory")
    zero_in = np.zeros(n)
    for _ in range(pre_eq_ticks):
        state = step_fn(state, connectome, zero_in, params, rng)

    T = real_recording.n_timepoints
    history = np.zeros((T, n), dtype=np.float32)
    sens = np.zeros(n)
    for t in range(T):
        sens[:] = 0.0
        sens[sens_idx] = rng.standard_normal(len(sens_idx)) * noise_level
        state = step_fn(state, connectome, sens, params, rng)
        history[t] = state.V

    digital_max_abs = float(np.max(np.abs(history)))
    digital_mean_abs = float(np.mean(np.abs(history)))

    real_cols = list(real_recording.labels.keys())
    real_names = [real_recording.labels[c] for c in real_cols]
    real_matrix = real_recording.traces[:, real_cols]

    res = pca_structure_similarity(
        history, connectome.neuron_names,
        real_matrix, real_names,
        n_components=N_PCA_COMPONENTS,
    )
    return {
        "combined": float(res.score),
        "explained_variance_cos": float(res.details["explained_variance_cos"]),
        "subspace_alignment": float(res.details["subspace_alignment"]),
        "n_matched": int(res.details["n_matched"]),
        "digital_max_abs": digital_max_abs,
        "digital_mean_abs": digital_mean_abs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("loading connectome…", flush=True)
    connectome = ConnectomeData.load()
    print("loading reference recording…", flush=True)
    reference = ReferenceDataset.from_atanas2023(
        recording_ids=[RECORDING_ID],
    )
    recording = reference.by_id(RECORDING_ID)
    print(f"  {RECORDING_ID}: T={recording.n_timepoints} "
          f"labeled={recording.n_labeled}\n")

    t0 = time.time()
    results = {
        "real": [],
        "shuffle": [],
        "transpose": [],
        "relu": [],
    }

    # --- real connectome, varying sim seed ---------------------------------
    print(f"[1/4] real connectome × {N_TRIALS_DIST} seeds…", flush=True)
    for k in range(N_TRIALS_DIST):
        r = run_one_trial(connectome, recording, sim_seed=1000 + k)
        results["real"].append(r)
    print(f"      mean combined: {np.mean([r['combined'] for r in results['real']]):.4f}\n")

    # --- shuffled connectome, varying shuffle and sim seed -----------------
    print(f"[2/4] shuffled connectome × {N_TRIALS_DIST} seeds…", flush=True)
    for k in range(N_TRIALS_DIST):
        shuf = shuffled_connectome(connectome, seed=2000 + k)
        r = run_one_trial(shuf, recording, sim_seed=2000 + k)
        results["shuffle"].append(r)
    print(f"      mean combined: {np.mean([r['combined'] for r in results['shuffle']]):.4f}\n")

    # --- transposed connectome ---------------------------------------------
    print(f"[3/4] transposed W_chem × {N_TRIALS_POINT} seeds…", flush=True)
    trans = transposed_connectome(connectome)
    for k in range(N_TRIALS_POINT):
        r = run_one_trial(trans, recording, sim_seed=3000 + k)
        results["transpose"].append(r)
    print(f"      mean combined: {np.mean([r['combined'] for r in results['transpose']]):.4f}\n")

    # --- ReLU activation on the real connectome ----------------------------
    print(f"[4/4] real connectome + ReLU × {N_TRIALS_POINT} seeds…", flush=True)
    for k in range(N_TRIALS_POINT):
        r = run_one_trial(connectome, recording, sim_seed=4000 + k,
                          step_fn=step_relu)
        results["relu"].append(r)
    print(f"      mean combined: {np.mean([r['combined'] for r in results['relu']]):.4f}\n")

    elapsed = time.time() - t0
    print(f"total runtime: {elapsed:.1f}s\n")

    # --- aggregate ----------------------------------------------------------
    def stats(rs: list[dict], key: str) -> dict:
        v = np.array([r[key] for r in rs])
        return {
            "n": int(v.size),
            "mean": float(v.mean()),
            "median": float(np.median(v)),
            "std": float(v.std()),
            "p2_5": float(np.percentile(v, 2.5)),
            "p97_5": float(np.percentile(v, 97.5)),
            "min": float(v.min()),
            "max": float(v.max()),
        }

    summary = {}
    for cond, rs in results.items():
        summary[cond] = {
            "combined": stats(rs, "combined"),
            "explained_variance_cos": stats(rs, "explained_variance_cos"),
            "subspace_alignment": stats(rs, "subspace_alignment"),
            "n_matched": rs[0]["n_matched"],
            "digital_max_abs_mean": float(np.mean([r["digital_max_abs"]
                                                   for r in rs])),
        }

    # --- significance -------------------------------------------------------
    real_combined = np.array([r["combined"] for r in results["real"]])
    shuf_combined = np.array([r["combined"] for r in results["shuffle"]])
    # Fraction of shuffle scores >= mean of real:
    frac_shuf_above_real_mean = float(
        np.mean(shuf_combined >= real_combined.mean())
    )
    # One-sided test: is real reliably higher than shuffle?
    real_p2_5 = float(np.percentile(real_combined, 2.5))
    shuf_p97_5 = float(np.percentile(shuf_combined, 97.5))
    distributions_overlap = bool(real_p2_5 <= shuf_p97_5)

    # --- JSON output --------------------------------------------------------
    json_path = OUTPUT_DIR / "pca_audit_results.json"
    payload = {
        "settings": {
            "recording_id": RECORDING_ID,
            "n_trials_dist": N_TRIALS_DIST,
            "n_trials_point": N_TRIALS_POINT,
            "n_pca_components": N_PCA_COMPONENTS,
            "pre_eq_ticks": PRE_EQ_TICKS,
            "sensory_noise": SENSORY_NOISE,
        },
        "summary": summary,
        "significance": {
            "frac_shuffle_>=_real_mean": frac_shuf_above_real_mean,
            "real_p2_5": real_p2_5,
            "shuffle_p97_5": shuf_p97_5,
            "distributions_overlap_95ci": distributions_overlap,
        },
        "trials": results,
        "runtime_sec": elapsed,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    # --- Text report --------------------------------------------------------
    report_path = OUTPUT_DIR / "pca_audit_report.txt"
    with report_path.open("w") as f:
        f.write("Phase 0.6 — PCA-similarity internal audit\n")
        f.write("=" * 60 + "\n")
        f.write(f"recording:     {RECORDING_ID} (labeled={recording.n_labeled})\n")
        f.write(f"runtime:       {elapsed:.1f}s\n\n")

        f.write(
            "                    n   combined        ev_cos          subspace_align\n"
        )
        f.write(
            "                       mean   95%CI      mean   95%CI      mean   95%CI\n"
        )
        for cond, s in summary.items():
            c = s["combined"]
            e = s["explained_variance_cos"]
            a = s["subspace_alignment"]
            f.write(
                f"  {cond:10s}  {c['n']:3d}   "
                f"{c['mean']:+.3f}±{c['std']:.3f}  "
                f"[{c['p2_5']:+.3f}, {c['p97_5']:+.3f}]    "
                f"{e['mean']:+.3f}±{e['std']:.3f}    "
                f"{a['mean']:+.3f}±{a['std']:.3f}\n"
            )
        f.write("\n")
        f.write(f"frac(shuffle >= real_mean):    {frac_shuf_above_real_mean:.3f}\n")
        f.write(f"real 2.5%:                     {real_p2_5:+.4f}\n")
        f.write(f"shuffle 97.5%:                 {shuf_p97_5:+.4f}\n")
        f.write(f"distributions overlap (95%CI): {distributions_overlap}\n")

    print(f"wrote {report_path}")
    print(f"wrote {json_path}\n")
    with report_path.open() as f:
        print(f.read())

    return 0


if __name__ == "__main__":
    sys.exit(main())
