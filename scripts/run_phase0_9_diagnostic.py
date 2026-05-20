"""Phase 0.9 — FC-gap diagnostic with RID modulator.

Reuses the per-pair functional-connectivity diagnosis from
`scripts/run_fc_gap_diagnosis.py`, but on three configurations:

  - real (mean over 10 Atanas 2023 recordings)
  - category (Phase 0.8.2)
  - rid (Phase 0.8.2 + RID modulator, tau=200, gain=0.5)

It then asks the specific questions from `logs/phase0.9_brief.md` §8:

  1. fc_similarity (already in `phase0_9_comparison_report`).
  2. Sign-reversal proportion (FC_real and FC_digital opposite signs)
     for category vs rid, on the strict-intersection neuron set used in
     Phase 0.8's diagnostic.
  3. Top-50 |diff| RID-hub improvement — how do pairs involving RID-
     adjacent neurons (RID itself + reversal pool + AVB/PVC) change?
  4. Anti-correlation production — does the digital model now produce
     FC values < -0.1?

Outputs:
  output/phase0_9_diagnostic_report.txt
  output/phase0_9_diagnostic_results.json
  output/phase0_9_diagnostic_heatmap.png
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from algos.config import OUTPUT_DIR
from algos.connectome import ConnectomeData
from algos.neural import (
    HeterogeneousNetwork,
    RIDModulator,
    REVERSAL_COMMAND_NEURONS,
    from_category_defaults,
)
from algos.validation.reference_data import ReferenceDataset


N_WORMS = 10
PRE_EQ_TICKS = 2000
SENSORY_NOISE = 0.1
TAU_RID = 200.0
MOD_GAIN = 0.5

RID_RELATED_NEURONS: tuple[str, ...] = (
    "RID",
    *REVERSAL_COMMAND_NEURONS,  # AVAL/AVAR/AVDL/AVDR/AVEL/AVER
    "AVBL", "AVBR", "PVCL", "PVCR",  # forward command — adjacent in circuit
)


def matched_neuron_set(recordings, connectome) -> list[str]:
    """Strict intersection: neurons labeled in *all* recordings and in connectome."""
    sets = [set(r.labels.values()) for r in recordings]
    inter = set.intersection(*sets)
    conn_set = set(connectome.neuron_names)
    return sorted(n for n in inter if n in conn_set)


def real_fc_per_recording(recording, neuron_names: list[str]) -> np.ndarray:
    cols, found = [], []
    for col, name in recording.labels.items():
        if name in neuron_names:
            cols.append(col)
            found.append(name)
    order = [found.index(n) for n in neuron_names]
    cols = [cols[i] for i in order]
    sub = recording.traces[:, cols]
    return np.corrcoef(sub, rowvar=False)


def digital_fc_per_recording(
    network: HeterogeneousNetwork,
    recording,
    neuron_names: list[str],
    *,
    seed: int,
    modulator: RIDModulator | None = None,
) -> np.ndarray:
    n = network.connectome.n_neurons
    state = network.initial_state(seed=seed)
    rng = np.random.default_rng(seed)
    sens_idx = network.connectome.get_neuron_indices_by_category("sensory")
    if modulator is not None:
        modulator.reset()
    zero = np.zeros(n)
    for _ in range(PRE_EQ_TICKS):
        state = network.step(state, zero, rng, modulator=modulator)
    T = recording.n_timepoints
    history = np.zeros((T, n), dtype=np.float32)
    sens = np.zeros(n)
    for t in range(T):
        sens[:] = 0.0
        sens[sens_idx] = rng.standard_normal(len(sens_idx)) * SENSORY_NOISE
        state = network.step(state, sens, rng, modulator=modulator)
        history[t] = state.V
    cols = [network.connectome.idx(nm) for nm in neuron_names]
    return np.corrcoef(history[:, cols], rowvar=False)


def sign_reversal_fraction(fc_real: np.ndarray, fc_digital: np.ndarray,
                           threshold: float = 0.05) -> float:
    """Fraction of off-diagonal pairs where FC_real and FC_digital have
    opposite signs, both with |FC| > threshold."""
    n = fc_real.shape[0]
    iu = np.triu_indices(n, k=1)
    a = fc_real[iu]
    b = fc_digital[iu]
    significant = (np.abs(a) > threshold) & (np.abs(b) > threshold)
    if significant.sum() == 0:
        return float("nan")
    flipped = (np.sign(a) != np.sign(b)) & significant
    return float(flipped.sum() / significant.sum())


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"loading top-{N_WORMS} Atanas 2023 recordings…", flush=True)
    reference = ReferenceDataset.from_atanas2023(max_recordings=N_WORMS)
    connectome = ConnectomeData.load()

    matched = matched_neuron_set(reference.recordings, connectome)
    n_m = len(matched)
    matched_cat = [connectome.category[connectome.idx(n)] for n in matched]
    print(f"matched neurons (strict intersection): {n_m}")
    print(f"categories: {Counter(matched_cat)}")

    # --- Real FC -------------------------------------------------------------
    real_fcs = [real_fc_per_recording(r, matched) for r in reference.recordings]
    fc_real = np.nanmean(np.stack(real_fcs), axis=0)

    # --- Digital FC: category baseline ---------------------------------------
    print("running category-baseline sims…", flush=True)
    net_cat = from_category_defaults(connectome)
    fc_cat_list = []
    for k_idx, r in enumerate(reference.recordings):
        seed = 1000 + k_idx
        fc_cat_list.append(digital_fc_per_recording(
            net_cat, r, matched, seed=seed, modulator=None
        ))
    fc_cat = np.nanmean(np.stack(fc_cat_list), axis=0)

    # --- Digital FC: with RID modulator --------------------------------------
    print("running rid-modulated sims…", flush=True)
    net_rid = from_category_defaults(connectome)
    fc_rid_list = []
    final_c_RIDs = []
    for k_idx, r in enumerate(reference.recordings):
        seed = 1000 + k_idx
        modulator = RIDModulator.from_connectome(
            connectome, tau=TAU_RID, gain=MOD_GAIN
        )
        fc_rid_list.append(digital_fc_per_recording(
            net_rid, r, matched, seed=seed, modulator=modulator
        ))
        final_c_RIDs.append(modulator.c_RID)
    fc_rid = np.nanmean(np.stack(fc_rid_list), axis=0)

    # --- Sign-reversal proportions -------------------------------------------
    sr_cat = sign_reversal_fraction(fc_real, fc_cat)
    sr_rid = sign_reversal_fraction(fc_real, fc_rid)

    # --- Anti-correlation production (digital pairs < -0.1) ------------------
    iu = np.triu_indices(n_m, k=1)
    anticorr_real = float(((fc_real[iu] < -0.1)).mean())
    anticorr_cat = float(((fc_cat[iu] < -0.1)).mean())
    anticorr_rid = float(((fc_rid[iu] < -0.1)).mean())

    # --- Per-pair diff vs real -----------------------------------------------
    diff_cat = fc_real - fc_cat
    diff_rid = fc_real - fc_rid

    pair_records = []
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        pair_records.append({
            "neuron_a": matched[i],
            "neuron_b": matched[j],
            "category_a": matched_cat[i],
            "category_b": matched_cat[j],
            "fc_real": float(fc_real[i, j]),
            "fc_cat": float(fc_cat[i, j]),
            "fc_rid": float(fc_rid[i, j]),
            "diff_cat": float(diff_cat[i, j]),
            "diff_rid": float(diff_rid[i, j]),
            "abs_diff_cat": float(abs(diff_cat[i, j])),
            "abs_diff_rid": float(abs(diff_rid[i, j])),
            "improvement_rid_vs_cat":
                float(abs(diff_cat[i, j]) - abs(diff_rid[i, j])),
        })

    # --- Top-50 |diff_cat| pairs — does RID modulation help these? ----------
    pair_records.sort(key=lambda x: -x["abs_diff_cat"])
    top50 = pair_records[:50]
    improvement_top50 = float(
        np.mean([p["improvement_rid_vs_cat"] for p in top50])
    )
    n_improved = sum(1 for p in top50 if p["improvement_rid_vs_cat"] > 0)
    n_worsened = sum(1 for p in top50 if p["improvement_rid_vs_cat"] < 0)

    # --- RID-related pairs specifically --------------------------------------
    rid_pairs = [
        p for p in pair_records
        if p["neuron_a"] in RID_RELATED_NEURONS
           or p["neuron_b"] in RID_RELATED_NEURONS
    ]
    rid_improvement = float(
        np.mean([p["improvement_rid_vs_cat"] for p in rid_pairs])
    ) if rid_pairs else float("nan")

    # Also, of the top-50 by |diff_cat|, which involve a RID-related neuron?
    rid_in_top50 = [
        p for p in top50
        if p["neuron_a"] in RID_RELATED_NEURONS
           or p["neuron_b"] in RID_RELATED_NEURONS
    ]
    rid_top50_improvement = float(
        np.mean([p["improvement_rid_vs_cat"] for p in rid_in_top50])
    ) if rid_in_top50 else float("nan")

    elapsed = time.time() - t0

    # --- Heatmap (3-panel) ---------------------------------------------------
    cat_order = [
        "sensory", "interneuron", "motor",
        "pharyngeal", "other_neuron", "sex_specific",
    ]
    order = sorted(
        range(n_m),
        key=lambda i: (
            cat_order.index(matched_cat[i])
            if matched_cat[i] in cat_order else 999,
            matched[i],
        ),
    )
    fc_real_s = fc_real[np.ix_(order, order)]
    fc_cat_s = fc_cat[np.ix_(order, order)]
    fc_rid_s = fc_rid[np.ix_(order, order)]
    matched_s = [matched[i] for i in order]
    cat_s = [matched_cat[i] for i in order]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    for ax, M, title in [
        (axes[0, 0], fc_real_s, "FC_real (10-worm mean)"),
        (axes[0, 1], fc_cat_s, "FC_category (Phase 0.8.2)"),
        (axes[0, 2], fc_rid_s, "FC_rid (Phase 0.9, g=0.5)"),
        (axes[1, 0], fc_real_s - fc_cat_s, "real − category"),
        (axes[1, 1], fc_real_s - fc_rid_s, "real − rid"),
        (axes[1, 2], fc_rid_s - fc_cat_s, "rid − category (Δ from RID)"),
    ]:
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1,
                       interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(range(n_m))
        ax.set_yticks(range(n_m))
        ax.set_xticklabels(matched_s, rotation=90, fontsize=5)
        ax.set_yticklabels(matched_s, fontsize=5)
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    cat_colors = {
        "sensory": "#1f77b4",
        "interneuron": "#2ca02c",
        "motor": "#d62728",
        "pharyngeal": "#888888",
        "other_neuron": "#bcbd22",
        "sex_specific": "#9467bd",
    }
    handles = [
        mpatches.Patch(color=cat_colors.get(c, "#cccccc"), label=c)
        for c in set(cat_s)
    ]
    axes[0, 0].legend(handles=handles, loc="upper left",
                      fontsize=6, framealpha=0.95)
    plt.tight_layout()
    heatmap_path = OUTPUT_DIR / "phase0_9_diagnostic_heatmap.png"
    fig.savefig(heatmap_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- JSON ----------------------------------------------------------------
    json_path = OUTPUT_DIR / "phase0_9_diagnostic_results.json"
    json_path.write_text(json.dumps({
        "settings": {
            "n_worms": N_WORMS,
            "n_matched_neurons": n_m,
            "pre_eq_ticks": PRE_EQ_TICKS,
            "sensory_noise": SENSORY_NOISE,
            "tau_rid": TAU_RID,
            "mod_gain": MOD_GAIN,
        },
        "matched_neurons": matched,
        "matched_categories": matched_cat,
        "sign_reversal_fraction": {
            "category": sr_cat,
            "rid": sr_rid,
            "delta": sr_rid - sr_cat,
        },
        "anticorr_fraction": {
            "real": anticorr_real,
            "category": anticorr_cat,
            "rid": anticorr_rid,
        },
        "top50_diff_cat_improvement_mean_abs": improvement_top50,
        "top50_n_improved": n_improved,
        "top50_n_worsened": n_worsened,
        "rid_related_pairs": {
            "n": len(rid_pairs),
            "mean_improvement": rid_improvement,
        },
        "rid_in_top50": {
            "n": len(rid_in_top50),
            "mean_improvement": rid_top50_improvement,
        },
        "final_c_RID_per_recording": final_c_RIDs,
        "top_pairs_by_abs_diff_cat": top50,
        "runtime_sec": elapsed,
    }, indent=2))

    # --- Text report ---------------------------------------------------------
    report_path = OUTPUT_DIR / "phase0_9_diagnostic_report.txt"
    with report_path.open("w") as f:
        f.write("Phase 0.9 — FC-gap diagnostic with RID modulator\n")
        f.write("=" * 78 + "\n")
        f.write(f"recordings:        {N_WORMS}\n")
        f.write(f"matched neurons:   {n_m}\n")
        f.write(f"matched pairs:     {len(iu[0])}\n")
        f.write(f"tau_rid:           {TAU_RID}\n")
        f.write(f"mod_gain:          {MOD_GAIN}\n")
        f.write(f"runtime:           {elapsed:.1f}s\n\n")

        f.write("category breakdown of matched neurons:\n")
        for c, k in Counter(matched_cat).most_common():
            f.write(f"  {c:14s}: {k}\n")
        f.write("\n")

        f.write("Sign-reversal fraction (|FC_real| > 0.05 and |FC_dig| > 0.05):\n")
        f.write(f"  category:  {sr_cat:.4f}\n")
        f.write(f"  rid:       {sr_rid:.4f}\n")
        f.write(f"  Δ:         {sr_rid - sr_cat:+.4f}  "
                f"(negative = RID modulator reduces sign reversals)\n\n")

        f.write("Anti-correlation production (fraction of pairs with FC < -0.1):\n")
        f.write(f"  real:      {anticorr_real:.4f}\n")
        f.write(f"  category:  {anticorr_cat:.4f}\n")
        f.write(f"  rid:       {anticorr_rid:.4f}\n\n")

        f.write("Top-50 |diff_cat| pairs — improvement under RID modulation:\n")
        f.write(f"  mean(|diff_cat| - |diff_rid|) = {improvement_top50:+.4f}\n")
        f.write(f"  n improved (improvement > 0):  {n_improved}/50\n")
        f.write(f"  n worsened (improvement < 0):  {n_worsened}/50\n\n")

        f.write("RID-related pairs (any pair touching "
                f"{', '.join(RID_RELATED_NEURONS)}):\n")
        f.write(f"  n_pairs:           {len(rid_pairs)}\n")
        f.write(f"  mean improvement:  {rid_improvement:+.4f}\n")
        f.write(f"  of these in top-50: {len(rid_in_top50)}\n")
        f.write(f"  mean improvement in top-50 subset: "
                f"{rid_top50_improvement:+.4f}\n\n")

        f.write(f"Final c_RID per recording (mean = "
                f"{float(np.mean(final_c_RIDs)):+.4f}, "
                f"std = {float(np.std(final_c_RIDs)):.4f}):\n")
        for k, c_val in enumerate(final_c_RIDs):
            f.write(f"  rec {k}: c_RID = {c_val:+.4f}\n")
        f.write("\n")

        f.write("Top-20 |diff_cat| pairs with RID-modulation effect:\n")
        f.write(
            f"  {'pair':22s}  {'cat_a':12s} {'cat_b':12s}  "
            f"{'FC_real':>7s} {'FC_cat':>7s} {'FC_rid':>7s}  "
            f"{'Δ_help':>7s}\n"
        )
        for p in top50[:20]:
            f.write(
                f"  {p['neuron_a']+'—'+p['neuron_b']:22s}  "
                f"{p['category_a']:12s} {p['category_b']:12s}  "
                f"{p['fc_real']:+7.3f} {p['fc_cat']:+7.3f} {p['fc_rid']:+7.3f}  "
                f"{p['improvement_rid_vs_cat']:+7.3f}\n"
            )

    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    print(f"wrote {heatmap_path}\n")
    with report_path.open() as f:
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
