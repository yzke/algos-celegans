"""Phase 0.8 — breakdown of where the functional-connectivity gap lives.

Phase 0.7 showed digital fc_similarity = +0.03 vs real cross-worm
fc_similarity ≈ +0.48 — a +0.45 gap. The gap is unlikely to be uniform.
This script:

  1. Computes the per-pair FC matrices on the 29-neuron strict
     intersection across the same 10 best-labeled Atanas 2023
     recordings used in Phase 0.7.
  2. Averages real FC across recordings, averages digital FC across
     length-matched bare-CTRNN simulations.
  3. Reports:
       a) heatmaps of FC_real, FC_digital, signed FC_diff
       b) top-20 highest-|diff| neuron pairs with categories
       c) category-combo aggregates (e.g. mean |diff| over all
          sensory–sensory pairs)
       d) hub analysis: neurons most frequently in the top-50 |diff|
          pairs
       e) per-pair `diff = FC_real - FC_digital` written as JSON

Outputs:
  output/fc_gap_diagnosis_report.txt
  output/fc_gap_diagnosis_results.json
  output/fc_gap_heatmap.png
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from algos.config import OUTPUT_DIR
from algos.connectome import ConnectomeData
from algos.neural import CTRNNParams, NeuralState, neural_step
from algos.validation.reference_data import ReferenceDataset


N_WORMS = 10
PRE_EQ_TICKS = 2000
SENSORY_NOISE = 0.1


def matched_neuron_set(recordings, connectome) -> list[str]:
    """Strict intersection: neurons labeled in *all* recordings AND in connectome."""
    sets = [set(r.labels.values()) for r in recordings]
    inter = set.intersection(*sets)
    conn_set = set(connectome.neuron_names)
    return sorted(n for n in inter if n in conn_set)


def real_fc_per_recording(recording, neuron_names: list[str]) -> np.ndarray:
    """Return (n, n) Pearson FC on the matched neurons for one recording."""
    cols, found = [], []
    for col, name in recording.labels.items():
        if name in neuron_names:
            cols.append(col)
            found.append(name)
    # Reorder columns to match neuron_names order.
    order = [found.index(n) for n in neuron_names]
    cols = [cols[i] for i in order]
    sub = recording.traces[:, cols]
    return np.corrcoef(sub, rowvar=False)


def digital_fc_per_recording(connectome, recording, neuron_names, *, seed: int):
    """Run length-matched bare-CTRNN sim and return FC on matched neurons."""
    n = connectome.n_neurons
    state = NeuralState.initialize(n, seed=seed)
    params = CTRNNParams()
    rng = np.random.default_rng(seed)
    sens_idx = connectome.get_neuron_indices_by_category("sensory")
    zero = np.zeros(n)
    for _ in range(PRE_EQ_TICKS):
        state = neural_step(state, connectome, zero, params, rng)
    T = recording.n_timepoints
    history = np.zeros((T, n), dtype=np.float32)
    sens = np.zeros(n)
    for t in range(T):
        sens[:] = 0.0
        sens[sens_idx] = rng.standard_normal(len(sens_idx)) * SENSORY_NOISE
        state = neural_step(state, connectome, sens, params, rng)
        history[t] = state.V
    cols = [connectome.idx(nm) for nm in neuron_names]
    return np.corrcoef(history[:, cols], rowvar=False)


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"loading top-{N_WORMS} Atanas 2023 recordings…", flush=True)
    reference = ReferenceDataset.from_atanas2023(max_recordings=N_WORMS)
    connectome = ConnectomeData.load()

    matched = matched_neuron_set(reference.recordings, connectome)
    print(f"matched neurons (strict intersection): {len(matched)}")
    n_m = len(matched)

    # Category labels.
    matched_cat = [connectome.category[connectome.idx(n)] for n in matched]
    print("categories:", Counter(matched_cat))

    # --- Real FC (mean across recordings) -----------------------------------
    real_fcs = []
    for r in reference.recordings:
        real_fcs.append(real_fc_per_recording(r, matched))
    fc_real = np.nanmean(np.stack(real_fcs), axis=0)

    # --- Digital FC (mean across length-matched sims) -----------------------
    print("running 10 digital sims…", flush=True)
    digital_fcs = []
    for r in reference.recordings:
        seed = hash(r.recording_id) & 0x7FFFFFFF
        digital_fcs.append(
            digital_fc_per_recording(connectome, r, matched, seed=seed)
        )
    fc_digital = np.nanmean(np.stack(digital_fcs), axis=0)

    fc_diff = fc_real - fc_digital
    abs_diff = np.abs(fc_diff)
    elapsed = time.time() - t0

    # --- Top-20 pairs by |diff| ---------------------------------------------
    iu = np.triu_indices(n_m, k=1)
    pair_diffs = []
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        pair_diffs.append({
            "neuron_a": matched[i],
            "neuron_b": matched[j],
            "category_a": matched_cat[i],
            "category_b": matched_cat[j],
            "fc_real": float(fc_real[i, j]),
            "fc_digital": float(fc_digital[i, j]),
            "diff": float(fc_diff[i, j]),
            "abs_diff": float(abs_diff[i, j]),
        })
    pair_diffs.sort(key=lambda x: -x["abs_diff"])

    # --- Category-combo aggregates ------------------------------------------
    cat_pairs: dict[tuple[str, str], list[float]] = {}
    for k in range(len(iu[0])):
        i, j = iu[0][k], iu[1][k]
        a, b = matched_cat[i], matched_cat[j]
        key = tuple(sorted((a, b)))
        cat_pairs.setdefault(key, []).append(float(abs_diff[i, j]))

    cat_stats = []
    for key, vals in cat_pairs.items():
        v = np.array(vals)
        cat_stats.append({
            "category_a": key[0],
            "category_b": key[1],
            "n_pairs": len(vals),
            "mean_abs_diff": float(v.mean()),
            "median_abs_diff": float(np.median(v)),
            "max_abs_diff": float(v.max()),
            "min_abs_diff": float(v.min()),
        })
    cat_stats.sort(key=lambda x: -x["mean_abs_diff"])

    # --- Hub analysis: how often does each neuron appear in top-K diffs? ----
    TOP_K = 50
    hub_counter: Counter = Counter()
    for p in pair_diffs[:TOP_K]:
        hub_counter[p["neuron_a"]] += 1
        hub_counter[p["neuron_b"]] += 1
    top_hubs = []
    for n in matched:
        count = hub_counter.get(n, 0)
        # Total possible occurrences in top-K is min(TOP_K, n_m-1) given each
        # pair only counts once. Use raw count for clarity.
        top_hubs.append({
            "neuron": n,
            "category": matched_cat[matched.index(n)],
            "count_in_top50": count,
        })
    top_hubs.sort(key=lambda x: -x["count_in_top50"])

    # --- Heatmaps -----------------------------------------------------------
    # Sort by category so structural patterns are visible.
    cat_order = ["sensory", "interneuron", "motor", "pharyngeal", "other_neuron",
                 "sex_specific"]
    order = sorted(
        range(n_m),
        key=lambda i: (cat_order.index(matched_cat[i])
                       if matched_cat[i] in cat_order else 999,
                       matched[i]),
    )
    fc_real_s = fc_real[np.ix_(order, order)]
    fc_dig_s = fc_digital[np.ix_(order, order)]
    fc_diff_s = fc_diff[np.ix_(order, order)]
    matched_s = [matched[i] for i in order]
    cat_s = [matched_cat[i] for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, M, title, vlim in [
        (axes[0], fc_real_s, "FC_real (mean over 10 worms)", 1.0),
        (axes[1], fc_dig_s, "FC_digital (mean over 10 sims)", 1.0),
        (axes[2], fc_diff_s, "FC_real - FC_digital", 1.0),
    ]:
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                       interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(range(n_m))
        ax.set_yticks(range(n_m))
        ax.set_xticklabels(matched_s, rotation=90, fontsize=6)
        ax.set_yticklabels(matched_s, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.045, pad=0.04)

    # Add category bands
    cat_colors = {
        "sensory": "#1f77b4",
        "interneuron": "#2ca02c",
        "motor": "#d62728",
        "pharyngeal": "#888888",
        "other_neuron": "#bcbd22",
        "sex_specific": "#9467bd",
    }
    handles = [mpatches.Patch(color=cat_colors.get(c, "#cccccc"), label=c)
               for c in set(cat_s)]
    axes[0].legend(handles=handles, loc="upper left", fontsize=6,
                   framealpha=0.95)
    plt.tight_layout()
    heatmap_path = OUTPUT_DIR / "fc_gap_heatmap.png"
    fig.savefig(heatmap_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    # --- JSON ---------------------------------------------------------------
    json_path = OUTPUT_DIR / "fc_gap_diagnosis_results.json"
    json_path.write_text(json.dumps({
        "settings": {
            "n_worms": N_WORMS,
            "n_matched_neurons": n_m,
            "pre_eq_ticks": PRE_EQ_TICKS,
            "sensory_noise": SENSORY_NOISE,
        },
        "matched_neurons": matched,
        "matched_categories": matched_cat,
        "fc_real": fc_real.tolist(),
        "fc_digital": fc_digital.tolist(),
        "fc_diff": fc_diff.tolist(),
        "top_pairs_by_abs_diff": pair_diffs[:50],
        "category_combo_stats": cat_stats,
        "top_hubs": top_hubs,
        "runtime_sec": elapsed,
    }, indent=2))

    # --- Text report --------------------------------------------------------
    report_path = OUTPUT_DIR / "fc_gap_diagnosis_report.txt"
    with report_path.open("w") as f:
        f.write("Phase 0.8 — FC gap diagnosis\n")
        f.write("=" * 78 + "\n")
        f.write(f"recordings:        {N_WORMS}\n")
        f.write(f"matched neurons:   {n_m}\n")
        f.write(f"matched pairs:     {len(iu[0])}\n")
        f.write(f"runtime:           {elapsed:.1f}s\n\n")

        f.write("category breakdown of matched neurons:\n")
        for c, k in Counter(matched_cat).most_common():
            f.write(f"  {c:14s}: {k}\n")
        f.write("\n")

        f.write(f"Mean |diff|:  {abs_diff[iu].mean():.4f}\n")
        f.write(f"Median |diff|: {np.median(abs_diff[iu]):.4f}\n")
        f.write(f"Max |diff|:   {abs_diff[iu].max():.4f}\n\n")

        f.write("Top-20 |diff| pairs:\n")
        f.write(f"  {'pair':28s}  {'cat_a':12s} {'cat_b':12s}  {'FC_real':>8s}  {'FC_dig':>8s}  {'diff':>8s}\n")
        for p in pair_diffs[:20]:
            f.write(
                f"  {p['neuron_a']+'—'+p['neuron_b']:28s}  "
                f"{p['category_a']:12s} {p['category_b']:12s}  "
                f"{p['fc_real']:+8.4f}  {p['fc_digital']:+8.4f}  "
                f"{p['diff']:+8.4f}\n"
            )
        f.write("\n")

        f.write("Category-combo aggregates (mean |diff|, sorted):\n")
        f.write(f"  {'combo':30s}  n_pairs   mean    median   max\n")
        for s in cat_stats:
            label = f"{s['category_a']:14s}× {s['category_b']:14s}"
            f.write(
                f"  {label:30s}  {s['n_pairs']:>5}  "
                f"{s['mean_abs_diff']:.4f}  "
                f"{s['median_abs_diff']:.4f}  {s['max_abs_diff']:.4f}\n"
            )
        f.write("\n")

        f.write(f"Top-10 hub neurons (appearances in top-{TOP_K} |diff| pairs):\n")
        f.write(f"  {'neuron':8s}  {'category':14s}  count\n")
        for h in top_hubs[:10]:
            f.write(
                f"  {h['neuron']:8s}  {h['category']:14s}  "
                f"{h['count_in_top50']}\n"
            )

    print(f"wrote {report_path}")
    print(f"wrote {json_path}")
    print(f"wrote {heatmap_path}")
    with report_path.open() as f:
        print()
        print(f.read())
    return 0


if __name__ == "__main__":
    sys.exit(main())
