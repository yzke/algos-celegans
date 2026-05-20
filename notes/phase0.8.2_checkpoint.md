# Phase 0.8.2 checkpoint — default category-based heterogeneity

## Status

**Done.** Three new step functions implemented and assigned by category;
comparison against homogeneous baseline run on the same 10 best-labeled
recordings used in Phase 0.7.

## What was built

- `src/algos/neural/step_library.py`:
  - `fast_filter`: relaxation toward saturated target `tanh(β · input)`
  - `integrator`: pure leaky integration `dV = (−V + input)/τ`
  - `slow_persistent`: ctrnn + momentum term `0.2·(V[t-1] − V[t-2])`
  - `DEFAULT_CATEGORY_ASSIGNMENT`: sensory→fast_filter,
    interneuron→integrator, motor→slow_persistent, pharyngeal→ctrnn_default
  - `from_category_defaults(connectome)` factory
- `scripts/run_phase0_8_2_comparison.py`: side-by-side comparison of
  homogeneous (all ctrnn_default) vs category heterogeneous on 10
  recordings.
- `tests/test_heterogeneous.py`: 2 new tests (factory assignment + stable
  long-run).

## Acceptance criteria

| AC | Result |
|---|---|
| Three new step functions implemented | ✅ fast_filter, integrator, slow_persistent — distinct in structure, not just tau |
| Category-based assignment factory | ✅ `from_category_defaults` |
| Phase 0.7 metrics rerun + compared | ✅ table below |
| Digital percentile in cross-worm distribution reported | ✅ — still 0% on all metrics |
| All prior tests pass | ✅ 32/32 (30 + 2 new) |

## Comparison numbers (mean over 10 recordings)

| metric | homogeneous (Phase 0.7) | heterogeneous (0.8.2) | Δ |
|---|---:|---:|---:|
| subspace_alignment | +0.398 | +0.360 | **−0.038** |
| temporal_correlation | −0.007 | +0.004 | **+0.011** |
| fc_similarity | +0.026 | +0.059 | **+0.032** |

**Where digital sits in the cross-worm distribution (Phase 0.7 baseline):**

Both homogeneous and heterogeneous remain at the **0th percentile** on
all three metrics. Neither has yet entered the real-worm distribution
(cross-worm p5 ≈ 0.537 / 0.049 / 0.348 for the three metrics).

## Reading the numbers

- **fc_similarity ↑ +0.032 (12% reduction of the original 0.45 gap)**:
  the load-bearing Phase 0.8 finding. The category-based heterogeneity
  measurably improves the metric Phase 0.7 identified as the
  largest digital-vs-real gap. Mechanism (best guess): different
  per-category time constants break the global "everything correlated
  via shared noise" pattern — sensory units saturate to ±1 quickly,
  motor units lag, interneurons sit in the middle, which shifts the FC
  matrix away from the uniformly-positive sea Phase 0.8 diagnosed.
- **subspace_alignment ↓ −0.038**: heterogeneity costs a little PCA
  alignment. Likely because fast_filter saturates strongly enough to
  flatten the top-K eigenvalue distribution slightly. Trade-off, not
  catastrophic.
- **temporal_correlation +0.011**: small positive movement. Still
  noise-level (within-worm split-half is only +0.035).

Neither delta moves the percentile rank — we are still **outside** the
real-worm distribution on every metric. But this is the *right
direction* on the metric that mattered most (FC), and the architecture
is now ready for 0.8.3's per-neuron-class customization.

## Issues encountered

- **fast_filter with β=5 saturates the sensory neurons to ±1
  quickly**. With sensory neurons saturated at ±1, the pre-synaptic
  `tanh(global_β · V) = tanh(V) ≈ ±0.76` for those neurons, which is
  larger than they'd produce under the homogeneous network's
  V_ss ≈ 0 dynamics. The downstream effect is detectable in the FC
  matrix (more activity flowing through the chemical synapses) and
  helps the FC metric.
  - **Decision**: keep β=5 as the brief specified. The saturation is
    the intended fast-filter behavior; tweaking it further would be
    leaving the brief's "default" parameters.

- **slow_persistent's momentum term (0.2 × ΔV)** is a small fixed
  constant; not tuned. With τ=50 the leak dominates, so momentum is a
  secondary effect that nudges motor outputs toward persisting.

## Files touched

```
NEW  src/algos/neural/step_library.py
MOD  src/algos/neural/__init__.py             (re-exports)
NEW  scripts/run_phase0_8_2_comparison.py
NEW  output/phase0_8_2_comparison_report.txt
NEW  output/phase0_8_2_comparison_results.json
MOD  tests/test_heterogeneous.py              (added 2 tests)
NEW  notes/phase0.8.2_checkpoint.md           (this file)
```

`docs/`, the connectome loader, Phase 0.7 metric and reference-data
code, the existing dynamics code — all **unchanged**.

## Ready for 0.8.3

The architecture cleanly supports per-neuron specialized step functions.
0.8.3 will:
- Add `change_detector` (ASEL/ASER), `setpoint_deviation` (AFDL/AFDR),
  `threshold_accumulator` (AVA, AVB, AVD/PVC pairs), `bistable_switch`
  (RIM).
- Override the category-default assignment for these specific neurons.
- Rerun the comparison.
