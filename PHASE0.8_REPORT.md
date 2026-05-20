# Phase 0.8 — Heterogeneous neuron architecture

> Generated: 2026-05-20
> Brief: `logs/phase0.8_heterogeneous.md`
> Status: 0.8.1, 0.8.2, 0.8.3 all complete; PHASE0.8_diagnostic.md
> (the earlier "where does the FC gap live?" work) preserved as a
> separate file.

This report covers the three-sub-phase architectural change from the
homogeneous Phase 0.7 CTRNN to a per-neuron-step-function dispatch
("matrix W + heterogeneous neuron functions"). The brief framed this
as the project's largest architectural change to date because
Phase 0.6/0.7 had falsified the assumption that a uniform CTRNN with
sufficient scale would produce real-worm-like dynamics.

---

## 1. Three sub-phases — completion status

| Sub-phase | Description | Status | Key result |
|---|---|---|---|
| **0.8.1** | Refactor: matrix + per-neuron step function dispatch | ✅ done | All-`ctrnn_default` matches Phase 0.7 bit-exactly (noise=0) and < 1e-6 (noise on); 30/30 tests pass; 0.094 ms/tick worst case (100× under budget) |
| **0.8.2** | Three category-default step functions | ✅ done | fc_similarity ↑ +0.044 vs homogeneous (largest single architectural gain in the project) |
| **0.8.3** | Specialized step functions for ASE/AFD/AVA/AVB/RIM (14 neurons) | ✅ done | **Regressed** fc by −0.016 vs 0.8.2 — synthetic per-neuron sketches don't help without body+modulators |

All commits used `[phase0.8.x]` prefix; the architecture refactor
(0.8.1), category-default heterogeneity (0.8.2), and key-neuron
specialization (0.8.3) each got their own commit + checkpoint note in
`notes/`.

---

## 2. Architecture (0.8.1)

```
src/algos/neural/heterogeneous.py:
  HeterogeneousNetwork(connectome, function_assignment, function_params, ...)
  HeterogeneousState(V_history, tick)
  STEP_LIBRARY: dict[str, StepFunction]
  ctrnn_default(V_current, total_input, V_history, params) → V_new
```

Per-tick algorithm:

1. Compute `chem_input = W_chem @ tanh(β·V)` globally.
2. Compute `gap_input = W_gap @ V − V · sum(W_gap)` globally.
3. Compute `total_input = chem + gap + sensory + noise` globally.
4. For each function group (neurons sharing a step function), call the
   group's step function on the group's slice of `(V_current,
   total_input, V_history, params)`.

`V_history` is a fixed-length circular buffer (default 5 ticks).
Step functions that need only V[t-1] (most) ignore it. Step functions
that need V[t-1] − V[t-2] (e.g. `slow_persistent`'s momentum term;
0.8.3's specialized functions) read it directly.

### Numerical equivalence

When every neuron uses `ctrnn_default`, the heterogeneous network
reproduces Phase 0.7's `neural_step` **bit-exactly** at noise=0 (max
|ΔV| < 1e-12) and with the same RNG sequence to <1e-6 with noise on.
Verified by `tests/test_heterogeneous.py::test_homogeneous_equivalence`
and `test_homogeneous_equivalence_no_noise`.

### Performance

| path | ms/tick |
|---|---:|
| Phase 0.7 `neural_step` | 0.075 |
| Heterogeneous, 1 group (= all `ctrnn_default`) | 0.070 |
| Heterogeneous, 5 groups | 0.094 |
| Heterogeneous, 8 groups (with 0.8.3 specializations) | ~0.10 |

Budget was < 10 ms/tick. We're 100× under. No bottleneck.

---

## 3. Category-default heterogeneity (0.8.2)

Three new step functions assigned by category:

| function | applied to | parameters | computational structure |
|---|---|---|---|
| `fast_filter` | sensory (83 neurons) | τ=5, β=5 | V ← tanh(β·input) target |
| `integrator` | interneuron (81) | τ=20 | leaky `dV = (−V + input)/τ` |
| `slow_persistent` | motor (108) | τ=50 | leaky + `0.2·(V[t-1] − V[t-2])` momentum |
| `ctrnn_default` | pharyngeal (20), sex_specific (8), other (2) | τ=15/20/20 | Phase 0.7 default |

### Comparison vs Phase 0.7 (10 best-labeled Atanas 2023 recordings)

| metric | homogeneous | **category (0.8.2)** | Δ |
|---|---:|---:|---:|
| subspace_alignment | +0.398 | +0.360 | −0.038 |
| temporal_correlation | −0.007 | +0.004 | +0.011 |
| **fc_similarity** | **+0.026** | **+0.059** | **+0.032** |

(0.8.2 standalone run, PYTHONHASHSEED-based RNG seeds; 0.8.3's stable-
seed re-run gives a slightly different `+0.061` for the same condition.
The qualitative finding is robust.)

The architectural change closes ~7% of the original Phase 0.7 FC gap
(+0.45 → +0.42 to enter the cross-worm 5%-tile). Notable but not
enough by itself.

---

## 4. Key-neuron specialization (0.8.3) — honest negative result

Four specialized step functions implemented and assigned to 14 neurons:

| function | neurons | shape |
|---|---|---|
| `change_detector` | ASEL (polarity+1), ASER (polarity−1) | `tanh(polarity·gain·(input − V))` integrated with τ |
| `setpoint_deviation` | AFDL (+1), AFDR (−1) | `tanh(gain·polarity·(input − setpoint))` |
| `threshold_accumulator` | AVAL/AVAR/AVDL/AVDR, AVBL/AVBR/PVCL/PVCR | leaky integrator + soft latch above threshold |
| `bistable_switch` | RIML, RIMR | drive + `self_gain · tanh(3·V)` (attractors near ±1) |

### Three-way comparison (stable seeds, 10 recordings)

| metric | homogeneous | category (0.8.2) | key_neuron (0.8.3) | Δ key vs cat |
|---|---:|---:|---:|---:|
| subspace_alignment | +0.394 | +0.353 | +0.349 | −0.005 |
| temporal_correlation | −0.003 | −0.014 | −0.016 | −0.002 |
| **fc_similarity** | **+0.017** | **+0.061** | **+0.044** | **−0.016** |

**Per-neuron specialization regressed FC by −0.016 relative to category
defaults.** This is the honest, important finding. The architectural
heterogeneity (going from 1 step function to 4) does the real work.
Going from 4 to 8 with finer per-neuron sketches doesn't help.

### Why the specializations didn't help

The synthetic step functions need supporting infrastructure the bare
Phase 0.8 network does not have:

- `change_detector` needs time-correlated sensory input → Phase 1.
- `threshold_accumulator` needs the network to push AVA past the
  threshold at biologically-meaningful moments → Phase 1 + Phase 3.
- `bistable_switch` picks an attractor on initial-condition noise →
  needs Phase 3 modulators to gate the choice biologically.
- `setpoint_deviation` with setpoint=0 is essentially a gain on input.

These step functions are mathematical sketches in the right shape, but
without the surrounding **input time-structure** (Phase 1) and
**cross-neuron mutual-exclusion mechanisms** (Phase 3 modulators), they
don't pay off.

---

## 5. Where the digital worm sits now

Phase 0.7's three target metrics, mean across 10 recordings, vs the
real-worm cross-worm baseline established in Phase 0.7:

| metric | digital (best of 3 configs) | cross-worm mean | cross-worm p5 | percentile |
|---|---:|---:|---:|---:|
| subspace_alignment | +0.398 (homogeneous) | +0.593 | +0.537 | **0%** |
| temporal_correlation | +0.011 (category) | +0.122 | +0.049 | **0%** |
| fc_similarity | +0.061 (category) | +0.478 | +0.348 | **0%** |

**The digital worm is still at the 0th percentile on all three metrics**
in the real-worm cross-worm distribution. Phase 0.8's architectural
change moved fc_similarity from +0.03 to +0.06 — a real improvement,
but +0.29 short of entering the real-worm distribution.

The architecture is now correct. The remaining gap is not about how
neurons compute; it is about what they compute *over*.

---

## 6. Summary of decisions

(`DECISIONS.md` `## [Phase 0.8.1/2/3]` for the full audit trail.)

| decision | rationale | location |
|---|---|---|
| New module, not replace `neural_step` | Coexist with Phase 0.7; equivalence test guards | `src/algos/neural/heterogeneous.py` |
| String function names, not int IDs | Readability; zero performance cost | `function_assignment` is `list[str]` |
| Per-neuron params as length-N ndarrays | Vectorized group dispatch; harmless extra keys | `function_params: dict[str, np.ndarray]` |
| V_history as fixed-length circular buffer | Cheap, enables derivative/momentum reads | `history_len = 5` |
| Keep brief's β=5 for `fast_filter` | Respect the brief's stated defaults | `step_library.py` |
| Stable seeds (`1000 + index`) for 0.8.3 onwards | Found that `hash(s)` is PYTHONHASHSEED-randomized | `scripts/run_phase0_8_3_comparison.py` |

---

## 7. Honest gap assessment

### What Phase 0.8 closed

- **fc_similarity gap: ~7% of the original closed** (+0.026 → +0.059 with
  category defaults; or +0.017 → +0.061 with the stable-seed re-run).
- **Architecture: completely refactored** to support per-neuron
  computation without modifying the connectome, normalization, or
  Phase 0.7 code paths.
- **Performance budget: 100× headroom.** No future scaling concern.

### What Phase 0.8 did not close

- **0th percentile remains 0th percentile.** No metric crossed into
  the real-worm distribution.
- **The +0.45 FC gap is now +0.42.** Most of it still there.
- **Per-neuron specialization didn't help.** The expected pay-off from
  treating ASE / AVA / RIM specially didn't appear, because their
  specialized functions need infrastructure the bare network lacks.

### What the gap is now made of

Phase 0.8_diagnostic (the prior "where is the FC gap" analysis)
identified: ~37% of all neuron-pair correlations sign-flipped between
real and digital, with the bare CTRNN systematically failing to
reproduce *anti-correlations* mediated by behavioral-state mutual
exclusion. Phase 0.8's architectural change reduced this by a small
amount (~7%); the rest of the gap continues to live in mechanisms the
unit-level step functions cannot capture by themselves:

1. **Time-structured sensory input.** Random Gaussian noise has no
   temporal coherence; real worms see correlated sensory streams.
   `change_detector` and similar derivative-based functions are
   useless on white noise. → Phase 1.
2. **Behavioral commitment / body mechanics.** A worm in motion has
   physical state that gates command-neuron transitions. Without it,
   `threshold_accumulator` for AVA cycles spuriously. → Phase 1.
3. **Modulator gating.** RID (the #1 problem hub from Phase 0.8
   diagnostic) is a neuropeptide releaser. A unit-level step function
   cannot represent its modulatory action on the rest of the network.
   → Phase 3.
4. **Mutual inhibition that is dynamic, not just structural.** The
   forward/reverse switch in real worms uses cross-modulator
   competition that doesn't reduce to per-unit dynamics. → Phase 3.

---

## 8. Recommendations for Phase 1 priorities

1. **Adopt the heterogeneous architecture as the default for Phase 1+.**
   `from_category_defaults(connectome)` gives a clean +0.03 to +0.04
   FC improvement at zero engineering cost. Use it.

2. **Do not invest more in per-neuron specialization until Phase 1 + 3
   are in place.** 0.8.3 demonstrated that synthetic step functions
   without supporting infrastructure regress, not improve.

3. **Phase 1's body delivers what 0.8.3's `change_detector` and
   `threshold_accumulator` were missing**: time-structured sensory
   input + behavioral state commitment. Once that lands, **re-run the
   0.8.3 specializations on the Phase 1 body** — they may finally start
   to help, and we'll have a clean comparison.

4. **Phase 3 (modulators) is more important than the Phase 0 schedule
   implied.** The Phase 0.8 diagnostic flagged RID (a neuropeptide
   releaser) as the #1 problem hub. The Phase 0.8.3 attempt at
   `bistable_switch` for RIM hit the limit of what unit-level
   dynamics can do. Modulators are not optional.

5. **Use stable seeds.** Every script from Phase 1 onward should set
   seeds explicitly (not via `hash()`). Phase 0.5/0.7/0.8.2 numbers
   are correct realizations but not bit-reproducible across runs.

6. **Use `subspace_alignment` alone as the headline PCA metric.**
   Phase 0.6 already established this. Phase 0.8 confirmed it remains
   sensitive: category heterogeneity moved it from 0.398 → 0.353 (a
   trade-off vs FC), telling us something real about how the
   architecture trades off correlation structure across the two metrics.

---

## 9. Single-sentence summary

Phase 0.8 refactored the project from "homogeneous CTRNN" to "matrix
+ per-neuron step functions" without breaking anything (30→32 tests
pass, 0.094 ms/tick, < 1e-6 equivalence at the homogeneous default),
recovered ~7% of the Phase 0.7 fc_similarity gap with simple
category-based defaults, and showed honestly that fine per-neuron
specializations actually regress without supporting body+modulator
infrastructure — confirming Phase 1 (body) and Phase 3 (modulators) as
the necessary next investments, in that order, with category
heterogeneity adopted as the default from Phase 1 onwards.

---

## 10. Artifacts

```
NEW  src/algos/neural/heterogeneous.py             # 0.8.1 architecture
NEW  src/algos/neural/step_library.py              # 0.8.2 + 0.8.3 step functions
MOD  src/algos/neural/__init__.py                  # re-exports only

NEW  tests/test_heterogeneous.py                   # 7 new tests, all pass

NEW  scripts/run_phase0_8_2_comparison.py
NEW  scripts/run_phase0_8_3_comparison.py
NEW  output/phase0_8_2_comparison_{report.txt,results.json}
NEW  output/phase0_8_3_comparison_{report.txt,results.json}

NEW  notes/phase0.8.1_checkpoint.md
NEW  notes/phase0.8.2_checkpoint.md
NEW  notes/phase0.8.3_checkpoint.md

MOD  DECISIONS.md                                  # ## [Phase 0.8.1/2/3]
NEW  PHASE0.8_REPORT.md                            # this file
RNM  PHASE0.8_REPORT.md → PHASE0.8_diagnostic.md   # the earlier diagnostic
```

Tests: **32 pass** (Phase 0 originals + Phase 0.5 specificity +
Phase 0.8.1/2 heterogeneous).

Total compute for Phase 0.8 comparison runs: ~25 seconds across both
scripts (10 recordings × 3 networks each).

---

*Last updated: 2026-05-20*
*Status: Phase 0.8 complete; architecture in place; ready for Phase 1*
*Recommendation: Phase 1 (body) → Phase 3 (modulators) → revisit 0.8.3 specializations.*
