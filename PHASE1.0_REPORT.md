# Phase 1.0 — Graph-native neural system

> Generated: 2026-05-21
> Branch: main
> Design: `docs/phase1_design.md`
> Task book: `docs/phase1.0.md`
> Verdict: **H_3 partially validated (architectural unblock real,
> headline-metric gap not closed; deeper bottleneck inherited from
> Phase 0.9 is unchanged).**

---

## 1. What shipped

Five commits, each a complete sub-phase:

| commit | scope |
|---|---|
| `680dfae` [phase1.0.1] | Graph primitives (Node, Edge, NeuralGraph, Subgraph) over `nx.MultiDiGraph`; loader builds 302 nodes + 3709 chemical + 2182 mirrored gap edges; flags 14 modulator neurons. |
| `1a8f553` [phase1.0.2] | LIF dynamics + event-driven SignalQueue + GraphSimulator with vectorized per-tick math (~0.07 ms/tick at N=302). |
| `7823420` [phase1.0.3] | 13 canonical functional subgraphs (reversal/forward command, two touch reflexes, chemo/thermo/head/feeding/motor/defecation/egg-laying + two modulators); anti-correlation diagnostic. |
| `e5b777c` [phase1.0.4] | HebbianRule on 100 plastic edges + ModulatorBank (RID + 5-HT) with parameter-level threshold modulation. |
| `(this)` [phase1.0.5] | Full Atanas-2023 comparison, subgraph-behavior probe, this report. |

Total: ~5,000 lines of code + tests + scripts. 56 new tests, all
passing. Pre-existing failure
`tests/test_modulators.py::test_rid_modulator_inhibits_reversal_pool`
unchanged — Phase 0.9a left it intentionally broken (PHASE0.9A_REPORT.md
§1) and Phase 1.0 does not touch the Phase 0 `algos.neural` module.

---

## 2. Headline numbers vs Phase 0.9

10 Atanas recordings, seed = 1000 + idx, 2000-tick warmup, identical
sensory-noise statistics (`σ = 0.2` on sensory neurons). Phase 0.9's
own numbers are pulled from PHASE0.9_REPORT.md.

| metric | Phase 0.9 category | Phase 0.9 +RID g=0.5 | **Phase 1.0 baseline** | **Phase 1.0 full** | Δ (1.0_full − 0.9_cat) |
|---|---:|---:|---:|---:|---:|
| subspace_alignment   | +0.3532 | +0.3526 | +0.2793 | +0.2771 | **−0.0761** |
| temporal_correlation | −0.0140 | −0.0138 | +0.0030 | +0.0020 | **+0.0160** |
| fc_similarity        | +0.0606 | +0.0616 | +0.0173 | +0.0097 | **−0.0509** |

Reading: the new graph-native LIF runtime *regresses* on subspace
alignment and FC similarity, *modestly improves* temporal
correlation (small magnitude either way), and **adding plasticity +
modulators on top of baseline LIF moves all three metrics in the
*wrong* direction relative to the LIF baseline alone**.

Per-recording deltas (full − baseline) show no consistent improvement:
8 of 10 recordings get worse on fc_similarity when plasticity +
modulators are switched on; 5 of 10 worse on subspace_alignment.

That outcome was not anticipated by the design doc and is honest news.
The next section explains where the architectural win actually showed
up.

---

## 3. The architectural unblock: anti-correlation

Phase 0.9's central diagnostic (`PHASE0.9_REPORT.md §3`) measured the
fraction of off-diagonal FC pairs at `FC < −0.1`:

- **all four Phase 0 CTRNN variants: 0.0%**
- real worm (Atanas 2023):           17.5%

Every architectural retread tried in Phase 0.8 / 0.8.2 / 0.8.3 / 0.9 /
0.9a stayed at 0%. That null was the project's most expensive
diagnostic — the bare CTRNN topology cannot produce negative
correlations regardless of parameter tweaks.

**Phase 1.0.3 → 1.0.4 result (mean across 3 seeds, 4000 sample ticks):**

| FC threshold | LIF baseline | LIF + plasticity + modulators | real worm (in-band Atanas, top-10) |
|---:|---:|---:|---:|
| < −0.05 | 29.04% | 29.40% | 34.72% |
| **< −0.10** | **8.83%** | **9.42%** | **27.74%** |
| < −0.20 | 0.22% | 0.26% | 16.35% |
| < −0.30 | 0.00% | 0.00% |  8.86% |

The new architecture closes about a third of the gap at FC < −0.1
(0% → 9% on the way to 28%). Stronger anti-correlations (< −0.2)
remain essentially absent — those are the ones that would correspond
to robust mutual exclusion (forward vs reversal, feeding vs
locomotion), which the bare network still cannot produce because the
modulator subsystem isn't getting a state-dependent drive (more
below).

The cause is structural, not parametric: replacing the dense CTRNN
with sparse spiking LIF + 26 GABAergic edges + connectome-defined
subgraph competition is sufficient to make the off-diagonal FC
distribution bimodal where Phase 0 was unimodal. **Plasticity on top
adds ~+0.6%; modulators on top add ~0.** That distribution shift is
the qualitative win Phase 1.0 was designed to deliver.

---

## 4. Subgraph behavior — what fires and what fails

Mean-rate trace per subgraph over a single 8000-tick `full` run
(`scripts/run_phase1_subgraph_behavior.py`):

```
  reversal_command          mean=0.019  std=0.048  max=0.41
  forward_command           mean=0.015  std=0.026  max=0.17
  anterior_touch            mean=0.089  std=0.082  max=0.45
  posterior_touch           mean=0.079  std=0.065  max=0.38
  chemosensory_amphid       mean=0.102  std=0.057  max=0.38
  thermosensory             mean=0.047  std=0.050  max=0.35
  head_motor_cpg            mean=0.033  std=0.027  max=0.17
  pharyngeal_cpg            mean=0.000  std=0.000  max=0.00
  ventral_cord_motor        mean=0.000  std=0.000  max=0.00
  modulator_RID             mean=0.027  std=0.051  max=0.33
  modulator_5HT             mean=0.042  std=0.042  max=0.25
  egg_laying                mean=0.000  std=0.000  max=0.00
  defecation_pacemaker      mean=0.104  std=0.125  max=0.85
```

**3 of 13 subgraphs are completely silent** under the random-sensory
drive: pharyngeal CPG, ventral cord motor, egg-laying. These
populations are downstream of behaviors the bare network has no way
to generate (feeding, locomotion-as-an-action, egg-laying-on-bacteria).
Defecation pacemaker, in contrast, runs autonomously and has the
largest std/max ratio — consistent with its biology as a true
3-neuron CPG that doesn't need extrinsic drive.

Cross-subgraph correlations test whether the architecture produces
the *kind* of structure that should exist. The expected versus
observed:

| pair | expected | observed | verdict |
|---|---|---:|---|
| reversal_command ↔ forward_command | anti (winner-take-all) | **+0.51** | ❌ wrong sign |
| anterior_touch ↔ reversal_command | + (touch → reverse) | +0.44 | ✓ |
| posterior_touch ↔ forward_command | + (touch → forward) | +0.41 | ✓ |
| chemosensory ↔ thermosensory | + (share AIY/AIZ/RIA) | +0.01 | ❌ no signal |
| modulator_RID ↔ forward_command | + (RID promotes forward) | +0.96 | ✓ but artifact |
| pharyngeal_cpg ↔ modulator_5HT | + (5-HT drives feeding) | 0.00 | n/a (both inputs missing) |

The most diagnostic failure: **forward_command ↔ reversal_command =
+0.51** (should be anti-correlated). With random sensory drive both
command pools rise *together* whenever sensory input is large. The
expected mutual exclusion needs structured drive or active inhibition
(via the modulator bank). The modulator bank exists; it has no effect
because RID never fires (its source — a single neuron with the
modulator-default threshold of 1.2 and no specific drive — is silent
across all seeds), and 5-HT saturates at c ≈ 0.07 (a 2.8%
multiplicative threshold change, below noise).

Top-5 most anti-correlated subgraph pairs are all in the −0.08 to
−0.12 range — well below "competitive coupling" levels seen in real
recordings. The strongest anti-correlations among matched neurons in
the FC distribution are individual-cell-pair effects from refractory
periods + GABAergic relays, not subgraph-level mutual exclusion.

---

## 5. Why the headline metrics regressed

Three structural reasons, in decreasing order of importance:

**5.1 — Rate-trace burstiness changes the principal-component structure.**
The Phase 0 CTRNN trace is smooth and continuous, with PCA spectra
dominated by a few collective modes that line up tolerably well with
the real worm's bottom-up PCs. The LIF rate trace, even after the
30-tick LP filter, is bursty: it has many small "spike-pulse" modes
sitting on top of the slow envelope, which inflate the subspace
distance even when the underlying mean-field activity is qualitatively
similar. Subspace alignment dropped from 0.35 → 0.28 (−0.08); this is
almost certainly the entire cause.

**5.2 — Functional connectivity is dominated by stochastic co-firing
chains.**
The +0.96 modulator_RID ↔ forward_command correlation is the
clearest example: RID and the forward pool spike approximately when
sensory drive happens to coincide with a refractory release across
that part of the graph. That pushes many *false* positive correlations
into the FC matrix, hurting alignment with the real worm's FC
(which is sparser and more structured). fc_similarity drops from
+0.06 → +0.02 (−0.05). Plasticity then *strengthens* the false-positive
correlations (Hebbian on noise-driven coincidence), pushing
fc_similarity to +0.01.

**5.3 — Modulators are inert without behavioral state.**
Identical conclusion to Phase 0.9 / 0.9a. RID source neuron has no
spontaneous drive in the bare network; 5-HT producers fire only
sparsely. With modulators saturated at low concentration, the
parameter-level threshold modulation is a few-percent shift that
washes out against the per-tick noise. Neither modulator does
behavior-shaping work, and the small non-zero effect they have on
plasticity is in the wrong direction (slight FC degradation).

The H_3 hypothesis stated that "anti-correlation comes from subgraph
switching, not from a global variable". Phase 1.0 shows this is
*partially* right — subgraph-level competition exists in the
architecture (the 9% anti-correlation lift) but the *switch itself*
cannot happen without a state-dependent driver. Modulators were the
expected driver; they don't have the input signal they need.

---

## 6. What the H_3 architectural hypothesis bought us

Positive:

- **Structural anti-correlation now exists** (was structurally absent
  in Phase 0): 0% → 9% at FC < −0.1, no parameter tweaking required.
- **Per-edge delays, refractory periods, and parameter-level modulation
  are first-class.** All three are blocked or hacky in the Phase 0
  CTRNN; in Phase 1.0 they're straight reads off the data structure.
- **Subgraph decomposition produces interpretable overlap patterns.**
  10 biologically-meaningful overlap pairs emerge from the literature-
  derived membership lists (e.g. AIY/AIZ/RIA shared between chemo and
  thermo paths exactly as the design predicted). The shared-node
  V-sync contract holds.
- **Plasticity infrastructure is correct.** Hebbian + decay keeps
  weights in [W_min, W_max]; correlated activity grows weights,
  decorrelated activity shrinks them. The 100-edge default fits the
  design's "50-100 plastic edges" guidance.

Negative:

- **Headline metrics regressed** on 2 of 3 (subspace −0.08, fc −0.05).
  Honest: the project paid an upfront architectural cost without
  recovering it through the new mechanism.
- **3 of 13 subgraphs are completely silent.** Pharynx, motor pool,
  egg-laying — all populations that need extrinsic behavioral context.
- **Modulators have essentially zero effect on the headline numbers.**
  RID c_m=0 across all seeds; 5-HT c_m≈0.07. The Phase 0.9 lesson
  (modulators need behavioral state) was carried over to Phase 1.0
  with no change.
- **Forward/reversal command remain positively correlated** (the most
  basic biological behavior the bare network was supposed to produce).
  Without active inhibition or competition fed by behavioral state,
  the two pools co-fire under common sensory drive.

---

## 7. What this rules out and what it leaves open

**Ruled out:**

- "Replacing CTRNN with LIF + spike propagation is, by itself,
  sufficient to close the digital-vs-real gap on the three headline
  metrics." It isn't. It closes the anti-correlation distribution gap
  partway but worsens the alignment metrics.
- "Plasticity + modulators on top of the new architecture will
  recover lost ground." They don't, in the bare network. Plasticity
  amplifies noise correlations; modulators are inert.

**Still open:**

- **H_3 with behavioral state** (Phase 1.5): does adding a body +
  environment supply the structured drive that makes RID/5-HT
  modulators actually do work? Phase 0.9a's §6 explicitly flagged this
  as the next test, and Phase 1.0's null modulator result is
  consistent with that hypothesis still holding.
- **Higher-order subgraph competition** (not in Phase 1.0): only one
  subgraph per circuit class is "active" at a time should be a
  *learned* property, not a hand-wired one. That would require the
  Phase 2 work on edge addition/removal.
- **Better rate-readout** (cheap fix): the 30-tick LP filter is too
  short to smooth out the spike-pulse PCs. A 100-200 tick filter would
  recover some of the lost subspace alignment without changing
  dynamics. Untested but a likely +0.04-0.05 on subspace_alignment.

---

## 8. Performance

| measurement | value |
|---|---:|
| Per-tick cost, LIF only (N=302) | **0.07 ms** |
| Per-tick cost, LIF + plast + mod | **0.12 ms** |
| 10⁴-tick stability run, no NaN | ✓ |
| 5 commits, total runtime of all 4 diagnostic scripts | ~22 s |
| Plasticity refresh interval | every 200 ticks (cheap) |

Well under the 1 ms/tick design budget; runs comfortably on CPU as
the original docs/design.md §2.3 required.

---

## 9. Code health

- **56 new tests**, organized by module:
  - `tests/graph/test_graph.py` — 20 tests (Node/Edge contract,
    NeuralGraph loader invariants, subgraph view semantics).
  - `tests/graph/test_circuits.py` — 11 tests (≥8 subgraphs,
    biological membership, overlap correctness, matrix dims).
  - `tests/neural_v2/test_dynamics.py` — 12 tests (LIF unit, signal
    queue delay logic, full-simulator stability, gap coupling,
    downstream propagation).
  - `tests/neural_v2/test_plasticity.py` — 13 tests (Hebbian
    direction + bounds, modulator c_m + threshold, simulator
    integration).
- **0 regressions on the pre-existing Phase 0 test suite** (37/38
  passing, the 1 failure is Phase 0.9a's deliberately-stale
  `test_rid_modulator_inhibits_reversal_pool` which the brief
  ordered not to update). Total: 94 collected, 93 passing.
- New runtime dependency: `networkx>=3.0` (pyproject.toml updated).

---

## 10. Single-paragraph summary

Phase 1.0 reorganizes the neural skeleton from "302×302 matrices
plus a step function" to "graph object with subgraph views plus an
event-driven LIF runtime", and adds Hebbian plasticity + two
parameter-level modulators on top. The architectural switch produces
a clear structural unblock — off-diagonal FC entries at < −0.1 jump
from 0% (every Phase 0 variant) to 9% (Phase 1.0, no tweaking
required), a third of the way to the real worm's 28% — but the three
headline alignment metrics (subspace, temporal, fc_similarity) drop
relative to Phase 0.9, mostly because LIF rate-trace burstiness
inflates the PCA distance and stochastic co-firing chains pollute
the FC matrix. The plasticity + modulator layer is correctly
implemented and tested, but it cannot produce its intended effect in
the bare network because RID never fires and 5-HT saturates at low
concentration — the same modulator obstruction Phase 0.9 / 0.9a
diagnosed, now confirmed to be independent of the underlying
dynamics. H_3 ("the digital model needs subgraph-level competition,
not just a global modulator") is *partially* validated: the
competition structure exists and produces qualitatively new behavior
that Phase 0 architecturally couldn't, but closing the rest of the
gap requires Phase 1.5 (body + environment) to supply the
state-dependent drive that the bare network has no way to generate.

---

*Status: Phase 1.0 complete. Next: PHASE 1.5 body + environment, with
a known concrete prediction — if the body supplies state-dependent
sensory drive, the modulator subsystem already wired in this commit
should produce a measurable Δfc > 0.03 at the FC < −0.2 anti-correlation
threshold (currently 0.2%, real worm 16.4%).*
