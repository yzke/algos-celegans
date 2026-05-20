# Phase 0.9 — Minimal RID modulator experiment

> Generated: 2026-05-20
> Brief: `logs/phase0.9_brief.md`
> Hypothesis under test: H_1.4 (a single global modulator gating the
> reversal command pool closes a meaningful fraction of the Phase 0.8 FC
> gap)
> Verdict: **P1 NOT SUPPORTED.** Δfc_similarity = −0.0019 at the default
> gain (well below the +0.03 lower bound for partial support).

This phase asked one focused question: if we add a single global slow
scalar `c_RID` to the Phase 0.8.2 category-default network and let it
inhibit the AVA/AVD/AVE reversal command pool, does fc_similarity vs
real-worm activity improve by ≥ +0.10? The answer is no — neither at
the default gain nor at the parameter neighbors we swept.

The bulk of this report is the honest characterization of *why* the
prediction failed, what the modulator does change, and what that
implies for the next step.

---

## 1. Implementation summary

### What was built

```
NEW  src/algos/neural/modulators.py        # RIDModulator class + defaults
MOD  src/algos/neural/heterogeneous.py     # step() accepts optional modulator
MOD  src/algos/neural/__init__.py          # re-export modulator API

NEW  tests/test_modulators.py              # 6 tests, all pass

NEW  scripts/run_phase0_9_comparison.py    # 4-way comparison (cat / g=0.2 / 0.5 / 1.0)
NEW  scripts/run_phase0_9_diagnostic.py    # FC gap diagnostic with modulator

NEW  output/phase0_9_comparison_{report.txt,results.json}
NEW  output/phase0_9_diagnostic_{report.txt,results.json}
NEW  output/phase0_9_diagnostic_heatmap.png

MOD  DECISIONS.md                          # ## [Phase 0.9]
NEW  PHASE0.9_REPORT.md                    # this file
```

### Mechanism

```python
# Per-tick, after total_input is assembled:
modulator.step(V)                          # c_RID += (V[RID] - c_RID) / tau
modulator.apply_modulation(total_input)    # total_input[reversal] -= gain * c_RID
```

Default parameters from the brief:
- `tau_RID = 200` ticks (~10× neural τ)
- `mod_gain = 0.5`
- reversal pool = `{AVAL, AVAR, AVDL, AVDR, AVEL, AVER}`

### Backward compatibility (AC0.9.1)

The test `test_no_modulator_equals_phase08_2` asserts the new code path
with `modulator=None` produces V values **bit-identical** (max |ΔV| =
0.0 — not just < 1e-12) to a separate run on the same RNG seed. The
modulator is an opt-in branch; nothing else moves when it's absent.

All 38 tests pass (32 prior + 6 new). RID and the full reversal pool
were all present in the Cook 2019 connectome (RID at index 175, all
others between 162-179, all `interneuron` category).

---

## 2. Headline result (AC0.9.2)

Means across 10 Atanas 2023 recordings, stable seeds `1000 + idx`:

| metric | category (baseline) | g=0.2 | **g=0.5 (default)** | g=1.0 | Δ(g=0.5 − cat) |
|---|---:|---:|---:|---:|---:|
| subspace_alignment | +0.3532 | +0.3548 | +0.3576 | +0.3611 | **+0.0044** |
| temporal_correlation | −0.0140 | −0.0136 | −0.0128 | −0.0110 | **+0.0012** |
| **fc_similarity** | **+0.0606** | **+0.0597** | **+0.0587** | **+0.0584** | **−0.0019** |

All three RID configurations remain at the **0th percentile** of the
Phase 0.7 cross-worm distribution on every metric.

### P1 verdict

> **P1**: `Δfc_similarity ≥ +0.10` ⇒ **NOT MET** (Δ = −0.0019 at default
> gain; −0.0009 at g=0.2; −0.0022 at g=1.0).

By the brief's own bands:
- +0.10 or better → strong support → **not reached.**
- +0.03 to +0.10 → partial support → **not reached.**
- < +0.03 → H_1.4 needs re-examination → **this is where we are.**

### Gain sensitivity

The four configurations differ smoothly and monotonically. The headline
fc_similarity slips slightly with stronger gain (−0.0019 → −0.0022),
while subspace_alignment rises (+0.0044 → +0.0079). This is **not** a
case of "0.5 is wrong, try harder." The modulation is doing *something*
to the dynamics, but its effect on FC structure is essentially
orthogonal to the gap we need to close.

---

## 3. FC-gap diagnostic (AC0.9.3)

Strict-intersection set: 29 neurons labeled in all 10 recordings and in
the connectome (11 sensory / 9 interneuron / 5 pharyngeal / 4 motor).
406 unordered pairs.

### Sign-reversal proportion

Pairs where `|FC_real| > 0.05` and `|FC_digital| > 0.05` but with
opposite signs:

| configuration | sign-reversal fraction |
|---|---:|
| category (Phase 0.8.2 baseline) | 0.300 |
| rid (Phase 0.9, g=0.5) | 0.313 |
| Δ | **+0.013** (slightly *worse*) |

The Phase 0.8 diagnostic reported a 0.369 sign-reversal proportion on
the bare Phase 0.7 homogeneous CTRNN. Phase 0.8.2's category
heterogeneity already brought this down to 0.300. The RID modulator
does not reduce it further; it nudges it up by 0.013.

### Anti-correlation production

Fraction of off-diagonal pairs with FC < −0.1:

| | fraction with FC < −0.1 |
|---|---:|
| real | **0.175** |
| category | 0.000 |
| rid (g=0.5) | 0.000 |

The brief asked, "数字模型现在能产生反相关吗?" — *can the digital
model now produce anti-correlations?* The answer is **no, still zero.**
Among 406 matched-neuron pairs, the digital model with or without the
RID modulator produces **zero** pairs at FC < −0.1. Real worms have
roughly 1 in 6 pairs in that range.

This is the single most important diagnostic in this Phase. The RID
modulator was supposed to introduce the kind of behavioral-state
mutual exclusion that produces strong anti-correlations. It does not.

### Top-50 pair improvement

Pairs ranked by |FC_real − FC_category| (i.e. the original FC gap):
how does adding the modulator change |FC_real − FC_rid| for those same
pairs?

| | value |
|---|---:|
| mean(|diff_cat| − |diff_rid|) over top-50 | +0.0001 |
| n improved (>0) | 24 / 50 |
| n worsened (<0) | 26 / 50 |

Effectively a coin-flip. The modulation perturbs FC values within ±0.04
on individual pairs but the perturbation is uncorrelated with the gap
direction.

### RID-hub specific

Pairs touching any RID-related neuron (RID, AVA{L,R}, AVD{L,R},
AVE{L,R}, AVB{L,R}, PVC{L,R}) — 106 of the 406 pairs:

| | value |
|---|---:|
| mean improvement (g=0.5 vs cat) | −0.0012 |
| of these in top-50 | 26 |
| mean improvement in top-50 ∩ RID-related | +0.0012 |

The brief expected ~11 RID-hub problem pairs from Phase 0.8 diagnostic.
We see 26 RID-related pairs in the top-50 |diff|, but they show no
meaningful improvement either as a group or zoomed in.

### Where the modulator does and doesn't help

Inspecting the top-20 by |diff_cat|, the modulator helps measurably on
a few pairs where the FC_real is negative and FC_cat is positive:

| pair | FC_real | FC_cat | FC_rid | improvement |
|---|---:|---:|---:|---:|
| AVER—RID | −0.609 | +0.143 | +0.109 | +0.034 |
| AVER—RMER | −0.539 | +0.075 | +0.054 | +0.021 |
| AVER—RMEL | −0.514 | +0.099 | +0.058 | +0.040 |

It hurts on a comparable number where FC_real is strongly positive:

| pair | FC_real | FC_cat | FC_rid | "improvement" |
|---|---:|---:|---:|---:|
| AIBR—AVER | +0.846 | +0.057 | +0.022 | −0.035 |
| AIBL—AVER | +0.827 | +0.088 | +0.072 | −0.016 |
| AIBL—RID | −0.519 | +0.106 | +0.123 | −0.017 |

The pattern is intelligible. When c_RID is positive (and the gain is
applied as −gain·c_RID at the reversal pool), AVA/AVE/AVD get pushed
down — which reduces their correlation with neurons positively coupled
to them (like AIB) but also reduces their correlation with neurons
negatively coupled to them. The modulator is a single-axis perturbation
that does not distinguish between "lower this because the real worm
shows anticorrelation here" and "leave this alone because the real worm
shows strong positive correlation here."

### The c_RID trajectory

```
recording 0: c_RID(final) = −0.561
recording 1:                 −0.561
recording 2:                 −0.561
recording 3:                 −0.561
recording 4:                 −0.562
recording 5:                 +0.561
recording 6:                 −0.561
recording 7:                 +0.561
recording 8:                 −0.562
recording 9:                 −0.562
```

c_RID lands at one of two attractors (≈ ±0.56) depending on the random
initial condition. RID itself is a single integrator-class interneuron
with no special structure in the bare network; over 1500-2000 ticks it
saturates to whatever sign the random initial drive pushes it toward.
The *modulator* then locks that initial-condition coin-flip in for the
whole recording.

That is the opposite of what RID does biologically. Real RID activity
varies with behavioral state (which itself depends on body, environment,
and feedback the bare network does not have). In our setup, the
modulator is a *passive amplifier of seed noise*, not a state-gating
mechanism.

---

## 4. The seven brief-§8 questions answered

1. **P1 (FC ≥ +0.10)?** No. Δfc_similarity = −0.0019.
2. **If not, what is the actual improvement and where does the gap
   live?** Improvement is null (−0.0019). The bulk of the gap (still
   +0.42) lives where Phase 0.8 diagnostic said it does: the digital
   model cannot produce anti-correlations (0% of pairs at FC < −0.1 vs
   17.5% in real), and the sign-reversal proportion is essentially
   unchanged (0.300 → 0.313).
3. **Did the RID-hub problem pairs improve?** No, mean improvement on
   the 106 RID-related pairs is −0.0012; on the 26 in the top-50, it
   is +0.0012. Both are noise around zero.
4. **Did the sign-reversal proportion (Phase 0.8: 0.369) drop?** No.
   The proportion is 0.300 → 0.313 (slightly worse). Note: the 0.369
   was on the homogeneous CTRNN; 0.300 is on Phase 0.8.2's category
   defaults, which had already absorbed most of that reduction.
5. **Does the digital model now produce anti-correlations?** No. Zero
   pairs at FC < −0.1, both with and without modulator.
6. **Support strength for H_1.4 (modulator necessity)?** **Refuted in
   its minimal form.** A single global slow scalar gating one command
   pool, on top of the bare category-default network, does not improve
   FC. We cannot conclude "modulators are unnecessary" from this — see
   §5 for what we *can* conclude.
7. **Phase 0.10 recommendation?** Do not invest more in modulators
   before Phase 1. See §6.

---

## 5. What this result does and does not tell us

### What it tells us

- **The bare network cannot produce anti-correlations at all.** Zero
  pairs at FC < −0.1 is a strong structural failure that is not about
  any particular modulator — it is about the fact that without
  time-structured sensory input, behavioral-state coupling, or recurrent
  reset dynamics, the network drifts to a unimodal noise distribution
  where everything moves together (or independently).
- **A single global modulator that locks to a noise-driven attractor
  cannot rescue this.** c_RID saturates at ±0.56 within a few hundred
  ticks based on initial conditions; the modulator then acts as a
  constant DC bias on the reversal pool. A constant DC bias does not
  introduce mutual exclusion.
- **The H_1.4 prediction as written was too optimistic about the
  minimum infrastructure.** The brief's logic was "modulator → mutual
  exclusion → anti-correlations → FC closes." But the chain breaks at
  the first link: in the bare network the modulator does not produce
  mutual exclusion, because there is no driving signal that pushes RID
  up *only when* the worm is in a forward state.

### What it does not tell us

- It does **not** falsify H_1 as a whole. H_1 says the connectome plus
  local transforms plus *global modulator state variables* are
  necessary. The current result is consistent with that statement plus
  an additional sub-claim: "modulators only work in the presence of the
  sensory/behavioral state that drives them." That sub-claim moves the
  next experiment back to Phase 1, not deeper into Phase 0.
- It does not rule out the possibility that **multiple** modulators
  together could behave differently (e.g. 5-HT + RID with opposing
  effects). But the brief was explicit: a single modulator should
  produce ≥ +0.10 if H_1.4 was right. It did not. Multi-modulator
  experiments would now need a revised hypothesis.

---

## 6. Recommendation for the next phase

**Do not skip to Phase 0.10 (multi-modulator).** The Phase 0.9
diagnostic indicates that the bottleneck is not the *number* of
modulators but the *absence of the input time-structure that drives
modulators biologically*. Specifically:

1. **Phase 1 (body + environment) is now load-bearing.** Without
   time-structured sensory input that itself depends on the worm's
   physical state, c_RID cannot do what it biologically does. The
   minimal Phase 1 should produce sensory streams whose correlations
   change with behavioral state.
2. **Re-run Phase 0.9's experiment on a Phase 1 network** before
   doing anything more elaborate with modulators. If the same single-
   modulator setup closes meaningful FC gap once it has structured
   input, H_1.4 is rescued. If not, the hypothesis needs another
   revision.
3. **Keep the RIDModulator code in tree.** It is small (~110 lines),
   well-tested, and the integration point in `HeterogeneousNetwork.step`
   is non-invasive. Phase 1 work can drop it in unchanged.
4. **The "zero anti-correlations" finding is the most actionable
   diagnostic of Phase 0.9.** Whatever Phase 1 produces, the first
   smoke test should be: "does the digital model now have some FC pairs
   at < −0.1?" If yes, we are on a different planet from Phase 0; if
   no, the gap remains the same gap.

---

## 7. Engineering notes

- Total runtime: 16.0 s for the 4-configuration comparison (10
  recordings × 4 configs); 6.9 s for the diagnostic. Total Phase 0.9
  compute: < 25 s.
- Single-tick performance impact of the modulator is ~10 µs (one scalar
  update + a 6-element subtract on the per-tick `total_input` array).
  Well within the < 1 ms/tick budget.
- The seed convention `1000 + recording_index` is stable across runs
  (per the Phase 0.8.3 PYTHONHASHSEED lesson). All Phase 0.9 numbers
  in this report are reproducible.
- `c_RID` saturates near ±0.56 in 9 of 10 recordings (one falls at
  +0.56). That bimodality is a side-effect of using random initial
  drives on a network that has no other source of structure for RID.

---

## 8. Single-sentence summary

A minimal RID modulator (single global slow scalar gating the reversal
command pool with default τ=200, gain=0.5) does not improve the
digital-vs-real fc_similarity on top of Phase 0.8.2 — Δ = −0.0019, well
below the +0.03 partial-support threshold, with the digital model
continuing to produce **zero strong anti-correlations** against the
real worm's 17.5% — falsifying P1 in its minimal form and confirming
that Phase 1 (body + structured sensory input) is the prerequisite for
any modulator experiment to be informative.

---

*Last updated: 2026-05-20*
*Status: Phase 0.9 complete; P1 refuted; pivoting to Phase 1.*
