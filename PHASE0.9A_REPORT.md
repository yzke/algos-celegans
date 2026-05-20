# Phase 0.9a — Reversed RID modulator experiment

> Generated: 2026-05-20
> Brief: `logs/phase0.9a_brief.md`
> Hypothesis under test: Explanation A — the Phase 0.9 RID direction was
> reversed; RID biologically *activates* the forward command pool
> (AVB/PVC) rather than *inhibiting* the reversal pool (AVA/AVD/AVE).
> Verdict: **Explanation A REJECTED.** Δfc_similarity = +0.0010 at the
> default gain — well below the +0.05 threshold for "direction was
> misunderstood," and indistinguishable from zero given the run-to-run
> magnitudes we see.

---

## 1. Change

One file modified, per the brief's "do not touch other code" constraint:

```
MOD  src/algos/neural/modulators.py
       - REVERSAL_COMMAND_NEURONS now contains (AVBL, AVBR, PVCL, PVCR)
       - apply_modulation: extra_input[pool] -= gain * c_RID
                       →   extra_input[pool] += gain * c_RID
       - docstrings updated; field name `reversal_indices` kept for
         backward-compat (it now indexes the forward pool)
       - τ = 200, gain = 0.5 unchanged
```

The Phase 0.9 test suite (`tests/test_modulators.py`) was deliberately
not updated — its `test_rid_modulator_inhibits_reversal_pool` is written
against the Phase 0.9 direction and will fail under the swap; per the
brief, no other code was changed.

---

## 2. Headline result

10 Atanas 2023 recordings, stable seeds `1000 + idx`, identical setup
to Phase 0.9 (PRE_EQ_TICKS=2000, sensory noise=0.1, τ_RID=200).

| metric | category (baseline) | g=0.2 | **g=0.5 (default)** | g=1.0 | Δ(g=0.5 − cat) |
|---|---:|---:|---:|---:|---:|
| subspace_alignment   | +0.3532 | +0.3530 | +0.3526 | +0.3522 | **−0.0006** |
| temporal_correlation | −0.0140 | −0.0140 | −0.0138 | −0.0136 | **+0.0002** |
| **fc_similarity**    | **+0.0606** | **+0.0609** | **+0.0616** | **+0.0630** | **+0.0010** |

All four configurations sit at the 0th percentile of the Phase 0.7
cross-worm distribution on every metric.

### Verdict against the brief's criteria

- FC improvement > **+0.05** → "explanation A correct, RID direction was misunderstood"
- FC improvement ≈ **0** → "exclude explanation A"

We measure Δfc = **+0.0010** — a factor of 50 below the support
threshold, and within the noise floor of the metric. **Explanation A
is excluded.**

---

## 3. Side-by-side with Phase 0.9

Same headline metric, identical seeds, only the modulator direction
differs:

| configuration | Δsubspace | Δtemporal | **Δfc** |
|---|---:|---:|---:|
| Phase 0.9   (−= AVA/AVD/AVE) g=0.5 | +0.0044 | +0.0012 | **−0.0019** |
| Phase 0.9a  (+= AVB/PVC)      g=0.5 | −0.0006 | +0.0002 | **+0.0010** |
| Net flip                            | −0.0050 | −0.0010 | **+0.0029** |

Flipping the direction does move fc_similarity in the right direction
(from −0.0019 to +0.0010, a +0.0029 swing) but the magnitude is
trivial. Subspace alignment moves the opposite way and loses Phase 0.9's
modest +0.0044 lift. Temporal correlation barely budges in either case.

In short: the direction matters slightly for sign, but not for scale.
Neither direction approaches the threshold the brief set.

---

## 4. Why both directions fail

Phase 0.9's diagnostic already established the underlying obstruction
(`PHASE0.9_REPORT.md §3`):

1. **c_RID saturates to one of two attractors (≈ ±0.56) within a few
   hundred ticks, driven by random initial conditions.** Phase 0.9a's
   per-recording finals (`output/phase0_9_comparison_report.txt`) show
   the same picture — 8 of 10 recordings at c_RID ≈ −0.636, 2 at
   ≈ +0.636. The seed-driven coin flip is direction-agnostic.
2. **Once c_RID is pinned, the modulator becomes a constant DC bias on a
   command pool.** In Phase 0.9 that bias suppresses AVA/AVD/AVE; in
   Phase 0.9a it raises AVB/PVC. Either way it cannot produce
   anti-correlations, which is the actual FC gap (Phase 0.9: digital
   model produces 0% of pairs at FC < −0.1 vs 17.5% in real recordings).
3. **Mutual exclusion between forward and reversal pools requires
   structured drive on RID itself.** The bare network has no such
   drive — RID is not coupled to behavioral state because there is no
   behavioral state. Flipping which pool is touched does not address
   this.

The +0.0010 fc lift in Phase 0.9a likely reflects a small structural
artifact: AVB/PVC happen to be more strongly coupled to high-activity
hub neurons (AIB, AVA, RIM), so adding a DC bias to them shifts a few
correlations in the same direction the real worm shows. It is not
behaviorally meaningful.

---

## 5. What this rules out

- **Direction-of-effect is not the bug.** The Phase 0.9 inhibition of
  the reversal pool was not "backwards" in a way that simple sign-flip
  recovers. Both directions land essentially at zero improvement.
- **Choice of target pool is not the bug.** Whether the modulator
  targets reversal or forward command, the bottleneck — c_RID becoming
  a noise-driven constant — does not change.

## 6. What this still leaves open

- **The brief's binary excludes explanation A, not the entire H_1.4
  family.** A modulator coupled to behavioral state (not just to its
  own membrane potential) could still produce the predicted effect.
  That requires Phase 1 (body + environment) to supply the
  state-dependent drive.
- The Phase 0.9 recommendation stands: do not stack more single-modulator
  variants on top of the bare network before Phase 1.

---

## 7. Single-sentence summary

Reversing the RID modulator's direction (activate AVB/PVC instead of
inhibit AVA/AVD/AVE, += instead of −=) changes Δfc_similarity from
−0.0019 to +0.0010 — a +0.0029 swing that confirms the sign matters
slightly but rules out "RID direction was misunderstood" as the
explanation for Phase 0.9's failure, because the magnitude in both
directions is dominated by the same underlying obstruction: c_RID
saturates to a seed-determined DC value in the absence of structured
behavioral input.

---

*Last updated: 2026-05-20*
*Status: Phase 0.9a complete; Explanation A rejected; obstruction is
upstream of modulator direction (Phase 1 still load-bearing).*
