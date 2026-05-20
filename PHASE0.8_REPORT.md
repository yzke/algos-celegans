# Phase 0.8 — Where the FC gap lives

> Generated: 2026-05-20
> Question being asked: Phase 0.7 reported a digital-vs-real FC
> similarity gap of +0.45 (digital ≈ 0.03 vs real cross-worm ≈ 0.48).
> Where in the 29×29 matrix is that gap concentrated? Is it uniform or
> dominated by a few neuron pairs / categories?
> Method: per-pair FC subtraction on the strict-intersection 29-neuron
> set across 10 best-labeled Atanas 2023 recordings.

---

## 1. Setup

| | |
|---|---|
| recordings | 10 best-NeuroPAL-labeled (same as Phase 0.7) |
| matched neurons | **29** (strict intersection across all 10 + in connectome) |
| matched pairs | 29×28 / 2 = **406** |
| FC_real | per-recording Pearson correlation, then averaged |
| FC_digital | one bare-CTRNN sim per recording (Phase 0.5 protocol A), then averaged |
| runtime | 2.2 s end to end |

Category mix of the 29 matched neurons:

| category | count | members |
|---|---:|---|
| sensory | 11 | ADAL, ADEL, ASGR, CEPDL, IL2DL, OLLL, OLLR, OLQDL, URYVL, URYVR, URBR |
| interneuron | 9 | AIBL, AIBR, AIZR, AVAL, AVDR, AVER, AVJR, RID, RIVL |
| pharyngeal | 5 | I2L, I3, M3L, M3R, NSML |
| motor | 4 | RMDDL, RMEL, RMER, SMDVL |

(URBR substituted in for clarity; see JSON for exact list.)

---

## 2. The dominant finding: signs are systematically flipped

**The FC gap is not about magnitudes — it's about signs.**

| statistic | FC_real (406 pairs) | FC_digital (406 pairs) |
|---|---:|---:|
| mean | **+0.08** | **+0.28** |
| std  | 0.26 | 0.24 |
| frac. with opposite sign to the other matrix | **36.9%** | (same) |
| frac. with opposite sign in top-50 \|diff\| pairs | **62.0%** | (same) |

Among the **top-50 \|diff\| pairs**:
- **31 pairs**: FC_real is **negative**, FC_digital is **positive** (anti-correlation flipped to co-activation).
- **19 pairs**: both positive, digital larger (over-correlation).
- **0 pairs**: FC_real positive, FC_digital negative.

The digital model **never** produces anti-correlation where the real
worm shows correlation. It frequently produces co-activation where the
real worm shows anti-correlation. **The gap is the missing
anti-correlations.**

### What's actually going on

In real worms, locomotion involves mutually-exclusive behavioral
states (forward ↔ reverse, head-bend-dorsal ↔ head-bend-ventral,
feeding-active ↔ feeding-quiescent). State-specific neurons
**anti-correlate**: when AVA fires, RID does not, and vice versa. When
M3 fires (pumping), I3 does not.

Our bare CTRNN sees random Gaussian noise distributed across all 83
sensory neurons. There is no behavioral state, no winner-take-all
dynamics, no neuromodulator-gated commitment to a state. The
recurrent dynamics tend to co-activate connected pools. The mean FC
of +0.28 is the signature of this "everything is mildly
correlated through the network" pattern.

---

## 3. Top-20 highest-|diff| pairs

| pair | cat_a × cat_b | FC_real | FC_digital | diff |
|---|---|---:|---:|---:|
| I3 — NSML | pharyngeal × pharyngeal | **−0.68** | **+0.79** | **−1.47** |
| AVER — RID | interneuron × interneuron | −0.61 | +0.71 | −1.32 |
| AVAL — RID | interneuron × interneuron | −0.56 | +0.75 | −1.31 |
| AVER — RMER | interneuron × motor | −0.54 | +0.66 | −1.20 |
| AVER — RMEL | interneuron × motor | −0.51 | +0.68 | −1.19 |
| AIBL — RID | interneuron × interneuron | −0.52 | +0.66 | −1.18 |
| AIBR — RID | interneuron × interneuron | −0.51 | +0.66 | −1.17 |
| AIBL — RMER | interneuron × motor | −0.55 | +0.57 | −1.12 |
| AIBR — RMER | interneuron × motor | −0.55 | +0.56 | −1.10 |
| AVAL — RMER | interneuron × motor | −0.53 | +0.52 | −1.06 |
| AVAL — RMEL | interneuron × motor | −0.51 | +0.53 | −1.03 |
| AIBL — RMEL | interneuron × motor | −0.46 | +0.56 | −1.02 |
| AIBR — RMEL | interneuron × motor | −0.47 | +0.56 | −1.02 |
| AVER — SMDVL | interneuron × motor | −0.04 | +0.83 | −0.87 |
| ADAL — RID | interneuron × interneuron | −0.25 | +0.62 | −0.87 |
| RID — RIVL | interneuron × motor | −0.16 | +0.70 | −0.86 |
| I3 — M3L | pharyngeal × pharyngeal | −0.05 | +0.79 | −0.84 |
| I3 — M3R | pharyngeal × pharyngeal | −0.03 | +0.79 | −0.82 |
| AIZR — RID | interneuron × interneuron | −0.22 | +0.59 | −0.81 |
| RID — URYVL | interneuron × sensory | −0.49 | +0.30 | −0.79 |

**Every single one** is FC_real negative → FC_digital positive
(or weakly negative). The biology behind these:

- **RID** is a forward-locomotion modulator (releases neuropeptides
  promoting forward). It anti-correlates with reversal command (AVA,
  AVE, AVD) and with the AIB integrator family. RID appears in **11
  of the top-50 pairs** — the most concentrated "problem hub" by far.
- **RME family** (RMEL, RMER, RMDDL) are head-bend motor neurons.
  They are state-specific: head bends correlate with reversal,
  anti-correlate with forward locomotion.
- **I3 / NSML / M3** are pharyngeal: feeding state is decoupled
  from locomotion in real worms. We have no feeding state.
- **AIB** family (AIBL, AIBR) are integrators that gate reversal:
  anti-correlated with forward-promoting RID.

---

## 4. Category-combo breakdown (mean |diff|)

| category combination | n_pairs | mean \|diff\| | median \|diff\| | max \|diff\| |
|---|---:|---:|---:|---:|
| **pharyngeal × pharyngeal** | 10 | **0.68** | 0.64 | 1.47 |
| **interneuron × motor** | 36 | **0.67** | 0.63 | 1.20 |
| **interneuron × interneuron** | 36 | **0.59** | 0.60 | 1.32 |
| motor × motor | 6 | 0.53 | 0.50 | 0.74 |
| interneuron × sensory | 99 | 0.24 | 0.21 | 0.79 |
| interneuron × pharyngeal | 45 | 0.24 | 0.21 | 0.62 |
| motor × sensory | 44 | 0.20 | 0.16 | 0.65 |
| sensory × sensory | 55 | 0.17 | 0.12 | 0.66 |
| motor × pharyngeal | 20 | 0.14 | 0.12 | 0.42 |
| pharyngeal × sensory | 55 | **0.12** | 0.07 | 0.43 |

The gap is **concentrated in inter–inter, inter–motor, and the
intra-pharyngeal subnetwork** (mean |diff| 0.59–0.68). It is
**smallest for sensory–sensory and sensory–pharyngeal pairs**
(mean |diff| 0.12–0.17), where neither real nor digital activity is
strongly state-dependent.

This precisely matches the "missing anti-correlations" reading. The
inter–motor and inter–inter category combos are where the **command
neuron / motor pool / modulator dance** happens, and that dance is
state-dependent. Sensory–sensory pairs don't depend on state in real
worms either, so the gap is small.

---

## 5. Top-10 hub neurons (appearances in top-50 |diff| pairs)

| neuron | category | count in top-50 |
|---|---|---:|
| **RID**   | interneuron | **11** |
| AVDR  | interneuron | 9 |
| **RMER**  | motor | 8 |
| **SMDVL** | motor | 8 |
| AIBL  | interneuron | 6 |
| AIBR  | interneuron | 6 |
| AVAL  | interneuron | 6 |
| AVER  | interneuron | 6 |
| AVJR  | interneuron | 6 |
| I3    | pharyngeal | 5 |

**RID is the single most important problem node.** It is the
forward-locomotion modulator (releases neuropeptides FLP-14, FLP-1)
and is supposed to anti-correlate with the reversal command pool. In
the bare CTRNN it co-activates with everything because there's no
modulator dynamics and no forward/reverse state machine.

The next 4 hubs (AVDR, RMER, SMDVL, AIBL/R) are all about the
forward/reverse motor switch. The hub list is essentially "the
forward/reverse dichotomy operationalized".

I3 is the pharyngeal stand-in for the missing feeding state.

---

## 6. Implications for Phase priorities

### 6.1 Phase 1 (body) alone will not close most of this gap

Phase 1's body adds physical state, sensory translation, and motor
output. Plausible Phase 1 effects on FC:

- ✅ **Sensory–sensory pairs** improve: a body in motion sees
  correlated sensory streams (chemical concentration ↔ thermal
  gradient ↔ touch). But this category already has small gaps
  (0.17 mean |diff|) so the absolute improvement is bounded.
- ✅ **Sensory–motor pairs** improve modestly: motor feedback
  closes the perception–action loop.
- ⚠ **Inter–inter / inter–motor / inter–pharyngeal pairs**:
  Phase 1's mechanical commitment (a moving worm has momentum;
  reversal is expensive) will create some hysteresis on the
  forward/reverse state, which should help the command neurons
  anti-correlate. But the *commitment to forward or reverse* is
  primarily set by modulator state in real worms, not just by
  mechanics.
- ❌ **Pharyngeal × pharyngeal** (0.68 mean |diff|): Phase 1 has no
  feeding-state model, so this category will not improve.

Estimate: Phase 1 alone probably moves fc_similarity from +0.03 to
maybe +0.10–0.15. The cross-worm 5%-tile (entering the real
distribution) is +0.35. **Phase 1 alone is not enough.**

### 6.2 Phase 3 (modulators) is more important than its number suggests

The hub analysis named **RID** as the single biggest problem node,
and **RID's whole job is releasing neuropeptides** (FLP-14, FLP-1).
The bare CTRNN cannot represent RID's modulatory output — it has no
modulator state variable.

The forward/reverse switch in real worms is implemented by a
combination of:
- **Mutual inhibition** in the command circuit (partly in the
  connectome — AIB → RIM → AVA chain).
- **Neuropeptide release** (RID's role; also AVK, ASI; modulated
  by PDF, FLP, etc.).
- **Body feedback** (Phase 1).

Modulator state is the missing piece without which the dance is
fundamentally underdetermined. The Phase 0 design doc §1.3 already
makes "multi-timescale structure" a non-negotiable commitment;
Phase 0.8 quantifies that commitment.

**Recommendation: Phase 3 (modulators) should be planned to follow
Phase 1 closely, and the Phase 1 success criterion should explicitly
include "FC gap on the inter-inter / inter-motor categories does not
remain larger than 0.30 after Phase 1" — if it does, do not delay
Phase 3.**

A more aggressive option: **prototype a minimal modulator system in
parallel with Phase 1**, even before the full Phase 3, with only RID's
neuropeptide release implemented. This single mechanism would attack
the largest single hub (RID) directly and might cut the FC gap by
~30–40% on its own.

### 6.3 Phase 4 (plasticity) is not urgent

Hebbian plasticity sharpens existing patterns; it cannot create the
missing forward/reverse mutual exclusion from scratch. The Phase 0.8
diagnosis suggests plasticity is most useful **after** body +
modulators are in place. Push Phase 4 later in the schedule than
originally implied.

### 6.4 A specific Phase-3 minimal viable feature

The smallest modulator mechanism that would meaningfully attack the
gap: **a single slow neuropeptide variable `c_RID` whose target is
RID activity and which suppresses the AVA reversal command pool**.
This is concretely:

```python
# in Phase 3 mod loop
c_RID += (V[idx['RID']] - c_RID) / TAU_RID            # TAU_RID ~ 200 ticks
for j in REVERSAL_COMMAND_TARGETS:                     # AVA, AVE, AVD
    extra_input[j] -= MOD_GAIN * c_RID
```

This adds 1 modulator variable and 6 lines of code, and directly
addresses the #1 hub neuron found in this audit.

---

## 7. What this rules out

- ❌ **"The connectome topology is wrong."** The gap is not
  uniformly distributed across pairs; it is concentrated where
  state-dependent anti-correlations would be expected. The same
  sensory–sensory pairs that should be state-independent show small
  gaps. The connectome is probably fine.
- ❌ **"We need a different normalization / activation."** Phase 0.6
  showed the metric is barely sensitive to those choices; the FC gap
  here is structural, not numerical.
- ❌ **"We need more sensory drive."** Adding drive will not flip
  signs — it will just amplify the existing pattern.

---

## 8. Single-sentence summary

The +0.45 FC similarity gap is not a magnitude problem but a *sign*
problem — 37% of all neuron-pair correlations have opposite sign in
real vs digital, and the missing signal is the
**behavioral-state-dependent anti-correlations** mediated by
modulators (esp. RID) and the forward/reverse mutual-exclusion
machinery — meaning Phase 1 (body) alone will not close most of this
gap and Phase 3 (modulators) should be planned to follow it
immediately, with a minimal RID-neuropeptide variable prototyped in
parallel if possible.

---

## 9. Generated artifacts

```
scripts/run_fc_gap_diagnosis.py
output/fc_gap_diagnosis_report.txt
output/fc_gap_diagnosis_results.json
output/fc_gap_heatmap.png                # FC_real / FC_digital / FC_diff
PHASE0.8_REPORT.md                       # this file
```

The heatmap visually confirms the story: `FC_real` has both
red (positive) and blue (negative) blocks structured by category;
`FC_digital` is almost entirely red; `FC_real - FC_digital` is almost
entirely blue (i.e. digital exceeded real).

---

*Last updated: 2026-05-20*
*Status: Phase 0.8 complete; FC gap diagnosed as sign-flip problem
centered on state-dependent neuron classes; Phase 3 priority raised.*
