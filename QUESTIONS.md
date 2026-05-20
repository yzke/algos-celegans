# Open questions for the project author

Items I had to decide on my own while the user was asleep. Worth revisiting
together.

---

## Q1. Centered sigmoid vs. tanh vs. per-neuron rest biases

I subtracted 0.5 from the logistic sigmoid to make V=0 a fixed point under
zero input (see DECISIONS.md). This is mathematically equivalent to using
a scaled `tanh`. Question: which formulation should the documented design
use going forward?

Three options:
1. **Keep centered logistic** (current): preserves the design-doc formula
   in literal form, just shifted.
2. **Switch to `tanh`** in the design doc: cleaner, no shift needed,
   matches Beer/Izquierdo CTRNN literature on C. elegans.
3. **Add per-neuron rest biases θ_i** so equilibrium is at `V*_i = θ_i`:
   biologically faithful but requires data we don't yet have.

I lean (2) for clarity. The shift in (1) is innocuous but obscures the
equation when read in isolation.

## Q2. Normalization choice: per-row L1 vs. spectral-radius scaling

Per-row L1 normalization satisfies "per-neuron total input O(1)" but is a
fairly aggressive simplification: a neuron with 10 inputs of weight 5 and
a neuron with 50 inputs of weight 1 end up with the same total drive.

Alternative: scale `W_chem` (and `W_gap`) by a global factor chosen so
the spectral radius is just below 1. This preserves *relative* input
magnitudes across neurons and would give a more accurate "input drive
distribution," at the cost of a few neurons having much larger inputs
than the average.

Per-row was the simplest stable choice. Phase 1+ should revisit when
modulator scaling enters the picture.

## Q3. CTRNN time constant differentiation

Phase 0 uses a uniform `tau=10` per design.md §3.6 ("initial version
unified"). Real C. elegans has slow modulatory neurons (e.g. RIM, AVK)
and fast command neurons (AVA, AVB). When we differentiate, what data
source do we use? c302 ships τ values; should we adopt those wholesale or
pick a small canonical set (sensory τ_s, inter τ_i, motor τ_m)?

## Q4. Cook 2019 vs. White 1986 reproducibility check

phase0.md §1.5 quotes "~7000 chemical, ~600 gap" — those are White 1986
era counts. Cook 2019 (corrected) actually reports closer to 3,700 chem
pairs and 1,100 unique gap pairs. I've adjusted the test bounds to match
the corrected data. Should we keep both reference numbers in the design
doc?

## Q5. CANL/CANR category label

I gave them their own `other_neuron` category. An alternative is to call
them "modulatory" or "secretory" (they secrete NLP-40 and shape body
volume). They aren't sensory/inter/motor in the classical sense. What
label do you want long-term?

## Q6. Should the Phase 0 demo stimulate a sensory neuron, an interneuron, or both?

The current `scripts/run_basic_simulation.py` stimulates ASEL then AVAL.
Both are biologically interesting (chemosensory; command interneuron for
backing). Open: would you prefer a stimulation pattern that more
graphically demonstrates a known C. elegans circuit (e.g. mechanical
touch on ALM/AVM activating PVC→backward circuit)?

## Q7. How aggressively to silence un-resolved warnings?

`matplotlib.tight_layout` raises a UserWarning because of the share-y
band layout. Harmless but noisy. Tolerable for Phase 0?

---

# Phase 0.5 — questions raised by the validation pass

Items I had to decide on my own; worth revisiting before Phase 1.

## Q8 (Phase 0.5). β = 1 versus a slightly-supercritical β

I landed on β=1 because it restores the original `max|V| < 0.1`
zero-input bound and matches the brief's literal `tanh(V)`. The
trade-off: β=1 is right at the bifurcation, so relaxation timescales
are long (~350 ticks). β=1.05 would put the system on the
super-critical side with a small non-trivial attractor at max|V|≈0.32
(essentially Phase 0's behavior under the new formulation) and faster
relaxation. Is the "clean V=0 attractor" property worth the slower
relaxation? Or would you prefer a slightly-super-critical β with a
non-trivial attractor and faster dynamics?

## Q9 (Phase 0.5). PCA-similarity metric definition

The PCA structure metric averages two quantities — top-K
explained-variance cosine and mean principal-angle cosine — which are
both in [-1, 1] but measure different things. Is averaging them the
right summary, or do you want them reported separately throughout?
Right now they're separated in the JSON but combined in the text
report.

## Q10 (Phase 0.5). Behavior-conditioned protocol — was simulating AVB during forward periods the right choice?

Protocol B drives backward command (AVA/AVD/AVE) when the real worm
reverses and forward command (AVB/PVC) otherwise. An alternative:
drive only AVA during reversals, leave the network un-driven
otherwise. Mine adds a constant forward-command floor, which may
over-saturate AVB downstream. I went with the symmetric version
because it matches the binary nature of the locomotion state in real
worms. Worth a check.

## Q11 (Phase 0.5). RMD anti-correlation — bug or biology?

The RMD family (head-bending motor neurons) shows mean per-neuron
temporal correlation of −0.35 to −0.48 under protocol B. Two
explanations are possible:
  (a) Our model's RMD activity is *wrong direction* — a circuit bug
      to fix in Phase 1.
  (b) RMD activity in real worms is driven by the head-angle feedback
      loop (RMD → head muscle → head turn → mechanosensory feedback →
      RMD) that our Phase 0.5 sim cannot reproduce. Without that loop
      we get an *un*-coordinated head signal.
I lean (b). But Phase 1 should explicitly test this once the motor
loop is in place. Reported as a Phase 1 todo, not a Phase 0.5 issue.

## Q12 (Phase 0.5). Should we include the low-confidence ("?"-suffixed) NeuroPAL labels?

Default loader filters them out. Including them ~30% boosts matched
neuron counts per recording, but adds noise. For Phase 1+ validation
we may want a confidence-weighted average instead of a hard cutoff.
Defer for now.

## Q13 (Phase 0.5). Phase 0 demo schedule

The basic_simulation script was regenerated after the tanh flip;
heatmap structure is now cleaner (clean V=0 baseline, distinct
ASEL and AVAL stimulus peaks at max\|V\|=0.40). Should the schedule
be reworked now that Phase 0.5 has identified specific circuits
worth showcasing (e.g. anterior touch reflex → AVA activation)?
Cosmetic.
