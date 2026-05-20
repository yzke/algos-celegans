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
