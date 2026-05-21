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

## Q14 (Phase 0.9). Missing `notes/algos_hypothesis_v1.md`

The Phase 0.9 brief references this file for the full H_1 / H_1.4
statement and the framework's revised hypothesis. The file is not in
the repo. I proceeded using only the brief's own §1.2 / §1.3 statement
of H_1. Question: was the v1 hypothesis document supposed to be
checked in, or is the brief intended to be the canonical source?

If there is a longer document somewhere, the Phase 0.9 report may
miss nuance in how H_1 differs from H_0 beyond the one-line summary.
The conclusions (P1 refuted at minimal form, Phase 1 first) don't
depend on that nuance, but follow-up phases might.

## Q15 (Phase 0.9). After P1 refutation, where does H_1 stand?

The brief defines three response bands: ≥+0.10 supports H_1.4,
+0.03–+0.10 partial, <+0.03 "H_1 may need re-examination." We're in
the third band. The Phase 0.9 report argues this does **not**
falsify H_1 as a whole, only its minimal-mechanism instantiation —
because the diagnostic ("zero anti-correlations from the bare
network") suggests the modulator never had the structured input that
would let it gate anything.

Open question for the project author: is that the right call? Or
does a clean failure of P1 in a model that has the connectome plus
local transforms but no sensory/body coupling argue that the
hypothesis needs more substantial revision before we try Phase 1?

I lean: keep H_1, sharpen its conditions ("modulators are necessary
*and* require structured input"), and re-test in Phase 1. But this
is the kind of choice where author input matters.

---

## [Phase 1.5+]

### Q-1.5+.1 — Innexin composition / rectification per electrical edge

Cook 2019 SI 5 records only contact counts for gap junctions; the
sheet does not encode rectification polarity. The current loader
treats every electrical edge as symmetric (`sign=+1`, mirrored both
directions), but the C. elegans literature documents asymmetric
(rectifying) gap junctions on AVA↔A-class motor neurons (Starich
2009, Liu 2017), AVB↔B-class motor neurons (Kawano 2011), and
several pharyngeal connections.

This is almost certainly contributing to the Phase 1.0 anomaly
"forward ↔ reversal command +0.51 correlation" (PHASE1.0_REPORT.md
§4): in the real worm, asymmetric coupling lets AVA depolarize
without fully feeding back into AVB, breaking command-pool synchrony.

Open question: do we want to assemble a per-edge rectification
table from the innexin literature for Phase 1.5? Estimated effort:
20-40 edges with published data, ~4-8 hours of literature work for
a comprehensive table.

### Q-1.5+.2 — Tyramine arm of RIM

RIM releases glutamate (modeled, sign +1 via `default`) AND
tyramine (NOT modeled). Tyramine via LGC-55 → chloride channel →
hyperpolarizes AVB, MC, RMD (Pirri 2009). This is a candidate
mechanism for forward-locomotion suppression during escape.

If included in Phase 1.5, options are:
  (a) add a tyramine modulator pool with RIM as producer
      (parameter-level threshold modulation on AVB/MC/RMD); or
  (b) add direct chemical edges from RIM with sign=-1 to those
      targets (state-level inhibition, fast).

Both are biologically plausible. Tyramine action is faster than
typical neuropeptide modulation but slower than direct chloride
channel opening. Phase 1.5 design should pick.

### Q-1.5+.3 — Self-gap entries (14 zeroed at load)

The corrected Cook 2019 gap sheet has 14 nonzero diagonal entries
that we zero at load with the rationale "data artifacts" (cancel
algebraically in the Laplacian). Some innexins (UNC-9 in particular)
form intracellular hemichannels at the cell membrane — these COULD
in principle produce self-coupling in EM. We didn't verify whether
any of the 14 self-entries match neurons with documented hemichannel
expression.

For Phase 1.0 dynamics this is moot. For Phase 2+ (structural
plasticity, channel-level detail) it might matter.

### Q-1.5+.4 — Co-release neurons: tonic vs spike-triggered

Several co-release neurons (HSN, NSM, RIM, VC4/5) are documented to
release peptides / monoamines via either spike-triggered exocytosis
or tonic / volume release. The current Phase 1.0 modulator subsystem
treats them as spike-rate-driven (the rate trace drives c_m). If
some of these are actually tonic, the modulator dynamics need a
different driver (e.g. baseline V level rather than spike rate).

Most relevant for: tyramine release by RIM (Alkema 2005 reports
graded release), serotonin tonic release by NSM during feeding (Sze
2000 reports basal release).


### Q-1.5+.5 — Tyramine: fast (LGC-55, chloride channel) vs slow (GPCR)?

RIM releases tyramine that has two distinct action timescales (Pirri
2009): the fast LGC-55-mediated chloride channel response (~ms) and
the slow GPCR-mediated SER-2/TYRA-3 modulation (~seconds). The fast
arm is properly modeled as direct inhibitory chemical edges
(sign=−1, low delay), but RIM is not GABAergic so our sign rule (per
the GABAERGIC list) treats RIM outputs as +1. The slow arm fits the
Modulator framework.

If Phase 1.5 wires tyramine, design needs to decide: fast (modify
loader to give RIM some output edges sign=−1 based on receptor
expression on the target) or slow (add Modulator entry) or both.
Both have biological basis. Pure-slow risks missing the actual
escape-response timing.

### Q-1.5+.6 — Dopamine antagonist receptor pairs

DOP-1 (D1-like, Gs/cAMP, excitatory) and DOP-3 (D2-like, Gi,
inhibitory) are expressed on overlapping target sets. The same
target neuron can receive both signals through different receptors,
producing direction-flipping effects depending on which receptor
dominates. Our Modulator framework only stores a *single*
`sensitivity` per (modulator, target) pair.

Phase 1.5 design needs to decide whether to:
  (a) split into `dopamine_excite` + `dopamine_inhibit` two modulators
      with disjoint targets (simple, may misrepresent the biology), or
  (b) make sensitivity a function of receptor expression (requires
      receptor data per target neuron — significant additional data
      collection).

### Q-1.5+.7 — Co-release: separate modulators per molecule?

NSM releases serotonin + NLP-3 + FLP-21 from the same vesicles
(Frooninckx 2012). In Phase 1.0 the entire NSM output is folded
into the 5-HT modulator. If we add NLP-3 and FLP-21 as separate
modulator pools with the same producer set, the c_m of each is
driven by the same producer rate — they'll be perfectly correlated
in time, which makes them mathematically redundant unless their
*target* sets differ.

Phase 1.5 design needs to decide: separate modulators only when
target sets are disjoint, OR maintain biological separation even
when targets overlap (clearer accounting, no extra information).

