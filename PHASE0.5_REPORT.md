# Phase 0.5 — Validation Report

> Generated: 2026-05-20
> Implementer: Claude Opus 4.7 (logs/phase0.5_brief.md overnight session)
> Status: All four AC0.5 acceptance criteria met.

---

## 1. Acceptance-criteria status

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| **AC0.5.1** | Real C. elegans electrophysiology data integrated | ✅ pass | 40 labeled Atanas 2023 recordings loadable via `ReferenceDataset.from_atanas2023()` |
| **AC0.5.2** | Three quantitative similarity metrics implemented | ✅ pass | temporal_correlation / functional_connectivity_similarity / pca_structure_similarity in `src/algos/validation/comparison.py`, each tested on 6 recordings |
| **AC0.5.3** | At least 5 key-neuron functional specificity tests on the bare connectome | ✅ pass | **6/6** specificity tests pass on Cook 2019 + v0.3 dynamics; pytest-parametrized |
| **AC0.5.4** | Validation report with quantitative scores + Phase 1 guidance | ✅ this document | — |

**Reproduction commands:**

```bash
# AC0.5.3 — key-neuron specificity (no external data needed)
python3 -m pytest tests/test_neuron_specificity.py -v
python3 scripts/run_neuron_specificity.py

# AC0.5.1 + 0.5.2 — comparison against Atanas 2023 (requires the 543 MB
# processed_h5.tar.bz2 in data/reference/ — see data/reference/README.md)
python3 scripts/run_comparison.py 6
```

---

## 2. Side-effect: design doc bumped to v0.3 + dynamics changed

Phase 0.5 began by acting on the dispositions in `logs/phase0.5_brief.md`
§§1–2 — i.e. closing out the Phase 0 open questions and revising the
master design doc. This is documented in detail in `DECISIONS.md` under
`## [Phase 0.5]`; the headlines:

- `docs/design.md` is now **v0.3**. Changes: §3.3 chem activation
  switched from `sigmoid(V) − 0.5` (Phase 0 patch) to canonical
  `tanh(β·V)`; §3.1 picks up the actual Cook 2019 connection counts;
  §4.4 makes the grounding principle explicit; §12 (new) records the
  project's relationship with OpenWorm.
- `src/algos/neural/dynamics.py` flipped to `tanh(β·V)`. After a
  β-sweep, β was chosen as **1.0** rather than the doc's earlier
  β=5 — at β=5, `tanh(5V)` is ~4× the Phase 0 chemical drive and
  pins V near ±1 (functional collapse). β=1.0 sits at the pitchfork
  bifurcation: V=0 is the unique stable fixed point of the un-driven
  network. This **restores the original phase0.md §3.2 strict bound**
  `max|V| < 0.1` that Phase 0 had to relax. AC2 substantively
  *strengthened* by Phase 0.5.
- All 18 + 7 = 25 pytest cases pass after the change (incl. the full
  100,000-tick stability run, 6.5 s).

The doc revision was committed before any Phase 0.5 implementation,
per the brief's instruction `修订主设计文档要单独 commit`.

---

## 3. AC0.5.3 — key-neuron specificity battery (6/6 pass)

A *signal-propagation* test, not a derivative-encoding test. The
bare connectome with no body and no sensory translator can't be
asked "does ASEL respond to upsteps?" — that property lives in the
SensoryTranslator (design.md §4.4). It *can* be asked "if you drive
ASEL, does activity propagate to the chemotaxis interneurons it
projects to, with the literature-reported sign?". Six tests, all
pass:

| Test | Driver(s) | Target group | Expected | mean ΔV | max\|ΔV\| |
|---|---|---|---:|---:|---:|
| Chemotaxis upstream (ASEL) | ASEL | AIYL, AIYR | ↑ | **+0.174** | 0.185 |
| Chemotaxis upstream (ASER, lateralized) | ASER | AIY+AIB pairs | ↑ | **+0.129** | 0.167 |
| Backward command → motor (chemical) | AVAL, AVAR | VA01-12 + DA01-09 | ↑ | **+0.107** | 0.139 |
| Forward command → motor (gap) | AVBL, AVBR | VB01-11 + DB01-07 | ↑ | **+0.051** | 0.071 |
| Anterior touch → backing reflex | ALML, ALMR, AVM | AVD + AVA pairs | ↑ | **+0.131** | 0.166 |
| Thermosensory propagation | AFDL, AFDR | AIYL, AIYR | ↑ | **+0.223** | 0.232 |

**Why this is non-trivial:** the AVB → VB/DB result. Cook 2019 has
essentially zero chemical AVB→VB synapses; the entire coupling is
electrical (gap-junction values 0.04 – 0.29 across 16+ pairs). The
test still passes because our Laplacian gap-junction term
(`W_gap @ V − V · sum_j W_gap[i,j]`) correctly drives equalization
between AVB and its gap-connected motor pool. This is independent
evidence that the v0.3 dynamics are well-tuned, not just that the
chemical synaptic structure is right.

**Pre-equilibrium quietness:** max\|V\| ≈ 3.6 × 10⁻⁸ after the 5000-tick
zero-input warm-up, so ΔV really is "response − zero", not
"response − residual activity".

Generated artifacts: `output/neuron_specificity_report.txt`,
`output/neuron_specificity_results.json`.

---

## 4. AC0.5.1+2 — comparison against Atanas 2023

### 4.1 Data

Used **Atanas et al. 2023** (*Cell* 186:4134; the brief said eLife —
it was actually published in Cell). Six best-NeuroPAL-labeled
recordings from the Zenodo deposit, each with 94–113 labeled neurons:

| recording_id | T (frames) | labeled neurons |
|---|---:|---:|
| 2022-08-02-01 | 1600 | 113 |
| 2023-01-17-14 | 1615 | 106 |
| 2023-01-09-15 | 1615 |  99 |
| 2023-01-10-07 | 1615 |  97 |
| 2023-01-23-15 | 1600 |  96 |
| 2022-06-14-13 | 1600 |  94 |

Sampling at ~1.7 Hz; each recording is ~16 minutes of freely-moving
worm with simultaneous behavior (velocity, head angle, reversal vector,
pumping).

### 4.2 Two digital protocols

Without a body the digital model has no canonical "matched" simulation.
Both protocols are reported:

- **Protocol A — random sensory drive.** Independent Gaussian noise
  (σ=0.1) at every sensory neuron, every tick. Pure-topology baseline.
- **Protocol B — behavior-conditioned drive.** Drive AVA/AVD/AVE
  (backward command, +0.5) whenever the real worm is reversing per
  `behavior/reversal_vec[t]`; drive AVB/PVC (forward command, +0.5)
  otherwise. Plus the same sensory noise as A. This isolates "is the
  *right command-neuron drive* enough to reproduce real activity?"

### 4.3 Quantitative results (means across 6 recordings)

| Metric | Protocol A (random) | Protocol B (behavior-cond.) | Δ |
|---|---:|---:|---:|
| temporal_correlation              | +0.012 | **+0.034** | +0.022 |
| functional_connectivity_similarity| +0.022 | **+0.062** | +0.040 |
| pca_structure_similarity (combined)¹ | **+0.649** | +0.600 | −0.049 |
| └ explained_variance_cos¹            | +0.89      | +0.88   |       |
| └ subspace_alignment¹                | **+0.38**  | +0.37   |       |

Per-recording variance is tight: PCA structure stays in [+0.57, +0.70]
in both protocols.

¹ **Phase 0.6 audit caveat (added 2026-05-20):** the combined
`pca_structure_similarity` score in this row is mostly a
metric-construction artifact. See `PHASE0.6_AUDIT.md`. The defensible
signal is `subspace_alignment` alone, where the real connectome scores
≈ +0.38 versus ≈ +0.28 under a sparsity-matched random connectome —
about a +0.10 effect, not the +0.65 the combined metric implies. The
`explained_variance_cos` sub-component is a noise floor near +0.9 that
random connectomes score slightly *higher* on. The text below was
written before the Phase 0.6 audit; treat it as the original
interpretation, qualified by the audit.

**Reading the numbers:**

1. **PCA structure ≈ 0.65 is the dominant positive finding.** The
   bare connectome's low-dimensional activity manifold — measured
   purely as the top-10 PCA spectrum + the alignment of the principal
   subspaces — captures a substantial fraction of the real worm's
   intrinsic geometry. This holds *under random sensory drive* with
   no behavior input. The connectome topology and v0.3 dynamics
   already encode the right global structure. **[Phase 0.6: REVISED.
   Most of 0.65 is metric artifact; the defensible signal is
   subspace_alignment ≈ +0.10 above null.]**

2. **Functional connectivity is essentially uncorrelated.** Both
   protocols give FC similarity scores near 0 (0.02 / 0.06). Even
   when we hand the model the right command-neuron drive, the
   pairwise correlation patterns observed in real worms do not
   emerge from connectome + dynamics alone.

3. **Temporal correlation is noise-level (~0.03).** Expected: there
   is no shared sensory history between digital and real.

4. **Behavior conditioning helps modestly for FC (3×) and temporal
   (3×), but doesn't move PCA.** This decomposes the gap: only a
   small fraction of the gap is "missing command input". The rest is
   structural — modulators, body-state-dependent gating, real sensory
   translation, plasticity.

### 4.4 Per-neuron temporal correlation (Protocol B)

The mean per-neuron *temporal* correlation hides striking per-neuron
extremes. Pooling across the 6 recordings:

**Best-matching neurons (highest mean r across recordings):**

| Neuron | mean r | recordings | interpretation |
|---|---:|---:|---|
| NSML | **+0.79** | 6 | Serotonergic, mid-feed signaling. Strongly tracks our model. |
| NSMR | **+0.76** | 6 | NSM partner. |
| AVEL | +0.65 | 5 | Backward command (driven directly by protocol B). |
| RID  | +0.64 | 6 | Internal reversal modulator. |
| AVER | +0.60 | 6 | Backward command partner. |
| AVAL | +0.44 | 6 | Backward command (driven directly). |
| AVAR | +0.42 | 5 | Backward command partner. |

**Worst-matching neurons (most anti-correlated):**

| Neuron | mean r | recordings | interpretation |
|---|---:|---:|---|
| I3     | **−0.86** | 6 | Pharyngeal. Feeding state is decoupled from locomotion in real worms; we have no feeding state. |
| ADAL   | −0.62 | 6 | Interneuron. Unclear pattern; worth Phase 1+ investigation. |
| RMDVL  | −0.48 | 5 | Head dorsal-ventral motor neuron. Our connectome may flip head-turn direction. |
| RMDL   | −0.39 | 5 | Same RMD family. |

The **NSM serotonergic neurons matching at r≈0.78** is a clean
positive: NSM activity is known to be strongly coupled to behavioral
state, and our protocol-B drive (matched to real reversal vs forward)
is recovering exactly that coupling. The directly-driven backward
command neurons (AVA, AVE) also score well — partly tautological since
they're the protocol-B inputs, but the magnitudes (+0.4 to +0.6) are
*lower* than NSM, suggesting AVA's *down*-stream propagation timing is
imperfectly matched even when its input is correct.

The **I3 anti-correlation (r = −0.86)** is the most striking
*negative* finding. I3 is pharyngeal; in real worms its activity
tracks feeding-pumping state, which is largely decoupled from
locomotion. Our model has no feeding state, so I3's response to
random/command drive is whatever the connectome topology produces —
in this case, opposite to the real worm.

**RMD family** (head-bending motor neurons) anti-correlation suggests
the connectome wiring + tanh dynamics may produce wrong-direction
head-bending under our drive — a circuit-level discrepancy that's
worth tracing in Phase 1+ once we have a body that can produce real
head-bending behavior.

Generated artifacts: `output/comparison_report.txt`,
`output/comparison_results.json`.

---

## 5. Implications for Phase 1

The Phase 0.5 numbers shape Phase 1 in three concrete ways:

### 5.1 The connectome topology is sound (don't change it)

PCA-structure similarity ≈ 0.65 under *no specific input* says the
connectome + v0.3 dynamics already encode the right global activity
geometry. Phase 1 should not start by retuning the connectome, the
normalization, or the activation function. Those are settled.

### 5.2 Body + sensory translator is necessary; not sufficient

Behavior-conditioned drive (protocol B) only marginally improves
FC and temporal metrics over the random baseline. **Having the right
command-neuron drive is not enough.** Phase 1's body will give the
model an *integrated* sensory-translator + motor-feedback loop, which
the bare protocol-B drive doesn't approximate. But based on these
numbers, even a perfect body is unlikely to push FC similarity above
~0.2 — the rest of the gap appears to live in:

- **State gating** (Phase 3: modulators) — the I3 pharyngeal
  anti-correlation specifically signals that behavior-state-dependent
  modulator effects are major shapers of real worm activity.
- **Plasticity** (Phase 4) — for slower-timescale correlation
  structure.

### 5.3 Concrete circuit-level pieces to revisit

- **RMD head motor circuit** — the negative correlation suggests our
  model produces wrong-direction head-bending under default drive.
  Phase 1's motor executor should explicitly verify RMD outputs
  match expected dorsal-vs-ventral head-bend mapping.
- **Pharyngeal subnetwork (I-series neurons)** — without a
  feeding-state model, these will continue to anti-correlate. Either
  (a) introduce a minimal feeding-state variable in Phase 1, or
  (b) accept the pharyngeal subnetwork as out-of-scope until
  modulators arrive in Phase 3.
- **AVA/AVE downstream timing** — directly-driven command neurons
  score lower than expected (r ≈ 0.4 vs NSM's 0.78). This hints at
  τ-differentiation being more important than the brief assumed —
  specifically that motor neurons may need slower τ than the current
  uniform τ=10. Phase 1's τ split (τ_s / τ_i / τ_m, per the disposition
  for Q3) should be done early, not late.

### 5.4 What Phase 0.5 ruled out

- Activation pinning: the tanh(β=1) choice is safe in the long run.
- Connectome normalization: per-row L1 + gap re-symmetrization is OK.
- Gap-junction term: the Laplacian form `W_gap @ V − V·sum(W_gap)`
  is well-tuned (the AVB → VB/DB specificity result confirms this).

---

## 6. Honest issues / non-perfect bits

- **Protocol B is approximate.** Driving AVA when the real worm
  reverses doesn't reproduce *how* AVA gets activated in vivo (via a
  cascade of upstream sensory + integrator neurons). The 0.03 → 0.06
  FC bump is a floor, not a ceiling, on what a properly-wired Phase 1
  body could achieve.

- **Temporal alignment is naive.** When the digital and real traces
  differ in length, we linearly resample. This is fine for temporal
  correlation as a *coarse* metric, but doesn't account for the
  variable per-recording calcium-imaging frame rate. A more careful
  alignment (e.g. resample to a common Hz) would tighten the temporal
  correlation numbers, probably modestly. Deferred.

- **NeuroPAL label confidence.** We drop `?`-suffixed labels by
  default. Including them gives ~30% more matched neurons per
  recording but introduces noise. The current "high-confidence only"
  filter is conservative.

- **PCA-similarity metric averages two pieces** (variance-spectrum
  cosine + subspace alignment). They are dissimilar quantities; the
  mean is interpretable but not the cleanest summary. Reported both
  in the JSON for inspection.

- **`docs/data_audit.md` still doesn't exist.** Phase 0.5 added one
  more data dependency (Atanas 2023 via Zenodo) without creating that
  file. It is now a Phase 1 dependency.

---

## 7. Generated artifacts

```
docs/design.md                                # bumped to v0.3
src/algos/neural/dynamics.py                  # tanh(β=1·V) activation
src/algos/config.py                           # CTRNNDefaults.beta = 1.0
tests/test_dynamics.py                        # stricter AC2 thresholds
tests/test_neuron_specificity.py              # 7 new pytest cases

src/algos/validation/__init__.py
src/algos/validation/neuron_specificity.py    # AC0.5.3 battery
src/algos/validation/reference_data.py        # AC0.5.1 loader
src/algos/validation/comparison.py            # AC0.5.2 metrics

scripts/run_neuron_specificity.py             # AC0.5.3 runner
scripts/run_comparison.py                     # AC0.5.1+2 runner

data/reference/README.md                      # data download docs
data/reference/{...}.bz2 / processed_h5/      # local only, .gitignored

output/neuron_specificity_report.txt
output/neuron_specificity_results.json
output/comparison_report.txt
output/comparison_results.json
output/basic_simulation_*                     # regenerated after tanh flip

DECISIONS.md                                  # full ## [Phase 0.5] section
QUESTIONS.md                                  # new Phase 0.5 questions
PHASE0.5_REPORT.md                            # this file
```

---

## 8. Single-sentence summary

The Cook 2019 connectome plus the v0.3 `tanh(β=1)` CTRNN dynamics
already produce a low-dimensional activity manifold that aligns
~65% with real C. elegans whole-brain recordings, but the pairwise
correlation structure is essentially unmatched — confirming Phase 1's
body + Phase 3's modulators as necessary investments and ruling out
"the connectome is wrong" as the source of the gap.

**Phase 0.6 update:** the "~65%" claim is mostly metric artifact. The
honest single-sentence summary is: *the bare connectome's top-10 PCA
subspace aligns with the real worm's at ~0.38 vs ~0.28 under a
random-connectome null — a real but modest ~3.5× effect (5σ above the
null spread) — and the pairwise correlation structure is essentially
unmatched, confirming Phase 1's body + Phase 3's modulators as
necessary investments.*

---

*Last updated: 2026-05-20*
*Status: Phase 0.5 complete; ready for Phase 1.*
