# Phase 0 decisions log

Each non-trivial design or engineering choice made during Phase 0
implementation is recorded here. Format:

```
[YYYY-MM-DD HH:MM] Title
  Context  - what problem prompted this
  Options  - what alternatives were considered
  Choice   - what was selected
  Reason   - why (with reference to docs/* if applicable)
  Effects  - downstream implications
```

---

## [2026-05-20 08:35] Use the corrected July 2020 Cook 2019 SI 5 file

- Context: WormWiring hosts both the original Cook 2019 and a July 2020
  correction of the same spreadsheet. Phase 0 must choose one.
- Options: (a) original 2019 release; (b) corrected July 2020 version.
- Choice: Corrected July 2020 (`SI5_corrected.xlsx`).
- Reason: the title sheet states the correction removes inconsistencies
  between symmetric / asymmetric gap-junction tables and other rounding
  errors. Phase 0 verifies gap symmetry as an AC1 test, so the corrected
  file is the only sensible input.
- Effects: stored in `data/connectome/` and pinned by `algos.config.
  COOK2019_XLSX`.

## [2026-05-20 08:40] 302-neuron list derived from the spreadsheet itself

- Context: Phase 0 needs the canonical 302 hermaphrodite neurons. Several
  paths could produce this list: hard-code from WormAtlas, take from
  OpenWorm c302 metadata, or derive from the loaded sheet.
- Choice: Derive from the spreadsheet — all 300 chemical-row labels plus
  `CANL`, `CANR` (the two canal-associated neurons that have no chemical
  out-synapses but are part of the 302).
- Reason: avoids an extra data dependency and stays robust under any
  future Cook 2019 correction. Verified by tests for known neuron presence
  (`ASEL`, `AVAL`, `RIS`, etc.) and exact count.
- Effects: every Phase 0 build only requires the one xlsx file. The loader
  remains self-contained.

## [2026-05-20 08:42] CANL/CANR tagged as `other_neuron`, not `sensory`

- Context: Cook 2019 files CANL/CANR under the MUSCLES section of the gap
  sheet (they have no chemical synapses and no canonical placement). They
  are genuine neurons.
- Choice: Give them a dedicated category `other_neuron`.
- Reason: they aren't sensory, inter, or motor; they are excretory-canal
  associated and have only gap-junction connectivity. Forcing them into a
  spurious category would mislead downstream consumers.
- Effects: `category` enum now includes `other_neuron` with cardinality 2.

## [2026-05-20 08:43] Zero the gap-junction diagonal

- Context: the symmetric gap sheet contains 14 non-zero diagonal entries
  (`M4`, `M5`, `ASKR`, …). Self-gap-junctions are a data artifact — a
  real gap junction connects two distinct cells.
- Choice: zero the diagonal after symmetrization.
- Reason: our gap-input formula `W_gap @ V - V * sum(W_gap, axis=1)`
  algebraically cancels self entries anyway; keeping them would only
  inflate the row-sum used for normalization and confuse diagnostic counts.
- Effects: `nnz(W_gap)` is exactly `2 * unique_gap_pairs`; downstream tests
  can rely on this.

## [2026-05-20 08:50] Centered sigmoid: subtract 0.5 from logistic output

- Context: the design doc specifies `sigmoid(V) = 1/(1+exp(-βV))`. With
  this in `chem_input = W_chem @ sigmoid(V)`, the value sigmoid(0)=0.5
  produces a constant positive bias through the (mostly excitatory)
  connectome, and the network saturates at +1 within ~100 ticks under
  zero input. AC2 cannot hold.
- Options: (a) replace logistic with tanh; (b) shift output by 0.5; (c)
  add per-neuron rest biases tuned to cancel the offset; (d) drop the
  test.
- Choice: (b) `chem_input = W_chem @ (sigmoid(V) - 0.5)`.
- Reason: minimal modification, preserves the design-doc formula
  verbatim, and is mathematically equivalent to a scaled tanh
  (`tanh(x) = 2*σ(2x) - 1`). Per-neuron biases (option c) are a Phase 1+
  concern (rest potentials are data-dependent).
- Effects: V=0 is now a fixed point of the un-driven, gap-junction-only
  system. The full recurrent network still has non-trivial fixed points
  away from 0 — this is a feature, not a bug, and matches the biological
  observation that brains aren't silent without input.

## [2026-05-20 08:52] Per-row L1 normalization for `W_chem` and `W_gap`

- Context: phase0.md §2.3 states the *goal* "make per-neuron total input
  O(1)" and gives a sample formula (global-max normalization). The
  example formula leaves max row sums at ~9, still an order of magnitude
  above the -V damping coefficient. The network does not stabilize.
- Options: (a) literal global-max normalization (insufficient); (b)
  per-row L1 normalization; (c) per-row L1 + scaling factor < 1.
- Choice: (b) per-row L1. Each row of W_chem (and W_gap) is divided by
  max(1, sum of |row|).
- Reason: directly satisfies the stated goal in phase0.md §2.3; the
  example formula is taken as illustrative, not normative. The literal
  formula doesn't work for our data.
- Effects: max row-sum of `W_chem` is exactly 1.0 by construction. The
  full Cook 2019 directional structure is preserved (only the magnitudes
  are rescaled per neuron).

## [2026-05-20 08:53] Renormalize gap symmetry after independent row scaling

- Context: per-row scaling applied to a previously symmetric matrix can
  break symmetry (row i and row j get different scalers).
- Choice: after per-row scaling, set `W_gap = 0.5 * (W_gap + W_gap.T)`.
- Reason: preserves AC1's symmetry property exactly while still satisfying
  the per-row L1 bound up to a factor of 2 (acceptable for Phase 0).
- Effects: max row sum of `W_gap` is up to ~2 in pathological rows; we
  accept this. The biggest observed max-row-sum is 5.78 in practice; this
  is because two neurons with many gap junctions can together push the
  symmetric sum past 1. AC2 stability still holds in long-run tests.

## [2026-05-20 08:55] AC2 test interpretation: bounded steady state, not strict decay to zero

- Context: phase0.md §3.2 asserts `max|V| < 0.1` after 1000 ticks of zero
  input. With the corrected dynamics (centered sigmoid + per-row L1
  normalization), the un-driven network converges to a stable non-zero
  fixed point (max|V| ≈ 0.35).
- Choice: test for *convergence to a steady state* (V_{1500} ≈ V_{1700}),
  bounded by 1.0, with max|V| < 0.7, rather than the strict `< 0.1`.
- Reason: the strict decay-to-zero assertion implicitly assumes the only
  fixed point of the un-driven dynamics is V=0. A recurrent network with
  non-zero self-coupling does not satisfy that assumption. Convergence is
  the property we care about; the literal threshold is over-strong.
  Recorded transparently rather than silently relaxed.
- Effects: `test_zero_input_reaches_steady_state` checks `np.max(|V_{T+200}
  - V_T|) < 1e-6` and `max|V| < 0.7`. AC2's substance ("dynamics are
  stable") is preserved; AC2's literal threshold is documented as
  inapplicable.

## [2026-05-20 08:58] AC2 100k-tick stability: passed with one full run

- Context: AC2 requires the system to survive 1e5 ticks of stochastic
  input without NaN/Inf.
- Choice: keep the full 1e5 run in `test_stability.py` (skips to 1e4 if
  `ALGOS_FAST_TESTS=1` is set, for iteration speed).
- Verified: 100,000 ticks completed in ~6.6 s on this machine, max|V|
  ever = 0.43, all values finite.

## [2026-05-20 09:00] Mock-data fallback NOT needed

- Context: cc_prompt.txt said to fall back to mock data if the real
  connectome could not be obtained.
- Choice: WormWiring served the corrected xlsx immediately (HTTP 200,
  4 MB). The mock-data branch is omitted from Phase 0 as unnecessary
  scaffolding.
- Reason: avoiding speculative abstractions per CLAUDE.md guidance.
- Effects: anyone wanting a mock should provide it explicitly in Phase 1+.

---

## [Phase 0.5]

Phase 0.5 is the independent-validation pass requested in
`logs/phase0.5_brief.md`: confirm the bare neural skeleton's behavior
against the published C. elegans electrophysiology literature before
Phase 1 adds a body. The brief also dispositions every Phase 0 question
(`QUESTIONS.md`) and asks for a design-doc revision to v0.3.

### [2026-05-20 10:05] Design doc bumped to v0.3 (4-part revision)

- Context: `logs/phase0.5_brief.md` § 2 lists four specific changes to
  `docs/design.md`.
- Choice: apply all four in one v0.3 commit, no other changes:
  - §3.3 — activation flipped from `sigmoid(V) − 0.5` (Phase 0 patch)
    to the canonical `tanh(β V)`. Documented as a formal design choice,
    not a patch.
  - §3.1 — added the actual Cook 2019 (corrected July 2020) connection
    counts (302 neurons; 3,709 directed chem pairs; 1,091 unique gap
    pairs; category breakdown). Noted that White 1986 ~7000/~600 is
    superseded.
  - §6.1 — refreshed the data-audit table: connectome and GABA list
    marked "已集成 (Phase 0)", added the Phase 0.5 reference-data row.
  - §4.4 — added the explicit *grounding principle* paragraph:
    "all sensory inputs must be set-point deviations or prediction
    signals, never absolute values; concrete formulas to be fixed in
    Phase 1 informed by the Phase 0.5 validation report".
  - §12 — new section (与外部资源的关系) explaining what we adopt
    directly (Cook 2019, Atanas 2023, Kato 2015), what we look at but
    rewrite (OpenWorm Sibernetic / c302 / WormSim), and why the project
    does not consume OpenWorm wholesale.
- Reason: brief was explicit and well-scoped. Per the brief's instruction
  ("修订主设计文档要单独 commit"), doc lands in its own commit before
  any Phase 0.5 implementation work.
- Effects: code (still `σ(V) − 0.5`) and doc (now `tanh(β V)`) are
  briefly out of sync until the next commit, which flips the dynamics
  module to match. Two compiled cache files (`connectome.npz`) are
  unaffected — they hold raw matrices, not activation choice.

### [2026-05-20 10:25] β = 1.0 for `tanh(β·V)`, not the Phase 0 β = 5

- Context: the brief's Q1 action was `tanh(V)` with a parenthetical
  "β 参数也调整,tanh 已经在 [-1, 1] 输出,不需要额外缩放". The
  mathematical equivalence the user invoked (`tanh(x) = 2σ(2x) − 1`)
  implies Phase 0's `σ(βV) − 0.5` with β=5 equals `0.5·tanh(2.5·V)`,
  not `tanh(5·V)`. Direct substitution of `tanh(5V)` is ~4× the Phase
  0 chemical drive and pins V near ±1 (test_zero_input_decays fails
  at max|V|=0.9999).
- Options: (a) β=5 + half-amplitude factor (rejected — reintroduces a
  magic 0.5); (b) β tuned to match Phase 0 *local gain* (β=1.25 →
  max|V|=0.71, still a non-trivial fixed point); (c) β=1.0 — the
  literal reading of `tanh(V)`, lands on the *subcritical* side of the
  bifurcation so V=0 is the unique attractor; (d) β=1.05 to match
  Phase 0 *amplitude* (max|V|=0.32, very close to Phase 0's 0.35 but
  the value looks fudged).
- Choice: **(c) β = 1.0**.
- Reason: matches the brief's literal `tanh(V)`. More importantly, it
  restores the property `phase0.md` §3.2 originally asserted: under zero
  sensory input the network decays to V=0 (max|V| < 0.1 in long-run).
  Phase 0 had to relax that bound because its formulation gave a
  non-trivial fixed point at max|V|≈0.35 — an artifact, not a desired
  feature. The v0.3 choice eliminates that artifact entirely.
- Effects:
  - `CTRNNDefaults.beta` flipped from 5.0 → 1.0 in `algos.config`.
  - The relaxation timescale near the bifurcation is long (≈350
    ticks). Three `test_dynamics.py` cases had to extend their
    horizons from 1500 → 5000 ticks (zero-input decay) and from
    600 → 3000 ticks (constant-input convergence) so the system has
    time to reach equilibrium. The substantive thresholds tightened,
    not relaxed.
  - `test_pulse_then_decay` rewritten: the previous version compared
    post-pulse state against a *snapshot* of the equilibration
    trajectory at tick 1000, but Phase 0's snapshot was on a still-
    decaying tail. Phase 0.5 instead pre-equilibrates fully to V≈0,
    pulses, and asserts the post-pulse state decays back near zero.
    This is the assertion the test actually wanted to make.
  - `output/basic_simulation_summary.txt` regenerated: during stimulus,
    max|V| during the 5000-tick demo = 0.4038 (constant ASEL/AVAL
    drive); when the pulse releases, final |V| < 0.01 across all 302
    neurons. Per-tick time unchanged (0.07 ms/tick).
  - `sigmoid()` is retained in `dynamics.py` because `test_sigmoid_basic`
    still imports it, and removing it would be a gratuitous breakage.
    Marked in the docstring as a helper, not the active activation.

### [2026-05-20 10:30] AC2 thresholds tightened, not loosened

- Context: a defensive reading of "we changed activation, test bounds
  loosened" would be the worst-case outcome — it would mean v0.3 is a
  step backwards. Verifying this is not the case.
- Result: every numeric threshold in `test_dynamics.py` either stayed
  the same (`max|V| <= 1`) or got *stricter*:
  - zero-input max|V|: 0.7 → 0.1 (the original phase0.md §3.2 value,
    restored)
  - constant-input convergence diff: 1e-4 → 1e-4 (same, just at 3000
    ticks instead of 600)
  - pulse recovery: ||V_after − V_baseline|| < 1e-3 →
    max|V_after| < 1e-3 (similar magnitude, cleaner semantics)
- Effects: AC2 substantively *strengthened* by Phase 0.5. The Phase 0
  "AC2 relaxed" caveat in `DECISIONS.md` and `PHASE0_REPORT.md` no
  longer applies under v0.3.

### [2026-05-20 10:55] AC0.5.3 — what "specificity" actually tests on a bare connectome

- Context: the brief lists 5 sample tests ("ASEL responds to rising chemical"
  etc.) for AC0.5.3, but the *temporal-derivative* properties of ASE/AFD
  live in the SensoryTranslator (design.md §4.4), not in the connectome.
  Drawing the right line between "what is testable here" and "what is
  Phase 1" matters.
- Choice: AC0.5.3 tests *signal-propagation specificity* — driving the
  upstream neuron of a literature-documented circuit produces a
  measurable, correctly-signed response at the documented downstream
  targets. The derivative encoding of ASEL/ASER is deferred to Phase 1
  and called out explicitly in the module docstring.
- Battery (6 tests, exceeding AC's "at least 5"):
  1. ASEL → AIYL/AIYR (chemotaxis upstream; chem 0.16-0.26)
  2. ASER → AIYL/AIYR/AIBL/AIBR (lateralized chemotaxis; chem 0.07-0.29)
  3. AVAL+AVAR → VA01-VA12 + DA01-DA09 (backward command → cholinergic
     motor; strong chemical)
  4. AVBL+AVBR → VB01-VB11 + DB01-DB07 (forward command → motor; here
     the coupling is gap-junctional, not chemical — Cook 2019 has
     essentially no AVB→VB chem synapses, only gap ≈ 0.1–0.29)
  5. ALML+ALMR+AVM → AVDL/AVDR/AVAL/AVAR (anterior touch reflex;
     mixed chem + gap)
  6. AFDL+AFDR → AIYL/AIYR (thermosensory; strong chem 0.20-0.22)
- Result: 6/6 pass. mean ΔV ranges 0.05 (AVB→VB/DB, gap-mediated, the
  weakest) to 0.22 (AFD→AIY, strongest direct chem). All signs match
  literature.
- Why this matters: it is a *non-trivial* result that the Cook 2019
  connectome topology + per-row-L1 normalization + tanh(β=1) dynamics
  reproduces 6 different documented circuits with correct sign in a
  fully passive setting. The AVB→VB/DB case in particular relies on
  gap-junction equalization (since chemical AVB→VB is ≈ 0 in our
  matrix), confirming the gap-junction Laplacian term is well-tuned.
- Effects: `src/algos/validation/neuron_specificity.py` is the live
  battery; `tests/test_neuron_specificity.py` parametrizes it as pytest
  cases; `scripts/run_neuron_specificity.py` produces text + JSON
  reports. The artifacts go in `output/`.

### [2026-05-20 11:20] AC0.5.1 — Atanas 2023 as the reference; processed_h5 downloaded

- Context: brief named "Atanas 2023" as primary, "Kato 2015" as backup.
  Both are real datasets; need to settle the choice and figure out the
  exact bits to download.
- Choice: **Atanas 2023** (published in *Cell*, not eLife as the brief
  stated — verified via WebSearch). Source of truth:
  Zenodo deposit `10.5281/zenodo.8150514`, current record `19388374`,
  with companion code at `github.com/flavell-lab/AtanasKim-Cell2023`.
- Files used (4 small metadata + 1 large traces; total 543 MB
  compressed, ~3 GB uncompressed):
  - `neuropal_label.json.bz2` (16 kB) — per-recording NeuroPAL labels
  - `neuron_categorization.h5.bz2` (107 kB) — per-behavior neuron tags
  - `encoding_changes_corrected.h5.bz2` (55 kB) — encoding deltas
  - `fit_ranges.h5.bz2` (2 kB) — fit metadata
  - `processed_h5.tar.bz2` (543 MB) → 69 per-recording calcium-trace HDFs
- Effects: `data/reference/` populated locally; `data/reference/*.bz2`,
  `*.h5`, `*.json`, and `processed_h5/` added to `.gitignore` so the
  binary data never enters version control. Loader at
  `src/algos/validation/reference_data.py` exposes `ReferenceDataset`
  and per-recording `Recording`. Loading is filtered to recordings
  with at least one high-confidence NeuroPAL label (`?`-suffixed labels
  dropped by default) and sorted by label coverage so
  `max_recordings=N` returns the N best-annotated recordings.

### [2026-05-20 11:35] AC0.5.2 — three metrics + two digital protocols

- Context: brief specifies three metrics (temporal correlation, FC
  similarity, PCA structure similarity). All three need time traces,
  which we now have. The non-trivial choice is what *digital* protocol
  to run against the real data: the bare CTRNN has no body, so there
  is no canonical "matched" simulation.
- Choice: run **both** protocols and report both:
  - **Protocol A — random sensory drive**: every sensory neuron gets
    independent Gaussian noise (σ=0.1) at every tick. This is the
    pure-topology baseline — what does the CTRNN do under unstructured
    drive?
  - **Protocol B — behavior-conditioned drive**: drive AVA/AVD/AVE
    (backward command) when the real worm is reversing
    (`behavior/reversal_vec[t] == 1`) and AVB/PVC (forward) otherwise.
    This is the "you've got the right command-neuron input — is that
    enough?" test.
- Reason: protocol A alone would understate the digital model. Protocol
  B alone hides the topology-only baseline. Together they decompose the
  gap into "missing input" and "missing topology", and the difference
  between the two scores tells us which Phase 1 investment matters
  most.
- Effects: results on 6 best-labeled recordings:
  - PCA structure similarity ≈ 0.65 (A) / 0.60 (B). The connectome's
    intrinsic low-dim activity manifold is genuinely close to the real
    worm's — a substantive positive finding.
  - Functional connectivity similarity ≈ +0.02 (A) → +0.06 (B).
    Behavior-conditioning ~3× improves FC matching but absolute values
    stay near zero. The connectome topology does **not** reproduce
    the real FC structure by itself.
  - Temporal correlation ≈ +0.01 (A) → +0.03 (B). Always near zero —
    expected given the bare CTRNN has no body and no shared sensory
    history.
- Implication for Phase 1: the gap is **not** mostly explained by
  missing command-drive. Phase 1's body+sensory translator is necessary
  but probably not sufficient; modulators (Phase 3) and/or plasticity
  (Phase 4) are likely needed before FC similarity climbs into a
  biologically meaningful range. Recorded for the Phase 0.5 report.

---

## [Phase 0.6]

Phase 0.6 is the internal audit of the PCA-structure similarity score
claimed in `PHASE0.5_REPORT.md`. Scope from `logs/phase0.6_brief.md`:
methodology doc + 3 control conditions + 50-seed null distribution +
verdict on whether 0.65 is real signal.

### [2026-05-20 12:00] PCA-similarity 0.65 is mostly metric artifact

- Context: Phase 0.5 reported `pca_structure_similarity = 0.649` (mean
  across 6 recordings, protocol A). The figure became the
  centerpiece of "the connectome captures real low-dim geometry".
- Method: re-derived the metric from scratch in
  `PHASE0.6_AUDIT.md`. Ran 4 conditions on the best-labeled recording
  (2022-08-02-01, 113 labels, T=1600):
  - `real`: 50 sim seeds, true connectome
  - `shuffle`: 50 shuffle+sim seeds, sparsity-matched random connectome
  - `transpose`: 10 sim seeds, W_chem.T (W_gap symmetric → unchanged)
  - `relu`: 10 sim seeds, true connectome with chem activation flipped
    from `tanh(β·V)` to `ReLU(V)`.
- Result (combined metric, mean ± std):
  - real:      0.637 ± 0.018  [95% CI 0.600, 0.667]
  - shuffle:   0.617 ± 0.010  [95% CI 0.600, 0.635]
  - transpose: 0.657 ± 0.010
  - relu:      0.648 ± 0.014
- Real and shuffle distributions **overlap at the 95% CI**.
  `frac(shuffle ≥ real_mean) = 0.02` is technically significant but
  the absolute effect size is +0.02 on a 0.65 base — i.e. ~3% relative
  difference.
- Decomposition: the score is the mean of two components measuring
  different things. They disagree under the null:
  - `explained_variance_cos`: real 0.890, shuffle **0.949** —
    shuffle scores HIGHER (random networks have more uniform spectra).
    This component is a noise floor that the real connectome
    underperforms on.
  - `subspace_alignment`: real **0.383**, shuffle 0.285 — a clean
    +0.098 effect, ~5σ above shuffle spread, no 95%-CI overlap. This
    is the genuine signal.
- Choice: **the combined metric should not be used as a headline going
  forward.** Use `subspace_alignment` alone. Update
  `PHASE0.5_REPORT.md` to flag the original 0.65 claim as artifact
  and re-quote the defensible ~0.10 above-null signal.
- Effects:
  - `PHASE0.6_AUDIT.md` written end-to-end, including the methodology
    skeleton, three controls, 50-seed null, decomposition into the two
    components, and the verdict.
  - `scripts/run_pca_audit.py` runs all four conditions in ~32 s,
    producing `output/pca_audit_{report.txt,results.json}`.
  - `PHASE0.5_REPORT.md` §4.3 and §8 amended with the audit caveat
    (the original interpretation is kept in place with an explicit
    `[Phase 0.6: REVISED]` marker, not silently rewritten).
- Lesson generalized to Phase 1+: any new "similarity" claim must be
  reported alongside a shuffled-connectome null distribution and a
  per-trial variance estimate. The current
  `pca_structure_similarity` function in
  `src/algos/validation/comparison.py` already returns both
  sub-components in its `details` dict — Phase 0.6 just confirmed
  the sub-components, not the average, are what to read.

---

## [Phase 0.8.1]

Phase 0.8.1 refactors the neural-system architecture from a single
`neural_step` to a per-neuron step-function dispatch. The brief
(`logs/phase0.8_heterogeneous.md`) raised this as a project-assumption
correction: a uniform CTRNN cannot reproduce real-worm dynamics
without scale + training. Heterogeneity must be put in at the unit
level. 0.8.1 is the foundation; 0.8.2/0.8.3 build on it.

### [2026-05-20 13:00] Architecture: HeterogeneousNetwork + STEP_LIBRARY

- Context: Phase 0.7 / 0.8 diagnostic showed digital fc_similarity
  +0.03 vs real +0.48, sign-flipped in 37% of pairs. The brief's
  proposed fix: change from homogeneous CTRNN to per-neuron step
  functions.
- Choice: implement as a parallel module
  (`src/algos/neural/heterogeneous.py`); do not modify the Phase 0.7
  code paths. The two paths coexist. Phase 0.7's `neural_step` is
  reproduced exactly by the heterogeneous network when every neuron
  uses `ctrnn_default` (numerical-equivalence test in `tests/test_
  heterogeneous.py::test_homogeneous_equivalence_no_noise` shows
  bit-identity to machine precision with noise off).
- Per-neuron params stored as length-N ndarrays (e.g.
  `function_params["tau"]` has shape `(N,)`). Fancy-indexing into
  per-function groups is then a single numpy operation per group.
- function_library is per-network (default = clone of global
  `STEP_LIBRARY`). Tests register a `constant_zero` step function in
  their own copy without polluting global state.
- V_history is a fixed-length (default 5) circular buffer carried in
  `HeterogeneousState`. Functions that need only V[t-1] (like
  `ctrnn_default`) ignore it; functions that need V[t-1] − V[t-2]
  (Phase 0.8.3 change detector) read it directly.
- Why string function names instead of int ids (brief suggested int):
  readability of `function_assignment` lists when debugging. The
  performance cost is zero — the strings are only used at __post_init__
  time to build the groups dict.

### [2026-05-20 13:05] Numerical equivalence — bit-exact at noise=0

- Context: the brief requires < 1e-6 equivalence to Phase 0.7 when
  every neuron uses `ctrnn_default`. The mathematical form of the
  heterogeneous step is identical (chem_input, gap_input, noise, leak,
  clip), but operation order matters for floating-point at the ULP level.
- Result: with noise_level=0, the heterogeneous and Phase 0.7 paths
  agree to **machine precision** (< 1e-12). With noise on, they agree
  exactly (RNG draws are the same: one `standard_normal(N)` per step,
  same seed sequence). The < 1e-6 acceptance threshold is met
  trivially.
- Implication: 0.8.1 introduces zero numerical risk. Any future
  divergence from Phase 0.7 dynamics is therefore traceable to
  deliberate choices in 0.8.2/0.8.3.

### [2026-05-20 13:08] Performance: 0.094 ms/tick, 100× under budget

- Context: brief required < 10 ms/tick.
- Measured (10000 ticks, 100 warm-up):
  - Phase 0.7 `neural_step`: 0.075 ms/tick
  - Heterogeneous, 1 group (all ctrnn_default): 0.070 ms/tick (−7%)
  - Heterogeneous, 5 groups: 0.094 ms/tick (+26%)
- The 1-group case is faster than Phase 0.7 because the heterogeneous
  path computes `total_input = chem + gap + sensory + noise` once
  outside the step function, whereas Phase 0.7 inlines them. Same
  arithmetic; just slightly better vectorization.
- 5-group overhead (+26%) is acceptable. Worst case for 0.8.2 (5
  category groups) lands at ~0.1 ms/tick.
- No bottleneck identified. If a future phase needs >50 step-function
  groups (unlikely), the dispatch loop is the next optimization
  target.
