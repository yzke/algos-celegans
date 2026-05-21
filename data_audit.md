# ALGOS Project — Data Audit

> Project-level ground truth for every biological data point currently
> consumed by `algos-celegans`. Built as Phase 1.5+.4 from the
> findings of `notes/edge_sign_audit.md`,
> `notes/subgraph_audit_part1.md`, `notes/subgraph_audit_part2.md`,
> and `data/modulators_full.md`. **This document is the canonical
> reference for "what is the source of X" questions.** When a
> downstream Phase finds an inconsistency between code and biology,
> the *first* place to check is here.
>
> Format conventions for each entry:
> - **Current value**: what the codebase actually uses today.
> - **Source**: specific publication or dataset reference (author,
>   year, journal, DOI when known).
> - **Evidence strength**: `direct` (measured / counted in the cited
>   work), `inferred` (extrapolation from related measurement),
>   `assumed` (no direct support; engineering choice).
> - **Project usage**: where in the code this datum is consumed.
> - **Uncertainty**: known gap, ambiguity, or known approximation.
>
> Documentation only — Phase 1.5+ does not modify `src/algos/`.

---

## Table of contents

1. [Connectome data](#1-connectome-data)
2. [Neuron identity and classification](#2-neuron-identity-and-classification)
3. [Subgraph (functional circuit) definitions](#3-subgraph-functional-circuit-definitions)
4. [Modulator system](#4-modulator-system)
5. [Plasticity data](#5-plasticity-data)
6. [Reference recordings (validation data)](#6-reference-recordings)
7. [Known data gaps and engineering assumptions](#7-known-data-gaps-and-engineering-assumptions)

---

## 1. Connectome data

### 1.1 Source workbook

- **Current value**: `data/connectome/SI5_corrected.xlsx`.
- **Source**: Cook SJ, Jarrell TA, Brittin CA, Bloniarz A, Hagmann D,
  Hobert O et al. (2019). *Whole-animal connectomes of both
  Caenorhabditis elegans sexes.* **Nature** 571(7763): 63–71.
  doi:10.1038/s41586-019-1352-7. Corrected July 2020 distribution from
  WormWiring (https://wormwiring.org).
- **Evidence strength**: `direct` — primary EM-derived adjacency
  matrices, the canonical hermaphrodite connectome.
- **Project usage**: `src/algos/connectome.py::ConnectomeData.load()`,
  cached as `data/connectome/connectome.npz`.
- **Uncertainty**: corrected-July-2020 is the latest known revision;
  any subsequent re-correction would invalidate the cache.

### 1.2 Neuron count

- **Current value**: 302 hermaphrodite neurons.
- **Source**: White et al. 1986 + Cook 2019 (the canonical 302).
  Sheet derivation: 300 chemical row labels + {CANL, CANR} from
  the gap sheet (these have no chemical out-synapses).
- **Evidence strength**: `direct`.
- **Project usage**: `algos.config.N_NEURONS = 302`.
- **Uncertainty**: none.

### 1.3 Chemical edge count

- **Current value**: 3,709 directed chemical edges (nonzero raw EM
  contacts).
- **Source**: Cook 2019 corrected sheet `hermaphrodite chemical`.
- **Evidence strength**: `direct`.
- **Project usage**: `src/algos/graph/loader.py::load_connectome_into_graph`.
- **Uncertainty**: ~5% of EM contacts at the smallest scale (1 serial
  section) are at the limit of detection — likely some have been
  misclassified or missed in the original tracing. Acceptable for
  network-level dynamics.

### 1.4 Gap-junction count

- **Current value**: 1,091 unique unordered pairs (2,182 mirrored
  directed entries in `W_gap_raw`).
- **Source**: Cook 2019 corrected sheet `hermaphrodite gap jn
  symmetric`.
- **Evidence strength**: `direct` (matrix entries), but
  **rectification not encoded** — see §1.5.
- **Project usage**: `src/algos/graph/loader.py` mirrors both
  directions.
- **Uncertainty**: 14 nonzero diagonal "self-gap" entries zeroed at
  load — algebraically cancel in the Laplacian, no dynamics impact;
  data interpretation flagged in QUESTIONS.md Q-1.5+.3.

### 1.5 Gap-junction rectification

- **Current value**: all gap edges symmetric (sign=+1, equal weight
  both directions).
- **Source**: Cook 2019 sheet does **not** record rectification.
- **Evidence strength**: `assumed` (engineering simplification).
- **Project usage**: `loader.py` mirrors symmetric entries; the
  runtime gap-input is `(W_gap @ V - V * sum(W_gap)) / tau`, fully
  symmetric.
- **Uncertainty**: **substantial**. Documented asymmetric junctions
  include AVA↔A-class motor (Starich 2009, Liu 2017), AVB↔B-class
  (Kawano 2011), several pharyngeal. The symmetric assumption likely
  contributes to the Phase 1.0 forward↔reversal +0.51 anomaly. See
  `notes/edge_sign_audit.md` §4 and QUESTIONS.md Q-1.5+.1.

### 1.6 Normalization mode

- **Current value**: `per_row` L1 normalization of W_chem and W_gap
  (each row divided by max(1, row_l1_sum)).
- **Source**: `phase0.md §2.3` stated goal "each neuron's total input
  should be O(1)"; per-row was the minimal correction that satisfied
  AC2 (zero-input decay, no NaN over 10⁵ ticks).
- **Evidence strength**: `assumed` (engineering choice, documented
  in DECISIONS.md Phase 0).
- **Project usage**: `algos.config.NORMALIZATION_MODE`.
- **Uncertainty**: normalization mode does not have a unique
  biological justification; alternatives (global_max, none) were
  tested in Phase 0.

---

## 2. Neuron identity and classification

### 2.1 Per-neuron category

- **Current value**: each of the 302 neurons tagged
  `'sensory' | 'interneuron' | 'motor' | 'pharyngeal' |
  'sex_specific' | 'other_neuron'` based on the Cook 2019 sheet's
  section headers (PHARYNX, SENSORY NEURONS, INTERNEURONS, MOTOR
  NEURONS, SEX SPECIFIC; CANL/R hand-tagged as `other_neuron`).
- **Source**: Cook 2019 sheet sections, which themselves derive
  from White 1986 anatomical classification.
- **Evidence strength**: `direct` for section assignment; `inferred`
  for some borderline cells (e.g. some "sensory" neurons also have
  interneuron-like properties).
- **Project usage**: `src/algos/connectome.py::ConnectomeData.category`;
  consumed by `algos.graph.node.CATEGORY_PARAM_DEFAULTS` for default
  τ/threshold per category.
- **Uncertainty**: distribution: 83 sensory, 81 interneuron, 108
  motor, 20 pharyngeal, 8 sex_specific, 2 other_neuron.
  Borderline cases (~5 neurons) where category is debatable; current
  Cook 2019 assignment used as authoritative.

### 2.2 Per-neuron neurotransmitter

- **Current value**: each neuron tagged `'GABA'` or `'default'`. The
  26 GABA neurons: AVL, DVB, RIS, RMED, RMEV, RMEL, RMER, DD01–DD06,
  VD01–VD13.
- **Source**:
  - McIntire SL, Jorgensen E, Kaplan J, Horvitz HR (1993). *The
    GABAergic nervous system of Caenorhabditis elegans.* **Nature**
    364: 337–341.
  - Schuske K, Beg AA, Jorgensen EM (2004). *The GABA nervous
    system in C. elegans.* **Trends Neurosci** 27(7): 407–414.
  - Gendrel M, Atlas EG, Hobert O (2016). *A cellular and
    regulatory map of the GABAergic nervous system of C. elegans.*
    **eLife** 5: e17686.
- **Evidence strength**: `direct` (immunohistochemistry, reporter
  lines, and genetic evidence in Gendrel 2016).
- **Project usage**:
  `src/algos/neurotransmitters.py::GABAERGIC`; consumed by
  `algos.connectome._from_xlsx` to set edge sign.
- **Uncertainty**: the 276 non-GABA neurons are coarsely tagged
  `'default'` (effective sign +1). Cholinergic, glutamatergic,
  monoaminergic, peptidergic distinctions are NOT encoded. For
  Phase 1.0's homogeneous excitatory treatment this is OK; for
  co-release (RIM, HSN, NSM, VC4/5) see §4 and
  `notes/edge_sign_audit.md` §3.

### 2.3 Modulator-tagged neurons

- **Current value**: 14 neurons flagged `is_modulator=True` at
  graph-load time:
  `RID, NSML/R, RICL/R, ADFL/R, HSNL/R, RIH, AVKL/R, PVT, DVA`.
- **Source**: composite from §4 entries below; primary references
  cited per modulator.
- **Evidence strength**: `direct` for producer identity; the
  promotion to `is_modulator=True` is a project tagging convention.
- **Project usage**: `src/algos/graph/loader.py::DEFAULT_MODULATOR_NEURONS`;
  read by `algos.graph.node.Node.from_connectome_row` to apply
  modulator-specific parameter defaults (longer τ, higher
  threshold).
- **Uncertainty**: **list incomplete**. Per `data/modulators_full.md`,
  missing modulator producers are: CEP/ADE/PDE (dopamine), RIML/R
  (tyramine), ALA (FLP-13), VC04/05 (5-HT+ACh co-release). Phase 1.5+
  should expand.

---

## 3. Subgraph (functional circuit) definitions

### 3.1 The 13 subgraphs in `src/algos/graph/circuits.py`

Audited individually in `notes/subgraph_audit_part1.md`. Summary:

| # | name | type | n | core citations |
|---|---|---|---:|---|
| 1 | reversal_command | recurrent | 6 | Chalfie 1985, Gray 2005, Piggott 2011 |
| 2 | forward_command | recurrent | 10 | Gray 2005, Kawano 2011, Wang 2020 |
| 3 | anterior_touch | feedforward | 7 | Chalfie 1985 |
| 4 | posterior_touch | feedforward | 9 | Chalfie 1985, Way & Chalfie 1989 |
| 5 | chemosensory_amphid | feedforward | 18 | Bargmann 2012, Tomioka 2006 |
| 6 | thermosensory | feedforward | 8 | Mori & Ohshima 1995, Hawk 2018 |
| 7 | head_motor_cpg | recurrent | 24 | Faumont 2011, Kawano 2011, Hendricks 2012 |
| 8 | pharyngeal_cpg | recurrent | 20 | Avery & Horvitz 1989, Trojanowski 2014 |
| 9 | ventral_cord_motor | recurrent | 58 | White 1986, Wen 2012 |
| 10 | modulator_RID | feedforward | 5 | Lim 2016 |
| 11 | modulator_5HT | feedforward | 9 | Sze 2000, Tanis 2008, Loer & Kenyon 1993 |
| 12 | egg_laying | feedforward | 8 | Schafer 2005, Collins 2016 |
| 13 | defecation_pacemaker | recurrent | 3 | Liu & Thomas 1994, Wang & Sieburth 2013 |

- **Evidence strength**: `direct` for membership of canonical
  neurons; `inferred` for some boundary cases (e.g. RIM's dual
  forward/reversal role).
- **Project usage**: `algos.graph.circuits.CIRCUIT_SPECS` →
  `build_canonical_subgraphs(graph)`.
- **Uncertainty**: per-circuit issues in
  `notes/subgraph_audit_part1.md`; recategorization candidates
  (PVD, RIM) and missing-member candidates (AIA, AVF, AS01–11)
  documented but not implemented.

### 3.2 Missing subgraph candidates

Per `notes/subgraph_audit_part2.md` §A:

| candidate | priority | rationale |
|---|---|---|
| inhibitory_command_gate | HIGH | RIS/ALA/AVL — root-cause fix for forward↔reversal +0.51 |
| aerotaxis | MEDIUM | URX/AQR/PQR/BAG → RID and command pool |
| sleep_quiescence | MED-LOW | ALA/RIS/AVK/PVT |
| harsh_touch_nociception | LOW | PVD/ASH/ADL recategorization |
| cross_modal_integration | LOW | AIA/AIM/AIN/RIG hubs |

- **Status**: `not_yet_implemented`. Phase 1.5+.2 deliverable was the
  identification, not the implementation.

### 3.3 Subgraph overlap topology

- **Current value**: 10 overlap pairs (cf. PHASE1.0_REPORT.md). E.g.
  reversal_command ∩ anterior_touch = {AVAL, AVAR, AVDL, AVDR};
  chemosensory_amphid ∩ thermosensory = {AIYL, AIYR, AIZL, AIZR,
  RIAL, RIAR}.
- **Source**: emerges from the membership lists; sanity-checked
  against the literature's named "shared hub" cells.
- **Evidence strength**: `direct` (set intersection of membership).
- **Project usage**: tested by
  `tests/graph/test_circuits.py::test_subgraph_overlap_nodes_share_V_across_subgraphs`.
- **Uncertainty**: none for the operation itself; biological
  consequence depends on the §3.2 missing subgraphs.

### 3.4 Subgraph "type" assignment (feedforward vs recurrent)

- **Current value**: 7 feedforward + 6 recurrent.
- **Source**: project classification based on whether the named
  circuit has documented internal feedback loops.
- **Evidence strength**: `inferred`. Even nominal "feedforward"
  circuits like chemosensory_amphid contain small feedback loops
  (e.g. AIYL↔AIYR weak gap). The classification reflects the
  *dominant* signal-flow direction.
- **Project usage**: read by `Subgraph.topological_order()` (returns
  ordering for feedforward, None for recurrent).
- **Uncertainty**: a few circuits (`posterior_touch`) include
  command-circuit nodes that are themselves recurrent; technically
  the inclusion makes the whole subgraph recurrent. Currently
  silently tolerated.

---

## 4. Modulator system

### 4.1 Wired in Phase 1.0

#### 4.1.1 RID neuropeptide (FLP-14)
- **Producers**: `{RID}` (1 neuron).
- **Targets**: `{AVBL, AVBR, PVCL, PVCR}`, sensitivity = −0.5
  (excitatory: lower threshold).
- **τ_m**: 500 ticks (project heuristic; design §7.2 says
  τ_m ≫ neuron τ ≈ 10).
- **Source**: Lim MA et al. (2016). *Neuroendocrine modulation
  sustains the C. elegans forward motor state.* **eLife** 5: e19887.
- **Evidence strength**: `direct` for producer identity; `inferred`
  for sensitivity magnitude (the −0.5 value is a project guess).
- **Project usage**: `algos.neural_v2.modulators.RID_*` constants
  +`build_default_modulator_bank()`.
- **Uncertainty**:
  - Phase 1.0 finding: c_RID = 0 across all seeds. RID is silent
    because its only upstream is PVC (also silent in the bare
    network). See PHASE1.0_REPORT.md §4.
  - The sensitivity magnitude 0.5 was not tuned, intentionally.

#### 4.1.2 Serotonin (5-HT)
- **Producers**: `{NSML, NSMR, ADFL, ADFR, HSNL, HSNR}` (6 neurons).
- **Targets (forward-suppression)**: `{AVBL, AVBR, AIBL, AIBR}`,
  sensitivity = +0.4 (raise threshold).
- **Targets (pharynx-excitation)**: `{M3L, M3R, MI, I1L, I1R}`,
  sensitivity = −0.4 (lower threshold).
- **τ_m**: 500 ticks.
- **Source**:
  - Sze JY et al. (2000). **Nature** 403: 560–564.
  - Tanis JE et al. (2008). **J Neurosci** 28(40): 10241.
  - Loer CM, Kenyon CJ (1993). **J Neurosci** 13(12): 5407.
- **Evidence strength**: `direct` for producer identity; `inferred`
  for sensitivity sign + magnitude.
- **Project usage**: `algos.neural_v2.modulators.SHT_*`.
- **Uncertainty**:
  - Producers missing: **VC04, VC05** (Schafer 2005) — vulval motor
    neurons that release 5-HT + ACh. Documented in
    `data/modulators_full.md` §1; not yet wired.
  - Phase 1.0 finding: c_5HT saturates at ≈ 0.07. Threshold
    multiplicative change is then 1 × (1 + 0.07 × 0.4) = 1.028 →
    2.8%, below noise. The signal is mechanically real but
    behaviorally negligible.

### 4.2 Documented but not wired (per `data/modulators_full.md`)

| modulator | producers | priority for Phase 1.5 |
|---|---|---|
| Dopamine | CEP, ADE, PDE (8) | HIGH (basal slowing) |
| Octopamine | RIC (2) | MEDIUM |
| Tyramine | RIM (2) | HIGH (likely fixes fwd↔rev +0.51) |
| FLP-1 | AVK, PVT, DVA | MED-LOW (sleep) |
| FLP-13 | ALA (1) | MED-LOW (stress sleep) |
| FLP-11 | RIS (peptide arm only) | MED-LOW |
| NLP-12 | DVA | DEFER (needs body) |
| INS family | AIA, ASI, ASJ | OUT OF SCOPE |

### 4.3 Cross-modulator general assumptions

- **Threshold modulation form**: `effective_threshold[i] = base[i] ×
  (1 + Σ_m c_m × sensitivity[i,m])`, clamped to `[0.1, 10] × base`.
- **Source**: design §7.3 + Phase 1.0.4 implementation choice
  (parameter-level modulation, gain control).
- **Evidence strength**: `assumed`. Multiplicative threshold scaling
  is a common abstraction; the specific functional form is a
  modeling choice, not derived from data.
- **Project usage**:
  `algos.neural_v2.modulators.ModulatorBank.apply_threshold_modulation`.
- **Uncertainty**: alternative formulations (additive bias on V;
  multiplicative gain on input; receptor-saturating kinetics) all
  have biological support. Phase 1.0's choice was made for
  consistency with design §7.3 alone.

---

## 5. Plasticity data

### 5.1 Plastic neuron set

- **Current value**: 18 plastic-source neurons; the top-100 outgoing
  chemical edges across these neurons are flagged plastic.
  Neurons: `AWCL/R, AIYL/R, AIBL/R, AIZL/R, RIAL/R, RIML/R,
  AFDL/R, PLML/R, AVDL/R`.
- **Source** (Phase 0.8 review, carried into Phase 1.0.4):
  - AWC chemosensory learning: Bargmann 2012, Tomioka 2006.
  - AIY/AIB/AIZ associative memory: Wakabayashi 2009 *PNAS* 106:
    14260; Iino 2009 *Neurosci Res* 64: 1.
  - RIA premotor integration: Hendricks 2012.
  - AFD thermal memory: Hawk 2018, Kobayashi 2016.
  - PLM mechanical habituation: Wicks & Rankin 1995 *J Neurosci*
    15: 2434.
  - AVD plasticity through touch-driven reversal: Wicks 1996.
- **Evidence strength**: `direct` for each neuron's plasticity role
  individually; `inferred` for the bundling-into-one-set treatment.
- **Project usage**:
  `algos.neural_v2.plasticity.DEFAULT_PLASTIC_PRE_NEURONS`.
- **Uncertainty**: 100-edge cap is engineering; biology has at least
  several hundred candidate-plastic synapses (every AWC-AIY contact,
  every AVD-AVA contact, etc.).

### 5.2 Hebbian rule parameters

- **Current value**: η (learning rate) = 5e-4; λ (decay) = 5e-5;
  w_min = 0.0, w_max = 1.0 per edge.
- **Source**: chosen to make plasticity slow relative to the spike
  rate (10²–10³ ticks for first observable change). No direct
  biological derivation.
- **Evidence strength**: `assumed`.
- **Project usage**: `algos.neural_v2.plasticity.DEFAULT_ETA`,
  `DEFAULT_LAMBDA`.
- **Uncertainty**: the values were chosen so the steady-state
  weight (`pre × post = λ/η = 0.1`) is moderate; any value that
  yields slow change without instability would work. Phase 1.0
  finding: weights mostly shrank (92/100 in seed=1000), suggesting
  the network's mean coactivation × η is generally below λ × W.

### 5.3 Drive signal for Hebbian

- **Current value**: leaky-integrator rate trace, not raw V.
- **Source**: project decision (DECISIONS.md Phase 1.0.4). Raw V in
  LIF is mostly 0 or v_reset, signal-starved for Hebbian.
- **Evidence strength**: `assumed`.
- **Project usage**: `HebbianRule.step(rate)`.
- **Uncertainty**: biologically, Hebbian-like rules operate on
  recent spike rate at both pre and post, which the rate trace
  captures well. The 30-tick filter (`DEFAULT_TAU_RATE`) is short;
  longer (~100 tick) might better match GCaMP / functional-rate
  observations.

---

## 6. Reference recordings

### 6.1 Calcium-imaging dataset

- **Current value**: 10 best-labeled hermaphrodite recordings from
  Atanas 2023, accessed via
  `algos.validation.reference_data.ReferenceDataset.from_atanas2023`.
- **Source**: Atanas AA, Kim J, Wang Z et al. (2023). *Brain-wide
  representations of behavior spanning multiple timescales and
  states in C. elegans.* **Cell** 186(19): 4134–4151.e31.
  doi:10.1016/j.cell.2023.07.035.
- **Evidence strength**: `direct` (whole-brain GCaMP6 imaging at
  ~3 Hz; co-imaged with behavior).
- **Project usage**: comparison target for all three Phase 0.7+
  metrics (subspace_alignment, temporal_correlation, fc_similarity);
  consumed by `scripts/run_phase1_comparison.py` etc.
- **Uncertainty**:
  - Labeled-neuron count per recording is 90–113 (selected by
    `from_atanas2023(max_recordings=10)` for best label coverage).
  - GCaMP is a slow Ca²⁺ indicator (~1 s rise / 2-3 s decay) —
    matches our `rate` trace concept (leaky integrator on spikes)
    but the exact temporal kernel differs.
  - Worm behavior in those 10 recordings is mostly free-moving on
    food; their structure reflects food-state biology, not
    naive/starved.

### 6.2 Atanas 2023 behavioral covariates

- **Current value**: per-tick velocity and reversal binary indicator
  available in `Recording.velocity` and `.reversal`.
- **Source**: same as §6.1.
- **Evidence strength**: `direct`.
- **Project usage**: **not yet consumed** by Phase 1.0 metrics.
  Phase 1.5 design should use these as ground-truth for matching
  behavioral states (forward/reverse periods).
- **Uncertainty**: behavioral classification by Atanas is at video
  frame rate; aligning to neural ticks needs careful time-base
  matching.

---

## 7. Known data gaps and engineering assumptions

> The honest list. Anything below is a place where the project uses
> a value/treatment that *will* matter for some downstream analysis
> but is not yet biology-validated. Phase 1.5 design should not
> assume any of these is "settled".

### 7.1 Gap-junction rectification (HIGH impact)

- Symmetric assumption (§1.5); known violations on AVA↔A-class,
  AVB↔B-class, several pharyngeal. Likely contributor to Phase 1.0
  forward↔reversal +0.51 anomaly.
- See QUESTIONS.md Q-1.5+.1.

### 7.2 RIM tyramine arm (HIGH impact)

- RIM's tyramine release (Pirri 2009) not modeled. Adding it as
  either fast (sign=−1 chemical edges to AVB/MC/RMD) or slow
  (Modulator entry) would automatically introduce the forward-
  suppression mechanism Phase 1.0 lacks.
- See QUESTIONS.md Q-1.5+.2 and Q-1.5+.5.

### 7.3 Sensory→Modulator bridges (HIGH impact)

- BAG → RID (3+3 contacts), URX → RIC (8 contacts) — these provide
  the *external* drive for RID and octopamine that the bare network
  cannot generate. Not currently wired in the Modulator framework.

### 7.4 Modulator sensitivity magnitudes (MEDIUM impact)

- RID sensitivity = −0.5, 5-HT sensitivity = ±0.4 — both project
  guesses (`assumed` evidence strength). No literature direct
  measurement of these in mV-per-c_m units.

### 7.5 Missing inhibitory command gate (HIGH impact)

- RIS/ALA/AVL not in any subgraph; ALA not even in
  `DEFAULT_MODULATOR_NEURONS`. See
  `notes/subgraph_audit_part2.md` §A.1.

### 7.6 Co-release accounting (MEDIUM impact)

- HSN (ACh + 5-HT): handled correctly via dual-system split.
- NSM (5-HT + NLP-3 + FLP-21): merged into 5-HT only.
- VC4/5 (ACh + 5-HT): 5-HT arm missing from `SHT_SOURCE_NEURONS`.
- RIM (Glu + tyramine): tyramine arm missing (§7.2).
- AVL (GABA + ACh): ACh arm not modeled (acceptable for current
  scope).
- See QUESTIONS.md Q-1.5+.7.

### 7.7 Excitatory-GABA via EXP-1 (LOW impact)

- AVL/DVB → enteric muscle is excitatory via EXP-1 (Beg 2003).
  Enteric muscle is out of 302; no impact until Phase 2+ adds
  muscle cells.

### 7.8 LIF parameter heterogeneity (MEDIUM impact)

- Per-category τ and threshold defaults
  (`algos.graph.node.CATEGORY_PARAM_DEFAULTS`) chosen by hand;
  modulator neurons get τ=40, others τ=5-12. No direct biology for
  these specific numbers.
- Project usage: `Node.from_connectome_row()`.

### 7.9 Chemical edge delay (MEDIUM impact)

- All chemical edges have `delay=1` tick (DEFAULT_CHEMICAL_DELAY).
  Real synapses have heterogeneous delays (axon length × ~0.5 m/s
  conduction velocity; sub-ms to ~10 ms in worm).
- Project usage: `algos.graph.edge.DEFAULT_CHEMICAL_DELAY`.
- The `SignalQueue.DelayBucket` machinery scales to per-edge
  delays; only the loader heuristic uses uniform 1.

### 7.10 Sensory neuron coupling to environment (BLOCKING for Phase 1.5)

- The 83 sensory neurons are driven by random Gaussian noise in
  every Phase 1.0 experiment (`SimulatorConfig.sensory_noise =
  0.2`). Real biology: ASE responds to NaCl in mM ranges, AFD to
  temperature with sub-degree precision, AWC to volatile attractants
  with parts-per-billion sensitivity. **Phase 1.5's central
  contribution** is replacing the noise with environment-driven
  inputs.

### 7.11 Self-gap entries (LOW impact)

- 14 nonzero diagonal gap entries zeroed at load — algebraically
  cancel; data interpretation unclear (see QUESTIONS.md Q-1.5+.3).

### 7.12 Receptor antagonism on common targets (MEDIUM impact)

- DOP-1 (excitatory) and DOP-3 (inhibitory) on overlapping
  dopaminergic targets — Phase 1.0's Modulator framework only
  records one sensitivity per (modulator, target). See QUESTIONS.md
  Q-1.5+.6.

### 7.13 Normalization mode unproven for spiking dynamics (LOW impact)

- Phase 0 chose `per_row` L1 normalization to keep CTRNN inputs
  O(1). Phase 1.0 inherits this for spiking LIF, which has
  different stability requirements. Empirically stable (10⁴ ticks,
  no NaN); no theoretical guarantee.

### 7.14 Rate trace as GCaMP analogue (LOW impact)

- Leaky integrator (τ = 30 ticks) on spike indicator. GCaMP has a
  more complex temporal kernel (rise ~50 ms, decay ~1 s, also
  saturates non-linearly above ~0.5 ΔF/F).

### 7.15 Plastic edge top-N selection (LOW impact)

- 100-edge cap chosen by initial weight magnitude. Real plastic
  edges are not specifically the strongest; many learning-relevant
  contacts are weak.

---

## 8. How to use this audit

When making any biology-touching change to the simulator:

1. **First** check this document for the relevant data point.
2. If the data point has `assumed` evidence strength, that change is
   a candidate for *also* triggering a re-audit of the related
   biology (file in QUESTIONS.md if you can't resolve).
3. If a new dataset or paper supersedes a cited source, update this
   document and bump the DECISIONS.md.

This audit will be updated by Phase 1.5+.5 with cross-references to
the Phase 1.5 design document. After Phase 1.5, any biology
modification (new modulator, new edge classification) must update the
corresponding §3–§5 entry here as part of the change.

---

## 9. Statistics summary

- **Audited data points** above the §7 line: **30** (1.1-1.6, 2.1-2.3,
  3.1-3.4 individually, 4.1.1, 4.1.2, 4.2, 4.3, 5.1-5.3, 6.1, 6.2).
- **Known gaps in §7**: **15** entries.
- **Forward references to QUESTIONS.md**: 6 (Q-1.5+.1 through .7,
  minus .3 which is informational).
- **Modulator entries in `data/modulators_full.md` not yet wired**:
  8 of 10 documented families.
- **Missing subgraph candidates in `notes/subgraph_audit_part2.md`**:
  5.

This document is canonical for the data points it covers. Updates
must come through DECISIONS.md.
