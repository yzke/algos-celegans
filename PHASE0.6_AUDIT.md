# Phase 0.6 — Internal audit of the PCA-structure similarity score

> Generated: 2026-05-20
> Question being asked: the Phase 0.5 report claims PCA-structure
> similarity ≈ 0.65 (digital ↔ real Atanas 2023, n=6 recordings).
> Is this a real signal about the C. elegans connectome, or an artifact
> of how the metric is defined / a value any reasonably-tuned network
> would produce?
> Method: independent re-derivation, 3 controls, 50-seed null distribution.

---

## 1. The metric, exactly

### 1.1 Input matrices

Two activity matrices go into the comparison after the
`match_matrices()` step (`src/algos/validation/comparison.py`):

| name | source | shape | content |
|---|---|---|---|
| `D` | digital simulation (`src/algos/validation/comparison.py`) | `(T_digital, N)` | `state.V[t]` over 1600 ticks across the 302 neurons, then sliced to the columns that are present in the real recording |
| `R` | real Atanas 2023 recording (`gcamp/traces_array_F_Fmean`) | `(T_real, N)` | ΔF/F traces over the same recording, sliced to the same `N` neurons that are labeled and matched by name |

`N` is the count of neurons that appear in *both* matrices — the
intersection of (a) all 302 neuron names in our connectome and (b) the
high-confidence NeuroPAL labels in the recording. For the
best-annotated recording (2022-08-02-01) that intersection is ~80–100
neurons.

`T_digital = 1600` ticks (one per real frame, with the digital sim
pre-equilibrated for 2000 ticks first). `T_real` is the recording's
frame count (1600 or 1615 depending on session). Phase 0.5's protocol
A uses random Gaussian sensory drive at every sensory-tagged neuron;
the digital sim does not see the real worm's behavior.

### 1.2 The pipeline

```
match_matrices(D_full, names_digital, R_full, names_real)
    → D : (T_digital, N)
    → R : (T_real, N)

# Per-neuron z-score along time (column-wise):
Dz = (D - mean(D, axis=0)) / std(D, axis=0)
Rz = (R - mean(R, axis=0)) / std(R, axis=0)

# Thin SVD on each:
_, S_d, Vt_d = np.linalg.svd(Dz, full_matrices=False)   # Vt_d: (N, N)
_, S_r, Vt_r = np.linalg.svd(Rz, full_matrices=False)

# Top K = min(10, N-1) right-singular vectors define the PC subspace.
var_d = (S_d**2)[:K] / sum(S_d**2)        # explained-variance ratios
var_r = (S_r**2)[:K] / sum(S_r**2)

# Component 1: spectrum-shape similarity
explained_variance_cos = dot(var_d, var_r) / (norm(var_d) * norm(var_r))

# Component 2: subspace alignment
overlap = Vt_d[:K] @ Vt_r[:K].T            # (K, K), each entry an inner
                                           # product of one PC of D with
                                           # one PC of R
s = svd(overlap, no UV)                    # singular values in [0, 1]
subspace_alignment = mean(s)

# Final score:
pca_similarity = 0.5 * (explained_variance_cos + subspace_alignment)
```

### 1.3 Two components, separately

The `pca_similarity` reported in the Phase 0.5 report (and the JSON)
is the **mean of two scalars** that measure different things:

1. **`explained_variance_cos`** — cosine of two length-K vectors of
   *explained-variance ratios*. Range `[-1, 1]`. Two completely
   uninformative spectra (both flat) would give cosine = 1. A real-but-
   shifted spectrum (e.g. digital eigenvalues drop slower than real)
   could give cosine ~ 0.95 even when the subspaces are orthogonal.
   This component is **insensitive to whether the PCs point the same
   way**.

2. **`subspace_alignment`** — mean singular value of the cross-basis
   inner-product matrix `Vt_d[:K] @ Vt_r[:K].T`. Range `[0, 1]`. This
   IS the principal-angle measure of how much the K-dimensional PCA
   subspaces of D and R overlap. If the digital and real PCs point in
   the same K-dim subspace (regardless of permutation or sign),
   alignment ≈ 1. If the subspaces are orthogonal, alignment = 0.

Averaging them was a choice in `pca_structure_similarity()`. Two
remarks worth recording:

- The mean is computationally clean but **mixes two quantities with
  different baselines under the null** (component 1 has a much higher
  noise floor than component 2 because flat spectra are common).
- The Phase 0.5 report acknowledged this ambiguity in §6 ("PCA-similarity
  metric averages two pieces"). Phase 0.6 audits both components
  separately and reports the geometric implication.

### 1.4 Time-axis alignment

Both `D` and `R` are passed to SVD column-wise (each *neuron* is a
variable, each *frame* is a sample). The metric does **not** assume
frame-by-frame temporal correspondence between digital and real — it
only uses each matrix's empirical covariance structure across time.
This is intentional: there is no biological reason a randomly-driven
digital sim should be time-aligned to a freely-moving worm.

The `temporal_correlation` metric (separate in Phase 0.5) *does* try
to align frame-by-frame (with linear resampling when lengths differ);
that metric is essentially zero in Phase 0.5 (mean +0.01) — confirming
that no per-frame correspondence exists. Phase 0.6 focuses only on the
PCA metric.

---

## 2. Controls

Three controls each producing one PCA-similarity number (or, for the
random-shuffle case, a distribution).

### 2.1 (a) Random-shuffled connectome

- **Intervention.** Replace `connectome.W_chem` with a matrix that has
  the same number of non-zero entries placed at uniformly-random
  positions, with the original non-zero *values* permuted into those
  positions. Same for `W_gap`, preserving symmetry by sampling
  upper-triangle positions only and mirroring. Diagonals zeroed
  (consistent with the loader convention).
- **Hypothesis.** A shuffled connectome destroys all biological
  topology but preserves the marginal degree/weight distributions and
  the per-row-L1-normalization invariant.
- **Statistical use.** 50 different shuffle seeds → null distribution
  for "what PCA-similarity does a random connectome produce?". Sim
  seed varies in parallel (each trial has its own seed) so the noise
  floor combines both connectome-shuffle and sim-noise variability —
  a stronger null than fixing the sim seed.

### 2.2 (b) Transposed connectome

- **Intervention.** Replace `W_chem` with `W_chem.T` (swap pre/post).
  `W_gap` is symmetric, so transpose is a no-op there.
- **Hypothesis.** If the directionality of synapses matters, the
  transposed connectome should produce different PCA structure than
  the real one. Note: the per-row-L1 normalization is recomputed on
  the transposed matrix so the row sums stay bounded.
- **Significance for the audit.** A clean degradation under
  transposition would say "the directional structure of the
  connectome matters". No degradation would say "PCA structure
  reflects sparsity / weight distribution / undirected topology
  only".

### 2.3 (c) ReLU activation on the unchanged connectome

- **Intervention.** Replace `chem_input = W_chem @ tanh(β · V)` with
  `chem_input = W_chem @ relu(V) = W_chem @ max(V, 0)`. Same network,
  same connectome, different non-linearity. β has no role for ReLU.
- **Hypothesis.** If the PCA-similarity score reflects something
  fundamental about how the connectome routes activity (independent of
  the specific saturating shape), ReLU should give a similar score.
  If ReLU gives a very different number, the score depends on the
  precise tanh choice (and is therefore less generalizable than
  Phase 0.5 implied).
- **Caveat.** ReLU dynamics on the same connectome can saturate
  differently. We monitor `max|V|` during each run and discard trials
  where the network saturates (`max|V| > 0.95` for >50% of frames).

---

## 3. Statistical test

The headline number reported in Phase 0.5 is 0.65 on protocol A
(random sensory drive). For Phase 0.6 we use a **single recording**
(`2022-08-02-01` — 113 high-confidence labels, the most data) and:

1. **Reference distribution (true connectome).** 50 trials with the
   real Cook 2019 connectome + v0.3 tanh dynamics, each with a
   different sim seed. → Distribution `H_real`.

2. **Null distribution (random shuffle).** 50 trials with a freshly
   shuffled connectome + v0.3 tanh dynamics, each trial has its own
   shuffle-seed and sim-seed. → Distribution `H_shuffle`.

3. **Point estimates.** Transposed connectome: one trial per sim seed,
   10 trials. ReLU activation: same 10 trials.

**Significance.** We report:
- Mean and 95% interval (2.5–97.5 percentile) of `H_real` and
  `H_shuffle`.
- The fraction of `H_shuffle` that exceeds the mean of `H_real`.
- The two components (`explained_variance_cos`, `subspace_alignment`)
  reported separately, not just the average.

The audit passes if (`mean(H_real) > 97.5%-tile of H_shuffle`); fails
if there is substantial overlap between the distributions.

---

## 4. Results

Recording: **2022-08-02-01** (113 high-confidence labels, T=1600).
Per-trial cost ≈ 0.32 s; total wall time **32.4 s** for 120 trials.

### 4.1 Combined PCA similarity (mean of two components)

| condition  | n  | mean   | std   | 95% CI         |
|---|---:|---:|---:|---|
| real       | 50 | **+0.637** | 0.018 | [+0.600, +0.667] |
| shuffle    | 50 | +0.617     | 0.010 | [+0.600, +0.635] |
| transpose  | 10 | +0.657     | 0.010 | [+0.638, +0.669] |
| relu       | 10 | +0.648     | 0.014 | [+0.622, +0.666] |

The Phase 0.5 headline number (0.65, n=6 recordings) lands inside the
real-connectome distribution observed here (mean 0.637 ± 0.018 on a
single recording). It is reproducible across sim seeds.

But:

- **The shuffled-connectome null distribution centers at 0.617** — only
  0.020 below the real-connectome mean.
- The two distributions' 95% confidence intervals **overlap**:
  real `[+0.600, +0.667]` vs shuffle `[+0.600, +0.635]`.
- `frac(shuffle ≥ real_mean) = 0.02`. Technically below the 5%
  threshold, but the *effect size is tiny relative to the absolute
  score*.
- **Transposing the connectome makes the score go up**, not down (0.657
  vs 0.637 real). If directional structure were what the metric
  measured, transpose should hurt — it doesn't.
- **Swapping tanh for ReLU barely moves the score** (0.648 vs 0.637).
  The combined metric is essentially insensitive to the specific
  nonlinearity.

The combined score `~0.65` does **not cleanly separate real from
control** for any of the three perturbations.

### 4.2 Component breakdown — where the artifact lives

| condition  | `explained_variance_cos` | `subspace_alignment` |
|---|---:|---:|
| real       | +0.890 ± 0.031   | **+0.383 ± 0.016** |
| shuffle    | +0.949 ± 0.008   | +0.285 ± 0.019   |
| transpose  | +0.955 ± 0.008   | +0.358 ± 0.016   |
| relu       | +0.906 ± 0.027   | +0.389 ± 0.013   |

The components disagree, sharply:

- **`explained_variance_cos` is HIGHER under the shuffled connectome**
  (0.949) than under the real one (0.890). Reason: a random network's
  PCA spectrum is more uniformly distributed than the real worm's
  spectrum, so cosine to a real ΔF/F spectrum (which is also fairly
  uniform across the top-10 components) gives a higher value than
  the real connectome (whose spectrum drops off faster). **The
  spectrum-cosine component is a noise floor near 0.9 that the real
  connectome scores *worse* on.**

- **`subspace_alignment` shows the real signal**: 0.383 (real) vs
  0.285 (shuffle) — a +0.098 effect, ~5σ above shuffle's spread. This
  is the *only* component that picks up genuine connectome structure.

The combined `0.5 * (ev_cos + align)` averages a +0.098 real signal
with a −0.059 anti-signal, producing the misleading +0.020 net effect.

### 4.3 Per-condition distributions

JSON: `output/pca_audit_results.json` contains every trial. Histogram-
worthy quick reads:

- **Real vs shuffle on `subspace_alignment` only**: clean separation.
  Real range [+0.355, +0.413]; shuffle range [+0.241, +0.328]. No
  overlap.
- **Real vs transpose on `subspace_alignment`**: real (0.383) >
  transpose (0.358) by 0.025 — modest evidence that directionality
  matters for the alignment component.
- **ReLU on `subspace_alignment`**: 0.389, essentially identical to
  tanh. The chemical activation choice barely affects the signal
  this metric picks up.

---

## 5. Conclusion

**The 0.65 figure in Phase 0.5 was misleading.**

It is reproducible (the metric, as defined, does give ~0.65 on the
real connectome and ~0.62 on a random one) but **most of it is a
metric-construction artifact**, not evidence about the connectome.

What actually held up:

- ✅ **`subspace_alignment` is a real signal.** The top-10 PCA
  subspace of the digitally-simulated bare CTRNN aligns with the
  real worm's top-10 subspace at ~0.38, versus ~0.28 under a random
  connectome with matched sparsity. That ~0.10 gap is robust across
  50 seeds and survives the transpose control with a smaller but
  positive effect (~0.025). This is the **defensible** finding.

- ❌ **`explained_variance_cos` is not a real signal.** The PCA
  spectrum of our digitally-simulated bare CTRNN, regardless of
  whether the connectome is real or shuffled, is approximately as
  flat as the real worm's. The cosine of two similarly-flat vectors
  is mechanically near 1.0. This component contributed +0.45
  (half of 0.89/0.95) to the published 0.65 *without measuring
  anything about the connectome at all*.

- ❌ **The averaged "PCA structure similarity" metric should not be
  used as published.** It compresses a real +0.10 effect and a
  spurious +0.95 floor into one number that looks impressive but
  measures mostly metric construction.

- ❌ **Insensitivity to non-linearity.** The combined metric gives
  almost the same value with ReLU as with tanh. While that suggests
  the underlying signal is structural rather than activation-specific
  (a positive interpretation), it also further confirms that the
  metric is not very discerning.

- ❓ **Transposed connectome scoring slightly *higher*** on the
  combined metric (0.657) is a cautionary flag. On the
  `subspace_alignment` sub-component, real (0.383) does edge out
  transpose (0.358), so directionality is *very modestly* preserved
  by that sub-component alone.

### Recommendations for Phase 1

1. **Replace the combined PCA-structure metric with
   `subspace_alignment` alone** going forward. Report
   `explained_variance_cos` separately as a sanity check, not as
   part of the headline.

2. **Revise `PHASE0.5_REPORT.md`** to clarify that the 0.65 figure
   is mostly artifact — and that the substantive remaining claim is
   "top-10 PCA subspace alignment ≈ 0.38 vs ≈ 0.28 null". This is
   still a positive finding about the connectome (~3.5× the null
   gap, ~5σ above null spread), just much more modest than the
   original framing.

3. **Use the proper null distribution from now on.** Any future
   similarity metric for digital ↔ real comparison should be reported
   alongside (a) a shuffled-connectome null and (b) a per-trial
   variance estimate. The audit script
   `scripts/run_pca_audit.py` is the template.

4. **Do not re-run Phase 0.5's similarity claims as evidence for
   "the connectome is right"** without re-deriving them on the
   subspace-alignment component. The combined-metric claim cannot
   support that load.

### Honest acknowledgement

The Phase 0.5 report did flag the combined-metric ambiguity in §6
("PCA-similarity metric averages two pieces"), but did not
quantify how much of the score came from the noise-floor component.
Phase 0.6 quantifies it: roughly half. The original report should
be updated.

---

*Last updated: 2026-05-20*
*Verdict: combined PCA-similarity = 0.65 is mostly metric construction.
The subspace-alignment sub-component (~0.38 real vs ~0.28 null) is the
defensible signal. Roughly half of the original 0.65 was noise floor.*
