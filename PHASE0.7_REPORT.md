# Phase 0.7 — Individual-variation baseline

> Generated: 2026-05-20
> Question being asked: how similar are real C. elegans worms to one
> another on the metrics we use to grade our digital worm? Without
> this baseline, the digital-vs-real scores from Phase 0.5/0.6 can't
> be interpreted.
> Method: pairwise comparisons across N=10 best-labeled Atanas 2023
> recordings (45 cross-pairs + 10 within-worm split-halves) +
> digital-vs-real on the same 10. All distributions reported with
> bootstrap 95% CIs.

---

## 1. Setup

| | |
|---|---|
| recordings | 10 best-NeuroPAL-labeled (92-113 high-confidence labels each) |
| cross-pairs | (10 choose 2) = 45 |
| within-worm | each recording split at midpoint (first half vs second half) |
| digital | one bare-CTRNN run per recording, length-matched, protocol A (random sensory noise) |
| bootstrap | B = 1000 resamples on mean, p5, p95 |
| metrics | `subspace_alignment` (Phase 0.6 validated), `temporal_correlation`, `fc_similarity` |
| runtime | 4.3 s end-to-end |

Recordings used:

| recording_id | T (frames) | high-conf labels |
|---|---:|---:|
| 2022-08-02-01 | 1600 | 113 |
| 2023-01-17-14 | 1615 | 106 |
| 2023-01-09-15 | 1615 |  99 |
| 2023-01-10-07 | 1615 |  97 |
| 2023-01-23-15 | 1600 |  96 |
| 2022-06-14-13 | 1600 |  94 |
| 2022-07-20-01 | 1600 |  94 |
| 2023-01-09-22 | 1615 |  94 |
| 2023-01-23-21 | 1600 |  94 |
| 2023-01-09-28 | 1615 |  92 |

---

## 2. The three distributions, side by side

Each row: a metric. Each column: a comparison condition. All values
mean ± std, with bootstrap 95% CI on the mean, and 5%/95% percentiles.

### 2.1 `subspace_alignment` — alignment of top-10 PCA subspaces

| condition | n | mean ± std | 95% CI (mean) | p5 — p95 |
|---|---:|---:|---|---|
| within-worm split-half | 10 | **+0.658 ± 0.033** | [+0.634, +0.677] | +0.604 — +0.697 |
| cross-worm pairs | 45 | **+0.593 ± 0.032** | [+0.583, +0.602] | +0.537 — +0.637 |
| digital vs real | 10 | **+0.399 ± 0.021** | [+0.386, +0.412] | +0.371 — +0.429 |

- **Within > cross > digital**, as one would hope. Within-worm
  similarity 0.658 is the empirical ceiling — what the same brain in
  motion looks like to itself.
- The cross-worm baseline (0.593) is **only 0.066 below** the
  within-worm ceiling. Different worms' top-10 PCA subspaces are
  almost as aligned with each other as a single worm is with its own
  later half. **C. elegans neural dynamics are highly stereotyped
  across individuals.**
- The digital worm sits at the **0th percentile** of the cross-worm
  distribution. Gap to the cross-worm 5%-tile = 0.139. The digital
  worm does not yet enter the real-worm distribution.

### 2.2 `temporal_correlation` — mean per-neuron Pearson r

| condition | n | mean ± std | 95% CI (mean) | p5 — p95 |
|---|---:|---:|---|---|
| within-worm split-half | 10 | **+0.035 ± 0.034** | [+0.014, +0.055] | −0.013 — +0.079 |
| cross-worm pairs | 45 | **+0.122 ± 0.045** | [+0.109, +0.135] | +0.049 — +0.192 |
| digital vs real | 10 | **−0.008 ± 0.020** | [−0.021, +0.004] | −0.040 — +0.018 |

- **Counter-intuitive: cross-worm > within-worm.** A single
  recording's first-half-vs-second-half temporal correlation (0.035)
  is *lower* than two different worms' correlation (0.122). Why?
  Within-worm: two consecutive epochs of the same trajectory — likely
  different behavior states (e.g. forward in half 1, reversal in
  half 2). Cross-worm: each recording averages over comparable
  behavior-state mixtures, so per-neuron means align. The metric is
  effectively dominated by behavior-state averages, not by trajectory
  detail.
- Phase 0.5 said "temporal_correlation ≈ 0 → no shared sensory
  history". Phase 0.7 says: even within the **same** worm split in
  two, the temporal correlation is only +0.035. The "≈ 0" finding
  was about right in absolute terms but was being compared to an
  imaginary ceiling of "should be high"; the real ceiling is also
  near zero.
- Digital still falls below: at −0.008, the digital worm is below
  even the within-worm baseline. Gap to cross-worm p5 (+0.049) =
  0.058.

### 2.3 `fc_similarity` — Pearson r of FC-matrix upper triangles

| condition | n | mean ± std | 95% CI (mean) | p5 — p95 |
|---|---:|---:|---|---|
| within-worm split-half | 10 | **+0.614 ± 0.094** | [+0.545, +0.668] | +0.463 — +0.718 |
| cross-worm pairs | 45 | **+0.478 ± 0.084** | [+0.454, +0.504] | +0.348 — +0.605 |
| digital vs real | 10 | **+0.030 ± 0.031** | [+0.013, +0.049] | −0.017 — +0.075 |

- Cross-worm FC similarity is **+0.478** — different real worms
  share substantial functional-connectivity structure. Within-worm
  ceiling is +0.614, only ~0.14 higher. Functional connectivity is
  *another* highly-stereotyped feature of C. elegans.
- Digital worm at **+0.030** is at the 0th percentile. The gap to
  the cross-worm 5%-tile (+0.348) is **+0.318** — by far the largest
  digital-real gap of any metric.
- This is the *load-bearing* gap. If Phase 1+ wants to ship a digital
  worm whose activity resembles a real worm, this is the metric to
  watch.

---

## 3. Where does the digital worm sit?

For each metric, the digital worm's mean score and its empirical
percentile in each baseline distribution:

| metric | digital mean | cross-worm mean | within-worm mean | percentile in cross-worm | gap to cross-worm mean |
|---|---:|---:|---:|---:|---:|
| `subspace_alignment` | +0.399 | +0.593 | +0.658 | **0.0%** | −0.194 |
| `temporal_correlation` | −0.008 | +0.122 | +0.035 | **0.0%** | −0.130 |
| `fc_similarity` | +0.030 | +0.478 | +0.614 | **0.0%** | −0.448 |

The digital worm is at the **0th percentile** of the cross-worm
distribution on every metric. No digital trial falls inside the
real-worm distribution.

This rules out one optimistic interpretation of Phase 0.5: that the
bare connectome was "starting to look like a worm". It is not. On the
correct baseline (real C. elegans variability), the digital worm is
clearly outside.

It does **not** rule out the other Phase 0.5 finding: the
connectome's structure is significantly better than a sparsity-matched
random null (Phase 0.6: subspace_alignment 0.38 real vs 0.28 null).
That gap is real. The point is just that ~0.38 above the random-null
floor still doesn't reach ~0.59 of the real-worm floor.

---

## 4. Project-level finding: real-worm individual variation is small

This was the question the brief asked to flag if it came up. It did.

- **Cross-worm subspace_alignment 0.59 vs within-worm 0.66** — only
  ~0.07 difference between "different worms" and "same worm at
  different times". The 95% CIs do not quite overlap (cross-worm
  [+0.583, +0.602] vs within-worm [+0.634, +0.677]) but the gap is
  small.
- **Cross-worm fc_similarity 0.48 vs within-worm 0.61** — gap of
  ~0.14. CIs do not overlap (cross [+0.454, +0.504] vs within
  [+0.545, +0.668]) but again the gap is moderate, not large.

**Implication for the project.** The Phase 0 design doc §1.2 framed
the digital worm as an instance with its own individuality — "每只
数字虫子是一个独立个体, 有自己的一生". This is still true at the
philosophical level (each digital worm dies and is not backed up),
but the *neural-activity individual variation* in real C. elegans
seems surprisingly small. A reasonable Phase 1 hypothesis: most of
the apparent individuality in C. elegans behavior comes from
*physical-state contingency* (where the worm is, what it just ate,
mechanical history of the body) and *random initial conditions*,
rather than from neural-circuit-level differences between
individuals.

This affects validation strategy:
1. Future "did the digital worm reproduce C. elegans neural
   activity?" claims should compare against the **cross-worm**
   baseline, not against a *single* recording. The within-worm
   baseline is too tight; the random-null is too loose; the
   cross-worm distribution is the right reference.
2. When grading whether a Phase 1+ change "moved the needle", the
   relevant comparison is "did we move into the cross-worm
   distribution?" not "did we improve by 0.05 over the previous
   version". The bar is high but well-defined.

---

## 5. Where Phase 1 should land

Three concrete numbers Phase 1 can aim at:

| metric | required to enter cross-worm distribution (p5) | digital current | gap |
|---|---:|---:|---:|
| `subspace_alignment` | +0.537 | +0.399 | **−0.139** |
| `fc_similarity` | +0.348 | +0.030 | **−0.318** |
| `temporal_correlation` | +0.049 | −0.008 | −0.058 |

`temporal_correlation` will be hard to move much because it requires
the digital worm to see *correlated sensory input* — and even
within-worm temporal correlation is only 0.035, so an ambitious
Phase 1 target is +0.04 to enter the distribution. Easier than it
sounds because real cross-worm p5 sits at +0.05.

`subspace_alignment` and `fc_similarity` are the two real targets.
The biggest single change Phase 1 could make on these would be
introducing **state-conditioned activity** — a body that has
metabolic states, a sensory translator that produces correlated
multi-neuron input rather than independent Gaussian noise. Phase 0.5
already showed that even *behavior-conditioned* command drive only
nudges fc_similarity from +0.02 to +0.06 — the rest of the gap
requires modulators (Phase 3) and likely plasticity (Phase 4) too.

---

## 6. Honest issues

- **N=10 is small for cross-worm baselines.** 45 pairs gives tight
  CIs on the mean (±0.01 on subspace_alignment), but the 5% and 95%
  percentiles are estimated from very few points; their bootstrap CIs
  are visibly wider. Phase 1 could re-run with N=20 (190 pairs) for
  more stable tails.
- **Within-worm split-half is a coarse proxy** for "same brain at
  different times". It captures sequential trajectory differences but
  not true epoch matching. A cleaner test would split by behavior
  state (forward vs reverse) and measure how similar a real worm's
  forward neurons are to its reverse neurons. Out of scope here.
- **Recordings come from different days and Atanas-lab sessions** —
  setup drift could inflate cross-worm differences. Probably minor,
  given how close cross- and within-worm scores are; if anything this
  suggests the real cross-worm baseline is even tighter than reported.
- **The temporal_correlation reading is dominated by behavior-state
  averages**, not by trajectory shape. A more behavior-aware metric
  (e.g. correlation conditioned on reversal state) would tell us
  more.
- The digital-vs-real numbers use Phase 0.5 protocol A (random
  sensory noise). Phase 0.5 also showed protocol B
  (behavior-conditioned drive) gives marginally better numbers; the
  qualitative conclusion (digital at 0th percentile) holds for both
  protocols.

---

## 7. Single-sentence summary

The cross-worm baseline on three Phase 0.5 metrics is *high*
(subspace_alignment ≈ 0.59, fc_similarity ≈ 0.48, both substantially
above the random-null and only ~0.1 below the within-worm ceiling) —
real C. elegans neural activity is strikingly stereotyped across
individuals, and the digital worm currently sits at the 0th
percentile of every cross-worm distribution: **outside** the real-worm
range on every metric, with the biggest gap on fc_similarity (+0.318
below the 5%-tile).

---

## 8. Generated artifacts

```
scripts/run_individual_variation.py
output/individual_variation_report.txt
output/individual_variation_results.json
PHASE0.7_REPORT.md                    # this file
```

---

*Last updated: 2026-05-20*
*Status: Phase 0.7 complete; baseline established for Phase 1.*
