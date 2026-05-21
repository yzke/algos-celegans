# Phase 1.0 — Findings

> Concise milestone record. Independent of `PHASE1.0_REPORT.md` (which
> is the full per-sub-phase narrative). This file captures only what
> the project should "formally remember" — the durable claims and
> the durable refutations, in a form a future Phase can cite.

---

## 1. What changed (one paragraph)

Phase 0 used 302×302 NumPy matrices with a CTRNN update; Phase 1.0
replaced the underlying representation with a NetworkX MultiDiGraph
of `Node` + `Edge` objects, swapped the update rule to leaky
integrate-and-fire with refractory periods, made spike propagation
event-driven through a per-delay `SignalQueue`, added 13 named
functional subgraphs as views onto the parent graph, and wired Hebbian
plasticity on 100 edges + two slow modulators (RID, 5-HT) with
parameter-level (threshold) modulation. All five sub-phases shipped
as independent commits with 56 new tests and 0 regressions on the
Phase 0 suite.

---

## 2. The one durable result

**Anti-correlation went from 0% to 9% at FC < −0.10, with no parameter
tuning.**

This is the project's first irreversible win. Every Phase 0 variant
(0.7 / 0.8.x / 0.9 / 0.9a) produced exactly **0%** of off-diagonal
FC entries below the −0.10 threshold. Real Atanas-2023 recordings
produce **~28%** on the same in-band metric. Phase 1.0 produces **9%**
(mean across 3 seeds, no plasticity / no modulators required).

That is approximately one-third of the gap closed by architecture
alone — not by tuning, not by adding mechanisms. The Phase 0 CTRNN +
homogeneous architecture was *structurally incapable* of producing
this distribution; the Phase 1.0 LIF + sparse-spike + GABA + sub-graph
architecture is.

The H_3 hypothesis ("anti-correlation comes from subgraph competition,
not from a global modulator") is **partially validated**: the
competition produces qualitatively new behavior, but does not by
itself close the full gap.

---

## 3. The three durable refutations

### 3.1 LIF + plasticity + modulators alone do **not** improve the headline metrics

Side-by-side, Phase 0.9 (CTRNN + RID @ gain 0.5) vs Phase 1.0 (full):

| metric | Phase 0.9 +RID | Phase 1.0 full | Δ |
|---|---:|---:|---:|
| subspace_alignment | +0.3526 | +0.2771 | **−0.0755** |
| temporal_correlation | −0.0138 | +0.0020 | +0.0158 |
| fc_similarity | +0.0616 | +0.0097 | **−0.0519** |

Two of three metrics regressed. The rate-trace burstiness of LIF
(vs. the continuous CTRNN trace) inflates PCA distance, and
stochastic co-firing chains pollute the FC matrix. **Phase 1.0 paid
an alignment-metric cost for the anti-correlation unlock; future
Phases must recover this without losing the unlock.**

### 3.2 Modulators are still inert without behavioral state

Same Phase 0.9 / 0.9a finding, now confirmed under a completely
different dynamical architecture:

- `c_RID = 0` across all 3 seeds (RID is silent because its only
  upstream is PVC, also silent).
- `c_5HT ≈ 0.07` (a 2.8% multiplicative threshold change at
  sensitivity = 0.4 — below noise).

The Phase 1.5 design must therefore treat modulator activation as a
**structural prerequisite**, not a dynamics-tuning problem. Either
the body must supply state-dependent drive to producer neurons, or
the modulator subsystem needs an alternative driver (sensory
→ modulator bridges like BAG→RID, currently unwired).

### 3.3 Three subgraphs silent under any bare-network condition

`pharyngeal_cpg`, `ventral_cord_motor`, `egg_laying` — mean rate
0.000 across an 8000-tick run. All three are downstream of
behaviors the bare network cannot generate (feeding, walking,
ovulation). No subgraph-membership tweak fixes this; only a body /
environment / behavioral-state input can.

---

## 4. Per-subgraph behavior — what worked, what didn't

From `output/phase1.0/subgraph_behavior.txt` (8000-tick, full
configuration):

**Worked (matching biology)**:
- anterior_touch → reversal_command: r = +0.44 ✓
- posterior_touch → forward_command: r = +0.41 ✓
- chemosensory_amphid + thermosensory share AIY/AIZ/RIA via overlap.

**Failed**:
- forward_command ↔ reversal_command: r = **+0.51** (should be
  anti-correlated; the central diagnostic failure).
- modulator_RID ↔ forward_command: r = +0.96 (artifactual; reflects
  shared noise + the tight PVC↔RID coupling, not modulatory drive).
- chemosensory_amphid ↔ thermosensory: r = 0.007 (should be
  positive — they share interneurons but AFD never fires in the bare
  simulation).
- Anti-correlation at FC < −0.20: 0.0% (real worm: 16.4%; strong
  mutual exclusion still inaccessible).

---

## 5. What this commits the project to

### Permanent decisions (cannot be revisited without explicit re-audit):

- **Graph-native is the canonical representation.** Matrices are
  cached subgraph views, not the underlying data structure. Reverting
  to the matrix-only Phase 0 architecture is off the table.
- **LIF + spike-rate observable is the canonical dynamics.** The
  `rate` leaky-integrator on spikes is the analog of GCaMP fluorescence
  for comparison purposes. CTRNN is retained in `algos.neural` only
  for backward compatibility.
- **Subgraph decomposition + node-overlap-for-sharing is the canonical
  modularization.** The 13 circuits in `algos.graph.circuits` are the
  baseline; additions/recategorizations happen by appending to
  `CIRCUIT_SPECS`, not by replacing them.
- **Plasticity and modulators are first-class but opt-in.** Both
  hooks (`attach_plasticity`, `attach_modulators`) are part of the
  simulator API; they default to off, so any new experiment can be
  run with or without them.

### Things Phase 1.5 must address:

1. **Behavioral state for modulators** — supply the input that makes
   RID and 5-HT do work.
2. **Inhibitory mechanism between forward and reversal command** —
   either tyramine arm of RIM (data ready, see
   `data/modulators_full.md` §4), or per-edge rectification on
   AVA/AVB↔motor gap junctions, or an explicit
   `inhibitory_command_gate` subgraph (see
   `notes/subgraph_audit_part2.md` §A.1).
3. **Activation of the 3 silent subgraphs** — pharyngeal, motor,
   egg-laying. Requires environmental/body context.
4. **Restoration of the 0.05–0.08 alignment-metric drop** — likely
   addressable by a longer rate-trace τ (~100 ticks instead of 30)
   and reducing stochastic co-firing through structured drive.

### Things the project will *not* attempt:

- A full muscle / fluid-dynamics body model. Phase 1.5 is a minimal
  body, "stimulus-conversion + actuation + energy" only.
- Reverting any Phase 1.0 architecture choice.
- Re-running Phase 0.9 experiments for direct comparison; the
  PHASE0.9_REPORT.md numbers are the canonical baseline.

---

## 6. Project memory artifacts produced

- `PHASE1.0_REPORT.md` — narrative, per-sub-phase.
- `PHASE1.0_FINDINGS.md` — this document, durable claims.
- `data_audit.md` — biology ground-truth (Phase 1.5+).
- `notes/edge_sign_audit.md`, `notes/subgraph_audit_part1.md`,
  `notes/subgraph_audit_part2.md`, `data/modulators_full.md` —
  Phase 1.5+ working notes.

These six documents together carry the durable Phase 1.0 + 1.5+
knowledge. Code without them is incomplete; they without code are
useful but incomplete. Treat the set as one deliverable.

---

## 7. One-sentence summary

The Phase 1.0 architectural shift unblocked the FC anti-correlation
gap from 0% to 9% at FC < −0.10 (about one-third of the way to real
biology), at the cost of a 0.05–0.08 drop on subspace and FC
similarity metrics that future phases must recover, and confirmed
under the new dynamics what Phase 0.9 found under the old: modulators
need behavioral state to do their job, which is exactly what Phase
1.5 is designed to supply.

---

*Status: Phase 1.0 closed. Phase 1.5+ audit closed. Phase 1.5
design draft in `docs/phase1.5_design.md`.*
