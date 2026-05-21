# ALGOS-Celegans Roadmap

> Current branch: main
> Last updated: 2026-05-21 (Phase 1.5+.5)
> Active phase: **Phase 1.5+ complete; Phase 1.5 implementation ready**

This is the canonical project roadmap. Each phase has a one-paragraph
description, status, key deliverables, and a link to its design
document and report. Phases are listed in chronological order.

---

## Phase 0 — Exploration (matrix + CTRNN paradigm) — **COMPLETE**

Sub-phases 0.5 through 0.9a (eight separate experiments). The Cook
2019 connectome was loaded into 302×302 NumPy matrices and driven with
continuous-time recurrent neural network dynamics; per-category
heterogeneity (Phase 0.8) and a single global modulator (Phase 0.9 /
0.9a) were tested as repairs for the architectural shortfall. The
project's central diagnostic — "the digital model produces 0% of FC
pairs at < −0.1 vs 17.5% in the real worm" — was identified in
Phase 0.9 and confirmed under every parameter sweep.

- **Deliverables**: `PHASE0_REPORT.md`, `PHASE0.5_REPORT.md`,
  `PHASE0.6_AUDIT.md`, `PHASE0.7_REPORT.md`, `PHASE0.8_REPORT.md`,
  `PHASE0.9_REPORT.md`, `PHASE0.9A_REPORT.md`,
  `src/algos/connectome.py`, `src/algos/neural/`.
- **Headline finding**: CTRNN + homogeneous (or category-heterogeneous)
  + single modulator architecture is **structurally incapable** of
  producing the anti-correlation distribution seen in real
  recordings.
- **Test count at close**: ~38 (45 passing after Phase 0.9a, with
  one deliberately-stale test left as a Phase 0.9a marker).
- **Design**: `docs/design.md` (v0.3).

---

## Phase 1.0 — Architecture period, first installment (graph-native + LIF) — **COMPLETE**

Reorganized the neural skeleton from "302×302 matrices plus a step
function" to "graph object with subgraph views plus an event-driven
LIF runtime", and added Hebbian plasticity + two parameter-level
modulators (RID, 5-HT) on top. Five sub-phases (1.0.1–1.0.5) shipped
as five independent commits.

- **Sub-phases**:
  - 1.0.1 — Graph primitives (Node, Edge, NeuralGraph, Subgraph) over
    `nx.MultiDiGraph`.
  - 1.0.2 — LIF dynamics + event-driven SignalQueue (0.07 ms/tick).
  - 1.0.3 — 13 functional subgraphs (reversal/forward command, two
    touch reflexes, chemo/thermo/head/feeding/motor/defecation/egg-
    laying + two modulators); anti-correlation diagnostic.
  - 1.0.4 — Hebbian on 100 edges + RID/5-HT modulators with
    threshold-level modulation.
  - 1.0.5 — Atanas-2023 comparison + subgraph behavior probe +
    PHASE1.0_REPORT.md.
- **Deliverables**: `src/algos/graph/`, `src/algos/neural_v2/`,
  `PHASE1.0_REPORT.md`, `PHASE1.0_FINDINGS.md`, 56 new tests.
- **Headline result**: Anti-correlation at FC < −0.10 went from
  **0% (all Phase 0 variants) → 9% (Phase 1.0, no tuning)**;
  real worm ≈ 28%. The H_3 hypothesis ("anti-correlation comes from
  subgraph competition, not from a global modulator") is **partially
  validated** — the competition produces qualitatively new behavior
  but does not close the full gap.
- **Headline failures**: subspace_alignment regressed by 0.08;
  fc_similarity regressed by 0.05 vs Phase 0.9 baseline; modulators
  remained inert (`c_RID = 0`, `c_5HT ≈ 0.07`); 3 of 13 subgraphs
  silent under any bare-network condition (pharynx, motor pool,
  egg-laying).
- **Test count at close**: 94 collected, 93 passing.
- **Design**: `docs/phase1_design.md` v1.1.

---

## Phase 1.5+ — Data audit + knowledge integration — **COMPLETE**

Between Phase 1.0 and Phase 1.5 (body integration), an 8-12 hour
audit + documentation pass that locks down the biology data the
project consumes. No code changes to `src/algos/` permitted (only
documented bug fixes, of which none were needed). Output is a set of
durable reference documents that all subsequent Phases cite.

- **Sub-phases**:
  - 1.5+.1 — Connectome data audit; edge sign rules verified.
    `notes/edge_sign_audit.md`.
  - 1.5+.2 — Subgraph definitions audited; 5 missing subgraphs
    identified. `notes/subgraph_audit_part1.md`,
    `notes/subgraph_audit_part2.md`.
  - 1.5+.3 — Modulator system data; 10 modulator families
    documented (2 wired, 8 unwired). `data/modulators_full.md`.
  - 1.5+.4 — Project-level data audit document. `data_audit.md`
    (30 audited points, 15 known gaps, 6 categories).
  - 1.5+.5 — Knowledge integration. `PHASE1.0_FINDINGS.md`,
    `docs/phase1_design.md` revised to v1.1,
    `docs/phase1.5_design.md` draft, this `ROADMAP.md`.
- **Key findings** documented for Phase 1.5 to act on:
  - Gap-junction rectification is the most likely single contributor
    to the forward↔reversal +0.51 anomaly.
  - RIM tyramine arm (well-attested, Pirri 2009) is missing — adding
    it would automatically introduce forward-suppression mechanism.
  - BAG → RID and URX → RIC sensory→modulator bridges exist in
    Cook 2019 but are not wired — root cause of modulator inertness.
  - `RIS / ALA / AVL`-centered inhibitory hub subgraph is missing;
    likely high-leverage fix for forward↔reversal +0.51.
- **No code changes**. Documentation-only deliverable.
- **Question log**: 7 entries added to `QUESTIONS.md` (Q-1.5+.1
  through Q-1.5+.7).

---

## Phase 1.5 — Minimal body integration — **READY TO START**

Add the minimum extrinsic input the Phase 1.0 system needs to
relieve the three documented failure modes: 3 silent subgraphs,
inert modulators, forward↔reversal +0.51 correlation. The body is
deliberately minimal — no fluid dynamics, no full muscle model — and
the design principle is "validation-oriented" rather than "biological
realism".

- **Sub-phases (planned)**:
  - 1.5.1 — `src/algos/env/` minimal environment (chemical/temperature
    scalar fields on a 2D grid).
  - 1.5.2 — `src/algos/body/` body state (position, heading, energy,
    death condition).
  - 1.5.3 — `src/algos/bridge/` three bridges: sensory (env → neural),
    motor (neural → body), feedback (body → modulator producers).
  - 1.5.4 — Augment circuits: `inhibitory_command_gate` subgraph;
    tyramine + dopamine modulator entries; AVA/AVB ↔ motor
    rectification.
  - 1.5.5 — Validation: chemotaxis, escape, sleep induction;
    PHASE1.5_REPORT.md.
- **Acceptance criteria** (from `docs/phase1.5_design.md` §8):
  - 3 silent subgraphs activate (mean rate > 0.01 in appropriate
    contexts).
  - `c_RID > 0.1` in CO2-elevated experiments; `c_5HT` significantly
    higher on food.
  - forward_command ↔ reversal_command Pearson r drops from +0.51
    to ≤ +0.10.
  - subspace_alignment recovers from +0.28 to ≥ +0.32; fc_similarity
    from +0.01 to ≥ +0.04.
  - Qualitative chemotaxis + repellent avoidance observable in
    trajectory.
- **Estimated effort**: 50-70 hours.
- **Design doc**: `docs/phase1.5_design.md` (v0.1 draft).

---

## Phase 2 — Structural plasticity — **NOT STARTED**

Allow the graph itself to evolve under environmental pressure:
edge addition / removal (synapse formation / pruning), node addition
(neurogenesis), topological rearrangement. This is the step that
moves the project from "fixed wiring + plastic weights" to
"wiring as a living object".

- **Prerequisites**: Phase 1.5 must succeed (otherwise we have no
  meaningful drive for structural plasticity).
- **Open questions**: structural plasticity rules in
  C. elegans are not extensively characterized in the adult — most
  rewiring happens during development. Phase 2 may need to invent
  rules rather than transcribe from biology.

---

## Phase 3 — Cross-species generalization — **NOT STARTED**

Apply the validated architecture to organisms beyond C. elegans:
larger connectomes (Drosophila, zebrafish, mouse cortical samples),
more complex subgraph structures, richer plasticity. The end goal
is "the architecture itself is a general digital-life operating
system" that can host organisms from worm-scale to brain-scale.

- **Prerequisites**: Phase 2 + a stable interface contract that does
  not implicitly assume C. elegans cardinality.
- **Currently no design work**.

---

## Documentation map

For project context, read in this order:

1. `README.md` — quick start.
2. `CLAUDE.md` — project instructions / structure.
3. `docs/design.md` — v0.3 master design (Phase 0 era).
4. `docs/phase1_design.md` — v1.1 (Phase 1.0 + Phase 1.5+ revisions).
5. `docs/phase1.5_design.md` — v0.1 (Phase 1.5 plan).
6. `PHASE1.0_FINDINGS.md` — durable Phase 1.0 conclusions.
7. `data_audit.md` — biology ground truth.
8. `DECISIONS.md` — per-phase decision log (chronological).
9. `QUESTIONS.md` — open questions, by phase.

For per-Phase narrative (longer, optional):

- `PHASE0_REPORT.md`, `PHASE0.5_REPORT.md`, `PHASE0.6_AUDIT.md`,
  `PHASE0.7_REPORT.md`, `PHASE0.8_REPORT.md`,
  `PHASE0.8_diagnostic.md`, `PHASE0.9_REPORT.md`,
  `PHASE0.9A_REPORT.md`, `PHASE1.0_REPORT.md`.

For Phase 1.5+ working notes:

- `notes/edge_sign_audit.md`, `notes/subgraph_audit_part1.md`,
  `notes/subgraph_audit_part2.md`, `data/modulators_full.md`.

---

## Phase status summary

| Phase | Status | Headline result |
|---|---|---|
| 0 (0.5 → 0.9a) | ✓ Complete | CTRNN paradigm refuted — FC anti-correlation gap structural |
| 1.0 (1.0.1 → 1.0.5) | ✓ Complete | Graph-native + LIF: anti-correlation 0% → 9%, alignment metrics regressed |
| 1.5+ (1.5+.1 → 1.5+.5) | ✓ Complete | Data audit + knowledge integration; Phase 1.5 design ready |
| 1.5 | ▷ Ready to start | Minimal body for behavioral state |
| 2 | — Not started | Structural plasticity |
| 3 | — Not started | Cross-species generalization |

---

*This file is updated at the close of every phase.*
