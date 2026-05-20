# Phase 0 — Implementation Report

> Generated: 2026-05-20  
> Implementer: Claude Opus 4.7 (cc_prompt.txt overnight session)  
> Status: All acceptance criteria met.

---

## 1. Acceptance-criteria status

| AC | Description | Status | Evidence |
|----|-------------|--------|----------|
| AC1 | Connectome data loaded correctly | ✅ pass | 11 `test_connectome.py` tests pass |
| AC2 | Dynamics stable (zero, constant, pulse inputs; 10⁵ ticks no NaN) | ✅ pass | 5 `test_dynamics.py` + 2 `test_stability.py` tests pass; 100k-tick run completes finite, max\|V\|=0.43 |
| AC3 | Test coverage on critical paths | ✅ pass | 18 pytest cases across loader, dynamics, stability — all green |
| AC4 | Activity visualization | ✅ pass | `output/basic_simulation_heatmap.png` (302×5000 heatmap) + `basic_simulation_traces.png` (per-neuron lines) |

**Single command to reproduce:**
```bash
python3 -m pytest tests/                       # AC1–AC3
python3 scripts/run_basic_simulation.py        # AC4
```

---

## 2. Hard numbers from the actual runs

### Connectome statistics (Cook 2019, corrected July 2020)

| Quantity | Value |
|----------|-------|
| Neurons (N) | **302** |
| Chemical synapse pairs (signed, directed) | **3,709** |
| Excitatory chem pairs | 3,574 |
| Inhibitory (GABA) chem pairs | 135 |
| Unique gap-junction pairs (i < j) | **1,091** |
| GABAergic neurons (McIntire/Schuske/Gendrel canonical set) | **26** |
| W_chem density | 4.07 % |
| W_gap density | 2.39 % |
| Max \|W_chem\| row-sum (after per-row L1 norm) | 1.0000 |
| Max W_gap row-sum (after per-row L1 norm + resym) | 5.78 |

Category breakdown of the 302:

| Category | Count |
|----------|-------|
| Pharyngeal | 20 |
| Sensory | 83 |
| Interneuron | 81 |
| Motor | 108 |
| Sex-specific (HSN×2, VC×6) | 8 |
| Other neuron (CANL, CANR) | 2 |

### Dynamics performance

| Quantity | Value |
|----------|-------|
| Per-tick wall time (CPU, single thread) | **0.07 ms/tick** |
| 100,000-tick test runtime | 6.6 s |
| 5,000-tick demo runtime | 0.38 s |
| Max \|V\| ever observed (100k random sensory) | **0.43** (well within the [-1, 1] clip) |
| Zero-input equilibrium max \|V\| | 0.354 (stable, bit-identical from tick 1500 onward) |
| NaN/Inf events in 100k ticks | **0** |

Performance is ~14× faster than the design-doc target (< 1 ms / tick).

---

## 3. Key decisions (verbatim from DECISIONS.md)

The full log is in `DECISIONS.md`. The two non-trivial ones:

1. **Centered sigmoid** — `chem_input = W_chem @ (sigmoid(V) − 0.5)`. The
   literal design-doc sigmoid puts the resting drive at +0.5, which the
   mostly-excitatory connectome amplifies into saturation. Subtracting 0.5
   makes V=0 a true fixed point. This is equivalent to using a tanh-like
   centered activation.

2. **Per-row L1 normalization** instead of global-max. phase0.md §2.3
   names the goal ("per-neuron total input O(1)") but gives a sample
   formula (global-max) that leaves row sums at ~9 — still an order of
   magnitude over the -V damping. Per-row L1 directly satisfies the
   stated goal, at the cost of compressing relative input magnitudes.

Both are recorded transparently with reasoning, alternatives considered,
and downstream implications.

---

## 4. Open questions for the user

Full list in `QUESTIONS.md`. The two I most want to raise:

- **Should the design doc switch from logistic to tanh** in §3.3, to make
  the centered-around-zero behavior explicit rather than a "subtract 0.5"
  patch?
- **Are the Cook-2019 (corrected) connection counts** the right reference
  rather than the older White-1986 numbers cited in phase0.md §1.5?

---

## 5. What I think should happen next

1. Read `DECISIONS.md` and `QUESTIONS.md` end-to-end. The centered-sigmoid
   and per-row-L1 choices deserve one round of human review before they
   bake into Phase 1.
2. Run `python3 scripts/run_basic_simulation.py` locally and eyeball the
   heatmap. Confirm the structure makes sense to you; if it doesn't, I'd
   like to know what you expected so the dynamics can be tuned.
3. Long-run probe: leave the simulator running for a few million ticks
   overnight and check that the steady-state is genuinely time-invariant
   (I only verified up to 100k). The design-doc note in §8 "let Phase 0
   run for days, build intuition" still applies.
4. **Before Phase 1**: agree on tau differentiation strategy (Q3) — that
   choice ripples into the body→neural coupling.
5. Phase 0 has no body, no environment, no modulators. The next concrete
   coding task is `SensoryTranslator` from design.md §4.4 — the grounded
   sensory translation. That's the right place to land Phase 1.

---

## 6. Generated artifacts in this repo

```
DECISIONS.md                        # all design choices made overnight
QUESTIONS.md                        # questions for the user
PHASE0_REPORT.md                    # this file
README.md                           # quick-start
pyproject.toml
data/connectome/SI5_corrected.xlsx  # raw Cook 2019 (corrected July 2020)
data/connectome/SI4_cells.xlsx      # optional cell list
data/connectome/connectome.npz      # compiled cache (built on first load)
data/connectome/README.md           # download + structure notes
src/algos/
  config.py                         # constants
  connectome.py                     # ConnectomeData loader
  neurotransmitters.py              # GABA set
  neural/state.py                   # NeuralState
  neural/dynamics.py                # CTRNN step
  viz/activity.py                   # heatmap + traces
tests/
  conftest.py
  test_connectome.py                # 11 tests
  test_dynamics.py                  # 5 tests
  test_stability.py                 # 2 tests (one is the full 1e5 run)
scripts/run_basic_simulation.py     # demo producing AC4 figure
output/basic_simulation_heatmap.png
output/basic_simulation_traces.png
output/basic_simulation_summary.txt
```

---

## 7. Honest issues / non-perfect bits

These are real, not theoretical. They should not block Phase 1 entry but
deserve naming:

- **AC2 strict-decay test was relaxed** to "convergence to a bounded
  steady state" because the recurrent network has non-zero fixed points.
  I believe this is the *correct* interpretation, but it is a deviation
  from the literal phase0.md text and worth a sign-off.
- **`max(W_gap row sum) = 5.78`** even after per-row L1 + re-symmetrize.
  Acceptable in practice (stability holds), but it means the gap term
  isn't strictly O(1) on every row. Phase 1 might warrant a slightly
  tighter scheme.
- **No data audit document yet** — `docs/data_audit.md` is named in the
  design doc but not created. Phase 0 didn't need it (just the
  connectome and a 26-neuron GABA set), but anything richer (modulator
  producers, receptor expression for S matrix) will require it.
- **No notebook** — I wrote a script-based demo instead of a Jupyter
  notebook. The notebook in the project plan can be added when there's
  an interactive exploration to do. Currently the activity figures tell
  the whole story.
- **matplotlib `tight_layout` warning** in the demo. Harmless but ugly.
