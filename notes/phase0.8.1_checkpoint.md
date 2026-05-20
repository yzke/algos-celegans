# Phase 0.8.1 checkpoint — heterogeneous architecture refactor

## Status

**Done.** Architecture refactored without modifying any Phase 0–0.7 code paths.

## What was built

- `src/algos/neural/heterogeneous.py`: `HeterogeneousNetwork`,
  `HeterogeneousState`, `STEP_LIBRARY`, `ctrnn_default`,
  `register_step_function`.
- `tests/test_heterogeneous.py`: 5 tests covering equivalence,
  per-neuron dispatch, history buffer, and unknown-function rejection.
- `src/algos/neural/__init__.py`: re-exports the new symbols alongside
  the existing CTRNN ones. **No existing imports broken.**

## Acceptance criteria

| AC | Result |
|---|---|
| All-`ctrnn_default` equivalence with Phase 0.7 < 1e-6 | **0.0** (deterministic test, no noise, max \|ΔV\| ≈ 0 to machine precision) |
| All Phase 0–0.7 tests pass | **30 / 30 pass** (25 prior + 5 new) |
| Per-neuron function assignment works | passes `test_can_assign_per_neuron_functions` |
| Performance: < 10 ms/tick | **0.094 ms/tick worst case** (5 groups). 100× under budget. |

## Performance numbers

| path | ms/tick |
|---|---:|
| Phase 0.7 `neural_step` baseline | 0.075 |
| Heterogeneous, 1 group (all `ctrnn_default`) | 0.070 |
| Heterogeneous, 5 groups | 0.094 |

The 1-group case is ~7% faster than Phase 0.7. The extra speed comes
from caching `total_input = chem + gap + sensory + noise` once outside
the step function call, where Phase 0.7 inlines this into a single
expression. The cumulative cost is identical; the heterogeneous form
just happens to vectorize slightly better.

## Issues encountered

None. The design fell out cleanly:

1. **Decision**: per-neuron params stored as length-N ndarrays
   (e.g. `function_params["tau"]` is shape `(N,)`), not per-function
   nested dicts. This makes fancy-indexing into groups simple and
   numpy-friendly.

2. **Decision**: V_history is a fixed-length circular buffer
   (default 5 ticks); written explicitly each tick by copying ahead.
   Avoids allocations.

3. **Decision**: function_library can be passed per-network (not just
   the global STEP_LIBRARY). Tests use this to register a
   `constant_zero` step function without polluting global state.

4. **Decision**: `from_homogeneous_ctrnn` factory mirrors Phase 0.7's
   `NeuralState.initialize` semantics (same seed, same V0). The
   equivalence test exploits this.

## Files touched

```
NEW  src/algos/neural/heterogeneous.py
MOD  src/algos/neural/__init__.py             (added re-exports only)
NEW  tests/test_heterogeneous.py
NEW  notes/phase0.8.1_checkpoint.md           (this file)
```

`docs/`, `src/algos/connectome.py`, `src/algos/neural/dynamics.py`,
`src/algos/neural/state.py`, and all earlier-phase scripts are
**unchanged**.

## Ready for 0.8.2

Architecture supports:
- Multiple step functions registered via `register_step_function`
- Per-neuron tau and any other named param via `function_params`
- Per-neuron history access for change-detector etc. in 0.8.3
