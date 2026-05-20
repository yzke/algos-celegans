"""AC2 long-run: 10^5 tick stability with no NaN/Inf (phase0.md §3.3)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from algos.neural import CTRNNParams, NeuralState, neural_step


@pytest.mark.parametrize("n_ticks", [100_000])
def test_no_nan_in_long_run(connectome, n_ticks):
    """phase0.md §3.3: 10^5 ticks of noisy input, no NaN/Inf, stays in clip."""
    if os.environ.get("ALGOS_FAST_TESTS"):
        # Allow quick runs during iteration. The full 1e5 run is what counts
        # for the AC2 sign-off — it is captured separately in the Phase 0
        # report.
        n_ticks = 10_000

    state = NeuralState.initialize(302, seed=42)
    params = CTRNNParams()
    rng = np.random.default_rng(42)

    max_abs_V = 0.0
    for t in range(n_ticks):
        sens = rng.standard_normal(302) * 0.1
        state = neural_step(state, connectome, sens, params, rng)
        if t % 5000 == 0:
            assert np.all(np.isfinite(state.V)), (
                f"NaN/Inf encountered at tick {state.tick}"
            )
        max_abs_V = max(max_abs_V, float(np.max(np.abs(state.V))))

    assert np.all(np.isfinite(state.V))
    assert max_abs_V <= 1.0 + 1e-12
    assert state.tick == n_ticks


def test_seed_determinism(connectome):
    """Same seed → identical trajectory (a basic reproducibility check)."""
    def run():
        state = NeuralState.initialize(302, seed=123)
        params = CTRNNParams()
        rng = np.random.default_rng(123)
        for _ in range(500):
            sens = rng.standard_normal(302) * 0.1
            state = neural_step(state, connectome, sens, params, rng)
        return state.V.copy()

    v1 = run()
    v2 = run()
    assert np.array_equal(v1, v2)
