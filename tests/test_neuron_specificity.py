"""AC0.5.3 — pytest layer for the key-neuron specificity battery.

Each test in `default_test_battery()` becomes one parametrized pytest case.
A failure here means the v0.3 dynamics on the Cook 2019 connectome do *not*
realize the literature-reported functional circuit for that neuron — i.e.
a substantive scientific finding, not a numerical glitch.

The expensive part of each case (pre-equilibration + drive = ~8000 ticks) is
deterministic and runs in ~0.5 s; the whole battery is ~5 s. We do not
session-cache results because (a) cost is low and (b) parametrization keeps
failures readable.
"""

from __future__ import annotations

import pytest

from algos.validation.neuron_specificity import (
    MIN_MAGNITUDE,
    default_test_battery,
    measure_specificity,
)


@pytest.mark.parametrize("test_spec", default_test_battery(), ids=lambda d: d["name"])
def test_specificity(connectome, test_spec):
    """Each entry in the default battery must pass: right sign, large enough."""
    result = measure_specificity(connectome, **test_spec)
    assert result.passed, (
        f"specificity test '{result.name}' failed:\n"
        f"  expected_sign={result.expected_sign}, got mean ΔV={result.mean_dv:+.4f}, "
        f"max|ΔV|={result.max_abs_dv:.4f} (threshold {MIN_MAGNITUDE})\n"
        f"  per-target ΔV: {result.per_target_dv}"
    )


def test_pre_equilibration_is_quiet(connectome):
    """Sanity check: pre-equilibration leaves the network essentially at V=0.

    Every battery test assumes this — if pre-eq isn't quiet, ΔV is not
    really `driven minus baseline`, it's `driven minus residual activity`.
    """
    # Trigger one specificity run just to inspect the diagnostics.
    result = measure_specificity(
        connectome,
        name="diagnostic",
        driver_neurons=["ASEL"],
        target_neurons=["ASER"],
    )
    assert result.baseline_max_abs_v < 1e-4, (
        f"pre-equilibration max|V|={result.baseline_max_abs_v:.2e} is not quiet"
    )
