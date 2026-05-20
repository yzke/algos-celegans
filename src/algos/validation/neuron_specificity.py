"""AC0.5.3 — Functional specificity of key neurons in the bare connectome.

The bare Phase 0 neural skeleton has no body, no environment, and no
sensory-translator. We can still ask: does the Cook 2019 connectome topology,
once run as a CTRNN with our v0.3 dynamics, actually realize the
well-documented functional circuits of the worm? Concretely: if we drive
ASEL, do its anatomically downstream chemotaxis interneurons activate? If we
drive AVA, do the backward-driving VA/DA motor neurons activate?

This module runs synthetic-input experiments — each test:

  1. pre-equilibrates the network to V≈0 (5000 ticks of zero input);
  2. applies a constant step input at the *driver* neurons;
  3. runs the dynamics until steady state (3000 ticks);
  4. measures the mean and per-neuron ΔV at the *target* neurons (vs the
     pre-equilibrium baseline, which is essentially zero).

A test passes when (a) the response sign matches the literature expectation,
and (b) the magnitude is above a small threshold (`MIN_MAGNITUDE = 0.01`) so
we don't accept numerical noise as a "response".

Critically: the *differentiator* behavior of ASE/ASER (ASEL prefers rising
NaCl, ASER prefers falling) is **not** testable at the connectome level. That
property lives in the SensoryTranslator (design.md §4.4), to be built in
Phase 1. What we can test here is propagation through the connectome —
"driving the input neuron of a known circuit causes the circuit's output to
respond in the right direction".

Literature anchors (in `expected_sign`):

- ASEL→AIY excitatory: White 1986 + Bargmann 2006 chemotaxis review.
- ASER→AIY/AIB excitatory: Suzuki 2008 (Nature, "Functional asymmetry…").
- AVAL/AVAR command backward → VA/DA cholinergic activation: Chalfie 1985,
  Kawano 2011 (Neuron, "An imbalancing act").
- AVBL/AVBR command forward → VB/DB activation via gap junctions: Kawano
  2011; the AVB→VB/DB coupling in Cook 2019 is overwhelmingly electrical.
- Anterior touch (ALM/AVM) → AVD/AVA backward escape reflex: Chalfie 1985
  ("Neural circuit for touch sensitivity"). Most of this propagates via
  the AVD gap-junction network in Cook 2019.
- AFD → AIY thermotaxis: Mori 1995, Hawk 2018. Chem weight in Cook 2019 is
  +0.2 (AFD is glutamatergic, but in our default-positive sign table it
  is treated as excitatory — see DECISIONS.md "GABA-only sign assignment").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from algos.connectome import ConnectomeData
from algos.neural import CTRNNParams, NeuralState, neural_step


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

PRE_EQ_TICKS: int = 5000        # zero-input ticks → V ≈ 0
DRIVE_TICKS: int = 3000         # constant-drive ticks → steady state
DRIVE_STRENGTH: float = 0.5     # input magnitude at each driver neuron
MIN_MAGNITUDE: float = 0.01     # ΔV must exceed this to count as a "response"
SEED: int = 42


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SpecificityResult:
    """Outcome of one specificity experiment."""

    name: str
    driver_neurons: list[str]
    target_neurons: list[str]
    expected_sign: int                  # +1, -1, or 0 (any non-zero response)
    drive_strength: float
    mean_dv: float                      # mean ΔV across target neurons
    max_abs_dv: float                   # max |ΔV| across target neurons
    per_target_dv: dict[str, float] = field(default_factory=dict)
    # Diagnostics for sanity-checking the experiment itself.
    baseline_max_abs_v: float = 0.0
    driven_max_abs_v: float = 0.0
    notes: str = ""

    @property
    def passed(self) -> bool:
        """A test passes when the response is large enough and the right sign."""
        if self.max_abs_dv < MIN_MAGNITUDE:
            return False
        if self.expected_sign == 0:
            return True   # only magnitude required
        return np.sign(self.mean_dv) == np.sign(self.expected_sign)

    def summary(self) -> str:
        sign_arrow = {1: "↑", -1: "↓", 0: "·"}[self.expected_sign]
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name:40s}  "
            f"drive=[{','.join(self.driver_neurons)}] (×{self.drive_strength}) → "
            f"target=[{','.join(self.target_neurons[:4])}{'…' if len(self.target_neurons) > 4 else ''}]  "
            f"expected {sign_arrow}  "
            f"mean ΔV={self.mean_dv:+.4f}  max|ΔV|={self.max_abs_dv:.4f}"
        )


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------


def measure_specificity(
    connectome: ConnectomeData,
    *,
    name: str,
    driver_neurons: Sequence[str],
    target_neurons: Sequence[str],
    expected_sign: int = 1,
    drive_strength: float = DRIVE_STRENGTH,
    pre_eq_ticks: int = PRE_EQ_TICKS,
    drive_ticks: int = DRIVE_TICKS,
    seed: int = SEED,
    notes: str = "",
) -> SpecificityResult:
    """Drive `driver_neurons` and measure response at `target_neurons`.

    Returns a `SpecificityResult` with mean and per-target ΔV (driven minus
    baseline). All input strengths are constant during the drive window;
    noise is disabled so the experiment is deterministic.
    """
    # Resolve indices upfront so unknown names fail fast.
    driver_idx = [connectome.idx(n) for n in driver_neurons]
    target_idx = [connectome.idx(n) for n in target_neurons]

    params = CTRNNParams(noise_level=0.0)

    # 1) Pre-equilibrate to V ≈ 0.
    state = NeuralState.initialize(connectome.n_neurons, seed=seed)
    rng = np.random.default_rng(seed)
    zero_in = np.zeros(connectome.n_neurons)
    for _ in range(pre_eq_ticks):
        state = neural_step(state, connectome, zero_in, params, rng)
    baseline_V = state.V.copy()
    baseline_max = float(np.max(np.abs(baseline_V)))

    # 2) Apply step input at the driver neurons.
    drive_input = np.zeros(connectome.n_neurons)
    for i in driver_idx:
        drive_input[i] = drive_strength
    for _ in range(drive_ticks):
        state = neural_step(state, connectome, drive_input, params, rng)
    driven_V = state.V.copy()
    driven_max = float(np.max(np.abs(driven_V)))

    # 3) Compute ΔV at the targets.
    dv = driven_V - baseline_V
    per_target = {connectome.neuron_names[i]: float(dv[i]) for i in target_idx}
    target_values = np.array(list(per_target.values()))

    return SpecificityResult(
        name=name,
        driver_neurons=list(driver_neurons),
        target_neurons=list(target_neurons),
        expected_sign=expected_sign,
        drive_strength=drive_strength,
        mean_dv=float(np.mean(target_values)),
        max_abs_dv=float(np.max(np.abs(target_values))),
        per_target_dv=per_target,
        baseline_max_abs_v=baseline_max,
        driven_max_abs_v=driven_max,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Test catalogue
# ---------------------------------------------------------------------------


def _va_da_targets() -> list[str]:
    """All backward-driving cholinergic motor neurons (VA + DA)."""
    return [f"VA{i:02d}" for i in range(1, 13)] + [f"DA{i:02d}" for i in range(1, 10)]


def _vb_db_targets() -> list[str]:
    """All forward-driving cholinergic motor neurons (VB + DB)."""
    # VB11 is the last VB in Cook 2019 (there is no VB12).
    return [f"VB{i:02d}" for i in range(1, 12)] + [f"DB{i:02d}" for i in range(1, 8)]


def default_test_battery() -> list[dict]:
    """The 6-test AC0.5.3 battery.

    Each dict has keys matching `measure_specificity` kwargs. Caller is
    expected to splat one of these into `measure_specificity(connectome, **d)`.
    """
    return [
        {
            "name": "ASEL → AIY (chemotaxis upstream)",
            "driver_neurons": ["ASEL"],
            "target_neurons": ["AIYL", "AIYR"],
            "expected_sign": +1,
            "notes": "ASEL is excitatory cholinergic to AIY (Suzuki 2008, Bargmann 2006).",
        },
        {
            "name": "ASER → AIY/AIB (chemotaxis upstream)",
            "driver_neurons": ["ASER"],
            "target_neurons": ["AIYL", "AIYR", "AIBL", "AIBR"],
            "expected_sign": +1,
            "notes": "ASER is excitatory to AIY (and weaker to AIB) per Cook 2019.",
        },
        {
            "name": "AVAL+AVAR → VA/DA (backward command → cholinergic motor)",
            "driver_neurons": ["AVAL", "AVAR"],
            "target_neurons": _va_da_targets(),
            "expected_sign": +1,
            "notes": "AVA drives backward locomotion via cholinergic chemical synapses (Kawano 2011).",
        },
        {
            "name": "AVBL+AVBR → VB/DB (forward command → motor, via gap)",
            "driver_neurons": ["AVBL", "AVBR"],
            "target_neurons": _vb_db_targets(),
            "expected_sign": +1,
            "notes": "AVB→VB/DB coupling is overwhelmingly electrical in Cook 2019.",
        },
        {
            "name": "anterior touch (ALM+AVM) → AVD/AVA (backing reflex)",
            "driver_neurons": ["ALML", "ALMR", "AVM"],
            "target_neurons": ["AVDL", "AVDR", "AVAL", "AVAR"],
            "expected_sign": +1,
            "notes": "Anterior touch reflex: ALM/AVM activate AVD/AVA via mixed chem+gap (Chalfie 1985).",
        },
        {
            "name": "AFD → AIY (thermosensory propagation)",
            "driver_neurons": ["AFDL", "AFDR"],
            "target_neurons": ["AIYL", "AIYR"],
            "expected_sign": +1,
            "notes": "AFD→AIY: dominant thermotaxis interneuron downstream of AFD (Mori 1995).",
        },
    ]


def run_default_battery(connectome: ConnectomeData) -> list[SpecificityResult]:
    """Execute the full default test battery and return per-test results."""
    return [
        measure_specificity(connectome, **kwargs)
        for kwargs in default_test_battery()
    ]


__all__ = [
    "SpecificityResult",
    "measure_specificity",
    "default_test_battery",
    "run_default_battery",
]
