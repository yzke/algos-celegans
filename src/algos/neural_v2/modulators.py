"""Modulator subsystem for Phase 1.0.4.

Per docs/phase1_design.md §7:

  调质生产者(如 RID, NSM, RIC)是图中的普通节点
  每个调质类型有一个全局浓度 c_m(t)
  target_m = mean(V of producer_nodes)
  c_m(t+1) = c_m(t) + (target_m - c_m(t)) / tau_m
  effective_threshold = threshold_base × (1 + sum_m: c_m × sensitivity[i, m])

Differences from Phase 0.9's ``algos.neural.modulators.RIDModulator``:

  * Drive uses the ``rate`` trace, not raw V — the spike-rate analog
    of "mean activity of producer neurons" (consistent with the rest
    of the runtime).
  * Effect is on the *threshold* (parameter-level modulation,
    §7.3), not on additive input. The runner re-pulls thresholds from
    the modulator each tick.
  * ``sign`` of sensitivity decides direction: positive sensitivity
    *raises* the threshold (suppresses spiking), negative
    *lowers* it (promotes spiking).

Two default modulators are wired:

  * RID neuropeptide: source = {RID}, targets = forward command pool
    (AVB, PVC). Sensitivity is NEGATIVE so an active RID lowers
    forward-pool thresholds → promotes forward locomotion. (Direction
    matches Phase 0.9a; here it's parameter-level rather than input-
    level.)
  * 5-HT serotonin: source = {NSML/R, ADFL/R, HSNL/R}, targets = forward
    command pool (AVB, AIB) with POSITIVE sensitivity → raises
    thresholds → suppresses forward locomotion (the canonical
    food-induced quiescence effect, Sze 2000), plus negative
    sensitivity on pharyngeal CPG (excites feeding, Trojanowski 2014).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from algos.graph import NeuralGraph


@dataclass
class Modulator:
    """One global modulator: one slow scalar concentration + targeted threshold scaling."""

    name: str
    producer_idx: np.ndarray        # (P,) int — neurons whose rate drives c_m
    target_idx: np.ndarray          # (T,) int — neurons whose threshold is modulated
    sensitivity: np.ndarray         # (T,) float — per-target multiplier in the threshold rule
    tau_m: float                    # time constant for c_m update (in ticks)
    c_m: float = 0.0
    record_history: bool = False
    history: list[float] = field(default_factory=list)

    @property
    def n_targets(self) -> int:
        return int(self.target_idx.size)

    def step(self, rate: np.ndarray) -> float:
        """Advance c_m one tick using the producer-neuron rate signal."""
        target_value = float(rate[self.producer_idx].mean())
        self.c_m += (target_value - self.c_m) / self.tau_m
        if self.record_history:
            self.history.append(self.c_m)
        return self.c_m

    def reset(self) -> None:
        self.c_m = 0.0
        self.history.clear()


@dataclass
class ModulatorBank:
    """Collection of Modulator objects + the threshold-modulation hook.

    The bank caches the base thresholds at construction time. Each
    tick, after every modulator's c_m is updated, the bank computes:

        effective_threshold[i] = threshold_base[i] *
            (1 + sum_m  c_m * sensitivity_at_i_for_m)

    and writes the result into the runtime's LIFParams.threshold array.
    """

    modulators: list[Modulator]
    base_threshold: np.ndarray        # (N,) frozen at construction
    # Per-target sensitivity, vectorized: (N, M). Filled at __post_init__.
    sensitivity_matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = self.base_threshold.shape[0]
        m = len(self.modulators)
        sm = np.zeros((n, m), dtype=np.float64)
        for mi, mod in enumerate(self.modulators):
            sm[mod.target_idx, mi] = mod.sensitivity
        self.sensitivity_matrix = sm

    @property
    def n_modulators(self) -> int:
        return len(self.modulators)

    def reset(self) -> None:
        for m in self.modulators:
            m.reset()

    def step_concentrations(self, rate: np.ndarray) -> np.ndarray:
        """Advance every modulator's c_m. Returns the (M,) c vector."""
        c = np.empty(self.n_modulators, dtype=np.float64)
        for mi, mod in enumerate(self.modulators):
            c[mi] = mod.step(rate)
        return c

    def apply_threshold_modulation(
        self,
        threshold_out: np.ndarray,
    ) -> np.ndarray:
        """Compute effective threshold per neuron given current c.

        Writes into ``threshold_out`` in-place and returns it.
        """
        c = np.array([m.c_m for m in self.modulators], dtype=np.float64)
        # (N,) = base * (1 + sum_m sm[:,m] * c[m])
        factor = 1.0 + self.sensitivity_matrix @ c
        # Clamp to a sensible range so a runaway concentration can't
        # produce zero or negative thresholds.
        factor = np.clip(factor, 0.1, 10.0)
        np.multiply(self.base_threshold, factor, out=threshold_out)
        return threshold_out


# ---------------------------------------------------------------------------
# Default bank constructor
# ---------------------------------------------------------------------------

# Tunables (design §7.2: tau_m >> tau).
DEFAULT_TAU_M: float = 500.0
RID_TARGET_NEURONS: tuple[str, ...] = ("AVBL", "AVBR", "PVCL", "PVCR")
RID_SENSITIVITY: float = -0.5         # negative → lower threshold → excite
SHT_SOURCE_NEURONS: tuple[str, ...] = (
    "NSML", "NSMR", "ADFL", "ADFR", "HSNL", "HSNR",
)
SHT_TARGET_FORWARD: tuple[str, ...] = ("AVBL", "AVBR", "AIBL", "AIBR")
SHT_SENSITIVITY_FORWARD: float = +0.4   # positive → raise threshold → suppress
SHT_TARGET_PHARYNX: tuple[str, ...] = ("M3L", "M3R", "MI", "I1L", "I1R")
SHT_SENSITIVITY_PHARYNX: float = -0.4   # negative → excite feeding


def build_default_modulator_bank(
    graph: NeuralGraph,
    base_threshold: np.ndarray,
    *,
    tau_m: float = DEFAULT_TAU_M,
    skip_missing: bool = True,
) -> ModulatorBank:
    """Construct the Phase 1.0.4 default bank: RID + 5-HT."""
    def _idx(names: Iterable[str]) -> np.ndarray:
        out = []
        for n in names:
            if graph.has_node(n):
                out.append(graph.index_of(n))
            elif not skip_missing:
                raise KeyError(f"modulator neuron {n!r} missing from graph")
        return np.array(out, dtype=np.int64)

    # RID modulator — single producer.
    rid_prod = _idx(["RID"])
    rid_tgt = _idx(RID_TARGET_NEURONS)
    rid_sens = np.full(rid_tgt.shape, RID_SENSITIVITY, dtype=np.float64)
    rid = Modulator(
        name="RID",
        producer_idx=rid_prod,
        target_idx=rid_tgt,
        sensitivity=rid_sens,
        tau_m=tau_m,
    )

    # 5-HT modulator — multiple producers and a heterogeneous target set
    # (forward-suppressing + pharynx-exciting).
    sht_prod = _idx(SHT_SOURCE_NEURONS)
    sht_fwd = _idx(SHT_TARGET_FORWARD)
    sht_phx = _idx(SHT_TARGET_PHARYNX)
    sht_tgt = np.concatenate([sht_fwd, sht_phx])
    sht_sens = np.concatenate([
        np.full(sht_fwd.shape, SHT_SENSITIVITY_FORWARD),
        np.full(sht_phx.shape, SHT_SENSITIVITY_PHARYNX),
    ])
    sht = Modulator(
        name="5HT",
        producer_idx=sht_prod,
        target_idx=sht_tgt,
        sensitivity=sht_sens,
        tau_m=tau_m,
    )

    return ModulatorBank(modulators=[rid, sht], base_threshold=base_threshold.copy())


__all__ = [
    "Modulator",
    "ModulatorBank",
    "build_default_modulator_bank",
    "DEFAULT_TAU_M",
    "RID_TARGET_NEURONS",
    "RID_SENSITIVITY",
    "SHT_SOURCE_NEURONS",
    "SHT_TARGET_FORWARD",
    "SHT_TARGET_PHARYNX",
    "SHT_SENSITIVITY_FORWARD",
    "SHT_SENSITIVITY_PHARYNX",
]
