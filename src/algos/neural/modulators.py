"""Phase 0.9 — minimal RID neuropeptide modulator.

A single global slow scalar `c_RID` that tracks the activity of the RID
neuron with a long time constant and exerts an inhibitory modulation on
the reversal command pool (AVA / AVD / AVE).

This is the minimal H_1.4 test described in `logs/phase0.9_brief.md`: not
a faithful neuropeptide diffusion model, but enough to ask whether a
single global state variable that mediates behavioral-state mutual
exclusion closes any of the Phase 0.8 FC gap.

Math:
    c_RID(t+1) = c_RID(t) + (V[idx_RID] - c_RID(t)) / tau
    extra_input[idx in REVERSAL_POOL] -= gain * c_RID

The modulator is opt-in: a HeterogeneousNetwork without one is
numerically identical to a Phase 0.8.2 network (enforced by
`tests/test_modulators.py::test_no_modulator_equals_phase08_2`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from algos.connectome import ConnectomeData


# Documented in §3.3 of logs/phase0.9_brief.md.
DEFAULT_TAU_RID: float = 200.0
DEFAULT_MOD_GAIN: float = 0.5

# Backward (reversal) command pool. Forward command (AVB, PVC) is
# explicitly excluded per the brief.
REVERSAL_COMMAND_NEURONS: tuple[str, ...] = (
    "AVAL", "AVAR",
    "AVDL", "AVDR",
    "AVEL", "AVER",
)


@dataclass
class RIDModulator:
    """Single global slow modulator driven by RID, inhibiting reversal pool.

    Attributes:
        idx_RID: connectome index of the RID neuron.
        reversal_indices: connectome indices of the reversal command pool.
        tau: time constant for the c_RID first-order lag (in ticks).
        gain: scalar weight applied to c_RID when computing the
            modulation subtracted from the reversal pool's extra_input.
        c_RID: current modulator state (kept in [-1, 1] by construction
            since V is clipped to that range).
        history: optional list of per-tick c_RID values for diagnostics;
            disabled by default to keep step cost negligible.
    """

    idx_RID: int
    reversal_indices: np.ndarray
    tau: float = DEFAULT_TAU_RID
    gain: float = DEFAULT_MOD_GAIN
    c_RID: float = 0.0
    record_history: bool = False
    history: list[float] = field(default_factory=list)

    @classmethod
    def from_connectome(
        cls,
        connectome: ConnectomeData,
        *,
        tau: float = DEFAULT_TAU_RID,
        gain: float = DEFAULT_MOD_GAIN,
        reversal_pool: tuple[str, ...] = REVERSAL_COMMAND_NEURONS,
        record_history: bool = False,
    ) -> "RIDModulator":
        """Build a RIDModulator from a loaded connectome.

        Raises KeyError if RID or any of the reversal pool is absent.
        """
        if "RID" not in connectome.neuron_to_idx:
            raise KeyError("RID neuron not present in connectome")
        missing = [
            n for n in reversal_pool if n not in connectome.neuron_to_idx
        ]
        if missing:
            raise KeyError(
                f"reversal pool neurons missing from connectome: {missing}"
            )
        idx_RID = connectome.idx("RID")
        reversal_indices = np.array(
            [connectome.idx(n) for n in reversal_pool], dtype=np.int64
        )
        return cls(
            idx_RID=idx_RID,
            reversal_indices=reversal_indices,
            tau=float(tau),
            gain=float(gain),
            record_history=record_history,
        )

    def reset(self) -> None:
        """Zero the modulator state and clear history."""
        self.c_RID = 0.0
        self.history.clear()

    def step(self, V: np.ndarray) -> float:
        """Advance c_RID one tick using the current V.

        Returns the new c_RID value.
        """
        self.c_RID += (V[self.idx_RID] - self.c_RID) / self.tau
        if self.record_history:
            self.history.append(self.c_RID)
        return self.c_RID

    def apply_modulation(self, extra_input: np.ndarray) -> np.ndarray:
        """Subtract `gain * c_RID` from extra_input at the reversal pool.

        Mutates `extra_input` in place and returns it for chaining.
        """
        extra_input[self.reversal_indices] -= self.gain * self.c_RID
        return extra_input


__all__ = [
    "RIDModulator",
    "REVERSAL_COMMAND_NEURONS",
    "DEFAULT_TAU_RID",
    "DEFAULT_MOD_GAIN",
]
