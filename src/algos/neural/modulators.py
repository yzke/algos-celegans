"""Phase 0.9a — RID modulator with flipped direction (forward activation).

A single global slow scalar `c_RID` that tracks the activity of the RID
neuron with a long time constant and exerts an **excitatory** modulation
on the **forward command pool** (AVB / PVC).

Phase 0.9 (refuted, Δfc = −0.0019) targeted the reversal pool
(AVA/AVD/AVE) with an inhibitory `-= gain * c_RID`. The Phase 0.9a
hypothesis is that the biological direction may be inverted: RID
*activates* forward motor command rather than *inhibiting* reversal
command. See `logs/phase0.9a_brief.md`.

Math (Phase 0.9a):
    c_RID(t+1) = c_RID(t) + (V[idx_RID] - c_RID(t)) / tau
    extra_input[idx in FORWARD_POOL] += gain * c_RID

The exported constant `REVERSAL_COMMAND_NEURONS` and the field name
`reversal_indices` are retained for backward-compatible imports; in
Phase 0.9a they hold the forward pool (AVBL/AVBR/PVCL/PVCR). The
docstring and test_modulators.py describe Phase 0.9 semantics — they
will not match this Phase 0.9a swap by design (the brief instructed no
other code changes).

The modulator is opt-in: a HeterogeneousNetwork without one is
numerically identical to a Phase 0.8.2 network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from algos.connectome import ConnectomeData


# Documented in §3.3 of logs/phase0.9_brief.md.
DEFAULT_TAU_RID: float = 200.0
DEFAULT_MOD_GAIN: float = 0.5

# Phase 0.9a: retargeted to the forward command pool (AVB / PVC). Name
# kept as REVERSAL_COMMAND_NEURONS for backward-compatible exports from
# `algos.neural`; semantically these are now the forward command neurons.
REVERSAL_COMMAND_NEURONS: tuple[str, ...] = (
    "AVBL", "AVBR",
    "PVCL", "PVCR",
)


@dataclass
class RIDModulator:
    """Phase 0.9a: global slow modulator driven by RID, *activating* forward pool.

    Attributes:
        idx_RID: connectome index of the RID neuron.
        reversal_indices: connectome indices of the modulated command pool.
            In Phase 0.9a this is the FORWARD pool (AVBL/AVBR/PVCL/PVCR);
            the field name is preserved from Phase 0.9 for import stability.
        tau: time constant for the c_RID first-order lag (in ticks).
        gain: scalar weight applied to c_RID; Phase 0.9a uses += so a
            positive c_RID excites the forward command pool.
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

        Phase 0.9a: the default `reversal_pool` constant now contains
        the FORWARD command neurons. Param name retained for backward
        compatibility with any existing callers.

        Raises KeyError if RID or any of the pool neurons is absent.
        """
        if "RID" not in connectome.neuron_to_idx:
            raise KeyError("RID neuron not present in connectome")
        missing = [
            n for n in reversal_pool if n not in connectome.neuron_to_idx
        ]
        if missing:
            raise KeyError(
                f"command-pool neurons missing from connectome: {missing}"
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
        """Phase 0.9a: ADD `gain * c_RID` to extra_input at the forward pool.

        (Field still named `reversal_indices` for backward-compat — in
        Phase 0.9a it points to AVBL/AVBR/PVCL/PVCR.)

        Mutates `extra_input` in place and returns it for chaining.
        """
        extra_input[self.reversal_indices] += self.gain * self.c_RID
        return extra_input


__all__ = [
    "RIDModulator",
    "REVERSAL_COMMAND_NEURONS",
    "DEFAULT_TAU_RID",
    "DEFAULT_MOD_GAIN",
]
