"""Edge (synapse) data structure for Phase 1.0 graph-native neural system.

Three edge types are supported (docs/phase1_design.md §3):

  * 'chemical'   — directed, signed (sign comes from pre-synaptic NT),
                   carries discrete spike events with explicit per-edge
                   delay. Most common.
  * 'electrical' — gap junction. Bidirectional in physiology but stored
                   as a *pair* of directed edges (one per direction) so
                   that every traversal is uniform. Sign is always +1.
                   Delay is 0 (instantaneous V coupling, §3.2).
  * 'modulatory' — from a modulator-class node to one or more targets;
                   changes a *parameter* of the target (threshold/tau)
                   rather than driving its state directly (§3.3, §7).

Plasticity flags (`is_plastic`, `plasticity_rule`) live on the edge —
this is where weight updates accumulate (§6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_EDGE_TYPES: frozenset[str] = frozenset({"chemical", "electrical", "modulatory"})

DEFAULT_CHEMICAL_DELAY: int = 1
DEFAULT_ELECTRICAL_DELAY: int = 0
DEFAULT_MODULATORY_DELAY: int = 0


@dataclass
class Edge:
    """A single synapse / gap junction / modulatory link.

    The pair (source, target, type) is the *identity* of an edge in the
    NetworkX MultiDiGraph. (The MultiDiGraph key — also exposed here as
    ``key`` — is "chem", "gap", or "mod" depending on type, allowing at
    most one of each between a given ordered pair.)
    """

    source: str
    target: str
    type: str                           # one of VALID_EDGE_TYPES
    weight: float                       # always positive magnitude
    sign: int                           # +1 (excitatory) or -1 (inhibitory)
    delay: int                          # in ticks
    is_plastic: bool = False
    plasticity_rule: str = "frozen"     # 'frozen' | 'hebbian' | 'stdp' | ...
    # Hard ceilings/floors used by plasticity to keep weights bounded (§6.2).
    w_min: float = 0.0
    w_max: float = 1.0
    # Modulator-specific: which target parameter the modulator scales,
    # and what type of operation (multiplicative vs additive). Ignored
    # for non-modulator edges.
    modulator_target_param: str = "threshold"
    modulator_op: str = "multiplicative"
    # Free-form metadata.
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.type not in VALID_EDGE_TYPES:
            raise ValueError(
                f"Edge type {self.type!r} not in {sorted(VALID_EDGE_TYPES)}"
            )
        if self.sign not in (+1, -1):
            raise ValueError(f"Edge sign must be +1 or -1, got {self.sign}")
        if self.weight < 0:
            raise ValueError(
                f"Edge weight must be non-negative magnitude (sign carries the "
                f"polarity); got {self.weight}"
            )
        if self.delay < 0:
            raise ValueError(f"Edge delay must be non-negative; got {self.delay}")

    @property
    def signed_weight(self) -> float:
        return self.sign * self.weight

    @property
    def key(self) -> str:
        """MultiDiGraph edge key — short string per type."""
        return {"chemical": "chem", "electrical": "gap", "modulatory": "mod"}[self.type]


def edge_key_for_type(edge_type: str) -> str:
    """Public helper — what MultiDiGraph key does an edge type get?"""
    return {"chemical": "chem", "electrical": "gap", "modulatory": "mod"}[edge_type]


__all__ = [
    "Edge",
    "VALID_EDGE_TYPES",
    "DEFAULT_CHEMICAL_DELAY",
    "DEFAULT_ELECTRICAL_DELAY",
    "DEFAULT_MODULATORY_DELAY",
    "edge_key_for_type",
]
