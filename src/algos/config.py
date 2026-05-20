"""Central configuration for ALGOS-Celegans Phase 0.

All values listed here are either:

  (a) Specified by the master design doc (`docs/design.md` §3.3, §3.6) — these
      should not be tweaked casually; or
  (b) Engineering knobs needed for the Phase 0 simulation only, clearly marked
      as such.

Keeping them in one place makes it easy to reproduce a run and to compare
against the documented values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
CONNECTOME_DIR: Path = DATA_DIR / "connectome"
OUTPUT_DIR: Path = PROJECT_ROOT / "output"
LOGS_DIR: Path = PROJECT_ROOT / "logs"

# Cook et al. 2019 source workbook (corrected July 2020 revision).
COOK2019_XLSX: Path = CONNECTOME_DIR / "SI5_corrected.xlsx"
CONNECTOME_CACHE: Path = CONNECTOME_DIR / "connectome.npz"


# ---------------------------------------------------------------------------
# Neural-system constants  (design.md §3.6)
# ---------------------------------------------------------------------------

N_NEURONS: int = 302


@dataclass(frozen=True)
class CTRNNDefaults:
    """Default CTRNN parameters from design.md §3.3 and §3.6."""

    tau: float = 10.0           # Time constant (in ticks). Initial version: uniform.
    beta: float = 5.0           # sigmoid steepness.
    noise_level: float = 0.01   # design.md §3.3.


# ---------------------------------------------------------------------------
# Connectome normalization
# ---------------------------------------------------------------------------
#
# Cook 2019 weights are integer "EM serial section" counts. Raw row sums can
# reach hundreds, which would make `chem_input = W_chem @ sigmoid(V)` dominate
# every other term in the CTRNN update and saturate the network within a few
# ticks.
#
# phase0.md §2.3 names the *goal* — "each neuron's total input should be
# O(1)" — and gives an example formula (global-max normalization). The example
# is insufficient: in our data, global-max normalization leaves max row sums at
# ~9, still an order of magnitude above the -V damping coefficient.
#
# We therefore use per-row L1 normalization, which directly satisfies the
# stated goal: each row of W_chem (and W_gap) is divided by max(1, sum of
# |row|), so the worst-case chem_input[i] is bounded by ~0.5 and gap_input
# stays in the same range. This is the minimal correction that lets AC2
# (zero-input decay, constant-input convergence, no NaN over 10^5 ticks) hold
# while preserving the directional structure of the connectome.
# Recorded in DECISIONS.md.
NORMALIZATION_MODE: str = "per_row"  # one of: "per_row", "global_max", "none"
