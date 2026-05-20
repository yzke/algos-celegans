"""Shared pytest fixtures for Phase 0 tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure src/ is importable when running pytest from repo root.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from algos.connectome import ConnectomeData  # noqa: E402


@pytest.fixture(scope="session")
def connectome() -> ConnectomeData:
    """Load the connectome once per pytest session (uses .npz cache)."""
    return ConnectomeData.load()
