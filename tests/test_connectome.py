"""AC1: data loading + structural checks (phase0.md §3.1)."""

from __future__ import annotations

import numpy as np
import pytest

from algos.config import N_NEURONS


def test_neuron_count(connectome):
    assert connectome.n_neurons == N_NEURONS
    assert len(connectome.neuron_names) == N_NEURONS
    assert connectome.W_chem.shape == (N_NEURONS, N_NEURONS)
    assert connectome.W_gap.shape == (N_NEURONS, N_NEURONS)


def test_neuron_to_idx_bijection(connectome):
    for name in connectome.neuron_names:
        idx = connectome.neuron_to_idx[name]
        assert connectome.neuron_names[idx] == name
    assert len(connectome.neuron_to_idx) == N_NEURONS


def test_gap_matrix_symmetric(connectome):
    assert np.allclose(connectome.W_gap, connectome.W_gap.T, atol=1e-12)


def test_gap_non_negative(connectome):
    assert (connectome.W_gap >= 0).all()


def test_chem_has_both_signs(connectome):
    """GABAergic neurons produce negative columns; verify both signs exist."""
    W = connectome.W_chem
    assert (W > 0).any()
    assert (W < 0).any()


def test_known_neurons_present(connectome):
    expected = ["ASEL", "ASER", "AVAL", "AVAR", "PLML", "PLMR",
                "AVBL", "AVBR", "RIS", "DD01", "VD13", "CANL", "CANR"]
    for name in expected:
        assert name in connectome.neuron_to_idx, f"Missing key neuron: {name}"


def test_connection_counts_match_cook_2019(connectome):
    """phase0.md §1.5 ranges: chem ~7000 raw synapses, gap ~600 (older) –
    Cook 2019 reports ~3,700 unique chem pairs and ~1,100 unique gap pairs.
    """
    n_chem = int(np.count_nonzero(connectome.W_chem))
    n_gap_total = int(np.count_nonzero(connectome.W_gap))
    n_gap_unique = int(np.count_nonzero(np.triu(connectome.W_gap, k=1)))

    # Permissive ranges — keep these bounds loose enough that an alternate
    # corrected/normalized Cook table won't break the test, but tight enough
    # to catch a structural load error (e.g. parsing only half the matrix).
    assert 3000 < n_chem < 5000, f"chem pair count {n_chem} outside expected"
    assert 800 < n_gap_unique < 1500, f"gap pair count {n_gap_unique}"
    assert n_gap_total == 2 * n_gap_unique  # symmetric, zero diagonal


def test_sparsity(connectome):
    """phase0.md §1.5: matrix sparsity < 10%."""
    density_chem = np.count_nonzero(connectome.W_chem) / (N_NEURONS ** 2)
    density_gap = np.count_nonzero(connectome.W_gap) / (N_NEURONS ** 2)
    assert density_chem < 0.10
    assert density_gap < 0.10


def test_categories_complete(connectome):
    """All neurons have a category."""
    for c in connectome.category:
        assert c in {"pharyngeal", "sensory", "interneuron", "motor",
                     "sex_specific", "other_neuron"}


def test_gabaergic_neurons_have_negative_columns(connectome):
    """Each known GABAergic neuron's outputs should be ≤ 0."""
    for name in ["DD01", "VD01", "VD13", "RIS", "AVL", "DVB", "RMED", "RMEL"]:
        j = connectome.idx(name)
        col = connectome.W_chem[:, j]
        assert (col <= 0).all(), f"{name} should have non-positive outputs"


def test_input_magnitudes_O1(connectome):
    """phase0.md §2.3 stated goal: per-neuron total input ~O(1)."""
    chem_l1 = np.abs(connectome.W_chem).sum(axis=1)
    assert chem_l1.max() <= 1.0 + 1e-9
