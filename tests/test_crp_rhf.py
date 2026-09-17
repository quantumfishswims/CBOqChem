"""Tests for CRPRHF (src/CRPqChem/crp_rhf.py)."""

import numpy as np
import pytest

from CRPqChem import CRPRHF

from conftest import make_crprhf


def test_zero_coupling_matches_canonical_rhf(crprhf_zero_coupling, bare_rhf):
    """coupling=0 must switch off every DSE term, exactly reproducing plain RHF."""
    assert crprhf_zero_coupling.e_tot == pytest.approx(bare_rhf.e_tot, abs=1e-10)


def test_nonzero_coupling_shifts_energy(crprhf_cavity, bare_rhf):
    """A real cavity coupling must perturb the mean-field energy measurably."""
    assert np.isfinite(crprhf_cavity.e_tot)
    assert abs(crprhf_cavity.e_tot - bare_rhf.e_tot) > 1e-6


def test_energy_invariant_under_polarization_sign_flip(mol_h2, pol_z):
    """The DSE correction enters quadratically in the polarization vector,
    so flipping its sign must leave the energy unchanged."""
    mf_plus = make_crprhf(mol_h2, pol_z, 0.1)
    mf_minus = make_crprhf(mol_h2, -pol_z, 0.1)
    assert mf_plus.e_tot == pytest.approx(mf_minus.e_tot, abs=1e-10)


def test_energy_depends_on_coupling_magnitude(mol_h2, pol_z):
    """Larger coupling should give a larger (in magnitude) energy shift for
    a fixed polarization direction; monotonicity sanity check."""
    e0 = make_crprhf(mol_h2, pol_z, 0.0).e_tot
    e_small = make_crprhf(mol_h2, pol_z, 0.05).e_tot
    e_large = make_crprhf(mol_h2, pol_z, 0.15).e_tot
    assert abs(e_small - e0) < abs(e_large - e0)


@pytest.mark.parametrize(
    "bad_polarization",
    [
        np.array([1.0, 0.0, 0.0, 0.0]),  # wrong shape
        np.array([1.0, 1.0, 0.0]),        # not normalized
    ],
)
def test_rejects_invalid_polarization(mol_h2, bad_polarization):
    with pytest.raises(ValueError):
        CRPRHF(mol_h2, bad_polarization, 0.1)


def test_copy_preserves_cavity_parameters(crprhf_cavity, mol_h2):
    mf2 = crprhf_cavity.copy(mol_h2)
    assert mf2.mol is mol_h2
    assert np.allclose(mf2.polarization, crprhf_cavity.polarization)
    assert mf2.coupling == crprhf_cavity.coupling


def test_no_cavity_arguments_behaves_like_plain_rhf(mol_h2, bare_rhf):
    mf = CRPRHF(mol_h2)
    mf.kernel()
    assert mf.e_tot == pytest.approx(bare_rhf.e_tot, abs=1e-10)
