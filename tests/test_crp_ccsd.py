"""Tests for CRPCCSD (src/CRPqChem/crp_ccsd.py), the iterative CRP-CCSD solver."""

import numpy as np
import pytest

from CRPqChem import CRPCCSD

from conftest import make_crprhf


def test_zero_coupling_matches_canonical_ccsd(mol_h2, pol_z, bare_ccsd_ecorr):
    """CRP-CCSD must reproduce  canonical CCSD already after its first macro-iteration
    for coupling=0."""
    crpmf = make_crprhf(mol_h2, pol_z, 0.0)
    mycc = CRPCCSD(crpmf, crp_max_cycle=10, crp_conv_tol=1e-10)
    ecorr, delta_ecorr, _ = mycc.solve_crpcc()

    assert ecorr == pytest.approx(bare_ccsd_ecorr, abs=1e-8)
    assert delta_ecorr[-1] < 1e-10


def test_converges_with_nonzero_coupling(mol_h2, pol_z):
    crpmf = make_crprhf(mol_h2, pol_z, 0.1)
    mycc = CRPCCSD(crpmf, crp_max_cycle=20, crp_conv_tol=1e-8)
    ecorr, delta_ecorr, delta_de = mycc.solve_crpcc()

    assert np.isfinite(ecorr)
    assert delta_ecorr[-1] < 1e-8
    assert len(delta_ecorr) <= 20


def test_cavity_correlation_energy_differs_from_bare(mol_h2, pol_z, bare_ccsd_ecorr):
    crpmf = make_crprhf(mol_h2, pol_z, 0.1)
    mycc = CRPCCSD(crpmf, crp_max_cycle=20, crp_conv_tol=1e-10)
    ecorr, _, _ = mycc.solve_crpcc()

    assert abs(ecorr - bare_ccsd_ecorr) > 1e-6


@pytest.mark.parametrize(
    "bad_polarization",
    [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ],
)
def test_rejects_invalid_polarization_from_mean_field(mol_h2, bare_rhf, bad_polarization):
    """CRPCCSD re-validates `.polarization`, independent of CRPRHF."""
    mf = bare_rhf.copy()
    mf.polarization = bad_polarization
    mf.coupling = 0.1
    with pytest.raises(ValueError):
        CRPCCSD(mf, crp_max_cycle=5, crp_conv_tol=1e-6)


def test_mean_field_without_cavity_attrs_is_treated_as_no_cavity(bare_rhf):
    """Plain scf.RHF lacks .polarization/.coupling; must fall back to None."""
    mycc = CRPCCSD(bare_rhf, crp_max_cycle=5, crp_conv_tol=1e-6)
    assert mycc.polarization is None
    assert mycc.coupling is None
