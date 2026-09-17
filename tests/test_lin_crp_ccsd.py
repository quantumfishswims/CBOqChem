"""Tests for LinCRPCCSD (src/CRPqChem/lin_crp_ccsd.py)."""

import numpy as np
import pytest
from pyscf import gto

from CRPqChem import LinCRPCCSD

from conftest import make_crprhf


@pytest.fixture(scope="module")
def mol_lih():
    """H2 has zero T1 by symmetry, which would make lambda1 trivially
    zero regardless of correctness; LiH has no such symmetry."""
    return gto.M(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)


@pytest.mark.parametrize(
    "kwargs, expected_label",
    [
        ({}, "mf-LinCRPCCSD"),
        ({"lambda0": True}, "lambda0-LinCRPCCSD"),
        ({"lambda1": True}, "lambda-LinCRPCCSD"),
    ],
)
def test_mode_label_naming(mol_h2, pol_z, kwargs, expected_label):
    crpmf = make_crprhf(mol_h2, pol_z, 0.1)
    mylin = LinCRPCCSD(crpmf, **kwargs)
    assert type(mylin).__name__ == expected_label


@pytest.mark.parametrize("kwargs", [{}, {"lambda0": True}, {"lambda1": True}])
def test_zero_coupling_matches_canonical_ccsd(mol_h2, pol_z, bare_ccsd_ecorr, kwargs):
    """All extra terms scale as coupling**2, so coupling=0 must
    collapse onto plain CCSD for every mode."""
    crpmf = make_crprhf(mol_h2, pol_z, 0.0)
    mylin = LinCRPCCSD(crpmf, **kwargs)
    mylin.kernel()

    assert mylin.e_corr == pytest.approx(bare_ccsd_ecorr, abs=1e-8)


def test_lambda_correction_changes_energy_for_nonzero_coupling(mol_lih, pol_z):
    """lambda1 adds a genuine T1/T2 correction, so it must differ from
    mean-field mode once coupling is nonzero."""
    crpmf = make_crprhf(mol_lih, pol_z, 0.1)

    mf_mode = LinCRPCCSD(crpmf)
    mf_mode.kernel()

    lambda_mode = LinCRPCCSD(crpmf, lambda1=True)
    lambda_mode.kernel()

    assert np.isfinite(mf_mode.e_corr)
    assert np.isfinite(lambda_mode.e_corr)
    assert abs(mf_mode.e_corr - lambda_mode.e_corr) > 1e-8


@pytest.mark.parametrize(
    "bad_polarization",
    [
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 0.0]),
    ],
)
def test_rejects_invalid_polarization_from_mean_field(bare_rhf, bad_polarization):
    mf = bare_rhf.copy()
    mf.polarization = bad_polarization
    mf.coupling = 0.1
    with pytest.raises(ValueError):
        LinCRPCCSD(mf)
