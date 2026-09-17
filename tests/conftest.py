"""Shared fixtures for the CRPqChem test suite.

All tests use H2/STO-3G: it is cheap enough to converge every method fast
while still giving a non-vanishing dipole self-energy (DSE)
correction. The DSE operator (int1e_r / int1e_rr) has non-zero matrix
elements in any basis with more than a single s function, independent of the
molecule's permanent dipole moment.
"""

import numpy as np
import pytest
from pyscf import gto, scf, cc

from CRPqChem import CRPRHF


@pytest.fixture(scope="session")
def mol_h2():
    return gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        verbose=0,
    )


@pytest.fixture(scope="session")
def pol_z():
    return np.array([0.0, 0.0, 1.0])


@pytest.fixture(scope="session")
def bare_rhf(mol_h2):
    """Canonical (non-cavity) PySCF RHF reference."""
    mf = scf.RHF(mol_h2)
    mf.kernel()
    return mf


@pytest.fixture(scope="session")
def bare_ccsd_ecorr(bare_rhf):
    """Canonical (non-cavity) PySCF CCSD correlation energy reference."""
    mycc = cc.CCSD(bare_rhf)
    mycc.kernel()
    return mycc.e_corr


def make_crprhf(mol, polarization, coupling):
    mf = CRPRHF(mol, polarization, coupling)
    mf.kernel()
    return mf


@pytest.fixture
def crprhf_zero_coupling(mol_h2, pol_z):
    """CRP-RHF with coupling=0: must reduce exactly to canonical RHF."""
    return make_crprhf(mol_h2, pol_z, 0.0)


@pytest.fixture
def crprhf_cavity(mol_h2, pol_z):
    """CRP-RHF with a non-trivial cavity coupling."""
    return make_crprhf(mol_h2, pol_z, 0.1)
