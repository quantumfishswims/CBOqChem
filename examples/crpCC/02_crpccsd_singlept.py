#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Singlepoint energy via Cavity Born-Oppenheimer (CBO) coupled cluster theory 
in cavity reaction potential (CRP) formulation with singles and doubles excitations (CRP-CCSD).

Iterative CRP-CCSD formulations minimize CBO electronic energy in cavity subspace.

Example provides correlated electronic dipole fluctuation correction 
for a water monomer coupled to a single cavity mode.

Literature:
Fischer, J. Chem. Phys. 161, 164112 (2024), doi:10.1063/5.0231528
"""

import bootstrap
import numpy as np
from pyscf import gto, scf, cc
from src.CRPqChem import CRPRHF, CRPCCSD

coupling            = 0.015                  # Light-matter coupling strength in sqrt(Eh)/e Bohr
polarization        = [np.array([0,0,1])]   # List of normalized polarization vectors
ao_basis            = 'augccpvdz'           # Basis set for electronic structure calculations

mol = gto.M(
atom=[["H",  0.00142,      -0.03800,        0.0],
      ["O",  0.00142,       0.05139,       -0.95281],
      ["H",  0.00142,       -0.84858,       -1.27824]],
    basis = ao_basis,
    charge = 0,
    verbose = 4
)

delta_crpccsd = []

# -- Canonical CCSD reference calculation ---
mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()
ecc = mf.e_tot + mycc.e_corr


# Iterative CRP-CCSD
crpmf = CRPRHF(mol, polarization[0], coupling)
crpmf.kernel()

mycrpccsd       = CRPCCSD(crpmf, 20, 5e-7)
ecorr_crpccsd, delta_ecorr_crpcc, delta_de_crpcc = mycrpccsd.solve_crpcc()
ecc_crpccsd     = crpmf.e_tot + ecorr_crpccsd
delta_crpccsd   = abs(ecc_crpccsd - ecc)

print("delta_crpccsd: \n", delta_crpccsd)
print("delta_ecorr_crpcc: \n", delta_ecorr_crpcc)
print("delta_de_crpcc: \n", delta_de_crpcc)