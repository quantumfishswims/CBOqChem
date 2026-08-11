#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Singlepoint energy via Cavity Born-Oppenheimer (CBO) coupled cluster theory 
in cavity reaction potential (CRP) formulation with singles and doubles excitations (CRP-CCSD).

Linearized CRP-CCSD formulations minimize CBO electronic energy in cavity subspace on
1) mean-field level of theory
2) lambda0-level of theory (energy correction from correlated dipole fluctuations)
3) lambda-level of theory  (energy & amplitude correction from correlated dipole fluctuations)

Example provides correlated electronic dipole fluctuation correction 
for a water dimer coupled to a single cavity mode.

Literature:
Fischer, J. Chem. Phys. 161, 164112 (2024), doi:10.1063/5.0231528
"""

import bootstrap
import numpy as np
from pyscf import gto, scf, cc
from src.CRPqChem import CRPRHF, LinCRPCCSD

coupling            = 0.03                  # Light-matter coupling strength in sqrt(Eh)/e Bohr
polarization        = [np.array([0,0,1])]   # List of normalized polarization vectors
ao_basis            = 'def2svpd'            # Basis set for electronic structure calculations

mol = gto.M(
atom=[["H",  0.00142,      -0.03800,        0.0],
      ["O",  0.00142,       0.05139,       -0.95281],
      ["H",  0.00142,       -0.84858,       -1.27824],
      ["H",  0.74281,        0.37695,        1.98765],
      ["O",  0.00142,       -0.03800,        1.54719],
      ["H", -0.76839,        0.40250,        1.90662]],
    basis = ao_basis,
    charge = 0,
    verbose = 4
)

delta_lincrpccsd_mf         = []
delta_lincrpccsd_lambda0    = []
delta_lincrpccsd_lambda     = []

# -- Canonical CCSD reference calculation ---
mf = scf.RHF(mol)
mf.kernel()
mycc = cc.CCSD(mf)
mycc.kernel()
ecc = mf.e_tot + mycc.e_corr


for i_pol in range(len(polarization)):
    # CRP-RHF
    crpmf = CRPRHF(mol, polarization[i_pol], coupling)
    crpmf.kernel()

    # mean-field linearised CRP-CCSD
    mylincrpccsd_mf = LinCRPCCSD(crpmf)
    mylincrpccsd_mf.kernel()
    ecc_mflin = crpmf.e_tot + mylincrpccsd_mf.e_corr
    delta_lincrpccsd_mf.append(np.abs(ecc_mflin - ecc))

    #lambda0-linearised CRP-CCSD
    mylincrpccsd_lambda0 = LinCRPCCSD(crpmf, lambda0=True)                 
    mylincrpccsd_lambda0.kernel()
    ecc_lambda0 = crpmf.e_tot + mylincrpccsd_lambda0.e_corr
    delta_lincrpccsd_lambda0.append(np.abs(ecc_lambda0 - ecc))  

    #lambda-linearised CRP-CCSD   
    mylincrpccsd_lambda = LinCRPCCSD(crpmf, lambda1=True)                               
    mylincrpccsd_lambda.kernel()
    ecc_lambda = crpmf.e_tot + mylincrpccsd_lambda.e_corr
    delta_lincrpccsd_lambda.append(np.abs(ecc_lambda - ecc))  

print("delta_mf_lincrpccsd: \n",    delta_lincrpccsd_mf)
print("delta_lambda0_lincrpccsd: \n",  delta_lincrpccsd_lambda0)     
print("delta_lambda_lincrpccsd: \n",   delta_lincrpccsd_lambda)