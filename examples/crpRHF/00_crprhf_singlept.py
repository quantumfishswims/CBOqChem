#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Singlepoint energies via Cavity Born-Oppenheimer (CBO) Restricted Hartree-Fock 
in cavity reaction potential (CRP) formulation (CRP-RHF).
CRP approach minimizes CBO electronic energy in cavity subspace

Example provides mean-field electronic dipole fluctuation correction 
for a water dimer coupled to a single cavity mode.

Literature:
Fischer, J. Chem. Phys. 161, 164112 (2024). doi:10.1063/5.0231528
"""

import numpy as np
from pyscf import gto, scf
import bootstrap
from src.CRPqChem import CRPRHF

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

delta_crprhf    = [] # Mean-field (RHF) dipole fluctuation correction (vanishes for coup = 0.0)

# --- Standard RHF calculation for reference ---
mf = scf.RHF(mol)   # *.density_fit(), *.newton()
mf.kernel()
e_tot_rhf = mf.e_tot

# --- CRP-RHF calculations for different polarization directions ---
for i in range(len(polarization)):
    crpmf = CRPRHF(mol, polarization[i], coupling) # *.density_fit(), *.newton()
    crpmf.kernel()
    e_crp_rhf = crpmf.e_tot
    delta_crprhf.append(np.abs(e_crp_rhf - e_tot_rhf))

print("delta_crprhf: \n", delta_crprhf)