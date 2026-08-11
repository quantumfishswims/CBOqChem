#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PES scan via Cavity Born-Oppenheimer (CBO) Restricted Hartree-Fock 
in cavity reaction potential (CRP) formulation (CRP-RHF).
CRP approach minimizes CBO electronic energy in cavity subspace

Example provides mean-field electronic dipole fluctuation correction 
for a dissociating hydrogen dimer coupled to a single cavity mode.

Literature:
Fischer, J. Chem. Phys. 161, 164112 (2024). doi:10.1063/5.0231528
"""

import numpy as np
from matplotlib import pyplot as plt
from pyscf import gto, scf
import bootstrap
from src.CRPqChem import CRPRHF

coupling = 0.03                         # Light-matter coupling strength in sqrt(Eh)/e Bohr
polarization = [np.array([1,0,0]),  
                np.array([0,1,0]), 
                np.array([0,0,1])]      # List of normalized polarization vectors
ao_basis = 'def2svpd'                   # Basis set for electronic structure calculations
nuc_grid = np.arange(2.0, 11.0, 1.0)    # Nuclear grid

pes_rhf_scan    = []
pes_crprhf_scan = []

# --- Mean-field energy in dissociation limit ---
dimerdiss = gto.M(
    atom = [["H", 0.0 ,  0.0,  0.0],
            ["H", 0.74,  0.0,  0.0],
            ["H", 0.0 ,  0.0, max(nuc_grid)],
            ["H", 0.74,  0.0, max(nuc_grid)]],
    basis = ao_basis,
    charge = 0,
    verbose = 2
)

mf_scan = scf.RHF(dimerdiss).nuc_grad_method().as_scanner()
mycrprhf_scan_xpol = CRPRHF(dimerdiss, polarization[0], coupling).nuc_grad_method().as_scanner()
mycrprhf_scan_ypol = CRPRHF(dimerdiss, polarization[1], coupling).nuc_grad_method().as_scanner()
mycrprhf_scan_zpol = CRPRHF(dimerdiss, polarization[2], coupling).nuc_grad_method().as_scanner()

for i_grid in range(len(nuc_grid)):
    dimer = gto.M(
    atom=[["H", 0.00000,        0.00,       0.00],
          ["H", 0.74000,        0.00,       0.00],
          ["H", 0.00000,        0.00,       nuc_grid[i_grid]],
          ["H", 0.74000,        0.00,       nuc_grid[i_grid]]],
        basis = ao_basis,
        charge = 0,
        verbose = 3
    )
    
    e_scan_rhf, _ = mf_scan(dimer)
    pes_rhf_scan.append(e_scan_rhf)

    e_scan_crprhf_xpol, _ = mycrprhf_scan_xpol(dimer)
    e_scan_crprhf_ypol, _ = mycrprhf_scan_ypol(dimer)
    e_scan_crprhf_zpol, _ = mycrprhf_scan_zpol(dimer)
    e_scan_crprhf         = [e_scan_crprhf_xpol, 
                            e_scan_crprhf_ypol, 
                            e_scan_crprhf_zpol]
    pes_crprhf_scan.append(e_scan_crprhf)


pes_rhf_shift = pes_rhf_scan - pes_rhf_scan[-1]
pes_crprhf_xpol_shift = np.asarray(pes_crprhf_scan)[0:len(nuc_grid),0] - np.asarray(pes_crprhf_scan)[-1,0]
pes_crprhf_ypol_shift = np.asarray(pes_crprhf_scan)[0:len(nuc_grid),1] - np.asarray(pes_crprhf_scan)[-1,1]
pes_crprhf_zpol_shift = np.asarray(pes_crprhf_scan)[0:len(nuc_grid),2] - np.asarray(pes_crprhf_scan)[-1,2]

plt.plot(nuc_grid, pes_rhf_shift, label='RHF')
plt.plot(nuc_grid, pes_crprhf_xpol_shift, label='CRP-RHF, pol = '+str(polarization[0]))
plt.plot(nuc_grid, pes_crprhf_ypol_shift, label='CRP-RHF, pol = '+str(polarization[1]))
plt.plot(nuc_grid, pes_crprhf_zpol_shift, label='CRP-RHF, pol = '+str(polarization[2]))
plt.legend(loc='upper right')
plt.show()















