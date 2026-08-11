#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PES scan via Cavity Born-Oppenheimer (CBO) coupled cluster theory 
in cavity reaction potential (CRP) formulation with singles and doubles excitations (CRP-CCSD).

Linearized CRP-CCSD on lambda0-level of theory accounting for energy correction from 
correlated dipole fluctuations while minimizing CBO electronic energy in cavity subspace

Example provides correlated electronic dipole fluctuation correction 
for a dissociating hydrogen dimer coupled to a single cavity mode.

Literature:
Fischer, J. Chem. Phys. 161, 164112 (2024), doi:10.1063/5.0231528
"""

import bootstrap
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, cc
from src.CRPqChem import CRPRHF, LinCRPCCSD

coupling            = 0.03                      # Light-matter coupling strength in sqrt(Eh)/e Bohr
polarization        = [np.array([1,0,0]),
                       np.array([0,1,0]),
                       np.array([0,0,1])]       # List of normalized polarization vectors
ao_basis            = 'def2svpd'                # Basis set for electronic structure calculations
nuc_grid            = np.arange(3.0, 11.0, 1.0) # Nuclear grid

dimerdiss = gto.M(
    atom = [["H", 0.0 ,  0.0,  0.0],
            ["H", 0.74,  0.0,  0.0],
            ["H", 0.0 ,  0.0, max(nuc_grid)],
            ["H", 0.74,  0.0, max(nuc_grid)]],
    basis = ao_basis,
    charge = 0,
    verbose = 3
)

pes_ccsd_scan               = []
pes_lambda0_lincrpccsd_scan = []

#CCSD energy at dissociation limit
mf_diss         = scf.RHF(dimerdiss)
crpmf_xpol_diss = CRPRHF(dimerdiss, polarization[0], coupling)
crpmf_ypol_diss = CRPRHF(dimerdiss, polarization[1], coupling)
crpmf_zpol_diss = CRPRHF(dimerdiss, polarization[2], coupling)

myccsd_scan = cc.CCSD(mf_diss).as_scanner()
my_lambda0_lincrpccsd_scan_xpol = LinCRPCCSD(crpmf_xpol_diss, lambda1=True).as_scanner()
my_lambda0_lincrpccsd_scan_ypol = LinCRPCCSD(crpmf_ypol_diss, lambda1=True).as_scanner()
my_lambda0_lincrpccsd_scan_zpol = LinCRPCCSD(crpmf_zpol_diss, lambda1=True).as_scanner()

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
    
    e_scan_ccsd = myccsd_scan(dimer)
    pes_ccsd_scan.append(e_scan_ccsd)

    e_scan_lambda0_lincrpccsd_xpol = my_lambda0_lincrpccsd_scan_xpol(dimer)
    e_scan_lambda0_lincrpccsd_ypol = my_lambda0_lincrpccsd_scan_ypol(dimer)
    e_scan_lambda0_lincrpccsd_zpol = my_lambda0_lincrpccsd_scan_zpol(dimer)

    e_scan_lambda0_lincrpccsd      = [e_scan_lambda0_lincrpccsd_xpol,
                                      e_scan_lambda0_lincrpccsd_ypol,
                                      e_scan_lambda0_lincrpccsd_zpol                                    
                                      ]
    
    pes_lambda0_lincrpccsd_scan.append(e_scan_lambda0_lincrpccsd)


pes_ccsd_shift = pes_ccsd_scan - pes_ccsd_scan[-1]
pes_lambda0_lincrpccsd_shift_xpol = np.asarray(pes_lambda0_lincrpccsd_scan)[0:len(nuc_grid),0] - np.asarray(pes_lambda0_lincrpccsd_scan)[-1,0]
pes_lambda0_lincrpccsd_shift_ypol = np.asarray(pes_lambda0_lincrpccsd_scan)[0:len(nuc_grid),1] - np.asarray(pes_lambda0_lincrpccsd_scan)[-1,1]
pes_lambda0_lincrpccsd_shift_zpol = np.asarray(pes_lambda0_lincrpccsd_scan)[0:len(nuc_grid),2] - np.asarray(pes_lambda0_lincrpccsd_scan)[-1,2]

plt.plot(nuc_grid, pes_ccsd_shift, label='CCSD')
plt.plot(nuc_grid, pes_lambda0_lincrpccsd_shift_xpol, label='Lambda0-LinCRPCCSD, pol = '+str(polarization[0]))
plt.plot(nuc_grid, pes_lambda0_lincrpccsd_shift_ypol, label='Lambda0-LinCRPCCSD, pol = '+str(polarization[1]))
plt.plot(nuc_grid, pes_lambda0_lincrpccsd_shift_zpol, label='Lambda0-LinCRPCCSD, pol = '+str(polarization[2]))
plt.legend(loc='upper right')
plt.show()



















