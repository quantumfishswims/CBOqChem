#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PES scan via Cavity Born-Oppenheimer (CBO) coupled cluster theory
in cavity reaction potential (CRP) formulation with doubles excitations (CRP-CCD).

Linearized CRP-CCD compared across its dring/ladder/mosaic diagram channels and
its MP2/MP3 limits, minimizing CBO electronic energy in cavity subspace.

Example provides correlated electronic dipole fluctuation correction
for a dissociating hydrogen dimer coupled to a single cavity mode.

Literature:
Fischer, J. Chem. Theory Comput. (2025) 21 (23): 12081-12093. doi:10.1021/acs.jctc.5c01604
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf
from CRPqChem import CRPRHF, CCD, LinCRPCCD

coupling            = 0.0                     # Light-matter coupling strength in sqrt(Eh)/e Bohr
polarization        = [np.array([1,0,0])]       # List of normalized polarization vectors
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

pes_ccd_scan                = []
pes_lincrpccd_scan          = []
pes_dring_lincrpccd_scan    = []
pes_ladder_lincrpccd_scan   = []
pes_mosaic_lincrpccd_scan   = []
pes_lincrpmp2_scan          = []
pes_lincrpmp3_scan          = []

mf_diss    = scf.RHF(dimerdiss)
crpmf_diss = CRPRHF(dimerdiss, polarization[0], coupling)

myccd_scan                  = CCD(mf_diss).as_scanner()
my_lincrpccd_scan           = LinCRPCCD(crpmf_diss).as_scanner()
my_dring_lincrpccd_scan     = LinCRPCCD(crpmf_diss, 'dring').as_scanner()
my_ladder_lincrpccd_scan    = LinCRPCCD(crpmf_diss, 'ladder').as_scanner()
my_mosaic_lincrpccd_scan    = LinCRPCCD(crpmf_diss, 'mosaic').as_scanner()
my_lincrpmp2_scan           = LinCRPCCD(crpmf_diss, 'mp2').as_scanner()
my_lincrpmp3_scan           = LinCRPCCD(crpmf_diss, 'mp3').as_scanner()

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

    e_scan_ccd = myccd_scan(dimer)
    pes_ccd_scan.append(e_scan_ccd)

    e_scan_lincrpccd = my_lincrpccd_scan(dimer)
    pes_lincrpccd_scan.append(e_scan_lincrpccd)

    e_scan_dring_lincrpccd = my_dring_lincrpccd_scan(dimer)
    pes_dring_lincrpccd_scan.append(e_scan_dring_lincrpccd)

    e_scan_ladder_lincrpccd = my_ladder_lincrpccd_scan(dimer)
    pes_ladder_lincrpccd_scan.append(e_scan_ladder_lincrpccd)

    e_scan_mosaic_lincrpccd = my_mosaic_lincrpccd_scan(dimer)
    pes_mosaic_lincrpccd_scan.append(e_scan_mosaic_lincrpccd)

    e_scan_lincrpmp2 = my_lincrpmp2_scan(dimer)
    pes_lincrpmp2_scan.append(e_scan_lincrpmp2)

    e_scan_lincrpmp3 = my_lincrpmp3_scan(dimer)
    pes_lincrpmp3_scan.append(e_scan_lincrpmp3)


pes_ccd_shift               = pes_ccd_scan - pes_ccd_scan[-1]
pes_lincrpccd_shift         = pes_lincrpccd_scan - pes_lincrpccd_scan[-1]
pes_dring_lincrpccd_shift   = pes_dring_lincrpccd_scan - pes_dring_lincrpccd_scan[-1]
pes_ladder_lincrpccd_shift  = pes_ladder_lincrpccd_scan - pes_ladder_lincrpccd_scan[-1]
pes_mosaic_lincrpccd_shift  = pes_mosaic_lincrpccd_scan - pes_mosaic_lincrpccd_scan[-1]
pes_lincrpmp2_shift         = pes_lincrpmp2_scan - pes_lincrpmp2_scan[-1]
pes_lincrpmp3_shift         = pes_lincrpmp3_scan - pes_lincrpmp3_scan[-1]


plt.plot(nuc_grid, pes_ccd_shift, label='CCD')
plt.plot(nuc_grid, pes_lincrpccd_shift, label='LinCRPCCD, pol = '+str(polarization[0]))
#plt.plot(nuc_grid, pes_dring_lincrpccd_shift, label='dring-LinCRPCCD, pol = '+str(polarization[0]))
#plt.plot(nuc_grid, pes_ladder_lincrpccd_shift, label='ladder-LinCRPCCD, pol = '+str(polarization[0]))
#plt.plot(nuc_grid, pes_mosaic_lincrpccd_shift, label='mosaic-LinCRPCCD, pol = '+str(polarization[0]))
#plt.plot(nuc_grid, pes_lincrpmp2_shift, label='LinCRPMP2, pol = '+str(polarization[0]))
#plt.plot(nuc_grid, pes_lincrpmp3_shift, label='LinCRPMP3, pol = '+str(polarization[0]))
plt.legend(loc='upper right')
plt.show()
