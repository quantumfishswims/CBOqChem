#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of Cavity Born-Oppenheimer (CBO) Hartree-Fock theory
in cavity reaction potential (CRP) formulation (CRP-HF).
CRP-HF determines cavity coordinates uniquely by 
minimizing CBO electronic energy self-consistently in cavity coordinate space.

CRP-RHF provides mean-field dipole fluctuation corrections for ab initio vibro-polaritonic chemistry.

Code exploits functionalities of PySCF for electronic structure calculations.

Literature CBO-RHF/CRP-RHF:
Fischer, J. Chem. Phys. 161, 164112 (2024). doi:10.1063/5.0231528
Fischer, J. Chem. Theory Comput. (2025) 21 (23): 12081-12093. doi:10.1021/acs.jctc.5c01604
"""

import numpy as np
from pyscf import scf
from ._cbo_common import validate_polarization


class CRPRHF(scf.hf.RHF):
    """Cavity Born-Oppenheimer restricted Hartree-Fock (CRP-RHF)."""

    def __init__(self, mol, polarization=None, coupling=None):
        super().__init__(mol)

        self.polarization   = polarization
        self.coupling       = coupling

        self._keys.update({'polarization', 'coupling'})

        validate_polarization(self.polarization)

    def get_hcore(self, mol=None):
        """
        Compute the CRP-corrected Hartree-Fock core Hamiltonian.

        Parameters
        ----------
        mol : pyscf.gto.M
            PySCF Molecule object.

        Returns
        -------
        hcore_crp : numpy.ndarray
            The CRP-corrected core Hamiltonian matrix (shape: (nao, nao)).
        """
        if mol is None:
            mol = self.mol

        hcore = super().get_hcore()

        int1e_rr_array = mol.intor_symmetric('int1e_rr').reshape(3,3,mol.nao, mol.nao)
        rr = np.einsum('i,ijkl,j->kl', self.polarization, int1e_rr_array, self.polarization, optimize=True)

        hdse = 0.5*self.coupling**2*rr
        hcore_crp = hcore + hdse

        return hcore_crp

    def get_veff(self, mol=None, dm=None, dm_last=0, vhf_last=0, hermi=1):
        """
        Compute the CRP effective two-index potential (veff).

        Parameters
        ----------
        mol : pyscf.gto.M
            PySCF Molecule object.
        dm : ndarray, shape (n_ao, n_ao)
            One-particle density matrix in AO basis.

        Returns
        -------
        veff_crp : ndarray, shape (n_ao, n_ao)
            The CRP-effective potential matrix (veff) in the AO basis.
        """
        if mol is None:
            mol = self.mol

        veff = super().get_veff(mol=mol,
                                    dm=dm,
                                    dm_last=dm_last,
                                    vhf_last=vhf_last,
                                    hermi=hermi)

        int1e_r_array = -mol.intor('int1e_r')
        de = np.einsum('i,ijk->jk', self.polarization, int1e_r_array)
        dde = np.outer(de,de).reshape((len(de), len(de), len(de), len(de)), order='C')

        # DSE exchange contribution
        Kdse = -0.5*self.coupling**2*np.einsum('pqrs,qr->ps', dde, dm, optimize=True)

        veff_crp = veff + Kdse

        return veff_crp

    def copy(self, mol=None):
        """Copy ensures compatibility with PySCF's density_fit(), newton(), etc."""
        if mol is None:
            mol = self.mol
        new_crprhf = super().copy(mol)
        new_crprhf.polarization = self.polarization
        new_crprhf.coupling = self.coupling
        return new_crprhf
