#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRP-RHF

Implementation of Cavity Born-Oppenheimer (CBO) Hartree-Fock in cavity reaction potential (CRP) formulation (CRP-HF).
CRP approach minimizes CBO electronic energy in cavity subspace self-consistently.

Mean-field dipole fluctuation corrections for ab initio vibro-polaritonic chemistry.

Code exploits functionalities of PySCF for electronic structure calculations.

Literature CBO-RHF/CRP-RHF:
Fischer, J. Chem. Phys. 161, 164112 (2024). doi:10.1063/5.0231528
Fischer, J. Chem. Theory Comput. (2025) 21 (23): 12081-12093. doi:10.1021/acs.jctc.5c01604
"""

import numpy as np
from pyscf import scf

def get_hcore_crp(mf, mol=None, polarization=None, coupling=None):
    """
    Compute the CRP-corrected Hartree-Fock core Hamiltonian.

    Parameters
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    polarization : array_like, shape (3,)
        Cavity polarization vector
    coupling : float
       Light–matter coupling strength (scalar) used to scale the DSE terms.

    Returns
    -------
    hcore_crp : numpy.ndarray
        The CRP-corrected core Hamiltonian matrix (shape: (nao, nao)).
    """
    if mol is None:
        mol = mf.mol

    hcore = super().get_hcore(mol)

    if polarization is not None and coupling is not None:
         
         int1e_rr_array = mol.intor_symmetric('int1e_rr').reshape(3,3,mol.nao, mol.nao)
         rr = np.einsum('i,ijkl,j->kl', polarization, int1e_rr_array, polarization, optimize=True)
    
         # 1e DSE correction
         hdse = 0.5*coupling**2*rr
    
         hcore_crp = hcore + hdse

    return hcore_crp


def get_veff_crp(mf, mol=None, dm=None, dm_last=0, vhf_last=0,
                 hermi=1, polarization=None, coupling=None):
    """
    Compute the CRP effective two-index potential (veff)
    ----------
    mol : pyscf.gto.M
        PySCF Molecule object.
    dm : ndarray, shape (n_ao, n_ao)
        One-particle density matrix in AO basis.
    polarization : array_like, shape (3,)
        Cavity polarization vector
    coupling : float
       Light–matter coupling strength (scalar) used to scale the DSE terms.

    Returns
    -------
    veff_crp : ndarray, shape (n_ao, n_ao)
        The CRP-effective potential matrix (veff) in the AO basis:
    """
    if mol is None:
        mol = mf.mol

    veff = super().get_veff(
        mol=mol,
        dm=dm,
        dm_last=dm_last,
        vhf_last=vhf_last,
        hermi=hermi
    )

    if polarization is not None and coupling is not None:
        int1e_r_array = -mol.intor('int1e_r')
        de = np.einsum('i,ijk->jk', polarization, int1e_r_array)
        dde = np.outer(de,de).reshape((len(de), len(de), len(de), len(de)), order='C')

        # DSE exchange contribution
        Kdse = 0.5*coupling**2*np.einsum('pqrs,qr->ps', dde, dm, optimize=True)

        veff_crp =  veff - Kdse
    
    return veff_crp


class CRPRHF(scf.hf.RHF):
    def __init__(self, mol, polarization=None, coupling=None):
        super().__init__(mol)

        self.polarization   = polarization
        self.coupling       = coupling

        self._keys.update({'polarization', 'coupling'})

        if self.polarization is not None:
            if self.polarization.shape != (3,):
                raise ValueError("Polarization vector must be of shape (3,).")
            if abs(1-np.dot(self.polarization, self.polarization)) > 1e-15:
                raise ValueError("Polarization vector must be normalized.")

    def get_hcore(self, mol=None):
        """
        Compute the CRP-corrected Hartree-Fock core Hamiltonian.

        Parameters
         ----------
        mol : pyscf.gto.M
        PySCF Molecule object.
        polarization : array_like, shape (3,)
        Cavity polarization vector
        coupling : float
        Light–matter coupling strength (scalar) used to scale the DSE terms.

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
        Compute the CRP effective two-index potential (veff)
        ----------
        mol : pyscf.gto.M
        PySCF Molecule object.
        dm : ndarray, shape (n_ao, n_ao)
        One-particle density matrix in AO basis.
        polarization : array_like, shape (3,)
        Cavity polarization vector
        coupling : float
        Light–matter coupling strength (scalar) used to scale the DSE terms.

        Returns
        -------
        veff_crp : ndarray, shape (n_ao, n_ao)
        The CRP-effective potential matrix (veff) in the AO basis:
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

        veff_crp = veff+ Kdse

        return veff_crp

    def copy(self, mol=None):
        # ensure compatility with PySCF's copy method to run density_fit(), newton(), etc.
        if mol is None:
            mol = self.mol
        new_crprhf = super().copy(mol)
        new_crprhf.polarization = self.polarization
        new_crprhf.coupling = self.coupling
        return new_crprhf


        
    
    
    



























