#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared Cavity Born-Oppenheimer (CBO) integral helpers.

Dipole self-energy (DSE) augmented AO/MO integral construction and
cavity-polarization vector validation reused by CRPCCSD (crp_ccsd.py) 
and LinCRPCCSD (lin_crp_ccsd.py). 
"""

import numpy as np
from pyscf import cc, ao2mo, lib


def validate_polarization(polarization):
    """Validate that a cavity polarization vector is a normalized 3-vector."""
    if polarization is not None:
        if polarization.shape != (3,):
            raise ValueError("Polarization vector must be of shape (3,).")
        if abs(1 - np.dot(polarization, polarization)) > 1e-15:
            raise ValueError("Polarization vector must be normalized.")


class CBOintegrals:
    """Shared DSE-augmented ERI construction for CBO coupled-cluster classes."""

    def _cbo_eri_ao(self, eri):
        """DSE-augmented ERIs in AO basis."""
        if self.polarization is None or self.coupling is None or self.mol is None:
            return eri

        int1e_r_array = -self.mol.intor('int1e_r')
        de  = np.einsum('i,ijk->jk', self.polarization, int1e_r_array, optimize=True)
        dde = np.einsum('pq,rs->pqrs', de, de, optimize=True)

        cbo_eri = eri + self.coupling**2*dde

        return cbo_eri

    def _dipole_ao_to_mo(self, mo_coeff=None):
        """AO-to-MO transformation dipole integrals."""
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        if self.mol is None:
            return None

        int1e_r_array = -self.mol.intor('int1e_r')
        de_polar_ao   = np.einsum('i,ijk->jk', self.polarization, int1e_r_array, optimize=True)
        de_polar_mo   = np.einsum('pi,pq,qj->ij', mo_coeff, de_polar_ao, mo_coeff, optimize=True)

        return de_polar_mo

    def _make_cbo_eris_incore(self, mo_coeff):
        """
        Build eris object with CBO-corrected two-electron integrals.
        Uses parent class to get a standard eris object, then overwrites integrals.

        The ERI-packing loop below (from "Compute CBO-corrected AO integrals"
        to the end of the method) is adapted from PySCF's own
        pyscf.cc.ccsd._make_eris_incore (Copyright 2014-2021 The PySCF
        Developers, Apache License, Version 2.0). See the NOTICE file at the repository root.
        """

        # Get standard eris object from parent class
        eris = cc.ccsd._ChemistsERIs()
        eris._common_init_(self, mo_coeff)
        nocc = eris.nocc
        nmo = eris.fock.shape[0]
        nvir = nmo - nocc

        # Compute CBO-corrected AO integrals
        cbo_eri = self._cbo_eri_ao(self.mol.intor('int2e'))

        # transform to MO basis
        eri1 = ao2mo.incore.full(cbo_eri, eris.mo_coeff)

        # Pack the integrals as in _make_eris_incore
        if eri1.ndim == 4:
            eri1 = ao2mo.restore(4, eri1, nmo)

        # Initialize canonical ERI blocks
        nvir_pair = nvir * (nvir + 1) // 2
        eris.oooo = np.empty((nocc, nocc, nocc, nocc))
        eris.ovoo = np.empty((nocc, nvir, nocc, nocc))
        eris.ovvo = np.empty((nocc, nvir, nvir, nocc))
        eris.ovov = np.empty((nocc, nvir, nocc, nvir))
        eris.ovvv = np.empty((nocc, nvir, nvir_pair))
        eris.vvvv = np.empty((nvir_pair, nvir_pair))

        # Assign occupied-occupied blocks
        ij = 0
        outbuf = np.empty((nmo, nmo, nmo))
        oovv = np.empty((nocc, nocc, nvir, nvir))
        for i in range(nocc):
            buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
            for j in range(i+1):
                eris.oooo[i,j] = eris.oooo[j,i] = buf[j,:nocc,:nocc]
                oovv[i,j] = oovv[j,i] = buf[j,nocc:,nocc:]
            ij += i + 1
        eris.oovv = oovv
        oovv = None

        # Assign mixed and virtual blocks
        ij1 = 0
        for i in range(nocc, nmo):
            buf = lib.unpack_tril(eri1[ij:ij+i+1], out=outbuf[:i+1])
            eris.ovoo[:,i-nocc] = buf[:nocc,:nocc,:nocc]
            eris.ovvo[:,i-nocc] = buf[:nocc,nocc:,:nocc]
            eris.ovov[:,i-nocc] = buf[:nocc,:nocc,nocc:]
            eris.ovvv[:,i-nocc] = lib.pack_tril(buf[:nocc,nocc:,nocc:])
            dij = i - nocc + 1
            lib.pack_tril(buf[nocc:i+1,nocc:,nocc:], out=eris.vvvv[ij1:ij1+dij])
            ij += i + 1
            ij1 += dij

        return eris

    def _make_cbo_eris_outcore(self, mo_coeff):
        """Out-of-core CBO ERI transformation (not yet implemented)."""
        raise NotImplementedError("Out-of-core CBO-CCSD ERI transformation is not implemented yet. "
                                  "Please use in-core mode or increase cc.max_memory.")

    def ao2mo(self, mo_coeff=None):
        """Build the CBO-corrected eris object, in-core or out-of-core as memory allows."""
        if mo_coeff is None:
            mo_coeff = self.mo_coeff

        nmo = self.nmo
        nao = self.mo_coeff.shape[0]
        nmo_pair = nmo * (nmo+1) // 2
        nao_pair = nao * (nao+1) // 2
        mem_incore = (max(nao_pair**2, nmo**4) + nmo_pair**2) * 8/1e6
        mem_now = lib.current_memory()[0]

        if (self._scf._eri is not None and
            (mem_incore + mem_now < self.max_memory or self.incore_complete)):
            return self._make_cbo_eris_incore(mo_coeff)

        else:
            return self._make_cbo_eris_outcore(mo_coeff)
