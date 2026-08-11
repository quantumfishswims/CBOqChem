#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eric Fischer, 10.08.2026, v2.0

Implementation of Cavity Born-Oppenheimer (CBO) coupled cluster theory with singles and doubles 
in cavity reaction potential (CRP) formulation (CRP-CCSD).
CRP-CCSD approach minimizes CBO electronic energy in cavity subspace self-consistently and is 
formally similar to implicit solvation CCSD models.

Correlated dipole fluctuation corrections for ab initio vibro-polaritonic chemistry.

Code exploits functionalities of PySCF for electronic structure calculations.

Literature CBO-CCSD/CRP-CCSD:
Fischer, J. Chem. Phys. 161, 164112 (2024). doi:10.1063/5.0231528
Fischer, J. Chem. Theory Comput. (2025) 21 (23): 12081-12093. doi:10.1021/acs.jctc.5c01604
"""

import numpy as np
from pyscf import cc, ao2mo, lib

   
# --- Linearised CRP-RCCSD approach ---

class LinCRPCCSD(cc.ccsd.CCSD):
    def __init__(self, mf, lambda0=None, lambda1=None, **kwargs):
        super().__init__(mf, **kwargs)

        self.polarization   = getattr(mf, 'polarization', None)
        self.coupling       = getattr(mf, 'coupling', None)
        self.lambda0        = bool(lambda0) if lambda0 is not None else None
        self.lambda1        = bool(lambda1) if lambda1 is not None else None

        self._keys.update(['polarization', 'coupling', 'lambda0', 'lambda1'])

        if self.lambda0 is None and self.lambda1 is None:
            mode_label = "mf-LinCRPCCSD"
        if self.lambda0 is True and self.lambda1 is None:
            mode_label = "lambda0-LinCRPCCSD"
        if self.lambda1 is True:
            mode_label = "lambda-LinCRPCCSD"

        self.__class__.__name__ = mode_label

        if self.polarization is not None:
            if self.polarization.shape != (3,):
                raise ValueError("Polarization vector must be of shape (3,).")
            if abs(1-np.dot(self.polarization, self.polarization)) > 1e-15:
                raise ValueError("Polarization vector must be normalized.")
  
    # --- Helper methods ---

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

        int1e_r_array   = -self.mol.intor('int1e_r')
        de_polar_ao     = np.einsum('i,ijk->jk', self.polarization, int1e_r_array, optimize=True)
        de_polar_mo     = np.einsum('pi,pq,qj->ij', mo_coeff, de_polar_ao, mo_coeff, optimize=True)

        return de_polar_mo

    
    def _make_cbo_eris_incore(self, mo_coeff):
        """
        Build eris object with CBO-corrected two-electron integrals.
        Uses parent class to get a standard eris object, then overwrites integrals.
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
        raise NotImplementedError("Out-of-core CBO-CCSD ERI transformation is not implemented yet."
                                  "Please use in-core mode or increase cc.max_memory.")


    def ao2mo(self, mo_coeff=None):
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
    
    # --- Lin CRP-CCSD energy and amplitudes
    
    def energy(self, t1=None, t2=None, eris=None):
        """Compute correlation energy with CRP corrections."""
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        if eris is None:
            eris = self.ao2mo(mo_coeff=self.mo_coeff)

        # Call parent energy method using proper super()
        ecorr = super().energy(t1, t2, eris)

        # Lambda0 correction
        if self.lambda0 is True or self.lambda1 is True:
            nocc            = self.nocc
            de_polar_mo     = self._dipole_ao_to_mo()
            de_ov           = de_polar_mo[:nocc, nocc:]
            delta_lambda0   = np.einsum('ia,ia', t1, de_ov)**2
            
            ecorr -= 2.0*self.coupling**2*delta_lambda0

        return ecorr
    
    def update_amps(self, t1=None, t2=None, eris=None):
        """Update T1 and T2 amplitudes with CRP corrections."""
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        if eris is None:
            eris = self.ao2mo(mo_coeff=self.mo_coeff)

        # Call parent update_amps method using proper super()
        t1_new, t2_new = super().update_amps(t1, t2, eris)
    
        # Lambda1 correction
        if self.lambda1 is True:
            nocc = self.nocc
            de_polar_mo = self._dipole_ao_to_mo(mo_coeff=self.mo_coeff)
            de_polar_mo_oo = de_polar_mo[:nocc, :nocc]
            de_polar_mo_ov = de_polar_mo[:nocc, nocc:]
            de_polar_mo_vv = de_polar_mo[nocc:, nocc:]

            de_polar_mo_factor = 2.0*np.einsum('ia,ia', t1, de_polar_mo_ov)

            # T1 update with Lambda1 correction
            t1_lambda  = de_polar_mo_ov
            t1_lambda += 2.0*np.einsum('ikac,kc->ia', t2, de_polar_mo_ov, optimize=True)
            t1_lambda +=     np.einsum('ic,ac->ia', t1, de_polar_mo_vv, optimize=True)
            t1_lambda -=     np.einsum('ka,ki->ia', t1, de_polar_mo_oo)
            t1_lambda -=     np.einsum('ic,ka,kc->ia', t1, t1, de_polar_mo_ov, optimize=True)
            
            t1_new -= self.coupling**2*de_polar_mo_factor*t1_lambda

            # T2 update with Lambda1 correction
            t2_lambda  =  np.einsum('ijac,bc->ijab', t2, de_polar_mo_vv, optimize=True)
            t2_lambda -=  np.einsum('ijbc,ac->ijab', t2, de_polar_mo_vv, optimize=True)
            t2_lambda -=  np.einsum('ikab,jk->ijab', t2, de_polar_mo_oo, optimize=True)
            t2_lambda +=  np.einsum('jkab,ik->ijab', t2, de_polar_mo_oo, optimize=True)
            t2_lambda -=  np.einsum('ic,kjab,kc->ijab', t1, t2, de_polar_mo_ov, optimize=True)
            t2_lambda +=  np.einsum('jc,kiab,kc->ijab', t1, t2, de_polar_mo_ov, optimize=True)
            t2_lambda -=  np.einsum('ka,ijcb,kc->ijab', t1, t2, de_polar_mo_ov, optimize=True)
            t2_lambda +=  np.einsum('kb,ijca,kc->ijab', t1, t2, de_polar_mo_ov, optimize=True)

            t2_new -= self.coupling**2*de_polar_mo_factor*t2_lambda
        
        return t1_new, t2_new

    def copy(self, mf=None):
        # ensure compatility with PySCF's copy method to run density_fit(), newton(), etc.
        new_lincrpccsd = super().copy(mf)
        new_lincrpccsd.lambda0 = self.lambda0
        new_lincrpccsd.lambda1 = self.lambda1
        return new_lincrpccsd
    



