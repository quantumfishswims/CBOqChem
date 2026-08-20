#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRP-CCSD (restricted)

Implementation of iterative Cavity Born-Oppenheimer (CBO) coupled cluster theory with singles and doubles 
in cavity reaction potential (CRP) formulation (CRP-CCSD).
CRP-CCSD minimizes CBO electronic energy in cavity subspace self-consistently and is 
formally similar to implicit solvation CCSD models.

Correlated dipole fluctuation corrections for ab initio vibro-polaritonic chemistry.

Code exploits functionalities of PySCF for electronic structure calculations.

Literature CRP-CCSD:
Fischer, J. Chem. Phys. 161, 164112 (2024). doi:10.1063/5.0231528
Fischer, J. Chem. Theory Comput. (2025) 21 (23): 12081-12093. doi:10.1021/acs.jctc.5c01604
"""

import copy
import numpy as np
from pyscf import scf, gto, cc, ao2mo, lib

# --- Iterative CRP-RCCSD approach ---

class CRPCCSD(cc.ccsd.CCSD):
    def __init__(self, mf, crp_max_cycle, crp_conv_tol, **kwargs):
        super().__init__(mf, **kwargs)

        self.polarization   = getattr(mf, 'polarization', None)
        self.coupling       = getattr(mf, 'coupling', None)
        self.crp_max_cycle  = int(crp_max_cycle)
        self.crp_conv_tol   = float(crp_conv_tol) 
        self.de_polar_mo    = None

        self._keys.update(['polarization', 'coupling', 'crp_max_cycle', 'crp_conv_tol', 'de_polar_mo'])

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

        int1e_r_array    = -self.mol.intor('int1e_r')
        de_polar_ao      = np.einsum('i,ijk->jk', self.polarization, int1e_r_array, optimize=True)
        de_polar_mo      = np.einsum('pi,pq,qj->ij', mo_coeff, de_polar_ao, mo_coeff, optimize=True)

        return de_polar_mo

    def _dipole_ccsd_expect(self, t1=None, t2=None, l1=None, l2=None):
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        if l1 is None:
            l1 = self.l1
        if l2 is None:
            l2 = self.l2

        nocc, nvir  = t1.shape
        de_polar_mo = self.de_polar_mo

        de_oo = de_polar_mo[:nocc,:nocc]
        de_vv = de_polar_mo[nocc:, nocc:]
        de_ov = de_polar_mo[:nocc, nocc:]
        de_vo = de_polar_mo[nocc:, :nocc]

        # --- gamma-1 intermediates
        goo  =    -np.einsum('jc,ic->ij', l1, t1, optimize=True)
        goo -= 0.5*np.einsum('jkcd,ikcd->ij', l2, t2, optimize=True)
    
        gvv  = np.einsum('kb,ka->ab',l1, t1, optimize=True)
        gvv += 0.5*np.einsum('klbc,klac->ab', l2, t2, optimize=True)
    
        gov = l1
            
        gvo         = t1.T # np.array[nvir, nocc]
        gvo_lambda  = np.einsum('jb,jiba->ai', l1, t2, optimize=True)
        gvo_lambda -= np.einsum('jb,ib,ja->ai', l1, t1, t1, optimize=True)

        gvo_x1 = np.einsum('kjcb,kicb->ji', l2, t2, optimize=True)
        gvo_x2 = np.einsum('kjcb,kjca->ba', l2, t2, optimize=True)
        gvo_lambda   -= 0.5*np.einsum('ji,ja->ai', gvo_x1, t1, optimize=True)
        gvo_lambda   += 0.5*np.einsum('ba,ib->ai', gvo_x2, t1, optimize=True)

        gvo += gvo_lambda

        # --- ccsd electronic dipole moment ---
        de_oo_ccsd      = np.einsum('ij,ij->', de_oo, goo, optimize=True)
        de_vv_ccsd      = np.einsum('ab,ab->', de_vv, gvv, optimize=True)
        de_ov_ccsd      = np.einsum('ia,ia->', de_ov, gov, optimize=True)
        de_vo_ccsd      = np.einsum('ai,ai->', de_vo, gvo, optimize=True)
        de_vo_lambda    = np.einsum('ai,ai->', de_vo, gvo_lambda, optimize=True)

        de_ccsd         = de_oo_ccsd + de_vv_ccsd + de_ov_ccsd + de_vo_ccsd
        de_ccsd_lambda  = de_oo_ccsd + de_vv_ccsd + de_ov_ccsd + de_vo_lambda

        return de_ccsd, de_ccsd_lambda

    
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


    # --- CRP-CCSD energy, amplitudes and multipliers

    def energy(self, t1=None, t2=None, eris=None):
        """Compute correlation energy with CRP corrections."""
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        if eris is None:
            eris = self.ao2mo(mo_coeff=self.mo_coeff)

        eris_ovov   = np.asarray(eris.ovov)
        ecorr       = 2.0*np.einsum('iajb, ijab', eris_ovov, t2, optimize=True)
        ecorr      -=     np.einsum('ibja, ijab', eris_ovov, t2, optimize=True)

        # Lambda0 correction
        nocc             = self.nocc
        de_mo_ov         = self.de_polar_mo[:nocc, nocc:]
        ecorr_lambda0    = np.einsum('ia,ia', t1, de_mo_ov)

        # Lambda correction
        ecorr_lambda     = self._dipole_ccsd_expect()[1]

        ecorr -= 2.0*self.coupling**2*ecorr_lambda0**2
        ecorr += 0.5*self.coupling**2*ecorr_lambda**2

        return ecorr

    def build_crp_eff_fock(self, eris):

        fock_crprhf     = eris.fock
        de_polar_mo     = self.de_polar_mo
        de_polar_cc     = self._dipole_ccsd_expect()[0]

        crp_eff_fock    = fock_crprhf 
        crp_eff_fock   -= self.coupling**2*de_polar_cc*de_polar_mo

        return crp_eff_fock

    def update_amps(self, t1=None, t2=None, eris=None):
        """Update T1 and T2 amplitudes with CRP corrections."""
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        if eris is None:
            eris = self.ao2mo(mo_coeff=self.mo_coeff)

        eris_crpcc       = copy.copy(eris)
        eris_crpcc.fock  = self.build_crp_eff_fock(eris)

        t1new, t2new = super().update_amps(t1, t2, eris_crpcc)

        return t1new, t2new

    def update_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        if t1 is None and t2 is None:
            t1 = self.t1
            t2 = self.t2
        if l1 is None and l2 is None:
            l1 = self.l1
            l2 = self.l2
        if eris is None:
            eris = self.ao2mo(mo_coeff=self.mo_coeff)

        eris_crpcc          = copy.copy(eris)
        eris_crpcc.fock     = self.build_crp_eff_fock(eris)
        
        l1new, l2new = super().update_lambda(t1, t2, l1, l2, eris_crpcc)

        return l1new, l2new

    # -------------------------------------------------------------------
    #                CRP-CC Macro-Iterator 
    # -------------------------------------------------------------------

    def solve_crpcc(self, crp_max_cycle=None, crp_conv_tol=None):
        if crp_max_cycle is None:
            crp_max_cycle = self.crp_max_cycle
        if crp_conv_tol is None:
            crp_conv_tol = self.crp_conv_tol
        if self.de_polar_mo is None:
            self.de_polar_mo = self._dipole_ao_to_mo()

        nocc = self.nocc
        nvir = self.nmo - nocc

        self.t1 = np.zeros((nocc, nvir))
        self.t2 = np.zeros((nocc, nocc, nvir, nvir))
        self.l1 = np.zeros((nocc, nvir))
        self.l2 = np.zeros((nocc, nocc, nvir, nvir))

        eris_bare   = self.ao2mo()
        de_mo_iter  = 2.0*np.einsum('ii', self.de_polar_mo)
        e_corr_iter = 2.0*self.crp_conv_tol

        e_corr_crpcc       = []
        delta_ecorr_crpcc  = []
        delta_de_crpcc     = []

        for crp_cycle in range(crp_max_cycle):
            print("-" * 75)
            print(f" CRP-CCSD macro-iteration {crp_cycle+1}.")
            print("-" * 75)
            
            self.kernel(eris=eris_bare)
            self.solve_lambda(eris=eris_bare)

            e_corr = self.energy()
            e_corr_crpcc.append(e_corr)

            de_mo_cc = self._dipole_ccsd_expect()[0]
            delta_de_crpcc.append(abs(de_mo_cc - de_mo_iter))

            delta_ecorr_crpcc.append(abs(e_corr-e_corr_iter))

            if delta_ecorr_crpcc[-1] < crp_conv_tol:
                print("-" * 75)
                print(f" CRP-CCSD converged in {crp_cycle+1} macro-iterations (g0 = {self.coupling} sqrt(Eh)/ea0).")
                print(f" CRP-CCSD Electronic Dipole  : {de_mo_cc:18.10f} a.u.")
                print(f" CRP-CCSD Correlation Energy : {e_corr_crpcc[-1]:18.10f} Eh")
                #print(f" CRP-CCSD energy convergence tolerance: {delta_e_crpcc[-1]:18.10f} Eh\n")
                return e_corr_crpcc[-1], delta_ecorr_crpcc, delta_de_crpcc
            
            elif crp_cycle+1 == crp_max_cycle and delta_ecorr_crpcc[-1] > crp_conv_tol:
                print("-" * 75)
                print(f" CRP-CCSD failed to converged in {crp_cycle+1} macro-iterations (g0 = {self.coupling} sqrt(Eh)/ea0).")
                print(f"delta_ecorr_crpcc: {delta_ecorr_crpcc[-2:]} a.u.")
                print(f"e_corr_crpcc: {e_corr_crpcc[-2:]} a.u.")
                return e_corr_crpcc[-1], delta_ecorr_crpcc, delta_de_crpcc

            e_corr_iter = e_corr
            de_mo_iter  = de_mo_cc



