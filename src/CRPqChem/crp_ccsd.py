#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of self-consistentcavity Born-Oppenheimer (CBO) coupled cluster theory with 
singles and doubles excitations in cavity reaction potential (CRP) formulation (CRP-CCSD).

CRP-CCSD determines cavity coordinates uniquely by self-consistently minimizing CBO electronic energy in cavity coordinate space 
conceptually exploring formal similarities to implicit solvation CCSD models.

CRP-CCSD provides correlated dipole fluctuation corrections for ab initio vibro-polaritonic chemistry.

Code exploits functionalities of PySCF for electronic structure calculations.

Literature CRP-CCSD:
Fischer, J. Chem. Phys. 161, 164112 (2024). doi:10.1063/5.0231528
Fischer, J. Chem. Theory Comput. (2025) 21 (23): 12081-12093. doi:10.1021/acs.jctc.5c01604
"""

import copy
import numpy as np
from pyscf import cc
from ._cbo_common import CBOintegrals, validate_polarization

# --- Iterative CRP-RCCSD approach ---

class CRPCCSD(CBOintegrals, cc.ccsd.CCSD):
    """Iterative CRP-CCSD"""

    def __init__(self, mf, crp_max_cycle, crp_conv_tol, **kwargs):
        super().__init__(mf, **kwargs)

        self.polarization   = getattr(mf, 'polarization', None)
        self.coupling       = getattr(mf, 'coupling', None)
        self.crp_max_cycle  = int(crp_max_cycle)
        self.crp_conv_tol   = float(crp_conv_tol)
        self.de_polar_mo    = None

        self._keys.update({'polarization', 'coupling', 'crp_max_cycle', 'crp_conv_tol', 'de_polar_mo'})

        validate_polarization(self.polarization)

    # --- Helper methods ---

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
        """Build the CRP-effective Fock matrix used in the amplitude/lambda updates."""
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
        """Run the CRP-CCSD self-consistent macro-iteration loop over the CCSD kernel."""
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
                return e_corr_crpcc[-1], delta_ecorr_crpcc, delta_de_crpcc

            elif crp_cycle+1 == crp_max_cycle and delta_ecorr_crpcc[-1] > crp_conv_tol:
                print("-" * 75)
                print(f" CRP-CCSD failed to converged in {crp_cycle+1} macro-iterations (g0 = {self.coupling} sqrt(Eh)/ea0).")
                print(f"delta_ecorr_crpcc: {delta_ecorr_crpcc[-2:]} a.u.")
                print(f"e_corr_crpcc: {e_corr_crpcc[-2:]} a.u.")
                return e_corr_crpcc[-1], delta_ecorr_crpcc, delta_de_crpcc

            e_corr_iter = e_corr
            de_mo_iter  = de_mo_cc
