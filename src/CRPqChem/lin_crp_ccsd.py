#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of linearized cavity Born-Oppenheimer (CBO) coupled cluster theory with 
singles and doubles excitations in cavity reaction potential (CRP) formulation (LinCRP-CCSD).

CRP-CCSD determines cavity coordinates uniquely by minimizing CBO electronic energy in cavity coordinate space 
conceptually exploring formal similarities to implicit solvation CCSD models.
LinCRP-CCSD exploits Langrangian linear in Lambda multiplier in analogy to canoncial CC theory rendering 
energy minimization approximate (non self-consistent).

LinCRP-CCSD provides correlated dipole fluctuation corrections for ab initio vibro-polaritonic chemistry.

Code exploits functionalities of PySCF for electronic structure calculations.

Literature LinCRP-CCSD:
Fischer, J. Chem. Phys. 161, 164112 (2024). doi:10.1063/5.0231528
Fischer, J. Chem. Theory Comput. (2025) 21 (23): 12081-12093. doi:10.1021/acs.jctc.5c01604
"""

import numpy as np
from pyscf import cc
from ._cbo_common import CBOintegrals, validate_polarization

# --- Linearised CRP-RCCSD approach ---

class LinCRPCCSD(CBOintegrals, cc.ccsd.CCSD):
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

        validate_polarization(self.polarization)

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
        t1new, t2new = super().update_amps(t1, t2, eris)
    
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
            
            t1new -= self.coupling**2*de_polar_mo_factor*t1_lambda

            # T2 update with Lambda1 correction
            t2_lambda  =  np.einsum('ijac,bc->ijab', t2, de_polar_mo_vv, optimize=True)
            t2_lambda -=  np.einsum('ijbc,ac->ijab', t2, de_polar_mo_vv, optimize=True)
            t2_lambda -=  np.einsum('ikab,jk->ijab', t2, de_polar_mo_oo, optimize=True)
            t2_lambda +=  np.einsum('jkab,ik->ijab', t2, de_polar_mo_oo, optimize=True)
            t2_lambda -=  np.einsum('ic,kjab,kc->ijab', t1, t2, de_polar_mo_ov, optimize=True)
            t2_lambda +=  np.einsum('jc,kiab,kc->ijab', t1, t2, de_polar_mo_ov, optimize=True)
            t2_lambda -=  np.einsum('ka,ijcb,kc->ijab', t1, t2, de_polar_mo_ov, optimize=True)
            t2_lambda +=  np.einsum('kb,ijca,kc->ijab', t1, t2, de_polar_mo_ov, optimize=True)

            t2new -= self.coupling**2*de_polar_mo_factor*t2_lambda
        
        return t1new, t2new

    def copy(self, mf=None):
        # ensure compatility with PySCF's copy method to run density_fit(), newton(), etc.
        new_lincrpccsd = super().copy(mf)
        new_lincrpccsd.lambda0 = self.lambda0
        new_lincrpccsd.lambda1 = self.lambda1
        return new_lincrpccsd
    



