# CBOqChem 

This repository contains PySCF-based implementations of the cavity Born-Oppenheimer (CBO) Hartree Fock 
and Coupled Cluster (CC) methods in the cavity reaction potential (CRP) formulation for applications in 
ab initio vibro-polaritonic chemistry. 

The theoretical background, implementation and model applications were presented in two publications:

- E.W. Fischer, J. Chem. Theory Comput. (2025) 21, 23, 12081-12093, doi:10.1021/acs.jctc.5c01604
- E.W. Fischer, J. Chem. Phys. 161, 164112 (2024), doi:10.1063/5.0231528

CBOqChem requires python, pyscf and numpy. 

## Cavity reaction potential (CRP) approach

The CRP approach provides access to electronic dipole fluctuation correction of molecular 
potential energy surfaces by integrating CBO energy optimization in cavity coordinate space self-consistently.
The CRP approach is formally similar implicit solvation methods in quantum chemistry.

## CRPqChem

### CRP-RHF

Mean-field approach to dipole-fluctuation corrected electronic energies in ab initio vibro-polaritonic chemistry. 

### CRP-CCSD

Correlated approach to dipole-fluctuation corrected electronic energies in ab initio vibro-polaritonic chemistry. 
CRP-CCSD solves coupled amplitude-multiplier equations self-consistently.

### LinCRP-CCSD

Approximate correlated approach to dipole-fluctuation corrected electronic energies in ab initio vibro-polaritonic chemistry. 
Linearized CRP (LinCRP) approximation based on decoupled amplitude and multiplier equations similar to canonical CC methods. 
Linearization does not refer to linearization of amplitude equations.
