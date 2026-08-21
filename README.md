# CBOqChem 

**CBOqChem** is a Python package providing PySCF-based implementations of the **Cavity Born-Oppenheimer (CBO) Hartree-Fock** and **Coupled Cluster** methods within the **Cavity Reaction Potential (CRP)** formulation. 

## Overview
The **CBOqChem** package enables ab initio vibro-polaritonic chemistry calculations for the electronic ground state of molecules in the vibrational strong coupling regime. Ab initio vibro-polaritonic chemistry 
combines quantum chemistry and non-relativistic quantum electrodynamics to model molecular systems and their reactivity in the presence of a low-frequency optical cavity. The **CRP** formulation directly
addresses cavity-induced electronic dipole fluctuations correcting the ground state potential energy surface.  

### Key Features
- **CRP Formulation**: Package exploits cavity-coordiante-free CRP formulation of the CBO framework, which ensures self-consistent energy optimization in cavity coordinate space.
- **CRP-RHF**: Self-consistent mean-field approach to dipole-fluctuation corrected electronic energies.
- **CRP-CCSD**: Self-consistent correlated approach to dipole-fluctuation corrected electronic energies relying on concepts from implicit solvation CCSD models.
- **LinCRP-CCSD**: Correlated approach to dipole-fluctuation corrected electronic energies similar to canonical CCSD.

## Literature 

The theoretical background and implementation details are described in:

1. **E.W. Fischer**. "Cavity Born–Oppenheimer coupled cluster theory: Toward electron correlation in the vibrational strong light-matter coupling regime."
    *J. Chem. Theory Comput.* (2025) 21(23), 12081-12093. 
	DOI: 10.1021/acs.jctc.5c01604
2. **E.W. Fischer**. "Cavity-modified local and non-local electronic interactions in molecular ensembles under vibrational strong coupling." 
    *J. Chem. Phys.* **161**, 164112 (2024). 
	DOI: 10.1063/5.0231528

Please cite these and the PySCF references therein when using CBOqChem in your research.

