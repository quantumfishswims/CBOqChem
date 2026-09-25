# CBOqChem 

**CBOqChem** is a Python package providing PySCF-based implementations of the **Cavity Born-Oppenheimer (CBO) restricted Hartree-Fock (RHF)** and **Coupled Cluster (CC)** methods within the **Cavity Reaction Potential (CRP)** formulation. 

## Overview
The **CBOqChem** package enables ab initio vibro-polaritonic chemistry calculations for the electronic ground state of molecules in the vibrational strong coupling regime. Ab initio vibro-polaritonic chemistry 
combines quantum chemistry and non-relativistic quantum electrodynamics to model molecular systems and their reactivity in the presence of a low-frequency optical cavity. The **CRP** formulation directly
addresses cavity-induced electronic dipole fluctuations correcting the ground state potential energy surface.  

### Key Features
- **CRP Formulation**: Package exploits cavity-coordiante-free CRP formulation of the CBO framework, which ensures self-consistent energy optimization in cavity coordinate space.
- **CRP-RHF**: Self-consistent mean-field approach to dipole-fluctuation corrected electronic energies.
- **CRP-CCSD**: Self-consistent correlated approach to dipole-fluctuation corrected electronic energies relying on concepts from implicit solvation CCSD models.
- **LinCRP-CCSD**: Correlated approach to dipole-fluctuation corrected electronic energies similar to canonical CCSD.

## Installation

Clone the repository and install it with pip:

```bash
git clone https://github.com/quantumfishswims/CBOqChem CBOqChem
cd CBOqChem
pip install .
```

For development (editable install, so local edits take effect without reinstalling):

```bash
pip install -e .
```

To also run the scripts under `examples/` (which use matplotlib for plotting), install the `examples` extra:

```bash
pip install -e ".[examples]"
```

Once installed, the package is importable as `CRPqChem`:

```python
from CRPqChem import CRPRHF, CRPCCSD, LinCRPCCSD
```

## Package Layout

The installable package lives under `src/CRPqChem/` (src layout), so it does not need to be on `PYTHONPATH` manually once installed:

```
src/CRPqChem/
├── __init__.py              # Public API: CRPRHF, CRPCCSD, LinCRPCCSD, CCD, LinCRPCCD
├── crp_rhf.py                # CRP-RHF: cavity Born-Oppenheimer mean-field theory
├── crp_ccsd.py                # CRP-CCSD: iterative (self-consistent) coupled cluster theory
├── lin_crp_ccsd.py            # LinCRP-CCSD: canonical coupled cluster theory
└── _cbo_common.py              # Shared DSE-augmented integral helpers reused by crp_ccsd.py and lin_crp_ccsd.py
```

## Examples

The `examples/` directory contains runnable scripts demonstrating each method on small molecular systems. They require the `examples` extra (for matplotlib) from the Installation section above.

### `examples/crpRHF/`
- **`00_crprhf_singlept.py`**: Singlepoint mean-field CRP-RHF electronic dipole fluctuation correction for a water dimer coupled to a single cavity mode.
- **`01_crprhf_scan.py`**: CRP-RHF potential energy surface scan for a dissociating hydrogen dimer coupled to a single cavity mode, comparing three cavity polarization directions and plotting the results.

### `examples/crpCC/`
- **`00_lin_crpccsd_singlept.py`**: Singlepoint LinCRP-CCSD electronic dipole fluctuation correction for a water dimer, comparing the mean-field, lambda0, and lambda levels of theory.
- **`01_lin_crpccsd_scan.py`**: Lambda0-level LinCRP-CCSD potential energy surface scan for a dissociating hydrogen dimer, comparing three cavity polarization directions and plotting the results.
- **`02_crpccsd_singlept.py`**: Singlepoint iterative (self-consistent) CRP-CCSD electronic dipole fluctuation correction for a water monomer.

Each script can be run directly once the package (and the `examples` extra) is installed, e.g.:

```bash
python examples/crpRHF/00_crprhf_singlept.py
```

## Testing

The `tests/` directory contains a pytest suite covering `CRPRHF`, `CRPCCSD`, `LinCRPCCSD`, `CCD` and `LinCRPCCD`, plus the shared `_cbo_common` mixin:

- **`test_cbo_common.py`**: polarization-vector validation shared by the CC-based classes.
- **`test_crp_rhf.py`**: `CRPRHF` reduces to canonical PySCF RHF at zero coupling, energy dependence on coupling magnitude/polarization sign, and polarization/cavity-argument validation.
- **`test_crp_ccsd.py`**: self-consistent `CRPCCSD` reduces to canonical CCSD at zero coupling, converges with nonzero coupling, and validates polarization input.
- **`test_lin_crp_ccsd.py`**: `LinCRPCCSD` reduces to canonical CCSD at zero coupling and correctly applies its lambda correction for nonzero coupling.
- **`test_lin_crp_ccd_channel.py`**: `LinCRPCCD` reduces to the local T1-pinned `CCD` class at zero coupling and to canonical MP2 for the `mp2` channel, checks channel labeling, and confirms convergence across all coupling channels.

Install the `test` extra and run pytest from the repository root:

```bash
pip install -e ".[test]"
pytest
```

## Literature 

The theoretical background and implementation details are described in:

1. **E.W. Fischer**. "Cavity Born–Oppenheimer coupled cluster theory: Toward electron correlation in the vibrational strong light-matter coupling regime."
    *J. Chem. Theory Comput.* (2025) 21(23), 12081-12093. 
	DOI: 10.1021/acs.jctc.5c01604
2. **E.W. Fischer**. "Cavity-modified local and non-local electronic interactions in molecular ensembles under vibrational strong coupling." 
    *J. Chem. Phys.* **161**, 164112 (2024). 
	DOI: 10.1063/5.0231528

Please cite these and the PySCF references therein when using CBOqChem in your research.

## License

CBOqChem is licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE). It depends on [PySCF](https://www.pyscf.org), which is also Apache-2.0 licensed. `src/CRPqChem/_cbo_common.py::CBOintegrals._make_cbo_eris_incore` additionally incorporates modified source code from PySCF's `pyscf.cc.ccsd._make_eris_incore`; see [NOTICE](NOTICE) for full attribution.

