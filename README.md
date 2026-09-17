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

## Installation

Clone the repository and install it with pip:

```bash
git clone <repo-url>
cd ab_initio_crp_qchem
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
├── lin_crp_ccsd.py            # LinCRP-CCSD: linearized coupled cluster theory
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

The `tests/` directory contains a pytest suite covering `CRPRHF`, `CRPCCSD` and `LinCRPCCSD`. It checks that every cavity-corrected method reduces to its canonical PySCF counterpart (RHF, CCSD) when the cavity coupling is switched off, checks polarization-vector validation, and documents a couple of known pre-existing bugs as expected failures (`xfail`).

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

