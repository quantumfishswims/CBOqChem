"""
Tests for the shared CBO helpers (src/CRPqChem/_cbo_common.py).
"""

import numpy as np
import pytest

from CRPqChem import CRPCCSD
from CRPqChem._cbo_common import CBOintegrals, validate_polarization


def test_validate_polarization_accepts_unit_vector():
    validate_polarization(np.array([0.0, 1.0, 0.0]))  # must not raise


def test_validate_polarization_rejects_wrong_shape():
    with pytest.raises(ValueError, match="shape"):
        validate_polarization(np.array([1.0, 0.0]))


def test_validate_polarization_rejects_non_unit_vector():
    with pytest.raises(ValueError, match="normalized"):
        validate_polarization(np.array([1.0, 1.0, 0.0]))


@pytest.mark.parametrize("cls", [CRPCCSD])
def test_cbo_using_classes_inherit_shared_mixin(cls):
    assert CBOintegrals in cls.__mro__
    for name in ("_cbo_eri_ao", "_dipole_ao_to_mo", "_make_cbo_eris_incore",
                 "_make_cbo_eris_outcore", "ao2mo"):
        assert getattr(cls, name) is getattr(CBOintegrals, name), (
            f"{cls.__name__}.{name} no longer resolves to the shared "
            f"CBOintegrals implementation; the dedup refactor may have "
            f"regressed back into a per-class duplicate."
        )
