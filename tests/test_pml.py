import numpy as np
import pytest

from pzid.grid import UniformGrid2D
from pzid.pml import axial_pml_stretch


def test_pml_is_symmetric_and_matches_interior() -> None:
    grid = UniformGrid2D(nx=4, nz=20, dx=0.1, dz=0.1)
    stretch = axial_pml_stretch(omega=2.0, grid=grid, cells=4)
    np.testing.assert_allclose(stretch, stretch[::-1])
    np.testing.assert_allclose(stretch[4:-4], 1.0)
    assert np.all(stretch.imag[:4] > 0.0)
    assert stretch.imag[0] > stretch.imag[3]


def test_zero_cell_pml_is_identity() -> None:
    grid = UniformGrid2D(nx=2, nz=4, dx=1.0, dz=1.0)
    np.testing.assert_array_equal(
        axial_pml_stretch(omega=1.0, grid=grid, cells=0),
        np.ones(grid.nz),
    )


@pytest.mark.parametrize(
    ("omega", "cells"),
    [(0.0, 1), (1.0, -1), (1.0, 5)],
)
def test_invalid_pml_parameters_raise(omega: complex, cells: int) -> None:
    grid = UniformGrid2D(nx=2, nz=8, dx=1.0, dz=1.0)
    with pytest.raises(ValueError):
        axial_pml_stretch(omega=omega, grid=grid, cells=cells)

