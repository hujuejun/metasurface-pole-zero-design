from math import pi

import numpy as np
import pytest

from pzid.fdfd import assemble_tm_operator
from pzid.grid import NonuniformGrid2D, UniformGrid2D
from pzid.pml import axial_pml_stretch


def test_uniform_operator_matches_discrete_fourier_sine_mode() -> None:
    grid = UniformGrid2D(nx=8, nz=7, dx=0.2, dz=0.15)
    epsilon = 2.25
    omega = 1.1
    transverse_order = 2
    axial_order = 3

    ix = np.arange(grid.nx)
    iz = np.arange(grid.nz)
    transverse = np.exp(2j * pi * transverse_order * ix / grid.nx)
    axial = np.sin(pi * axial_order * (iz + 1) / (grid.nz + 1))
    field = np.outer(axial, transverse).ravel()

    operator = assemble_tm_operator(omega, epsilon, grid)
    expected_eigenvalue = (
        4.0
        / epsilon
        * np.sin(pi * transverse_order / grid.nx) ** 2
        / grid.dx**2
        + 4.0
        / epsilon
        * np.sin(pi * axial_order / (2.0 * (grid.nz + 1))) ** 2
        / grid.dz**2
        - omega**2
    )
    np.testing.assert_allclose(
        operator @ field,
        expected_eigenvalue * field,
        rtol=2e-13,
        atol=2e-13,
    )


def test_real_lossless_bloch_operator_is_hermitian() -> None:
    grid = UniformGrid2D(nx=5, nz=4, dx=0.2, dz=0.25)
    epsilon = np.linspace(1.5, 3.0, grid.size).reshape(grid.shape)
    operator = assemble_tm_operator(
        omega=0.8,
        epsilon_r=epsilon,
        grid=grid,
        bloch_kx=0.37,
    )
    difference = operator - operator.getH()
    assert np.max(np.abs(difference.data), initial=0.0) < 1e-13


def test_uniform_edges_reproduce_legacy_uniform_operator() -> None:
    uniform = UniformGrid2D(nx=5, nz=7, dx=0.13, dz=0.08)
    nonuniform = NonuniformGrid2D(
        nx=uniform.nx,
        dx=uniform.dx,
        z_edges=uniform.z_edges,
    )
    epsilon = np.linspace(1.3, 3.2, uniform.size).reshape(uniform.shape)
    stretch = 1.0 + 0.03j * np.arange(uniform.nz)
    first = assemble_tm_operator(
        1.2 - 0.04j, epsilon, uniform, stretch_z=stretch
    )
    second = assemble_tm_operator(
        1.2 - 0.04j, epsilon, nonuniform, stretch_z=stretch
    )
    np.testing.assert_allclose(first.toarray(), second.toarray(), atol=1e-13)


def test_nonuniform_lossless_operator_is_hermitian() -> None:
    grid = NonuniformGrid2D(
        nx=5,
        dx=0.2,
        z_edges=np.asarray([0.0, 0.1, 0.22, 0.29, 0.43, 0.6]),
    )
    epsilon = np.linspace(1.5, 3.0, grid.size).reshape(grid.shape)
    operator = assemble_tm_operator(
        omega=0.8,
        epsilon_r=epsilon,
        grid=grid,
        bloch_kx=0.37,
    )
    difference = operator - operator.getH()
    assert np.max(np.abs(difference.data), initial=0.0) < 1e-13


def test_nonuniform_axial_faces_preserve_constant_flux_across_material_steps() -> None:
    grid = NonuniformGrid2D(
        nx=3,
        dx=0.2,
        z_edges=np.asarray([0.0, 0.08, 0.19, 0.25, 0.42, 0.55]),
    )
    epsilon_z = np.asarray([1.4, 2.3, 3.1, 1.7, 2.6])
    epsilon = np.repeat(epsilon_z[:, np.newaxis], grid.nx, axis=1)
    coefficient = 1.0 / epsilon_z
    field_z = np.zeros(grid.nz)
    for index in range(grid.nz - 1):
        field_z[index + 1] = field_z[index] + (
            0.5 * grid.dz_cells[index] / coefficient[index]
            + 0.5 * grid.dz_cells[index + 1] / coefficient[index + 1]
        )
    field = np.repeat(field_z[:, np.newaxis], grid.nx, axis=1)
    residual = (assemble_tm_operator(0.0, epsilon, grid) @ field.ravel()).reshape(
        grid.shape
    )
    np.testing.assert_allclose(residual[1:-1], 0.0, atol=2e-13)


def test_pml_operator_is_complex_symmetric_at_normal_incidence() -> None:
    grid = UniformGrid2D(nx=4, nz=10, dx=0.1, dz=0.1)
    stretch = axial_pml_stretch(omega=2.0, grid=grid, cells=3)
    operator = assemble_tm_operator(
        omega=2.0,
        epsilon_r=np.ones(grid.shape),
        grid=grid,
        stretch_z=stretch,
    )
    difference = operator - operator.T
    assert np.max(np.abs(difference.data), initial=0.0) < 1e-13
    assert np.iscomplexobj(operator.data)


@pytest.mark.parametrize(
    "epsilon",
    [0.0, np.ones((2, 2)), np.ones((3, 4))],
)
def test_invalid_material_shape_or_value_raises(epsilon: object) -> None:
    grid = UniformGrid2D(nx=3, nz=4, dx=1.0, dz=1.0)
    with pytest.raises(ValueError):
        assemble_tm_operator(1.0, epsilon, grid)
