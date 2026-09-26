import numpy as np
import pytest

from pzid.adjoint3d import maxwell_bilinear_epsilon_gradient_3d
from pzid.fdfd3d import (
    assemble_maxwell_operator_3d,
    assemble_maxwell_system_3d,
    curl_operators_3d,
    derivative_operators_3d,
    epsilon_interpolation_3d,
)
from pzid.grid3d import UniformGrid3D
from pzid.pml3d import AxialPML3D


def test_bloch_derivatives_have_expected_plane_wave_eigenvalues() -> None:
    grid = UniformGrid3D(nx=4, ny=3, nz=5, dx=0.2, dy=0.3, dz=0.1)
    kx = 0.37
    ky = -0.22
    forward, backward = derivative_operators_3d(
        grid, bloch_kx=kx, bloch_ky=ky
    )
    ix = np.arange(grid.nx)
    iy = np.arange(grid.ny)
    transverse = np.exp(
        1j * (ky * iy[:, None] * grid.dy + kx * ix[None, :] * grid.dx)
    )
    field = np.tile(transverse[None, :, :], (grid.nz, 1, 1)).ravel()
    expected_forward_x = (np.exp(1j * kx * grid.dx) - 1.0) / grid.dx
    expected_backward_y = (1.0 - np.exp(-1j * ky * grid.dy)) / grid.dy
    np.testing.assert_allclose(forward[0] @ field, expected_forward_x * field)
    np.testing.assert_allclose(backward[1] @ field, expected_backward_y * field)


def test_zero_bloch_curls_are_discrete_transposes_without_pml() -> None:
    grid = UniformGrid3D(nx=3, ny=2, nz=4, dx=0.2, dy=0.3, dz=0.1)
    curl_e, curl_h = curl_operators_3d(grid)
    difference = curl_h - curl_e.T
    assert difference.nnz == 0 or np.max(np.abs(difference.data)) < 1e-14


def test_uniform_epsilon_is_preserved_on_all_yee_components() -> None:
    grid = UniformGrid3D(nx=3, ny=2, nz=4, dx=0.2, dy=0.3, dz=0.1)
    for mapping in epsilon_interpolation_3d(grid):
        np.testing.assert_allclose(
            mapping @ np.full(grid.cell_count, 2.7), 2.7
        )


@pytest.mark.parametrize("omega", (1.7 + 0.04j, 1.7 - 0.04j))
def test_maxwell_epsilon_gradient_matches_directional_difference(
    omega: complex,
) -> None:
    grid = UniformGrid3D(nx=3, ny=2, nz=8, dx=0.12, dy=0.13, dz=0.09)
    pml = AxialPML3D(cells=2, target_round_trip_amplitude=1e-2)
    rng = np.random.default_rng(53)
    epsilon = 1.2 + rng.random(grid.shape)
    direction = rng.normal(size=grid.shape)
    left = rng.normal(size=grid.field_size) + 1j * rng.normal(
        size=grid.field_size
    )
    right = rng.normal(size=grid.field_size) + 1j * rng.normal(
        size=grid.field_size
    )
    system = assemble_maxwell_system_3d(omega, epsilon, grid, pml=pml)
    gradient = maxwell_bilinear_epsilon_gradient_3d(system, left, right)
    step = 1e-6
    plus = assemble_maxwell_operator_3d(
        omega, epsilon + step * direction, grid, pml=pml
    )
    minus = assemble_maxwell_operator_3d(
        omega, epsilon - step * direction, grid, pml=pml
    )
    numerical = left @ ((plus - minus) @ right) / (2.0 * step)
    analytic = np.sum(gradient * direction)
    assert analytic == pytest.approx(numerical, rel=3e-8, abs=5e-7)


def test_pml_staggering_has_correct_interface_samples() -> None:
    grid = UniformGrid3D(nx=2, ny=2, nz=12, dx=0.1, dy=0.1, dz=0.1)
    pml = AxialPML3D(cells=3, target_round_trip_amplitude=1e-3)
    sigma_node = pml.conductivity(grid, stagger="node")
    sigma_half = pml.conductivity(grid, stagger="half")
    assert sigma_half[0] == pytest.approx(sigma_half[-1])
    assert sigma_node[2] > 0.0
    assert sigma_node[-3] == 0.0
    assert sigma_node[0] > sigma_node[2]
