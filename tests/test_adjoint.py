import numpy as np
import pytest

from pzid.adjoint import (
    scattering_matrix_with_jacobian,
    tm_bilinear_epsilon_gradient,
)
from pzid.fdfd import assemble_tm_operator
from pzid.grid import NonuniformGrid2D, UniformGrid2D
from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering


def test_tm_operator_bilinear_gradient_matches_directional_difference() -> None:
    grid = UniformGrid2D(nx=4, nz=7, dx=0.13, dz=0.09)
    rng = np.random.default_rng(14)
    epsilon = 1.4 + 0.7 * rng.random(grid.shape)
    direction = rng.normal(size=grid.shape)
    left = rng.normal(size=grid.size) + 1j * rng.normal(size=grid.size)
    right = rng.normal(size=grid.size) + 1j * rng.normal(size=grid.size)
    stretch = 1.0 + 0.05j * np.linspace(0.0, 1.0, grid.nz)
    gradient = tm_bilinear_epsilon_gradient(
        epsilon, grid, left, right, stretch_z=stretch
    )

    step = 1e-6
    plus = assemble_tm_operator(
        1.7 + 0.04j, epsilon + step * direction, grid, stretch_z=stretch
    )
    minus = assemble_tm_operator(
        1.7 + 0.04j, epsilon - step * direction, grid, stretch_z=stretch
    )
    finite_difference = left @ ((plus - minus) @ right) / (2.0 * step)
    analytic = np.sum(gradient * direction)
    assert analytic == pytest.approx(finite_difference, rel=2e-8, abs=2e-7)


def test_nonuniform_tm_gradient_matches_directional_difference() -> None:
    grid = NonuniformGrid2D(
        nx=4,
        dx=0.13,
        z_edges=np.asarray([0.0, 0.08, 0.17, 0.23, 0.37, 0.48, 0.65]),
    )
    rng = np.random.default_rng(21)
    epsilon = 1.4 + 0.7 * rng.random(grid.shape)
    direction = rng.normal(size=grid.shape)
    left = rng.normal(size=grid.size) + 1j * rng.normal(size=grid.size)
    right = rng.normal(size=grid.size) + 1j * rng.normal(size=grid.size)
    stretch = 1.0 + 0.05j * np.linspace(0.0, 1.0, grid.nz)
    gradient = tm_bilinear_epsilon_gradient(
        epsilon, grid, left, right, stretch_z=stretch
    )

    step = 1e-6
    plus = assemble_tm_operator(
        1.7 + 0.04j, epsilon + step * direction, grid, stretch_z=stretch
    )
    minus = assemble_tm_operator(
        1.7 + 0.04j, epsilon - step * direction, grid, stretch_z=stretch
    )
    finite_difference = left @ ((plus - minus) @ right) / (2.0 * step)
    analytic = np.sum(gradient * direction)
    assert analytic == pytest.approx(finite_difference, rel=2e-8, abs=2e-7)


def _setup() -> TwoPortScatteringSetup:
    grid = UniformGrid2D(nx=4, nz=120, dx=0.09, dz=0.025)
    return TwoPortScatteringSetup(
        grid=grid,
        lower_source_plane=15,
        upper_source_plane=104,
        lower_monitor_planes=(26, 34),
        upper_monitor_planes=(85, 93),
    )


@pytest.mark.parametrize("omega", (1.9 + 0.07j, 1.9 - 0.07j))
def test_scattering_adjoint_matches_directional_difference(omega: complex) -> None:
    setup = _setup()
    epsilon = np.ones(setup.grid.shape)
    epsilon[49:72, :] = 2.7
    rng = np.random.default_rng(8)
    direction = np.zeros(setup.grid.shape)
    direction[47:74, :] = rng.normal(size=(27, setup.grid.nx))

    solution = scattering_matrix_with_jacobian(omega, epsilon, setup)
    analytic = np.einsum("abij,ij->ab", solution.jacobian, direction)
    step = 2e-6
    plus = solve_two_port_scattering(
        omega, epsilon + step * direction, setup
    ).matrix
    minus = solve_two_port_scattering(
        omega, epsilon - step * direction, setup
    ).matrix
    finite_difference = (plus - minus) / (2.0 * step)
    assert np.max(np.abs(analytic - finite_difference)) < 2e-7


def test_nonuniform_scattering_adjoint_matches_directional_difference() -> None:
    widths = np.asarray(
        [0.04] * 12
        + [0.05, 0.07, 0.09, 0.12, 0.12, 0.09, 0.07, 0.05]
        + [0.04] * 12
    )
    grid = NonuniformGrid2D(
        nx=4,
        dx=0.08,
        z_edges=np.concatenate(([0.0], np.cumsum(widths))),
    )
    setup = TwoPortScatteringSetup(
        grid=grid,
        lower_source_plane=3,
        upper_source_plane=grid.nz - 4,
        lower_monitor_planes=(6, 9),
        upper_monitor_planes=(grid.nz - 10, grid.nz - 7),
    )
    epsilon = np.ones(grid.shape)
    center = grid.nz // 2
    epsilon[center - 3 : center + 3, :] = 2.7
    rng = np.random.default_rng(18)
    direction = np.zeros(grid.shape)
    direction[center - 4 : center + 4, :] = rng.normal(size=(8, grid.nx))
    omega = 1.9 - 0.07j
    solution = scattering_matrix_with_jacobian(omega, epsilon, setup)
    analytic = np.einsum("abij,ij->ab", solution.jacobian, direction)
    step = 2e-6
    plus = solve_two_port_scattering(
        omega, epsilon + step * direction, setup
    ).matrix
    minus = solve_two_port_scattering(
        omega, epsilon - step * direction, setup
    ).matrix
    finite_difference = (plus - minus) / (2.0 * step)
    assert np.max(np.abs(analytic - finite_difference)) < 2e-7
