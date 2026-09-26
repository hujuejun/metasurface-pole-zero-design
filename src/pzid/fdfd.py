"""Sparse cell-centered FDFD assembly for the scalar TM equation."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix

from pzid.grid import Grid2D


ComplexArray: TypeAlias = NDArray[np.complex128]


def _as_grid_array(
    value: ArrayLike,
    grid: Grid2D,
    *,
    name: str,
) -> ComplexArray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim == 0:
        return np.full(grid.shape, array, dtype=np.complex128)
    if array.shape != grid.shape:
        raise ValueError(f"{name} must be scalar or have shape {grid.shape}")
    return array


def _harmonic_face(left: complex, right: complex) -> complex:
    """Effective flux coefficient between two cell centers."""

    denominator = left + right
    if denominator == 0.0:
        raise ValueError("adjacent flux coefficients have zero harmonic sum")
    return 2.0 * left * right / denominator


def _axial_face_conductance(
    lower: complex,
    upper: complex,
    lower_width: float,
    upper_width: float,
) -> complex:
    """Flux conductance across unequal half-cell distances."""

    denominator = lower_width * upper + upper_width * lower
    if denominator == 0.0:
        raise ValueError("adjacent axial coefficients have zero weighted sum")
    return 2.0 * lower * upper / denominator


def assemble_tm_operator(
    omega: complex,
    epsilon_r: ArrayLike,
    grid: Grid2D,
    *,
    mu_r: ArrayLike = 1.0,
    bloch_kx: float = 0.0,
    stretch_z: ArrayLike = 1.0,
    speed_of_light: float = 1.0,
) -> csr_matrix:
    r"""Assemble the TM operator on a cell-centered finite-volume grid.

    The discretized equation is

    .. math::

       [-\partial_x(a_x\partial_x)-\partial_z(a_z\partial_z)
        - k_0^2 m] H_y = J,

    where ``a_x = s_z / epsilon_r``, ``a_z = 1/(s_z epsilon_r)``,
    and ``m = mu_r s_z`` for an axial coordinate stretch. The transverse
    boundary is Bloch periodic. Homogeneous Dirichlet termination is used
    beyond the two axial edge cells; in scattering calculations it lies behind
    the PML. Flux coefficients at material faces use their harmonic mean,
    which enforces the TM interface condition for cell-centered materials.
    """

    if speed_of_light <= 0.0:
        raise ValueError("speed_of_light must be positive")
    epsilon = _as_grid_array(epsilon_r, grid, name="epsilon_r")
    permeability = _as_grid_array(mu_r, grid, name="mu_r")
    if np.any(epsilon == 0.0):
        raise ValueError("epsilon_r must be nonzero")

    stretch = np.asarray(stretch_z, dtype=np.complex128)
    if stretch.ndim == 0:
        stretch = np.full(grid.nz, stretch, dtype=np.complex128)
    if stretch.shape != (grid.nz,):
        raise ValueError(f"stretch_z must be scalar or have shape {(grid.nz,)}")
    if np.any(stretch == 0.0):
        raise ValueError("stretch_z must be nonzero")

    stretch_2d = stretch[:, np.newaxis]
    coefficient_x = stretch_2d / epsilon
    coefficient_z = 1.0 / (stretch_2d * epsilon)
    # The axial cell-volume weighting produces a complex-symmetric weak-form
    # matrix on nonuniform grids.  Division by one fixed reference length is
    # a harmless global equation scaling and recovers the legacy uniform-grid
    # matrix exactly.
    axial_volume_weights = grid.dz_cells / grid.reference_dz
    mass = permeability * stretch_2d * axial_volume_weights[:, np.newaxis]

    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    diagonal = np.zeros(grid.size, dtype=np.complex128)
    phase = np.exp(1j * bloch_kx * grid.period)

    def add_off_diagonal(row: int, column: int, value: complex) -> None:
        rows.append(row)
        columns.append(column)
        data.append(value)

    for iz in range(grid.nz):
        for ix in range(grid.nx):
            row = grid.flat_index(iz, ix)

            # Add each transverse face once. The paired entries enforce the
            # Bloch phases H(x+L) = exp(i k_x L) H(x).
            right_ix = (ix + 1) % grid.nx
            neighbor = grid.flat_index(iz, right_ix)
            face = _harmonic_face(
                coefficient_x[iz, ix], coefficient_x[iz, right_ix]
            ) * axial_volume_weights[iz] / grid.dx**2
            diagonal[row] += face
            diagonal[neighbor] += face
            if ix == grid.nx - 1:
                add_off_diagonal(row, neighbor, -face * phase)
                add_off_diagonal(neighbor, row, -face / phase)
            else:
                add_off_diagonal(row, neighbor, -face)
                add_off_diagonal(neighbor, row, -face)

            # Add each interior axial face once.
            if iz < grid.nz - 1:
                neighbor = grid.flat_index(iz + 1, ix)
                face = _axial_face_conductance(
                    coefficient_z[iz, ix],
                    coefficient_z[iz + 1, ix],
                    float(grid.dz_cells[iz]),
                    float(grid.dz_cells[iz + 1]),
                ) / grid.reference_dz
                diagonal[row] += face
                diagonal[neighbor] += face
                add_off_diagonal(row, neighbor, -face)
                add_off_diagonal(neighbor, row, -face)

            # Homogeneous Dirichlet ghost cells terminate the axial domain.
            if iz == 0:
                diagonal[row] += coefficient_z[iz, ix] / (
                    grid.boundary_dz("lower") * grid.reference_dz
                )
            elif iz == grid.nz - 1:
                diagonal[row] += coefficient_z[iz, ix] / (
                    grid.boundary_dz("upper") * grid.reference_dz
                )

    wave_number = complex(omega) / speed_of_light
    diagonal -= wave_number**2 * mass.ravel()
    indices = np.arange(grid.size)
    rows.extend(indices.tolist())
    columns.extend(indices.tolist())
    data.extend(diagonal.tolist())
    return coo_matrix(
        (np.asarray(data), (np.asarray(rows), np.asarray(columns))),
        shape=(grid.size, grid.size),
        dtype=np.complex128,
    ).tocsr()
