"""Full-vector Yee-grid FDFD for doubly periodic three-dimensional cells."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import (
    bmat,
    coo_matrix,
    csr_matrix,
    diags,
    eye,
    kron,
)

from pzid.grid3d import UniformGrid3D
from pzid.pml3d import AxialPML3D


ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass(frozen=True)
class MaxwellOperator3D:
    """Assembled Maxwell operator and material maps at one frequency."""

    omega: complex
    matrix: csr_matrix
    curl_e: csr_matrix
    curl_h: csr_matrix
    cell_epsilon: ComplexArray
    edge_epsilon: ComplexArray
    epsilon_scale: ComplexArray
    inverse_mu: ComplexArray
    interpolation: tuple[csr_matrix, csr_matrix, csr_matrix]
    grid: UniformGrid3D
    speed_of_light: float


def _periodic_forward_difference(
    count: int,
    spacing: float,
    bloch_phase: complex,
) -> csr_matrix:
    rows = np.repeat(np.arange(count), 2)
    columns = np.empty(2 * count, dtype=int)
    data = np.empty(2 * count, dtype=np.complex128)
    for index in range(count):
        columns[2 * index] = index
        data[2 * index] = -1.0 / spacing
        columns[2 * index + 1] = (index + 1) % count
        data[2 * index + 1] = (
            bloch_phase / spacing if index == count - 1 else 1.0 / spacing
        )
    return coo_matrix((data, (rows, columns)), shape=(count, count)).tocsr()


def _periodic_backward_difference(
    count: int,
    spacing: float,
    bloch_phase: complex,
) -> csr_matrix:
    rows = np.repeat(np.arange(count), 2)
    columns = np.empty(2 * count, dtype=int)
    data = np.empty(2 * count, dtype=np.complex128)
    for index in range(count):
        columns[2 * index] = index
        data[2 * index] = 1.0 / spacing
        columns[2 * index + 1] = (index - 1) % count
        data[2 * index + 1] = (
            -1.0 / (bloch_phase * spacing)
            if index == 0
            else -1.0 / spacing
        )
    return coo_matrix((data, (rows, columns)), shape=(count, count)).tocsr()


def _open_forward_difference(count: int, spacing: float) -> csr_matrix:
    """Forward difference with a zero ghost sample above the domain."""

    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    for index in range(count):
        rows.append(index)
        columns.append(index)
        data.append(-1.0 / spacing)
        if index + 1 < count:
            rows.append(index)
            columns.append(index + 1)
            data.append(1.0 / spacing)
    return coo_matrix((data, (rows, columns)), shape=(count, count)).tocsr()


def _open_backward_difference(count: int, spacing: float) -> csr_matrix:
    """Backward difference with a zero ghost sample below the domain."""

    rows: list[int] = []
    columns: list[int] = []
    data: list[complex] = []
    for index in range(count):
        rows.append(index)
        columns.append(index)
        data.append(1.0 / spacing)
        if index > 0:
            rows.append(index)
            columns.append(index - 1)
            data.append(-1.0 / spacing)
    return coo_matrix((data, (rows, columns)), shape=(count, count)).tocsr()


def derivative_operators_3d(
    grid: UniformGrid3D,
    *,
    bloch_kx: float = 0.0,
    bloch_ky: float = 0.0,
    speed_of_light: float = 1.0,
) -> tuple[
    tuple[csr_matrix, csr_matrix, csr_matrix],
    tuple[csr_matrix, csr_matrix, csr_matrix],
]:
    """Return forward and backward Yee derivatives in ``x, y, z`` order."""

    if speed_of_light <= 0.0:
        raise ValueError("speed_of_light must be positive")
    phase_x = np.exp(1j * bloch_kx * grid.period_x)
    phase_y = np.exp(1j * bloch_ky * grid.period_y)
    forward_x_1d = _periodic_forward_difference(grid.nx, grid.dx, phase_x)
    backward_x_1d = _periodic_backward_difference(grid.nx, grid.dx, phase_x)
    forward_y_1d = _periodic_forward_difference(grid.ny, grid.dy, phase_y)
    backward_y_1d = _periodic_backward_difference(grid.ny, grid.dy, phase_y)
    forward_z_1d = _open_forward_difference(grid.nz, grid.dz)
    backward_z_1d = _open_backward_difference(grid.nz, grid.dz)

    identity_x = eye(grid.nx, format="csr", dtype=np.complex128)
    identity_y = eye(grid.ny, format="csr", dtype=np.complex128)
    identity_z = eye(grid.nz, format="csr", dtype=np.complex128)
    forward_x = kron(
        identity_z, kron(identity_y, forward_x_1d, format="csr"), format="csr"
    )
    backward_x = kron(
        identity_z, kron(identity_y, backward_x_1d, format="csr"), format="csr"
    )
    forward_y = kron(
        identity_z, kron(forward_y_1d, identity_x, format="csr"), format="csr"
    )
    backward_y = kron(
        identity_z, kron(backward_y_1d, identity_x, format="csr"), format="csr"
    )
    forward_z = kron(
        forward_z_1d, kron(identity_y, identity_x, format="csr"), format="csr"
    )
    backward_z = kron(
        backward_z_1d, kron(identity_y, identity_x, format="csr"), format="csr"
    )

    return (
        (forward_x, forward_y, forward_z),
        (backward_x, backward_y, backward_z),
    )


def curl_operators_3d(
    grid: UniformGrid3D,
    *,
    bloch_kx: float = 0.0,
    bloch_ky: float = 0.0,
    speed_of_light: float = 1.0,
) -> tuple[csr_matrix, csr_matrix]:
    """Return electric-to-magnetic and magnetic-to-electric curls."""

    forward, backward = derivative_operators_3d(
        grid,
        bloch_kx=bloch_kx,
        bloch_ky=bloch_ky,
        speed_of_light=speed_of_light,
    )
    dxf, dyf, dzf = forward
    dxb, dyb, dzb = backward
    zero = csr_matrix((grid.cell_count, grid.cell_count), dtype=np.complex128)
    curl_e = bmat(
        ((zero, -dzf, dyf), (dzf, zero, -dxf), (-dyf, dxf, zero)),
        format="csr",
    )
    curl_h = bmat(
        ((zero, -dzb, dyb), (dzb, zero, -dxb), (-dyb, dxb, zero)),
        format="csr",
    )
    return curl_e, curl_h


@lru_cache(maxsize=32)
def _epsilon_interpolation_cached(
    nx: int, ny: int, nz: int
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Cell-to-Yee interpolation for isotropic material permittivity."""

    count = nx * ny * nz

    def flat(iz: int, iy: int, ix: int) -> int:
        return (iz * ny + iy) * nx + ix

    def component_matrix(component: int) -> csr_matrix:
        rows: list[int] = []
        columns: list[int] = []
        data: list[float] = []
        for iz in range(nz):
            z_neighbors = (iz,) if iz == 0 else (iz - 1, iz)
            for iy in range(ny):
                for ix in range(nx):
                    if component == 0:  # Ex: average perpendicular y-z cells.
                        candidates = [
                            (kz, jy, ix)
                            for kz in z_neighbors
                            for jy in ((iy - 1) % ny, iy)
                        ]
                    elif component == 1:  # Ey: average perpendicular x-z cells.
                        candidates = [
                            (kz, iy, jx)
                            for kz in z_neighbors
                            for jx in ((ix - 1) % nx, ix)
                        ]
                    else:  # Ez: average perpendicular x-y cells.
                        candidates = [
                            (iz, jy, jx)
                            for jy in ((iy - 1) % ny, iy)
                            for jx in ((ix - 1) % nx, ix)
                        ]
                    unique = tuple(dict.fromkeys(candidates))
                    weight = 1.0 / len(unique)
                    row = flat(iz, iy, ix)
                    for cell in unique:
                        rows.append(row)
                        columns.append(flat(*cell))
                        data.append(weight)
        return coo_matrix(
            (data, (rows, columns)), shape=(count, count), dtype=float
        ).tocsr()

    return component_matrix(0), component_matrix(1), component_matrix(2)


def epsilon_interpolation_3d(
    grid: UniformGrid3D,
) -> tuple[csr_matrix, csr_matrix, csr_matrix]:
    """Return sparse maps from cell epsilon to each electric Yee component."""

    return _epsilon_interpolation_cached(grid.nx, grid.ny, grid.nz)


def _as_cell_array(value: ArrayLike, grid: UniformGrid3D) -> ComplexArray:
    values = np.asarray(value, dtype=np.complex128)
    if values.ndim == 0:
        return np.full(grid.shape, values, dtype=np.complex128)
    if values.shape != grid.shape:
        raise ValueError(f"epsilon_r must be scalar or have shape {grid.shape}")
    return values


def assemble_maxwell_system_3d(
    omega: complex,
    epsilon_r: ArrayLike,
    grid: UniformGrid3D,
    *,
    mu_r: float = 1.0,
    bloch_kx: float = 0.0,
    bloch_ky: float = 0.0,
    pml: AxialPML3D | None = None,
    speed_of_light: float = 1.0,
) -> MaxwellOperator3D:
    r"""Assemble ``curl mu^-1 curl E - (omega/c)^2 epsilon E``.

    Materials are isotropic and supplied on cell centers.  Arithmetic maps
    transfer permittivity to the three Yee electric-component locations.  The
    transverse directions are Bloch periodic and the axial zero-ghost edges
    are placed behind the PML.
    """

    if complex(omega) == 0.0:
        raise ValueError("omega must be nonzero")
    if speed_of_light <= 0.0:
        raise ValueError("speed_of_light must be positive")
    if not np.isfinite(mu_r) or mu_r <= 0.0:
        raise ValueError("mu_r must be a finite positive scalar")
    epsilon = _as_cell_array(epsilon_r, grid)
    if np.any(epsilon == 0.0) or not np.all(np.isfinite(epsilon)):
        raise ValueError("epsilon_r must be finite and nonzero")
    if pml is not None:
        pml.validate(grid)

    curl_e, curl_h = curl_operators_3d(
        grid,
        bloch_kx=bloch_kx,
        bloch_ky=bloch_ky,
        speed_of_light=speed_of_light,
    )
    interpolation = epsilon_interpolation_3d(grid)
    flattened = epsilon.ravel()
    interpolated_epsilon = np.vstack(
        [np.asarray(mapping @ flattened) for mapping in interpolation]
    )
    if pml is None or pml.cells == 0:
        stretch_node = np.ones(grid.nz, dtype=np.complex128)
        stretch_half = np.ones(grid.nz, dtype=np.complex128)
    else:
        stretch_node, stretch_half = pml.stretches(
            omega, grid, speed_of_light=speed_of_light
        )
    repeat = grid.nx * grid.ny
    stretch_node_full = np.repeat(stretch_node, repeat)
    stretch_half_full = np.repeat(stretch_half, repeat)
    # Uniaxial transformation-optics PML tensors for a z stretch:
    # epsilon = eps * diag(s, s, 1/s), and likewise for mu.
    epsilon_scale = np.vstack(
        (
            stretch_node_full,
            stretch_node_full,
            1.0 / stretch_half_full,
        )
    )
    inverse_mu = np.vstack(
        (
            1.0 / (mu_r * stretch_half_full),
            1.0 / (mu_r * stretch_half_full),
            stretch_node_full / mu_r,
        )
    )
    edge_epsilon = interpolated_epsilon * epsilon_scale
    mass = diags(edge_epsilon.ravel(), format="csr")
    inverse_mu_matrix = diags(inverse_mu.ravel(), format="csr")
    wave_number = complex(omega) / speed_of_light
    matrix = (
        curl_h @ inverse_mu_matrix @ curl_e - wave_number**2 * mass
    ).tocsr()
    matrix.eliminate_zeros()
    return MaxwellOperator3D(
        omega=complex(omega),
        matrix=matrix,
        curl_e=curl_e,
        curl_h=curl_h,
        cell_epsilon=epsilon,
        edge_epsilon=np.asarray(edge_epsilon),
        epsilon_scale=np.asarray(epsilon_scale),
        inverse_mu=np.asarray(inverse_mu),
        interpolation=interpolation,
        grid=grid,
        speed_of_light=float(speed_of_light),
    )


def assemble_maxwell_operator_3d(
    omega: complex,
    epsilon_r: ArrayLike,
    grid: UniformGrid3D,
    **kwargs: object,
) -> csr_matrix:
    """Return only the sparse matrix from :func:`assemble_maxwell_system_3d`."""

    return assemble_maxwell_system_3d(
        omega, epsilon_r, grid, **kwargs
    ).matrix
