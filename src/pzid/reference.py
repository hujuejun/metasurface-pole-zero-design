"""Reduced uniform-port reference solves for scattering calibration."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import spsolve

from pzid.fdfd import assemble_tm_operator
from pzid.grid import Grid2D, NonuniformGrid2D, UniformGrid2D
from pzid.radiation import apply_radiation_boundaries


ComplexArray = NDArray[np.complex128]


def uniform_reference_fields(
    omega: complex,
    grid: Grid2D,
    source_planes: ArrayLike,
    *,
    refractive_index: float,
    stretch_z: ArrayLike = 1.0,
    radiation_boundaries: bool = True,
    bloch_kx: float = 0.0,
) -> ComplexArray:
    """Return full-grid fields from uniform line sources in a uniform cell.

    Only the transverse-constant Fourier component is excited. Solving it on
    a two-cell transverse grid is discretely equivalent to a full-width empty
    cell, after which the result is tiled over the requested period. Columns
    of the returned matrix correspond to ``source_planes``.
    """

    planes = np.asarray(source_planes, dtype=int).ravel()
    if planes.size == 0 or np.any(planes < 0) or np.any(planes >= grid.nz):
        raise ValueError("source_planes must lie inside the grid")
    if refractive_index <= 0.0:
        raise ValueError("refractive_index must be positive")
    if bloch_kx == 0.0 and isinstance(grid, UniformGrid2D):
        reduced_grid: Grid2D = UniformGrid2D(
            nx=2,
            nz=grid.nz,
            dx=0.5 * grid.period,
            dz=grid.dz,
        )
    elif bloch_kx == 0.0:
        reduced_grid = NonuniformGrid2D(
            nx=2,
            dx=0.5 * grid.period,
            z_edges=grid.z_edges,
            normalization_dz=grid.reference_dz,
        )
    else:
        reduced_grid = grid
    epsilon = np.full(
        reduced_grid.shape, refractive_index**2, dtype=np.complex128
    )
    operator = assemble_tm_operator(
        omega,
        epsilon,
        reduced_grid,
        stretch_z=stretch_z,
        bloch_kx=bloch_kx,
    )
    if radiation_boundaries:
        operator = apply_radiation_boundaries(
            operator,
            omega,
            reduced_grid,
            lower_index=refractive_index,
            upper_index=refractive_index,
            bloch_kx=bloch_kx,
        )
    sources = np.zeros((reduced_grid.size, planes.size), dtype=np.complex128)
    transverse_profile = np.exp(1j * float(bloch_kx) * reduced_grid.x)
    for column, plane in enumerate(planes):
        start = plane * reduced_grid.nx
        sources[start : start + reduced_grid.nx, column] = (
            transverse_profile / grid.reference_dz
        )
    reduced = np.asarray(spsolve(operator, sources))
    if reduced.ndim == 1:
        reduced = reduced[:, np.newaxis]
    if bloch_kx != 0.0:
        return np.asarray(reduced.reshape(grid.size, planes.size))
    axial = reduced.reshape(grid.nz, reduced_grid.nx, planes.size)[:, 0, :]
    full = np.broadcast_to(
        axial[:, np.newaxis, :], (grid.nz, grid.nx, planes.size)
    )
    return np.asarray(full.reshape(grid.size, planes.size))
