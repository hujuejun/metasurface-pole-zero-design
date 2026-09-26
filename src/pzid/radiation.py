"""Exact discrete Rayleigh radiation boundaries for uniform periodic ports."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

from pzid.grid import Grid2D


ComplexArray = NDArray[np.complex128]


def transverse_propagation_operator(
    omega: complex,
    grid: Grid2D,
    *,
    refractive_index: float,
    speed_of_light: float = 1.0,
    axial_spacing: float | None = None,
    bloch_kx: float = 0.0,
) -> ComplexArray:
    r"""Map a boundary row to its next outward ghost row.

    The operator diagonalizes in the discrete transverse Fourier basis. For
    order ``m``, its eigenvalue is ``exp(i k_z,m dz)``, where ``k_z,m`` obeys
    the same finite-difference dispersion as the interior operator. The branch
    is propagating outward or evanescently decaying for real positive
    frequency, and analytically continues into the nearby complex-frequency
    plane.

    The transverse basis includes the requested Bloch wavevector, so this
    remains an exact discrete radiation map at oblique incidence.
    """

    if refractive_index <= 0.0 or speed_of_light <= 0.0:
        raise ValueError("refractive_index and speed_of_light must be positive")
    if axial_spacing is None:
        lower_spacing = grid.boundary_dz("lower")
        upper_spacing = grid.boundary_dz("upper")
        if not np.isclose(lower_spacing, upper_spacing, rtol=1e-12, atol=1e-14):
            raise ValueError(
                "axial_spacing is required when boundary cell widths differ"
            )
        axial_spacing = lower_spacing
    if axial_spacing <= 0.0:
        raise ValueError("axial_spacing must be positive")
    transverse_k = (
        float(bloch_kx)
        + 2.0 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx)
    )
    discrete_kx_sq = (
        2.0 * np.sin(0.5 * transverse_k * grid.dx) / grid.dx
    ) ** 2
    axial_argument = (
        (complex(omega) * refractive_index / speed_of_light) ** 2
        - discrete_kx_sq.astype(np.complex128)
    )
    axial_discrete_k = np.sqrt(axial_argument)
    # Continue every open order from positive real k_z and every closed order
    # from positive imaginary k_z.  The explicit sheet choice is needed for
    # pole searches in the lower-half complex-frequency plane and also keeps
    # wide-supercell (multi-order) controls physically outgoing.
    real_frequency = abs(float(np.real(complex(omega))))
    open_orders = (
        (real_frequency * refractive_index / speed_of_light) ** 2
        > np.real(discrete_kx_sq)
    )
    for index, is_open in enumerate(open_orders):
        if is_open:
            if axial_discrete_k[index].real < 0.0:
                axial_discrete_k[index] *= -1.0
        elif axial_discrete_k[index].imag < 0.0:
            axial_discrete_k[index] *= -1.0
    axial_k = (
        2.0
        * np.arcsin(0.5 * axial_spacing * axial_discrete_k)
        / axial_spacing
    )
    propagation = np.exp(1j * axial_k * axial_spacing)

    fourier = np.exp(
        1j * np.outer(grid.x, transverse_k)
    ) / np.sqrt(grid.nx)
    return np.asarray(
        fourier @ np.diag(propagation) @ fourier.conj().T,
        dtype=np.complex128,
    )


def apply_radiation_boundaries(
    operator: csr_matrix,
    omega: complex,
    grid: Grid2D,
    *,
    lower_index: float,
    upper_index: float,
    speed_of_light: float = 1.0,
    bloch_kx: float = 0.0,
) -> csr_matrix:
    """Replace axial Dirichlet ghost values with outgoing port fields."""

    if operator.shape != (grid.size, grid.size):
        raise ValueError("operator shape does not match grid")
    updates: list[csr_matrix] = []
    for iz, index, side in (
        (0, lower_index, "lower"),
        (grid.nz - 1, upper_index, "upper"),
    ):
        boundary_dz = grid.boundary_dz(side)
        propagation = transverse_propagation_operator(
            omega,
            grid,
            refractive_index=index,
            speed_of_light=speed_of_light,
            axial_spacing=boundary_dz,
            bloch_kx=bloch_kx,
        )
        boundary_indices = np.asarray(
            [grid.flat_index(iz, ix) for ix in range(grid.nx)]
        )
        rows = np.repeat(boundary_indices, grid.nx)
        columns = np.tile(boundary_indices, grid.nx)
        data = (
            -(1.0 / index**2)
            * propagation.ravel()
            / (boundary_dz * grid.reference_dz)
        )
        updates.append(
            coo_matrix(
                (data, (rows, columns)),
                shape=operator.shape,
                dtype=np.complex128,
            ).tocsr()
        )
    return (operator + updates[0] + updates[1]).tocsr()
