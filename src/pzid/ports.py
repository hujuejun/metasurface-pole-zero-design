"""Analytic specular-port modes in locally uniform axial regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzid.grid import Grid2D


@dataclass(frozen=True)
class SpecularAmplitudes:
    """Global coefficients of forward and backward scalar ``H_y`` waves."""

    forward: complex
    backward: complex


ComplexArray = NDArray[np.complex128]


def discrete_axial_wavenumber(
    omega: complex,
    *,
    refractive_index: float,
    dz: float,
    transverse_wavenumber: float = 0.0,
    transverse_spacing: float | None = None,
    speed_of_light: float = 1.0,
) -> complex:
    r"""Return the axial wavenumber of the uniform discrete TM operator.

    At normal incidence, the cell-centered finite-volume dispersion is

    .. math::

       4\sin^2(k_z\Delta z/2)/\Delta z^2
       = n^2 (\omega/c)^2.

    The principal analytic branch is appropriate around positive frequencies
    below the grid's first axial Nyquist threshold.
    """

    if refractive_index <= 0.0 or dz <= 0.0 or speed_of_light <= 0.0:
        raise ValueError("index, dz, and speed_of_light must be positive")
    if transverse_spacing is None:
        if transverse_wavenumber != 0.0:
            raise ValueError(
                "transverse_spacing is required for nonzero transverse_wavenumber"
            )
        discrete_kx_sq = 0.0
    else:
        if transverse_spacing <= 0.0:
            raise ValueError("transverse_spacing must be positive")
        discrete_kx_sq = (
            2.0
            * np.sin(0.5 * float(transverse_wavenumber) * transverse_spacing)
            / transverse_spacing
        ) ** 2
    axial_discrete_k = np.sqrt(
        (complex(omega) * refractive_index / speed_of_light) ** 2
        - discrete_kx_sq
    )
    if axial_discrete_k.real < 0.0:
        axial_discrete_k *= -1.0
    return complex(2.0 * np.arcsin(0.5 * dz * axial_discrete_k) / dz)


def decompose_specular_h(
    field: ArrayLike,
    grid: Grid2D,
    *,
    planes: tuple[int, int],
    omega: complex,
    refractive_index: float,
    bloch_kx: float = 0.0,
    speed_of_light: float = 1.0,
) -> SpecularAmplitudes:
    r"""Decompose the zeroth Fourier order into ``+z`` and ``-z`` waves.

    The returned coefficients ``A+`` and ``A-`` use a global axial phase
    origin and satisfy

    .. math::

       \langle H_y(z)\rangle_x = A_+ e^{i k_z z} + A_- e^{-i k_z z}.

    Monitor planes must lie in the same uniform, non-PML port region.
    """

    values = np.asarray(field, dtype=np.complex128)
    if values.shape != grid.shape:
        raise ValueError(f"field must have shape {grid.shape}")
    first, second = planes
    if not 0 <= first < grid.nz or not 0 <= second < grid.nz:
        raise ValueError("monitor plane is outside the grid")
    if first == second:
        raise ValueError("two distinct monitor planes are required")

    forward_row, backward_row = specular_projection_rows(
        grid,
        planes=planes,
        omega=omega,
        refractive_index=refractive_index,
        bloch_kx=bloch_kx,
        speed_of_light=speed_of_light,
    )
    flattened = values.ravel()
    return SpecularAmplitudes(
        complex(forward_row @ flattened),
        complex(backward_row @ flattened),
    )


def specular_projection_rows(
    grid: Grid2D,
    *,
    planes: tuple[int, int],
    omega: complex,
    refractive_index: float,
    bloch_kx: float = 0.0,
    speed_of_light: float = 1.0,
) -> tuple[ComplexArray, ComplexArray]:
    r"""Return analytic linear rows for the two specular coefficients.

    If the flattened field is ``u``, the returned rows satisfy
    ``A_plus = forward_row @ u`` and ``A_minus = backward_row @ u``.
    The products are bilinear rather than Hermitian: no complex conjugation is
    introduced, so the projections remain analytic in complex frequency.
    """

    first, second = planes
    if not 0 <= first < grid.nz or not 0 <= second < grid.nz:
        raise ValueError("monitor plane is outside the grid")
    if first == second:
        raise ValueError("two distinct monitor planes are required")

    port_dz = grid.uniform_dz_for_planes((first, second))
    wave_number = discrete_axial_wavenumber(
        omega,
        refractive_index=refractive_index,
        dz=port_dz,
        transverse_wavenumber=bloch_kx,
        transverse_spacing=grid.dx,
        speed_of_light=speed_of_light,
    )
    z = np.asarray([grid.z[first], grid.z[second]])
    mode_matrix = np.column_stack(
        (np.exp(1j * wave_number * z), np.exp(-1j * wave_number * z))
    )
    condition = np.linalg.cond(mode_matrix)
    if not np.isfinite(condition) or condition > 1e10:
        raise ValueError("monitor planes give an ill-conditioned decomposition")
    inverse = np.linalg.inv(mode_matrix)
    sampling = np.zeros((2, grid.size), dtype=np.complex128)
    transverse_projection = np.exp(-1j * float(bloch_kx) * grid.x) / grid.nx
    for sample, plane in enumerate((first, second)):
        start = plane * grid.nx
        sampling[sample, start : start + grid.nx] = transverse_projection
    rows = inverse @ sampling
    return np.asarray(rows[0]), np.asarray(rows[1])
