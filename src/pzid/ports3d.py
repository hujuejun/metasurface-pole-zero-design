"""Specular plane-wave ports for doubly periodic 3D unit cells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzid.grid3d import UniformGrid3D
from pzid.ports import discrete_axial_wavenumber


ComplexArray: TypeAlias = NDArray[np.complex128]
Polarization = Literal["x", "y"]


@dataclass(frozen=True)
class SpecularElectricAmplitudes3D:
    """Global coefficients of forward and backward electric plane waves."""

    forward: complex
    backward: complex


def _component_index(polarization: Polarization) -> int:
    if polarization == "x":
        return 0
    if polarization == "y":
        return 1
    raise ValueError("polarization must be 'x' or 'y'")


def specular_e_projection_rows_3d(
    grid: UniformGrid3D,
    *,
    planes: tuple[int, int],
    omega: complex,
    refractive_index: float,
    polarization: Polarization,
    speed_of_light: float = 1.0,
) -> tuple[ComplexArray, ComplexArray]:
    r"""Return rows extracting the normal-incidence zeroth diffraction order.

    The rows are complex-bilinear (no field conjugation), so they remain
    analytic functions of complex frequency.  A two-plane fit separates
    ``exp(+i k_z z)`` and ``exp(-i k_z z)`` electric-field amplitudes.
    """

    first, second = planes
    if not 0 <= first < grid.nz or not 0 <= second < grid.nz:
        raise ValueError("monitor plane is outside the grid")
    if first == second:
        raise ValueError("two distinct monitor planes are required")
    if refractive_index <= 0.0:
        raise ValueError("refractive_index must be positive")

    wave_number = discrete_axial_wavenumber(
        omega,
        refractive_index=refractive_index,
        dz=grid.dz,
        speed_of_light=speed_of_light,
    )
    z = np.asarray(
        (grid.tangential_e_z[first], grid.tangential_e_z[second])
    )
    mode_matrix = np.column_stack(
        (np.exp(1j * wave_number * z), np.exp(-1j * wave_number * z))
    )
    condition = np.linalg.cond(mode_matrix)
    if not np.isfinite(condition) or condition > 1e10:
        raise ValueError("monitor planes give an ill-conditioned decomposition")

    sampling = np.zeros((2, grid.field_size), dtype=np.complex128)
    component_offset = _component_index(polarization) * grid.cell_count
    plane_size = grid.nx * grid.ny
    for row, plane in enumerate((first, second)):
        start = component_offset + plane * plane_size
        sampling[row, start : start + plane_size] = 1.0 / plane_size
    rows = np.linalg.solve(mode_matrix, sampling)
    return np.asarray(rows[0]), np.asarray(rows[1])


def decompose_specular_e_3d(
    field: ArrayLike,
    grid: UniformGrid3D,
    *,
    planes: tuple[int, int],
    omega: complex,
    refractive_index: float,
    polarization: Polarization,
    speed_of_light: float = 1.0,
) -> SpecularElectricAmplitudes3D:
    """Decompose one polarization of a vector electric field."""

    values = np.asarray(field, dtype=np.complex128)
    if values.size != grid.field_size:
        raise ValueError("field must contain grid.field_size values")
    forward, backward = specular_e_projection_rows_3d(
        grid,
        planes=planes,
        omega=omega,
        refractive_index=refractive_index,
        polarization=polarization,
        speed_of_light=speed_of_light,
    )
    flattened = values.reshape(grid.field_size)
    return SpecularElectricAmplitudes3D(
        forward=complex(forward @ flattened),
        backward=complex(backward @ flattened),
    )
