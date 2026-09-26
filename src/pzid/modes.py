"""Diagnostics for classifying two-dimensional TM resonances."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzid.grid import UniformGrid2D
from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering


ComplexArray = NDArray[np.complex128]


def resonant_residue_field(
    pole: complex,
    coupling_ratio: complex,
    epsilon_r: ArrayLike,
    setup: TwoPortScatteringSetup,
    *,
    relative_offset: float = 1e-5,
) -> ComplexArray:
    """Return a pole-residue field from a nearby complex-frequency solve.

    Multiplication by the complex-frequency offset suppresses the analytic
    driven background and leaves the resonant field as the offset tends to
    zero. The returned normalization is arbitrary.
    """

    if complex(pole).imag >= 0.0:
        raise ValueError("pole must lie below the real axis")
    if relative_offset <= 0.0:
        raise ValueError("relative_offset must be positive")
    scale = max(abs(complex(pole).imag), abs(complex(pole).real), 1.0)
    offset = relative_offset * scale * (1.0 + 0.37j)
    solution = solve_two_port_scattering(
        complex(pole) + offset, epsilon_r, setup
    )
    incoming = np.asarray((1.0, complex(coupling_ratio)))
    return np.asarray(
        offset * (solution.fields @ incoming).reshape(setup.grid.shape)
    )


def scattering_pole_residue(
    pole: complex,
    epsilon_r: ArrayLike,
    setup: TwoPortScatteringSetup,
    *,
    relative_offset: float = 1e-5,
) -> ComplexArray:
    """Sample the rank-one scattering residue of an isolated simple pole."""

    if complex(pole).imag >= 0.0:
        raise ValueError("pole must lie below the real axis")
    if relative_offset <= 0.0:
        raise ValueError("relative_offset must be positive")
    scale = max(abs(complex(pole).imag), abs(complex(pole).real), 1.0)
    offset = relative_offset * scale * (1.0 + 0.37j)
    matrix = solve_two_port_scattering(
        complex(pole) + offset, epsilon_r, setup
    ).matrix
    residue = offset * matrix
    return np.asarray(0.5 * (residue + residue.T))


def two_port_residue_diagnostics(residue: ArrayLike) -> dict[str, float | complex]:
    """Return rank and signed radiation-parity diagnostics for a 2x2 residue.

    In the project's global port convention an axially even ``H_y`` mode has
    an upper/lower coupling ratio of ``-1`` and an odd ``H_y`` mode has ratio
    ``+1``.  The signed index is therefore +1 for even and -1 for odd.
    """

    values = np.asarray(residue, dtype=np.complex128)
    if values.shape != (2, 2):
        raise ValueError("residue must have shape (2, 2)")
    vectors, singular_values, _ = np.linalg.svd(values)
    if singular_values[0] == 0.0 or vectors[0, 0] == 0.0:
        raise ValueError("residue must have a nonzero dominant coupling vector")
    coupling = np.asarray(vectors[:, 0])
    ratio = complex(coupling[1] / coupling[0])
    even_basis = np.asarray((1.0, -1.0), dtype=np.complex128) / np.sqrt(2.0)
    odd_basis = np.asarray((1.0, 1.0), dtype=np.complex128) / np.sqrt(2.0)
    even_fraction = float(abs(np.vdot(even_basis, coupling)) ** 2)
    odd_fraction = float(abs(np.vdot(odd_basis, coupling)) ** 2)
    normalization = max(even_fraction + odd_fraction, np.finfo(float).tiny)
    even_fraction /= normalization
    odd_fraction /= normalization
    return {
        "upper_to_lower_coupling_ratio": ratio,
        "even_hy_radiation_fraction": even_fraction,
        "odd_hy_radiation_fraction": odd_fraction,
        "signed_hy_parity_index": even_fraction - odd_fraction,
        "second_to_first_singular_value": float(
            singular_values[1] / singular_values[0]
        ),
    }


def phase_align(field: ArrayLike) -> ComplexArray:
    """Rotate a complex field so its largest sample is positive real."""

    values = np.asarray(field, dtype=np.complex128)
    index = np.unravel_index(np.argmax(np.abs(values)), values.shape)
    return np.asarray(values * np.exp(-1j * np.angle(values[index])))


def parity_errors(field: ArrayLike) -> tuple[float, float]:
    """Return normalized even and odd errors under axial reflection."""

    values = np.asarray(field, dtype=np.complex128)
    norm = max(float(np.linalg.norm(values)), np.finfo(float).tiny)
    return (
        float(np.linalg.norm(values - np.flipud(values)) / norm),
        float(np.linalg.norm(values + np.flipud(values)) / norm),
    )


def tm_electric_field(
    h_y: ArrayLike,
    omega: complex,
    epsilon_r: ArrayLike,
    grid: UniformGrid2D,
) -> tuple[ComplexArray, ComplexArray]:
    r"""Recover ``(E_x, E_z)`` for the project's ``exp(-i omega t)`` TM field."""

    magnetic = np.asarray(h_y, dtype=np.complex128)
    epsilon = np.asarray(epsilon_r, dtype=np.complex128)
    if magnetic.shape != grid.shape or epsilon.shape != grid.shape:
        raise ValueError("field and epsilon_r must match the grid")
    if complex(omega) == 0.0 or np.any(epsilon == 0.0):
        raise ValueError("omega and epsilon_r must be nonzero")
    derivative_x = (
        np.roll(magnetic, -1, axis=1) - np.roll(magnetic, 1, axis=1)
    ) / (2.0 * grid.dx)
    derivative_z = np.gradient(magnetic, grid.dz, axis=0, edge_order=2)
    e_x = -1j * derivative_z / (complex(omega) * epsilon)
    e_z = 1j * derivative_x / (complex(omega) * epsilon)
    return np.asarray(e_x), np.asarray(e_z)


def polarization_diagnostics(
    h_y: ArrayLike,
    omega: complex,
    epsilon_r: ArrayLike,
    grid: UniformGrid2D,
    *,
    background_index: float,
) -> tuple[dict[str, float], tuple[ComplexArray, ComplexArray]]:
    """Return origin-declared 2D polarization-current moment diagnostics.

    The moment measures classify symmetry-allowed dipolar radiation. They are
    not asserted to be a complete multipole expansion of a periodic lattice.
    """

    epsilon = np.asarray(epsilon_r, dtype=np.complex128)
    e_x, e_z = tm_electric_field(h_y, omega, epsilon, grid)
    contrast = epsilon - float(background_index) ** 2
    p_x_density = contrast * e_x
    p_z_density = contrast * e_z
    area = grid.dx * grid.dz
    norm_x = max(float(np.sum(np.abs(p_x_density)) * area), np.finfo(float).tiny)
    norm_z = max(float(np.sum(np.abs(p_z_density)) * area), np.finfo(float).tiny)
    vector_norm = max(
        float(
            np.sum(np.sqrt(np.abs(p_x_density) ** 2 + np.abs(p_z_density) ** 2))
            * area
        ),
        np.finfo(float).tiny,
    )
    electric_x = np.sum(p_x_density) * area
    electric_z = np.sum(p_z_density) * area
    x = grid.x[np.newaxis, :] - 0.5 * grid.period
    z = grid.z[:, np.newaxis] - 0.5 * grid.height
    magnetic_y_integral = np.sum(z * p_x_density - x * p_z_density) * area
    length_scale = max(grid.period, np.finfo(float).tiny)
    diagnostics = {
        "electric_x_coherence": float(abs(electric_x) / norm_x),
        "electric_z_coherence": float(abs(electric_z) / norm_z),
        "magnetic_y_moment_normalized": float(
            abs(magnetic_y_integral) / (length_scale * vector_norm)
        ),
        "polarization_norm": vector_norm,
    }
    return diagnostics, (np.asarray(p_x_density), np.asarray(p_z_density))


def transverse_harmonic_fractions(
    field: ArrayLike, rows: tuple[int, int]
) -> dict[str, float]:
    """Partition field energy into zeroth, first, and higher harmonics."""

    values = np.asarray(field, dtype=np.complex128)
    if values.ndim != 2:
        raise ValueError("field must be two-dimensional")
    start, stop = rows
    if not 0 <= start < stop <= values.shape[0]:
        raise ValueError("rows must select a nonempty field interval")
    spectrum = np.fft.fft(values[start:stop], axis=1) / values.shape[1]
    power = np.sum(np.abs(spectrum) ** 2, axis=0)
    total = max(float(np.sum(power)), np.finfo(float).tiny)
    first = float(power[1] + power[-1]) if values.shape[1] > 2 else 0.0
    zeroth_fraction = float(power[0] / total)
    first_fraction = first / total
    return {
        "zeroth_harmonic_fraction": zeroth_fraction,
        "first_harmonic_pair_fraction": first_fraction,
        "higher_harmonic_fraction": max(
            0.0, 1.0 - zeroth_fraction - first_fraction
        ),
    }


def periodic_material_connectivity(material_mask: ArrayLike) -> dict[str, Any]:
    """Label 4-connected material components with transverse periodic wrap."""

    mask = np.asarray(material_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("material_mask must be two-dimensional")
    nz, nx = mask.shape
    visited = np.zeros_like(mask)
    sizes: list[int] = []
    seam_components = 0
    for seed_z, seed_x in np.argwhere(mask):
        if visited[seed_z, seed_x]:
            continue
        queue: deque[tuple[int, int]] = deque(((int(seed_z), int(seed_x)),))
        visited[seed_z, seed_x] = True
        size = 0
        touches_left = False
        touches_right = False
        while queue:
            iz, ix = queue.popleft()
            size += 1
            touches_left |= ix == 0
            touches_right |= ix == nx - 1
            for neighbor_z, neighbor_x in (
                (iz - 1, ix),
                (iz + 1, ix),
                (iz, (ix - 1) % nx),
                (iz, (ix + 1) % nx),
            ):
                if 0 <= neighbor_z < nz and mask[neighbor_z, neighbor_x]:
                    if not visited[neighbor_z, neighbor_x]:
                        visited[neighbor_z, neighbor_x] = True
                        queue.append((neighbor_z, neighbor_x))
        sizes.append(size)
        seam_components += int(touches_left and touches_right)
    material_cells = int(np.count_nonzero(mask))
    return {
        "component_count": len(sizes),
        "largest_component_fraction": (
            float(max(sizes) / material_cells) if material_cells else 0.0
        ),
        "periodic_seam_component_count": seam_components,
        "has_periodically_connected_component": bool(seam_components),
    }
