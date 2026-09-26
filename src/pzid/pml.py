"""Analytic complex-coordinate stretching for axial PML regions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pzid.grid import UniformGrid2D


def axial_pml_stretch(
    omega: complex,
    grid: UniformGrid2D,
    *,
    cells: int,
    target_round_trip_amplitude: float = 1e-10,
    polynomial_order: int = 3,
    speed_of_light: float = 1.0,
) -> NDArray[np.complex128]:
    """Return cell-centered stretch ``s_z = 1 + i sigma(z)/omega``.

    Equal PMLs are placed at the lower and upper axial boundaries. The
    conductivity profile is zero at the interface and polynomial toward the
    outer edge. ``omega`` may be complex but must be nonzero.
    """

    if omega == 0.0:
        raise ValueError("PML stretch is undefined at zero frequency")
    if cells < 0 or 2 * cells >= grid.nz:
        raise ValueError("cells must leave at least one non-PML axial cell")
    if not 0.0 < target_round_trip_amplitude < 1.0:
        raise ValueError("target_round_trip_amplitude must lie between 0 and 1")
    if polynomial_order < 1:
        raise ValueError("polynomial_order must be positive")
    if speed_of_light <= 0.0:
        raise ValueError("speed_of_light must be positive")
    if cells == 0:
        return np.ones(grid.nz, dtype=np.complex128)

    thickness = cells * grid.dz
    sigma_max = -(
        (polynomial_order + 1)
        * speed_of_light
        * np.log(target_round_trip_amplitude)
        / (2.0 * thickness)
    )
    depth = (np.arange(cells, dtype=float) + 0.5) / cells
    sigma_edge_to_interface = sigma_max * depth[::-1] ** polynomial_order
    sigma = np.zeros(grid.nz, dtype=float)
    sigma[:cells] = sigma_edge_to_interface
    sigma[-cells:] = sigma_edge_to_interface[::-1]
    return 1.0 + 1j * sigma / complex(omega)

