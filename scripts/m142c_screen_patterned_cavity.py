from __future__ import annotations

import numpy as np

from pzid.scattering import SymmetricScatteringSetup, solve_symmetric_scattering

def add_centered_cavity_ridge(
    epsilon: np.ndarray,
    setup: SymmetricScatteringSetup,
    layers: tuple[tuple[float, float], ...],
    pair_count: int,
    high_index: float,
    low_index: float,
    *,
    fill_factor: float,
    height_um: float,
) -> np.ndarray:
    """Add an exactly area-averaged rectangular ridge inside the cavity."""

    if not 0.0 < fill_factor < 1.0 or height_um <= 0.0:
        raise ValueError("ridge fill and height must be positive and subperiodic")
    grid = setup.grid
    stack_length = sum(d for _, d in layers)
    base_z = 0.5 * (grid.height - stack_length)
    mirror_length = sum(d for _, d in layers[: 2 * pair_count])
    cavity_length = layers[2 * pair_count][1]
    if height_um >= cavity_length:
        raise ValueError("ridge height must remain inside cavity")
    center_z = base_z + mirror_length + 0.5 * cavity_length
    z_lower, z_upper = center_z - 0.5 * height_um, center_z + 0.5 * height_um
    z_cell_lower = np.arange(grid.nz, dtype=float) * grid.dz
    z_fraction = np.maximum(
        0.0, np.minimum(z_cell_lower + grid.dz, z_upper) - np.maximum(z_cell_lower, z_lower)
    ) / grid.dz
    width = fill_factor * grid.period
    x_lower, x_upper = 0.5 * (grid.period - width), 0.5 * (grid.period + width)
    x_cell_lower = np.arange(grid.nx, dtype=float) * grid.dx
    x_fraction = np.maximum(
        0.0, np.minimum(x_cell_lower + grid.dx, x_upper) - np.maximum(x_cell_lower, x_lower)
    ) / grid.dx
    result = np.asarray(epsilon, dtype=float).copy()
    result += (high_index**2 - low_index**2) * z_fraction[:, None] * x_fraction[None, :]
    return result
