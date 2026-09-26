from __future__ import annotations

import numpy as np

from pzid.grid import UniformGrid2D

from pzid.scattering import SymmetricScatteringSetup, solve_symmetric_scattering

def rasterize_layers(
    layers: tuple[tuple[float, float], ...],
    grid: UniformGrid2D,
    port_index: float,
    *,
    base_z_um: float,
) -> np.ndarray:
    """Exact axial cell-volume average of a physical stack, no grid transfer."""

    epsilon = np.full(grid.shape, port_index**2, dtype=float)
    lower_edges = np.arange(grid.nz, dtype=float) * grid.dz
    upper_edges = lower_edges + grid.dz
    cursor = float(base_z_um)
    for n, thickness in layers:
        upper = cursor + thickness
        overlap = np.maximum(0.0, np.minimum(upper_edges, upper) - np.maximum(lower_edges, cursor))
        fraction = overlap / grid.dz
        epsilon += (n**2 - port_index**2) * fraction[:, np.newaxis]
        cursor = upper
    if not np.isclose(cursor - base_z_um, sum(d for _, d in layers), atol=1e-12):
        raise AssertionError("layer raster extent differs from physical geometry")
    return epsilon

def make_problem(
    layers: tuple[tuple[float, float], ...],
    *,
    period_um: float,
    lateral_cells: int,
    dz_um: float,
    port_index: float,
    padding_um: float,
) -> tuple[UniformGrid2D, np.ndarray, SymmetricScatteringSetup]:
    length = sum(d for _, d in layers)
    nz = int(np.ceil((length + 2.0 * padding_um) / dz_um))
    grid = UniformGrid2D(lateral_cells, nz, period_um / lateral_cells, dz_um)
    base_z = 0.5 * (grid.height - length)
    epsilon = rasterize_layers(layers, grid, port_index, base_z_um=base_z)
    source = max(4, int(round(0.13 / dz_um)))
    near = max(source + 4, int(round(0.32 / dz_um)))
    far = max(source + 2, int(round(0.24 / dz_um)))
    setup = SymmetricScatteringSetup(
        grid=grid,
        source_plane=source,
        input_monitor_planes=(far, near),
        output_monitor_planes=(nz - 1 - near, nz - 1 - far),
        port_index=port_index,
    )
    if np.any(epsilon[setup.input_monitor_planes[1]] != port_index**2):
        raise ValueError("lower monitor intersects structure")
    if np.any(epsilon[setup.output_monitor_planes[0]] != port_index**2):
        raise ValueError("upper monitor intersects structure")
    return grid, epsilon, setup
