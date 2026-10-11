"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
from typing import Any
import numpy as np
from pzid.benchmarks import CoupledRectangularBarArray
from pzid.grid import UniformGrid2D
from pzid.twoport import TwoPortScatteringSetup

def build_problem(config: dict[str, Any], width_um: float, bar_height_um: float, central_gap_um: float, *, cells_per_period: int) -> tuple[CoupledRectangularBarArray, UniformGrid2D, np.ndarray, TwoPortScatteringSetup]:
    benchmark = config['benchmark']
    mesh = config['mesh']
    period = float(benchmark['period_um'])
    spacing = period / cells_per_period
    nz = int(mesh['axial_cells_per_period'] * cells_per_period)
    if nz % 2:
        nz += 1
    grid = UniformGrid2D(nx=cells_per_period, nz=nz, dx=spacing, dz=spacing)
    geometry = CoupledRectangularBarArray(period=period, width=width_um, bar_height=bar_height_um, central_gap=central_gap_um, refractive_index=float(benchmark['refractive_index']), background_index=float(benchmark['background_index']))
    epsilon = geometry.relative_permittivity(grid, subpixels=int(mesh['subpixels']))
    occupied = np.flatnonzero(np.any(epsilon > float(benchmark['background_index']) ** 2 + 1e-12, axis=1))
    margin = int(mesh['port_margin_cells'])
    near = int(mesh['near_monitor_offset_cells'])
    far = int(mesh['far_monitor_offset_cells'])
    if not 1 < near < far < margin:
        raise ValueError('port offsets must satisfy 1 < near < far < margin')
    lower_source = max(2, int(occupied[0]) - margin)
    lower_monitors = (lower_source + near, lower_source + far)
    upper_monitors = (grid.nz - 1 - lower_monitors[1], grid.nz - 1 - lower_monitors[0])
    setup = TwoPortScatteringSetup(grid=grid, lower_source_plane=lower_source, upper_source_plane=grid.nz - 1 - lower_source, lower_monitor_planes=lower_monitors, upper_monitor_planes=upper_monitors, port_index=float(benchmark['background_index']))
    return (geometry, grid, epsilon, setup)
