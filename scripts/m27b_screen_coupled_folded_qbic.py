"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
from typing import Any
import numpy as np
from pzid.hybrid import build_period_doubled_coupled_surface_relief_slabs
from pzid.twoport import TwoPortScatteringSetup

def _build(config: dict[str, Any], gap: int, rib: int, average: int, asymmetry: int) -> tuple[Any, np.ndarray, Any, TwoPortScatteringSetup]:
    benchmark = config['benchmark']
    mesh = config['mesh']
    grid, epsilon, layout = build_period_doubled_coupled_surface_relief_slabs(period_um=float(benchmark['period_um']), cells_per_period=int(mesh['cells_per_period']), port_margin_rows=int(mesh['port_margin_rows']), single_slab_rows=int(benchmark['single_slab_rows']), inner_grating_rib_rows=rib, spacer_rows=gap, left_grating_width_cells=average - asymmetry, right_grating_width_cells=average + asymmetry, port_index=float(benchmark['port_index']), slab_index=float(benchmark['slab_index']))
    lower_source = int(mesh['lower_source_row'])
    monitors = tuple((int(value) for value in mesh['lower_monitor_rows']))
    setup = TwoPortScatteringSetup(grid=grid, lower_source_plane=lower_source, upper_source_plane=grid.nz - 1 - lower_source, lower_monitor_planes=monitors, upper_monitor_planes=tuple((grid.nz - 1 - value for value in reversed(monitors))), port_index=float(benchmark['port_index']))
    return (grid, epsilon, layout, setup)
