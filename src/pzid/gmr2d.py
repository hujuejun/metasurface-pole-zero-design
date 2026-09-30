"""Physical binary slab-and-tooth 1D guided-mode-resonance geometry.

The period is along x; invariant y makes this a 2D TM (Hy) structure.  The
slab and ridge have the same high index and are surrounded by one low-index
medium. The two optimization coordinates are physical slab thickness and
ridge/etch height, not unconstrained per-cell permittivities.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pzid.grid import Grid2D


@dataclass(frozen=True)
class SingleToothGMR:
    period_um: float
    slab_thickness_um: float
    ridge_height_um: float
    ridge_fill_factor: float
    high_index: float
    port_index: float

    def __post_init__(self) -> None:
        if self.period_um <= 0 or self.slab_thickness_um <= 0 or self.ridge_height_um <= 0:
            raise ValueError("all physical lengths must be positive")
        if not 0 < self.ridge_fill_factor < 1:
            raise ValueError("ridge fill must lie strictly between zero and one")
        if self.port_index <= 0 or self.high_index <= self.port_index:
            raise ValueError("indices must satisfy high > port > 0")

    def rasterize_with_derivatives(
        self, grid: Grid2D, *, base_z_um: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return area-averaged epsilon and exact off-edge shape derivatives.

        Derivative axis 0 is slab thickness, axis 1 is ridge height. Interfaces
        are horizontal in z and vertical in x, so cell overlap is exact rather
        than Monte Carlo/subpixel sampling. At a parameter value exactly on a
        cell edge the map is nonsmooth: the returned derivative is the centered
        subgradient, which lets an optimizer cross the edge without crashing.
        """

        if not np.isclose(grid.period, self.period_um, rtol=0, atol=1e-12):
            raise ValueError("grid period differs from physical grating period")
        top = base_z_um + self.slab_thickness_um + self.ridge_height_um
        if base_z_um <= 0 or top >= grid.height:
            raise ValueError("structure must lie entirely inside the axial domain")
        z_edges = np.asarray(grid.z_edges, dtype=float)
        z_lower, z_upper = z_edges[:-1], z_edges[1:]
        widths = np.asarray(grid.dz_cells, dtype=float)

        def axial_fraction(lower: float, upper: float) -> NDArray[np.float64]:
            overlap = np.maximum(0.0, np.minimum(z_upper, upper) - np.maximum(z_lower, lower))
            return np.clip(overlap / widths, 0.0, 1.0)

        x_edges = np.arange(grid.nx + 1, dtype=float) * grid.dx
        x_left = 0.5 * self.period_um * (1.0 - self.ridge_fill_factor)
        x_right = 0.5 * self.period_um * (1.0 + self.ridge_fill_factor)
        x_fill = np.maximum(
            0.0, np.minimum(x_edges[1:], x_right) - np.maximum(x_edges[:-1], x_left)
        ) / grid.dx
        x_fill = np.clip(x_fill, 0.0, 1.0)
        slab_top = base_z_um + self.slab_thickness_um
        slab_fill = axial_fraction(base_z_um, slab_top)
        ridge_fill = axial_fraction(slab_top, top)
        contrast = self.high_index**2 - self.port_index**2
        epsilon = self.port_index**2 + contrast * (
            slab_fill[:, None] + ridge_fill[:, None] * x_fill[None, :]
        )

        def boundary_slope(location: float) -> NDArray[np.float64]:
            slope = ((location > z_lower) & (location < z_upper)).astype(float) / widths
            matching = np.flatnonzero(np.isclose(location, z_edges, rtol=0, atol=1e-12))
            if matching.size:
                edge = int(matching[0])
                if not 0 < edge < grid.nz:
                    raise ValueError("physical interface left interior axial region")
                slope[edge - 1] = 0.5 / widths[edge - 1]
                slope[edge] = 0.5 / widths[edge]
            return slope

        lower_slope = boundary_slope(slab_top)
        upper_slope = boundary_slope(top)
        derivative = np.empty((2, *grid.shape), dtype=float)
        derivative[0] = contrast * (
            lower_slope[:, None] * (1.0 - x_fill[None, :])
            + upper_slope[:, None] * x_fill[None, :]
        )
        derivative[1] = contrast * upper_slope[:, None] * x_fill[None, :]
        return np.asarray(epsilon, dtype=float), derivative
