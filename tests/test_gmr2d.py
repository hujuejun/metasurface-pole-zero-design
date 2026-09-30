"""Geometry and adjoint prerequisites for the physical single-tooth GMR."""

from dataclasses import replace

import numpy as np
from pzid.gmr2d import SingleToothGMR
from pzid.grid import UniformGrid2D


def test_binary_area_average_and_physical_shape_derivatives() -> None:
    geometry = SingleToothGMR(0.8023, 0.0988, 0.0937, 0.66, 2.0, 1.0)
    grid = UniformGrid2D(48, 160, 0.8023 / 48, 0.0067)
    base = 0.41
    epsilon, derivative = geometry.rasterize_with_derivatives(grid, base_z_um=base)
    assert epsilon.shape == grid.shape
    assert derivative.shape == (2, *grid.shape)
    assert np.min(epsilon) == 1.0
    assert np.max(epsilon) == 4.0
    for index, name in enumerate(("slab_thickness_um", "ridge_height_um")):
        step = 1e-5
        plus = replace(geometry, **{name: getattr(geometry, name) + step})
        minus = replace(geometry, **{name: getattr(geometry, name) - step})
        numeric = (plus.rasterize_with_derivatives(grid, base_z_um=base)[0]
                   - minus.rasterize_with_derivatives(grid, base_z_um=base)[0]) / (2 * step)
        np.testing.assert_allclose(derivative[index], numeric, rtol=2e-10, atol=1e-8)


def test_boundary_on_cell_edge_uses_centered_subgradient() -> None:
    grid = UniformGrid2D(40, 100, 0.8 / 40, 0.01)
    geometry = SingleToothGMR(0.8, 0.10, 0.05, 0.6, 2.0, 1.0)
    _, derivative = geometry.rasterize_with_derivatives(grid, base_z_um=0.40)
    step = 1e-6
    for index, name in enumerate(("slab_thickness_um", "ridge_height_um")):
        plus = replace(geometry, **{name: getattr(geometry, name) + step})
        minus = replace(geometry, **{name: getattr(geometry, name) - step})
        numeric = (plus.rasterize_with_derivatives(grid, base_z_um=0.40)[0]
                   - minus.rasterize_with_derivatives(grid, base_z_um=0.40)[0]) / (2 * step)
        np.testing.assert_allclose(derivative[index], numeric, rtol=2e-10, atol=1e-8)
