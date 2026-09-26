import numpy as np

from pzid.grid3d import UniformGrid3D


def test_grid3d_component_layout_and_views() -> None:
    grid = UniformGrid3D(nx=3, ny=2, nz=4, dx=0.1, dy=0.2, dz=0.3)
    assert grid.shape == (4, 2, 3)
    assert grid.cell_count == 24
    assert grid.field_size == 72
    assert grid.cell_flat_index(2, 1, 1) == 16
    values = np.arange(grid.field_size, dtype=float).astype(np.complex128)
    assert grid.component_view(values, "y").shape == grid.shape
    assert grid.component_view(values, "y")[0, 0, 0] == grid.cell_count
    np.testing.assert_allclose(
        grid.tangential_e_z, np.arange(grid.nz) * grid.dz
    )
