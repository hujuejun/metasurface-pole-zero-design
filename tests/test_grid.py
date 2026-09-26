import numpy as np
import pytest

from pzid.grid import NonuniformGrid2D, UniformGrid2D


def test_grid_coordinates_and_indexing() -> None:
    grid = UniformGrid2D(nx=4, nz=3, dx=0.2, dz=0.3)
    assert grid.shape == (3, 4)
    assert grid.size == 12
    assert grid.period == pytest.approx(0.8)
    assert grid.height == pytest.approx(0.9)
    np.testing.assert_allclose(grid.x, [0.1, 0.3, 0.5, 0.7])
    np.testing.assert_allclose(grid.z, [0.15, 0.45, 0.75])
    assert grid.flat_index(2, 3) == 11


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nx": 1, "nz": 3, "dx": 1.0, "dz": 1.0},
        {"nx": 3, "nz": 1, "dx": 1.0, "dz": 1.0},
        {"nx": 3, "nz": 3, "dx": 0.0, "dz": 1.0},
        {"nx": 3, "nz": 3, "dx": 1.0, "dz": -1.0},
    ],
)
def test_invalid_grid_raises(kwargs: dict[str, float | int]) -> None:
    with pytest.raises(ValueError):
        UniformGrid2D(**kwargs)


def test_nonuniform_grid_coordinates_and_local_port_spacing() -> None:
    grid = NonuniformGrid2D(
        nx=4,
        dx=0.2,
        z_edges=np.asarray([0.0, 0.1, 0.2, 0.35, 0.5]),
    )
    assert grid.shape == (4, 4)
    assert grid.period == pytest.approx(0.8)
    assert grid.height == pytest.approx(0.5)
    assert grid.reference_dz == pytest.approx(0.1)
    np.testing.assert_allclose(grid.z, [0.05, 0.15, 0.275, 0.425])
    np.testing.assert_allclose(grid.dz_cells, [0.1, 0.1, 0.15, 0.15])
    np.testing.assert_allclose(grid.z_center_spacings, [0.1, 0.125, 0.15])
    assert grid.uniform_dz_for_planes((0, 1)) == pytest.approx(0.1)
    assert grid.uniform_dz_for_planes((2, 3)) == pytest.approx(0.15)
    with pytest.raises(ValueError, match="uniform axial region"):
        grid.uniform_dz_for_planes((1, 2))


@pytest.mark.parametrize(
    "edges",
    (
        [0.1, 0.2, 0.3],
        [0.0, 0.2, 0.1],
        [0.0, np.nan, 0.2],
        [0.0, 0.1],
    ),
)
def test_invalid_nonuniform_grid_raises(edges: list[float]) -> None:
    with pytest.raises(ValueError):
        NonuniformGrid2D(nx=3, dx=0.1, z_edges=np.asarray(edges))
