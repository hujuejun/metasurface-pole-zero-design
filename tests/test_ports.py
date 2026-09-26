import numpy as np
import pytest

from pzid.grid import NonuniformGrid2D, UniformGrid2D
from pzid.ports import (
    decompose_specular_h,
    discrete_axial_wavenumber,
    specular_projection_rows,
)


def test_decomposition_recovers_discrete_forward_and_backward_waves() -> None:
    grid = UniformGrid2D(nx=5, nz=30, dx=0.1, dz=0.04)
    omega = 2.3 - 0.04j
    index = 1.4
    wave_number = discrete_axial_wavenumber(
        omega, refractive_index=index, dz=grid.dz
    )
    expected_forward = 0.8 - 0.3j
    expected_backward = -0.2 + 0.1j
    axial = (
        expected_forward * np.exp(1j * wave_number * grid.z)
        + expected_backward * np.exp(-1j * wave_number * grid.z)
    )
    field = np.repeat(axial[:, np.newaxis], grid.nx, axis=1)
    found = decompose_specular_h(
        field,
        grid,
        planes=(7, 18),
        omega=omega,
        refractive_index=index,
    )
    assert found.forward == pytest.approx(expected_forward, rel=1e-13)
    assert found.backward == pytest.approx(expected_backward, rel=1e-13)


def test_discrete_wavenumber_converges_to_continuum() -> None:
    omega = 1.7
    index = 2.1
    coarse = discrete_axial_wavenumber(
        omega, refractive_index=index, dz=0.02
    )
    fine = discrete_axial_wavenumber(
        omega, refractive_index=index, dz=0.01
    )
    exact = omega * index
    assert abs(fine - exact) < abs(coarse - exact)


def test_projection_rows_match_direct_decomposition() -> None:
    grid = UniformGrid2D(nx=3, nz=12, dx=0.2, dz=0.1)
    rng = np.random.default_rng(4)
    field = rng.normal(size=grid.shape) + 1j * rng.normal(size=grid.shape)
    arguments = {
        "planes": (3, 8),
        "omega": 1.2 + 0.03j,
        "refractive_index": 1.0,
    }
    direct = decompose_specular_h(field, grid, **arguments)
    forward, backward = specular_projection_rows(grid, **arguments)
    assert forward @ field.ravel() == pytest.approx(direct.forward)
    assert backward @ field.ravel() == pytest.approx(direct.backward)


def test_nonuniform_grid_decomposes_wave_in_uniform_port_region() -> None:
    widths = np.asarray([0.04] * 8 + [0.06, 0.08, 0.06] + [0.04] * 8)
    grid = NonuniformGrid2D(
        nx=4, dx=0.1, z_edges=np.concatenate(([0.0], np.cumsum(widths)))
    )
    omega = 2.1 - 0.03j
    wave_number = discrete_axial_wavenumber(
        omega, refractive_index=1.2, dz=0.04
    )
    expected_forward = 0.7 + 0.2j
    expected_backward = -0.1 + 0.3j
    axial = (
        expected_forward * np.exp(1j * wave_number * grid.z)
        + expected_backward * np.exp(-1j * wave_number * grid.z)
    )
    field = np.repeat(axial[:, np.newaxis], grid.nx, axis=1)
    found = decompose_specular_h(
        field, grid, planes=(2, 6), omega=omega, refractive_index=1.2
    )
    assert found.forward == pytest.approx(expected_forward, rel=1e-13)
    assert found.backward == pytest.approx(expected_backward, rel=1e-13)
    with pytest.raises(ValueError, match="uniform axial region"):
        specular_projection_rows(
            grid, planes=(7, 9), omega=omega, refractive_index=1.2
        )


def test_identical_monitor_planes_raise() -> None:
    grid = UniformGrid2D(nx=2, nz=4, dx=1.0, dz=1.0)
    with pytest.raises(ValueError):
        decompose_specular_h(
            np.ones(grid.shape),
            grid,
            planes=(1, 1),
            omega=1.0,
            refractive_index=1.0,
        )


def test_oblique_decomposition_recovers_bloch_order() -> None:
    grid = UniformGrid2D(nx=9, nz=30, dx=0.08, dz=0.04)
    omega = 3.1 - 0.03j
    bloch_kx = 0.37
    wave_number = discrete_axial_wavenumber(
        omega,
        refractive_index=1.2,
        dz=grid.dz,
        transverse_wavenumber=bloch_kx,
        transverse_spacing=grid.dx,
    )
    expected_forward = 0.6 - 0.2j
    expected_backward = -0.15 + 0.08j
    axial = (
        expected_forward * np.exp(1j * wave_number * grid.z)
        + expected_backward * np.exp(-1j * wave_number * grid.z)
    )
    field = axial[:, np.newaxis] * np.exp(1j * bloch_kx * grid.x)[np.newaxis, :]
    found = decompose_specular_h(
        field,
        grid,
        planes=(7, 18),
        omega=omega,
        refractive_index=1.2,
        bloch_kx=bloch_kx,
    )
    assert found.forward == pytest.approx(expected_forward, rel=1e-12)
    assert found.backward == pytest.approx(expected_backward, rel=1e-12)
