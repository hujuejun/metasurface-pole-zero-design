import numpy as np

from pzid.grid import NonuniformGrid2D, UniformGrid2D
from pzid.ports import discrete_axial_wavenumber
from pzid.twoport import TwoPortScatteringSetup, solve_two_port_scattering


def _setup() -> TwoPortScatteringSetup:
    grid = UniformGrid2D(nx=4, nz=180, dx=0.08, dz=0.02)
    return TwoPortScatteringSetup(
        grid=grid,
        lower_source_plane=24,
        upper_source_plane=155,
        lower_monitor_planes=(38, 50),
        upper_monitor_planes=(129, 141),
    )


def test_empty_cell_has_exchange_scattering_matrix() -> None:
    setup = _setup()
    solution = solve_two_port_scattering(
        2.0 + 0.07j, np.ones(setup.grid.shape), setup
    )
    wave_number = discrete_axial_wavenumber(
        2.0 + 0.07j, refractive_index=1.0, dz=setup.grid.dz
    )
    phase = np.exp(
        1j
        * wave_number
        * (
            setup.resolved_upper_reference_z
            - setup.resolved_lower_reference_z
        )
    )
    expected = np.asarray(((0.0, phase), (phase, 0.0)))
    assert np.max(np.abs(solution.matrix - expected)) < 1e-12
    assert np.max(
        np.abs(solution.system.reference_scattering - expected)
    ) < 1e-12


def test_lossless_slab_is_reciprocal_and_unitary() -> None:
    setup = _setup()
    epsilon = np.ones(setup.grid.shape)
    epsilon[78:102, :] = 4.0
    matrix = solve_two_port_scattering(2.0, epsilon, setup).matrix
    assert abs(matrix[0, 1] - matrix[1, 0]) < 1e-12
    assert abs(matrix[0, 0] - matrix[1, 1]) < 1e-12
    assert np.max(np.abs(matrix.conj().T @ matrix - np.eye(2))) < 1e-12


def test_asymmetric_structure_remains_reciprocal_at_complex_frequency() -> None:
    setup = _setup()
    epsilon = np.ones(setup.grid.shape)
    epsilon[72:87, :] = 2.25
    epsilon[87:105, :] = 4.0
    matrix = solve_two_port_scattering(2.1 - 0.06j, epsilon, setup).matrix
    assert abs(matrix[0, 1] - matrix[1, 0]) < 1e-11


def test_setup_rejects_invalid_plane_order() -> None:
    grid = UniformGrid2D(nx=2, nz=20, dx=0.1, dz=0.1)
    try:
        TwoPortScatteringSetup(
            grid=grid,
            lower_source_plane=2,
            upper_source_plane=17,
            lower_monitor_planes=(4, 13),
            upper_monitor_planes=(9, 15),
        )
    except ValueError:
        return
    raise AssertionError("interleaved monitor pairs should be rejected")


def _nonuniform_setup() -> TwoPortScatteringSetup:
    widths = np.asarray(
        [0.04] * 12
        + [0.05, 0.07, 0.09, 0.12, 0.12, 0.09, 0.07, 0.05]
        + [0.04] * 12
    )
    grid = NonuniformGrid2D(
        nx=4, dx=0.08, z_edges=np.concatenate(([0.0], np.cumsum(widths)))
    )
    return TwoPortScatteringSetup(
        grid=grid,
        lower_source_plane=3,
        upper_source_plane=grid.nz - 4,
        lower_monitor_planes=(6, 9),
        upper_monitor_planes=(grid.nz - 10, grid.nz - 7),
    )


def test_nonuniform_empty_cell_is_conservative_and_reciprocal() -> None:
    setup = _nonuniform_setup()
    solution = solve_two_port_scattering(
        2.0, np.ones(setup.grid.shape), setup
    )
    # A change in discrete spacing has a small truncation-error reflection;
    # it must not be subtracted because the raw discrete network is unitary.
    assert abs(solution.matrix[0, 0]) < 5e-3
    assert abs(solution.matrix[1, 1]) < 5e-3
    assert abs(solution.matrix[0, 1] - solution.matrix[1, 0]) < 2e-12
    assert np.max(
        np.abs(solution.matrix.conj().T @ solution.matrix - np.eye(2))
    ) < 2e-12
    np.testing.assert_allclose(
        solution.matrix, solution.system.reference_scattering, atol=2e-12
    )


def test_nonuniform_lossless_slab_is_reciprocal_and_unitary() -> None:
    setup = _nonuniform_setup()
    epsilon = np.ones(setup.grid.shape)
    center = setup.grid.nz // 2
    epsilon[center - 2 : center + 2] = 4.0
    matrix = solve_two_port_scattering(2.0, epsilon, setup).matrix
    assert abs(matrix[0, 1] - matrix[1, 0]) < 2e-12
    assert abs(matrix[0, 0] - matrix[1, 1]) < 2e-12
    assert np.max(np.abs(matrix.conj().T @ matrix - np.eye(2))) < 2e-12


def test_nonuniform_setup_rejects_monitor_across_mesh_transition() -> None:
    widths = np.asarray([0.04] * 6 + [0.06, 0.08] + [0.04] * 6)
    grid = NonuniformGrid2D(
        nx=2, dx=0.1, z_edges=np.concatenate(([0.0], np.cumsum(widths)))
    )
    with np.testing.assert_raises(ValueError):
        TwoPortScatteringSetup(
            grid=grid,
            lower_source_plane=2,
            upper_source_plane=grid.nz - 3,
            lower_monitor_planes=(4, 6),
            upper_monitor_planes=(grid.nz - 7, grid.nz - 5),
        )


def test_oblique_empty_cell_has_exchange_scattering_matrix() -> None:
    base = _setup()
    setup = TwoPortScatteringSetup(
        grid=base.grid,
        lower_source_plane=base.lower_source_plane,
        upper_source_plane=base.upper_source_plane,
        lower_monitor_planes=base.lower_monitor_planes,
        upper_monitor_planes=base.upper_monitor_planes,
        bloch_kx=0.41,
    )
    omega = 2.0 + 0.07j
    solution = solve_two_port_scattering(
        omega, np.ones(setup.grid.shape), setup
    )
    wave_number = discrete_axial_wavenumber(
        omega,
        refractive_index=1.0,
        dz=setup.grid.dz,
        transverse_wavenumber=setup.bloch_kx,
        transverse_spacing=setup.grid.dx,
    )
    phase = np.exp(
        1j
        * wave_number
        * (
            setup.resolved_upper_reference_z
            - setup.resolved_lower_reference_z
        )
    )
    expected = np.asarray(((0.0, phase), (phase, 0.0)))
    assert np.max(np.abs(solution.matrix - expected)) < 2e-12


def test_oblique_lossless_slab_is_unitary() -> None:
    base = _setup()
    setup = TwoPortScatteringSetup(
        grid=base.grid,
        lower_source_plane=base.lower_source_plane,
        upper_source_plane=base.upper_source_plane,
        lower_monitor_planes=base.lower_monitor_planes,
        upper_monitor_planes=base.upper_monitor_planes,
        bloch_kx=0.41,
    )
    epsilon = np.ones(setup.grid.shape)
    epsilon[78:102, :] = 4.0
    matrix = solve_two_port_scattering(2.0, epsilon, setup).matrix
    assert np.max(np.abs(matrix.conj().T @ matrix - np.eye(2))) < 2e-12
