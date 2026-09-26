"""Calibrated four-channel scattering for periodic 3D metasurface cells."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import splu

from pzid.fdfd3d import MaxwellOperator3D, assemble_maxwell_system_3d
from pzid.grid3d import UniformGrid3D
from pzid.pml3d import AxialPML3D
from pzid.ports3d import Polarization, specular_e_projection_rows_3d
from pzid.ports import discrete_axial_wavenumber


ComplexArray: TypeAlias = NDArray[np.complex128]

# Channel order is lower-x, lower-y, upper-x, upper-y for both incoming and
# outgoing amplitudes.  At a given side, incoming points toward the structure
# and outgoing points away from it.
CHANNELS_3D = ("lower_x", "lower_y", "upper_x", "upper_y")


def polarization_channel_indices(
    polarization: Polarization,
) -> tuple[int, int]:
    """Return lower and upper channel indices for one linear polarization."""

    if polarization == "x":
        return 0, 2
    if polarization == "y":
        return 1, 3
    raise ValueError("polarization must be 'x' or 'y'")


def _scattering_deembedding_phases_3d(
    omega: complex,
    setup: "ScatteringSetup3D",
) -> tuple[ComplexArray, ComplexArray]:
    """Return outgoing and incoming phases for local port references."""

    lower_reference = float(
        np.mean(setup.grid.z[list(setup.lower_monitor_planes)])
    )
    upper_reference = float(
        np.mean(setup.grid.z[list(setup.upper_monitor_planes)])
    )
    wave_number = discrete_axial_wavenumber(
        omega,
        refractive_index=setup.port_index,
        dz=setup.grid.dz,
        speed_of_light=setup.speed_of_light,
    )
    incoming_phase = np.asarray(
        (
            np.exp(1j * wave_number * lower_reference),
            np.exp(1j * wave_number * lower_reference),
            np.exp(-1j * wave_number * upper_reference),
            np.exp(-1j * wave_number * upper_reference),
        )
    )
    outgoing_phase = np.asarray(
        (
            np.exp(-1j * wave_number * lower_reference),
            np.exp(-1j * wave_number * lower_reference),
            np.exp(1j * wave_number * upper_reference),
            np.exp(1j * wave_number * upper_reference),
        )
    )
    return np.asarray(outgoing_phase), np.asarray(incoming_phase)


def deembed_scattering_matrix_3d(
    scattering_matrix: ArrayLike,
    omega: complex,
    setup: "ScatteringSetup3D",
) -> ComplexArray:
    """Reference all port amplitudes to symmetric local monitor planes.

    The projection rows return coefficients of global ``exp(+/- i k z)``
    waves.  Their raw reflection phases therefore differ between the two
    sides even for an axially symmetric structure.  This diagonal basis
    change evaluates the waves at cell-centered monitor reference planes and
    is analytic at complex frequency.  It changes phases but not channel
    powers, poles, or zeros.
    """

    matrix = np.asarray(scattering_matrix, dtype=np.complex128)
    if matrix.shape != (4, 4):
        raise ValueError("scattering_matrix must have shape (4, 4)")
    outgoing_phase, incoming_phase = _scattering_deembedding_phases_3d(
        omega, setup
    )
    return np.asarray(
        outgoing_phase[:, np.newaxis]
        * matrix
        / incoming_phase[np.newaxis, :]
    )


def deembed_scattering_jacobian_3d(
    scattering_jacobian: ArrayLike,
    omega: complex,
    setup: "ScatteringSetup3D",
) -> ComplexArray:
    """Apply the local-reference basis change to an ``S`` Jacobian.

    The deembedding phases are independent of material and geometry, so the
    same diagonal factors multiply every trailing derivative coordinate.
    """

    jacobian = np.asarray(scattering_jacobian, dtype=np.complex128)
    if jacobian.ndim < 2 or jacobian.shape[:2] != (4, 4):
        raise ValueError("scattering_jacobian must start with shape (4, 4)")
    outgoing_phase, incoming_phase = _scattering_deembedding_phases_3d(
        omega, setup
    )
    trailing = (1,) * (jacobian.ndim - 2)
    return np.asarray(
        outgoing_phase.reshape((4, 1, *trailing))
        * jacobian
        / incoming_phase.reshape((1, 4, *trailing))
    )


def local_to_global_incoming_3d(
    incoming_amplitudes: ArrayLike,
    omega: complex,
    setup: "ScatteringSetup3D",
) -> ComplexArray:
    """Convert local-reference incoming amplitudes to global wave coefficients."""

    local = np.asarray(incoming_amplitudes, dtype=np.complex128).reshape(-1)
    if local.shape != (4,):
        raise ValueError("incoming_amplitudes must contain four channels")
    _, incoming_phase = _scattering_deembedding_phases_3d(omega, setup)
    return np.asarray(local / incoming_phase)


@dataclass(frozen=True)
class ScatteringSetup3D:
    """Fixed sources, monitors, ports, and PML for a 3D unit cell."""

    grid: UniformGrid3D
    lower_source_plane: int
    upper_source_plane: int
    lower_monitor_planes: tuple[int, int]
    upper_monitor_planes: tuple[int, int]
    port_index: float = 1.0
    pml: AxialPML3D = AxialPML3D(cells=12)
    bloch_kx: float = 0.0
    bloch_ky: float = 0.0
    speed_of_light: float = 1.0
    require_subwavelength: bool = True

    def __post_init__(self) -> None:
        if self.port_index <= 0.0:
            raise ValueError("port_index must be positive")
        if self.speed_of_light <= 0.0:
            raise ValueError("speed_of_light must be positive")
        if self.bloch_kx != 0.0 or self.bloch_ky != 0.0:
            raise ValueError(
                "the current four-channel ports require normal incidence"
            )
        self.pml.validate(self.grid)
        if (
            len(self.lower_monitor_planes) != 2
            or len(self.upper_monitor_planes) != 2
        ):
            raise ValueError("each port requires exactly two monitor planes")
        planes = (
            self.lower_source_plane,
            *self.lower_monitor_planes,
            *self.upper_monitor_planes,
            self.upper_source_plane,
        )
        if any(not 0 <= plane < self.grid.nz for plane in planes):
            raise ValueError("source or monitor plane is outside the grid")
        if not (
            self.pml.cells <= self.lower_source_plane
            < min(self.lower_monitor_planes)
            < max(self.lower_monitor_planes)
            < min(self.upper_monitor_planes)
            < max(self.upper_monitor_planes)
            < self.upper_source_plane
            < self.grid.nz - self.pml.cells
        ):
            raise ValueError(
                "planes must be ordered PML, lower source, lower monitors, "
                "upper monitors, upper source, PML"
            )

    @property
    def first_diffraction_frequency(self) -> float:
        """Lowest normal-incidence transverse diffraction threshold."""

        return min(
            2.0
            * np.pi
            * self.speed_of_light
            / (self.port_index * self.grid.period_x),
            2.0
            * np.pi
            * self.speed_of_light
            / (self.port_index * self.grid.period_y),
        )


@dataclass(frozen=True)
class ScatteringLinearSystem3D:
    """Maxwell system and calibrated four-channel input/output maps."""

    omega: complex
    epsilon_r: ComplexArray
    setup: ScatteringSetup3D
    maxwell: MaxwellOperator3D
    sources: ComplexArray
    output_projector: ComplexArray
    output_offset: ComplexArray
    reference_fields: ComplexArray
    reference_scattering: ComplexArray


@dataclass(frozen=True)
class ScatteringSolution3D:
    """Four-by-four specular scattering matrix and unit-input fields."""

    matrix: ComplexArray
    fields: ComplexArray
    system: ScatteringLinearSystem3D

    def polarization_block(self, polarization: Polarization) -> ComplexArray:
        """Return the two-side co-polarized block in lower/upper order."""

        lower, upper = polarization_channel_indices(polarization)
        return np.asarray(self.matrix[np.ix_((lower, upper), (lower, upper))])

    def cross_polarized_block(self, polarization: Polarization) -> ComplexArray:
        """Return cross-polarized outputs for co-polarized incidence."""

        lower, upper = polarization_channel_indices(polarization)
        other = "y" if polarization == "x" else "x"
        cross_lower, cross_upper = polarization_channel_indices(other)
        return np.asarray(
            self.matrix[np.ix_((cross_lower, cross_upper), (lower, upper))]
        )


@dataclass(frozen=True)
class ReferenceCalibration3D:
    """Design-independent source and port calibration at one frequency."""

    sources: ComplexArray
    output_projector: ComplexArray
    output_offset: ComplexArray
    reference_fields: ComplexArray
    reference_scattering: ComplexArray


def _validate_port_material(
    epsilon: ComplexArray, setup: ScatteringSetup3D
) -> None:
    expected = setup.port_index**2
    fixed_end = max(setup.lower_monitor_planes)
    fixed_start = min(setup.upper_monitor_planes)
    if not np.allclose(epsilon[: fixed_end + 1], expected, atol=1e-13, rtol=0.0):
        raise ValueError("the lower PML, source, and monitors must be uniform port")
    if not np.allclose(epsilon[fixed_start:], expected, atol=1e-13, rtol=0.0):
        raise ValueError("the upper PML, source, and monitors must be uniform port")


def _raw_sources(setup: ScatteringSetup3D) -> ComplexArray:
    sources = np.zeros((setup.grid.field_size, 4), dtype=np.complex128)
    plane_size = setup.grid.nx * setup.grid.ny
    amplitude = 1.0 / setup.grid.dz
    for side, plane in enumerate(
        (setup.lower_source_plane, setup.upper_source_plane)
    ):
        for polarization_index in (0, 1):
            column = 2 * side + polarization_index
            component_offset = polarization_index * setup.grid.cell_count
            start = component_offset + plane * plane_size
            sources[start : start + plane_size, column] = amplitude
    return sources


def _port_projection_rows(
    omega: complex, setup: ScatteringSetup3D
) -> tuple[ComplexArray, ComplexArray, ComplexArray, ComplexArray]:
    rows: list[ComplexArray] = []
    for polarization in ("x", "y"):
        lower_forward, lower_backward = specular_e_projection_rows_3d(
            setup.grid,
            planes=setup.lower_monitor_planes,
            omega=omega,
            refractive_index=setup.port_index,
            polarization=polarization,
            speed_of_light=setup.speed_of_light,
        )
        upper_forward, upper_backward = specular_e_projection_rows_3d(
            setup.grid,
            planes=setup.upper_monitor_planes,
            omega=omega,
            refractive_index=setup.port_index,
            polarization=polarization,
            speed_of_light=setup.speed_of_light,
        )
        rows.append(lower_forward)
        rows.append(lower_backward)
        rows.append(upper_forward)
        rows.append(upper_backward)
    # Return arrays indexed by polarization: lower plus/minus, upper plus/minus.
    return (
        np.vstack((rows[0], rows[4])),
        np.vstack((rows[1], rows[5])),
        np.vstack((rows[2], rows[6])),
        np.vstack((rows[3], rows[7])),
    )


@lru_cache(maxsize=8)
def _reference_calibration(
    omega: complex, setup: ScatteringSetup3D
) -> ReferenceCalibration3D:
    """Solve and cache the homogeneous reference for fixed setup/frequency.

    At normal incidence the uniform reference and sheet sources occupy only
    the transverse-constant Fourier subspace.  A minimal 2 x 2 transverse grid is
    therefore an exact reduction of the reference solve (with the identical
    axial stencil and PML), avoiding a second full 3D factorization at every
    new complex frequency.
    """

    lower_plus, lower_minus, upper_plus, upper_minus = _port_projection_rows(
        omega, setup
    )
    output_projector = np.vstack((lower_minus, upper_plus))
    raw_sources = _raw_sources(setup)
    reduced_grid = UniformGrid3D(
        nx=2,
        ny=2,
        nz=setup.grid.nz,
        dx=setup.grid.dx,
        dy=setup.grid.dy,
        dz=setup.grid.dz,
    )
    reduced_setup = ScatteringSetup3D(
        grid=reduced_grid,
        lower_source_plane=setup.lower_source_plane,
        upper_source_plane=setup.upper_source_plane,
        lower_monitor_planes=setup.lower_monitor_planes,
        upper_monitor_planes=setup.upper_monitor_planes,
        port_index=setup.port_index,
        pml=setup.pml,
        speed_of_light=setup.speed_of_light,
        require_subwavelength=False,
    )
    reduced_lower_plus, _, _, reduced_upper_minus = _port_projection_rows(
        omega, reduced_setup
    )
    reduced_sources = _raw_sources(reduced_setup)
    reference_epsilon = np.full(reduced_grid.shape, setup.port_index**2)
    reference_system = assemble_maxwell_system_3d(
        omega,
        reference_epsilon,
        reduced_grid,
        pml=setup.pml,
        speed_of_light=setup.speed_of_light,
    )
    reference_factorization = splu(reference_system.matrix.tocsc())
    raw_reduced_reference = np.asarray(
        reference_factorization.solve(reduced_sources)
    )
    incident = np.asarray(
        (
            reduced_lower_plus[0] @ raw_reduced_reference[:, 0],
            reduced_lower_plus[1] @ raw_reduced_reference[:, 1],
            reduced_upper_minus[0] @ raw_reduced_reference[:, 2],
            reduced_upper_minus[1] @ raw_reduced_reference[:, 3],
        ),
        dtype=np.complex128,
    )
    if np.any(np.abs(incident) < 1e-14):
        raise RuntimeError("reference source calibration produced zero amplitude")
    scales = 1.0 / incident
    sources = raw_sources * scales[np.newaxis, :]
    reduced_fields = raw_reduced_reference * scales[np.newaxis, :]
    reduced_fields = reduced_fields.reshape(3, setup.grid.nz, 2, 2, 4)
    reference_fields = np.broadcast_to(
        reduced_fields[:, :, :1, :1, :],
        (3, setup.grid.nz, setup.grid.ny, setup.grid.nx, 4),
    ).reshape(setup.grid.field_size, 4)
    reference_scattering = output_projector @ reference_fields
    # Do not subtract the residual PML reflection. An additive correction
    # would alter physical scattering zeros and invalidate exact time-reversal
    # identities. PML error is instead controlled by an explicit convergence
    # gate; source normalization remains multiplicative and analytic.
    output_offset = np.zeros((4, 4), dtype=np.complex128)
    return ReferenceCalibration3D(
        sources=np.asarray(sources),
        output_projector=np.asarray(output_projector),
        output_offset=np.asarray(output_offset),
        reference_fields=np.asarray(reference_fields),
        reference_scattering=np.asarray(reference_scattering),
    )


def clear_reference_cache_3d() -> None:
    """Release cached homogeneous reference fields (useful between meshes)."""

    _reference_calibration.cache_clear()


def assemble_scattering_system_3d(
    omega: complex,
    epsilon_r: ArrayLike,
    setup: ScatteringSetup3D,
) -> ScatteringLinearSystem3D:
    """Assemble calibrated four-channel sources and outgoing projections."""

    epsilon = np.asarray(epsilon_r, dtype=np.complex128)
    if epsilon.shape != setup.grid.shape:
        raise ValueError(f"epsilon_r must have shape {setup.grid.shape}")
    _validate_port_material(epsilon, setup)
    if (
        setup.require_subwavelength
        and abs(complex(omega).real) >= setup.first_diffraction_frequency
    ):
        raise ValueError(
            "frequency reaches an additional diffraction channel; the "
            "four-channel specular basis is incomplete"
        )

    calibration = _reference_calibration(complex(omega), setup)

    maxwell = assemble_maxwell_system_3d(
        omega,
        epsilon,
        setup.grid,
        bloch_kx=setup.bloch_kx,
        bloch_ky=setup.bloch_ky,
        pml=setup.pml,
        speed_of_light=setup.speed_of_light,
    )
    return ScatteringLinearSystem3D(
        omega=complex(omega),
        epsilon_r=epsilon,
        setup=setup,
        maxwell=maxwell,
        sources=calibration.sources,
        output_projector=calibration.output_projector,
        output_offset=calibration.output_offset,
        reference_fields=calibration.reference_fields,
        reference_scattering=calibration.reference_scattering,
    )


def solve_scattering_3d(
    omega: complex,
    epsilon_r: ArrayLike,
    setup: ScatteringSetup3D,
) -> ScatteringSolution3D:
    """Solve the full four-channel specular scattering problem."""

    system = assemble_scattering_system_3d(omega, epsilon_r, setup)
    factorization = splu(system.maxwell.matrix.tocsc())
    fields = np.asarray(factorization.solve(system.sources))
    matrix = system.output_projector @ fields + system.output_offset
    return ScatteringSolution3D(
        matrix=np.asarray(matrix), fields=fields, system=system
    )
