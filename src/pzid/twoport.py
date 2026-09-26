"""Calibrated two-sided scattering for identical periodic ports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from pzid.fdfd import assemble_tm_operator
from pzid.grid import Grid2D
from pzid.ports import discrete_axial_wavenumber, specular_projection_rows
from pzid.radiation import apply_radiation_boundaries
from pzid.reference import uniform_reference_fields


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class TwoPortScatteringSetup:
    """Source and monitor planes for two identical uniform ports.

    Port 1 is the lower (negative-``z``) port and port 2 is the upper
    (positive-``z``) port. Incoming amplitudes point toward the structure;
    outgoing amplitudes point away from it. All amplitudes use one global
    electric-field phase convention.
    """

    grid: Grid2D
    lower_source_plane: int
    upper_source_plane: int
    lower_monitor_planes: tuple[int, int]
    upper_monitor_planes: tuple[int, int]
    port_index: float = 1.0
    lower_reference_z: float | None = None
    upper_reference_z: float | None = None
    bloch_kx: float = 0.0

    def __post_init__(self) -> None:
        if self.port_index <= 0.0:
            raise ValueError("port_index must be positive")
        if self.lower_reference_z is not None and self.lower_reference_z < 0.0:
            raise ValueError("lower_reference_z must be nonnegative")
        if self.upper_reference_z is not None and self.upper_reference_z < 0.0:
            raise ValueError("upper_reference_z must be nonnegative")
        planes = (
            self.lower_source_plane,
            *self.lower_monitor_planes,
            *self.upper_monitor_planes,
            self.upper_source_plane,
        )
        if any(not 0 <= plane < self.grid.nz for plane in planes):
            raise ValueError("source or monitor plane is outside the grid")
        if not (
            self.lower_source_plane
            < min(self.lower_monitor_planes)
            < max(self.lower_monitor_planes)
            < min(self.upper_monitor_planes)
            < max(self.upper_monitor_planes)
            < self.upper_source_plane
        ):
            raise ValueError(
                "planes must be ordered lower source, lower monitors, "
                "upper monitors, upper source"
            )
        # Exact modal projections and radiation boundaries require a locally
        # translation-invariant discrete port from each domain edge through
        # its outermost monitor plane.
        lower_dz = self.grid.uniform_dz_for_planes(
            tuple(range(0, max(self.lower_monitor_planes) + 1))
        )
        upper_dz = self.grid.uniform_dz_for_planes(
            tuple(range(min(self.upper_monitor_planes), self.grid.nz))
        )
        if not np.isclose(lower_dz, upper_dz, rtol=1e-12, atol=1e-14):
            raise ValueError("the two identical ports must use the same axial spacing")

    @property
    def port_dz(self) -> float:
        return self.grid.uniform_dz_for_planes(
            tuple(range(0, max(self.lower_monitor_planes) + 1))
        )

    @property
    def resolved_lower_reference_z(self) -> float:
        if self.lower_reference_z is not None:
            return self.lower_reference_z
        return float(self.grid.z[max(self.lower_monitor_planes)])

    @property
    def resolved_upper_reference_z(self) -> float:
        if self.upper_reference_z is not None:
            return self.upper_reference_z
        return float(self.grid.z[min(self.upper_monitor_planes)])


@dataclass(frozen=True)
class TwoPortLinearSystem:
    """Discrete system and calibrated input/output maps at one frequency."""

    omega: complex
    epsilon_r: ComplexArray
    setup: TwoPortScatteringSetup
    operator: csr_matrix
    sources: ComplexArray
    output_projector: ComplexArray
    output_offset: ComplexArray
    reference_fields: ComplexArray
    reference_scattering: ComplexArray


@dataclass(frozen=True)
class TwoPortScatteringSolution:
    """Full two-port scattering matrix and unit-input fields."""

    matrix: ComplexArray
    fields: ComplexArray
    system: TwoPortLinearSystem


def _radiating_operator(
    omega: complex,
    epsilon_r: ComplexArray,
    setup: TwoPortScatteringSetup,
) -> csr_matrix:
    operator = assemble_tm_operator(
        omega, epsilon_r, setup.grid, bloch_kx=setup.bloch_kx
    )
    return apply_radiation_boundaries(
        operator,
        omega,
        setup.grid,
        lower_index=setup.port_index,
        upper_index=setup.port_index,
        bloch_kx=setup.bloch_kx,
    )


def _raw_sources(setup: TwoPortScatteringSetup) -> ComplexArray:
    sources = np.zeros((setup.grid.size, 2), dtype=np.complex128)
    amplitude = 1.0 / setup.grid.reference_dz
    transverse_profile = np.exp(1j * setup.bloch_kx * setup.grid.x)
    for column, plane in enumerate(
        (setup.lower_source_plane, setup.upper_source_plane)
    ):
        start = plane * setup.grid.nx
        sources[start : start + setup.grid.nx, column] = (
            amplitude * transverse_profile
        )
    return sources


def _validate_uniform_port_rows(
    epsilon_r: ComplexArray,
    setup: TwoPortScatteringSetup,
) -> None:
    port_epsilon = setup.port_index**2
    rows = {
        0,
        setup.grid.nz - 1,
        setup.lower_source_plane,
        setup.upper_source_plane,
        *setup.lower_monitor_planes,
        *setup.upper_monitor_planes,
    }
    if any(
        not np.allclose(
            epsilon_r[row], port_epsilon, rtol=0.0, atol=1e-13
        )
        for row in rows
    ):
        raise ValueError("sources, monitors, and boundaries must be uniform ports")


def assemble_two_port_system(
    omega: complex,
    epsilon_r: ArrayLike,
    setup: TwoPortScatteringSetup,
) -> TwoPortLinearSystem:
    """Assemble calibrated unit-amplitude sources and outgoing projections.

    Incident amplitudes are calibrated with the empty-cell reference solve.
    No reflected reference field is subtracted: on a nonuniform mesh that
    field contains the conservative discretization error at mesh-spacing
    transitions, and subtracting it would break the exact lossless energy
    balance. Propagation between the two explicit reference planes is retained.
    """

    epsilon = np.asarray(epsilon_r, dtype=np.complex128)
    if epsilon.shape != setup.grid.shape:
        raise ValueError(f"epsilon_r must have shape {setup.grid.shape}")
    _validate_uniform_port_rows(epsilon, setup)

    forward_lower, backward_lower = specular_projection_rows(
        setup.grid,
        planes=setup.lower_monitor_planes,
        omega=omega,
        refractive_index=setup.port_index,
        bloch_kx=setup.bloch_kx,
    )
    forward_upper, backward_upper = specular_projection_rows(
        setup.grid,
        planes=setup.upper_monitor_planes,
        omega=omega,
        refractive_index=setup.port_index,
        bloch_kx=setup.bloch_kx,
    )
    wave_number = discrete_axial_wavenumber(
        omega,
        refractive_index=setup.port_index,
        dz=setup.port_dz,
        transverse_wavenumber=setup.bloch_kx,
        transverse_spacing=setup.grid.dx,
    )
    lower_z = setup.resolved_lower_reference_z
    upper_z = setup.resolved_upper_reference_z
    forward_lower = forward_lower * np.exp(1j * wave_number * lower_z)
    backward_lower = backward_lower * np.exp(-1j * wave_number * lower_z)
    forward_upper = forward_upper * np.exp(1j * wave_number * upper_z)
    backward_upper = backward_upper * np.exp(-1j * wave_number * upper_z)
    # Global E_x amplitudes equal +H_y for +z waves and -H_y for -z
    # waves, up to the same fixed impedance factor in identical ports.
    output_projector = np.vstack((-backward_lower, forward_upper))

    raw_sources = _raw_sources(setup)
    raw_reference_fields = uniform_reference_fields(
        omega,
        setup.grid,
        (setup.lower_source_plane, setup.upper_source_plane),
        refractive_index=setup.port_index,
        bloch_kx=setup.bloch_kx,
    )

    lower_incident = forward_lower @ raw_reference_fields[:, 0]
    upper_incident = -backward_upper @ raw_reference_fields[:, 1]
    if abs(lower_incident) == 0.0 or abs(upper_incident) == 0.0:
        raise RuntimeError("source calibration produced zero incident amplitude")
    scales = np.asarray((1.0 / lower_incident, 1.0 / upper_incident))
    sources = raw_sources * scales[np.newaxis, :]
    reference_fields = raw_reference_fields * scales[np.newaxis, :]

    output_offset = np.zeros((2, 2), dtype=np.complex128)
    reference_scattering = output_projector @ reference_fields

    return TwoPortLinearSystem(
        omega=complex(omega),
        epsilon_r=epsilon,
        setup=setup,
        operator=_radiating_operator(omega, epsilon, setup),
        sources=np.asarray(sources),
        output_projector=np.asarray(output_projector),
        output_offset=output_offset,
        reference_fields=np.asarray(reference_fields),
        reference_scattering=np.asarray(reference_scattering),
    )


def solve_two_port_scattering(
    omega: complex,
    epsilon_r: ArrayLike,
    setup: TwoPortScatteringSetup,
) -> TwoPortScatteringSolution:
    """Return the full two-by-two scattering matrix at complex frequency."""

    system = assemble_two_port_system(omega, epsilon_r, setup)
    lu = splu(system.operator.tocsc())
    fields = np.asarray(lu.solve(system.sources))
    matrix = system.output_projector @ fields + system.output_offset
    return TwoPortScatteringSolution(
        matrix=np.asarray(matrix), fields=fields, system=system
    )
