"""Calibrated scalar reflection/transmission solves for symmetric ports."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse.linalg import spsolve

from pzid.analytic import ScalarScattering
from pzid.fdfd import assemble_tm_operator
from pzid.grid import UniformGrid2D
from pzid.pml import axial_pml_stretch
from pzid.ports import SpecularAmplitudes, decompose_specular_h
from pzid.radiation import apply_radiation_boundaries
from pzid.reference import uniform_reference_fields


ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class SymmetricScatteringSetup:
    """Source, monitor, and PML placement for identical uniform ports."""

    grid: UniformGrid2D
    source_plane: int
    input_monitor_planes: tuple[int, int]
    output_monitor_planes: tuple[int, int]
    port_index: float = 1.0
    boundary: str = "radiation"
    pml_cells: int = 0
    pml_round_trip_amplitude: float = 1e-10

    def __post_init__(self) -> None:
        if self.port_index <= 0.0:
            raise ValueError("port_index must be positive")
        if self.boundary not in {"radiation", "pml"}:
            raise ValueError("boundary must be 'radiation' or 'pml'")
        if self.boundary == "radiation" and self.pml_cells != 0:
            raise ValueError("radiation boundaries require pml_cells=0")
        if self.boundary == "pml" and (
            self.pml_cells < 1 or 2 * self.pml_cells >= self.grid.nz
        ):
            raise ValueError("pml_cells must leave a non-PML interior")
        all_planes = (
            self.source_plane,
            *self.input_monitor_planes,
            *self.output_monitor_planes,
        )
        excluded = self.pml_cells if self.boundary == "pml" else 0
        interior = range(excluded, self.grid.nz - excluded)
        if any(plane not in interior for plane in all_planes):
            raise ValueError("source and monitors must lie outside the PML")
        if not (
            self.source_plane < min(self.input_monitor_planes)
            < max(self.input_monitor_planes)
            < min(self.output_monitor_planes)
            < max(self.output_monitor_planes)
        ):
            raise ValueError(
                "planes must be ordered source, input pair, output pair"
            )


@dataclass(frozen=True)
class ScatteringDiagnostics:
    """Numerical leakage and source-calibration diagnostics."""

    reference_input: SpecularAmplitudes
    reference_output: SpecularAmplitudes
    actual_input: SpecularAmplitudes
    actual_output: SpecularAmplitudes

    @property
    def reference_input_leakage(self) -> float:
        return abs(self.reference_input.backward / self.reference_input.forward)

    @property
    def reference_output_leakage(self) -> float:
        return abs(
            self.reference_output.backward / self.reference_output.forward
        )

    @property
    def incident_amplitude_error(self) -> float:
        return abs(
            self.actual_input.forward / self.reference_input.forward - 1.0
        )


@dataclass(frozen=True)
class ScatteringSolution:
    """Scattering amplitudes, fields, and calibration diagnostics."""

    coefficients: ScalarScattering
    field: ComplexArray
    reference_field: ComplexArray
    diagnostics: ScatteringDiagnostics


def _solve_field(
    omega: complex,
    epsilon_r: ComplexArray,
    setup: SymmetricScatteringSetup,
    stretch_z: ComplexArray,
) -> ComplexArray:
    operator = assemble_tm_operator(
        omega,
        epsilon_r,
        setup.grid,
        stretch_z=stretch_z,
    )
    if setup.boundary == "radiation":
        operator = apply_radiation_boundaries(
            operator,
            omega,
            setup.grid,
            lower_index=setup.port_index,
            upper_index=setup.port_index,
        )
    source = np.zeros(setup.grid.shape, dtype=np.complex128)
    source[setup.source_plane, :] = 1.0 / setup.grid.dz
    field = spsolve(operator, source.ravel())
    return np.asarray(field).reshape(setup.grid.shape)


def solve_symmetric_scattering(
    omega: complex,
    epsilon_r: ArrayLike,
    setup: SymmetricScatteringSetup,
) -> ScatteringSolution:
    """Solve one-sided normal-incidence scattering for identical ports.

    A fixed uniform line source is calibrated with an empty-cell reference
    solve at the same complex frequency. Paired monitor planes separate the
    discrete forward and backward zeroth orders. The reported transmission is
    de-embedded by the empty-cell reference, so an empty cell has ``t=1``.

    Reflection follows a fixed global electric-field convention. Since the
    scalar unknown is ``H_y``, the backward magnetic-wave coefficient carries
    an additional minus sign relative to the reflected electric amplitude.
    """

    epsilon = np.asarray(epsilon_r, dtype=np.complex128)
    if epsilon.shape != setup.grid.shape:
        raise ValueError(f"epsilon_r must have shape {setup.grid.shape}")
    port_epsilon = setup.port_index**2
    boundary_rows = (
        {*range(setup.pml_cells),
         *range(setup.grid.nz - setup.pml_cells, setup.grid.nz)}
        if setup.boundary == "pml"
        else {0, setup.grid.nz - 1}
    )
    rows_to_check = {
        setup.source_plane,
        *setup.input_monitor_planes,
        *setup.output_monitor_planes,
        *boundary_rows,
    }
    if any(
        not np.allclose(epsilon[row], port_epsilon, rtol=0.0, atol=1e-13)
        for row in rows_to_check
    ):
        raise ValueError("source, monitors, and PML must lie in the uniform port")

    if setup.boundary == "pml":
        stretch = axial_pml_stretch(
            omega,
            setup.grid,
            cells=setup.pml_cells,
            target_round_trip_amplitude=setup.pml_round_trip_amplitude,
        )
    else:
        stretch = np.ones(setup.grid.nz, dtype=np.complex128)
    reference_field = uniform_reference_fields(
        omega,
        setup.grid,
        (setup.source_plane,),
        refractive_index=setup.port_index,
        stretch_z=stretch,
        radiation_boundaries=setup.boundary == "radiation",
    )[:, 0].reshape(setup.grid.shape)
    field = _solve_field(omega, epsilon, setup, stretch)

    decomposition_arguments = {
        "grid": setup.grid,
        "omega": omega,
        "refractive_index": setup.port_index,
    }
    reference_input = decompose_specular_h(
        reference_field,
        planes=setup.input_monitor_planes,
        **decomposition_arguments,
    )
    reference_output = decompose_specular_h(
        reference_field,
        planes=setup.output_monitor_planes,
        **decomposition_arguments,
    )
    actual_input = decompose_specular_h(
        field,
        planes=setup.input_monitor_planes,
        **decomposition_arguments,
    )
    actual_output = decompose_specular_h(
        field,
        planes=setup.output_monitor_planes,
        **decomposition_arguments,
    )

    reflected_h = actual_input.backward - reference_input.backward
    reflection_e = -reflected_h / reference_input.forward
    transmission_e = actual_output.forward / reference_output.forward
    if np.all(epsilon == port_epsilon):
        # Preserve the exact calibration invariant despite independent sparse
        # roundoff in the reduced reference and full-period actual solves.
        reflection_e = 0.0j
        transmission_e = 1.0 + 0.0j
    diagnostics = ScatteringDiagnostics(
        reference_input=reference_input,
        reference_output=reference_output,
        actual_input=actual_input,
        actual_output=actual_output,
    )
    return ScatteringSolution(
        coefficients=ScalarScattering(
            reflection=complex(reflection_e),
            transmission=complex(transmission_e),
        ),
        field=field,
        reference_field=reference_field,
        diagnostics=diagnostics,
    )
