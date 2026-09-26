"""Reusable multi-frequency 3D pole-zero inverse-design problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzid.adjoint3d import (
    ScatteringJacobianSolution3D,
    scattering_matrix_with_jacobian_3d,
)
from pzid.constraints3d import DensityConstraintEvaluation3D
from pzid.density3d import DensityParameterization3D
from pzid.optimize import LMOptions, LMResult, complex_system_to_real, solve_underdetermined_lm
from pzid.poles import PoleTarget
from pzid.poles3d import time_reversal_incoming_3d
from pzid.ports3d import Polarization
from pzid.scattering3d import (
    ScatteringSetup3D,
    deembed_scattering_jacobian_3d,
    deembed_scattering_matrix_3d,
    polarization_channel_indices,
)


ComplexArray: TypeAlias = NDArray[np.complex128]
RealArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class PoleConstraintSpec3D:
    """One prescribed passive pole and its full-channel time-reversal weight."""

    target: PoleTarget
    polarization: Polarization = "x"
    weight: float = 1.0
    local_reference: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("pole weight must be finite and positive")


@dataclass(frozen=True)
class ChannelConstraintSpec3D:
    """One selected co-polarized channel equality at complex frequency."""

    frequency: complex
    channel: Literal["reflection", "transmission"]
    target_value: complex = 0.0
    polarization: Polarization = "x"
    weight: float = 1.0
    local_reference: bool = False

    def __post_init__(self) -> None:
        if self.channel not in {"reflection", "transmission"}:
            raise ValueError("channel must be 'reflection' or 'transmission'")
        if not np.isfinite(complex(self.frequency)):
            raise ValueError("constraint frequency must be finite")
        if not np.isfinite(complex(self.target_value)):
            raise ValueError("target value must be finite")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("channel weight must be finite and positive")


@dataclass(frozen=True)
class ChannelSlopeConstraintSpec3D:
    """Finite-difference channel slope equality at two complex frequencies."""

    lower_frequency: complex
    upper_frequency: complex
    channel: Literal["reflection", "transmission"]
    target_value: complex = 0.0
    polarization: Polarization = "x"
    weight: float = 1.0
    local_reference: bool = False

    def __post_init__(self) -> None:
        if self.channel not in {"reflection", "transmission"}:
            raise ValueError("channel must be 'reflection' or 'transmission'")
        lower = complex(self.lower_frequency)
        upper = complex(self.upper_frequency)
        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("slope frequencies must be finite")
        if upper == lower:
            raise ValueError("slope frequencies must differ")
        if not np.isfinite(complex(self.target_value)):
            raise ValueError("slope target must be finite")
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("slope weight must be finite and positive")


@dataclass
class PoleZeroInverseProblem3D:
    """Map raw density to all prescribed pole, zero, and background rows."""

    parameterization: DensityParameterization3D
    setup: ScatteringSetup3D
    poles: tuple[PoleConstraintSpec3D, ...] = ()
    channels: tuple[ChannelConstraintSpec3D, ...] = ()
    channel_slopes: tuple[ChannelSlopeConstraintSpec3D, ...] = ()

    def __post_init__(self) -> None:
        if self.parameterization.grid != self.setup.grid:
            raise ValueError("parameterization and scattering setup need one grid")
        if not self.poles and not self.channels and not self.channel_slopes:
            raise ValueError("at least one pole or channel constraint is required")

    def evaluate(self, raw_density: ArrayLike) -> DensityConstraintEvaluation3D:
        """Evaluate all equalities and exact density-space Jacobians."""

        mapping = self.parameterization.map(raw_density)
        residual_blocks: list[ComplexArray] = []
        epsilon_jacobian_blocks: list[ComplexArray] = []
        frequency_cache: dict[complex, ScatteringJacobianSolution3D] = {}

        def scattering_at(frequency: complex) -> ScatteringJacobianSolution3D:
            key = complex(frequency)
            if key not in frequency_cache:
                frequency_cache[key] = scattering_matrix_with_jacobian_3d(
                    key, mapping.epsilon_r, self.setup
                )
            return frequency_cache[key]

        def represented(
            solution: ScatteringJacobianSolution3D,
            frequency: complex,
            local_reference: bool,
        ) -> tuple[ComplexArray, ComplexArray]:
            if not local_reference:
                return solution.matrix, solution.jacobian
            return (
                deembed_scattering_matrix_3d(
                    solution.matrix, frequency, self.setup
                ),
                deembed_scattering_jacobian_3d(
                    solution.jacobian, frequency, self.setup
                ),
            )

        for specification in self.poles:
            solution = scattering_at(
                np.conj(specification.target.frequency)
            )
            frequency = np.conj(specification.target.frequency)
            matrix, scattering_jacobian = represented(
                solution, frequency, specification.local_reference
            )
            incoming = time_reversal_incoming_3d(
                specification.target.coupling_ratio,
                specification.polarization,
            )
            residual_blocks.append(
                specification.weight * (matrix @ incoming)
            )
            epsilon_jacobian_blocks.append(
                specification.weight
                * np.einsum("abcde,b->acde", scattering_jacobian, incoming)
            )

        for specification in self.channels:
            solution = scattering_at(specification.frequency)
            matrix, scattering_jacobian = represented(
                solution,
                specification.frequency,
                specification.local_reference,
            )
            lower, upper = polarization_channel_indices(
                specification.polarization
            )
            outgoing = (
                lower if specification.channel == "reflection" else upper
            )
            residual_blocks.append(
                np.asarray(
                    (
                        specification.weight
                        * (
                            matrix[outgoing, lower]
                            - specification.target_value
                        ),
                    )
                )
            )
            epsilon_jacobian_blocks.append(
                specification.weight
                * scattering_jacobian[outgoing, lower][np.newaxis, ...]
            )

        for specification in self.channel_slopes:
            lower_solution = scattering_at(specification.lower_frequency)
            upper_solution = scattering_at(specification.upper_frequency)
            lower_matrix, lower_jacobian = represented(
                lower_solution,
                specification.lower_frequency,
                specification.local_reference,
            )
            upper_matrix, upper_jacobian = represented(
                upper_solution,
                specification.upper_frequency,
                specification.local_reference,
            )
            incident, transmitted = polarization_channel_indices(
                specification.polarization
            )
            outgoing = (
                incident if specification.channel == "reflection" else transmitted
            )
            spacing = specification.upper_frequency - specification.lower_frequency
            slope = (
                upper_matrix[outgoing, incident]
                - lower_matrix[outgoing, incident]
            ) / spacing
            slope_jacobian = (
                upper_jacobian[outgoing, incident]
                - lower_jacobian[outgoing, incident]
            ) / spacing
            residual_blocks.append(
                np.asarray(
                    (
                        specification.weight
                        * (slope - specification.target_value),
                    )
                )
            )
            epsilon_jacobian_blocks.append(
                specification.weight * slope_jacobian[np.newaxis, ...]
            )

        complex_residual = np.concatenate(residual_blocks)
        epsilon_jacobian = np.concatenate(epsilon_jacobian_blocks, axis=0)
        raw_jacobian = self.parameterization.pullback(
            epsilon_jacobian, mapping
        )
        residual, jacobian = complex_system_to_real(
            complex_residual, raw_jacobian
        )
        return DensityConstraintEvaluation3D(
            residual=residual,
            jacobian=jacobian,
            complex_residual=np.asarray(complex_residual),
            complex_jacobian=np.asarray(raw_jacobian),
            mapping=mapping,
        )

    def residual_system(self, raw_flat: RealArray) -> tuple[RealArray, RealArray]:
        """Adapter consumed by the dual-space LM implementation."""

        result = self.evaluate(raw_flat)
        return result.residual, result.jacobian.reshape(
            result.residual.size, -1
        )

    def optimize(
        self,
        initial_raw: ArrayLike,
        *,
        options: LMOptions | None = None,
    ) -> LMResult:
        """Run bounded Chen-style underdetermined LM continuation."""

        initial = np.asarray(initial_raw, dtype=float)
        if initial.size != self.parameterization.variable_count:
            raise ValueError("initial_raw size does not match the design volume")
        return solve_underdetermined_lm(
            self.residual_system,
            initial.ravel(),
            lower_bound=0.0,
            upper_bound=1.0,
            options=options,
        )
