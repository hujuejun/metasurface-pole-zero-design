"""Density-space pole and scalar-channel constraints for 3D inverse design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzid.adjoint3d import scattering_matrix_with_jacobian_3d
from pzid.density3d import (
    DensityMapping3D,
    DensityParameterization3D,
)
from pzid.optimize import complex_system_to_real
from pzid.poles import PoleTarget
from pzid.poles3d import solve_pole_constraint_3d
from pzid.ports3d import Polarization
from pzid.scattering3d import ScatteringSetup3D, polarization_channel_indices


ComplexArray: TypeAlias = NDArray[np.complex128]
RealArray: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class DensityConstraintEvaluation3D:
    """Real optimizer system and underlying complex 3D equalities."""

    residual: RealArray
    jacobian: RealArray
    complex_residual: ComplexArray
    complex_jacobian: ComplexArray
    mapping: DensityMapping3D


def _package(
    complex_residual: ComplexArray,
    epsilon_jacobian: ComplexArray,
    parameterization: DensityParameterization3D,
    mapping: DensityMapping3D,
) -> DensityConstraintEvaluation3D:
    raw_jacobian = parameterization.pullback(epsilon_jacobian, mapping)
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


def pole_density_constraint_3d(
    raw_density: ArrayLike,
    parameterization: DensityParameterization3D,
    target: PoleTarget,
    setup: ScatteringSetup3D,
    *,
    polarization: Polarization = "x",
) -> DensityConstraintEvaluation3D:
    """Return the full-channel Chen pole constraint in raw variables."""

    mapping = parameterization.map(raw_density)
    evaluation = solve_pole_constraint_3d(
        target, mapping.epsilon_r, setup, polarization=polarization
    )
    return _package(
        evaluation.residual,
        evaluation.jacobian,
        parameterization,
        mapping,
    )


def channel_value_density_constraint_3d(
    raw_density: ArrayLike,
    parameterization: DensityParameterization3D,
    frequency: complex,
    channel: Literal["reflection", "transmission"],
    target_value: complex,
    setup: ScatteringSetup3D,
    *,
    polarization: Polarization = "x",
) -> DensityConstraintEvaluation3D:
    """Constrain a lower-incident co-polarized channel value."""

    mapping = parameterization.map(raw_density)
    evaluation = scattering_matrix_with_jacobian_3d(
        frequency, mapping.epsilon_r, setup
    )
    lower, upper = polarization_channel_indices(polarization)
    outgoing = lower if channel == "reflection" else upper
    complex_residual = np.asarray(
        (evaluation.matrix[outgoing, lower] - complex(target_value),)
    )
    epsilon_jacobian = evaluation.jacobian[outgoing, lower][np.newaxis, ...]
    return _package(
        complex_residual,
        epsilon_jacobian,
        parameterization,
        mapping,
    )


def channel_zero_density_constraint_3d(
    raw_density: ArrayLike,
    parameterization: DensityParameterization3D,
    frequency: complex,
    channel: Literal["reflection", "transmission"],
    setup: ScatteringSetup3D,
    *,
    polarization: Polarization = "x",
) -> DensityConstraintEvaluation3D:
    """Return a co-polarized scalar ``r=0`` or ``t=0`` constraint."""

    return channel_value_density_constraint_3d(
        raw_density,
        parameterization,
        frequency,
        channel,
        0.0,
        setup,
        polarization=polarization,
    )


def combine_density_constraints_3d(
    *evaluations: DensityConstraintEvaluation3D,
    weights: ArrayLike | None = None,
) -> tuple[RealArray, RealArray]:
    """Concatenate compatible 3D constraint systems with optional weights."""

    if not evaluations:
        raise ValueError("at least one constraint evaluation is required")
    residual = np.concatenate([item.residual for item in evaluations])
    jacobian = np.vstack([item.jacobian for item in evaluations])
    if weights is None:
        return residual, jacobian
    scaling = np.asarray(weights, dtype=float).ravel()
    if scaling.size != residual.size:
        raise ValueError("weights must contain one value per real residual")
    return scaling * residual, scaling[:, np.newaxis] * jacobian
