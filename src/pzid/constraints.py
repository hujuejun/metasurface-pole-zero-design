"""Pole and scalar-channel constraints on raw density variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pzid.adjoint import scattering_matrix_with_jacobian
from pzid.density import DensityMapping, DensityParameterization
from pzid.optimize import complex_system_to_real
from pzid.poles import PoleTarget, solve_pole_constraint
from pzid.twoport import TwoPortScatteringSetup


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


@dataclass(frozen=True)
class DensityConstraintEvaluation:
    """Real optimizer system plus its underlying complex equalities."""

    residual: RealArray
    jacobian: RealArray
    complex_residual: ComplexArray
    complex_jacobian: ComplexArray
    mapping: DensityMapping


def _package(
    complex_residual: ComplexArray,
    epsilon_jacobian: ComplexArray,
    parameterization: DensityParameterization,
    mapping: DensityMapping,
) -> DensityConstraintEvaluation:
    raw_jacobian = parameterization.pullback(epsilon_jacobian, mapping)
    residual, jacobian = complex_system_to_real(
        complex_residual, raw_jacobian
    )
    return DensityConstraintEvaluation(
        residual=residual,
        jacobian=jacobian,
        complex_residual=complex_residual,
        complex_jacobian=np.asarray(raw_jacobian),
        mapping=mapping,
    )


def pole_density_constraint(
    raw_density: ArrayLike,
    parameterization: DensityParameterization,
    target: PoleTarget,
    setup: TwoPortScatteringSetup,
) -> DensityConstraintEvaluation:
    """Return the two-complex-equation Chen pole constraint in raw variables."""

    mapping = parameterization.map(raw_density)
    evaluation = solve_pole_constraint(target, mapping.epsilon_r, setup)
    return _package(
        evaluation.residual,
        evaluation.jacobian,
        parameterization,
        mapping,
    )


def channel_zero_density_constraint(
    raw_density: ArrayLike,
    parameterization: DensityParameterization,
    frequency: complex,
    channel: Literal["reflection", "transmission"],
    setup: TwoPortScatteringSetup,
) -> DensityConstraintEvaluation:
    """Return a lower-incidence scalar ``r=0`` or ``t=0`` constraint."""

    return channel_value_density_constraint(
        raw_density,
        parameterization,
        frequency,
        channel,
        0.0,
        setup,
    )


def channel_value_density_constraint(
    raw_density: ArrayLike,
    parameterization: DensityParameterization,
    frequency: complex,
    channel: Literal["reflection", "transmission"],
    target_value: complex,
    setup: TwoPortScatteringSetup,
) -> DensityConstraintEvaluation:
    """Constrain a lower-incidence scalar channel to a complex target value."""

    mapping = parameterization.map(raw_density)
    evaluation = scattering_matrix_with_jacobian(
        frequency, mapping.epsilon_r, setup
    )
    outgoing = 0 if channel == "reflection" else 1
    complex_residual = np.asarray(
        (evaluation.matrix[outgoing, 0] - complex(target_value),)
    )
    epsilon_jacobian = evaluation.jacobian[outgoing, 0][np.newaxis, ...]
    return _package(
        complex_residual,
        epsilon_jacobian,
        parameterization,
        mapping,
    )


def combine_density_constraints(
    *evaluations: DensityConstraintEvaluation,
    weights: ArrayLike | None = None,
) -> tuple[RealArray, RealArray]:
    """Concatenate compatible real residual systems with optional weights."""

    if not evaluations:
        raise ValueError("at least one constraint evaluation is required")
    residual = np.concatenate([evaluation.residual for evaluation in evaluations])
    jacobian = np.vstack([evaluation.jacobian for evaluation in evaluations])
    if weights is None:
        return residual, jacobian
    scaling = np.asarray(weights, dtype=float).ravel()
    if scaling.size != residual.size:
        raise ValueError("weights must contain one value per real residual")
    return scaling * residual, scaling[:, np.newaxis] * jacobian
