from __future__ import annotations

from typing import Any

import numpy as np

from pzid.adjoint import scattering_matrix_with_jacobian

from pzid.optimize import complex_system_to_real

from scripts.m142e_optimize_patterned_q_poles import (
    _verify_final,
    build_fixed_grid,
    physical_epsilon_and_derivatives,
)

def direct_complex_transmission_system(
    coordinates: np.ndarray,
    frequencies: np.ndarray,
    targets: np.ndarray,
    geometry_config: dict[str, Any],
    source: dict[str, Any],
    scalar: Any,
    twoport: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Complex-t residual and exact 2D adjoint Jacobian in physical coords."""

    epsilon, d_epsilon = physical_epsilon_and_derivatives(
        coordinates, geometry_config, source, scalar
    )
    residual = np.empty(len(frequencies), dtype=complex)
    jacobian = np.empty((len(frequencies), 2), dtype=complex)
    for index, omega in enumerate(frequencies):
        evaluation = scattering_matrix_with_jacobian(float(omega), epsilon, twoport)
        residual[index] = evaluation.matrix[1, 0] - targets[index]
        jacobian[index] = np.einsum(
            "zx,pzx->p", evaluation.jacobian[1, 0], d_epsilon
        )
    return complex_system_to_real(residual, jacobian)

def check_direct_adjoint(
    coordinates: np.ndarray,
    frequencies: np.ndarray,
    targets: np.ndarray,
    geometry_config: dict[str, Any],
    source: dict[str, Any],
    scalar: Any,
    twoport: Any,
    step: float,
) -> dict[str, Any]:
    residual, jacobian = direct_complex_transmission_system(
        coordinates, frequencies, targets, geometry_config, source, scalar, twoport
    )
    errors = []
    for coordinate in range(2):
        displacement = np.eye(2)[coordinate] * step
        plus, _ = direct_complex_transmission_system(
            coordinates + displacement, frequencies, targets,
            geometry_config, source, scalar, twoport
        )
        minus, _ = direct_complex_transmission_system(
            coordinates - displacement, frequencies, targets,
            geometry_config, source, scalar, twoport
        )
        numeric = (plus - minus) / (2.0 * step)
        errors.append(
            float(np.linalg.norm(numeric - jacobian[:, coordinate]) / max(np.linalg.norm(numeric), 1.0))
        )
    return {"relative_errors": errors, "initial_residual_norm": float(np.linalg.norm(residual))}
