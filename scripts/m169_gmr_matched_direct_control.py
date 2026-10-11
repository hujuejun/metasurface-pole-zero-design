"""Shared numerical definitions required by the published studies."""
from __future__ import annotations
import numpy as np
from pzid.adjoint import scattering_matrix_with_jacobian
from scripts.m164_gmr_pole_adjoint import physical_epsilon_and_derivatives

def direct_system(values: np.ndarray, frequencies: np.ndarray, targets: np.ndarray, config: dict, setup) -> tuple[np.ndarray, np.ndarray]:
    epsilon, derivative = physical_epsilon_and_derivatives(values, config, setup)
    residual = np.empty(2 * len(frequencies), dtype=float)
    jacobian = np.empty((2 * len(frequencies), len(values)), dtype=float)
    for index, (frequency, target) in enumerate(zip(frequencies, targets, strict=True)):
        result = scattering_matrix_with_jacobian(float(frequency), epsilon, setup)
        error = result.matrix[1, 0] - target
        gradient = np.einsum('zx,pzx->p', result.jacobian[1, 0], derivative)
        residual[2 * index:2 * index + 2] = (error.real, error.imag)
        jacobian[2 * index] = gradient.real
        jacobian[2 * index + 1] = gradient.imag
    return (residual, jacobian)
